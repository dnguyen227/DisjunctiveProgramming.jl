# [GDP Models](@id model_guide)

A guide for creating and working with generalized disjunctive programming
models. See the [API](@ref) for the technical details of each method.

## Overview

Generalized disjunctive programming (GDP) expresses a discrete decision as a
choice between alternative sets of constraints rather than as an algebraic
relationship between binary variables. Each alternative is called a *disjunct*,
a group of mutually exclusive disjuncts is a *disjunction*, and a Boolean
*logical variable* records which disjunct is selected. Writing a model this way
keeps the modeling layer close to the way the problem is actually described, and
it defers the choice of algebraic encoding to solution time.

A [`GDPModel`](@ref) is a JuMP model that carries the extra bookkeeping needed to
support this. It holds ordinary JuMP variables and constraints exactly as a
`Model` does, and adds storage for logical variables, disjunct constraints,
disjunctions, and logical constraints. None of that structure is passed to a
solver directly. Instead, calling `optimize!` first *reformulates* the model into
an equivalent mixed-integer program using a method of your choosing, and then
solves that program. The reformulation is described in
[Solution Methods](@ref methods_guide).

## Basic Usage

A GDP model is created with [`GDPModel`](@ref), which accepts the same arguments
as JuMP's `Model`:

```@example gdp_model
using DisjunctiveProgramming, HiGHS

model = GDPModel(HiGHS.Optimizer)
set_silent(model)
```

Ordinary variables and constraints are added with the usual JuMP macros. Logical
variables use the [`Logical`](@ref) variable type, and a constraint is assigned
to a disjunct by tagging it with [`Disjunct`](@ref):

```@example gdp_model
@variable(model, 0 <= production <= 20)
@variable(model, 0 <= cost <= 100)
@variable(model, Y[1:2], Logical)

@constraint(model, production <= 12, Disjunct(Y[1]))
@constraint(model, cost >= 5 + 2 * production, Disjunct(Y[1]))

@constraint(model, production <= 20, Disjunct(Y[2]))
@constraint(model, cost >= 12 + production, Disjunct(Y[2]))

@disjunction(model, Y)
nothing # hide
```

Constraints that hold no matter which disjunct is chosen are added without a
`Disjunct` tag, in the normal way:

```@example gdp_model
@constraint(model, demand, production >= 10)
@objective(model, Min, cost)
nothing # hide
```

Solving requires choosing a reformulation. The `gdp_method` keyword of
`optimize!` selects one, and defaults to [`BigM`](@ref):

```@example gdp_model
optimize!(model, gdp_method = BigM())

println("cost       = ", objective_value(model))
println("production = ", value(production))
println("process 1  = ", value(Y[1]))
println("process 2  = ", value(Y[2]))
```

Note that `value` applied to a logical variable returns a `Bool`, not a
floating-point number, so the selected disjunct can be read off directly.

!!! note
    An optimizer is required for every reformulation, and some methods need one
    to solve auxiliary subproblems as well. [`MBM`](@ref) and
    [`CuttingPlanes`](@ref) both take an optimizer as their first argument for
    exactly this reason.

## How a GDP Model is Stored

A `GDPModel` is a JuMP `Model` with a [`GDPData`](@ref) object attached to its
extension dictionary. That object records the logical variables, the disjunct
constraints grouped by their indicator, the disjunctions, and the logical
constraints, together with the mappings produced during reformulation. The
mappings are what allow a solved model to be queried in terms of the original
logical variables rather than the binary variables that replaced them.

Because reformulation adds variables and constraints to the same model object,
it is performed at most once for a given method. The model records which method
was applied and whether it is still current, and repeating an `optimize!` call
with the same method reuses the existing reformulation instead of rebuilding it.
Changing the method, or adding new disjunctive structure, marks the model as
needing to be reformulated again.

!!! warning
    Reformulation mutates the model in place. If you need the original GDP
    structure preserved for a later experiment, build the model inside a
    function so it can be constructed fresh for each method, rather than
    reformulating one model repeatedly.

## Queries

The data attached to a model is reached with [`gdp_data`](@ref), and
[`is_gdp_model`](@ref) reports whether a given JuMP model carries it:

```@example gdp_model
is_gdp_model(model)
```

```@example gdp_model
gdp_data(model) isa DisjunctiveProgramming.GDPData
```

All the standard JuMP result queries apply unchanged, since after reformulation
the model is an ordinary mixed-integer program:

```@example gdp_model
termination_status(model)
```

## Modification

A model can be reformulated explicitly, without solving, using
[`reformulate_model`](@ref). This is useful for inspecting the mixed-integer
program that a given method produces:

```@example gdp_model
inspection = GDPModel()
@variable(inspection, 0 <= x <= 20)
@variable(inspection, W[1:2], Logical)
@constraint(inspection, x <= 5, Disjunct(W[1]))
@constraint(inspection, x >= 15, Disjunct(W[2]))
@disjunction(inspection, W)

reformulate_model(inspection, BigM())
print(inspection)
```

The same model reformulated with [`Hull`](@ref) produces a larger but tighter
program, because each variable appearing in a disjunct is disaggregated into one
copy per disjunct:

```@example gdp_model
comparison = GDPModel()
@variable(comparison, 0 <= x <= 20)
@variable(comparison, W[1:2], Logical)
@constraint(comparison, x <= 5, Disjunct(W[1]))
@constraint(comparison, x >= 15, Disjunct(W[2]))
@disjunction(comparison, W)

reformulate_model(comparison, Hull())
num_variables(comparison), num_variables(inspection)
```

## Next Steps

- [Logical Variables](@ref variables_guide) covers Boolean decisions and their
  properties.
- [Logical Constraints](@ref logic_guide) covers propositions and cardinality
  requirements over those decisions.
- [Disjunctions](@ref constraints_guide) covers disjunct constraints, nesting,
  and disjunction construction.
- [Solution Methods](@ref methods_guide) covers each reformulation in detail.
