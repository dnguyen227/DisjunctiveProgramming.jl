# [Disjunctions](@id constraints_guide)

A guide for disjunct constraints and disjunctions. See the [API](@ref) for the
technical details of each method.

## Overview

A *disjunct constraint* is an ordinary algebraic constraint that is enforced only
when its associated logical variable is true. A *disjunction* is a group of
logical variables among which exactly one may be selected, so that exactly one
group of disjunct constraints is enforced.

The two are declared separately. Constraints are tagged with the logical
variable that governs them, and the disjunction is then declared over those
logical variables. Nothing prevents a logical variable from carrying constraints
without ever appearing in a disjunction, which is how conditional constraints
that are not part of an either-or choice are expressed.

## Basic Usage

### Disjunct Constraints

A constraint becomes a disjunct constraint when a [`Disjunct`](@ref) tag naming
its logical variable is passed to `@constraint`. Any constraint JuMP accepts may
be tagged, including nonlinear and vector constraints:

```@example gdp_cons
using DisjunctiveProgramming, HiGHS

model = GDPModel(HiGHS.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:2] <= 10)
@variable(model, Y[1:2], Logical)

@constraint(model, x[1] + x[2] <= 8, Disjunct(Y[1]))
@constraint(model, x[1] - x[2] >= 2, Disjunct(Y[1]))

@constraint(model, x[1] + 3 * x[2] <= 12, Disjunct(Y[2]))
nothing # hide
```

The tag is the last argument, after the constraint expression. Named and
containerized constraints follow JuMP's usual syntax, with the name or index
expression preceding the constraint:

```@example gdp_cons
@constraint(model, capacity[i = 1:2], x[i] <= 6, Disjunct(Y[1]))
nothing # hide
```

!!! note
    A quadratic or otherwise nonlinear disjunct constraint is supported by
    [`BigM`](@ref) and [`Hull`](@ref), but the resulting program is only as
    tractable as the underlying solver makes it. `Hull` additionally requires a
    perspective reformulation for nonlinear terms, controlled by its epsilon
    parameter.

### Declaring a Disjunction

[`@disjunction`](@ref) takes a vector of logical variables. Passing a container
directly is equivalent to passing its elements:

```@example gdp_cons
@disjunction(model, Y)
nothing # hide
```

A disjunction may be named, in which case it is registered on the model and can
be retrieved later:

```@example gdp_cons
named = GDPModel()
@variable(named, 0 <= z <= 10)
@variable(named, W[1:2], Logical)
@constraint(named, z <= 3, Disjunct(W[1]))
@constraint(named, z >= 7, Disjunct(W[2]))

@disjunction(named, choice, W)
named[:choice]
```

The function form [`disjunction`](@ref) does the same without a macro, which is
convenient when disjunctions are built programmatically:

```@example gdp_cons
programmatic = GDPModel()
@variable(programmatic, 0 <= w <= 10)
@variable(programmatic, V[1:2], Logical)
@constraint(programmatic, w <= 3, Disjunct(V[1]))
@constraint(programmatic, w >= 7, Disjunct(V[2]))

disjunction(programmatic, V)
nothing # hide
```

Several disjunctions may be declared in one block with
[`@disjunctions`](@ref):

```@example gdp_cons
several = GDPModel()
@variable(several, 0 <= v <= 10)
@variable(several, Z[1:2, 1:2], Logical)
for i in 1:2, j in 1:2
    @constraint(several, v <= i + j, Disjunct(Z[i, j]))
end

@disjunctions(several, begin
    Z[1, :]
    Z[2, :]
end)
nothing # hide
```

### Exclusivity

By default a disjunction adds a constraint requiring that exactly one of its
disjuncts be selected. The `exactly1` keyword relaxes this, permitting any
number of disjuncts to be active, which is occasionally wanted when the
disjunction encodes a set of independently available options:

```@example gdp_cons
relaxed = GDPModel()
@variable(relaxed, 0 <= u <= 10)
@variable(relaxed, R[1:2], Logical)
@constraint(relaxed, u <= 3, Disjunct(R[1]))
@constraint(relaxed, u <= 7, Disjunct(R[2]))

@disjunction(relaxed, R, exactly1 = false)
nothing # hide
```

!!! warning
    Some reformulations rely on the exclusivity constraint for correctness or
    for tightness. Setting `exactly1 = false` is appropriate only when the
    modeling intent genuinely permits several disjuncts at once.

## Nested Disjunctions

A disjunction may itself sit inside a disjunct, giving a hierarchy of decisions.
This is done by passing a `Disjunct` tag to `@disjunction`, exactly as for a
constraint. The inner disjunction is then enforced only when the outer disjunct
is selected:

```@example gdp_cons
nested = GDPModel(HiGHS.Optimizer)
set_silent(nested)

@variable(nested, 0 <= capacity <= 500)
@variable(nested, build, Logical)
@variable(nested, outsource, Logical, logical_complement = build)
@variable(nested, size_choice[1:2], Logical)

@constraint(nested, capacity <= 100, Disjunct(size_choice[1]))
@constraint(nested, capacity <= 500, Disjunct(size_choice[2]))
@constraint(nested, capacity <= 50, Disjunct(outsource))

@disjunction(nested, inner, size_choice, Disjunct(build))
@disjunction(nested, [build, outsource])

@objective(nested, Max, capacity)
optimize!(nested, gdp_method = BigM())

value(build), value(capacity)
```

Note that the inner disjunction is declared before the outer one. The logical
variables of the inner disjunction carry constraints of their own, and the outer
disjunction governs whether that whole sub-decision is active.

## Next Steps

- [Logical Constraints](@ref logic_guide) covers requirements relating logical
  variables directly.
- [Solution Methods](@ref methods_guide) covers how disjunctions become
  mixed-integer constraints.
