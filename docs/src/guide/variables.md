# [Logical Variables](@id variables_guide)

A guide for logical variables in `DisjunctiveProgramming`. See the [API](@ref)
for the technical details of each method.

## Overview

A logical variable is a Boolean decision. It takes the value `true` or `false`,
and its role in a GDP model is to indicate whether the constraints of a
particular disjunct are enforced. Logical variables are distinct from JuMP's
binary variables: they cannot appear inside algebraic constraints, and they are
not handed to the solver as they stand. During reformulation each one is
replaced by a binary variable, and the constraints tagged to it are encoded in
terms of that binary.

Keeping the two kinds of variable separate is what allows a single model to be
reformulated several different ways. The logical layer records the decision
structure, and the reformulation decides how that structure becomes algebra.

## Basic Usage

Logical variables are declared with `@variable` using the [`Logical`](@ref)
variable type, and support the same container syntax as any JuMP variable:

```@example gdp_vars
using DisjunctiveProgramming

model = GDPModel()

@variable(model, Y, Logical)
@variable(model, Z[1:3], Logical)
@variable(model, W[1:2, 1:3], Logical)
nothing # hide
```

A start value may be supplied at declaration:

```@example gdp_vars
@variable(model, V[1:2], Logical, start = true)
nothing # hide
```

!!! warning
    There is no `fix` keyword on the `@variable` declaration. To hold a logical
    variable at a value, declare it and then call `fix` on it, as shown under
    [Modification](@ref var_modification) below.

### Logical Complements

When a decision is genuinely binary, the second alternative can be declared as
the complement of the first. The complement is not an independent variable: its
value is constrained to be the negation of the variable it complements, so no
separate exclusivity constraint is needed.

```@example gdp_vars
@variable(model, use_process, Logical)
@variable(model, skip_process, Logical, logical_complement = use_process)

has_logical_complement(skip_process)
```

Using a complement rather than two free logical variables reduces the number of
binaries in the reformulated program, and is worth doing whenever a disjunction
has exactly two disjuncts.

## Queries

The Boolean state of a variable and its relationship to the reformulated model
are both queryable:

```@example gdp_vars
start_value(V[1])
```

```@example gdp_vars
is_fixed(Y)
```

The binary variable that stands in for a logical variable after reformulation is
reached with [`binary_variable`](@ref):

```@example gdp_vars
binary_variable(Y)
```

After a solve, `value` returns a `Bool`:

```@example gdp_vars
using HiGHS

solved = GDPModel(HiGHS.Optimizer)
set_silent(solved)
@variable(solved, 0 <= x <= 10)
@variable(solved, S[1:2], Logical)
@constraint(solved, x <= 3, Disjunct(S[1]))
@constraint(solved, x >= 7, Disjunct(S[2]))
@disjunction(solved, S)
@objective(solved, Max, x)
optimize!(solved)

value.(S)
```

## [Modification](@id var_modification)

Logical variables support the usual JuMP modification methods. `fix` holds a
variable at a Boolean value, which is how a disjunct is forced in or out of the
solution:

```@example gdp_vars
fix(Y, true)
is_fixed(Y), fix_value(Y)
```

```@example gdp_vars
unfix(Y)
is_fixed(Y)
```

Start values and names may also be set after declaration:

```@example gdp_vars
set_start_value(Y, false)
set_name(Y, "use_first_process")
name(Y)
```

!!! tip
    Fixing a logical variable is the correct way to assert that a single
    decision is true or false. Writing a logical constraint on a lone variable,
    such as `@constraint(model, Y := true)`, is rejected precisely because `fix`
    already expresses it, and does so without adding a constraint to the
    reformulated program.

## Reformulation Tags

The binary variable created for a logical variable can be given a tag, which is
forwarded to `@variable` as the variable type when the binary is built. This is
an extension mechanism: it lets another package attach its own variable
behaviour to the binaries that reformulation produces.

A tag is any type for which `JuMP.build_variable` is defined:

```@example gdp_vars
struct MyTag end

function JuMP.build_variable(::Function, info::JuMP.VariableInfo, ::MyTag; kwargs...)
    return JuMP.ScalarVariable(info)
end

tagged = GDPModel()
@variable(tagged, T[1:2], Logical(MyTag()))
nothing # hide
```

Every binary variable created for `T` during reformulation is then built through
the `MyTag` method rather than the default one.

## Next Steps

- [Logical Constraints](@ref logic_guide) covers propositions and cardinality
  requirements relating several logical variables.
- [Disjunctions](@ref constraints_guide) covers attaching constraints to these
  variables.
