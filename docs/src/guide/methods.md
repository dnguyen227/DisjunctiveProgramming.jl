# [Solution Methods](@id methods_guide)

A guide for the reformulations that turn a GDP model into a mixed-integer
program. See the [API](@ref) for the technical details of each method.

## Overview

A `GDPModel` is not solved directly. Before a solver sees it, the disjunctions
and logical constraints are replaced by algebraic constraints on binary
variables, producing an ordinary mixed-integer program. The method that performs
this replacement is chosen by the user, and the choice matters: different
reformulations of the same model are equivalent in their integer solutions but
differ sharply in the strength of their continuous relaxation, and therefore in
how long a branch-and-bound solver takes.

The method is passed through the `gdp_method` keyword of `optimize!`:

```@example gdp_methods
using DisjunctiveProgramming, HiGHS

function production_model()
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= production <= 20)
    @variable(model, 0 <= cost <= 100)
    @variable(model, Y[1:2], Logical)
    @constraint(model, production <= 12, Disjunct(Y[1]))
    @constraint(model, cost >= 5 + 2 * production, Disjunct(Y[1]))
    @constraint(model, production <= 20, Disjunct(Y[2]))
    @constraint(model, cost >= 12 + production, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, production >= 10)
    @objective(model, Min, cost)
    return model
end

model = production_model()
optimize!(model, gdp_method = BigM())
objective_value(model)
```

Six methods are available.

| Method | Extra solver needed | Relaxation strength | Program size |
|---|---|---|---|
| [`BigM`](@ref) | no | weakest | smallest |
| [`MBM`](@ref) | yes | tighter than `BigM` | same as `BigM` |
| [`PSplit`](@ref) | no | between `BigM` and `Hull` | tunable |
| [`Hull`](@ref) | no | tightest of the algebraic methods | largest |
| [`CuttingPlanes`](@ref) | yes | strengthens a base method | grows with iterations |
| [`Indicator`](@ref) | no, but solver must support it | solver-dependent | smallest |

!!! tip
    Start with `BigM` while building a model, because it is the cheapest to
    construct and the easiest to read when printed. Move to `Hull` or `MBM` once
    the model is correct and solve time becomes the constraint.

## Big-M

The big-M reformulation relaxes each disjunct constraint by an amount large
enough to make it vacuous when its disjunct is not selected. For a constraint
``r(x) \leq 0`` governed by indicator ``Y`` with binary ``y``, it produces
``r(x) \leq M(1 - y)``.

The method takes the value of ``M`` and a flag controlling whether tighter
values are derived from variable bounds:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = BigM())
objective_value(model)
```

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = BigM(1e4))
objective_value(model)
```

By default the method attempts to tighten the supplied value using the bounds of
the variables appearing in each constraint. Tightening can be disabled, which is
occasionally useful for reproducing a specific formulation:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = BigM(1e4, false))
objective_value(model)
```

!!! warning
    A big-M value that is too small silently removes valid solutions, and one
    that is far too large produces a relaxation so weak that branch and bound
    makes little progress. Bound every variable that appears in a disjunct
    constraint so that tightening has something to work with.

## Multiple Big-M

Big-M applies one value per constraint. Multiple big-M computes a separate value
for each constraint-disjunct pair, by maximizing the constraint's left-hand side
over the region defined by the *other* disjunct. The values obtained are no
weaker than a single big-M and usually a good deal tighter, at the cost of
solving one small optimization problem per pair.

Because those subproblems must be solved, the method requires an optimizer:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = MBM(HiGHS.Optimizer))
objective_value(model)
```

A fallback value is used for any pair whose subproblem does not yield a finite
bound, and it may be set as the second argument:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = MBM(HiGHS.Optimizer, 1e6))
objective_value(model)
```

The resulting program has the same number of variables and constraints as the
big-M program. The improvement is entirely in the coefficients, which makes this
an inexpensive substitution wherever big-M would otherwise be used.

## Convex Hull

The hull reformulation disaggregates every variable appearing in a disjunction
into one copy per disjunct, links the copies to the original by a summation
constraint, and scales each disjunct's constraints by its binary variable. For
disjuncts described by convex constraints, the continuous relaxation it produces
is the convex hull of the disjunction, which is the tightest relaxation
available. For nonconvex disjuncts it gives the hull of the convexified
disjuncts, which is still far tighter than big-M but no longer exact.

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = Hull())
objective_value(model)
```

The price is size. The disaggregated program has substantially more variables
and constraints:

```@example gdp_methods
big_m = production_model()
hull = production_model()
reformulate_model(big_m, BigM())
reformulate_model(hull, Hull())

(bigm = num_variables(big_m), hull = num_variables(hull))
```

Nonlinear disjunct constraints require a perspective function, which is singular
when the binary variable is zero. The method's parameter is the epsilon used to
regularize it:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = Hull(1e-8))
objective_value(model)
```

!!! note
    A smaller epsilon gives a more faithful perspective reformulation but a
    worse-conditioned problem. The default of `1e-6` is a reasonable compromise
    for most models, and only needs revisiting when a nonlinear solver reports
    numerical trouble.

## P-Split

P-split sits between big-M and hull. Rather than disaggregating every variable,
it partitions the variables into groups and disaggregates at the level of
groups. The number of groups is a dial between formulation strength and
formulation size: more groups give a stronger relaxation and a larger program.

The two ends of the dial are known. A single group gives a formulation whose
continuous relaxation admits the same set of feasible original variables as
big-M. Splitting on every variable gives the convex hull of the disjunction,
provided the disjunct constraints are affine and every variable is bounded.

!!! note
    The hull guarantee at the finest partition holds for affine disjunct
    constraints over a bounded box. When a disjunct contains nonlinear
    constraints, no P-split formulation is guaranteed to recover the true convex
    hull of the disjunction, though finer partitions still tighten the
    relaxation.

A partition may be given explicitly as a vector of variable vectors:

```@example gdp_methods
model = production_model()
partition = [[model[:production]], [model[:cost]]]
optimize!(model, gdp_method = PSplit(partition))
objective_value(model)
```

Alternatively a number of groups may be given, and the variables are divided
among them automatically:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = PSplit(2, model))
objective_value(model)
```

## Cutting Planes

The cutting planes method does not produce a formulation on its own. It solves a
sequence of separation problems, each of which yields a cut that tightens the
relaxation, and then applies a final reformulation to the strengthened model.
The cuts are valid for the hull, so the effect is to approach hull strength
while keeping a program closer to big-M in size.

An optimizer is required for the separation problems:

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = CuttingPlanes(HiGHS.Optimizer))
objective_value(model)
```

The number of separation rounds, the tolerance below which a cut is considered
unhelpful, the reformulation applied at the end, and the big-M value used by it
are all adjustable:

```@example gdp_methods
model = production_model()
method = CuttingPlanes(
    HiGHS.Optimizer;
    max_iter = 10,
    seperation_tolerance = 1e-8,
    final_reform_method = BigM(),
    M_value = 1e5
)
optimize!(model, gdp_method = method)
objective_value(model)
```

!!! note
    Each round adds constraints permanently to the model, so a large `max_iter`
    trades a tighter relaxation for a larger program and a longer setup time.
    The default of three rounds captures most of the available tightening on
    typical models.

## Indicator Constraints

Rather than encoding the disjunction algebraically, this method passes it to the
solver as indicator constraints, leaving the solver to handle the implication
internally. Modern mixed-integer solvers implement these natively and can often
propagate them more effectively than any big-M encoding.

```@example gdp_methods
model = production_model()
optimize!(model, gdp_method = Indicator())
objective_value(model)
```

!!! warning
    Indicator constraints are supported only for linear disjunct constraints,
    and only by solvers that implement them. A solver without support will
    reject the model rather than fall back to another encoding.

## Reformulating Without Solving

[`reformulate_model`](@ref) applies a method and stops, which is the way to
inspect or export the mixed-integer program that a method produces:

```@example gdp_methods
model = production_model()
reformulate_model(model, BigM())
num_variables(model), num_constraints(model, count_variable_in_set_constraints = true)
```

Once reformulated, the model is an ordinary JuMP model and every JuMP facility
applies to it, including `write_to_file` for exporting and `print` for reading
the formulation directly.

## Next Steps

- [GDP Models](@ref model_guide) covers model construction and result queries.
- [Disjunctions](@ref constraints_guide) covers the structure these methods
  operate on.
