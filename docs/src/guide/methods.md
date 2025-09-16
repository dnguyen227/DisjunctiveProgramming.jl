# Solution Methods

A guide for solution methods and reformulation techniques in DisjunctiveProgramming.jl.

## Overview

DisjunctiveProgramming.jl automatically reformulates GDP models into mixed-integer programming (MIP) problems that can be solved by standard MIP solvers. Three main reformulation methods are available:

- **Big-M Method**: Uses large constants to deactivate constraints
- **Convex Hull Method**: Creates tighter linear relaxations through variable disaggregation
- **Indicator Constraints**: Uses solver-native indicator constraints (when supported)

## Basic Usage

### Default Behavior

By default, DisjunctiveProgramming.jl uses the Big-M method:

```julia
using DisjunctiveProgramming
using HiGHS

model = GDPModel(HiGHS.Optimizer)

# ... build your model ...

# Solve with default Big-M method
optimize!(model)
```

### Specifying Solution Methods

You can explicitly choose the solution method:

```julia
# Big-M method with custom parameters
optimize!(model, gdp_method = BigM(1e6, tighten = true))

# Convex Hull method
optimize!(model, gdp_method = Hull())

# Indicator constraints
optimize!(model, gdp_method = Indicator())
```

## Big-M Method

The Big-M method is the most widely applicable reformulation technique.

### How It Works

For a disjunctive constraint `f(x) ≤ 0` in disjunct `Y`, the Big-M method creates:
```
f(x) ≤ M(1 - Y)
```
where `M` is a sufficiently large constant.

### Usage

```julia
# Default Big-M
optimize!(model, gdp_method = BigM())

# Custom Big-M value
optimize!(model, gdp_method = BigM(1e8))

# With bound tightening (default: true)
optimize!(model, gdp_method = BigM(1e6, tighten = true))

# Without bound tightening
optimize!(model, gdp_method = BigM(1e6, tighten = false))
```

### Choosing Big-M Values

The Big-M value should be:
- Large enough that constraints are effectively deactivated when `Y = 0`
- Small enough to avoid numerical issues

```julia
# For problems with bounded variables, estimate reasonable M values
@variable(model, 0 <= x <= 100)
@variable(model, 0 <= y <= 50)

# For constraint x + y <= 10, M = 150 would be sufficient
optimize!(model, gdp_method = BigM(150))
```

### Example: Big-M Method

```julia
using DisjunctiveProgramming
using HiGHS

model = GDPModel(HiGHS.Optimizer)

@variable(model, 0 <= x <= 100)
@variable(model, 0 <= y <= 100)
@variable(model, Y[1:2], Logical)

# Disjunctive constraints
@constraint(model, x + y <= 50, Disjunct(Y[1]))
@constraint(model, x - y >= 10, Disjunct(Y[2]))

@disjunction(model, [Y[1], Y[2]])
@objective(model, Max, x + y)

# Solve with Big-M
optimize!(model, gdp_method = BigM(200, tighten = true))

println("Optimal value: ", objective_value(model))
println("x = ", value(x), ", y = ", value(y))
```

## Convex Hull Method

The Hull method provides the tightest linear relaxation but requires more variables and constraints.

### How It Works

For each variable `x` in a disjunction, the Hull method creates disaggregated variables `x_k` for each disjunct `k`, with:
```
x = Σ x_k
x_k ≤ M_k * Y_k
```

### Usage

```julia
# Default Hull method
optimize!(model, gdp_method = Hull())

# Custom epsilon for nonlinear constraints
optimize!(model, gdp_method = Hull(1e-8))
```

### When to Use Hull

The Hull method is particularly effective for:
- Problems with tight variable bounds
- Linear disjunctive constraints
- Cases where the linear relaxation quality is important

### Example: Hull Method

```julia
using DisjunctiveProgramming
using HiGHS

model = GDPModel(HiGHS.Optimizer)

@variable(model, 0 <= x <= 10)
@variable(model, 0 <= y <= 10)
@variable(model, Y[1:2], Logical)

# Disjunctive constraints
@constraint(model, x + 2y <= 15, Disjunct(Y[1]))
@constraint(model, 3x + y <= 18, Disjunct(Y[2]))

@disjunction(model, [Y[1], Y[2]])
@objective(model, Max, x + y)

# Solve with Hull method
optimize!(model, gdp_method = Hull())

println("Optimal value: ", objective_value(model))
println("x = ", value(x), ", y = ", value(y))
```

## Indicator Constraints

Indicator constraints use solver-native support for logical implications.

### How It Works

For a disjunctive constraint `f(x) ≤ 0` in disjunct `Y`, an indicator constraint directly represents:
```
Y = 1 → f(x) ≤ 0
```

### Usage

```julia
# Use indicator constraints (if solver supports them)
optimize!(model, gdp_method = Indicator())
```

### Example: Indicator Constraints

```julia
using DisjunctiveProgramming
using Gurobi  # Solver that supports indicators

model = GDPModel(Gurobi.Optimizer)

@variable(model, x >= 0)
@variable(model, y >= 0)
@variable(model, Y[1:2], Logical)

# Linear disjunctive constraints only
@constraint(model, x + y <= 10, Disjunct(Y[1]))
@constraint(model, 2x + y <= 12, Disjunct(Y[2]))

@disjunction(model, [Y[1], Y[2]])
@objective(model, Max, x + y)

# Solve with indicator constraints
optimize!(model, gdp_method = Indicator())
```
## Next Steps

- Learn about [Logical Operations](logic.md) for complex logical expressions
- Explore the [API Reference](../api.md) for detailed method documentation
