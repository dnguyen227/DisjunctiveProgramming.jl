# GDP Models

A guide for creating and working with generalized disjunctive programming models.

## Overview

Generalized Disjunctive Programming (GDP) models extend traditional optimization by incorporating logical decision-making through disjunctions. A disjunction represents an "either-or" choice between different sets of constraints, where exactly one alternative (disjunct) can be active at a time.

DisjunctiveProgramming.jl provides the `GDPModel` type, which extends JuMP's modeling capabilities to handle logical variables, disjunctive constraints, and various reformulation methods for solving GDP problems.

## Basic Usage

### Creating a GDP Model

GDP models are created using the `GDPModel()` constructor, similar to JuMP's `Model()`:

```julia
using DisjunctiveProgramming
using HiGHS  # or any other supported optimizer

# Create a basic GDP model
model = GDPModel()

# Create a GDP model with an optimizer
model = GDPModel(HiGHS.Optimizer)
```

### Setting an Optimizer

You can set or change the optimizer using JuMP's standard methods:

```julia
model = GDPModel()
set_optimizer(model, HiGHS.Optimizer)

# Or with optimizer attributes
set_optimizer_with_attributes(model, HiGHS.Optimizer, "presolve" => "on")
```

### Model Information

You can check if a model is a GDP model and access its data:

```julia
# Check if a model is a GDP model
is_gdp_model(model)  # returns true

# Access the GDP-specific data
data = gdp_data(model)
```

## Example: Simple GDP Model

Let's create a simple GDP model with a choice between two production processes:

```julia
using DisjunctiveProgramming
using HiGHS

# Create the model
model = GDPModel(HiGHS.Optimizer)

# Regular variables
@variable(model, x >= 0)  # production amount
@variable(model, cost >= 0)  # total cost

# Logical variables for process selection
@variable(model, Y[1:2], Logical)

# Disjunctive constraints
@constraint(model, [i=1], x <= 100, Disjunct(Y[1]))  # Process 1 capacity
@constraint(model, [i=1], cost == 2*x, Disjunct(Y[1]))  # Process 1 cost

@constraint(model, [i=2], x <= 80, Disjunct(Y[2]))   # Process 2 capacity  
@constraint(model, [i=2], cost == 1.5*x, Disjunct(Y[2]))  # Process 2 cost

# Create the disjunction (exactly one process must be selected)
@disjunction(model, [Y[1], Y[2]])

# Objective
@objective(model, Min, cost)

# Solve
optimize!(model)

# Check results
println("Optimal cost: ", objective_value(model))
println("Production: ", value(x))
println("Process 1 selected: ", value(Y[1]))
println("Process 2 selected: ", value(Y[2]))
```

## Solution Methods

DisjunctiveProgramming.jl supports multiple reformulation methods to solve GDP problems:

### Big-M Method

The Big-M method is the default approach:

```julia
# Solve with Big-M (default)
optimize!(model)

# Explicitly specify Big-M with custom parameters
optimize!(model, gdp_method = BigM(1e6, true))
```

### Convex Hull Method

The Hull method provides tighter relaxations:

```julia
# Solve with convex hull reformulation
optimize!(model, gdp_method = Hull())

# With custom epsilon for nonlinear constraints
optimize!(model, gdp_method = Hull(1e-8))
```

### Indicator Constraints

For linear problems with supporting solvers:

```julia
# Solve with indicator constraints
optimize!(model, gdp_method = Indicator())
```

## Model Queries

You can query various aspects of your GDP model:

```julia
# Check if model is ready to optimize
_ready_to_optimize(model)

# Get the current solution method
_solution_method(model)

# Access logical variables
logical_vars = _logical_variables(model)

# Access disjunctions
disjunctions = _disjunctions(model)
```

## Next Steps

- Learn about [Logical Variables](variables.md) for modeling binary decisions
- Explore [Constraints](constraints.md) for building disjunctive constraints
- See [Solution Methods](methods.md) for detailed reformulation options
