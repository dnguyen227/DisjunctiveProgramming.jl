# Logical Variables

A guide for working with logical variables in DisjunctiveProgramming.jl.

## Overview

Logical variables are the foundation of generalized disjunctive programming. They represent binary decisions that determine which disjuncts (sets of constraints) are active in a disjunction. Unlike regular binary variables, logical variables are specifically designed to work with disjunctive constraints and are automatically reformulated into appropriate binary variables during the solution process.

## Basic Usage

### Creating Logical Variables

Logical variables are created using the `@variable` macro with the `Logical` type:

```julia
using DisjunctiveProgramming

model = GDPModel()

# Single logical variable
@variable(model, Y, Logical)

# Array of logical variables
@variable(model, Z[1:3], Logical)

# Multi-dimensional arrays
@variable(model, W[1:2, 1:3], Logical)
```

### Logical Variable Properties

You can set various properties for logical variables:

```julia
# With a starting value
@variable(model, Y, Logical, start = true)

# With a fixed value  
@variable(model, Y, Logical, fix = false)
```

### Logical Complements

For disjunctions with exactly two disjuncts, you can create logical complement variables:

```julia
# Create the first logical variable
@variable(model, Y1, Logical)

# Create its complement (Y2 = 1 - Y1)
@variable(model, Y2, Logical, logical_complement = Y1)

# Now Y1 + Y2 = 1 is automatically enforced
```

## Working with Logical Variables

### Querying Logical Variables

```julia
# Check if a variable has a logical complement
has_logical_complement(Y2)  # returns true

# Get the underlying binary variable
binary_var = binary_variable(Y1)

# Check if fixed
is_fixed(Y1)

# Get fixed value (if any)
if is_fixed(Y1)
    fixed_val = fix_value(Y1)
end

# Get start value
start_val = start_value(Y1)
```

### Modifying Logical Variables

```julia
# Set start value
set_start_value(Y1, true)

# Fix a logical variable
fix(Y1, false)

# Unfix a logical variable
unfix(Y1)

# Set name
set_name(Y1, "use_process_1")
```

### Getting Solution Values

After solving, you can query the values of logical variables:

```julia
optimize!(model)

# Get the boolean value
y1_value = value(Y1)  # returns true or false
y2_value = value(Y2)
```

## Example: Production Planning

Here's a complete example showing logical variables in a production planning context:

```julia
using DisjunctiveProgramming
using HiGHS

# Create model
model = GDPModel(HiGHS.Optimizer)

# Decision variables
@variable(model, production >= 0)
@variable(model, cost >= 0)

# Logical variables for facility selection
@variable(model, use_facility[1:3], Logical)

# Disjunctive constraints for each facility
# Facility 1: Low capacity, low cost
@constraint(model, production <= 50, Disjunct(use_facility[1]))
@constraint(model, cost >= 10 + 2*production, Disjunct(use_facility[1]))

# Facility 2: Medium capacity, medium cost  
@constraint(model, production <= 100, Disjunct(use_facility[2]))
@constraint(model, cost >= 20 + 1.5*production, Disjunct(use_facility[2]))

# Facility 3: High capacity, high cost
@constraint(model, production <= 200, Disjunct(use_facility[3]))
@constraint(model, cost >= 50 + 1*production, Disjunct(use_facility[3]))

# Exactly one facility must be selected
@disjunction(model, use_facility)

# Demand constraint
@constraint(model, production >= 75)

# Objective: minimize cost
@objective(model, Min, cost)

# Solve
optimize!(model)

# Display results
println("Optimal production: ", value(production))
println("Optimal cost: ", value(cost))
for i in 1:3
    if value(use_facility[i])
        println("Selected facility: ", i)
    end
end
```

## Logical Variable Tags

You can associate tags with logical variables to control how they are reformulated:

```julia
# Define a custom tag type
struct MyCustomTag end

# Create logical variable with tag
@variable(model, Y, Logical(MyCustomTag()))

# The tag will be passed to the reformulated binary variable
```

## Advanced Usage

### Nested Logical Variables

Logical variables can be used in nested disjunctions:

```julia
# Main process selection
@variable(model, main_process[1:2], Logical)

# Sub-process selection within each main process
@variable(model, sub_process[1:2, 1:3], Logical)

# Main disjunction
@disjunction(model, main_process)

# Nested disjunctions
for i in 1:2
    @disjunction(model, sub_process[i, :], Disjunct(main_process[i]))
end
```

### Integration with Regular Variables

Logical variables can be used alongside regular JuMP variables:

```julia
# Regular variables
@variable(model, x >= 0)
@variable(model, y, Bin)  # Regular binary variable

# Logical variables
@variable(model, Z, Logical)

# Mix in constraints (using binary_variable to access the binary form)
@constraint(model, x + binary_variable(Z) <= 10)
```

## Next Steps

- Learn about [Constraints](constraints.md) for building disjunctive constraints
- Explore [Logical Operations](logic.md) for combining logical variables
- See [Solution Methods](methods.md) for optimization approaches and how logical variables are reformulated
