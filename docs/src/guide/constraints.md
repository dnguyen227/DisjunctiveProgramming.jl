# Constraints

A guide for creating and working with disjunctive constraints.

## Overview

DisjunctiveProgramming.jl supports several types of constraints that work together to model complex logical decision-making problems:

- **Disjunctive constraints**: Regular constraints that are active only when their associated logical variable is true
- **Disjunctions**: Collections of disjuncts where exactly one must be selected
- **Logical constraints**: Constraints involving logical expressions and operators
- **Cardinality constraints**: Constraints that control how many logical variables can be true

## Basic Usage

### Disjunctive Constraints

Disjunctive constraints are regular JuMP constraints tagged with a `Disjunct` to associate them with a logical variable:

```julia
using DisjunctiveProgramming
using HiGHS

model = GDPModel(HiGHS.Optimizer)

# Variables
@variable(model, x >= 0)
@variable(model, y >= 0)
@variable(model, Y[1:2], Logical)

# Disjunctive constraints
@constraint(model, x + y <= 10, Disjunct(Y[1]))
@constraint(model, 2x + y <= 15, Disjunct(Y[1]))

@constraint(model, x + 3y <= 12, Disjunct(Y[2]))
@constraint(model, x - y >= 2, Disjunct(Y[2]))
```

### Creating Disjunctions

Disjunctions group logical variables where exactly one must be true:

```julia
# Create a disjunction using the macro
@disjunction(model, [Y[1], Y[2]])

# Or using the function
disjunction(model, [Y[1], Y[2]])
```

### Multiple Disjunctions

You can create multiple disjunctions at once:

```julia
@variable(model, Z[1:3, 1:2], Logical)

@disjunctions(model, begin
    [Z[1,1], Z[1,2]]  # First disjunction
    [Z[2,1], Z[2,2]]  # Second disjunction  
    [Z[3,1], Z[3,2]]  # Third disjunction
end)
```

## Logical Constraints

### Cardinality Constraints

Control how many logical variables can be true:

```julia
@variable(model, select[1:5], Logical)

# Exactly 2 must be selected
@constraint(model, select in Exactly(2))

# At least 1 must be selected
@constraint(model, select in AtLeast(1))

# At most 3 can be selected
@constraint(model, select in AtMost(3))
```

### Logical Expressions

Build complex logical relationships:

```julia
@variable(model, A, Logical)
@variable(model, B, Logical) 
@variable(model, C, Logical)

# Logical AND: A ∧ B ⟹ C
@constraint(model, (A ∧ B) ⟹ C := true)

# Logical OR: A ∨ B
@constraint(model, A ∨ B := true)

# Logical equivalence: A ⇔ B
@constraint(model, A ⇔ B := true)

# Negation: ¬A
@constraint(model, ¬A := true)
```


## Nested Disjunctions

Create hierarchical decision structures:

```julia
# Main choice: Build factory or outsource
@variable(model, build_factory, Logical)
@variable(model, outsource, Logical, logical_complement = build_factory)

# If building factory, choose size
@variable(model, small_factory, Logical)
@variable(model, large_factory, Logical)

# Factory size constraints
@constraint(model, capacity <= 100, Disjunct(small_factory))
@constraint(model, capacity <= 500, Disjunct(large_factory))

# Nested disjunction: Factory size choice is only relevant if building
@disjunction(model, [small_factory, large_factory], Disjunct(build_factory))

# Main disjunction
@disjunction(model, [build_factory, outsource])
```

## Next Steps

- Learn about [Solution Methods](methods.md) to understand how constraints are reformulated
- Explore [Logical Operations](logic.md) for complex logical relationships
