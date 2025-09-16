# Logical Operations

A guide for logical operations and expressions in DisjunctiveProgramming.jl.

## Overview

DisjunctiveProgramming.jl supports rich logical expressions that allow you to model complex logical relationships between decisions. These expressions use logical variables and operators to create constraints that capture the logical structure of your problem.

## Logical Operators

### Basic Operators

DisjunctiveProgramming.jl provides several logical operators:

| Operator | Symbol | Unicode | Description |
|----------|--------|---------|-------------|
| AND | `&&` | `∧` | Logical conjunction |
| OR | `\|\|` | `∨` | Logical disjunction |
| NOT | `!` | `¬` | Logical negation |
| IMPLIES | | `⟹` | Logical implication |
| IFF | | `⇔` | Logical equivalence (if and only if) |

### Using Operators

```julia
using DisjunctiveProgramming

model = GDPModel()

@variable(model, A, Logical)
@variable(model, B, Logical)
@variable(model, C, Logical)

# AND operation
@constraint(model, A ∧ B := true)

# OR operation  
@constraint(model, A ∨ B := true)

# NOT operation
@constraint(model, ¬A := true)

# IMPLIES operation
@constraint(model, A ⟹ B := true)

# IFF (equivalence) operation
@constraint(model, A ⇔ B := true)
```

## Building Logical Expressions

### Simple Expressions

```julia
@variable(model, use_machine[1:3], Logical)

# At least one machine must be used
@constraint(model, use_machine[1] ∨ use_machine[2] ∨ use_machine[3] := true)

# Machines 1 and 2 cannot be used together
@constraint(model, ¬(use_machine[1] ∧ use_machine[2]) := true)
```

### Complex Expressions

```julia
@variable(model, process_A, Logical)
@variable(model, process_B, Logical)
@variable(model, quality_check, Logical)
@variable(model, premium_grade, Logical)

# If using process A or B, quality check is required
@constraint(model, (process_A ∨ process_B) ⟹ quality_check := true)

# Premium grade requires process A and quality check
@constraint(model, premium_grade ⟹ (process_A ∧ quality_check) := true)

# Process A and B are mutually exclusive
@constraint(model, ¬(process_A ∧ process_B) := true)
```

### Nested Expressions

```julia
@variable(model, X[1:4], Logical)

# Complex nested logical expression
@constraint(model, ((X[1] ∨ X[2]) ∧ (X[3] ∨ X[4])) ⟹ (X[1] ⇔ X[3]) := true)
```
## Next Steps

- Learn about [Solution Methods](methods.md) for understanding how logical expressions are reformulated
- See the [API Reference](../api.md) for complete function documentation
