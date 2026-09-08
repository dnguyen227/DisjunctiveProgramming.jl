# [Logical Constraints](@id logic_guide)

A guide for propositional and cardinality constraints over logical variables.
See the [API](@ref) for the technical details of each method.

## Overview

Disjunctions express which constraints apply. Logical constraints express which
*combinations of decisions* are permitted, independently of any algebraic
constraint. A logical constraint relates logical variables to one another, and
comes in two forms.

A *proposition* is a Boolean expression asserted to be true, such as requiring
that selecting one process implies selecting a quality check. A *cardinality
constraint* requires that a given number of logical variables from a collection
be true, such as selecting exactly two suppliers out of five.

Both forms are reformulated into linear constraints on the binary variables that
replace the logical variables. Propositions are first converted to conjunctive
normal form, which yields one linear inequality per clause.

## Logical Operators

Five operators are available. Each has a Unicode form, a written alias, and in
most cases an ASCII operator that behaves identically inside a logical
expression.

| Operation | Unicode | Alias | ASCII |
|---|---|---|---|
| Conjunction | `∧` | `logical_and` | `&&` |
| Disjunction | `∨` | `logical_or` | `\|\|` |
| Negation | `¬` | `logical_not` | `!` |
| Implication | `⟹` | `implies` | |
| Equivalence | `⇔` | `iff` | |

The Unicode symbols are entered in the Julia REPL and in most editors by typing
the LaTeX name followed by tab: `\wedge` for `∧`, `\vee` for `∨`, `\neg` for
`¬`, `\Longrightarrow` for `⟹`, and `\Leftrightarrow` for `⇔`.

## Basic Usage

A proposition is written as a Boolean expression followed by `:= true` inside
`@constraint`:

```@example gdp_logic
using DisjunctiveProgramming

model = GDPModel()
@variable(model, process_a, Logical)
@variable(model, process_b, Logical)
@variable(model, quality_check, Logical)
@variable(model, premium_grade, Logical)

@constraint(model, (process_a ∨ process_b) ⟹ quality_check := true)
@constraint(model, premium_grade ⟹ (process_a ∧ quality_check) := true)
@constraint(model, ¬(process_a ∧ process_b) := true)
nothing # hide
```

The written aliases and ASCII operators produce the same constraints, and are
useful when a source file is meant to stay in plain ASCII:

```@example gdp_logic
@constraint(model, logical_or(process_a, process_b) := true)
@constraint(model, (process_a && quality_check) := true)
nothing # hide
```

!!! warning
    A proposition must relate more than one variable. Asserting a single
    variable, as in `@constraint(model, process_a := true)` or
    `@constraint(model, ¬process_a := true)`, is an error. Use
    `fix(process_a, true)` or `fix(process_a, false)` instead, which expresses
    the same thing without adding a constraint to the reformulated program.

### Cardinality Constraints

[`Exactly`](@ref), [`AtLeast`](@ref), and [`AtMost`](@ref) constrain how many
variables in a collection are true. They are used as JuMP sets, with the
collection on the left of `in`:

```@example gdp_logic
selection = GDPModel()
@variable(selection, suppliers[1:5], Logical)

@constraint(selection, suppliers in Exactly(2))
@constraint(selection, suppliers in AtLeast(1))
@constraint(selection, suppliers in AtMost(3))
nothing # hide
```

The count may itself be a logical variable rather than an integer, which makes
the requirement conditional on another decision:

```@example gdp_logic
@variable(selection, expand, Logical)
@constraint(selection, suppliers in AtLeast(expand))
nothing # hide
```

Here at least one supplier is required when `expand` is true, and no requirement
is imposed when it is false.

!!! note
    Every disjunction added with [`@disjunction`](@ref) already carries an
    exclusivity requirement equivalent to `Exactly(1)` over its disjuncts, added
    automatically. You do not need to write one yourself, and the behaviour can
    be turned off with the `exactly1` keyword when a disjunction should permit
    more than one disjunct to be selected.

## Building Larger Expressions

Operators nest, so a proposition can describe a structured requirement in one
constraint:

```@example gdp_logic
nested = GDPModel()
@variable(nested, X[1:4], Logical)

@constraint(nested, ((X[1] ∨ X[2]) ∧ (X[3] ∨ X[4])) ⟹ (X[1] ⇔ X[3]) := true)
nothing # hide
```

Because the reformulation converts each proposition to conjunctive normal form,
a deeply nested expression may expand into a considerable number of linear
constraints. Where a requirement can be stated either as one large proposition
or as several small ones, the several small ones usually produce a tighter and
smaller program.

Splatting works for requirements over a whole container:

```@example gdp_logic
@constraint(nested, logical_or(X...) := true)
nothing # hide
```

## Next Steps

- [Disjunctions](@ref constraints_guide) covers attaching algebraic constraints
  to logical variables.
- [Solution Methods](@ref methods_guide) covers how these constraints are
  encoded in the mixed-integer program.
