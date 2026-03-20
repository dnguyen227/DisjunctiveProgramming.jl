"""
    InfiniteGDPModel(args...; kwargs...)

Creates an `InfiniteOpt.InfiniteModel` that is compatible with the
capabiltiies provided by DisjunctiveProgramming.jl. This requires
that InfiniteOpt be imported first.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt

julia> InfiniteGDPModel()

```
"""
function InfiniteGDPModel end

"""
    InfiniteLogical(prefs...)

Allows users to create infinite logical variables. This is a tag
for the `@variable` macro that is a combination of `InfiniteOpt.Infinite`
and `DisjunctiveProgramming.Logical`. This requires that InfiniteOpt be
first imported.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt

julia> model = InfiniteGDPModel();

julia> @infinite_parameter(model, t in [0, 1]);

julia> @infinite_parameter(model, x[1:2] in [-1, 1]);

julia> @variable(model, Y, InfiniteLogical(t, x)) # creates Y(t, x) in {True, False}
Y(t, x)
```
"""
function InfiniteLogical end

"""
    transcribe(
        model::InfiniteOpt.InfiniteModel;
        [method::AbstractReformulationMethod]
    )::NamedTuple

Transcribe an `InfiniteModel` into a flat `JuMP.Model` and
build forward variable mappings. Requires InfiniteOpt.

## Returns
A `NamedTuple` with fields:
- `inf_model`: the original `InfiniteModel`
- `flat_model`: independent flat `JuMP.Model` copy
- `fwd`: `Dict{var, Vector{JuMP.VariableRef}}` mapping
  infinite variables to flat transcribed variables
"""
function transcribe end

"""
    jump_model(tmap)::JuMP.Model

Return the flat model from a transcription result.
"""
function jump_model end

"""
    transcribed_variable(tmap, var)

Return the transcribed variable(s) corresponding to
an infinite variable. Returns a `Vector{JuMP.VariableRef}`
flattened in column-major order (length = product of
support sizes for the variable's parameters; length 1
for finite vars).
"""
function transcribed_variable end

"""
    infinite_variable(
        tmap,
        flat_var::JuMP.VariableRef
    )::Tuple{var, Int}

Return the original infinite variable and support index
corresponding to a transcribed variable. Finite
variables have support index 1.

**Note**: This performs an O(n) scan over the forward
map. Intended for debugging and testing, not hot paths.
"""
function infinite_variable end

"""
    support_values(tmap)::Dict{var, Vector{Float64}}

Return the support point values. Returns a dictionary
mapping each infinite parameter reference to its
vector of support values.
"""
function support_values end

"""
    lift_constraint(
        tmap,
        terms::Vector{Tuple{var, Vector{Float64}}},
        sense::Symbol,
        rhs::Vector{Float64}
    )

Add a constraint to both the flat model and the
InfiniteModel. Coefficients at support points are
automatically interpolated into `@parameter_function`
for the infinite constraint.

## Arguments
- `terms`: pairs of `(infinite_var, coeff_vector)` where
  `coeff_vector` has length K = product of all parameter
  support sizes. Entries are ordered by the flattened
  column-major Cartesian product of supports.
- `sense`: `:>=`, `:<=`, or `:==`.
- `rhs`: right-hand side values at each support point
  (length K).
"""
function lift_constraint end
