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
    sample_M_values(sampler, objectives, sub, method, support_grids)

Compute the MBM M values at the transcription supports of an infinite
model. `sampler` is the `sampler` field of [`MBM`](@ref),
`objectives` is the array of per-support objective expressions, `sub`
is the transcribed submodel wrapped as a `GDPSubmodel`, `method` is
the `_MBM` data, and `support_grids` is a function returning the
support vectors of the infinite parameters. It is a function because
the supports are only well defined once M is known to vary over them,
so methods that return early (or never need coordinates) must not
call it. Returns an array of M values shaped like `objectives`, a
scalar when M is uniform across the supports, or `nothing` if an M
subproblem is infeasible. Extensions implement methods that dispatch
on `sampler`: `nothing` solves an M subproblem at every support,
while a [`GPSampler`](@ref) solves a subset of the supports and
fills the rest with a Gaussian-process upper confidence bound.
"""
function sample_M_values(sampler, objectives, sub, method, support_grids)
    error("Unrecognized `sampler` value `$(repr(sampler))` for MBM " *
          "on an infinite model. Use `nothing` to solve an M " *
          "subproblem at every support, or a `GPSampler` (with " *
          "AbstractGPs loaded) to estimate M values with a " *
          "Gaussian process.")
end
