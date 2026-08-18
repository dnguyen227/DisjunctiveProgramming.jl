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
    sample_M_values(gp, objectives, sub, method, support_grids)

Compute the MBM M values at the transcription supports of an infinite
model. `gp` is the `gp` field of [`MBM`](@ref), `objectives` is the
array of per-support objective expressions, `sub` is the transcribed
submodel wrapped as a `GDPSubmodel`, `method` is the `_MBM` data
(which carries the sampling settings `kappa`, `budget`, `min_solves`,
`detect_uniform_M`, `lengthscales`, `jitter`, `n_seeds`, and `seeds`),
and `support_grids` is a function returning the
support vectors of the infinite parameters. It is a function because
the supports are only well defined once M is known to vary over them,
so methods that return early (or never need coordinates) must not
call it. Returns an array of M values shaped like `objectives`, a
scalar when M is uniform across the supports, or `nothing` if an M
subproblem is infeasible. Extensions implement methods that dispatch
on `gp`: `nothing` solves an M subproblem at every support, while an
`AbstractGPs.AbstractGP` or a `KernelFunctions.Kernel` solves a
subset of the supports selected by an upper-confidence-bound
acquisition and fills the rest with the posterior upper confidence
bound `mean + kappa * sd`. The filled values are heuristic upper
estimates of the exact M values, not certificates.
"""
function sample_M_values(gp, objectives, sub, method, support_grids)
    error("Unrecognized `gp` value `$(repr(gp))` for MBM on an " *
          "infinite model. Use `nothing` to solve an M subproblem " *
          "at every support, or load AbstractGPs and pass an " *
          "`AbstractGPs.AbstractGP` or a `KernelFunctions.Kernel` " *
          "to estimate M values with a Gaussian process.")
end
