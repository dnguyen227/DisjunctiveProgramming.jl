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
    GPSampler(; kappa = 2.5, budget = 0.25, min_solves = 6,
              kernel = nothing, detect_uniform_M = true)

Creates a Gaussian-process M sampler for [`MBM`](@ref) on infinite
models. Instead of solving an M subproblem at every support of the
infinite parameters, the sampler solves a subset of the supports
selected by an upper-confidence-bound acquisition and fills the
remaining supports with the posterior upper confidence bound
`mean + kappa * sd`. The filled values are heuristic upper estimates
of the exact M values, not certificates; increase `kappa` for more
conservative estimates or use `MBM(...; M_sampler = :exact)` to solve
every support. This requires that InfiniteOpt and AbstractGPs be
imported first, in which case it is also the default M sampler (see
the `M_sampler` field of [`MBM`](@ref)).

**Keyword Arguments**
- `kappa::Real`: Upper-confidence-bound multiplier used to select the
  next support to solve and to fill unsolved supports (2.5).
- `budget::Real`: Fraction of the supports to solve exactly, in
  `(0, 1]` (0.25).
- `min_solves::Int`: Minimum number of exactly solved supports (6).
- `kernel`: Covariance kernel for the GP fit. Defaults to a squared
  exponential kernel whose lengthscale is selected by maximizing the
  marginal likelihood; pass a `KernelFunctions` kernel to override.
- `detect_uniform_M::Bool`: If `true` (the default), M values that
  agree at the first few supports are taken to be uniform and used
  for every support. This is cheap and is what makes infinite
  parameters without a support grid (e.g. dependent ones) workable,
  but it assumes M does not vary elsewhere. Set it to `false` to
  always fit the GP, which leaves the usual `kappa * sd` cushion on
  the unsolved supports at the cost of the extra solves, and which
  requires that every infinite parameter have a support grid.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt, AbstractGPs, HiGHS

julia> method = MBM(HiGHS.Optimizer, M_sampler = GPSampler(kappa = 4.0))
```
"""
function GPSampler end

"""
    sample_M_values(sampler, objectives, sub, method, support_grids)

Compute the MBM M values at the transcription supports of an infinite
model. `objectives` is the array of per-support objective expressions,
`sub` is the transcribed submodel wrapped as a `GDPSubmodel`, `method`
is the `_MBM` data, and `support_grids` is a function returning the
support vectors of the infinite parameters. It is a function because
the supports are only well defined once M is known to vary over them,
so samplers that return early (or never need coordinates) must not
call it. Returns an array of M values shaped like
`objectives`, a scalar when M is uniform across the supports, or
`nothing` if an M subproblem is infeasible. Extensions implement
methods for their sampler types (e.g. [`GPSampler`](@ref)); the
`:exact` sampler solves every support.
"""
function sample_M_values end
