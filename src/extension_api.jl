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
    GPSampler(
        kernel = nothing;
        kappa::Real = 2.5,
        budget::Real = 0.25,
        detect_uniform_M::Bool = true,
        lengthscales = [0.05, 0.1, 0.2, 0.4, 0.8],
        jitter::Real = 1e-8,
        seeds = 4
        )

A Gaussian-process M sampler for the `sampler` field of [`MBM`](@ref)
on infinite models. Instead of solving an M subproblem at every
support, it solves the seed supports, then the supports selected by
an upper-confidence-bound acquisition until the budget is spent, and
fills the remaining supports with the posterior upper confidence
bound `mean + kappa * sd`. The filled values are heuristic upper
estimates of the exact M values, not certificates. Using it requires
that AbstractGPs be loaded.

**Arguments**
- `kernel`: Covariance kernel for the GP fit; `nothing` (the
  default) uses a squared exponential kernel. The lengthscale is
  selected from `lengthscales` by marginal likelihood.
- `kappa::Real`: Upper-confidence-bound multiplier used to select the
  next support to solve and to fill unsolved supports (2.5).
- `budget::Real`: Fraction of the supports to solve exactly, in
  `(0, 1]` (0.25). The seed supports are always solved;
  `budget = 1.0` solves every support.
- `detect_uniform_M::Bool`: If `true` (the default), M values that
  agree at the seed supports are taken to be uniform and used for
  every support. Set it to `false` to always fit the GP, which
  leaves the usual `kappa * sd` cushion on the unsolved supports at
  the cost of the extra solves.
- `lengthscales`: Candidate lengthscales for the marginal-likelihood
  kernel fit, relative to support coordinates normalized to
  `[0, 1]` ([0.05, 0.1, 0.2, 0.4, 0.8]). Give a single candidate to
  pin the lengthscale.
- `jitter::Real`: Observation-noise nugget added to the GP prior
  when fitting (1e-8).
- `seeds`: Number of evenly spaced supports solved before the first
  GP fit (4), or a vector of fractions in `[0, 1]` giving their
  positions along the supports. With `detect_uniform_M`, evenly
  spaced seeds can read a periodic M as uniform; pass unevenly
  spaced fractions to guard against that.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt, AbstractGPs, HiGHS

julia> method = MBM(HiGHS.Optimizer, sampler = GPSampler(kappa = 4.0))
```
"""
struct GPSampler{K} <: AbstractMBMSampler
    # Parametric so base needs no AbstractGPs dependency
    kernel::K
    kappa::Float64
    budget::Float64
    detect_uniform_M::Bool
    lengthscales::Vector{Float64}
    jitter::Float64
    seeds::Union{Int, Vector{Float64}}

    function GPSampler(
        kernel::K = nothing;
        kappa::Real = 2.5,
        budget::Real = 0.25,
        detect_uniform_M::Bool = true,
        lengthscales = [0.05, 0.1, 0.2, 0.4, 0.8],
        jitter::Real = 1e-8,
        seeds = 4
        ) where {K}
        kappa >= 0 || error("`kappa` must be nonnegative.")
        0 < budget <= 1 || error("`budget` must be in `(0, 1]`.")
        lengthscales = collect(Float64, lengthscales)
        (!isempty(lengthscales) && all(>(0), lengthscales)) ||
            error("`lengthscales` must be positive and nonempty.")
        jitter >= 0 || error("`jitter` must be nonnegative.")
        if seeds isa Int
            seeds >= 2 || error("`seeds` must be at least 2.")
        else
            seeds = collect(Float64, seeds)
            (!isempty(seeds) && all(f -> 0 <= f <= 1, seeds)) ||
                error("`seeds` must be fractions in `[0, 1]`.")
        end
        new{K}(kernel, Float64(kappa), Float64(budget),
            detect_uniform_M, lengthscales, Float64(jitter), seeds)
    end
end

"""
    sample_M_values(sampler, objectives, sub, method, support_grids)

Compute the MBM M values at the transcription supports of an infinite
model. `sampler` is the `sampler` field of [`MBM`](@ref),
`objectives` is the array of per-support objective expressions, `sub`
is the transcribed submodel wrapped as a `GDPSubmodel`, `method` is
the `_MBM` data, and `support_grids` is a tuple of the supports of
the infinite parameters (a sorted vector per independent parameter,
a matrix of joint supports per dependent group).
Returns an array of M values shaped like `objectives`, a
scalar when M is uniform across the supports, or `nothing` if an M
subproblem is infeasible. Extensions implement methods that dispatch
on `sampler`: an [`ExhaustiveSampler`](@ref) solves an M subproblem
at every support, while a [`GPSampler`](@ref) solves a subset of the
supports and
fills the rest with a Gaussian-process upper confidence bound.
"""
function sample_M_values(sampler, objectives, sub, method, support_grids)
    error("Unrecognized `sampler` value `$(repr(sampler))` for MBM " *
          "on an infinite model. Use `ExhaustiveSampler()` to solve " *
          "an M subproblem at every support, or a `GPSampler` (with " *
          "AbstractGPs loaded) to estimate M values with a " *
          "Gaussian process.")
end
