module InfiniteGPDisjunctiveProgramming

import InfiniteOpt, JuMP
import AbstractGPs, KernelFunctions
import DisjunctiveProgramming as DP

################################################################################
#                                 GP SAMPLER
################################################################################
# See the `GPSampler` docstring in `src/extension_api.jl`
struct GPSampler{K}
    kappa::Float64
    budget::Float64
    min_solves::Int
    kernel::K
end

function DP.GPSampler(;
    kappa::Real = 2.5,
    budget::Real = 0.25,
    min_solves::Int = 6,
    kernel = nothing
    )
    kappa >= 0 || error("`kappa` must be nonnegative.")
    0 < budget <= 1 || error("`budget` must be in `(0, 1]`.")
    min_solves >= 1 || error("`min_solves` must be at least 1.")
    return GPSampler(Float64(kappa), Float64(budget), min_solves, kernel)
end

################################################################################
#                                 GP FITTING
################################################################################
# Lengthscale candidates on the [0, 1]-normalized support coordinates
const _LENGTHSCALES = (0.05, 0.1, 0.2, 0.4, 0.8)

# Observation jitter for the GP fit
const _JITTER = 1e-8

# Coordinates of every support in linear index order, normalized to
# [0, 1]^d so one isotropic lengthscale works across dimensions
function _support_coords(grids)
    idxs = CartesianIndices(length.(grids))
    los = [minimum(g) for g in grids]
    rng = [max(maximum(g) - minimum(g), eps()) for g in grids]
    return [[(grids[d][I[d]] - los[d]) / rng[d] for d in 1:length(grids)]
            for I in vec(idxs)]
end

# Fit the GP posterior on the solved coordinates; `y` is standardized
# by the caller. With no user kernel, select the lengthscale of a
# squared exponential kernel by maximizing the marginal likelihood.
function _fit_posterior(sampler::GPSampler, X, y)
    isnothing(sampler.kernel) || return AbstractGPs.posterior(
        AbstractGPs.GP(sampler.kernel)(X, _JITTER), y)
    best_post, best_lp = nothing, -Inf
    for ls in _LENGTHSCALES
        kern = KernelFunctions.with_lengthscale(
            KernelFunctions.SqExponentialKernel(), ls)
        fx = AbstractGPs.GP(kern)(X, _JITTER)
        lp = AbstractGPs.logpdf(fx, y)
        if lp > best_lp
            best_post, best_lp = AbstractGPs.posterior(fx, y), lp
        end
    end
    return best_post
end

# Posterior mean and sd at all coords given the solved (index => M)
# samples, destandardized back to M units
function _mean_sd(sampler::GPSampler, X, solved)
    lis = collect(keys(solved))
    y = [solved[li] for li in lis]
    ybar = sum(y) / length(y)
    ystd = max(sqrt(sum(abs2, y .- ybar) / max(length(y) - 1, 1)), 1e-8)
    post = _fit_posterior(sampler, X[lis], (y .- ybar) ./ ystd)
    mz = AbstractGPs.mean(post, X)
    vz = max.(AbstractGPs.var(post, X), 0.0)
    return mz .* ystd .+ ybar, sqrt.(vz) .* ystd
end

################################################################################
#                              M VALUE SAMPLING
################################################################################
# Solve M at actively-selected supports (max-UCB acquisition) and fill
# the rest with the upper confidence bound mean + kappa * sd, a
# heuristic over-estimate. Returns a scalar when the seed M values are
# uniform (e.g. M does not vary over the supports).
function DP.sample_M_values(
    sampler::GPSampler,
    objectives::AbstractArray,
    sub::DP.GDPSubmodel,
    method::DP._MBM,
    support_grids
    )
    idxs = collect(CartesianIndices(objectives))
    n = length(idxs)
    solved = Dict{Int, Float64}()
    solve_at(li) = begin
        m = DP.raw_M(sub, objectives[idxs[li]], method)
        m === nothing && return false
        solved[li] = m
        return true
    end
    for s in unique([1, cld(n + 1, 2), n])
        solve_at(s) || return nothing
    end
    seed = collect(values(solved))
    all(==(first(seed)), seed) && return first(seed)
    budget = clamp(
        ceil(Int, sampler.budget * n), min(sampler.min_solves, n), n)
    X = _support_coords(support_grids())
    while length(solved) < budget
        ms, ss = _mean_sd(sampler, X, solved)
        acq = ms .+ sampler.kappa .* ss
        for li in keys(solved)
            acq[li] = -Inf
        end
        solve_at(argmax(acq)) || return nothing
    end
    M_vals = Array{Float64}(undef, size(objectives))
    if length(solved) == n # nothing left to estimate
        for (li, I) in enumerate(idxs)
            M_vals[I] = solved[li]
        end
        return M_vals
    end
    ms, ss = _mean_sd(sampler, X, solved)
    for (li, I) in enumerate(idxs)
        # exact M values are nonnegative, so the fill is too
        M_vals[I] = get(solved, li, max(ms[li] + sampler.kappa * ss[li], 0.0))
    end
    return M_vals
end

end
