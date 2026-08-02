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
    detect_uniform_M::Bool
end

function DP.GPSampler(;
    kappa::Real = 2.5,
    budget::Real = 0.25,
    min_solves::Int = 6,
    kernel = nothing,
    detect_uniform_M::Bool = true
    )
    kappa >= 0 || error("`kappa` must be nonnegative.")
    0 < budget <= 1 || error("`budget` must be in `(0, 1]`.")
    min_solves >= 1 || error("`min_solves` must be at least 1.")
    return GPSampler(Float64(kappa), Float64(budget), min_solves,
        kernel, detect_uniform_M)
end

################################################################################
#                                 GP FITTING
################################################################################
const _LENGTHSCALES = (0.05, 0.1, 0.2, 0.4, 0.8)
const _JITTER = 1e-8

# Normalized to [0, 1]^d so one lengthscale works across dimensions
function _support_coords(grids)
    idxs = CartesianIndices(length.(grids))
    los = [minimum(g) for g in grids]
    rng = [max(maximum(g) - minimum(g), eps()) for g in grids]
    return [[(grids[d][I[d]] - los[d]) / rng[d] for d in 1:length(grids)]
            for I in vec(idxs)]
end

# Lengthscale by marginal likelihood unless the user gave a kernel
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

function _mean_sd(sampler::GPSampler, X, solved)
    lis = collect(keys(solved))
    y = [solved[li] for li in lis]
    ybar = sum(y) / length(y)
    # floored so near-equal solved values still cushion the filled ones
    ystd = max(sqrt(sum(abs2, y .- ybar) / max(length(y) - 1, 1)),
        1e-2 * abs(ybar), 1e-8)
    post = _fit_posterior(sampler, X[lis], (y .- ybar) ./ ystd)
    mz = AbstractGPs.mean(post, X)
    vz = max.(AbstractGPs.var(post, X), 0.0)
    return mz .* ystd .+ ybar, sqrt.(vz) .* ystd
end

################################################################################
#                              M VALUE SAMPLING
################################################################################
# Solve M at max-UCB selected supports, fill the rest with the bound
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
    # golden fractions; even spacing aliases with a periodic M
    for s in unique([1, n, 1 + floor(Int, 0.618 * (n - 1)),
                     1 + floor(Int, 0.382 * (n - 1))])
        solve_at(s) || return nothing
    end
    if sampler.detect_uniform_M
        # a uniform M needs no fit, and so no support grid either
        probes = collect(values(solved))
        all(==(first(probes)), probes) && return first(probes)
    end
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
    for (li, I) in enumerate(idxs) # exact M values are nonnegative
        M_vals[I] = get(solved, li, max(ms[li] + sampler.kappa * ss[li], 0.0))
    end
    return M_vals
end

end
