module AbstractGPsDisjunctiveProgramming

import AbstractGPs
import AbstractGPs.KernelFunctions
import DisjunctiveProgramming as DP

################################################################################
#                                 GP FITTING
################################################################################
# Normalized to [0, 1]^d so one lengthscale works across dimensions
function _support_coords(grids)
    idxs = CartesianIndices(length.(grids))
    los = [minimum(g) for g in grids]
    rng = [max(maximum(g) - minimum(g), eps()) for g in grids]
    return [[(grids[d][I[d]] - los[d]) / rng[d] for d in 1:length(grids)]
            for I in vec(idxs)]
end

# Lengthscale selected by marginal likelihood over the candidates
function _fit_posterior(kernel, X, y, sampler)
    best_post, best_lp = nothing, -Inf
    for ls in sampler.lengthscales
        kern = KernelFunctions.with_lengthscale(kernel, ls)
        fx = AbstractGPs.GP(kern)(X, sampler.jitter)
        lp = AbstractGPs.logpdf(fx, y)
        if lp > best_lp
            best_post, best_lp = AbstractGPs.posterior(fx, y), lp
        end
    end
    return best_post
end

function _mean_sd(sampler, X, solved)
    lis = collect(keys(solved))
    y = [solved[li] for li in lis]
    ybar = sum(y) / length(y)
    # floored so near-equal solved values still cushion the filled ones
    ystd = max(sqrt(sum(abs2, y .- ybar) / max(length(y) - 1, 1)),
        1e-2 * abs(ybar), 1e-8)
    kernel = something(sampler.kernel,
        KernelFunctions.SqExponentialKernel())
    post = _fit_posterior(kernel, X[lis], (y .- ybar) ./ ystd, sampler)
    mz = AbstractGPs.mean(post, X)
    vz = max.(AbstractGPs.var(post, X), 0.0)
    return mz .* ystd .+ ybar, sqrt.(vz) .* ystd
end

################################################################################
#                              M VALUE SAMPLING
################################################################################
# Solve M at max-UCB selected supports, fill the rest with the bound
function DP.sample_M_values(
    sampler::DP.GPSampler,
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
    # an evenly spaced seed count, or user-given seed fractions
    fracs = sampler.seeds isa Int ?
        range(0, 1, length = sampler.seeds) : sampler.seeds
    for s in unique(1 .+ round.(Int, fracs .* (n - 1)))
        solve_at(s) || return nothing
    end
    if sampler.detect_uniform_M
        # a uniform M needs no fit, and so no support grid either
        probes = collect(values(solved))
        all(==(first(probes)), probes) && return first(probes)
    end
    budget = min(ceil(Int, sampler.budget * n), n)
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
        M_vals[I] = get(solved, li,
            max(ms[li] + sampler.kappa * ss[li], 0.0))
    end
    return M_vals
end

end
