module AbstractGPsDisjunctiveProgramming

import AbstractGPs
import AbstractGPs.KernelFunctions
import DisjunctiveProgramming as DP

################################################################################
#                                 GP FITTING
################################################################################
# Normalized to [0, 1]^d so one lengthscale works across dimensions.
# An independent parameter contributes one coordinate; a dependent
# group contributes its joint-support column.
function _support_coords(grids::Tuple)::Vector{Vector{Float64}}
    axis_coords = map(grids) do g
        g isa AbstractMatrix ? [g[:, j] for j in axes(g, 2)] :
            [[v] for v in g]
    end
    coords = [reduce(vcat, getindex.(axis_coords, Tuple(I)))
              for I in vec(CartesianIndices(length.(axis_coords)))]
    dims = eachindex(first(coords))
    mins = [minimum(c[d] for c in coords) for d in dims]
    ranges = [max(maximum(c[d] for c in coords) - mins[d], eps())
              for d in dims]
    return [[(c[d] - mins[d]) / ranges[d] for d in dims]
            for c in coords]
end

# Lengthscale selected by marginal likelihood over the candidates
function _fit_posterior(
    kernel::KernelFunctions.Kernel,
    X::Vector{Vector{Float64}},
    y::Vector{Float64},
    sampler::DP.GPSampler
    )
    best_posterior, best_log_prob = nothing, -Inf
    for lengthscale in sampler.lengthscales
        scaled_kernel = KernelFunctions.with_lengthscale(
            kernel, lengthscale)
        finite_gp = AbstractGPs.GP(scaled_kernel)(X, sampler.jitter)
        log_prob = AbstractGPs.logpdf(finite_gp, y)
        if log_prob > best_log_prob
            best_posterior = AbstractGPs.posterior(finite_gp, y)
            best_log_prob = log_prob
        end
    end
    return best_posterior
end

function _mean_sd(
    sampler::DP.GPSampler,
    X::Vector{Vector{Float64}},
    solved::Dict{Int, Float64}
    )
    solved_indices = collect(keys(solved))
    y = [solved[i] for i in solved_indices]
    y_mean = sum(y) / length(y)
    # floored so near-equal solved values still cushion the filled ones
    y_scale = max(sqrt(sum(abs2, y .- y_mean) / max(length(y) - 1, 1)),
        1e-2 * abs(y_mean), 1e-8)
    kernel = something(sampler.kernel,
        KernelFunctions.SqExponentialKernel())
    posterior = _fit_posterior(
        kernel, X[solved_indices], (y .- y_mean) ./ y_scale, sampler)
    posterior_mean = AbstractGPs.mean(posterior, X)
    posterior_var = max.(AbstractGPs.var(posterior, X), 0.0)
    return posterior_mean .* y_scale .+ y_mean,
        sqrt.(posterior_var) .* y_scale
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
    support_grids::Tuple
    )
    indices = collect(CartesianIndices(objectives))
    n = length(indices)
    solved = Dict{Int, Float64}()
    solve_at(index::Int) = begin
        M_val = DP.raw_M(sub, objectives[indices[index]], method)
        M_val === nothing && return false
        solved[index] = M_val
        return true
    end
    # an evenly spaced seed count, or user-given seed fractions
    fractions = sampler.seeds isa Int ?
        range(0, 1, length = sampler.seeds) : sampler.seeds
    for index in unique(1 .+ round.(Int, fractions .* (n - 1)))
        solve_at(index) || return nothing
    end
    if sampler.detect_uniform_M
        # a uniform M needs no fit
        probes = collect(values(solved))
        all(==(first(probes)), probes) && return first(probes)
    end
    budget = min(ceil(Int, sampler.budget * n), n)
    X = _support_coords(support_grids)
    while length(solved) < budget
        means, sds = _mean_sd(sampler, X, solved)
        acquisition = means .+ sampler.kappa .* sds
        for index in keys(solved)
            acquisition[index] = -Inf
        end
        solve_at(argmax(acquisition)) || return nothing
    end
    M_vals = Array{Float64}(undef, size(objectives))
    if length(solved) == n # nothing left to estimate
        for (index, I) in enumerate(indices)
            M_vals[I] = solved[index]
        end
        return M_vals
    end
    means, sds = _mean_sd(sampler, X, solved)
    for (index, I) in enumerate(indices) # exact M values are nonnegative
        M_vals[I] = get(solved, index,
            max(means[index] + sampler.kappa * sds[index], 0.0))
    end
    return M_vals
end

end
