################################################################################
#                        LINEAR MODEL TREE -- GREEDY FIT
################################################################################
# In-house greedy trainer for `LinearModelTree`: CART-style axis-aligned splits
# with a least-squares linear model at each leaf. Training is a separable layer
# from the GDP embedding (`linear_tree.jl`); external trainers (scikit-learn,
# IAI) belong in weak-dependency extensions that emit the same `LinearModelTree`.

# Least-squares leaf fit (tiny ridge for stability). Returns the slopes, the
# intercept, and the leaf's residual sum of squares.
function _fit_leaf(X, y, idx)
    Z = hcat(X[idx, :], ones(length(idx)))
    G = Z' * Z
    for k in 1:size(G, 1)
        G[k, k] += 1e-8
    end
    beta = G \ (Z' * y[idx])
    resid = y[idx] .- Z * beta
    return beta[1:end-1], beta[end], sum(abs2, resid)
end

# Best axis-aligned split by total child SSE; `nothing` if none is valid.
function _best_split(X, y, idx, min_leaf)
    best, best_sse = nothing, Inf
    for j in 1:size(X, 2)
        vals = sort!(unique(X[idx, j]))
        length(vals) < 2 && continue
        for k in 1:length(vals)-1
            t = (vals[k] + vals[k+1]) / 2
            lidx = [i for i in idx if X[i, j] <= t]
            ridx = [i for i in idx if X[i, j] > t]
            (length(lidx) < min_leaf || length(ridx) < min_leaf) && continue
            sse = _fit_leaf(X, y, lidx)[3] + _fit_leaf(X, y, ridx)[3]
            sse < best_sse && ((best, best_sse) = ((j, t, lidx, ridx), sse))
        end
    end
    return best, best_sse
end

function _grow(X, y, idx, depth, max_depth, min_leaf, min_gain)
    coef, b, sse = _fit_leaf(X, y, idx)
    leaf = LinearTreeLeaf(coef, b)
    (depth >= max_depth || length(idx) < 2 * min_leaf) && return leaf
    best, best_sse = _best_split(X, y, idx, min_leaf)
    (best === nothing || sse - best_sse <= min_gain * max(sse, 1.0)) &&
        return leaf
    j, t, lidx, ridx = best
    left = _grow(X, y, lidx, depth + 1, max_depth, min_leaf, min_gain)
    right = _grow(X, y, ridx, depth + 1, max_depth, min_leaf, min_gain)
    return LinearTreeSplit(j, t, left, right)
end

"""
    build_linear_tree(
        X::AbstractMatrix,
        y::AbstractVector;
        [max_depth::Int = 3],
        [min_samples_leaf::Int = 10],
        [min_gain::Real = 1e-6]
        )::LinearModelTree

Fit a linear model decision tree greedily (CART-style splits with a
least-squares linear model at each leaf). `X` is `n_samples` by
`n_features`; `y` is length `n_samples`. A node splits only if the best
axis-aligned split reduces the residual sum of squares by more than
`min_gain` (relative to the node's own SSE), the depth is below
`max_depth`, and both children keep at least `min_samples_leaf` samples.

## Keyword Arguments
- `max_depth::Int`: Maximum tree depth.
- `min_samples_leaf::Int`: Minimum samples per leaf.
- `min_gain::Real`: Minimum relative SSE reduction to justify a split.

## Returns
- `LinearModelTree`: The fitted tree.
"""
function build_linear_tree(
    X::AbstractMatrix,
    y::AbstractVector;
    max_depth::Int = 3,
    min_samples_leaf::Int = 10,
    min_gain::Real = 1e-6,
    )
    size(X, 1) == length(y) || error("`X` and `y` size mismatch.")
    root = _grow(X, y, collect(1:size(X, 1)), 1, max_depth,
        min_samples_leaf, Float64(min_gain))
    return LinearModelTree(root, size(X, 2))
end
