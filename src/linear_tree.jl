################################################################################
#                          LINEAR MODEL DECISION TREES
################################################################################
# A linear model decision tree is a decision tree whose leaves hold a linear
# model y = a'x + b, giving a piecewise-linear surrogate. It embeds exactly as
# a GDP disjunction: one disjunct per leaf, carrying the root-to-leaf split
# inequalities and the leaf's linear model. `LinearModelTree` is the canonical
# type and `add_linear_tree` embeds it; the greedy trainer that fits one from
# data lives in `linear_tree_fit.jl`.

################################################################################
#                              TREE REPRESENTATION
################################################################################
abstract type LinearTreeNode end

"""
    LinearTreeSplit(feature::Int, threshold::Real, left, right)

Internal node routing a point to `left` if `x[feature] <= threshold` and to
`right` otherwise. `left` and `right` are `LinearTreeNode`s.
"""
struct LinearTreeSplit <: LinearTreeNode
    feature::Int
    threshold::Float64
    left::LinearTreeNode
    right::LinearTreeNode
end

"""
    LinearTreeLeaf(coef::Vector{<:Real}, intercept::Real)

Leaf holding the linear model `y = coef' x + intercept`.
"""
struct LinearTreeLeaf <: LinearTreeNode
    coef::Vector{Float64}
    intercept::Float64
end

"""
    LinearModelTree(root::LinearTreeNode, n_features::Int)

A fitted linear model decision tree over `n_features` inputs. This is the
canonical type consumed by [`add_linear_tree`](@ref) and [`predict`](@ref);
adapters for external trainers (e.g. scikit-learn, IAI) should build this
same type so everything downstream is reused.
"""
struct LinearModelTree
    root::LinearTreeNode
    n_features::Int
end

################################################################################
#                              PREDICT / LEAVES
################################################################################
_eval(node::LinearTreeLeaf, x) =
    sum(node.coef[j] * x[j] for j in eachindex(x)) + node.intercept
_eval(node::LinearTreeSplit, x) = x[node.feature] <= node.threshold ?
    _eval(node.left, x) : _eval(node.right, x)

"""
    predict(tree::LinearModelTree, X::AbstractMatrix)::Vector

Evaluate the surrogate at each row of `X` (`n_samples` by `n_features`).
"""
predict(tree::LinearModelTree, X::AbstractMatrix) =
    [_eval(tree.root, view(X, i, :)) for i in 1:size(X, 1)]

# Flatten the tree to (path, coef, intercept) per leaf. Each path element is
# (feature, threshold, :left/:right).
function _tree_leaves(tree::LinearModelTree)
    out = Tuple{Vector{Tuple{Int,Float64,Symbol}},
        Vector{Float64}, Float64}[]
    _collect_leaves!(out, tree.root, Tuple{Int,Float64,Symbol}[])
    return out
end
_collect_leaves!(out, n::LinearTreeLeaf, path) =
    push!(out, (copy(path), n.coef, n.intercept))
function _collect_leaves!(out, n::LinearTreeSplit, path)
    _collect_leaves!(out, n.left, [path; (n.feature, n.threshold, :left)])
    _collect_leaves!(out, n.right, [path; (n.feature, n.threshold, :right)])
end

################################################################################
#                            GDP SURROGATE EMBEDDING
################################################################################
function _input_lb(xi)
    JuMP.has_lower_bound(xi) && return JuMP.lower_bound(xi)
    error("Input `$xi` needs a lower bound to embed a tree surrogate.")
end
function _input_ub(xi)
    JuMP.has_upper_bound(xi) && return JuMP.upper_bound(xi)
    error("Input `$xi` needs an upper bound to embed a tree surrogate.")
end

"""
    add_linear_tree(
        model::JuMP.AbstractModel,
        tree::LinearModelTree,
        x::AbstractVector;
        [output_name::String = ""],
        [xlb = lower bounds of `x`],
        [xub = upper bounds of `x`]
        )

Embed a linear model decision tree `tree` as a GDP disjunction over the
input variables `x` and return the surrogate output variable `y`. Each leaf
becomes a disjunct carrying its root-to-leaf split inequalities and its
linear model `y = a'x + b`; the disjunction selects exactly one leaf. The
output variable is bounded by the envelope of the leaves' linear models over
the input box `[xlb, xub]` (needed for the Hull reformulation). A single-leaf
tree is added as a plain linear equality with no disjunction.

## Keyword Arguments
- `output_name::String`: Base name for the output variable.
- `xlb`, `xub`: Lower/upper input bounds; default to the bounds of `x`.

## Returns
- The surrogate output variable `y`.
"""
function add_linear_tree(
    model::JuMP.AbstractModel,
    tree::LinearModelTree,
    x::AbstractVector;
    output_name::String = "",
    xlb = _input_lb.(x),
    xub = _input_ub.(x),
    )
    length(x) == tree.n_features ||
        error("`x` has $(length(x)) inputs, tree expects $(tree.n_features).")
    leaves = _tree_leaves(tree)
    ylo, yhi = Inf, -Inf
    for (_, a, b) in leaves
        lo = hi = b
        for j in eachindex(x)
            lo += min(a[j] * xlb[j], a[j] * xub[j])
            hi += max(a[j] * xlb[j], a[j] * xub[j])
        end
        ylo, yhi = min(ylo, lo), max(yhi, hi)
    end
    y = JuMP.@variable(model, base_name = output_name,
        lower_bound = ylo, upper_bound = yhi)
    if length(leaves) == 1                       # pure linear model
        a, b = leaves[1][2], leaves[1][3]
        JuMP.@constraint(model,
            y == sum(a[j] * x[j] for j in eachindex(x)) + b)
        return y
    end
    Y = JuMP.@variable(model, [1:length(leaves)], Logical)
    for (l, (path, a, b)) in enumerate(leaves)
        for (j, t, dir) in path
            dir === :left ?
                JuMP.@constraint(model, x[j] <= t, Disjunct(Y[l])) :
                JuMP.@constraint(model, x[j] >= t, Disjunct(Y[l]))
        end
        JuMP.@constraint(model,
            y == sum(a[j] * x[j] for j in eachindex(x)) + b, Disjunct(Y[l]))
    end
    disjunction(model, Y)
    return y
end
