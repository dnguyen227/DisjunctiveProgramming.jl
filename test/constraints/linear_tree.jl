using HiGHS

# True piecewise-linear target with axis splits at 5 (min -10 @ (5,5),
# max 15 @ x1=5). Exactly representable by a depth-2 linear model tree.
_f_pwl(x1, x2) = x1 <= 5 ? (x2 <= 5 ? x1 : x2) :
                           (x2 <= 5 ? 20 - x1 : 2x1 + 2x2 - 30)

# Hand-built canonical tree (the path future sklearn/IAI adapters take).
function _pwl_tree()
    A = LinearTreeLeaf([1.0, 0.0], 0.0)
    B = LinearTreeLeaf([0.0, 1.0], 0.0)
    C = LinearTreeLeaf([-1.0, 0.0], 20.0)
    D = LinearTreeLeaf([2.0, 2.0], -30.0)
    root = LinearTreeSplit(1, 5.0,
        LinearTreeSplit(2, 5.0, A, B),
        LinearTreeSplit(2, 5.0, C, D))
    return LinearModelTree(root, 2)
end

function test_build_and_predict()
    grid = collect(0.0:0.4:10.0)          # 5.0 not on grid -> clean splits
    pts = [(a, b) for a in grid for b in grid]
    X = [pts[k][j] for k in eachindex(pts), j in 1:2]
    y = [_f_pwl(p...) for p in pts]
    tree = build_linear_tree(X, y; max_depth = 3, min_samples_leaf = 30)
    @test tree isa LinearModelTree
    @test tree.n_features == 2
    yhat = predict(tree, X)
    ybar = sum(y) / length(y)
    r2 = 1 - sum(abs2, y .- yhat) / sum(abs2, y .- ybar)
    @test r2 > 0.999                      # near-exact fit of a PWL target
    for p in ([2.0 2.0], [8.0 8.0], [2.0 8.0], [9.0 6.0])
        @test isapprox(predict(tree, p)[1], _f_pwl(p[1], p[2]); atol = 0.05)
    end
end

function test_embedding_exactness()
    tree = _pwl_tree()
    for method in (BigM(), Hull())
        for p in ([2.0, 2.0], [8.0, 8.0], [2.0, 8.0], [8.0, 2.0])
            model = GDPModel(HiGHS.Optimizer)
            set_attribute(model, MOI.Silent(), true)
            @variable(model, 0 <= x[1:2] <= 10)
            y = add_linear_tree(model, tree, x)
            @constraint(model, x .== p)
            @objective(model, Min, 0)
            optimize!(model, gdp_method = method)
            @test isapprox(value(y), _f_pwl(p[1], p[2]); atol = 1e-6)
        end
    end
end

function test_optimize_recovers_optimum()
    tree = _pwl_tree()
    for method in (BigM(), Hull())
        model = GDPModel(HiGHS.Optimizer)
        set_attribute(model, MOI.Silent(), true)
        @variable(model, 0 <= x[1:2] <= 10)
        y = add_linear_tree(model, tree, x)
        @objective(model, Min, y)
        optimize!(model, gdp_method = method)
        @test termination_status(model) == MOI.OPTIMAL
        @test isapprox(objective_value(model), -10.0; atol = 1e-6)

        model = GDPModel(HiGHS.Optimizer)
        set_attribute(model, MOI.Silent(), true)
        @variable(model, 0 <= x[1:2] <= 10)
        y = add_linear_tree(model, tree, x)
        @objective(model, Max, y)
        optimize!(model, gdp_method = method)
        @test isapprox(objective_value(model), 15.0; atol = 1e-6)
    end
end

function test_single_leaf_tree()
    tree = LinearModelTree(LinearTreeLeaf([1.0, 2.0], 3.0), 2)
    model = GDPModel(HiGHS.Optimizer)
    set_attribute(model, MOI.Silent(), true)
    @variable(model, 0 <= x[1:2] <= 10)
    y = add_linear_tree(model, tree, x)
    @test isempty(DP._disjunctions(model))       # no disjunction created
    @constraint(model, x .== [4.0, 5.0])
    @objective(model, Min, 0)
    optimize!(model, gdp_method = BigM())
    @test isapprox(value(y), 1*4 + 2*5 + 3; atol = 1e-6)  # = 17
end

function test_linear_tree_errors()
    tree = _pwl_tree()
    model = GDPModel()
    @variable(model, x[1:2])                      # no bounds
    @test_throws ErrorException add_linear_tree(model, tree, x)
    @variable(model, 0 <= xb[1:3] <= 1)
    @test_throws ErrorException add_linear_tree(model, tree, xb)  # wrong dim
end

################################################################################
#                     INDEPENDENT ORACLE + FIT HELPERS
################################################################################
# Enumerate leaves by walking the public node types directly (independent of
# the package's own flatten, so a shared bug can't hide).
_walk(n::LinearTreeLeaf, path, out) =
    push!(out, (copy(path), n.coef, n.intercept))
function _walk(n::LinearTreeSplit, path, out)
    _walk(n.left, [path; (n.feature, n.threshold, :left)], out)
    _walk(n.right, [path; (n.feature, n.threshold, :right)], out)
end
function _leaf_list(tree)
    out = Tuple{Vector{Tuple{Int,Float64,Symbol}}, Vector{Float64}, Float64}[]
    _walk(tree.root, Tuple{Int,Float64,Symbol}[], out)
    return out
end

_setobj!(m, sense, e) = sense === MOI.MIN_SENSE ?
    (@objective(m, Min, e)) : (@objective(m, Max, e))

# Oracle: optimum of the surrogate over the box by solving one LP per leaf
# (its path polyhedron intersect the box) and taking the best.
function _enum_opt(tree, lb, ub, sense; add_extra! = (m, x) -> nothing)
    best = sense === MOI.MIN_SENSE ? Inf : -Inf
    for (path, a, b) in _leaf_list(tree)
        m = Model(HiGHS.Optimizer)
        set_attribute(m, MOI.Silent(), true)
        @variable(m, lb[j] <= x[j = 1:tree.n_features] <= ub[j])
        for (j, t, d) in path
            d === :left ? @constraint(m, x[j] <= t) : @constraint(m, x[j] >= t)
        end
        add_extra!(m, x)
        _setobj!(m, sense, sum(a[j] * x[j] for j in 1:tree.n_features) + b)
        optimize!(m)
        if termination_status(m) == MOI.OPTIMAL
            v = objective_value(m)
            best = sense === MOI.MIN_SENSE ? min(best, v) : max(best, v)
        end
    end
    return best
end

function _collect_splits(node, out)
    if node isa LinearTreeSplit
        push!(out, (node.feature, node.threshold))
        _collect_splits(node.left, out)
        _collect_splits(node.right, out)
    end
end

_pwl_grid(step) = begin
    grid = collect(0.0:step:10.0)
    pts = [(a, b) for a in grid for b in grid]
    X = [pts[k][j] for k in eachindex(pts), j in 1:2]
    (X, [_f_pwl(p...) for p in pts], pts)
end

# Deterministic pseudo-random noise in about [-0.05, 0.05] (no Random dep).
_noise(i) = 0.1 * (mod(sin(i * 12.9898) * 43758.5453, 1.0) - 0.5)

################################################################################
#                     POINT 2: GDP REPRESENTATION VALIDITY
################################################################################
function test_gdp_matches_enumeration()
    X, yv, _ = _pwl_grid(0.4)
    learned = build_linear_tree(X, yv; max_depth = 3, min_samples_leaf = 30)
    for tree in (_pwl_tree(), learned)
        for method in (BigM(), Hull(), Indicator())
            for sense in (MOI.MIN_SENSE, MOI.MAX_SENSE)
                model = GDPModel(HiGHS.Optimizer)
                set_attribute(model, MOI.Silent(), true)
                @variable(model, 0 <= x[1:2] <= 10)
                y = add_linear_tree(model, tree, x)
                _setobj!(model, sense, y)
                optimize!(model, gdp_method = method)
                oracle = _enum_opt(tree, [0.0, 0.0], [10.0, 10.0], sense)
                @test isapprox(objective_value(model), oracle; atol = 1e-4)
            end
        end
    end
end

function test_reformulation_agreement()
    tree = _pwl_tree()
    objs, xs = Float64[], Vector{Float64}[]
    for method in (BigM(), Hull(), Indicator())
        model = GDPModel(HiGHS.Optimizer)
        set_attribute(model, MOI.Silent(), true)
        @variable(model, 0 <= x[1:2] <= 10)
        y = add_linear_tree(model, tree, x)
        @objective(model, Min, y)
        optimize!(model, gdp_method = method)
        push!(objs, objective_value(model))
        push!(xs, value.(x))
    end
    @test all(isapprox(objs[1], o; atol = 1e-6) for o in objs)  # same optimum
    @test all(isapprox(xs[1], p; atol = 1e-4) for p in xs)      # same argmin
end

function test_embedded_in_model_matches_enumeration()
    tree = _pwl_tree()
    side!(m, x) = @constraint(m, x[1] >= 6)           # forces into leaves C/D
    for method in (BigM(), Hull())
        model = GDPModel(HiGHS.Optimizer)
        set_attribute(model, MOI.Silent(), true)
        @variable(model, 0 <= x[1:2] <= 10)
        y = add_linear_tree(model, tree, x)
        @constraint(model, x[1] >= 6)
        @objective(model, Min, y)
        optimize!(model, gdp_method = method)
        oracle = _enum_opt(tree, [0.0, 0.0], [10.0, 10.0], MOI.MIN_SENSE;
            add_extra! = side!)
        @test isapprox(objective_value(model), oracle; atol = 1e-4)
    end
end

################################################################################
#                        POINT 1: FIT QUALITY
################################################################################
function test_structure_recovery()
    X, yv, _ = _pwl_grid(0.4)                          # 5.0 not on grid
    tree = build_linear_tree(X, yv; max_depth = 3, min_samples_leaf = 30)
    @test length(_leaf_list(tree)) == 4
    sp = Tuple{Int,Float64}[]
    _collect_splits(tree.root, sp)
    @test all(4.5 <= t <= 5.5 for (_, t) in sp)        # breakpoints near 5
    @test Set(f for (f, _) in sp) == Set([1, 2])
    for p in ([2.0 2.0], [2.0 8.0], [8.0 2.0], [8.0 8.0])
        @test isapprox(predict(tree, p)[1], _f_pwl(p[1], p[2]); atol = 1e-6)
    end
end

function test_noisy_generalization()
    X, _, pts = _pwl_grid(0.35)
    n = length(pts)
    ynoisy = [_f_pwl(pts[k]...) + _noise(k) for k in 1:n]
    tr = [k for k in 1:n if iseven(k)]
    te = [k for k in 1:n if isodd(k)]
    tree = build_linear_tree(X[tr, :], ynoisy[tr];
        max_depth = 3, min_samples_leaf = 20)
    yhat = predict(tree, X[te, :])
    ytrue = [_f_pwl(pts[k]...) for k in te]
    ybar = sum(ytrue) / length(ytrue)
    r2 = 1 - sum(abs2, ytrue .- yhat) / sum(abs2, ytrue .- ybar)
    @test r2 > 0.98                                    # generalizes to truth
    sp = Tuple{Int,Float64}[]
    _collect_splits(tree.root, sp)
    @test all(4.3 <= t <= 5.7 for (_, t) in sp)
end

function test_monotone_improvement()
    grid = collect(0.0:0.5:10.0)
    pts = [(a, b) for a in grid for b in grid]
    n = length(pts)
    X = [pts[k][j] for k in 1:n, j in 1:2]
    yv = [pts[k][1]^2 + pts[k][2]^2 for k in 1:n]      # smooth, non-PWL
    sse(d) = sum(abs2, yv .- predict(
        build_linear_tree(X, yv; max_depth = d,
            min_samples_leaf = 5, min_gain = 0.0), X))
    @test sse(1) > sse(2) > sse(3)                     # depth genuinely helps
end

function test_hyperparameters_honored()
    grid = collect(0.0:0.5:10.0)
    pts = [(a, b) for a in grid for b in grid]
    n = length(pts)
    X = [pts[k][j] for k in 1:n, j in 1:2]
    yv = [pts[k][1]^2 + pts[k][2]^2 for k in 1:n]
    for d in 1:3
        t = build_linear_tree(X, yv; max_depth = d,
            min_samples_leaf = 20, min_gain = 0.0)
        @test length(_leaf_list(t)) <= 2^d             # depth bound
    end
    ylin = [3 + 2 * pts[k][1] - pts[k][2] for k in 1:n]
    t = build_linear_tree(X, ylin; max_depth = 4, min_samples_leaf = 5)
    @test length(_leaf_list(t)) == 1                   # linear -> no splits
end

@testset "Linear Model Trees" begin
    test_build_and_predict()
    test_embedding_exactness()
    test_optimize_recovers_optimum()
    test_single_leaf_tree()
    test_linear_tree_errors()
    test_gdp_matches_enumeration()
    test_reformulation_agreement()
    test_embedded_in_model_matches_enumeration()
    test_structure_recovery()
    test_noisy_generalization()
    test_monotone_improvement()
    test_hyperparameters_honored()
end
