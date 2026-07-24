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

@testset "Linear Model Trees" begin
    test_build_and_predict()
    test_embedding_exactness()
    test_optimize_recovers_optimum()
    test_single_leaf_tree()
    test_linear_tree_errors()
end
