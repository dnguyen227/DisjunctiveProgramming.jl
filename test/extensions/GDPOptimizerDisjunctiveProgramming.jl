using HiGHS, Ipopt, Juniper, InfiniteOpt
import GDPOptimizer
import DisjunctiveProgramming as DP

function _gdp_optimizer_factory()
    return () -> GDPOptimizer.Optimizer(nlp_solver = Ipopt.Optimizer,
        mip_solver = HiGHS.Optimizer)
end

# The lowering emits one vector constraint per disjunction with the
# activation first, then the indicators, then the rows grouped by
# disjunct.
function test_disjunction_set_lowering()
    model = GDPModel()
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 == 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    reformulate_model(model, MOIDisjunction())
    crefs = DP._reformulation_constraints(model)
    vector_crefs = filter(crefs) do cref
        constraint_object(cref).set isa GDPOptimizer.DisjunctionSet
    end
    @test length(vector_crefs) == 1
    constraint = constraint_object(only(vector_crefs))
    set = constraint.set
    @test set.inner_sets ==
        [[MOI.LessThan(3.0)], [MOI.EqualTo(64.0)]]
    @test length(constraint.func) == MOI.dimension(set) == 5
    @test isequal_canonical(constraint.func[1],
        one(constraint.func[1])) # top-level activation is the constant 1
    @test coefficient(constraint.func[2], binary_variable(Y[1])) == 1.0
end

# A nested disjunction lowers to its own flat constraint whose
# activation is the parent indicator; the parent carries no rows for it.
function test_lowering_nested()
    model = GDPModel()
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 5, Disjunct(W[1]))
    @constraint(model, x <= 6, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]))
    @disjunction(model, Y)
    reformulate_model(model, MOIDisjunction())
    objs = [constraint_object(cref)
            for cref in DP._reformulation_constraints(model)
            if constraint_object(cref).set isa GDPOptimizer.DisjunctionSet]
    @test length(objs) == 2
    inner = only(filter(c -> c.set.inner_sets ==
        [[MOI.LessThan(5.0)], [MOI.LessThan(6.0)]], objs))
    outer = only(filter(c -> c.set.inner_sets ==
        [[MOI.LessThan(3.0)], MOI.AbstractScalarSet[]], objs))
    @test isequal_canonical(outer.func[1], one(outer.func[1]))
    @test coefficient(inner.func[1], binary_variable(Y[2])) == 1.0
    @test length(outer.func) == MOI.dimension(outer.set) == 4
end

function test_lowering_nested_requires_exactly1()
    model = GDPModel()
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 5, Disjunct(W[1]))
    @constraint(model, x <= 6, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]), exactly1 = false)
    @disjunction(model, Y)
    @test_throws ErrorException reformulate_model(model,
        MOIDisjunction())
end

# Nested solve: max x with x <= 2, or a nested mode choice between
# x^2 <= 25 and x <= 8.
function test_lowering_solve_nested()
    model = GDPModel(_gdp_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 2, Disjunct(Y[1]))
    @constraint(model, x^2 <= 25, Disjunct(W[1]))
    @constraint(model, x <= 8, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = MOIDisjunction())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2])
    @test value(W[2])
end

# The same GDP solved through a BigM reformulation and through the
# lowering into GDPOptimizer must agree: max x with (x <= 3) or
# (x <= 7).
function test_lowering_solve_linear()
    model = GDPModel(_gdp_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = MOIDisjunction())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-4
    @test value(x) ≈ 7.0 atol = 1e-4
    @test value(Y[2])

    reference = GDPModel(HiGHS.Optimizer)
    set_silent(reference)
    @variable(reference, 0 <= x2 <= 10)
    @variable(reference, Y2[1:2], Logical)
    @constraint(reference, x2 <= 3, Disjunct(Y2[1]))
    @constraint(reference, x2 <= 7, Disjunct(Y2[2]))
    @disjunction(reference, Y2)
    @objective(reference, Max, x2)
    optimize!(reference, gdp_method = BigM())
    @test objective_value(model) ≈ objective_value(reference) atol = 1e-6
end

# Nonlinear disjunct row: max x with (x <= 3) or (x^2 == 64); the
# unique optimum 8 is checked directly
function test_lowering_solve_nonlinear()
    model = GDPModel(_gdp_optimizer_factory())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 == 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = MOIDisjunction())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2])
end

# An InfiniteGDPModel lowers through the same method: the disjunction
# transcribes to one DisjunctionSet constraint per support.
function test_lowering_infinite()
    model = InfiniteGDPModel(_gdp_optimizer_factory())
    set_silent(model)
    @infinite_parameter(model, t in [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, integral(x, t))
    optimize!(model, gdp_method = MOIDisjunction())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 10.0 atol = 1e-4
    @test all(value(x) .>= 5.0 .- 1e-4)
end

# A nested infinite disjunction transcribes with the parent indicator
# as its activation, so the rebuilt constraint function is a uniform
# vector of scalar expressions (no constant to promote).
function test_lowering_infinite_nested()
    model = InfiniteGDPModel(_gdp_optimizer_factory())
    set_silent(model)
    @infinite_parameter(model, t in [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @variable(model, W[1:2], InfiniteLogical(t))
    @constraint(model, x <= 2, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(W[1]))
    @constraint(model, x >= 3, Disjunct(W[2]))
    @disjunction(model, W, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, integral(x, t))
    optimize!(model, gdp_method = MOIDisjunction())
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 10.0 atol = 1e-4
end

@testset "GDPOptimizer lowering" begin
    test_disjunction_set_lowering()
    test_lowering_nested()
    test_lowering_nested_requires_exactly1()
    test_lowering_solve_linear()
    test_lowering_solve_nonlinear()
    test_lowering_solve_nested()
    test_lowering_infinite()
    test_lowering_infinite_nested()
end
