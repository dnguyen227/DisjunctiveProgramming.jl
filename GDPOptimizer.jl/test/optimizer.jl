# A two-disjunct toy stored straight into a model: binaries z1, z2 and
# continuous x with x <= 0 (disjunct 1) or x >= 1 (disjunct 2).
function _build_toy_disjunction(model)
    x = MOI.add_variable(model)
    z = MOI.add_variables(model, 2)
    for zi in z
        MOI.add_constraint(model, zi, MOI.ZeroOne())
    end
    set = GDPO.DisjunctionSet([[MOI.LessThan(0.0)], [MOI.GreaterThan(1.0)]])
    func = MOI.VectorAffineFunction(
        [MOI.VectorAffineTerm(2, MOI.ScalarAffineTerm(1.0, z[1])),
         MOI.VectorAffineTerm(3, MOI.ScalarAffineTerm(1.0, z[2])),
         MOI.VectorAffineTerm(4, MOI.ScalarAffineTerm(1.0, x)),
         MOI.VectorAffineTerm(5, MOI.ScalarAffineTerm(1.0, x))],
        [1.0, 0.0, 0.0, 0.0, 0.0]) # constant activation in component 1
    ci = MOI.add_constraint(model, func, set)
    return x, z, set, ci
end

function test_optimizer_scaffold()
    optimizer = GDPO.Optimizer()
    @test MOI.is_empty(optimizer)
    @test MOI.get(optimizer, MOI.SolverName()) == "GDPOptimizer"
    @test MOI.get(optimizer, MOI.TerminationStatus()) ==
        MOI.OPTIMIZE_NOT_CALLED
    @test MOI.get(optimizer, MOI.ResultCount()) == 0
    @test MOI.get(optimizer, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(optimizer, MOI.DualStatus()) == MOI.NO_SOLUTION
end

function test_optimizer_options()
    optimizer = GDPO.Optimizer(max_iter = 25)
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("max_iter")) == 25
    MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 5)
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("max_iter")) == 5
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("M_value")) == 1e9
    bogus = MOI.RawOptimizerAttribute("bogus")
    @test !MOI.supports(optimizer, bogus)
    @test_throws MOI.UnsupportedAttribute MOI.get(optimizer, bogus)
    @test_throws MOI.UnsupportedAttribute MOI.set(optimizer, bogus, 1)
    @test_throws ArgumentError GDPO.Optimizer(bogus = 1)
    MOI.set(optimizer, MOI.Silent(), true)
    @test MOI.get(optimizer, MOI.Silent())
    MOI.set(optimizer, MOI.TimeLimitSec(), 100)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) == 100.0
    @test MOI.get(optimizer, MOI.RawOptimizerAttribute("time_limit")) == 100.0
    MOI.set(optimizer, MOI.TimeLimitSec(), nothing)
    @test MOI.get(optimizer, MOI.TimeLimitSec()) === nothing
end

function test_model_building()
    optimizer = GDPO.Optimizer()
    x, z, set, ci = _build_toy_disjunction(optimizer)
    @test MOI.is_valid(optimizer, ci)
    @test MOI.get(optimizer, MOI.NumberOfVariables()) == 3
    @test (MOI.VectorAffineFunction{Float64}, GDPO.DisjunctionSet) in
        MOI.get(optimizer, MOI.ListOfConstraintTypesPresent())
    @test MOI.get(optimizer, MOI.ConstraintSet(), ci) == set
    objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, x)], 0.0)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(optimizer,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        objective)
    @test MOI.get(optimizer, MOI.ObjectiveSense()) == MOI.MIN_SENSE
    @test !MOI.is_empty(optimizer)
    MOI.empty!(optimizer)
    @test MOI.is_empty(optimizer)
end

function test_copy_to()
    src = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    x, z, set, ci = _build_toy_disjunction(src)
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.0))
    dest = GDPO.Optimizer()
    index_map = MOI.copy_to(dest, src)
    @test MOI.get(dest, MOI.NumberOfVariables()) == 3
    @test (MOI.VectorAffineFunction{Float64}, GDPO.DisjunctionSet) in
        MOI.get(dest, MOI.ListOfConstraintTypesPresent())
    @test MOI.get(dest, MOI.ConstraintSet(), index_map[ci]) == set
    dest_func = MOI.get(dest, MOI.ConstraintFunction(), index_map[ci])
    @test dest_func.terms[1].scalar_term.variable == index_map[z[1]]
end

@testset "Optimizer scaffold" begin
    test_optimizer_scaffold()
    test_optimizer_options()
    test_model_building()
    test_copy_to()
end
