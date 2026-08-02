using InfiniteOpt, HiGHS, AbstractGPs, KernelFunctions
import DisjunctiveProgramming as DP

# Helpers to access internal functions of the two extensions
const IGDP = Base.get_extension(DP, :InfiniteGPDisjunctiveProgramming)
const IDP = Base.get_extension(DP, :InfiniteDisjunctiveProgramming)

function test_gp_sampler_creation()
    sampler = GPSampler()
    @test sampler isa IGDP.GPSampler
    @test sampler.kappa == 2.5
    @test sampler.budget == 0.25
    @test sampler.min_solves == 6
    @test isnothing(sampler.kernel)
    @test sampler.detect_uniform_M
    kern = with_lengthscale(SqExponentialKernel(), 0.3)
    sampler = GPSampler(kappa = 4.0, budget = 0.1, min_solves = 3,
        kernel = kern, detect_uniform_M = false)
    @test sampler.kappa == 4.0
    @test sampler.budget == 0.1
    @test sampler.min_solves == 3
    @test sampler.kernel === kern
    @test !sampler.detect_uniform_M
    @test_throws ErrorException GPSampler(kappa = -1)
    @test_throws ErrorException GPSampler(budget = 0)
    @test_throws ErrorException GPSampler(budget = 1.5)
    @test_throws ErrorException GPSampler(min_solves = 0)
end

function test_gp_sampler_resolution()
    # the GP extension is loaded, so :auto resolves to a GPSampler
    @test IDP._resolve_M_sampler(:auto) isa IGDP.GPSampler
    @test IDP._resolve_M_sampler(:exact) === :exact
    sampler = GPSampler(kappa = 3.0)
    @test IDP._resolve_M_sampler(sampler) === sampler
    @test MBM(HiGHS.Optimizer).M_sampler === :auto
    @test MBM(HiGHS.Optimizer, M_sampler = :exact).M_sampler === :exact
end

# Mirror of test_raw_M_infinite_scalar: uniform seed M values collapse
# to the exactly-solved scalar under the GP sampler
function test_gp_raw_M_scalar()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, con, x >= 5, Disjunct(Y[1]))
    @constraint(model, con2, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    mbm = DP._MBM(MBM(HiGHS.Optimizer), model)
    sub = DP.copy_model_with_constraints(
        model, DP.DisjunctConstraintRef[con2], mbm)
    obj = DP.prepare_max_M_objective(
        model, JuMP.constraint_object(con), sub)
    @test DP.raw_M(sub, obj, mbm) == 5.0
end

# With few supports the budget floor covers every support, so the GP
# sampler solves all of them exactly and must reproduce the exact
# grid parameter function
function test_gp_raw_M_matches_exact()
    function pfunc_values(M_sampler, supports)
        model = InfiniteGDPModel()
        @infinite_parameter(model, t ∈ [0, 1], supports = supports)
        @variable(model, 0 <= x <= 10, Infinite(t))
        @variable(model, Y[1:2], InfiniteLogical(t))
        @parameter_function(model, f == t -> 2*t)
        @constraint(model, con, x <= f, Disjunct(Y[1]))
        @constraint(model, con2, x >= 0.5, Disjunct(Y[2]))
        @disjunction(model, Y)
        mbm = DP._MBM(
            MBM(HiGHS.Optimizer, M_sampler = M_sampler), model)
        sub = DP.copy_model_with_constraints(
            model, DP.DisjunctConstraintRef[con2], mbm)
        obj = DP.prepare_max_M_objective(
            model, JuMP.constraint_object(con), sub)
        M = DP.raw_M(sub, obj, mbm)
        @test M isa InfiniteOpt.GeneralVariableRef
        return [InfiniteOpt.raw_function(M)(t_val) for t_val in supports]
    end
    supports = [0.0, 0.25, 0.5, 0.75, 1.0]
    exact_vals = pfunc_values(:exact, supports)
    @test pfunc_values(GPSampler(), supports) == exact_vals
    # a user kernel skips the lengthscale fit but solves the same supports
    kern = with_lengthscale(SqExponentialKernel(), 0.2)
    @test pfunc_values(GPSampler(kernel = kern), supports) == exact_vals
end

# an empty disjunct region makes the M subproblems infeasible; both
# samplers propagate that up to the reformulation error
function test_gp_infeasible_disjunct()
    function build()
        model = InfiniteGDPModel(HiGHS.Optimizer)
        set_silent(model)
        @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
        @variable(model, 0 <= x <= 10, Infinite(t))
        @variable(model, Y[1:2], InfiniteLogical(t))
        @parameter_function(model, f == t -> 2*t)
        @constraint(model, x <= f, Disjunct(Y[1]))
        @constraint(model, x >= 8, Disjunct(Y[2]))
        @constraint(model, x <= 3, Disjunct(Y[2]))
        @disjunction(model, Y)
        @objective(model, Max, 𝔼(x, t))
        return model
    end
    for sampler in (:exact, GPSampler())
        model = build()
        @test_throws ErrorException optimize!(model,
            gdp_method = MBM(HiGHS.Optimizer, M_sampler = sampler))
    end
end

# optimum (10) needs M(t) >= 10 - 2t pointwise; the GP fill is heuristic
function test_gp_mbm_solve_equivalence()
    function solve_with(M_sampler)
        model = InfiniteGDPModel(HiGHS.Optimizer)
        set_silent(model)
        @infinite_parameter(model, t ∈ [0, 1], num_supports = 20)
        @variable(model, 0 <= x <= 10, Infinite(t))
        @variable(model, Y[1:2], InfiniteLogical(t))
        @parameter_function(model, f == t -> 2*t)
        @constraint(model, x <= f, Disjunct(Y[1]))
        @constraint(model, x >= 0.5, Disjunct(Y[2]))
        @disjunction(model, Y)
        @objective(model, Max, 𝔼(x, t))
        optimize!(model,
            gdp_method = MBM(HiGHS.Optimizer, M_sampler = M_sampler))
        @test termination_status(model) == MOI.OPTIMAL
        return objective_value(model)
    end
    obj_exact = solve_with(:exact)
    obj_auto = solve_with(:auto)
    obj_gp = solve_with(GPSampler(kappa = 4.0, budget = 0.2))
    @test obj_exact ≈ 10.0 atol = 1e-4
    # over-M can't raise the optimum, under-M can only shave it a bit
    @test obj_auto <= obj_exact + 1e-6
    @test obj_auto ≈ obj_exact atol = 1e-2
    @test obj_gp <= obj_exact + 1e-6
    @test obj_gp ≈ obj_exact atol = 1e-2
end

# A periodic M must not read as uniform. With f(t) = 2|cos(2*pi*t)|
# on these supports, M = 10 - f is 8 at supports 1, 3, 5 and 10 at
# supports 2, 4, so evenly spaced probes alias and collapse M to 8,
# which caps x at 8 and cuts the optimum from 10 down to 9.
function test_gp_periodic_M_not_uniform()
    supports = [0.0, 0.25, 0.5, 0.75, 1.0]
    function solve_with(M_sampler)
        model = InfiniteGDPModel(HiGHS.Optimizer)
        set_silent(model)
        @infinite_parameter(model, t ∈ [0, 1], supports = supports)
        @variable(model, 0 <= x <= 10, Infinite(t))
        @variable(model, Y[1:2], InfiniteLogical(t))
        @parameter_function(model, f == t -> 2 * abs(cos(2 * pi * t)))
        @constraint(model, x <= f, Disjunct(Y[1]))
        @constraint(model, x >= 0.5, Disjunct(Y[2]))
        @disjunction(model, Y)
        @objective(model, Max, 𝔼(x, t))
        optimize!(model, gdp_method = MBM(
            HiGHS.Optimizer, M_sampler = M_sampler))
        return objective_value(model)
    end
    @test solve_with(:exact) ≈ 10.0 atol = 1e-6
    @test solve_with(GPSampler()) ≈ 10.0 atol = 1e-6
end

# With detection off the uniform M is not collapsed to a scalar: the
# GP is fit and the unsolved supports keep their kappa * sd cushion,
# which must sit above the M that detection would have returned.
function test_gp_detect_uniform_M_off()
    function raw_M_with(detect)
        model = InfiniteGDPModel()
        @infinite_parameter(model, t ∈ [0, 1], num_supports = 20)
        @variable(model, 0 <= x <= 10, Infinite(t))
        @variable(model, Y[1:2], InfiniteLogical(t))
        @constraint(model, con, x >= 5, Disjunct(Y[1]))
        @constraint(model, con2, x <= 3, Disjunct(Y[2]))
        @disjunction(model, Y)
        mbm = DP._MBM(MBM(HiGHS.Optimizer,
            M_sampler = GPSampler(detect_uniform_M = detect)), model)
        sub = DP.copy_model_with_constraints(
            model, DP.DisjunctConstraintRef[con2], mbm)
        obj = DP.prepare_max_M_objective(
            model, JuMP.constraint_object(con), sub)
        return DP.raw_M(sub, obj, mbm)
    end
    @test raw_M_with(true) == 5.0
    M = raw_M_with(false)
    @test M isa InfiniteOpt.GeneralVariableRef
    raw_fn = InfiniteOpt.raw_function(M)
    vals = [raw_fn(t) for t in range(0, 1, length = 20)]
    @test all(vals .>= 5.0 - 1e-6)
    @test maximum(vals) > 5.0
end

# Dependent parameters have no support grid, so turning detection off
# leaves the GP with nothing to fit over
function test_gp_detect_uniform_M_off_dependent()
    model = InfiniteGDPModel()
    @infinite_parameter(model, ξ[1:2] ∈ [0, 1], num_supports = 4)
    @variable(model, 0 <= x <= 10, Infinite(ξ))
    @variable(model, Y[1:2], InfiniteLogical(ξ))
    @constraint(model, con, x >= 5, Disjunct(Y[1]))
    @constraint(model, con2, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    mbm = DP._MBM(MBM(HiGHS.Optimizer,
        M_sampler = GPSampler(detect_uniform_M = false)), model)
    sub = DP.copy_model_with_constraints(
        model, DP.DisjunctConstraintRef[con2], mbm)
    obj = DP.prepare_max_M_objective(
        model, JuMP.constraint_object(con), sub)
    @test_throws ErrorException DP.raw_M(sub, obj, mbm)
end

function test_gp_unknown_sampler_error()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @parameter_function(model, f == t -> 2*t)
    @constraint(model, x <= f, Disjunct(Y[1]))
    @constraint(model, x >= 0.5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, 𝔼(x, t))
    @test_throws ErrorException optimize!(model,
        gdp_method = MBM(HiGHS.Optimizer, M_sampler = :grid))
end

@testset "InfiniteGPDisjunctiveProgramming" begin
    test_gp_sampler_creation()
    test_gp_sampler_resolution()
    test_gp_raw_M_scalar()
    test_gp_raw_M_matches_exact()
    test_gp_mbm_solve_equivalence()
    test_gp_periodic_M_not_uniform()
    test_gp_detect_uniform_M_off()
    test_gp_detect_uniform_M_off_dependent()
    test_gp_unknown_sampler_error()
    test_gp_infeasible_disjunct()
end
