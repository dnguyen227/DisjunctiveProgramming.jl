using JuMP
import HiGHS, Ipopt

function _loa_optimizer(; kwargs...)
    return () -> GDPO.Optimizer(; nlp_solver = Ipopt.Optimizer,
        mip_solver = HiGHS.Optimizer, kwargs...)
end

# min x with x >= 2 (disjunct 1) or x >= 5 (disjunct 2). The loop
# enumerates both combinations and keeps the incumbent at 2.
function test_linear_disjunction()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(2.0)], [MOI.GreaterThan(5.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test objective_value(model) ≈ 2.0 atol = 1e-5
    @test value(x) ≈ 2.0 atol = 1e-5
    @test value(z[1]) ≈ 1.0 atol = 1e-5
    @test value(z[2]) ≈ 0.0 atol = 1e-5
    @test occursin("LOA finished", raw_status(model))
    @test solve_time(model) > 0.0
    @test dual_status(model) == MOI.NO_SOLUTION
end

# Convex quadratic objective over linear disjuncts: y >= x or
# y >= 2 - x. The optimum sits at x = 3, y = 0 in disjunct 2.
function test_quadratic_objective()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 4)
    @variable(model, 0 <= y <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], y - x, y - (2 - x)] in
        GDPO.DisjunctionSet([
            [MOI.GreaterThan(0.0)], [MOI.GreaterThan(0.0)]]))
    @objective(model, Min, (x - 3)^2 + y)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 0.0 atol = 1e-5
    @test value(z[2]) ≈ 1.0 atol = 1e-5
    @test value(x) ≈ 3.0 atol = 1e-4
    @test value(y) ≈ 0.0 atol = 1e-5
    # exhaustion path: the bound is the last master's, still valid
    @test objective_bound(model) <= objective_value(model) + 1e-6
    @test !isnan(relative_gap(model))
end

# max x with (x <= 3) or (x <= 7): port of DP.jl test_loa_solve_simple.
function test_max_sense_linear()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-4
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# Two disjunctions: port of DP.jl test_loa_solve_two_disjunctions.
function test_two_disjunctions()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= w <= 10)
    @variable(model, zx[1:2], Bin)
    @variable(model, zw[1:2], Bin)
    @constraint(model, [1, zx[1], zx[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @constraint(model, [1, zw[1], zw[2], w, w] in GDPO.DisjunctionSet([
        [MOI.LessThan(2.0)], [MOI.LessThan(5.0)]]))
    @objective(model, Max, x + w)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 12.0 atol = 1e-4
end

# Nonlinear global x^2 <= 25 binds before the chosen disjunct: port of
# DP.jl test_loa_nonlinear_global (no Juniper needed - the layer's NLP
# has its binaries fixed).
function test_nonlinear_global()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, x^2 <= 25)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
    @test isfinite(relative_gap(model))
end

# Nonlinear global equality: one seed combination is NLP-infeasible, so
# the NLPF path supplies the linearization site. Port of DP.jl
# test_loa_nonlinear_equality_global.
function test_nonlinear_equality_global()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, x^2 == 25)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(x) ≈ 5.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# Nonlinear equality inside a disjunct: set covering must activate it
# and the cut emits both gated directions. Port of DP.jl
# test_loa_nonlinear_equality_disjunct.
function test_nonlinear_equality_disjunct()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(3.0)], [MOI.EqualTo(64.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
    # deterministic gap convergence: set covering visits the nonlinear
    # disjunct first, then the master proves the other is worse
    @test occursin("converged", raw_status(model))
    @test relative_gap(model) <= 1e-5
end

# Nonlinear Interval row inside a disjunct: port of DP.jl
# test_loa_nonlinear_interval_disjunct.
function test_nonlinear_interval_disjunct()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, x^2] in
        GDPO.DisjunctionSet([
            [MOI.LessThan(3.0)], [MOI.Interval(36.0, 64.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# A single binary drives both disjuncts through its complement: port of
# DP.jl test_loa_complement_indicator_nonlinear_disjunct.
function test_complement_indicator()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z, Bin)
    @constraint(model, [1, 1.0z, 1 - z, 1.0x, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(3.0)], [MOI.EqualTo(64.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(z) ≈ 0.0 atol = 1e-5
end

# `use_nlpf = false` still solves by enumeration when the seeds are
# feasible.
function test_nlpf_disabled()
    model = Model(_loa_optimizer(use_nlpf = false))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(3.0)], [MOI.EqualTo(64.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
end

# Big-M master gating solves the same instances as indicator gating
# (interval split included).
function test_bigm_master_gating()
    model = Model(_loa_optimizer(master_gating = "bigm", M_value = 100.0))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, 1.0x] in
        GDPO.DisjunctionSet([
            [MOI.LessThan(2.0)], [MOI.Interval(3.0, 7.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-4
end

# Integer-typed options convert at their use sites, including the
# Bool read of use_nlpf on the infeasible-seed path.
function test_integer_options()
    model = Model(_loa_optimizer(use_nlpf = 0, M_value = 10^9,
        time_limit = 3600))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(3.0)], [MOI.EqualTo(64.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
end

# A binary that is not a disjunction indicator keeps its integrality
# in the subproblem, which the nlp_solver must then handle (HiGHS
# both roles here since the model is linear).
function test_non_indicator_binary()
    factory = () -> GDPO.Optimizer(nlp_solver = HiGHS.Optimizer,
        mip_solver = HiGHS.Optimizer)
    model = Model(factory)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @variable(model, w, Bin)
    @constraint(model, 1.0x + w <= 8)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-5
end

# Globally infeasible model: the master is infeasible before any
# incumbent exists.
function test_infeasible_no_incumbent()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, 1.0x >= 20)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
    @test primal_status(model) == MOI.NO_SOLUTION
    @test result_count(model) == 0
end

# A zero time limit exits before any solve.
function test_time_limit()
    model = Model(_loa_optimizer(time_limit = 0.0))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.TIME_LIMIT
    @test result_count(model) == 0
end

# Nonconvex objective: the OA bound is no certificate, so the layer
# must never report OPTIMAL and the raw status carries the gap record.
function test_nonconvex_never_optimal()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 2)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, 1.0x] in
        GDPO.DisjunctionSet([[MOI.LessThan(1.0)], [MOI.LessThan(2.0)]]))
    @objective(model, Min, -x^2)
    optimize!(model)
    @test termination_status(model) != MOI.OPTIMAL
    @test termination_status(model) in
        (MOI.LOCALLY_SOLVED, MOI.ITERATION_LIMIT, MOI.TIME_LIMIT)
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test occursin("LOA finished", raw_status(model))
end

# A feasibility model (no objective) minimizes a constant zero.
function test_feasibility_sense()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    set_start_value(x, 6.0)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(5.0)], [MOI.LessThan(1.0)]]))
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test value(x) >= 5.0 - 1e-6 || value(x) <= 1.0 + 1e-6
end

# A linear Interval disjunct row reaches the master as two one-sided
# indicator constraints (Indicator{A}(Interval) has no MILP bridge).
function test_linear_interval_disjunct()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, 1.0z[1], 1.0z[2], 1.0x, 1.0x] in
        GDPO.DisjunctionSet([
            [MOI.LessThan(2.0)], [MOI.Interval(3.0, 7.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 7.0 atol = 1e-4
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# A second solve on the same optimizer must not leak the first solve's
# bound, gap, or incumbent.
function test_reoptimize_resets_results()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(2.0)], [MOI.GreaterThan(5.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 2.0 atol = 1e-5
    @constraint(model, 1.0x >= 20)
    optimize!(model)
    @test termination_status(model) == MOI.INFEASIBLE
    @test result_count(model) == 0
    @test objective_bound(model) == -Inf
    @test isnan(relative_gap(model))
end

# An unsupported constraint type is rewritten by the JuMP bridge layer
# into supported rows instead of erroring inside the solve.
function test_bridged_vector_constraint()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1.0x - 5.0] in MOI.Nonnegatives(1))
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-4
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

################################################################################
#                            UNIT TESTS
################################################################################
function test_cut_term_directions()
    x = MOI.VariableIndex(1)
    lin = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(2.0, x)], 1.0)
    @test length(GDPO._oa_cut_terms(MOI.LessThan(5.0), lin)) == 1
    @test length(GDPO._oa_cut_terms(MOI.GreaterThan(5.0), lin)) == 1
    @test length(GDPO._oa_cut_terms(MOI.EqualTo(5.0), lin)) == 2
    @test length(GDPO._oa_cut_terms(MOI.Interval(0.0, 5.0), lin)) == 2
    less = only(GDPO._oa_cut_terms(MOI.LessThan(5.0), lin))
    @test less.constant == -4.0
    greater = only(GDPO._oa_cut_terms(MOI.GreaterThan(5.0), lin))
    @test greater.constant == 4.0
    @test only(greater.terms).coefficient == -2.0
end

function test_activation_binary()
    z = MOI.VariableIndex(1)
    saf(a, c) = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(a, z)], c)
    @test GDPO._activation_binary(saf(1.0, 0.0)) == (z, true)
    @test GDPO._activation_binary(saf(-1.0, 1.0)) == (z, false)
    @test_throws ErrorException GDPO._activation_binary(saf(2.0, 0.0))
    @test_throws ErrorException GDPO._activation_binary(saf(1.0, 3.0))
end

function test_sense_primitives()
    @test GDPO._penalty_sign(MOI.MIN_SENSE) == 1
    @test GDPO._penalty_sign(MOI.MAX_SENSE) == -1
    @test GDPO._worst_objective(MOI.MIN_SENSE) == Inf
    @test GDPO._worst_objective(MOI.MAX_SENSE) == -Inf
    @test GDPO._is_better(MOI.MIN_SENSE, 1.0, 2.0)
    @test GDPO._is_better(MOI.MAX_SENSE, 2.0, 1.0)
    @test GDPO._gap(MOI.MIN_SENSE, 5.0, 3.0) == 2.0
    @test GDPO._gap(MOI.MAX_SENSE, 3.0, 5.0) == 2.0
end

function test_linearize_quadratic()
    x = MOI.VariableIndex(1)
    func = MOI.ScalarQuadraticFunction(
        [MOI.ScalarQuadraticTerm(2.0, x, x)],
        MOI.ScalarAffineTerm{Float64}[], 0.0)  # x^2
    linearizer = GDPO._Linearizer()
    lin = GDPO._linearize(linearizer, func, Dict(x => 3.0))
    # x^2 at x = 3: 9 + 6 (x - 3) = 6 x - 9
    @test lin.constant ≈ -9.0
    @test only(lin.terms).coefficient ≈ 6.0
    @test length(linearizer.evaluators) == 1
    GDPO._linearize(linearizer, func, Dict(x => 4.0))
    @test length(linearizer.evaluators) == 1
end

# Nested disjunction: the inner disjunction's activation component is
# the outer indicator, so it selects a mode only while the outer
# disjunct is active and is vacuous otherwise.
function test_nested_disjunction()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, zout[1:2], Bin)
    @variable(model, zin[1:2], Bin)
    @constraint(model, [1, zout[1], zout[2], x] in GDPO.DisjunctionSet([
        [MOI.LessThan(2.0)], MOI.AbstractScalarSet[]]))
    @constraint(model, [1.0zout[2], zin[1], zin[2], x^2, x] in
        GDPO.DisjunctionSet([[MOI.LessThan(25.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(zout[2]) ≈ 1.0 atol = 1e-5
    @test value(zin[2]) ≈ 1.0 atol = 1e-5
end

# When the parent disjunct is off, the inner indicators sum to zero
# and the inner rows impose nothing.
function test_nested_disjunction_vacuous()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, zout[1:2], Bin)
    @variable(model, zin[1:2], Bin)
    @constraint(model, [1, zout[1], zout[2], x] in GDPO.DisjunctionSet([
        [MOI.LessThan(2.0)], MOI.AbstractScalarSet[]]))
    @constraint(model, [1.0zout[2], zin[1], zin[2], x, x^2] in
        GDPO.DisjunctionSet([
            [MOI.GreaterThan(5.0)], [MOI.GreaterThan(36.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 0.0 atol = 1e-4
    @test value(zout[1]) ≈ 1.0 atol = 1e-5
    @test value(zin[1]) + value(zin[2]) ≈ 0.0 atol = 1e-5
end

@testset "LOA loop" begin
    test_linear_disjunction()
    test_nested_disjunction()
    test_nested_disjunction_vacuous()
    test_quadratic_objective()
    test_max_sense_linear()
    test_two_disjunctions()
    test_nonlinear_global()
    test_nonlinear_equality_global()
    test_nonlinear_equality_disjunct()
    test_nonlinear_interval_disjunct()
    test_complement_indicator()
    test_nlpf_disabled()
    test_bigm_master_gating()
    test_integer_options()
    test_non_indicator_binary()
    test_infeasible_no_incumbent()
    test_time_limit()
    test_nonconvex_never_optimal()
    test_feasibility_sense()
    test_linear_interval_disjunct()
    test_reoptimize_resets_results()
    test_bridged_vector_constraint()
end

@testset "LOA units" begin
    test_cut_term_directions()
    test_activation_binary()
    test_sense_primitives()
    test_linearize_quadratic()
end
