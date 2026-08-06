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

# Genuinely nonlinear (non-quadratic) global rows stay
# `ScalarNonlinearFunction`s through demotion and the linearizer walks
# their argument trees (variable and affine arguments).
function test_nonlinear_exp_global()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, exp(x) <= exp(5.0))
    @constraint(model, exp(x + 1.0) <= exp(6.5))
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# `use_nlpf = false` with an infeasible combination: the solve keeps
# only the no-good cut and still finishes from the other disjunct.
function test_nlpf_disabled_infeasible_combination()
    model = Model(_loa_optimizer(use_nlpf = false))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, 1.0x >= 5)
    @constraint(model, [1, 1.0z[1], 1.0z[2], x^2, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(9.0)], [MOI.LessThan(64.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# The NLPF slacks GreaterThan rows and copies the global linear rows
# when restoring an infeasible combination.
function test_nlpf_slacked_rows()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, 1.0x >= 4)
    @constraint(model, x^2 >= 25)
    @constraint(model, [1, 1.0z[1], 1.0z[2], x^2, x^2] in
        GDPO.DisjunctionSet([[MOI.LessThan(9.0)], [MOI.LessThan(64.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(z[2]) ≈ 1.0 atol = 1e-5
end

# A quadratic-typed global row with no quadratic terms demotes to
# affine and lands in the master directly.
function test_demoted_affine_global()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, zero(QuadExpr) + 1.0x <= 6)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 6.0 atol = 1e-4
end

# An all-variable disjunction arrives as `VectorOfVariables`; its raw
# variable rows are promoted to affine rows rather than bounds.
function test_vector_of_variables_disjunction()
    model = Model(_loa_optimizer())
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, zout[1:2], Bin)
    @variable(model, zin[1:2], Bin)
    @constraint(model, [1, zout[1], zout[2], x] in GDPO.DisjunctionSet([
        [MOI.LessThan(2.0)], MOI.AbstractScalarSet[]]))
    @constraint(model, [zout[2], zin[1], zin[2], x, x] in
        GDPO.DisjunctionSet([[MOI.LessThan(5.0)], [MOI.LessThan(8.0)]]))
    @objective(model, Max, x)
    optimize!(model)
    @test termination_status(model) == MOI.LOCALLY_SOLVED
    @test objective_value(model) ≈ 8.0 atol = 1e-4
    @test value(zout[2]) ≈ 1.0 atol = 1e-5
    @test value(zin[2]) ≈ 1.0 atol = 1e-5
end

# Zero iteration budgets exit before any solve, without an incumbent.
function test_iteration_limit_no_incumbent()
    model = Model(_loa_optimizer(set_cover_max_iter = 0, max_iter = 0))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    @test result_count(model) == 0
end

# `max_iter = 0` keeps the set-covering incumbent but produces no
# master bound.
function test_iteration_limit_with_incumbent()
    model = Model(_loa_optimizer(max_iter = 0))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(2.0)], [MOI.GreaterThan(5.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.ITERATION_LIMIT
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test occursin("limit hit", raw_status(model))
    @test occursin("no bound", raw_status(model))
    @test objective_bound(model) == -Inf
end

# An unbounded master (no OA cuts yet bound alpha_oa) surfaces its
# status instead of looping.
function test_master_abnormal_status()
    model = Model(_loa_optimizer(set_cover_max_iter = 0))
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.LessThan(3.0)], [MOI.LessThan(7.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.OTHER_LIMIT
    @test result_count(model) == 0
    @test occursin("master solve finished", raw_status(model))
end

################################################################################
#                            MOCK SOLVER
################################################################################
# Delegates every MOI call to a wrapped optimizer, but sleeps
# `sleep_time` seconds in each solve and, from solve `fail_from` on,
# skips the inner solve and reports `fail_status` with no solution.
# Deterministic triggers for the deadline and abnormal-master paths.
mutable struct MockSolver <: MOI.AbstractOptimizer
    inner::MOI.AbstractOptimizer
    sleep_time::Float64
    fail_from::Int
    fail_status::MOI.TerminationStatusCode
    solves::Int
    failing::Bool
end

function MockSolver(
    factory;
    sleep_time::Float64 = 0.0,
    fail_from::Int = typemax(Int),
    fail_status::MOI.TerminationStatusCode = MOI.NODE_LIMIT
    )
    return MockSolver(MOI.instantiate(factory), sleep_time, fail_from,
        fail_status, 0, false)
end

function MOI.optimize!(model::MockSolver)
    model.solves += 1
    model.sleep_time > 0 && sleep(model.sleep_time)
    model.failing = model.solves >= model.fail_from
    model.failing || MOI.optimize!(model.inner)
    return
end

const _WrappedAttr = Union{MOI.AbstractModelAttribute,
    MOI.AbstractOptimizerAttribute}
const _WrappedIndexAttr = Union{MOI.AbstractVariableAttribute,
    MOI.AbstractConstraintAttribute}
const _WrappedIndex = Union{MOI.VariableIndex, MOI.ConstraintIndex}

function MOI.get(model::MockSolver, attr::_WrappedAttr)
    if model.failing
        attr isa MOI.TerminationStatus && return model.fail_status
        attr isa MOI.PrimalStatus && return MOI.NO_SOLUTION
    end
    return MOI.get(model.inner, attr)
end

MOI.is_empty(model::MockSolver) = MOI.is_empty(model.inner)
MOI.empty!(model::MockSolver) = MOI.empty!(model.inner)
MOI.supports_incremental_interface(::MockSolver) = true
MOI.copy_to(model::MockSolver, src::MOI.ModelLike) =
    MOI.copy_to(model.inner, src)
MOI.add_variable(model::MockSolver) = MOI.add_variable(model.inner)
MOI.delete(model::MockSolver, index) = MOI.delete(model.inner, index)
MOI.is_valid(model::MockSolver, index) = MOI.is_valid(model.inner, index)

function MOI.add_constraint(
    model::MockSolver,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet
    )
    return MOI.add_constraint(model.inner, func, set)
end

function MOI.supports_constraint(
    model::MockSolver,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet}
    )
    return MOI.supports_constraint(model.inner, F, S)
end

MOI.supports(model::MockSolver, attr::_WrappedAttr) =
    MOI.supports(model.inner, attr)

MOI.set(model::MockSolver, attr::_WrappedAttr, value) =
    MOI.set(model.inner, attr, value)

function MOI.supports(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    I::Type{<:_WrappedIndex}
    )
    return MOI.supports(model.inner, attr, I)
end

function MOI.get(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    index::_WrappedIndex
    )
    return MOI.get(model.inner, attr, index)
end

function MOI.set(
    model::MockSolver,
    attr::_WrappedIndexAttr,
    index::_WrappedIndex,
    value
    )
    return MOI.set(model.inner, attr, index, value)
end

function _mock_time_limit_model(limit::Float64; kwargs...)
    factory = () -> GDPO.Optimizer(
        nlp_solver = () -> MockSolver(Ipopt.Optimizer; kwargs...),
        mip_solver = HiGHS.Optimizer,
        iteration_time_limit = limit)
    model = Model(factory)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(2.0)], [MOI.GreaterThan(5.0)]]))
    @objective(model, Min, x)
    return model
end

# The loop deadline passes right after the covering incumbent: the
# mock NLP sleeps past `iteration_time_limit`, so the main loop never
# starts and the incumbent is reported against the time limit. The
# first solve warms up the mock stack so compilation latency cannot
# eat the deadline before the covering pass runs.
function test_time_limit_with_incumbent()
    warmup = _mock_time_limit_model(Inf)
    optimize!(warmup)
    @test objective_value(warmup) ≈ 2.0 atol = 1e-5
    model = _mock_time_limit_model(0.5, sleep_time = 1.5)
    optimize!(model)
    @test termination_status(model) == MOI.TIME_LIMIT
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test occursin("limit hit", raw_status(model))
end

# The master finishes abnormally after an incumbent exists: the mock
# master reports a node limit on its second solve.
function test_master_abnormal_status_with_incumbent()
    factory = () -> GDPO.Optimizer(
        nlp_solver = Ipopt.Optimizer,
        mip_solver = () -> MockSolver(HiGHS.Optimizer, fail_from = 2))
    model = Model(factory)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, z[1:2], Bin)
    @constraint(model, [1, z[1], z[2], x, x] in GDPO.DisjunctionSet([
        [MOI.GreaterThan(2.0)], [MOI.GreaterThan(5.0)]]))
    @objective(model, Min, x)
    optimize!(model)
    @test termination_status(model) == MOI.OTHER_LIMIT
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test occursin("limit hit", raw_status(model))
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
    test_nonlinear_exp_global()
    test_nonlinear_equality_global()
    test_nonlinear_equality_disjunct()
    test_nonlinear_interval_disjunct()
    test_complement_indicator()
    test_nlpf_disabled()
    test_nlpf_disabled_infeasible_combination()
    test_nlpf_slacked_rows()
    test_demoted_affine_global()
    test_vector_of_variables_disjunction()
    test_bigm_master_gating()
    test_integer_options()
    test_non_indicator_binary()
    test_infeasible_no_incumbent()
    test_time_limit()
    test_iteration_limit_no_incumbent()
    test_iteration_limit_with_incumbent()
    test_master_abnormal_status()
    test_time_limit_with_incumbent()
    test_master_abnormal_status_with_incumbent()
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
