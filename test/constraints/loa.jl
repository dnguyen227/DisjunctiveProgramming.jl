using HiGHS, Ipopt, Juniper

function test_loa_datatype()
    method = LOA(HiGHS.Optimizer)
    @test method.nlp_optimizer == HiGHS.Optimizer
    @test method.mip_optimizer == HiGHS.Optimizer
    @test method.max_iter == 10
    @test method.set_cover_max_iter == 8
    @test method.M_value == 1e9
    @test method.max_slack == 1000.0
    @test method.oa_penalty == 1000.0
    @test method.convergence_tol == 1e-6
    @test method.slack_tol == 1e-4
    @test method.inner_method isa BigM
    @test method.inner_method.value == 1e9

    method = LOA(HiGHS.Optimizer; max_iter = 50, M_value = 1e6,
        max_slack = 500.0, oa_penalty = 200.0,
        convergence_tol = 1e-4, slack_tol = 1e-3)
    @test method.max_iter == 50
    @test method.M_value == 1e6
    @test method.max_slack == 500.0
    @test method.oa_penalty == 200.0
    @test method.convergence_tol == 1e-4
    @test method.slack_tol == 1e-3
    @test method.inner_method isa BigM
    @test method.inner_method.value == 1e6

    method = LOA(HiGHS.Optimizer; inner_method = MBM(HiGHS.Optimizer))
    @test method.inner_method isa MBM
end

function test_cover_disjuncts()
    # Only disjuncts owning a nonlinear constraint need covering: the two
    # `x^2` disjuncts do, the two linear `x` disjuncts do not.
    model = GDPModel()
    @variable(model, -5 <= x <= 5)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x^2 <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 1, Disjunct(W[1]))
    @constraint(model, x >= -1, Disjunct(W[2]))
    @disjunction(model, W)

    DP.reformulate_model(model, BigM(1e9))
    method = LOA(HiGHS.Optimizer)
    problem = DP.build_loa_problem(model, method)
    disjuncts = DP._cover_disjuncts(problem)

    # Two nonlinear disjuncts to cover; the linear W disjuncts omitted.
    @test length(disjuncts) == 2
    binary_map = DP._indicator_to_binary(model)
    underlying = Set(DP._underlying_binary(dref) for dref in disjuncts)
    @test underlying == Set([DP._underlying_binary(binary_map[Y[1]]),
        DP._underlying_binary(binary_map[Y[2]])])
end

function test_oa_cut_terms()
    # The `<= 0` cut directions per set, at a scalar linearization value
    # of 5: LessThan/GreaterThan give one signed direction, EqualTo and
    # Interval give both.
    @test DP._oa_cut_terms(MOI.LessThan(2.0), 5.0) == (3.0,)
    @test DP._oa_cut_terms(MOI.GreaterThan(2.0), 5.0) == (-3.0,)
    @test DP._oa_cut_terms(MOI.EqualTo(2.0), 5.0) == (3.0, -3.0)
    @test DP._oa_cut_terms(MOI.Interval(1.0, 4.0), 5.0) == (1.0, -4.0)
end

function test_is_linear_F()
    # Scalar and vector variable-ref / affine function types are linear
    # (copied into the master at build time); anything else, e.g. a
    # quadratic, is nonlinear and enters only as an OA cut.
    @test DP._is_linear_F(JuMP.VariableRef)
    @test DP._is_linear_F(JuMP.AffExpr)
    @test DP._is_linear_F(Vector{JuMP.VariableRef})
    @test DP._is_linear_F(Vector{JuMP.AffExpr})
    @test !DP._is_linear_F(JuMP.QuadExpr)
end

function test_no_good_cut()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    DP.reformulate_model(model, BigM(1e9))
    method = LOA(HiGHS.Optimizer)
    problem = DP.build_loa_problem(model, method)
    master = DP._build_loa_master(problem, method)
    master_model = master.model

    binary_map = DP._indicator_to_binary(model)
    combo = Dict(
        DP._underlying_binary(binary_map[Y[1]]) =>
            DP._underlying_value(binary_map[Y[1]], true),
        DP._underlying_binary(binary_map[Y[2]]) =>
            DP._underlying_value(binary_map[Y[2]], false))

    num_cons_before = length(JuMP.all_constraints(
        master_model;
        include_variable_in_set_constraints = false))
    cref = DP.avoid_combination(master.model, combo, master.variable_map)
    num_cons_after = length(JuMP.all_constraints(
        master_model;
        include_variable_in_set_constraints = false))

    @test num_cons_after == num_cons_before + 1
    # The cut is (1 - y1) + y2 >= 1, i.e. normalized -y1 + y2 >= 0:
    # excludes exactly the (Y1 active, Y2 inactive) combination.
    y1 = master.variable_map[binary_map[Y[1]]]
    y2 = master.variable_map[binary_map[Y[2]]]
    @test JuMP.normalized_coefficient(cref, y1) == -1.0
    @test JuMP.normalized_coefficient(cref, y2) == 1.0
    @test JuMP.normalized_rhs(cref) == 0.0
end

function test_loa_reformulate_simple()
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)

    method = LOA(HiGHS.Optimizer)
    DP.reformulate_model(model, method)

    @test DP._ready_to_optimize(model)
    # The committed model solves to the LOA incumbent: x = 7 via Y[2].
    JuMP.optimize!(model, ignore_optimize_hook = true)
    @test objective_value(model) ≈ 7.0 atol = 1e-6
    @test value(x) ≈ 7.0 atol = 1e-6
end

function test_loa_solve_simple()
    # Simple GDP: max x s.t. (x <= 3) OR (x <= 7), 0 <= x <= 10
    # Optimal: x = 7 (select second disjunct)
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)

    optimize!(model, gdp_method = LOA(HiGHS.Optimizer))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol=1e-4
end

function test_loa_solve_simple_with_mbm()
    # Same GDP as test_loa_solve_simple, but inner_method = MBM so the
    # NLP model is built with per-constraint tight Ms instead of BigM.
    # Optimum unchanged: x = 7 via the second disjunct.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)

    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        inner_method = MBM(HiGHS.Optimizer)))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol=1e-4
end

function test_loa_solve_two_disjunctions()
    # Two disjunctions: max x + z
    # D1: (x <= 3) OR (x <= 7)
    # D2: (z <= 2) OR (z <= 5)
    # Optimal: x = 7, z = 5
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= z <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, z <= 2, Disjunct(W[1]))
    @constraint(model, z <= 5, Disjunct(W[2]))
    @disjunction(model, W)
    @objective(model, Max, x + z)

    optimize!(model, gdp_method = LOA(HiGHS.Optimizer))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 12.0 atol=1e-4
end

function test_loa_error_fallback()
    method = LOA(HiGHS.Optimizer)
    @test_throws ErrorException DP.reformulate_model(42, method)
end

function test_loa_nonlinear_global()
    # max x s.t. x^2 <= 25 (global), (x <= 3) ∨ (x <= 8), 0 <= x <= 10.
    # Disjunct Y[2] permits x up to 8 but the global x^2 <= 25 bounds
    # x to 5. Verifies that the global-cut pass emits the
    # linearization of the global into the master without breaking
    # the loop. Result must hit the global-binding optimum.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @constraint(model, x^2 <= 25)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-3
end

function test_loa_nonlinear_equality_global()
    # max x s.t. x^2 == 25 (global nonlinear equality), (x <= 3) ∨
    # (x <= 8), 0 <= x <= 10. The Y[1] seed is NLP-infeasible (x <= 3
    # contradicts x = 5), so NLPF slacks the inequality while keeping
    # the equality exact. The equality emits BOTH cut directions into
    # the master. Optimum: x = 5 via Y[2].
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = GDPModel(ipopt)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @constraint(model, x^2 == 25)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(ipopt; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(x) ≈ 5.0 atol = 1e-3
    @test value(Y[2]) ≈ 1.0 atol = 1e-6
end

function test_loa_nonlinear_equality_disjunct()
    # Nonlinear equality inside a disjunct: Y1: x <= 3, Y2: x^2 == 64.
    # The disjunct cut emits both gated directions. Optimum: x = 8.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 == 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2]) ≈ 1.0 atol = 1e-6
end

function test_loa_nonlinear_interval_disjunct()
    # Nonlinear Interval constraint inside a disjunct: Y1: x <= 3,
    # Y2: 36 <= x^2 <= 64. The disjunct cut emits both gated
    # directions. Optimum: x = 8 via Y2.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, 36 <= x^2 <= 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-3
    @test value(Y[2]) ≈ 1.0 atol = 1e-6
end

function test_loa_restores_prior_time_limit()
    # A time limit set by the user before LOA must survive the loop's
    # per-solve caps when only `iteration_time_limit` is finite (the
    # `_restore_time_limit(::Real)` path).
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    set_time_limit_sec(model, 90.0)
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        iteration_time_limit = 60.0, time_limit = Inf))
    @test time_limit_sec(model) == 90.0
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol = 1e-4
end

function test_loa_limit_hit_report()
    # Stop the main loop on `max_iter = 1` after the master has produced
    # a bound: the report must label the run "limit hit" (not converged),
    # and the single loop iteration still finds the off-diagonal optimum.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= z <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 4, Disjunct(Y[1]))
    @constraint(model, x >= 6, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, z <= 4, Disjunct(W[1]))
    @constraint(model, z >= 6, Disjunct(W[2]))
    @disjunction(model, W)
    @objective(model, Min, x - z)
    method = LOA(HiGHS.Optimizer; max_iter = 1)
    @test_logs (:info, r"limit hit") match_mode = :any begin
        DP.reformulate_model(model, method)
    end
    JuMP.optimize!(model, ignore_optimize_hook = true)
    @test objective_value(model) ≈ -10.0 atol = 1e-4
end

function test_loa_complement_indicator_nonlinear_disjunct()
    # Regression: complement-form indicators carry `1 - y_base` (an
    # AffExpr) as their binary reference. When the complement disjunct has a
    # nonlinear constraint, the disjunct-cut pass feeds that AffExpr
    # straight into the OA cut gating term `M(1 - binary)`, which must
    # accept an AffExpr binary (not just a plain variable ref).
    #
    # Setup: 0 <= x <= 10, Y2 ≡ ¬Y1.
    #   Disjunct(Y1): x <= 3     (linear)
    #   Disjunct(Y2): x^2 <= 64  (nonlinear, optimum: x = 8 with Y2)
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y1, Logical)
    @variable(model, Y2, Logical, logical_complement = Y1)
    @constraint(model, x <= 3, Disjunct(Y1))
    @constraint(model, x^2 <= 64, Disjunct(Y2))
    @disjunction(model, [Y1, Y2])
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-3
end

function test_loa_nlpf_infeasible_disjunct()
    # Y1 disjunct constraint x^2 >= 200 is NLP-infeasible against the
    # variable bound x in [0, 10] (max x^2 = 100). The primary NLP at
    # the Y1=true seed therefore fails and NLPF (V&G 1990 eq. 8) must
    # kick in: slack `u` on the GreaterThan, minimize u, return the
    # x = 10 / u = 100 primal as the linearization site for the OA cut
    # on x^2 >= 200. With the master's per-cut penalized slack
    # absorbing the resulting (invalid in isolation) linearization and
    # the no-good cut forbidding Y1, LOA should still converge to the
    # Y2=true optimum x = 5.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x^2 >= 200, Disjunct(Y[1]))
    @constraint(model, x <= 5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-3
    @test value(Y[2]) ≈ 1.0 atol = 1e-6
    @test value(Y[1]) ≈ 0.0 atol = 1e-6
end

function test_loa_solve_nlp_infeasible_fallback()
    # `_solve_nlp` fallback: both the primary NLP and NLPF are
    # infeasible at the fixed combination, so it returns the infeasible
    # sentinel (objective = Inf, feasible = false, empty linearization
    # point, combination echoed back unchanged).
    #
    # The global x^2 == 400 is unsatisfiable under 0 <= x <= 10 (needs
    # x = 20). It is NONLINEAR, so it never enters the master and the
    # primary NLP is the one that catches it as infeasible. It is an
    # EQUALITY, so NLPF does not slack it and NLPF is infeasible too.
    # This is what distinguishes the path from
    # test_loa_no_feasible_incumbent, whose LINEAR equality makes the
    # master infeasible first, short-circuiting the NLP solve entirely.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = GDPModel()
    @variable(model, 0 <= x <= 10)
    @constraint(model, x^2 == 400)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)

    DP.reformulate_model(model, BigM(1e9))
    method = LOA(ipopt)
    problem = DP.build_loa_problem(model, method)
    DP._relax_binaries(problem)
    JuMP.set_optimizer(problem.nlp, method.nlp_optimizer)
    JuMP.set_silent(problem.nlp)

    binary_map = DP._indicator_to_binary(model)
    combination = Dict(
        DP._underlying_binary(binary_map[Y[1]]) => false,
        DP._underlying_binary(binary_map[Y[2]]) => true)

    result = DP._solve_nlp(problem, combination, method)
    @test result.feasible == false
    @test result.objective == Inf
    @test isempty(result.linearization_point)
    @test result.combination === combination
end

function test_loa_sense_primitives()
    # Sense-dispatched primitives the main loop reads through. The
    # finite/infinite solve tests cover every branch except the
    # minimize-sense gap (no minimizing model reaches a master bound in
    # those tests), so pin all six here for both senses directly.
    minv = Val(MOI.MIN_SENSE)
    maxv = Val(MOI.MAX_SENSE)
    @test DP._penalty_sign(minv) == 1
    @test DP._penalty_sign(maxv) == -1
    @test DP._worst_objective(minv) == Inf
    @test DP._worst_objective(maxv) == -Inf
    @test DP._is_better(minv, 3.0, 5.0)
    @test !DP._is_better(minv, 5.0, 3.0)
    @test DP._is_better(maxv, 5.0, 3.0)
    @test !DP._is_better(maxv, 3.0, 5.0)
    @test DP._gap(minv, 5.0, 3.0) == 2.0
    @test DP._gap(maxv, 3.0, 5.0) == 2.0
end

function test_loa_iteration_loop()
    # Force the master/NLP refinement loop to run its body rather than
    # exhaust the master on the seeds or converge on the first master
    # solve. Two disjunctions over disjoint x/z half-lines put the
    # optimum on an OFF-diagonal combination the set-covering seeds (the
    # diagonal (Y1,W1) and (Y2,W2)) never try. After seeding, the master
    # stays feasible and its bound beats the seed incumbent, so LOA
    # enters the loop body, extracts the off-diagonal combination, solves
    # its NLP, and updates the incumbent before converging. `Min` sense
    # also drives the minimize-sense gap path.
    #   min x - z
    #   D1: (x <= 4) [Y1]  v  (x >= 6) [Y2]
    #   D2: (z <= 4) [W1]  v  (z >= 6) [W2]
    # Optimum: Y1 & W2 -> x = 0, z = 10, obj = -10 (off-diagonal).
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= z <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 4, Disjunct(Y[1]))
    @constraint(model, x >= 6, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, z <= 4, Disjunct(W[1]))
    @constraint(model, z >= 6, Disjunct(W[2]))
    @disjunction(model, W)
    @objective(model, Min, x - z)
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ -10.0 atol = 1e-4
    @test value(Y[1]) ≈ 1.0 atol = 1e-6
    @test value(W[2]) ≈ 1.0 atol = 1e-6
end

function test_loa_time_limits()
    # Exercise the wall-clock budget paths. A finite `time_limit`
    # (overall cap) drives `_cap_remaining_time` during the subproblem
    # solves and the final-solve budget reset; a finite
    # `iteration_time_limit` (loop-only budget) drives the post-loop
    # time-limit restore from no prior limit (the `::Nothing` path).
    # Both limits are generous: they govern the path taken, not the
    # result.
    function build_model()
        model = GDPModel(HiGHS.Optimizer)
        set_silent(model)
        @variable(model, 0 <= x <= 10)
        @variable(model, Y[1:2], Logical)
        @constraint(model, x <= 3, Disjunct(Y[1]))
        @constraint(model, x <= 7, Disjunct(Y[2]))
        @disjunction(model, Y)
        @objective(model, Max, x)
        return model
    end

    model = build_model()
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer; time_limit = 60.0))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol = 1e-4

    model = build_model()
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        iteration_time_limit = 60.0, time_limit = Inf))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol = 1e-4
end

function test_loa_no_feasible_incumbent()
    # Every disjunct combination is NLP-infeasible: the global x == 20
    # contradicts the x <= 10 bound under any selection. NLPF does not
    # slack equalities, so NLPF also fails (no values) and `_solve_nlp`
    # returns its infeasible fallback. No seed or loop NLP yields a
    # feasible incumbent, so `best_result` stays `nothing`, the master
    # is infeasible (no bound), and LOA exits on the "no feasible
    # incumbent" warning from `_report_loa_gap`.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @constraint(model, x == 20)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    method = LOA(HiGHS.Optimizer)
    @test_logs (:warn, r"no feasible incumbent") match_mode = :any begin
        DP.reformulate_model(model, method)
    end
    @test DP._ready_to_optimize(model)
end

function test_loa_hull_linear()
    # Same GDP as test_loa_solve_simple, but inner_method = Hull so the
    # NLP and master carry disaggregated variables and the disjunct
    # cuts are convex-hull cuts rather than big-M. Optimum unchanged:
    # x = 7 via the second disjunct.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        inner_method = Hull()))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 7.0 atol = 1e-4
end

function test_loa_hull_two_disjunctions()
    # Two independent disjunctions under Hull; optimum x = 7, z = 5.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, 0 <= z <= 10)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, z <= 2, Disjunct(W[1]))
    @constraint(model, z <= 5, Disjunct(W[2]))
    @disjunction(model, W)
    @objective(model, Max, x + z)
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        inner_method = Hull()))
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 12.0 atol = 1e-4
end

function test_loa_hull_nonlinear_global()
    # max x s.t. x^2 <= 25 (global), (x <= 3) ∨ (x <= 8) under Hull.
    # Linear disjuncts enter the master as exact Hull perspectives; the
    # global nonlinear enters via global OA cuts. Optimum: x = 5.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @constraint(model, x^2 <= 25)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-3
end

function test_loa_hull_nonlinear_disjunct()
    # Nonlinear disjunct constraint under Hull: this is the case big-M
    # and Hull genuinely differ on. 0 <= x <= 10, Y1: x <= 3,
    # Y2: x^2 <= 64. The convex-hull OA cut disaggregates the
    # linearization of x^2. Optimum: x = 8 via Y2.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 <= 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-3
end

function test_loa_hull_complement_nonlinear()
    # Hull with a complement indicator Y2 ≡ ¬Y1 and a nonlinear
    # disjunct under Y2. The disaggregator is keyed by the complement
    # binary `1 - y1` (an AffExpr); the cut must disaggregate against
    # it. Optimum: x = 8 via Y2.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = GDPModel(juniper)
    set_silent(model)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y1, Logical)
    @variable(model, Y2, Logical, logical_complement = Y1)
    @constraint(model, x <= 3, Disjunct(Y1))
    @constraint(model, x^2 <= 64, Disjunct(Y2))
    @disjunction(model, [Y1, Y2])
    @objective(model, Max, x)
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-3
end

function test_loa_hull_nested_disaggregation_map()
    # The inner disjunction (over y) is a disjunct of the outer (gated by
    # z[1]). Hull's nested redispatch must propagate the LOA
    # disaggregation map so the inner disaggregations are recorded, else
    # LOA-Hull emits the nested cut on the aggregated variable.
    model = GDPModel()
    @variable(model, 0 <= x <= 10)
    @variable(model, y[1:2], Logical)
    @variable(model, z[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(y[1]))
    @constraint(model, x <= 8, Disjunct(y[2]))
    @disjunction(model, y, Disjunct(z[1]))
    @constraint(model, x <= 1, Disjunct(z[2]))
    @disjunction(model, z)
    disaggregations = Dict{Any, Any}()
    DP.reformulate_model(model, Hull(; disaggregation_map = disaggregations))
    @test haskey(disaggregations, (x, y[1]))
    @test haskey(disaggregations, (x, y[2]))
end

function test_reformulate_after_loa_restores_binaries()
    # LOA relaxes the logical binaries and fixes them to its incumbent, so
    # a later reformulation of the same model must restore binary
    # integrality. Regression: a nested disjunction's tight-M looked up a
    # relaxed binary's bounds and threw a KeyError.
    model = GDPModel(HiGHS.Optimizer)
    set_silent(model)
    @variable(model, 1 <= x <= 9)
    @variable(model, Y[1:2], Logical)
    @variable(model, W[1:2], Logical)
    @constraint(model, 1 <= x <= 2, Disjunct(W[1]))
    @constraint(model, 2 <= x <= 3, Disjunct(W[2]))
    @constraint(model, x >= 8, Disjunct(Y[2]))
    @disjunction(model, [W[1], W[2]], Disjunct(Y[1]))
    @disjunction(model, [Y[1], Y[2]])
    @objective(model, Max, x)

    optimize!(model, gdp_method = LOA(HiGHS.Optimizer))
    bins = [v for (_, v) in DP._indicator_to_binary(model)
        if v isa JuMP.VariableRef]
    @test all(!JuMP.is_binary, bins)          # LOA left them relaxed

    @test optimize!(model, gdp_method = BigM()) isa Nothing
    @test all(JuMP.is_binary, bins)           # restored by reformulation
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ 9
end

@testset "LOA" begin
    test_loa_datatype()
    test_cover_disjuncts()
    test_oa_cut_terms()
    test_is_linear_F()
    test_no_good_cut()
    test_loa_reformulate_simple()
    test_loa_solve_simple()
    test_loa_solve_simple_with_mbm()
    test_loa_solve_two_disjunctions()
    test_loa_error_fallback()
    test_loa_nonlinear_global()
    test_loa_nonlinear_equality_global()
    test_loa_nonlinear_equality_disjunct()
    test_loa_nonlinear_interval_disjunct()
    test_loa_restores_prior_time_limit()
    test_loa_limit_hit_report()
    test_loa_complement_indicator_nonlinear_disjunct()
    test_loa_nlpf_infeasible_disjunct()
    test_loa_solve_nlp_infeasible_fallback()
    test_loa_sense_primitives()
    test_loa_iteration_loop()
    test_loa_time_limits()
    test_loa_no_feasible_incumbent()
    test_loa_hull_linear()
    test_loa_hull_two_disjunctions()
    test_loa_hull_nonlinear_global()
    test_loa_hull_nonlinear_disjunct()
    test_loa_hull_complement_nonlinear()
    test_loa_hull_nested_disaggregation_map()
    test_reformulate_after_loa_restores_binaries()
end
