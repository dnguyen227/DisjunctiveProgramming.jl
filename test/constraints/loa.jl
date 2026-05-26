using HiGHS, Ipopt, Juniper

function test_loa_datatype()
    method = LOA(HiGHS.Optimizer)
    @test method.nlp_optimizer == HiGHS.Optimizer
    @test method.mip_optimizer == HiGHS.Optimizer
    @test method.max_iter == 10
    @test method.atol == 1e-6
    @test method.rtol == 1e-4
    @test method.M_value == 1e9
    @test method.max_slack == 1000.0
    @test method.OA_penalty_factor == 1000.0
    @test method.inner_method isa BigM
    @test method.inner_method.value == 1e9

    method = LOA(HiGHS.Optimizer; max_iter = 50, atol = 1e-8,
        rtol = 1e-6, M_value = 1e6, max_slack = 500.0,
        OA_penalty_factor = 200.0)
    @test method.max_iter == 50
    @test method.atol == 1e-8
    @test method.rtol == 1e-6
    @test method.M_value == 1e6
    @test method.max_slack == 500.0
    @test method.OA_penalty_factor == 200.0
    @test method.inner_method isa BigM
    @test method.inner_method.value == 1e6

    method = LOA(HiGHS.Optimizer; inner_method = MBM(HiGHS.Optimizer))
    @test method.inner_method isa MBM
end

function test_set_covering_combos()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    combos = DP._set_covering_combinations(model)

    # Should cover both Y[1] and Y[2]
    all_active = Set()
    for combo in combos
        for (ind, active) in combo
            active && push!(all_active, ind)
        end
    end
    @test length(all_active) == 2
end

function test_no_good_cut()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    DP.reformulate_model(model, BigM(1e9))
    master = DP.build_loa_master(
        model, LOA(HiGHS.Optimizer))
    master_model = master.model

    combo = Dict(Y[1] => true, Y[2] => false)

    num_cons_before = length(JuMP.all_constraints(
        master_model;
        include_variable_in_set_constraints = false))
    DP.avoid_combination(master.model, combo, master.binary_map)
    num_cons_after = length(JuMP.all_constraints(
        master_model;
        include_variable_in_set_constraints = false))

    @test num_cons_after == num_cons_before + 1
end

function test_loa_convergence_check()
    method = LOA(HiGHS.Optimizer; atol = 1e-6, rtol = 1e-4)

    sense = Val(MOI.MIN_SENSE)
    @test DP._loa_converged(1.0, 1.0, sense, method) == true
    @test DP._loa_converged(1.0, 0.9999, sense, method) == true
    @test DP._loa_converged(1.0, 0.5, sense, method) == false
    @test DP._loa_converged(1e-8, 0.0, sense, method) == true
end

function test_loa_reformulate_simple()
    model = GDPModel(HiGHS.Optimizer)
    @variable(model, 0 <= x <= 10)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 7, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, x)

    method = LOA(HiGHS.Optimizer)
    DP.reformulate_model(model, method)

    @test DP._ready_to_optimize(model)
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
    # x to 5. Verifies that `_add_global_oa_cuts` emits the
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

function test_loa_complement_indicator_nonlinear_disjunct()
    # Regression: complement-form indicators store `1 - y_base` (an
    # AffExpr) in `binary_map`. When the complement disjunct has a
    # nonlinear constraint, `add_disjunct_oa_cuts` calls `cut_info`
    # on that AffExpr; the AffExpr method must dispatch.
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

function test_linearize_nonlinear_exp()
    # exp(x) + y at (1, 2):
    # f = e + 2, ∇f = [e, 1]
    # linear: e*(x-1) + 1*(y-2) + (e+2) = e*x + y
    model = GDPModel()
    @variable(model, x)
    @variable(model, y)
    func = @expression(model, exp(x) + y)
    xk = Dict{JuMP.AbstractVariableRef, Float64}(
        x => 1.0, y => 2.0)
    id_map = Dict(x => x, y => y)
    lin = DP._linearize_at(func, xk, id_map)
    @test JuMP.constant(lin) ≈ 0.0 atol = 1e-8
    @test JuMP.coefficient(lin, x) ≈ exp(1.0) atol = 1e-8
    @test JuMP.coefficient(lin, y) ≈ 1.0 atol = 1e-8
end

function test_linearize_nonlinear_sin()
    # sin(x) at x = π/6:
    # f = 0.5, f' = cos(π/6) = √3/2
    # linear: 0.5 + (√3/2)(x - π/6)
    model = GDPModel()
    @variable(model, x)
    func = @expression(model, sin(x))
    xk = Dict{JuMP.AbstractVariableRef, Float64}(
        x => π / 6)
    id_map = Dict(x => x)
    lin = DP._linearize_at(func, xk, id_map)
    expected_const = 0.5 - (√3 / 2) * (π / 6)
    @test JuMP.constant(lin) ≈ expected_const atol = 1e-8
    @test JuMP.coefficient(lin, x) ≈ √3 / 2 atol = 1e-8
end

function test_linearize_nonlinear_multivar()
    # exp(x) * sin(y) at (1, π/2):
    # f = e*1 = e, ∂f/∂x = e*sin(π/2) = e, ∂f/∂y = e*cos(π/2) = 0
    # linear: e + e*(x-1) + 0*(y-π/2) = e*x
    model = GDPModel()
    @variable(model, x)
    @variable(model, y)
    func = @expression(model, exp(x) * sin(y))
    xk = Dict{JuMP.AbstractVariableRef, Float64}(
        x => 1.0, y => π / 2)
    id_map = Dict(x => x, y => y)
    lin = DP._linearize_at(func, xk, id_map)
    @test JuMP.constant(lin) ≈ 0.0 atol = 1e-8
    @test JuMP.coefficient(lin, x) ≈ exp(1.0) atol = 1e-8
    @test JuMP.coefficient(lin, y) ≈ 0.0 atol = 1e-8
end

function test_to_nlp_expr()
    model = GDPModel()
    @variable(model, x)
    @variable(model, y)
    idx = Dict(x => 1, y => 2)

    # NonlinearExpr
    nl = @expression(model, exp(x))
    e = DP._to_nlp_expr(nl, idx)
    @test e == Expr(:call, :exp, MOI.VariableIndex(1))

    # AffExpr
    aff = @expression(model, 2x + 3y + 1)
    e = DP._to_nlp_expr(aff, idx)
    @test e isa Expr
    @test e.head == :call && e.args[1] == :+

    # Number
    @test DP._to_nlp_expr(42, idx) == 42
end

@testset "LOA" begin
    test_loa_datatype()
    test_set_covering_combos()
    test_no_good_cut()
    test_loa_convergence_check()
    test_loa_reformulate_simple()
    test_loa_solve_simple()
    test_loa_solve_simple_with_mbm()
    test_loa_solve_two_disjunctions()
    test_loa_error_fallback()
    test_loa_nonlinear_global()
    test_loa_complement_indicator_nonlinear_disjunct()
    test_linearize_nonlinear_exp()
    test_linearize_nonlinear_sin()
    test_linearize_nonlinear_multivar()
    test_to_nlp_expr()
end
