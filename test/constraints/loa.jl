using HiGHS

function test_loa_datatype()
    method = LOA(HiGHS.Optimizer)
    @test method.optimizer == HiGHS.Optimizer
    @test method.max_iter == 100
    @test method.atol == 1e-6
    @test method.rtol == 1e-4
    @test method.M_value == 1e9

    method = LOA(HiGHS.Optimizer; max_iter = 50, atol = 1e-8, rtol = 1e-6, M_value = 1e6)
    @test method.max_iter == 50
    @test method.atol == 1e-8
    @test method.rtol == 1e-6
    @test method.M_value == 1e6
end

function test_collect_disjunction_info()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    info = DP._collect_disjunction_info(model)
    @test length(info.disjunction_indices) == 1
    @test length(info.indicators[info.disjunction_indices[1]]) == 2
end

function test_set_covering_combos()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    info = DP._collect_disjunction_info(model)
    method = LOA(HiGHS.Optimizer)
    combos = DP._set_covering_combos(info, method)

    # Should cover both Y[1] and Y[2]
    all_active = Set()
    for combo in combos
        for (ind, active) in combo
            if active
                push!(all_active, ind)
            end
        end
    end
    # Both indicators should appear as active in at least one combo
    @test length(all_active) == 2
end

function test_no_good_cut()
    model = GDPModel()
    @variable(model, x)
    @variable(model, Y[1:2], Logical)
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x >= 5, Disjunct(Y[2]))
    @disjunction(model, Y)

    info = DP._collect_disjunction_info(model)
    method = LOA(HiGHS.Optimizer)

    # Build a master and add a no-good cut
    master, ref_map, lv_map = DP.copy_gdp_model(model)
    JuMP.set_optimizer(master, HiGHS.Optimizer)
    DP.reformulate_model(master, BigM(1e9))

    master_maps = (
        ref_map = ref_map,
        lv_map = lv_map,
        ind_to_bin = DP._indicator_to_binary(master),
    )

    # Combo where Y[1] = true, Y[2] = false
    combo = Dict{Any, Bool}(Y[1] => true, Y[2] => false)

    num_cons_before = length(JuMP.all_constraints(master; include_variable_in_set_constraints = false))
    DP._add_no_good_cut_to_master!(master, master_maps, combo, info)
    num_cons_after = length(JuMP.all_constraints(master; include_variable_in_set_constraints = false))

    # Should have added exactly 1 constraint
    @test num_cons_after == num_cons_before + 1
end

function test_loa_convergence_check()
    method = LOA(HiGHS.Optimizer; atol = 1e-6, rtol = 1e-4)

    # Should converge when gap is small
    @test DP._loa_converged(1.0, 1.0, method) == true
    @test DP._loa_converged(1.0, 0.9999, method) == true
    @test DP._loa_converged(1.0, 0.5, method) == false

    # Absolute tolerance
    @test DP._loa_converged(1e-8, 0.0, method) == true
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

    # After reformulation, model should be ready to optimize
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

@testset "LOA" begin
    test_loa_datatype()
    test_collect_disjunction_info()
    test_set_covering_combos()
    test_no_good_cut()
    test_loa_convergence_check()
    test_loa_reformulate_simple()
    test_loa_solve_simple()
    test_loa_solve_two_disjunctions()
    test_loa_error_fallback()
end
