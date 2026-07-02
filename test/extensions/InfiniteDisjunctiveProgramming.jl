using InfiniteOpt, HiGHS, Ipopt, Juniper
import DisjunctiveProgramming as DP

# Helper to access internal function
const IDP = Base.get_extension(DP, :InfiniteDisjunctiveProgramming)

function test_infiniteopt_extension()
    # Initialize the model
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_attribute(model, MOI.Silent(), true)

    # Create the infinite variables
    I = 1:4
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 100)
    @variable(model, 0 <= g[I] <= 10, Infinite(t))

    # Add the disjunctions and their indicator variables
    @variable(model, G[I, 1:2], InfiniteLogical(t))
    @test all(isa.(@constraint(model, [i ∈ I, j ∈ 1:2], 0 <= g[i], 
        Disjunct(G[i, 1])), DisjunctConstraintRef{InfiniteModel})
        )
    @test all(isa.(@constraint(model, [i ∈ I, j ∈ 1:2], g[i] <= 0, 
        Disjunct(G[i, 2])), DisjunctConstraintRef{InfiniteModel})
        )
    @test all(isa.(@disjunction(model, [i ∈ I], G[i, :]), 
        DisjunctionRef{InfiniteModel})
        )

    # Add the logical propositions
    @variable(model, W, InfiniteLogical(t))
    @test @constraint(model, G[1, 1] ∨ G[2, 1] ∧ G[3, 1] == W := true) isa 
        LogicalConstraintRef{InfiniteModel}
    @constraint(model, 𝔼(binary_variable(W), t) >= 0.95)

    # Reformulate and solve 
    @test optimize!(model, gdp_method = Hull()) isa Nothing

    # check the results
    @test all(value(W))
end

function test_infinite_gdp_model_creation()
    model = InfiniteGDPModel()
    @test model isa InfiniteModel
    @test is_gdp_model(model)
    
end

function test_infinite_logical()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    
    @variable(model, y, InfiniteLogical(t))
    @test y isa DP.LogicalVariableRef{InfiniteModel}
    @test binary_variable(y) isa InfiniteOpt.GeneralVariableRef
end

function test_is_parameter()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s[1:2] ∈ [0, 1], independent = true)
    @finite_parameter(model, p == 1.0)
    @variable(model, x, Infinite(t))
    @variable(model, y)
    
    # Test DependentParameterRef
    @test IDP._is_parameter(t) == true
    
    # Test IndependentParameterRef
    @test IDP._is_parameter(s[1]) == true
    
    # Test FiniteParameterRef
    @test IDP._is_parameter(p) == true
    
    # Test non-parameter variables (else branch)
    @test IDP._is_parameter(x) == false
    @test IDP._is_parameter(y) == false
end

# _is_parameter on unwrapped concrete dispatch types
# (DependentParameterRef, IndependentParameterRef, FiniteParameterRef,
# ParameterFunctionRef, Any fallback).
function test_is_parameter_concrete_dispatches()
    model = InfiniteGDPModel()
    # Scalar + `independent = true` array both give IndependentParameterRef;
    # a default array parameter gives DependentParameterRef.
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s[1:2] ∈ [0, 1], independent = true)
    @infinite_parameter(model, q[1:2] ∈ [0, 1])
    @finite_parameter(model, p == 1.0)
    @variable(model, x, Infinite(t))
    @parameter_function(model, pf == t -> 2*t)
    dvr = InfiniteOpt.dispatch_variable_ref
    # Verify each ref hits the intended dispatch.
    @test dvr(t) isa InfiniteOpt.IndependentParameterRef
    @test dvr(s[1]) isa InfiniteOpt.IndependentParameterRef
    @test dvr(q[1]) isa InfiniteOpt.DependentParameterRef
    @test dvr(p) isa InfiniteOpt.FiniteParameterRef
    @test dvr(pf) isa InfiniteOpt.ParameterFunctionRef
    @test IDP._is_parameter(dvr(t)) == true
    @test IDP._is_parameter(dvr(s[1])) == true
    @test IDP._is_parameter(dvr(q[1])) == true
    @test IDP._is_parameter(dvr(p)) == true
    @test IDP._is_parameter(dvr(pf)) == true
    @test IDP._is_parameter(dvr(x)) == false    # Any fallback
end

function test_requires_disaggregation()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @finite_parameter(model, p == 1.0)
    @variable(model, x, Infinite(t))
    @variable(model, y)
    
    # Parameters should NOT require disaggregation
    @test DP.requires_disaggregation(t) == false
    @test DP.requires_disaggregation(p) == false
    
    # Variables SHOULD require disaggregation
    @test DP.requires_disaggregation(x) == true
    @test DP.requires_disaggregation(y) == true
end

function test_all_variables_infiniteopt()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, x, Infinite(t))
    @variable(model, y)
    @variable(model, dx, Infinite(t))
    @deriv(dx, t)
    
    all_vars = DP.collect_all_vars(model)
    @test x in all_vars
    @test y in all_vars
    @test dx in all_vars
    
    # Verify derivatives are included
    derivs = collect(InfiniteOpt.all_derivatives(model))
    for d in derivs
        @test d in all_vars
    end
end

function test_get_constant()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @finite_parameter(model, p == 2.0)
    @variable(model, x, Infinite(t))
    
    # Test expression with parameter terms (_is_parameter branch)
    expr2 = @expression(model, 3.0 + 2*t + x)
    constant2 = DP.get_constant(expr2)
    @test JuMP.constant(constant2) == 3.0
    @test haskey(constant2.terms, t)
    @test constant2.terms[t] == 2.0
    @test !haskey(constant2.terms, x)
    
    # Test expression with finite parameter
    expr3 = @expression(model, 1.0 + 3*p)
    constant3 = DP.get_constant(expr3)
    @test JuMP.constant(constant3) == 1.0
    @test haskey(constant3.terms, p)
    @test constant3.terms[p] == 3.0
end

function test_disaggregate_expression_infiniteopt()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, z, InfiniteLogical(t))
    @variable(model, w, Bin)
    
    bvrefs = DP._indicator_to_binary(model)
    bvref = bvrefs[z]
    
    vrefs = Set([x, w])
    DP._variable_bounds(model)[x] = DP.set_variable_bound_info(x, Hull())
    method = DP._Hull(Hull(1e-3), vrefs)
    DP._disaggregate_variables(model, z, vrefs, method)
    
    aff_bin = @expression(model, 2*w + 1)
    result_bin = DP.disaggregate_expression(model, aff_bin, bvref, method)
    @test haskey(result_bin.terms, w)
    
    aff_param = @expression(model, 3*t + 1)
    result_param = DP.disaggregate_expression(model, aff_param, bvref, method)
    @test result_param isa JuMP.GenericQuadExpr
    
    aff_expr = @expression(model, 2*x + 1)
    result_expr = DP.disaggregate_expression(model, aff_expr, bvref, method)
    dvref = method.disjunct_variables[x, bvref]
    @test result_expr == bvref + 2*dvref
    
    @variable(model, 0 <= y <= 5, Infinite(t))
    aff_not_disagg = @expression(model, 3*y + 1)
    result_not_disagg = DP.disaggregate_expression(model, aff_not_disagg, bvref, method)
    @test haskey(result_not_disagg.terms, y)
end

function test_variable_properties_infiniteopt()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, 1 <= x <= 10, Infinite(t), start = 5.0)
    @variable(model, y)
    
    props_x = DP.VariableProperties(x)
    @test props_x.info.has_lb == true
    @test props_x.info.lower_bound == 1.0
    @test props_x.info.has_ub == true
    @test props_x.info.upper_bound == 10.0
    @test props_x.name == "x"
    @test props_x.set === nothing
    @test t in InfiniteOpt.parameter_refs(x)
    
    props_y = DP.VariableProperties(y)
    @test props_y.name == "y"
    @test props_y.variable_type === nothing
end

function test_variable_properties_from_expr()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s ∈ [0, 2])
    @variable(model, x, Infinite(t))
    @variable(model, y, Infinite(s))
    
    expr = @expression(model, 2*x + y)
    props = DP.VariableProperties(expr)
    @test props.name == ""
    @test props.variable_type isa InfiniteOpt.Infinite
    @test Set(props.variable_type.parameter_refs) == Set((t, s))
    var1 = DP.create_variable(model, props)
    JuMP.set_name(var1, "inferred_var")
    @test JuMP.name(var1) == "inferred_var"
    @test InfiniteOpt.parameter_refs(var1) == (t, s)
end

function test_variable_properties_from_quad_expr()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s ∈ [0, 2])
    @variable(model, x, Infinite(t))
    @variable(model, y, Infinite(s))
    
    expr = @expression(model, x^2 + x*y)
    props = DP.VariableProperties(expr)
    @test props.name == ""
    @test props.variable_type isa InfiniteOpt.Infinite
    @test Set(props.variable_type.parameter_refs) == Set((t, s))
    var1 = DP.create_variable(model, props)
    JuMP.set_name(var1, "quad_inferred_var")
    @test JuMP.name(var1) == "quad_inferred_var"
    @test Set(InfiniteOpt.parameter_refs(var1)) == Set((t, s))
end

function test_variable_properties_from_nonlinear_expr()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s ∈ [0, 2])
    @variable(model, x, Infinite(t))
    @variable(model, y, Infinite(s))

    expr = @expression(model, exp(x) + sin(y))
    props = DP.VariableProperties(expr)
    @test props.name == ""
    @test props.variable_type isa InfiniteOpt.Infinite
    @test Set(props.variable_type.parameter_refs) == Set((t, s))
    var1 = DP.create_variable(model, props)
    JuMP.set_name(var1, "nl_inferred_var")
    @test JuMP.name(var1) == "nl_inferred_var"
    @test Set(InfiniteOpt.parameter_refs(var1)) == Set((t, s))
end

function test_variable_properties_from_vector()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @infinite_parameter(model, s ∈ [0, 2])
    @infinite_parameter(model, r ∈ [0, 3])
    @variable(model, x, Infinite(t))
    @variable(model, y, Infinite(s))
    @variable(model, z, Infinite(r))

    exprs = [
        @expression(model, x + 1),
        @expression(model, y + 2),
        @expression(model, exp(z))
    ]
    props = DP.VariableProperties(exprs)
    var1 = DP.create_variable(model, props)
    JuMP.set_name(var1, "vector_var")
    @test JuMP.name(var1) == "vector_var"
    prefs = InfiniteOpt.parameter_refs(var1)
    @test length(prefs) == 3
    @test Set(prefs) == Set((t, s, r))
end

function test_add_cardinality_constraint()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, y[1:3], InfiniteLogical(t))
    
    LCR = DP.LogicalConstraintRef{InfiniteModel}
    @test @constraint(model, y in Exactly(1)) isa LCR
    @test @constraint(model, y in AtLeast(1)) isa LCR
    @test @constraint(model, y in AtMost(2)) isa LCR
end

function test_add_logical_constraint()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, y[1:2], InfiniteLogical(t))
    
    LCR = DP.LogicalConstraintRef{InfiniteModel}
    @test @constraint(model, y[1] ∨ y[2] := true) isa LCR
    @test @constraint(model, y[1] ∧ y[2] := true) isa LCR
    @test @constraint(model, y[1] ⟹ y[2] := true) isa LCR
end

function test_add_constraint_single_logical_error()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, y, InfiniteLogical(t))
    
    c = JuMP.ScalarConstraint(y, MOI.EqualTo(true))
    @test_throws ErrorException JuMP.add_constraint(model, c, "")
end

function test_add_constraint_affine_logical_error()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, y[1:2], InfiniteLogical(t))

    aff_expr = 1.0 * y[1] + 1.0 * y[2]
    c = JuMP.ScalarConstraint(aff_expr, MOI.EqualTo(1.0))
    @test_throws ErrorException JuMP.add_constraint(model, c, "")
end

function test_add_constraint_quad_logical_error()
    model = InfiniteGDPModel()
    @infinite_parameter(model, t ∈ [0, 1])
    @variable(model, y[1:2], InfiniteLogical(t))
    
    quad_expr = 1.0 * y[1] * y[2]
    c = JuMP.ScalarConstraint(quad_expr, MOI.EqualTo(1.0))
    @test_throws ErrorException JuMP.add_constraint(model, c, "")
end

function test_logical_value()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, y[1:2], InfiniteLogical(t))
    
    @constraint(model, x >= 5, Disjunct(y[1]))
    @constraint(model, x <= 5, Disjunct(y[2]))
    @disjunction(model, [y[1], y[2]])
    
    @objective(model, Min, 𝔼(x, t))
    
    optimize!(model, gdp_method = Hull())
    
    val = value(y[2])
    @test eltype(val) == Bool
end

# raw_M against an InfiniteModel where M is constant across supports.
# Setup: x(t) ∈ [0, 10], disj1: x ≥ 5, disj2: x ≤ 3.
# For disj1 slack r(x) = 5 - x maximized over disj2's region x ∈ [0, 3]:
# max(5 - x) = 5 at x = 0. Same at every support ⇒ scalar M = 5.
function test_raw_M_infinite_scalar()
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
    @test length(InfiniteOpt.parameter_refs(obj)) == 1
    @test DP.raw_M(sub, obj, mbm) == 5.0
end

# raw_M with a support-varying M. Setup: x(t) ∈ [0, 10], disj1: x ≤ 2t,
# disj2: x ≥ 0.5. Slack r(x) = x - 2t maximized over x ∈ [0.5, 10]:
# max(x - 2t) = 10 - 2t. Varies with t ⇒ raw_M returns a pfunc whose
# raw values at supports are max-of-cell upper bounds for 10 - 2t.
function test_raw_M_infinite_param_function()
    model = InfiniteGDPModel()
    supports = [0.0, 0.25, 0.5, 0.75, 1.0]
    @infinite_parameter(model, t ∈ [0, 1], supports = supports)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @parameter_function(model, f == t -> 2*t)
    @constraint(model, con, x <= f, Disjunct(Y[1]))
    @constraint(model, con2, x >= 0.5, Disjunct(Y[2]))
    @disjunction(model, Y)
    mbm = DP._MBM(MBM(HiGHS.Optimizer), model)
    sub = DP.copy_model_with_constraints(
        model, DP.DisjunctConstraintRef[con2], mbm)
    obj = DP.prepare_max_M_objective(
        model, JuMP.constraint_object(con), sub)
    M = DP.raw_M(sub, obj, mbm)
    @test M isa InfiniteOpt.GeneralVariableRef
    raw_fn = InfiniteOpt.raw_function(M)
    # max-of-corners is conservative: raw_fn(t) ≥ 10 - 2t at supports.
    for t_val in supports
        @test raw_fn(t_val) >= 10.0 - 2*t_val - 1e-6
    end
end

# Piecewise-constant max-of-corners: returns the maximum value over
# the 2^n corners of the cell containing the query.
function test_interpolate()
    grid1 = [0.0, 1.0, 2.0, 3.0]
    vals1 = [10.0, 20.0, 40.0, 50.0]
    f = IDP._interpolate((grid1,), vals1)
    # At grid points: max over the cell to the right (or last cell).
    @test f(0.0) == 20.0   # max(vals[1], vals[2])
    @test f(1.0) == 40.0   # max(vals[2], vals[3])
    @test f(3.0) == 50.0   # last cell: max(vals[3], vals[4])
    # Between grid points: max of the surrounding two values.
    @test f(0.5) == 20.0
    @test f(1.5) == 40.0
    @test f(2.25) == 50.0
    # Out-of-range clamps to the boundary cell.
    @test f(-1.0) == 20.0
    @test f(4.0) == 50.0

    # 2D: max over the 4 surrounding corners.
    gx = [0.0, 1.0, 2.0]
    gy = [0.0, 10.0]
    vals2 = [x * y for x in gx, y in gy]   # 3x2 matrix
    g = IDP._interpolate((gx, gy), vals2)
    @test g(0.0, 0.0) == 10.0   # corners (0,0)=0, (1,0)=0, (0,10)=0, (1,10)=10
    @test g(2.0, 10.0) == 20.0  # last cell, max corner is (2,10)=20
    @test g(0.5, 5.0) == 10.0   # corners 0,0,0,10 -> 10
    @test g(1.5, 5.0) == 20.0   # corners 0,0,10,20 -> 20
end

# extract_solution returns per-support values from the transformation
# backend. Setup: force disj 2 active (x ≤ 3), BigM-reformulate, solve
# min ∫x ⇒ x = 0 at every support.
function test_extract_solution_infinite()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    K = 4
    @infinite_parameter(model, t ∈ [0, 1], num_supports = K)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    JuMP.fix(Y[2], true)  # force disj 2 active
    @objective(model, Min, ∫(x, t))
    DP.reformulate_model(model, BigM(10.0))
    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    optimize!(model, ignore_optimize_hook = true)
    sol = DP.extract_solution(model)
    @test haskey(sol, x)
    @test length(sol[x]) == K
    @test all(v -> isapprox(v, 0.0; atol=1e-6), sol[x])
end

# add_cut adds one pointwise-sum cut to the transformation backend and
# marks the backend ready so the next optimize! does NOT re-transcribe.
function test_add_cut_infinite()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    K = 3
    @infinite_parameter(model, t ∈ [0, 1], num_supports = K)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    DP.reformulate_model(model, BigM(10.0))
    InfiniteOpt.build_transformation_backend!(model)
    transcribed = InfiniteOpt.transformation_model(model)
    n_before = JuMP.num_constraints(transcribed;
        count_variable_in_set_constraints = false)
    rBM_sol = Dict(x => [1.0, 2.0, 3.0])
    sep_sol = Dict(x => [0.5, 1.5, 2.5])
    DP.add_cut(model, [x], rBM_sol, sep_sol)
    n_after = JuMP.num_constraints(transcribed;
        count_variable_in_set_constraints = false)
    @test n_after == n_before + 1
    # set_transformation_backend_ready(true) — next optimize! should
    # reuse without re-transcribing (otherwise our cut would be lost)
    @test InfiniteOpt.transformation_backend_ready(model)
end

# MBM with finite + integer variables in an InfiniteModel.
function test_mbm_finite_and_integer_var()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= w <= 5, Int)
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x + w >= 5, Disjunct(Y[1]))
    @constraint(model, x + w <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Min, ∫(x, t) + w)
    @test optimize!(model,
        gdp_method = MBM(HiGHS.Optimizer)) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
end

function test_mbm_infinite_simple()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    K = 4
    @infinite_parameter(model, t ∈ [0, 1], num_supports = K)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    JuMP.fix(Y[2], true)  # force disj 2 active
    @objective(model, Min, ∫(x, t))
    DP.reformulate_model(model, BigM(10.0))
    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    optimize!(model, ignore_optimize_hook = true)
    sol = DP.extract_solution(model)
    @test haskey(sol, x)
    @test length(sol[x]) == K
    @test all(v -> isapprox(v, 0.0; atol=1e-6), sol[x])
end

# add_cut adds one pointwise-sum cut to the transformation backend and
# marks the backend ready so the next optimize! does NOT re-transcribe.
function test_add_cut_infinite()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    K = 3
    @infinite_parameter(model, t ∈ [0, 1], num_supports = K)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    DP.reformulate_model(model, BigM(10.0))
    InfiniteOpt.build_transformation_backend!(model)
    transcribed = InfiniteOpt.transformation_model(model)
    n_before = JuMP.num_constraints(transcribed;
        count_variable_in_set_constraints = false)
    rBM_sol = Dict(x => [1.0, 2.0, 3.0])
    sep_sol = Dict(x => [0.5, 1.5, 2.5])
    DP.add_cut(model, [x], rBM_sol, sep_sol)
    n_after = JuMP.num_constraints(transcribed;
        count_variable_in_set_constraints = false)
    @test n_after == n_before + 1
    # set_transformation_backend_ready(true) — next optimize! should
    # reuse without re-transcribing (otherwise our cut would be lost)
    @test InfiniteOpt.transformation_backend_ready(model)
end

# MBM with finite + integer variables in an InfiniteModel.
function test_mbm_finite_and_integer_var()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= w <= 5, Int)
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x + w >= 5, Disjunct(Y[1]))
    @constraint(model, x + w <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Min, ∫(x, t) + w)
    @test optimize!(model,
        gdp_method = MBM(HiGHS.Optimizer)) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
end

function test_mbm_infinite_simple()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))

    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)

    @objective(model, Min, ∫(x, t))

    @test optimize!(model, gdp_method = MBM(HiGHS.Optimizer)) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    # x=0 with disjunct 2 active (x <= 3) gives min
    @test objective_value(model) ≈ 0.0 atol = 0.1
end

function test_mbm_infinite_param_dependent()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 20)
    @variable(model, -10 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))

    # Parameter-dependent constraints:
    # Disjunct 1: x(t) <= 2*t
    # Disjunct 2: x(t) >= 1 - t
    @parameter_function(model, f1 == t -> 2*t)
    @parameter_function(model, f2 == t -> 1 - t)
    @constraint(model, x <= f1, Disjunct(Y[1]))
    @constraint(model, x >= f2, Disjunct(Y[2]))
    @disjunction(model, Y)

    @objective(model, Min, ∫(x, t))

    @test optimize!(model, gdp_method = MBM(HiGHS.Optimizer)) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
end

function test_mbm_vs_bigm_infinite()
    # Compare MBM and BigM: should give same
    # feasible set and optimal value.
    for method_pair in [
        (BigM(100), MBM(HiGHS.Optimizer))
    ]
        model1 = InfiniteGDPModel(HiGHS.Optimizer)
        set_silent(model1)
        @infinite_parameter(model1, t ∈ [0, 1], num_supports = 10)
        @variable(model1, 0 <= x1 <= 10, Infinite(t))
        @variable(model1, Y1[1:2], InfiniteLogical(t))
        @constraint(model1, x1 >= 5, Disjunct(Y1[1]))
        @constraint(model1, x1 <= 3, Disjunct(Y1[2]))
        @disjunction(model1, Y1)
        @objective(model1, Min, ∫(x1, t))
        optimize!(model1, gdp_method = method_pair[1])
        obj1 = objective_value(model1)

        model2 = InfiniteGDPModel(HiGHS.Optimizer)
        set_silent(model2)
        @infinite_parameter(model2, t2 ∈ [0, 1], num_supports = 10)
        @variable(model2, 0 <= x2 <= 10, Infinite(t2))
        @variable(model2, Y2[1:2], InfiniteLogical(t2))
        @constraint(model2, x2 >= 5, Disjunct(Y2[1]))
        @constraint(model2, x2 <= 3, Disjunct(Y2[2]))
        @disjunction(model2, Y2)
        @objective(model2, Min, ∫(x2, t2))
        optimize!(model2, gdp_method = method_pair[2])
        obj2 = objective_value(model2)

        @test obj1 ≈ obj2 atol = 0.5
    end
end

function test_methods()
    I = 1:3
    J = 1:6
    period_bounds = collect(0:1:6)
    expected_obj = 4.504541662743021
    expected_z = -1.3634301575859131
    tol = 0.1

    # Use Juniper for MIQP support (HiGHS cannot solve MIQP)
    ipopt = optimizer_with_attributes(Ipopt.Optimizer, 
        "print_level" => 0, "sb" => "yes"
    )
    optimizer = optimizer_with_attributes(Juniper.Optimizer, "nl_solver" => ipopt)
    model = InfiniteGDPModel(optimizer)
    set_attribute(model, MOI.Silent(), true)

    @infinite_parameter(
        model, τ[j in J] in [period_bounds[j], period_bounds[j+1]], 
        num_supports = 5, independent = true, container = Array
    )
    @variable(model, -5 ≤ y[j in J] ≤ 5, Infinite(τ[j]), container = Array)
    @variable(model, -4 ≤ z ≤ 4)
    @objective(model, Min, 10 * sum(∫(y[j]^2, τ[j]) for j in J))

    @constraint(model, y[1](0) == 1)
    @constraint(model, [j = 2:6], y[j](period_bounds[j]) == y[j-1](period_bounds[j]))

    @variable(model, W[i = I, j = J], Logical)
    @constraint(model, [j in J], ∂(y[j], τ[j]) == -2*τ[j] + 0.3*z - 20*y[j], Disjunct(W[1, j]))
    @constraint(model, [j in J], ∂(y[j], τ[j]) == -2*z + 0.4*τ[j] - 4, Disjunct(W[2, j]))
    @constraint(model, [j in J], ∂(y[j], τ[j]) == 2*z + 4*(τ[j] - y[j] - 1), Disjunct(W[3, j]))
    @disjunction(model, [j in J], W[:, j])

    for j in J
        set_upper_bound(∂(y[j], τ[j]), 100)
        set_lower_bound(∂(y[j], τ[j]), -100)
    end

    @test optimize!(model, gdp_method = BigM()) isa Nothing
    @test objective_value(model) ≈ expected_obj atol=tol
    @test value(z) ≈ expected_z atol=tol

    @test optimize!(model, gdp_method = Hull()) isa Nothing
    @test objective_value(model) ≈ expected_obj atol=tol
    @test value(z) ≈ expected_z atol=tol

    @test optimize!(model, gdp_method = PSplit(3, model)) isa Nothing
    @test objective_value(model) ≈ expected_obj atol=tol
    @test value(z) ≈ expected_z atol=tol
end

function test_mbm_with_derivatives()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, -5 <= x <= 5, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))

    @constraint(model, ∂(x, t) >= 1, Disjunct(Y[1]))
    @constraint(model, ∂(x, t) <= -1, Disjunct(Y[2]))
    @disjunction(model, Y)

    set_upper_bound(∂(x, t), 10)
    set_lower_bound(∂(x, t), -10)

    @objective(model, Min, ∫(x^2, t))

    juniper = JuMP.optimizer_with_attributes(
        Juniper.Optimizer,
        "nl_solver" => JuMP.optimizer_with_attributes(
            Ipopt.Optimizer, "print_level" => 0),
        "log_levels" => []
    )
    set_optimizer(model, juniper)
    @test optimize!(model, gdp_method = MBM(juniper)) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED,
         MOI.ALMOST_LOCALLY_SOLVED]
end

function test_CuttingPlanes_infinite_simple()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))

    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)

    @objective(model, Min, ∫(x, t))

    # Should not throw
    @test optimize!(model,
        gdp_method = CuttingPlanes(
            HiGHS.Optimizer; max_iter = 5)
    ) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
end

function test_CuttingPlanes_infinite_two_disj()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x[1:2] <= 10, Infinite(t))
    @variable(model, W1[1:2], InfiniteLogical(t))
    @variable(model, W2[1:2], InfiniteLogical(t))

    @constraint(model, x[1] >= 2, Disjunct(W1[1]))
    @constraint(model, x[1] <= 1, Disjunct(W1[2]))
    @disjunction(model, W1)

    @constraint(model, x[2] >= 3, Disjunct(W2[1]))
    @constraint(model, x[2] <= 2, Disjunct(W2[2]))
    @disjunction(model, W2)

    @objective(model, Min, ∫(x[1] + x[2], t))

    # Compare cutting planes vs BigM
    optimize!(model,
        gdp_method = CuttingPlanes(
            HiGHS.Optimizer; max_iter = 10)
    )
    cp_obj = objective_value(model)

    model2 = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model2)
    @infinite_parameter(model2, t2 ∈ [0, 1], num_supports = 10)
    @variable(model2, 0 <= x2[1:2] <= 10, Infinite(t2))
    @variable(model2, V1[1:2], InfiniteLogical(t2))
    @variable(model2, V2[1:2], InfiniteLogical(t2))
    @constraint(model2, x2[1] >= 2, Disjunct(V1[1]))
    @constraint(model2, x2[1] <= 1, Disjunct(V1[2]))
    @disjunction(model2, V1)
    @constraint(model2, x2[2] >= 3, Disjunct(V2[1]))
    @constraint(model2, x2[2] <= 2, Disjunct(V2[2]))
    @disjunction(model2, V2)
    @objective(model2, Min, ∫(x2[1] + x2[2], t2))
    optimize!(model2, gdp_method = BigM())
    bigm_obj = objective_value(model2)

    @test cp_obj ≈ bigm_obj atol = 1.0
end



function test_CuttingPlanes_with_cuts()
    # Maximization with single-constraint disjuncts where Hull
    # is strictly tighter than BigM. BigM allows x+y up to
    # variable bounds (20), Hull limits to max(5,8)=8 — this
    # forces cuts to tighten the relaxation. Finite var w exercises
    # the isempty(var_prefs) branch in add_cut.
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= y <= 10, Infinite(t))
    @variable(model, 0 <= w <= 10)
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x + y <= 5, Disjunct(Y[1]))
    @constraint(model, x + y <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x + y, t) + w)
    cutting_planes = CuttingPlanes(HiGHS.Optimizer;
        max_iter = 30, seperation_tolerance = 1e-6)
    @test optimize!(model, gdp_method = cutting_planes) isa Nothing
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
end

function test_CuttingPlanes_multiparameter()
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)

    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @infinite_parameter(model, s ∈ [0, 2], num_supports = 4)
    @variable(model, 0 <= x <= 10, Infinite(t, s))
    @variable(model, Y[1:2], InfiniteLogical(t, s))

    @constraint(model, x >= 5, Disjunct(Y[1]))
    @constraint(model, x <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)

    @objective(model, Min, ∫(∫(x, t), s))

    optimize!(model,
        gdp_method = CuttingPlanes(HiGHS.Optimizer; max_iter = 5))
    @test termination_status(model) in
        [MOI.OPTIMAL, MOI.LOCALLY_SOLVED]
    # Y[2] (x <= 3) is the minimizer: x = 0 across (t, s) → ∫∫x = 0.
    @test objective_value(model) ≈ 0.0 atol = 1e-4
    @test all(isapprox.(value(x), 0.0; atol = 1e-6))
end

function test_loa_infinite_nonlinear_global()
    # max ∫x dt s.t. x(t)^2 <= 25 (global, per-support nonlinear),
    #   (x <= 3) ∨ (x <= 8), 0 <= x <= 10 over t ∈ [0, 1].
    # Disjunct Y[2] permits x up to 8 but the global x^2 <= 25 caps
    # x at 5. The per-support global transcribes to an `AbstractArray`
    # of scalar constraints, so this exercises the per-support fan-out
    # of the transcribed global constraints. Without the global cut the
    # master would allow x = 8 and report 8.0; the binding optimum
    # is ∫5 dt = 5.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = InfiniteGDPModel(ipopt)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @constraint(model, x^2 <= 25)
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t))
    optimize!(model,
        gdp_method = LOA(ipopt; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-2
    @test all(value(Y[2]))
    @test all(isapprox.(value(x), 5.0; atol = 1e-2))
end

function test_loa_infinite_complement_indicator()
    # Regression: `_indicator_to_binary(model)[Y2]` returns the AffExpr
    # `1 - binary(Y1)` for a logical-complement indicator. Fixing and
    # gating must resolve it onto the underlying binary's transcription
    # with the value inverted.
    #
    # max ∫ x dt with 0 ≤ x ≤ 10 over t ∈ [0,1]:
    #   Y1: x ≤ 3       Y2 ≡ ¬Y1: x ≤ 8
    # Y2 is the maximizer (x = 8 across t), giving ∫8 dt = 8.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = InfiniteGDPModel(ipopt)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y1, Logical)
    @variable(model, Y2, Logical, logical_complement = Y1)
    @constraint(model, x <= 3, Disjunct(Y1))
    @constraint(model, x <= 8, Disjunct(Y2))
    @disjunction(model, [Y1, Y2])
    @objective(model, Max, ∫(x, t))
    optimize!(model,
        gdp_method = LOA(ipopt; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
end

function test_loa_infinite_complement_nonlinear_disjunct()
    # Regression: a finite (complement-form AffExpr) indicator gating a
    # NONLINEAR constraint on an infinite variable. The constraint
    # transcribes to one scalar row per support while the binary
    # reference is a single expression, so the fan-out must broadcast
    # it across the constraint's supports.
    #
    # max ∫ x dt with 0 ≤ x ≤ 10 over t ∈ [0,1]:
    #   Y1: x ≤ 3        Y2 ≡ ¬Y1: x² ≤ 64
    # Y2 (complement) is the maximizer (x = 8 across t), giving 8.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = InfiniteGDPModel(juniper)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y1, Logical)
    @variable(model, Y2, Logical, logical_complement = Y1)
    @constraint(model, x <= 3, Disjunct(Y1))
    @constraint(model, x^2 <= 64, Disjunct(Y2))
    @disjunction(model, [Y1, Y2])
    @objective(model, Max, ∫(x, t))
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
end

function test_loa_infinite_multidim_parameter()
    # Regression: multi-D dependent parameter group. Transcription
    # expands a variable over ξ[1:2] across its joint supports; the
    # transcribed binaries and constraint rows must stay aligned.
    #
    # max w with 0 ≤ x ≤ 10 over ξ[1:2] ∈ [0,1]², w ≤ x at every joint
    # support:
    #   Y[1]: x ≤ 3       Y[2]: x ≤ 8
    # Y[2] is the maximizer (x = 8 across ξ allows w = 8).
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = InfiniteGDPModel(ipopt)
    set_silent(model)
    @infinite_parameter(model, ξ[1:2] ∈ [0, 1], num_supports = 4)
    @variable(model, 0 <= x <= 10, Infinite(ξ))
    @variable(model, 0 <= w <= 10)
    @constraint(model, w <= x)
    @variable(model, Y[1:2], InfiniteLogical(ξ))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, w)
    optimize!(model,
        gdp_method = LOA(ipopt; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
end

function test_loa_infinite_nlpf_infeasible_disjunct()
    # InfiniteOpt LOA: Y[1]'s per-support constraint x >= 200 is
    # NLP-infeasible against the bound x in [0, 10]. With infinite
    # indicators `Y[1:2], InfiniteLogical(t)`, the infeasible
    # Y[1]-everywhere seed makes the primary NLP fail, so NLPF runs: it
    # copies the transcribed NLP (per-support binary fixes included),
    # slacks the inequality, and returns a linearization point. The
    # disjunct is deliberately LINEAR:
    # the model is then convex, so LOA's master bound is rigorous and
    # it converges to Y[2] active everywhere (x = 5), giving ∫x dt = 5.
    # (A nonconvex infeasible disjunct such as x^2 >= 200 would make LOA
    # a heuristic and let the local NLP solver report a constraint-
    # violating point as solved — out of scope for the NLPF path here.)
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = InfiniteGDPModel(juniper)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x >= 200, Disjunct(Y[1]))
    @constraint(model, x <= 5, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t))
    optimize!(model,
        gdp_method = LOA(juniper; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 5.0 atol = 1e-2
end

function test_loa_infinite_aggregate_global()
    # min ∫x dt s.t. ∫(x^2, t) <= 4 (aggregate global), x >= y,
    #   (y >= 1) ∨ (y >= 3), 0 <= x, y <= 10 over t ∈ [0, 1].
    # The aggregate global transcribes to a single scalar (the
    # measure is expanded), exercising the single-row branch of
    # the transcribed global constraints. Y[1] (y >= 1) is the cheaper
    # disjunct: x = 1 satisfies x >= y and ∫x^2 = 1 <= 4, giving
    # objective ∫1 dt = 1.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = InfiniteGDPModel(ipopt)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= y <= 10)
    @constraint(model, ∫(x^2, t) <= 4)
    @constraint(model, x >= y)
    @variable(model, Y[1:2], Logical)
    @constraint(model, y >= 1, Disjunct(Y[1]))
    @constraint(model, y >= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Min, ∫(x, t))
    optimize!(model,
        gdp_method = LOA(ipopt; mip_optimizer = HiGHS.Optimizer))
    @test termination_status(model) in
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 1.0 atol = 1e-2
end

function test_loa_infinite_aggregate_disjunct()
    # An aggregate (measure) constraint INSIDE a disjunct: it
    # transcribes to a single scalar row that must be gated by the
    # indicator's binary. max ∫x dt with Y[1]: ∫(x^2, t) <= 4 and
    # Y[2]: x <= 0. Y[1] is optimal: constant x = 2 gives ∫x^2 = 4
    # and ∫x dt = 2 (Y[2] caps the objective at 0). Big-M tightening
    # is disabled: it needs bounds of every constraint variable and a
    # `MeasureRef` has none.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    model = InfiniteGDPModel(ipopt)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 10)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], Logical)
    @constraint(model, ∫(x^2, t) <= 4, Disjunct(Y[1]))
    @constraint(model, x <= 0, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t))
    optimize!(model, gdp_method = LOA(ipopt;
        mip_optimizer = HiGHS.Optimizer, inner_method = BigM(1e4, false)))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 2.0 atol = 1e-2
    @test value(Y[1])
end

function test_loa_infinite_iteration_loop()
    # Force the InfiniteModel LOA main loop to fix combinations that
    # vary per support. Two disjunctions over disjoint x/z half-lines
    # put the optimum on an off-diagonal combination the set-covering
    # seeds never try, so the master stays feasible and the loop body
    # runs with master-extracted per-support values.
    #   min ∫(x - z) dt
    #   D1: (x <= 4) [Y1]  ∨  (x >= 6) [Y2]
    #   D2: (z <= 4) [W1]  ∨  (z >= 6) [W2]
    # Optimum: Y1 & W2 everywhere → x = 0, z = 10, ∫(x - z) = -10.
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 3)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= z <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @variable(model, W[1:2], InfiniteLogical(t))
    @constraint(model, x <= 4, Disjunct(Y[1]))
    @constraint(model, x >= 6, Disjunct(Y[2]))
    @disjunction(model, Y)
    @constraint(model, z <= 4, Disjunct(W[1]))
    @constraint(model, z >= 6, Disjunct(W[2]))
    @disjunction(model, W)
    @objective(model, Min, ∫(x - z, t))
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer))
    @test termination_status(model) in
        (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ -10.0 atol = 1e-3
    @test all(value(Y[1]))
    @test all(value(W[2]))
    @test all(isapprox.(value(x), 0.0; atol = 1e-6))
    @test all(isapprox.(value(z), 10.0; atol = 1e-6))
end

function test_loa_infinite_hull_linear()
    # InfiniteOpt LOA with inner_method = Hull. Linear disjuncts enter
    # the master as exact per-support Hull perspectives over
    # disaggregated infinite variables. max ∫x dt, Y1: x <= 3,
    # Y2: x <= 8. Optimum: x = 8 across t → ∫8 dt = 8.
    model = InfiniteGDPModel(HiGHS.Optimizer)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x <= 8, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t))
    optimize!(model, gdp_method = LOA(HiGHS.Optimizer;
        inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
end

function test_loa_infinite_hull_nonlinear_disjunct()
    # The infinite Hull case big-M differs from: a NONLINEAR constraint
    # on an infinite variable inside a disjunct, under a plain infinite
    # indicator. The convex-hull OA cut is built per support over the
    # transcribed disaggregated infinite variable. max ∫x dt,
    # Y1: x <= 3, Y2: x^2 <= 64. Optimum: x = 8 via Y2.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = InfiniteGDPModel(juniper)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x <= 3, Disjunct(Y[1]))
    @constraint(model, x^2 <= 64, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t))
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
    @test all(value(Y[2]))
    @test all(isapprox.(value(x), 8.0; atol = 1e-2))
end

function test_loa_infinite_hull_complement_nonlinear()
    # Hull with a complement indicator Y2 ≡ ¬Y1 gating a nonlinear
    # constraint on an infinite variable. The per-support disaggregator
    # is keyed by the transcribed complement binary `1 - y1_k`
    # (an AffExpr). max ∫x dt, Y1: x <= 3, Y2: x^2 <= 64. Optimum 8.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = InfiniteGDPModel(juniper)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 5)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, Y1, Logical)
    @variable(model, Y2, Logical, logical_complement = Y1)
    @constraint(model, x <= 3, Disjunct(Y1))
    @constraint(model, x^2 <= 64, Disjunct(Y2))
    @disjunction(model, [Y1, Y2])
    @objective(model, Max, ∫(x, t))
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 8.0 atol = 1e-2
end

function test_loa_infinite_hull_finite_var()
    # A FINITE variable disaggregated inside a nonlinear disjunct
    # constraint gated by an INFINITE indicator. The fan-out slices the
    # binary per support, so the disaggregator must key the finite
    # variable's single copy by each per-support binary (driven off the
    # binary, not the variable). max ∫x dt + w over t ∈ [0,1]:
    #   Y1: x^2 + w^2 <= 25      Y2: x <= 0, w <= 3
    # Y1 is active (Y2 caps the objective at 3); x(t) = w = sqrt(12.5),
    # so the optimum is ∫sqrt(12.5) dt + sqrt(12.5) = 2 sqrt(12.5)
    # ≈ 7.0711.
    ipopt = optimizer_with_attributes(Ipopt.Optimizer,
        "print_level" => 0, "sb" => "yes")
    juniper = optimizer_with_attributes(Juniper.Optimizer,
        "nl_solver" => ipopt, "log_levels" => [])
    model = InfiniteGDPModel(juniper)
    set_silent(model)
    @infinite_parameter(model, t ∈ [0, 1], num_supports = 4)
    @variable(model, 0 <= x <= 10, Infinite(t))
    @variable(model, 0 <= w <= 10)
    @variable(model, Y[1:2], InfiniteLogical(t))
    @constraint(model, x^2 + w^2 <= 25, Disjunct(Y[1]))
    @constraint(model, x <= 0, Disjunct(Y[2]))
    @constraint(model, w <= 3, Disjunct(Y[2]))
    @disjunction(model, Y)
    @objective(model, Max, ∫(x, t) + w)
    optimize!(model, gdp_method = LOA(juniper;
        mip_optimizer = HiGHS.Optimizer, inner_method = Hull()))
    @test termination_status(model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
    @test objective_value(model) ≈ 7.0711 atol = 1e-2
end

@testset "InfiniteDisjunctiveProgramming" begin

    @testset "Model" begin
        test_infinite_gdp_model_creation()
    end

    @testset "all_variables" begin
        test_all_variables_infiniteopt()
    end

    @testset "Variables" begin
        test_infinite_logical()
        test_is_parameter()
        test_is_parameter_concrete_dispatches()
        test_requires_disaggregation()
        test_variable_properties_infiniteopt()
        test_variable_properties_from_expr()
        test_variable_properties_from_quad_expr()
        test_variable_properties_from_nonlinear_expr()
        test_variable_properties_from_vector()
    end

    @testset "Constraints" begin
        test_add_cardinality_constraint()
        test_add_logical_constraint()
    end

    @testset "JuMP Overloads" begin
        test_logical_value()
        test_add_constraint_single_logical_error()
        test_add_constraint_affine_logical_error()
        test_add_constraint_quad_logical_error()
    end

    @testset "Methods" begin
        test_get_constant()
        test_disaggregate_expression_infiniteopt()
    end

    @testset "MBM" begin
        test_interpolate()
        test_raw_M_infinite_scalar()
        test_raw_M_infinite_param_function()
        test_mbm_finite_and_integer_var()
        test_mbm_infinite_simple()
        test_mbm_infinite_param_dependent()
        test_mbm_vs_bigm_infinite()
        test_mbm_with_derivatives()
    end

    @testset "Integration" begin
        test_infiniteopt_extension()
        test_methods()
    end

    @testset "Cutting Planes" begin
        test_extract_solution_infinite()
        test_add_cut_infinite()
        test_CuttingPlanes_infinite_simple()
        test_CuttingPlanes_infinite_two_disj()
        test_CuttingPlanes_with_cuts()
        test_CuttingPlanes_multiparameter()
    end

    # LOA runs on the transcribed backend, a plain JuMP model, so it
    # needs no `JuMP.copy_model(::InfiniteModel)` from the InfiniteOpt
    # fork (unlike MBM).
    @testset "LOA" begin
        test_loa_infinite_nonlinear_global()
        test_loa_infinite_aggregate_global()
        test_loa_infinite_aggregate_disjunct()
        test_loa_infinite_complement_indicator()
        test_loa_infinite_complement_nonlinear_disjunct()
        test_loa_infinite_multidim_parameter()
        test_loa_infinite_nlpf_infeasible_disjunct()
        test_loa_infinite_iteration_loop()
        test_loa_infinite_hull_linear()
        test_loa_infinite_hull_nonlinear_disjunct()
        test_loa_infinite_hull_complement_nonlinear()
        test_loa_infinite_hull_finite_var()
    end

end
