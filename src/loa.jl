################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Implementation of the LOA algorithm from:
#   Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
# With augmented-penalty OA master from:
#   Viswanathan & Grossmann (1990), Comp. & Chem. Eng. 14(7), 769-782
# Closely follows Pyomo's GDPopt LOA implementation.
################################################################################

"""
    LOA{O} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models.

**Fields**
- `optimizer::O`: Optimizer for solving NLP subproblems and MILP
  master problems (required).
- `max_iter::Int`: Maximum LOA iterations (default = `10`).
- `atol::Float64`: Absolute convergence tolerance (default = `1e-6`).
- `rtol::Float64`: Relative convergence tolerance (default = `1e-4`).
- `M_value::Float64`: Big-M value for master reformulation
  (default = `1e9`).
- `max_slack::Float64`: Upper bound on each OA cut slack
  (default = `1000.0`).
- `OA_penalty_factor::Float64`: Multiplier on the slack sum in
  the augmented master objective (default = `1000.0`).
"""
struct LOA{O} <: AbstractReformulationMethod
    optimizer::O
    max_iter::Int
    atol::Float64
    rtol::Float64
    M_value::Float64
    max_slack::Float64
    OA_penalty_factor::Float64
    function LOA(
        optimizer::O;
        max_iter::Int = 10,
        atol::Float64 = 1e-6,
        rtol::Float64 = 1e-4,
        M_value::Float64 = 1e9,
        max_slack::Float64 = 1000.0,
        OA_penalty_factor::Float64 = 1000.0
        ) where {O}
        new{O}(optimizer, max_iter, atol, rtol, M_value,
               max_slack, OA_penalty_factor)
    end
end

################################################################################
#                          DATA STRUCTURES
################################################################################
# Result from solving an NLP subproblem.
struct _LOAIterationResult{M <: JuMP.AbstractModel}
    combo::Dict{LogicalVariableRef{M}, Bool}
    x_values::Dict{JuMP.AbstractVariableRef, Float64}
    duals::Dict{DisjunctConstraintRef{M}, Float64}
    objective::Float64
    feasible::Bool
end

# Master problem state.
mutable struct _LOAMaster{M <: JuMP.AbstractModel}
    model::M
    ref_map::JuMP.GenericReferenceMap
    # orig indicator → master binary variable (or nothing)
    bin_map::Dict{LogicalVariableRef, Any}
    slack_vars::Vector{JuMP.VariableRef}
    original_obj::JuMP.AbstractJuMPScalar
    obj_sense::_MOI.OptimizationSense
end

################################################################################
#                           MAIN ALGORITHM
################################################################################
function reformulate_model(
    model::JuMP.AbstractModel, method::LOA
    )
    _clear_reformulations(model)

    # Step 1: Set covering initialization
    combos = _set_covering_combos(model)

    # Step 2: Solve init subproblems
    z_upper = Inf
    M = typeof(model)
    init_results = _LOAIterationResult{M}[]
    for combo in combos
        result = solve_loa_subproblem(model, combo, method)
        push!(init_results, result)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
        end
    end

    # Step 3: Build MILP master
    master = _build_loa_master(model, init_results, method)

    # Step 4: Main LOA loop
    z_lower = -Inf
    for iter in 1:method.max_iter
        _update_augmented_objective!(master, method)

        JuMP.optimize!(master.model, ignore_optimize_hook = true)
        if !JuMP.is_solved_and_feasible(master.model)
            break
        end
        z_lower = JuMP.objective_value(master.model)
        _loa_converged(z_upper, z_lower, method) && break

        combo = _extract_combo_from_master(model, master)
        result = solve_loa_subproblem(model, combo, method)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
        end

        result.feasible && add_oa_cuts(
            master, result, model, method)
        _add_no_good_cut!(master, combo)
    end

    # Step 5: Final reformulation
    reformulate_model(model, BigM(method.M_value))
    return
end

################################################################################
#                      SET COVERING INITIALIZATION
################################################################################
# Find a minimum set of disjunct combos such that every disjunct
# appears True in at least one combo (Türkay & Grossmann 1996).
function _set_covering_combos(model::JuMP.AbstractModel)
    M = typeof(model)
    per_disj = Vector{Tuple{DisjunctionIndex, LogicalVariableRef{M}}}[]
    for (idx, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        push!(per_disj,
            [(idx, ind) for ind in disj_data.constraint.indicators])
    end
    isempty(per_disj) && return Dict{LogicalVariableRef{M}, Bool}[]

    all_combos = [
        Tuple{DisjunctionIndex, LogicalVariableRef{M}}[c...]
        for c in Iterators.product(per_disj...)]

    uncovered = Set{LogicalVariableRef{M}}()
    for group in per_disj
        for (_, ind) in group
            push!(uncovered, ind)
        end
    end

    selected = Dict{LogicalVariableRef{M}, Bool}[]
    while !isempty(uncovered)
        best_combo = nothing
        best_count = 0
        for combo in all_combos
            count = sum(ind in uncovered for (_, ind) in combo)
            if count > best_count
                best_count = count
                best_combo = combo
            end
        end
        best_combo === nothing && break
        combo_dict = Dict{LogicalVariableRef{M}, Bool}()
        for (disj_idx, active_ind) in best_combo
            disj_data = _disjunctions(model)[disj_idx]
            for ind in disj_data.constraint.indicators
                combo_dict[ind] = (ind == active_ind)
            end
        end
        push!(selected, combo_dict)
        for (_, active_ind) in best_combo
            delete!(uncovered, active_ind)
        end
    end
    return selected
end

################################################################################
#                     SUBPROBLEM HELPERS
################################################################################

# Build a direct orig indicator → master binary variable mapping.
function _build_bin_map(master_model, lv_map)
    ind_to_bin = _indicator_to_binary(master_model)
    bin_map = Dict{LogicalVariableRef, Any}()
    for (orig_ind, mapped_ind) in lv_map
        if haskey(ind_to_bin, mapped_ind)
            bv = ind_to_bin[mapped_ind]
            bin_map[orig_ind] = bv isa JuMP.AbstractVariableRef ?
                bv : nothing
        else
            bin_map[orig_ind] = nothing
        end
    end
    return bin_map
end

# Collect the active disjunct constraint refs for a combo.
function _active_constraints(
    model::M, combo::Dict{LogicalVariableRef{M}, Bool}
    ) where {M <: JuMP.AbstractModel}
    crefs = DisjunctConstraintRef{M}[]
    for (ind, active) in combo
        !active && continue
        haskey(_indicator_to_constraints(model), ind) || continue
        for cref in _indicator_to_constraints(model)[ind]
            cref isa DisjunctConstraintRef || continue
            push!(crefs, cref)
        end
    end
    return crefs
end

################################################################################
#                      NLP SUBPROBLEM
################################################################################

# Build a lightweight submodel with only the given constraints
# + variable bounds + objective. Returns (GDPSubmodel, con_refs)
# where con_refs are the submodel constraint refs in the same
# order as the input constraints (for dual extraction).
function copy_model_with_constraints(
    model::JuMP.AbstractModel,
    constraints::Vector{<:DisjunctConstraintRef},
    method::LOA
    )
    var_type = JuMP.variable_ref_type(model)
    sub_model = _copy_model(model)
    dec_vars = collect_all_vars(model)
    fwd_map = Dict{var_type, Vector{var_type}}()

    for var in dec_vars
        copy_var = variable_copy(sub_model, var)
        fwd_map[var] = [copy_var]
    end

    flat_map = Dict(v => ws[1] for (v, ws) in fwd_map)

    # Copy objective
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = _replace_variables_in_constraint(obj, flat_map)
    JuMP.@objective(sub_model, sense, new_obj)

    # Copy constraints, track refs for dual extraction
    sub_crefs = JuMP.ConstraintRef[]
    for cref in constraints
        con = JuMP.constraint_object(cref)
        expr = _replace_variables_in_constraint(
            con.func, flat_map)
        T = one(JuMP.value_type(typeof(sub_model)))
        sub_cref = JuMP.@constraint(
            sub_model, expr * T in con.set)
        push!(sub_crefs, sub_cref)
    end

    JuMP.set_optimizer(sub_model, method.optimizer)
    JuMP.set_silent(sub_model)

    return GDPSubmodel(sub_model, dec_vars, fwd_map), sub_crefs
end

# Solve the NLP subproblem for a given combo. Builds a
# lightweight model with only the active constraints.
function solve_loa_subproblem(
    model::M, combo::Dict{LogicalVariableRef{M}, Bool},
    method::LOA
    ) where {M <: JuMP.AbstractModel}
    active_crefs = _active_constraints(model, combo)
    sub, sub_crefs = copy_model_with_constraints(
        model, active_crefs, method)

    JuMP.optimize!(sub.model)

    if !JuMP.is_solved_and_feasible(sub.model)
        return _LOAIterationResult{M}(
            combo,
            Dict{JuMP.AbstractVariableRef, Float64}(),
            Dict{DisjunctConstraintRef{M}, Float64}(),
            Inf, false)
    end

    x_vals = Dict{JuMP.AbstractVariableRef, Float64}(
        var => JuMP.value(sub.fwd_map[var][1])
        for var in sub.dec_vars)

    # Duals directly from submodel constraints (no BigM indirection)
    duals = Dict{DisjunctConstraintRef{M}, Float64}()
    has_d = JuMP.has_duals(sub.model)
    for (orig_cref, sub_cref) in zip(active_crefs, sub_crefs)
        if has_d
            duals[orig_cref] = JuMP.dual(sub_cref)
        end
    end

    return _LOAIterationResult{M}(
        combo, x_vals, duals,
        JuMP.objective_value(sub.model), true)
end

################################################################################
#                      MASTER PROBLEM BUILDER
################################################################################
function _build_loa_master(
    model::M, init_results, method::LOA
    ) where {M <: JuMP.AbstractModel}
    master_model, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(master_model, method.optimizer)
    JuMP.set_silent(master_model)
    reformulate_model(master_model, BigM(method.M_value))

    master = _LOAMaster{typeof(master_model)}(
        master_model, ref_map,
        _build_bin_map(master_model, lv_map),
        JuMP.VariableRef[],
        JuMP.objective_function(master_model),
        JuMP.objective_sense(master_model))

    for result in init_results
        result.feasible && add_oa_cuts(
            master, result, model, method)
        _add_no_good_cut!(master, result.combo)
    end
    return master
end

# Rebuild the augmented penalty objective.
function _update_augmented_objective!(
    master::_LOAMaster, method::LOA
    )
    sign_adjust = master.obj_sense == _MOI.MIN_SENSE ? 1 : -1
    penalty = zero(JuMP.AffExpr)
    for s in master.slack_vars
        JuMP.add_to_expression!(penalty, 1.0, s)
    end
    new_obj = master.original_obj +
        sign_adjust * method.OA_penalty_factor * penalty
    JuMP.set_objective_function(master.model, new_obj)
    JuMP.set_objective_sense(master.model, master.obj_sense)
    return
end

################################################################################
#                        OA CUT GENERATION
################################################################################
# Add outer approximation cuts at solution point xk.
# Pyomo GDPopt cut form:
#   sign(sign_adjust * λ) * [r(xk) - rhs + ∇r(xk)ᵀ(x - xk)]
#     - slack <= M*(1 - y)
function add_oa_cuts(
    master::_LOAMaster,
    result::_LOAIterationResult{M},
    model::M,
    method::LOA
    ) where {M <: JuMP.AbstractModel}
    sign_adjust = master.obj_sense == _MOI.MIN_SENSE ? -1 : 1

    for (ind, active) in result.combo
        !active && continue
        haskey(_indicator_to_constraints(model), ind) || continue
        bin_var = get(master.bin_map, ind, nothing)

        for orig_cref in _indicator_to_constraints(model)[ind]
            orig_cref isa DisjunctConstraintRef || continue
            con_data = _disjunct_constraints(model)[
                JuMP.index(orig_cref)]
            con = con_data.constraint

            con.func isa JuMP.GenericAffExpr && continue

            dual_val = get(result.duals, orig_cref, nothing)
            dual_val === nothing && continue

            lin_expr = _linearize_at(
                con.func, result.x_values, master.ref_map)
            rhs = _set_rhs(con.set)
            s = sign(sign_adjust * dual_val)
            s == 0 && continue

            slack = JuMP.@variable(master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack)
            push!(master.slack_vars, slack)

            if bin_var !== nothing
                cref = JuMP.@constraint(master.model,
                    s * (lin_expr - rhs) - slack <=
                    method.M_value * (1 - bin_var))
            else
                cref = JuMP.@constraint(master.model,
                    s * (lin_expr - rhs) - slack <= 0)
            end
            push!(_reformulation_constraints(master.model),
                cref)
        end
    end
end

# Linearize a QuadExpr at point xk (first-order Taylor).
function _linearize_at(
    func::JuMP.GenericQuadExpr, xk::Dict, ref_map
    )
    grad = Dict{Any, Float64}()
    for (var, coef) in func.aff.terms
        grad[var] = get(grad, var, 0.0) + coef
    end
    for (pair, coef) in func.terms
        vi, vj = pair.a, pair.b
        if vi == vj
            grad[vi] = get(grad, vi, 0.0) +
                2 * coef * get(xk, vi, 0.0)
        else
            grad[vi] = get(grad, vi, 0.0) +
                coef * get(xk, vj, 0.0)
            grad[vj] = get(grad, vj, 0.0) +
                coef * get(xk, vi, 0.0)
        end
    end

    f_xk = func.aff.constant
    for (var, coef) in func.aff.terms
        f_xk += coef * get(xk, var, 0.0)
    end
    for (pair, coef) in func.terms
        f_xk += coef * get(xk, pair.a, 0.0) *
                        get(xk, pair.b, 0.0)
    end

    constant = f_xk
    for (var, g) in grad
        constant -= g * get(xk, var, 0.0)
    end
    result = JuMP.AffExpr(constant)
    for (var, g) in grad
        JuMP.add_to_expression!(result, g, ref_map[var])
    end
    return result
end

################################################################################
#                  NONLINEAR EXPRESSION LINEARIZATION
################################################################################
# Convert JuMP expression tree to Julia Expr with x[i]
# variable references for MOI.Nonlinear evaluation.
function _to_nlp_expr(expr::JuMP.GenericNonlinearExpr, idx::Dict)
    args = Any[_to_nlp_expr(a, idx) for a in expr.args]
    return Expr(:call, expr.head, args...)
end

function _to_nlp_expr(expr::JuMP.GenericAffExpr, idx::Dict)
    parts = Any[expr.constant]
    for (var, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef,
            Expr(:ref, :x, idx[var])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end

function _to_nlp_expr(expr::JuMP.GenericQuadExpr, idx::Dict)
    parts = Any[_to_nlp_expr(expr.aff, idx)]
    for (pair, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef,
            Expr(:ref, :x, idx[pair.a]),
            Expr(:ref, :x, idx[pair.b])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end

function _to_nlp_expr(var::JuMP.AbstractVariableRef, idx::Dict)
    return _MOI.VariableIndex(idx[var])
end

_to_nlp_expr(x::Number, ::Dict) = x

# Linearize a GenericNonlinearExpr at point xk using
# MOI.Nonlinear reverse-mode AD.
function _linearize_at(func::JuMP.GenericNonlinearExpr, xk::Dict, ref_map)
    vars = JuMP.AbstractVariableRef[]
    _interrogate_variables(v -> push!(vars, v), func)
    unique!(vars)
    if isempty(vars)
        return JuMP.AffExpr(JuMP.value(v -> 0.0, func))
    end

    n = length(vars)
    T = JuMP.value_type(
        typeof(JuMP.owner_model(vars[1])))
    idx = Dict(vars[i] => i for i in 1:n)
    nlp = _MOI.Nonlinear.Model()
    _MOI.Nonlinear.set_objective(
        nlp, _to_nlp_expr(func, idx))
    ord = [_MOI.VariableIndex(i) for i in 1:n]
    evaluator = _MOI.Nonlinear.Evaluator(
        nlp, _MOI.Nonlinear.SparseReverseMode(), ord)
    _MOI.initialize(evaluator, [:Grad])

    xk_vec = [get(xk, v, zero(T)) for v in vars]
    f_xk = _MOI.eval_objective(evaluator, xk_vec)
    grad = zeros(T, n)
    _MOI.eval_objective_gradient(
        evaluator, grad, xk_vec)

    constant = T(f_xk)
    for i in 1:n
        constant -= grad[i] * xk_vec[i]
    end
    V = typeof(ref_map[vars[1]])
    result = JuMP.GenericAffExpr{T, V}(constant)
    for i in 1:n
        iszero(grad[i]) && continue
        JuMP.add_to_expression!(
            result, grad[i], ref_map[vars[i]])
    end
    return result
end

# Extract RHS value from an MOI set.
_set_rhs(set::_MOI.LessThan) = set.upper
_set_rhs(set::_MOI.GreaterThan) = set.lower
_set_rhs(set::_MOI.EqualTo) = set.value
_set_rhs(::Any) = 0.0

################################################################################
#                       NO-GOOD CUT GENERATION
################################################################################
function _add_no_good_cut!(master::_LOAMaster, combo)
    cut_expr = JuMP.AffExpr(0.0)
    for (ind, active) in combo
        bin_var = get(master.bin_map, ind, nothing)
        bin_var === nothing && continue
        if active
            JuMP.add_to_expression!(cut_expr, -1.0, bin_var)
            JuMP.add_to_expression!(cut_expr, 1.0)
        else
            JuMP.add_to_expression!(cut_expr, 1.0, bin_var)
        end
    end
    cref = JuMP.@constraint(master.model, cut_expr >= 1.0)
    push!(_reformulation_constraints(master.model), cref)
end

################################################################################
#                     MASTER SOLUTION EXTRACTION
################################################################################
function _extract_combo_from_master(
    model::M, master::_LOAMaster
    ) where {M <: JuMP.AbstractModel}
    combo = Dict{LogicalVariableRef{M}, Bool}()
    for (_, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        for ind in disj_data.constraint.indicators
            bin_var = get(master.bin_map, ind, nothing)
            combo[ind] = bin_var !== nothing ?
                JuMP.value(bin_var) > 0.5 : false
        end
    end
    return combo
end

################################################################################
#                       CONVERGENCE CHECK
################################################################################
function _loa_converged(z_upper, z_lower, method::LOA)
    gap = z_upper - z_lower
    gap <= method.atol && return true
    abs(z_upper) > 1e-10 &&
        gap / abs(z_upper) <= method.rtol && return true
    return false
end

################################################################################
#                          ERROR FALLBACK
################################################################################
function reformulate_model(::M, ::LOA) where {M}
    error("reformulate_model not implemented for " *
          "model type `$(M)` with LOA.")
end
