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
    LOA{O, P} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models.

**Fields**
- `nlp_optimizer::O`: Optimizer for NLP subproblems (required).
- `mip_optimizer::P`: Optimizer for the MILP master problem. Defaults to
  `nlp_optimizer` when omitted.
- `max_iter::Int`: Maximum LOA iterations (default = `10`).
- `atol::Float64`: Absolute convergence tolerance (default = `1e-6`).
- `rtol::Float64`: Relative convergence tolerance (default = `1e-4`).
- `M_value::Float64`: Big-M value for master reformulation
  (default = `1e9`).
- `max_slack::Float64`: Upper bound on each OA cut slack
  (default = `1000.0`).
- `OA_penalty_factor::Float64`: Multiplier on the slack sum in the
  augmented master objective (default = `1000.0`).
"""
struct LOA{O, P} <: AbstractReformulationMethod
    nlp_optimizer::O
    mip_optimizer::P
    max_iter::Int
    atol::Float64
    rtol::Float64
    M_value::Float64
    max_slack::Float64
    OA_penalty_factor::Float64
    function LOA(
        nlp_optimizer::O;
        mip_optimizer::P = nlp_optimizer,
        max_iter::Int = 10,
        atol::Float64 = 1e-6,
        rtol::Float64 = 1e-4,
        M_value::Float64 = 1e9,
        max_slack::Float64 = 1000.0,
        OA_penalty_factor::Float64 = 1000.0
        ) where {O, P}
        new{O, P}(
            nlp_optimizer, mip_optimizer, max_iter, atol, rtol,
            M_value, max_slack, OA_penalty_factor
        )
    end
end

################################################################################
#                          DATA STRUCTURES
################################################################################
# Result from solving an NLP subproblem. X is Float64 for finite models,
# Vector{Float64} for infinite (per-support).
struct _LOAIterationResult{M <: JuMP.AbstractModel, X}
    combo::Dict{LogicalVariableRef{M}, Bool}
    x_values::Dict{JuMP.AbstractVariableRef, X}
    duals::Dict{DisjunctConstraintRef{M}, X}
    objective::Float64
    feasible::Bool
end

# Master problem state. R is the ref_map type: GenericReferenceMap for
# finite, Dict for infinite (flat transcribed). lv_map maps original
# indicators to master indicators — use with `binary_variable` or
# `JuMP.value` to access the master's binary/solution. nl_globals holds
# (func, set) pairs from the original model for nonlinear non-disjunct
# constraints (stripped from master, OA-cut instead). The master's
# objective is augmented incrementally by `_apply_slack_penalty!` as
# slacks are created in `_add_oa_cuts`.
mutable struct _LOAMaster{M <: JuMP.AbstractModel, R}
    model::M
    ref_map::R
    lv_map::Dict{LogicalVariableRef{M}, LogicalVariableRef{M}}
    obj_sense::_MOI.OptimizationSense
    nl_globals::Vector{Tuple{Any, Any}}
end

################################################################################
#                           MAIN ALGORITHM
################################################################################
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)

    # Step 1: Set covering initialization
    combos = _set_covering_combos(model)

    # Step 2: Solve initial NLP subproblems
    z_upper = Inf
    M = typeof(model)
    best_result = nothing
    init_results = _LOAIterationResult[]
    for combo in combos
        result = _solve_loa_subproblem(model, combo, method)
        push!(init_results, result)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
            best_result = result
        end
    end

    # Step 3: Build MILP master
    master = _build_loa_master(model, init_results, method)

    # Step 4: Main LOA loop
    z_lower = -Inf
    for iter in 1:method.max_iter
        JuMP.optimize!(master.model, ignore_optimize_hook = true)
        if !JuMP.is_solved_and_feasible(master.model)
            break
        end
        z_lower = JuMP.objective_value(master.model)
        if _loa_converged(z_upper, z_lower, method)
            break
        end
        combo = _extract_combo(model, master)
        result = _solve_loa_subproblem(model, combo, method)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
            best_result = result
        end
        result.feasible && _add_oa_cuts(master, result, model, method)
        _add_no_good_cut(master, combo)
    end

    # Step 5: Final reformulation
    _finalize_model(model, best_result, method)
    return
end

################################################################################
#                      SET COVERING INITIALIZATION
################################################################################
# Minimum set of disjunct combos covering every disjunct at least once
# (Türkay & Grossmann 1996).
function _set_covering_combos(model::JuMP.AbstractModel)
    M = typeof(model)
    LVR = LogicalVariableRef{M}
    per_disj = Vector{Tuple{DisjunctionIndex, LVR}}[]
    for (idx, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        push!(per_disj, [
            (idx, ind) for ind in disj_data.constraint.indicators
        ])
    end
    isempty(per_disj) && return Dict{LVR, Bool}[]

    all_combos = [
        Tuple{DisjunctionIndex, LVR}[c...]
        for c in Iterators.product(per_disj...)
    ]
    uncovered = Set{LVR}()
    for group in per_disj
        for (_, ind) in group
            push!(uncovered, ind)
        end
    end

    selected = Dict{LVR, Bool}[]
    while !isempty(uncovered)
        best_combo = nothing
        best_count = 0
        for combo in all_combos
            cnt = sum(ind in uncovered for (_, ind) in combo)
            if cnt > best_count
                best_count = cnt
                best_combo = combo
            end
        end
        best_combo === nothing && break
        combo_dict = Dict{LVR, Bool}()
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
#                        SUBPROBLEM HELPERS
################################################################################
# Active disjunct constraint refs for a given combo.
function _active_constraints(
    model::M,
    combo::Dict{LogicalVariableRef{M}, Bool}
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
#                         NLP SUBPROBLEM
################################################################################
# LOA dispatch of copy_model_with_constraints: builds a submodel like MBM
# but also copies objective + global constraints, relaxes integrality, and
# returns tracked constraint refs for dual extraction.
function copy_model_with_constraints(
    model::JuMP.AbstractModel,
    constraints::Vector{<:DisjunctConstraintRef},
    method::LOA
    )
    var_type = JuMP.variable_ref_type(model)
    sub_model = _copy_model(model)
    decision_vars = collect_all_vars(model)
    fwd_map = Dict{var_type, Vector{var_type}}(
        var => [variable_copy(sub_model, var)]
        for var in decision_vars)
    JuMP.relax_integrality(sub_model)
    flat_map = Dict(v => ws[1] for (v, ws) in fwd_map)

    # Copy objective
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = _replace_variables_in_constraint(obj, flat_map)
    JuMP.@objective(sub_model, sense, new_obj)

    # Copy global (non-disjunct) constraints
    _copy_global_constraints(model, sub_model, flat_map)

    # Copy disjunct constraints (tracked for duals)
    sub_crefs = JuMP.ConstraintRef[]
    for cref in constraints
        con = JuMP.constraint_object(cref)
        expr = _replace_variables_in_constraint(con.func, flat_map)
        T = one(JuMP.value_type(typeof(sub_model)))
        sub_cref = JuMP.@constraint(sub_model, expr * T in con.set)
        push!(sub_crefs, sub_cref)
    end

    JuMP.set_optimizer(sub_model, method.nlp_optimizer)
    JuMP.set_silent(sub_model)
    return (
        GDPSubmodel(sub_model, decision_vars, fwd_map), sub_crefs
    )
end

# Solve the NLP subproblem for a given combo.
function _solve_loa_subproblem(
    model::M,
    combo::Dict{LogicalVariableRef{M}, Bool},
    method::LOA
    ) where {M <: JuMP.AbstractModel}
    active_crefs = _active_constraints(model, combo)
    sub, sub_crefs = copy_model_with_constraints(
        model, active_crefs, method
    )

    JuMP.optimize!(sub.model)

    if !JuMP.is_solved_and_feasible(sub.model)
        return _LOAIterationResult{M, Float64}(
            combo,
            Dict{JuMP.AbstractVariableRef, Float64}(),
            Dict{DisjunctConstraintRef{M}, Float64}(),
            Inf, false
        )
    end

    x_vals = Dict{JuMP.AbstractVariableRef, Float64}(
        var => JuMP.value(sub.fwd_map[var][1])
        for var in sub.decision_vars
    )

    duals = Dict{DisjunctConstraintRef{M}, Float64}()
    has_d = JuMP.has_duals(sub.model)
    for (orig, sub_c) in zip(active_crefs, sub_crefs)
        has_d && (duals[orig] = JuMP.dual(sub_c))
    end

    return _LOAIterationResult{M, Float64}(
        combo, x_vals, duals, JuMP.objective_value(sub.model), true
    )
end

################################################################################
#                      NONLINEAR DISJUNCT HELPERS
################################################################################
# True if a constraint function is nonlinear (GenericNonlinearExpr only —
# quadratic constraints are kept in the master for solvers that handle
# them).
_is_nonlinear(::JuMP.GenericNonlinearExpr) = true
_is_nonlinear(::Any) = false

# True if model has any nonlinear disjunct constraints.
function _has_nonlinear_disjuncts(model)
    for (_, dc) in _disjunct_constraints(model)
        _is_nonlinear(dc.constraint.func) && return true
    end
    return false
end

# Nonlinear non-disjunct constraints as (func, set) pairs. If
# `delete=true`, also removes them from the model (used on the master
# copy before BigM reformulation).
function _nonlinear_global_constraints(model; delete::Bool = false)
    pairs = Tuple{Any, Any}[]
    to_delete = Any[]
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: Union{JuMP.VariableRef, _MOI.VariableIndex} && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref isa DisjunctConstraintRef && continue
            con = JuMP.constraint_object(cref)
            _is_nonlinear(con.func) || continue
            push!(pairs, (con.func, con.set))
            delete && push!(to_delete, cref)
        end
    end
    for cref in to_delete
        JuMP.delete(model, cref)
    end
    return pairs
end

# Remove nonlinear disjunct constraints from GDP data so that BigM only
# reforms linear/quadratic ones.
function _remove_nonlinear_disjuncts(model)
    gd = gdp_data(model)
    nl_idxs = Set{DisjunctConstraintIndex}()
    for (idx, dc) in gd.disjunct_constraints
        _is_nonlinear(dc.constraint.func) && push!(nl_idxs, idx)
    end
    for idx in nl_idxs
        dcref = DisjunctConstraintRef(model, idx)
        ind = gd.constraint_to_indicator[dcref]
        if haskey(gd.indicator_to_constraints, ind)
            filter!(
                r -> !(r isa DisjunctConstraintRef &&
                       JuMP.index(r) == idx),
                gd.indicator_to_constraints[ind]
            )
        end
        delete!(gd.constraint_to_indicator, dcref)
        delete!(gd.disjunct_constraints, idx)
    end
    return nl_idxs
end

################################################################################
#                      MASTER PROBLEM BUILDER
################################################################################
function _build_loa_master(
    model::M, init_results, method::LOA
    ) where {M <: JuMP.AbstractModel}
    # Capture nonlinear globals from original model before copying (need
    # original var refs so linearization maps via master.ref_map).
    nl_globals = _nonlinear_global_constraints(model)

    master_model, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(master_model, method.mip_optimizer)
    JuMP.set_silent(master_model)
    _remove_nonlinear_disjuncts(master_model)
    _nonlinear_global_constraints(master_model; delete = true)
    reformulate_model(master_model, BigM(method.M_value))

    master = _LOAMaster{typeof(master_model), typeof(ref_map)}(
        master_model, ref_map, lv_map,
        JuMP.objective_sense(master_model),
        nl_globals
    )

    for result in init_results
        result.feasible && _add_oa_cuts(master, result, model, method)
        _add_no_good_cut(master, result.combo)
    end
    return master
end

# Add `sgn * OA_penalty_factor * slack` to the master objective, where
# `sgn = +1` for MIN (penalty increases cost) and `-1` for MAX.
function _apply_slack_penalty!(
    master::_LOAMaster, method::LOA, slack
    )
    sgn = master.obj_sense == _MOI.MIN_SENSE ? 1 : -1
    JuMP.set_objective_function(
        master.model,
        JuMP.objective_function(master.model) +
        sgn * method.OA_penalty_factor * slack)
    return
end

################################################################################
#                        OA CUT GENERATION
################################################################################
# Add outer approximation cuts at solution point xk. Pyomo GDPopt cut
# form:
#   sign(s * λ) * [r(xk) - rhs + ∇r(xk)ᵀ(x - xk)] - slack <= M*(1 - y)
function _add_oa_cuts(
    master::_LOAMaster,
    result::_LOAIterationResult{M, <:Any},
    model::M,
    method::LOA
    ) where {M <: JuMP.AbstractModel}
    sgn = master.obj_sense == _MOI.MIN_SENSE ? -1 : 1

    for (ind, active) in result.combo
        !active && continue
        haskey(_indicator_to_constraints(model), ind) || continue
        bin_var = binary_variable(master.lv_map[ind])

        for orig_cref in _indicator_to_constraints(model)[ind]
            orig_cref isa DisjunctConstraintRef || continue
            con_data = _disjunct_constraints(model)[JuMP.index(orig_cref)]
            con = con_data.constraint
            # Skip linear constraints (no OA needed)
            con.func isa JuMP.GenericAffExpr && continue

            dual_val = get(result.duals, orig_cref, nothing)
            dual_val === nothing && continue

            lin_expr = _linearize_at(
                con.func, result.x_values, master.ref_map
            )
            rhs = _set_rhs(con.set)
            s = sign(sgn * dual_val)
            s == 0 && continue

            slack = JuMP.@variable(
                master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack
            )
            _apply_slack_penalty!(master, method, slack)

            lhs = s * (lin_expr - rhs) - slack
            cref = JuMP.@constraint(
                master.model,
                lhs <= method.M_value * (1 - bin_var)
            )
            push!(_reformulation_constraints(master.model), cref)
        end
    end

    # Unconditional OA cuts for nonlinear global constraints (no
    # indicator gating, form: sign * (lin_expr - rhs) - slack <= 0).
    # Equality sets get both signs; other sets get one.
    for (func, set) in master.nl_globals
        lin_expr = _linearize_at(
            func, result.x_values, master.ref_map)
        rhs = _set_rhs(set)
        for s in _oa_global_signs(set)
            slack = JuMP.@variable(master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack)
            _apply_slack_penalty!(master, method, slack)
            lhs = s * (lin_expr - rhs) - slack
            cref = JuMP.@constraint(master.model, lhs <= 0)
            push!(_reformulation_constraints(master.model), cref)
        end
    end
end

# Sign coefficients for global OA cuts based on constraint set type.
# Equality needs both directions.
_oa_global_signs(::_MOI.LessThan) = (1,)
_oa_global_signs(::_MOI.GreaterThan) = (-1,)
_oa_global_signs(::_MOI.EqualTo) = (1, -1)
_oa_global_signs(::Any) = ()

################################################################################
#                       LINEARIZATION HELPERS
################################################################################
# First-order Taylor linearization of QuadExpr at xk.
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
            grad[vi] = get(grad, vi, 0.0) + 2 * coef * get(xk, vi, 0.0)
        else
            grad[vi] = get(grad, vi, 0.0) + coef * get(xk, vj, 0.0)
            grad[vj] = get(grad, vj, 0.0) + coef * get(xk, vi, 0.0)
        end
    end

    f_xk = func.aff.constant
    for (var, coef) in func.aff.terms
        f_xk += coef * get(xk, var, 0.0)
    end
    for (pair, coef) in func.terms
        f_xk += coef * get(xk, pair.a, 0.0) * get(xk, pair.b, 0.0)
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

# Convert JuMP expression tree to MOI Nonlinear Expr.
function _to_nlp_expr(expr::JuMP.GenericNonlinearExpr, idx::Dict)
    args = Any[_to_nlp_expr(a, idx) for a in expr.args]
    return Expr(:call, expr.head, args...)
end

function _to_nlp_expr(expr::JuMP.GenericAffExpr, idx::Dict)
    parts = Any[expr.constant]
    for (var, coef) in expr.terms
        vi = _MOI.VariableIndex(idx[var])
        push!(parts, Expr(:call, :*, coef, vi))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end

function _to_nlp_expr(expr::JuMP.GenericQuadExpr, idx::Dict)
    parts = Any[_to_nlp_expr(expr.aff, idx)]
    for (pair, coef) in expr.terms
        va = _MOI.VariableIndex(idx[pair.a])
        vb = _MOI.VariableIndex(idx[pair.b])
        push!(parts, Expr(:call, :*, coef, va, vb))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end

function _to_nlp_expr(var::JuMP.AbstractVariableRef, idx::Dict)
    return _MOI.VariableIndex(idx[var])
end

_to_nlp_expr(x::Number, ::Dict) = x

# Linearize a GenericNonlinearExpr at xk via MOI.Nonlinear reverse-mode
# AD.
function _linearize_at(
    func::JuMP.GenericNonlinearExpr, xk::Dict, ref_map
    )
    vars = JuMP.AbstractVariableRef[]
    _interrogate_variables(v -> push!(vars, v), func)
    unique!(vars)
    if isempty(vars)
        return JuMP.AffExpr(JuMP.value(v -> 0.0, func))
    end

    n = length(vars)
    T = JuMP.value_type(typeof(JuMP.owner_model(vars[1])))
    idx = Dict(vars[i] => i for i in 1:n)
    nlp = _MOI.Nonlinear.Model()
    _MOI.Nonlinear.set_objective(nlp, _to_nlp_expr(func, idx))
    ord = [_MOI.VariableIndex(i) for i in 1:n]
    evaluator = _MOI.Nonlinear.Evaluator(
        nlp, _MOI.Nonlinear.SparseReverseMode(), ord
    )
    _MOI.initialize(evaluator, [:Grad])

    xk_vec = [get(xk, v, zero(T)) for v in vars]
    f_xk = _MOI.eval_objective(evaluator, xk_vec)
    grad = zeros(T, n)
    _MOI.eval_objective_gradient(evaluator, grad, xk_vec)

    constant = T(f_xk)
    for i in 1:n
        constant -= grad[i] * xk_vec[i]
    end
    V = typeof(ref_map[vars[1]])
    result = JuMP.GenericAffExpr{T, V}(constant)
    for i in 1:n
        iszero(grad[i]) && continue
        JuMP.add_to_expression!(result, grad[i], ref_map[vars[i]])
    end
    return result
end

# Extract RHS value from an MOI set. Interval and other sets fall back
# to 0.0 (unused by current OA cut dispatches).
_set_rhs(set::Union{_MOI.LessThan, _MOI.GreaterThan, _MOI.EqualTo}) =
    _MOI.constant(set)
_set_rhs(::Any) = 0.0

################################################################################
#                       NO-GOOD CUT GENERATION
################################################################################
function _add_no_good_cut(master::_LOAMaster, combo)
    T = JuMP.value_type(typeof(master.model))
    V = JuMP.variable_ref_type(master.model)
    cut_expr = zero(JuMP.GenericAffExpr{T, V})
    for (ind, active) in combo
        haskey(master.lv_map, ind) || continue
        bin_var = binary_variable(master.lv_map[ind])
        if active
            JuMP.add_to_expression!(
                cut_expr, -one(T), bin_var)
            JuMP.add_to_expression!(
                cut_expr, one(T))
        else
            JuMP.add_to_expression!(
                cut_expr, one(T), bin_var)
        end
    end
    cref = JuMP.@constraint(
        master.model, cut_expr >= one(T))
    push!(
        _reformulation_constraints(master.model), cref)
end

################################################################################
#                     MASTER SOLUTION EXTRACTION
################################################################################
function _extract_combo(
    model::M, master::_LOAMaster
    ) where {M <: JuMP.AbstractModel}
    combo = Dict{LogicalVariableRef{M}, Bool}()
    for (_, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        for ind in disj_data.constraint.indicators
            combo[ind] = JuMP.value(master.lv_map[ind])
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
#                       FINAL MODEL SETUP
################################################################################
# Collect active nonlinear constraints from best combo, fix indicators,
# strip nonlinear disjuncts, BigM-reform linear ones, then re-add the
# nonlinear constraints directly with an NLP solver.
function _finalize_model(model, best_result, method)
    has_nl = _has_nonlinear_disjuncts(model) ||
        !isempty(_nonlinear_global_constraints(model))
    if !has_nl
        # Pure linear/quadratic: standard BigM
        reformulate_model(model, BigM(method.M_value))
        return
    end

    # Fix indicators, collect active nonlinear constraints.
    nl_active = Tuple{Any, Any}[]
    if best_result !== nothing
        for (ind, active) in best_result.combo
            bv = _indicator_to_binary(model)[ind]
            if bv isa JuMP.AbstractVariableRef
                JuMP.fix(bv, active ? 1.0 : 0.0; force = true)
            end
        end
        for cref in _active_constraints(model, best_result.combo)
            c = _disjunct_constraints(model)[JuMP.index(cref)].constraint
            _is_nonlinear(c.func) && push!(nl_active, (c.func, c.set))
        end
        for (var, val) in best_result.x_values
            JuMP.is_valid(model, var) &&
                JuMP.set_start_value(var, val)
        end
    end

    # BigM-reform only linear/quadratic disjuncts
    _remove_nonlinear_disjuncts(model)
    reformulate_model(model, BigM(method.M_value))

    # Relax integrality and switch to NLP solver
    for var in JuMP.all_variables(model)
        JuMP.is_binary(var) && JuMP.unset_binary(var)
        JuMP.is_integer(var) && JuMP.unset_integer(var)
    end
    JuMP.set_optimizer(model, method.nlp_optimizer)

    # Re-add active nonlinear constraints directly
    for (func, set) in nl_active
        cref = JuMP.@constraint(model, func in set)
        push!(_reformulation_constraints(model), cref)
    end
    return
end
