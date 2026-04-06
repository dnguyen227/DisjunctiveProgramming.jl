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
        result = _solve_loa_subproblem(model, combo, method)
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
        z_lower = objective_value(master.model)
        _loa_converged(z_upper, z_lower, method) && break

        combo = _extract_combo_from_master(model, master)
        result = _solve_loa_subproblem(model, combo, method)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
        end

        result.feasible && _add_oa_cuts!(
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
# Number of reformulated constraints BigM produces per disjunct
# constraint. Matches reformulate_disjunct_constraint in bigm.jl.
_num_reform_cons(::JuMP.ScalarConstraint{<:Any, <:_MOI.LessThan}) = 1
_num_reform_cons(::JuMP.ScalarConstraint{<:Any, <:_MOI.GreaterThan}) = 1
_num_reform_cons(::JuMP.ScalarConstraint{<:Any, <:_MOI.EqualTo}) = 2
_num_reform_cons(::JuMP.ScalarConstraint{<:Any, <:_MOI.Interval}) = 2
_num_reform_cons(::JuMP.VectorConstraint) = 1
_num_reform_cons(::Any) = 0

# Build mapping: sub DisjunctConstraintRef → Vector of BigM-
# reformulated constraint refs.
function _build_reform_map(sub::JuMP.AbstractModel)
    ref_cons = _reformulation_constraints(sub)
    mapping = Dict{DisjunctConstraintRef, Vector}()
    idx = 1
    for (_, disj_data) in _disjunctions(sub)
        disj_data.constraint.nested && continue
        for ind in disj_data.constraint.indicators
            haskey(_indicator_to_constraints(sub), ind) || continue
            for cref in _indicator_to_constraints(sub)[ind]
                cref isa DisjunctConstraintRef || continue
                con = JuMP.constraint_object(cref)
                n = _num_reform_cons(con)
                if n > 0 && idx + n - 1 <= length(ref_cons)
                    mapping[cref] = ref_cons[idx:idx+n-1]
                end
                idx += n
            end
        end
    end
    return mapping
end

# Sum duals across reformulated constraint refs for one disjunct
# constraint. Returns nothing if unavailable.
function _sum_duals(reform_map, sub_cref)
    haskey(reform_map, sub_cref) || return nothing
    try
        return sum(JuMP.dual(rc) for rc in reform_map[sub_cref])
    catch
        return nothing
    end
end

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

# Fix binary variables according to a Boolean combo.
function _fix_loa_binaries!(sub, combo, lv_map)
    ind_to_bin = _indicator_to_binary(sub)
    for (ind, active) in combo
        mapped_ind = lv_map[ind]
        haskey(ind_to_bin, mapped_ind) || continue
        bin_var = ind_to_bin[mapped_ind]
        bin_var isa JuMP.AbstractVariableRef || continue
        JuMP.fix(bin_var, active ? 1.0 : 0.0; force = true)
    end
end

################################################################################
#                      NLP SUBPROBLEM SOLVER
################################################################################
function _solve_loa_subproblem(
    model::M, combo::Dict{LogicalVariableRef{M}, Bool},
    method::LOA
    ) where {M <: JuMP.AbstractModel}
    sub, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(sub, method.optimizer)
    JuMP.set_silent(sub)
    reformulate_model(sub, BigM(method.M_value))
    reform_map = _build_reform_map(sub)
    _fix_loa_binaries!(sub, combo, lv_map)

    JuMP.optimize!(sub, ignore_optimize_hook = true)

    if !JuMP.is_solved_and_feasible(sub)
        return _LOAIterationResult{M}(
            combo,
            Dict{JuMP.AbstractVariableRef, Float64}(),
            Dict{DisjunctConstraintRef{M}, Float64}(),
            Inf, false)
    end

    x_vals = Dict{JuMP.AbstractVariableRef, Float64}(
        var => JuMP.value(ref_map[var])
        for var in collect_all_vars(model))

    duals = Dict{DisjunctConstraintRef{M}, Float64}()
    for (ind, active) in combo
        !active && continue
        haskey(_indicator_to_constraints(model), ind) || continue
        for orig_cref in _indicator_to_constraints(model)[ind]
            orig_cref isa DisjunctConstraintRef || continue
            sub_cref = DisjunctConstraintRef(sub, orig_cref.index)
            d = _sum_duals(reform_map, sub_cref)
            d !== nothing && (duals[orig_cref] = d)
        end
    end

    return _LOAIterationResult{M}(
        combo, x_vals, duals, objective_value(sub), true)
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
        result.feasible && _add_oa_cuts!(
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
function _add_oa_cuts!(
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

            slack = @variable(master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack)
            push!(master.slack_vars, slack)

            if bin_var !== nothing
                @constraint(master.model,
                    s * (lin_expr - rhs) - slack <=
                    method.M_value * (1 - bin_var))
            else
                @constraint(master.model,
                    s * (lin_expr - rhs) - slack <= 0)
            end
        end
    end
end

# Linearize an AffExpr at point xk (exact).
function _linearize_at(
    func::JuMP.GenericAffExpr, xk::Dict, ref_map
    )
    result = JuMP.AffExpr(func.constant)
    for (var, coef) in func.terms
        JuMP.add_to_expression!(result, coef, ref_map[var])
    end
    return result
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
    @constraint(master.model, cut_expr >= 1.0)
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
