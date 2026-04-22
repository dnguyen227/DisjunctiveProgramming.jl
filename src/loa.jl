################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
# With augmented-penalty OA master from:
# Viswanathan & Grossmann (1990), Comp. & Chem. Eng. 14(7), 769-782
################################################################################

"""
    LOA{O, P} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models. Uses two models: the
original (BigM-reformulated, binaries fixed per iteration as an NLP) and a
master MILP copy that accumulates OA and no-good cuts.
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
        max_iter::Int = 10, atol::Float64 = 1e-6,
        rtol::Float64 = 1e-4, M_value::Float64 = 1e9,
        max_slack::Float64 = 1000.0, OA_penalty_factor::Float64 = 1000.0
        ) where {O, P}
        new{O, P}(nlp_optimizer, mip_optimizer, max_iter, atol, rtol,
            M_value, max_slack, OA_penalty_factor)
    end
end

################################################################################
#                          DATA STRUCTURES
################################################################################
# Master MILP state. orig_obj is the original objective (for OA cuts);
# alpha_oa replaces it (Türkay & Grossmann 1996). obj_ref_map maps the
# variables in orig_obj to master variables.
mutable struct _LOAMaster{M <: JuMP.AbstractModel, B, V}
    model::M
    bin_map::B
    var_map::V
    obj_sense::_MOI.OptimizationSense
    orig_obj::Any
    alpha_oa::Any
    obj_ref_map::Any
end

# Feasibility-restoration submodel (Viswanathan & Grossmann 1990, NLPF).
# Standalone model with a shared scalar slack `u` embedded into every JuMP
# constraint and `min u` as the objective. When the primary NLP is
# infeasible, binaries are fixed here and the submodel is solved to
# produce a least-infeasible point for OA cut generation. Keeping it
# separate leaves the original NLP model clean (no `u`, no slackened
# constraints).
#
# fwd_map: original-model var -> feas var, used for binary fixing and
# value extraction.
# obj_ref_map: original-objective var -> feas var, used to linearize the
# original objective at the feas point.
struct _LOAFeasSubmodel{M <: JuMP.AbstractModel}
    model::M
    fwd_map::Any
    obj_ref_map::Any
end

################################################################################
#                           MAIN ALGORITHM
################################################################################
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    combos = _set_covering_combos(model)
    reformulate_model(model, BigM(method.M_value))

    #build the feasibility-restoration submodel as a separate copy, then
    #embed the shared scalar slack `u` in every constraint (V&G 1990,
    #NLPF). The original model stays clean: no slacks, no integrality
    #relaxation, no objective change.
    feas = copy_model_with_constraints(model, method)
    _embed_feas_slack(feas)

    master = build_loa_master(model, method)
    reform_map = _build_reform_map(model)
    JuMP.relax_integrality(model)
    JuMP.set_optimizer(model, method.nlp_optimizer)
    JuMP.set_silent(model)

    sense_token = Val(JuMP.objective_sense(model))
    best_obj = _worst_obj(sense_token)
    is_better(o) = _is_better(sense_token, o, best_obj)
    best_result = nothing
    for combo in combos
        result = _solve_nlp(model, combo, method, reform_map, feas)
        _add_no_good_cut(model, master, combo)
        _add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_obj = result.objective
            best_result = result
        end
    end

    master_bound = _worst_obj(_flip_sense(sense_token))
    for iter in 1:method.max_iter
        JuMP.optimize!(master.model)
        JuMP.is_solved_and_feasible(master.model) || break
        master_bound = JuMP.objective_value(master.model)
        _loa_converged(best_obj, master_bound, sense_token, method) && break
        combo = _extract_combo(model, master)
        result = _solve_nlp(model, combo, method, reform_map, feas)
        _add_no_good_cut(model, master, combo)
        _add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_obj = result.objective
            best_result = result
        end
    end

    _finalize_model(model, best_result)
    return
end

#fix best combo permanently and set start values
function _finalize_model(model, best_result)
    best_result === nothing && return
    fix_combo_binaries(model, best_result.combo)
    for (var, v) in best_result.x_values
        JuMP.is_valid(model, var) || continue
        v isa Number || continue
        JuMP.set_start_value(var, v)
    end
end

################################################################################
#                      EXTENSION POINTS
################################################################################
#build master MILP from the BigM-reformulated original
function build_loa_master(model::JuMP.AbstractModel, method::LOA)
    orig_obj = JuMP.objective_function(model)
    master, copy_map = JuMP.copy_model(model)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)
    bin_map = Dict{LogicalVariableRef, Any}()
    for (ind, bv) in _indicator_to_binary(model)
        bin_map[ind] = copy_map[bv]
    end
    #replace objective with alpha_oa; OA cuts are added each iteration
    obj_sense = JuMP.objective_sense(master)
    alpha_oa = JuMP.@variable(master, base_name = "alpha_oa")
    JuMP.@objective(master, obj_sense, alpha_oa)
    return _LOAMaster(
        master, bin_map, copy_map, obj_sense, orig_obj, alpha_oa, copy_map)
end

"""
    copy_model_with_constraints(model, method::LOA)

Build an LOA feas submodel: a fresh deep copy of `model` with every
non-bound constraint copied over, wired up with `method.nlp_optimizer`.
The returned submodel is a raw copy; slack embedding and the `min u`
objective are applied separately via [`_embed_feas_slack`]. Shares the
dispatch entry point with MBM's [`copy_model_with_constraints`](@ref),
which instead takes an explicit subset of disjunct constraints.
"""
function copy_model_with_constraints(
    model::JuMP.AbstractModel, method::LOA
    )
    var_type = JuMP.variable_ref_type(model)
    sub_model = _copy_model(model)
    fwd_map = Dict{var_type, var_type}()
    for var in collect_all_vars(model)
        fwd_map[var] = variable_copy(sub_model, var)
    end
    VT = JuMP.variable_ref_type(typeof(model))
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === VT && continue
        for cref in JuMP.all_constraints(model, F, S)
            con = JuMP.constraint_object(cref)
            expr = _replace_variables_in_constraint(con.func, fwd_map)
            JuMP.@constraint(sub_model, expr in con.set)
        end
    end
    JuMP.set_optimizer(sub_model, method.nlp_optimizer)
    JuMP.set_silent(sub_model)
    return _LOAFeasSubmodel(sub_model, fwd_map, fwd_map)
end

#embed shared scalar slack `u` into every constraint of the feas
#submodel, relax integrality, and set `min u` as the objective. Converts
#the raw copy produced by `copy_model_with_constraints` into a V&G 1990
#NLPF feasibility-restoration problem.
function _embed_feas_slack(feas::_LOAFeasSubmodel)
    u = JuMP.@variable(feas.model, base_name = "_loa_u", lower_bound = 0.0)
    _slacken_model_constraints(feas.model, u)
    JuMP.relax_integrality(feas.model)
    JuMP.@objective(feas.model, Min, u)
    return
end

#replace every JuMP constraint in `model` (excluding variable bounds) with
#its slackened counterpart via `_slacken`. Shared by finite and InfiniteOpt
#feas submodel builders.
function _slacken_model_constraints(model, u)
    VT = JuMP.variable_ref_type(typeof(model))
    to_slacken = Any[]
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === VT && continue
        for cref in JuMP.all_constraints(model, F, S)
            push!(to_slacken, cref)
        end
    end
    for cref in to_slacken
        JuMP.is_valid(model, cref) || continue
        con = JuMP.constraint_object(cref)
        for (sf, ss) in _slacken(con.func, con.set, u)
            JuMP.@constraint(model, sf in ss)
        end
        JuMP.delete(model, cref)
    end
    return
end

#fix/unfix indicator binaries on the original model
function fix_combo_binaries(model, combo)
    for (ind, active) in combo
        bv = _indicator_to_binary(model)[ind]
        JuMP.fix(bv, active ? 1.0 : 0.0; force = true)
    end
end
function unfix_combo_binaries(model, combo)
    for (ind, _) in combo
        bv = _indicator_to_binary(model)[ind]
        JuMP.is_fixed(bv) && JuMP.unfix(bv)
    end
end

################################################################################
#                      SET COVERING INITIALIZATION
################################################################################
function _set_covering_combos(model::JuMP.AbstractModel)
    M = typeof(model)
    LVR = LogicalVariableRef{M}
    per_disj = Vector{Tuple{DisjunctionIndex, LVR}}[]
    for (idx, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        push!(per_disj, [(idx, ind) for ind in disj_data.constraint.indicators])
    end
    isempty(per_disj) && return Dict{LVR, Bool}[]
    all_combos = [Tuple{DisjunctionIndex, LVR}[c...]
        for c in Iterators.product(per_disj...)]
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
#                      NLP SUBPROBLEM
################################################################################
# Termination tokens. `_solve_nlp` and `_run_feas_restoration` dispatch on
# these to build the result tuple without inspecting the solver status at
# multiple call sites.
struct _Feasible end
struct _Infeasible end
_nlp_status(model) = JuMP.is_solved_and_feasible(model) ?
    _Feasible() : _Infeasible()

#solve primary NLP for a fixed combo; on infeasibility, dispatch to the
#feasibility-restoration submodel (min u) to get a least-infeasible point
#for OA cut generation (Viswanathan & Grossmann 1990)
function _solve_nlp(
    model::M, combo, method::LOA, reform_map, feas::_LOAFeasSubmodel
    ) where {M <: JuMP.AbstractModel}
    fix_combo_binaries(model, combo)
    JuMP.optimize!(model, ignore_optimize_hook = true)
    result = _nlp_primary_result(
        _nlp_status(model), model, combo, reform_map, feas)
    unfix_combo_binaries(model, combo)
    return result
end

#feasible primary NLP: pack up x-values, duals, objective
function _nlp_primary_result(
    ::_Feasible, model::M, combo, reform_map, feas
    ) where {M <: JuMP.AbstractModel}
    x_vals, obj_xv = extract_primary_x_values(model)
    duals = _collect_nlp_duals(model, combo, reform_map)
    return (combo = combo, x_values = x_vals, duals = duals,
        objective = JuMP.objective_value(model), feasible = true,
        obj_x_values = obj_xv)
end

#infeasible primary NLP: hand off to the feas-restoration submodel
function _nlp_primary_result(
    ::_Infeasible, model, combo, reform_map, feas
    )
    return _run_feas_restoration(model, feas, combo)
end

#fix binaries on the feas submodel, solve min u, and dispatch on outcome
function _run_feas_restoration(
    model::M, feas::_LOAFeasSubmodel, combo
    ) where {M <: JuMP.AbstractModel}
    _fix_feas_combo_binaries(feas, model, combo)
    JuMP.optimize!(feas.model)
    result = _nlp_feas_result(_nlp_status(feas.model), model, feas, combo)
    _unfix_feas_combo_binaries(feas, model, combo)
    return result
end

#feas submodel solved: return the least-infeasible point for OA cuts
function _nlp_feas_result(
    ::_Feasible, model::M, feas, combo
    ) where {M <: JuMP.AbstractModel}
    x_vals, obj_xv = extract_feas_x_values(model, feas)
    return (combo = combo, x_values = x_vals,
        duals = Dict{DisjunctConstraintRef{M}, Any}(),
        objective = Inf, feasible = false, obj_x_values = obj_xv)
end

#feas submodel also infeasible: return empty result (no OA cut)
function _nlp_feas_result(
    ::_Infeasible, model::M, feas, combo
    ) where {M <: JuMP.AbstractModel}
    empty = Dict{JuMP.AbstractVariableRef, Any}()
    return (combo = combo, x_values = empty,
        duals = Dict{DisjunctConstraintRef{M}, Any}(),
        objective = Inf, feasible = false, obj_x_values = empty)
end

#collect duals of reformulated disjunct constraints for active disjuncts
function _collect_nlp_duals(
    model::M, combo, reform_map
    ) where {M <: JuMP.AbstractModel}
    duals = Dict{DisjunctConstraintRef{M}, Any}()
    JuMP.has_duals(model) || return duals
    for (ind, active) in combo
        active || continue
        haskey(_indicator_to_constraints(model), ind) || continue
        for cref in _indicator_to_constraints(model)[ind]
            cref isa DisjunctConstraintRef || continue
            duals[cref] = _sum_duals(reform_map, cref)
        end
    end
    return duals
end

#extract x-values from the primary NLP. Returns (x_vals, obj_x_values)
#where obj_x_values is keyed by whatever variables the original objective
#uses (same as x_vals for finite; flat-transcription vars for infinite).
function extract_primary_x_values(model::JuMP.AbstractModel)
    x_vals = Dict{JuMP.AbstractVariableRef, Any}()
    for v in JuMP.all_variables(model)
        JuMP.is_fixed(v) && continue
        x_vals[v] = JuMP.value(v)
    end
    return x_vals, x_vals
end

#extract x-values from the feasibility submodel, keyed by original-model
#variables via `feas.fwd_map`. Same return contract as the primary path.
function extract_feas_x_values(
    model::JuMP.AbstractModel, feas::_LOAFeasSubmodel
    )
    x_vals = Dict{JuMP.AbstractVariableRef, Any}()
    for v in JuMP.all_variables(model)
        JuMP.is_fixed(v) && continue
        haskey(feas.fwd_map, v) || continue
        x_vals[v] = JuMP.value(feas.fwd_map[v])
    end
    return x_vals, x_vals
end

#fix/unfix indicator binaries on the feas submodel via fwd_map. Dispatches
#through `fix_fv`/`unfix_fv` to handle scalar feas vars (finite) and
#per-support vectors (InfiniteOpt flat transcription).
function _fix_feas_combo_binaries(feas::_LOAFeasSubmodel, model, combo)
    for (ind, val) in combo
        orig_bv = _indicator_to_binary(model)[ind]
        haskey(feas.fwd_map, orig_bv) || continue
        fix_fv(feas.fwd_map[orig_bv], val)
    end
end
function _unfix_feas_combo_binaries(feas::_LOAFeasSubmodel, model, combo)
    for (ind, _) in combo
        orig_bv = _indicator_to_binary(model)[ind]
        haskey(feas.fwd_map, orig_bv) || continue
        unfix_fv(feas.fwd_map[orig_bv])
    end
end

#fix/unfix a feas-submodel binary variable at a scalar Bool value
fix_fv(bv, val::Bool) = JuMP.fix(bv, val ? 1.0 : 0.0; force = true)
function unfix_fv(bv)
    JuMP.is_fixed(bv) && JuMP.unfix(bv)
    return
end

################################################################################
#                       REFORM MAP (BigM constraint index)
################################################################################
function _build_reform_map(model::M) where {M <: JuMP.AbstractModel}
    ref_cons = _reformulation_constraints(model)
    isempty(ref_cons) && return Dict{DisjunctConstraintRef{M}, Vector{Any}}()
    CRT = eltype(ref_cons)
    rmap = Dict{DisjunctConstraintRef{M}, Vector{CRT}}()
    idx = 1
    for (_, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        for ind in disj_data.constraint.indicators
            haskey(_indicator_to_constraints(model), ind) || continue
            for cref in _indicator_to_constraints(model)[ind]
                cref isa DisjunctConstraintRef || continue
                con = _disjunct_constraints(model)[JuMP.index(cref)].constraint
                n = _num_reform_cons(con)
                idx + n - 1 <= length(ref_cons) || break
                rmap[cref] = collect(ref_cons[idx:idx + n - 1])
                idx += n
            end
        end
    end
    return rmap
end

_num_reform_cons(::JuMP.ScalarConstraint{T, S}) where {
    T <: Union{Number, JuMP.AbstractJuMPScalar},
    S <: Union{_MOI.EqualTo, _MOI.Interval}} = 2
_num_reform_cons(::JuMP.ScalarConstraint) = 1
_num_reform_cons(con::JuMP.VectorConstraint{T, S}) where {
    T <: Union{Number, JuMP.AbstractJuMPScalar},
    S <: _MOI.Zeros} = 2 * _MOI.dimension(con.set)
_num_reform_cons(con::JuMP.VectorConstraint) = _MOI.dimension(con.set)

function _sum_duals(reform_map, cref)
    haskey(reform_map, cref) || return 0.0
    rcs = reform_map[cref]
    isempty(rcs) && return 0.0
    total = JuMP.dual(rcs[1])
    for i in 2:length(rcs)
        total = total .+ JuMP.dual(rcs[i])
    end
    return total
end

################################################################################
#                      COMBO EXTRACTION + CUTS
################################################################################
#extract combo from master solution. `combo_val` dispatches on scalar vs
#array bin_map values (extension adds the array method for per-support).
function _extract_combo(
    model::M, master::_LOAMaster
    ) where {M <: JuMP.AbstractModel}
    combo = Dict{LogicalVariableRef{M}, Any}()
    for (_, disj_data) in _disjunctions(model)
        disj_data.constraint.nested && continue
        for ind in disj_data.constraint.indicators
            haskey(master.bin_map, ind) || continue
            combo[ind] = combo_val(master.bin_map[ind])
        end
    end
    return combo
end
combo_val(bv) = round(JuMP.value(bv)) > 0.5

#no-good cut on the master excluding the current combo. `add_ng_terms`
#dispatches on scalar vs array bin_map / active values.
function _add_no_good_cut(model, master::_LOAMaster, combo)
    cut_expr = JuMP.AffExpr(0.0)
    for (ind, active) in combo
        haskey(master.bin_map, ind) || continue
        add_ng_terms(cut_expr, master.bin_map[ind], active)
    end
    JuMP.@constraint(master.model, cut_expr >= 1.0)
end
function add_ng_terms(cut, bv, active::Bool)
    if active
        JuMP.add_to_expression!(cut, -1.0, bv)
        JuMP.add_to_expression!(cut, 1.0)
    else
        JuMP.add_to_expression!(cut, 1.0, bv)
    end
    return
end

#predicate used by OA cut generation: is there any active indicator in
#this combo entry? Scalar Bool case; extension adds `AbstractVector{Bool}`
any_active(active::Bool) = active

################################################################################
#                       LINEARIZATION HELPERS
################################################################################
function _linearize_at(var::JuMP.AbstractVariableRef, xk, ref_map)
    return JuMP.AffExpr(0.0, ref_map[var] => 1.0)
end
function _linearize_at(func::JuMP.GenericAffExpr, xk, ref_map)
    result = JuMP.AffExpr(func.constant)
    for (var, coef) in func.terms
        JuMP.add_to_expression!(result, coef, ref_map[var])
    end
    return result
end
################################################################################
#                    SLACK HELPERS
################################################################################
_slacken(f, set::_MOI.LessThan, u) = [(f - u, set)]
_slacken(f, set::_MOI.GreaterThan, u) = [(f + u, set)]
function _slacken(f, set::_MOI.EqualTo, u)
    b = _MOI.constant(set)
    return [(f - u, _MOI.LessThan(b)), (f + u, _MOI.GreaterThan(b))]
end
_slacken(f, set, u) = [(f, set)]

################################################################################
#                        OA CUT GENERATION
################################################################################
#sense-dependent coefficients dispatched on Val(obj_sense). See usage in
#`_add_objective_oa_cut` and `_add_disjunct_oa_cuts`.
_disjunct_cut_coeffs(::Val{_MOI.MIN_SENSE}) = (-1, 1)
_disjunct_cut_coeffs(::Val{_MOI.MAX_SENSE}) = (1, -1)
_worst_obj(::Val{_MOI.MIN_SENSE}) = Inf
_worst_obj(::Val{_MOI.MAX_SENSE}) = -Inf
_is_better(::Val{_MOI.MIN_SENSE}, new, best) = new < best
_is_better(::Val{_MOI.MAX_SENSE}, new, best) = new > best
_gap(::Val{_MOI.MIN_SENSE}, best, bound) = best - bound
_gap(::Val{_MOI.MAX_SENSE}, best, bound) = bound - best
_flip_sense(::Val{_MOI.MIN_SENSE}) = Val(_MOI.MAX_SENSE)
_flip_sense(::Val{_MOI.MAX_SENSE}) = Val(_MOI.MIN_SENSE)

function _add_oa_cuts(model, master::_LOAMaster, result, method::LOA)
    isempty(result.x_values) && return
    _add_objective_oa_cut(master, result)
    _add_disjunct_oa_cuts(model, master, result, method)
    return
end

#linearize the original objective at `result.obj_x_values` and add it as
#a cut on `alpha_oa` with direction determined by `master.obj_sense`
function _add_objective_oa_cut(master::_LOAMaster, result)
    lin = _linearize_at(
        master.orig_obj, result.obj_x_values, master.obj_ref_map)
    _add_obj_cut(Val(master.obj_sense), master, lin)
    return
end
_add_obj_cut(::Val{_MOI.MIN_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin <= master.alpha_oa)
_add_obj_cut(::Val{_MOI.MAX_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin >= master.alpha_oa)

#add V&G 1990 augmented-penalty OA cuts for each active disjunct's
#nonlinear constraints. `cut_sites` yields the per-cut tuples: 1 site
#for finite, K sites for the InfiniteOpt case (extension override).
function _add_disjunct_oa_cuts(
    model, master::_LOAMaster, result, method::LOA
    )
    sgn, pen = _disjunct_cut_coeffs(Val(master.obj_sense))
    for (ind, active) in result.combo
        any_active(active) || continue
        haskey(master.bin_map, ind) || continue
        haskey(_indicator_to_constraints(model), ind) || continue
        for orig_cref in _indicator_to_constraints(model)[ind]
            orig_cref isa DisjunctConstraintRef || continue
            con = _disjunct_constraints(model)[
                JuMP.index(orig_cref)].constraint
            con.func isa JuMP.GenericAffExpr && continue
            d = get(result.duals, orig_cref, nothing)
            d === nothing && continue
            for (bv, xk, smap, d_k) in cut_sites(
                master.bin_map[ind], active,
                result.x_values, master.var_map, d)
                dv = _dv(d_k)
                s = sign(sgn * dv)
                s == 0 && continue
                _add_one_disjunct_oa_cut(
                    master, con, method, bv, xk, smap, s, pen)
            end
        end
    end
end

#one OA cut site: (binary, x_values, var_map, dual). Scalar case yields
#a single-element tuple collection; extension's array dispatch yields
#K per-support sites with sliced x_values and var_map.
cut_sites(bv, active::Bool, x_values, var_map, d) =
    ((bv, x_values, var_map, d),)

#extract a scalar dual value from a dual "cell" that may be a vector
#(from a vector-valued reform constraint in the infinite case)
_dv(d::Number) = d
_dv(d) = sum(d)

#add a single disjunct OA cut with augmented-penalty slack
function _add_one_disjunct_oa_cut(
    master::_LOAMaster, con, method, bv, x_values, var_map, s, pen
    )
    lin_expr = _linearize_at(con.func, x_values, var_map)
    rhs = _set_rhs(con.set)
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
        pen * method.OA_penalty_factor * slack)
    JuMP.@constraint(master.model,
        s * (lin_expr - rhs) - slack <=
        method.M_value * (1 - bv))
    return
end

################################################################################
#                       CONVERGENCE CHECK
################################################################################
function _loa_converged(best_obj, master_bound, sense_token, method::LOA)
    (isinf(best_obj) || isinf(master_bound)) && return false
    gap = _gap(sense_token, best_obj, master_bound)
    gap <= method.atol && return true
    abs(best_obj) > 1e-10 && gap / abs(best_obj) <= method.rtol && return true
    return false
end
_loa_converged(z_upper, z_lower, method::LOA) =
    _loa_converged(z_upper, z_lower, Val(_MOI.MIN_SENSE), method)

################################################################################
#                          ERROR FALLBACK
################################################################################
function reformulate_model(::M, ::LOA) where {M}
    error("reformulate_model not implemented for model type `$(M)` with LOA.")
end
