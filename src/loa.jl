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
#                           MAIN ALGORITHM
################################################################################
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    combos = _set_covering_combos(model)
    reformulate_model(model, BigM(method.M_value))

    master = build_loa_master(model, method)
    reform_map = _build_reform_map(model)
    JuMP.set_optimizer(model, method.nlp_optimizer)
    JuMP.set_silent(model)

    sense_token = Val(JuMP.objective_sense(model))
    best_obj = _worst_obj(sense_token)
    is_better(o) = _is_better(sense_token, o, best_obj)
    best_result = nothing

    #Initialization Procedure (Türkay & Grossmann 1996 §2.2): solve the
    #set-covering NLPs to seed the master with at least one OA cut per
    #disjunct before the main iteration.
    for combo in combos
        result = _solve_nlp(model, combo, method, reform_map)
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
        result = _solve_nlp(model, combo, method, reform_map)
        _add_no_good_cut(model, master, combo)
        _add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_obj = result.objective
            best_result = result
        end
    end

    _finalize_model(model, best_result)
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
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
"""
    build_loa_master(model::JuMP.AbstractModel, method::LOA)

Build the LOA master MILP as a deep copy of the BigM-reformulated
`model`, install the `alpha_oa` objective auxiliary, and wire up
`method.mip_optimizer`. OA and no-good cuts are added to the returned
master each iteration of the main LOA loop.

## Returns
- `NamedTuple` with `model` (master MILP), `bin_map`
  (indicator→binary), `var_map` (original→master var),
  `obj_sense`, `orig_obj`, `alpha_oa`, and `obj_ref_map`
  (objective-side map for linearization).
"""
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
    return (model = master, bin_map = bin_map, var_map = copy_map,
        obj_sense = obj_sense, orig_obj = orig_obj,
        alpha_oa = alpha_oa, obj_ref_map = copy_map)
end

"""
    copy_model_with_constraints(
        model::JuMP.AbstractModel,
        disjunct_crefs::Vector{<:DisjunctConstraintRef},
        method::LOA
        )

Build a raw feasibility-restoration submodel by deep-copying `model`'s
decision variables, then including only the problem's global
constraints (those not added by BigM reformulation) and the original
pre-BigM constraints of `disjunct_crefs`. The shared slack `u` and
`min u` objective are applied separately by [`_embed_feas_slack`](@ref).
Mirrors MBM's subset-taking [`copy_model_with_constraints`](@ref).

## Returns
- `NamedTuple` with `sub::GDPSubmodel` (copied model and
  original→copy forward map) and `obj_ref_map` (objective-side
  reference map for linearizing the original objective at the feas
  point).
"""
function copy_model_with_constraints(
    model::JuMP.AbstractModel,
    disjunct_crefs::Vector{<:DisjunctConstraintRef},
    method::LOA
    )
    V = JuMP.variable_ref_type(model)
    sub_model = _copy_model(model)
    decision_vars = collect_all_vars(model)
    fwd_map = Dict{V, V}()
    for var in decision_vars
        fwd_map[var] = variable_copy(sub_model, var)
    end
    VT = JuMP.variable_ref_type(typeof(model))
    reform_set = is_gdp_model(model) ?
        Set(_reformulation_constraints(model)) : Set()
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === VT && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            con = JuMP.constraint_object(cref)
            expr = _replace_variables_in_constraint(con.func, fwd_map)
            JuMP.@constraint(sub_model, expr in con.set)
        end
    end
    for cref in disjunct_crefs
        con = JuMP.constraint_object(cref)
        expr = _replace_variables_in_constraint(con.func, fwd_map)
        JuMP.@constraint(sub_model, expr in con.set)
    end
    JuMP.set_optimizer(sub_model, method.nlp_optimizer)
    JuMP.set_silent(sub_model)
    return (sub = GDPSubmodel(sub_model, decision_vars, fwd_map),
        obj_ref_map = fwd_map)
end

# Convert the raw copy from `copy_model_with_constraints` into a V&G 1990
# NLPF problem: shared slack `u` embedded in every constraint via
# `_slacken`, integrality relaxed, `min u` as the objective.
function _embed_feas_slack(feas)
    m = feas.sub.model
    u = JuMP.@variable(m, base_name = "_loa_u", lower_bound = 0.0)
    VT = JuMP.variable_ref_type(typeof(m))
    to_slacken = Any[]
    for (F, S) in JuMP.list_of_constraint_types(m)
        F === VT && continue
        for cref in JuMP.all_constraints(m, F, S)
            push!(to_slacken, cref)
        end
    end
    for cref in to_slacken
        JuMP.is_valid(m, cref) || continue
        con = JuMP.constraint_object(cref)
        for (sf, ss) in _slacken(con.func, con.set, u)
            JuMP.@constraint(m, sf in ss)
        end
        JuMP.delete(m, cref)
    end
    JuMP.relax_integrality(m)
    JuMP.@objective(m, Min, u)
    return
end

"""
    fix_combo_binaries(model::JuMP.AbstractModel, combo)::Nothing

Fix every indicator binary in `combo` to its active / inactive value
on `model`. Complement indicators (stored as `1 - other_bv`) are
handled by fixing the underlying variable to the complement value via
[`fix_fv`](@ref).
"""
function fix_combo_binaries(model, combo)
    for (ind, active) in combo
        fix_fv(_indicator_to_binary(model)[ind], active)
    end
end

"""
    unfix_combo_binaries(model::JuMP.AbstractModel, combo)::Nothing

Undo the effect of [`fix_combo_binaries`](@ref): unfix every indicator
binary in `combo` on `model`.
"""
function unfix_combo_binaries(model, combo)
    for (ind, _) in combo
        unfix_fv(_indicator_to_binary(model)[ind])
    end
end

################################################################################
#                      SET COVERING INITIALIZATION
################################################################################
#Türkay & Grossmann (1996) §2.2: pick a minimal set of combos that
#activates every indicator at least once, so the master starts with an
#OA cut for each disjunct. We enumerate nested disjunctions alongside
#top-level ones — inconsistent combos (nested active under inactive
#parent) are handled by feasibility restoration at NLP-solve time.
function _set_covering_combos(model::JuMP.AbstractModel)
    M = typeof(model)
    LVR = LogicalVariableRef{M}
    per_disj = Vector{Tuple{DisjunctionIndex, LVR}}[]
    for (idx, disj_data) in _disjunctions(model)
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
# Solve the primary NLP for a fixed combo. If infeasible, build a fresh
# V&G 1990 NLPF submodel (globals + active-disjunct originals, with
# shared slack `u` and `min u` objective), solve it for a least-
# infeasible point to generate OA cuts from, then discard it. Returns
# a named tuple with
# `combo / x_values / duals / objective / feasible / obj_x_values`.
function _solve_nlp(
    model::M, combo, method::LOA, reform_map
    ) where {M <: JuMP.AbstractModel}
    DCRef = DisjunctConstraintRef{M}
    empty_duals = Dict{DCRef, Any}()
    fix_combo_binaries(model, combo)
    JuMP.optimize!(model, ignore_optimize_hook = true)
    if JuMP.is_solved_and_feasible(model)
        x_vals, obj_xv = read_primary_solution(model)
        duals = _collect_nlp_duals(model, combo, reform_map)
        obj_val = JuMP.objective_value(model)
        unfix_combo_binaries(model, combo)
        return (combo = combo, x_values = x_vals, duals = duals,
            objective = obj_val, feasible = true, obj_x_values = obj_xv)
    end
    unfix_combo_binaries(model, combo)
    # Build a fresh per-combo NLPF: only the active-disjunct originals
    # + globals, slackened with a shared `u`. Discarded after solve.
    active_crefs = _active_disjunct_crefs(model, combo)
    feas = copy_model_with_constraints(model, active_crefs, method)
    _embed_feas_slack(feas)
    feas_fwd = feas.sub.fwd_map
    for (ind, val) in combo
        orig_bv = _indicator_to_binary(model)[ind]
        haskey(feas_fwd, orig_bv) || continue
        fix_fv(feas_fwd[orig_bv], val)
    end
    JuMP.optimize!(feas.sub.model)
    feas_ok = JuMP.is_solved_and_feasible(feas.sub.model)
    x_vals = feas_ok ? first(read_feas_solution(model, feas)) :
        Dict{JuMP.AbstractVariableRef, Any}()
    return (combo = combo, x_values = x_vals, duals = empty_duals,
        objective = Inf, feasible = false, obj_x_values = x_vals)
end

# Collect the `DisjunctConstraintRef`s of every active indicator in
# `combo`. Used to feed `copy_model_with_constraints` the minimal
# per-combo subset for NLPF construction.
function _active_disjunct_crefs(model::M, combo) where {M}
    crefs = DisjunctConstraintRef{M}[]
    for (ind, active) in combo
        any_active(active) || continue
        haskey(_indicator_to_constraints(model), ind) || continue
        for cref in _indicator_to_constraints(model)[ind]
            cref isa DisjunctConstraintRef || continue
            push!(crefs, cref)
        end
    end
    return crefs
end

# Sum the duals of BigM-reformulated constraints for each active
# disjunct's original constraint ref. Used for OA cut generation.
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

"""
    read_primary_solution(model::JuMP.AbstractModel)::Tuple{Dict, Dict}

Read the primal solution of the primary NLP after a feasible solve.
Returns `(x_values, obj_x_values)` where both dicts are keyed by
variable reference; `obj_x_values` matches `x_values` for finite
models. The InfiniteOpt extension overrides this to return the
flat-transcription dict as the second element.
"""
function read_primary_solution(model::JuMP.AbstractModel)
    x_vals = Dict{JuMP.AbstractVariableRef, Any}()
    for v in JuMP.all_variables(model)
        JuMP.is_fixed(v) && continue
        x_vals[v] = JuMP.value(v)
    end
    return x_vals, x_vals
end

"""
    read_feas_solution(
        model::JuMP.AbstractModel, feas
        )::Tuple{Dict, Dict}

Read the primal solution of the feasibility-restoration NLPF after a
feasible solve, keyed by original-model variables via `extract_solution`
on `feas.sub`. Same return contract as [`read_primary_solution`](@ref).
"""
function read_feas_solution(model::JuMP.AbstractModel, feas)
    x_vals = extract_solution(feas.sub)
    return x_vals, x_vals
end

"""
    fix_fv(bv, val::Bool)::Nothing

Fix a binary indicator reference `bv` to `val`. Dispatches on:
- `AbstractVariableRef`: calls `JuMP.fix(bv, val ? 1.0 : 0.0; force = true)`.
- `GenericAffExpr`: the complement-indicator form `1 - other_bv`; fix
  `other_bv` to the complement of `val`.

The InfiniteOpt extension adds an `AbstractArray` dispatch for
per-support indicator vectors.
"""
fix_fv(bv, val::Bool) = JuMP.fix(bv, val ? 1.0 : 0.0; force = true)
function fix_fv(bv::JuMP.GenericAffExpr, val::Bool)
    under, coeff = only(bv.terms)
    JuMP.fix(under, val ? 0.0 : 1.0; force = true)
end

"""
    unfix_fv(bv)::Nothing

Undo [`fix_fv`](@ref) on `bv`. No-op if `bv` is not currently fixed.
For complement AffExprs, unfixes the underlying variable.
"""
function unfix_fv(bv)
    JuMP.is_fixed(bv) && JuMP.unfix(bv)
    return
end
function unfix_fv(bv::JuMP.GenericAffExpr)
    under = only(keys(bv.terms))
    JuMP.is_fixed(under) && JuMP.unfix(under)
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
    model::M, master
    ) where {M <: JuMP.AbstractModel}
    combo = Dict{LogicalVariableRef{M}, Any}()
    for (_, disj_data) in _disjunctions(model)
        for ind in disj_data.constraint.indicators
            haskey(master.bin_map, ind) || continue
            combo[ind] = combo_val(master.bin_map[ind])
        end
    end
    return combo
end
"""
    combo_val(bv)::Bool

Round the master's current binary solution for indicator ref `bv` to a
`Bool`. The InfiniteOpt extension adds an `AbstractArray` dispatch
that returns a `Vector{Bool}` per support.
"""
combo_val(bv) = round(JuMP.value(bv)) > 0.5

function _add_no_good_cut(model, master, combo)
    cut_expr = JuMP.AffExpr(0.0)
    for (ind, active) in combo
        haskey(master.bin_map, ind) || continue
        add_ng_terms(cut_expr, master.bin_map[ind], active)
    end
    JuMP.@constraint(master.model, cut_expr >= 1.0)
end

"""
    add_ng_terms(cut, bv, active::Bool)::Nothing

Assemble one indicator's contribution to the no-good cut `cut_expr >=
1`: add `1 - y_j` if the indicator was active in the excluded combo,
or `y_j` otherwise. The InfiniteOpt extension adds an
`AbstractArray`/`AbstractArray{Bool}` dispatch that folds per-support
contributions.
"""
function add_ng_terms(cut, bv, active::Bool)
    if active
        JuMP.add_to_expression!(cut, -1.0, bv)
        JuMP.add_to_expression!(cut, 1.0)
    else
        JuMP.add_to_expression!(cut, 1.0, bv)
    end
    return
end

"""
    any_active(active)::Bool

Return `true` if any indicator value in `active` is truthy. The base
dispatch is the trivial scalar `Bool` case; the InfiniteOpt extension
adds an `AbstractVector{Bool}` dispatch for per-support indicators.
"""
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
# Sense-dependent coefficients for OA and convergence calls. Dispatched
# on `Val(master.obj_sense)` so downstream call sites can read like
# regular Julia without branching on the sense.
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

function _add_oa_cuts(model, master, result, method::LOA)
    isempty(result.x_values) && return
    _add_objective_oa_cut(master, result)
    _add_disjunct_oa_cuts(model, master, result, method)
    return
end

# Linearize the original objective at the NLP's solution and add the
# bounding cut `lin ≤ α` (min) or `lin ≥ α` (max) on the master.
function _add_objective_oa_cut(master, result)
    lin = _linearize_at(
        master.orig_obj, result.obj_x_values, master.obj_ref_map)
    _add_obj_cut(Val(master.obj_sense), master, lin)
    return
end
_add_obj_cut(::Val{_MOI.MIN_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin <= master.alpha_oa)
_add_obj_cut(::Val{_MOI.MAX_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin >= master.alpha_oa)

# Add V&G 1990 augmented-penalty OA cuts for each active disjunct's
# nonlinear constraints: fresh per-cut slack `σ_ik` with a penalty term
# in the master objective, and cut body `s (lin - rhs) - σ ≤ M(1 - y)`.
function _add_disjunct_oa_cuts(
    model, master, result, method::LOA
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
            rhs = _set_rhs(con.set)
            for (bv, xk, smap, d_k) in cut_info(
                master.bin_map[ind], active,
                result.x_values, master.var_map, d)
                s = sign(sgn * _dv(d_k))
                s == 0 && continue
                lin_expr = _linearize_at(con.func, xk, smap)
                slack = JuMP.@variable(master.model,
                    lower_bound = 0.0, upper_bound = method.max_slack)
                JuMP.set_objective_function(master.model,
                    JuMP.objective_function(master.model) +
                    pen * method.OA_penalty_factor * slack)
                JuMP.@constraint(master.model,
                    s * (lin_expr - rhs) - slack <=
                    method.M_value * (1 - bv))
            end
        end
    end
end

"""
    cut_info(bv, active, x_values, var_map, d)

Yield the inputs `(bv, xk, smap, d_k)` needed to emit each OA cut for
one active disjunct constraint. The scalar base returns one tuple
(one cut per constraint). The InfiniteOpt extension adds an
`AbstractArray` dispatch that yields K tuples with per-support-sliced
`x_values`, `var_map`, and dual (one cut per active support).
"""
cut_info(bv, active::Bool, x_values, var_map, d) =
    ((bv, x_values, var_map, d),)

# Collapse a dual value (scalar for a single reformulated constraint,
# vector for reform constraints that produced multiple JuMP constraints
# like Interval / EqualTo / Zeros) to a scalar sign-carrier.
_dv(d::Number) = d
_dv(d) = sum(d)

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
