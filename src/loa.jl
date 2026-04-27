################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
# With augmented-penalty OA master from:
# Viswanathan & Grossmann (1990), Comp. & Chem. Eng. 14(7), 769-782
################################################################################

################################################################################
#                              METHOD TYPE
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
        max_iter::Int = 10,
        atol::Float64 = 1e-6,
        rtol::Float64 = 1e-4,
        M_value::Float64 = 1e9,
        max_slack::Float64 = 1000.0,
        OA_penalty_factor::Float64 = 1000.0
        ) where {O, P}
        new{O, P}(nlp_optimizer, mip_optimizer, max_iter, atol, rtol,
            M_value, max_slack, OA_penalty_factor)
    end
end

################################################################################
#                            SENSE PRIMITIVES
################################################################################
# Val(MIN/MAX)-dispatched primitives so the algorithm reads sense-agnostic.
_disjunct_cut_coefficients(::Val{_MOI.MIN_SENSE}) = (-1, 1)
_disjunct_cut_coefficients(::Val{_MOI.MAX_SENSE}) = (1, -1)
_worst_objective(::Val{_MOI.MIN_SENSE}) = Inf
_worst_objective(::Val{_MOI.MAX_SENSE}) = -Inf
_is_better(::Val{_MOI.MIN_SENSE}, new, best) = new < best
_is_better(::Val{_MOI.MAX_SENSE}, new, best) = new > best
_gap(::Val{_MOI.MIN_SENSE}, best, bound) = best - bound
_gap(::Val{_MOI.MAX_SENSE}, best, bound) = bound - best
_flip_sense(::Val{_MOI.MIN_SENSE}) = Val(_MOI.MAX_SENSE)
_flip_sense(::Val{_MOI.MAX_SENSE}) = Val(_MOI.MIN_SENSE)

################################################################################
#                            MAIN ALGORITHM
################################################################################
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    combinations = _set_covering_combinations(model)
    reformulate_model(model, BigM(method.M_value))

    master = build_loa_master(model, method)
    reformulation_map = _build_reformulation_map(model)
    JuMP.set_optimizer(model, method.nlp_optimizer)
    JuMP.set_silent(model)
    # Cached for repeated use in `_worst_objective`, `_is_better`,
    # `_flip_sense`, and `_loa_converged`.
    sense_token = Val(JuMP.objective_sense(model))
    best_objective = _worst_objective(sense_token)
    is_better(candidate) = _is_better(sense_token, candidate, best_objective)
    best_result = nothing

    # Initialization Procedure (Türkay & Grossmann 1996, sec. 2.2): solve
    # the set-covering NLPs to seed the master with at least one OA cut
    # per disjunct before the main iteration.
    for combination in combinations
        result = _solve_nlp(model, combination, method, reformulation_map)
        avoid_combination(master.model, combination, master.binary_map)
        _add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_objective = result.objective
            best_result = result
        end
    end

    master_bound = _worst_objective(_flip_sense(sense_token))
    for iter in 1:method.max_iter
        JuMP.optimize!(master.model)
        JuMP.is_solved_and_feasible(master.model) || break
        master_bound = JuMP.objective_value(master.model)
        _loa_converged(best_objective, master_bound, sense_token, method) && break
        combination = _extract_combination(model, master)
        result = _solve_nlp(model, combination, method, reformulation_map)
        avoid_combination(master.model, combination, master.binary_map)
        _add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_objective = result.objective
            best_result = result
        end
    end

    if best_result !== nothing
        fix_combination_binaries(model, best_result.combination)
        set_start_values(model, best_result.linearization_point)
    end
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
    return
end

# Convergence: absolute gap ≤ atol or relative gap ≤ rtol. Distinct
# from CP's `separation_obj ≤ tolerance` check (different convergence
# shapes — gap vs. single-threshold), so not shared.
function _loa_converged(
    best_objective::Real,
    master_bound::Real,
    sense_token::Val,
    method::LOA
    )
    (isinf(best_objective) || isinf(master_bound)) && return false
    gap = _gap(sense_token, best_objective, master_bound)
    gap <= method.atol && return true
    abs(best_objective) > 1e-10 &&
        gap / abs(best_objective) <= method.rtol && return true
    return false
end

# Error fallback for unsupported model types.
function reformulate_model(::M, ::LOA) where {M}
    error("reformulate_model not implemented for model type `$(M)` with LOA.")
end

################################################################################
#                       SET-COVERING INITIALIZATION
################################################################################
# Türkay & Grossmann (1996), sec. 2.2: produce a minimal set of
# combinations that activates every indicator at least once, so the
# master starts with an OA cut per disjunct. Nested disjunctions are
# enumerated alongside top-level ones — inconsistent combinations
# (nested active under inactive parent) are handled by feasibility
# restoration at NLP-solve time.
#
# `K = max(disjunction sizes)` combinations suffice: combination `k`
# activates the `k`-th indicator of each disjunction, cycling via
# `mod1` for disjunctions shorter than `K`. Every indicator gets
# activated at least once over k=1..K.
#
# Note: keeps the less-specific `model::JuMP.AbstractModel`
# signature instead of `model::M where {M <: JuMP.AbstractModel}` to
# avoid ambiguity with InfiniteOpt overrides; this function is
# internal so no public-API impact.
function _set_covering_combinations(model::JuMP.AbstractModel)
    LogicalRef = LogicalVariableRef{typeof(model)}
    indicator_lists = [collect(d.constraint.indicators)
        for (_, d) in _disjunctions(model)]
    isempty(indicator_lists) && return Dict{LogicalRef, Bool}[]
    K = maximum(length, indicator_lists)
    return [
        Dict{LogicalRef, Bool}(
            indicator => (indicator == indicators[mod1(k, length(indicators))])
            for indicators in indicator_lists
            for indicator in indicators)
        for k in 1:K
    ]
end

################################################################################
#                         MASTER CONSTRUCTION
################################################################################
# True for linear constraint function types (scalar/vector variable refs
# and affine expressions). LOA master per Türkay & Grossmann (1996) is
# pure MILP; nonlinear `f`, `g`, `h_{ij}` enter as OA cuts after each
# NLP solve.
_is_linear_F(::Type{<:JuMP.AbstractVariableRef}) = true
_is_linear_F(::Type{<:JuMP.GenericAffExpr}) = true
_is_linear_F(::Type{<:AbstractVector{<:JuMP.AbstractVariableRef}}) = true
_is_linear_F(::Type{<:AbstractVector{<:JuMP.GenericAffExpr}}) = true
_is_linear_F(::Type) = false

# OVERRIDABLE. Build the LOA master MILP `M^b_{LA}` (Türkay & Grossmann
# 1996, eq. 12): copy decision variables and only the linear
# constraints, install `alpha_oa` as the objective auxiliary. Nonlinear
# objective and disjunct constraints enter as OA cuts after each NLP
# solve. Returns a NamedTuple of (model, binary_map, variable_map,
# objective_sense, original_objective, alpha_oa, objective_ref_map).
function build_loa_master(model::JuMP.AbstractModel, method::LOA)
    original_objective = JuMP.objective_function(model)
    objective_sense = JuMP.objective_sense(model)
    variable_type = JuMP.variable_ref_type(typeof(model))

    master = _copy_model(model)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)

    variable_map = copy_variables_onto_model(master, model)

    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        _is_linear_F(F) || continue
        for constraint_ref in JuMP.all_constraints(model, F, S)
            constraint = JuMP.constraint_object(constraint_ref)
            new_func = replace_variables_in_constraint(
                constraint.func, variable_map)
            JuMP.@constraint(master, new_func in constraint.set)
        end
    end

    binary_map = Dict{LogicalVariableRef, Any}()
    for (indicator, binary_ref) in _indicator_to_binary(model)
        binary_map[indicator] = _remap_indicator_to_binary(
            binary_ref, variable_map)
    end

    alpha_oa = JuMP.@variable(master, base_name = "alpha_oa")
    JuMP.@objective(master, objective_sense, alpha_oa)

    return (model = master, binary_map = binary_map,
        variable_map = variable_map, objective_sense = objective_sense,
        original_objective = original_objective, alpha_oa = alpha_oa,
        objective_ref_map = variable_map)
end

################################################################################
#                          NLP SUBPROBLEM
################################################################################
# Solve the primary NLP for a fixed combination. If feasible, read the
# primal point, duals, and objective and return. If infeasible, build a
# fresh V&G 1990 NLPF submodel (active-disjunct originals + globals,
# slackened with shared `u` and `min u` objective), solve it for a
# least-infeasible point, and return that. The NLPF is discarded.
function _solve_nlp(
    model::M, combination, method::LOA, reformulation_map
    ) where {M <: JuMP.AbstractModel}
    fix_combination_binaries(model, combination)
    JuMP.optimize!(model, ignore_optimize_hook = true)
    if JuMP.is_solved_and_feasible(model)
        lin_point = extract_solution(model)
        duals = _collect_nlp_duals(model, combination, reformulation_map)
        objective_val = JuMP.objective_value(model)
        unfix_combination_binaries(model, combination)
        return (combination = combination,
            linearization_point = lin_point, duals = duals,
            objective = objective_val, feasible = true)
    end
    unfix_combination_binaries(model, combination)
    active_refs = _active_disjunct_constraint_refs(model, combination)
    feas = copy_model_with_constraints(model, active_refs, method)
    _embed_feas_slack(feas)
    for (indicator, value) in combination
        orig_binary_ref = _indicator_to_binary(model)[indicator]
        fix_indicator(_remap_indicator_to_binary(
            orig_binary_ref, feas.sub.fwd_map), value)
    end
    JuMP.optimize!(feas.sub.model)
    lin_point = JuMP.is_solved_and_feasible(feas.sub.model) ?
        extract_solution(feas.sub) :
        Dict{JuMP.AbstractVariableRef, Any}()
    return (combination = combination,
        linearization_point = lin_point,
        duals = Dict{DisjunctConstraintRef{M}, Any}(),
        objective = Inf, feasible = false)
end

# `DisjunctConstraintRef`s of every active indicator in `combination`.
function _active_disjunct_constraint_refs(
    model::M,
    combination::AbstractDict
    ) where {M <: JuMP.AbstractModel}
    refs = DisjunctConstraintRef{M}[]
    for (indicator, active) in combination
        any_active(active) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef && push!(refs, cref)
        end
    end
    return refs
end

# OVERRIDABLE. Fix every indicator in `combination` on `model`.
function fix_combination_binaries(
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    for (indicator, active) in combination
        fix_indicator(model, indicator, active)
    end
end

# OVERRIDABLE. Inverse of `fix_combination_binaries`.
function unfix_combination_binaries(
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    for (indicator, _) in combination
        unfix_indicator(model, indicator)
    end
end

# OVERRIDABLE. Build a raw V&G 1990 NLPF submodel: copy decision
# variables, copy global (non-reformulation) constraints and the active
# disjuncts' original pre-BigM constraints. The shared slack `u` and
# `min u` objective are layered on by `_embed_feas_slack`. Returns
# `(sub::GDPSubmodel, objective_ref_map)`.
function copy_model_with_constraints(
    model::JuMP.AbstractModel,
    disjunct_constraint_refs::Vector{<:DisjunctConstraintRef},
    method::LOA
    )
    sub_model = _copy_model(model)
    fwd_map = copy_variables_onto_model(sub_model, model)
    variable_type = JuMP.variable_ref_type(typeof(model))
    reform_set = is_gdp_model(model) ?
        Set(_reformulation_constraints(model)) : Set()
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            con = JuMP.constraint_object(cref)
            expr = replace_variables_in_constraint(con.func, fwd_map)
            JuMP.@constraint(sub_model, expr in con.set)
        end
    end
    for cref in disjunct_constraint_refs
        con = JuMP.constraint_object(cref)
        expr = replace_variables_in_constraint(con.func, fwd_map)
        JuMP.@constraint(sub_model, expr in con.set)
    end
    JuMP.set_optimizer(sub_model, method.nlp_optimizer)
    JuMP.set_silent(sub_model)
    return (
        sub = GDPSubmodel(sub_model, collect_all_vars(model), fwd_map),
        objective_ref_map = fwd_map
        )
end

# Convert the raw copy into a V&G 1990 NLPF problem: shared slack
# embedded in every constraint via `_slacken`, `min slack` objective.
function _embed_feas_slack(feas::NamedTuple)
    model = feas.sub.model
    slack = JuMP.@variable(model, base_name = "_feas_slack", lower_bound = 0.0)
    variable_type = JuMP.variable_ref_type(typeof(model))
    to_slacken = Any[]
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        append!(to_slacken, JuMP.all_constraints(model, F, S))
    end
    for constraint_ref in to_slacken
        constraint = JuMP.constraint_object(constraint_ref)
        for (slack_func, slack_set) in _slacken(
            constraint.func, constraint.set, slack)
            JuMP.@constraint(model, slack_func in slack_set)
        end
        JuMP.delete(model, constraint_ref)
    end
    JuMP.@objective(model, Min, slack)
    return
end

# Apply slack `u` to a constraint based on its set type. ≤: `f − u`;
# ≥: `f + u`; ==: split into ≤ and ≥; otherwise passthrough.
_slacken(f, set::_MOI.LessThan, u) = [(f - u, set)]
_slacken(f, set::_MOI.GreaterThan, u) = [(f + u, set)]
function _slacken(f, set::_MOI.EqualTo, u)
    b = _MOI.constant(set)
    return [(f - u, _MOI.LessThan(b)), (f + u, _MOI.GreaterThan(b))]
end
_slacken(f, set, u) = [(f, set)]

################################################################################
#                       BIGM DUAL COLLECTION
################################################################################
# Build a map `DisjunctConstraintRef → Vector{<reformulated cref>}` so
# `_collect_nlp_duals` can sum BigM duals per original disjunct
# constraint. Uses `_num_reform_constraints` to slice the flat
# reformulation-constraint list per original. (Brittle: assumes BigM
# emits constraints in disjunction-iteration order; a cleaner
# alternative would be to have BigM record this map directly.)
function _build_reformulation_map(model::M) where {M <: JuMP.AbstractModel}
    ref_cons = _reformulation_constraints(model)
    isempty(ref_cons) && return Dict{DisjunctConstraintRef{M}, Vector{Any}}()
    CRT = eltype(ref_cons)
    rmap = Dict{DisjunctConstraintRef{M}, Vector{CRT}}()
    cursor = 1
    for (_, disjunction_data) in _disjunctions(model)
        for indicator in disjunction_data.constraint.indicators
            haskey(_indicator_to_constraints(model), indicator) || continue
            for constraint_ref in _indicator_to_constraints(model)[indicator]
                constraint_ref isa DisjunctConstraintRef || continue
                constraint = _disjunct_constraints(model)[
                    JuMP.index(constraint_ref)].constraint
                num_reform = _num_reform_constraints(constraint)
                cursor + num_reform - 1 <= length(ref_cons) || break
                rmap[constraint_ref] = collect(
                    ref_cons[cursor:cursor + num_reform - 1])
                cursor += num_reform
            end
        end
    end
    return rmap
end

# Number of BigM-reformulated JuMP constraints per original constraint.
_num_reform_constraints(::JuMP.ScalarConstraint{T, S}) where {
    T <: Union{Number, JuMP.AbstractJuMPScalar},
    S <: Union{_MOI.EqualTo, _MOI.Interval}} = 2
_num_reform_constraints(::JuMP.ScalarConstraint) = 1
_num_reform_constraints(constraint::JuMP.VectorConstraint{T, S}) where {
    T <: Union{Number, JuMP.AbstractJuMPScalar},
    S <: _MOI.Zeros} = 2 * _MOI.dimension(constraint.set)
_num_reform_constraints(constraint::JuMP.VectorConstraint) =
    _MOI.dimension(constraint.set)

# Sum the duals of all reformulation constraints for one disjunct cref.
function _sum_duals(
    reformulation_map::AbstractDict,
    constraint_ref::DisjunctConstraintRef
    )
    haskey(reformulation_map, constraint_ref) || return 0.0
    rcs = reformulation_map[constraint_ref]
    isempty(rcs) && return 0.0
    total = JuMP.dual(rcs[1])
    for i in 2:length(rcs)
        total = total .+ JuMP.dual(rcs[i])
    end
    return total
end

# Sum BigM-reformulation duals per active disjunct constraint ref.
function _collect_nlp_duals(
    model::M,
    combination::AbstractDict,
    reformulation_map::AbstractDict
    ) where {M <: JuMP.AbstractModel}
    duals = Dict{DisjunctConstraintRef{M}, Any}()
    JuMP.has_duals(model) || return duals
    for (indicator, active) in combination
        any_active(active) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef || continue
            duals[cref] = _sum_duals(reformulation_map, cref)
        end
    end
    return duals
end

################################################################################
#                       COMBO EXTRACTION (master → NLP)
################################################################################
# Read each indicator's binary value from the master MILP solution.
# `combination_val` dispatches per binary type (scalar in base;
# extensions add per-support).
function _extract_combination(
    model::M,
    master::NamedTuple
    ) where {M <: JuMP.AbstractModel}
    combination = Dict{LogicalVariableRef{M}, Any}()
    for (_, disjunction_data) in _disjunctions(model)
        for indicator in disjunction_data.constraint.indicators
            haskey(master.binary_map, indicator) || continue
            combination[indicator] = combination_val(master.binary_map[indicator])
        end
    end
    return combination
end

# OVERRIDABLE. One-shot float→Bool conversion; downstream dispatches on Bool.
combination_val(binary_ref) = round(Bool, JuMP.value(binary_ref))

################################################################################
#                          OA CUT EMISSION
################################################################################
function _add_oa_cuts(
    model::JuMP.AbstractModel,
    master::NamedTuple,
    result::NamedTuple,
    method::LOA
    )
    isempty(result.linearization_point) && return
    linearization = _linearize_at(master.original_objective,
        result.linearization_point, master.objective_ref_map)
    _add_objective_cut(Val(master.objective_sense), master, linearization)
    add_disjunct_oa_cuts(model, master, result, method)
    return
end

# Bounding cut `lin ≤ α` (min) or `lin ≥ α` (max) on the master.
_add_objective_cut(::Val{_MOI.MIN_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin <= master.alpha_oa)
_add_objective_cut(::Val{_MOI.MAX_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin >= master.alpha_oa)

# OVERRIDABLE. Add V&G 1990 augmented-penalty OA cuts for each active
# disjunct's nonlinear constraints: fresh per-cut slack `σ_ik` with a
# penalty term in the master objective, and cut body
# `s (lin − rhs) − σ ≤ M(1 − y)`.
function add_disjunct_oa_cuts(
    model::JuMP.AbstractModel,
    master::NamedTuple,
    result::NamedTuple,
    method::LOA
    )
    sign_factor, penalty_sign = _disjunct_cut_coefficients(
        Val(master.objective_sense))
    for (indicator, active) in result.combination
        any_active(active) || continue
        haskey(master.binary_map, indicator) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef || continue
            constraint = _disjunct_constraints(model)[
                JuMP.index(cref)].constraint
            dual = get(result.duals, cref, nothing)
            dual === nothing && continue
            for (binary_ref, lin_point, var_map, dual_value) in cut_info(
                    master.binary_map[indicator], active,
                    result.linearization_point, master.variable_map, dual)
                _add_oa_cut_for_constraint(
                    constraint, master, binary_ref, lin_point,
                    var_map, dual_value, method, sign_factor, penalty_sign)
            end
        end
    end
end

# Linearize constraint at `linearization_point`, append a fresh per-cut
# slack with V&G penalty, gate by `M(1 − binary)`. Linear constraints
# are exact via BigM and skipped.
function _add_oa_cut_for_constraint(
    constraint::JuMP.AbstractConstraint,
    master::NamedTuple,
    binary_ref,
    linearization_point::AbstractDict,
    var_map::AbstractDict,
    dual_value,
    method::LOA,
    sign_factor::Int,
    penalty_sign::Int
    )
    _is_linear_F(typeof(constraint.func)) && return
    sign_value = sign(sign_factor * _collapse_dual(dual_value))
    sign_value == 0 && return
    rhs = _set_rhs(constraint.set)
    linearization = _linearize_at(constraint.func, linearization_point, var_map)
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
            penalty_sign * method.OA_penalty_factor * slack)
    JuMP.@constraint(master.model,
        sign_value * (linearization - rhs) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# OVERRIDABLE. Yield `(binary_ref, lin_point, var_map, dual)` per OA
# cut to emit. Scalar binary returns one tuple; InfiniteOpt overrides
# for per-support `Vector` binaries to yield multiple sliced tuples.
cut_info(
    binary_ref::JuMP.AbstractVariableRef,
    active::Bool,
    linearization_point::AbstractDict,
    variable_map::AbstractDict,
    dual
    ) = ((binary_ref, linearization_point, variable_map, dual),)

# OVERRIDABLE. Truthiness of an active descriptor. InfiniteOpt
# overrides for per-support `Vector{Bool}`.
any_active(active::Bool) = active

# Collapse a per-constraint dual (scalar or vector for Interval /
# EqualTo / Zeros) to a scalar sign-carrier.
_collapse_dual(dual::Number) = dual
_collapse_dual(dual) = sum(dual)

# First-order Taylor for a single var or affine expression mapped into
# master space. (Quad/nonlinear `_linearize_at` lives in `utilities.jl`
# and uses MOI Nonlinear AD.)
function _linearize_at(
    variable::JuMP.AbstractVariableRef,
    ::AbstractDict,
    ref_map::AbstractDict
    )
    target = ref_map[variable]
    return JuMP.GenericAffExpr{Float64, typeof(target)}(0.0, target => 1.0)
end
function _linearize_at(
    func::JuMP.GenericAffExpr,
    ::AbstractDict,
    ref_map::AbstractDict
    )
    V = valtype(ref_map)
    T = JuMP.value_type(V) <: Number ? JuMP.value_type(V) : Float64
    result = JuMP.GenericAffExpr{T, V}(T(func.constant))
    for (variable, coefficient) in func.terms
        JuMP.add_to_expression!(result, coefficient, ref_map[variable])
    end
    return result
end

################################################################################
#                            FINALIZATION
################################################################################
# OVERRIDABLE. Set JuMP start values from a scalar-valued
# linearization-point dict. Warm-starts the post-hook `optimize!`
# that JuMP fires after `reformulate_model` — without this the final
# NLP solve restarts from wherever the last LOA iteration left the
# model. InfiniteOpt overrides for per-support Vector values.
function set_start_values(
    ::JuMP.AbstractModel, linearization_point::AbstractDict
    )
    for (variable, value) in linearization_point
        JuMP.set_start_value(variable, _unwrap_scalar(value))
    end
end
