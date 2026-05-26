################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
# With augmented-penalty OA master from:
# Viswanathan & Grossmann (1990), Comp. & Chem. Eng. 14(7), 769-782
#
# Infeasible primary NLPs are handled solely via the master's augmented
# penalty: the combination is forbidden by a no-good cut and no OA cut
# is emitted from that iteration. No separate feasibility-restoration
# (NLPF) subproblem is built.
################################################################################

################################################################################
#                              METHOD TYPE
################################################################################
"""
    LOA{O, P, R} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models. Uses two models: the
original (reformulated by `inner_method`, binaries fixed per iteration as an
NLP) and a master MILP copy that accumulates OA and no-good cuts.

`inner_method` defaults to `BigM(M_value)`; pass `MBM(optimizer)` to use
tighter per-constraint Ms. Other reformulations (`Hull`, `PSplit`) are not
yet supported — they emit constraints whose shapes the LOA dual-collection
logic can't slice.
"""
struct LOA{O, P, R} <: AbstractReformulationMethod
    nlp_optimizer::O
    mip_optimizer::P
    inner_method::R
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
        OA_penalty_factor::Float64 = 1000.0,
        inner_method::R = BigM(M_value)
        ) where {O, P, R <: AbstractReformulationMethod}
        new{O, P, R}(nlp_optimizer, mip_optimizer, inner_method, max_iter,
            atol, rtol, M_value, max_slack, OA_penalty_factor)
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
    reformulate_model(model, method.inner_method)

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
        add_oa_cuts(model, master, result, method)
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
        @info "LOA iter $iter: LB=$master_bound  UB=$best_objective  " *
              "gap=$(_gap(sense_token, best_objective, master_bound))"
        _loa_converged(best_objective, master_bound, sense_token, method) && break
        combination = _extract_combination(model, master)
        result = _solve_nlp(model, combination, method, reformulation_map)
        avoid_combination(master.model, combination, master.binary_map)
        add_oa_cuts(model, master, result, method)
        if result.feasible && is_better(result.objective)
            best_objective = result.objective
            best_result = result
        end
    end

    if best_result !== nothing
        commit_combination(model, best_result.combination,
            best_result.linearization_point)
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
# primal point, duals, and objective. If infeasible, return an empty
# result — the master's augmented penalty (slacks `σ_ik` on disjunct OA
# cuts) absorbs infeasibility and a no-good cut forbids the combination.
function _solve_nlp(
    model::M, combination, method::LOA, reformulation_map
    ) where {M <: JuMP.AbstractModel}
    return with_fixed_combination(model, combination) do
        JuMP.optimize!(model, ignore_optimize_hook = true)
        if JuMP.is_solved_and_feasible(model)
            lin_point = extract_solution(model)
            duals = _collect_nlp_duals(
                model, combination, reformulation_map)
            objective_val = JuMP.objective_value(model)
            return (combination = combination,
                linearization_point = lin_point, duals = duals,
                objective = objective_val, feasible = true)
        end
        return (combination = combination,
            linearization_point = Dict{JuMP.AbstractVariableRef, Any}(),
            duals = Dict{DisjunctConstraintRef{M}, Any}(),
            objective = Inf, feasible = false)
    end
end

# Fix `combination`, run `f()`, unfix. Logical binaries are relaxed
# from ZeroOne for the duration so a pure NLP solver (Ipopt) can
# handle the inner solve; restored on exit. Extensions override
# `_fix_combination` to handle per-support fixing without needing
# their own try/finally lifecycle.
function with_fixed_combination(
    f,
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    relaxed = relax_logical_vars(model)
    undo = _fix_combination(model, combination)
    try
        return f()
    finally
        undo()
        unrelax_logical_vars(relaxed)
    end
end

# Finalize the model with the LOA-optimal combination: relax logical
# binaries (stripping ZeroOne so a pure NLP solver can handle the
# post-hook `JuMP.optimize!`), fix indicators at the committed values,
# and warm-start from the linearization point. After this, the model
# is no longer in a state suitable for re-running with a different
# `gdp_method`.
function commit_combination(
    model::JuMP.AbstractModel,
    combination::AbstractDict,
    linearization_point::AbstractDict
    )
    relax_logical_vars(model)
    _fix_combination(model, combination)
    for (variable, values) in linearization_point
        _set_warm_start!(variable, values)
    end
    return
end

# OVERRIDABLE. Apply the combination's fixes and return a closure that
# undoes them. Base loops `fix_indicator`/`unfix_indicator`. The
# InfiniteOpt extension overrides this to handle per-support
# `Vector{Bool}` values via point-equality constraints, keeping all
# cleanup state captured in the returned closure.
function _fix_combination(
    model::JuMP.AbstractModel, combination::AbstractDict
    )
    for (indicator, value) in combination
        fix_indicator(model, indicator, value)
    end
    return function ()
        for (indicator, _) in combination
            unfix_indicator(model, indicator)
        end
    end
end

# OVERRIDABLE. Write the LOA linearization point into a variable's
# warm start. `values` is always per-support shape (length-1 vector
# for finite vars); base unwraps. The InfiniteOpt extension overrides
# for `GeneralVariableRef` to broadcast across transcribed supports.
_set_warm_start!(variable, values::AbstractVector) =
    JuMP.set_start_value(variable, only(values))

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
function add_oa_cuts(
    model::JuMP.AbstractModel,
    master::NamedTuple,
    result::NamedTuple,
    method::LOA
    )
    isempty(result.linearization_point) && return
    linearization = _linearize_at(master.original_objective,
        result.linearization_point, master.objective_ref_map)
    _add_objective_cut(Val(master.objective_sense), master, linearization)
    _add_global_oa_cuts(model, master, result, method)
    add_disjunct_oa_cuts(model, master, result, method)
    return
end

# Bounding cut `lin ≤ α` (min) or `lin ≥ α` (max) on the master.
_add_objective_cut(::Val{_MOI.MIN_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin <= master.alpha_oa)
_add_objective_cut(::Val{_MOI.MAX_SENSE}, master, lin) =
    JuMP.@constraint(master.model, lin >= master.alpha_oa)

# Add the OA cut `g(x^l) + ∇g(x^l)^T (x − x^l) in con.set` for every
# nonlinear global constraint of `model` — the third cut class in
# Türkay & Grossmann (1996, eq. 12) alongside the objective and the
# disjunct cuts. Walks `JuMP.list_of_constraint_types`, skipping
# variable bounds, linear functions (already in the master), and
# BigM-reformulated forms (in `_reformulation_constraints`).
# `LessThan` and `GreaterThan` are supported; equalities and vector
# constraints are passed through to `JuMP.@constraint` as-is —
# valid for affine-after-linearization sets.
#
# Finite-only. InfiniteOpt's `add_oa_cuts(::InfiniteModel, ...)`
# override uses `_add_global_oa_cuts_infinite`, which routes through
# transcription to handle per-support / aggregate-ref globals.
function _add_global_oa_cuts(
    model::JuMP.AbstractModel,
    master::NamedTuple,
    result::NamedTuple,
    method::LOA
    )
    _, penalty_sign = _disjunct_cut_coefficients(
        Val(master.objective_sense))
    variable_type = JuMP.variable_ref_type(typeof(model))
    reform_set = is_gdp_model(model) ?
        Set(_reformulation_constraints(model)) : Set()
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        _is_linear_F(F) && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            con = JuMP.constraint_object(cref)
            con isa JuMP.ScalarConstraint || continue
            linearization = _linearize_at(con.func,
                result.linearization_point, master.variable_map)
            _add_global_oa_row!(master, linearization, con.set,
                method, penalty_sign)
        end
    end
    return
end

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

# Fresh nonnegative slack added to the master objective with the V&G
# 1990 penalty. Shared by the disjunct and global OA cuts so both get
# the same augmented-penalty treatment: a nonconvex linearization can
# be an invalid relaxation, and the penalized slack keeps the master
# feasible instead of letting accumulated cuts make it infeasible.
function _penalized_slack(
    master::NamedTuple, method::LOA, penalty_sign::Int
    )
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
            penalty_sign * method.OA_penalty_factor * slack)
    return slack
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
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        sign_value * (linearization - rhs) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# Slacked global OA row(s) (V&G 1990). The nonconvex linearization may
# be an invalid relaxation, so each row carries a penalized slack
# rather than being a hard constraint — without this the accumulated
# global cuts make the master infeasible on nonconvex models.
# `EqualTo` / `Interval` get a two-sided pair sharing one slack.
# Unknown set types fall back to the prior hard cut.
function _add_global_oa_row!(
    master::NamedTuple, lin, set::_MOI.LessThan,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - _MOI.constant(set) <= slack)
    return
end
function _add_global_oa_row!(
    master::NamedTuple, lin, set::_MOI.GreaterThan,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, _MOI.constant(set) - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::NamedTuple, lin, set::_MOI.EqualTo,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    c = _MOI.constant(set)
    JuMP.@constraint(master.model, lin - c <= slack)
    JuMP.@constraint(master.model, c - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::NamedTuple, lin, set::_MOI.Interval,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - set.upper <= slack)
    JuMP.@constraint(master.model, set.lower - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::NamedTuple, lin, set, ::LOA, ::Int
    )
    JuMP.@constraint(master.model, lin in set)
    return
end

# OVERRIDABLE. Yield `(binary_ref, lin_point, var_map, dual)` per OA
# cut to emit. Scalar binary returns one tuple; InfiniteOpt overrides
# for per-support `Vector` binaries to yield multiple sliced tuples.
# `GenericAffExpr` covers complement-form indicators (`1 - y`) stored
# in `binary_map` when a `Logical` is declared with `logical_complement`.
cut_info(
    binary_ref::JuMP.AbstractVariableRef,
    active::Bool,
    linearization_point::AbstractDict,
    variable_map::AbstractDict,
    dual
    ) = ((binary_ref, linearization_point, variable_map, dual),)
cut_info(
    binary_ref::JuMP.GenericAffExpr,
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

