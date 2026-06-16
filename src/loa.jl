################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
# With augmented-penalty OA master from:
# Viswanathan & Grossmann (1990), Comp. & Chem. Eng. 14(7), 769-782
#
# Infeasible primary NLPs fall through to NLPF (V&G 1990 eq. 8): a
# slacked feasibility version of the same problem whose primal becomes
# the linearization site for OA cuts. A no-good cut still forbids that
# combination on the master. NLPF is bypassed for per-support
# Vector{Bool} combinations (InfiniteOpt multi-resolution master) —
# those fall back to no-good-only.
################################################################################

################################################################################
#                              METHOD TYPE
################################################################################
"""
    LOA{O, P, R, T} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models. Iterates between a
primary NLP (the original model reformulated by `inner_method` with
indicator binaries fixed per iteration) and a master MILP that
accumulates outer-approximation and no-good cuts.

`inner_method` defaults to `BigM(M_value)`. `MBM(optimizer)` is also
supported. Other reformulations (`Hull`, `PSplit`) are not yet supported.

## Fields
- `nlp_optimizer::O`: Solver used for the primary NLP.
- `mip_optimizer::P`: Solver used for the master MILP (defaults to
  `nlp_optimizer`).
- `inner_method::R`: Reformulation applied to the primary NLP — `BigM`
  or `MBM`.
- `max_iter::Int`: Maximum LOA iterations after set-covering seeding.
- `atol::T`, `rtol::T`: Absolute / relative gap tolerances for
  convergence.
- `M_value::T`: Big-M used in the disjunct OA cut gating term.
- `max_slack::T`: Upper bound for each per-cut slack variable.
- `oa_penalty::T`: V&G 1990 penalty coefficient applied to slacks in
  the master objective.
"""
struct LOA{O, P, R, T} <: AbstractReformulationMethod
    nlp_optimizer::O
    mip_optimizer::P
    inner_method::R
    max_iter::Int
    atol::T
    rtol::T
    M_value::T
    max_slack::T
    oa_penalty::T
    verbose::Bool
    supports_schedule::Union{Nothing, Vector{Int}}
    coarse_builder::Union{Nothing, Function}
    function LOA(
        nlp_optimizer::O;
        mip_optimizer::P = nlp_optimizer,
        max_iter::Int = 10,
        atol::T = 1e-6,
        rtol::T = 1e-4,
        M_value::T = 1e9,
        max_slack::T = 1e3,
        oa_penalty::T = 1e3,
        inner_method::R = BigM(M_value),
        verbose::Bool = false,
        supports_schedule::Union{Nothing, Vector{Int}} = nothing,
        coarse_builder::Union{Nothing, Function} = nothing
        ) where {O, P, R <: AbstractReformulationMethod, T}
        R <: Union{BigM, MBM} || error(
            "LOA inner_method must be BigM or MBM (got $R). " *
            "Hull and PSplit are not yet supported.")
        supports_schedule === nothing || coarse_builder !== nothing || error(
            "LOA `supports_schedule` requires a `coarse_builder = N -> " *
            "InfiniteModel` that constructs a fresh model at each " *
            "warmup resolution.")
        new{O, P, R, T}(nlp_optimizer, mip_optimizer, inner_method,
            max_iter, atol, rtol, M_value, max_slack, oa_penalty, verbose,
            supports_schedule, coarse_builder)
    end
end

################################################################################
#                              LOA MASTER
################################################################################
# TODO: Move `_LOAMaster` to `src/datatypes.jl` alongside `GDPSubmodel`
# once the LOA API stabilizes. Keeping it co-located with the algorithm
# while the field set is still in flux.
#
# Bundles the LOA master MILP and the maps the algorithm needs to
# translate original-model refs into master-model refs.
# `objective_ref_map` is split from `variable_map` because the
# InfiniteOpt aggregate-objective path uses a transcribed-
# `JuMP.VariableRef`-keyed map for the objective only, while constraint
# linearization uses the InfiniteOpt-keyed `variable_map`.
mutable struct _LOAMaster{M <: JuMP.AbstractModel, OF, AO, BM, VM, RM}
    model::M
    binary_map::BM
    variable_map::VM
    objective_sense::_MOI.OptimizationSense
    original_objective::OF
    alpha_oa::AO
    objective_ref_map::RM
    # All penalized slacks ever appended to the master, in emission
    # order — disjunct, global, and objective cuts. Iterated by
    # `_check_slacks` after each master solve to diagnose whether
    # `max_slack` is binding.
    slacks::Vector{Any}
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
    method.supports_schedule === nothing ||
        error("`supports_schedule` is only meaningful for InfiniteOpt " *
            "models; load InfiniteOpt to enable the multi-resolution path.")
    _reformulate_loa_single_level(model, method)
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
    return
end

# Single-resolution LOA. Always uses set-covering for the
# initialization seeds; the InfiniteOpt multi-resolution wrapper
# carries trajectory information across levels via primal warm starts
# on transcribed JuMP variables (set before this is called), not by
# overriding the seed-combination loop. Returns `best_result`
# (NamedTuple with `combination` / `linearization_point` / etc.) or
# `nothing` if every NLP was infeasible.
function _reformulate_loa_single_level(
    model::JuMP.AbstractModel, method::LOA
    )
    _clear_reformulations(model)
    combinations = _set_covering_combinations(model)
    reformulate_model(model, method.inner_method)

    master = build_loa_master(model, method)
    reformulation_map = _build_reformulation_map(model)
    JuMP.set_optimizer(model, method.nlp_optimizer)
    JuMP.set_silent(model)
    sense_token = Val(JuMP.objective_sense(model))
    best_objective = _worst_objective(sense_token)
    best_result = nothing

    # Initialization Procedure (Türkay & Grossmann 1996, sec. 2.2):
    # solve K set-covering NLPs with cycling indicator combinations to
    # seed the master with at least one OA cut per disjunct.
    # `previous_result` carries the most recent FEASIBLE NLP primal
    # forward as the warm start for the next NLP (T&G §2.3); NLPF
    # solutions are skipped since their primal is slack-distorted.
    previous_result = nothing
    for (i, combination) in enumerate(combinations)
        _set_nlp_warm_start!(previous_result)
        t_seed = @elapsed result = _solve_nlp(
            model, combination, method, reformulation_map)
        avoid_combination(master.model, combination, master.binary_map)
        add_oa_cuts(model, master, result, method)
        _check_penalty(result, method)
        if result.feasible && _is_better(sense_token, result.objective,
                best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
        method.verbose && _log_seed(i, t_seed, result, best_objective)
    end

    # The master objective is a rigorous bound only for a convex (here,
    # linear) inner problem. With nonlinear constraints the bound may be
    # invalid, so terminate on V&G (1990) criterion 5 — stop once a
    # feasible NLP no longer improves on the previous feasible NLP —
    # rather than on the untrustworthy master gap.
    rigorous = !_has_nonlinear_constraints(model)
    warned = false
    produced_bound = false
    prev_nlp_objective = previous_result === nothing ? nothing :
        previous_result.objective
    for iter in 1:method.max_iter
        t_master = @elapsed JuMP.optimize!(master.model)
        feasible = JuMP.is_solved_and_feasible(master.model)
        method.verbose && _log_master(iter, master.model, t_master)
        if !feasible
            # An infeasible/unbounded master before any bound is a
            # degenerate exit, not convergence: LOA falls back to the
            # best set-covering seed without ever iterating. Common
            # when most NLP subproblems are infeasible, so the master
            # lacks the OA cuts to bound `alpha_oa`.
            produced_bound || @warn "LOA: master returned status " *
                "$(JuMP.termination_status(master.model)) before any " *
                "bound was produced; exiting with the best seed NLP " *
                "incumbent ($best_objective). This is NOT a converged " *
                "or certified optimum — often a sign the NLP " *
                "subproblems are mostly infeasible."
            break
        end
        _check_slacks(master, method)
        master_bound = JuMP.objective_value(master.model)
        produced_bound = true
        method.verbose &&
            _log_iter(iter, master_bound, best_objective, sense_token)
        if rigorous
            _loa_converged(
                best_objective, master_bound, sense_token, method) &&
                break
        elseif !warned && _bound_crossed(
                best_objective, master_bound, sense_token, method)
            @warn "LOA: master bound ($master_bound) crossed the " *
                "incumbent ($best_objective). The model appears " *
                "nonconvex, so the OA cuts are not valid supports and " *
                "the dual bound is NOT rigorous. Returning the best " *
                "NLP incumbent as a heuristic (Viswanathan & " *
                "Grossmann 1990)."
            warned = true
        end
        combination = _extract_combination(model, master)
        _set_nlp_warm_start!(previous_result)
        t_nlp = @elapsed result = _solve_nlp(
            model, combination, method, reformulation_map)
        avoid_combination(master.model, combination, master.binary_map)
        add_oa_cuts(model, master, result, method)
        _check_penalty(result, method)
        if result.feasible && _is_better(sense_token, result.objective,
                best_objective)
            best_objective = result.objective
            best_result = result
        end
        method.verbose && _log_nlp(iter, t_nlp, result, best_objective)
        # V&G (1990) criterion 5: without a rigorous bound, stop once a
        # feasible NLP fails to improve on the previous feasible NLP.
        if !rigorous && result.feasible
            prev_nlp_objective !== nothing && !_is_better(sense_token,
                result.objective, prev_nlp_objective) && break
            prev_nlp_objective = result.objective
        end
        result.feasible && (previous_result = result)
    end

    if best_result !== nothing
        commit_combination(model, best_result.combination,
            best_result.linearization_point)
    end
    return best_result
end

# `verbose = true` trace helpers. Print to stdout so a `tee`-d log
# captures them; structured key=value so they're easy to grep.
function _log_seed(i::Int, t::Real, result::NamedTuple, best::Real)
    println("LOA_SEED ", i, ": time=", round(t, digits = 3),
        "s feasible=", result.feasible, " obj=", result.objective,
        " best=", best)
    flush(stdout)
end
function _log_master(iter::Int, master::JuMP.AbstractModel, t::Real)
    println("LOA_MASTER iter=", iter,
        " status=", JuMP.termination_status(master),
        " feasible=", JuMP.is_solved_and_feasible(master),
        " time=", round(t, digits = 3), "s")
    flush(stdout)
end
function _log_iter(iter::Int, lb::Real, ub::Real, sense_token::Val)
    println("LOA_ITER ", iter, ": LB=", lb, " UB=", ub,
        " gap=", _gap(sense_token, ub, lb))
    flush(stdout)
end
function _log_nlp(iter::Int, t::Real, result::NamedTuple, best::Real)
    println("LOA_NLP iter=", iter,
        " time=", round(t, digits = 3), "s feasible=", result.feasible,
        " obj=", result.objective, " best=", best)
    flush(stdout)
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

# A genuine bound crossing: the master "bound" is worse than the
# incumbent by more than the convergence tolerance (gap strongly
# negative). In valid convex OA, LB ≤ UB always, so this can only
# arise from invalid OA cuts on a nonconvex problem — V&G (1990) report
# the master objective exceeding the integer NLP optimum on every
# nonconvex test problem. The tolerance band separates a real crossing
# from floating-point / solver noise.
function _bound_crossed(
    best_objective::Real,
    master_bound::Real,
    sense_token::Val,
    method::LOA
    )
    (isinf(best_objective) || isinf(master_bound)) && return false
    gap = _gap(sense_token, best_objective, master_bound)
    return gap < -(method.atol + method.rtol * abs(best_objective))
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
# (nested active under inactive parent) are handled by the no-good
# cut emitted from the infeasible NLP solve.
#
# `K = max(disjunction sizes)` combinations suffice: combination `k`
# activates the `k`-th indicator of each disjunction, cycling via
# `mod1` for disjunctions shorter than `K`. Every indicator gets
# activated at least once over k = 1..K.
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

# True if any constraint carries a non-affine function. LOA's master
# objective is a rigorous lower bound only when every constraint is
# linear: the inner problem is then a (linear, hence convex) GDP and
# the OA cuts are exact supporting hyperplanes. With a nonlinear
# constraint the problem may be nonconvex, the OA linearizations need
# not support the feasible set, and the master bound is not rigorous —
# so LOA terminates on V&G (1990) subproblem-improvement instead.
function _has_nonlinear_constraints(model::JuMP.AbstractModel)
    variable_type = JuMP.variable_ref_type(typeof(model))
    for (F, _) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        _is_linear_F(F) || return true
    end
    return false
end

# OVERRIDABLE. Build the LOA master MILP `M^b_{LA}` (Türkay & Grossmann
# 1996, eq. 12): copy decision variables and only the linear
# constraints, install `alpha_oa` as the objective auxiliary. Nonlinear
# objective and disjunct constraints enter as OA cuts after each NLP
# solve. Returns an `_LOAMaster`.
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

    return _LOAMaster(master, binary_map, variable_map, objective_sense,
        original_objective, alpha_oa, variable_map, Any[])
end

################################################################################
#                          NLP SUBPROBLEM
################################################################################
# Solve the primary NLP for a fixed combination. If feasible, read the
# primal point, duals, and objective. If infeasible, fall through to
# the NLPF (V&G 1990 eq. 8) approximation: a slacked version of the
# same problem that always solves, whose primal becomes the
# linearization site for OA cuts. The master still learns shape
# information from the failed combination instead of only adding a
# no-good cut.
function _solve_nlp(
    model::M, combination, method::LOA, reformulation_map
    ) where {M <: JuMP.AbstractModel}
    primary = with_fixed_combination(model, combination) do
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
        return nothing
    end
    primary === nothing || return primary

    # Primary NLP infeasible — try NLPF.
    nlpf = _solve_nlpf(model, combination, method)
    nlpf === nothing || return nlpf

    return (combination = combination,
        linearization_point = Dict{JuMP.AbstractVariableRef, Any}(),
        duals = Dict{DisjunctConstraintRef{M}, Any}(),
        objective = Inf, feasible = false)
end

################################################################################
#                       NLPF (FEASIBILITY SUBPROBLEM)
################################################################################
# V&G 1990 eq. 8: when the primary NLP is infeasible at a fixed
# combination, copy the model, slack every scalar inequality with a
# single nonnegative slack `u`, minimize `u`, and return the resulting
# primal point as an approximate linearization site. The point
# satisfies equalities and variable bounds exactly; inequalities can
# be violated by at most `u`. The "duals" returned are sign-only — the
# OA cut emitter uses `sign(dual)` to pick a direction, and for a
# slacked feasibility solve we know each active constraint's
# linearization is informative in the standard direction.
function _solve_nlpf(
    model::M, combination, method::LOA
    ) where {M <: JuMP.AbstractModel}
    copy, ref_map = JuMP.copy_model(model)
    JuMP.set_optimizer(copy, method.nlp_optimizer)
    JuMP.set_silent(copy)

    var_type = JuMP.variable_ref_type(typeof(copy))
    u = JuMP.@variable(copy, lower_bound = 0.0, base_name = "_nlpf_u")

    for (F, S) in JuMP.list_of_constraint_types(copy)
        F === var_type && continue
        _nlpf_should_slack(S) || continue
        for cref in JuMP.all_constraints(copy, F, S)
            con = JuMP.constraint_object(cref)
            JuMP.delete(copy, cref)
            new_func = _nlpf_slacked_func(con.func, u, con.set)
            JuMP.@constraint(copy, new_func in con.set)
        end
    end

    JuMP.@objective(copy, Min, u)

    # Translate each original-model indicator binary to its
    # counterpart on the copy, then pin it to the combination value.
    # `_nlpf_fix_on_copy` dispatches on value type: scalar `Bool`
    # paths use `JuMP.fix`; per-support `AbstractVector{Bool}` paths
    # require an `InfiniteModel`-side override (per-support point-
    # equality) — see `ext/InfiniteDisjunctiveProgramming.jl`.
    for (indicator, value) in combination
        binary = _binary_on_copy(
            _indicator_to_binary(model)[indicator], ref_map)
        _nlpf_fix_on_copy(copy, binary, value)
    end

    try
        JuMP.optimize!(copy, ignore_optimize_hook = true)
    catch
        return nothing
    end
    JuMP.has_values(copy) || return nothing

    linearization_point = _nlpf_extract_primal(model, ref_map)
    duals = _nlpf_sign_duals(model, combination)
    return (combination = combination,
        linearization_point = linearization_point,
        duals = duals,
        objective = Inf, feasible = false)
end

# Translate a binary reference from the original model to its
# counterpart on the copy. Direct refs go through `ref_map`;
# complement-form `1 - y_orig` rebuilds as `1 - ref_map[y_orig]`.
_binary_on_copy(binary::JuMP.AbstractVariableRef, ref_map) =
    ref_map[binary]
function _binary_on_copy(
    binary::JuMP.GenericAffExpr, ref_map
    )
    underlying = only(keys(binary.terms))
    return 1.0 - ref_map[underlying]
end

# Pin a copy-side binary to a combination value. Plain `JuMP.fix`
# (no `force = true`) — `force` triggers `delete_upper_bound` which
# fails on InfiniteOpt copies whose ZeroOne constraint refs didn't
# survive `JuMP.copy_model`. The fix pins the value to 0/1 which
# satisfies the ZeroOne anyway. Solvers that can't handle binaries
# (Ipopt) will error out — the caller catches and returns `nothing`
# so the iteration falls back to no-good-only behavior.
function _nlpf_fix_on_copy(
    copy, binary::JuMP.AbstractVariableRef, value::Bool
    )
    JuMP.is_fixed(binary) && JuMP.unfix(binary)
    JuMP.fix(binary, value ? 1.0 : 0.0)
    return
end
# Complement form `1 - y_underlying`: indicator=value means
# underlying=!value. Recursion routes both scalar `Bool` and per-
# support `AbstractVector{Bool}` to the underlying-binary dispatch.
function _nlpf_fix_on_copy(
    copy, binary::JuMP.GenericAffExpr, value
    )
    underlying = only(keys(binary.terms))
    _nlpf_fix_on_copy(copy, underlying, _flip(value))
    return
end

_flip(v::Bool) = !v
_flip(v::AbstractVector{Bool}) = .!v

_nlpf_should_slack(::Type{<:_MOI.LessThan}) = true
_nlpf_should_slack(::Type{<:_MOI.GreaterThan}) = true
_nlpf_should_slack(::Type) = false

_nlpf_slacked_func(func, u, ::_MOI.LessThan) = func - u
_nlpf_slacked_func(func, u, ::_MOI.GreaterThan) = func + u

# Read primal values from `copy` keyed by the original model's
# variables, in the same per-support shape `extract_solution` would
# have produced on the original.
function _nlpf_extract_primal(
    model::JuMP.AbstractModel, ref_map
    )
    V = JuMP.variable_ref_type(typeof(model))
    T = JuMP.value_type(typeof(model))
    result = Dict{V, Vector{T}}()
    for v in collect_all_vars(model)
        JuMP.is_fixed(v) && continue
        target = try
            ref_map[v]
        catch
            continue  # var not in the copy's ref_map (e.g., parameter only)
        end
        val = JuMP.value(target)
        result[v] = val isa AbstractArray ? vec(val) : [val]
    end
    return result
end

# Sign-only "duals" on active disjunct constraints — the OA cut
# emitter only needs `sign(dual)` to pick the linearization direction,
# and for a slacked-feasibility solve each active constraint's
# linearization is meaningful in its standard direction.
function _nlpf_sign_duals(
    model::M, combination::AbstractDict
    ) where {M <: JuMP.AbstractModel}
    duals = Dict{DisjunctConstraintRef{M}, Any}()
    for (indicator, active) in combination
        any_active(active) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef || continue
            con = _disjunct_constraints(model)[
                JuMP.index(cref)].constraint
            duals[cref] = _nlpf_dual_sign(con.set)
        end
    end
    return duals
end

_nlpf_dual_sign(::_MOI.LessThan) = 1.0
_nlpf_dual_sign(::_MOI.GreaterThan) = -1.0
_nlpf_dual_sign(::_MOI.EqualTo) = [1.0, 1.0]
_nlpf_dual_sign(::_MOI.Interval) = [1.0, 1.0]
_nlpf_dual_sign(s::_MOI.Nonpositives) = ones(_MOI.dimension(s))
_nlpf_dual_sign(s::_MOI.Nonnegatives) = -ones(_MOI.dimension(s))
_nlpf_dual_sign(s::_MOI.Zeros) = ones(_MOI.dimension(s))
_nlpf_dual_sign(::Any) = 0.0

# Fix `combination`, run `f()`, unfix. Logical binaries are relaxed
# from ZeroOne for the duration so a pure NLP solver (Ipopt) can
# handle the inner solve; restored on exit. Extensions override
# `fix_combination` to handle per-support fixing without needing
# their own try/finally lifecycle.
function with_fixed_combination(
    f,
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    relaxed = relax_logical_vars(model)
    undo = fix_combination(model, combination)
    try
        return f()
    finally
        undo()
        unrelax_logical_vars(relaxed)
    end
end

# Iter-to-iter NLP warm start (T&G 1996 §2.3): seed the next primary
# NLP solve from the most recent FEASIBLE solution's primal point.
# No-op on the first seed iteration when `previous` is `nothing`, and
# no-op on iterations following an NLPF fall-through (caller does not
# update `previous_result` then), since NLPF's primal is slack-
# distorted and a worse start than the prior real NLP. Routes through
# the `set_warm_start!` dispatch so the InfiniteOpt extension's
# per-support broadcast handles `GeneralVariableRef` correctly.
function _set_nlp_warm_start!(previous)
    previous === nothing && return
    for (variable, values) in previous.linearization_point
        set_warm_start!(variable, values)
    end
    return
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
    fix_combination(model, combination)
    for (variable, values) in linearization_point
        set_warm_start!(variable, values)
    end
    return
end

# OVERRIDABLE. Apply the combination's fixes and return a closure that
# undoes them. Base loops `fix_indicator`/`unfix_indicator`. The
# InfiniteOpt extension overrides this to handle per-support
# `Vector{Bool}` values via point-equality constraints, keeping all
# cleanup state captured in the returned closure.
function fix_combination(
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
set_warm_start!(variable, values::AbstractVector) =
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
#                              DIAGNOSTICS
################################################################################
# V&G 1990 requires ρ > ‖λ*‖_∞ for the penalty term to dominate any
# admissible Lagrange multiplier — otherwise a slack can absorb a real
# violation and the OA cut becomes an invalid relaxation. Check after
# every feasible primary NLP and warn if the bound is at risk. NLPF
# returns sign-only duals (`±1`) so this is a no-op when `!feasible`.
function _check_penalty(result::NamedTuple, method::LOA)
    result.feasible || return
    isempty(result.duals) && return
    max_dual = 0.0
    for (_, d) in result.duals
        max_dual = max(max_dual, _max_abs_dual(d))
    end
    if max_dual >= method.oa_penalty
        @warn "LOA: oa_penalty = $(method.oa_penalty) ≤ max |dual| " *
            "= $max_dual. V&G 1990 requires ρ > ‖λ‖_∞; slacks may " *
            "absorb real OA-cut violations. Raise `oa_penalty`."
    end
    return
end
_max_abs_dual(d::Number) = abs(d)
_max_abs_dual(d::AbstractArray) = isempty(d) ? 0.0 : maximum(abs, d)
_max_abs_dual(::Nothing) = 0.0

# `max_slack` is a numerical guard (V&G 1990 leaves slacks unbounded;
# in practice an unbounded slack lets the master trivially feasible-
# itself out of the OA constraints). When a slack is at the cap, the
# master is effectively solving a stricter problem than V&G's master.
# Diagnose so the user can raise `max_slack` if the cap binds often.
function _check_slacks(master::_LOAMaster, method::LOA)
    JuMP.has_values(master.model) || return
    isempty(master.slacks) && return
    cap = method.max_slack
    threshold = 0.99 * cap
    max_value = 0.0
    binding = 0
    for slack in master.slacks
        val = JuMP.value(slack)
        max_value = max(max_value, val)
        val >= threshold && (binding += 1)
    end
    if binding > 0
        @warn "LOA: $binding slack(s) at max_slack = $cap (max " *
            "value $max_value). Master is solving a stricter problem " *
            "than V&G's; raise `max_slack` if this persists."
    end
    return
end

################################################################################
#                       COMBO EXTRACTION (master → NLP)
################################################################################
# Read each indicator's binary value from the master MILP solution.
# `combination_val` dispatches per binary type (scalar in base;
# extensions add per-support).
function _extract_combination(
    model::M,
    master::_LOAMaster
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
    master::_LOAMaster,
    result::NamedTuple,
    method::LOA
    )
    isempty(result.linearization_point) && return
    linearization = _linearize_at(master.original_objective,
        result.linearization_point, master.objective_ref_map)
    _add_objective_cut(
        Val(master.objective_sense), master, linearization, method)
    _add_global_oa_cuts(model, master, result, method)
    add_disjunct_oa_cuts(model, master, result, method)
    return
end

# Slacked objective cut (V&G 1990 eq. 6). For MIN, `lin ≤ α + σ` with
# `σ ≥ 0` penalized in the master objective so the slack drives to
# zero at convergence; the linearization is then `α ≥ lin` (standard
# OA bound). MAX is symmetric: `lin ≥ α − σ`. Without the slack, an
# accumulated bad linearization on a nonconvex objective can make the
# master infeasible — the disjunct and global cuts already carry
# slacks but the objective cut did not.
function _add_objective_cut(
    sense_token::Val, master::_LOAMaster, lin, method::LOA
    )
    _, penalty_sign = _disjunct_cut_coefficients(sense_token)
    slack = _penalized_slack(master, method, penalty_sign)
    _add_objective_cut_body(sense_token, master, lin, slack)
end
_add_objective_cut_body(::Val{_MOI.MIN_SENSE}, master, lin, slack) =
    JuMP.@constraint(master.model, lin <= master.alpha_oa + slack)
_add_objective_cut_body(::Val{_MOI.MAX_SENSE}, master, lin, slack) =
    JuMP.@constraint(master.model, lin >= master.alpha_oa - slack)

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
    master::_LOAMaster,
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
    master::_LOAMaster,
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
                    master.binary_map[indicator], active, constraint,
                    result.linearization_point, master.variable_map, dual)
                _add_oa_cut_for_constraint(
                    constraint, master, binary_ref, lin_point,
                    var_map, dual_value, method, sign_factor,
                    penalty_sign, result.feasible)
            end
        end
    end
end

# Fresh nonnegative slack added to the master objective with the V&G
# 1990 penalty. Shared by the disjunct, global, and objective OA cuts
# so all three carry the same augmented-penalty treatment: a nonconvex
# linearization can be an invalid relaxation, and the penalized slack
# keeps the master feasible instead of letting accumulated cuts make
# it infeasible. The slack is recorded on `master.slacks` so the
# `_check_slacks` diagnostic can see whether `max_slack` is binding.
function _penalized_slack(
    master::_LOAMaster, method::LOA, penalty_sign::Int
    )
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    push!(master.slacks, slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
            penalty_sign * method.oa_penalty * slack)
    return slack
end

# Linearize constraint at `linearization_point`, then dispatch on the
# constraint's set to emit slacked OA cut(s) gated by `M(1 − binary)`.
# Linear constraints are exact via BigM and skipped. `feasible` flags
# whether the linearization point came from a real NLP (true) or from
# NLPF (false); equality cuts behave differently in those two regimes.
function _add_oa_cut_for_constraint(
    constraint::JuMP.AbstractConstraint,
    master::_LOAMaster,
    binary_ref,
    linearization_point::AbstractDict,
    var_map::AbstractDict,
    dual_value,
    method::LOA,
    sign_factor::Int,
    penalty_sign::Int,
    feasible::Bool
    )
    _is_linear_F(typeof(constraint.func)) && return
    linearization = _linearize_at(
        constraint.func, linearization_point, var_map)
    _emit_disjunct_oa_cut(constraint.set, master, binary_ref,
        linearization, dual_value, method, sign_factor,
        penalty_sign, feasible)
    return
end

# Inequality sets — one signed cut per Türkay-Grossmann 1996 / Duran-
# Grossmann 1986 OA convention. Primary NLP gives the real Lagrange
# multiplier (whose sign picks the active side); NLPF returns ±1 from
# `_nlpf_dual_sign(set)`.
function _emit_disjunct_oa_cut(
    set::Union{_MOI.LessThan, _MOI.GreaterThan},
    master::_LOAMaster, binary_ref, linearization, dual_value,
    method::LOA, sign_factor::Int, penalty_sign::Int, ::Bool
    )
    sign_value = sign(sign_factor * _collapse_dual(dual_value))
    sign_value == 0 && return
    rhs = _set_rhs(set)
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        sign_value * (linearization - rhs) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# Equality — feasible NLP: the summed BigM duals give the OA/ER
# multiplier sign (Duran-Grossmann 1986), so one signed cut suffices.
# Infeasible (NLPF): sign is uninformative, so emit both directions
# sharing one slack — mirrors the global-OA equality treatment in
# `_add_global_oa_row!(::EqualTo)`.
function _emit_disjunct_oa_cut(
    set::_MOI.EqualTo,
    master::_LOAMaster, binary_ref, linearization, dual_value,
    method::LOA, sign_factor::Int, penalty_sign::Int, feasible::Bool
    )
    c = _MOI.constant(set)
    slack = _penalized_slack(master, method, penalty_sign)
    if feasible
        sign_value = sign(sign_factor * _collapse_dual(dual_value))
        sign_value == 0 && return
        JuMP.@constraint(master.model,
            sign_value * (linearization - c) - slack <=
                method.M_value * (1 - binary_ref))
        return
    end
    JuMP.@constraint(master.model,
        (linearization - c) - slack <=
            method.M_value * (1 - binary_ref))
    JuMP.@constraint(master.model,
        (c - linearization) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# Interval — always two-sided regardless of dual: there is no single
# rhs to sign against (`_set_rhs(::Interval) = 0` is wrong here), and
# any active boundary needs its linearization to gate the right side.
# Both rows share one slack so the V&G penalty is not double-counted.
function _emit_disjunct_oa_cut(
    set::_MOI.Interval,
    master::_LOAMaster, binary_ref, linearization, ::Any,
    method::LOA, ::Int, penalty_sign::Int, ::Bool
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        (linearization - set.upper) - slack <=
            method.M_value * (1 - binary_ref))
    JuMP.@constraint(master.model,
        (set.lower - linearization) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# Fallback for vector sets and any future set types — preserve the
# original single-signed behavior with `_collapse_dual` for the sign.
function _emit_disjunct_oa_cut(
    set, master::_LOAMaster, binary_ref, linearization, dual_value,
    method::LOA, sign_factor::Int, penalty_sign::Int, ::Bool
    )
    sign_value = sign(sign_factor * _collapse_dual(dual_value))
    sign_value == 0 && return
    rhs = _set_rhs(set)
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
    master::_LOAMaster, lin, set::_MOI.LessThan,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - _MOI.constant(set) <= slack)
    return
end
function _add_global_oa_row!(
    master::_LOAMaster, lin, set::_MOI.GreaterThan,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, _MOI.constant(set) - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::_LOAMaster, lin, set::_MOI.EqualTo,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    c = _MOI.constant(set)
    JuMP.@constraint(master.model, lin - c <= slack)
    JuMP.@constraint(master.model, c - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::_LOAMaster, lin, set::_MOI.Interval,
    method::LOA, penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - set.upper <= slack)
    JuMP.@constraint(master.model, set.lower - lin <= slack)
    return
end
function _add_global_oa_row!(
    master::_LOAMaster, lin, set, ::LOA, ::Int
    )
    JuMP.@constraint(master.model, lin in set)
    return
end

# OVERRIDABLE. Yield `(binary_ref, lin_point, var_map, dual)` per OA
# cut to emit for `constraint`. Scalar binary returns one tuple; the
# InfiniteOpt extension overrides to fan out across supports of any
# infinite variable found in `binary_ref` OR `constraint.func`. The
# constraint must factor in because a finite indicator on an
# InfiniteModel can still gate a constraint that depends on an
# infinite var — one cut per support is needed, even though the
# indicator is scalar.
#
# `GenericAffExpr` covers complement-form indicators (`1 - y`) stored
# in `binary_map` when a `Logical` is declared with `logical_complement`.
cut_info(
    binary_ref::JuMP.AbstractVariableRef,
    active::Bool,
    constraint::JuMP.AbstractConstraint,
    linearization_point::AbstractDict,
    variable_map::AbstractDict,
    dual
    ) = ((binary_ref, linearization_point, variable_map, dual),)
cut_info(
    binary_ref::JuMP.GenericAffExpr,
    active::Bool,
    constraint::JuMP.AbstractConstraint,
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

################################################################################
#                    LINEARIZATION & EXPRESSION CONVERSION
################################################################################
# TODO: Move this section out of `loa.jl` (likely back to
# `src/utilities.jl`) when a second OA-style method lands and these
# helpers earn their generality. Currently only LOA uses them, so
# they live next to the algorithm that consumes them.

# First-order Taylor for a single var or affine expression mapped into
# master space.
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

# Convert JuMP expression trees to Julia Expr with
# MOI.VariableIndex leaves for MOI.Nonlinear evaluation.
function _to_nlp_expr(expr::JuMP.GenericNonlinearExpr, idx::Dict)
    args = Any[_to_nlp_expr(a, idx) for a in expr.args]
    return Expr(:call, expr.head, args...)
end
function _to_nlp_expr(expr::JuMP.GenericAffExpr, idx::Dict)
    parts = Any[expr.constant]
    for (var, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef, _MOI.VariableIndex(idx[var])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end
function _to_nlp_expr(expr::JuMP.GenericQuadExpr, idx::Dict)
    parts = Any[_to_nlp_expr(expr.aff, idx)]
    for (pair, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef,
            _MOI.VariableIndex(idx[pair.a]),
            _MOI.VariableIndex(idx[pair.b])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end
function _to_nlp_expr(var::JuMP.AbstractVariableRef, idx::Dict)
    return _MOI.VariableIndex(idx[var])
end
_to_nlp_expr(x::Number, ::Dict) = x

# First-order Taylor linearization of a quadratic or nonlinear
# expression at point xk via MOI.Nonlinear reverse-mode AD.
function _linearize_at(
    func::Union{JuMP.GenericQuadExpr, JuMP.GenericNonlinearExpr},
    xk::Dict, ref_map
    )
    vars = JuMP.AbstractVariableRef[]
    _interrogate_variables(v -> push!(vars, v), func)
    unique!(vars)
    isempty(vars) && return JuMP.AffExpr(JuMP.value(v -> 0.0, func))

    n = length(vars)
    T = JuMP.value_type(typeof(JuMP.owner_model(vars[1])))
    idx = Dict(vars[i] => i for i in 1:n)
    nlp = _MOI.Nonlinear.Model()
    _MOI.Nonlinear.set_objective(nlp, _to_nlp_expr(func, idx))
    ord = [_MOI.VariableIndex(i) for i in 1:n]
    evaluator = _MOI.Nonlinear.Evaluator(
        nlp, _MOI.Nonlinear.SparseReverseMode(), ord)
    _MOI.initialize(evaluator, [:Grad])

    xk_vec = [_unwrap_scalar(get(xk, v, zero(T))) for v in vars]
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

# Extract RHS from an MOI set.
_set_rhs(s::Union{_MOI.LessThan, _MOI.GreaterThan, _MOI.EqualTo}) =
    _MOI.constant(s)
_set_rhs(::Any) = 0.0

# Unwrap a 1-element per-support `Vector` to its scalar value;
# scalars pass through. `extract_solution` returns per-support
# `Vector`s uniformly (length-1 for finite, length-K for InfiniteOpt).
# AD pipelines and `set_start_value` need a scalar in the finite
# case; per-support consumers slice out a scalar themselves.
_unwrap_scalar(v::Real) = v
_unwrap_scalar(v::AbstractVector) = only(v)
