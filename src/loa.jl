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
# combination on the master. NLPF is bypassed for vector-valued
# `Vector{Bool}` combinations (which an extension may supply) — those
# fall back to no-good-only.
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
- `M_value::T`: Big-M used in the disjunct OA cut gating term.
- `max_slack::T`: Upper bound for each per-cut slack variable.
- `oa_penalty::T`: V&G 1990 penalty coefficient applied to slacks in
  the master objective.
- `convergence_tol::Float64`: Relative gap tolerance for the early
  stop. The main loop exits once the master bound meets the incumbent
  within this relative gap, provided the slacks also pass `slack_tol`.
- `slack_tol::Float64`: Largest total OA-cut slack (summed slack
  variables) for which the master bound still counts as a valid
  convergence certificate. Positive slack means the master is
  violating its own cuts (the nonconvex case), so the bound is not
  trustworthy and the loop keeps running to `max_iter` rather than
  stopping on a spurious crossing.
- `iteration_time_limit::Float64`: Wall-clock budget (seconds) for the
  LOA iteration loop (seeds + main loop). `Inf` (default) is no limit;
  each subproblem solve is capped to the budget left so one solve can't
  overrun it.
- `time_limit::Float64`: Overall wall-clock cap (seconds) covering the
  iteration loop AND the final committed solve. `Inf` (default) is no
  cap; when set, the final solve gets only the budget left under it.
"""
struct LOA{O, P, R, T} <: AbstractReformulationMethod
    nlp_optimizer::O
    mip_optimizer::P
    inner_method::R
    max_iter::Int
    M_value::T
    max_slack::T
    oa_penalty::T
    convergence_tol::Float64
    slack_tol::Float64
    iteration_time_limit::Float64
    time_limit::Float64
    function LOA(
        nlp_optimizer::O;
        mip_optimizer::P = nlp_optimizer,
        max_iter::Int = 10,
        M_value::T = 1e9,
        max_slack::T = 1e3,
        oa_penalty::T = 1e3,
        inner_method::R = BigM(M_value),
        convergence_tol::Float64 = 1e-6,
        slack_tol::Float64 = 1e-4,
        iteration_time_limit::Float64 = Inf,
        time_limit::Float64 = Inf
        ) where {O, P, R <: AbstractReformulationMethod, T}
        R <: Union{BigM, MBM} || error(
            "LOA inner_method must be BigM or MBM (got $R). " *
            "Hull and PSplit are not yet supported.")
        new{O, P, R, T}(nlp_optimizer, mip_optimizer, inner_method,
            max_iter, M_value, max_slack, oa_penalty,
            convergence_tol, slack_tol,
            iteration_time_limit, time_limit)
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
# `objective_ref_map` is split from `variable_map` so an extension can
# map the objective's linearization variables differently from the
# constraint ones; in base the two maps are identical.
mutable struct _LOAMaster{M <: JuMP.AbstractModel, OF, AO, BM, VM, RM}
    model::M
    binary_map::BM
    variable_map::VM
    objective_sense::_MOI.OptimizationSense
    original_objective::OF
    alpha_oa::AO
    objective_ref_map::RM
end

################################################################################
#                            SENSE PRIMITIVES
################################################################################
# Val(MIN/MAX)-dispatched primitives so the algorithm reads sense-agnostic.
_penalty_sign(::Val{_MOI.MIN_SENSE}) = 1
_penalty_sign(::Val{_MOI.MAX_SENSE}) = -1
_worst_objective(::Val{_MOI.MIN_SENSE}) = Inf
_worst_objective(::Val{_MOI.MAX_SENSE}) = -Inf
_is_better(::Val{_MOI.MIN_SENSE}, new, best) = new < best
_is_better(::Val{_MOI.MAX_SENSE}, new, best) = new > best
_gap(::Val{_MOI.MIN_SENSE}, best, bound) = best - bound
_gap(::Val{_MOI.MAX_SENSE}, best, bound) = bound - best

################################################################################
#                            MAIN ALGORITHM
################################################################################
# LOA reformulation entry point: set-covering seeds, the master/NLP
# iteration loop, commit the best combination, and mark the model ready.
# This driver is model-agnostic — extensions override the inner solve
# steps (master build, NLP, cuts), not this loop.
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    combinations = _set_covering_combinations(model)
    reformulate_model(model, method.inner_method)

    master = build_loa_master(model, method)
    # Relax the original model's logical binaries once, permanently. The
    # original model is only ever solved as the NLP (never as a MILP —
    # the master is a separate copy), so its ZeroOne is never used.
    # Stripping it here lets a pure NLP solver handle every solve and
    # makes per-iteration fixing overwrite in place (no relax/unrelax,
    # no fix/undo). Must follow build_loa_master so the master keeps its
    # ZeroOne binaries.
    relax_logical_vars(model)
    JuMP.set_optimizer(model, method.nlp_optimizer)
    JuMP.set_silent(model)
    t_start = time()
    overall_deadline = t_start + method.time_limit
    loop_deadline =
        min(t_start + method.iteration_time_limit, overall_deadline)
    original_time_limit = JuMP.time_limit_sec(model)
    sense_token = Val(JuMP.objective_sense(model))
    best_objective = _worst_objective(sense_token)
    best_result = nothing
    master_bound = nothing

    # Initialization Procedure (Türkay & Grossmann 1996, sec. 2.2):
    # solve K set-covering NLPs with cycling indicator combinations to
    # seed the master with at least one OA cut per disjunct.
    # `previous_result` carries the most recent FEASIBLE NLP primal
    # forward as the warm start for the next NLP (T&G §2.3); NLPF
    # solutions are skipped since their primal is slack-distorted.
    previous_result = nothing
    for combination in combinations
        time() < loop_deadline || break
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(model, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.binary_map)
        add_oa_cuts(model, master, result, method)
        if result.feasible &&
                _is_better(sense_token, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
    end

    # Master/NLP loop. Each iteration solves the master MILP for a lower
    # bound (`master_bound` is the `alpha_oa` auxiliary, NOT the penalized
    # objective the slacks inflate), then solves the NLP at the master's
    # combination and updates the incumbent. The loop exits on whichever
    # comes first: convergence (the bound meets the incumbent within
    # `convergence_tol` while total slack is below `slack_tol`), an
    # infeasible master (every combination forbidden by a no-good cut),
    # `max_iter`, or the time budget. The slack gate is what makes the
    # convergence stop safe without a convexity flag: on a convex inner
    # problem the slacks collapse to zero and the crossing is a real
    # optimality proof; on a nonconvex one the master pays slack to
    # violate its own cuts, the gate stays shut, and the loop runs on to
    # `max_iter` instead of stopping at a spurious crossing.
    converged = false
    for _ in 1:method.max_iter
        time() < loop_deadline || break
        _cap_remaining_time(master.model, loop_deadline)
        JuMP.optimize!(master.model)
        JuMP.is_solved_and_feasible(master.model) || break
        master_bound = JuMP.value(master.alpha_oa)
        if best_result !== nothing
            gap = _gap(sense_token, best_objective, master_bound)
            total_slack = abs(JuMP.objective_value(master.model) -
                master_bound) / method.oa_penalty
            tol = method.convergence_tol * max(abs(best_objective), 1.0)
            if gap <= tol && total_slack <= method.slack_tol
                converged = true
                break
            end
        end
        combination = _extract_combination(model, master)
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(model, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.binary_map)
        add_oa_cuts(model, master, result, method)
        if result.feasible &&
                _is_better(sense_token, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
    end

    if best_result !== nothing
        commit_combination(model, best_result.combination,
            best_result.linearization_point)
    end
    if isfinite(method.time_limit)
        # Overall cap: the final committed solve gets the budget left.
        JuMP.set_time_limit_sec(model, max(0.0, overall_deadline - time()))
    elseif isfinite(method.iteration_time_limit)
        # Loop-only budget: restore so the final solve isn't crippled.
        _restore_time_limit(model, original_time_limit)
    end
    _report_loa_gap(best_objective, best_result, master_bound, sense_token,
        converged)
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
    return
end

# Report the final gap once the loop ends. `[converged]` = the loop hit
# its convergence test (the master bound met the incumbent within
# `convergence_tol` with total slack below `slack_tol`); `[limit hit]` =
# it stopped on `max_iter`, the time budget, or an exhausted master with
# a gap still open; `[no bound]` = the master never produced a bound
# (fell back to the best seed). `master_bound` is `alpha_oa`, a rigorous
# dual bound only for a convex (linear) inner problem; on a nonconvex
# model it can cross the incumbent, which the slack gate is meant to
# catch and the reported gap surfaces.
function _report_loa_gap(
    best_objective::Real,
    best_result,
    master_bound,
    sense_token::Val,
    converged::Bool
    )
    if best_result === nothing
        @warn "LOA finished: no feasible incumbent found."
        return
    end
    if master_bound === nothing
        @info "LOA finished [no bound]: incumbent $best_objective " *
            "(master produced no bound; best seed kept)."
        return
    end
    gap = _gap(sense_token, best_objective, master_bound)
    relative = abs(best_objective) > 1e-10 ?
        gap / abs(best_objective) : gap
    label = converged ? "converged" : "limit hit"
    @info "LOA finished [$label]: incumbent $best_objective, master " *
        "bound $master_bound, gap $gap (relative $relative)."
    return
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
        original_objective, alpha_oa, variable_map)
end

################################################################################
#                          NLP SUBPROBLEM
################################################################################
# Cap `target`'s solver to the wall-clock budget left before `deadline`
# (seconds, `time()` clock). No-op when `deadline` is `Inf` (no LOA time
# limit). Keeps one in-flight solve from overrunning the whole-loop
# budget; the loop also breaks once `deadline` passes.
function _cap_remaining_time(target::JuMP.AbstractModel, deadline::Float64)
    isfinite(deadline) || return
    JuMP.set_time_limit_sec(target, max(0.0, deadline - time()))
    return
end

# Restore the model's solver time limit after the loop — the factory
# value captured before capping, or unset if the factory set none.
_restore_time_limit(model::JuMP.AbstractModel, ::Nothing) =
    JuMP.unset_time_limit_sec(model)
_restore_time_limit(model::JuMP.AbstractModel, seconds::Real) =
    JuMP.set_time_limit_sec(model, seconds)

# Solve the primary NLP for a fixed combination. If feasible, read the
# primal point and objective. If infeasible, fall through to the NLPF
# (V&G 1990 eq. 8) approximation: a slacked version of the same problem
# that always solves, whose primal becomes the linearization site for
# OA cuts. The master still learns shape information from the failed
# combination instead of only adding a no-good cut.
function _solve_nlp(
    model::M,
    combination,
    method::LOA;
    deadline::Float64 = Inf
    ) where {M <: JuMP.AbstractModel}
    _cap_remaining_time(model, deadline)
    primary = with_fixed_combination(model, combination) do
        JuMP.optimize!(model, ignore_optimize_hook = true)
        if JuMP.is_solved_and_feasible(model)
            lin_point = extract_solution(model)
            objective_val = JuMP.objective_value(model)
            return (combination = combination,
                linearization_point = lin_point,
                objective = objective_val, feasible = true)
        end
        return nothing
    end
    primary === nothing || return primary

    # Primary NLP infeasible — try NLPF.
    nlpf = _solve_nlpf(model, combination, method; deadline = deadline)
    nlpf === nothing || return nlpf

    return (combination = combination,
        linearization_point = Dict{JuMP.AbstractVariableRef, Any}(),
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
# be violated by at most `u`.
function _solve_nlpf(
    model::M, combination, method::LOA; deadline::Float64 = Inf
    ) where {M <: JuMP.AbstractModel}
    # The original model's logical binaries are permanently relaxed
    # (see reformulate_model), so this copy is continuous — any NLP
    # solver, including pure-NLP ones like Ipopt, can solve NLPF.
    copy, ref_map = JuMP.copy_model(model)
    JuMP.set_optimizer(copy, method.nlp_optimizer)
    JuMP.set_silent(copy)
    _cap_remaining_time(copy, deadline)

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

    # Pin the copy's indicator binaries to the combination. The infinite
    # `with_fixed_combination` undoes the original's fix after each
    # solve, so this NLPF copy arrives unfixed and its `nlpf_fix_on_copy`
    # override re-pins it. The finite path leaves the original fixed in
    # place, so the copy inherits the fix and the base `nlpf_fix_on_copy`
    # is a no-op.
    for (indicator, value) in combination
        binary = _binary_on_copy(
            _indicator_to_binary(model)[indicator], ref_map)
        nlpf_fix_on_copy(copy, binary, value)
    end

    JuMP.optimize!(copy, ignore_optimize_hook = true)
    JuMP.has_values(copy) || return nothing

    linearization_point = _nlpf_extract_primal(model, ref_map)
    return (combination = combination,
        linearization_point = linearization_point,
        objective = Inf, feasible = false)
end

# Translate a binary reference from the original model to its
# counterpart on the copy. Direct refs go through `ref_map`;
# complement-form `1 - y_orig` rebuilds as `1 - ref_map[y_orig]`.
_binary_on_copy(binary::JuMP.AbstractVariableRef, ref_map) =
    ref_map[binary]
function _binary_on_copy(binary::JuMP.GenericAffExpr, ref_map)
    underlying = only(keys(binary.terms))
    return 1.0 - ref_map[underlying]
end

# OVERRIDABLE. Pin a copy-side binary to a combination value. Base
# no-op: the finite original stays fixed in place (its
# `with_fixed_combination` does not undo), so its NLPF copy inherits the
# fix. The InfiniteOpt extension overrides this — its original is
# cleaned up each iteration, so the copy needs pinning — using
# constraints rather than `JuMP.fix` to avoid force-deleting bounds on
# the relaxed copy.
nlpf_fix_on_copy(copy, binary, value) = nothing

_nlpf_should_slack(::Type{<:_MOI.LessThan}) = true
_nlpf_should_slack(::Type{<:_MOI.GreaterThan}) = true
_nlpf_should_slack(::Type) = false

_nlpf_slacked_func(func, u, ::_MOI.LessThan) = func - u
_nlpf_slacked_func(func, u, ::_MOI.GreaterThan) = func + u

# Read primal values from `copy` keyed by the original model's
# variables, in the same vector shape `extract_solution` would have
# produced on the original.
function _nlpf_extract_primal(model::JuMP.AbstractModel, ref_map)
    V = JuMP.variable_ref_type(typeof(model))
    T = JuMP.value_type(typeof(model))
    result = Dict{V, Vector{T}}()
    for v in collect_all_vars(model)
        JuMP.is_fixed(v) && continue
        target = ref_map[v]
        val = JuMP.value(target)
        result[v] = val isa AbstractArray ? vec(val) : [val]
    end
    return result
end

# Fix `combination` in place, then run `f()`. No relax/restore (the
# model's logical binaries are relaxed once, permanently, in
# reformulate_model) and no fix/undo: `fix_combination` overwrites in
# place, so there is no per-iteration state to unwind — hence no
# try/finally. On a solve error the model is left fixed to the last
# combination; the next `fix_combination` (or `commit_combination`)
# overwrites it.
function with_fixed_combination(
    f,
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    fix_combination(model, combination)
    return f()
end

# Iter-to-iter NLP warm start (T&G 1996 §2.3): seed the next primary
# NLP solve from the most recent FEASIBLE solution's primal point.
# No-op on the first seed iteration when `previous` is `nothing`, and
# no-op on iterations following an NLPF fall-through (caller does not
# update `previous_result` then), since NLPF's primal is slack-
# distorted and a worse start than the prior real NLP. Routes through
# the `set_linearization_start` dispatch so an extension can broadcast
# a vector-valued start across its variables.
function _set_nlp_warm_start(previous)
    previous === nothing && return
    for (variable, values) in previous.linearization_point
        set_linearization_start(variable, values)
    end
    return
end

# Finalize the model with the LOA-optimal combination: fix indicators
# at the committed values (in place; the model is already permanently
# relaxed by reformulate_model) and warm-start from the linearization
# point. After this, the model is no longer in a state suitable for
# re-running with a different `gdp_method`.
function commit_combination(
    model::JuMP.AbstractModel,
    combination::AbstractDict,
    linearization_point::AbstractDict
    )
    fix_combination(model, combination)
    for (variable, values) in linearization_point
        set_linearization_start(variable, values)
    end
    return
end

# OVERRIDABLE. Apply the combination's fixes in place. No undo closure:
# `fix_indicator` uses `force = true` so each call overwrites the prior
# fix, and the model is permanently relaxed, so there is nothing to
# restore. An extension overrides this for vector-valued `Vector{Bool}`
# values (per-support pins reused in place across iterations).
function fix_combination(model::JuMP.AbstractModel, combination::AbstractDict)
    for (indicator, value) in combination
        fix_indicator(model, indicator, value)
    end
    return
end

# OVERRIDABLE. Write the LOA linearization point into a variable's warm
# start. This is deliberately NOT a direct `JuMP.set_start_value` call.
# The linearization point is stored per-variable as a vector — length-1
# for a scalar/finite variable, length-K for a variable carrying K
# support points (see `extract_solution`). For an InfiniteOpt infinite
# variable that per-support vector must be written to each of the K
# *transcribed* JuMP variables (`transformation_variable(v)` is an array
# of per-support refs); `JuMP.set_start_value` on the single infinite
# `GeneralVariableRef` cannot place a distinct start at each support.
# Base handles the scalar case by unwrapping `only(values)`; the
# extension overrides this seam to broadcast the vector across the
# transcribed instances. Hence the dispatch rather than a bare
# `set_start_value`.
set_linearization_start(variable, values::AbstractVector) =
    JuMP.set_start_value(variable, only(values))

################################################################################
#                       COMBO EXTRACTION (master → NLP)
################################################################################
# Read each indicator's binary value from the master MILP solution.
# `combination_val` returns a `Bool` (finite indicator) or per-support
# `Vector{Bool}` (infinite); downstream dispatches on that.
function _extract_combination(
    model::M,
    master::_LOAMaster
    ) where {M <: JuMP.AbstractModel}
    combination = Dict{LogicalVariableRef{M}, Any}()
    for (_, disjunction_data) in _disjunctions(model)
        for indicator in disjunction_data.constraint.indicators
            combination[indicator] =
                combination_val(master.binary_map[indicator])
        end
    end
    return combination
end

# Read the master binary and round to `Bool`. Rounding is required (not
# defensive): a MILP solver returns binaries within its
# integer-feasibility tolerance — e.g. 2.75e-40 for a "0", 1.0000002 for
# a "1" — never exactly 0/1, so `Bool(value)` would throw `InexactError`.
# Not dispatched: the ref type is the same whether the indicator is
# finite or infinite; only `JuMP.value`'s return differs (scalar vs a
# per-support array), so we branch on the value and handle both here.
function combination_val(binary_ref)
    val = JuMP.value(binary_ref)
    return val isa AbstractArray ? vec(round.(Bool, val)) : round(Bool, val)
end

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
    sense_token::Val,
    master::_LOAMaster,
    lin,
    method::LOA
    )
    penalty_sign = _penalty_sign(sense_token)
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
# Base (scalar-model) version. An extension that overrides `add_oa_cuts`
# supplies its own global-cut handling for constraints whose
# linearization is vector-valued or derived.
function _add_global_oa_cuts(
    model::JuMP.AbstractModel,
    master::_LOAMaster,
    result::NamedTuple,
    method::LOA
    )
    penalty_sign = _penalty_sign(Val(master.objective_sense))
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
            _add_global_oa_row(master, linearization, con.set,
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
    penalty_sign = _penalty_sign(Val(master.objective_sense))
    for (indicator, active) in result.combination
        is_active(active) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef || continue
            constraint = _disjunct_constraints(model)[
                JuMP.index(cref)].constraint
            _add_oa_cut_for_constraint(
                constraint, master, master.binary_map[indicator],
                result.linearization_point, master.variable_map,
                method, penalty_sign)
        end
    end
end

# Fresh nonnegative slack added to the master objective with the V&G
# 1990 penalty. Shared by the disjunct, global, and objective OA cuts
# so all three carry the same augmented-penalty treatment: a nonconvex
# linearization can be an invalid relaxation, and the penalized slack
# keeps the master feasible instead of letting accumulated cuts make
# it infeasible.
function _penalized_slack(master::_LOAMaster, method::LOA, penalty_sign::Int)
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
            penalty_sign * method.oa_penalty * slack)
    return slack
end

# Linearize constraint at `linearization_point`, then dispatch on the
# constraint's set to emit slacked OA cut(s) gated by `M(1 − binary)`.
# Linear constraints are exact via BigM and skipped.
function _add_oa_cut_for_constraint(
    constraint::JuMP.AbstractConstraint,
    master::_LOAMaster,
    binary_ref,
    linearization_point::AbstractDict,
    var_map::AbstractDict,
    method::LOA,
    penalty_sign::Int
    )
    _is_linear_F(typeof(constraint.func)) && return
    linearization = _linearize_at(
        constraint.func, linearization_point, var_map)
    _emit_disjunct_oa_cut(
        constraint.set, master, binary_ref, linearization, method,
        penalty_sign)
    return
end

# Emit the slacked, gated OA cut(s) for a disjunct constraint. The cut
# direction is set-determined (Duran-Grossmann 1986 OA convention): a
# `≤` constraint linearizes in place, a `≥` constraint is negated, and
# equality / interval emit both directions sharing one slack. No dual is
# needed — the constraint's set fixes the direction.
function _emit_disjunct_oa_cut(
    set::_MOI.LessThan,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        (linearization - _set_rhs(set)) - slack <=
            method.M_value * (1 - binary_ref))
    return
end
function _emit_disjunct_oa_cut(
    set::_MOI.GreaterThan,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        (_set_rhs(set) - linearization) - slack <=
            method.M_value * (1 - binary_ref))
    return
end
function _emit_disjunct_oa_cut(
    set::_MOI.EqualTo,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    c = _MOI.constant(set)
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        (linearization - c) - slack <= method.M_value * (1 - binary_ref))
    JuMP.@constraint(master.model,
        (c - linearization) - slack <= method.M_value * (1 - binary_ref))
    return
end
function _emit_disjunct_oa_cut(
    set::_MOI.Interval,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
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
# Fallback for vector / future set types: emit the `≤`-direction cut
# against the set's RHS (`_set_rhs` defaults to 0).
function _emit_disjunct_oa_cut(
    set,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model,
        (linearization - _set_rhs(set)) - slack <=
            method.M_value * (1 - binary_ref))
    return
end

# Slacked global OA row(s) (V&G 1990). The nonconvex linearization may
# be an invalid relaxation, so each row carries a penalized slack
# rather than being a hard constraint — without this the accumulated
# global cuts make the master infeasible on nonconvex models.
# `EqualTo` / `Interval` get a two-sided pair sharing one slack.
# Unknown set types fall back to the prior hard cut.
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set::_MOI.LessThan,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - _MOI.constant(set) <= slack)
    return
end
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set::_MOI.GreaterThan,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, _MOI.constant(set) - lin <= slack)
    return
end
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set::_MOI.EqualTo,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    c = _MOI.constant(set)
    JuMP.@constraint(master.model, lin - c <= slack)
    JuMP.@constraint(master.model, c - lin <= slack)
    return
end
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set::_MOI.Interval,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    JuMP.@constraint(master.model, lin - set.upper <= slack)
    JuMP.@constraint(master.model, set.lower - lin <= slack)
    return
end
function _add_global_oa_row(master::_LOAMaster, lin, set, ::LOA, ::Int)
    JuMP.@constraint(master.model, lin in set)
    return
end

# Is this indicator active in the combination? A finite indicator carries
# a `Bool`; an infinite one carries a per-support `Vector{Bool}` (active
# at some supports, not others). Used to skip disjuncts that are off
# everywhere when emitting OA cuts.
is_active(active::Bool) = active
is_active(active::AbstractVector{Bool}) = any(active)

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
    xk::Dict,
    ref_map
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

# Unwrap a 1-element `Vector` to its scalar value; scalars pass
# through. `extract_solution` returns `Vector`s uniformly (length-1 for
# a scalar variable, length-K for a vector-valued one). AD pipelines
# and `set_start_value` need a scalar in the scalar case; vector-valued
# consumers slice out a scalar themselves.
_unwrap_scalar(v::Real) = v
_unwrap_scalar(v::AbstractVector) = only(v)
