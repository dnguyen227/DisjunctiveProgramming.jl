################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################

################################################################################
#                              METHOD TYPE
################################################################################
"""
    LOA{O, P, R, T} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models. Iterates a primary
NLP (original model reformulated by `inner_method`, binaries fixed per
iteration) and a master MILP accumulating OA and no-good cuts.
`inner_method` is `BigM` (default), `MBM`, or `Hull`.

## Fields
- `nlp_optimizer::O`: solver for the primary NLP.
- `mip_optimizer::P`: solver for the master MILP (default `nlp_optimizer`).
- `inner_method::R`: NLP reformulation — `BigM`, `MBM`, or `Hull`.
- `max_iter::Int`: max iterations after set-covering seeding.
- `M_value::T`: big-M for the disjunct OA cut gating term.
- `max_slack::T`: upper bound per slack variable.
- `oa_penalty::T`: penalty on slacks in the master objective.
- `convergence_tol::Float64`: relative gap tolerance for the early stop.
- `slack_tol::Float64`: max total slack for which the bound still counts
  as converged (positive slack = nonconvex crossing, keep iterating).
- `iteration_time_limit::Float64`: budget (s) for the iteration loop.
- `time_limit::Float64`: overall budget (s) incl. the final solve
  (default 3600; `Inf` disables).
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
        time_limit::Float64 = 3600.0
        ) where {O, P, R <: AbstractReformulationMethod, T}
        R <: Union{BigM, MBM, Hull} || error(
            "LOA inner_method must be BigM, MBM, or Hull (got $R). " *
            "PSplit is not yet supported.")
        new{O, P, R, T}(nlp_optimizer, mip_optimizer, inner_method,
            max_iter, M_value, max_slack, oa_penalty,
            convergence_tol, slack_tol,
            iteration_time_limit, time_limit)
    end
end

################################################################################
#                              LOA MASTER
################################################################################
# The LOA master MILP plus maps from original- to master-model refs.
# `objective_ref_map` splits from `variable_map` so an extension can map
# objective vars separately (identical in base); `disaggregator` is the
# master `_Hull` for Hull cuts, `nothing` for Big-M / MBM.
mutable struct _LOAMaster{M <: JuMP.AbstractModel, OF, AO, BM, VM, RM, DG}
    model::M
    binary_map::BM
    variable_map::VM
    objective_sense::_MOI.OptimizationSense
    original_objective::OF
    alpha_oa::AO
    objective_ref_map::RM
    disaggregator::DG
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
# LOA entry point: set-covering seeds, the master/NLP loop, commit the
# best combination. Model-agnostic — extensions override the inner steps.
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    combinations = _set_covering_combinations(model)
    # Hull needs the disaggregation map; thread a LOA-owned sink through
    # the inner reformulation to collect it (kept off GDPData).
    inner, sink = _loa_inner_method(model, method.inner_method)
    reformulate_model(model, inner)

    master = build_loa_master(model, method, sink)
    # The original is only ever the NLP (build_loa_master keeps the
    # master's binaries), so relax it once; fixing overwrites in place.
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

    # Seed the master with an OA cut per set-covering combination, each
    # NLP warm-started from the last feasible primal.
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

    # Master/NLP loop: `alpha_oa` is the bound, the NLP refines the
    # incumbent. Exit on convergence (bound meets incumbent and slack
    # settled), infeasible master, max_iter, or time.
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

# Report the final gap. The bound is rigorous only for a convex inner
# problem; on a nonconvex one it can cross the incumbent (negative gap).
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
# `K = max disjunction size` combinations that activate every indicator
# at least once: combination `k` activates the `k`-th indicator of each
# disjunction, cycling via `mod1`. Inconsistent nested combinations are
# caught by the no-good cut from the infeasible NLP.
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
# True for linear constraint function types (variable refs / affine).
# Nonlinear functions enter the master as OA cuts, not at copy time.
_is_linear_F(::Type{<:JuMP.AbstractVariableRef}) = true
_is_linear_F(::Type{<:JuMP.GenericAffExpr}) = true
_is_linear_F(::Type{<:AbstractVector{<:JuMP.AbstractVariableRef}}) = true
_is_linear_F(::Type{<:AbstractVector{<:JuMP.GenericAffExpr}}) = true
_is_linear_F(::Type) = false

# OVERRIDABLE. Build the master MILP: copy the variables and linear
# constraints, install `alpha_oa` as the objective auxiliary. Nonlinear
# objective and disjunct constraints enter as OA cuts per NLP solve.
# `sink` is the disaggregation map collected by the inner Hull
# reformulation (`nothing` for Big-M / MBM).
function build_loa_master(
    model::JuMP.AbstractModel,
    method::LOA,
    sink = nothing
    )
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

    disaggregator = _build_disaggregator(model, method.inner_method,
        variable_map, binary_map, sink)
    return _LOAMaster(master, binary_map, variable_map, objective_sense,
        original_objective, alpha_oa, variable_map, disaggregator)
end

# The inner reformulation method LOA runs, plus the disaggregation sink to
# collect (Big-M / MBM need none). For Hull, return a sink-carrying copy
# so the reformulation records its `(variable, indicator) -> disaggregated
# variable` map into a fresh LOA-owned `Dict`.
_loa_inner_method(::JuMP.AbstractModel, inner::Union{BigM, MBM}) =
    (inner, nothing)
function _loa_inner_method(model::JuMP.AbstractModel, inner::Hull)
    V = JuMP.variable_ref_type(typeof(model))
    sink = Dict{Tuple{V, LogicalVariableRef{typeof(model)}}, V}()
    return (Hull(inner.value; sink = sink), sink)
end

# Master-space Hull disaggregator for the convex-hull cut emitter
# (`nothing` for Big-M / MBM). Remaps the collected `(variable, indicator)
# -> disaggregated variable` sink into master refs.
_build_disaggregator(::JuMP.AbstractModel, ::Union{BigM, MBM},
    variable_map, binary_map, sink) = nothing
function _build_disaggregator(
    model::JuMP.AbstractModel,
    inner::Hull,
    variable_map::AbstractDict,
    binary_map::AbstractDict,
    sink
    )
    hull = _Hull(inner, Set{valtype(variable_map)}())
    for ((variable, indicator), disaggregated) in sink
        record_disaggregation(hull, model, variable_map[variable],
            binary_map[indicator], variable_map[disaggregated])
    end
    return hull
end

# OVERRIDABLE. Record one master-space disaggregation in the Hull
# disaggregator. Base keys it directly by `(variable, binary)`; the
# InfiniteOpt extension keys per support so per-support cut emission
# matches.
function record_disaggregation(
    hull::_Hull,
    ::JuMP.AbstractModel,
    variable,
    binary,
    disaggregated
    )
    hull.disjunct_variables[(variable, binary)] = disaggregated
    return
end

################################################################################
#                          NLP SUBPROBLEM
################################################################################
# Cap `target`'s solver to the budget left before `deadline` so one solve
# can't overrun the loop. No-op when `deadline` is `Inf`.
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

# Solve the primary NLP at a fixed combination. If infeasible, fall
# through to NLPF (a slacked version that always solves) so the master
# still gets a linearization site, not just a no-good cut.
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
            return (combination = combination,
                linearization_point = extract_solution(model),
                objective = JuMP.objective_value(model), feasible = true)
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
# When the primary NLP is infeasible, copy the model, slack every scalar
# inequality with one nonnegative `u`, minimize `u`, and return the
# primal as the linearization site (equalities/bounds exact, inequalities
# violated by at most `u`).
function _solve_nlpf(
    model::M,
    combination,
    method::LOA;
    deadline::Float64 = Inf
    ) where {M <: JuMP.AbstractModel}
    # The original's binaries are permanently relaxed, so this copy is
    # continuous — any NLP solver (e.g. Ipopt) can solve NLPF.
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

    # Fix the combination on the copy so NLPF solves at the chosen
    # binaries, not by leaning on the copy inheriting the original's fix.
    fix_combination_on_copy(copy, model, combination, ref_map)

    JuMP.optimize!(copy, ignore_optimize_hook = true)
    JuMP.has_values(copy) || return nothing

    linearization_point = _nlpf_extract_primal(model, ref_map)
    return (combination = combination,
        linearization_point = linearization_point,
        objective = Inf, feasible = false)
end

# Fix each indicator's binary on the NLPF copy at its combination value,
# mapped into copy space. Explicit so NLPF is self-contained rather than
# relying on the copy inheriting the original's fix.
function fix_combination_on_copy(copy, model, combination, ref_map)
    for (indicator, value) in combination
        binary = _binary_on_copy(
            _indicator_to_binary(model)[indicator], ref_map)
        _fix_binary_on_copy(copy, binary, value)
    end
    return
end

# Map an original binary (or its `1 - y` complement) into NLPF-copy space.
_binary_on_copy(binary::JuMP.AbstractVariableRef, ref_map) = ref_map[binary]
_binary_on_copy(binary::JuMP.GenericAffExpr, ref_map) =
    1.0 - ref_map[only(keys(binary.terms))]

# OVERRIDABLE. Fix one binary on the copy at a scalar value via an
# equality (the copy is discarded, so the fix needs no teardown). The
# InfiniteOpt extension adds a per-support method for a `Vector{Bool}`.
_fix_binary_on_copy(copy, binary, value::Bool) =
    JuMP.@constraint(copy, binary == (value ? 1.0 : 0.0))

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

# Fix `combination`, run `f()`, then undo. The finite undo is a no-op
# (fixes overwrite in place); the infinite `fix_combination` clears its
# per-support pins.
function with_fixed_combination(
    f,
    model::JuMP.AbstractModel,
    combination::AbstractDict
    )
    undo = fix_combination(model, combination)
    try
        return f()
    finally
        undo()
    end
end

# Iter-to-iter NLP warm start: seed the next NLP from the last FEASIBLE
# primal (no-op on the first seed and after an NLPF fall-through, whose
# primal is slack-distorted). Routes through `set_linearization_start`.
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

# OVERRIDABLE. Apply the combination's fixes in place and return an undo
# closure (a no-op here, since the finite fixes overwrite in place). An
# extension overrides this for per-support `Vector{Bool}` values.
function fix_combination(model::JuMP.AbstractModel, combination::AbstractDict)
    for (indicator, value) in combination
        fix_indicator(model, indicator, value)
    end
    return () -> nothing
end

# OVERRIDABLE. Warm-start a variable from the linearization point (stored
# as a per-support vector). Base unwraps the scalar; the InfiniteOpt
# extension broadcasts across the transcribed per-support refs.
set_linearization_start(variable, values::AbstractVector) =
    JuMP.set_start_value(variable, only(values))

################################################################################
#                       COMBO EXTRACTION (master -> NLP)
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

# Read the master binary and round to `Bool` (a MILP solver returns
# values within its integer tolerance, never exactly 0/1). Branches on
# the value: scalar (finite) or per-support array (infinite).
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
        objective_linearization_point(model, result.linearization_point),
        master.objective_ref_map)
    _add_objective_cut(
        Val(master.objective_sense), master, linearization, method)
    add_global_oa_cuts(model, master, result, method)
    add_disjunct_oa_cuts(model, master, result, method)
    return
end

# OVERRIDABLE. The point at which the master objective is linearized.
# Base uses the raw point; an extension whose `original_objective` is
# transcribed or derived overrides this to match its shape.
objective_linearization_point(::JuMP.AbstractModel, linearization_point) =
    linearization_point

# Slacked objective cut. MIN: `lin <= alpha_oa + sigma`; MAX symmetric.
# The slack (like the disjunct/global cuts) keeps a nonconvex objective
# linearization from making the master infeasible.
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

# OVERRIDABLE. Add an OA cut for every nonlinear global constraint,
# skipping variable bounds, linear functions, and reformulation
# constraints. Base (scalar) version; an extension overrides it for
# vector-valued / transcribed linearizations.
function add_global_oa_cuts(
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

# Add slacked OA cuts for each active disjunct's nonlinear constraints.
# This driver (iterate active disjuncts and their constraints) is shared;
# the per-constraint emission is the OVERRIDABLE seam below. `cache` is a
# per-pass scratch passed to each seam call (used by the extension).
function add_disjunct_oa_cuts(
    model::JuMP.AbstractModel,
    master::_LOAMaster,
    result::NamedTuple,
    method::LOA
    )
    penalty_sign = _penalty_sign(Val(master.objective_sense))
    cache = Ref{Any}(nothing)
    for (indicator, active) in result.combination
        is_active(active) || continue
        haskey(_indicator_to_constraints(model), indicator) || continue
        for cref in _indicator_to_constraints(model)[indicator]
            cref isa DisjunctConstraintRef || continue
            constraint = _disjunct_constraints(model)[
                JuMP.index(cref)].constraint
            add_disjunct_constraint_oa_cuts(model, constraint, master,
                master.binary_map[indicator], active, result, method,
                penalty_sign, cache)
        end
    end
end

# OVERRIDABLE. Emit the OA cut(s) for one active disjunct constraint.
# Base emits a single cut; the InfiniteOpt extension fans out per
# support. `active` is unused in base but drives the extension's fan-out;
# `cache` is a per-pass scratch the extension memoizes transcription into.
function add_disjunct_constraint_oa_cuts(
    ::JuMP.AbstractModel,
    constraint::JuMP.AbstractConstraint,
    master::_LOAMaster,
    binary_ref,
    active,
    result::NamedTuple,
    method::LOA,
    penalty_sign::Int,
    cache
    )
    _add_oa_cut_for_constraint(
        constraint, master, binary_ref, result.linearization_point,
        master.variable_map, method, penalty_sign)
    return
end

# Fresh penalized slack added to the master objective. Shared by the
# disjunct, global, and objective cuts so a nonconvex (invalid)
# linearization can't make the master infeasible.
function _penalized_slack(master::_LOAMaster, method::LOA, penalty_sign::Int)
    slack = JuMP.@variable(master.model,
        lower_bound = 0.0, upper_bound = method.max_slack)
    JuMP.set_objective_function(master.model,
        JuMP.objective_function(master.model) +
            penalty_sign * method.oa_penalty * slack)
    return slack
end

# Linearize the constraint at `linearization_point`, then emit slacked OA
# cut(s) gated to match the inner reformulation (Big-M / MBM or Hull).
# Linear constraints are exact via the reformulation and skipped.
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
    _emit_disjunct_oa_cut(method.inner_method,
        constraint.set, master, binary_ref, linearization, method,
        penalty_sign)
    return
end

# The `<= 0` directions of an OA cut for `set`: `lin - rhs` for LessThan,
# `rhs - lin` for GreaterThan, both for EqualTo / Interval. The caller
# adds the gating, slack, and (for Hull) disaggregation per direction.
_oa_cut_terms(set::_MOI.GreaterThan, lin) = (_set_rhs(set) - lin,)
_oa_cut_terms(set::_MOI.EqualTo, lin) =
    (lin - _MOI.constant(set), _MOI.constant(set) - lin)
_oa_cut_terms(set::_MOI.Interval, lin) = (lin - set.upper, set.lower - lin)
_oa_cut_terms(set, lin) = (lin - _set_rhs(set),)

# Big-M / MBM disjunct cut: each direction slacked and gated by
# `M(1 - binary)`.
function _emit_disjunct_oa_cut(
    ::Union{BigM, MBM},
    set,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    for term in _oa_cut_terms(set, linearization)
        JuMP.@constraint(master.model,
            term - slack <= method.M_value * (1 - binary_ref))
    end
    return
end

# Convex-hull disjunct cut: `disaggregate_expression` rewrites each
# direction into disaggregated space (variables -> per-disjunct copies,
# constant scaled by the binary), so the cut switches off at `y = 0` with
# no big-M. It is linear in its argument, so the two-sided sets need no
# special casing.
function _emit_disjunct_oa_cut(
    ::Hull,
    set,
    master::_LOAMaster,
    binary_ref,
    linearization,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    for term in _oa_cut_terms(set, linearization)
        body = disaggregate_expression(
            master.model, term, binary_ref, master.disaggregator)
        JuMP.@constraint(master.model, body - slack <= 0)
    end
    return
end

# Slacked global OA row(s): each direction carries a penalized slack so a
# nonconvex linearization can't make the master infeasible. Unknown sets
# fall back to a hard cut.
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set::Union{_MOI.LessThan, _MOI.GreaterThan, _MOI.EqualTo, _MOI.Interval},
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    for term in _oa_cut_terms(set, lin)
        JuMP.@constraint(master.model, term <= slack)
    end
    return
end
function _add_global_oa_row(master::_LOAMaster, lin, set, ::LOA, ::Int)
    JuMP.@constraint(master.model, lin in set)
    return
end

# Is this indicator active anywhere? Finite: a `Bool`; infinite: a
# per-support `Vector{Bool}`. Skips disjuncts that are off everywhere.
is_active(active::Bool) = active
is_active(active::AbstractVector{Bool}) = any(active)

################################################################################
#                    LINEARIZATION & EXPRESSION CONVERSION
################################################################################
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

# Unwrap a 1-element `Vector` to its scalar; scalars pass through
# (`extract_solution` returns length-1 vectors for scalar variables).
_unwrap_scalar(v::Real) = v
_unwrap_scalar(v::AbstractVector) = only(v)
