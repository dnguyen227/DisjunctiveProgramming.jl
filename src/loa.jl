################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################

################################################################################
#                            SENSE HANDLERS
################################################################################
_penalty_sign(::Val{_MOI.MIN_SENSE}) = 1
_penalty_sign(::Val{_MOI.MAX_SENSE}) = -1
_worst_objective(::Val{_MOI.MIN_SENSE}) = Inf
_worst_objective(::Val{_MOI.MAX_SENSE}) = -Inf
_is_better(::Val{_MOI.MIN_SENSE}, new, best) = new < best
_is_better(::Val{_MOI.MAX_SENSE}, new, best) = new > best
_gap(::Val{_MOI.MIN_SENSE}, best, bound) = best - bound
_gap(::Val{_MOI.MAX_SENSE}, best, bound) = bound - best
# A valid overall bound is the weaker of two partial bounds.
_loosest_bound(::Val{_MOI.MIN_SENSE}, best, bound) = min(best, bound)
_loosest_bound(::Val{_MOI.MAX_SENSE}, best, bound) = max(best, bound)

# One progress record per NLP solve, for external convergence traces.
function _log_loa_progress(t_start::Float64, best_objective, master_bound)
    bound = master_bound === nothing ? NaN : master_bound
    @info "LOA progress: elapsed=$(time() - t_start) " *
        "incumbent=$best_objective bound=$bound"
    return
end

################################################################################
#                            MAIN LOOP
################################################################################
# LOA optimize hook: build the problem, seed with set-covering
# combinations, iterate the master/NLP loop, then load the results into
# the model by injecting every feasible combination found (best first).
function _optimize_hook(model::JuMP.AbstractModel, method::LOA; kwargs...)
    _clear_reformulations(model)
    reformulate_model(model, method.inner_method)

    problem = build_loa_problem(model, method)
    # The master copies the NLP while its binaries still carry
    # integrality; the NLP is then relaxed once and each iteration
    # overwrites the binary fixes in place.
    master = _build_loa_master(problem, method)
    _relax_binaries(problem)
    nlp = problem.nlp
    displaced_optimizer = _install_nlp_optimizer(nlp, method.nlp_optimizer,
        gdp_data(model).displaced_optimizer)
    JuMP.set_silent(nlp)
    t_start = time()
    overall_deadline = t_start + method.time_limit
    loop_deadline =
        min(t_start + method.iteration_time_limit, overall_deadline)
    original_time_limit = JuMP.time_limit_sec(nlp)
    sense_token = Val(master.objective_sense)
    best_objective = _worst_objective(sense_token)
    best_result = nothing
    master_bound = nothing
    feasible_results = NamedTuple[]
    master_failure = nothing
    cuts_added = false
    converged = false

    # Set-covering seed. This mimics Pyomo GDPopt's set-covering
    # initialization: borrow the master as a covering MILP (only its
    # objective changes), let it pick a combination that activates the
    # nonlinear disjuncts still lacking a linearization, solve the NLP
    # there, and seed the resulting OA and no-good cuts. Each NLP
    # warm-starts from the last feasible primal.
    previous_result = nothing
    cover_disjuncts = _cover_disjuncts(problem)
    needs_cover = trues(length(cover_disjuncts))
    num_covered = 0
    for iteration in 1:method.set_cover_max_iter
        (iteration == 1 || any(needs_cover)) || break
        time() < loop_deadline || break
        # Swap in the covering objective, solve, read off the combination,
        # then restore the OA objective (with its accumulated slack
        # penalties) before emitting cuts against it.
        oa_objective = JuMP.objective_function(master.model)
        JuMP.@objective(master.model, Max,
            _cover_objective(master, cover_disjuncts, needs_cover,
                num_covered))
        _cap_remaining_time(master.model, loop_deadline)
        JuMP.optimize!(master.model)
        solved = JuMP.is_solved_and_feasible(master.model)
        combination = solved ? _extract_combination(problem, master) : nothing
        JuMP.set_objective_sense(master.model, master.objective_sense)
        JuMP.set_objective_function(master.model, oa_objective)
        if !solved
            master_failure = JuMP.termination_status(master.model)
            break
        end
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(problem, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.variable_map)
        _add_oa_cuts(problem, master, result, method)
        cuts_added = true
        if result.feasible
            push!(feasible_results, result)
            previous_result = result
            if _is_better(sense_token, result.objective, best_objective)
                best_objective = result.objective
                best_result = result
            end
        end
        _log_loa_progress(t_start, best_objective, master_bound)
        # A disjunct counts as covered only once it is active in a
        # feasible NLP. An infeasible combination still contributes its
        # no-good cut above but leaves the coverage targets untouched.
        if result.feasible
            for i in eachindex(cover_disjuncts)
                needs_cover[i] || continue
                _disjunct_active(result.combination, cover_disjuncts[i]) &&
                    (needs_cover[i] = false)
            end
            num_covered = count(!, needs_cover)
        end
    end

    # Master/NLP loop: `alpha_oa` is the bound, the NLP refines the
    # incumbent. Exit on convergence, infeasible master, max_iter, or time.
    for _ in 1:method.max_iter
        time() < loop_deadline || break
        _cap_remaining_time(master.model, loop_deadline)
        JuMP.optimize!(master.model)
        if !JuMP.is_solved_and_feasible(master.model)
            master_failure = JuMP.termination_status(master.model)
            break
        end
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
        combination = _extract_combination(problem, master)
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(problem, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.variable_map)
        _add_oa_cuts(problem, master, result, method)
        cuts_added = true
        if result.feasible
            push!(feasible_results, result)
            previous_result = result
            if _is_better(sense_token, result.objective, best_objective)
                best_objective = result.objective
                best_result = result
            end
        end
        _log_loa_progress(t_start, best_objective, master_bound)
    end

    _restore_time_limit(nlp, original_time_limit)
    status = _loa_termination_status(best_result !== nothing, converged,
        master_failure, cuts_added, time() >= loop_deadline)
    message = _loa_status_message(status, best_objective, best_result,
        master_bound, sense_token)
    best_result === nothing ? (@warn message) : (@info message)
    # Must be read before the injection replaces the NLP's optimizer.
    solver = "LOA(" * string(nameof(typeof(method.inner_method))) *
        ", nlp = " * JuMP.solver_name(nlp) *
        ", master = " * JuMP.solver_name(master.model) * ")"
    sort!(feasible_results; by = result -> result.objective,
        rev = master.objective_sense == _MOI.MAX_SENSE)
    # No-good cuts mean the master bound only covers unvisited
    # combinations, so the reported bound folds in the incumbent
    # (`raw_status` above keeps the unfolded master bound).
    bound = if best_result === nothing
        master_bound
    elseif _infeasible_master(master_failure)
        best_objective
    elseif master_bound === nothing
        nothing
    else
        _loosest_bound(sense_token, best_objective, master_bound)
    end
    outcome = (results = feasible_results, status = status,
        raw_status = message, bound = bound,
        solve_time = time() - t_start, solver_name = solver,
        displaced_optimizer = displaced_optimizer)
    # Injection is not a reformulation; a later method must rebuild.
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, false)
    _load_solution(model, problem, outcome, method)
    return
end

# Master statuses that report infeasibility. A local claim
# (LOCALLY_INFEASIBLE, e.g. from an NLP-based MILP solver like Juniper)
# counts for combination exhaustion, but only a global certificate
# justifies the global INFEASIBLE report.
_infeasible_master(::Nothing) = false
_infeasible_master(status::_MOI.TerminationStatusCode) = status in
    (_MOI.INFEASIBLE, _MOI.INFEASIBLE_OR_UNBOUNDED,
        _MOI.LOCALLY_INFEASIBLE)
_proven_infeasible_master(::Nothing) = false
_proven_infeasible_master(status::_MOI.TerminationStatusCode) = status in
    (_MOI.INFEASIBLE, _MOI.INFEASIBLE_OR_UNBOUNDED)

# LOA-level termination status, reported the way a native solver would:
# convergence and combination exhaustion (master infeasible with an
# incumbent in hand) are locally solved (a local NLP solver cannot
# certify global optimality), a master proven infeasible before any cut
# proves the linear relaxation infeasible, exhaustion without an
# incumbent is only a local claim, and otherwise the binding limit is
# reported (`OTHER_LIMIT` for a master solve that failed without an
# infeasibility status, e.g. on its own time cap).
function _loa_termination_status(
    found_incumbent::Bool,
    converged::Bool,
    master_failure::Union{Nothing, _MOI.TerminationStatusCode},
    cuts_added::Bool,
    timed_out::Bool
    )
    (converged || (_infeasible_master(master_failure) && found_incumbent)) &&
        return _MOI.LOCALLY_SOLVED
    _proven_infeasible_master(master_failure) && !cuts_added &&
        return _MOI.INFEASIBLE
    _infeasible_master(master_failure) && return _MOI.LOCALLY_INFEASIBLE
    timed_out && return _MOI.TIME_LIMIT
    master_failure === nothing || return _MOI.OTHER_LIMIT
    return _MOI.ITERATION_LIMIT
end

# Human-readable run summary: logged at the end of the hook and stored
# as the model's `RawStatusString`. The bound is rigorous only for a
# convex inner problem; on a nonconvex one it can cross the incumbent
# (negative gap).
function _loa_status_message(
    status::_MOI.TerminationStatusCode,
    best_objective::Real,
    best_result,
    master_bound,
    sense_token::Val
    )
    best_result === nothing &&
        return "LOA finished [$status]: no feasible incumbent found."
    master_bound === nothing &&
        return "LOA finished [$status]: incumbent $best_objective " *
            "(master produced no bound; best seed kept)."
    gap = _gap(sense_token, best_objective, master_bound)
    relative = abs(best_objective) > 1e-10 ?
        gap / abs(best_objective) : gap
    label = status == _MOI.LOCALLY_SOLVED ? "converged" : "limit hit"
    return "LOA finished [$label]: incumbent $best_objective, master " *
        "bound $master_bound, gap $gap (relative $relative)."
end

################################################################################
#                       SET-COVERING INITIALIZATION
################################################################################
# The nonlinear disjuncts to cover: one entry per distinct indicator that
# owns a nonlinear disjunct constraint, carrying that indicator's
# `binary_ref`. Keyed by `(underlying binary, active value)` so the two
# disjuncts of a single-binary disjunction (`y` and its `1 - y`
# complement) stay distinct. Purely linear disjuncts need no cover: the
# inner reformulation already places them in the master exactly.
function _cover_disjuncts(problem::_LOAProblem{M, V, T}) where {M, V, T}
    seen = Set{Tuple{V, Bool}}()
    disjuncts = Union{V, JuMP.GenericAffExpr{T, V}}[]
    for (binary_ref, _, _) in problem.disjunct_constraints
        key = (_underlying_binary(binary_ref),
            _underlying_value(binary_ref, true))
        key in seen && continue
        push!(seen, key)
        push!(disjuncts, binary_ref)
    end
    return disjuncts
end

# The set-covering objective (master space), mimicking Pyomo GDPopt:
# maximize active disjuncts weighted `num_covered + 1` if still uncovered
# else `1`, so one uncovered disjunct outweighs every covered one and each
# solve must activate a new disjunct when the logic allows. Empty
# `disjuncts` gives the zero expression: a constant objective that still
# seeds one feasible combination (needed to bound `alpha_oa`).
function _cover_objective(
    master::_LOAMaster,
    disjuncts,
    needs_cover,
    num_covered::Int
    )
    T = JuMP.value_type(typeof(master.model))
    V = JuMP.variable_ref_type(typeof(master.model))
    expr = JuMP.GenericAffExpr{T, V}(zero(T))
    for i in eachindex(disjuncts)
        weight = needs_cover[i] ? num_covered + 1 : 1
        activation = _remap_indicator_to_binary(disjuncts[i],
            master.variable_map)
        JuMP.add_to_expression!(expr, T(weight), activation)
    end
    return expr
end

################################################################################
#                       FINITE PROBLEM CONSTRUCTION
################################################################################
# The concrete variable behind an indicator's binary reference (the
# underlying variable of a `1 - y` complement expression), and the value
# it takes when the indicator is active.
_underlying_binary(binary_ref::JuMP.AbstractVariableRef) = binary_ref
_underlying_binary(binary_ref::JuMP.GenericAffExpr) =
    only(keys(binary_ref.terms))
_underlying_value(::JuMP.AbstractVariableRef, active::Bool) = active
_underlying_value(::JuMP.GenericAffExpr, active::Bool) = !active

"""
    build_loa_problem(
        model::JuMP.AbstractModel,
        method::LOA
        )::_LOAProblem

Build the problem the LOA loop operates on from `model` (already
reformulated by the LOA inner method): the NLP subproblem, the binary
variables backing the indicators, the nonlinear disjunct
`(binary_ref, function, set)` triples, the nonlinear global
`(function, set)` pairs, and the Hull disaggregation map recorded in
the GDP data during reformulation. The NLP is `model` itself; the
InfiniteOpt extension overloads this to build the problem from the
transcribed backend instead.

## Returns
- `_LOAProblem`: the problem.
"""
function build_loa_problem(model::JuMP.AbstractModel, method::LOA)
    V = JuMP.variable_ref_type(typeof(model))
    T = JuMP.value_type(typeof(model))
    binary_map = _indicator_to_binary(model)

    binaries = V[_underlying_binary(binary_ref)
        for (_, binary_ref) in binary_map]
    unique!(binaries)

    disjunct_constraints = Tuple{Union{V, JuMP.GenericAffExpr{T, V}},
        JuMP.AbstractJuMPScalar, _MOI.AbstractScalarSet}[]
    for (_, disjunction) in _disjunctions(model)
        for indicator in disjunction.constraint.indicators
            haskey(_indicator_to_constraints(model), indicator) || continue
            binary_ref = binary_map[indicator]
            for cref in _indicator_to_constraints(model)[indicator]
                cref isa DisjunctConstraintRef || continue
                constraint = _disjunct_constraints(model)[
                    JuMP.index(cref)].constraint
                constraint isa JuMP.ScalarConstraint || continue
                _is_linear_F(typeof(constraint.func)) && continue
                push!(disjunct_constraints,
                    (binary_ref, constraint.func, constraint.set))
            end
        end
    end

    global_constraints = Tuple{JuMP.AbstractJuMPScalar,
        _MOI.AbstractScalarSet}[]
    reform_set = Set(_reformulation_constraints(model))
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === V && continue
        _is_linear_F(F) && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            constraint = JuMP.constraint_object(cref)
            constraint isa JuMP.ScalarConstraint || continue
            push!(global_constraints, (constraint.func, constraint.set))
        end
    end

    binary_disaggregations =
        Dict{Tuple{V, Union{V, JuMP.GenericAffExpr{T, V}}}, V}(
            (variable, binary_map[indicator]) => disaggregated
            for ((variable, indicator), disaggregated)
                in _disaggregation_map(model))
    return _LOAProblem(model, binaries, disjunct_constraints,
        global_constraints, binary_disaggregations)
end

# Relax the binaries to continuous in [0, 1]; each NLP solve then
# overwrites their fixes in place.
function _relax_binaries(problem::_LOAProblem)
    for binary in problem.binaries
        JuMP.unset_binary(binary)
        JuMP.set_lower_bound(binary, 0.0)
        JuMP.set_upper_bound(binary, 1.0)
    end
    return
end

# Undo `_relax_binaries` and the loop's last combination fix: drop the
# fix and the relaxation bounds, then restore integrality so the model
# is handed back with its binaries exactly as the reformulation created
# them. (A fixed binary carries no bounds — `fix(; force = true)`
# deleted them — while an unfixed one still carries the relaxation's.)
function _restore_binaries(problem::_LOAProblem)
    for binary in problem.binaries
        JuMP.is_fixed(binary) && JuMP.unfix(binary)
        JuMP.has_lower_bound(binary) && JuMP.delete_lower_bound(binary)
        JuMP.has_upper_bound(binary) && JuMP.delete_upper_bound(binary)
        JuMP.set_binary(binary)
    end
    return
end

################################################################################
#                         OPTIMIZER HANDOVER
################################################################################
# Install the NLP solver, returning the optimizer it displaces. LOA
# solves the NLP on the model itself, so that one is the user's own. A
# re-solve displaces the last run's cache instead, so `previous` (the
# optimizer already stashed) carries through untouched.
function _install_nlp_optimizer(
    nlp::JuMP.AbstractModel,
    nlp_optimizer,
    previous
    )
    backend = JuMP.backend(nlp)
    displaced = if backend isa _MOI.Utilities.CachingOptimizer &&
            backend.optimizer !== nothing
        JuMP.unsafe_backend(nlp)
    else
        nothing
    end
    JuMP.set_optimizer(nlp, nlp_optimizer)
    return displaced isa _LOAResultCache ? previous : displaced
end

# Model types without a MOI backend of their own never carry a cache.
function _has_loa_cache(model::JuMP.GenericModel)
    backend = JuMP.backend(model)
    return backend isa _MOI.Utilities.CachingOptimizer &&
        backend.optimizer isa _LOAResultCache
end
_has_loa_cache(::JuMP.AbstractModel) = false

# Restore the displaced optimizer, emptied so the re-solve copies the
# model in. Without one, drop the cache rather than serve its results.
function _restore_displaced_optimizer(model::JuMP.AbstractModel)
    displaced = gdp_data(model).displaced_optimizer
    if displaced === nothing
        _MOI.Utilities.drop_optimizer(JuMP.backend(model))
    else
        _MOI.empty!(displaced)
        JuMP.set_optimizer(model, () -> displaced)
    end
    return
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

# Build the LOA master MILP: copy the NLP's variables and linear
# constraints, install the `alpha_oa` objective auxiliary, and record
# the NLP-to-master variable map. Nonlinear constraints are not
# copied; they enter later as OA cuts per NLP solve.
function _build_loa_master(problem::_LOAProblem, method::LOA)
    nlp = problem.nlp
    objective = JuMP.objective_function(nlp)
    objective_sense = JuMP.objective_sense(nlp)
    variable_type = JuMP.variable_ref_type(typeof(nlp))

    master = _copy_model(nlp)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)

    variable_map = copy_variables_onto_model(master, nlp)

    for (F, S) in JuMP.list_of_constraint_types(nlp)
        F === variable_type && continue
        _is_linear_F(F) || continue
        for constraint_ref in JuMP.all_constraints(nlp, F, S)
            constraint = JuMP.constraint_object(constraint_ref)
            new_func = replace_variables_in_constraint(
                constraint.func, variable_map)
            JuMP.@constraint(master, new_func in constraint.set)
        end
    end

    alpha_oa = JuMP.@variable(master, base_name = "alpha_oa")
    JuMP.@objective(master, objective_sense, alpha_oa)

    disaggregator = _build_disaggregator(problem, variable_map,
        method.inner_method)
    return _LOAMaster(master, variable_map, objective_sense, objective,
        alpha_oa, disaggregator)
end

# Master-space Hull disaggregator for the convex-hull cut emitter
# (`nothing` for Big-M / MBM): remap the `(variable, binary_ref) ->
# disaggregated variable` map into master refs.
_build_disaggregator(::_LOAProblem, ::Dict, ::Union{BigM, MBM}) = nothing
function _build_disaggregator(
    problem::_LOAProblem,
    variable_map::Dict{V, V},
    inner::Hull
    ) where {V <: JuMP.AbstractVariableRef}
    hull = _Hull(inner, Set{valtype(variable_map)}())
    for ((variable, binary_ref), disaggregated) in problem.disaggregation_map
        hull.disjunct_variables[(variable_map[variable],
            _remap_indicator_to_binary(binary_ref, variable_map))] =
            variable_map[disaggregated]
    end
    return hull
end

################################################################################
#                          NLP SUBPROBLEM
################################################################################
function _cap_remaining_time(target::JuMP.AbstractModel, deadline::Float64)
    isfinite(deadline) || return
    JuMP.set_time_limit_sec(target, max(0.0, deadline - time()))
    return
end
_restore_time_limit(model::JuMP.AbstractModel, ::Nothing) =
    JuMP.unset_time_limit_sec(model)
_restore_time_limit(model::JuMP.AbstractModel, seconds::Real) =
    JuMP.set_time_limit_sec(model, seconds)

# Fix each binary at its combination value, overwriting any previous
# fix in place.
function _fix_combination(combination::AbstractDict)
    for (binary, value) in combination
        JuMP.fix(binary, value ? 1.0 : 0.0; force = true)
    end
    return
end

# Primal values keyed by the NLP's variables, skipping fixed ones
# (the combination binaries and any user-fixed variables).
function _extract_solution(nlp::JuMP.AbstractModel)
    V = JuMP.variable_ref_type(typeof(nlp))
    T = JuMP.value_type(typeof(nlp))
    return Dict{V, T}(v => JuMP.value(v)
        for v in JuMP.all_variables(nlp) if !JuMP.is_fixed(v))
end

# Solve the NLP at a fixed combination. If infeasible, fall
# through to NLPF (a slacked version that always solves) so the master
# still gets a linearization site, not just a no-good cut. NLPF can be
# switched off via `use_nlpf = false`.
function _solve_nlp(
    problem::_LOAProblem,
    combination,
    method::LOA;
    deadline::Float64 = Inf
    )
    nlp = problem.nlp
    _cap_remaining_time(nlp, deadline)
    _fix_combination(combination)
    JuMP.optimize!(nlp, ignore_optimize_hook = true)
    if JuMP.is_solved_and_feasible(nlp)
        return (combination = combination,
            linearization_point = _extract_solution(nlp),
            objective = JuMP.objective_value(nlp), feasible = true)
    end

    # Primary NLP infeasible — try NLPF (unless disabled, in which case
    # the combination contributes only its no-good cut).
    if method.use_nlpf
        nlpf = _solve_nlpf(problem, combination, method; deadline = deadline)
        nlpf === nothing || return nlpf
    end

    V = JuMP.variable_ref_type(typeof(nlp))
    T = JuMP.value_type(typeof(nlp))
    return (combination = combination,
        linearization_point = Dict{V, T}(),
        objective = Inf, feasible = false)
end

################################################################################
#                       NLPF (FEASIBILITY SUBPROBLEM)
################################################################################
# When the primary NLP is infeasible, copy the NLP (fixes included),
# slack every scalar inequality with one nonnegative `u`, minimize `u`,
# and return the primal as the linearization site (equalities/bounds
# exact, inequalities violated by at most `u`).
function _solve_nlpf(
    problem::_LOAProblem,
    combination,
    method::LOA;
    deadline::Float64 = Inf
    )
    copy, ref_map = JuMP.copy_model(problem.nlp)
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

    JuMP.optimize!(copy, ignore_optimize_hook = true)
    # Use the primal only at a genuine feasible point; a solver can report
    # `has_values` with a nonfeasible/NaN primal that would poison the cut.
    JuMP.is_solved_and_feasible(copy) || return nothing

    V = JuMP.variable_ref_type(typeof(problem.nlp))
    T = JuMP.value_type(typeof(problem.nlp))
    linearization_point = Dict{V, T}(v => JuMP.value(ref_map[v])
        for v in JuMP.all_variables(problem.nlp) if !JuMP.is_fixed(v))
    return (combination = combination,
        linearization_point = linearization_point,
        objective = Inf, feasible = false)
end

_nlpf_should_slack(::Type{<:_MOI.LessThan}) = true
_nlpf_should_slack(::Type{<:_MOI.GreaterThan}) = true
_nlpf_should_slack(::Type) = false

_nlpf_slacked_func(func, u, ::_MOI.LessThan) = func - u
_nlpf_slacked_func(func, u, ::_MOI.GreaterThan) = func + u

# Iter-to-iter NLP warm start: seed the next NLP from the last FEASIBLE
# primal (no-op on the first seed and after an NLPF fall-through, whose
# primal is slack-distorted).
function _set_nlp_warm_start(previous)
    previous === nothing && return
    for (variable, value) in previous.linearization_point
        JuMP.set_start_value(variable, value)
    end
    return
end

################################################################################
#                       LOA RESULT CACHE (MOI LAYER)
################################################################################
# `_LOAResultCache` delegates the MOI interface to its inner mock. Only
# `SolverName` (the mock hardcodes "Mock"), the bound attributes (a
# raw `KeyError` from the mock's generic storage when the master
# produced no bound), and `optimize!` are intercepted.
_MOI.is_empty(cache::_LOAResultCache) = _MOI.is_empty(cache.mock)
_MOI.empty!(cache::_LOAResultCache) = _MOI.empty!(cache.mock)
# "Solving" the cache would clear JuMP's dirty flag and serve the stored
# results again, so an edited model would report a stale point as fresh.
function _MOI.optimize!(cache::_LOAResultCache)
    return error("This model holds the results of an LOA solve and " *
        "cannot be re-solved directly (`ignore_optimize_hook = true`). " *
        "Call `optimize!(model, gdp_method = ...)` to rebuild and solve.")
end
_MOI.copy_to(cache::_LOAResultCache, src::_MOI.ModelLike) =
    _MOI.copy_to(cache.mock, src)
_MOI.add_variable(cache::_LOAResultCache) = _MOI.add_variable(cache.mock)
function _MOI.add_constraint(
    cache::_LOAResultCache,
    func::_MOI.AbstractFunction,
    set::_MOI.AbstractSet
    )
    return _MOI.add_constraint(cache.mock, func, set)
end
_MOI.delete(cache::_LOAResultCache, index::_MOI.Index) =
    _MOI.delete(cache.mock, index)
_MOI.is_valid(cache::_LOAResultCache, index::_MOI.Index) =
    _MOI.is_valid(cache.mock, index)
function _MOI.modify(
    cache::_LOAResultCache,
    ci::_MOI.ConstraintIndex,
    change::_MOI.AbstractFunctionModification
    )
    return _MOI.modify(cache.mock, ci, change)
end
function _MOI.modify(
    cache::_LOAResultCache,
    attr::_MOI.ObjectiveFunction,
    change::_MOI.AbstractFunctionModification
    )
    return _MOI.modify(cache.mock, attr, change)
end
function _MOI.supports_constraint(
    cache::_LOAResultCache,
    F::Type{<:_MOI.AbstractFunction},
    S::Type{<:_MOI.AbstractSet}
    )
    return _MOI.supports_constraint(cache.mock, F, S)
end
function _MOI.supports(
    cache::_LOAResultCache,
    attr::Union{_MOI.AbstractModelAttribute,
        _MOI.AbstractOptimizerAttribute}
    )
    return _MOI.supports(cache.mock, attr)
end
function _MOI.supports(
    cache::_LOAResultCache,
    attr::Union{_MOI.AbstractVariableAttribute,
        _MOI.AbstractConstraintAttribute},
    IdxType::Type{<:_MOI.Index}
    )
    return _MOI.supports(cache.mock, attr, IdxType)
end
function _MOI.get(
    cache::_LOAResultCache,
    attr::Union{_MOI.AbstractModelAttribute,
        _MOI.AbstractOptimizerAttribute}
    )
    return _MOI.get(cache.mock, attr)
end
function _MOI.get(
    cache::_LOAResultCache,
    attr::_MOI.AbstractVariableAttribute,
    index::_MOI.VariableIndex
    )
    return _MOI.get(cache.mock, attr, index)
end
function _MOI.get(
    cache::_LOAResultCache,
    attr::_MOI.AbstractConstraintAttribute,
    index::_MOI.ConstraintIndex
    )
    return _MOI.get(cache.mock, attr, index)
end
function _MOI.set(
    cache::_LOAResultCache,
    attr::Union{_MOI.AbstractModelAttribute,
        _MOI.AbstractOptimizerAttribute},
    value
    )
    return _MOI.set(cache.mock, attr, value)
end
function _MOI.set(
    cache::_LOAResultCache,
    attr::_MOI.AbstractVariableAttribute,
    index::_MOI.VariableIndex,
    value
    )
    return _MOI.set(cache.mock, attr, index, value)
end
function _MOI.set(
    cache::_LOAResultCache,
    attr::_MOI.AbstractConstraintAttribute,
    index::_MOI.ConstraintIndex,
    value
    )
    return _MOI.set(cache.mock, attr, index, value)
end
_MOI.get(cache::_LOAResultCache, ::_MOI.SolverName) = cache.name
# A real solver throws `GetAttributeNotAllowed` when no bound exists.
function _MOI.get(
    cache::_LOAResultCache,
    attr::Union{_MOI.ObjectiveBound, _MOI.RelativeGap}
    )
    haskey(cache.mock.model_attributes, attr) ||
        throw(_MOI.GetAttributeNotAllowed(attr,
            "The LOA master did not produce a bound."))
    return _MOI.get(cache.mock, attr)
end
# No duals from an MI(N)LP solve; matches `dual_status = NO_SOLUTION`.
# Both methods are needed: the caching layer swallows the first throw
# and recomputes `DualObjectiveValue` from per-constraint duals.
_MOI.get(cache::_LOAResultCache, attr::_MOI.DualObjectiveValue) =
    throw(_MOI.GetAttributeNotAllowed(attr, "LOA does not provide duals."))
function _MOI.get(
    cache::_LOAResultCache,
    attr::_MOI.ConstraintDual,
    index::_MOI.ConstraintIndex
    )
    return throw(_MOI.GetAttributeNotAllowed(attr,
        "LOA does not provide duals."))
end

################################################################################
#                       LOAD SOLUTION BY INJECTION
################################################################################
# Complete primal for one pool result. The binaries were fixed during
# the loop (absent from the linearization point), so the combination
# supplies them.
function _result_primal(problem::_LOAProblem, result)
    nlp = problem.nlp
    V = JuMP.variable_ref_type(typeof(nlp))
    T = JuMP.value_type(typeof(nlp))
    primal = Dict{V, T}()
    for v in JuMP.all_variables(nlp)
        if haskey(result.linearization_point, v)
            primal[v] = result.linearization_point[v]
        elseif JuMP.is_fixed(v)
            primal[v] = JuMP.fix_value(v)
        else
            primal[v] = zero(T)
        end
    end
    for (binary, active) in result.combination
        primal[binary] = active ? one(T) : zero(T)
    end
    return primal
end

# Attach a `_LOAResultCache` preloaded with the outcome so standard
# JuMP result queries answer with no solve (an empty pool injects only
# the statuses). `is_model_dirty = false` re-opens the result queries
# JuMP guards behind OptimizeNotCalled.
function _inject_solution(
    nlp::JuMP.AbstractModel,
    problem::_LOAProblem,
    outcome::NamedTuple
    )
    T = JuMP.value_type(typeof(nlp))
    primals = [_result_primal(problem, result)
        for result in outcome.results]
    # The mock is the one MOI optimizer that accepts written results.
    # Serve stored objectives rather than re-evaluating at the primal.
    JuMP.set_optimizer(nlp, () -> _LOAResultCache(
        _MOI.Utilities.MockOptimizer(
            _MOI.Utilities.UniversalFallback(_MOI.Utilities.Model{T}()),
            T; eval_objective_value = false),
        outcome.solver_name); add_bridges = false)
    # Attach copies the model in and creates the `optimizer_index` map.
    _MOI.Utilities.attach_optimizer(JuMP.backend(nlp))
    # Result attributes are read-only through the caching layer, so they
    # are written on the inner optimizer directly.
    cache = JuMP.unsafe_backend(nlp)
    _MOI.set(cache, _MOI.TerminationStatus(), outcome.status)
    _MOI.set(cache, _MOI.RawStatusString(), outcome.raw_status)
    _MOI.set(cache, _MOI.SolveTimeSec(), Float64(outcome.solve_time))
    _MOI.set(cache, _MOI.ResultCount(), length(primals))
    if outcome.bound !== nothing
        _MOI.set(cache, _MOI.ObjectiveBound(), outcome.bound)
        isempty(primals) || _MOI.set(cache, _MOI.RelativeGap(),
            abs(outcome.bound - outcome.results[1].objective) /
                abs(outcome.results[1].objective))
    end
    indices = Dict(variable => JuMP.optimizer_index(variable)
        for variable in JuMP.all_variables(nlp))
    for k in eachindex(primals)
        _MOI.set(cache, _MOI.PrimalStatus(k), _MOI.FEASIBLE_POINT)
        _MOI.set(cache, _MOI.ObjectiveValue(k),
            outcome.results[k].objective)
        for (variable, value) in primals[k]
            _MOI.set(cache, _MOI.VariablePrimal(k),
                indices[variable], value)
        end
    end
    # No public API for this; must stay the last mutation.
    nlp.is_model_dirty = false
    return
end

# Finalize a finite GDP. Dispatchable so the InfiniteOpt extension
# overrides it.
function _load_solution(
    model::JuMP.AbstractModel,
    problem::_LOAProblem,
    outcome::NamedTuple,
    method::LOA
    )
    # Drop the solver first: an NLP-only solver fails JuMP's ZeroOne
    # supports check during the binary restore, even detached.
    _MOI.Utilities.drop_optimizer(JuMP.backend(problem.nlp))
    _restore_binaries(problem)
    _inject_solution(problem.nlp, problem, outcome)
    gdp_data(model).displaced_optimizer = outcome.displaced_optimizer
    return
end

################################################################################
#                       COMBO EXTRACTION (master -> NLP)
################################################################################
# Read each binary's master value, rounded to `Bool` (a MILP solver
# returns values within its integer tolerance, never exactly 0/1).
function _extract_combination(problem::_LOAProblem, master::_LOAMaster)
    return Dict(binary => round(Bool, JuMP.value(master.variable_map[binary]))
        for binary in problem.binaries)
end

################################################################################
#                          OA CUT EMISSION
################################################################################
# Whether the disjunct behind `binary_ref` is active in `combination`
# (a complement reference inverts its underlying binary's value).
_disjunct_active(combination::AbstractDict,
    binary_ref::JuMP.AbstractVariableRef) = combination[binary_ref]
_disjunct_active(combination::AbstractDict, binary_ref::JuMP.GenericAffExpr) =
    !combination[_underlying_binary(binary_ref)]

# Emit all OA cuts for one NLP result: the objective cut, a slacked row
# per nonlinear global, and a gated cut per active nonlinear disjunct
# constraint (Big-M / MBM or Hull gating per the inner method).
function _add_oa_cuts(
    problem::_LOAProblem,
    master::_LOAMaster,
    result::NamedTuple,
    method::LOA
    )
    isempty(result.linearization_point) && return
    sense_token = Val(master.objective_sense)
    penalty_sign = _penalty_sign(sense_token)
    linearization = _linearize_at(master.objective,
        result.linearization_point, master.variable_map)
    _add_objective_cut(sense_token, master, linearization, method)
    for (func, set) in problem.global_constraints
        linearization = _linearize_at(func, result.linearization_point,
            master.variable_map)
        _add_global_oa_row(master, linearization, set, method, penalty_sign)
    end
    for (binary_ref, func, set) in problem.disjunct_constraints
        _disjunct_active(result.combination, binary_ref) || continue
        linearization = _linearize_at(func, result.linearization_point,
            master.variable_map)
        _emit_disjunct_oa_cut(method.inner_method, set, master,
            _remap_indicator_to_binary(binary_ref, master.variable_map),
            linearization, method, penalty_sign)
    end
    return
end

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

# Extract RHS from an MOI set.
_set_rhs(s::Union{_MOI.LessThan, _MOI.GreaterThan, _MOI.EqualTo}) =
    _MOI.constant(s)

# The `<= 0` directions of an OA cut for `set`: `lin - rhs` for LessThan,
# `rhs - lin` for GreaterThan, both for EqualTo / Interval. The caller
# adds the gating, slack, and (for Hull) disaggregation per direction.
_oa_cut_terms(set::_MOI.LessThan, lin) = (lin - _set_rhs(set),)
_oa_cut_terms(set::_MOI.GreaterThan, lin) = (_set_rhs(set) - lin,)
_oa_cut_terms(set::_MOI.EqualTo, lin) =
    (lin - _MOI.constant(set), _MOI.constant(set) - lin)
_oa_cut_terms(set::_MOI.Interval, lin) = (lin - set.upper, set.lower - lin)

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
# nonconvex linearization can't make the master infeasible.
function _add_global_oa_row(
    master::_LOAMaster,
    lin,
    set,
    method::LOA,
    penalty_sign::Int
    )
    slack = _penalized_slack(master, method, penalty_sign)
    for term in _oa_cut_terms(set, lin)
        JuMP.@constraint(master.model, term <= slack)
    end
    return
end
