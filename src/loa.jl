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

################################################################################
#                            MAIN LOOP
################################################################################
# LOA entry: build the problem, seed with set-covering combinations,
# iterate the master/NLP loop, and commit the best combination.
function reformulate_model(model::JuMP.AbstractModel, method::LOA)
    _clear_reformulations(model)
    # Hull needs the disaggregation map if its used as inner function
    inner, disaggregation_map = _loa_inner_method(model, method.inner_method)
    reformulate_model(model, inner)

    problem = build_loa_problem(model, method, disaggregation_map)
    # The master copies the NLP while its binaries still carry
    # integrality; the NLP is then relaxed once and each iteration
    # overwrites the binary fixes in place.
    master = _build_loa_master(problem, method)
    _relax_binaries(problem)
    nlp = problem.nlp
    JuMP.set_optimizer(nlp, method.nlp_optimizer)
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

    # Seed the master with an OA cut per set-covering combination, each
    # NLP warm-started from the last feasible primal.
    previous_result = nothing
    for combination in problem.covering_combinations
        time() < loop_deadline || break
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(problem, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.variable_map)
        _add_oa_cuts(problem, master, result, method)
        if result.feasible &&
                _is_better(sense_token, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
    end

    # Master/NLP loop: `alpha_oa` is the bound, the NLP refines the
    # incumbent. Exit on convergence, infeasible master, max_iter, or time.
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
        combination = _extract_combination(problem, master)
        _set_nlp_warm_start(previous_result)
        result = _solve_nlp(problem, combination, method;
            deadline = loop_deadline)
        avoid_combination(master.model, combination, master.variable_map)
        _add_oa_cuts(problem, master, result, method)
        if result.feasible &&
                _is_better(sense_token, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
    end

    best_result === nothing || _commit_combination(best_result)
    if isfinite(method.time_limit)
        # Overall cap: the final committed solve gets the budget left.
        JuMP.set_time_limit_sec(nlp, max(0.0, overall_deadline - time()))
    elseif isfinite(method.iteration_time_limit)
        # Loop-only budget: restore so the final solve isn't crippled.
        _restore_time_limit(nlp, original_time_limit)
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
#                       SET-COVERING INITIALIZATION (simple version from pyomo)
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

# Translate an indicator-level combination onto the underlying binary
# variables, inverting the value for complement-form indicators.
function _binary_combination(model::JuMP.AbstractModel, combination)
    V = JuMP.variable_ref_type(typeof(model))
    binary_map = _indicator_to_binary(model)
    result = Dict{V, Bool}()
    for (indicator, active) in combination
        binary_ref = binary_map[indicator]
        result[_underlying_binary(binary_ref)] =
            _underlying_value(binary_ref, active)
    end
    return result
end

"""
    build_loa_problem(
        model::JuMP.AbstractModel,
        method::LOA,
        [disaggregation_map = nothing]
        )::_LOAProblem

Build the problem the LOA loop operates on from `model` (already
reformulated by the LOA inner method): the NLP subproblem, the binary
variables backing the indicators, the set-covering seed combinations,
the nonlinear disjunct `(binary_ref, function, set)` triples, the
nonlinear global `(function, set)` pairs, and the Hull disaggregation
map. The NLP is `model` itself; the InfiniteOpt extension overloads
this to build the problem from the transcribed backend instead.

## Returns
- `_LOAProblem`: the problem.
"""
function build_loa_problem(
    model::JuMP.AbstractModel,
    method::LOA,
    disaggregation_map = nothing
    )
    V = JuMP.variable_ref_type(typeof(model))
    binary_map = _indicator_to_binary(model)

    binaries = V[_underlying_binary(binary_ref)
        for (_, binary_ref) in binary_map]
    unique!(binaries)
    combinations = [_binary_combination(model, combination)
        for combination in _set_covering_combinations(model)]

    disjunct_constraints = Tuple{Any, Any, Any}[]
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

    global_constraints = Tuple{Any, Any}[]
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

    binary_disaggregations = disaggregation_map === nothing ? nothing :
        Dict((variable, binary_map[indicator]) => disaggregated
            for ((variable, indicator), disaggregated) in disaggregation_map)
    return _LOAProblem(model, binaries, combinations, disjunct_constraints,
        global_constraints, binary_disaggregations)
end

# The inner reformulation method LOA runs, plus the disaggregation map
# to collect (Big-M / MBM need none). For Hull, return a map-carrying
# copy so the reformulation records its `(variable, indicator) ->
# disaggregated variable` map into a fresh LOA-owned `Dict`.
_loa_inner_method(::JuMP.AbstractModel, inner::Union{BigM, MBM}) =
    (inner, nothing)
function _loa_inner_method(model::JuMP.AbstractModel, inner::Hull)
    V = JuMP.variable_ref_type(typeof(model))
    disaggregations = Dict{Tuple{V, LogicalVariableRef{typeof(model)}}, V}()
    return (Hull(inner.value; disaggregation_map = disaggregations),
        disaggregations)
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
_build_disaggregator(::_LOAProblem, ::AbstractDict, ::Union{BigM, MBM}) =
    nothing
function _build_disaggregator(
    problem::_LOAProblem,
    variable_map::AbstractDict,
    inner::Hull
    )
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
# still gets a linearization site, not just a no-good cut.
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

    # Primary NLP infeasible — try NLPF.
    nlpf = _solve_nlpf(problem, combination, method; deadline = deadline)
    nlpf === nothing || return nlpf

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

# Finalize the NLP with the LOA-optimal combination: fix the
# binaries at the committed values (in place; the NLP is already
# permanently relaxed) and warm-start from the linearization point.
# After this, the model is no longer in a state suitable for re-running
# with a different `gdp_method`.
function _commit_combination(result)
    _fix_combination(result.combination)
    for (variable, value) in result.linearization_point
        JuMP.set_start_value(variable, value)
    end
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
