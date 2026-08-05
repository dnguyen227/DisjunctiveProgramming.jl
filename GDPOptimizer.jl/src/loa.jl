################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION
################################################################################
_worst_objective(sense::MOI.OptimizationSense) =
    sense == MOI.MAX_SENSE ? -Inf : Inf
_is_better(sense::MOI.OptimizationSense, new, best) =
    sense == MOI.MAX_SENSE ? new > best : new < best
_gap(sense::MOI.OptimizationSense, best, bound) =
    sense == MOI.MAX_SENSE ? bound - best : best - bound

# one record per NLP solve, for convergence traces
function _log_progress(
    model::Optimizer,
    t_start::Float64,
    best_objective,
    master_bound
    )
    model.silent && return
    bound = master_bound === nothing ? NaN : master_bound
    @info "LOA progress: elapsed=$(time() - t_start) " *
        "incumbent=$best_objective bound=$bound"
    return
end

# one target per (binary, value); linear disjuncts need no cover
# since the master already carries them exactly
function _cover_disjuncts(problem::_Problem)
    seen = Set{Tuple{MOI.VariableIndex, Bool}}()
    cover = _Disjunct[]
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _is_nonlinear_disjunct(disjunct) || continue
        key = (disjunct.binary, disjunct.active_value)
        key in seen && continue
        push!(seen, key)
        push!(cover, disjunct)
    end
    return cover
end

# GDPopt's covering weights: an uncovered disjunct outweighs all
# covered ones
function _cover_objective(
    master::_Master,
    cover::Vector{_Disjunct},
    needs_cover,
    num_covered::Int
    )
    objective = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm{Float64}[], 0.0)
    for i in eachindex(cover)
        weight = Float64(needs_cover[i] ? num_covered + 1 : 1)
        activation = _map_to(master.variable_map, cover[i].activation)
        objective = MOI.Utilities.operate(+, Float64, objective,
            MOI.Utilities.operate(*, Float64, weight, activation))
    end
    return objective
end

# variable starts warm start the first NLP; later iterations use
# the last feasible primal
function _user_start_values(model::Optimizer, problem::_Problem)
    point = Dict{MOI.VariableIndex, Float64}()
    for vi in problem.variables
        start = MOI.get(model.cache, MOI.VariablePrimalStart(), vi)
        start === nothing || (point[vi] = start)
    end
    return isempty(point) ? nothing : (point = point,)
end

function _set_master_objective(master::_Master, sense, objective)
    MOI.set(master.model, MOI.ObjectiveSense(), sense)
    MOI.set(master.model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        objective)
    return
end

################################################################################
#                              MAIN LOOP
################################################################################
function MOI.optimize!(model::Optimizer)
    t_start = time()
    _reset_results(model)
    problem = _build_problem(model)
    master = _build_master(model, problem)
    subproblem = _build_subproblem(model, problem)
    linearizer = _Linearizer()
    sense = problem.sense
    overall_deadline = t_start + Float64(_option(model, "time_limit"))
    loop_deadline = min(overall_deadline,
        t_start + Float64(_option(model, "iteration_time_limit")))

    best_objective = _worst_objective(sense)
    best_result = nothing
    previous_result = _user_start_values(model, problem)
    master_bound = nothing
    master_status = nothing
    converged = false

    # Shared iteration tail: no-good cut, OA cuts, incumbent update.
    process_result = result -> begin
        _avoid_combination(master, result.combination)
        _add_oa_cuts(model, problem, master, linearizer, result)
        if result.feasible &&
                _is_better(sense, result.objective, best_objective)
            best_objective = result.objective
            best_result = result
        end
        result.feasible && (previous_result = result)
        _log_progress(model, t_start, best_objective, master_bound)
        return
    end
    warm_start = () ->
        previous_result === nothing ? nothing : previous_result.point

    # set covering: reuse the master with a coverage objective so
    # every nonlinear disjunct gets visited once
    cover = _cover_disjuncts(problem)
    needs_cover = trues(length(cover))
    num_covered = 0
    for iteration in 1:_option(model, "set_cover_max_iter")
        (iteration == 1 || any(needs_cover)) || break
        time() < loop_deadline || break
        _set_master_objective(master, MOI.MAX_SENSE,
            _cover_objective(master, cover, needs_cover, num_covered))
        _cap_remaining_time(master.model, loop_deadline)
        MOI.optimize!(master.model)
        solved = _solved_and_feasible(master.model)
        # capture the status before the objective restore invalidates it
        status = MOI.get(master.model, MOI.TerminationStatus())
        combination = solved ? _extract_combination(problem, master) : nothing
        _set_master_objective(master, master.sense, master.oa_objective)
        if !solved
            master_status = status
            break
        end
        result = _solve_nlp(model, problem, subproblem, combination,
            warm_start(); deadline = loop_deadline)
        process_result(result)
        # covered only once active in a feasible NLP; infeasible
        # combinations just leave their no-good cut
        if result.feasible
            for i in eachindex(cover)
                needs_cover[i] || continue
                _disjunct_active(result.combination, cover[i]) &&
                    (needs_cover[i] = false)
            end
            num_covered = count(!, needs_cover)
        end
    end

    # main loop: alpha_oa gives the bound, the NLP the incumbent
    if master_status === nothing
        for _ in 1:_option(model, "max_iter")
            time() < loop_deadline || break
            _cap_remaining_time(master.model, loop_deadline)
            MOI.optimize!(master.model)
            if !_solved_and_feasible(master.model)
                master_status = MOI.get(master.model, MOI.TerminationStatus())
                break
            end
            master_bound = MOI.get(master.model, MOI.VariablePrimal(),
                master.alpha_oa)
            if best_result !== nothing
                gap = _gap(sense, best_objective, master_bound)
                total_slack = abs(MOI.get(master.model,
                    MOI.ObjectiveValue()) - master_bound) /
                    Float64(_option(model, "oa_penalty"))
                tol = Float64(_option(model, "convergence_tol")) *
                    max(abs(best_objective), 1.0)
                if gap <= tol &&
                        total_slack <= Float64(_option(model, "slack_tol"))
                    converged = true
                    break
                end
            end
            combination = _extract_combination(problem, master)
            result = _solve_nlp(model, problem, subproblem, combination,
                warm_start(); deadline = loop_deadline)
            process_result(result)
        end
    end

    _store_results(model, sense, best_objective, best_result, master_bound,
        master_status, converged, loop_deadline)
    model.solve_time = time() - t_start
    return
end

################################################################################
#                          RESULT SYNTHESIS
################################################################################
# the OA bound is valid only for convex problems, so report
# LOCALLY_SOLVED, never OPTIMAL
function _store_results(
    model::Optimizer,
    sense::MOI.OptimizationSense,
    best_objective::Float64,
    best_result,
    master_bound,
    master_status,
    converged::Bool,
    loop_deadline::Float64
    )
    timed_out = time() >= loop_deadline
    if best_result === nothing
        model.primal_status = MOI.NO_SOLUTION
        model.objective_value = NaN
        model.raw_status = "No feasible incumbent found."
        if master_status == MOI.INFEASIBLE
            model.termination_status = MOI.INFEASIBLE
            model.raw_status = "No feasible incumbent: the master " *
                "problem is infeasible."
        elseif timed_out
            model.termination_status = MOI.TIME_LIMIT
        elseif master_status === nothing
            model.termination_status = MOI.ITERATION_LIMIT
        else
            model.termination_status = MOI.OTHER_LIMIT
            model.raw_status = "No feasible incumbent: the master " *
                "solve finished with status $master_status."
        end
        return
    end
    model.primal_status = MOI.FEASIBLE_POINT
    model.incumbent = best_result.point
    model.objective_value = best_objective
    model.objective_bound = master_bound === nothing ? nothing :
        Float64(master_bound)
    if converged
        model.termination_status = MOI.LOCALLY_SOLVED
    elseif timed_out
        model.termination_status = MOI.TIME_LIMIT
    elseif master_status == MOI.INFEASIBLE
        # all combinations visited; the incumbent is best over all
        # of them, but the OA bound is gone
        model.termination_status = MOI.LOCALLY_SOLVED
    elseif master_status !== nothing
        model.termination_status = MOI.OTHER_LIMIT
    else
        model.termination_status = MOI.ITERATION_LIMIT
    end
    label = converged ? "converged" : (master_status == MOI.INFEASIBLE ?
        "combinations exhausted" : "limit hit")
    if master_bound === nothing
        model.raw_status = "LOA finished [$label]: incumbent " *
            "$best_objective (master produced no bound)."
    else
        gap = _gap(sense, best_objective, master_bound)
        relative = abs(best_objective) > 1e-10 ?
            gap / abs(best_objective) : gap
        model.relative_gap = relative
        model.raw_status = "LOA finished [$label]: incumbent " *
            "$best_objective, master bound $master_bound, gap $gap " *
            "(relative $relative)."
    end
    return
end
