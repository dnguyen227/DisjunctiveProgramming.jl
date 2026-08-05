################################################################################
#                          NLP SUBPROBLEM
################################################################################
# Built once; each iteration overwrites the binary fixes in place
# and swaps the active disjuncts' rows. No big-M anywhere.
struct _Subproblem
    model::MOI.ModelLike
    variable_map::Dict{MOI.VariableIndex, MOI.VariableIndex}
    fixes::Dict{MOI.VariableIndex,
        MOI.ConstraintIndex{MOI.VariableIndex, MOI.EqualTo{Float64}}}
    rows::Vector{MOI.ConstraintIndex}
end

function _build_subproblem(model::Optimizer, problem::_Problem)
    nlp = _instantiate(model.nlp_solver, "nlp_solver")
    variable_map = Dict{MOI.VariableIndex, MOI.VariableIndex}(
        vi => MOI.add_variable(nlp) for vi in problem.variables)
    indicators = Set(problem.binaries)
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        vi in indicators && continue
        MOI.add_constraint(nlp, variable_map[vi],
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    fixes = Dict(binary => MOI.add_constraint(nlp, variable_map[binary],
        MOI.EqualTo(0.0)) for binary in problem.binaries)
    for ci in problem.linear_cis
        func = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        MOI.add_constraint(nlp, _map_to(variable_map, func),
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    for (func, set) in problem.nonlinear_rows
        MOI.add_constraint(nlp, _map_to(variable_map, func), set)
    end
    MOI.set(nlp, MOI.ObjectiveSense(), problem.sense)
    objective = _map_to(variable_map, problem.objective)
    MOI.set(nlp, MOI.ObjectiveFunction{typeof(objective)}(), objective)
    return _Subproblem(nlp, variable_map, fixes, MOI.ConstraintIndex[])
end

function _set_warm_start(nlp::MOI.ModelLike, variable_map::AbstractDict, point)
    point === nothing && return
    for (vi, value) in point
        MOI.set(nlp, MOI.VariablePrimalStart(), variable_map[vi], value)
    end
    return
end

function _extract_point(
    nlp::MOI.ModelLike,
    problem::_Problem,
    variable_map::AbstractDict
    )
    return Dict{MOI.VariableIndex, Float64}(
        vi => MOI.get(nlp, MOI.VariablePrimal(), variable_map[vi])
        for vi in problem.variables)
end

# Solve the NLP at a fixed combination: overwrite the binary fixes,
# swap the active disjuncts' rows, and optimize. If infeasible, fall
# through to NLPF (a slacked version that always solves) so the master
# still gets a linearization site, not just a no-good cut.
function _solve_nlp(
    model::Optimizer,
    problem::_Problem,
    sub::_Subproblem,
    combination::AbstractDict,
    warm_start;
    deadline::Float64 = Inf
    )
    for (binary, value) in combination
        MOI.set(sub.model, MOI.ConstraintSet(), sub.fixes[binary],
            MOI.EqualTo(value ? 1.0 : 0.0))
    end
    for ci in sub.rows
        MOI.delete(sub.model, ci)
    end
    empty!(sub.rows)
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _disjunct_active(combination, disjunct) || continue
        for (func, set) in zip(disjunct.functions, disjunct.sets)
            push!(sub.rows, MOI.add_constraint(sub.model,
                _map_to(sub.variable_map, func), set))
        end
    end
    _set_warm_start(sub.model, sub.variable_map, warm_start)
    _cap_remaining_time(sub.model, deadline)
    MOI.optimize!(sub.model)
    if _solved_and_feasible(sub.model)
        return (combination = combination,
            point = _extract_point(sub.model, problem, sub.variable_map),
            objective = MOI.get(sub.model, MOI.ObjectiveValue()),
            feasible = true)
    end
    if Bool(_option(model, "use_nlpf"))
        result = _solve_nlpf(model, problem, combination, warm_start;
            deadline = deadline)
        result === nothing || return result
    end
    return (combination = combination,
        point = nothing, objective = Inf, feasible = false)
end

################################################################################
#                       NLPF (FEASIBILITY SUBPROBLEM)
################################################################################
_nlpf_slacked(func, u, ::MOI.LessThan{Float64}) =
    MOI.Utilities.operate(-, Float64, func, u)
_nlpf_slacked(func, u, ::MOI.GreaterThan{Float64}) =
    MOI.Utilities.operate(+, Float64, func, u)
_nlpf_slacked(func, u, ::MOI.AbstractScalarSet) = nothing

# The slacked feasibility NLP: one nonnegative `u` relaxes every scalar
# inequality row (bounds and equalities stay exact) and is minimized.
# Its solution is a linearization site for an infeasible combination.
function _solve_nlpf(
    model::Optimizer,
    problem::_Problem,
    combination::AbstractDict,
    warm_start;
    deadline::Float64 = Inf
    )
    nlp = _instantiate(model.nlp_solver, "nlp_solver")
    variable_map = Dict{MOI.VariableIndex, MOI.VariableIndex}(
        vi => MOI.add_variable(nlp) for vi in problem.variables)
    u = MOI.add_variable(nlp)
    MOI.add_constraint(nlp, u, MOI.GreaterThan(0.0))
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        haskey(combination, vi) && continue
        MOI.add_constraint(nlp, variable_map[vi],
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    for (binary, value) in combination
        MOI.add_constraint(nlp, variable_map[binary],
            MOI.EqualTo(value ? 1.0 : 0.0))
    end
    rows = Tuple{MOI.AbstractScalarFunction, MOI.AbstractScalarSet}[]
    for ci in problem.linear_cis
        push!(rows, (MOI.get(model.cache, MOI.ConstraintFunction(), ci),
            MOI.get(model.cache, MOI.ConstraintSet(), ci)))
    end
    append!(rows, problem.nonlinear_rows)
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _disjunct_active(combination, disjunct) || continue
        append!(rows, zip(disjunct.functions, disjunct.sets))
    end
    for (func, set) in rows
        mapped = _map_to(variable_map, func)
        slacked = _nlpf_slacked(mapped, u, set)
        if slacked === nothing
            MOI.add_constraint(nlp, mapped, set)
        else
            MOI.add_constraint(nlp, slacked, set)
        end
    end
    MOI.set(nlp, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(nlp, MOI.ObjectiveFunction{MOI.VariableIndex}(), u)
    _set_warm_start(nlp, variable_map, warm_start)
    _cap_remaining_time(nlp, deadline)
    MOI.optimize!(nlp)
    # Use the primal only at a genuine feasible point; a solver can
    # report values at a nonfeasible/NaN primal that poisons the cut.
    _solved_and_feasible(nlp) || return nothing
    return (combination = combination,
        point = _extract_point(nlp, problem, variable_map),
        objective = Inf, feasible = false)
end
