################################################################################
#                         MASTER CONSTRUCTION
################################################################################
# `alpha_oa` carries the objective; `oa_objective` also tracks the
# slack penalties so set covering can swap objectives out and back
mutable struct _Master
    model::MOI.ModelLike
    variable_map::Dict{MOI.VariableIndex, MOI.VariableIndex}
    sense::MOI.OptimizationSense
    objective::MOI.AbstractScalarFunction
    alpha_oa::MOI.VariableIndex
    oa_objective::MOI.ScalarAffineFunction{Float64}
end

function _instantiate(factory, name::String)
    factory === nothing &&
        error("GDPOptimizer requires the `$name` optimizer factory.")
    solver = MOI.instantiate(factory;
        with_cache_type = Float64, with_bridge_type = Float64)
    MOI.set(solver, MOI.Silent(), true)
    return solver
end

_map_to(variable_map::AbstractDict, func) =
    MOI.Utilities.map_indices(vi -> variable_map[vi], func)

function _solved_and_feasible(solver::MOI.ModelLike)
    status = MOI.get(solver, MOI.TerminationStatus())
    return status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) &&
        MOI.get(solver, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
end

function _cap_remaining_time(solver::MOI.ModelLike, deadline::Float64)
    isfinite(deadline) || return
    MOI.set(solver, MOI.TimeLimitSec(), max(0.0, deadline - time()))
    return
end

function _build_master(model::Optimizer, problem::_Problem)
    mip = _instantiate(model.mip_solver, "mip_solver")
    variable_map = Dict{MOI.VariableIndex, MOI.VariableIndex}(
        vi => MOI.add_variable(mip) for vi in problem.variables)
    for ci in problem.variable_cis
        vi = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        MOI.add_constraint(mip, variable_map[vi],
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    for ci in problem.linear_cis
        func = MOI.get(model.cache, MOI.ConstraintFunction(), ci)
        MOI.add_constraint(mip, _map_to(variable_map, func),
            MOI.get(model.cache, MOI.ConstraintSet(), ci))
    end
    # nonlinear-typed rows that demoted to affine stay in the master
    for (func, set) in problem.nonlinear_rows
        _is_linear(func) || continue
        MOI.add_constraint(mip, _map_to(variable_map, _to_affine(func)), set)
    end
    for disjunction in problem.disjunctions
        _add_exactly_one(mip, variable_map, disjunction)
        for disjunct in disjunction.disjuncts
            _add_gated_rows(mip, variable_map, disjunct, model.options)
        end
    end
    alpha_oa = MOI.add_variable(mip)
    sense = problem.sense
    oa_objective = MOI.ScalarAffineFunction(
        [MOI.ScalarAffineTerm(1.0, alpha_oa)], 0.0)
    MOI.set(mip, MOI.ObjectiveSense(), sense)
    MOI.set(mip,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        oa_objective)
    return _Master(mip, variable_map, sense, problem.objective, alpha_oa,
        oa_objective)
end

# Indicators sum to the activation (1 top-level, parent indicator
# nested); a complement pair normalizes to a trivial `1 == 1` row
function _add_exactly_one(
    mip::MOI.ModelLike,
    variable_map::AbstractDict,
    disjunction::_Disjunction
    )
    total = MOI.Utilities.operate(+, Float64,
        (_map_to(variable_map, disjunct.activation)
            for disjunct in disjunction.disjuncts)...)
    body = MOI.Utilities.operate(-, Float64, total,
        _map_to(variable_map, disjunction.activation))
    MOI.Utilities.normalize_and_add_constraint(mip, body, MOI.EqualTo(0.0))
    return
end

# split Interval rows; the indicator bridge takes one-sided sets only
_indicator_sets(set::MOI.Interval{Float64}) =
    (MOI.LessThan(set.upper), MOI.GreaterThan(set.lower))
_indicator_sets(set::MOI.AbstractScalarSet) = (set,)

# Gate each linear row with an indicator constraint or big-M per
# `master_gating`; nonlinear rows enter the master only as OA cuts
function _add_gated_rows(
    mip::MOI.ModelLike,
    variable_map::AbstractDict,
    disjunct::_Disjunct,
    options::Dict{String, Any}
    )
    gating = options["master_gating"]
    gating in ("indicator", "bigm") ||
        error("Unknown `master_gating` value `$gating`.")
    activate = disjunct.active_value ? MOI.ACTIVATE_ON_ONE :
        MOI.ACTIVATE_ON_ZERO
    binary = variable_map[disjunct.binary]
    activation = _map_to(variable_map, disjunct.activation)
    M = Float64(options["M_value"])
    # M * (1 - activation) moved left, as in the disjunct OA cuts
    gate = MOI.Utilities.operate(-, Float64,
        MOI.Utilities.operate(*, Float64, M, activation), M)
    for (func, set) in zip(disjunct.functions, disjunct.sets)
        _is_linear(func) || continue
        row = _map_to(variable_map, _to_affine(func))
        if gating == "bigm"
            for term in _oa_cut_terms(set, row)
                body = MOI.Utilities.operate(+, Float64, term, gate)
                MOI.Utilities.normalize_and_add_constraint(mip, body,
                    MOI.LessThan(0.0))
            end
        else
            for inner in _indicator_sets(set)
                gated = MOI.Utilities.operate(vcat, Float64, binary, row)
                MOI.add_constraint(mip, gated,
                    MOI.Indicator{activate}(inner))
            end
        end
    end
    return
end

# bounded penalized slack so an invalid cut cannot blow up the master
function _add_penalized_slack(
    master::_Master,
    options::Dict{String, Any},
    penalty_sign::Int
    )
    slack = MOI.add_variable(master.model)
    MOI.add_constraint(master.model, slack, MOI.GreaterThan(0.0))
    MOI.add_constraint(master.model, slack,
        MOI.LessThan(Float64(options["max_slack"])))
    penalty = penalty_sign * Float64(options["oa_penalty"])
    master.oa_objective = MOI.Utilities.operate(+, Float64,
        master.oa_objective, MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(penalty, slack)], 0.0))
    MOI.set(master.model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        master.oa_objective)
    return slack
end

# round to Bool; MILP values are only within integer tolerance
function _extract_combination(problem::_Problem, master::_Master)
    return Dict{MOI.VariableIndex, Bool}(
        binary => round(Bool, MOI.get(master.model, MOI.VariablePrimal(),
            master.variable_map[binary]))
        for binary in problem.binaries)
end

# No-good cut: active `1 - z` plus inactive `z` terms must reach 1
function _avoid_combination(master::_Master, combination::AbstractDict)
    terms = MOI.ScalarAffineTerm{Float64}[]
    constant = 0.0
    for (binary, value) in combination
        mapped = master.variable_map[binary]
        if value
            push!(terms, MOI.ScalarAffineTerm(-1.0, mapped))
            constant += 1.0
        else
            push!(terms, MOI.ScalarAffineTerm(1.0, mapped))
        end
    end
    cut = MOI.ScalarAffineFunction(terms, constant)
    MOI.Utilities.normalize_and_add_constraint(master.model, cut,
        MOI.GreaterThan(1.0))
    return
end
