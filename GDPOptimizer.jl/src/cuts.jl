################################################################################
#                           LINEARIZATION
################################################################################
# First-order Taylor expansions of the nonlinear rows, sharing one
# `MOI.Nonlinear` reverse-mode evaluator per row across iterations
# (only the evaluation point changes).
struct _Linearizer
    evaluators::Dict{UInt64,
        Tuple{MOI.Nonlinear.Evaluator, Vector{MOI.VariableIndex}}}
end

_Linearizer() = _Linearizer(Dict{UInt64,
    Tuple{MOI.Nonlinear.Evaluator, Vector{MOI.VariableIndex}}}())

function _append_variables(
    variables::Vector{MOI.VariableIndex},
    func::MOI.VariableIndex
    )
    return push!(variables, func)
end
function _append_variables(
    variables::Vector{MOI.VariableIndex},
    func::MOI.ScalarAffineFunction{Float64}
    )
    return append!(variables, term.variable for term in func.terms)
end
function _append_variables(
    variables::Vector{MOI.VariableIndex},
    func::MOI.ScalarQuadraticFunction{Float64}
    )
    append!(variables, term.variable for term in func.affine_terms)
    for term in func.quadratic_terms
        push!(variables, term.variable_1, term.variable_2)
    end
    return variables
end
function _append_variables(
    variables::Vector{MOI.VariableIndex},
    func::MOI.ScalarNonlinearFunction
    )
    for arg in func.args
        arg isa Real || _append_variables(variables, arg)
    end
    return variables
end

function _evaluator(linearizer::_Linearizer, func)
    return get!(linearizer.evaluators, objectid(func)) do
        variables = _append_variables(MOI.VariableIndex[], func)
        unique!(variables)
        nonlinear = MOI.Nonlinear.Model()
        MOI.Nonlinear.set_objective(nonlinear, func)
        evaluator = MOI.Nonlinear.Evaluator(nonlinear,
            MOI.Nonlinear.SparseReverseMode(), variables)
        MOI.initialize(evaluator, [:Grad])
        return (evaluator, variables)
    end
end

# Exact for affine functions, first-order Taylor at `point` otherwise.
# The result stays in cache space; callers remap when adding cuts.
_linearize(::_Linearizer, func::MOI.ScalarAffineFunction{Float64}, point) = func
_linearize(::_Linearizer, func::MOI.VariableIndex, point) = _to_affine(func)
function _linearize(
    linearizer::_Linearizer,
    func::Union{MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction},
    point::AbstractDict
    )
    evaluator, variables = _evaluator(linearizer, func)
    x = [point[vi] for vi in variables]
    value = MOI.eval_objective(evaluator, x)
    gradient = zeros(length(x))
    MOI.eval_objective_gradient(evaluator, gradient, x)
    constant = value - sum(gradient[i] * x[i] for i in eachindex(x);
        init = 0.0)
    terms = [MOI.ScalarAffineTerm(gradient[i], variables[i])
        for i in eachindex(variables) if gradient[i] != 0.0]
    return MOI.ScalarAffineFunction(terms, constant)
end

################################################################################
#                          OA CUT EMISSION
################################################################################
_penalty_sign(sense::MOI.OptimizationSense) = sense == MOI.MAX_SENSE ? -1 : 1

# The `<= 0` directions of an OA cut for `set`: `lin - rhs` for
# LessThan, `rhs - lin` for GreaterThan, both for EqualTo / Interval.
_oa_cut_terms(set::MOI.LessThan{Float64}, lin) =
    (MOI.Utilities.operate(-, Float64, lin, set.upper),)
_oa_cut_terms(set::MOI.GreaterThan{Float64}, lin) =
    (MOI.Utilities.operate(-, Float64, set.lower, lin),)
_oa_cut_terms(set::MOI.EqualTo{Float64}, lin) =
    (MOI.Utilities.operate(-, Float64, lin, set.value),
        MOI.Utilities.operate(-, Float64, set.value, lin))
_oa_cut_terms(set::MOI.Interval{Float64}, lin) =
    (MOI.Utilities.operate(-, Float64, lin, set.upper),
        MOI.Utilities.operate(-, Float64, set.lower, lin))

# Emit all OA cuts for one NLP result: the objective cut, a slacked row
# per nonlinear global, and a gated cut per active nonlinear disjunct
# row. Every slack keeps a nonconvex linearization from making the
# master infeasible.
function _add_oa_cuts(
    model::Optimizer,
    problem::_Problem,
    master::_Master,
    linearizer::_Linearizer,
    result::NamedTuple
    )
    result.point === nothing && return
    sign = _penalty_sign(master.sense)
    _add_objective_cut(model, master, linearizer, result.point, sign)
    for (func, set) in problem.nonlinear_rows
        _is_linear(func) && continue
        lin = _master_linearization(master, linearizer, func, result.point)
        _add_global_oa_row(model, master, lin, set, sign)
    end
    for disjunction in problem.disjunctions, disjunct in disjunction.disjuncts
        _disjunct_active(result.combination, disjunct) || continue
        for (func, set) in zip(disjunct.functions, disjunct.sets)
            _is_linear(func) && continue
            lin = _master_linearization(master, linearizer, func, result.point)
            _add_disjunct_oa_cut(model, master, disjunct, lin, set, sign)
        end
    end
    return
end

function _master_linearization(
    master::_Master,
    linearizer::_Linearizer,
    func,
    point
    )
    return _map_to(master.variable_map, _linearize(linearizer, func, point))
end

# Slacked objective cut. MIN: `lin <= alpha_oa + slack`; MAX symmetric.
function _add_objective_cut(
    model::Optimizer,
    master::_Master,
    linearizer::_Linearizer,
    point,
    sign::Int
    )
    lin = _master_linearization(master, linearizer, master.objective, point)
    slack = _add_penalized_slack(master, model.options, sign)
    alpha = _to_affine(master.alpha_oa)
    if master.sense == MOI.MAX_SENSE
        body = MOI.Utilities.operate(-, Float64,
            MOI.Utilities.operate(-, Float64, alpha, lin), _to_affine(slack))
    else
        body = MOI.Utilities.operate(-, Float64,
            MOI.Utilities.operate(-, Float64, lin, alpha), _to_affine(slack))
    end
    MOI.Utilities.normalize_and_add_constraint(master.model, body,
        MOI.LessThan(0.0))
    return
end

# Slacked global OA row(s): each direction `term - slack <= 0`.
function _add_global_oa_row(
    model::Optimizer,
    master::_Master,
    lin::MOI.ScalarAffineFunction{Float64},
    set::MOI.AbstractScalarSet,
    sign::Int
    )
    slack = _add_penalized_slack(master, model.options, sign)
    for term in _oa_cut_terms(set, lin)
        body = MOI.Utilities.operate(-, Float64, term, _to_affine(slack))
        MOI.Utilities.normalize_and_add_constraint(master.model, body,
            MOI.LessThan(0.0))
    end
    return
end

# Gated disjunct cut: each direction `term - slack <= M * (1 - z)` for
# an activation `z` (its complement gates on `M * z`).
function _add_disjunct_oa_cut(
    model::Optimizer,
    master::_Master,
    disjunct::_Disjunct,
    lin::MOI.ScalarAffineFunction{Float64},
    set::MOI.AbstractScalarSet,
    sign::Int
    )
    M = Float64(_option(model, "M_value"))
    activation = _map_to(master.variable_map, disjunct.activation)
    # M * (1 - activation) moved left: term - slack + M * activation - M
    gate = MOI.Utilities.operate(-, Float64,
        MOI.Utilities.operate(*, Float64, M, activation), M)
    slack = _add_penalized_slack(master, model.options, sign)
    for term in _oa_cut_terms(set, lin)
        body = MOI.Utilities.operate(+, Float64,
            MOI.Utilities.operate(-, Float64, term, _to_affine(slack)), gate)
        MOI.Utilities.normalize_and_add_constraint(master.model, body,
            MOI.LessThan(0.0))
    end
    return
end
