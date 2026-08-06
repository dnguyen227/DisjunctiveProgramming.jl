################################################################################
#                                OPTIMIZER
################################################################################
const _Cache = MOI.Utilities.UniversalFallback{MOI.Utilities.Model{Float64}}

# Raw options, mirroring DisjunctiveProgramming.jl's LOA defaults.
const _DEFAULT_OPTIONS = Dict{String, Any}(
    "max_iter" => 10,
    "set_cover_max_iter" => 8,
    "M_value" => 1e9,
    "master_gating" => "indicator",
    "max_slack" => 1e3,
    "oa_penalty" => 1e3,
    "use_nlpf" => true,
    "convergence_tol" => 1e-6,
    "slack_tol" => 1e-4,
    "iteration_time_limit" => Inf,
    "time_limit" => 3600.0,
)

"""
    Optimizer(; nlp_solver, mip_solver = nlp_solver, kwargs...)

Logic-based outer approximation solver for models containing
[`DisjunctionSet`](@ref) constraints. `nlp_solver` and `mip_solver`
are optimizer factories as accepted by `MOI.instantiate`. The
remaining keyword arguments set raw options (also reachable through
`MOI.RawOptimizerAttribute`):

- `max_iter = 10`: master/NLP iterations after the set-covering seed.
- `set_cover_max_iter = 8`: set-covering initialization iterations.
- `M_value = 1e9`: big-M gating the disjunct OA cuts in the master.
- `master_gating = "indicator"`: how linear disjunct rows enter the
  master, `"indicator"` (constraint gated by the binary) or `"bigm"`
  (rows relaxed by `M_value * (1 - z)`, tighter for solvers that
  cannot strengthen indicators, e.g. with presolve disabled).
- `max_slack = 1e3`: upper bound of each OA cut slack.
- `oa_penalty = 1e3`: objective penalty per unit of cut slack.
- `use_nlpf = true`: solve a slacked feasibility NLP when the primary
  NLP is infeasible, so its point still seeds OA cuts.
- `convergence_tol = 1e-6`: relative incumbent/bound gap tolerance.
- `slack_tol = 1e-4`: total cut slack tolerance for convergence.
- `iteration_time_limit = Inf`: seconds allotted to the LOA loop.
- `time_limit = 3600.0`: overall seconds budget.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    nlp_solver::Any
    mip_solver::Any
    cache::_Cache
    options::Dict{String, Any}
    silent::Bool
    # results (filled by MOI.optimize!)
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    incumbent::Dict{MOI.VariableIndex, Float64}
    objective_value::Float64
    objective_bound::Union{Nothing, Float64}
    relative_gap::Float64
    raw_status::String
    solve_time::Float64
end

function Optimizer(;
    nlp_solver = nothing,
    mip_solver = nlp_solver,
    kwargs...
    )
    options = copy(_DEFAULT_OPTIONS)
    for (key, value) in kwargs
        haskey(options, string(key)) || throw(ArgumentError(
            "Unknown option `$key`."))
        options[string(key)] = value
    end
    return Optimizer(nlp_solver, mip_solver,
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        options, false, MOI.OPTIMIZE_NOT_CALLED, MOI.NO_SOLUTION,
        Dict{MOI.VariableIndex, Float64}(), NaN, nothing, NaN, "", NaN)
end

_option(model::Optimizer, name::String) = model.options[name]

MOI.get(::Optimizer, ::MOI.SolverName) = "GDPOptimizer"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.1.0"

MOI.is_empty(model::Optimizer) = MOI.is_empty(model.cache)

function MOI.empty!(model::Optimizer)
    MOI.empty!(model.cache)
    _reset_results(model)
    return
end

# The non-cache part of MOI.empty!, also run at the top of every solve
# so a re-optimize cannot leak the previous solve's bound, gap, or
# point.
function _reset_results(model::Optimizer)
    model.termination_status = MOI.OPTIMIZE_NOT_CALLED
    model.primal_status = MOI.NO_SOLUTION
    empty!(model.incumbent)
    model.objective_value = NaN
    model.objective_bound = nothing
    model.relative_gap = NaN
    model.raw_status = ""
    model.solve_time = NaN
    return
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

################################################################################
#                        MODEL-BUILDING FORWARDING
################################################################################
# Everything model-building lands in the cache; the solve partitions it
# in MOI.optimize!, so incremental additions need no invalidation.
const _Index = Union{MOI.VariableIndex, MOI.ConstraintIndex}

MOI.add_variable(model::Optimizer) = MOI.add_variable(model.cache)

MOI.add_variables(model::Optimizer, n::Int) = MOI.add_variables(model.cache, n)

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.AbstractScalarSet
    )
    return MOI.add_constrained_variable(model.cache, set)
end

function MOI.add_constrained_variables(
    model::Optimizer,
    set::MOI.AbstractVectorSet
    )
    return MOI.add_constrained_variables(model.cache, set)
end

function MOI.add_constraint(
    model::Optimizer,
    func::MOI.AbstractFunction,
    set::MOI.AbstractSet
    )
    return MOI.add_constraint(model.cache, func, set)
end

# Honest constraint support (the cache would claim everything, which
# disables the bridges that rewrite e.g. vector cones into supported
# scalar rows): only what `_build_problem` actually partitions.
const _ScalarFunction = Union{MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64}, MOI.ScalarNonlinearFunction}
const _VectorFunction = Union{MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    MOI.VectorQuadraticFunction{Float64}, MOI.VectorNonlinearFunction}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:MOI.AbstractFunction},
    ::Type{<:MOI.AbstractSet}
    )
    return false
end

# Non-indicator discrete variables pass through to the subproblems
# with their integrality intact, so the nlp_solver must handle them.
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VariableIndex},
    ::Type{<:Union{_SupportedInnerSet, MOI.ZeroOne, MOI.Integer}}
    )
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_ScalarFunction},
    ::Type{<:_SupportedInnerSet}
    )
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:_VectorFunction},
    ::Type{DisjunctionSet}
    )
    return true
end

# A `DisjunctionSet` never constrains variables on creation: without
# this, `copy_to` turns a `VectorOfVariables` disjunction (which may
# repeat a variable across rows) into `add_constrained_variables`.
function MOI.supports_add_constrained_variables(
    ::Optimizer,
    ::Type{DisjunctionSet}
    )
    return false
end

MOI.is_valid(model::Optimizer, index::_Index) =
    MOI.is_valid(model.cache, index)

MOI.delete(model::Optimizer, index::_Index) = MOI.delete(model.cache, index)

function MOI.get(model::Optimizer, T::Type{<:_Index}, name::String)
    return MOI.get(model.cache, T, name)
end

# Non-result attributes forward to the cache; the result attributes
# below override these generic methods, and any solve-set attribute
# without an override (duals, ConstraintPrimal) is refused rather than
# forwarded to the cache, which never holds results.
# Forward to the wrapped inner model (honest), not the UniversalFallback,
# which claims support for every attribute and so never rejects an
# unsupported one on copy_to.
MOI.supports(model::Optimizer, attr::MOI.AbstractModelAttribute) =
    MOI.supports(model.cache.model, attr)

function MOI.set(model::Optimizer, attr::MOI.AbstractModelAttribute, value)
    MOI.supports(model, attr) || throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, value)
end

function MOI.get(model::Optimizer, attr::MOI.AbstractModelAttribute)
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex}
    )
    return MOI.supports(model.cache.model, attr, MOI.VariableIndex)
end

# The LOA loop consumes warm starts, so accept them even though the
# honest inner model does not store them (the fallback does).
function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex}
    )
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value
    )
    MOI.supports(model, attr, MOI.VariableIndex) ||
        throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, vi, value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex
    )
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr, vi)
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    C::Type{<:MOI.ConstraintIndex}
    )
    return MOI.supports(model.cache.model, attr, C)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex,
    value
    )
    MOI.supports(model, attr, typeof(ci)) ||
        throw(MOI.UnsupportedAttribute(attr))
    return MOI.set(model.cache, attr, ci, value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex
    )
    MOI.is_set_by_optimize(attr) && throw(MOI.GetAttributeNotAllowed(attr))
    return MOI.get(model.cache, attr, ci)
end

################################################################################
#                          OPTIMIZER ATTRIBUTES
################################################################################
MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value::Bool)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(
    model::Optimizer,
    ::MOI.TimeLimitSec,
    value::Union{Nothing, Real}
    )
    model.options["time_limit"] = value === nothing ? Inf : Float64(value)
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    limit = model.options["time_limit"]
    return isfinite(limit) ? limit : nothing
end

function MOI.supports(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    return haskey(model.options, attr.name)
end

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    MOI.supports(model, attr) || throw(MOI.UnsupportedAttribute(attr))
    model.options[attr.name] = value
    return
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    MOI.supports(model, attr) || throw(MOI.UnsupportedAttribute(attr))
    return model.options[attr.name]
end

################################################################################
#                           RESULT ATTRIBUTES
################################################################################
MOI.get(model::Optimizer, ::MOI.TerminationStatus) = model.termination_status

MOI.get(model::Optimizer, ::MOI.RawStatusString) = model.raw_status

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return model.primal_status == MOI.NO_SOLUTION ? 0 : 1
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    return attr.result_index == 1 ? model.primal_status : MOI.NO_SOLUTION
end

MOI.get(model::Optimizer, ::MOI.DualStatus) = MOI.NO_SOLUTION

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.objective_value
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    bound = model.objective_bound
    if bound === nothing
        sense = MOI.get(model.cache, MOI.ObjectiveSense())
        return sense == MOI.MAX_SENSE ? Inf : -Inf
    end
    return bound
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex
    )
    MOI.check_result_index_bounds(model, attr)
    return model.incumbent[vi]
end

MOI.get(model::Optimizer, ::MOI.RelativeGap) = model.relative_gap

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time
