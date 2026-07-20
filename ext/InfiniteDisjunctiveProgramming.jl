module InfiniteDisjunctiveProgramming

import JuMP.MOI as _MOI
import InfiniteOpt, JuMP
import DisjunctiveProgramming as DP

################################################################################
#                                   MODEL
################################################################################
function DP.InfiniteGDPModel(args...; kwargs...)
    return DP.GDPModel{
        InfiniteOpt.InfiniteModel,
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.InfOptConstraintRef
        }(args...; kwargs...)
end

function DP.collect_all_vars(model::InfiniteOpt.InfiniteModel)
    vars = JuMP.all_variables(model)
    derivs = InfiniteOpt.all_derivatives(model)
    return append!(vars, derivs)
end

################################################################################
#                                 VARIABLES
################################################################################
DP.InfiniteLogical(prefs...) = DP.Logical(InfiniteOpt.Infinite(prefs...))

_is_parameter(vref::InfiniteOpt.GeneralVariableRef) =
    _is_parameter(InfiniteOpt.dispatch_variable_ref(vref))
_is_parameter(::InfiniteOpt.DependentParameterRef) = true
_is_parameter(::InfiniteOpt.IndependentParameterRef) = true
_is_parameter(::InfiniteOpt.ParameterFunctionRef) = true
_is_parameter(::InfiniteOpt.FiniteParameterRef) = true
_is_parameter(::Any) = false

function DP.requires_disaggregation(vref::InfiniteOpt.GeneralVariableRef)
    return !_is_parameter(vref)
end

function DP.VariableProperties(vref::InfiniteOpt.GeneralVariableRef)
    info = DP.get_variable_info(vref)
    name = JuMP.name(vref)
    set = nothing
    prefs = InfiniteOpt.parameter_refs(vref)
    var_type = !isempty(prefs) ? InfiniteOpt.Infinite(prefs...) : nothing
    return DP.VariableProperties(info, name, set, var_type)
end

# Extract parameter refs from expression and return VariableProperties with Infinite type
function DP.VariableProperties(
    expr::Union{
        JuMP.GenericAffExpr{C, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericQuadExpr{C, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericNonlinearExpr{InfiniteOpt.GeneralVariableRef}
    }
) where C
    prefs = InfiniteOpt.parameter_refs(expr)
    info = DP._free_variable_info()
    var_type = !isempty(prefs) ? InfiniteOpt.Infinite(prefs...) : nothing
    return DP.VariableProperties(info, "", nothing, var_type)
end

function DP.VariableProperties(
    exprs::Vector{<:Union{
        InfiniteOpt.GeneralVariableRef,
        JuMP.GenericAffExpr{<:Any, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericQuadExpr{<:Any, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericNonlinearExpr{InfiniteOpt.GeneralVariableRef}
    }}
)
    all_prefs = Set{InfiniteOpt.GeneralVariableRef}()
    for expr in exprs
        for pref in InfiniteOpt.parameter_refs(expr)
            push!(all_prefs, pref)
        end
    end
    prefs = Tuple(all_prefs)
    info = DP._free_variable_info()
    var_type = !isempty(prefs) ? InfiniteOpt.Infinite(prefs...) : nothing
    return DP.VariableProperties(info, "", nothing, var_type)
end

function JuMP.value(vref::DP.LogicalVariableRef{InfiniteOpt.InfiniteModel})
    return JuMP.value(DP.binary_variable(vref)) .>= 0.5
end

################################################################################
#                                CONSTRAINTS
################################################################################
function JuMP.add_constraint(
    model::InfiniteOpt.InfiniteModel,
    c::JuMP.VectorConstraint{F, S},
    name::String = ""
) where {F, S <: DP.AbstractCardinalitySet}
    return DP._add_cardinality_constraint(model, c, name)
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{DP._LogicalExpr{M}, S},
    name::String = ""
) where {S, M <: InfiniteOpt.InfiniteModel}
    return DP._add_logical_constraint(model, c, name)
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{DP.LogicalVariableRef{M}, S},
    name::String = ""
) where {M <: InfiniteOpt.InfiniteModel, S}
    error("Cannot define constraint on single logical variable, use `fix` instead.")
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{
        JuMP.GenericAffExpr{C, DP.LogicalVariableRef{M}}, S
    },
    name::String = ""
) where {M <: InfiniteOpt.InfiniteModel, S, C}
    error("Cannot add, subtract, or multiply with logical variables.")
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{
        JuMP.GenericQuadExpr{C, DP.LogicalVariableRef{M}}, S
    },
    name::String = ""
) where {M <: InfiniteOpt.InfiniteModel, S, C}
    error("Cannot add, subtract, or multiply with logical variables.")
end

################################################################################
#                                  METHODS
################################################################################
function DP.get_constant(
    expr::JuMP.GenericAffExpr{T, InfiniteOpt.GeneralVariableRef}
) where {T}
    constant = JuMP.constant(expr)
    param_expr = zero(typeof(expr))
    for (var, coeff) in expr.terms
        if _is_parameter(var)
            JuMP.add_to_expression!(param_expr, coeff, var)
        end
    end
    return constant + param_expr
end

function DP.disaggregate_expression(
    model::M,
    aff::JuMP.GenericAffExpr,
    bvref::Union{JuMP.AbstractVariableRef, JuMP.GenericAffExpr},
    method::DP._Hull
) where {M <: InfiniteOpt.InfiniteModel}
    terms = Any[aff.constant * bvref]
    for (vref, coeff) in aff.terms
        if JuMP.is_binary(vref)
            push!(terms, coeff * vref)
        elseif vref isa InfiniteOpt.GeneralVariableRef && _is_parameter(vref)
            push!(terms, coeff * vref * bvref)
        elseif !haskey(method.disjunct_variables, (vref, bvref))
            push!(terms, coeff * vref)
        else
            dvref = method.disjunct_variables[vref, bvref]
            push!(terms, coeff * dvref)
        end
    end
    return JuMP.@expression(model, sum(terms))
end

################################################################################
#                          MBM FOR INFINITEMODEL
################################################################################

# Copy the InfiniteModel, strip everything but VariableInfo bounds,
# add back the selected disjunct constraints, transcribe, and return
# only the other disjunct's constraints plus variable bounds.
function DP.copy_model_with_constraints(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP._MBM
    )
    # Filter out every source constraint at copy time instead of
    # copying then deleting. Equivalent end state, fewer allocations.
    mini, ref_map = JuMP.copy_model(
        model; filter_constraints = cref -> false
        )

    for cref in constraints
        con = JuMP.constraint_object(cref)
        T = one(JuMP.value_type(typeof(mini)))
        JuMP.@constraint(mini, ref_map[con.func] * T in con.set)
    end

    InfiniteOpt.build_transformation_backend!(mini)
    transcribed = InfiniteOpt.transformation_model(mini)
    JuMP.set_optimizer(transcribed, method.optimizer)
    JuMP.set_silent(transcribed)

    # fwd_map needs every ref reachable from disjunct constraints —
    # decision vars + parameters + parameter functions so the
    # objective substitution in `prepare_max_M_objective` can look up
    # any term it sees.
    decision_vars = DP.collect_all_vars(model)
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef,
        Vector{InfiniteOpt.GeneralVariableRef}}()
    for v in decision_vars
        fwd_map[v] = [ref_map[v]]
    end
    for p in InfiniteOpt.all_parameters(model)
        fwd_map[p] = [ref_map[p]]
    end
    for pf in InfiniteOpt.all_parameter_functions(model)
        fwd_map[pf] = [ref_map[pf]]
    end
    return DP.GDPSubmodel(mini, decision_vars, fwd_map)
end

function DP.prepare_max_M_objective(
    ::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.LessThan}
    flat_map = Dict(v => ws[1] for (v, ws) in sub.fwd_map)
    obj_func = DP.replace_variables_in_constraint(obj.func, flat_map)
    return obj_func - obj.set.upper
end

function DP.prepare_max_M_objective(
    ::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.GreaterThan}
    flat_map = Dict(v => ws[1] for (v, ws) in sub.fwd_map)
    obj_func = DP.replace_variables_in_constraint(obj.func, flat_map)
    return obj.set.lower - obj_func
end

# Constant interpolation
function _interpolate(
    grids::NTuple{N, AbstractVector{<:Real}},
    values::AbstractArray{<:Real, N}
    ) where {N}
    # mimic the call form of Interpolations.jl's interpolation
    return (args...) -> _interpolate_at(grids, values, args)
end

function _interpolate_at(
    grids::NTuple{N, AbstractVector{<:Real}},
    values::AbstractArray{<:Real, N},
    args::NTuple{N, <:Real}
    ) where {N}
    # lower-corner cell index per dimension
    idx_lo = ntuple(d ->
        clamp(searchsortedlast(grids[d], args[d]),1, length(grids[d]) - 1), N
    )
    # max over the 2^N corners; bit d of k picks lower or upper
    return maximum(
        values[ntuple(d -> idx_lo[d] +((k >> (d - 1)) & 1), N)...]
        for k in 0:(2^N - 1)
        )
end

# Transcribe mini_expr, solve per support on the transcribed JuMP
# model, and aggregate to a scalar if uniform, else to a parameter
# function on main.
function DP.raw_M(
    sub::DP.GDPSubmodel{<:InfiniteOpt.InfiniteModel},
    mini_expr::JuMP.AbstractJuMPScalar,
    method::DP._MBM
    )
    objectives = InfiniteOpt.transformation_expression(mini_expr)
    transcribed = InfiniteOpt.transformation_model(sub.model)
    inner_sub = DP.GDPSubmodel(transcribed,JuMP.VariableRef[],
        Dict{JuMP.VariableRef, Vector{JuMP.VariableRef}}()
        )
    M_vals = Array{typeof(method.default_M)}(undef, size(objectives))
    for I in eachindex(objectives)
        m = DP.raw_M(inner_sub, objectives[I], method)
        m === nothing && return nothing
        M_vals[I] = m
    end
    all(==(first(M_vals)), M_vals) && return first(M_vals)
    mini_prefs = InfiniteOpt.parameter_refs(mini_expr)
    reverse_map = Dict(ws[1] => v for (v, ws) in sub.fwd_map)
    prefs = Tuple(reverse_map[p] for p in mini_prefs)
    main = JuMP.owner_model(first(prefs))
    grids = Tuple(InfiniteOpt.supports(p) for p in prefs)
    param_func = InfiniteOpt.build_parameter_function(
        error, _interpolate(grids, M_vals), prefs)
    return InfiniteOpt.add_parameter_function(main, param_func)
end

################################################################################
#                    CUTTING PLANES FOR INFINITEMODEL
################################################################################

# Build CP subproblem: reformulate the InfiniteModel in-place, transcribe,
# copy, and wrap in GDPSubmodel with forward variable map.
function DP.copy_and_reformulate(
    model::InfiniteOpt.InfiniteModel,
    decision_vars::Vector{InfiniteOpt.GeneralVariableRef},
    reform_method::DP.AbstractReformulationMethod,
    method::DP.CuttingPlanes
    )
    DP.reformulate_model(model, reform_method)
    InfiniteOpt.build_transformation_backend!(model)
    transcribed = InfiniteOpt.transformation_model(model)
    transcription_fwd = Dict{InfiniteOpt.GeneralVariableRef,
        Vector{JuMP.VariableRef}}()
    for v in DP.collect_all_vars(model)
        transcription_var = InfiniteOpt.transformation_variable(v)
        var_prefs = InfiniteOpt.parameter_refs(v)
        transcription_fwd[v] = isempty(var_prefs) ?
            [transcription_var] : vec(transcription_var)
    end
    sub_copy, copy_map = JuMP.copy_model(transcribed)
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}()
    for v in decision_vars
        haskey(transcription_fwd, v) || continue
        fwd_map[v] = [copy_map[transcribed_var] for transcribed_var in transcription_fwd[v]]
    end
    sub = DP.GDPSubmodel(sub_copy, decision_vars, fwd_map)
    JuMP.set_optimizer(sub.model, method.optimizer)
    # NOTE: previously called JuMP.set_silent(sub.model) here, but
    # that hides the Gurobi B&B trace even when the caller passes
    # OutputFlag=1 / LogToConsole=1 via optimizer_with_attributes.
    # The caller is now responsible for silencing if desired.
    return sub
end

# Read per-support values from the transformation backend.
function DP.extract_solution(model::InfiniteOpt.InfiniteModel)
    dvars = DP.collect_cutting_planes_vars(model)
    V = eltype(dvars)
    T = JuMP.value_type(typeof(model))
    sol = Dict{V, Vector{T}}()
    for v in dvars
        transcription_var = InfiniteOpt.transformation_variable(v)
        var_prefs = InfiniteOpt.parameter_refs(v)
        sol[v] = isempty(var_prefs) ? [JuMP.value(transcription_var)] :
            JuMP.value.(vec(transcription_var))
    end
    return sol
end

# Add a pointwise-sum cut directly to the transformation backend and mark
# it ready so the next optimize! doesn't re-transcribe and wipe the cut.
function DP.add_cut(
    model::InfiniteOpt.InfiniteModel,
    decision_vars::Vector{InfiniteOpt.GeneralVariableRef},
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}},
    sep_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    transcribed = InfiniteOpt.transformation_model(model)
    cut_expr = zero(JuMP.GenericAffExpr{
        JuMP.value_type(typeof(transcribed)),
        JuMP.variable_ref_type(transcribed)})
    for var in decision_vars
        haskey(rBM_sol, var) || continue
        haskey(sep_sol, var) || continue
        rbm_vals = rBM_sol[var]
        sep_vals = sep_sol[var]
        transcription_var = InfiniteOpt.transformation_variable(var)
        transcribed_vars = transcription_var isa AbstractArray ?
            vec(transcription_var) : [transcription_var]
        for k in eachindex(transcribed_vars)
            xi = 2 * (sep_vals[k] - rbm_vals[k])
            JuMP.add_to_expression!(cut_expr, xi, transcribed_vars[k])
            JuMP.add_to_expression!(cut_expr, -xi * sep_vals[k])
        end
    end
    JuMP.@constraint(transcribed, cut_expr >= 0)
    InfiniteOpt.set_transformation_backend_ready(model, true)
    return
end

################################################################################
#                       LOA FOR INFINITEMODEL
################################################################################
# LOA override for InfiniteModel: after the inner reformulation, build
# the transformation backend and hand the LOA loop the backend model —
# a plain scalar JuMP model whose variables, measures, and constraints
# are already transcribed per support. The loop runs entirely on base
# code and solves the backend directly, which never invalidates it, so
# the committed fixes and warm starts survive the final `optimize!`.

# Per-support transcribed refs of an InfiniteOpt object, as a Vector
# (finite objects give one element).
_transcribed_refs(x::AbstractArray) = vec(x)
_transcribed_refs(x) = [x]

# Index a per-support vector, broadcasting a single-element (finite)
# vector across all supports.
_at(values::Vector, k::Int) =
    length(values) == 1 ? values[1] : values[k]

# Per-support transcribed binary references for an indicator: the
# transcribed underlying binary, complemented for a complement-form
# indicator.
function _transcribed_binary_refs(model::InfiniteOpt.InfiniteModel, indicator)
    binary_ref = DP._indicator_to_binary(model)[indicator]
    binaries = _transcribed_refs(InfiniteOpt.transformation_variable(
        DP._underlying_binary(binary_ref)))
    binary_ref isa JuMP.GenericAffExpr || return binaries
    return [1.0 - binary for binary in binaries]
end

# Transcribe the inner Hull disaggregation map per support. A finite
# variable under an infinite indicator keys its single disaggregated
# copy by each per-support binary reference, mirroring the per-support
# disjunct cuts that look it up.
function _transcribed_disaggregation_map(model::InfiniteOpt.InfiniteModel)
    result = Dict{Tuple{JuMP.VariableRef,
        Union{JuMP.VariableRef, JuMP.AffExpr}}, JuMP.VariableRef}()
    for ((variable, indicator), disaggregated) in DP._disaggregation_map(model)
        binary_refs = _transcribed_binary_refs(model, indicator)
        variables = _transcribed_refs(
            InfiniteOpt.transformation_variable(variable))
        disaggregateds = _transcribed_refs(InfiniteOpt.transformation_variable(
            disaggregated))
        n = max(length(binary_refs), length(variables), length(disaggregateds))
        for k in 1:n
            result[(_at(variables, k), _at(binary_refs, k))] =
                _at(disaggregateds, k)
        end
    end
    return result
end

function DP.build_loa_problem(
    model::InfiniteOpt.InfiniteModel,
    method::DP.LOA
    )
    InfiniteOpt.build_transformation_backend!(model)
    nlp = InfiniteOpt.transformation_model(model)

    binaries = JuMP.VariableRef[]
    for (_, binary_ref) in DP._indicator_to_binary(model)
        append!(binaries, _transcribed_refs(
            InfiniteOpt.transformation_variable(
                DP._underlying_binary(binary_ref))))
    end
    unique!(binaries)

    disjunct_constraints = Tuple{Union{JuMP.VariableRef, JuMP.AffExpr},
        JuMP.AbstractJuMPScalar, _MOI.AbstractScalarSet}[]
    for (_, disjunction) in DP._disjunctions(model)
        for indicator in disjunction.constraint.indicators
            haskey(DP._indicator_to_constraints(model), indicator) ||
                continue
            binary_refs = _transcribed_binary_refs(model, indicator)
            for cref in DP._indicator_to_constraints(model)[indicator]
                cref isa DP.DisjunctConstraintRef || continue
                constraint = DP._disjunct_constraints(model)[
                    JuMP.index(cref)].constraint
                constraint isa JuMP.ScalarConstraint || continue
                DP._is_linear_F(typeof(constraint.func)) && continue
                funcs = _transcribed_refs(InfiniteOpt.transformation_expression(
                    constraint.func))
                for k in 1:max(length(binary_refs), length(funcs))
                    push!(disjunct_constraints,
                        (_at(binary_refs, k), _at(funcs, k), constraint.set))
                end
            end
        end
    end

    global_constraints = Tuple{JuMP.AbstractJuMPScalar,
        _MOI.AbstractScalarSet}[]
    reform_set = Set(DP._reformulation_constraints(model))
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === InfiniteOpt.GeneralVariableRef && continue
        DP._is_linear_F(F) && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            constraint = JuMP.constraint_object(cref)
            constraint isa JuMP.ScalarConstraint || continue
            for func in _transcribed_refs(InfiniteOpt.transformation_expression(
                constraint.func))
                push!(global_constraints, (func, constraint.set))
            end
        end
    end

    return DP._LOAProblem(nlp, binaries,
        disjunct_constraints, global_constraints,
        _transcribed_disaggregation_map(model))
end

# Inject the outcome into the transcription backend and mark it ready so
# InfiniteOpt value/objective queries return it. No binary restore (the
# relaxed binaries are backend-transcribed and discarded on rebuild) and no
# solver stash (a later reformulation mutates the InfiniteModel, invalidates
# the backend, and rebuilds it fresh with the real optimizer).
function DP._load_solution(
    model::InfiniteOpt.InfiniteModel,
    problem::DP._LOAProblem,
    outcome::NamedTuple,
    method::DP.LOA
    )
    DP._inject_solution(problem.nlp, problem, outcome)
    InfiniteOpt.set_transformation_backend_ready(model, true)
    return
end

end
