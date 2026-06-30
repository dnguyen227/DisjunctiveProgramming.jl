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

# InfiniteOpt-internal primitive: for an InfiniteOpt variable, return
# its per-support values as a Vector. `JuMP.value` returns a scalar
# (finite vars), Vector (1 infinite param), or N-D Array (multiple
# independent param groups). `vcat` lifts a scalar to a 1-element
# Vector; `vec` flattens any N-D Array to a Vector.
_per_support_values(variable::InfiniteOpt.GeneralVariableRef) =
    vec(vcat(JuMP.value(variable)))

# Read per-support values from the transformation backend, keyed by
# InfiniteOpt vars. Skips fixed vars. The objective-side translation
# (transcribe-then-AD when the objective has aggregate refs) lives in
# the `objective_linearization_point(::InfiniteModel, ...)` override
# below — base `extract_solution` doesn't need to anticipate it.
function DP.extract_solution(model::InfiniteOpt.InfiniteModel)
    return Dict(var => _per_support_values(var)
        for var in DP.collect_all_vars(model)
        if !JuMP.is_fixed(var))
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
# Dispatch overrides for InfiniteModel. The base LOA in src/loa.jl is
# written for finite (scalar) models. The InfiniteOpt master and feas
# submodel are themselves InfiniteModels, with per-support handling
# via point evaluation on infinite `GeneralVariableRef`s.

# Supports of the infinite parameter group `v` depends on. For a 1-D
# parameter, `supports(p)` is a `Vector{Float64}`; for a dependent
# group of dimension k, it is a k × N_supports `Matrix`. Returns each
# joint support point as one element — scalar for 1-D, column view for
# multi-D — so callers can iterate uniformly. Pair with `_at_support`,
# which splats vector supports into the variable's call form.
function _supports_of(v::InfiniteOpt.GeneralVariableRef)
    p = only(InfiniteOpt.parameter_refs(v))
    sup = InfiniteOpt.supports(p)
    ndims(sup) == 1 && return sup
    # Materialize each column as a concrete `Vector{Float64}`;
    # `eachcol` yields `SubArray` views that InfiniteOpt's
    # `VectorTuple` constructor refuses.
    return [sup[:, k] for k in axes(sup, 2)]
end

_is_point_var(v::InfiniteOpt.GeneralVariableRef) =
    InfiniteOpt.dispatch_variable_ref(v) isa InfiniteOpt.PointVariableRef

# A `MeasureRef` collapses `f(x, t)` over an infinite parameter into
# one ref; a `ParameterFunctionRef` collapses a parameter-dependent
# function value into one ref. Either hides decision-variable
# dependence from MOI Nonlinear AD, so LOA needs to transcribe the
# enclosing expression to recover correct gradients.
_is_aggregate_ref(v::InfiniteOpt.GeneralVariableRef) =
    InfiniteOpt.dispatch_variable_ref(v) isa Union{
        InfiniteOpt.MeasureRef, InfiniteOpt.ParameterFunctionRef}

function _has_aggregate_ref(expr)
    found = Ref(false)
    DP._interrogate_variables(expr) do v
        found[] || (found[] = _is_aggregate_ref(v))
    end
    return found[]
end

# Resolve a `PointVariableRef` (e.g., `L(0)` from a boundary
# condition) to the master's corresponding point variable: look up
# the underlying infinite var in `var_map`, then point-evaluate at
# the same support values. For all other variable refs (parameters,
# infinite/finite decision vars), fall back to direct lookup.
function DP.replace_variables_in_constraint(
    v::InfiniteOpt.GeneralVariableRef, var_map::AbstractDict
    )
    if _is_point_var(v)
        underlying = InfiniteOpt.infinite_variable_ref(v)
        return var_map[underlying](InfiniteOpt.parameter_values(v)...)
    end
    return var_map[v]
end

# Bool active arises from `_set_covering_combinations`, which keys
# combinations on `LogicalVariableRef -> Bool` regardless of whether
# the indicator is infinite. Broadcast over all supports for an
# infinite indicator; for a finite (or point-variable) ref, fall
# through to the base scalar dispatch.
function DP.add_no_good_terms(
    cut, binary_ref::InfiniteOpt.GeneralVariableRef, active::Bool
    )
    isempty(InfiniteOpt.parameter_refs(binary_ref)) &&
        return invoke(DP.add_no_good_terms,
            Tuple{Any, Any, Bool}, cut, binary_ref, active)
    for support in _supports_of(binary_ref)
        DP.add_no_good_terms(cut, _at_support(binary_ref, support), active)
    end
    return
end

# AbstractVector active arises from `combination_val` after a master
# solve, which yields per-support `Vector{Bool}`.
function DP.add_no_good_terms(
    cut, binary_ref::InfiniteOpt.GeneralVariableRef,
    actives::AbstractVector
    )
    supports = _supports_of(binary_ref)
    for (k, support) in enumerate(supports)
        DP.add_no_good_terms(
            cut, _at_support(binary_ref, support), actives[k])
    end
    return
end

# Complement-form AffExpr (`1 - y_underlying`) with per-support active
# descriptor — same fan-out as the variable-ref form, but the
# underlying var lives inside the AffExpr's terms. Use the underlying
# var's supports to drive the loop.
function DP.add_no_good_terms(
    cut, binary_ref::JuMP.GenericAffExpr,
    actives::AbstractVector
    )
    underlying = only(keys(binary_ref.terms))
    underlying isa InfiniteOpt.GeneralVariableRef || return invoke(
        DP.add_no_good_terms,
        Tuple{Any, Any, AbstractVector}, cut, binary_ref, actives)
    supports = _supports_of(underlying)
    for (k, support) in enumerate(supports)
        DP.add_no_good_terms(
            cut, _at_support(binary_ref, support), actives[k])
    end
    return
end

# Fan out across supports if either the binary or the constraint
# expression involves an infinite variable; otherwise emit one
# un-sliced site. The chosen supports come from the first infinite
# variable found in either expression.
function _infinite_cut_info(
    binary_ref,
    active,
    constraint_func,
    linearization_point,
    variable_map
    )
    supports = _relevant_supports(binary_ref, constraint_func)
    supports === nothing &&
        return ((binary_ref, linearization_point, variable_map),)
    actives = active isa AbstractVector ? active :
        fill(active, length(supports))
    return _cut_sites(binary_ref, supports, actives,
        linearization_point, variable_map)
end

# Supports governing per-support fan-out — pick the first infinite
# variable found in the binary expression, falling back to the
# constraint expression. `nothing` means everything is finite, so the
# caller emits a single un-sliced site.
function _relevant_supports(binary_ref, constraint_func)
    var = _find_infinite_var(binary_ref)
    var === nothing && (var = _find_infinite_var(constraint_func))
    var === nothing && return nothing
    return _supports_of(var)
end

_find_infinite_var(v::InfiniteOpt.GeneralVariableRef) =
    isempty(InfiniteOpt.parameter_refs(v)) ? nothing : v

function _find_infinite_var(expr)
    found = Ref{Any}(nothing)
    DP._interrogate_variables(expr) do v
        found[] === nothing || return
        v isa InfiniteOpt.GeneralVariableRef || return
        isempty(InfiniteOpt.parameter_refs(v)) && return
        found[] = v
    end
    return found[]
end

function _cut_sites(
    binary_ref,
    supports::AbstractVector,
    actives::AbstractVector,
    linearization_point::AbstractDict,
    variable_map::AbstractDict
    )
    sites = Any[]
    for (k, support) in enumerate(supports)
        actives[k] || continue
        point_var_map = Dict{
            InfiniteOpt.GeneralVariableRef,
            InfiniteOpt.GeneralVariableRef
            }()
        for (variable, master_var) in variable_map
            point_var_map[variable] = _at_support(master_var, support)
        end
        point = Dict{InfiniteOpt.GeneralVariableRef, Float64}()
        for (variable, point_value) in linearization_point
            point[variable] = _at(point_value, k)
        end
        push!(sites, (
            _at_support(binary_ref, support),
            point,
            point_var_map
            ))
    end
    return sites
end

# Slice a per-support container at index `k`. Length-1 vectors are
# finite-shape (one value applies at every support, per
# `_per_support_values` wrapping `JuMP.value` of a finite var as
# `[scalar]`) — return the single value regardless of `k`. Scalars
# pass through.
_at(values::AbstractArray, k::Integer) =
    length(values) == 1 ? values[1] : values[k]
_at(scalar, ::Integer) = scalar

# Point-evaluate an InfiniteOpt var at `support` if it's infinite;
# return the var as-is if it's finite. `support` is one joint support
# point — a scalar for 1-D parameters, a vector for a multi-D
# dependent group — and `v(support)` matches both call forms.
_at_support(v::InfiniteOpt.GeneralVariableRef, support) =
    isempty(InfiniteOpt.parameter_refs(v)) ? v : v(support)

# Rebuild a complement-form AffExpr (`1 - y_underlying`) with its
# variables point-evaluated, so per-support fan-out can use the
# AffExpr directly in the gating term `M(1 - binary_at_support)`.
function _at_support(
    expr::JuMP.GenericAffExpr{C, V}, support
    ) where {C, V <: InfiniteOpt.GeneralVariableRef}
    result = JuMP.GenericAffExpr{C, V}(expr.constant)
    for (var, coef) in expr.terms
        JuMP.add_to_expression!(result, coef, _at_support(var, support))
    end
    return result
end

# Map transcribed input JuMP var -> master point variable, used as
# `objective_ref_map`. Needed because MOI AD can't walk a `MeasureRef`,
# so a measure objective is transcribed, AD'd flat, and the gradient
# mapped back to master point vars. Per-support disjunct cuts skip this
# (they linearize natively via `_infinite_cut_info`).
function _transcribed_to_master_point(
    model::InfiniteOpt.InfiniteModel,
    ref_map::AbstractDict
    )
    result = Dict{JuMP.VariableRef, InfiniteOpt.GeneralVariableRef}()
    for v in DP.collect_all_vars(model)
        # Point vars share the infinite var's transcription, so skipping
        # them loses no mappings.
        _is_point_var(v) && continue
        master_var = ref_map[v]
        transcribed = InfiniteOpt.transformation_variable(v)
        prefs = InfiniteOpt.parameter_refs(v)
        if isempty(prefs)
            result[transcribed] = master_var
        else
            # `_supports_of`/`_at_support` handle 1-D (scalar) and
            # multi-D (vector) supports; `vec(supports(...))` would
            # flatten multi-D and break point evaluation.
            for (k, support) in enumerate(_supports_of(v))
                result[vec(transcribed)[k]] =
                    _at_support(master_var, support)
            end
        end
    end
    return result
end

# Build the master from `JuMP.copy_model(model)`, keeping linear
# constraints and the InfiniteOpt vars (per-support handling is done
# downstream by point evaluation). The objective branches on
# `_has_aggregate_ref`: aggregate-free objectives AD the InfiniteOpt
# expression directly; aggregate ones are transcribed flat and mapped
# back via `_transcribed_to_master_point`.
function DP.build_loa_master(
    model::InfiniteOpt.InfiniteModel, method::DP.LOA, sink = nothing
    )::DP._LOAMaster
    # Keep only linear, non-aggregate constraints. Nonlinear and
    # aggregate-wrapped ones (e.g. `∫(x^2,t) ≤ c`) re-enter as OA cuts,
    # so they are dropped at copy time. Variable bounds survive on
    # VariableInfo regardless.
    variable_type = InfiniteOpt.GeneralVariableRef
    master, copy_ref_map = JuMP.copy_model(
        model;
        filter_constraints = function (cref)
            con = JuMP.constraint_object(cref)
            F = typeof(con.func)
            F === variable_type && return true
            DP._is_linear_F(F) || return false
            return !_has_aggregate_ref(con.func)
        end
        )
    # `copy_model` copies the GDP optimize-hook; clear it so
    # `optimize!(master)` doesn't re-trigger reformulation on the
    # (empty-GDP-data) master copy.
    JuMP.set_optimize_hook(master, nothing)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)

    # InfiniteReferenceMap supports indexing but not iteration; build
    # a Dict so downstream LOA code can `haskey` / iterate over the
    # source-side refs LOA cares about.
    ref_map = Dict{InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()
    for v in DP.collect_all_vars(model)
        _is_point_var(v) && continue
        ref_map[v] = copy_ref_map[v]
    end
    for p in InfiniteOpt.all_parameters(model)
        ref_map[p] = copy_ref_map[p]
    end
    for pfunc in InfiniteOpt.all_parameter_functions(model)
        ref_map[pfunc] = copy_ref_map[pfunc]
    end

    raw_objective = JuMP.objective_function(model)
    objective_sense = JuMP.objective_sense(model)
    if _has_aggregate_ref(raw_objective)
        InfiniteOpt.build_transformation_backend!(model)
        transcribed_input = InfiniteOpt.transformation_model(model)
        original_objective = JuMP.objective_function(transcribed_input)
        objective_ref_map = _transcribed_to_master_point(
            model, ref_map)
    else
        original_objective = raw_objective
        objective_ref_map = ref_map
    end

    alpha_oa = JuMP.@variable(master, base_name = "alpha_oa")
    JuMP.@objective(master, objective_sense, alpha_oa)

    binary_map = Dict{DP.LogicalVariableRef, Any}()
    for (indicator, binary_ref) in DP._indicator_to_binary(model)
        binary_map[indicator] = DP._remap_indicator_to_binary(
            binary_ref, ref_map)
    end
    variable_map = Dict{InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()
    for v in DP.collect_all_vars(model)
        _is_point_var(v) && continue
        variable_map[v] = ref_map[v]
    end

    # Aggregate-wrapped LINEAR constraints (e.g. `𝔼(W, ξ) ≥ α`) were
    # dropped at copy time and the OA-cut path skips linear `F`, so add
    # each one to the master directly as a transcribed flat scalar.
    _add_aggregate_linear_constraints(
        master, model, ref_map, variable_type)

    return DP._LOAMaster(master, binary_map, variable_map,
        objective_sense, original_objective, alpha_oa,
        objective_ref_map, DP._build_disaggregator(
            model, method.inner_method, variable_map, binary_map, sink))
end

# Record one master-space disaggregation for `InfiniteModel`. The Hull
# cut emitter runs through the per-support fan-out over point-evaluated
# variables, so key the map by `(point variable, point binary)` at each
# support — `disaggregate_expression` then substitutes the point
# disaggregated variable per support. Finite (parameter-free) entries
# key once, unsliced.
function DP.record_disaggregation(
    hull::DP._Hull,
    ::InfiniteOpt.InfiniteModel,
    variable,
    binary,
    disaggregated
    )
    supports = _disaggregator_supports(variable, binary)
    if supports === nothing
        hull.disjunct_variables[(variable, binary)] = disaggregated
    else
        for support in supports
            hull.disjunct_variables[(_at_support(variable, support),
                _at_support(binary, support))] =
                    _at_support(disaggregated, support)
        end
    end
    return
end

# Supports the per-support fan-out gates this disjunct on (mirrors
# `_relevant_supports`): the binary's if the indicator is infinite, else
# the variable's if it is infinite, else `nothing` (both finite, one
# unsliced key). Keying off the binary lets a finite variable under an
# infinite indicator match the sliced binary the emitter looks up.
function _disaggregator_supports(master_var, master_bin)
    binary_var = _find_infinite_var(master_bin)
    binary_var === nothing || return _supports_of(binary_var)
    isempty(InfiniteOpt.parameter_refs(master_var)) && return nothing
    return _supports_of(master_var)
end

# Transcribe each linear constraint containing a `MeasureRef` /
# `ParameterFunctionRef` and append the flat scalar form to the master.
# Already affine post-transcription, so `_linearize_at` just substitutes
# variables via `transcribed_to_master`.
function _add_aggregate_linear_constraints(
    master::InfiniteOpt.InfiniteModel,
    model::InfiniteOpt.InfiniteModel,
    ref_map::AbstractDict,
    variable_type::Type
    )
    has_any = false
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        DP._is_linear_F(F) || continue
        for cref in JuMP.all_constraints(model, F, S)
            con = JuMP.constraint_object(cref)
            _has_aggregate_ref(con.func) || continue
            has_any = true
            break
        end
        has_any && break
    end
    has_any || return

    InfiniteOpt.build_transformation_backend!(model)
    transcribed_to_master = _transcribed_to_master_point(model, ref_map)
    # Finite parameters survive transcription as scalar JuMP vars and can
    # appear in transcribed constraints, but `_transcribed_to_master_point`
    # only walks decision vars. Map them to the master's parameters so the
    # constraint stays parameter-relative (honors `set_value` without
    # rebuild).
    for p in InfiniteOpt.all_parameters(model)
        InfiniteOpt.dispatch_variable_ref(p) isa
            InfiniteOpt.FiniteParameterRef || continue
        transcribed_p = InfiniteOpt.transformation_variable(p)
        transcribed_p isa JuMP.VariableRef || continue
        haskey(ref_map, p) || continue
        transcribed_to_master[transcribed_p] = ref_map[p]
    end
    # `_linearize_at(::GenericAffExpr, ...)` ignores its
    # `linearization_point` arg; pass any empty dict.
    empty_point = Dict{JuMP.VariableRef, Float64}()
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        DP._is_linear_F(F) || continue
        for cref in JuMP.all_constraints(model, F, S)
            con = JuMP.constraint_object(cref)
            _has_aggregate_ref(con.func) || continue
            con isa JuMP.ScalarConstraint || continue
            transcribed_func = InfiniteOpt.transformation_expression(
                con.func)
            if transcribed_func isa AbstractArray
                for tf in vec(transcribed_func)
                    master_expr = DP._linearize_at(
                        tf, empty_point, transcribed_to_master)
                    JuMP.@constraint(master, master_expr in con.set)
                end
            else
                master_expr = DP._linearize_at(
                    transcribed_func, empty_point, transcribed_to_master)
                JuMP.@constraint(master, master_expr in con.set)
            end
        end
    end
    return
end

# Objective linearization point for `InfiniteModel` (the seam the shared
# base `add_oa_cuts` calls). The master objective is either the raw
# InfiniteOpt expression (non-aggregate, expects a scalar `xk[v]` keyed
# on InfiniteOpt vars) or the transcribed flat scalar expression
# (aggregate, expects a transcribed-`JuMP.VariableRef`-keyed scalar
# dict). Translate the base per-support `Vector`-valued point into
# whichever shape `original_objective` needs.
function DP.objective_linearization_point(
    model::InfiniteOpt.InfiniteModel,
    linearization_point::AbstractDict
    )
    _has_aggregate_ref(JuMP.objective_function(model)) &&
        return _transcribe_linearization_point(model, linearization_point)
    T = eltype(valtype(linearization_point))
    return Dict{InfiniteOpt.GeneralVariableRef, T}(
        var => values[1]
        for (var, values) in linearization_point
        if isempty(InfiniteOpt.parameter_refs(var)))
end

# Global OA cuts for `InfiniteModel`: route through transcription so
# per-support / aggregate-ref expressions reach `_linearize_at` as flat
# scalars over `JuMP.VariableRef`s. Aggregate-containing constraints
# (e.g. `∫f(x,t)dt ≤ 0`) transcribe to a single scalar expression.
# Constraints with infinite-parameter dependence (e.g. `x(t) ≥ 0`)
# transcribe to a per-support `AbstractArray` of scalar expressions —
# one OA cut is emitted per support.
function DP.add_global_oa_cuts(
    model::InfiniteOpt.InfiniteModel,
    master::DP._LOAMaster,
    result::NamedTuple,
    method::DP.LOA
    )
    penalty_sign = DP._penalty_sign(Val(master.objective_sense))
    variable_type = InfiniteOpt.GeneralVariableRef
    reform_set = DP.is_gdp_model(model) ?
        Set(DP._reformulation_constraints(model)) : Set()
    transcribed_xk = Ref{Any}(nothing)
    transcribed_to_master = Ref{Any}(nothing)
    ensure_transcribed = function ()
        transcribed_xk[] === nothing || return
        InfiniteOpt.build_transformation_backend!(model)
        transcribed_xk[] = _transcribe_linearization_point(
            model, result.linearization_point)
        transcribed_to_master[] = _transcribed_to_master_point(
            model, master.variable_map)
        return
    end
    for (F, S) in JuMP.list_of_constraint_types(model)
        F === variable_type && continue
        DP._is_linear_F(F) && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref in reform_set && continue
            con = JuMP.constraint_object(cref)
            con isa JuMP.ScalarConstraint || continue
            ensure_transcribed()
            transcribed_func = InfiniteOpt.transformation_expression(
                con.func)
            if transcribed_func isa AbstractArray
                for tf in vec(transcribed_func)
                    lin = DP._linearize_at(tf, transcribed_xk[],
                        transcribed_to_master[])
                    DP._add_global_oa_row(master, lin, con.set,
                        method, penalty_sign)
                end
            else
                lin = DP._linearize_at(transcribed_func,
                    transcribed_xk[], transcribed_to_master[])
                DP._add_global_oa_row(master, lin, con.set,
                    method, penalty_sign)
            end
        end
    end
    return
end

# Per-constraint OA cut emission for `InfiniteModel` (the seam the base
# `add_disjunct_oa_cuts` driver calls). Non-aggregate constraints fan out
# per support via `_infinite_cut_info`; aggregate ones (`MeasureRef`) are
# transcribed flat and handed to the base `_add_oa_cut_for_constraint`.
# `cache` memoizes the transcription maps once per pass (lazily, on the
# first aggregate constraint) so they are not rebuilt per constraint.
function DP.add_disjunct_constraint_oa_cuts(
    model::InfiniteOpt.InfiniteModel,
    constraint::JuMP.AbstractConstraint,
    master::DP._LOAMaster,
    binary_ref,
    active,
    result::NamedTuple,
    method::DP.LOA,
    penalty_sign::Int,
    cache
    )
    if _has_aggregate_ref(constraint.func)
        if cache[] === nothing
            InfiniteOpt.build_transformation_backend!(model)
            cache[] = (_transcribe_linearization_point(
                    model, result.linearization_point),
                _transcribed_to_master_point(model, master.variable_map))
        end
        transcribed_xk, transcribed_to_master = cache[]
        transcribed_func = InfiniteOpt.transformation_expression(
            constraint.func)
        transcribed_constraint = JuMP.ScalarConstraint(
            transcribed_func, constraint.set)
        for (point_binary, _, _) in _infinite_cut_info(binary_ref,
            active, transcribed_func, result.linearization_point,
            master.variable_map)
            DP._add_oa_cut_for_constraint(transcribed_constraint, master,
                point_binary, transcribed_xk, transcribed_to_master,
                method, penalty_sign)
        end
        return
    end
    for (point_binary, point, var_map) in _infinite_cut_info(binary_ref,
        active, constraint.func, result.linearization_point,
        master.variable_map)
        DP._add_oa_cut_for_constraint(constraint, master, point_binary,
            point, var_map, method, penalty_sign)
    end
    return
end

# Apply the combination's fixes and return a closure that reverses them.
# A `Bool` delegates to base `fix_indicator`; a per-support `Vector{Bool}`
# pins each support of the infinite binary with a point-equality
# constraint. The closure (used by `with_fixed_combination` to tear down
# each iteration) deletes those pins and unfixes; `commit_combination`
# ignores it to persist the committed fix.
function DP.fix_combination(
    model::InfiniteOpt.InfiniteModel, combination::AbstractDict
    )
    constraint_refs = InfiniteOpt.InfOptConstraintRef[]
    fixed_indicators = DP.LogicalVariableRef{
        InfiniteOpt.InfiniteModel}[]
    for (indicator, value) in combination
        if value isa AbstractVector
            binary_ref = DP._indicator_to_binary(model)[indicator]
            target, target_for_active = binary_ref isa JuMP.GenericAffExpr ?
                (only(keys(binary_ref.terms)), 0.0) :
                (binary_ref, 1.0)
            target_for_inactive = 1.0 - target_for_active
            for (k, support) in enumerate(_supports_of(target))
                push!(constraint_refs, JuMP.@constraint(model,
                    _at_support(target, support) ==
                        (value[k] ? target_for_active : target_for_inactive)))
            end
        else
            DP.fix_indicator(model, indicator, value)
            push!(fixed_indicators, indicator)
        end
    end
    return function ()
        for ref in constraint_refs
            JuMP.is_valid(model, ref) && JuMP.delete(model, ref)
        end
        for indicator in fixed_indicators
            DP.unfix_indicator(model, indicator)
        end
    end
end

# Per-support fix for an infinite logical's `Vector{Bool}` value: an
# equality at each support. `with_fixed_combination` undid the original's
# fix before the NLPF fall-through, so this is the only fix on the copy.
# Equalities (not `JuMP.fix`) avoid force-deleting the relaxed copy's
# bounds, whose refs may not survive `copy_model`.
function DP._fix_binary_on_copy(copy, binary, value::AbstractVector)
    underlying = binary isa JuMP.GenericAffExpr ?
        only(keys(binary.terms)) : binary
    for (k, support) in enumerate(_supports_of(underlying))
        JuMP.@constraint(copy,
            _at_support(binary, support) == (value[k] ? 1.0 : 0.0))
    end
    return
end

# Broadcast a per-support warm start across an infinite var's
# transcribed instances; for a finite var on an InfiniteModel,
# `transcription_variable` returns a single ref and `values` is
# length-1.
function DP.set_linearization_start(
    variable::InfiniteOpt.GeneralVariableRef,
    values::AbstractVector
    )
    transcribed = InfiniteOpt.transformation_variable(variable)
    if transcribed isa AbstractArray
        for (k, ref) in enumerate(vec(transcribed))
            JuMP.set_start_value(ref, values[k])
        end
    else
        JuMP.set_start_value(transcribed, only(values))
    end
    return
end

# Convert an InfiniteModel-var-keyed per-support point into a
# transcribed-JuMP-var-keyed scalar point. Companion to
# `_transcribed_to_master_point`: feeds the AD walker for the
# objective OA cut. See the WHY note above
# `_transcribed_to_master_point` for the MeasureRef/AD constraint
# that motivates this whole transcribe-then-AD layer.
function _transcribe_linearization_point(
    model::InfiniteOpt.InfiniteModel,
    linearization_point::AbstractDict
    )
    T = eltype(valtype(linearization_point))
    transcribed = Dict{JuMP.VariableRef, T}()
    for (variable, values) in linearization_point
        _add_to_transcribed_dict(
            transcribed,
            InfiniteOpt.transformation_variable(variable),
            values
            )
    end
    return transcribed
end

# Infinite var: per-support transcribed array, values per-support
function _add_to_transcribed_dict(
    d::AbstractDict,
    ts::AbstractArray,
    values::AbstractVector
    )
    for (k, ref) in enumerate(vec(ts))
        d[ref] = values[k]
    end
    return
end

# Finite var: single transcribed ref, values 1-element
_add_to_transcribed_dict(
    d::AbstractDict,
    ts::JuMP.AbstractVariableRef,
    values::AbstractVector
    ) =
    (d[ts] = values[1]; nothing)

end
