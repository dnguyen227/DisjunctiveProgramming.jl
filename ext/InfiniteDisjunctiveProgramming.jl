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
    mini, ref_map = JuMP.copy_model(model)

    # Drop global constraints.
    for cref in JuMP.all_constraints(mini)
        JuMP.delete(mini, cref)
    end

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
# (transcribe-then-AD when the objective has aggregate refs) lives
# in the `add_oa_cuts(::InfiniteModel, ...)` override below — base
# `extract_solution` doesn't need to anticipate it.
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

DP.any_active(actives::AbstractVector{Bool}) = any(actives)

# `JuMP.value` returns a per-support `Array` for infinite vars and a
# scalar for finite vars. The `> 0.5` cutoff handles solver-side
# integer-feasibility slack (e.g. HiGHS can return 2.75e-40 for a
# "0" binary), where direct `Bool(val)` would `InexactError`.
# `JuMP.value` returns a per-support `Array` for infinite vars and a
# scalar for finite vars; `round(Bool, ·)` handles both via broadcast
# and absorbs solver integer-feasibility slack.
function DP.combination_val(v::InfiniteOpt.GeneralVariableRef)
    val = JuMP.value(v)
    return val isa AbstractArray ? vec(round.(Bool, val)) : round(Bool, val)
end

_supports_of(v::InfiniteOpt.GeneralVariableRef) =
    vec(InfiniteOpt.supports(only(InfiniteOpt.parameter_refs(v))))

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
# combinations on `LogicalVariableRef → Bool` regardless of whether
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
        DP.add_no_good_terms(cut, binary_ref(support), active)
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
        DP.add_no_good_terms(cut, binary_ref(support), actives[k])
    end
    return
end

function DP.cut_info(
    binary_ref::InfiniteOpt.GeneralVariableRef, active::Bool,
    linearization_point, variable_map, dual
    )
    supports = _supports_of(binary_ref)
    return _cut_sites(binary_ref, supports,
        fill(true, length(supports)),
        linearization_point, variable_map, dual)
end

function DP.cut_info(
    binary_ref::InfiniteOpt.GeneralVariableRef,
    actives::AbstractVector,
    linearization_point, variable_map, dual
    )
    return _cut_sites(binary_ref, _supports_of(binary_ref), actives,
        linearization_point, variable_map, dual)
end

function _cut_sites(
    binary_ref::InfiniteOpt.GeneralVariableRef,
    supports::AbstractVector,
    actives::AbstractVector,
    linearization_point::AbstractDict,
    variable_map::AbstractDict,
    dual
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
            binary_ref(support),
            point,
            point_var_map,
            _at(dual, k)
            ))
    end
    return sites
end

# Slice a per-support container at index `k`; pass scalars through.
_at(values::AbstractArray, k::Integer) = values[k]
_at(scalar, ::Integer) = scalar

# Point-evaluate an InfiniteOpt var at `support` if it's infinite;
# return the var as-is if it's finite.
_at_support(v::InfiniteOpt.GeneralVariableRef, support) =
    isempty(InfiniteOpt.parameter_refs(v)) ? v : v(support)

# Build map: transcribed input JuMP var → master point variable.
# For an infinite input var v, every transcribed support `v_k`
# maps to `ref_map[v](d_k)` (master point variable). Used as
# `objective_ref_map` so the objective OA cut can linearize the
# (already flat) transcribed objective and land in master point
# variables.
#
# Why the transcribe-then-AD detour exists: MOI Nonlinear AD has
# no walker for `InfiniteOpt.MeasureRef`, so an objective like
# `∫(f(z, t), t)` cannot be differentiated directly. The objective
# OA cut therefore transcribes the input model once, ADs the
# resulting flat scalar objective, and uses this map to translate
# the gradient back into master point variables. Per-support
# DISJUNCT cuts do NOT need this — they linearize natively in
# InfiniteModel space via `cut_info`. If the objective contains no
# measures, this whole layer is dead weight; killing it would
# require either banning measure objectives or hand-writing a
# `_linearize_at` that walks `MeasureRef` symbolically.
function _transcribed_to_master_point(
    model::InfiniteOpt.InfiniteModel,
    ref_map::AbstractDict
    )
    result = Dict{JuMP.VariableRef, InfiniteOpt.GeneralVariableRef}()
    for v in DP.collect_all_vars(model)
        # Point vars share their transcribed instance with the
        # underlying infinite var's per-support transcription, so
        # skipping them here doesn't lose any transcribed→master
        # mappings.
        _is_point_var(v) && continue
        master_var = ref_map[v]
        transcribed = InfiniteOpt.transformation_variable(v)
        prefs = InfiniteOpt.parameter_refs(v)
        if isempty(prefs)
            result[transcribed] = master_var
        else
            supports = vec(InfiniteOpt.supports(only(prefs)))
            for (k, ref) in enumerate(vec(transcribed))
                result[ref] = master_var(supports[k])
            end
        end
    end
    return result
end

# Build the LOA master from `JuMP.copy_model(model)` and strip the
# nonlinear constraints (they re-enter via OA cuts). `binary_map[
# indicator]` and `variable_map[v]` hold single InfiniteOpt vars on
# the master; per-support handling happens downstream via point
# evaluation on those refs. OA cuts added in the LOA loop are
# point-evaluated scalar constraints on the master InfiniteModel;
# transcription is rebuilt before each master solve.
#
# Objective handling branches on `_has_aggregate_ref`. When the
# objective is aggregate-free (no MeasureRef / ParameterFunctionRef),
# `original_objective` is the InfiniteOpt objective itself and
# `objective_ref_map = ref_map`, so AD walks the InfiniteOpt
# expression directly. When it contains an aggregate, AD cannot see
# inside it; we fall back to transcribing the input model and using
# the flat scalar objective with a transcribed-to-master point map.
function DP.build_loa_master(
    model::InfiniteOpt.InfiniteModel, method::DP.LOA
    )
    master, copy_ref_map = JuMP.copy_model(model)
    # `copy_model` copies the GDP optimize-hook; clear it so
    # `optimize!(master)` doesn't re-trigger reformulation on the
    # (empty-GDP-data) master copy.
    JuMP.set_optimize_hook(master, nothing)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)

    # Strip nonlinear constraints; they re-enter via OA cuts.
    # Variable bounds (F=GeneralVariableRef) are kept by `copy_model`.
    # An aggregate-wrapped constraint (e.g. `∫(x^2,t) ≤ c`) is
    # structurally affine over a `MeasureRef` so `_is_linear_F` reads
    # it as linear, but it expands to a nonlinear form on
    # transcription — strip it too and let the OA cut re-add the
    # linearized version.
    variable_type = InfiniteOpt.GeneralVariableRef
    for (F, S) in JuMP.list_of_constraint_types(master)
        F === variable_type && continue
        is_linear = DP._is_linear_F(F)
        for cref in JuMP.all_constraints(master, F, S)
            if is_linear
                con = JuMP.constraint_object(cref)
                _has_aggregate_ref(con.func) || continue
            end
            JuMP.delete(master, cref)
        end
    end

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

    return (model = master, binary_map = binary_map,
        variable_map = variable_map,
        objective_sense = objective_sense,
        original_objective = original_objective,
        alpha_oa = alpha_oa,
        objective_ref_map = objective_ref_map)
end

# Override the disjunct-cut loop for `InfiniteModel`. Same shape as
# Override `add_oa_cuts` for `InfiniteModel` to translate the
# linearization point into the form the master's `original_objective`
# expects, and to route global OA cuts through transcription so they
# work over infinite vars and aggregate refs. The base
# `result.linearization_point` has per-support `Vector` values keyed
# on InfiniteOpt vars; the master's objective is either the raw
# InfiniteOpt expression (non-aggregate, expects scalar `xk[v]`) or
# the transcribed flat scalar expression (aggregate, expects
# transcribed-`JuMP.VariableRef`-keyed scalar dict). The translation
# produces whichever shape `_linearize_at` needs.
function DP.add_oa_cuts(
    model::InfiniteOpt.InfiniteModel,
    master::NamedTuple,
    result::NamedTuple,
    method::DP.LOA
    )
    isempty(result.linearization_point) && return
    obj_point = if _has_aggregate_ref(JuMP.objective_function(model))
        _transcribe_linearization_point(
            model, result.linearization_point)
    else
        T = eltype(valtype(result.linearization_point))
        Dict{InfiniteOpt.GeneralVariableRef, T}(
            var => values[1]
            for (var, values) in result.linearization_point
            if isempty(InfiniteOpt.parameter_refs(var)))
    end
    linearization = DP._linearize_at(master.original_objective,
        obj_point, master.objective_ref_map)
    DP._add_objective_cut(
        Val(master.objective_sense), master, linearization)
    _add_global_oa_cuts_infinite(model, master, result, method)
    DP.add_disjunct_oa_cuts(model, master, result, method)
    return
end

# Global OA cuts for `InfiniteModel`. Mirrors base `_add_global_oa_cuts`
# but routes through transcription so per-support / aggregate-ref
# expressions reach `_linearize_at` as flat scalars over `JuMP.VariableRef`s.
# Aggregate-containing constraints (e.g. `∫f(x,t)dt ≤ 0`) transcribe
# to a single scalar expression. Constraints with infinite-parameter
# dependence (e.g. `x(t) ≥ 0`) transcribe to a per-support `AbstractArray`
# of scalar expressions — one OA cut is emitted per support.
function _add_global_oa_cuts_infinite(
    model::InfiniteOpt.InfiniteModel,
    master::NamedTuple,
    result::NamedTuple,
    method::DP.LOA
    )
    _, penalty_sign = DP._disjunct_cut_coefficients(
        Val(master.objective_sense))
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
                    DP._add_global_oa_row!(master, lin, con.set,
                        method, penalty_sign)
                end
            else
                lin = DP._linearize_at(transcribed_func,
                    transcribed_xk[], transcribed_to_master[])
                DP._add_global_oa_row!(master, lin, con.set,
                    method, penalty_sign)
            end
        end
    end
    return
end

# Override `add_disjunct_oa_cuts` for `InfiniteModel`. Same shape as
# the base loop, but each constraint is checked for aggregate refs.
# Aggregate constraints (e.g. those containing a `MeasureRef`) are
# transcribed via `InfiniteOpt.transformation_expression`, then
# handed back to the base `_add_oa_cut_for_constraint` as a regular
# `JuMP.ScalarConstraint` over flat `JuMP.VariableRef`s — which the
# base AD pipeline can linearize correctly.
#
# `transcribed_to_master` and `transcribed_xk` are built lazily once
# per `add_disjunct_oa_cuts` call and shared across all aggregate
# constraints in the iteration.
function DP.add_disjunct_oa_cuts(
    model::InfiniteOpt.InfiniteModel,
    master::NamedTuple,
    result::NamedTuple,
    method::DP.LOA
    )
    sign_factor, penalty_sign = DP._disjunct_cut_coefficients(
        Val(master.objective_sense))
    transcribed_to_master = Ref{Any}(nothing)
    transcribed_xk = Ref{Any}(nothing)
    ensure_transcribed = function ()
        transcribed_to_master[] === nothing || return
        InfiniteOpt.build_transformation_backend!(model)
        transcribed_to_master[] = _transcribed_to_master_point(
            model, master.variable_map)
        transcribed_xk[] = _transcribe_linearization_point(
            model, result.linearization_point)
        return
    end
    for (indicator, active) in result.combination
        DP.any_active(active) || continue
        haskey(master.binary_map, indicator) || continue
        haskey(DP._indicator_to_constraints(model), indicator) ||
            continue
        for orig_constraint_ref in
                DP._indicator_to_constraints(model)[indicator]
            orig_constraint_ref isa DP.DisjunctConstraintRef ||
                continue
            constraint = DP._disjunct_constraints(model)[
                JuMP.index(orig_constraint_ref)].constraint
            dual = get(result.duals, orig_constraint_ref, nothing)
            dual === nothing && continue
            if _has_aggregate_ref(constraint.func)
                ensure_transcribed()
                transcribed_func =
                    InfiniteOpt.transformation_expression(
                        constraint.func)
                transcribed_constraint = JuMP.ScalarConstraint(
                    transcribed_func, constraint.set)
                for (binary_ref, _, _, dual_value) in DP.cut_info(
                    master.binary_map[indicator], active,
                    result.linearization_point,
                    master.variable_map, dual)
                    DP._add_oa_cut_for_constraint(
                        transcribed_constraint, master, binary_ref,
                        transcribed_xk[], transcribed_to_master[],
                        dual_value, method, sign_factor,
                        penalty_sign)
                end
                continue
            end
            for (binary_ref, linearization_point, var_map,
                    dual_value) in DP.cut_info(
                    master.binary_map[indicator], active,
                    result.linearization_point,
                    master.variable_map, dual)
                DP._add_oa_cut_for_constraint(
                    constraint, master, binary_ref,
                    linearization_point, var_map, dual_value,
                    method, sign_factor, penalty_sign)
            end
        end
    end
end

# Fix indicators for `combination`, run `f()`, then undo. Bool values
# are whole-var fixes via `JuMP.fix`; AbstractVector values are
# per-support point-equality constraints. Refs and fixed binaries are
# tracked in local state so cleanup runs from the same closure — no
# `model.ext` stash.
function DP.with_fixed_combination(
    f,
    model::InfiniteOpt.InfiniteModel,
    combination::AbstractDict
    )
    constraint_refs = InfiniteOpt.InfOptConstraintRef[]
    fixed_binaries = InfiniteOpt.GeneralVariableRef[]
    for (indicator, value) in combination
        binary_ref = DP._indicator_to_binary(model)[indicator]
        _apply_fix!(
            constraint_refs, fixed_binaries, model, binary_ref, value)
    end
    try
        return f()
    finally
        for ref in constraint_refs
            JuMP.is_valid(model, ref) && JuMP.delete(model, ref)
        end
        for binary_ref in fixed_binaries
            JuMP.is_fixed(binary_ref) && JuMP.unfix(binary_ref)
        end
    end
end

# Finalize the LOA-best combination + warm start, leaving the model
# in a state the post-hook `optimize!` will accept. Mirrors
# `with_fixed_combination` for the fixing half but skips bookkeeping
# (these fixes stay).
function DP.commit_combination(
    model::InfiniteOpt.InfiniteModel,
    combination::AbstractDict,
    linearization_point::AbstractDict
    )
    constraint_refs = InfiniteOpt.InfOptConstraintRef[]
    fixed_binaries = InfiniteOpt.GeneralVariableRef[]
    for (indicator, value) in combination
        binary_ref = DP._indicator_to_binary(model)[indicator]
        _apply_fix!(
            constraint_refs, fixed_binaries, model, binary_ref, value)
    end
    for (variable, values) in linearization_point
        _set_starts_for_transcribed(
            InfiniteOpt.transformation_variable(variable),
            values
            )
    end
    return
end

# Bool: fix the whole infinite var across all supports.
function _apply_fix!(
    ::AbstractVector,
    fixed::AbstractVector,
    ::InfiniteOpt.InfiniteModel,
    binary_ref::InfiniteOpt.GeneralVariableRef,
    value::Bool
    )
    JuMP.fix(binary_ref, value ? 1.0 : 0.0; force = true)
    push!(fixed, binary_ref)
    return
end

# Vector{Bool}: per-support point-equality constraints.
function _apply_fix!(
    refs::AbstractVector,
    ::AbstractVector,
    model::InfiniteOpt.InfiniteModel,
    binary_ref::InfiniteOpt.GeneralVariableRef,
    values::AbstractVector{Bool}
    )
    for (k, support) in enumerate(_supports_of(binary_ref))
        push!(refs, JuMP.@constraint(model,
            binary_ref(support) == (values[k] ? 1.0 : 0.0)))
    end
    return
end

# Infinite var: per-support transcribed array
function _set_starts_for_transcribed(
    transcribed::AbstractArray,
    values::AbstractVector
    )
    for (k, ref) in enumerate(vec(transcribed))
        JuMP.set_start_value(ref, values[k])
    end
    return
end

# Finite var: single transcribed ref, `values` is 1-element
_set_starts_for_transcribed(
    transcribed::JuMP.AbstractVariableRef,
    values::AbstractVector
    ) =
    JuMP.set_start_value(transcribed, values[1])

# Finite var: single transcribed ref, scalar value (feas-side path)
_set_starts_for_transcribed(
    transcribed::JuMP.AbstractVariableRef,
    value::Real
    ) =
    JuMP.set_start_value(transcribed, value)

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

# Finite var: single transcribed ref, scalar value (feas-side path)
_add_to_transcribed_dict(
    d::AbstractDict,
    ts::JuMP.AbstractVariableRef,
    value::Real
    ) =
    (d[ts] = value; nothing)

end
