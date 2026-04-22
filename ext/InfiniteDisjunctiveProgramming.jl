module InfiniteDisjunctiveProgramming

import JuMP.MOI as _MOI
import InfiniteOpt, JuMP, Interpolations
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

# Build a mini InfiniteModel holding only the given disjunct constraints,
# transcribe it, and return as a GDPSubmodel.
function DP.copy_model_with_constraints(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP._MBM
    )
    mini = InfiniteOpt.InfiniteModel()
    ref_map = Dict{InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()

    # 1. Copy infinite parameters with their supports
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        supports = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(error, domain; supports = supports)
        new_param = InfiniteOpt.add_parameter(mini, param, JuMP.name(p))
        ref_map[p] = new_param
    end

    # 2. Copy decision variables with bounds
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        var_type = isempty(prefs) ? nothing :
            InfiniteOpt.Infinite(Tuple(ref_map[p] for p in prefs)...)
        props = DP.VariableProperties(
            DP.get_variable_info(v), "", nothing, var_type)
        ref_map[v] = DP.create_variable(mini, props)
    end

    # 3. Copy derivatives with their bounds
    for d in InfiniteOpt.all_derivatives(model)
        vref = InfiniteOpt.derivative_argument(d)
        pref = InfiniteOpt.operator_parameter(d)
        new_d = InfiniteOpt.deriv(ref_map[vref], ref_map[pref])
        info = DP.get_variable_info(d)
        info.has_lb && JuMP.set_lower_bound(new_d, info.lower_bound)
        info.has_ub && JuMP.set_upper_bound(new_d, info.upper_bound)
        ref_map[d] = new_d
    end

    # 4. Copy parameter functions (needed by ref_map substitution)
    for pfunc in InfiniteOpt.all_parameter_functions(model)
        func = InfiniteOpt.raw_function(pfunc)
        prefs = InfiniteOpt.parameter_refs(pfunc)
        mapped_prefs = Tuple(ref_map[p] for p in prefs)
        pref_arg = length(mapped_prefs) == 1 ?
            only(mapped_prefs) : mapped_prefs
        param_func = InfiniteOpt.build_parameter_function(
            error, func, pref_arg)
        ref_map[pfunc] = InfiniteOpt.add_parameter_function(
            mini, param_func)
    end

    # 5. Add disjunct constraints using existing ref_map
    for cref in constraints
        cref isa DP.DisjunctConstraintRef || continue
        con = JuMP.constraint_object(cref)
        new_func = DP._replace_variables_in_constraint(con.func, ref_map)
        T = one(JuMP.value_type(typeof(mini)))
        JuMP.@constraint(mini, new_func * T in con.set)
    end

    # 6. Transcribe mini InfiniteModel
    InfiniteOpt.build_transformation_backend!(mini)
    transcribed = InfiniteOpt.transformation_model(mini)
    JuMP.set_optimizer(transcribed, method.optimizer)
    JuMP.set_silent(transcribed)
    # Stash for prepare_max_M_objective / raw_M.
    transcribed.ext[:inf_mbm_main] = model
    transcribed.ext[:inf_mbm_ref_map] = ref_map
    # GDPSubmodel's fwd_map / decision_vars are CP-only; unused here.
    return DP.GDPSubmodel(transcribed, InfiniteOpt.GeneralVariableRef[],
        Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}())
end

function DP.prepare_max_M_objective(
    ::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.LessThan}
    ref_map = sub.model.ext[:inf_mbm_ref_map]
    mini_expr = DP._replace_variables_in_constraint(
        obj.func, ref_map) - obj.set.upper
    sub.model.ext[:inf_mbm_obj_expr] = obj.func
    return InfiniteOpt.transformation_expression(mini_expr)
end

function DP.prepare_max_M_objective(
    ::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.GreaterThan}
    ref_map = sub.model.ext[:inf_mbm_ref_map]
    mini_expr = obj.set.lower - DP._replace_variables_in_constraint(
        obj.func, ref_map)
    sub.model.ext[:inf_mbm_obj_expr] = obj.func
    return InfiniteOpt.transformation_expression(mini_expr)
end

# Per-support solve, delegating to scalar base raw_M. Aggregated to a
# scalar if uniform, else to a parameter function.
function DP.raw_M(
    sub::DP.GDPSubmodel,
    objectives::AbstractArray{<:Union{JuMP.AbstractJuMPScalar, Real}},
    method::DP._MBM
    )
    M_vals = similar(objectives, typeof(method.default_M))
    for I in eachindex(objectives)
        JuMP.set_start_value.(JuMP.all_variables(sub.model), nothing)
        m = DP.raw_M(sub, objectives[I], method)
        m === nothing && return nothing
        M_vals[I] = m
    end
    all(==(first(M_vals)), M_vals) && return first(M_vals)
    main = sub.model.ext[:inf_mbm_main]
    expr = sub.model.ext[:inf_mbm_obj_expr]
    prefs = InfiniteOpt.parameter_refs(expr)
    grids = Tuple(InfiniteOpt.supports(p) for p in prefs)
    interp = Interpolations.linear_interpolation(grids, M_vals)
    param_func = InfiniteOpt.build_parameter_function(
        error, (args...) -> interp(args...), prefs)
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
    JuMP.set_silent(sub.model)
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
# Dispatch overrides for InfiniteModel. The base LOA algorithm in src/loa.jl
# is written for finite (scalar) models; these overrides handle transcription
# to a flat master and per-support binary/variable indexing.

# Helper: map an InfiniteOpt var to its flat master var(s) via transcription +
# copy_map. Returns a vector for infinite vars, scalar for finite.
function _tv_map(v, copy_map)
    tv = InfiniteOpt.transformation_variable(v)
    return tv isa AbstractArray ? [copy_map[fv] for fv in vec(tv)] : copy_map[tv]
end

# Convert InfiniteModel x_values to flat-var x_values (for objective OA cut).
function _flat_xk(model::InfiniteOpt.InfiniteModel, x_values)
    fxk = Dict{JuMP.VariableRef, Float64}()
    for (v, val) in x_values
        tv = InfiniteOpt.transformation_variable(v)
        if tv isa AbstractArray
            vals = val isa AbstractVector ? val : fill(Float64(val), length(tv))
            for (i, fv) in enumerate(vec(tv))
                fxk[fv] = vals[i]
            end
        else
            fxk[tv] = val isa Number ? Float64(val) : Float64(first(val))
        end
    end
    return fxk
end

#per-support dispatch methods: the base DP.jl LOA code calls scalar-form
#helpers; these array methods let the same code path handle vector-valued
#bin_map / var_map / x_values / active entries without any extra branches
#in the base file.

DP.fix_fv(bvs::AbstractArray, val::Bool) =
    (for bv in bvs; DP.fix_fv(bv, val); end; return)
DP.fix_fv(bvs::AbstractArray, val::AbstractArray) =
    (for (bv, v) in zip(bvs, val); DP.fix_fv(bv, v); end; return)
DP.unfix_fv(bvs::AbstractArray) =
    (for bv in bvs; DP.unfix_fv(bv); end; return)

DP.any_active(v::AbstractVector{Bool}) = any(v)

#combo extraction: round per-support binary values to a Vector{Bool}
DP.combo_val(bvs::AbstractArray) = Bool.(round.(JuMP.value.(bvs)))

#no-good cut: fold one scalar term per (binary, active) pair. The scalar-
#active method handles the set-covering phase where combos are Bool-valued.
DP.add_ng_terms(cut, bvs::AbstractArray, active::Bool) =
    DP.add_ng_terms(cut, bvs, fill(active, length(bvs)))
function DP.add_ng_terms(cut, bvs::AbstractArray, actives::AbstractArray)
    for (bv, a) in zip(bvs, actives)
        DP.add_ng_terms(cut, bv, a)
    end
    return
end

#OA cut sites: one site per active support, with per-support restrictions
#of `x_values` and `var_map`
DP.cut_sites(bvs::AbstractArray, active::Bool, x_values, var_map, d) =
    DP.cut_sites(bvs, fill(active, length(bvs)), x_values, var_map, d)
function DP.cut_sites(
    bvs::AbstractArray, actives::AbstractArray,
    x_values, var_map, d
    )
    sites = Any[]
    for k in 1:length(bvs)
        actives[k] || continue
        smap_k = Dict{Any, Any}(
            v => (mv isa AbstractVector ? mv[k] : mv)
            for (v, mv) in var_map)
        x_k = Dict{Any, Any}(
            v => (xv isa AbstractVector ? xv[k] : xv)
            for (v, xv) in x_values)
        d_k = d isa AbstractVector ? d[k] : d
        push!(sites, (bvs[k], x_k, smap_k, d_k))
    end
    return sites
end

#detect number of supports from a bin_map; used below in `build_loa_master`
function _detect_K(bin_map)
    for (_, bvs) in bin_map
        bvs isa AbstractVector && return length(bvs)
    end
    return 1
end

#transcribe the BigM'd InfiniteModel to flat, copy it, create alpha_oa.
#Number of supports K is stashed in `master.model.ext[:_loa_K]` for the
#per-support OA cut and combo overrides below.
function DP.build_loa_master(model::InfiniteOpt.InfiniteModel, method::DP.LOA)
    InfiniteOpt.build_transformation_backend!(model)
    flat = InfiniteOpt.transformation_model(model)
    orig_obj = JuMP.objective_function(flat)
    master, copy_map = JuMP.copy_model(flat)
    JuMP.set_optimizer(master, method.mip_optimizer)
    JuMP.set_silent(master)
    bin_map = Dict{DP.LogicalVariableRef, Any}()
    for (ind, bv) in DP._indicator_to_binary(model)
        bin_map[ind] = _tv_map(bv, copy_map)
    end
    var_map = Dict{InfiniteOpt.GeneralVariableRef, Any}()
    for v in DP.collect_all_vars(model)
        var_map[v] = _tv_map(v, copy_map)
    end
    #also store a flat→master copy_map for objective OA cuts
    flat_copy_map = Dict{JuMP.VariableRef, JuMP.VariableRef}()
    for v in JuMP.all_variables(flat)
        flat_copy_map[v] = copy_map[v]
    end
    obj_sense = JuMP.objective_sense(master)
    alpha_oa = JuMP.@variable(master, base_name = "alpha_oa")
    JuMP.@objective(master, obj_sense, alpha_oa)
    m = DP._LOAMaster(master, bin_map, var_map, obj_sense, orig_obj,
        alpha_oa, flat_copy_map)
    master.ext[:_loa_K] = _detect_K(bin_map)
    master.ext[:_loa_flat_copy_map] = flat_copy_map
    return m
end

#fix per-support via point equality constraints
function DP.fix_combo_binaries(model::InfiniteOpt.InfiniteModel, combo)
    crefs = InfiniteOpt.InfOptConstraintRef[]
    for (ind, val) in combo
        bv = DP._indicator_to_binary(model)[ind]
        if val isa Bool
            JuMP.fix(bv, val ? 1.0 : 0.0; force = true)
        else
            sups = InfiniteOpt.supports(first(InfiniteOpt.parameter_refs(bv)))
            for (k, s) in enumerate(vec(sups))
                push!(crefs,
                    JuMP.@constraint(model, bv(s) == (val[k] ? 1.0 : 0.0)))
            end
        end
    end
    model.ext[:_loa_fix_crefs] = crefs
end

function DP.unfix_combo_binaries(model::InfiniteOpt.InfiniteModel, combo)
    if haskey(model.ext, :_loa_fix_crefs)
        for c in model.ext[:_loa_fix_crefs]
            JuMP.is_valid(model, c) && JuMP.delete(model, c)
        end
        delete!(model.ext, :_loa_fix_crefs)
    end
    for (ind, val) in combo
        val isa Bool || continue
        bv = DP._indicator_to_binary(model)[ind]
        JuMP.is_fixed(bv) && JuMP.unfix(bv)
    end
end

#Transcribe the BigM'd InfiniteModel, then hand off to the base
#`copy_model_with_constraints` on the flat model. fwd_map is rekeyed to
#InfiniteModel vars for use by `_extract_*_x_values`; obj_ref_map stays
#keyed by flat vars (the flat-level objective OA cut lives there).
function DP.copy_model_with_constraints(
    model::InfiniteOpt.InfiniteModel, method::DP.LOA
    )
    InfiniteOpt.build_transformation_backend!(model)
    flat = InfiniteOpt.transformation_model(model)
    base = DP.copy_model_with_constraints(flat, method)
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef, Any}()
    for v in DP.collect_all_vars(model)
        fwd_map[v] = _tv_map(v, base.fwd_map)
    end
    return DP._LOAFeasSubmodel(base.model, fwd_map, base.fwd_map)
end

#read per-support JuMP values from either a vector of flat vars or a
#single scalar. Dispatch handles both transformation_variable (may be
#N-dim) and fwd_map (already flat) shapes.
_read_values(v::AbstractArray) = Float64[JuMP.value(fv) for fv in vec(v)]
_read_values(v) = Float64(JuMP.value(v))

#extract per-support x-values from the InfiniteModel NLP; objective
#x-values are keyed by the flat transcription vars for the objective cut
function DP.extract_primary_x_values(model::InfiniteOpt.InfiniteModel)
    x_vals = Dict{JuMP.AbstractVariableRef, Any}()
    for v in DP.collect_all_vars(model)
        JuMP.is_fixed(v) && continue
        x_vals[v] = _read_values(InfiniteOpt.transformation_variable(v))
    end
    return x_vals, _flat_xk(model, x_vals)
end

#extract per-support x-values from the flat feas submodel. x_vals keys are
#InfiniteModel vars (via fwd_map); obj_x_values keys are flat vars.
function DP.extract_feas_x_values(
    model::InfiniteOpt.InfiniteModel, feas::DP._LOAFeasSubmodel
    )
    x_vals = Dict{JuMP.AbstractVariableRef, Any}()
    for v in DP.collect_all_vars(model)
        JuMP.is_fixed(v) && continue
        haskey(feas.fwd_map, v) || continue
        x_vals[v] = _read_values(feas.fwd_map[v])
    end
    obj_xv = Dict{JuMP.VariableRef, Float64}()
    for (flat_v, feas_v) in feas.obj_ref_map
        obj_xv[flat_v] = Float64(JuMP.value(feas_v))
    end
    return x_vals, obj_xv
end

end
