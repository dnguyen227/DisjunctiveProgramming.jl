module InfiniteDisjunctiveProgramming

import JuMP.MOI as _MOI
import InfiniteOpt, JuMP, Interpolations
import DisjunctiveProgramming as DP

################################################################################
#                                   MODEL
################################################################################
function DP.InfiniteGDPModel(args...; kwargs...)
    return DP.GDPModel{InfiniteOpt.InfiniteModel,
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.InfOptConstraintRef}(args...; kwargs...)
end

function DP.collect_all_vars(model::InfiniteOpt.InfiniteModel)
    vars = JuMP.all_variables(model)
    return append!(vars, InfiniteOpt.all_derivatives(model))
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
    prefs = InfiniteOpt.parameter_refs(vref)
    var_type = !isempty(prefs) ? InfiniteOpt.Infinite(prefs...) : nothing
    return DP.VariableProperties(info, name, nothing, var_type)
end

# Extract parameter refs from expression, return VariableProperties
# with Infinite type
function DP.VariableProperties(
    expr::Union{
        JuMP.GenericAffExpr{C, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericQuadExpr{C, InfiniteOpt.GeneralVariableRef},
        JuMP.GenericNonlinearExpr{InfiniteOpt.GeneralVariableRef}}
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
        JuMP.GenericNonlinearExpr{InfiniteOpt.GeneralVariableRef}}}
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
    c::JuMP.VectorConstraint{F, S}, name::String = ""
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
    error("Cannot define constraint on single logical variable, " *
          "use `fix` instead.")
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{
        JuMP.GenericAffExpr{C, DP.LogicalVariableRef{M}}, S},
    name::String = ""
    ) where {M <: InfiniteOpt.InfiniteModel, S, C}
    error("Cannot add, subtract, or multiply with " *
          "logical variables.")
end

function JuMP.add_constraint(
    model::M,
    c::JuMP.ScalarConstraint{
        JuMP.GenericQuadExpr{C, DP.LogicalVariableRef{M}}, S},
    name::String = ""
    ) where {M <: InfiniteOpt.InfiniteModel, S, C}
    error("Cannot add, subtract, or multiply with " *
          "logical variables.")
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
    model::M, aff::JuMP.GenericAffExpr,
    bvref::Union{JuMP.AbstractVariableRef, JuMP.GenericAffExpr},
    method::DP._Hull
    ) where {M <: InfiniteOpt.InfiniteModel}
    terms = Any[aff.constant * bvref]
    for (vref, coeff) in aff.terms
        if JuMP.is_binary(vref)
            push!(terms, coeff * vref)
        elseif vref isa InfiniteOpt.GeneralVariableRef &&
               _is_parameter(vref)
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

# Quadratic expression: handle parameter × parameter, parameter × variable,
# and variable × variable terms.
function DP.disaggregate_expression(
    model::M, quad::JuMP.GenericQuadExpr,
    bvref::Union{JuMP.AbstractVariableRef, JuMP.GenericAffExpr},
    method::DP._Hull
    ) where {M <: InfiniteOpt.InfiniteModel}
    # Affine part (uses InfiniteOpt override above)
    new_expr = DP.disaggregate_expression(model, quad.aff, bvref, method)
    ϵ = method.value
    for (pair, coeff) in quad.terms
        a_param = pair.a isa InfiniteOpt.GeneralVariableRef &&
            _is_parameter(pair.a)
        b_param = pair.b isa InfiniteOpt.GeneralVariableRef &&
            _is_parameter(pair.b)
        if a_param && b_param
            # param × param: constant, scale by y
            new_expr += coeff * pair.a * pair.b * bvref
        elseif a_param
            # param × var: perspective cancels y
            db = method.disjunct_variables[pair.b, bvref]
            new_expr += coeff * pair.a * db
        elseif b_param
            # var × param: perspective cancels y
            da = method.disjunct_variables[pair.a, bvref]
            new_expr += coeff * da * pair.b
        else
            # var × var: standard perspective
            da = method.disjunct_variables[pair.a, bvref]
            db = method.disjunct_variables[pair.b, bvref]
            new_expr += coeff * da * db / ((1 - ϵ) * bvref + ϵ)
        end
    end
    return new_expr
end

################################################################################
#                          MBM FOR INFINITEMODEL
################################################################################
# Reuses the finite MBM infrastructure by overriding:
#   create_submodel  — build mini InfiniteModel +
#       transcribe to flat JuMP model
#   _prepare_objectives — expand infinite constraint
#       into K flat objectives via _build_flat_map
#   condense_values — interpolate flat values ->
#       parameter function (shared with CP)

# Collect all parameter function refs from all disjunct
# constraints in the model.
function _all_param_functions(
    model::InfiniteOpt.InfiniteModel
    )
    pf_set = Set{InfiniteOpt.GeneralVariableRef}()
    for (_, crefs) in DP._indicator_to_constraints(model)
        for cref in crefs
            cref isa DP.DisjunctConstraintRef || continue
            con = JuMP.constraint_object(cref)
            for v in InfiniteOpt.all_expression_variables(
                    con.func)
                dv = InfiniteOpt.dispatch_variable_ref(v)
                if dv isa InfiniteOpt.ParameterFunctionRef
                    push!(pf_set, v)
                end
            end
        end
    end
    return pf_set
end

# Build a flat map for support point k. Maps decision
# variables to their flat JuMP.VariableRef at support k
# (handling multi-parameter indexing) and evaluates
# parameter functions to their numerical values.
function _build_flat_map(
    model::InfiniteOpt.InfiniteModel,
    sub::DP.GDPSubmodel, k::Int,
    prefs, supports, full_shape
    )
    ci = CartesianIndices(full_shape)[k]

    # Decision variables: variable-local index
    flat_map = Dict{InfiniteOpt.GeneralVariableRef, Any}()
    for (v, ws) in sub.fwd
        if length(ws) == 1
            flat_map[v] = ws[1]
        else
            vprefs = InfiniteOpt.parameter_refs(v)
            var_shape = Tuple(
                length(supports[p]) for p in vprefs)
            var_idx = Tuple(
                ci[findfirst(==(p), prefs)]
                for p in vprefs)
            local_k = LinearIndices(
                var_shape)[var_idx...]
            flat_map[v] = ws[local_k]
        end
    end

    # Parameter functions: evaluate at support k
    sup_vals = Dict(
        prefs[i] => supports[prefs[i]][ci[i]]
        for i in 1:length(prefs))
    for pf in _all_param_functions(model)
        fn = InfiniteOpt.raw_function(pf)
        pf_prefs = InfiniteOpt.parameter_refs(pf)
        pf_vals = Tuple(sup_vals[p] for p in pf_prefs)
        flat_map[pf] = fn(pf_vals...)
    end
    return flat_map
end

# Build mini InfiniteModel with only the given disjunct
# constraints, transcribe to flat JuMP model, return
# GDPSubmodel with forward map.
function DP.create_submodel(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP._MBM
    )
    mini = InfiniteOpt.InfiniteModel()
    ref_map = Dict{InfiniteOpt.GeneralVariableRef,
                   InfiniteOpt.GeneralVariableRef}()

    # 1. Copy infinite parameters
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        sups = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(
            error, domain; supports = sups)
        new_p = InfiniteOpt.add_parameter(
            mini, param, JuMP.name(p))
        ref_map[p] = new_p
    end

    # 2. Copy decision variables (skip parameters)
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        info = DP.get_variable_info(v)
        prefs = InfiniteOpt.parameter_refs(v)
        if isempty(prefs)
            new_v = JuMP.@variable(mini)
        else
            mapped = Tuple(ref_map[p] for p in prefs)
            new_v = JuMP.@variable(mini,
                variable_type =
                    InfiniteOpt.Infinite(mapped...))
        end
        if info.has_lb
            JuMP.set_lower_bound(new_v, info.lower_bound)
        end
        if info.has_ub
            JuMP.set_upper_bound(new_v, info.upper_bound)
        end
        if info.binary
            JuMP.set_binary(new_v)
        end
        if info.integer
            JuMP.set_integer(new_v)
        end
        ref_map[v] = new_v
    end

    # 3. Copy derivatives
    for d in InfiniteOpt.all_derivatives(model)
        vref = InfiniteOpt.derivative_argument(d)
        pref = InfiniteOpt.operator_parameter(d)
        new_d = InfiniteOpt.deriv(
            ref_map[vref], ref_map[pref])
        info = DP.get_variable_info(d)
        if info.has_lb
            JuMP.set_lower_bound(new_d, info.lower_bound)
        end
        if info.has_ub
            JuMP.set_upper_bound(new_d, info.upper_bound)
        end
        ref_map[d] = new_d
    end

    # 4. Copy parameter functions from ALL disjuncts
    # (needed for constraint transcription)
    pf_set = _all_param_functions(model)
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        prefs = InfiniteOpt.parameter_refs(pf)
        mapped_prefs = Tuple(
            ref_map[p] for p in prefs)
        new_pf = _make_parameter_function(
            mini, fn, mapped_prefs...)
        ref_map[pf] = new_pf
    end

    # 5. Add constraints using existing remap
    for cref in constraints
        cref isa DP.DisjunctConstraintRef || continue
        con = JuMP.constraint_object(cref)
        new_func = DP._replace_variables_in_constraint(
            con.func, ref_map)
        T = one(JuMP.value_type(typeof(mini)))
        JuMP.@constraint(mini, new_func * T in con.set)
    end

    # 6. Transcribe to flat JuMP model
    tr = transcribe_to_flat(mini)

    # 7. Remap fwd: orig var → flat VarRef
    fwd = Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}()
    for (orig, mapped) in ref_map
        _is_parameter(orig) && continue
        haskey(tr.fwd, mapped) || continue
        fwd[orig] = tr.fwd[mapped]
    end

    dec_vars = collect(keys(fwd))
    JuMP.set_optimizer(
        tr.flat_model, method.optimizer)
    JuMP.set_silent(tr.flat_model)
    return DP.GDPSubmodel(
        tr.flat_model, dec_vars, fwd)
end

# Prepare objectives for all support points. Expands
# an infinite constraint into K flat objectives via
# _build_flat_map with multi-parameter indexing and
# parameter function evaluation.
function DP._prepare_objectives(
    model::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.LessThan}
    prefs, supports = _collect_parameters(model)
    full_shape = Tuple(
        length(supports[p]) for p in prefs)
    K = prod(full_shape)
    objectives = Vector{Any}(undef, K)
    for k in 1:K
        flat_map = _build_flat_map(
            model, sub, k,
            prefs, supports, full_shape)
        objectives[k] = -obj.set.upper +
            DP._replace_variables_in_constraint(
                obj.func, flat_map)
    end
    return objectives
end
function DP._prepare_objectives(
    model::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.GreaterThan}
    prefs, supports = _collect_parameters(model)
    full_shape = Tuple(
        length(supports[p]) for p in prefs)
    K = prod(full_shape)
    objectives = Vector{Any}(undef, K)
    for k in 1:K
        flat_map = _build_flat_map(
            model, sub, k,
            prefs, supports, full_shape)
        objectives[k] = obj.set.lower -
            DP._replace_variables_in_constraint(
                obj.func, flat_map)
    end
    return objectives
end

# Condense flat per-support values to final form.
# Uses all model parameters (MBM path).
function DP.condense_values(
    model::InfiniteOpt.InfiniteModel,
    vals::AbstractVector{<:Real}
    )
    if all(==(vals[1]), vals)
        return vals[1]
    end
    prefs, supports = _collect_parameters(model)
    return condense_to_pf(
        model, vals, prefs, supports)
end

# Interpolate a flat vector of per-support values into
# a parameter function.
function _vals_to_pf(
    model::InfiniteOpt.InfiniteModel,
    vals::AbstractVector{<:Real}, prefs,
    grids::Tuple, shape::Tuple
    )
    nd = reshape(vals, shape)
    fn = _linear_interpolation(grids, nd)
    return _make_parameter_function(model, fn, prefs...)
end

# Shared grids/shape computation + _vals_to_pf call.
# Caller decides parameter scope (all model params vs
# variable-specific).
function condense_to_pf(
    model::InfiniteOpt.InfiniteModel,
    vals::AbstractVector{<:Real},
    prefs, supports
    )
    grids = Tuple(supports[p] for p in prefs)
    shape = Tuple(
        length(supports[p]) for p in prefs)
    return _vals_to_pf(
        model, vals, prefs, grids, shape)
end

################################################################################
#                            TRANSCRIPTION MAP
################################################################################

# Transcription result type alias. transcribe() returns
# a NamedTuple with these three fields.
const TMap = NamedTuple{
    (:inf_model, :flat_model, :fwd),
    Tuple{InfiniteOpt.InfiniteModel, JuMP.Model,
          Dict{InfiniteOpt.GeneralVariableRef,
               Vector{JuMP.VariableRef}}}}

# Create a parameter function programmatically. Uses build_parameter_function
# + add_parameter_function (the lower-level API behind @parameter_function)
# since the macro doesn't support programmatic use. The closure wrapper
# handles non-Function callables like Interpolations.Extrapolation.
# Accepts any number of prefs via varargs:
#   _make_parameter_function(m, f, t)     -> 1D
#   _make_parameter_function(m, f, t, x)  -> 2D
function _make_parameter_function(
    model::InfiniteOpt.InfiniteModel, fn,
    prefs::InfiniteOpt.GeneralVariableRef...
    )
    f = fn isa Function ? fn : ((args...) -> fn(args...))
    pref_arg = length(prefs) == 1 ? prefs[1] : prefs
    pfunc = InfiniteOpt.build_parameter_function(error, f, pref_arg)
    return InfiniteOpt.add_parameter_function(model, pfunc)
end

# Collect all infinite parameters and their supports.
function _collect_parameters(model::InfiniteOpt.InfiniteModel)
    params = collect(InfiniteOpt.all_parameters(model))
    if isempty(params)
        error("Model has no infinite parameters.")
    end
    prefs = InfiniteOpt.GeneralVariableRef[p for p in params]
    supports = Dict{InfiniteOpt.GeneralVariableRef, Vector{Float64}}(
        p => Float64.(InfiniteOpt.supports(p)) for p in prefs)
    return prefs, supports
end

# Transcribe an InfiniteModel to a flat JuMP.Model with
# forward variable map. Shared by MBM and CP paths.
function transcribe_to_flat(
    model::InfiniteOpt.InfiniteModel
    )
    prefs, supports = _collect_parameters(model)
    InfiniteOpt.build_transformation_backend!(model)
    flat = InfiniteOpt.transformation_model(model)
    inf_vars, fin_vars =
        _collect_decision_vars(model)
    fwd = Dict{InfiniteOpt.GeneralVariableRef,
               Vector{JuMP.VariableRef}}()
    for var in inf_vars
        tv = InfiniteOpt.transformation_variable(
            var)
        fwd[var] = vec(tv)
    end
    for var in fin_vars
        tv = InfiniteOpt.transformation_variable(
            var)
        fwd[var] = [tv]
    end
    return (flat_model = flat, fwd = fwd,
            prefs = prefs, supports = supports)
end

function DP.transcribe(
    model::InfiniteOpt.InfiniteModel;
    method::DP.AbstractReformulationMethod = DP.BigM()
    )
    DP.reformulate_model(model, method)
    tr = transcribe_to_flat(model)
    sub_copy, copy_map = JuMP.copy_model(
        tr.flat_model)
    fwd = Dict{InfiniteOpt.GeneralVariableRef,
               Vector{JuMP.VariableRef}}()
    for (var, tvars) in tr.fwd
        fwd[var] = [copy_map[tv] for tv in tvars]
    end
    return (inf_model = model,
            flat_model = sub_copy, fwd = fwd)
end

function DP.jump_model(tmap::TMap)
    return tmap.flat_model
end

function DP.transcribed_variable(
    tmap::TMap,
    var::InfiniteOpt.GeneralVariableRef
    )
    haskey(tmap.fwd, var) || error(
        "Variable not found in transcription.")
    return tmap.fwd[var]
end

# O(n) scan — only used in tests/debugging.
function DP.infinite_variable(
    tmap::TMap, flat_var::JuMP.VariableRef
    )
    for (var, tvars) in tmap.fwd
        for (k, tv) in enumerate(tvars)
            tv == flat_var && return (var, k)
        end
    end
    error("Variable not found in transcription.")
end

function DP.support_values(tmap::TMap)
    _, supports = _collect_parameters(
        tmap.inf_model)
    return supports
end

function DP.lift_constraint(
    tmap::TMap,
    terms::Vector{<:Tuple{InfiniteOpt.GeneralVariableRef,
        Vector{Float64}}},
    sense::Symbol, rhs::Vector{Float64}
    )
    model = tmap.inf_model
    flat = tmap.flat_model
    prefs, supports = _collect_parameters(model)

    full_shape = Tuple(
        length(supports[p]) for p in prefs)
    K = prod(full_shape)
    full_ci = CartesianIndices(full_shape)

    # 1. Add K constraints to the flat model
    for k in 1:K
        expr = JuMP.AffExpr(0.0)
        ci = full_ci[k]
        for (var, coeffs) in terms
            haskey(tmap.fwd, var) || continue
            jvars = tmap.fwd[var]
            jk = _global_to_var_index(
                ci, var, jvars,
                prefs, supports)
            JuMP.add_to_expression!(
                expr, coeffs[k], jvars[jk])
        end
        if sense == :>=
            JuMP.@constraint(flat, expr >= rhs[k])
        elseif sense == :<=
            JuMP.@constraint(flat, expr <= rhs[k])
        elseif sense == :(==)
            JuMP.@constraint(flat, expr == rhs[k])
        end
    end

    # 2. Add 1 infinite constraint via interpolation
    cut_terms = Any[]
    for (var, coeffs) in terms
        vprefs = InfiniteOpt.parameter_refs(var)
        if isempty(vprefs)
            push!(cut_terms, coeffs[1] * var)
        else
            c_pf = condense_to_pf(
                model, coeffs, prefs, supports)
            push!(cut_terms, c_pf * var)
        end
    end

    rhs_pf = condense_to_pf(
        model, rhs, prefs, supports)

    if !isempty(cut_terms)
        lhs = JuMP.@expression(
            model, sum(cut_terms))
        if sense == :>=
            JuMP.@constraint(model, lhs >= rhs_pf)
        elseif sense == :<=
            JuMP.@constraint(model, lhs <= rhs_pf)
        elseif sense == :(==)
            JuMP.@constraint(model, lhs == rhs_pf)
        end
    end
    return
end

# Map a full-grid CartesianIndex to a variable's own
# flat index based on which parameters it depends on.
function _global_to_var_index(
    ci::CartesianIndex,
    var::InfiniteOpt.GeneralVariableRef,
    jvars::Vector{JuMP.VariableRef},
    prefs::Vector{InfiniteOpt.GeneralVariableRef},
    supports::Dict{InfiniteOpt.GeneralVariableRef,
                   Vector{Float64}}
    )
    if length(jvars) == 1
        return 1  # finite variable
    end
    var_prefs = InfiniteOpt.parameter_refs(var)
    var_shape = Tuple(
        length(supports[p]) for p in var_prefs)
    var_idx = Tuple(
        ci[findfirst(==(p), prefs)]
        for p in var_prefs)
    return LinearIndices(var_shape)[var_idx...]
end

################################################################################
#                    CUTTING PLANES FOR INFINITEMODEL
################################################################################

# Linear interpolation via Interpolations.jl, consistent with InfiniteOpt's
# InfiniteInterpolations extension. Uses linear extrapolation at boundaries.
# Handles both 1-D and N-D grids:
#   1D: grids = (t_vals,), y_vals = Vector
#   ND: grids = (t_vals, x_vals, ...), y_vals = Array
function _linear_interpolation(
    grids::Tuple, y_vals::AbstractArray{<:Real}
    )
    if length(grids) == 1
        return Interpolations.linear_interpolation(
            grids[1], vec(y_vals),
            extrapolation_bc = Interpolations.Line())
    else
        return Interpolations.linear_interpolation(
            grids, y_vals,
            extrapolation_bc = Interpolations.Line())
    end
end

# Collect all transcribable decision variables using the existing
# _is_parameter dispatch to filter out parameters. Returns
# (infinite_vars, finite_vars) where infinite vars transcribe to Vector
# and finite vars to scalar.
function _collect_decision_vars(model::InfiniteOpt.InfiniteModel)
    inf_vars = InfiniteOpt.GeneralVariableRef[]
    fin_vars = InfiniteOpt.GeneralVariableRef[]
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        if isempty(prefs)
            push!(fin_vars, v)
        else
            push!(inf_vars, v)
        end
    end
    for d in InfiniteOpt.all_derivatives(model)
        push!(inf_vars, d)
    end
    return inf_vars, fin_vars
end

# Extension point: collect decision variables
function DP.collect_cp_vars(model::InfiniteOpt.InfiniteModel)
    inf_vars, fin_vars = _collect_decision_vars(model)
    return vcat(inf_vars, fin_vars)
end

# Build subproblem via transcribe. Returns a GDPSubmodel
# with forward map (infinite var -> flat var vector).
function DP.build_cp_subproblem(
    model::InfiniteOpt.InfiniteModel,
    dec_vars::Vector{InfiniteOpt.GeneralVariableRef},
    reform_method::DP.AbstractReformulationMethod,
    method::DP.cutting_planes
    )
    tmap = DP.transcribe(model; method = reform_method)
    fwd = Dict(v => tmap.fwd[v] for v in dec_vars)
    sub = DP.GDPSubmodel(
        DP.jump_model(tmap), dec_vars, fwd)
    DP.configure_optimizer(sub, method)
    return sub
end

# rBM setup: build a separate transcribed copy (cannot
# reuse original InfiniteModel in-place for solving).
function DP.setup_rbm(
    model::InfiniteOpt.InfiniteModel,
    dec_vars::Vector{InfiniteOpt.GeneralVariableRef},
    method::DP.cutting_planes
    )
    rBM = DP.build_cp_subproblem(
        model, dec_vars, DP.BigM(method.M_value), method)
    JuMP.relax_integrality(rBM.model)
    return rBM, nothing
end

# Integral cut on the original InfiniteModel. Separates
# infinite and finite terms, wraps infinite part in
# InfiniteOpt.integral for a single scalar constraint.
function DP.add_original_model_cut(
    model::InfiniteOpt.InfiniteModel,
    dec_vars::Vector{InfiniteOpt.GeneralVariableRef},
    rBM_sol::Dict{InfiniteOpt.GeneralVariableRef,
        Vector{T1}},
    sep_sol::Dict{InfiniteOpt.GeneralVariableRef,
        Vector{T2}}
    ) where {T1, T2}
    prefs, sups = _collect_parameters(model)
    inf_terms = Any[]
    fin_terms = Any[]

    for var in dec_vars
        haskey(rBM_sol, var) || continue
        haskey(sep_sol, var) || continue
        vprefs = InfiniteOpt.parameter_refs(var)

        if isempty(vprefs)
            xi = 2 * (sep_sol[var][1] -
                rBM_sol[var][1])
            abs(xi) < 1e-10 && continue
            sp = sep_sol[var][1]
            push!(fin_terms, xi * (var - sp))
        else
            xi_vals = 2 .* (
                sep_sol[var] .- rBM_sol[var])
            sp_vals = sep_sol[var]
            xi_pf = condense_to_pf(
                model, xi_vals, vprefs, sups)
            sp_pf = condense_to_pf(
                model, sp_vals, vprefs, sups)
            push!(inf_terms,
                xi_pf * var - xi_pf * sp_pf)
        end
    end

    cut_scalar = DP._zero_aff(model)

    if !isempty(inf_terms)
        inf_expr = JuMP.@expression(model, sum(inf_terms))
        for p in prefs
            inf_expr = InfiniteOpt.integral(inf_expr, p)
        end
        cut_scalar += inf_expr
    end

    if !isempty(fin_terms)
        for ft in fin_terms
            cut_scalar += ft
        end
    end

    if !iszero(cut_scalar)
        JuMP.@constraint(model, cut_scalar >= 0)
    end
    return
end

end
