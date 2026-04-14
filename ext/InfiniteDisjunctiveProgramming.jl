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

# Extract parameter refs from expression, return VariableProperties with
# Infinite type.
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

# Quadratic expression: handle parameter x parameter, parameter x variable,
# and variable x variable terms.
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
# copy_model_with_constraints (build mini InfiniteModel +
# transcribe to flat JuMP model), prepare_max_M_objective
# (expand infinite constraint into K flat objectives via
# _build_flat_map), and aggregate_M_values (interpolate flat
# values to parameter function).

# Collect all parameter function refs from all disjunct constraints in
# the model.
# Collect parameter functions from the model.
# scan_all=false: only disjunct constraints (MBM/CP).
# scan_all=true: also objective + global constraints
# (LOA needs the full model).
function _all_param_functions(
    model::InfiniteOpt.InfiniteModel;
    scan_all::Bool = false
    )
    pf_set = Set{InfiniteOpt.GeneralVariableRef}()

    _scan(expr) = begin
        for v in InfiniteOpt.all_expression_variables(
                expr)
            dv = InfiniteOpt.dispatch_variable_ref(v)
            if dv isa InfiniteOpt.ParameterFunctionRef
                push!(pf_set, v)
            elseif dv isa InfiniteOpt.MeasureRef
                _scan(
                    InfiniteOpt.measure_function(dv))
            end
        end
    end

    if scan_all
        _scan(JuMP.objective_function(model))
        for (F, S) in JuMP.list_of_constraint_types(
                model)
            F <: Union{
                JuMP.VariableRef, _MOI.VariableIndex,
                InfiniteOpt.GeneralVariableRef
            } && continue
            for cref in JuMP.all_constraints(
                    model, F, S)
                con = JuMP.constraint_object(cref)
                _scan(con.func)
            end
        end
    end

    # Always scan disjunct constraints
    for (_, crefs) in DP._indicator_to_constraints(
            model)
        for cref in crefs
            cref isa DP.DisjunctConstraintRef ||
                continue
            con = JuMP.constraint_object(cref)
            _scan(con.func)
        end
    end

    return pf_set
end

# Build a flat map for support point k. Maps decision variables to their
# flat JuMP.VariableRef at support k (handling multi-parameter indexing)
# and evaluates parameter functions to their numerical values. pf_set is
# precomputed by the caller to avoid rescanning all disjunct constraints
# on every support point.
function _build_flat_map(
    sub::DP.GDPSubmodel, k::Int,
    prefs::Vector{InfiniteOpt.GeneralVariableRef},
    supports::Dict{InfiniteOpt.GeneralVariableRef,Vector{Float64}},
    full_shape::Tuple,
    pf_set::Set{InfiniteOpt.GeneralVariableRef}
    )
    ci = CartesianIndices(full_shape)[k]

    # Decision variables: map each to its variable-local index
    flat_map = Dict{InfiniteOpt.GeneralVariableRef, Any}()
    for (v, ws) in sub.fwd_map
        if length(ws) == 1
            flat_map[v] = ws[1]
        else
            vp = InfiniteOpt.parameter_refs(v)
            shape = Tuple(length(supports[p]) for p in vp)
            idx = Tuple(ci[findfirst(==(p), prefs)] for p in vp)
            flat_map[v] = ws[LinearIndices(shape)[idx...]]
        end
    end

    # Parameter functions: evaluate at support point k
    sup_vals = Dict(
        prefs[i] => supports[prefs[i]][ci[i]]
        for i in 1:length(prefs))
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        pf_prefs = InfiniteOpt.parameter_refs(pf)
        pf_vals = Tuple(sup_vals[p] for p in pf_prefs)
        flat_map[pf] = fn(pf_vals...)
    end
    return flat_map
end

# Build mini InfiniteModel with only the given disjunct constraints,
# transcribe to flat JuMP model, return GDPSubmodel with forward map.
function DP.copy_model_with_constraints(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP._MBM
    )
    mini = InfiniteOpt.InfiniteModel()
    ref_map = Dict{InfiniteOpt.GeneralVariableRef,InfiniteOpt.GeneralVariableRef}()

    # 1. Copy infinite parameters with their supports
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        sups = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(error, domain; supports = sups)
        new_p = InfiniteOpt.add_parameter(mini, param, JuMP.name(p))
        ref_map[p] = new_p
    end

    # 2. Copy decision variables with bounds (skip parameters)
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        var_type = isempty(prefs) ? nothing :
            InfiniteOpt.Infinite(Tuple(ref_map[p] for p in prefs)...)
        props = DP.VariableProperties(DP.get_variable_info(v),"", nothing, var_type)
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

    # 4. Copy parameter functions from ALL disjuncts (needed for
    # constraint transcription)
    pf_set = _all_param_functions(model)
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        prefs = InfiniteOpt.parameter_refs(pf)
        mapped_prefs = Tuple(ref_map[p] for p in prefs)
        new_pf = _make_parameter_function(mini, fn, mapped_prefs...)
        ref_map[pf] = new_pf
    end

    # 5. Add disjunct constraints using existing ref_map
    for cref in constraints
        cref isa DP.DisjunctConstraintRef || continue
        con = JuMP.constraint_object(cref)
        new_func = DP._replace_variables_in_constraint(con.func, ref_map)
        T = one(JuMP.value_type(typeof(mini)))
        JuMP.@constraint(mini, new_func * T in con.set)
    end

    # 6. Transcribe mini InfiniteModel to flat JuMP model
    flat, tr_fwd = transcribe_to_flat(mini)

    # 7. Remap fwd_map: original model var -> flat JuMP VarRef
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}()
    for (orig, mapped) in ref_map
        _is_parameter(orig) && continue
        haskey(tr_fwd, mapped) || continue
        fwd_map[orig] = tr_fwd[mapped]
    end

    decision_vars = collect(keys(fwd_map))
    JuMP.set_optimizer(flat, method.optimizer)
    JuMP.set_silent(flat)
    return DP.GDPSubmodel(flat, decision_vars, fwd_map)
end

# Prepare objectives for all support points. Expands an infinite
# constraint into K flat objectives via _build_flat_map with
# multi-parameter indexing and parameter function evaluation.
function DP.prepare_max_M_objective(
    model::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.LessThan}
    prefs, supports = _collect_parameters(model)
    full_shape = Tuple(length(supports[p]) for p in prefs)
    K = prod(full_shape)
    pf_set = _all_param_functions(model)
    objectives = Vector{JuMP.AbstractJuMPScalar}(undef, K)
    for k in 1:K
        flat_map = _build_flat_map(sub, k, prefs, supports, full_shape, pf_set)
        objectives[k] = -obj.set.upper +
            DP._replace_variables_in_constraint(obj.func, flat_map)
    end
    return objectives
end
function DP.prepare_max_M_objective(
    model::InfiniteOpt.InfiniteModel,
    obj::JuMP.ScalarConstraint{T, S},
    sub::DP.GDPSubmodel
    ) where {T, S <: _MOI.GreaterThan}
    prefs, supports = _collect_parameters(model)
    full_shape = Tuple(length(supports[p]) for p in prefs)
    K = prod(full_shape)
    pf_set = _all_param_functions(model)
    objectives = Vector{JuMP.AbstractJuMPScalar}(undef, K)
    for k in 1:K
        flat_map = _build_flat_map(sub, k, prefs, supports,full_shape, pf_set)
        objectives[k] = obj.set.lower -
            DP._replace_variables_in_constraint(obj.func, flat_map)
    end
    return objectives
end

# Solve the submodel for a vector of objectives (one per
# support point). Returns aggregated result or nothing.
function DP._raw_M(
    sub::DP.GDPSubmodel,
    objectives::Vector{<:JuMP.AbstractJuMPScalar},
    method::DP._MBM
    )
    M_vals = typeof(method.default_M)[]
    for obj_expr in objectives
        # Clear primal starts to avoid NaN from prior solve
        for var in JuMP.all_variables(sub.model)
            JuMP.set_start_value(var, nothing)
        end
        JuMP.@objective(sub.model, Max, obj_expr)
        JuMP.optimize!(sub.model)
        if JuMP.is_solved_and_feasible(sub.model)
            push!(M_vals, max(
                JuMP.objective_value(sub.model),
                zero(method.default_M)))
        elseif JuMP.termination_status(sub.model) ==
                JuMP.MOI.INFEASIBLE
            return nothing
        else
            push!(M_vals, method.default_M)
        end
    end
    model = JuMP.owner_model(
        first(keys(sub.fwd_map)))
    return aggregate_M_values(model, M_vals)
end

# Condense flat per-support values to final form (MBM path).
function aggregate_M_values(
    model::InfiniteOpt.InfiniteModel,
    vals::AbstractVector{<:Real}
    )
    if all(==(vals[1]), vals)
        return vals[1]
    end
    prefs, supports = _collect_parameters(model)
    return condense_to_pf(model, vals, prefs, supports)
end

# Interpolate flat per-support values into a parameter function. Computes
# grids/shape from supports, reshapes, interpolates, and registers on
# the model.
function condense_to_pf(
    model::InfiniteOpt.InfiniteModel,
    vals::AbstractVector{<:Real},
    prefs::Union{Vector{InfiniteOpt.GeneralVariableRef},
        Tuple{Vararg{InfiniteOpt.GeneralVariableRef}}},
    supports::Dict{InfiniteOpt.GeneralVariableRef,
        Vector{Float64}}
    )
    grids = Tuple(supports[p] for p in prefs)
    shape = Tuple(length(supports[p]) for p in prefs)
    nd = reshape(vals, shape)
    fn = Interpolations.linear_interpolation(grids, nd,
        extrapolation_bc = Interpolations.Line())
    return _make_parameter_function(model, fn, prefs...)
end

################################################################################
#                          TRANSCRIPTION HELPERS
################################################################################

# Create a parameter function programmatically. Uses
# build_parameter_function + add_parameter_function (the lower-level
# API behind @parameter_function) since the macro doesn't support
# programmatic use. The closure wrapper handles non-Function callables
# like Interpolations.Extrapolation. Accepts any number of prefs via
# varargs: _make_parameter_function(m, f, t) for 1D, (m, f, t, x) for 2D.
function _make_parameter_function(
    model::InfiniteOpt.InfiniteModel, fn,
    prefs::InfiniteOpt.GeneralVariableRef...
    )
    f = fn isa Function ? fn : ((args...) -> fn(args...))
    pref_arg = length(prefs) == 1 ? prefs[1] : prefs
    pfunc = InfiniteOpt.build_parameter_function(error, f, pref_arg)
    return InfiniteOpt.add_parameter_function(model, pfunc)
end

# Collect all infinite parameters and their supports from the model.
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

# Transcribe an InfiniteModel to a flat JuMP.Model with forward variable
# map. Shared by MBM and CP paths.
function transcribe_to_flat(model::InfiniteOpt.InfiniteModel)
    InfiniteOpt.build_transformation_backend!(model)
    flat = InfiniteOpt.transformation_model(model)
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}()
    for v in DP.collect_all_vars(model)
        tv = InfiniteOpt.transformation_variable(v)
        vprefs = InfiniteOpt.parameter_refs(v)
        fwd_map[v] = isempty(vprefs) ? [tv] : vec(tv)
    end
    return flat, fwd_map
end

################################################################################
#                    CUTTING PLANES FOR INFINITEMODEL
################################################################################

# Build CP subproblem: reformulate the InfiniteModel, transcribe to a flat
# JuMP copy, and wrap in GDPSubmodel with forward variable map.
function DP.copy_and_reformulate(
    model::InfiniteOpt.InfiniteModel,
    decision_vars::Vector{InfiniteOpt.GeneralVariableRef},
    reform_method::DP.AbstractReformulationMethod,
    method::DP.CuttingPlanes
    )
    DP.reformulate_model(model, reform_method)
    flat, tr_fwd = transcribe_to_flat(model)
    sub_copy, copy_map = JuMP.copy_model(flat)
    fwd_map = Dict{InfiniteOpt.GeneralVariableRef, Vector{JuMP.VariableRef}}()
    for v in decision_vars
        haskey(tr_fwd, v) || continue
        fwd_map[v] = [copy_map[tv] for tv in tr_fwd[v]]
    end
    sub = DP.GDPSubmodel(sub_copy, decision_vars, fwd_map)
    JuMP.set_optimizer(sub.model, method.optimizer)
    JuMP.set_silent(sub.model)
    return sub
end

# rBM setup: build a separate transcribed copy (cannot reuse original
# InfiniteModel in-place for solving).
function DP.reformulate_and_relax(
    model::InfiniteOpt.InfiniteModel,
    decision_vars::Vector{InfiniteOpt.GeneralVariableRef},
    reform_method::DP.AbstractReformulationMethod,
    method::DP.CuttingPlanes
    )
    rBM = DP.copy_and_reformulate(model, decision_vars, reform_method, method)
    JuMP.relax_integrality(rBM.model)
    return rBM, nothing
end

# Add separating cut for the infinite CP path. Adds the linear cut
# to the flat transcribed rBM (so the next CP iteration benefits) AND
# an integral cut to the original InfiniteModel (so the final
# reformulation includes the cuts). Dispatched via the decision var
# ref type of the GDPSubmodel.
function DP._add_cut(
    sub::DP.GDPSubmodel{<:Any, <:InfiniteOpt.GeneralVariableRef, <:Any},
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}},
    sep_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    # --- Linear cut on the flat rBM model ---
    cut_expr = zero(JuMP.GenericAffExpr{
        JuMP.value_type(typeof(sub.model)),
        JuMP.variable_ref_type(sub.model)})
    for var in sub.decision_vars
        sub_vars = sub.fwd_map[var]
        rbm_vals = rBM_sol[var]
        sep_vals = sep_sol[var]
        for k in 1:length(sub_vars)
            xi = 2 * (sep_vals[k] - rbm_vals[k])
            JuMP.add_to_expression!(cut_expr, xi, sub_vars[k])
            JuMP.add_to_expression!(cut_expr, -xi * sep_vals[k])
        end
    end
    JuMP.@constraint(sub.model, cut_expr >= 0)

    # --- Integral cut on the original InfiniteModel ---
    original = JuMP.owner_model(first(sub.decision_vars))
    prefs, sups = _collect_parameters(original)
    inf_terms = Any[]
    cut_scalar = zero(JuMP.GenericAffExpr{
        JuMP.value_type(typeof(original)),
        JuMP.variable_ref_type(original)})
    for var in sub.decision_vars
        #TODO: Candidate for dispatch?
        haskey(rBM_sol, var) || continue
        haskey(sep_sol, var) || continue
        vprefs = InfiniteOpt.parameter_refs(var)
        if isempty(vprefs)
            xi = 2 * (sep_sol[var][1] - rBM_sol[var][1])
            sp = sep_sol[var][1]
            cut_scalar += xi * (var - sp)
        else
            xi_vals = 2 .* (sep_sol[var] .- rBM_sol[var])
            sp_vals = sep_sol[var]
            xi_pf = condense_to_pf(original, xi_vals, vprefs, sups)
            sp_pf = condense_to_pf(original, sp_vals, vprefs, sups)
            push!(inf_terms, xi_pf * var - xi_pf * sp_pf)
        end
    end

    if !isempty(inf_terms)
        inf_expr = JuMP.@expression(original, sum(inf_terms))
        for p in prefs
            inf_expr = InfiniteOpt.integral(inf_expr, p)
        end
        cut_scalar += inf_expr
    end

    JuMP.@constraint(original, cut_scalar >= 0)
    return
end

################################################################################
#                        LOA FOR INFINITEMODEL
################################################################################

# InfiniteModel dispatch for _copy_global_constraints:
# also skip GeneralVariableRef constraint types.
function DP._copy_global_constraints(
    model::InfiniteOpt.InfiniteModel,
    sub_model, var_map
    )
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: Union{
            JuMP.VariableRef, _MOI.VariableIndex,
            InfiniteOpt.GeneralVariableRef
        } && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref isa DP.DisjunctConstraintRef &&
                continue
            con = JuMP.constraint_object(cref)
            new_f = DP._replace_variables_in_constraint(
                con.func, var_map)
            JuMP.@constraint(
                sub_model, new_f in con.set)
        end
    end
    return
end

# Extend _replace_variables_in_constraint to handle
# InfiniteOpt measures (integrals). When a
# GeneralVariableRef is a MeasureRef, rebuild the
# integral with the remapped integrand and parameter.
function DP._replace_variables_in_constraint(
    expr::InfiniteOpt.GeneralVariableRef,
    var_map::AbstractDict
    )
    haskey(var_map, expr) && return var_map[expr]
    dv = InfiniteOpt.dispatch_variable_ref(expr)
    if dv isa InfiniteOpt.MeasureRef
        mf = InfiniteOpt.measure_function(dv)
        md = InfiniteOpt.measure_data(dv)
        new_mf = DP._replace_variables_in_constraint(
            mf, var_map)
        pref = InfiniteOpt.parameter_refs(md)
        new_pref = var_map[pref]
        return InfiniteOpt.integral(new_mf, new_pref)
    end
    return var_map[expr]
end

# Build a mini InfiniteModel NLP subproblem with the
# given disjunct constraints + all global constraints.
# Stays as InfiniteModel (transcribed automatically
# on optimize!). Returns (mini, ref_map, mini_crefs).
function _build_inf_subproblem(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP.LOA
    )
    mini = InfiniteOpt.InfiniteModel()
    ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()

    # Copy parameters, variables, derivatives, param
    # functions using the shared MBM pattern.
    _copy_infinite_structure(
        model, mini, ref_map; scan_all = true,
        relax_integrality = true)

    # Copy objective
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = DP._replace_variables_in_constraint(
        obj, ref_map)
    JuMP.@objective(mini, sense, new_obj)

    # Copy global constraints
    DP._copy_global_constraints(model, mini, ref_map)

    # Copy active disjunct constraints
    mini_crefs = InfiniteOpt.InfOptConstraintRef[]
    for cref in constraints
        con = JuMP.constraint_object(cref)
        new_func = DP._replace_variables_in_constraint(
            con.func, ref_map)
        T = one(JuMP.value_type(typeof(mini)))
        mini_cref = JuMP.@constraint(
            mini, new_func * T in con.set)
        push!(mini_crefs, mini_cref)
    end

    return mini, ref_map, mini_crefs
end

# Shared helper: copy infinite parameters, decision
# variables, derivatives, and parameter functions from
# model to target using ref_map.
function _copy_infinite_structure(
    model::InfiniteOpt.InfiniteModel,
    target::InfiniteOpt.InfiniteModel,
    ref_map::Dict;
    scan_all::Bool = false,
    relax_integrality::Bool = false
    )
    # Parameters
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        sups = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(
            error, domain; supports = sups)
        new_p = InfiniteOpt.add_parameter(
            target, param, JuMP.name(p))
        ref_map[p] = new_p
    end

    # Decision variables
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        var_type = isempty(prefs) ? nothing :
            InfiniteOpt.Infinite(
                Tuple(ref_map[p] for p in prefs)...)
        kw = relax_integrality ?
            (; has_binary = false,
               has_integer = false,
               has_start = false) :
            (; has_start = false)
        info = DP.get_variable_info(v; kw...)
        props = DP.VariableProperties(
            info, JuMP.name(v), nothing, var_type)
        ref_map[v] = DP.create_variable(target, props)
    end

    # Derivatives
    for d in InfiniteOpt.all_derivatives(model)
        vref = InfiniteOpt.derivative_argument(d)
        pref = InfiniteOpt.operator_parameter(d)
        new_d = InfiniteOpt.deriv(
            ref_map[vref], ref_map[pref])
        info = DP.get_variable_info(d)
        info.has_lb &&
            JuMP.set_lower_bound(
                new_d, info.lower_bound)
        info.has_ub &&
            JuMP.set_upper_bound(
                new_d, info.upper_bound)
        ref_map[d] = new_d
    end

    # Parameter functions
    pf_set = _all_param_functions(
        model; scan_all = scan_all)
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        prefs = InfiniteOpt.parameter_refs(pf)
        mapped = Tuple(ref_map[p] for p in prefs)
        new_pf = _make_parameter_function(
            target, fn, mapped...)
        ref_map[pf] = new_pf
    end
    return
end

# Solve NLP subproblem for InfiniteModel. Solves
# directly as InfiniteModel (auto-transcribed).
# Extracts per-support-point x-values and duals.
function DP._solve_loa_subproblem(
    model::InfiniteOpt.InfiniteModel,
    combo::Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        Bool},
    method::DP.LOA
    )
    M = InfiniteOpt.InfiniteModel
    active_crefs = DP._active_constraints(model, combo)
    mini, ref_map, mini_crefs = _build_inf_subproblem(
        model, active_crefs, method)

    JuMP.set_optimizer(mini, method.nlp_optimizer)
    JuMP.set_silent(mini)
    JuMP.optimize!(mini)

    XV = Vector{Float64}
    if !JuMP.is_solved_and_feasible(mini)
        return DP._LOAIterationResult{M, XV}(
            combo,
            Dict{JuMP.AbstractVariableRef, XV}(),
            Dict{DP.DisjunctConstraintRef{M}, XV}(),
            Inf, false)
    end

    # Extract per-support-point x values
    x_vals = Dict{JuMP.AbstractVariableRef, XV}()
    for (orig, mapped) in ref_map
        _is_parameter(orig) && continue
        v = JuMP.value(mapped)
        x_vals[orig] = v isa AbstractVector ?
            Float64.(v) : [Float64(v)]
    end

    # Extract duals (per-support-point)
    duals = Dict{DP.DisjunctConstraintRef{M}, XV}()
    has_d = JuMP.has_duals(mini)
    if has_d
        for (i, orig) in enumerate(active_crefs)
            d = JuMP.dual(mini_crefs[i])
            duals[orig] = d isa AbstractVector ?
                Float64.(d) : [Float64(d)]
        end
    end

    return DP._LOAIterationResult{M, XV}(
        combo, x_vals, duals,
        JuMP.objective_value(mini), true)
end

# Build the MILP master for InfiniteModel LOA.
# Stays as InfiniteModel — transcribed automatically
# on optimize!. No manual transcription needed.
function DP._build_loa_master(
    model::InfiniteOpt.InfiniteModel,
    init_results, method::DP.LOA
    )
    nl_globals = DP._nonlinear_global_constraints(
        model)

    # Build master as InfiniteModel (no transcription).
    # InfiniteOpt transcribes automatically on optimize!.
    master_inf = InfiniteOpt.InfiniteModel()
    master_inf.ext[:GDP] = DP.GDPData{
        InfiniteOpt.InfiniteModel,
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.InfOptConstraintRef}()
    ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()

    _copy_infinite_structure(
        model, master_inf, ref_map; scan_all = true)

    # Copy objective
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = DP._replace_variables_in_constraint(
        obj, ref_map)
    JuMP.@objective(master_inf, sense, new_obj)

    # Copy global constraints
    DP._copy_global_constraints(
        model, master_inf, ref_map)

    # Copy GDP data
    _copy_gdp_for_loa(model, master_inf, ref_map)

    # Strip nonlinear disjuncts and globals, BigM-reform
    DP._remove_nonlinear_disjuncts(master_inf)
    DP._remove_nonlinear_globals(master_inf)
    DP.reformulate_model(
        master_inf, DP.BigM(method.M_value))

    JuMP.set_optimizer(
        master_inf, method.mip_optimizer)
    JuMP.set_silent(master_inf)

    # Build bin_map: orig indicator → master binary
    lv_map = Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel}
        }()
    orig_gdp = DP.gdp_data(model)
    for (idx, _) in orig_gdp.logical_variables
        lv_map[DP.LogicalVariableRef(model, idx)] =
            DP.LogicalVariableRef(master_inf, idx)
    end
    bin_map = Dict{DP.LogicalVariableRef, Any}()
    ind_to_bin = DP._indicator_to_binary(master_inf)
    for (orig_ind, copy_ind) in lv_map
        bv = get(ind_to_bin, copy_ind, nothing)
        bin_map[orig_ind] = bv isa
            JuMP.AbstractVariableRef ? bv : nothing
    end

    # ref_map for OA cuts: orig var → master var
    oa_ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()
    for (orig, copy) in ref_map
        _is_parameter(orig) && continue
        oa_ref_map[orig] = copy
    end

    master = DP._LOAMaster{
        typeof(master_inf), typeof(oa_ref_map)}(
        master_inf, oa_ref_map, bin_map,
        JuMP.VariableRef[],
        JuMP.objective_function(master_inf),
        JuMP.objective_sense(master_inf),
        nl_globals)

    for result in init_results
        result.feasible && DP._add_oa_cuts(
            master, result, model, method)
        DP._add_no_good_cut(master, result.combo)
    end
    return master
end

# Copy GDP data from model to copy InfiniteModel.
# Creates infinite binary backing variables that
# match the original's parameter dependencies.
function _copy_gdp_for_loa(
    model::InfiniteOpt.InfiniteModel,
    copy::InfiniteOpt.InfiniteModel,
    ref_map::Dict
    )
    orig_gdp = DP.gdp_data(model)
    copy_gdp = DP.gdp_data(copy)

    lv_map = Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel}
        }()
    orig_ind_to_bin = DP._indicator_to_binary(model)
    for (idx, var_data) in orig_gdp.logical_variables
        copy_gdp.logical_variables[idx] =
            DP.LogicalVariableData(
                var_data.variable, var_data.name)
        orig_lv = DP.LogicalVariableRef(model, idx)
        copy_lv = DP.LogicalVariableRef(copy, idx)
        lv_map[orig_lv] = copy_lv
        # Create binary with same infinite params
        orig_bv = orig_ind_to_bin[orig_lv]
        var_type = nothing
        if orig_bv isa InfiniteOpt.GeneralVariableRef
            prefs = InfiniteOpt.parameter_refs(orig_bv)
            if !isempty(prefs)
                var_type = InfiniteOpt.Infinite(
                    Tuple(ref_map[p]
                        for p in prefs)...)
            end
        end
        if var_type === nothing
            bvref = JuMP.@variable(copy,
                base_name = var_data.name,
                binary = true)
        else
            info = JuMP.VariableInfo(
                false, NaN, false, NaN, false, NaN,
                false, NaN, true, false)
            props = DP.VariableProperties(
                info, var_data.name,
                nothing, var_type)
            bvref = DP.create_variable(copy, props)
        end
        DP._indicator_to_binary(
            copy)[copy_lv] = bvref
    end

    for (idx, dc_data) in orig_gdp.disjunct_constraints
        old_con = dc_data.constraint
        old_dc_ref = DP.DisjunctConstraintRef(
            model, idx)
        old_ind = orig_gdp.constraint_to_indicator[
            old_dc_ref]
        new_func = DP._replace_variables_in_constraint(
            old_con.func, ref_map)
        new_con = JuMP.build_constraint(error,
            new_func, old_con.set,
            DP.Disjunct(lv_map[old_ind]))
        JuMP.add_constraint(copy, new_con, dc_data.name)
    end

    disj_map = Dict{
        DP.DisjunctionRef{InfiniteOpt.InfiniteModel},
        DP.DisjunctionRef{InfiniteOpt.InfiniteModel}}()
    for (idx, disj_data) in orig_gdp.disjunctions
        old_disj = disj_data.constraint
        new_inds = [lv_map[ind]
            for ind in old_disj.indicators]
        copy_gdp.disjunctions[idx] =
            DP.ConstraintData(
                DP.Disjunction(
                    new_inds, old_disj.nested),
                disj_data.name)
        disj_map[DP.DisjunctionRef(model, idx)] =
            DP.DisjunctionRef(copy, idx)
    end

    for (idx, lc_data) in orig_gdp.logical_constraints
        old_con = lc_data.constraint
        new_func =
            DP._replace_variables_in_constraint(
                old_con.func, lv_map)
        new_con = JuMP.build_constraint(
            error, new_func, old_con.set)
        JuMP.add_constraint(
            copy, new_con, lc_data.name)
    end

    for (d_ref, lc_ref) in
            orig_gdp.exactly1_constraints
        copy_gdp.exactly1_constraints[
            disj_map[d_ref]] =
            DP.LogicalConstraintRef(copy,
                DP.LogicalConstraintIndex(
                    JuMP.index(lc_ref).value))
    end
    return lv_map
end

# OA cuts on the InfiniteModel master. Uses
# condense_to_pf to turn per-support xk values
# into parameter functions for linearization
# coefficients, producing infinite OA constraints.
function DP._add_oa_cuts(
    master::DP._LOAMaster{
        <:InfiniteOpt.InfiniteModel, <:Any},
    result::DP._LOAIterationResult{
        InfiniteOpt.InfiniteModel, Vector{Float64}},
    model::InfiniteOpt.InfiniteModel,
    method::DP.LOA
    )
    sgn = master.obj_sense ==
        _MOI.MIN_SENSE ? -1 : 1
    prefs, sups = _collect_parameters(master.model)

    # Linearize a QuadExpr at xk where xk values may
    # be parameter functions. Returns an infinite expr.
    function _linearize_quad_inf(func, xk, ref_map)
        # f(xk) + ∇f(xk)ᵀ(x - xk)
        # = Σ_ij c_ij*(xk_j*x_i + xk_i*x_j - xk_i*xk_j)
        #   + Σ_i a_i*x_i + constant
        m = master.model
        result = JuMP.@expression(m, 0.0)
        # Affine terms: a_i * x_i + constant
        result += func.aff.constant
        for (var, coef) in func.aff.terms
            result += coef * ref_map[var]
        end
        # Quadratic terms: c*(xk_j*x_i + xk_i*x_j - xk_i*xk_j)
        for (pair, coef) in func.terms
            vi, vj = pair.a, pair.b
            xk_i = get(xk, vi, 0.0)
            xk_j = get(xk, vj, 0.0)
            if vi == vj
                # c*x^2 → c*(2*xk*x - xk^2)
                result += coef * (
                    2 * xk_i * ref_map[vi] -
                    xk_i * xk_i)
            else
                result += coef * (
                    xk_j * ref_map[vi] +
                    xk_i * ref_map[vj] -
                    xk_i * xk_j)
            end
        end
        return result
    end

    # Helper: build xk dict with parameter functions
    # from per-support solution vectors.
    function _build_pf_xk(result, master)
        xk = Dict{Any, Any}()
        id_map = Dict{Any, Any}()
        for (orig, vals) in result.x_values
            mapped = get(master.ref_map, orig, nothing)
            mapped === nothing && continue
            vprefs = InfiniteOpt.parameter_refs(mapped)
            if isempty(vprefs) || length(vals) == 1
                xk[orig] = vals[1]
            else
                xk[orig] = condense_to_pf(
                    master.model, Float64.(vals),
                    vprefs, sups)
            end
            id_map[orig] = mapped
        end
        return xk, id_map
    end

    xk, id_map = _build_pf_xk(result, master)

    # Disjunct OA cuts (skipped for linear biodiesel)
    for (ind, active) in result.combo
        !active && continue
        haskey(DP._indicator_to_constraints(
            model), ind) || continue
        bin_var = get(master.bin_map, ind, nothing)

        for orig_cref in DP._indicator_to_constraints(
                model)[ind]
            orig_cref isa DP.DisjunctConstraintRef ||
                continue
            con = DP._disjunct_constraints(model)[
                JuMP.index(orig_cref)].constraint
            con.func isa JuMP.GenericAffExpr && continue

            dual_vec = get(
                result.duals, orig_cref, nothing)
            dual_vec === nothing && continue
            # Use mean dual for sign determination
            mean_dual = sum(dual_vec) / length(dual_vec)
            s = sign(sgn * mean_dual)
            s == 0 && continue

            lin_expr = _linearize_quad_inf(
                con.func, xk, id_map)
            rhs = DP._set_rhs(con.set)

            slack = JuMP.@variable(master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack,
                variable_type = InfiniteOpt.Infinite(
                    prefs...))
            push!(master.slack_vars, slack)

            lhs = s * (lin_expr - rhs) - slack
            if bin_var !== nothing
                cref = JuMP.@constraint(
                    master.model, lhs <=
                    method.M_value * (1 - bin_var))
            else
                cref = JuMP.@constraint(
                    master.model, lhs <= 0)
            end
            push!(DP._reformulation_constraints(
                master.model), cref)
        end
    end

    # Global OA cuts
    for (func, set) in master.nl_globals
        lin_expr = _linearize_quad_inf(func, xk, id_map)
        rhs = DP._set_rhs(set)
        for s in DP._oa_global_signs(set)
            slack = JuMP.@variable(master.model,
                lower_bound = 0.0,
                upper_bound = method.max_slack,
                variable_type = InfiniteOpt.Infinite(
                    prefs...))
            push!(master.slack_vars, slack)
            lhs = s * (lin_expr - rhs) - slack
            cref = JuMP.@constraint(
                master.model, lhs <= 0)
            push!(DP._reformulation_constraints(
                master.model), cref)
        end
    end
end

# Final model setup for InfiniteModel LOA.
function DP._finalize_model(
    model::InfiniteOpt.InfiniteModel,
    best_result, method::DP.LOA
    )
    has_nl = DP._has_nonlinear_disjuncts(model) ||
        !isempty(
            DP._nonlinear_global_constraints(model))
    if !has_nl
        DP.reformulate_model(
            model, DP.BigM(method.M_value))
        JuMP.set_optimizer(
            model, method.mip_optimizer)
        return
    end

    # Collect active nonlinear constraints, fix
    # indicators to best combo
    nl_active = Tuple{Any, Any}[]
    if best_result !== nothing
        for (ind, active) in best_result.combo
            bv = DP._indicator_to_binary(model)[ind]
            if bv isa JuMP.AbstractVariableRef
                JuMP.fix(bv, active ? 1.0 : 0.0;
                    force = true)
            end
            !active && continue
            haskey(DP._indicator_to_constraints(
                model), ind) || continue
            for cref in DP._indicator_to_constraints(
                    model)[ind]
                cref isa DP.DisjunctConstraintRef ||
                    continue
                c = DP._disjunct_constraints(model)[
                    JuMP.index(cref)].constraint
                DP._is_nonlinear(c.func) &&
                    push!(nl_active, (c.func, c.set))
            end
        end
    end

    DP._remove_nonlinear_disjuncts(model)
    DP.reformulate_model(
        model, DP.BigM(method.M_value))
    JuMP.set_optimizer(model, method.mip_optimizer)

    for (func, set) in nl_active
        cref = JuMP.@constraint(model, func in set)
        push!(
            DP._reformulation_constraints(model), cref)
    end
    return
end

end
