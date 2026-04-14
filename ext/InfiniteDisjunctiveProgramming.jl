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

# Collect all parameter functions referenced anywhere
# in the model (objective, global constraints, and
# disjunct constraints). Needed by LOA because the
# subproblem and master copy the full model, not
# just disjunct constraints.
function _all_param_functions_full(
    model::InfiniteOpt.InfiniteModel
    )
    pf_set = Set{InfiniteOpt.GeneralVariableRef}()

    # Scan helper: walk an expression and collect pfs
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

    # Objective
    _scan(JuMP.objective_function(model))

    # All constraints (global + disjunct)
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: Union{
            JuMP.VariableRef, _MOI.VariableIndex,
            InfiniteOpt.GeneralVariableRef
        } && continue
        for cref in JuMP.all_constraints(model, F, S)
            con = JuMP.constraint_object(cref)
            _scan(con.func)
        end
    end

    # Disjunct constraints
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

# Remap an expression using ref_map, rebuilding
# measures (integrals) on the target model.
function _remap_expression(
    expr::InfiniteOpt.GeneralVariableRef,
    ref_map::Dict, target_model
    )
    if haskey(ref_map, expr)
        return ref_map[expr]
    end
    dv = InfiniteOpt.dispatch_variable_ref(expr)
    if dv isa InfiniteOpt.MeasureRef
        mf = InfiniteOpt.measure_function(dv)
        md = InfiniteOpt.measure_data(dv)
        new_mf = _remap_expression(
            mf, ref_map, target_model)
        pref = InfiniteOpt.parameter_refs(md)
        new_pref = ref_map[pref]
        return InfiniteOpt.integral(
            new_mf, new_pref)
    end
    return ref_map[expr]
end

function _remap_expression(
    expr::JuMP.GenericAffExpr{C,
        InfiniteOpt.GeneralVariableRef},
    ref_map::Dict, target_model
    ) where {C}
    result = JuMP.GenericAffExpr{C,
        InfiniteOpt.GeneralVariableRef}(
        expr.constant)
    for (var, coef) in expr.terms
        new_var = _remap_expression(
            var, ref_map, target_model)
        JuMP.add_to_expression!(
            result, coef, new_var)
    end
    return result
end

function _remap_expression(
    expr::JuMP.GenericQuadExpr{C,
        InfiniteOpt.GeneralVariableRef},
    ref_map::Dict, target_model
    ) where {C}
    aff = _remap_expression(
        expr.aff, ref_map, target_model)
    result = JuMP.GenericQuadExpr(aff)
    for (pair, coef) in expr.terms
        va = _remap_expression(
            pair.a, ref_map, target_model)
        vb = _remap_expression(
            pair.b, ref_map, target_model)
        JuMP.add_to_expression!(
            result, coef, va, vb)
    end
    return result
end

function _remap_expression(
    expr::JuMP.GenericNonlinearExpr, ref_map::Dict,
    target_model
    )
    new_args = Any[]
    for a in expr.args
        push!(new_args, _remap_expression(
            a, ref_map, target_model))
    end
    return JuMP.GenericNonlinearExpr(
        expr.head, new_args)
end

_remap_expression(x::Number, ::Dict, _) = x

# Fallback: use _replace_variables_in_constraint
function _remap_expression(expr, ref_map::Dict, _)
    return DP._replace_variables_in_constraint(
        expr, ref_map)
end

# Build a mini InfiniteModel NLP subproblem with only the
# given disjunct constraints + all global constraints,
# transcribe to flat JuMP model for solving.
function DP._copy_subproblem(
    model::InfiniteOpt.InfiniteModel,
    constraints::Vector{<:DP.DisjunctConstraintRef},
    method::DP.LOA
    )
    mini = InfiniteOpt.InfiniteModel()
    ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()

    # 1. Copy infinite parameters with supports
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        sups = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(
            error, domain; supports = sups)
        new_p = InfiniteOpt.add_parameter(
            mini, param, JuMP.name(p))
        ref_map[p] = new_p
    end

    # 2. Copy decision variables (relax integrality)
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        var_type = isempty(prefs) ? nothing :
            InfiniteOpt.Infinite(
                Tuple(ref_map[p] for p in prefs)...)
        info = DP.get_variable_info(v;
            has_binary = false, has_integer = false,
            has_start = false)
        props = DP.VariableProperties(
            info, "", nothing, var_type)
        ref_map[v] = DP.create_variable(mini, props)
    end

    # 3. Copy derivatives with bounds
    for d in InfiniteOpt.all_derivatives(model)
        vref = InfiniteOpt.derivative_argument(d)
        pref = InfiniteOpt.operator_parameter(d)
        new_d = InfiniteOpt.deriv(
            ref_map[vref], ref_map[pref])
        info = DP.get_variable_info(d)
        info.has_lb &&
            JuMP.set_lower_bound(new_d, info.lower_bound)
        info.has_ub &&
            JuMP.set_upper_bound(new_d, info.upper_bound)
        ref_map[d] = new_d
    end

    # 4. Copy parameter functions from everywhere
    # (objective, global cons, disjunct cons)
    pf_set = _all_param_functions_full(model)
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        prefs = InfiniteOpt.parameter_refs(pf)
        mapped = Tuple(ref_map[p] for p in prefs)
        new_pf = _make_parameter_function(
            mini, fn, mapped...)
        ref_map[pf] = new_pf
    end

    # 5. Copy objective (rebuild measures if present)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = _remap_expression(obj, ref_map, mini)
    JuMP.@objective(mini, sense, new_obj)

    # 6. Copy global (non-disjunct) constraints
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: Union{
            JuMP.VariableRef, _MOI.VariableIndex,
            InfiniteOpt.GeneralVariableRef
        } && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref isa DP.DisjunctConstraintRef &&
                continue
            con = JuMP.constraint_object(cref)
            new_f = _remap_expression(
                con.func, ref_map, mini)
            JuMP.@constraint(mini, new_f in con.set)
        end
    end

    # 7. Copy active disjunct constraints (tracked for
    #    dual extraction after transcription)
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

    # 8. Transcribe to flat JuMP model
    flat, tr_fwd = transcribe_to_flat(mini)

    # 9. Build fwd_map: orig var → flat vars
    fwd_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        Vector{JuMP.VariableRef}}()
    for (orig, mapped) in ref_map
        _is_parameter(orig) && continue
        haskey(tr_fwd, mapped) || continue
        fwd_map[orig] = tr_fwd[mapped]
    end

    # 10. Map mini constraint refs to flat refs
    sub_crefs = JuMP.ConstraintRef[]
    for mc in mini_crefs
        tc = InfiniteOpt.transformation_constraint(mc)
        if tc isa AbstractVector
            append!(sub_crefs, tc)
        else
            push!(sub_crefs, tc)
        end
    end

    decision_vars = collect(keys(fwd_map))
    JuMP.set_optimizer(flat, method.nlp_optimizer)
    JuMP.set_silent(flat)
    return (
        DP.GDPSubmodel(flat, decision_vars, fwd_map),
        sub_crefs
    )
end

# Solve NLP subproblem for InfiniteModel. Extracts
# per-support-point x-values and duals.
function DP._solve_loa_subproblem(
    model::InfiniteOpt.InfiniteModel,
    combo::Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        Bool},
    method::DP.LOA
    )
    M = InfiniteOpt.InfiniteModel
    active_crefs = DP._active_constraints(model, combo)
    sub, sub_crefs = DP._copy_subproblem(
        model, active_crefs, method)

    JuMP.optimize!(sub.model)

    XV = Vector{Float64}
    if !JuMP.is_solved_and_feasible(sub.model)
        return DP._LOAIterationResult{M, XV}(
            combo,
            Dict{JuMP.AbstractVariableRef, XV}(),
            Dict{DP.DisjunctConstraintRef{M}, XV}(),
            Inf, false)
    end

    # Per-support-point x values
    x_vals = Dict{JuMP.AbstractVariableRef, XV}()
    for var in sub.decision_vars
        x_vals[var] = [
            JuMP.value(fv) for fv in sub.fwd_map[var]]
    end

    # Per-support-point duals (flat constraints are
    # ordered: K flat crefs per original constraint)
    K = length(first(values(sub.fwd_map)))
    n_orig = length(active_crefs)
    duals = Dict{DP.DisjunctConstraintRef{M}, XV}()
    has_d = JuMP.has_duals(sub.model)
    if has_d && length(sub_crefs) == n_orig * K
        for (i, orig) in enumerate(active_crefs)
            duals[orig] = [
                JuMP.dual(sub_crefs[(i-1)*K + k])
                for k in 1:K]
        end
    elseif has_d
        # Fallback: single dual per constraint
        for (i, orig) in enumerate(active_crefs)
            i <= length(sub_crefs) || break
            duals[orig] = [JuMP.dual(sub_crefs[i])]
        end
    end

    return DP._LOAIterationResult{M, XV}(
        combo, x_vals, duals,
        JuMP.objective_value(sub.model), true)
end

# Build the MILP master for InfiniteModel LOA.
# Copies the model manually, strips nonlinear disjuncts,
# BigM-reforms, then transcribes to flat MILP.
function DP._build_loa_master(
    model::InfiniteOpt.InfiniteModel,
    init_results, method::DP.LOA
    )
    # Capture nonlinear globals from original model
    # (stored for OA cut generation; stripped from
    # the master copy before BigM reformulation).
    nl_globals = DP._nonlinear_global_constraints(
        model)

    # Use copy_and_reformulate from CP path to get a
    # flat BigM-reformed copy. First strip nonlinear
    # disjuncts before reformulation.
    master_inf = InfiniteOpt.InfiniteModel()
    master_inf.ext[:GDP] = DP.GDPData{
        InfiniteOpt.InfiniteModel,
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.InfOptConstraintRef}()
    ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        InfiniteOpt.GeneralVariableRef}()

    # Copy parameters
    for p in InfiniteOpt.all_parameters(model)
        domain = InfiniteOpt.infinite_domain(p)
        sups = Float64.(InfiniteOpt.supports(p))
        param = InfiniteOpt.build_parameter(
            error, domain; supports = sups)
        new_p = InfiniteOpt.add_parameter(
            master_inf, param, JuMP.name(p))
        ref_map[p] = new_p
    end

    # Copy variables (keep binary/integer for BigM)
    for v in JuMP.all_variables(model)
        _is_parameter(v) && continue
        prefs = InfiniteOpt.parameter_refs(v)
        var_type = isempty(prefs) ? nothing :
            InfiniteOpt.Infinite(
                Tuple(ref_map[p] for p in prefs)...)
        info = DP.get_variable_info(
            v; has_start = false)
        props = DP.VariableProperties(
            info, JuMP.name(v), nothing, var_type)
        ref_map[v] = DP.create_variable(
            master_inf, props)
    end

    # Copy derivatives
    for d in InfiniteOpt.all_derivatives(model)
        vref = InfiniteOpt.derivative_argument(d)
        pref = InfiniteOpt.operator_parameter(d)
        new_d = InfiniteOpt.deriv(
            ref_map[vref], ref_map[pref])
        info = DP.get_variable_info(d)
        info.has_lb &&
            JuMP.set_lower_bound(new_d, info.lower_bound)
        info.has_ub &&
            JuMP.set_upper_bound(new_d, info.upper_bound)
        ref_map[d] = new_d
    end

    # Copy parameter functions (full scan)
    pf_set = _all_param_functions_full(model)
    for pf in pf_set
        fn = InfiniteOpt.raw_function(pf)
        prefs = InfiniteOpt.parameter_refs(pf)
        mapped = Tuple(ref_map[p] for p in prefs)
        new_pf = _make_parameter_function(
            master_inf, fn, mapped...)
        ref_map[pf] = new_pf
    end

    # Copy objective (rebuild measures if present)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    new_obj = _remap_expression(
        obj, ref_map, master_inf)
    JuMP.@objective(master_inf, sense, new_obj)

    # Copy global constraints
    for (F, S) in JuMP.list_of_constraint_types(model)
        F <: Union{
            JuMP.VariableRef, _MOI.VariableIndex,
            InfiniteOpt.GeneralVariableRef
        } && continue
        for cref in JuMP.all_constraints(model, F, S)
            cref isa DP.DisjunctConstraintRef &&
                continue
            con = JuMP.constraint_object(cref)
            new_f = _remap_expression(
                con.func, ref_map, master_inf)
            JuMP.@constraint(
                master_inf, new_f in con.set)
        end
    end

    # Copy GDP data manually
    _copy_gdp_for_loa(model, master_inf, ref_map)

    # Strip nonlinear disjuncts and globals,
    # then BigM-reform
    DP._remove_nonlinear_disjuncts(master_inf)
    DP._remove_nonlinear_globals(master_inf)
    DP.reformulate_model(
        master_inf, DP.BigM(method.M_value))

    # Transcribe to flat MILP, copy to get
    # independent model from InfiniteOpt backend
    raw_flat, raw_fwd = transcribe_to_flat(master_inf)
    flat, copy_map = JuMP.copy_model(raw_flat)
    flat.ext[:GDP] = DP.GDPData{
        JuMP.Model, JuMP.VariableRef,
        JuMP.ConstraintRef}()
    JuMP.set_optimizer(flat, method.mip_optimizer)
    JuMP.set_silent(flat)
    # Remap tr_fwd through copy_map
    tr_fwd = Dict(
        k => [copy_map[v] for v in vs]
        for (k, vs) in raw_fwd)

    # Build bin_map: orig indicator → representative
    # flat binary (first support point)
    lv_map = Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel}
        }()
    orig_gdp = DP.gdp_data(model)
    copy_gdp = DP.gdp_data(master_inf)
    for (idx, _) in orig_gdp.logical_variables
        orig_lv = DP.LogicalVariableRef(model, idx)
        copy_lv = DP.LogicalVariableRef(
            master_inf, idx)
        lv_map[orig_lv] = copy_lv
    end
    ind_to_bin = DP._indicator_to_binary(master_inf)
    bin_map = Dict{DP.LogicalVariableRef, Any}()
    for (orig_ind, copy_ind) in lv_map
        if haskey(ind_to_bin, copy_ind)
            bv = ind_to_bin[copy_ind]
            if bv isa JuMP.AbstractVariableRef &&
                    haskey(tr_fwd, bv)
                # Representative: first flat binary
                bin_map[orig_ind] = tr_fwd[bv][1]
            else
                bin_map[orig_ind] = nothing
            end
        else
            bin_map[orig_ind] = nothing
        end
    end

    # Build ref_map for OA cuts: orig var → flat vars
    # Store tr_fwd keyed by original vars
    oa_ref_map = Dict{
        InfiniteOpt.GeneralVariableRef,
        Vector{JuMP.VariableRef}}()
    for (orig, copy) in ref_map
        _is_parameter(orig) && continue
        haskey(tr_fwd, copy) || continue
        oa_ref_map[orig] = tr_fwd[copy]
    end

    master = DP._LOAMaster{
        typeof(flat), typeof(oa_ref_map)}(
        flat, oa_ref_map, bin_map,
        JuMP.VariableRef[],
        JuMP.objective_function(flat),
        JuMP.objective_sense(flat),
        nl_globals)

    for result in init_results
        result.feasible && DP._add_oa_cuts(
            master, result, model, method)
        DP._add_no_good_cut(master, result.combo)
    end
    return master
end

# Copy GDP data (logical vars, disjunctions, disjunct
# constraints, logical constraints) from model to copy
# using ref_map for variable substitution.
function _copy_gdp_for_loa(
    model::InfiniteOpt.InfiniteModel,
    copy::InfiniteOpt.InfiniteModel,
    ref_map::Dict
    )
    orig_gdp = DP.gdp_data(model)
    copy_gdp = DP.gdp_data(copy)

    # Logical variables (reuse same indices).
    # Create the binary backing variable with the
    # same infinite parameters as the original
    # logical var (if any) so transcription expands
    # them per support point.
    lv_map = Dict{
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel},
        DP.LogicalVariableRef{InfiniteOpt.InfiniteModel}
        }()
    orig_ind_to_bin = DP._indicator_to_binary(model)
    for (idx, var_data) in orig_gdp.logical_variables
        new_data = DP.LogicalVariableData(
            var_data.variable, var_data.name)
        copy_gdp.logical_variables[idx] = new_data
        orig_lv = DP.LogicalVariableRef(model, idx)
        copy_lv = DP.LogicalVariableRef(copy, idx)
        lv_map[orig_lv] = copy_lv
        # Preserve infinite parameters of the
        # original binary (if infinite logical).
        orig_bv = orig_ind_to_bin[orig_lv]
        var_type = nothing
        if orig_bv isa
                InfiniteOpt.GeneralVariableRef
            prefs = InfiniteOpt.parameter_refs(
                orig_bv)
            if !isempty(prefs)
                var_type = InfiniteOpt.Infinite(
                    Tuple(ref_map[p] for p in prefs)...)
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
        DP._indicator_to_binary(copy)[copy_lv] = bvref
    end

    # Disjunct constraints
    dc_map = Dict{
        DP.DisjunctConstraintRef{
            InfiniteOpt.InfiniteModel},
        DP.DisjunctConstraintRef{
            InfiniteOpt.InfiniteModel}}()
    for (idx, dc_data) in orig_gdp.disjunct_constraints
        old_con = dc_data.constraint
        old_dc_ref = DP.DisjunctConstraintRef(
            model, idx)
        old_ind = orig_gdp.constraint_to_indicator[
            old_dc_ref]
        new_ind = lv_map[old_ind]
        new_func = DP._replace_variables_in_constraint(
            old_con.func, ref_map)
        new_con = JuMP.build_constraint(
            error, new_func, old_con.set,
            DP.Disjunct(new_ind))
        new_dc_ref = JuMP.add_constraint(
            copy, new_con, dc_data.name)
        dc_map[old_dc_ref] = new_dc_ref
    end

    # Disjunctions
    disj_map = Dict{
        DP.DisjunctionRef{InfiniteOpt.InfiniteModel},
        DP.DisjunctionRef{InfiniteOpt.InfiniteModel}
        }()
    for (idx, disj_data) in orig_gdp.disjunctions
        old_disj = disj_data.constraint
        new_inds = [lv_map[ind]
            for ind in old_disj.indicators]
        new_disj = DP.Disjunction(
            new_inds, old_disj.nested)
        copy_gdp.disjunctions[idx] =
            DP.ConstraintData(new_disj, disj_data.name)
        disj_map[
            DP.DisjunctionRef(model, idx)
        ] = DP.DisjunctionRef(copy, idx)
    end

    # Logical constraints (including exactly-1)
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

    # Exactly1 constraints mapping (reuse same index
    # since logical_constraints were added in order)
    for (d_ref, lc_ref) in
            orig_gdp.exactly1_constraints
        new_d = disj_map[d_ref]
        copy_gdp.exactly1_constraints[new_d] =
            DP.LogicalConstraintRef(copy,
                DP.LogicalConstraintIndex(
                    JuMP.index(lc_ref).value))
    end

    return lv_map
end

# Add per-support-point OA cuts on the flat master.
function DP._add_oa_cuts(
    master::DP._LOAMaster{
        <:JuMP.AbstractModel,
        <:Dict{InfiniteOpt.GeneralVariableRef,
               Vector{JuMP.VariableRef}}},
    result::DP._LOAIterationResult{
        InfiniteOpt.InfiniteModel, Vector{Float64}},
    model::InfiniteOpt.InfiniteModel,
    method::DP.LOA
    )
    sgn = master.obj_sense ==
        _MOI.MIN_SENSE ? -1 : 1
    oa_fwd = master.ref_map

    for (ind, active) in result.combo
        !active && continue
        haskey(DP._indicator_to_constraints(
            model), ind) || continue
        bin_var = get(master.bin_map, ind, nothing)

        for orig_cref in DP._indicator_to_constraints(
                model)[ind]
            orig_cref isa DP.DisjunctConstraintRef ||
                continue
            con_data = DP._disjunct_constraints(
                model)[JuMP.index(orig_cref)]
            con = con_data.constraint
            con.func isa JuMP.GenericAffExpr && continue

            dual_vec = get(
                result.duals, orig_cref, nothing)
            dual_vec === nothing && continue

            K = length(first(values(oa_fwd)))
            rhs = DP._set_rhs(con.set)

            for k in 1:K
                dual_val = k <= length(dual_vec) ?
                    dual_vec[k] : 0.0
                s = sign(sgn * dual_val)
                s == 0 && continue

                # Build xk and ref_map for support k
                xk_k = Dict{
                    JuMP.AbstractVariableRef, Float64}()
                ref_k = Dict{Any, Any}()
                for (v, fvs) in oa_fwd
                    haskey(result.x_values, v) ||
                        continue
                    xv = result.x_values[v]
                    idx = min(k, length(xv))
                    fidx = min(k, length(fvs))
                    xk_k[fvs[fidx]] = xv[idx]
                    ref_k[fvs[fidx]] = fvs[fidx]
                end

                # Linearize the constraint func at xk
                # on flat variables
                flat_func =
                    DP._replace_variables_in_constraint(
                        con.func,
                        Dict(v => fvs[min(k,length(fvs))]
                            for (v, fvs) in oa_fwd))
                lin_expr = DP._linearize_at(
                    flat_func, xk_k, ref_k)

                slack = JuMP.@variable(master.model,
                    lower_bound = 0.0,
                    upper_bound = method.max_slack)
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
    end

    # OA cuts for nonlinear global constraints.
    # Per-support-point: linearize on flat vars.
    K = isempty(oa_fwd) ? 0 :
        length(first(values(oa_fwd)))
    for (func, set) in master.nl_globals
        rhs = DP._set_rhs(set)
        signs = DP._oa_global_signs(set)
        for k in 1:K
            # Flat variable map at support k
            flat_sub = Dict(
                v => fvs[min(k, length(fvs))]
                for (v, fvs) in oa_fwd)
            # xk and identity ref_map on flat vars
            xk_k = Dict{
                JuMP.AbstractVariableRef, Float64}()
            ref_k = Dict{Any, Any}()
            for (v, fvs) in oa_fwd
                haskey(result.x_values, v) || continue
                xv = result.x_values[v]
                idx = min(k, length(xv))
                fidx = min(k, length(fvs))
                xk_k[fvs[fidx]] = xv[idx]
                ref_k[fvs[fidx]] = fvs[fidx]
            end
            flat_func =
                DP._replace_variables_in_constraint(
                    func, flat_sub)
            lin_expr = DP._linearize_at(
                flat_func, xk_k, ref_k)
            for s in signs
                slack = JuMP.@variable(master.model,
                    lower_bound = 0.0,
                    upper_bound = method.max_slack)
                push!(master.slack_vars, slack)
                lhs = s * (lin_expr - rhs) - slack
                cref = JuMP.@constraint(
                    master.model, lhs <= 0)
                push!(
                    DP._reformulation_constraints(
                        master.model), cref)
            end
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
        @info "FINALIZE combo: $(best_result.combo)"
        for (ind, active) in best_result.combo
            bv = DP._indicator_to_binary(model)[ind]
            @info "  $ind => $active, bv=$bv"
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
        # Set start values from best subproblem
        # (per-support-point → InfiniteOpt vars)
        for (v, vals) in best_result.x_values
            JuMP.is_valid(model, v) || continue
            vals isa AbstractVector || continue
            isempty(vals) && continue
            try
                JuMP.set_start_value(v, vals[1])
            catch
            end
        end
    end

    # BigM-reform only linear/quadratic disjuncts
    DP._remove_nonlinear_disjuncts(model)
    DP.reformulate_model(
        model, DP.BigM(method.M_value))

    # Re-add active nonlinear constraints directly
    for (func, set) in nl_active
        cref = JuMP.@constraint(model, func in set)
        push!(
            DP._reformulation_constraints(model), cref)
    end
    return
end

end
