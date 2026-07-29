module InfiniteDisjunctiveProgramming

import JuMP.MOI as _MOI
import InfiniteOpt, JuMP
import DisjunctiveProgramming as DP
import AbstractGPs, KernelFunctions

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

# ------ GP active-learning M(d) (hard-coded for the mbm_gp experiments) ------
# Coordinates of every support, normalized to [0,1]^d so one isotropic
# lengthscale works across dimensions.
function _gp_support_coords(grids)
    idxs = CartesianIndices(length.(grids))
    los = [minimum(g) for g in grids]
    rng = [max(maximum(g) - minimum(g), eps()) for g in grids]
    X = [[(grids[d][I[d]] - los[d]) / rng[d] for d in 1:length(grids)]
         for I in idxs]
    return X, collect(idxs)
end

# Posterior mean and sd at all coords, given solved (index => M) samples.
# Outputs are standardized so the k*sd term scales with the M spread.
function _gp_mean_sd(X, solved)
    lis = collect(keys(solved))
    yt = [solved[li] for li in lis]
    ybar = sum(yt) / length(yt)
    ystd = max(sqrt(sum(abs2, yt .- ybar) / max(length(yt) - 1, 1)), 1e-8)
    kern = KernelFunctions.with_lengthscale(
        KernelFunctions.SqExponentialKernel(), 0.1)
    post = AbstractGPs.posterior(
        AbstractGPs.GP(kern)(X[lis], 1e-8), (yt .- ybar) ./ ystd)
    mz = AbstractGPs.mean(post, X)
    vz = max.(AbstractGPs.var(post, X), 0.0)
    return mz .* ystd .+ ybar, sqrt.(vz) .* ystd
end

# Count of per-support M subproblems solved (for the grid-vs-GP comparison).
const _M_SOLVE_COUNT = Ref(0)

# Original workflow: solve M at every support (grid). Kept for the mbm_gp
# vs grid comparison, gated by ENV["DP_MBM_GRID"].
function _grid_M_vals(objectives, inner_sub, method)
    M_vals = Array{Float64}(undef, size(objectives))
    for I in eachindex(objectives)
        _M_SOLVE_COUNT[] += 1
        m = DP.raw_M(inner_sub, objectives[I], method)
        m === nothing && return nothing
        M_vals[I] = m
    end
    return M_vals
end

# Solve M at actively-selected supports (max-UCB acquisition) and fill the
# rest with the UCB (mean + k*sd), a valid over-estimate. Returns a scalar
# when M is uniform (e.g. dependent multi-dim parameters where M does not
# vary), matching the grid workflow's early return.
function _gp_active_M_vals(objectives, inner_sub, method, sub, mini_expr)
    idxs = collect(CartesianIndices(objectives))
    n = length(idxs)
    k = 2.5
    budget = min(n, max(6, cld(n, 4)))
    solved = Dict{Int, Float64}()
    solve_at!(li) = begin
        _M_SOLVE_COUNT[] += 1
        m = DP.raw_M(inner_sub, objectives[idxs[li]], method)
        m === nothing && return false
        solved[li] = m
        true
    end
    for s in unique([1, cld(n + 1, 2), n])
        solve_at!(s) || return nothing
    end
    seed = collect(values(solved))
    all(==(first(seed)), seed) && return first(seed)   # uniform M -> scalar
    mini_prefs = InfiniteOpt.parameter_refs(mini_expr)
    reverse_map = Dict(ws[1] => v for (v, ws) in sub.fwd_map)
    prefs = Tuple(reverse_map[p] for p in mini_prefs)
    grids = Tuple(InfiniteOpt.supports(p) for p in prefs)
    X, _ = _gp_support_coords(grids)
    while length(solved) < budget
        ms, ss = _gp_mean_sd(X, solved)
        acq = ms .+ k .* ss
        for li in keys(solved)
            acq[li] = -Inf
        end
        solve_at!(argmax(acq)) || return nothing
    end
    ms, ss = _gp_mean_sd(X, solved)
    M_vals = Array{Float64}(undef, size(objectives))
    for (li, I) in enumerate(idxs)
        M_vals[I] = get(solved, li, ms[li] + k * ss[li])
    end
    return M_vals
end

# Transcribe mini_expr, then approximate M(d) with an actively-sampled GP
# instead of solving at every support; aggregate to a scalar if uniform.
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
    M_vals = haskey(ENV, "DP_MBM_GRID") ?
        _grid_M_vals(objectives, inner_sub, method) :
        _gp_active_M_vals(objectives, inner_sub, method, sub, mini_expr)
    M_vals === nothing && return nothing
    M_vals isa Number && return M_vals
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

end
