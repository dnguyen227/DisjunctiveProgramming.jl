################################################################################
#                        CUTTING PLANES SUBPROBLEM
################################################################################

# Configure optimizer on a subproblem.
function configure_optimizer(
    sub::GDPSubmodel, method::CuttingPlanes
    )
    JuMP.set_optimizer(sub.model, method.optimizer)
    JuMP.set_silent(sub.model)
    return
end

################################################################################
#                       EXTENSION POINT FUNCTIONS
################################################################################

# Collect decision variables for cutting planes. Extensions may override to
# customize variable collection.
function collect_cp_vars(model::JuMP.AbstractModel)
    return collect_all_vars(model)
end

# Build one cutting planes subproblem (SEP). dec_vars is the shared key space
# (collected once from the clean model). Extensions may override for custom
# subproblem construction.
function build_cp_subproblem(
    model::JuMP.AbstractModel,
    dec_vars::AbstractVector,
    reform_method::AbstractReformulationMethod,
    method::CuttingPlanes
    )
    copy, ref_map, _ = copy_gdp_model(model)
    reformulate_model(copy, reform_method)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    V = JuMP.variable_ref_type(model)
    m2c = Dict{V, V}(v => ref_map[v] for v in dec_vars)
    JuMP.@objective(copy, sense, _replace_variables_in_constraint(obj, m2c))
    fwd_map = Dict{V, Vector{V}}(v => [ref_map[v]] for v in dec_vars)
    sub = GDPSubmodel(copy, dec_vars, fwd_map)
    configure_optimizer(sub, method)
    return sub
end

# Set up the rBM (relaxed Big-M) subproblem. Reformulates the model in-place
# (no copy). Returns (rBM, undo_fn). Extensions may override for custom setup.
function setup_rbm(
    model::JuMP.AbstractModel,
    dec_vars::AbstractVector,
    method::CuttingPlanes
    )
    reformulate_model(model, BigM(method.M_value))
    V = JuMP.variable_ref_type(model)
    fwd_map = Dict{V, Vector{V}}(v => [v] for v in dec_vars)
    sub = GDPSubmodel(model, dec_vars, fwd_map)
    configure_optimizer(sub, method)
    undo_relax = JuMP.relax_integrality(model)
    return sub, undo_relax
end

# Extract solution from a solved subproblem, keyed by original dec_vars.
function _extract_solution(sub::GDPSubmodel)
    V = eltype(sub.dec_vars)
    first_var = first(values(sub.fwd_map))[1]
    T = JuMP.value_type(typeof(JuMP.owner_model(first_var)))
    sol = Dict{V, Vector{T}}()
    for var in sub.dec_vars
        tvars = sub.fwd_map[var]
        sol[var] = [JuMP.value(tv) for tv in tvars]
    end
    return sol
end

# Set quadratic separation objective: min Σ (x_k - rBM_k)².
function _set_sep_objective(
    sub::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    obj_expr = _zero_quad(sub.model)
    for var in sub.dec_vars
        tvars = sub.fwd_map[var]
        vals = rBM_sol[var]
        for k in 1:length(tvars)
            JuMP.add_to_expression!(obj_expr,
                (tvars[k] - vals[k]) *
                (tvars[k] - vals[k]))
        end
    end
    JuMP.@objective(sub.model, Min, obj_expr)
    return
end

# Solve the separation problem. Returns (sep_obj, sep_sol).
function _solve_separation(
    sep::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    _set_sep_objective(sep, rBM_sol)
    JuMP.optimize!(sep.model, ignore_optimize_hook = true)
    sep_obj = JuMP.objective_value(sep.model)
    sep_sol = _extract_solution(sep)
    return sep_obj, sep_sol
end

# Add cut to subproblem: Σ_var Σ_k 2*(sep_k - rBM_k)*(x_k - sep_k) ≥ 0
function _add_cut(
    sub::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}},
    sep_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    cut_expr = _zero_aff(sub.model)
    for var in sub.dec_vars
        tvars = sub.fwd_map[var]
        rbm_vals = rBM_sol[var]
        sep_vals = sep_sol[var]
        for k in 1:length(tvars)
            xi = 2 * (sep_vals[k] - rbm_vals[k])
            JuMP.add_to_expression!(cut_expr, xi, tvars[k])
            JuMP.add_to_expression!(cut_expr, -xi * sep_vals[k])
        end
    end
    JuMP.@constraint(sub.model, cut_expr >= 0)
    return
end

# Add a cut to the original model. Extensions may override for custom cut
# representations.
function add_original_model_cut(
    model::JuMP.AbstractModel,
    dec_vars::AbstractVector,
    rBM_sol::Dict, sep_sol::Dict
    )
    cut_expr = _zero_aff(model)
    for var in dec_vars
        xi = 2 * (sep_sol[var][1] - rBM_sol[var][1])
        sp = sep_sol[var][1]
        JuMP.add_to_expression!(cut_expr, xi, var)
        JuMP.add_to_expression!(cut_expr, -xi * sp)
    end
    JuMP.@constraint(model, cut_expr >= 0)
    return
end

################################################################################
#                        UNIFIED CUTTING PLANES LOOP
################################################################################

function reformulate_model(
    model::JuMP.AbstractModel,
    method::CuttingPlanes
    )
    _clear_reformulations(model)
    dec_vars = collect_cp_vars(model)

    # Build SEP subproblem first from the clean (unreformulated) model
    sep = build_cp_subproblem(model, dec_vars, Hull(), method)
    JuMP.relax_integrality(sep.model)

    # Set up rBM on the original model via in-place BigM reformulation
    rBM, undo_relax = setup_rbm(model, dec_vars, method)

    # Cutting plane loop: rBM <-> SEP until convergence (Trespalacios &
    # Grossmann 2016, Prop. 3.4)
    for iter in 1:method.max_iter
        # 1. Solve the relaxed Big-M (rBM) master problem for current iterate
        JuMP.optimize!(rBM.model, ignore_optimize_hook = true)
        rBM_sol = _extract_solution(rBM)

        # 2. Solve the SEP (separation subproblem) using the rBM solution
        sep_obj, sep_sol = _solve_separation(sep, rBM_sol)

        # 3. Check convergence: sep_obj ≈ 0 means rBM solution is already
        # Hull-feasible, so no separating cut exists
        if sep_obj <= method.seperation_tolerance
            break
        end

        # 4. Add separating cuts to both the rBM and the original model
        _add_cut(rBM, rBM_sol, sep_sol)
        if rBM.model !== model
            add_original_model_cut(model, dec_vars, rBM_sol, sep_sol)
        end
    end

    if undo_relax !== nothing
        undo_relax()
    end
    
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
    return
end

################################################################################
#                              ERROR MESSAGES
################################################################################

function reformulate_model(
    ::M, ::CuttingPlanes
    ) where {M}
    error("reformulate_model not implemented for model type `$(M)`.")
end
