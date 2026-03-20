################################################################################
#                        CUTTING PLANES SUBPROBLEM
################################################################################

# Configure optimizer on a subproblem.
function configure_optimizer(sub::GDPSubmodel,
    method::cutting_planes)
    JuMP.set_optimizer(sub.model, method.optimizer)
    JuMP.set_silent(sub.model)
    return
end

################################################################################
#                       EXTENSION POINT FUNCTIONS
################################################################################

# Collect decision variables for cutting planes.
# Extensions may override to customize variable
# collection.
function collect_cp_vars(model::JuMP.AbstractModel)
    return collect_all_vars(model)
end

# Build one cutting planes subproblem (SEP). dec_vars is
# the shared key space (collected once from the clean
# model). Extensions may override for custom subproblem
# construction.
function build_cp_subproblem(
    model::JuMP.AbstractModel,
    dec_vars::Vector{V},
    reform_method::AbstractReformulationMethod,
    method::cutting_planes
    ) where {V}
    copy, ref_map, _ = copy_gdp_model(model)
    reformulate_model(copy, reform_method)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    m2c = Dict(v => ref_map[v] for v in dec_vars)
    JuMP.@objective(copy, sense,
        _replace_variables_in_constraint(obj, m2c))
    fwd = Dict(v => [ref_map[v]] for v in dec_vars)
    sub = GDPSubmodel(copy, dec_vars, fwd)
    configure_optimizer(sub, method)
    return sub
end

# Set up the rBM (relaxed Big-M) subproblem.
# Reformulates the model in-place (no copy).
# Returns (rBM::GDPSubmodel, undo_fn). Extensions
# may override for custom rBM setup.
function setup_rbm(
    model::JuMP.AbstractModel,
    dec_vars::Vector{V},
    method::cutting_planes
    ) where {V}
    reformulate_model(model, BigM(method.M_value))
    fwd = Dict(v => [v] for v in dec_vars)
    sub = GDPSubmodel(model, dec_vars, fwd)
    configure_optimizer(sub, method)
    undo_relax = JuMP.relax_integrality(model)
    return sub, undo_relax
end

# Extract solution from a solved subproblem. Keyed by
# original dec_vars.
function _extract_solution(sub::GDPSubmodel)
    V = eltype(sub.dec_vars)
    first_var = first(values(sub.fwd))[1]
    T = JuMP.value_type(typeof(JuMP.owner_model(first_var)))
    sol = Dict{V, Vector{T}}()
    for var in sub.dec_vars
        tvars = sub.fwd[var]
        sol[var] = [JuMP.value(tv) for tv in tvars]
    end
    return sol
end

# Set the quadratic separation objective:
# min sum_var sum_k (x_k - rBM_k)^2.
function _set_sep_objective(sub::GDPSubmodel, rBM_sol)
    obj_expr = _zero_quad(sub.model)
    for var in sub.dec_vars
        tvars = sub.fwd[var]
        vals = rBM_sol[var]
        for k in 1:length(tvars)
            JuMP.add_to_expression!(
                obj_expr,
                (tvars[k] - vals[k]) *
                (tvars[k] - vals[k])
                )
        end
    end
    JuMP.@objective(sub.model, Min, obj_expr)
    return
end

# Solve the separation problem. Returns (sep_obj, sep_sol).
function _solve_separation(sep::GDPSubmodel, rBM_sol)
    _set_sep_objective(sep, rBM_sol)
    JuMP.optimize!(sep.model, ignore_optimize_hook = true)
    sep_obj = JuMP.objective_value(sep.model)
    sep_sol = _extract_solution(sep)
    return sep_obj, sep_sol
end

# Add a cut to a subproblem.
# cut: sum_var sum_k 2*(sep_k-rBM_k)*(x_k-sep_k) >= 0
function _add_cut(sub::GDPSubmodel, rBM_sol, sep_sol)
    cut_expr = _zero_aff(sub.model)
    for var in sub.dec_vars
        tvars = sub.fwd[var]
        rbm_vals = rBM_sol[var]
        sep_vals = sep_sol[var]
        for k in 1:length(tvars)
            xi = 2 * (sep_vals[k] - rbm_vals[k])
            JuMP.add_to_expression!(cut_expr, xi,
                tvars[k])
            JuMP.add_to_expression!(cut_expr,
                -xi * sep_vals[k])
        end
    end
    JuMP.@constraint(sub.model, cut_expr >= 0)
    return
end

# Add a cut to the original model. Extensions may
# override for custom cut representations.
function add_original_model_cut(
    model::JuMP.AbstractModel, dec_vars::Vector{V},
    rBM_sol::Dict{V, Vector{T1}},
    sep_sol::Dict{V, Vector{T2}}
    ) where {V, T1, T2}
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
    model::JuMP.AbstractModel, method::cutting_planes
    )
    _clear_reformulations(model)
    dec_vars = collect_cp_vars(model)

    # Build SEP first (from clean model)
    sep = build_cp_subproblem(model, dec_vars,
        Hull(), method)
    JuMP.relax_integrality(sep.model)

    # Set up rBM
    rBM, undo_relax = setup_rbm(model, dec_vars, method)

    # Cutting plane loop
    prev_sep_obj = Inf
    for iter in 1:method.max_iter
        # 1. Solve rBM
        JuMP.optimize!(rBM.model,
            ignore_optimize_hook = true)
        rBM_sol = _extract_solution(rBM)

        # 2. Solve SEP
        sep_obj, sep_sol = _solve_separation(sep,
            rBM_sol)

        # 3. Check convergence
        if sep_obj <= method.seperation_tolerance
            break
        end

        # 4. Check stalling
        rel_improvement = (prev_sep_obj - sep_obj) /
            max(abs(prev_sep_obj), 1e-10)
        if iter > 1 && rel_improvement < 0.01
            break
        end
        prev_sep_obj = sep_obj

        # 5. Add cuts
        _add_cut(rBM, rBM_sol, sep_sol)
        if rBM.model !== model
            add_original_model_cut(model, dec_vars,
                rBM_sol, sep_sol)
        end
    end

    if undo_relax !== nothing
        undo_relax()
    end

    # Mark model as ready (inner reformulate_model calls
    # set the solution method to BigM; override it here)
    _set_solution_method(model, method)
    _set_ready_to_optimize(model, true)
    return
end

################################################################################
#                              ERROR MESSAGES
################################################################################

function reformulate_model(
    ::M, ::cutting_planes
    ) where {M}
    error("reformulate_model not implemented for " *
          "model type `$(M)`.")
end
