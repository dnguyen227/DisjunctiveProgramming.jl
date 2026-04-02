################################################################################
#                       FUNCTIONS FOR LOOP
################################################################################

# Collect decision variables for cutting planes. Extensions may override to
# customize variable collection.
function collect_cutting_planes_vars(model::JuMP.AbstractModel)
    return collect_all_vars(model)
end

# Extract solution from a solved subproblem, keyed by original decision_vars.
function _extract_solution(sub::GDPSubmodel)
    V = eltype(sub.decision_vars)
    T = JuMP.value_type(typeof(sub.model))
    sol = Dict{V, Vector{T}}()
    for var in sub.decision_vars
        sol[var] = JuMP.value.(sub.fwd_map[var])
    end
    return sol
end

# Set quadratic separation objective: min Σ (x_k - rBM_k)².
function _set_separation_objective(
    sub::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    obj_expr = zero(JuMP.GenericQuadExpr{
        JuMP.value_type(typeof(sub.model)),
        JuMP.variable_ref_type(sub.model)}
        )
    for var in sub.decision_vars
        sub_vars = sub.fwd_map[var]
        vals = rBM_sol[var]
        for k in 1:length(sub_vars)
            JuMP.add_to_expression!(obj_expr,
                (sub_vars[k] - vals[k]) *
                (sub_vars[k] - vals[k])
                )
        end
    end
    JuMP.@objective(sub.model, Min, obj_expr)
    return
end

# Solve the separation problem. Returns (separation_obj, separation_sol).
function _solve_separation(
    separation::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    _set_separation_objective(separation, rBM_sol)
    JuMP.optimize!(separation.model, ignore_optimize_hook = true)
    separation_obj = JuMP.objective_value(separation.model)
    separation_sol = _extract_solution(separation)
    return separation_obj, separation_sol
end

# Add cut: Σ_var Σ_k 2*(separation_k - rBM_k)*(x_k - separation_k) ≥ 0
function _add_cut(
    sub::GDPSubmodel,
    rBM_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}},
    separation_sol::Dict{<:JuMP.AbstractVariableRef, <:Vector{<:Number}}
    )
    cut_expr = zero(JuMP.GenericAffExpr{
        JuMP.value_type(typeof(sub.model)),
        JuMP.variable_ref_type(sub.model)}
        )
    for var in sub.decision_vars
        sub_vars = sub.fwd_map[var]
        rbm_vals = rBM_sol[var]
        separation_vals = separation_sol[var]
        for k in 1:length(sub_vars)
            xi = 2 * (separation_vals[k] - rbm_vals[k])
            JuMP.add_to_expression!(cut_expr, xi, sub_vars[k])
            JuMP.add_to_expression!(cut_expr, -xi * separation_vals[k])
        end
    end
    JuMP.@constraint(sub.model, cut_expr >= 0)
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
    decision_vars = collect_cutting_planes_vars(model)

    # Build separation subproblem from the clean (unreformulated) model
    separation = copy_and_reformulate(
        model, decision_vars, Hull(), method)
    JuMP.relax_integrality(separation.model)

    # Set up rBM on the original model via in-place BigM reformulation
    rBM, undo_relax = reformulate_and_relax(
        model, decision_vars, BigM(method.M_value), method)

    # Cutting plane loop: rBM <-> SEP until convergence
    for iter in 1:method.max_iter
        # 1. Solve the relaxed Big-M (rBM) master problem for current iterate
        JuMP.optimize!(rBM.model, ignore_optimize_hook = true)
        rBM_sol = _extract_solution(rBM)

        # 2. Solve the SEP (separation subproblem) using the rBM solution
        separation_obj, separation_sol = _solve_separation(
            separation, rBM_sol)

        # 3. Check convergence: separation_obj ≈ 0 means rBM solution is already
        # Hull-feasible, so no separating cut exists
        if separation_obj <= method.seperation_tolerance
            break
        end

        # 4. Add separating cut to rBM
        _add_cut(rBM, rBM_sol, separation_sol)
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
