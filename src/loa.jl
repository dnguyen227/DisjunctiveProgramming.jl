################################################################################
#                    LOGIC-BASED OUTER APPROXIMATION (LOA)
################################################################################
# Implementation of the LOA algorithm from:
#   Türkay & Grossmann (1996), Comp. & Chem. Eng. 20(8), 959-978
#   Trespalacios & Grossmann (2014), Review of MINLP & GDP Methods
# Closely follows Pyomo's GDPopt LOA implementation.
################################################################################

"""
    LOA{O} <: AbstractReformulationMethod

Logic-based Outer Approximation solver for GDP models.

**Fields**
- `optimizer::O`: Optimizer for solving NLP subproblems and MILP
  master problems (required).
- `max_iter::Int`: Maximum LOA iterations (default = `100`).
- `atol::Float64`: Absolute convergence tolerance (default = `1e-6`).
- `rtol::Float64`: Relative convergence tolerance (default = `1e-4`).
- `M_value::Float64`: Big-M value for master reformulation
  (default = `1e9`).
"""
struct LOA{O} <: AbstractReformulationMethod
    optimizer::O
    max_iter::Int
    atol::Float64
    rtol::Float64
    M_value::Float64
    function LOA(
        optimizer::O;
        max_iter::Int = 100,
        atol::Float64 = 1e-6,
        rtol::Float64 = 1e-4,
        M_value::Float64 = 1e9
        ) where {O}
        new{O}(optimizer, max_iter, atol, rtol, M_value)
    end
end

################################################################################
#                           MAIN ALGORITHM
################################################################################
# Override reformulate_model for LOA (iterative method pattern,
# same as cutting_planes).
function reformulate_model(
    model::JuMP.AbstractModel, method::LOA
    )
    _clear_reformulations(model)
    disj_info = _collect_disjunction_info(model)

    # Step 1: Set covering initialization
    combos = _set_covering_combos(disj_info, method)

    # Step 2: Solve init subproblems, collect solutions
    z_upper = Inf
    best_sol = nothing
    init_results = _LOAIterationResult[]
    for combo in combos
        result = _solve_loa_subproblem(
            model, combo, disj_info, method)
        push!(init_results, result)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
            best_sol = result.x_values
        end
    end

    # Step 3: Build MILP master
    master, master_maps = _build_loa_master(
        model, disj_info, init_results, method)

    # Step 4: Main LOA loop
    z_lower = -Inf
    for iter in 1:method.max_iter
        JuMP.optimize!(master, ignore_optimize_hook = true)
        if termination_status(master) != MOI.OPTIMAL &&
            termination_status(master) != MOI.LOCALLY_SOLVED
            break
        end
        z_lower = objective_value(master)
        if _loa_converged(z_upper, z_lower, method)
            break
        end

        # Extract selected combo from master solution
        combo = _extract_combo_from_master(
            master, master_maps, disj_info)

        # Solve NLP subproblem
        result = _solve_loa_subproblem(
            model, combo, disj_info, method)
        if result.feasible && result.objective < z_upper
            z_upper = result.objective
            best_sol = result.x_values
        end

        # Add OA cuts and no-good cut to master
        _add_oa_cuts_to_master!(
            master, master_maps, result, disj_info)
        _add_no_good_cut_to_master!(
            master, master_maps, combo, disj_info)
    end

    # Step 5: Final reformulation — apply BigM so the original
    # model is in a state JuMP can optimize
    reformulate_model(model, BigM(method.M_value))
    return
end

################################################################################
#                          DATA STRUCTURES
################################################################################
# Store info about each disjunction and its disjuncts.
struct _DisjunctionInfo
    disjunction_indices::Vector{DisjunctionIndex}
    # disjunction index => vector of indicator LogicalVariableRefs
    indicators::Dict{DisjunctionIndex, Vector}
    # LogicalVariableRef => vector of DisjunctConstraintRefs
    disjunct_constraints::Dict{Any, Vector}
end

# Result from solving an NLP subproblem.
struct _LOAIterationResult
    combo::Dict{Any, Bool}   # indicator => true/false
    x_values::Dict           # variable => value
    objective::Float64
    feasible::Bool
    B_set::Vector  # indicators that were True (infeasible only)
    N_set::Vector  # indicators that were False (infeasible only)
end

################################################################################
#                     DISJUNCTION INFO COLLECTION
################################################################################
# Gather the disjunction structure from the GDP model.
function _collect_disjunction_info(model::JuMP.AbstractModel)
    disj_indices = DisjunctionIndex[]
    indicators_map = Dict{DisjunctionIndex, Vector}()
    dc_map = Dict{Any, Vector}()
    ind_to_cons = _indicator_to_constraints(model)

    for (idx, disj_data) in _disjunctions(model)
        disj = disj_data.constraint
        disj.nested && continue
        push!(disj_indices, idx)
        inds = disj.indicators
        indicators_map[idx] = collect(inds)
        for ind in inds
            if haskey(ind_to_cons, ind)
                dc_map[ind] = [ref for ref in ind_to_cons[ind]
                               if ref isa DisjunctConstraintRef]
            else
                dc_map[ind] = []
            end
        end
    end
    return _DisjunctionInfo(disj_indices, indicators_map, dc_map)
end

################################################################################
#                      SET COVERING INITIALIZATION
################################################################################
# Find a minimum set of disjunct combos such that every disjunct
# appears True in at least one combo.
# Following Türkay & Grossmann (1996) eq. 14.
function _set_covering_combos(
    info::_DisjunctionInfo, method::LOA
    )
    all_combos = _cartesian_product(info)

    # Collect all individual disjuncts that need covering
    all_disjuncts = []
    for idx in info.disjunction_indices
        for ind in info.indicators[idx]
            push!(all_disjuncts, ind)
        end
    end

    # Greedy set covering: pick combos covering most uncovered
    uncovered = Set(all_disjuncts)
    selected_combos = Dict{Any, Bool}[]

    while !isempty(uncovered)
        best_combo = nothing
        best_count = 0
        for combo in all_combos
            count = sum((ind in uncovered ? 1 : 0)
                        for (_, ind) in combo)
            if count > best_count
                best_count = count
                best_combo = combo
            end
        end
        best_combo === nothing && break
        # Convert to Dict{indicator => Bool}
        combo_dict = Dict{Any, Bool}()
        for (disj_idx, active_ind) in best_combo
            for ind in info.indicators[disj_idx]
                combo_dict[ind] = (ind == active_ind)
            end
        end
        push!(selected_combos, combo_dict)
        for (_, active_ind) in best_combo
            delete!(uncovered, active_ind)
        end
    end
    return selected_combos
end

# Cartesian product over disjunctions. Returns vector of vectors
# of (DisjunctionIndex, indicator) tuples.
function _cartesian_product(info::_DisjunctionInfo)
    combos = Vector{Tuple}[]
    _cartesian_helper!(
        combos, info.disjunction_indices,
        info.indicators, Tuple[], 1)
    return combos
end

function _cartesian_helper!(
    result, disj_indices, indicators_map, current, depth
    )
    if depth > length(disj_indices)
        push!(result, copy(current))
        return
    end
    idx = disj_indices[depth]
    for ind in indicators_map[idx]
        push!(current, (idx, ind))
        _cartesian_helper!(
            result, disj_indices, indicators_map,
            current, depth + 1)
        pop!(current)
    end
end

################################################################################
#                      NLP SUBPROBLEM SOLVER
################################################################################
# Build and solve the NLP subproblem for a fixed Boolean assignment.
# Matches (sub-LBOA) from Trespalacios (2014).
function _solve_loa_subproblem(
    model::JuMP.AbstractModel,
    combo::Dict{Any, Bool},
    info::_DisjunctionInfo,
    method::LOA
    )
    # Copy model, reformulate with BigM, fix binaries
    sub, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(sub, method.optimizer)
    JuMP.set_silent(sub)
    reformulate_model(sub, BigM(method.M_value))

    # Fix logical variable binaries according to combo
    ind_to_bin = _indicator_to_binary(sub)
    for (ind, active) in combo
        mapped_ind = lv_map[ind]
        if haskey(ind_to_bin, mapped_ind)
            bin_var = ind_to_bin[mapped_ind]
            if bin_var isa JuMP.AbstractVariableRef
                JuMP.fix(bin_var, active ? 1.0 : 0.0;
                         force = true)
            end
        end
    end

    JuMP.optimize!(sub, ignore_optimize_hook = true)
    status = termination_status(sub)
    feasible = (status == MOI.OPTIMAL ||
                status == MOI.LOCALLY_SOLVED ||
                status == MOI.FEASIBLE_POINT)

    if feasible
        x_vals = Dict{Any, Float64}()
        for var in collect_all_vars(model)
            x_vals[var] = JuMP.value(ref_map[var])
        end
        return _LOAIterationResult(
            combo, x_vals, objective_value(sub), true, [], [])
    else
        return _solve_loa_feasibility(
            model, combo, info, method)
    end
end

################################################################################
#                    FEASIBILITY SUBPROBLEM
################################################################################
# Solve (feas-LBOA): minimize max constraint violation.
function _solve_loa_feasibility(
    model::JuMP.AbstractModel,
    combo::Dict{Any, Bool},
    info::_DisjunctionInfo,
    method::LOA
    )
    feas, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(feas, method.optimizer)
    JuMP.set_silent(feas)
    reformulate_model(feas, BigM(method.M_value))

    # Fix logical variable binaries
    ind_to_bin = _indicator_to_binary(feas)
    for (ind, active) in combo
        mapped_ind = lv_map[ind]
        if haskey(ind_to_bin, mapped_ind)
            bin_var = ind_to_bin[mapped_ind]
            if bin_var isa JuMP.AbstractVariableRef
                JuMP.fix(bin_var, active ? 1.0 : 0.0;
                         force = true)
            end
        end
    end

    # Add slack variable and modify objective
    @variable(feas, _loa_slack)
    # TODO: Properly relax constraints by u.
    @objective(feas, Min, _loa_slack)
    JuMP.optimize!(feas, ignore_optimize_hook = true)

    # Extract solution if available, otherwise use zeros
    x_vals = Dict{Any, Float64}()
    has_sol = (termination_status(feas) == MOI.OPTIMAL ||
               termination_status(feas) == MOI.LOCALLY_SOLVED ||
               termination_status(feas) == MOI.FEASIBLE_POINT)
    for var in collect_all_vars(model)
        if has_sol
            val = try JuMP.value(ref_map[var]) catch; 0.0 end
            x_vals[var] = isnan(val) ? 0.0 : val
        else
            x_vals[var] = 0.0
        end
    end

    B_set = [ind for (ind, active) in combo if active]
    N_set = [ind for (ind, active) in combo if !active]
    return _LOAIterationResult(
        combo, x_vals, Inf, false, B_set, N_set)
end

################################################################################
#                      MASTER PROBLEM BUILDER
################################################################################
# Build the MILP master with initial OA + no-good cuts.
function _build_loa_master(
    model::JuMP.AbstractModel,
    info::_DisjunctionInfo,
    init_results::Vector{_LOAIterationResult},
    method::LOA
    )
    master, ref_map, lv_map = copy_gdp_model(model)
    JuMP.set_optimizer(master, method.optimizer)
    JuMP.set_silent(master)
    reformulate_model(master, BigM(method.M_value))

    master_maps = (ref_map = ref_map, lv_map = lv_map,
                   ind_to_bin = _indicator_to_binary(master))

    for result in init_results
        _add_oa_cuts_to_master!(
            master, master_maps, result, info)
    end
    for result in init_results
        _add_no_good_cut_to_master!(
            master, master_maps, result.combo, info)
    end
    return master, master_maps
end

################################################################################
#                        OA CUT GENERATION
################################################################################
# Add outer approximation cuts at solution point xk.
#
# For each disjunct constraint r_ki(x) <= 0 that was active:
#   r_ki(xk) + ∇r_ki(xk)ᵀ(x - xk) <= M*(1 - y_ki)
#
# AffExpr: linearization = the constraint itself (exact).
# QuadExpr: linearization is the first-order Taylor approx.
function _add_oa_cuts_to_master!(
    master::JuMP.AbstractModel, master_maps,
    result::_LOAIterationResult,
    info::_DisjunctionInfo
    )
    ref_map = master_maps.ref_map
    lv_map = master_maps.lv_map
    ind_to_bin = master_maps.ind_to_bin
    xk = result.x_values

    for (ind, active) in result.combo
        !active && continue
        !haskey(info.disjunct_constraints, ind) && continue

        mapped_ind = lv_map[ind]
        bin_var = nothing
        if haskey(ind_to_bin, mapped_ind)
            bv = ind_to_bin[mapped_ind]
            if bv isa JuMP.AbstractVariableRef
                bin_var = bv
            end
        end

        for con_ref in info.disjunct_constraints[ind]
            con_data = _disjunct_constraints(
                JuMP.owner_model(ind))[JuMP.index(con_ref)]
            con = con_data.constraint
            lin_expr = _linearize_at(con.func, xk, ref_map)
            rhs = _set_rhs(con.set)

            if bin_var !== nothing
                M = 1e9
                @constraint(master,
                    lin_expr <= rhs + M * (1 - bin_var))
            else
                @constraint(master, lin_expr <= rhs)
            end
        end
    end
end

# Linearize an AffExpr at point xk (exact, no Taylor needed).
function _linearize_at(
    func::JuMP.GenericAffExpr, xk::Dict, ref_map
    )
    result = JuMP.AffExpr(func.constant)
    for (var, coef) in func.terms
        JuMP.add_to_expression!(
            result, coef, ref_map[var])
    end
    return result
end

# Linearize a QuadExpr at point xk (first-order Taylor).
# f(xk) + ∇f(xk)ᵀ(x - xk)
# = -xk'Q*xk + (2Q*xk + a)'x + b
function _linearize_at(
    func::JuMP.GenericQuadExpr, xk::Dict, ref_map
    )
    grad = Dict{Any, Float64}()

    # Affine contribution: a_i
    for (var, coef) in func.aff.terms
        grad[var] = get(grad, var, 0.0) + coef
    end

    # Quadratic contribution: 2*Q_ij*xk_j
    for (pair, coef) in func.terms
        vi, vj = pair.a, pair.b
        xk_vi = get(xk, vi, 0.0)
        xk_vj = get(xk, vj, 0.0)
        if vi == vj
            grad[vi] = get(grad, vi, 0.0) + 2 * coef * xk_vi
        else
            grad[vi] = get(grad, vi, 0.0) + coef * xk_vj
            grad[vj] = get(grad, vj, 0.0) + coef * xk_vi
        end
    end

    # Evaluate f(xk)
    f_xk = func.aff.constant
    for (var, coef) in func.aff.terms
        f_xk += coef * get(xk, var, 0.0)
    end
    for (pair, coef) in func.terms
        f_xk += coef * get(xk, pair.a, 0.0) *
                        get(xk, pair.b, 0.0)
    end

    # Build: f(xk) + ∑ grad_i * (x_i - xk_i)
    constant = f_xk
    for (var, g) in grad
        constant -= g * get(xk, var, 0.0)
    end
    result = JuMP.AffExpr(constant)
    for (var, g) in grad
        JuMP.add_to_expression!(
            result, g, ref_map[var])
    end
    return result
end

# Extract RHS value from an MOI set.
_set_rhs(set::_MOI.LessThan) = set.upper
_set_rhs(set::_MOI.GreaterThan) = set.lower
_set_rhs(set::_MOI.EqualTo) = set.value
_set_rhs(::Any) = 0.0

################################################################################
#                       NO-GOOD CUT GENERATION
################################################################################
# Add no-good cut to prevent revisiting the same Boolean combo.
# Cut: ∑(1 - y_i for i in B) + ∑(y_i for i in N) >= 1
function _add_no_good_cut_to_master!(
    master::JuMP.AbstractModel, master_maps,
    combo::Dict{Any, Bool}, info::_DisjunctionInfo
    )
    ind_to_bin = master_maps.ind_to_bin
    lv_map = master_maps.lv_map
    cut_expr = JuMP.AffExpr(0.0)

    for (ind, active) in combo
        mapped_ind = lv_map[ind]
        if haskey(ind_to_bin, mapped_ind)
            bin_var = ind_to_bin[mapped_ind]
            if bin_var isa JuMP.AbstractVariableRef
                if active  # (1 - y_i) term
                    JuMP.add_to_expression!(
                        cut_expr, -1.0, bin_var)
                    JuMP.add_to_expression!(cut_expr, 1.0)
                else  # y_i term
                    JuMP.add_to_expression!(
                        cut_expr, 1.0, bin_var)
                end
            end
        end
    end
    @constraint(master, cut_expr >= 1.0)
end

################################################################################
#                     MASTER SOLUTION EXTRACTION
################################################################################
# Read binary variable values from the master solution and
# determine the active combo.
function _extract_combo_from_master(
    master::JuMP.AbstractModel, master_maps,
    info::_DisjunctionInfo
    )
    ind_to_bin = master_maps.ind_to_bin
    lv_map = master_maps.lv_map
    combo = Dict{Any, Bool}()

    for idx in info.disjunction_indices
        for ind in info.indicators[idx]
            mapped_ind = lv_map[ind]
            if haskey(ind_to_bin, mapped_ind)
                bin_var = ind_to_bin[mapped_ind]
                if bin_var isa JuMP.AbstractVariableRef
                    combo[ind] = JuMP.value(bin_var) > 0.5
                else
                    combo[ind] = false
                end
            else
                combo[ind] = false
            end
        end
    end
    return combo
end

################################################################################
#                       CONVERGENCE CHECK
################################################################################
function _loa_converged(
    z_upper::Float64, z_lower::Float64, method::LOA
    )
    gap = z_upper - z_lower
    gap <= method.atol && return true
    if abs(z_upper) > 1e-10 &&
        gap / abs(z_upper) <= method.rtol
        return true
    end
    return false
end

################################################################################
#                          ERROR FALLBACK
################################################################################
function reformulate_model(::M, ::LOA) where {M}
    error("reformulate_model not implemented for " *
          "model type `$(M)` with LOA.")
end
