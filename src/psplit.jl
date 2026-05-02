################################################################################
#                              BUILD PARTITIONED EXPRESSION
################################################################################

function _build_partitioned_expression(
    expr::T,
    partition_variables::Vector{<:JuMP.AbstractVariableRef}
) where {T <: JuMP.GenericAffExpr}
    constant = get_constant(expr)
    new_affexpr = zero(T)
    for var in partition_variables
        JuMP.add_to_expression!(new_affexpr, JuMP.coefficient(expr, var), var) 
    end
    return new_affexpr, constant
end

function _build_partitioned_expression(
    expr::T,
    partition_variables::Vector{<:JuMP.AbstractVariableRef}
) where {T <: JuMP.GenericQuadExpr}
    new_quadexpr = zero(T)
    constant = get_constant(expr)
    for var in partition_variables
        for (pair, coeff) in expr.terms
            if pair.a == var && pair.b == var
                JuMP.add_to_expression!(new_quadexpr, coeff, var, var)
            elseif pair.a == var || pair.b == var
                error("Quadratic expression contains 
                bilinear term ($(pair.a), $(pair.b))")
            end
            
        end
    end
    new_aff, _ = _build_partitioned_expression(expr.aff, partition_variables)
    JuMP.add_to_expression!(new_quadexpr, new_aff)
    return new_quadexpr, constant
end

function _build_partitioned_expression(
    expr::T,
    partition_variables::Vector{<:JuMP.AbstractVariableRef}
) where {T <: JuMP.AbstractVariableRef}

    if expr in partition_variables
        return expr, zero(T)
    else
        return zero(T), zero(T)
    end
end

function _build_partitioned_expression(
    expr::T,
    partition_variables::Vector{<:JuMP.AbstractVariableRef}
) where {T <: Number}
    return expr, zero(T)
end

function _build_partitioned_expression(
    expr::T,
    partition_variables::Vector{<:JuMP.AbstractVariableRef}
) where {T <: JuMP.GenericNonlinearExpr}
    error("P-Split does not currently support nonlinear expressions $(expr)")
end


################################################################################
#                       BILINEAR LIFTING (Section 5)
################################################################################
# Implements the convex-underestimator escape hatch from Kronqvist, Misener,
# Tsay (2022) Section 5 "Beyond convex disjuncts" with one extra ingredient:
# rather than using the McCormick envelope as a stand-alone relaxation of the
# disjunct constraint, we lift each bilinear `c*x*y` to a fresh global
# auxiliary variable `w_xy`, pin `w_xy == x*y` exactly via a global
# definitional constraint (the underlying solver enforces it -- e.g. Gurobi
# `NonConvex=2` spatial branch-and-bound), and add the four McCormick cuts
# globally as valid LP-tightening inequalities. The disjunct then sees only
# `c*w_xy` (affine), satisfying P-split Assumption 1 while preserving
# objective parity with BigM/MBM/Hull on the original problem.

# Walk a quadratic expression and return (coeff, x, y) for every off-diagonal
# bilinear term. Square terms (a == b) and the affine part are skipped.
function _extract_bilinear_pairs(quad::JuMP.GenericQuadExpr{T, V}) where {T, V}
    pairs = Vector{Tuple{T, V, V}}()
    for (vars, coeff) in quad.terms
        if vars.a !== vars.b
            push!(pairs, (coeff, vars.a, vars.b))
        end
    end
    return pairs
end

# Read a variable's box bounds, erroring early if either side is missing.
# This is the single source of truth for "must have both bounds" in the
# convexify path.
function _box_bounds(v::JuMP.AbstractVariableRef)
    JuMP.has_lower_bound(v) && JuMP.has_upper_bound(v) || error(
        "PSplit convexify: variable $v requires both lower and upper " *
        "bounds for the McCormick lifting"
        )
    return JuMP.lower_bound(v), JuMP.upper_bound(v)
end

# Locate a cached lift for the unordered pair (x, y), or `nothing`.
function _lookup_lift(
    cache::Dict{Tuple{V, V}, V},
    x::V,
    y::V
    ) where {V <: JuMP.AbstractVariableRef}
    for ((a, b), w) in cache
        if (a === x && b === y) || (a === y && b === x)
            return w
        end
    end
    return nothing
end

# Lift the bilinear x*y to a fresh global variable w with:
#   1. an exact definitional equality `w == x*y` (solver-enforced),
#   2. four McCormick cuts as global valid inequalities,
#   3. box bounds on w derived from the corner products.
# Returns the (cached or freshly created) auxiliary variable.
function _lift_bilinear!(
    model::JuMP.AbstractModel,
    x::V,
    y::V,
    method::PSplit{V}
    ) where {V <: JuMP.AbstractVariableRef}
    cached = _lookup_lift(method.bilinear_lifts, x, y)
    cached === nothing || return cached

    xL, xU = _box_bounds(x)
    yL, yU = _box_bounds(y)
    corners = (xL * yL, xL * yU, xU * yL, xU * yU)
    wL, wU = minimum(corners), maximum(corners)

    # `VariableProperties(x*y)` carries the parameter refs through the
    # InfiniteOpt extension overload, so `w` is `Infinite(...)` whenever
    # x and y are infinite variables.
    props = VariableProperties(x * y)
    w = create_variable(model, props)
    JuMP.set_name(w, "w_lift_$(hash((x, y)))")
    JuMP.set_lower_bound(w, wL)
    JuMP.set_upper_bound(w, wU)

    JuMP.@constraint(model, w == x * y)
    JuMP.@constraint(model, w >= yL * x + xL * y - xL * yL)
    JuMP.@constraint(model, w >= yU * x + xU * y - xU * yU)
    JuMP.@constraint(model, w <= yU * x + xL * y - xL * yU)
    JuMP.@constraint(model, w <= yL * x + xU * y - xU * yL)

    # `w` must live in some partition block, otherwise
    # `_build_partitioned_expression` silently drops its coefficient and the
    # disjunct constraint loses the lifted term. Place it next to `x` (the
    # first variable of the bilinear pair) so the partition stays
    # interpretable; fall back to block 1 if `x` is in no block (e.g. a
    # hand-crafted partition that intentionally excludes it).
    blk = findfirst(p -> any(v -> v === x, p), method.partition)
    push!(method.partition[blk === nothing ? 1 : blk], w)

    # Register w with the bound-info dictionary that `_bound_auxiliary`
    # reads from. The pre-reformulation pass populated it before w existed.
    _variable_bounds(model)[w] = set_variable_bound_info(w, method)

    method.bilinear_lifts[(x, y)] = w
    return w
end

# Substitute every bilinear term `c*x*y` in `con.func` with `c*w_xy` (lifting
# as needed) and return the resulting affine-only `ScalarConstraint` carrying
# the same set. Square terms (x^2) pass through unchanged for the existing
# diagonal-quadratic dispatch downstream.
function _convexify_quad(
    con::JuMP.ScalarConstraint{F, S},
    model::JuMP.AbstractModel,
    method::PSplit
    ) where {F <: JuMP.GenericQuadExpr, S}
    bilinears = _extract_bilinear_pairs(con.func)
    isempty(bilinears) && return con

    new_quad = zero(F)
    for (vars, coeff) in con.func.terms
        if vars.a === vars.b
            JuMP.add_to_expression!(new_quad, coeff, vars.a, vars.b)
        end
    end
    JuMP.add_to_expression!(new_quad, con.func.aff)

    for (coeff, x, y) in bilinears
        w = _lift_bilinear!(model, x, y, method)
        JuMP.add_to_expression!(new_quad, coeff, w)
    end

    return JuMP.ScalarConstraint(new_quad, con.set)
end


################################################################################
#                              BOUND AUXILIARY
################################################################################
function _bound_auxiliary(
    model::M,
    v::JuMP.AbstractVariableRef,
    func::JuMP.GenericAffExpr{T,V},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, V}

    lower_bound = has_lower_bound(v) ? lower_bound(v) : zero(T)
    upper_bound = has_upper_bound(v) ? upper_bound(v) : zero(T)

    for (var, coeff) in func.terms
        if var != v
            JuMP.is_binary(var) && continue
            var_lb, var_ub = variable_bound_info(var)
            if coeff > 0.0
                lower_bound += coeff * var_lb
                upper_bound += coeff * var_ub
            else
                lower_bound += coeff * var_ub
                upper_bound += coeff * var_lb
            end
        end
    end
    JuMP.set_lower_bound(v, lower_bound)
    JuMP.set_upper_bound(v, upper_bound)
    _variable_bounds(model)[v] = set_variable_bound_info(v, method)
end    

function _bound_auxiliary(
    model::M,
    v::JuMP.AbstractVariableRef,
    func::Number,
    method::PSplit
) where {M <: JuMP.AbstractModel}
    JuMP.set_lower_bound(v, func)
    JuMP.set_upper_bound(v, func)
    _variable_bounds(model)[v] = set_variable_bound_info(v, method)
    return
end

function _bound_auxiliary(
    model::M,
    v::JuMP.AbstractVariableRef,
    func::JuMP.GenericQuadExpr{T,V},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, V}
    
    # Handle linear terms
    _bound_auxiliary(model, v, func.aff, method)
    lower_bound = JuMP.lower_bound(v)
    upper_bound = JuMP.upper_bound(v)
    
    # Handle quadratic terms
    for (vars, coeff) in func.terms
        var = vars.a 
        if var != v
            JuMP.is_binary(var) && continue
            lb, ub = variable_bound_info(var)
            
            # For x^2 terms
            sq_min = min(lb^2, ub^2, zero(T))
            sq_max = max(lb^2, ub^2, zero(T))
            
            if coeff > 0.0
                lower_bound += coeff * sq_min
                upper_bound += coeff * sq_max
            else
                lower_bound += coeff * sq_max
                upper_bound += coeff * sq_min
            end
        end
    end
    
    # Add constant term
    const_term = func.aff.constant
    # lower_bound += const_term
    # upper_bound += const_term
    
    JuMP.set_lower_bound(v, lower_bound)
    JuMP.set_upper_bound(v, upper_bound)
    _variable_bounds(model)[v] = set_variable_bound_info(v, method)
end

function _bound_auxiliary(
    model::M,
    v::JuMP.AbstractVariableRef,
    func::JuMP.AbstractVariableRef,
    method::PSplit
) where {M <: JuMP.AbstractModel}
    T = JuMP.value_type(M)
    lower_bound = zero(T)
    upper_bound = zero(T)
    if func != v
        lower_bound = variable_bound_info(func)[1]
        upper_bound = variable_bound_info(func)[2]
        JuMP.set_lower_bound(v, lower_bound)
        JuMP.set_upper_bound(v, upper_bound)
    else
        JuMP.set_lower_bound(v,lower_bound)
        JuMP.set_upper_bound(v,upper_bound)
    end
    _variable_bounds(model)[v] = set_variable_bound_info(v, method)
end

requires_variable_bound_info(method::Union{PSplit, _PSplit}) = true

function set_variable_bound_info(
    vref::JuMP.AbstractVariableRef, 
    ::Union{PSplit, _PSplit}
    )
    if !has_lower_bound(vref) || !has_upper_bound(vref)
        error("Variable $vref must have both lower and upper bounds defined when
         using the PSplit reformulation."
         )
    else
        lb = min(0, lower_bound(vref))
        ub = max(0, upper_bound(vref))
    end
    return lb, ub
end

################################################################################
#                              REFORMULATE DISJUNCT
################################################################################

function reformulate_disjunction(
    model::JuMP.AbstractModel, 
    disj::Disjunction, 
    method::PSplit{V}
) where {V <: JuMP.AbstractVariableRef}
    ref_cons = Vector{JuMP.AbstractConstraint}() #store reformulated constraints
    disj_vrefs = _get_disjunction_variables(model, disj)
    sum_constraints = Dict{LogicalVariableRef, Vector{<:JuMP.AbstractConstraint}}()
    aux_vars = Set{V}()
    for d in disj.indicators
        partitioned_constraints, sum_constraints[d], vars = _partition_disjunct(model, d, method)
        append!(ref_cons, partitioned_constraints)
        union!(aux_vars, vars)
    end

    psplit = _PSplit(method, model)
    psplit.hull = _Hull(Hull(), union(disj_vrefs, aux_vars))
    psplit.sum_constraints = sum_constraints
    for d in disj.indicators
        bvref = binary_variable(d)
        for vref in disj_vrefs
            push!(psplit.hull.disjunction_variables[vref], vref)
            psplit.hull.disjunct_variables[vref, bvref] = vref
        end
        _disaggregate_variables(model, d, aux_vars, psplit.hull)
        _reformulate_disjunct(model, ref_cons, d, psplit)
    end
    for vref in aux_vars
        _aggregate_variable(model, ref_cons, vref, psplit.hull)
    end
    
    return ref_cons
end

function reformulate_disjunction(
    model::JuMP.AbstractModel,
    disj::Disjunction,
    method::_PSplit
)
    return reformulate_disjunction(
        model, disj,
        PSplit(method.partition; convexify = method.convexify)
        )
end

function _reformulate_disjunct(
    model::JuMP.AbstractModel, 
    ref_cons::Vector{JuMP.AbstractConstraint}, 
    lvref::LogicalVariableRef, 
    method::_PSplit
    )
    #reformulate each constraint and add to the model
    bvref = binary_variable(lvref)
    haskey(method.sum_constraints, lvref) || return
    constraints = method.sum_constraints[lvref]
    for con in constraints
        append!(ref_cons, reformulate_disjunct_constraint(model, con, bvref, method.hull))
    end
    return
end

function _partition_disjunct(
    model::M, 
    lvref::LogicalVariableRef, 
    method::PSplit
) where {M <: JuMP.AbstractModel}
    !haskey(_indicator_to_constraints(model), lvref) && return #skip if disjunct is empty
    
    partitioned_constraints = Vector{AbstractConstraint}()
    sum_constraints = Vector{AbstractConstraint}()
    aux_vars = Set{JuMP.AbstractVariableRef}()
    for cref in _indicator_to_constraints(model)[lvref]
        con = JuMP.constraint_object(cref)
        if !(con isa Disjunction)
            if method.convexify == :mccormick &&
                con isa JuMP.ScalarConstraint &&
                con.func isa JuMP.GenericQuadExpr
                con = _convexify_quad(con, model, method)
            end
            part_con, sum_con, new_aux_vars = _build_partitioned_constraint(model, con, method)
            append!(partitioned_constraints, part_con)
            append!(sum_constraints, sum_con)
            union!(aux_vars, new_aux_vars)
        end
    end
    return partitioned_constraints, sum_constraints, aux_vars
end

#################################################################################
#                              BUILD PARTITIONED CONSTRAINT
#################################################################################
function _build_partitioned_constraint(
    model::M,
    con::JuMP.ScalarConstraint{T, S},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: _MOI.LessThan}
    val_type = JuMP.value_type(M)
    p = length(method.partition)
    v = Vector{JuMP.variable_ref_type(M)}(undef, p)
    _, constant = _build_partitioned_expression(con.func, method.partition[p])
    part_con = Vector{JuMP.AbstractConstraint}(undef, p)
    for i in 1:p
        func, _ = _build_partitioned_expression(con.func, method.partition[i])
        v[i] = create_variable(model, VariableProperties(func))
        JuMP.set_name(v[i], "v_$(hash(con))_$(i)")
        part_con[i] = JuMP.build_constraint(error, func - v[i], 
        MOI.LessThan(zero(val_type))
        )
        _bound_auxiliary(model, v[i], func, method)
    end
    sum_con = JuMP.@build_constraint(sum(v[i] for i in 1:p) + constant <= con.set.upper)

    return part_con, [sum_con], v
end

function _build_partitioned_constraint(
    model::M,
    con::JuMP.ScalarConstraint{T, S},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: _MOI.GreaterThan}
    val_type = JuMP.value_type(M)
    p = length(method.partition)
    part_con = Vector{JuMP.AbstractConstraint}(undef, p)
    v = Vector{JuMP.variable_ref_type(M)}(undef, p)
    _, constant = _build_partitioned_expression(con.func, method.partition[p])
    for i in 1:p
        func, _ = _build_partitioned_expression(con.func, method.partition[i])
        v[i] = create_variable(model, VariableProperties(func))
        JuMP.set_name(v[i], "v_$(hash(con))_$(i)")
        part_con[i] = JuMP.build_constraint(error, -func - v[i], 
        MOI.LessThan(zero(val_type))
        )
        _bound_auxiliary(model, v[i], -func, method)
    end
    sum_con = JuMP.@build_constraint(sum(v[i] for i in 1:p) - constant <= -con.set.lower)
    return part_con, [sum_con], v
end

function _build_partitioned_constraint(
    model::M,
    con::JuMP.ScalarConstraint{T, S},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: Union{_MOI.Interval, _MOI.EqualTo}}
    val_type = JuMP.value_type(M)
    p = length(method.partition)
    part_con_lt = Vector{JuMP.AbstractConstraint}(undef, p)
    part_con_gt = Vector{JuMP.AbstractConstraint}(undef, p)
    #let [_, 1] be the upper bound and [_, 2] be the lower bound
    _, constant = _build_partitioned_expression(con.func, method.partition[p]) 
    v = Matrix{JuMP.variable_ref_type(M)}(undef, p, 2)
    for i in 1:p
        func, _= _build_partitioned_expression(con.func, method.partition[i])
        v[i,1] = create_variable(model, VariableProperties(func))
        v[i,2] = create_variable(model, VariableProperties(func))
        JuMP.set_name(v[i,1], "v_$(hash(con))_$(i)_1")
        JuMP.set_name(v[i,2], "v_$(hash(con))_$(i)_2")
        part_con_lt[i] = JuMP.build_constraint(error, 
        func - v[i,1], MOI.LessThan(zero(val_type))
        )
        part_con_gt[i] = JuMP.build_constraint(error, 
        -func - v[i,2], MOI.LessThan(zero(val_type))
        )
        _bound_auxiliary(model, v[i,1], func, method)
        _bound_auxiliary(model, v[i,2], -func, method)
    end
    set_values = _set_values(con.set)
    sum_con_lt = JuMP.@build_constraint(sum(v[i,1] for i in 1:p) + constant <= set_values[2])
    sum_con_gt = JuMP.@build_constraint(sum(v[i,2] for i in 1:p) - constant <= -set_values[1])
    return vcat(part_con_lt, part_con_gt), [sum_con_lt, sum_con_gt], vec(v)
end
function _build_partitioned_constraint(
    model::M,
    con::JuMP.VectorConstraint{T, S, R},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: _MOI.Nonpositives, R}
    p = length(method.partition)
    d = con.set.dimension
    v = Matrix{JuMP.variable_ref_type(M)}(undef, p, d)
    part_con = Vector{JuMP.AbstractConstraint}(undef, p)
    constants = Vector{Number}(undef, d)
    for i in 1:p
        part_expr = [_build_partitioned_expression(con.func[j],
        method.partition[i]) for j in 1:d
        ]
        func = JuMP.@expression(model, [j = 1:d], part_expr[j][1])
        constants .= [part_expr[j][2] for j in 1:d]
        for j in 1:d
            v[i,j] = create_variable(model, VariableProperties(func[j]))
            JuMP.set_name(v[i,j], "v_$(hash(con))_$(i)_$(j)")
            _bound_auxiliary(model, v[i,j], func[j], method)
        end
        part_con[i] = JuMP.build_constraint(error, 
        func - v[i,:], _MOI.Nonpositives(d)
        )
    end
    new_func = JuMP.@expression(model,[j = 1:d], 
    sum(v[i,j] for i in 1:p) + constants[j]
    )
    sum_con = JuMP.build_constraint(error, new_func, _MOI.Nonpositives(d))
    return part_con, [sum_con], vec(v)
end

function _build_partitioned_constraint(
    model::M,
    con::JuMP.VectorConstraint{T, S, R},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: _MOI.Nonnegatives, R}
    p = length(method.partition)
    d = con.set.dimension
    v = Matrix{JuMP.variable_ref_type(M)}(undef, p, d)
    part_con = Vector{JuMP.AbstractConstraint}(undef, p)
    constants = Vector{Number}(undef, d)
    for i in 1:p
        part_expr = [
            _build_partitioned_expression(con.func[j], method.partition[i]) 
            for j in 1:d
        ]
        func = JuMP.@expression(model, [j = 1:d], -part_expr[j][1])
        constants .= [-part_expr[j][2] for j in 1:d]
        for j in 1:d
            v[i,j] = create_variable(model, VariableProperties(func[j]))
            JuMP.set_name(v[i,j], "v_$(hash(con))_$(i)_$(j)")
            _bound_auxiliary(model, v[i,j], func[j], method)
        end
        part_con[i] = JuMP.build_constraint(error, func - v[i,:], _MOI.Nonpositives(d))
    end
    new_func = JuMP.@expression(model,[j = 1:d], 
    sum(v[i,j] for i in 1:p) + constants[j]
    )
    sum_con = JuMP.build_constraint(error,new_func,_MOI.Nonpositives(d))
    return part_con, [sum_con], vec(v)
end

function _build_partitioned_constraint(
    model::M,
    con::JuMP.VectorConstraint{T, S, R},
    method::PSplit
) where {M <: JuMP.AbstractModel, T, S <: _MOI.Zeros, R}
    p = length(method.partition)
    d = con.set.dimension
    part_con_np = Vector{JuMP.AbstractConstraint}(undef, p)  # nonpositive (≤ 0)
    part_con_nn = Vector{JuMP.AbstractConstraint}(undef, p)  # nonnegative (≥ 0)
    v = Array{JuMP.variable_ref_type(M)}(undef, p, d, 2)
    constants = Vector{Number}(undef, d)
    for i in 1:p
        part_expr = [
            _build_partitioned_expression(con.func[j], method.partition[i]) 
            for j in 1:d
        ]
        func = JuMP.@expression(model, [j = 1:d], part_expr[j][1])        
        constants .= [part_expr[j][2] for j in 1:d]
        for j in 1:d
            v[i,j,1] = create_variable(model, VariableProperties(func[j]))
            v[i,j,2] = create_variable(model, VariableProperties(func[j]))
            JuMP.set_name(v[i,j,1], "v_$(hash(con))_$(i)_$(j)_1")
            JuMP.set_name(v[i,j,2], "v_$(hash(con))_$(i)_$(j)_2")
            _bound_auxiliary(model, v[i,j,1], func[j], method)
            _bound_auxiliary(model, v[i,j,2], -func[j], method)
        end
        part_con_np[i] = JuMP.build_constraint(error, 
        func - v[i,:,1], _MOI.Nonpositives(d)
        )
        part_con_nn[i] = JuMP.build_constraint(error, 
        -func - v[i,:,2], _MOI.Nonpositives(d)
        )
    end
    new_func_np = JuMP.@expression(model,[j = 1:d], 
    sum(v[i,j,1] for i in 1:p) + constants[j]
    )
    new_func_nn = JuMP.@expression(model,[j = 1:d], 
    -sum(v[i,j,2] for i in 1:p) - constants[j]
    )
    sum_con_np = JuMP.build_constraint(error, 
    new_func_np, _MOI.Nonpositives(d)
    )
    sum_con_nn = JuMP.build_constraint(error, 
    new_func_nn, _MOI.Nonpositives(d)
    )
    return vcat(part_con_np, part_con_nn), vcat(sum_con_np, sum_con_nn), vec(v)
end

################################################################################
#                          FALLBACK WARNING DISPATCHES
################################################################################

# Generic fallback for _build_partitioned_expression
function _build_partitioned_expression(
    expr::F,
    ::Vector{<:JuMP.AbstractVariableRef}
) where F
    error("PSplit: _build_partitioned_expression not implemented 
    for expression type $F.")
end

# Generic fallback for _bound_auxiliary
function _bound_auxiliary(
    ::JuMP.AbstractModel,
    v::JuMP.AbstractVariableRef,
    func::F,
    ::PSplit
) where F
    error("PSplit: _bound_auxiliary not implemented for function 
    type $F.")
end