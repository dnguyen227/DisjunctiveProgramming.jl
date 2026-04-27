################################################################################
#                                MODEL COPYING
################################################################################
# Extension point for model copying (creates empty model).
function _copy_model(
    model::M
    ) where {M <: JuMP.AbstractModel}
    return M()
end

# OVERRIDABLE. Copy `source`'s decision variables into `target` (with
# bounds and integrality) and return a `source → target` ref map.
# Constraints and objective are NOT copied — the caller adds whichever
# subset it wants. InfiniteOpt overrides to additionally copy
# parameters, derivatives, and parameter functions.
function copy_variables_onto_model(
    target::JuMP.AbstractModel,
    source::JuMP.AbstractModel
    )
    V = JuMP.variable_ref_type(typeof(source))
    ref_map = Dict{V, V}()
    for variable in JuMP.all_variables(source)
        ref_map[variable] = variable_copy(target, variable)
    end
    return ref_map
end

"""
    copy_and_reformulate(model, decision_vars, reform_method, method)

Copy the GDP model, reformulate the copy with `reform_method`,
and wrap in a `GDPSubmodel`. The original model is not
modified. The copy's objective is rewritten in terms of the
copied variables.
"""
function copy_and_reformulate(
    model::JuMP.AbstractModel,
    decision_vars::Vector{<:JuMP.AbstractVariableRef},
    reform_method::AbstractReformulationMethod,
    method::AbstractReformulationMethod
    )
    copy, ref_map, _ = copy_gdp_model(model)
    reformulate_model(copy, reform_method)
    obj = JuMP.objective_function(model)
    sense = JuMP.objective_sense(model)
    V = JuMP.variable_ref_type(model)
    orig_to_copy = Dict{V, V}(
        v => ref_map[v] for v in decision_vars)
    JuMP.@objective(copy, sense,
        replace_variables_in_constraint(obj, orig_to_copy)
        )
    fwd_map = Dict{V, Vector{V}}(v => [ref_map[v]] for v in decision_vars)
    sub = GDPSubmodel(copy, decision_vars, fwd_map)
    JuMP.set_optimizer(sub.model, method.optimizer)
    JuMP.set_silent(sub.model)
    return sub
end

"""
    reformulate_and_relax(model, decision_vars, reform_method, method)

Reformulate the model in-place with `reform_method` and relax
integrality. Returns `(GDPSubmodel, undo_fn)` where `undo_fn`
restores integrality.
"""
function reformulate_and_relax(
    model::JuMP.AbstractModel,
    decision_vars::Vector{<:JuMP.AbstractVariableRef},
    reform_method::AbstractReformulationMethod,
    method::AbstractReformulationMethod
    )
    reformulate_model(model, reform_method)
    V = JuMP.variable_ref_type(model)
    fwd_map = Dict{V, Vector{V}}(v => [v] for v in decision_vars)
    sub = GDPSubmodel(model, decision_vars, fwd_map)
    JuMP.set_optimizer(sub.model, method.optimizer)
    JuMP.set_silent(sub.model)
    undo_relax = JuMP.relax_integrality(model)
    return sub, undo_relax
end

"""
    extract_solution(sub::GDPSubmodel)

Read the primal solution of `sub.model` after a solve, keyed by the
parent-model decision variables via `sub.fwd_map`. Shape follows
`fwd_map` values: `Vector`-valued fwd_maps (CP/MBM) yield per-support
`Vector`s; scalar fwd_maps (LOA feas) yield scalars.
"""
function extract_solution(sub::GDPSubmodel)
    return Dict(
        var => JuMP.value.(sub.fwd_map[var]) for var in sub.decision_vars)
end

################################################################################
#                          INDICATOR FIXING
################################################################################
"""
    fix_indicator(model, indicator::LogicalVariableRef, value::Bool)
    fix_indicator(binary_ref, value::Bool)

Fix a logical indicator's binary backing variable to `value` (true →
1.0, false → 0.0). The 3-arg form is the user-facing API: pass the
model and the `LogicalVariableRef`. The 2-arg form takes the binary
backing reference directly (or its complement-form expression
`1 - other_binary` when the indicator was declared as a complement).

Mirrors [`relax_logical_vars`](@ref) for selectively fixing rather
than relaxing.
"""
fix_indicator(model::JuMP.AbstractModel,
    indicator::LogicalVariableRef, value::Bool) =
    fix_indicator(_indicator_to_binary(model)[indicator], value)
fix_indicator(binary_ref::JuMP.AbstractVariableRef, value::Bool) =
    JuMP.fix(binary_ref, value ? 1.0 : 0.0; force = true)
function fix_indicator(
    binary_ref::JuMP.GenericAffExpr, value::Bool
    )
    underlying, _ = only(binary_ref.terms)
    JuMP.fix(underlying, value ? 0.0 : 1.0; force = true)
    return
end

"""
    unfix_indicator(model, indicator::LogicalVariableRef)
    unfix_indicator(binary_ref)

Undo [`fix_indicator`](@ref). No-op if not currently fixed.
"""
unfix_indicator(model::JuMP.AbstractModel,
    indicator::LogicalVariableRef) =
    unfix_indicator(_indicator_to_binary(model)[indicator])
function unfix_indicator(binary_ref)
    JuMP.is_fixed(binary_ref) && JuMP.unfix(binary_ref)
    return
end
function unfix_indicator(binary_ref::JuMP.GenericAffExpr)
    underlying = only(keys(binary_ref.terms))
    JuMP.is_fixed(underlying) && JuMP.unfix(underlying)
    return
end

################################################################################
#                          NO-GOOD CUTS
################################################################################
"""
    avoid_combination(model, combination)
    avoid_combination(model, combination, binary_map)

Add a no-good cut to `model` excluding `combination` from any future
solution. The added constraint
    Σ_{j active} (1 - y_j) + Σ_{j inactive} y_j ≥ 1
forces at least one indicator to differ from `combination`.

`combination` maps `LogicalVariableRef` → `Bool` (whether each
indicator was active). The 3-arg form lets you supply an explicit
`binary_map` (defaulting to `_indicator_to_binary(model)`) when the
binaries used in the cut belong to a copy of the original model
(e.g. an LOA master).

Returns the constraint reference of the added cut.
"""
avoid_combination(model::JuMP.AbstractModel, combination) =
    avoid_combination(
        model, combination, _indicator_to_binary(model))
function avoid_combination(model::JuMP.AbstractModel,
    combination, binary_map
    )
    V = JuMP.variable_ref_type(typeof(model))
    T = JuMP.value_type(typeof(model))
    cut = JuMP.GenericAffExpr{T, V}(zero(T))
    for (indicator, active) in combination
        haskey(binary_map, indicator) || continue
        add_no_good_terms(cut, binary_map[indicator], active)
    end
    return JuMP.@constraint(model, cut >= one(T))
end

"""
    add_no_good_terms(cut, binary_ref, active::Bool)

Append one indicator's contribution to a no-good cut expression:
adds `1 - y_j` if the indicator was active in the excluded
combination, or `y_j` otherwise. Used by [`avoid_combination`](@ref).
"""
function add_no_good_terms(cut, binary_ref, active::Bool)
    if active
        JuMP.add_to_expression!(cut, -1.0, binary_ref)
        JuMP.add_to_expression!(cut, 1.0)
    else
        JuMP.add_to_expression!(cut, 1.0, binary_ref)
    end
    return
end

################################################################################
#                          LOGICAL VARIABLE RELAXATION
################################################################################
"""
    relax_logical_vars(model::JuMP.AbstractModel)

Relax the binary variables associated with logical indicators
to continuous variables in `[0, 1]`. Returns a vector of the
relaxed variable references, which can be passed to
[`unrelax_logical_vars`](@ref) to restore integrality.
"""
function relax_logical_vars(model::JuMP.AbstractModel)
    V = JuMP.variable_ref_type(model)
    binary_refs = V[]
    for (_, bvar) in _indicator_to_binary(model)
        bvar isa V || continue
        push!(binary_refs, bvar)
        JuMP.unset_binary(bvar)
        JuMP.set_lower_bound(bvar, 0.0)
        JuMP.set_upper_bound(bvar, 1.0)
    end
    return binary_refs
end

"""
    unrelax_logical_vars(
        binary_refs::Vector{<:JuMP.AbstractVariableRef}
        )

Restore the binary constraint on variables previously relaxed
by [`relax_logical_vars`](@ref).
"""
function unrelax_logical_vars(
    binary_refs::Vector{<:JuMP.AbstractVariableRef}
    )
    for v in binary_refs
        JuMP.set_binary(v)
    end
end

################################################################################
#                              ALL VARIABLES
################################################################################
"""
    collect_all_vars(model::JuMP.AbstractModel)

Returns all variable references in the model.
Extend this for model types that have additional ref types (e.g., derivatives).
"""
collect_all_vars(model::JuMP.AbstractModel) = JuMP.all_variables(model)

################################################################################
#                              GET CONSTANT
################################################################################
"""
    get_constant(expr)

Returns the constant portion of an expression. Extendable for model types where
additional terms should be treated as constants.
"""
get_constant(expr::JuMP.GenericAffExpr) = JuMP.constant(expr)
get_constant(expr::JuMP.GenericQuadExpr) = JuMP.constant(expr)
get_constant(expr::Number) = expr
function get_constant(expr::JuMP.AbstractVariableRef)
    return zero(JuMP.value_type(typeof(JuMP.owner_model(expr))))
end

################################################################################
#                              MODEL COPYING
################################################################################
"""
    JuMP.copy_extension_data(
        data::GDPData,
        new_model::JuMP.AbstractModel,
        old_model::JuMP.AbstractModel
    )::GDPData

Extend `JuMP.copy_extension_data` to initialize an empty [`GDPData`](@ref) object 
for the copied model. This is the first step in the model copying process and is 
automatically called by `JuMP.copy_model`. The actual GDP data (logical variables, 
disjunctions, etc.) is copied separately via [`copy_gdp_data`](@ref).
"""
function JuMP.copy_extension_data(
    data::GDPData{M, V, C, T},
    new_model::JuMP.AbstractModel,
    old_model::JuMP.AbstractModel
) where {M, V, C, T}
    return GDPData{M, V, C}()
end

"""
    copy_gdp_data(
        model::JuMP.AbstractModel,
        new_model::JuMP.AbstractModel,
        ref_map::JuMP.GenericReferenceMap
    )::Dict{LogicalVariableRef, LogicalVariableRef}

Copy all GDP-specific data from `model` to `new_model`, including logical variables, 
logical constraints, disjunct constraints, and disjunctions. This function is called 
automatically by [`copy_gdp_model`](@ref) after `JuMP.copy_model` has copied the base 
model structure.

**Arguments**
- `model::JuMP.AbstractModel`: The source model containing GDP data to copy.
- `new_model::JuMP.AbstractModel`: The destination model that will receive the copied GDP data.
- `ref_map::JuMP.GenericReferenceMap`: The reference map from `JuMP.copy_model` that maps 
  old variable references to new ones.

**Returns**
- `Dict{LogicalVariableRef, LogicalVariableRef}`: A mapping from old logical variable 
  references to new logical variable references.
"""
function copy_gdp_data(
    model::M,
    new_model::M,
    ref_map::GenericReferenceMap
    ) where {M <: JuMP.AbstractModel}
    
    old_gdp = model.ext[:GDP]

    # GDPData contains the following fields.
    # DICTIONARIES (for loops below)
    # - logical_variables
    # - logical_constraints
    # - disjunct_constraints
    # - disjunctions
    # - exactly1_constraints
    # - indicator_to_binary
    # - indicator_to_constraints
    # - constraint_to_indicator
    # - variable_bounds
    # SINGLE VALUES (copy directly)
    # - solution_method
    # - ready_to_optimize

    new_gdp = new_model.ext[:GDP]

    # Creating maps from old to new model.
    var_map = Dict(v => ref_map[v] for v in collect_all_vars(model))
    lv_map = Dict{LogicalVariableRef{M}, LogicalVariableRef{M}}()
    lc_map = Dict{LogicalConstraintRef{M}, LogicalConstraintRef{M}}()
    disj_map = Dict{DisjunctionRef{M}, DisjunctionRef{M}}()
    disj_con_map = Dict{DisjunctConstraintRef{M}, DisjunctConstraintRef{M}}()

    # Copying logical variables
    for (idx, var) in old_gdp.logical_variables
        old_var_ref = LogicalVariableRef(model, idx)
        new_var_data = LogicalVariableData(var.variable, var.name)
        new_var = LogicalVariableRef(new_model, idx)
        lv_map[old_var_ref] = new_var
        # Update to new_gdp.logical_variables
        new_gdp.logical_variables[idx] = new_var_data
    end

    # Copying logical constraints
    for (idx, lc_data) in old_gdp.logical_constraints
        old_con_ref = LogicalConstraintRef(model, idx)
        new_con_ref = LogicalConstraintRef(new_model, idx)
        c = lc_data.constraint
        expr = replace_variables_in_constraint(c.func, lv_map)
        new_con = JuMP.build_constraint(error, expr, c.set)
        JuMP.add_constraint(new_model, new_con, lc_data.name)
        lc_map[old_con_ref] = new_con_ref
    end

    # Copying disjunct constraints
    for (idx, disj_con_data) in old_gdp.disjunct_constraints
        old_constraint = disj_con_data.constraint
        old_dc_ref = DisjunctConstraintRef(model, idx)
        old_indicator = old_gdp.constraint_to_indicator[old_dc_ref]
        new_indicator = lv_map[old_indicator]
        new_expr = replace_variables_in_constraint(old_constraint.func, 
        var_map
        )
        # Update to new_gdp.disjunct_constraints
        new_con = JuMP.build_constraint(error, new_expr, 
        old_constraint.set, Disjunct(new_indicator)
        )
        new_dc_ref = JuMP.add_constraint(new_model, new_con, disj_con_data.name)
        disj_con_map[old_dc_ref] = new_dc_ref
    end

    # Copying disjunctions
    for (idx, disj_data) in old_gdp.disjunctions
        old_disj = disj_data.constraint
        new_indicators = [replace_variables_in_constraint(indicator, lv_map) 
        for indicator in old_disj.indicators
            ]
        new_disj = Disjunction(new_indicators, old_disj.nested)
        disj_map[DisjunctionRef(model, idx)] = DisjunctionRef(new_model, idx)
        # Update to new_gdp.disjunctions
        new_gdp.disjunctions[idx] = ConstraintData(new_disj, disj_data.name)
    end

    # Copying exactly1 constraints
    for (d_ref, lc_ref) in old_gdp.exactly1_constraints
        new_lc_ref = lc_map[lc_ref]
        new_d_ref = disj_map[d_ref]
        # Update to new_gdp.exactly1_constraints
        new_gdp.exactly1_constraints[new_d_ref] = new_lc_ref
    end

    # Copying indicator to binary
    for (lv_ref, bref) in old_gdp.indicator_to_binary
        new_bref = _remap_indicator_to_binary(bref, var_map)
        # Update to new_gdp.indicator_to_binary
        new_gdp.indicator_to_binary[lv_map[lv_ref]] = new_bref
    end

    # Copying indicator to constraints
    for (lv_ref, con_refs) in old_gdp.indicator_to_constraints
        new_lvar_ref = lv_map[lv_ref]
        new_con_refs = Vector{
            Union{DisjunctConstraintRef{M}, DisjunctionRef{M}}
        }()
        for con_ref in con_refs
            new_con_ref = _remap_indicator_to_constraint(con_ref, 
            disj_con_map, disj_map
            )
            push!(new_con_refs, new_con_ref)
        end
        # Update to new_gdp.indicator_to_constraints
        new_gdp.indicator_to_constraints[new_lvar_ref] = new_con_refs
    end

    # Copying constraint to indicator
    for (con_ref, lv_ref) in old_gdp.constraint_to_indicator
        # Update to new_gdp.constraint_to_indicator
        new_gdp.constraint_to_indicator[
            _remap_constraint_to_indicator(con_ref, disj_con_map, disj_map)
            ] = lv_map[lv_ref]
    end

    # Copying variable bounds
    for (v, bounds) in old_gdp.variable_bounds
        # Update to new_gdp.variable_bounds
        new_gdp.variable_bounds[var_map[v]] = bounds
    end

    # Copying solution method and ready to optimize
    new_gdp.solution_method = old_gdp.solution_method
    new_gdp.ready_to_optimize = old_gdp.ready_to_optimize

    return lv_map
end

"""
    copy_gdp_model(model::JuMP.AbstractModel)

Create a copy of a [`GDPModel`](@ref), including all variables, constraints, and 
GDP-specific data (logical variables, disjunctions, etc.).

**Arguments**
- `model::JuMP.AbstractModel`: The GDP model to copy.

**Returns**
A tuple `(new_model, ref_map, lv_map)` where:
- `new_model`: The copied model.
- `ref_map::JuMP.GenericReferenceMap`: Maps old variable and constraint references to new ones.
- `lv_map::Dict{LogicalVariableRef, LogicalVariableRef}`: Maps old logical variable 
  references to new ones.

## Example
```julia
using DisjunctiveProgramming, HiGHS
model = GDPModel(HiGHS.Optimizer)
@variable(model, x)
@variable(model, Y[1:2], LogicalVariable)
@constraint(model, x <= 10, Disjunct(Y[1]))
@constraint(model, x >= 20, Disjunct(Y[2]))
@disjunction(model, Y)

new_model, ref_map, lv_map = copy_gdp_model(model)
```
"""
function copy_gdp_model(model::M) where {M <: JuMP.AbstractModel}
    new_model, ref_map = JuMP.copy_model(model)
    lv_map = copy_gdp_data(model, new_model, ref_map)
    return new_model, ref_map, lv_map
end
################################################################################
#                                GDP REMAPPING
################################################################################
# These remapping functions use multiple dispatch to handle different types that
# can appear in GDP data structures during model copying.
#
# Indicators can be represented by a variable or an affine expression to 
# indicate a complementary relationship with another variable.
# This translates to a binary or affine expression in its binary reformulation.
#
# Depending on the above, different mappings are required for indicator_to_binary,
# indicator_to_constraints, and constraint_to_indicator.
################################################################################

function _remap_indicator_to_constraint(
    con_ref::DisjunctConstraintRef,
    disj_con_map::Dict{DisjunctConstraintRef{M}, DisjunctConstraintRef{M}},
    ::Dict{DisjunctionRef{M}, DisjunctionRef{M}}
) where {M <: JuMP.AbstractModel}
    return disj_con_map[con_ref]   
end

function _remap_indicator_to_constraint(
    con_ref::DisjunctionRef,
    ::Dict{DisjunctConstraintRef{M}, DisjunctConstraintRef{M}},
    disj_map::Dict{DisjunctionRef{M}, DisjunctionRef{M}}
) where {M <: JuMP.AbstractModel}
    return disj_map[con_ref]   
end

function _remap_indicator_to_binary(
    bref::JuMP.AbstractVariableRef,
    var_map::Dict{V, V}
) where {V <: JuMP.AbstractVariableRef}
    return var_map[bref]
end

function _remap_indicator_to_binary(
    bref::JuMP.GenericAffExpr,
    var_map::Dict{V, V}
) where {V <: JuMP.AbstractVariableRef}
    return replace_variables_in_constraint(bref, var_map)
end

function _remap_constraint_to_indicator(
    con_ref::DisjunctConstraintRef,
    disj_con_map::Dict{DisjunctConstraintRef{M}, DisjunctConstraintRef{M}},
    ::Dict{DisjunctionRef{M}, DisjunctionRef{M}}
) where {M <: JuMP.AbstractModel}
    return disj_con_map[con_ref]   
end

function _remap_constraint_to_indicator(
    con_ref::DisjunctionRef,
    ::Dict{DisjunctConstraintRef{M}, DisjunctConstraintRef{M}},
    disj_map::Dict{DisjunctionRef{M}, DisjunctionRef{M}}
) where {M <: JuMP.AbstractModel}
    return disj_map[con_ref]   
end
################################################################################
#                    LINEARIZATION & EXPRESSION CONVERSION
################################################################################
# First-order Taylor approximation and MOI expression building
# for outer approximation methods (LOA, future OA variants).
################################################################################

################################################################################
#                    AGGREGATE-REF DETECTION
################################################################################
# Predicate: does the variable ref aggregate multiple decision
# variables behind a single leaf? An "aggregate" ref is one that AD
# cannot see inside — e.g. an InfiniteOpt `MeasureRef` (`∫ f(x,t) dt`
# is one ref but depends on `x(t_1), …, x(t_K)`) or a
# `ParameterFunctionRef`. Base returns false; the InfiniteOpt
# extension overrides for aggregate ref types.
#
# When `has_aggregate_ref(expr)` is true, MOI Nonlinear AD on `expr`
# would treat the aggregate as a single opaque variable and produce a
# meaningless gradient. The LOA pipeline falls back to transcription
# in that case (flat scalar expression, AD on the flat form, then map
# back to master refs).
#
# #suggestions for names are welcome
is_aggregate_ref(::JuMP.AbstractVariableRef) = false

function has_aggregate_ref(expr)
    found = Ref(false)
    _interrogate_variables(expr) do v
        found[] || (found[] = is_aggregate_ref(v))
    end
    return found[]
end

################################################################################
#                    MOI NONLINEAR EXPRESSION CONVERSION
################################################################################
# Convert JuMP expression trees to Julia Expr with
# MOI.VariableIndex leaves for MOI.Nonlinear evaluation.
function _to_nlp_expr(expr::JuMP.GenericNonlinearExpr, idx::Dict)
    args = Any[_to_nlp_expr(a, idx) for a in expr.args]
    return Expr(:call, expr.head, args...)
end
function _to_nlp_expr(expr::JuMP.GenericAffExpr, idx::Dict)
    parts = Any[expr.constant]
    for (var, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef, _MOI.VariableIndex(idx[var])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end
function _to_nlp_expr(expr::JuMP.GenericQuadExpr, idx::Dict)
    parts = Any[_to_nlp_expr(expr.aff, idx)]
    for (pair, coef) in expr.terms
        push!(parts, Expr(:call, :*, coef,
            _MOI.VariableIndex(idx[pair.a]),
            _MOI.VariableIndex(idx[pair.b])))
    end
    length(parts) == 1 && return parts[1]
    return Expr(:call, :+, parts...)
end
function _to_nlp_expr(var::JuMP.AbstractVariableRef, idx::Dict)
    return _MOI.VariableIndex(idx[var])
end
_to_nlp_expr(x::Number, ::Dict) = x

# First-order Taylor linearization of a quadratic or nonlinear
# expression at point xk via MOI.Nonlinear reverse-mode AD.
function _linearize_at(
    func::Union{JuMP.GenericQuadExpr, JuMP.GenericNonlinearExpr},
    xk::Dict, ref_map
    )
    vars = JuMP.AbstractVariableRef[]
    _interrogate_variables(v -> push!(vars, v), func)
    unique!(vars)
    isempty(vars) && return JuMP.AffExpr(JuMP.value(v -> 0.0, func))

    n = length(vars)
    T = JuMP.value_type(typeof(JuMP.owner_model(vars[1])))
    idx = Dict(vars[i] => i for i in 1:n)
    nlp = _MOI.Nonlinear.Model()
    _MOI.Nonlinear.set_objective(nlp, _to_nlp_expr(func, idx))
    ord = [_MOI.VariableIndex(i) for i in 1:n]
    evaluator = _MOI.Nonlinear.Evaluator(
        nlp, _MOI.Nonlinear.SparseReverseMode(), ord)
    _MOI.initialize(evaluator, [:Grad])

    xk_vec = [_unwrap_scalar(get(xk, v, zero(T))) for v in vars]
    f_xk = _MOI.eval_objective(evaluator, xk_vec)
    grad = zeros(T, n)
    _MOI.eval_objective_gradient(evaluator, grad, xk_vec)

    constant = T(f_xk)
    for i in 1:n
        constant -= grad[i] * xk_vec[i]
    end
    V = typeof(ref_map[vars[1]])
    result = JuMP.GenericAffExpr{T, V}(constant)
    for i in 1:n
        iszero(grad[i]) && continue
        JuMP.add_to_expression!(result, grad[i], ref_map[vars[i]])
    end
    return result
end

# Extract RHS from an MOI set.
_set_rhs(s::Union{_MOI.LessThan, _MOI.GreaterThan, _MOI.EqualTo}) =
    _MOI.constant(s)
_set_rhs(::Any) = 0.0

# Unwrap a 1-element per-support `Vector` to its scalar value;
# scalars pass through. `extract_solution` returns per-support
# `Vector`s uniformly (length-1 for finite, length-K for InfiniteOpt).
# AD pipelines and `set_start_value` need a scalar in the finite
# case; per-support consumers slice out a scalar themselves.
_unwrap_scalar(v::Real) = v
_unwrap_scalar(v::AbstractVector) = only(v)
