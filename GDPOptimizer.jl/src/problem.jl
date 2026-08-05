################################################################################
#                          PROBLEM PARTITION
################################################################################
# activation is `z` or `1 - z`; rows stay in cache space
struct _Disjunct
    activation::MOI.ScalarAffineFunction{Float64}
    binary::MOI.VariableIndex
    active_value::Bool
    functions::Vector{MOI.AbstractScalarFunction}
    sets::Vector{MOI.AbstractScalarSet}
end

# activation: 1 at top level, the parent indicator when nested
struct _Disjunction
    activation::MOI.ScalarAffineFunction{Float64}
    disjuncts::Vector{_Disjunct}
end

# Constraint indices stay in cache space; the builders remap them
struct _Problem
    variables::Vector{MOI.VariableIndex}
    binaries::Vector{MOI.VariableIndex}
    disjunctions::Vector{_Disjunction}
    variable_cis::Vector{MOI.ConstraintIndex}
    linear_cis::Vector{MOI.ConstraintIndex}
    # stable function objects so the linearizer can cache evaluators
    nonlinear_rows::Vector{Tuple{MOI.AbstractScalarFunction,
        MOI.AbstractScalarSet}}
    sense::MOI.OptimizationSense
    objective::MOI.AbstractScalarFunction
end

_is_linear(::Union{MOI.VariableIndex, MOI.ScalarAffineFunction}) = true
_is_linear(::MOI.AbstractScalarFunction) = false

_to_affine(func::MOI.ScalarAffineFunction{Float64}) = func
function _to_affine(func::MOI.AbstractScalarFunction)
    return convert(MOI.ScalarAffineFunction{Float64}, func)
end

# a raw `VariableIndex` row becomes a bound and collides with the
# variable's own bounds in the subproblem
_as_row(func::MOI.VariableIndex) = _to_affine(func)
_as_row(func::MOI.AbstractScalarFunction) = func

# demote rows the enclosing vector function promoted
_demote(func::MOI.AbstractScalarFunction) = func
function _demote(func::MOI.ScalarQuadraticFunction{Float64})
    return _try_convert(MOI.ScalarAffineFunction{Float64}, func)
end
function _demote(func::MOI.ScalarNonlinearFunction)
    return _try_convert(MOI.ScalarAffineFunction{Float64},
        _try_convert(MOI.ScalarQuadraticFunction{Float64}, func))
end

function _try_convert(T::Type, func)
    return try
        convert(T, func)
    catch
        func
    end
end

_scalarize(func::MOI.AbstractVectorFunction) =
    collect(MOI.Utilities.eachscalar(func))

# `z` -> (z, true), `1 - z` -> (z, false)
function _activation_binary(activation::MOI.ScalarAffineFunction{Float64})
    canonical = MOI.Utilities.canonical(activation)
    if length(canonical.terms) == 1
        term = only(canonical.terms)
        if term.coefficient == 1.0 && canonical.constant == 0.0
            return term.variable, true
        elseif term.coefficient == -1.0 && canonical.constant == 1.0
            return term.variable, false
        end
    end
    return error("Unsupported indicator expression `$activation`: each " *
        "`DisjunctionSet` indicator must be a binary variable `z` or " *
        "its complement `1 - z`.")
end

function _parse_disjunction(
    cache::_Cache,
    ci::MOI.ConstraintIndex{F, DisjunctionSet}
    ) where {F}
    set = MOI.get(cache, MOI.ConstraintSet(), ci)
    rows = _scalarize(MOI.get(cache, MOI.ConstraintFunction(), ci))
    disjunction_activation = _to_affine(
        _demote(rows[activation_index(set)]))
    disjuncts = _Disjunct[]
    for (i, j) in enumerate(indicator_indices(set))
        activation = _to_affine(_demote(rows[j]))
        binary, active_value = _activation_binary(activation)
        zero_one = MOI.ConstraintIndex{MOI.VariableIndex, MOI.ZeroOne}(
            binary.value)
        MOI.is_valid(cache, zero_one) || error("The indicator variable " *
            "of a `DisjunctionSet` disjunct must be `MOI.ZeroOne`.")
        functions = MOI.AbstractScalarFunction[
            _as_row(_demote(rows[k])) for k in row_indices(set, i)]
        push!(disjuncts, _Disjunct(activation, binary, active_value,
            functions, set.inner_sets[i]))
    end
    return _Disjunction(disjunction_activation, disjuncts)
end

# partition the cache for the LOA loop
function _build_problem(model::Optimizer)
    cache = model.cache
    sense = MOI.get(cache, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        # feasibility model: minimize a constant zero objective
        sense = MOI.MIN_SENSE
        objective = MOI.ScalarAffineFunction(
            MOI.ScalarAffineTerm{Float64}[], 0.0)
    else
        F = MOI.get(cache, MOI.ObjectiveFunctionType())
        objective = _demote(MOI.get(cache, MOI.ObjectiveFunction{F}()))
    end
    variables = MOI.get(cache, MOI.ListOfVariableIndices())
    disjunctions = _Disjunction[]
    variable_cis = MOI.ConstraintIndex[]
    linear_cis = MOI.ConstraintIndex[]
    nonlinear_rows = Tuple{MOI.AbstractScalarFunction,
        MOI.AbstractScalarSet}[]
    for (FC, S) in MOI.get(cache, MOI.ListOfConstraintTypesPresent())
        cis = MOI.get(cache, MOI.ListOfConstraintIndices{FC, S}())
        if S === DisjunctionSet
            append!(disjunctions,
                _parse_disjunction(cache, ci) for ci in cis)
        elseif FC === MOI.VariableIndex && S <: MOI.AbstractScalarSet
            append!(variable_cis, cis)
        elseif FC === MOI.ScalarAffineFunction{Float64} &&
                S <: MOI.AbstractScalarSet
            append!(linear_cis, cis)
        elseif FC <: Union{MOI.ScalarQuadraticFunction{Float64},
                MOI.ScalarNonlinearFunction} && S <: MOI.AbstractScalarSet
            append!(nonlinear_rows,
                (_demote(MOI.get(cache, MOI.ConstraintFunction(), ci)),
                    MOI.get(cache, MOI.ConstraintSet(), ci))
                for ci in cis)
        else
            error("GDPOptimizer does not support `$FC`-in-`$S` constraints.")
        end
    end
    binaries = unique!([disjunct.binary for disjunction in disjunctions
        for disjunct in disjunction.disjuncts])
    # non-indicator discrete variables keep their integrality; the
    # nlp_solver must handle whatever remains
    return _Problem(variables, binaries, disjunctions, variable_cis,
        linear_cis, nonlinear_rows, sense, objective)
end

function _disjunct_active(combination::AbstractDict, disjunct::_Disjunct)
    return combination[disjunct.binary] == disjunct.active_value
end

_is_nonlinear_disjunct(disjunct::_Disjunct) =
    any(!_is_linear(func) for func in disjunct.functions)
