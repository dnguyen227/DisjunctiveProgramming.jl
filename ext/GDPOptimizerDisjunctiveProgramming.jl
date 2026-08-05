module GDPOptimizerDisjunctiveProgramming

import DisjunctiveProgramming as DP
import GDPOptimizer
import JuMP
import JuMP.MOI as _MOI

################################################################################
#                        DISJUNCTION SET LOWERING
################################################################################
# One flat constraint per disjunction: [activation, indicators,
# rows]. Nested disjunctions get their own constraint with the parent
# indicator as the activation. InfiniteOpt transcribes per support.
function DP.reformulate_disjunction(
    model::JuMP.AbstractModel,
    disj::DP.Disjunction,
    method::DP.MOIDisjunction
    )
    constraints = JuMP.AbstractConstraint[]
    _lower_disjunction(constraints, model, disj, 1.0)
    return constraints
end

function _lower_disjunction(
    constraints::Vector{JuMP.AbstractConstraint},
    model::JuMP.AbstractModel,
    disj::DP.Disjunction,
    activation
    )
    indicators = JuMP.AbstractJuMPScalar[]
    rows = JuMP.AbstractJuMPScalar[]
    inner_sets = Vector{_MOI.AbstractScalarSet}[]
    for lvref in disj.indicators
        indicator = 1.0 * DP.binary_variable(lvref)
        push!(indicators, indicator)
        sets = _MOI.AbstractScalarSet[]
        for cref in get(DP._indicator_to_constraints(model), lvref, [])
            constraint = JuMP.constraint_object(cref)
            if constraint isa DP.Disjunction
                haskey(DP._exactly1_constraints(model), cref) || error(
                    "`MOIDisjunction` requires nested " *
                    "disjunctions created with `exactly1 = true`.")
                _lower_disjunction(constraints, model, constraint,
                    indicator)
            else
                push!(rows, constraint.func)
                push!(sets, constraint.set)
            end
        end
        push!(inner_sets, sets)
    end
    push!(constraints, JuMP.VectorConstraint(
        collect(promote(1.0 * activation, indicators..., rows...)),
        GDPOptimizer.DisjunctionSet(inner_sets)))
    return
end

# transcription may collapse the constant activation to a number;
# promote back to a uniform expression vector
function JuMP.build_constraint(
    ::Function,
    func::AbstractVector{<:JuMP.AbstractJuMPScalar},
    set::GDPOptimizer.DisjunctionSet
    )
    return JuMP.VectorConstraint(func, set)
end

function JuMP.build_constraint(
    ::Function,
    func::AbstractVector,
    set::GDPOptimizer.DisjunctionSet
    )
    return JuMP.VectorConstraint(collect(promote(func...)), set)
end

end
