# MOI contract conformance. MOI.Test cannot build a `DisjunctionSet`, so it
# only exercises the API-contract surface (model API + optimizer attributes),
# not the LOA solving logic -- that is covered by the other test files. The
# solving families and the dual/basis attributes are out of scope by design.
import HiGHS
import Ipopt

@testset "MOI contract (model API + attributes)" begin
    optimizer = MOI.instantiate(
        () -> GDPO.Optimizer(
            nlp_solver = Ipopt.Optimizer,
            mip_solver = HiGHS.Optimizer,
            time_limit = 20.0,
        );
        with_cache_type = Float64,
        with_bridge_type = Float64,
    )
    config = MOI.Test.Config(
        atol = 1e-3,
        rtol = 1e-3,
        optimal_status = MOI.LOCALLY_SOLVED,   # never reports OPTIMAL
        exclude = Any[
            MOI.ConstraintDual,
            MOI.DualObjectiveValue,
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
        ],
    )
    MOI.Test.runtests(
        optimizer,
        config;
        include = Regex[r"test_model_", r"test_attribute_"],
        warn_unsupported = false,
    )
end
