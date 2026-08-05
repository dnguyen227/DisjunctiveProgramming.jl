function test_set_construction()
    set = GDPO.DisjunctionSet([
        [MOI.LessThan(1.0), MOI.GreaterThan(0.0)],
        [MOI.EqualTo(2.0)],
    ])
    @test GDPO.num_disjuncts(set) == 2
    @test MOI.dimension(set) == 1 + 2 + 3
    @test GDPO.activation_index(set) == 1
    @test GDPO.indicator_indices(set) == 2:3
    @test GDPO.row_indices(set, 1) == 4:5
    @test GDPO.row_indices(set, 2) == 6:6
    copied = copy(set)
    @test copied == set
    @test copied.inner_sets !== set.inner_sets
end

function test_set_interval_inner()
    set = GDPO.DisjunctionSet([[MOI.Interval(0.0, 1.0)], [MOI.LessThan(0.0)]])
    @test MOI.dimension(set) == 5
end

function test_set_empty_disjunct()
    set = GDPO.DisjunctionSet([
        MOI.AbstractScalarSet[],
        [MOI.LessThan(0.0)],
    ])
    @test MOI.dimension(set) == 4
    @test GDPO.row_indices(set, 1) == 4:3
    @test isempty(GDPO.row_indices(set, 1))
    @test GDPO.row_indices(set, 2) == 4:4
end

function test_set_validation()
    @test_throws ArgumentError GDPO.DisjunctionSet(
        Vector{MOI.AbstractScalarSet}[])
    @test_throws ArgumentError GDPO.DisjunctionSet([[MOI.ZeroOne()]])
    @test_throws ArgumentError GDPO.DisjunctionSet([
        [MOI.LessThan(0.0)],
        [MOI.Nonnegatives(2)],
    ])
end

@testset "DisjunctionSet" begin
    test_set_construction()
    test_set_interval_inner()
    test_set_empty_disjunct()
    test_set_validation()
end
