"""
    InfiniteGDPModel(args...; kwargs...)

Creates an `InfiniteOpt.InfiniteModel` that is compatible with the 
capabiltiies provided by DisjunctiveProgramming.jl. This requires 
that InfiniteOpt be imported first.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt

julia> InfiniteGDPModel()

```
"""
function InfiniteGDPModel end

"""
    MOIDisjunction()

Reformulation method that lowers each disjunction to a single vector
constraint in `GDPOptimizer.DisjunctionSet` so the model can be solved
by the GDPOptimizer.jl MOI solver layer (logic-based outer
approximation). Requires GDPOptimizer to be imported first and the
model's optimizer to be a `GDPOptimizer.Optimizer`.

**Example**
```julia
julia> using DisjunctiveProgramming, GDPOptimizer, HiGHS, Ipopt

julia> model = GDPModel(() -> GDPOptimizer.Optimizer(
           nlp_solver = Ipopt.Optimizer, mip_solver = HiGHS.Optimizer));

julia> optimize!(model, gdp_method = MOIDisjunction())
```
"""
struct MOIDisjunction <: AbstractReformulationMethod end

requires_exactly1(::MOIDisjunction) = true

"""
    InfiniteLogical(prefs...)

Allows users to create infinite logical variables. This is a tag 
for the `@variable` macro that is a combination of `InfiniteOpt.Infinite` 
and `DisjunctiveProgramming.Logical`. This requires that InfiniteOpt be 
first imported.

**Example**
```julia
julia> using DisjunctiveProgramming, InfiniteOpt

julia> model = InfiniteGDPModel();

julia> @infinite_parameter(model, t in [0, 1]);

julia> @infinite_parameter(model, x[1:2] in [-1, 1]);

julia> @variable(model, Y, InfiniteLogical(t, x)) # creates Y(t, x) in {True, False}
Y(t, x)
```
"""
function InfiniteLogical end
