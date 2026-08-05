using Aqua
using DisjunctiveProgramming

# GDPOptimizer is a strong dep only because it is unregistered (its
# extension loads through it); DP.jl src never imports it directly.
Aqua.test_all(DisjunctiveProgramming, deps_compat = false,
    ambiguities = false, stale_deps = (ignore = [:GDPOptimizer],))
Aqua.test_deps_compat(DisjunctiveProgramming, check_extras = (ignore=[:Test, :HiGHS],))
Aqua.test_ambiguities(DisjunctiveProgramming)