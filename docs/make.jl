using DisjunctiveProgramming
using Documenter

makedocs(
    sitename = "DisjunctiveProgramming.jl",
    modules  = [DisjunctiveProgramming],
    pages = [
        "Home" => "index.md",
        "User Guide" => [
            "GDP Models" => "guide/model.md",
            "Logical Variables" => "guide/variables.md",
            "Logical Constraints" => "guide/logic.md",
            "Disjunctions" => "guide/constraints.md",
            "Solution Methods" => "guide/methods.md"
        ],
        "API" => "api.md"
    ],
    checkdocs = :none,
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1
    )
)
deploydocs(;
    repo="github.com/infiniteopt/DisjunctiveProgramming.jl",
)
