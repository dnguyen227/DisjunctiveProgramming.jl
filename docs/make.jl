using DisjunctiveProgramming
using Documenter

makedocs(
    sitename = "DisjunctiveProgramming.jl",
    modules  = [DisjunctiveProgramming],
    pages = [
        "Home" => "index.md",
        "Quick Start" => "tutorials/quick_start.md",
        "User Guide" => [
            "GDP Models" => "guide/model.md",
            "Logical Variables" => "guide/variables.md",
            "Logical Constraints" => "guide/logic.md",
            "Disjunctions" => "guide/constraints.md",
            "Solution Methods" => "guide/methods.md"
        ],
        "API Manual" => [
            "GDP Models" => "manual/model.md",
            "Logical Variables" => "manual/variables.md",
            "Logical Constraints" => "manual/logic.md",
            "Disjunctions" => "manual/constraints.md",
            "Solution Methods" => "manual/methods.md"
        ],
        "Development" => [
            "Getting Started" => "develop/start_guide.md",
            "Extensions" => "develop/extensions.md",
            "Style Guide" => "develop/style.md"
        ]
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
