using DisjunctiveProgramming
using Documenter
makedocs(
    sitename = "DisjunctiveProgramming.jl",
    modules  = [DisjunctiveProgramming],
    pages=[
        "Home" => "index.md",
        "User Guide" => [
            "GDP Models" => "guide/model.md",
            "Logical Variables" => "guide/variables.md", 
            "Constraints" => "guide/constraints.md",
            "Solution Methods" => "guide/methods.md",
            "Logical Operations" => "guide/logic.md"
        ],
        "API" => "api.md"
    ],
    checkdocs = :none
)
deploydocs(;
    repo="github.com/hdavid16/DisjunctiveProgramming.jl",
)
