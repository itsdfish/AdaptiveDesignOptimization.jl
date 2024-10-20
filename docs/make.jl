using Documenter
using AdaptiveDesignOptimization

makedocs(
    warnonly = true,
    sitename = "AdaptiveDesignOptimization",
    format = Documenter.HTML(
        assets = [
            asset(
            "https://fonts.googleapis.com/css?family=Montserrat|Source+Code+Pro&display=swap",
            class = :css
        )
        ],
        collapselevel = 1
    ),
    modules = [
        AdaptiveDesignOptimization
        # Base.get_extension(AdaptiveDesignOptimization, :TuringExt),
        # Base.get_extension(AdaptiveDesignOptimization, :NamedArraysExt)
    ],
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(repo = "github.com/itsdfish/AdaptiveDesignOptimization.jl.git")
