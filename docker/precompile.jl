# Installs and precompiles the packages of one notebook environment.
# Usage: julia precompile.jl <folder containing Project.toml + Manifest.toml>
using Pkg

src = ARGS[1]
dir = mktempdir()
cp(joinpath(src, "Project.toml"), joinpath(dir, "Project.toml"))
cp(joinpath(src, "Manifest.toml"), joinpath(dir, "Manifest.toml"))

Pkg.activate(dir)
Pkg.instantiate()
Pkg.precompile()
println("Installed and precompiled packages for ", basename(src))
