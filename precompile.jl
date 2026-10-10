# Build-time script: installs and precompiles the EXACT packages saved inside the
# Pluto notebook (its embedded Project.toml and Manifest.toml), so nobody waits for
# this when the notebook is opened later.
using Pkg

nb = read(ARGS[1], String)

function grab(name)
    m = match(Regex(name * " = \"\"\"\n(.*?)\"\"\"", "s"), nb)
    m === nothing && error("could not find $name in the notebook")
    return m.captures[1]
end

dir = mktempdir()
write(joinpath(dir, "Project.toml"), grab("PLUTO_PROJECT_TOML_CONTENTS"))
write(joinpath(dir, "Manifest.toml"), grab("PLUTO_MANIFEST_TOML_CONTENTS"))

Pkg.activate(dir)
Pkg.instantiate()
Pkg.precompile()
println("Notebook packages are installed and precompiled.")
