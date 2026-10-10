# Build-time script: installs and precompiles the EXACT packages saved inside a
# Pluto notebook (its embedded Project.toml and Manifest.toml), so nobody waits for
# this when the notebook is opened later. Files that are not Pluto notebooks with
# saved packages are skipped.
using Pkg

path = ARGS[1]
nb = read(path, String)

function grab(name)
    m = match(Regex(name * " = \"\"\"\n(.*?)\"\"\"", "s"), nb)
    return m === nothing ? nothing : m.captures[1]
end

project = grab("PLUTO_PROJECT_TOML_CONTENTS")
manifest = grab("PLUTO_MANIFEST_TOML_CONTENTS")

if project === nothing || manifest === nothing
    println("Skipping $(basename(path)): no saved packages found.")
    exit(0)
end

dir = mktempdir()
write(joinpath(dir, "Project.toml"), project)
write(joinpath(dir, "Manifest.toml"), manifest)

Pkg.activate(dir)
Pkg.instantiate()
Pkg.precompile()
println("Packages of $(basename(path)) are installed and precompiled.")
