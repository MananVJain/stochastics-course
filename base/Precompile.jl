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
