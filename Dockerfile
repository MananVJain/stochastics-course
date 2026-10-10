FROM ghcr.io/juliapluto/pluto-jupyter:latest

# Put the notebook into the image, owned by the notebook user
COPY --chown=${NB_UID} 00_introduction.jl ${HOME}/00_introduction.jl

# BUILD TIME: run the notebook once, so that its packages
# (PlutoUI, HypertextLiteral, Distributions) are installed and precompiled
# inside the image. Students then never wait for this.
RUN julia --project=/tmp/ps -e 'using Pkg; Pkg.add("PlutoSliderServer"); using PlutoSliderServer; PlutoSliderServer.export_notebook(joinpath(ENV["HOME"], "00_introduction.jl"))'
