FROM ghcr.io/juliapluto/pluto-jupyter:latest

# 1. The base image ships Julia 1.6.1 and Pluto 0.14.7, far too old for these notebooks.
#    Install Julia 1.13.1 and put it first in PATH, so Pluto is started with it.
USER root
RUN mkdir -p /opt/julia-1.13.1 && \
    ( wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.1-linux-x86_64.tar.gz \
      || curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.1-linux-x86_64.tar.gz ) \
    | tar -xz -C /opt/julia-1.13.1 --strip-components=1
ENV PATH=/opt/julia-1.13.1/bin:${PATH}
USER ${NB_UID}

# 2. Install and precompile a current Pluto for Julia 1.13.1
RUN julia -e 'using Pkg; Pkg.add("Pluto"); Pkg.precompile()'

# 3. Copy ALL files of the repository (every notebook) into the image
COPY --chown=${NB_UID} . ${HOME}/

# 4. Precompile the exact packages of EVERY notebook (*.jl)
RUN for f in ${HOME}/*.jl; do julia ${HOME}/precompile.jl "$f"; done
