FROM ghcr.io/juliapluto/pluto-jupyter:latest

USER root
RUN mkdir -p /opt/julia-1.13.1 && \
    ( wget -qO- https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.1-linux-x86_64.tar.gz \
      || curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/1.13/julia-1.13.1-linux-x86_64.tar.gz ) \
    | tar -xz -C /opt/julia-1.13.1 --strip-components=1
ENV PATH=/opt/julia-1.13.1/bin:${PATH}
USER ${NB_UID}

RUN julia -e 'using Pkg; Pkg.add("Pluto"); Pkg.precompile()'

COPY --chown=${NB_UID} base/precompile.jl /tmp/precompile.jl
COPY --chown=${NB_UID} *.jl /tmp/notebooks/
RUN for f in /tmp/notebooks/*.jl; do julia /tmp/precompile.jl "$f"; done
