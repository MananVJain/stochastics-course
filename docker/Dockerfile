FROM quay.io/jupyter/base-notebook:latest

LABEL org.opencontainers.image.source=https://github.com/MananVJain/stochastics-course

ARG JULIA_VERSION=1.13.1
ARG JULIA_MINOR=1.13
ARG PLUTO_VERSION=1.0.3

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/julia \
 && curl -fsSL --retry 5 --retry-delay 3 \
      "https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MINOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    | tar -xz -C /opt/julia --strip-components=1 \
 && ln -s /opt/julia/bin/julia /usr/local/bin/julia

USER ${NB_UID}
WORKDIR ${HOME}

RUN julia --startup-file=no -e 'using Pkg; Pkg.add(name="Pluto", version=ENV["PLUTO_VERSION"]); Pkg.precompile()'

COPY --chown=${NB_UID}:${NB_GID} docker/pluto_proxy /tmp/pluto_proxy
RUN pip install --no-cache-dir jupyter-server-proxy /tmp/pluto_proxy \
 && rm -rf /tmp/pluto_proxy

COPY --chown=${NB_UID}:${NB_GID} docker/precompile.jl /tmp/precompile.jl
COPY --chown=${NB_UID}:${NB_GID} envs/ /tmp/envs/
RUN for d in /tmp/envs/*/; do julia --startup-file=no /tmp/precompile.jl "$d"; done \
 && rm -rf /tmp/envs /tmp/precompile.jl

COPY --chown=${NB_UID}:${NB_GID} *.jl ${HOME}/

COPY --chown=${NB_UID}:${NB_GID} docker/smoke_test.sh /tmp/smoke_test.sh
RUN bash /tmp/smoke_test.sh && rm /tmp/smoke_test.sh
