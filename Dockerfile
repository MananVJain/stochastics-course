FROM ghcr.io/mananvjain/stochastics-base:latest

COPY --chown=${NB_UID} *.jl ${HOME}/

RUN julia -e 'import Pluto' && \
    for f in ${HOME}/*.jl; do julia /tmp/precompile.jl "$f"; done
