FROM ghcr.io/mananvjain/stochastics-base:latest

COPY --chown=${NB_UID} *.jl ${HOME}/
