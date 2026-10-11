# Stochastics course

Interactive Pluto notebooks. Click a link, wait for Binder to start, and the notebook opens by itself.

| Notebook | Link |
| --- | --- |
| 00 Introduction | [Open on Binder](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2Fopen%3Fpath%3D%252Fhome%252Fjovyan%252F00_introduction.jl) |
| 01 Probability | [Open on Binder](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2Fopen%3Fpath%3D%252Fhome%252Fjovyan%252F01_probability.jl) |
| 02 Random numbers | [Open on Binder](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2Fopen%3Fpath%3D%252Fhome%252Fjovyan%252F02_random_numbers.jl) |
| 03 Markov chains | [Open on Binder](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2Fopen%3Fpath%3D%252Fhome%252Fjovyan%252F03_markov_chains.jl) |
| 04 Monte Carlo | [Open on Binder](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2Fopen%3Fpath%3D%252Fhome%252Fjovyan%252F04_monte_carlo.jl) |

Or open [Pluto's start page](https://mybinder.org/v2/gh/MananVJain/stochastics-course/binder?urlpath=pluto%2F) and pick a notebook from the list.

## How this repo works (for the maintainer)

- `docker/Dockerfile` defines the single image: Julia 1.13.1, Pluto 1.0.3, every notebook's packages precompiled, and the notebooks.
- GitHub Actions builds it (`.github/workflows/build-course-image.yml`), publishes it to `ghcr.io/mananvjain/stochastics-course`, and rewrites the one-line `Dockerfile` on the `binder` branch to point at that exact image.
- Binder only builds that one line, so it takes seconds and never compiles anything.
- If you ever change a notebook or anything in `docker/`, push to `main` and everything updates by itself.
