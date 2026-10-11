### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c53da7f3-8dc9-bae2-85a9-6da83b211fc7
# packages used by this notebook
begin
	using PlutoUI, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions, SHA, Base64
end

# ╔═╡ 8c99463c-3198-d27e-ca3e-fd548d5cf8a2
md"""
# 4 · Monte Carlo Integration and Importance Sampling

*Stochastic Modeling and Simulation*
"""

# ╔═╡ e444a9e4-2eaf-7df2-f017-5fef67da466b
md"""
Solutions password $(@bind pw PasswordField(default = ""))
"""

# ╔═╡ 545007f7-ca6b-c1f0-b034-a020653c0d02
md"""
In the previous notebooks, random numbers were used to simulate random processes. This notebook uses random numbers to compute quantities that are not random: areas, integrals and expected values. The **Monte Carlo method** estimates such a quantity by an average over random samples. 

!!! hook "Guiding problem"
    For many years, a factory has discharged waste water into a lake. The authorities want to know the **total mass of pollutant on the lake bed** (in tonnes). The lake has an irregular shape, and the pollution is uneven: high near the outlet of the factory, possibly high at places that nobody knows. A drone can fly to any point of the lake, land a probe and measure the local concentration (in kg per hectare); each flight costs time and battery. 

    **How should the drone choose its measurement points, and how many flights are needed to determine the total mass within ± 10 %?**

    Your estimate of the number of flights: $(@bind guess_n NumberField(0:10:100000; default = 0))
"""

# ╔═╡ 969086ae-15cd-a847-acb1-206abbcef36f
md"""
!!! info "Key points"
    - The law of large numbers guarantees that the average of i.i.d. samples converges to their expected value ``\mu``. This is the reason why Monte Carlo methods work.
    - Weak law: at each large ``N``, a large deviation is improbable. Strong law: each run converges with probability one.
    - The law requires a finite mean, and it says nothing about the speed of convergence.
"""

# ╔═╡ fde1cec7-d538-6f67-c864-99b74aa612fd
md"""

### 3.2 Unbiasedness and variance of the estimator

**Question.** For a finite ``N``, how far is ``\theta_N`` from ``F``?

**Unbiasedness.** By linearity of the expected value and because every ``x_i`` has the same distribution,

```math
E[\theta_N] = \frac{b - a}{N}\sum_{i=1}^N E[f(X_i)] = (b - a)\,E[f(X)] = F .
```

The estimator is **unbiased**: on average over many runs it gives exactly ``F``. Its error is purely random.

**Variance.** Using ``\mathrm{Var}(cY) = c^2\,\mathrm{Var}(Y)`` and the fact that the variances of **independent** terms add,

```math
\mathrm{Var}(\theta_N) = \frac{(b-a)^2}{N^2}\sum_{i=1}^N \mathrm{Var}\big(f(X_i)\big) = \frac{(b-a)^2}{N^2}\,N\,\sigma_f^2 = \frac{(b-a)^2\,\sigma_f^2}{N}, \qquad \sigma_f^2 = \mathrm{Var}\big(f(X)\big).
```

The **standard error** is the standard deviation of the estimator,

```math
\mathrm{SE} = \sqrt{\mathrm{Var}(\theta_N)} = \frac{(b - a)\,\sigma_f}{\sqrt N} .
```

``\sigma_f^2`` is usually unknown, but it can be estimated from the same samples by the sample variance

```math
s_f^2 = \frac{1}{N - 1}\sum_{i=1}^N \big(f(x_i) - \langle f \rangle\big)^2, \qquad \langle f \rangle = \frac1N\sum_{i=1}^N f(x_i) .
```

So,
1. **The error decreases like ``1/\sqrt N``.** Ten times more accuracy requires one hundred times more samples.
2. **The constant ``\sigma_f`` matters.** The error is proportional to the variation of ``f(X)``. Besides increasing ``N``, the only way to reduce the error is to reduce this constant. This is the purpose of importance sampling (Section 5).

Samples per run ``N`` $(@bind s3_N Select([10, 100, 1000]; default = 100))
"""

# ╔═╡ 5acc51bd-a5d8-c73c-2186-673f925545bc
md"""
!!! think "Think"
    With 10 samples a classmate obtains an estimate that is closer to the exact value than yours with 10 000 samples. Is the classmate's estimate better?
"""

# ╔═╡ ca32279f-116a-45ef-58aa-2c1a9b20a9e9
md"""
!!! info "Key points"
    - Every integral over a bounded interval is an expected value: ``\int_a^b f\,dx = (b-a)\,E[f(X)]`` with ``X \sim \mathcal U(a, b)``.
    - The Monte Carlo estimator is unbiased; its standard error is ``(b-a)\sigma_f/\sqrt N``. The error decreases only like ``1/\sqrt N``.
"""

# ╔═╡ d67ea20b-8d7c-0207-df85-ca1f9e6a4219
md"""
## 4. Integrals in several dimensions: the drone survey

### 4.1 The multidimensional estimator

**Question.** The total mass of pollutant is an integral over a two-dimensional region. Does the method carry over?

**The integral.** Let ``\Omega \subset \mathbb R^2`` be the lake and ``c(\mathbf x)`` the concentration at the point ``\mathbf x = (x, y)`` (in kg/ha; 1 km² = 100 ha). The total mass is

```math
M = \int_\Omega c(\mathbf x)\,d^2\mathbf x .
```

Geometrically, ``M`` is the **volume** under the surface ``c(\mathbf x)`` above the lake. Figure 4.1 shows the integrand. Move the slice across the lake: each west–east slice has an area (right panel), and the volume is the sum of these slice areas times their thickness.

Slice at north $(@bind s4_y Slider(0.5:0.1:2.5; default = 1.8, show_value = true)) km
"""

# ╔═╡ 8491345c-04aa-5ae1-d4c0-29dc6c3f503d
md"""
The estimate converges, but the error band is wide. Most flights measure the low background concentration, while the few flights that land in the plume measure values much larger. Each of them makes the estimate jump. The single-flight estimates ``y_i`` vary strongly, so ``\mathrm{Var}(c(\mathbf X))`` is large.

!!! info "Key points"
    - In ``m`` dimensions, the estimator is ``V`` times the average of the integrand at uniform points of the domain, and its variance is ``V^2\,\mathrm{Var}(f(\mathbf X))/N``.
    - **An integrand that is large on a small region gives a large variance.**

### 4.3 Why random points instead of a grid?

**Question.** One could also measure on a regular grid. Why use random points?

**Grids.** A grid with ``n`` points per axis in ``m`` dimensions has ``N = n^m`` points with spacing ``h = 1/n`` (on the unit cube). For a smooth integrand, the error of the midpoint rule is proportional to ``h^2 = n^{-2}``. Expressed in the number of function evaluations,

```math
\text{grid error} \propto n^{-2} = N^{-2/m}, \qquad \text{Monte Carlo error} = \frac{V\sigma_f}{\sqrt N} \propto N^{-1/2} .
```

The Monte Carlo rate does not depend on the dimension. For ``m < 4`` the grid converges faster, and for ``m > 4`` Monte Carlo is faster. Moreover, a grid needs ``n^m`` points, with only 10 points per axis, a 20-dimensional grid has ``10^{20}`` points. 

Figure 4.3 compares both for the integral of ``e^{-|\mathbf x|^2}`` over the cube ``[0, 1]^m``. Change the dimension ``m`` and compare the slopes of the red and blue lines.

Dimension ``m`` $(@bind s4_m Select([1, 2, 4, 8, 16]; default = 2))
"""

# ╔═╡ a2e6e5ca-75dc-aa7b-791f-14cdf06df9d9
md"""
!!! info "Key points"
    - The Monte Carlo error decreases like ``N^{-1/2}`` in every dimension; the error of a grid decreases like ``N^{-2/m}`` for smooth integrands.
    - In low dimensions grids are better for smooth integrands; in high dimensions Monte Carlo is the only feasible method.
    - The constant ``\sigma_f`` may still depend on the dimension, but not the rate.
"""

# ╔═╡ 8578d6d1-252a-8d48-11b6-5f10e518ea53
md"""

- **A, uniform on ``[0, 4]``.** Valid, but most samples fall where ``f \approx 0`` and get weight almost zero; the few in the bump get large weights. Variance per sample about 1.56.
- **B, similar shape, slightly wider than ``f``.** The weights are nearly equal. Variance per sample about 0.063, about 25 times smaller than A.
- **C, narrower than ``f``.** In the tails of ``f``, ``q`` decays faster than ``f``, so ``f/q`` becomes large there. 
- **D, shifted.** Most samples fall where ``f`` is small; the few in the bump carry very large weights.
- **E, zero on half of the support of ``f``.** The condition ``q > 0`` where ``f \neq 0`` is violated. The estimator converges to ``\int_2^4 f\,dx = F/2``: it is biased.
"""

# ╔═╡ 96c69e43-2c05-127a-5719-630b00a82df4
md"""
!!! info "Key points"
    - Importance sampling draws from a proposal ``q`` and weights each sample by ``f/q``. It is unbiased **only if ``q > 0`` wherever ``f \neq 0``**; otherwise it is biased.
    - The variance is zero for ``q \propto f``. A good ``q`` has approximately the shape of ``f``, so that the weights are nearly equal.
"""

# ╔═╡ db81f0a8-1f2b-78ef-2a10-4a67a23173c4
md"""
With the default plan, most flights go to the plume, where the concentration is high, and the estimates of 300 surveys scatter much less than with uniform flights. ``\delta = 1`` gives back uniform flights.
"""

# ╔═╡ db30c858-4699-00c5-7b50-b965c9154615
md"""
!!! think "Think"
    With a narrow proposal around the outlet and ``\delta = 0``, the histogram of the estimates is no longer centred at the true value. Explain why, with reference to the map.
"""

# ╔═╡ dd479f0a-a0c4-dd34-f5cf-07ba0a960878
md"""
## 6. Solution of the guiding problem

The standard error of ``N`` flights is ``\sigma/\sqrt N``, where ``\sigma`` is the standard deviation of the estimate from a single flight (``y_i`` for uniform flights, ``w_i`` for importance sampling). A 95 % error band of ± 10 % requires ``2\sigma/\sqrt N = 0.1\,M``, that is, ``N = (2\sigma/(0.1\,M))^2``. The table gives ``\sigma`` for both flight plans, estimated from 20 000 simulated flights each.
"""

# ╔═╡ 6efd2c36-f857-de55-92cd-3be26c63ec0a
md"""
Importance sampling does not change the ``1/\sqrt N`` law; it reduces the constant ``\sigma``. 

## 7. Exercises

### 7.1 Calculation problems

**Problem 1.** Which law guarantees that a Monte Carlo estimator converges as ``N \to \infty``? State it in words and as an equation, and name its assumptions.
"""

# ╔═╡ 6d0f2434-cac6-7a87-00fd-0714e94a2dcf
md"""
**Problem 2.** Compute analytically the variance of the Monte Carlo estimator of ``\int_0^1 x^3\,dx`` with ``N`` samples from ``\mathcal U(0, 1)``. Give a closed-form expression.
"""

# ╔═╡ d9c4bd58-0ba5-f994-4202-18482d70966c
md"""
**Problem 3.** Compute analytically the variance of the importance sampling estimator of ``\int_0^1 x^3\,dx`` with the proposal ``q(x) = 2x`` on ``[0, 1]``, for ``N`` samples. By what factor is it smaller than in Problem 2? How are samples from ``q`` generated?
"""

# ╔═╡ c00b6b12-2f42-414b-e52b-5c79b8015d12
md"""
**Problem 5.** The integral ``\int_0^\infty e^{-x}\cos(x)\,dx`` is to be estimated. (a) Why can it not be estimated with uniform samples as in Section 3.1? (b) Write it as an expectation with respect to the exponential distribution and give the estimator. (c) Is the variance of this estimator finite?
"""

# ╔═╡ 3829776e-d93a-895f-b637-202a0c5e8b34
md"""
### 7.2 Programming exercises

**Exercise 1 · Debugging: estimating ``\pi``.** `estimate_pi(N, rng)` should estimate ``\pi`` with ``N`` random points in the unit square. The code uses the same random number for both coordinates. Predict where the points lie and which value the function returns; then correct it.
"""

# ╔═╡ 4b5b546f-196b-fe76-a330-3a20044a2194
function estimate_pi(N, rng)
	inside = 0
	for _ in 1:N
		x = y = rand(rng)
		x^2 + y^2 <= 1 && (inside += 1)
	end
	return 4inside / N
end;

# ╔═╡ 82df10b9-7bc3-ad36-97e2-c6da0d582441
md"""
**Exercise 2 · Debugging: the standard error.** `mc_integrate(f, a, b, N)` should return the estimate and its standard error ``(b - a)\,s_f/\sqrt N``. Predict by which factor the standard error must decrease when ``N`` is multiplied by 100; then correct the code.
"""

# ╔═╡ 646faf41-c93f-9579-6d05-99b81a3a316b
function mc_integrate(f, a, b, N)
	x = a .+ (b - a) .* rand(N)
	fx = f.(x)
	return (estimate = (b - a) * mean(fx), se = (b - a) * std(fx) / N)
end;

# ╔═╡ ba37f52d-2959-d794-d844-8e3129b23b32
md"""
**Exercise 3 · Debugging: importance sampling.** `is_x2(N)` should estimate ``\int_0^1 x^2\,dx = 1/3`` by importance sampling with ``q(x) = 2x``: samples ``X = \sqrt U`` and weights ``x^2/(2x)``. Predict which value is estimated if the samples are uniform while the weights still use ``q(x) = 2x``; then correct the code.
"""

# ╔═╡ 296d4abe-987f-c750-d0c6-d62ac781acd6
function is_x2(N)
	x = rand(N)
	w = x .^ 2 ./ (2 .* x)
	return (estimate = mean(w), se = std(w) / sqrt(N))
end;

# ╔═╡ ad58a80b-7990-6d1c-5409-b7bd2dfebfb7
md"""
## Further reading

| resource | content |
|:--|:--|
| [A. B. Owen, *Monte Carlo Theory, Methods and Examples*, Chapters 1–2 and 9](https://artowen.su.domains/mc/) | Monte Carlo integration and importance sampling |
| [N. Metropolis, *The beginning of the Monte Carlo method*, Los Alamos Science (1987)](https://sgp.fas.org/othergov/doe/lanl/pubs/00326866.pdf) | the history |
| [M. Pharr, W. Jakob and G. Humphreys, *Physically Based Rendering*, Chapter 13](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration) | Monte Carlo integration and importance sampling in computer graphics |

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 4.
- A. B. Owen, *Monte Carlo Theory, Methods and Examples*, 2013.
- C. P. Robert and G. Casella, *Monte Carlo Statistical Methods*, 2nd ed., Springer, 2004, Chapter 3.
- N. Metropolis and S. Ulam, "The Monte Carlo method", *Journal of the American Statistical Association* 44 (1949) 335–341.
"""

# ╔═╡ b4dc5b17-3cbd-c88d-597d-5c7db40b9053
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 3ff4f2a8-879b-357c-5aa2-cb9830b2eb7a
# Drawing figures (SVG)
begin
	const COL = (blue = "#1f5aa6", red = "#c2462e", ochre = "#b8860b", green = "#2f7d55", purple = "#6a4c93",
		grey = "#8c8c8c", sky = "#7fa8d8", ink = "#1b1b1b", light = "#d9e4f2", pink = "#e7a493", teal = "#2b6f74")

	fmt(x::Integer) = string(x)

	function fmt(x::Real; d = 3)
		isnan(x) && return "–"
		isinf(x) && return x > 0 ? "∞" : "−∞"
		x == 0 && return "0"
		a = abs(x)
		if a < 1e-3 || a >= 1e6
			e = floor(Int, log10(a)); m = x / 10.0^e
			return string(round(m, sigdigits = d)) * " × 10" * supnum(e)
		end
		s = string(round(x, sigdigits = d))
		endswith(s, ".0") ? s[1:end-2] : s
	end

	pct(x; d = 1) = (d == 0 ? string(round(Int, 100x)) : string(round(100x, digits = d))) * " %"

	function supnum(e::Integer)
		m = Dict('-' => '⁻', '0' => '⁰', '1' => '¹', '2' => '²', '3' => '³', '4' => '⁴', '5' => '⁵', '6' => '⁶', '7' => '⁷', '8' => '⁸', '9' => '⁹')
		join(m[c] for c in string(e))
	end

	xesc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")

	B(x) = Markdown.Bold(Any[string(x)])

	mutable struct Fig
		w::Int; h::Int
		l::Int; r::Int; t::Int; b::Int
		xr::Tuple{Float64, Float64}; yr::Tuple{Float64, Float64}
		xlog::Bool; ylog::Bool
		axes::Bool; frame::Bool
		xlabel::String; ylabel::String; title::String
		xticks::Any; yticks::Any; xtl::Any; ytl::Any
		grid::Bool
		body::IOBuffer; over::IOBuffer
		id::String
	end

	function fig(; w = 640, h = 300, xr = (0, 1), yr = (0, 1), xlabel = "", ylabel = "", title = "",
		xlog = false, ylog = false, axes = true, frame = false, xticks = nothing, yticks = nothing,
		xticklabels = nothing, yticklabels = nothing, grid = false, l = nothing, r = 16, t = nothing, b = nothing)
		ll = l === nothing ? (axes ? (ylabel == "" ? 46 : 62) : 10) : l
		tt = t === nothing ? (title == "" ? 14 : 30) : t
		bb = b === nothing ? (axes ? (xlabel == "" ? 28 : 46) : 10) : b
		Fig(w, h, ll, r, tt, bb, (Float64(xr[1]), Float64(xr[2])), (Float64(yr[1]), Float64(yr[2])), xlog, ylog, axes, frame,
			xlabel, ylabel, title, xticks, yticks, xticklabels, yticklabels, grid, IOBuffer(), IOBuffer(), string(rand(UInt32), base = 36))
	end

	_tx(f, x) = f.xlog ? log10(max(x, 1e-300)) : x

	_ty(f, y) = f.ylog ? log10(max(y, 1e-300)) : y

	_lim(f, r, lg) = lg ? (log10(r[1]), log10(r[2])) : r

	X(f::Fig, x) = (a = _lim(f, f.xr, f.xlog); f.l + (_tx(f, x) - a[1]) / (a[2] - a[1]) * (f.w - f.l - f.r))

	Y(f::Fig, y) = (a = _lim(f, f.yr, f.ylog); f.h - f.b - (_ty(f, y) - a[1]) / (a[2] - a[1]) * (f.h - f.t - f.b))

	_n(v) = @sprintf("%.1f", v)

	function _st(s, o)
		a = ""
		s === nothing || (a *= " data-s=\"$(s)\"")
		if o isa Tuple
			a *= " data-ra=\"$(o[1])\" data-rb=\"$(o[2])\""
		elseif o !== nothing
			a *= " data-o=\"$(o)\""
		end
		a
	end

	_dash(d) = d === nothing ? "" : " stroke-dasharray=\"$(d)\""

	_pick(v, i) = v isa AbstractVector ? v[i] : v

	function line!(f::Fig, x, y; c = COL.blue, w = 1.8, dash = nothing, op = 1.0, s = nothing, o = nothing, steps = false, over = false)
		d = IOBuffer(); pen = false; px = NaN
		for (xi, yi) in zip(x, y)
			if !(isfinite(xi) && isfinite(yi))
				pen = false; continue
			end
			if steps && pen
				print(d, "H", _n(X(f, xi)))
			end
			print(d, pen ? "L" : "M", _n(X(f, xi)), ",", _n(Y(f, yi))); pen = true
		end
		print(over ? f.over : f.body, "<path d=\"", String(take!(d)), "\" fill=\"none\" stroke=\"$(c)\" stroke-width=\"$(w)\" stroke-linejoin=\"round\" stroke-linecap=\"round\" opacity=\"$(op)\"", _dash(dash), _st(s, o), "/>")
		f
	end

	function dots!(f::Fig, x, y; c = COL.blue, r = 3.2, op = 0.9, stroke = nothing, sw = 1, s = nothing, o = nothing, over = false)
		io = over ? f.over : f.body
		for i in eachindex(x)
			(isfinite(x[i]) && isfinite(y[i])) || continue
			print(io, "<circle cx=\"", _n(X(f, x[i])), "\" cy=\"", _n(Y(f, y[i])), "\" r=\"", _pick(r, i), "\" fill=\"", _pick(c, i), "\" opacity=\"", _pick(op, i), "\"",
				stroke === nothing ? "" : " stroke=\"$(_pick(stroke, i))\" stroke-width=\"$(sw)\"", _st(_pick(s, i), _pick(o, i)), "/>")
		end
		f
	end

	function hist!(f::Fig, edges, hgt; c = COL.blue, op = 0.75, stroke = "white", s = nothing, o = nothing)
		for i in eachindex(hgt)
			x0 = X(f, edges[i]); x1 = X(f, edges[i+1]); y0 = Y(f, 0); y1 = Y(f, hgt[i])
			print(f.body, "<rect x=\"", _n(x0), "\" y=\"", _n(min(y0, y1)), "\" width=\"", _n(max(x1 - x0, 0)), "\" height=\"", _n(abs(y0 - y1)),
				"\" fill=\"$(c)\" opacity=\"$(op)\" stroke=\"$(stroke)\" stroke-width=\"0.6\"", _st(s, o), "/>")
		end
		f
	end

	function area!(f::Fig, x, y; c = COL.blue, op = 0.18, base = 0.0, s = nothing, o = nothing)
		pts = [(X(f, xi), Y(f, yi)) for (xi, yi) in zip(x, y) if isfinite(yi)]
		isempty(pts) && return f
		d = "M" * _n(pts[1][1]) * "," * _n(Y(f, base)) * join(["L" * _n(p[1]) * "," * _n(p[2]) for p in pts]) * "L" * _n(pts[end][1]) * "," * _n(Y(f, base)) * "Z"
		print(f.body, "<path d=\"$(d)\" fill=\"$(c)\" opacity=\"$(op)\" stroke=\"none\"", _st(s, o), "/>")
		f
	end

	function band!(f::Fig, x, lo, hi; c = COL.blue, op = 0.15, s = nothing, o = nothing)
		p1 = ["$(_n(X(f, x[i]))),$(_n(Y(f, hi[i])))" for i in eachindex(x)]
		p2 = ["$(_n(X(f, x[i]))),$(_n(Y(f, lo[i])))" for i in reverse(eachindex(x))]
		print(f.body, "<polygon points=\"", join(vcat(p1, p2), " "), "\" fill=\"$(c)\" opacity=\"$(op)\" stroke=\"none\"", _st(s, o), "/>")
		f
	end

	function poly!(f::Fig, x, y; fill = COL.blue, op = 0.2, stroke = "none", sw = 1, s = nothing, o = nothing)
		print(f.body, "<polygon points=\"", join(["$(_n(X(f, a))),$(_n(Y(f, b)))" for (a, b) in zip(x, y)], " "), "\" fill=\"$(fill)\" opacity=\"$(op)\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\"", _st(s, o), "/>")
		f
	end

	function seg!(f::Fig, x0, y0, x1, y1; c = COL.grey, w = 1.2, dash = nothing, op = 1.0, s = nothing, o = nothing, over = false)
		print(over ? f.over : f.body, "<line x1=\"$(_n(X(f, x0)))\" y1=\"$(_n(Y(f, y0)))\" x2=\"$(_n(X(f, x1)))\" y2=\"$(_n(Y(f, y1)))\" stroke=\"$(c)\" stroke-width=\"$(w)\" opacity=\"$(op)\" stroke-linecap=\"round\"", _dash(dash), _st(s, o), "/>")
		f
	end

	vline!(f::Fig, x; c = "#555", w = 1.1, dash = "4 3", s = nothing, o = nothing) = seg!(f, x, f.yr[1], x, f.yr[2]; c, w, dash, s, o)

	hline!(f::Fig, y; c = "#555", w = 1.1, dash = "4 3", s = nothing, o = nothing) = seg!(f, f.xr[1], y, f.xr[2], y; c, w, dash, s, o)

	function text!(f::Fig, x, y, str; c = COL.ink, size = 12, anchor = "start", italic = false, bold = false, dx = 0, dy = 0, px = false, s = nothing, o = nothing, rot = 0, bg = false)
		a = px ? x : X(f, x); b = px ? y : Y(f, y)
		a += dx; b += dy
		tr = rot == 0 ? "" : " transform=\"rotate($(rot) $(_n(a)) $(_n(b)))\""
		sty = (italic ? " font-style=\"italic\"" : "") * (bold ? " font-weight=\"600\"" : "")
		halo = bg ? " paint-order=\"stroke\" stroke=\"white\" stroke-width=\"3.5\" stroke-linejoin=\"round\"" : ""
		print(f.over, "<text x=\"$(_n(a))\" y=\"$(_n(b))\" font-size=\"$(size)\" fill=\"$(c)\" text-anchor=\"$(anchor)\" dominant-baseline=\"middle\"$(sty)$(tr)$(halo)", _st(s, o), ">", xesc(str), "</text>")
		f
	end

	function rect!(f::Fig, x0, y0, x1, y1; fill = COL.blue, op = 0.15, stroke = "none", sw = 1, rx = 0, s = nothing, o = nothing, over = false)
		a0, a1 = sort([X(f, x0), X(f, x1)]); b0, b1 = sort([Y(f, y0), Y(f, y1)])
		print(over ? f.over : f.body, "<rect x=\"$(_n(a0))\" y=\"$(_n(b0))\" width=\"$(_n(a1 - a0))\" height=\"$(_n(b1 - b0))\" rx=\"$(rx)\" fill=\"$(fill)\" opacity=\"$(op)\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\"", _st(s, o), "/>")
		f
	end

	function circle!(f::Fig, x, y, rpx; fill = "white", stroke = COL.ink, sw = 1.5, op = 1.0, s = nothing, o = nothing, over = false)
		print(over ? f.over : f.body, "<circle cx=\"$(_n(X(f, x)))\" cy=\"$(_n(Y(f, y)))\" r=\"$(rpx)\" fill=\"$(fill)\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\" opacity=\"$(op)\"", _st(s, o), "/>")
		f
	end

	function legend!(f::Fig, items; at = :tr, size = 12)
		if at == :top
			x = Float64(f.l); yy = max(f.t / 2, 9.0)
			for it in items
				lab, c = it[1], it[2]; kind = length(it) > 2 ? it[3] : :line
				if kind == :bar
					print(f.over, "<rect x=\"$(_n(x))\" y=\"$(_n(yy - 6))\" width=\"14\" height=\"12\" fill=\"$(c)\" opacity=\"0.8\"/>")
				elseif kind == :dot
					print(f.over, "<circle cx=\"$(_n(x + 7))\" cy=\"$(_n(yy))\" r=\"4\" fill=\"$(c)\"/>")
				else
					print(f.over, "<line x1=\"$(_n(x))\" x2=\"$(_n(x + 18))\" y1=\"$(_n(yy))\" y2=\"$(_n(yy))\" stroke=\"$(c)\" stroke-width=\"2.2\"" * (kind == :dash ? " stroke-dasharray=\"4 3\"" : "") * "/>")
				end
				print(f.over, "<text x=\"$(_n(x + 24))\" y=\"$(_n(yy))\" font-size=\"$(size)\" fill=\"#222\" dominant-baseline=\"middle\">", xesc(lab), "</text>")
				x += 24 + 6.4 * length(lab) * size / 12 + 18
			end
			return f
		end
		n = length(items); lh = 18; wmax = maximum(length(first(it)) for it in items) * 6.6 + 34
		x0 = at in (:tr, :br) ? f.w - f.r - wmax - 4 : f.l + 10
		y0 = at in (:tr, :tl) ? f.t + 8 : f.h - f.b - n * lh - 4
		print(f.over, "<rect x=\"$(_n(x0 - 6))\" y=\"$(_n(y0 - 4))\" width=\"$(_n(wmax + 8))\" height=\"$(n * lh + 4)\" fill=\"white\" opacity=\"0.85\" rx=\"3\"/>")
		for (i, it) in enumerate(items)
			lab, c = it[1], it[2]; kind = length(it) > 2 ? it[3] : :line
			yy = y0 + (i - 0.5) * lh
			if kind == :line
				print(f.over, "<line x1=\"$(_n(x0))\" x2=\"$(_n(x0 + 20))\" y1=\"$(_n(yy))\" y2=\"$(_n(yy))\" stroke=\"$(c)\" stroke-width=\"2.2\"/>")
			elseif kind == :dash
				print(f.over, "<line x1=\"$(_n(x0))\" x2=\"$(_n(x0 + 20))\" y1=\"$(_n(yy))\" y2=\"$(_n(yy))\" stroke=\"$(c)\" stroke-width=\"1.6\" stroke-dasharray=\"4 3\"/>")
			elseif kind == :dot
				print(f.over, "<circle cx=\"$(_n(x0 + 10))\" cy=\"$(_n(yy))\" r=\"4\" fill=\"$(c)\"/>")
			else
				print(f.over, "<rect x=\"$(_n(x0 + 3))\" y=\"$(_n(yy - 6))\" width=\"14\" height=\"12\" fill=\"$(c)\" opacity=\"0.8\"/>")
			end
			print(f.over, "<text x=\"$(_n(x0 + 28))\" y=\"$(_n(yy))\" font-size=\"$(size)\" fill=\"#222\" dominant-baseline=\"middle\">", xesc(lab), "</text>")
		end
		f
	end

	function niceticks(lo, hi, n = 5)
		span = hi - lo; span > 0 || return [lo]
		raw = span / n; mag = 10.0^floor(log10(raw)); nr = raw / mag
		step = (nr < 1.5 ? 1 : nr < 3 ? 2 : nr < 7 ? 5 : 10) * mag
		t = ceil(lo / step) * step; out = Float64[]
		while t <= hi + 1e-9 * span
			push!(out, abs(t) < 1e-12 * span ? 0.0 : t); t += step
		end
		out
	end

	function tickfmt(ts)
		d = 0
		for t in ts
			k = something(findfirst(k -> abs(t - round(t, digits = k)) < 1e-9 * max(1, abs(t)), 0:4), 5) - 1
			d = max(d, k)
		end
		v -> v == 0 ? "0" : (a = abs(v); (a != 0 && (a < 1e-3 || a >= 1e5)) ? fmt(v) : (d == 0 ? string(round(Int, v)) : @sprintf("%.*f", d, v)))
	end

	function _axes(io, f::Fig)
		x0, x1 = f.l, f.w - f.r; y0, y1 = f.t, f.h - f.b
		ink = "#333"; tc = "#555"
		if f.xlog
			xt = f.xticks === nothing ? [10.0^k for k in ceil(Int, log10(f.xr[1])):floor(Int, log10(f.xr[2]))] : f.xticks
			xl = f.xtl === nothing ? ["10" * supnum(round(Int, log10(t))) for t in xt] : f.xtl
		else
			xt = f.xticks === nothing ? niceticks(f.xr..., clamp(round(Int, (f.w - f.l - f.r) / 80), 2, 7)) : f.xticks
			xl = f.xtl === nothing ? map(tickfmt(xt), xt) : f.xtl
		end
		if f.ylog
			yt = f.yticks === nothing ? [10.0^k for k in ceil(Int, log10(f.yr[1])):floor(Int, log10(f.yr[2]))] : f.yticks
			length(yt) > 7 && (yt = yt[1:2:end])
			yl = f.ytl === nothing ? ["10" * supnum(round(Int, log10(t))) for t in yt] : f.ytl
		else
			yt = f.yticks === nothing ? niceticks(f.yr..., 4) : f.yticks
			yl = f.ytl === nothing ? map(tickfmt(yt), yt) : f.ytl
		end
		if f.grid
			for t in yt
				yy = Y(f, t); print(io, "<line x1=\"$(x0)\" x2=\"$(x1)\" y1=\"$(_n(yy))\" y2=\"$(_n(yy))\" stroke=\"#ececE8\" stroke-width=\"1\"/>")
			end
		end
		print(io, "<line x1=\"$(x0)\" x2=\"$(x1)\" y1=\"$(y1)\" y2=\"$(y1)\" stroke=\"$(ink)\" stroke-width=\"0.9\"/>")
		print(io, "<line x1=\"$(x0)\" x2=\"$(x0)\" y1=\"$(y0)\" y2=\"$(y1)\" stroke=\"$(ink)\" stroke-width=\"0.9\"/>")
		if f.frame
			print(io, "<line x1=\"$(x0)\" x2=\"$(x1)\" y1=\"$(y0)\" y2=\"$(y0)\" stroke=\"$(ink)\" stroke-width=\"0.6\"/>")
			print(io, "<line x1=\"$(x1)\" x2=\"$(x1)\" y1=\"$(y0)\" y2=\"$(y1)\" stroke=\"$(ink)\" stroke-width=\"0.6\"/>")
		end
		for (t, s) in zip(xt, xl)
			xx = X(f, t); (xx < x0 - 1 || xx > x1 + 1) && continue
			print(io, "<line x1=\"$(_n(xx))\" x2=\"$(_n(xx))\" y1=\"$(y1)\" y2=\"$(y1 + 4)\" stroke=\"$(ink)\" stroke-width=\"0.9\"/>")
			hw = 3.3 * length(s)
			anc = xx + hw > f.w - 1 ? "end" : xx - hw < 1 ? "start" : "middle"
			tx = anc == "end" ? min(xx + 3, f.w - 1.0) : anc == "start" ? max(xx - 3, 1.0) : xx
			print(io, "<text x=\"$(_n(tx))\" y=\"$(y1 + 16)\" font-size=\"11.5\" fill=\"$(tc)\" text-anchor=\"$(anc)\">", xesc(s), "</text>")
		end
		for (t, s) in zip(yt, yl)
			yy = Y(f, t); (yy < y0 - 1 || yy > y1 + 1) && continue
			print(io, "<line x1=\"$(x0 - 4)\" x2=\"$(x0)\" y1=\"$(_n(yy))\" y2=\"$(_n(yy))\" stroke=\"$(ink)\" stroke-width=\"0.9\"/>")
			print(io, "<text x=\"$(x0 - 7)\" y=\"$(_n(yy + 4))\" font-size=\"11.5\" fill=\"$(tc)\" text-anchor=\"end\">", xesc(s), "</text>")
		end
		f.xlabel == "" || print(io, "<text x=\"$(_n((x0 + x1) / 2))\" y=\"$(f.h - 8)\" font-size=\"13\" fill=\"#222\" text-anchor=\"middle\">", xesc(f.xlabel), "</text>")
		f.ylabel == "" || print(io, "<text transform=\"translate(14,$(_n((y0 + y1) / 2))) rotate(-90)\" font-size=\"13\" fill=\"#222\" text-anchor=\"middle\">", xesc(f.ylabel), "</text>")
	end

	function svgstring(f::Fig)
		io = IOBuffer()
		print(io, "<svg class=\"fig\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $(f.w) $(f.h)\" width=\"$(round(Int, 0.97 * f.w))\" font-family=\"system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif\">")
		print(io, "<defs><clipPath id=\"c$(f.id)\"><rect x=\"$(f.l)\" y=\"$(f.t - 2)\" width=\"$(f.w - f.l - f.r + 2)\" height=\"$(f.h - f.t - f.b + 4)\"/></clipPath></defs>")
		f.title == "" || print(io, "<text x=\"$(f.l)\" y=\"16\" font-size=\"13.5\" font-weight=\"600\" fill=\"#1b1b1b\">", xesc(f.title), "</text>")
		print(io, "<g clip-path=\"url(#c$(f.id))\">", String(copy(f.body.data[1:f.body.size])), "</g>")
		f.axes && _axes(io, f)
		print(io, String(copy(f.over.data[1:f.over.size])), "</svg>")
		String(take!(io))
	end

	Base.show(io::IO, ::MIME"text/html", f::Fig) = print(io, "<div class=\"fg\">", svgstring(f), "</div>")

	struct Panels
		items::Vector{Any}
		caption::Any
		vertical::Bool
	end

	function _html(x)
		x isa Fig && return svgstring(x)
		io = IOBuffer(); show(io, MIME"text/html"(), x); String(take!(io))
	end

	figure(items...; caption = nothing) = Panels(collect(items), caption, false)

	vstack(items...; caption = nothing) = Panels(collect(items), caption, true)

	function Base.show(io::IO, m::MIME"text/html", p::Panels)
		print(io, p.vertical ? "<div class=\"fg\"><div>" : "<div class=\"fg\"><div class=\"fg-row\">")
		for it in p.items
			print(io, "<div>", _html(it), "</div>")
		end
		print(io, "</div>")
		if p.caption !== nothing
			print(io, "<div class=\"fg-cap\">", _html(p.caption isa AbstractString ? Markdown.parse(p.caption) : p.caption), "</div>")
		end
		print(io, "</div>")
	end

	_tcell(c) = c isa AbstractString ? (q = Markdown.parse(c).content; isempty(q) ? Any[""] : q[1] isa Markdown.Paragraph ? q[1].content : Any[c]) :
		c isa Number ? Any[string(c)] : Any[c]

	function tbl(header, rows)
		Markdown.MD(Any[Markdown.Table(vcat([Any[_tcell(h) for h in header]], [Any[_tcell(c) for c in r] for r in rows]),
			vcat([:l], fill(:r, length(header) - 1)))])
	end

	function histdata(xs, edges; density = true)
		c = zeros(length(edges) - 1)
		for x in xs
			i = searchsortedlast(edges, x)
			(1 <= i <= length(c)) && (c[i] += 1)
			x == edges[end] && (c[end] += 1)
		end
		density ? c ./ (length(xs) .* diff(edges)) : c
	end
end;

# ╔═╡ 32afd1f8-1018-5749-a0f2-ce44586ef525
# Animations, walkthroughs and sound
begin
	struct Player
		items::Vector{Any}
		n::Int
		start::Int
		fps::Float64
		reads::Vector{String}
		label::String
		caption::Any
		loop::Bool
	end

	player(items...; n, start = 0, fps = 12, reads = String[], label = "step", caption = nothing, loop = false) =
		Player(collect(items), n, start, fps, reads, label, caption, loop)

	function Base.show(io::IO, m::MIME"text/html", p::Player)
		inner = join(["<div>" * _html(it) * "</div>" for it in p.items])
		capt = p.caption === nothing ? "" : "<div class=\"fg-cap\">" * _html(p.caption isa AbstractString ? Markdown.parse(p.caption) : p.caption) * "</div>"
		show(io, m, @htl("""<div class="pl"><div class="fg-row pl-stage">$(HTML(inner))</div>
		<div class="pl-bar"><button class="pl-reset" title="back to the start">⏮</button><button class="pl-play" title="play or pause">▶</button><button class="pl-step" title="one step">⏭</button><button class="pl-end" title="jump to the end">⏭⏭</button>
		<input class="pl-range" type="range" min="0" max="$(p.n)" step="1" value="$(p.start)"><span class="pl-lab"></span>
		<label>speed <select class="pl-speed"><option value="1">1×</option><option value="2">2×</option><option value="5">5×</option><option value="20">20×</option></select></label></div>
		<div class="pl-read"></div>$(HTML(capt))
		<script>
		const root = currentScript.parentElement;
		const n = $(p.n), fps = $(p.fps), reads = $(p.reads), lab = $(p.label), loop = $(p.loop);
		const els = Array.from(root.querySelectorAll("[data-s],[data-o],[data-ra]"));
		const S = els.map(e => e.hasAttribute("data-s") ? +e.getAttribute("data-s") : null);
		const O = els.map(e => e.hasAttribute("data-o") ? +e.getAttribute("data-o") : null);
		const RA = els.map(e => e.hasAttribute("data-ra") ? +e.getAttribute("data-ra") : null);
		const RB = els.map(e => e.hasAttribute("data-rb") ? +e.getAttribute("data-rb") : null);
		const range = root.querySelector(".pl-range"), lb = root.querySelector(".pl-lab"), rd = root.querySelector(".pl-read");
		const play = root.querySelector(".pl-play");
		let k = +range.value, timer = null;
		function show(j) {
			k = Math.max(0, Math.min(n, j)); range.value = k;
			for (let i = 0; i < els.length; i++) {
				const vis = (S[i] === null || S[i] <= k) && (O[i] === null || O[i] === k) && (RA[i] === null || (RA[i] <= k && k <= RB[i]));
				els[i].style.display = vis ? "" : "none";
			}
			lb.textContent = lab + " " + k + " / " + n;
			if (reads.length > 0) rd.innerHTML = reads[Math.min(k, reads.length - 1)];
		}
		function stop() { if (timer) { clearInterval(timer); timer = null; } play.textContent = "▶"; }
		function start() {
			if (k >= n) show(0);
			play.textContent = "⏸";
			timer = setInterval(() => { const sp = +root.querySelector(".pl-speed").value; if (k >= n) { if (loop) show(0); else stop(); } else show(Math.min(n, k + sp)); }, 1000 / fps);
		}
		play.addEventListener("click", () => timer ? stop() : start());
		root.querySelector(".pl-step").addEventListener("click", () => { stop(); show(k + 1); });
		root.querySelector(".pl-reset").addEventListener("click", () => { stop(); show(0); });
		root.querySelector(".pl-end").addEventListener("click", () => { stop(); show(n); });
		range.addEventListener("input", () => { stop(); show(+range.value); });
		invalidation.then(stop);
		show(k);
		</script></div>"""))
	end
end;

# ╔═╡ 468ac6f3-fd09-6aad-5cc5-2c203a5245aa
# Questions, answers and checks
begin
	adm(kind, title, body) = Markdown.MD(Markdown.Admonition(kind, title, [body]))

	function mcq(question, options::Pair...; correct::Int, head = "Quick check")
		opts = [@htl("""<div class="quiz-opt" data-i="$(i)" tabindex="0" role="button"><span class="quiz-key">$(string('A' + i - 1))</span><span>$(lab)</span></div>""") for (i, (lab, _)) in enumerate(options)]
		fbs = [@htl("""<div class="quiz-fb $(i == correct ? "yes" : "no")" data-i="$(i)" style="display:none"><b>$(i == correct ? "Correct." : "Not quite.")</b> $(fb)</div>""") for (i, (_, fb)) in enumerate(options)]
		@htl("""<div class="quiz"><div class="quiz-head">$(head)</div><div class="quiz-q">$(question)</div>
		<div class="quiz-opts">$(opts)</div>$(fbs)
		<script>
		const root = currentScript.parentElement, correct = $(correct);
		root.querySelectorAll(".quiz-opt").forEach(o => {
			const pick = () => {
				const i = +o.dataset.i;
				root.querySelectorAll(".quiz-fb").forEach(f => { f.style.display = (+f.dataset.i === i) ? "block" : "none"; });
				root.querySelectorAll(".quiz-opt").forEach(x => x.classList.remove("picked", "right", "wrong"));
				o.classList.add("picked", i === correct ? "right" : "wrong");
			};
			o.addEventListener("click", pick);
			o.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
		});
		</script></div>""")
	end

	function numq(question, answer::Real; rtol = 0.02, atol = 0.0, unit = "", why = md"", head = "Quick check")
		@htl("""<div class="quiz"><div class="quiz-head">$(head)</div><div class="quiz-q">$(question)</div>
		<div class="quiz-num"><input type="text" inputmode="decimal" placeholder="your answer" size="14"> <span>$(unit)</span>
		<button class="q-check">Check</button> <button class="q-show" style="display:none">Show the answer</button></div>
		<div class="quiz-fb no q-msg" style="display:none"></div>
		<div class="quiz-fb yes q-why" style="display:none"><b class="q-why-head">Correct.</b> $(why)</div>
		<script>
		const root = currentScript.parentElement, ans = $(Float64(answer)), rtol = $(Float64(rtol)), atol = $(Float64(atol));
		const inp = root.querySelector("input"), msg = root.querySelector(".q-msg"), why = root.querySelector(".q-why"), show = root.querySelector(".q-show");
		let tries = 0;
		const value = s => {
			s = s.trim().replace(/−/g, "-").replace(/×/g, "*").replace(/π/g, "pi").replace(/\\^/g, "**").replace(/,/g, ".");
			if (/[^0-9+\\-*/().eE\\s]/.test(s.replace(/pi|sqrt|exp|ln|log/g, ""))) return NaN;
			s = s.replace(/pi/g, "Math.PI").replace(/sqrt/g, "Math.sqrt").replace(/exp/g, "Math.exp").replace(/ln|log/g, "Math.log");
			try { return Function("return (" + s + ")")(); } catch (e) { return NaN; }
		};
		const check = () => {
			const v = value(inp.value); tries++;
			if (!isFinite(v)) { msg.style.display = "block"; msg.textContent = "This could not be read as a number. Examples: 0.75, 3/4, 2*pi, 1e-3."; return; }
			if (Math.abs(v - ans) <= Math.max(atol, rtol * Math.abs(ans))) {
				msg.style.display = "none"; why.style.display = "block"; inp.classList.add("ok"); show.style.display = "none";
			} else {
				why.style.display = "none"; msg.style.display = "block";
				msg.textContent = (v > ans ? "Too high." : "Too low.") + (tries >= 2 ? " Check the reasoning, or show the answer." : " Try again.");
				if (tries >= 2) show.style.display = "inline-block";
			}
		};
		root.querySelector(".q-check").addEventListener("click", check);
		inp.addEventListener("keydown", e => { if (e.key === "Enter") check(); });
		show.addEventListener("click", () => { msg.style.display = "none"; why.style.display = "block";
			root.querySelector(".q-why-head").textContent = "Answer: " + (+ans.toPrecision(6)) + ".";
			show.style.display = "none"; });
		</script></div>""")
	end

	reveal(body; title = "Answer") = Markdown.MD(Markdown.Admonition("hint", title, Any[body]))

	proof(show::Bool, title, body) = @htl("""<details class="proof" open=$(show)><summary>$(title)</summary>$(body)</details>""")

	_msg(m) = m isa Function ? m() : m

	struct _Failed end

	const _LASTERR = Ref("")

	_errtext(e) = (t = sprint(showerror, e); length(t) > 300 ? first(t, 300) * " …" : t)

	_box(kind, title, body, hint) = @htl("""<div class="admonition $(kind)"><p class="admonition-title">$(title)</p>$(body)$(hint === nothing ? "" : @htl("<div class='admonition hint'><p class='admonition-title'>Hint</p>$(hint)</div>"))</div>""")

	function check(result, good::Bool, ok_msg, bad_msg; hint = md"")
		result === missing && return _box("warning", "No answer yet", md"Complete the code in the cell above; the check runs as soon as the function returns a value.", hint)
		result isa _Failed && return _box("danger", "Not yet: the code stops with an error", @htl("<pre style='white-space:pre-wrap'>$(_LASTERR[])</pre>"), hint)
		good ? adm("correct", "Correct", _msg(ok_msg)) : _box("danger", "Not yet", _msg(bad_msg), hint)
	end

	"""
	    guarded(f)

	Run a check; if the code under test throws while it is being checked, report the error as an answer that is not yet right.
	"""
	function guarded(f)
		_LASTERR[] = ""
		try
			f()
		catch e
			msg = _LASTERR[] == "" ? _errtext(e) : _LASTERR[]
			_box("danger", "Not yet: the code stops with an error", @htl("<pre style='white-space:pre-wrap'>$(msg)</pre>"), nothing)
		end
	end

	_hasmissing(x) = x === missing || (x isa Union{Tuple, AbstractArray, NamedTuple} && any(_hasmissing, x))

	tryrun(f) = try
		r = f()
		_hasmissing(r) ? missing : r
	catch e
		_LASTERR[] = _errtext(e)
		_Failed()
	end

	function _pbkdf2(pw::Vector{UInt8}, salt::Vector{UInt8}, rounds::Int)
		u = hmac_sha256(pw, vcat(salt, UInt8[0, 0, 0, 1])); t = copy(u)
		for _ in 2:rounds
			u = hmac_sha256(pw, u); t = t .⊻ u
		end
		t
	end

	function _keystream(key, n)
		out = UInt8[]; c = 0
		while length(out) < n
			append!(out, sha256(vcat(key, collect(reinterpret(UInt8, [UInt64(c)])))))
			c += 1
		end
		out[1:n]
	end

	const _KEYCACHE = Dict{Tuple{String, Vector{UInt8}}, Vector{UInt8}}()

	function _decrypt(pw::AbstractString, blob::AbstractString)
		raw = base64decode(blob); salt = raw[1:16]; chk = raw[17:48]; ct = raw[49:end]
		key = get!(() -> _pbkdf2(Vector{UInt8}(strip(pw)), salt, 20_000), _KEYCACHE, (String(strip(pw)), salt))
		sha256(vcat(key, Vector{UInt8}("check"))) == chk || return nothing
		String(ct .⊻ _keystream(key, length(ct)))
	end

	"""
	    solution(pw, blob; title = "Solution")

	A solution that opens when the password `pw` typed at the top of the notebook is right.
	"""
	function solution(pw::AbstractString, blob::AbstractString; title = "Solution")
		text = isempty(strip(pw)) ? nothing : _decrypt(pw, blob)
		if text === nothing
			@htl("""<div class="sol-locked">$(title): locked. Enter the password at the top of the notebook to open it.</div>""")
		else
			@htl("""<details class="sol"><summary>$(title)</summary>$(Markdown.parse(text))</details>""")
		end
	end
end;

# ╔═╡ 21441567-4811-fe79-3396-20c1dabd45f8
reveal(md"""No. A single estimate can be close by luck. The quality of an estimator is the spread of its results over many repetitions. Methods are compared by their standard errors, or by repeating each many times, not by one outcome.""")

# ╔═╡ 3cf54315-740c-9dfd-bc4a-920d47cdd849
numq(md"For ``\int_0^1 e^x\,dx`` with uniform samples, ``\sigma_f^2 = 0.2420``. How many samples give a standard error of 0.001?", ((ℯ^2 - 1) / 2 - (ℯ - 1)^2) / 1e-6; rtol = 0.02,
	why = md"``\sigma_f/\sqrt N = 0.001`` requires ``N = 0.2420/10^{-6} \approx 242\,000``.", head = "Check your understanding")

# ╔═╡ 11e3b2b8-c4db-798c-718f-7f0e3109bc7d
mcq(md"A Monte Carlo run with ``N = 10^4`` samples gives a standard error of 0.01. How many samples are needed for a standard error of 0.001?",
	md"``10^5``" => md"The standard error decreases like ``1/\sqrt N``. Ten times more samples reduce it only by ``\sqrt{10} \approx 3.2``.",
	md"``10^6``" => md"Right. A ten times smaller error requires ``10^2 = 100`` times more samples.",
	md"``10^8``" => md"That would reduce the error by a factor 100, to ``10^{-4}``.";
	correct = 2, head = "Pitfall")

# ╔═╡ b898ca97-8643-8393-de4c-c26b1ed0b865
mcq(md"An importance sampler reports an estimate with a very small standard error. Which conclusion is justified?",
	md"The estimate is accurate." => md"Not necessarily. If the proposal is zero, or extremely small, in a region where the integrand is large, that region is never sampled: the estimate is biased, and the standard error, computed from the samples, does not show it (proposal E).",
	md"The estimate scatters little around its own expected value, but it may be biased if the proposal does not cover the support of the integrand." => md"Right. The standard error measures only the scatter of the estimator. Coverage of the support (``q > 0`` wherever ``f \neq 0``) must be ensured by the choice of ``q``, not by the data.",
	md"The proposal is close to optimal." => md"A small standard error can also result from a biased estimator.";
	correct = 2, head = "Pitfall")

# ╔═╡ 78498a7b-acee-30d7-e48b-d4bee929d80d
reveal(md"""The narrow proposal almost never visits the second, unknown source in the south-east of the lake, nor most of the background. In these regions ``q`` is practically zero. Most surveys therefore miss them entirely and underestimate the total; the rare survey that does visit them obtains an enormous weight ``c/q``. The histogram is shifted to low values with a long tail to the right, and the error estimate of each survey is misleadingly small. The defensive uniform share bounds every weight by ``c\,V/\delta`` and removes the problem, at a small cost in efficiency. The general rule: **``q`` must not be much smaller than the integrand anywhere.**""")

# ╔═╡ 207eae68-a0a3-fc86-0351-87e9ae855cf8
solution(pw, "N+yHUnutSJmsKAvjzGoiRt5X0OUzIfCFAdt5cnMeJ/thhXNjaEbWO78Z68Z8TbdaEWlv6o/GqL0howkKXW4O6FldPyZ4l+ISairjHPOydZEvw1vNul+nFXMtfJpM0j/zqzLPszprBAcDUw4G6Rt8j7TxcqJVigZ8peXWE5IgnIS5SHgN8GtqMg3XYxp9jgnIHzUMlWT/cA41WPkSpBGnwYX7/9boOWf9lVzpOD9NgR6TJa9VAJ2ngesmopLAgZsRX6JoS4Jg7cfSb0FRT3QjbTeEW7m4yFHVnHaf4iAvuOx8re/HwSnJmNKBo03uGMvVCuC12vz7J6VKn6Z/6Y2+yZ7kbbx4jTfj5fuNqbr3TuoIiWH9uC5p9cZDB7gPegKqIyab4oT+pfrUqLj3yiqXSbqhwTebsBpteTzrxBYKunKpizZRSaw0m6fk5oj4FrSLspxQHpKEhCNSjp4X/3n1utsTsrPVSkbVHqDPOwD3bYJVKZ6mZ5/+5Ub6D2DzWTcjM8awVBzXZ5/66uvh7S9JvmvkvVDS/0XfEy7VR3Y9LmITltE03w=="; title = "Solution to Problem 1")

# ╔═╡ 1581a3a7-b318-7058-3e50-4f60cfd3016b
solution(pw, "oCJTjTiezw2mqZtQtpLDTUIxko+zNYAvdxPcHF30fMivnvaec+Rwfs0pTEJRcJxQbgpHjgA9qsYpzxi4c+g+FJw1kv0TY5+37eA2yKsxwLW350G4xLnTAHoZVfxqaWguyXSN4yWc5vDLw+sv3q0rVm7OCq3wX/TI3Om4hC0lPvJVPhc3sFrQQUoERHpouy107JdVTW1yAOEa7ht5Lmty44+LZzU0E4neTkYGzVzdmrQT8puJIOGGXvF2UGdp7tmTEgxFAwHAMnJ0FuSArzuR7hmOCCqZNu2U8/fpBd5z/Aecg/tXdB0xCnleNJOLEy6VD4a0IfNvaRCLRgY+Q1MCPg8uYD16vV/35Jel3dbmH49numW4P4ERfh/bnGBz5sYYsoKD/aHdGuAo3VsUCFN6VldOh4hDHCz2LlIvBPNECMvUnhtyEhjOUroo"; title = "Solution to Problem 2")

# ╔═╡ ff7180a4-cc16-ccac-688a-b2a202256575
solution(pw, "tNCbAXS00k6T4QBiCPWOT4bEk1niaeCuyjJHKuH0wc0z1vJKorjAb4P7nfjQQ13jbmatY7qLL4Xevyh2EWKiT8Eup8MV2Aar8T3T9Ay1/mjeDmevGyDwLIyU+8BHpdl7I7ChEAdHpROQFgyIhC11v+Xnqrgjh1IxJVAvOJhIokZqfPDJb9TdzEw9ntFl6FfLX5vs45UaciB5luVrM30b6qYdDyybVPFaI+WYnEoPzc9NlZTqdlSQ3Zcb6eGhCWVe1/9ZqAFbk8pclAefJIFu0NSeSZ1NrepLNfqAoL/XmOc0nZUlXTVxRWJvhD83M6Y/6s70sfScOFjbKXo6GPf8eSGqJoqc8wGUQ6uTqXQP9VsKi/BRgfGjMujLgdgcQKghv+MZACrnG4n1hBNrnsxS55Yyq9Mq2Nalc4kjmxdInLbygLdHK26MQAVqAKA6+i0c+G9y3JX/tqVxjsqwCYxKgu3kvRKMbUXHzwPFUMYxCx0d0mqDVAVPpEU8hKJAf+JLs5jhl2Kh7YOJGF3iHIuSPkIp3ByZysvFJG8uJmn7TRUjtRLxkYyKLaJAuBLofSYT79THUSoOOKbsnIsC9FGfRiGwCp6LGwerhgp6YzXKXZ2ymNlRFQjtCLOmo9hb5w2NYDG7F/oQEhV+omWaeicC8x3RaQwja6w+JZMIaLy3tILu9w=="; title = "Solution to Problem 3")

# ╔═╡ 270f711a-650e-a84a-1446-d1caad08fa51
solution(pw, "rRGBVEwRwMcO/dOAfnQuZEU9sMqIpoeXYpJpJSWPuYElMnFx7fPXrziJyGwKFCrBTu1F+EBOxcgiGUThbmSIUuypp7RJtmJZWLih/vom9RziIsKUPupqwp76l0qAgzRkILl2YMJOJ6GTLbduf5qGnVUAIUuEY2oR6DRlnYTgSWlLy4lkSJHuI/P1fYs7HdTth3xxD2y56Z/z8vGDko88d9ZKM3vmdsLdaMPXj93dmWJMV5rVvlHDlUDNvFy8/K9uteH6Qhsy6Mz1ORyJZUaPzy4UNdykUREM7TL3bBS0OOEDRhy1ObZv53NqarXqXc1gyEEx3S/yLjJ7Lo6xoGAfTKbxiuizlWdGPa7+DaAyWZufV79sh0guKVrFc+9D/vfGLOU7KYCkH7mG/bszeBOKnpZ/vjaOReSqmmACQ5FVYO9AeseCdCT7Yl9i2V16r/6sY/LfFJ0ckhO/o5we2+C/JwNL/7moP/HCo0aiFtQmEOey7rXmErRARfo+4zqkv2Ki7t9qgovqXuoXHQeO"; title = "Solution to Problem 5")

# ╔═╡ 6ec18bfd-9161-399b-8d26-85abde2400a4
solution(pw, "Gfl3ykUWTeBYyaiOoiP2dgwkqffTddwjStF6scWdd1t2gdJQRNyDjGAg64X2+iqhTFTPfw9jJ/EE1aPMhq7WX5BEfV3JiDRyQ5oPAy5duNg3evJCtu3v8U4v/AscmundKVZo2REF8LMvXA6LETenX7aH6GvoQAUwZYRHPdWz25xCSkKF0NM12v+eINZzBfWJoSOtpdm1g6SEB8Q0jJLXI9BSvupJssC5hv/ooaoBe74+gu0CWevXwuh2thmzq+jARIl23DCbQUfaQenDLVmpPDvi9LDzk1lk8BXmUDj9a6ATMhDqWqxeC0Nv1fzILSadGEK1mtRlBGGjwJFyff4jeau67jIAT/09vWmx6MFczMxDtxcs34OO0ZBZ+Ge9rxg94PuRPkxXfsd/nVCMDS4="; title = "Solution to Exercise 1")

# ╔═╡ 095b319f-2f26-701b-0235-58c99f8f4b5a
solution(pw, "e36OY427p85D05zHX/K4OfFh84e+B7QZpVc3+bKqXyXOdc1nZG7q6jRnh3kJVVS2LIZuWF5gu2pwYEZYD16hkWceDzqJQNyzGkCT371diC7tYtsc0Wi+XD79eqJDRbr3vXJj5D8om46N0T/CeVAwYhMDvElMXQVpnpudkrBGwtgU2dNXWaJ51hnxHuLuj0R9ZnXdvZRr+8bFAqXKwSxnwsIrNW/MJZMSLAC5rFMMFPdSZXdUjlZCFqEdasjlHoH0ezxXQqB0Pp13ANg8TsQOccVURIVREWrboObKEdO/ZxCPuD7t67jIXssos4aajTZZqGC9BWiMOCAxIddjZXZH9Rm3+NHRGRHru2B3ZwLsdQ0WtMIxynaSurh22HfxcUuarkI56iwxCJ1i2wcnkMM+djqgew=="; title = "Solution to Exercise 2")

# ╔═╡ a9944d94-57bf-708a-c426-05af0acd2d85
solution(pw, "5xeBpsuHeQXmAkMdAK6VYCn5HOQHgzmyQkkJaEOSVZTL5Df0zj0Jm9D8vfFjllPqmy2Xvnu1Jxs1ZjTQxd7ni5QKHdoNCvwi411wJW3iFjilzuQVw8QJuIEw2FYzdi67T3WXcI8H8IdmUlJQGYEqG8zvZH7Vn/QuA27e5rTL89bEWtcRJbA66uf0Vv29sFvTkMzNM/wQ0R5YtmvpVeosGZ8J7QWfUsLacOFNTOlfqgm4Z43kYd1544KXU9XeAbAV5eKU8Fdab8PZ7A3PN2S4ejDYkyvPNRyb5cpIwHEyvhPk5DYuUJYT8CurUeH3ysZwxi3sn/xr/G9jzB3xd6Ji5DY1YfKAGUWZrLa9NeNujww/aXWyvKJbkXu0xbVoMhTq6Yf0b3cO82VFwF97OGBxh4sLgwFnYOrPFQZVuK+ABq9VLgMqS7hDI05V/qt0ny4G"; title = "Solution to Exercise 3")

# ╔═╡ 7169378f-d01b-0643-46af-4f926f785f66
begin
	function inlake(x, y)
		u = (x - 2.0) / 1.6; v = (y - 1.5) / 1.1
		ρ = hypot(u, v); θ = atan(v, u)
		ρ <= 1 + 0.15cos(3θ + 0.5) + 0.08sin(5θ)
	end

	conc(x, y) = 2.0 + 80exp(-((x - 1.0)^2 + (y - 1.8)^2) / (2 * 0.3^2)) + 40exp(-((x - 3.05)^2 + (y - 0.95)^2) / (2 * 0.12^2))

	const GX = range(0.005, 3.995, length = 400); const GY = range(0.005, 2.995, length = 300); const DA = 0.01 * 0.01

	const MASK = [inlake(x, y) for x in GX, y in GY]

	const LAKE_AREA = count(MASK) * DA

	const TRUE_KG = sum(conc(x, y) for x in GX, y in GY if inlake(x, y)) * DA * 100

	function lake_point(rng)
		while true
			x = 4rand(rng); y = 3rand(rng)
			inlake(x, y) && return (x, y)
		end
	end

	function lake_map(; w = 390, h = 300, title = "", show = true)
		f = fig(w = w, h = h, xr = (0, 4), yr = (0, 3), xlabel = "east (km)", ylabel = "north (km)", title = title, frame = true)
		if show
			for x in 0.05:0.1:3.95, y in 0.05:0.1:2.95
				inlake(x, y) || continue
				c = conc(x, y); a = min(1.0, (c - 2) / 60)
				rect!(f, x - 0.05, y - 0.05, x + 0.05, y + 0.05; fill = COL.red, op = 0.06 + 0.7a, stroke = "none")
			end
		end
		θs = range(0, 2π, length = 241)
		bx = Float64[]; by = Float64[]
		for t in θs
			ρ = 1 + 0.15cos(3t + 0.5) + 0.08sin(5t)
			push!(bx, 2.0 + 1.6ρ * cos(t)); push!(by, 1.5 + 1.1ρ * sin(t))
		end
		line!(f, bx, by; c = COL.blue, w = 2)
		rect!(f, 0.28, 1.78, 0.48, 2.02; fill = COL.ink, op = 1); text!(f, 0.38, 2.15, "factory"; anchor = "middle", size = 10.5, bg = true)
		f
	end

	function figure_1_2(s1_NL, sd04_1)
		rng = MersenneTwister(12 + 1009sd04_1); N = s1_NL
		x = 4 .* rand(rng, N); y = 3 .* rand(rng, N); hit = inlake.(x, y)
		f = lake_map(; show = false, title = "random points on the map")
		dots!(f, x[hit], y[hit]; c = COL.blue, r = N > 1000 ? 1.4 : 2.4, op = 0.8)
		dots!(f, x[.!hit], y[.!hit]; c = COL.grey, r = N > 1000 ? 1.2 : 2, op = 0.4)
		est = 12 * mean(hit); se = 12 * sqrt(mean(hit) * (1 - mean(hit)) / N)
		txt = Markdown.parse("$(count(hit)) of $(N) points lie in the lake, so its area is approximately ``12 \\times $(count(hit))/$(N) = `` **$(fmt(est; d = 3)) km²** (standard error $(fmt(se; d = 2))). The true value is $(fmt(LAKE_AREA; d = 3)) km².")
		figure(f, txt; caption = md"**Figure 1.2.** The same method measures the area of the lake: points are drawn uniformly in the 4 km × 3 km map, and the area of the map is multiplied by the fraction of points that lie on the water.")
	end

	function is_survey(rng, N, cx, cy, wd, δ, Zg)
		tot = 0.0; ws = zeros(N); pts = Vector{Tuple{Float64,Float64}}(undef, N)
		for i in 1:N
			if rand(rng) < δ
				p = lake_point(rng)
			else
				p = (0.0, 0.0)
				while true
					p = (cx + wd * randn(rng), cy + wd * randn(rng))
					inlake(p...) && break
				end
			end
			g = exp(-((p[1] - cx)^2 + (p[2] - cy)^2) / (2wd^2)) / (2π * wd^2) / Zg
			q = δ / LAKE_AREA + (1 - δ) * g
			ws[i] = conc(p...) / q * 100 / 1000; pts[i] = p
		end
		ws, pts
	end

	function figure_5_4(s5_cx, s5_cy, s5_w, s5_d, sd04_5)
		cx, cy, wd, δ = s5_cx, s5_cy, s5_w, s5_d; N = 200
		Zg = sum(exp(-((x - cx)^2 + (y - cy)^2) / (2wd^2)) / (2π * wd^2) for x in GX, y in GY if inlake(x, y)) * DA
		Zg < 1e-6 && return md"*The proposal lies outside the lake. Move its centre onto the water.*"
		rng = MersenneTwister(51 + 1009sd04_5)
		ws, pts = is_survey(rng, N, cx, cy, wd, δ, Zg)
		f = lake_map(; w = 370, title = "proposal q (purple) and one survey", show = true)
		ts = range(0, 2π, length = 121)
		for (k, rr) in enumerate((wd, 2wd))
			pts_ = [(cx + rr * cos(t), cy + rr * sin(t)) for t in ts]
			line!(f, first.(pts_), last.(pts_); c = COL.purple, w = 2, dash = k == 1 ? nothing : "5 3", op = δ < 1 ? 1 : 0)
		end
		δ < 1 && text!(f, cx, cy, "+"; anchor = "middle", size = 16, bold = true, c = COL.purple)
		mw = maximum(ws)
		for (i, p) in enumerate(pts)
			circle!(f, p[1], p[2], 1.4 + 4.5sqrt(ws[i] / mw); fill = COL.ink, stroke = "white", sw = 0.6, op = 0.8)
		end
		R = 300
		uni = [mean(LAKE_AREA * 100 / 1000 * conc(lake_point(r)...) for _ in 1:N) for r in (MersenneTwister(5000 + k) for k in 1:R)]
		isr = [mean(is_survey(MersenneTwister(9000 + k), N, cx, cy, wd, δ, Zg)[1]) for k in 1:R]
		T = TRUE_KG / 1000
		lo = min(quantile(uni, 0.005), quantile(isr, 0.005)); hi = max(quantile(uni, 0.995), quantile(isr, 0.995))
		edges = collect(range(lo, hi, length = 36))
		hu = histdata(uni, edges); hi_ = histdata(isr, edges)
		g = fig(w = 290, h = 300, xr = (lo, hi), yr = (0, 1.7max(maximum(hu), maximum(hi_))), xlabel = "estimated total (tonnes)", ylabel = "density", title = "300 surveys each")
		hist!(g, edges, hu; c = COL.grey, op = 0.45); hist!(g, edges, hi_; c = COL.purple, op = 0.55)
		seg!(g, T, 0, T, 1.15max(maximum(hu), maximum(hi_)); c = COL.ink, w = 1.6)   # stops below the legend
		legend!(g, [("uniform flights", COL.grey, :bar), ("this flight plan", COL.purple, :bar), ("true total", COL.ink)]; at = :tl, size = 10.5)
		su, si = std(uni), std(isr)
		share = maximum(ws) / sum(ws)
		verdict = si < su ? "The plan is **$(fmt(su^2 / si^2; d = 3)) times** as efficient as uniform flights: it requires that many times fewer flights for the same accuracy." :
			"The plan is **less** efficient than uniform flights: its estimates scatter $(fmt(si / su; d = 3)) times as widely."
		bias = abs(mean(isr) - T) > 3si / sqrt(R) ? " Its estimates are also **not centred** at the true value: part of the lake is almost never visited, and when it is, the weight is very large." : ""
		figure(f, g; caption = md"**Figure 5.2.** Importance sampling for the drone. Left: the true concentration field (red), the normal part of the proposal ``q`` and the 200 flights of one survey (black dots; the larger the dot, the larger its weight ``c/q``). Right: the distribution of the estimate over 300 complete surveys, for this flight plan (purple) and for uniform flights (grey).")
	end

	function figure_6_1()
		σu = std(LAKE_AREA * 100 / 1000 * conc(lake_point(r)...) for r in (MersenneTwister(70000 + k) for k in 1:20000))
		cx, cy, wd, δ = 1.0, 1.8, 0.45, 0.3
		Zg = sum(exp(-((x - cx)^2 + (y - cy)^2) / (2wd^2)) / (2π * wd^2) for x in GX, y in GY if inlake(x, y)) * DA
		ws, _ = is_survey(MersenneTwister(71), 20000, cx, cy, wd, δ, Zg); σi = std(ws)
		T = TRUE_KG / 1000; target = 0.05T
		Nu = ceil(Int, (σu / target)^2); Ni = ceil(Int, (σi / target)^2)
		vstack(tbl(["", "uniform flights", "proposal around the outlet (30 % uniform)"], [
			["standard deviation σ of the estimate from one flight", "$(fmt(σu; d = 3)) t", "$(fmt(σi; d = 3)) t"],
			["flights needed for ± 10 % (two standard errors)", string(Nu), string(Ni)]]);
			caption = md"**Table 6.1.** Comparison of the two flight plans, estimated from 20 000 simulated flights each. The number of flights is ``N = (2\sigma / (0.1 M))^2``.")
	end

	function answer(guess_n)
		g = guess_n > 0 ? "Your estimate: $(guess_n) flights. " : ""
		adm("key", "Result", Markdown.parse(g * "The lake bed holds about **$(fmt(TRUE_KG / 1000; d = 3)) tonnes** of pollutant. About **1100 uniformly distributed flights** are required for ± 10 %. A plan that concentrates the flights around the outlet, with 30 % uniform flights, requires about **four times fewer**."))
	end
end;

# ╔═╡ 4b962447-df83-f6f0-76ca-782cb800d432
figure_6_1()

# ╔═╡ f9c61d30-0d56-9bf1-b85a-e806ef6cb254
answer(guess_n)

# ╔═╡ 82b77ce8-fd4b-2902-d6b8-4758c4956eda
begin
	function figure_1_1(s1_new, s1_N)
		let
			rng = MersenneTwister(1777 + 11s1_new); N = s1_N
			x = rand(rng, N); y = rand(rng, N); inside = x .^ 2 .+ y .^ 2 .<= 1
			est = 4 .* cumsum(inside) ./ (1:N)
			f = fig(w = 330, h = 340, xr = (0, 1), yr = (0, 1), frame = true, xticks = [0, 0.5, 1], yticks = [0, 0.5, 1])
			θ = range(0, π / 2, length = 100)
			area!(f, cos.(θ), sin.(θ); c = COL.blue, op = 0.08)
			line!(f, cos.(θ), sin.(θ); c = COL.ink, w = 1.6)
			dots!(f, x, y; c = [i ? COL.blue : COL.red for i in inside], r = 2.4, op = 0.85, s = collect(1:N))
			g = fig(w = 340, h = 340, xr = (1, N), yr = (2.6, 3.8), xlog = true, xlabel = "number of points N", ylabel = "estimate of π", xticks = [t for t in [1, 10, 100, 1000] if t <= N])
			hline!(g, π; c = COL.ink, dash = nothing)
			text!(g, N, π - 0.06, "π = 3.14159…"; anchor = "end", size = 11.5, bg = true)
			for i in 2:N
				seg!(g, i - 1, est[i-1], i, est[i]; c = COL.blue, w = 1.6, s = i)
			end
			dots!(g, collect(1:N), est; c = COL.blue, r = 4, o = collect(1:N))
			reads = ["Points: <b>$(k)</b>" * (k == 0 ? "" : " · inside: <b>$(count(inside[1:k]))</b> · estimate 4 × $(count(inside[1:k]))/$(k) = <b>$(fmt(est[k]; d = 4))</b>") for k in 0:N]
			note = Markdown.parse("After $(N) points: $(count(inside)) inside, estimate **$(fmt(est[N]; d = 4))**, error $(fmt(abs(est[N] - π); d = 2)).")
			player(vstack(figure(f, g), note); n = N, start = 0, fps = 40, reads = reads, label = "point",
				caption = md"**Figure 1.1.** Random points in the unit square; blue points fall inside the quarter circle. Right: the running estimate of π (logarithmic axis).")
		end
	end

	function figure_2_1(s2_new, s2_src, s2_eps)
		let
			rng = MersenneTwister(2 + 17s2_new); R = 50; N = 2000
			draw, μ, σ = Dict("a fair die (μ = 3.5)" => ((m) -> rand(rng, 1:6, m), 3.5, sqrt(35 / 12)), "a coin, 1 for heads (μ = 0.5)" => ((m) -> Float64.(rand(rng, Bool, m)), 0.5, 0.5),
				"an exponential waiting time (μ = 1)" => ((m) -> randexp(rng, m), 1.0, 1.0))[s2_src]
			runs = [cumsum(draw(N)) ./ (1:N) for _ in 1:R]
			idx = unique(round.(Int, 10 .^ range(0, log10(N), length = 160)))
			f = fig(w = 450, h = 320, xr = (1, N), yr = (μ - 2.2σ, μ + 2.2σ), xlog = true, xlabel = "number of samples N", ylabel = "running average")
			band!(f, [1, N], [μ - s2_eps, μ - s2_eps], [μ + s2_eps, μ + s2_eps]; c = COL.green, op = 0.22)
			hline!(f, μ - s2_eps; c = COL.green, w = 1.4); hline!(f, μ + s2_eps; c = COL.green, w = 1.4)
			for r in runs
				line!(f, idx, clamp.(r[idx], μ - 2.2σ, μ + 2.2σ); c = COL.blue, w = 0.8, op = 0.45)
			end
			hline!(f, μ; c = COL.ink, dash = nothing)
			text!(f, N, μ + s2_eps, "μ + ε"; anchor = "end", dy = -9, size = 11, c = COL.green, bold = true, bg = true)
			text!(f, N, μ - s2_eps, "μ − ε"; anchor = "end", dy = 10, size = 11, c = COL.green, bold = true, bg = true)
			Ns = idx
			frac = [count(r -> abs(r[n] - μ) > s2_eps, runs) / R for n in Ns]
			cheb = [min(1.0, σ^2 / (n * s2_eps^2)) for n in Ns]
			g = fig(w = 230, h = 320, xr = (1, N), yr = (0, 1), xlog = true, xlabel = "N", ylabel = "fraction of runs outside", l = 60)
			line!(g, Ns, frac; c = COL.red, w = 2)
			line!(g, Ns, cheb; c = COL.grey, w = 1.4, dash = "4 3")
			nin = something(findfirst(n -> all(r -> all(abs.(r[n:end] .- μ) .<= s2_eps), runs), 1:N), N + 1)
			figure(f, g; caption = md"**Figure 2.1.** Left: 50 independent runs; each blue curve is the running average ``\bar X_N`` of one run, the black line is ``\mu``, the green band is ``\mu \pm \varepsilon``. Right: the fraction of runs outside the band at each ``N`` (red), an estimate of ``P(|\bar X_N - \mu| > \varepsilon)``.")
		end
	end

	function _check_estimate_pi(estimate_pi)
		let
			r = tryrun(() -> estimate_pi(10, MersenneTwister(1)))
			ok = r !== missing && let
				es = [estimate_pi(10_000, MersenneTwister(k)) for k in 1:200]
				abs(mean(es) - π) < 0.01 && 0.012 < std(es) < 0.02
			end
			check(r, ok, () -> md"200 runs with ``N = 10^4``: mean $(fmt(mean(estimate_pi(10_000, MersenneTwister(k)) for k in 1:200); d = 5)), spread between runs $(fmt(std([estimate_pi(10_000, MersenneTwister(k)) for k in 1:200]))).",
				md"Your estimates are not centred at π with the expected spread of about 0.016.";
				hint = md"Plot a few of the points the code generates. Do they fill the square?")
		end
	end

	function compute_s3_fun(s3_f)
		try
			g = eval(Meta.parse("x -> " * s3_f))
			Base.invokelatest(g, 0.3) isa Real ? g : nothing
		catch
			nothing
		end
	end

	function _check_mc_integrate(mc_integrate)
		let
			r = tryrun(() -> mc_integrate(exp, 0, 1, 100))
			ok = r !== missing && hasproperty(r, :estimate) && let
				r1 = mc_integrate(exp, 0, 1, 200_000)
				abs(r1.estimate - (ℯ - 1)) < 5r1.se && isapprox(r1.se, sqrt(0.2420 / 2e5); rtol = 0.05)
			end
			check(r, ok, () -> md"``\int_0^1 e^x dx`` with ``N = 2 \times 10^5``: $(fmt(mc_integrate(exp, 0, 1, 200_000).estimate; d = 5)) (exact 1.71828), standard error about 0.0011.",
				md"Expected a named tuple with the estimate of ``e - 1`` and a standard error near 0.0011 for ``N = 2 \times 10^5``.";
				hint = md"The error of Monte Carlo varies like ``1/\sqrt N``")
		end
	end

	function _check_is_x2(is_x2)
		let
			r = tryrun(() -> is_x2(100))
			ok = r !== missing && hasproperty(r, :se) && let
				r1 = is_x2(200_000)
				abs(r1.estimate - 1 / 3) < 5r1.se && isapprox(r1.se, sqrt(1 / 72 / 2e5); rtol = 0.05)
			end
			check(r, ok, md"Estimate close to ``1/3`` with standard error ``\sqrt{1/72}/\sqrt{N}``.",
				md"Expected a named tuple with estimate ≈ 1/3 and standard error ≈ ``\sqrt{1/72}/\sqrt{N}``.";
				hint = md"Which density do the samples follow, and which density do the weights assume?")
		end
	end

	function figure_3_1(s3_fun, s3_b, s3_a, s3_new)
		(s3_fun === nothing || s3_b <= s3_a) && return md"*The integrand could not be read, or b ≤ a.*"
		f(x) = Float64(Base.invokelatest(s3_fun, x))
		a, b = Float64(s3_a), Float64(s3_b); N = 100
		rng = MersenneTwister(30 + 1009s3_new)
		x = a .+ (b - a) .* rand(rng, N); fx = f.(x)
		xs = range(a, b, length = 400); fs = f.(xs)
		ref = (b - a) * mean(f.(range(a, b, length = 200_001)))
		lo = min(0, minimum(fs)); hi = max(maximum(fs), 0) * 1.18 + 1e-9
		runm = cumsum(fx) ./ (1:N)
		F = fig(w = 330, h = 290, xr = (a, b), yr = (lo, hi), xlabel = "x", ylabel = "f(x)", title = "area under f and the rectangle")
		area!(F, xs, fs; c = COL.blue, op = 0.15); line!(F, xs, fs; c = COL.blue, w = 2.2)
		for i in 1:N
			seg!(F, x[i], 0, x[i], fx[i]; c = COL.red, w = 1, op = 0.45, s = i)
			dots!(F, [x[i]], [fx[i]]; c = COL.red, r = 3, s = i)
			rect!(F, a, 0, b, runm[i]; fill = COL.ochre, op = 0.22, stroke = COL.ochre, sw = 1.5, o = i)
		end
		G = fig(w = 330, h = 290, xr = (0, N), yr = (min(0, (b - a) * minimum(runm), ref) - 0.05abs(ref), max((b - a) * maximum(runm), ref) * 1.15 + 1e-9), xlabel = "number of samples N", ylabel = "estimate θ_N", title = "running estimate")
		hline!(G, ref; c = COL.ink, w = 1.4)
		for i in 2:N
			seg!(G, i - 1, (b - a) * runm[i-1], i, (b - a) * runm[i]; c = COL.ochre, w = 2, s = i)
		end
		legend!(G, [("θ_N = (b − a) × mean height", COL.ochre), ("accurate value", COL.ink)]; at = :br, size = 10.5)
		reads = ["Samples: <b>$(k)</b>" * (k == 0 ? "" : " · newest point x = $(fmt(x[k]; d = 3)), f(x) = $(fmt(fx[k]; d = 3)) · mean height $(fmt(runm[k]; d = 4)) · rectangle area θ = $(fmt(b - a; d = 3)) × $(fmt(runm[k]; d = 4)) = <b>$(fmt((b - a) * runm[k]; d = 4))</b>") for k in 0:N]
		txt = Markdown.parse("After $(N) samples: ``\\theta_N`` = **$(fmt((b - a) * runm[N]; d = 5))**, standard error $(fmt((b - a) * std(fx) / sqrt(N); d = 2)); accurate value $(fmt(ref; d = 6)).")
		player(vstack(figure(F, G), txt); n = N, start = 0, fps = 6, reads = reads, label = "sample",
			caption = md"**Figure 3.1.** Monte Carlo integration. Each sample ``x_i`` gives a height ``f(x_i)`` (red). The orange rectangle has the width ``b - a`` and the mean of the heights sampled so far as its height; its area is the estimate ``\theta_N``. Right: the estimate as a function of the number of samples.")
	end

	function figure_3_3(s3_fun, s3_b, s3_a, s3_N, s3_f)
		ok = s3_fun !== nothing && s3_b > s3_a
		f(x) = ok ? Float64(Base.invokelatest(s3_fun, x)) : exp(-x^2)
		a, b = ok ? (Float64(s3_a), Float64(s3_b)) : (0.0, 2.0)
		N = s3_N; R = 2000
		ref = (b - a) * mean(f.(range(a, b, length = 400_001)))
		σf = std(f.(range(a, b, length = 100_001))); se = (b - a) * σf / sqrt(N)
		ests = zeros(R); ses = zeros(R)
		for k in 1:R
			v = f.(a .+ (b - a) .* rand(MersenneTwister(1000 + k + 7919N), N))
			ests[k] = (b - a) * mean(v); ses[k] = (b - a) * std(v) / sqrt(N)
		end
		cover = mean(abs.(ests .- ref) .<= 2 .* ses)
		sg = max(se, 1e-9); edges = collect(range(ref - 4.5sg, ref + 4.5sg, length = 37))
		hd = histdata(ests, edges)
		F = fig(w = 330, h = 280, xr = (edges[1], edges[end]), yr = (0, 1.3max(maximum(hd), pdf(Normal(ref, sg), ref))), xlabel = "estimate θ_N", ylabel = "density", title = "$(R) runs with N = $(N)")
		rect!(F, ref - 2sg, 0, ref + 2sg, 1.3max(maximum(hd), pdf(Normal(ref, sg), ref)); fill = COL.green, op = 0.08)
		hist!(F, edges, hd; c = COL.purple, op = 0.55)
		zz = range(edges[1], edges[end], length = 200); line!(F, zz, pdf.(Normal(ref, sg), zz); c = COL.ink, w = 1.8)
		vline!(F, ref; c = COL.ink, dash = nothing, w = 1.2)
		Ns = round.(Int, 10 .^ range(1, 4.5, length = 12))
		rms = [sqrt(mean(((b - a) * mean(f.(a .+ (b - a) .* rand(MersenneTwister(500M + k), M))) - ref)^2 for k in 1:80)) for M in Ns]
		th = (b - a) * σf ./ sqrt.(Ns)
		ylo = minimum(vcat(rms, th)) / 2; yhi = maximum(vcat(rms, th)) * 2
		G = fig(w = 330, h = 280, xr = (10, Ns[end]), yr = (ylo, yhi), xlog = true, ylog = true, xlabel = "number of samples N", ylabel = "root-mean-square error", title = "error against N")
		line!(G, Ns, th; c = COL.ink, w = 1.6, dash = "5 3"); dots!(G, Ns, rms; c = COL.red, r = 4)
		legend!(G, [("80 runs per N", COL.red, :dot), ("(b − a)σ_f/√N", COL.ink, :dash)]; at = :tr, size = 10.5)
		figure(F, G; caption = md"**Figure 3.2.** Left: the distribution of the Monte Carlo estimate over many independent runs (purple), the normal approximation with standard deviation ``(b-a)\sigma_f/\sqrt N`` (black curve), and the band of ``\pm 2`` standard errors around the accurate value (green). Right: the root-mean-square error as a function of ``N`` on logarithmic axes.")
	end

	function figure_4_1(s4_y)
		y0 = s4_y; xs = range(0, 4, length = 401)
		f = lake_map(; w = 360, title = "concentration c(x, y): darker = higher")
		wet = [inlake(x, y0) for x in xs]
		i1 = findfirst(wet); i2 = findlast(wet)
		seg!(f, xs[i1], y0, xs[i2], y0; c = COL.ink, w = 2.4, dash = "6 3")
		text!(f, xs[i2], y0, "slice"; dx = 6, size = 11.5, bold = true, bg = true)
		cv = [w ? conc(x, y0) : 0.0 for (x, w) in zip(xs, wet)]
		G = fig(w = 300, h = 300, xr = (0, 4), yr = (0, 90), xlabel = "east x (km)", ylabel = "concentration (kg/ha)", title = "c along the slice at north = $(fmt(y0; d = 2)) km")
		area!(G, xs, cv; c = COL.red, op = 0.2); line!(G, xs, cv; c = COL.red, w = 2)
		sl = sum(cv) * (xs[2] - xs[1])
		txt = Markdown.parse("Area under the slice: ``\\int c(x, $(fmt(y0; d = 2)))\\,dx`` = $(fmt(sl; d = 3)) kg/ha · km. Adding up the areas of all slices, each multiplied by its thickness ``dy``, gives the volume under the surface: the total mass ``M`` = $(fmt(TRUE_KG / 1000; d = 3)) t.")
		figure(f, G, txt; caption = md"**Figure 4.1.** The integrand of the guiding problem. Left: the concentration on the lake bed, with the factory outlet in the west and a second, unknown source in the south-east; the dashed line is one west–east slice. Right: the concentration along this slice; the shaded area is the integral along the slice. The total mass is the volume under the surface ``c(x, y)``, that is, the sum of all slice areas times their thickness.")
	end

	function figure_4_2(s4_N, sd04_4)
		rng = MersenneTwister(41 + 1009sd04_4); N = s4_N
		pts = [lake_point(rng) for _ in 1:N]; c = [conc(p...) for p in pts]
		vals = LAKE_AREA * 100 .* c ./ 1000
		f = lake_map(; w = 370, title = "red: high, orange: medium, white: low")
		for (i, p) in enumerate(pts)
			circle!(f, p[1], p[2], 3.2; fill = c[i] > 20 ? COL.red : c[i] > 6 ? COL.ochre : "#ffffff", stroke = COL.ink, sw = 0.8, s = i)
		end
		runm = cumsum(vals) ./ (1:N)
		runse = [k < 2 ? NaN : std(vals[1:k]) / sqrt(k) for k in 1:N]
		ymax = max(TRUE_KG / 1000 * 3, maximum(runm) * 1.1)
		g = fig(w = 290, h = 300, xr = (0, N), yr = (0, ymax), xlabel = "number of flights N", ylabel = "estimated total (tonnes)", title = "running estimate θ_N")
		ks = collect(2:N)
		band!(g, ks, max.(runm[2:end] .- 2runse[2:end], 0), runm[2:end] .+ 2runse[2:end]; c = COL.blue, op = 0.15, s = N)
		hline!(g, TRUE_KG / 1000; c = COL.ink, dash = "5 3", w = 1.5)
		for k in 2:N
			seg!(g, k - 1, runm[k-1], k, runm[k]; c = COL.blue, w = 1.8, s = k)
		end
		legend!(g, [("θ_N ± 2 SE", COL.blue), ("true total", COL.ink, :dash)]; at = :tr, size = 10.5)
		reads = vcat(["No flight yet."], ["Flight $(k): measured c = $(fmt(c[k]; d = 3)) kg/ha, so V·c = $(fmt(vals[k]; d = 3)) t · running mean θ = <b>$(fmt(runm[k]; d = 3)) t</b>" * (k > 1 ? " ± $(fmt(2runse[k]; d = 2))" : "") for k in 1:N])
		txt = Markdown.parse("After $(N) flights: **$(fmt(runm[end]; d = 3)) ± $(fmt(2runse[end]; d = 2)) tonnes** (± two standard errors). The true total is $(fmt(TRUE_KG / 1000; d = 3)) t.")
		player(f, g, txt; n = N, start = 0, fps = 12, reads = reads, label = "flight",
			caption = md"**Figure 4.2.** Plain Monte Carlo for the drone. Left: the flights, coloured by the measured concentration, over the true concentration field (red shading, which the drone does not know). Right: the running estimate ``\theta_N``, the mean of the values ``V c(\mathbf x_i)``, with a band of ± two standard errors; the dashed line is the true total.")
	end

	const DIM_I1 = sqrt(π) / 2 * 0.8427007929497149        # ∫₀¹ exp(-x²) dx

	const DIM_I2 = sqrt(π / 8) * 0.9544997361036416        # ∫₀¹ exp(-2x²) dx

	_mid1(n) = sum(exp(-((k - 0.5) / n)^2) for k in 1:n) / n

	function figure_4_3(s4_m)
		m = s4_m
		rs = sqrt((DIM_I2 / DIM_I1^2)^m - 1)          # relative standard deviation of f(X)
		Nmax = 1e7
		ns = [n for n in 1:4000 if float(n)^m <= Nmax]
		gN = [float(n)^m for n in ns]; gE = [abs((_mid1(n) / DIM_I1)^m - 1) for n in ns]
		keep = gE .> 1e-9; gN = gN[keep]; gE = gE[keep]
		Ms = [10, 100, 1000, 10_000, 100_000]
		rng = MersenneTwister(4 + m)
		mcE = [sqrt(mean(((mean(prod(exp(-rand(rng)^2) for _ in 1:m) for _ in 1:M) / DIM_I1^m) - 1)^2 for _ in 1:20)) for M in Ms]
		F = fig(w = 640, h = 330, xr = (1, Nmax), yr = (1e-9, 3), xlog = true, ylog = true, xlabel = "number of function evaluations N", ylabel = "relative error", t = 34)
		NN = 10 .^ range(0, 7, length = 50)
		line!(F, NN, rs ./ sqrt.(NN); c = COL.blue, w = 2, dash = "5 3")
		dots!(F, Ms, mcE; c = COL.blue, r = 4.5)
		sel = unique(vcat([findlast(gN .<= t) for t in 10 .^ range(0, 7, length = 22) if any(gN .<= t)], length(gN)))
		line!(F, gN, gE; c = COL.red, w = 2); dots!(F, gN[sel], gE[sel]; c = COL.red, r = 3)
		text!(F, 1.3, 1e-8, "dimension m = $(m)"; size = 13, bold = true)
		legend!(F, [("grid (midpoint rule), N = n^m", COL.red), ("Monte Carlo, 20 runs per N", COL.blue, :dot), ("Monte Carlo theory σ_f/(F√N)", COL.blue, :dash)]; at = :top, size = 11)
		figure(F; caption = md"**Figure 4.3.** Grid against random points for the integral of ``f(\mathbf x) = e^{-|\mathbf x|^2}`` over the unit cube ``[0, 1]^m``. Red: the relative error of the midpoint rule on a grid with ``n`` points per axis. Blue: the relative error of plain Monte Carlo with ``N`` uniform points.")
	end
end;

# ╔═╡ 2b2d73b1-ad81-af4c-2da4-901ccd1028e0
figure_4_1(s4_y)

# ╔═╡ 7f5c299a-f91c-082d-e475-69a2344bcdf3
figure_4_3(s4_m)

# ╔═╡ bcc26dda-0a7f-524d-92fe-5315356d99c9
begin
	IS_F(x) = exp(-(x - 2)^2 / (2 * 0.3^2))

	const IS_TRUE = 0.3sqrt(2π)

	function is_prop(case)
		c = case[1]
		c == 'A' ? Uniform(0, 4) : c == 'B' ? Normal(2, 0.4) : c == 'C' ? Normal(2, 0.15) : c == 'D' ? Normal(3, 0.4) : Uniform(2, 4)
	end

	function is_run(q, N, rng)
		x = rand(rng, q, N); w = [pdf(q, xi) > 0 ? IS_F(xi) / pdf(q, xi) : 0.0 for xi in x]
		x, w
	end

	function figure_is(s5_case, s5_N, sd04_6)
		q = is_prop(s5_case); N = s5_N
		rng = MersenneTwister(60 + 1009sd04_6)
		x, w = is_run(q, N, rng)
		xs = range(-0.5, 4.5, length = 400)
		ft = IS_F.(xs) ./ IS_TRUE; qt = pdf.(q, xs)
		ymax = 1.15max(maximum(ft), maximum(qt))
		F = fig(w = 220, h = 270, xr = (-0.5, 4.5), yr = (0, ymax), xlabel = "x", ylabel = "density", title = "integrand and q", l = 50)
		area!(F, xs, ft; c = COL.blue, op = 0.15); line!(F, xs, ft; c = COL.blue, w = 2)
		line!(F, xs, qt; c = COL.red, w = 2)
		for xi in x[1:min(N, 200)]; seg!(F, xi, 0, xi, 0.04ymax; c = COL.red, w = 1, op = 0.6); end
		legend!(F, [("f / F", COL.blue), ("q", COL.red)]; at = :tl, size = 10.5)
		wmax = max(maximum(w), 1e-9)
		G = fig(w = 220, h = 270, xr = (-0.5, 4.5), yr = (0, 1.15wmax), xlabel = "x", ylabel = "weight f(x)/q(x)", title = "weights of one run", l = 50)
		dots!(G, x, w; c = COL.purple, r = 2.2, op = 0.7)
		hline!(G, IS_TRUE; c = COL.ink, dash = "4 3", w = 1)
		R = 1000
		ests = [mean(is_run(q, N, MersenneTwister(7000 + k + 13N))[2]) for k in 1:R]
		lo = min(quantile(ests, 0.005), IS_TRUE * 0.4); hi = max(quantile(ests, 0.995), IS_TRUE * 1.4)
		edges = collect(range(lo, hi, length = 34)); hd = histdata(ests, edges)
		H = fig(w = 220, h = 270, xr = (lo, hi), yr = (0, 1.2maximum(hd)), xlabel = "estimate", ylabel = "density", title = "$(R) runs", l = 46)
		hist!(H, edges, hd; c = COL.purple, op = 0.55); vline!(H, IS_TRUE; c = COL.ink, dash = nothing, w = 1.6)
		refsd = std([mean(is_run(Uniform(0, 4), N, MersenneTwister(9000 + k + 13N))[2]) for k in 1:R])
		eff = (refsd / std(ests))^2
		share = maximum(w) / max(sum(w), 1e-300)

		figure(F, G, H; caption = md"**Figure 5.1.** Importance sampling of ``\int f(x)\,dx`` for a narrow bump. Left: the shape of the integrand (normalised to area 1, blue) and the proposal density ``q`` (red). Middle: the weights ``f(x_i)/q(x_i)`` of these samples; the dashed line is the exact integral. Right: the distribution of the estimate over many independent runs; the black line is the exact value.")
	end

	check_estimate_pi(estimate_pi) = guarded(() -> _check_estimate_pi(estimate_pi))

	check_mc_integrate(mc_integrate) = guarded(() -> _check_mc_integrate(mc_integrate))

	check_is_x2(is_x2) = guarded(() -> _check_is_x2(is_x2))
end;

# ╔═╡ 32132fb7-2b55-24bb-9843-f1a14a8ce450
check_estimate_pi(estimate_pi)

# ╔═╡ 3794a3eb-1caa-2ced-7bd6-a8c73fea0c6b
check_mc_integrate(mc_integrate)

# ╔═╡ 06555cb2-1486-e734-1af9-bab1177872e8
check_is_x2(is_x2)

# ╔═╡ f3a8c8d4-48f5-39ca-f47c-662c638125e7
# Controls
begin
	Seed(default::Integer = 0; label = "New random numbers") = CounterButton(label)
end;

# ╔═╡ 6157a2db-7044-1b80-b015-f415474e87bd
md"""
## 1. The idea: estimating ``\pi`` with random points

**Question.** How can random numbers compute a number that is not random, such as ``\pi``?

**Idea.** Draw a square of side 1 and, inside it, a quarter of the unit circle (Figure 1.1, left). If points are thrown at the square completely at random, the share of points that land inside the quarter circle is close to the share of the square's **area** that the quarter circle covers. That share is known in terms of ``\pi``, so counting points gives ``\pi``.

Draw ``N`` independent points ``(u_1^{(i)}, u_2^{(i)})`` with ``u_1^{(i)}, u_2^{(i)} \sim \mathcal U(0, 1)``, ``i = 1, \dots, N``. A point lies inside the quarter circle if ``(u_1^{(i)})^2 + (u_2^{(i)})^2 \le 1``. The probability of this event is the ratio of the areas,

```math
P(\text{inside}) = \frac{\text{area of the quarter circle}}{\text{area of the square}} = \frac{\pi r^2 / 4}{r^2} = \frac{\pi}{4} \qquad (r = 1).
```

Hence

```math
\pi \approx 4\,\frac{\text{number of points inside the quarter circle}}{N} .
```

Figure 1.1 draws the points one by one. Press ▶ and watch the right panel: the estimate jumps up for every blue point and down for every red point. Then increase ``N`` or draw new points, and compare the final estimate with ``\pi``.

Points $(@bind s1_N Select([100, 1000, 3000]; default = 1000)) ``\qquad`` $(@bind s1_new Seed(0; label = "New points"))
"""

# ╔═╡ e3bb564d-06e0-adb8-3f31-ff72db83a251
figure_1_1(s1_new, s1_N)

# ╔═╡ 79740486-fc58-382f-af93-a98c35227077
md"""
The estimate approaches ``\pi``, but irregularly, and a new set of points gives a different value.

The argument does not depend on the shape of the circle. Any region for which one can decide whether a point lies inside can be measured in the same way. This gives the first piece of the guiding problem, the **area ``V`` of the lake**: draw points uniformly on the 4 km × 3 km map (area 12 km²) and multiply 12 km² by the fraction of points that lie on the water.

Points $(@bind s1_NL Select([100, 1000, 5000]; default = 1000)) ``\qquad`` $(@bind sd04_1 Seed(0))
"""

# ╔═╡ ce5d7f25-850f-0fce-db52-5eefa14a9001
figure_1_2(s1_NL, sd04_1)

# ╔═╡ b1472a85-0568-1b23-7597-31d17058d0bd
md"""
!!! info "Key points"
    - A Monte Carlo estimate is an **average of random quantities**. It is itself random: a repetition with other random numbers gives a slightly different value.
    - The estimate approaches the exact value as ``N`` grows (law of large numbers (Section 2)).

## 2. The law of large numbers

**Question.** Why does the average of random samples approach the exact value?

**Setting.** Let ``X_1, X_2, \dots, X_N`` be independent and identically distributed (i.i.d.) random variables with finite expected value ``E[X_i] = \mu``, and let

```math
\bar X_N = \frac1N \sum_{i=1}^N X_i
```

be their average. The law of large numbers states that ``\bar X_N \to \mu`` as ``N \to \infty``. 

!!! theorem "Laws of large numbers"
    **Weak law:** for every ``\varepsilon > 0``,
    ```math
    \lim_{N \to \infty} P\big(|\bar X_N - \mu| > \varepsilon\big) = 0 .
    ```
    **Strong law:**
    ```math
    P\Big(\lim_{N \to \infty} \bar X_N = \mu\Big) = 1 .
    ```

**Weak law.** Fix a tolerance ``\varepsilon``. The probability that the average misses ``\mu`` by more than ``\varepsilon`` becomes arbitrarily small for large ``N`` (*convergence in probability*). 

**Strong law.** With probability one, the sequence ``\bar X_1, \bar X_2, \dots`` of a single run converges to ``\mu``: after some (random) ``N``, it stays inside the band (*almost sure convergence*). 


Half-width of the band ``\varepsilon`` $(@bind s2_eps Slider(0.05:0.05:0.5; default = 0.2, show_value = true)) ``\qquad`` samples from $(@bind s2_src Select(["a fair die (μ = 3.5)", "a coin, 1 for heads (μ = 0.5)", "an exponential waiting time (μ = 1)"]; default = "a fair die (μ = 3.5)")) ``\qquad`` $(@bind s2_new Seed(0; label = "New runs"))
"""

# ╔═╡ 23e1102e-5eee-3e42-5940-37b196c82fba
figure_2_1(s2_new, s2_src, s2_eps)

# ╔═╡ eef6abd0-c772-7bc3-2c98-f32a780d4385
md"""
## 3. Monte Carlo integration

### 3.1 An integral as an expected value

**Question.** The law of large numbers computes expected values by averaging. How can an **integral** be computed in this way?

**Idea.** Rewrite the integral as the expected value of some random variable; then average samples of it.

**Derivation.** Consider ``F = \int_a^b f(x)\,dx``. Multiplying the integrand by ``1 = (b - a)/(b - a)`` does not change the integral:

```math
F = \int_a^b f(x)\,dx = (b - a)\int_a^b f(x)\,\frac{1}{b - a}\,dx = (b - a)\int_a^b f(x)\,p(x)\,dx = (b - a)\,E[f(X)] .
```

Here ``p(x) = 1/(b - a)`` for ``x \in (a, b)`` is the density of the uniform distribution, so ``X \sim \mathcal U(a, b)``, and the last integral is the definition of ``E[f(X)]``. Replacing the expected value by a sample average gives the **Monte Carlo estimator**

```math
\theta_N = \frac{b - a}{N}\sum_{i=1}^{N} f(x_i), \qquad x_i \sim \mathcal U(a, b) \text{ i.i.d.}, \quad i = 1, \dots, N .
```

The values ``f(x_1), \dots, f(x_N)`` are i.i.d. with finite mean, so by the law of large numbers ``\theta_N \to F`` as ``N \to \infty``.

**Geometric meaning.** The integral is the area under the curve. ``\frac1N \sum f(x_i)`` is the **mean height** of the curve at the sampled points, and ``\theta_N`` is the area of a **rectangle** with width ``b - a`` and this mean height.

Figure 3.1 adds one sample at a time. Press ▶ and watch how each new red stick changes the height of the orange rectangle (left) and the estimate (right). The integrand and the interval can be changed (Julia syntax).

``f(x) = `` $(@bind s3_f TextField(30; default = "exp(-x^2)")) ``\quad a = `` $(@bind s3_a NumberField(-10:0.5:10; default = 0)) ``\quad b = `` $(@bind s3_b NumberField(-10:0.5:10; default = 2)) ``\quad`` $(@bind s3_new Seed(0; label = "New samples"))
"""

# ╔═╡ 3ce614b2-d36e-f2ec-f717-948cc2f2776c
s3_fun = compute_s3_fun(s3_f);

# ╔═╡ 0bef0984-95fc-4660-235a-20c6722dadd1
figure_3_1(s3_fun, s3_b, s3_a, s3_new)

# ╔═╡ a203525f-a3a2-821d-2ab2-edc2179f8cfc
figure_3_3(s3_fun, s3_b, s3_a, s3_N, s3_f)

# ╔═╡ d72fb612-7224-5316-5207-83b8f1408bfd
md"""
The concentration is low almost everywhere, very high near the outlet and moderately high at a second, small source in the south-east.

**The estimator.** The derivation of Section 3.1 carries over without change. The volume (here the area) of the domain is ``V = \int_\Omega d^2\mathbf x``, and ``1/V`` is the density of the uniform distribution on ``\Omega``. Multiplying and dividing by ``V`` gives ``M = V\,E[c(\mathbf X)]`` with ``\mathbf X`` uniform on ``\Omega``, and therefore

```math
M \approx \theta_N = \frac{V}{N}\sum_{i=1}^N c(\mathbf x_i), \qquad \mathrm{Var}(\theta_N) = \frac{V^2}{N}\,\mathrm{Var}\big(c(\mathbf X)\big), \qquad \mathbf x_i \text{ i.i.d. uniform on } \Omega .
```

In one dimension, ``V = b - a``.

### 4.2 Uniform flights

The area of the lake, ``V = 5.61`` km², is known from the map (Figure 1.2 estimates it).

!!! note "Algorithm: plain Monte Carlo survey"
    For ``i = 1, \dots, N``:
    1. Draw a point uniformly in the rectangle ``[0, 4] \times [0, 3]`` km of the map; if it is not on the water, draw again. The accepted points are uniform on ``\Omega``.
    2. Fly there and measure ``c(\mathbf x_i)``.
    3. Compute the single-flight estimate ``y_i = V\,c(\mathbf x_i)`` (converted to tonnes: ``\times 100`` ha/km² ``\div 1000`` kg/t).

    Report ``\theta_N = \frac1N \sum_i y_i`` with the standard error ``s_y/\sqrt N``, where ``s_y`` is the sample standard deviation of the ``y_i``.

Each ``y_i`` is on its own an unbiased estimate of ``M``; ``\theta_N`` averages them. Figure 4.2 performs one survey. Press ▶ and follow the right panel: watch what happens to the estimate when a flight lands in the plume near the factory.

Flights $(@bind s4_N Select([50, 200, 400]; default = 200)) ``\qquad`` $(@bind sd04_4 Seed(0; label = "New survey"))
"""

# ╔═╡ 223a776e-130c-9177-6b7b-674571037920
figure_4_2(s4_N, sd04_4)

# ╔═╡ b2bd9ab7-401a-a399-34b8-3b66530a209c
md"""
## 5. Importance sampling

### 5.1 The idea and the estimator

**Question.** Uniform flights spend most measurements at the low concentration area. Can the samples be placed preferably where the integrand is large, without making the estimate wrong?

**Idea.** Draw the samples from a density ``q`` that is high in the "important" regions. Regions that are visited more often are then over-represented; to correct this, each sample is divided by the density with which it was drawn.

**Derivation.** Consider ``F = \int_\Omega f(\mathbf x)\,d^m\mathbf x``. Let ``q`` be a probability density, the **proposal**, whose support ``S = \{\mathbf x : q(\mathbf x) > 0\}`` contains ``\Omega``. Multiplying the integrand by ``1 = q(\mathbf x)/q(\mathbf x)``, which is allowed because ``q > 0`` on ``\Omega``,

```math
F = \int_\Omega f(\mathbf x)\,\frac{q(\mathbf x)}{q(\mathbf x)}\,d^m\mathbf x = \int_S \frac{f(\mathbf x)}{q(\mathbf x)}\,\mathbb 1_{\mathbf x \in \Omega}\; q(\mathbf x)\,d^m\mathbf x = E_q\!\left[\frac{f(\mathbf X)}{q(\mathbf X)}\,\mathbb 1_{\mathbf X \in \Omega}\right], \qquad \mathbf X \sim q .
```

The indicator ``\mathbb 1_{\mathbf x \in \Omega}`` (1 inside ``\Omega``, 0 outside) restricts the integral to the original domain. Replacing the expected value by an average over ``N`` i.i.d. samples ``\mathbf x_i \sim q`` gives the **importance sampling estimator**

```math
\theta_N = \frac1N\sum_{i=1}^N w_i, \qquad w_i = \frac{f(\mathbf x_i)}{q(\mathbf x_i)}\,\mathbb 1_{\mathbf x_i \in \Omega}, \qquad \mathrm{Var}(\theta_N) = \frac1N\,\mathrm{Var}_q\!\left(\frac{f(\mathbf X)}{q(\mathbf X)}\,\mathbb 1_{\mathbf X \in \Omega}\right).
```

``w_i`` is the **weight** of sample ``i``. A sample from a region where ``q`` is high (visited often) gets a small weight; a sample from a region where ``q`` is low (visited rarely) gets a large weight. The estimator is unbiased by the derivation above, and its variance depends on the choice of ``q``.

### 5.2 The optimal proposal

**Question.** Which ``q`` gives the smallest variance?

Let ``f \ge 0`` and choose ``q`` proportional to ``f``, ``q(\mathbf x) = K f(\mathbf x)``, with the constant ``K`` that makes ``q`` integrate to one. Then every weight is ``w_i = f/(Kf) = 1/K``, a constant, and

```math
\mathrm{Var}(\theta_N) = \frac1N\,\mathrm{Var}\!\left(\frac1K\right) = 0 .
```

A single sample would give the exact result. However, ``q`` integrates to one only if ``1/K = \int_\Omega f\,d^m\mathbf x = F``: to evaluate the weight one must already know the answer. This proposal cannot be used, but **a good ``q`` follows the shape of ``f`` as closely as possible, up to a constant factor.** Then all weights are nearly equal.

### 5.3 Good and poor proposals

Figure 5.1 estimates the integral of a narrow bump, ``f(x) = e^{-(x-2)^2/(2 \cdot 0.3^2)}`` with exact value ``F = 0.3\sqrt{2\pi} \approx 0.752``, with five different proposals. Look first at the left panel: how well does the red proposal match the blue shape of ``f``? Then at the middle panel: a good proposal gives weights close to the dashed line, the weight of the ideal proposal. The right panel shows how the estimate scatters over 1000 independent runs. Go through all five proposals.

Proposal $(@bind s5_case Select(["A: uniform on [0, 4]", "B: normal, centre 2, width 0.4 (good match)", "C: normal, centre 2, width 0.15 (too narrow)", "D: normal, centre 3, width 0.4 (shifted)", "E: uniform on [2, 4] (misses half of f)"]; default = "B: normal, centre 2, width 0.4 (good match)")) ``\qquad`` samples per run ``N`` $(@bind s5_N Select([100, 1000]; default = 100)) ``\qquad`` $(@bind sd04_6 Seed(0))
"""

# ╔═╡ e417d0c3-f551-0ef6-cbeb-913e863c88e6
figure_is(s5_case, s5_N, sd04_6)

# ╔═╡ 59dfc236-2f1b-7f4a-ecbf-1a000bb8bc54
md"""
### 5.4 A flight plan for the drone

The location of the outlet is known, the second source is not. The proposal for the drone is therefore a **mixture** of two parts:

```math
q(\mathbf x) = \delta\,\frac{1}{V} + (1 - \delta)\,\frac{g(\mathbf x)}{Z}, \qquad g(\mathbf x) = \frac{1}{2\pi w^2}\,e^{-|\mathbf x - \mathbf x_c|^2/(2w^2)}, \qquad \mathbf x \in \Omega .
```

- With probability ``\delta`` the drone flies to a **uniform** point of the lake (density ``1/V``).
- With probability ``1 - \delta`` it flies to a point drawn from a **normal distribution** with centre ``\mathbf x_c`` and width ``w``; points that fall outside the lake are drawn again. ``Z = \int_\Omega g\,d^2\mathbf x`` is the probability that a normal point lands on the lake; dividing by ``Z`` makes this part a density on ``\Omega``.

Since ``q`` is known at every point, each measurement gets the weight ``w_i = c(\mathbf x_i)/q(\mathbf x_i)`` (in tonnes after the unit conversion), and ``\theta_N`` is the average weight. The uniform part, the **defensive** share ``\delta > 0``, guarantees ``q \ge \delta/V > 0`` on the whole lake. This ensures the condition of Section 5.1, including the unknown source, and it bounds every weight by ``c\,V/\delta``.

!!! game "Experiment · the best flight plan"
    With 200 flights per survey, find the proposal for which the estimates scatter least. The figure reports how many times more efficient the plan is than uniform flights. Then try a plan without uniform flights (``\delta = 0``) and a narrow distribution around the outlet, and compare the histograms.

Centre east $(@bind s5_cx Slider(0.2:0.1:3.8; default = 1.0, show_value = true)) km ``\quad`` centre north $(@bind s5_cy Slider(0.2:0.1:2.8; default = 1.8, show_value = true)) km ``\quad`` width $(@bind s5_w Select([0.15, 0.3, 0.45, 0.6, 1.0]; default = 0.45)) km ``\quad`` uniform share ``\delta`` $(@bind s5_d Select([0.0, 0.1, 0.3, 0.5, 1.0]; default = 0.3)) ``\quad`` $(@bind sd04_5 Seed(0))
"""

# ╔═╡ 4711e7db-56ba-aefe-b509-d199c32b2604
figure_5_4(s5_cx, s5_cy, s5_w, s5_d, sd04_5)

# ╔═╡ 2f42b6e4-1d51-177f-1c7b-78bdbaec96ae
# layout of figures, animations and questions
html"""<style>
.fg{margin:10px 0 6px 0}
.fg svg.fig,.pl svg.fig{display:block;max-width:100%;height:auto;background:#fff}
.fg-row{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:flex-start}
.fg-cap{font-size:0.9rem;color:#333;line-height:1.4;margin:6px 2px 0 2px;max-width:720px}
.fg-cap p{margin:0}
.fg-cap b.fg-num{font-weight:700;color:#1b1b1b}
.pl{border:1px solid #e3e3df;border-radius:4px;padding:8px 4px 6px;background:#fff;margin:8px 0}
.pl-bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px;font-family:system-ui,sans-serif;font-size:13px}
.pl-bar button{font:inherit;padding:2px 10px;border:1px solid #cfcfca;border-radius:3px;background:#fafaf8;cursor:pointer;min-width:2.4em}
.pl-bar button:hover{border-color:#1f5aa6}
.pl-bar input[type=range]{flex:1;min-width:160px;accent-color:#1f5aa6}
.pl-lab{font-variant-numeric:tabular-nums;color:#444;min-width:7em}
.pl-read{font-family:system-ui,sans-serif;font-size:13px;color:#333;min-height:1.4em;margin-top:4px;font-variant-numeric:tabular-nums}
.quiz{border:1px solid #e3e3df;border-radius:4px;padding:10px 14px;margin:14px 0;background:#fdfdfc}
.quiz-head{font-size:0.74rem;letter-spacing:0.09em;text-transform:uppercase;font-weight:700;color:#6a4c93;margin-bottom:4px}
.quiz-q p{margin:2px 0 6px}
.quiz-opts{display:flex;flex-direction:column;gap:6px;margin:6px 0}
.quiz-opt{display:flex;gap:10px;align-items:baseline;border:1px solid #e3e3df;border-radius:4px;padding:5px 10px;cursor:pointer;background:#fff}
.quiz-opt:hover{border-color:#1f5aa6}
.quiz-opt p{margin:0}
.quiz-key{font-weight:700;color:#8c8c8c;min-width:1.2em;font-family:system-ui,sans-serif}
.quiz-opt.picked.right{border-color:#2f7d55;background:#eef6f0}
.quiz-opt.picked.wrong{border-color:#c2462e;background:#fbf1ee}
.quiz-fb{border-radius:3px;padding:7px 12px;margin:6px 0 2px;font-size:0.95rem}
.quiz-fb p{margin:2px 0}
.quiz-fb.yes{background:#eef6f0;border-left:3px solid #2f7d55}
.quiz-fb.no{background:#fbf1ee;border-left:3px solid #c2462e}
.quiz-num{display:flex;gap:8px;align-items:center;flex-wrap:wrap;font-family:system-ui,sans-serif}
.quiz-num input{font:inherit;padding:3px 8px;border:1px solid #cfcfca;border-radius:3px}
.quiz-num input.ok{border-color:#2f7d55;background:#f3faf5}
.quiz-num button{font:inherit;padding:2px 10px;border:1px solid #cfcfca;border-radius:3px;background:#fafaf8;cursor:pointer}
details.sol{border:1px solid #2f7d55;border-left:5px solid #2f7d55;border-radius:4px;padding:.4em .7em;margin:.2em 0 .8em}
details.sol summary{cursor:pointer;font-weight:600;color:#2f7d55}
div.sol-locked{border:1px dashed #bbb;border-radius:4px;padding:.35em .7em;color:#777;font-size:.9em;margin:.2em 0 .8em}
details.proof{border-left:3px solid #16427a;padding:2px 0 2px 12px;margin:8px 0}
details.proof summary{cursor:pointer;color:#16427a;font-weight:600}
details.hint{margin-top:4px}
details.hint summary{cursor:pointer;font-weight:600}
</style>"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Base64 = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SHA = "ea8e919c-243c-51af-8825-aaa63cd721ce"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
Distributions = "~0.25.131"
HypertextLiteral = "~1.0.0"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.1"
manifest_format = "2.1"
project_hash = "ccac36e3d7f3a9ac8cd2fdfdc91a1a838abe8fa0"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
registries = "General"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
registries = "General"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CommonSolve]]
deps = ["PrecompileTools"]
git-tree-sha1 = "6c389fa857f6ca5a95474b52a52023fd77f24cb7"
registries = "General"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.14"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
registries = "General"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
registries = "General"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
registries = "General"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
registries = "General"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a958ab3a40c755563f5e1405c0846cb0446bf19d"
registries = "General"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.131"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
registries = "General"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "086b5fbd032baf544678cc15b52b015e4c4aceb8"
registries = "General"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.1"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
registries = "General"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
registries = "General"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
registries = "General"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "d1a86724f81bcd184a38fd284ce183ec067d71a0"
registries = "General"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "1.0.0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
registries = "General"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
registries = "General"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
registries = "General"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
registries = "General"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.18.0+1"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "OpenSSL_jll", "Zlib_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.104+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.13.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "b85e2797b2409570e84c4de46238c0ed5f6476ae"
registries = "General"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.2"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
registries = "General"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
registries = "General"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2026.8.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.30+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
registries = "General"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "5005266de4bfe50e53ff44a5cb5c540b6e47a254"
registries = "General"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.6.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
registries = "General"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
registries = "General"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
registries = "General"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
registries = "General"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "971f04b3780c0da4edf230573fc1aadc2c6e649e"
registries = "General"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.10"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
registries = "General"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.13.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "429071b23f4c9a13fb6582f807cc2ef454082408"
registries = "General"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.9.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
registries = "General"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
registries = "General"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
registries = "General"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.10.1+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
registries = "General"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "908fec9df6c5de98548ead82a468c95ccf6cd263"
registries = "General"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.7.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["CompilerSupportLibraries_jll", "Libdl"]
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.67.1+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─8c99463c-3198-d27e-ca3e-fd548d5cf8a2
# ╟─e444a9e4-2eaf-7df2-f017-5fef67da466b
# ╟─545007f7-ca6b-c1f0-b034-a020653c0d02
# ╟─6157a2db-7044-1b80-b015-f415474e87bd
# ╟─e3bb564d-06e0-adb8-3f31-ff72db83a251
# ╟─79740486-fc58-382f-af93-a98c35227077
# ╟─ce5d7f25-850f-0fce-db52-5eefa14a9001
# ╟─b1472a85-0568-1b23-7597-31d17058d0bd
# ╟─23e1102e-5eee-3e42-5940-37b196c82fba
# ╟─969086ae-15cd-a847-acb1-206abbcef36f
# ╟─eef6abd0-c772-7bc3-2c98-f32a780d4385
# ╟─3ce614b2-d36e-f2ec-f717-948cc2f2776c
# ╟─0bef0984-95fc-4660-235a-20c6722dadd1
# ╟─fde1cec7-d538-6f67-c864-99b74aa612fd
# ╟─a203525f-a3a2-821d-2ab2-edc2179f8cfc
# ╟─5acc51bd-a5d8-c73c-2186-673f925545bc
# ╟─21441567-4811-fe79-3396-20c1dabd45f8
# ╟─3cf54315-740c-9dfd-bc4a-920d47cdd849
# ╟─11e3b2b8-c4db-798c-718f-7f0e3109bc7d
# ╟─ca32279f-116a-45ef-58aa-2c1a9b20a9e9
# ╟─d67ea20b-8d7c-0207-df85-ca1f9e6a4219
# ╟─2b2d73b1-ad81-af4c-2da4-901ccd1028e0
# ╟─d72fb612-7224-5316-5207-83b8f1408bfd
# ╟─223a776e-130c-9177-6b7b-674571037920
# ╟─8491345c-04aa-5ae1-d4c0-29dc6c3f503d
# ╟─7f5c299a-f91c-082d-e475-69a2344bcdf3
# ╟─a2e6e5ca-75dc-aa7b-791f-14cdf06df9d9
# ╟─b2bd9ab7-401a-a399-34b8-3b66530a209c
# ╟─e417d0c3-f551-0ef6-cbeb-913e863c88e6
# ╟─8578d6d1-252a-8d48-11b6-5f10e518ea53
# ╟─b898ca97-8643-8393-de4c-c26b1ed0b865
# ╟─96c69e43-2c05-127a-5719-630b00a82df4
# ╟─59dfc236-2f1b-7f4a-ecbf-1a000bb8bc54
# ╟─4711e7db-56ba-aefe-b509-d199c32b2604
# ╟─db81f0a8-1f2b-78ef-2a10-4a67a23173c4
# ╟─db30c858-4699-00c5-7b50-b965c9154615
# ╟─78498a7b-acee-30d7-e48b-d4bee929d80d
# ╠═dd479f0a-a0c4-dd34-f5cf-07ba0a960878
# ╟─4b962447-df83-f6f0-76ca-782cb800d432
# ╟─f9c61d30-0d56-9bf1-b85a-e806ef6cb254
# ╟─6efd2c36-f857-de55-92cd-3be26c63ec0a
# ╟─207eae68-a0a3-fc86-0351-87e9ae855cf8
# ╟─6d0f2434-cac6-7a87-00fd-0714e94a2dcf
# ╟─1581a3a7-b318-7058-3e50-4f60cfd3016b
# ╟─d9c4bd58-0ba5-f994-4202-18482d70966c
# ╟─ff7180a4-cc16-ccac-688a-b2a202256575
# ╟─c00b6b12-2f42-414b-e52b-5c79b8015d12
# ╟─270f711a-650e-a84a-1446-d1caad08fa51
# ╟─3829776e-d93a-895f-b637-202a0c5e8b34
# ╠═4b5b546f-196b-fe76-a330-3a20044a2194
# ╟─32132fb7-2b55-24bb-9843-f1a14a8ce450
# ╟─6ec18bfd-9161-399b-8d26-85abde2400a4
# ╟─82df10b9-7bc3-ad36-97e2-c6da0d582441
# ╠═646faf41-c93f-9579-6d05-99b81a3a316b
# ╟─3794a3eb-1caa-2ced-7bd6-a8c73fea0c6b
# ╟─095b319f-2f26-701b-0235-58c99f8f4b5a
# ╟─ba37f52d-2959-d794-d844-8e3129b23b32
# ╠═296d4abe-987f-c750-d0c6-d62ac781acd6
# ╟─06555cb2-1486-e734-1af9-bab1177872e8
# ╟─a9944d94-57bf-708a-c426-05af0acd2d85
# ╟─ad58a80b-7990-6d1c-5409-b7bd2dfebfb7
# ╟─c53da7f3-8dc9-bae2-85a9-6da83b211fc7
# ╟─b4dc5b17-3cbd-c88d-597d-5c7db40b9053
# ╟─82b77ce8-fd4b-2902-d6b8-4758c4956eda
# ╟─bcc26dda-0a7f-524d-92fe-5315356d99c9
# ╟─7169378f-d01b-0643-46af-4f926f785f66
# ╟─3ff4f2a8-879b-357c-5aa2-cb9830b2eb7a
# ╟─32afd1f8-1018-5749-a0f2-ce44586ef525
# ╟─468ac6f3-fd09-6aad-5cc5-2c203a5245aa
# ╟─f3a8c8d4-48f5-39ca-f47c-662c638125e7
# ╟─2f42b6e4-1d51-177f-1c7b-78bdbaec96ae
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
