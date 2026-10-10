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

# ╔═╡ 3e02678a-dfa6-7e13-17f1-11258e9ebca7
begin
	using PlutoUI, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions
	import AbstractPlutoDingetjes
end

# ╔═╡ a78f4338-148b-8ea3-b73d-1dbf9c1844a0
md"""
# 2 · Random Variate Generation

*Stochastic Modeling and Simulation*
"""

# ╔═╡ 13b07c56-4e67-1515-5f9f-9931dbd7766d
md"""
Stochastic simulation needs random numbers, often millions or billions of random numbers. A computer, however, runs deterministic instructions. This notebook explores random variate generation: how to generate numbers that are independent and uniformly distributed on ``[0, 1)``, and how to transform such numbers into samples of any other distribution.

!!! hook "Guiding questions"
    **(1)** How can a deterministic machine generate numbers that can be used as random numbers? **(2)** How to generate random numerbs from any given distribution?
"""

# ╔═╡ 05fbffb6-e4be-7ee3-061e-db4844f44486
md"""
## 1. Algorithmic and data-driven generators

There are two approaches to generate uniform random numbers on a computer.

- **Data-driven** generators measure a physical process that is unpredictable in practice such as thermal noise in electronic circuits, the times between keystrokes or network packets, radioactive decay. The numbers are unpredictable, which cryptography requires, but they are produced slowly and a sequence cannot be reproduced.
- **Algorithmic** generators compute a sequence with a deterministic recursion from a seed. They are fast, require no special hardware, and are **reproducible**: the same seed gives the same sequence, so that a simulation can be repeated and checked.

Simulations therefore use algorithmic generators. Such a generator does not produce random numbers; it produces numbers that *behave* like independent uniform random numbers. 

## 2. Pseudo-random numbers

### 2.1 The linear congruential generator

The linear congruential generator (LCG), proposed by D. H. Lehmer in 1949, computes

```math
z_i = (a\,z_{i-1} + c) \bmod m, \qquad u_i = \frac{z_i}{m} \in [0, 1),
```

with modulus ``m``, multiplier ``a``, increment ``c`` and seed ``z_0``. Arithmetic modulo ``m`` can be pictured as arithmetic on a clock with ``m`` positions: each step jumps from the current position to the next. Choose small parameters and follow the jumps.

``m`` $(@bind s2_m NumberField(2:64; default = 31)) ``\qquad`` ``a`` $(@bind s2_a NumberField(0:200; default = 3)) ``\qquad`` ``c`` $(@bind s2_c NumberField(0:200; default = 0)) ``\qquad`` seed ``z_0`` $(@bind s2_z0 NumberField(0:200; default = 1))
"""

# ╔═╡ 20ffbdf2-0dcd-6905-f64f-a4d729528b21
md"""
The figure shows two properties of a pseudo-random generator.

1. **Periodicity.** The state can take only ``m`` values. Since each value depends on the previous one, the sequence repeats as soon as a value occurs for the second time, and from then on it repeats. The sequence is therefore eventually periodic, and its **period** (or **cycle length**, the number of steps after which the sequence repeats) is at most ``m``. A simulation that draws more numbers than the period reuses the same numbers in the same order.

2. **Dependence on the parameters.** With ``m = 31`` and ``c = 0``, the multiplier ``a = 3`` visits all 30 non-zero values, ``a = 5`` only three.

Practical generators use large, carefully chosen parameters. The *minimal standard* generator uses ``a = 48\,271`` and ``m = 2^{31} - 1``, with a cycle length of about ``2.1 \cdot 10^9``. Modern generators replace the multiplication by bit operations on a larger state: the Mersenne Twister (1998) has a cycle length of ``2^{19937} - 1``, and Julia's default generator Xoshiro256++ (2018) has ``2^{256} - 1``.

The xorshift generator (G. Marsaglia, 2003) is a simple example of such a generator. 
"""

# ╔═╡ 8afaa8a6-31b3-5834-c6b1-e496f456fc51
function xorshift32(seed, n)
	x = UInt32(seed)
	u = zeros(n)
	for i in 1:n
		x ⊻= x << 13
		x ⊻= x >> 17
		x ⊻= x << 5
		u[i] = x / 2.0^32
	end
	return u
end;

# ╔═╡ 4439b501-4289-1100-bd44-7e4864e9c7e3
md"""
### 2.2 Testing a generator

A long cycle is necessary but not sufficient. The numbers must also be **uniformly distributed and behave as if they were independent**. Two simple graphical tests can be used to detect some defects: the **histogram** of the numbers should be flat, and the **pairs** ``(u_i, u_{i+1})`` should fill the unit square without any visible pattern.

Generator $(@bind s2_gen Select(["the generator of Figure 2.1", "small LCG: a = 7, m = 8191", "minstd: a = 48271, m = 2³¹ − 1", "RANDU: a = 65539, m = 2³¹", "Julia default (Xoshiro256++)"]; default = "small LCG: a = 7, m = 8191")) ``\qquad`` numbers $(@bind s2_n Select([500, 2000, 10_000]; default = 2000))
"""

# ╔═╡ 3e086307-18a7-b5ad-500b-9a92ba0eeaac
md"""
!!! think "Think"
    Two runs of a simulation started with the same seed give exactly the same result. A colleague reports this as a bug in the random number generator. Is it?
"""

# ╔═╡ f9ce07c7-8013-bb1b-f790-070ee8dfe900
md"""
The small generator passes the histogram test and fails the pair test: its pairs lie on a few lines. The pairs, triples and longer tuples of *every* linear congruential generator lie on a limited number of parallel hyperplanes. Good parameters make these planes so many and so close to each other that a simulation is not affected.

The generator RANDU, by IBM in the 1960s and used in many simulations, illustrates the problem. Its pairs appear uniform; its triples ``(u_i, u_{i+1}, u_{i+2})`` do not. Rotate the cube until the pattern becomes visible.

Generator $(@bind s2_g3 Select(["RANDU: a = 65539, m = 2³¹", "minstd: a = 48271, m = 2³¹ − 1", "Julia default (Xoshiro256++)"]; default = "RANDU: a = 65539, m = 2³¹")) ``\qquad`` rotation $(@bind s2_yaw Slider(0:1:180; default = 30, show_value = true))° ``\qquad`` tilt $(@bind s2_pitch Slider(-90:1:90; default = 20, show_value = true))°
"""

# ╔═╡ 3864ded5-4243-970d-39f0-c2d1de65456f
md"""
### 2.3 The middle-square method

The first generator for an electronic computer was proposed by John von Neumann in 1949: a four-digit number is squared, giving eight digits, and the **middle four digits** form the next number. For example, ``1234^2 = 01\,5227\,56``, so the next number is 5227.

!!! game "Experiment · the longest-lived seed"
    Find a four-digit seed whose sequence lasts as long as possible before it reaches 0 or enters a cycle.

Seed $(@bind s2_ms NumberField(0:9999; default = 1234))
"""

# ╔═╡ 7edcde5d-3807-12c8-af71-b7143da81a4f
md"""
Most seeds reach a short cycle (for example 6100 → 2100 → 4100 → 8100 → 6100) or the value 0 within a few dozen steps.
"""

# ╔═╡ 8d31b823-8dc3-9f89-7449-8326e98a7c13
md"""
Quasi-random numbers are **not** independent: consecutive values differ by the fixed step ``\alpha``. They are therefore suitable for integration, but not for simulations that require independent random numbers.
"""

# ╔═╡ 72b1562a-d62c-4b5d-d6a7-f4ca54c2fcf0
md"""
**Why the distribution changes.** Every number in a small interval around ``x`` is mapped to a small interval around ``y = g(x)``. The two intervals therefore has **the same draws** and have the **same probability**. Their widths, however, differ: ``g`` stretches the interval where it is steep and compresses it where it is flat. 
"""

# ╔═╡ 2f83d373-6aa5-cbf6-9485-bff5d7024957
md"""
A strip of width ``dx`` around ``x`` has probability ``f_X(x)\,dx``, and a strip of width ``dy = |g'(x)|\,dx`` around ``y = g(x)``. The probability is conserved, so the density, probability per unit length, becomes

```math
f_Y(y) = \frac{f_X(x)}{|g'(x)|} = f_X\big(g^{-1}(y)\big)\,\left|\frac{d}{dy}g^{-1}(y)\right| \qquad \text{for a monotone } g .
```

**The absolute value.** A density must be non-negative. For a decreasing ``g``, the derivative of ``g^{-1}`` is negative; the absolute value makes sure the density is non-negative. Equivalently, for an increasing ``g``, ``F_Y(y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))``, and for a decreasing ``g`` the inequality is reversed, ``F_Y(y) = 1 - F_X(g^{-1}(y))``; differentiating either expression gives the formula with the absolute value.

**Example.** Let ``f_X(x) = 2x`` on ``[0, 1]`` and ``Y = 5 - X``. Then ``g^{-1}(y) = 5 - y`` and ``\frac{d}{dy}g^{-1}(y) = -1``, so ``f_Y(y) = 2(5 - y) \cdot |-1| = 2(5-y)`` on ``[4, 5]``. Without the absolute value, the result would be negative everywhere.

Two important cases: ``Y = 1 - X`` is again uniform if ``X`` is uniform, and ``Y = -\ln X`` has the density ``e^{-y}``.
"""

# ╔═╡ aba50892-b7dc-ac95-8273-18e596e2073b
md"""
!!! think "Think"
    A program generates exponential waiting times with rate ``\lambda = 10`` per hour as `-log(rand()) * λ`. The waiting times look plausible: positive, many short, a few long. What is wrong, and which simple check reveals the error?
"""

# ╔═╡ 97b6e3a0-e743-c7b8-19ee-0e9094eaf3d9
md"""
**Exponential distribution.** Solving ``u = 1 - e^{-\lambda x}`` gives ``x = -\ln(1 - u)/\lambda``. Since ``1 - U`` is also uniform, ``X = -\ln(U)/\lambda`` can be used as well.

**Discrete distributions.** For values ``x_1 < x_2 < \dots`` with probabilities ``p_1, p_2, \dots``, return the first ``x_k`` with ``p_1 + \dots + p_k > u``. 

**Any distribution with a known quantile function.** The package `Distributions` gives ``F^{-1}`` as `quantile`, so that `quantile.(D, rand(n))` produces ``n`` samples of `D`. The cell below uses a Weibull distribution, a common model for the lifetime of components; replace it by `Gamma(3, 1)`, `LogNormal(0, 0.5)` or `Beta(2, 5)`.
"""

# ╔═╡ b8e69f66-e5d7-5ddd-2d89-0a595ce225bf
md"""
!!! think "Think"
    In Figure 4.3, all five stripes of the square have the same area, but the rings in the plane have very different widths. Why must the outermost ring be the widest, and what would happen to the tails of ``X_1`` if ``U_1`` were generated by a generator that never returns values below ``10^{-6}``?
"""

# ╔═╡ 1b7bd338-7795-38fc-ccb2-1b5cc4bd94f2
md"""
!!! think "Think"
    An accept–reject sampler accepts only 0.1 % of its proposals. Is its output wrong, and what would you change?
"""

# ╔═╡ cb3147ba-3d23-9081-f82d-68299f7decd6
md"""
The method applies to any density that can be evaluated, even one drawn by hand. Its cost is the expected number of attempts per accepted sample, ``\mu``. The envelope fits poorly, and ``\mu`` is large, when

- the target is sharply peaked, so that the box must be tall and is mostly empty, so many wasted proposals;
- the target has **heavy tails**: a box cannot fit a density with unbounded support, and an envelope with narrow tails than the target cannot satisfy ``\mu g \ge f`` for any finite ``\mu``;
- the dimension is high.
"""

# ╔═╡ 9d409c93-6870-45ce-b6bd-5d2946233d32
md"""
!!! think "Think"
    Composition–rejection requires a set-up step before the first sample: the levels and boxes must be computed. In which situation is this worthwhile, and in which is it not? Consider a simulation that draws ``10^9`` samples from one fixed density, and one that draws a few samples from a density that changes after every sample.
"""

# ╔═╡ a557a30b-8610-38da-c4c3-3f8fb19c04af
md"""
| situation | method | remark |
|:--|:--|:--|
| inverse CDF available (exponential, Weibull) | inversion | one uniform number per sample, exact |
| normal distribution | Box–Muller (or `randn`) |  |
| only the density is known, low dimension | accept–reject with a close envelope | peaks and heavy tails make ``\mu`` large |
| sharply peaked density, many samples | composition–rejection | set-up cost |
| complex, high-dimensional distributions | Markov chain Monte Carlo | correlated samples |

"""

# ╔═╡ 7b60936c-bb3b-508f-3a07-326af90fe25f
md"""
## 7. Answers to the guiding questions

**(1)** A deterministic machine produces pseudo-random numbers with a recursion that starts from a seed. The sequence is periodic and predictable, but with a well-designed recursion the cycle is extremely long and the numbers pass statistical tests. Determinism is an advantage, to make simulations reproducible.

**(2)** Uniform numbers are transformed: by the inverse CDF (exponential: ``-\ln U/\lambda``), by a suitable change of variables (normal: Box–Muller), or by accept–reject when only the density is known.
"""

# ╔═╡ 6a67d0d7-f6ca-4d0e-352a-a9b79c220dda
md"""
## Further reading

| resource | content |
|:--|:--|
| [G. Marsaglia, *Random numbers fall mainly in the planes* (1968)](https://pmc.ncbi.nlm.nih.gov/articles/PMC285899/) | the lattice structure of linear congruential generators |
| [M. E. O'Neill, PCG random number generators](https://www.pcg-random.org/) | period, speed, statistical quality and predictability of generators |
| [S. Vigna, xoshiro and xoroshiro generators](https://prng.di.unimi.it/) | the family of Julia's default generator |
| [L. Devroye, *Non-Uniform Random Variate Generation*, Chapter II](https://luc.devroye.org/chapter_two.pdf) | inversion and rejection |

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 2.
- G. E. P. Box and M. E. Muller, "A note on the generation of random normal deviates", *Annals of Mathematical Statistics* 29 (1958) 610–611.
- P. L'Ecuyer and R. Simard, "TestU01: a C library for empirical testing of random number generators", *ACM Transactions on Mathematical Software* 33 (2007) 22.
- G. Marsaglia, "Random numbers fall mainly in the planes", *Proceedings of the National Academy of Sciences* 61 (1968) 25–28.
- S. K. Park and K. W. Miller, "Random number generators: good ones are hard to find", *Communications of the ACM* 31 (1988) 1192–1201; the multiplier 48271 of minstd: S. K. Park, K. W. Miller and P. K. Stockmeyer, *Communications of the ACM* 36(7) (1993) 105–110.
- M. Matsumoto and T. Nishimura, "Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator", *ACM Transactions on Modeling and Computer Simulation* 8 (1998) 3–30.
- G. Marsaglia, "Xorshift RNGs", *Journal of Statistical Software* 8(14) (2003) 1–6.
- D. Blackman and S. Vigna, "Scrambled linear pseudorandom number generators", *ACM Transactions on Mathematical Software* 47(4) (2021) 36 (Xoshiro256++; preprint 2018).
- J. H. Halton, "On the efficiency of certain quasi-random sequences of points in evaluating multi-dimensional integrals", *Numerische Mathematik* 2 (1960) 84–90; I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals", *USSR Computational Mathematics and Mathematical Physics* 7(4) (1967) 86–112.
"""

# ╔═╡ d873d3c8-a66f-1821-a9e5-842bdd89fa8e
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 5472fa1d-a48c-d08d-cbf7-d185da690229
begin
	function lcg_internal(a, c, m, seed, n)
		u = zeros(n); z = Int128(seed)
		for i in 1:n
			z = mod(a * z + c, m); u[i] = Float64(z) / m
		end
		u
	end

	function discrepancy1(x)
		s = sort(x); n = length(s)
		maximum(max(abs(i / n - s[i]), abs((i - 1) / n - s[i])) for i in 1:n)
	end
end;

# ╔═╡ 42706196-7709-0faa-d7d6-1e4e3c90e190
# SVG
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

	function bars!(f::Fig, x, hgt; width = 0.8, c = COL.blue, op = 0.85, base = 0.0, stroke = "white", sw = 0.8, s = nothing, o = nothing, labels = nothing, lsize = 11)
		for i in eachindex(x)
			x0 = X(f, x[i] - width / 2); x1 = X(f, x[i] + width / 2)
			y0 = Y(f, base); y1 = Y(f, hgt[i])
			print(f.body, "<rect x=\"", _n(min(x0, x1)), "\" y=\"", _n(min(y0, y1)), "\" width=\"", _n(abs(x1 - x0)), "\" height=\"", _n(abs(y1 - y0)),
				"\" fill=\"", _pick(c, i), "\" opacity=\"", _pick(op, i), "\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\"", _st(_pick(s, i), _pick(o, i)), "/>")
			if labels !== nothing
				text!(f, x[i], hgt[i], labels[i]; anchor = "middle", dy = -4, size = lsize, c = "#444", s = _pick(s, i), o = _pick(o, i))
			end
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

	function seg!(f::Fig, x0, y0, x1, y1; c = COL.grey, w = 1.2, dash = nothing, op = 1.0, s = nothing, o = nothing, over = false)
		print(over ? f.over : f.body, "<line x1=\"$(_n(X(f, x0)))\" y1=\"$(_n(Y(f, y0)))\" x2=\"$(_n(X(f, x1)))\" y2=\"$(_n(Y(f, y1)))\" stroke=\"$(c)\" stroke-width=\"$(w)\" opacity=\"$(op)\" stroke-linecap=\"round\"", _dash(dash), _st(s, o), "/>")
		f
	end

	vline!(f::Fig, x; c = "#555", w = 1.1, dash = "4 3", s = nothing, o = nothing) = seg!(f, x, f.yr[1], x, f.yr[2]; c, w, dash, s, o)

	hline!(f::Fig, y; c = "#555", w = 1.1, dash = "4 3", s = nothing, o = nothing) = seg!(f, f.xr[1], y, f.xr[2], y; c, w, dash, s, o)

	function arrow!(f::Fig, x0, y0, x1, y1; c = COL.ink, w = 1.6, head = 8, op = 1.0, dash = nothing, s = nothing, o = nothing, over = true)
		a0, b0, a1, b1 = X(f, x0), Y(f, y0), X(f, x1), Y(f, y1)
		L = hypot(a1 - a0, b1 - b0); L < 0.5 && return f
		ux, uy = (a1 - a0) / L, (b1 - b0) / L; h = min(head, 0.45L)
		io = over ? f.over : f.body
		print(io, "<g opacity=\"$(op)\"", _st(s, o), ">")
		print(io, "<line x1=\"$(_n(a0))\" y1=\"$(_n(b0))\" x2=\"$(_n(a1 - ux * h * 0.7))\" y2=\"$(_n(b1 - uy * h * 0.7))\" stroke=\"$(c)\" stroke-width=\"$(w)\" stroke-linecap=\"round\"", _dash(dash), "/>")
		print(io, "<polygon points=\"$(_n(a1)),$(_n(b1)) $(_n(a1 - ux * h - uy * h * 0.42)),$(_n(b1 - uy * h + ux * h * 0.42)) $(_n(a1 - ux * h + uy * h * 0.42)),$(_n(b1 - uy * h - ux * h * 0.42))\" fill=\"$(c)\"/></g>")
		f
	end

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

	function _pad(a, b)
		a == b && return (a - 1, b + 1)
		d = 0.04 * (b - a); (a - d, b + d)
	end

	function histplot(xs; bins = 30, xlabel = "", title = "", c = COL.blue, w = 520, h = 240, density = false, ymax = nothing)
		v = filter(isfinite, collect(Float64, xs))
		isempty(v) && return md"*No finite values to plot.*"
		lo, hi = minimum(v), maximum(v)
		if all(isinteger, v) && hi - lo <= 60
			ks = collect(lo:hi); cs = [count(==(k), v) for k in ks] ./ (density ? length(v) : 1)
			f = fig(; w, h, xr = (lo - 0.7, hi + 0.7), yr = (0, 1.1maximum(cs)), xlabel, ylabel = density ? "fraction" : "count", title)
			return bars!(f, ks, cs; c, width = hi - lo > 25 ? 0.95 : 0.8)
		end
		lo == hi && (lo -= 0.5; hi += 0.5)
		e = collect(range(lo, hi; length = bins + 1)); hgt = histdata(v, e; density)
		f = fig(; w, h, xr = (lo, hi), yr = (0, ymax === nothing ? 1.1maximum(hgt) : ymax), xlabel, ylabel = density ? "density" : "count", title)
		hist!(f, e, hgt; c)
	end

	function scatterplot(x, y; xlabel = "", ylabel = "", title = "", w = 360, h = 320, c = COL.blue, r = 2.2, op = 0.6, xr = nothing, yr = nothing)
		xv = collect(Float64, x); yv = collect(Float64, y)
		f = fig(; w, h, xr = xr === nothing ? _pad(extrema(xv)...) : xr, yr = yr === nothing ? _pad(extrema(yv)...) : yr, xlabel, ylabel, title)
		dots!(f, xv, yv; c, r, op)
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

# ╔═╡ f8418f96-04c9-9794-25f3-94e145770c57
let
	u = xorshift32(2463534242, 3000)
	scatterplot(u[1:end-1], u[2:end]; xlabel = "uᵢ", ylabel = "uᵢ₊₁", title = "xorshift32: consecutive pairs", r = 1.4)
end

# ╔═╡ 184f7345-c519-5ed0-9322-bd0e9ad5a86c
let
	D = Weibull(1.5, 2.0)
	x = quantile.(D, rand(20_000))
	t = range(minimum(x), maximum(x); length = 200)
	f = histplot(x; bins = 40, density = true, ymax = 1.15maximum(pdf.(D, t)), xlabel = "x", title = "samples by inversion (bars) and the density (red)")
	line!(f, t, pdf.(D, t); c = COL.red, w = 2)
end

# ╔═╡ b19f5437-9612-dbb9-c293-c89f71af4333
# 
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

	struct PseudoCode
		lines::Vector{String}
		active::Vector{Vector{Int}}
		title::String
	end

	function Base.show(io::IO, ::MIME"text/html", p::PseudoCode)
		print(io, "<div class=\"pc\">")
		p.title == "" || print(io, "<div class=\"pc-head\">", xesc(p.title), "</div>")
		for (i, l) in enumerate(p.lines)
			print(io, "<div class=\"pc-ln\">")
			for (k, act) in enumerate(p.active)
				i in act && print(io, "<span class=\"pc-mark\" data-o=\"$(k - 1)\"></span>")
			end
			print(io, "<span class=\"pc-no\">$(i)</span><code>", xesc(l), "</code></div>")
		end
		print(io, "</div>")
	end

	_md_html(s) = s isa AbstractString ? _html(Markdown.parse(replace(s, r"``([^`]+)``" => s"*\1*"))) : _html(s)

	"""
	    walk(code, steps, panels...; title, caption, label, fps)

	A step-by-step walkthrough of an algorithm. `code` is a vector of pseudocode lines; `steps`
	is a vector with one entry per frame (frame 0, 1, …), each a named tuple with fields
	`line` (an Int or a vector of Ints: the active lines), `state` (what the variables are now)
	and `why` (why this step is done); an optional field `fail` adds a "what goes wrong" note.
	`panels` are figures whose elements are tagged with `o = k` (frame k only) or `s = k`
	(from frame k on).
	"""
	function walk(code::Vector{String}, steps, panels...; title = "algorithm", caption = nothing, label = "step", fps = 1.0)
		act = [s.line isa Integer ? [s.line] : collect(s.line) for s in steps]
		reads = map(steps) do s
			r = "<div class=\"wk-state\">" * _md_html(s.state) * "</div>"
			replace(r, "<p>" => "<span>", "</p>" => "</span> ")
		end
		p = player(PseudoCode(code, act, title), panels...; n = length(steps) - 1, start = 0, fps = fps, reads = reads, label = label, caption = caption)
		p
	end
end;

# ╔═╡ 9c4c7b32-e21e-c34c-1326-30e845b977d1
# figures helpers
begin
	function compute_s2_seq(s2_m, s2_a, s2_c, s2_z0)
		let
			m, a, c = s2_m, mod(s2_a, s2_m), mod(s2_c, s2_m); z = mod(s2_z0, m)
			seen = Dict(z => 0); zs = [z]
			cyc = 0; start = 0
			for i in 1:m+1
				z = mod(a * z + c, m)
				if haskey(seen, z)
					start = seen[z]; cyc = i - seen[z]; push!(zs, z); break
				end
				seen[z] = i; push!(zs, z)
			end
			(zs = zs, cyc = cyc, start = start, m = m, a = a, c = c)
		end
	end

	function compute_s2_u(s2_n, s2_seq, s2_gen)
		let
			n = s2_n + 2
			Dict("small LCG: a = 7, m = 8191" => () -> lcg_internal(7, 0, 8191, 1, n),
				"minstd: a = 48271, m = 2³¹ − 1" => () -> lcg_internal(48271, 0, 2^31 - 1, 1, n),
				"RANDU: a = 65539, m = 2³¹" => () -> lcg_internal(65539, 0, 2^31, 1, n),
				"Julia default (Xoshiro256++)" => () -> rand(Xoshiro(1), n),
				"the generator of Figure 2.1" => () -> lcg_internal(s2_seq.a, s2_seq.c, s2_seq.m, s2_seq.zs[1], n))[s2_gen]()
		end
	end

	function figure_2_2(s2_u)
		let
			u = s2_u; n = length(u) - 2
			edges = collect(0:0.05:1)
			f = fig(w = 330, h = 320, xr = (0, 1), yr = (0, 1.6), xlabel = "u", ylabel = "density", title = "histogram")
			hist!(f, edges, histdata(u[1:n], edges); c = COL.blue, op = 0.7)
			hline!(f, 1.0; c = COL.red, dash = nothing)
			text!(f, 0.98, 1.0, "flat = uniform"; anchor = "end", dy = -40, size = 11.5, c = COL.red, bold = true, bg = true)
			g = fig(w = 330, h = 320, xr = (0, 1), yr = (0, 1), xlabel = "uᵢ (a number)", ylabel = "uᵢ₊₁ (the next one)", title = "consecutive pairs", frame = true)
			dots!(g, u[1:n], u[2:n+1]; c = COL.ink, r = n > 5000 ? 1.0 : 1.6, op = 0.7)
			np = length(unique(zip(round.(u[1:n], digits = 9), round.(u[2:n+1], digits = 9))))
			cnts = histdata(u[1:n], edges; density = false); exp_c = n / 20
			chi2 = sum((cnts .- exp_c) .^ 2 ./ exp_c)
			txt = Markdown.parse("$(n) numbers give only **$(np)** different pairs" * (np < n ? " (the generator repeats itself)" : "") * ". " * (np < 0.5n ? "Few distinct pairs mean a short cycle: the scatter plot shows the same points again and again." : ""))
			figure(f, g, txt; caption = md"**Figure 2.2.** Left: histogram of the numbers. Right: consecutive pairs ``(u_i, u_{i+1})``.")
		end
	end

	function figure_2_3(s2_g3, s2_yaw, s2_pitch)
		let
			u = Dict("RANDU: a = 65539, m = 2³¹" => () -> lcg_internal(65539, 0, 2^31, 1, 9002), "minstd: a = 48271, m = 2³¹ − 1" => () -> lcg_internal(48271, 0, 2^31 - 1, 1, 9002),
				"Julia default (Xoshiro256++)" => () -> rand(Xoshiro(1), 9002))[s2_g3]()
			P = [u[3i+1:3i+3] .- 0.5 for i in 0:2999]
			ψ, φ = deg2rad(s2_yaw), deg2rad(s2_pitch)
			proj(p) = (x = cos(ψ) * p[1] - sin(ψ) * p[2]; y0 = sin(ψ) * p[1] + cos(ψ) * p[2]; (x, cos(φ) * p[3] + sin(φ) * y0))
			f = fig(w = 400, h = 400, xr = (-0.9, 0.9), yr = (-0.9, 0.9), axes = false, l = 6, r = 6, t = 6, b = 6)
			for (a, b) in [((-1, -1, -1), (1, -1, -1)), ((-1, 1, -1), (1, 1, -1)), ((-1, -1, 1), (1, -1, 1)), ((-1, 1, 1), (1, 1, 1)),
				((-1, -1, -1), (-1, 1, -1)), ((1, -1, -1), (1, 1, -1)), ((-1, -1, 1), (-1, 1, 1)), ((1, -1, 1), (1, 1, 1)),
				((-1, -1, -1), (-1, -1, 1)), ((1, -1, -1), (1, -1, 1)), ((-1, 1, -1), (-1, 1, 1)), ((1, 1, -1), (1, 1, 1))]
				pa = proj(0.5 .* collect(a)); pb = proj(0.5 .* collect(b))
				seg!(f, pa[1], pa[2], pb[1], pb[2]; c = "#c8c8c2", w = 1)
			end
			for (e, lab) in (((1, -1, -1), "uᵢ"), ((-1, 1, -1), "uᵢ₊₁"), ((-1, -1, 1), "uᵢ₊₂"))
				pa = proj(0.5 .* [-1, -1, -1]); pb = proj(0.5 .* collect(e))
				seg!(f, pa[1], pa[2], pb[1], pb[2]; c = "#666", w = 1.6, over = true)
				pm = proj(0.62 .* (collect(e) .- 1) ./ 2)  # the middle of the edge, pushed outwards
				text!(f, pm[1], pm[2], lab; anchor = "middle", size = 13, bold = true, c = "#444", bg = true)
			end
			q = proj.(P)
			dots!(f, first.(q), last.(q); c = COL.blue, r = 1.3, op = 0.6)
			isR = startswith(s2_g3, "RANDU")
			near = abs(s2_yaw - 34) <= 2 && abs(s2_pitch) <= 2
			txt = Markdown.parse(isR ? (near ? "You are looking along the planes: all 3000 triples lie on **15 parallel planes**." : "From this direction RANDU looks uniform. Try rotation 34° and tilt 0°.") : "$(s2_g3) shows no planes visible at this resolution from any direction. Its lattice structure (minstd) or its absence (Xoshiro) is far finer than the dots.")
			figure(f, txt; caption = md"**Figure 2.3.** Three thousand consecutive triples from the chosen generator in the unit cube. From most directions the cloud looks uniform. At rotation 34° and tilt 0° the points form onto **15 parallel planes**.")
		end
	end

	const ALPHAS = Dict("golden ratio: α = (√5 − 1)/2" => ((sqrt(5) - 1) / 2, sqrt(2) - 1), "α = √2 − 1" => (sqrt(2) - 1, sqrt(3) - 1), "a rational α = 0.3" => (0.3, 0.7), "α = 0.01 (small steps)" => (0.01, sqrt(2) - 1))

	function figure_3_1(s3_new, s3_n, s3_al)
		n = s3_n; rng = MersenneTwister(3 + 19s3_new)
		α1, α2 = ALPHAS[s3_al]
		pr = (rand(rng, n), rand(rng, n)); qr = ([mod(k * α1, 1) for k in 1:n], [mod(k * α2, 1) for k in 1:n])
		function mk(p, title, c)
			f = fig(w = 330, h = 330, xr = (0, 1), yr = (0, 1), title = title, frame = true, xticks = [0, 0.5, 1], yticks = [0, 0.5, 1])
			fo = fill(n + 1, 10, 10)
			for i in n:-1:1
				fo[min(10, 1 + floor(Int, 10p[1][i])), min(10, 1 + floor(Int, 10p[2][i]))] = i
			end
			for a in 1:10, b in 1:10
				rect!(f, (a - 1) / 10, (b - 1) / 10, a / 10, b / 10; fill = COL.ochre, op = 0.28, o = (0, fo[a, b] - 1))
			end
			for t in 0.1:0.1:0.9
				seg!(f, t, 0, t, 1; c = "#e4e4df", w = 0.8); seg!(f, 0, t, 1, t; c = "#e4e4df", w = 0.8)
			end
			dots!(f, p[1], p[2]; c = c, r = n > 500 ? 1.6 : 2.6, op = 0.85, s = collect(1:n))
			for i in 1:n
				circle!(f, p[1][i], p[2][i], 6; fill = "none", stroke = COL.blue, sw = 2, o = i, over = true)
			end
			f
		end
		empt(p, k) = (occ = falses(10, 10); for i in 1:k; occ[min(10, 1 + floor(Int, 10p[1][i])), min(10, 1 + floor(Int, 10p[2][i]))] = true; end; 100 - count(occ))
		F = mk(pr, "pseudo-random", COL.ink); G = mk(qr, "additive recurrence", COL.red)
		reads = ["Points: <b>$(k)</b>" * (k == 0 ? "" : " · empty cells of a 10 × 10 grid: pseudo-random <b>$(empt(pr, k))</b>, additive recurrence <b>$(empt(qr, k))</b>") for k in 0:n]
		note = Markdown.parse(occursin("rational", s3_al) ? "With a **rational** step the sequence repeats after 10 points, and all points lie on a few lines: the step must be irrational." :
			occursin("small", s3_al) ? "" :
			"")
		player(F, G, note; n = n, start = n, fps = n > 500 ? 60 : 20, reads = reads, label = "point",
			caption = md"**Figure 3.1.** Pseudo-random points (black) form clusters and leave gaps; the additive recurrence ``u_{k} = k\alpha \bmod 1`` (red, one irrational ``\alpha`` per coordinate) places each new point into a gap.")
	end

	function ms_step(x)
		sq = x^2
		(sq, (sq ÷ 100) % 10_000)
	end

	function figure_2_4(s2_ms)
		x = clamp(s2_ms, 0, 9999); seen = Dict{Int,Int}(); states = Int[x]; seen[x] = 0; rows = []
		fate = ""; first_at = -1
		for k in 1:200
			sq, nx = ms_step(states[end])
			push!(rows, (states[end], sq, nx))
			if haskey(seen, nx)
				first_at = seen[nx]
				fate = nx == 0 ? "dies at 0 after **$(k)** steps" : "falls into a cycle after **$(seen[nx])** steps: from step $(seen[nx]) on, the same $(k - seen[nx]) value$(k - seen[nx] == 1 ? "" : "s") repeat forever"
				push!(states, nx); break
			end
			seen[nx] = k; push!(states, nx)
		end
		fate == "" && (fate = "is still running after 200 steps")
		n = length(states)
		f = fig(w = 680, h = 220, xr = (0, max(n - 1, 10)), yr = (0, 10_000), xlabel = "step", ylabel = "state", title = "von Neumann's middle-square method from seed $(lpad(x, 4, '0'))")
		line!(f, collect(0:n-1), states; c = COL.blue, w = 1.4)
		dots!(f, collect(0:n-1), states; c = COL.blue, r = 2.6)
		if first_at >= 0
			seg!(f, first_at, states[end], n - 1, states[end]; c = COL.red, w = 1.4, dash = "4 3")
			first_at < n - 1 && line!(f, collect(first_at:n-1), states[first_at+1:n]; c = COL.red, w = 2)
			circle!(f, first_at, states[end], 7; fill = "none", stroke = COL.red, sw = 2, over = true)
		end
		dots!(f, [n - 1], [states[end]]; c = COL.red, r = 5.5, over = true)
		shown = rows[1:min(6, length(rows))]
		t = tbl(["state", "squared (8 digits)", "middle four → next state"], [[lpad(a, 4, '0'), HTML(lpad(b, 8, '0')[1:2] * "<b style='color:#c2462e'>" * lpad(b, 8, '0')[3:6] * "</b>" * lpad(b, 8, '0')[7:8]), lpad(c, 4, '0')] for (a, b, c) in shown])
		figure(f, vstack(t, Markdown.parse("This seed " * fate * ".")); caption = md"**Figure 2.4.** Each state is squared, and the middle four of the eight digits become the next state.")
	end

	function figure_5_3(s5_pw, s5_lv, s5_cr)
		f(x) = 0.08 + exp(-((x - 0.35) / s5_pw)^2 / 2)
		grid = range(0, 1, length = 2001); fv = f.(grid); Z = sum(fv) / length(fv); fv ./= Z; fmax = maximum(fv); fn(x) = f(x) / Z
		nl = s5_lv; a = fmax / 2.0^(nl - 1); levels = nl == 1 ? [0.0, fmax] : [0.0; a .* 2.0 .^ (0:nl-1)]
		F = fig(w = 680, h = 300, xr = (0, 1), yr = (0, fmax * 1.12), xlabel = "x", ylabel = "density")
		cols = [COL.sky, COL.blue, COL.purple, COL.green, COL.ochre, COL.teal, COL.pink]
		rect!(F, 0, 0, 1, fmax; fill = "none", op = 1, stroke = COL.red, sw = 1.5)
		rect!(F, 0, 0, 1, fmax; fill = COL.red, op = 0.07, o = 0)
		text!(F, 0.99, fmax * 1.05, "one box around everything"; anchor = "end", size = 11.5, c = COL.red, bg = true)
		boxes = []
		for k in 1:length(levels)-1
			lo, hi = levels[k], levels[k+1]
			inside = fv .> lo; i = 1
			while i <= length(grid)
				if inside[i]
					j = i
					while j < length(grid) && inside[j+1]; j += 1; end
					top = min(hi, maximum(fv[i:j]))
					push!(boxes, (grid[i], lo, grid[j], top, k))
					rect!(F, grid[i], lo, grid[j], top; fill = cols[k], op = 0.35, stroke = cols[k], sw = 0.8, s = k)
					i = j + 1
				else
					i += 1
				end
			end
			(min(hi, fmax) - lo) > 0.09fmax && text!(F, 0.99, (lo + min(hi, fmax)) / 2, "level $(k)"; anchor = "end", size = 11.5, c = cols[k], s = k, bg = true)
		end
		line!(F, grid, fv; c = COL.ink, w = 2)
		areas = [(b[3] - b[1]) * (b[4] - b[2]) for b in boxes]; covered = sum(areas)
		rng = MersenneTwister(5 + s5_cr); D = 25; acc = 0
		cum = cumsum(areas) ./ covered
		for d in 1:D
			k = findfirst(>(rand(rng)), cum); bx = boxes[k]
			x = bx[1] + rand(rng) * (bx[3] - bx[1]); v = bx[2] + rand(rng) * (bx[4] - bx[2]); ok = v <= fn(x); acc += ok
			fr = nl + d
			rect!(F, bx[1], bx[2], bx[3], bx[4]; fill = "none", op = 1, stroke = COL.ink, sw = 2, o = fr)
			dots!(F, [x], [v]; c = ok ? COL.green : COL.red, r = 3.6, s = fr)
		end
		reads = vcat(["The density, and one bounding box: on average $(fmt(fmax)) darts per accepted sample."],
			["Level $(k) added: the boxes cover the parts of the density between $(fmt(levels[k]; d = 2)) and $(fmt(levels[k+1]; d = 2))." for k in 1:nl],
			["Dart $(d): choose a box with probability proportional to its area (black outline), throw a dart into it, accept if below the curve (green)." for d in 1:D])
		txt = Markdown.parse("Total area of the boxes: **$(fmt(covered))**, so on average $(fmt(covered)) darts per sample instead of $(fmt(fmax)) with one box: a saving by a factor **$(fmt(fmax / covered))**.")
		player(F, txt; n = nl + D, start = 0, fps = 2, reads = reads, label = "frame",
			caption = md"**Figure 5.2.** Composition–rejection.")
	end

	function compute_s4_T(s4_g)
		Dict(
			"y = 1 − x" => (g = x -> 1 - x, fy = y -> 0 <= y <= 1 ? 1.0 : 0.0, yr = (0.0, 1.0), ymax = 2.5),
			"y = x²" => (g = x -> x^2, fy = y -> 0 < y <= 1 ? 1 / (2sqrt(y)) : 0.0, yr = (0.0, 1.0), ymax = 5.0),
			"y = √x" => (g = x -> sqrt(x), fy = y -> 0 <= y <= 1 ? 2y : 0.0, yr = (0.0, 1.0), ymax = 2.5),
			"y = x³" => (g = x -> x^3, fy = y -> 0 < y <= 1 ? 1 / (3y^(2 / 3)) : 0.0, yr = (0.0, 1.0), ymax = 6.0),
			"y = −ln(x)" => (g = x -> -log(x), fy = y -> y >= 0 ? exp(-y) : 0.0, yr = (0.0, 5.0), ymax = 1.2),
			"y = tan(π(x − ½))" => (g = x -> tan(π * (x - 0.5)), fy = y -> 1 / (π * (1 + y^2)), yr = (-6.0, 6.0), ymax = 0.45))[s4_g]
	end

	function figure_4_1(sd02_7, s4_T, s4_x0)
		let
			T = s4_T; rng = MersenneTwister(4 + 1009sd02_7)
			x = rand(rng, 20000); y = T.g.(x)
			y0, y1 = T.yr
			cw = 360; side = 200
			c = fig(w = cw, h = 320, xr = (0, 1), yr = (y0, y1), xlabel = "x (uniform)", ylabel = "y = g(x)", l = 52)
			xs = range(0.0005, 0.9995, length = 400)
			line!(c, xs, T.g.(xs); c = COL.ink, w = 2.4)
			for xi in range(0.05, 0.95, length = 10)
				yi = T.g(xi)
				y0 <= yi <= y1 || continue
				seg!(c, xi, y0, xi, yi; c = COL.blue, w = 1.2, op = 0.75)
				seg!(c, 0, yi, xi, yi; c = COL.red, w = 1.2, op = 0.75)
				dots!(c, [xi], [y0]; c = COL.blue, r = 3.6, over = true)
				dots!(c, [0], [yi]; c = COL.red, r = 3.6, over = true)
			end
			edges = collect(range(y0, y1, length = 41))
			h = histdata(y, edges)
			l = fig(w = side, h = 320, xr = (0, T.ymax), yr = (y0, y1), xlabel = "density of Y", l = 40, r = 8)
			for i in 1:40
				rect!(l, 0, edges[i], min(h[i], T.ymax), edges[i+1]; fill = COL.red, op = 0.55, stroke = "white", sw = 0.5)
			end
			ys = range(y0, y1, length = 400)
			line!(l, min.(T.fy.(ys), T.ymax), ys; c = COL.ink, w = 1.8)
			x0 = s4_x0; yv = T.g(x0); hh = 1e-5
			slope = (T.g(x0 + hh) - T.g(x0 - hh)) / (2hh)
			if y0 <= yv <= y1
				δ = 0.03; xa, xb = max(x0 - δ, 1e-4), min(x0 + δ, 1 - 1e-4); ya, yb = minmax(T.g(xa), T.g(xb))
				ya = clamp(ya, y0, y1); yb = clamp(yb, y0, y1)
				rect!(c, xa, y0, xb, y1; fill = COL.ochre, op = 0.2)
				rect!(c, 0, ya, 1, yb; fill = COL.ochre, op = 0.2)
				seg!(c, x0, y0, x0, yv; c = COL.ochre, w = 2.2); seg!(c, 0, yv, x0, yv; c = COL.ochre, w = 2.2)
				dots!(c, [x0], [yv]; c = COL.ochre, r = 5)
				dots!(l, [min(T.fy(yv), T.ymax)], [yv]; c = COL.ochre, r = 5)
				rect!(l, 0, ya, min((xb - xa) / max(yb - ya, 1e-9), T.ymax), yb; fill = COL.ochre, op = 0.5)
			end
			txt = Markdown.parse("At ``x_0 = $(x0)`` the curve has slope ``g'(x_0) = $(fmt(slope))`` and maps ``x_0`` to ``y = $(fmt(yv))``. The orange strip has the same probability on both axes, but on the ``y`` axis it is stretched by the factor ``|g'(x_0)|``. Probability per unit length, the density, is therefore divided by that factor: ``f_Y(y) = 1/|g'(x_0)| = `` **$(fmt(1 / abs(slope)))**. " * (abs(slope) > 1 ? "" : ""))
			figure(l, c, txt; caption = md"**Figure 4.1.** Uniform numbers ``x`` are mapped through ``g``.")
		end
	end

	
	function figure_5_1(s5_new, s5_f, s5_env, s5_N)
		let
			f = s5_f; box = startswith(s5_env, "box"); low = s5_env == "box that is too low"
			g = box ? (x -> 1.0) : (x -> pdf(Beta(2, 4), x))
			grid = range(0.0005, 0.9995, length = 4000)
			μ = (low ? 0.6 : 1.02) * maximum(f(x) / g(x) for x in grid)
			rng = MersenneTwister(51 + 13s5_new); N = s5_N
			ys = box ? rand(rng, N) : rand(rng, Beta(2, 4), N)
			us = rand(rng, N)
			h = us .* μ .* g.(ys); acc = h .<= f.(ys)
			top = max(min(1.08 * maximum(μ * g(x) for x in grid), 4maximum(f.(grid))), 1.08maximum(f.(grid)))
			F = fig(w = 680, h = 316, xr = (0, 1), yr = (0, top), xlabel = "x", ylabel = "height", t = 30)
			area!(F, grid, f.(grid); c = COL.blue, op = 0.12)
			band!(F, grid, f.(grid), max.(f.(grid), min.(μ .* g.(grid), top)); c = COL.red, op = 0.07)
			let mid = [0.15 <= x <= 0.85 for x in grid], xa = grid[argmax(f.(grid))], gapv = (min.(μ .* g.(grid), top) .- f.(grid)) .* mid, xb = grid[argmax(gapv)]
				text!(F, xa, 0.3f(xa), "accepted"; anchor = "middle", size = 13, bold = true, c = COL.blue, bg = true, o = 0)
				maximum(gapv) > 0.15top && text!(F, xb, f(xb) + 0.7 * (min(μ * g(xb), top) - f(xb)), "rejected"; anchor = "middle", size = 13, bold = true, c = COL.red, bg = true, o = 0)
			end
			line!(F, grid, f.(grid); c = COL.blue, w = 2.2)
			line!(F, grid, μ .* g.(grid); c = COL.ink, w = 1.6, dash = "6 3")
			for i in 1:N
				if acc[i]
					dots!(F, [ys[i]], [h[i]]; c = COL.blue, r = 3.2, s = i)
				else
					text!(F, ys[i], h[i], "×"; anchor = "middle", c = COL.red, size = 13, s = i)
				end
			end
			legend!(F, [("target f", COL.blue), ("envelope μg, μ = $(fmt(μ))", COL.ink, :dash)]; at = :top)
			edges = collect(0:0.05:1)
			G = fig(w = 680, h = 180, xr = (0, 1), yr = (0, 1.25maximum(f.(grid))), xlabel = "accepted x", ylabel = "density")
			for i in 1:N
				k = count(acc[1:i]); k == 0 && continue
				hist!(G, edges, histdata(ys[1:i][acc[1:i]], edges); c = COL.blue, op = 0.55, o = i)
			end
			line!(G, grid, f.(grid); c = COL.ink, w = 1.6)
			if low
				cut = [f(x) > μ * g(x) for x in grid]
				area!(F, grid[cut], f.(grid[cut]); c = COL.red, op = 0.25)
			end
			lost = low ? sum(max.(f.(grid) .- μ .* g.(grid), 0)) / length(grid) : 0.0
			code = ["repeat",
				"    draw Y from the proposal g",
				"    draw U, uniform on (0, 1)",
				"    h ← U·μ·g(Y)       (dart height under the envelope)",
				"    if h ≤ f(Y):  accept, output X = Y",
				"    else:         reject, throw again"]
			walk(code, F, G, note; title = "accept–reject", label = "dart", fps = N > 500 ? 40 : 15,
				caption = md"**Figure 5.1.** Accept–reject.")
		end
	end

	function figure_4_2(s4_new, s4_N, s4_d)
	let
		rng = MersenneTwister(42 + 3s4_new); N = s4_N
		D = Dict("exponential (λ = 1)" => (d = Exponential(1), lo = 0.0, hi = 5.0, disc = false),
			"normal (numerical inverse)" => (d = Normal(), lo = -3.5, hi = 3.5, disc = false),
			"a loaded die (discrete)" => (d = Categorical([0.1, 0.1, 0.1, 0.1, 0.2, 0.4]), lo = 0.0, hi = 7.0, disc = true),
			"triangular on [0, 2]" => (d = TriangularDist(0, 2, 1), lo = -0.2, hi = 2.2, disc = false))[s4_d]
		u = rand(rng, N); x = [quantile(D.d, ui) for ui in u]
		xt = D.disc ? (1:6) : nothing
		xs = range(D.lo, D.hi, length = 300)
		fade = N > 50 ? 0.25 : 0.6

		# top: the CDF with the samples
		f = fig(w = 680, h = 280, xr = (D.lo, D.hi), yr = (0, 1), xlabel = "x", ylabel = "u = F(x)", xticks = xt)
		if D.disc
			Fs = [0.0; cumsum(probs(D.d)); 1.0]; bx = [0.0; 1:6; 7.0]
			for k in 1:7
				seg!(f, bx[k], Fs[k], bx[k+1], Fs[k]; c = COL.ink, w = 2.2)
			end
			for k in 1:6
				seg!(f, k, Fs[k], k, Fs[k+1]; c = COL.ink, w = 1, dash = "2 2")
			end
		else
			line!(f, xs, cdf.(D.d, xs); c = COL.ink, w = 2.2)
		end
		for i in 1:N
			seg!(f, D.lo, u[i], x[i], u[i]; c = COL.red, w = 1.4, op = fade)
			seg!(f, x[i], u[i], x[i], 0; c = COL.blue, w = 1.4, op = fade)
			dots!(f, [D.lo], [u[i]]; c = COL.red, r = 3.5, op = fade, over = true)
		end
		# one highlighted example
		ue = 0.7; xe = quantile(D.d, ue)
		arrow!(f, D.lo, ue, xe, ue; c = COL.ink, w = 1.8, dash = "5 4", head = 9)
		arrow!(f, xe, ue, xe, 0.0; c = COL.ink, w = 1.8, dash = "5 4", head = 9)
		text!(f, D.lo, ue, "u"; dx = 10, dy = -10, size = 13, bold = true, c = COL.red, bg = true)
		text!(f, xe, 0.0, "x = F⁻¹(u)"; dx = 8, dy = -12, size = 13, bold = true, c = COL.blue, bg = true)

		# bottom: histogram of the samples against the target
		edges = D.disc ? collect(0.5:1:6.5) : collect(range(D.lo, D.hi, length = 31))
		ymax = D.disc ? 0.6 : 1.25 * maximum(pdf.(D.d, xs))
		g = fig(w = 680, h = 190, xr = (D.lo, D.hi), yr = (0, ymax), xlabel = "x", ylabel = "density", xticks = xt)
		hist!(g, edges, histdata(x, edges); c = COL.blue, op = 0.6)
		if D.disc
			bars!(g, 1:6, probs(D.d); width = 0.25, c = COL.ink, op = 0.9)
		else
			line!(g, xs, pdf.(D.d, xs); c = COL.ink, w = 2)
		end

		figure(f, g; caption = md"**Figure 4.2.** Inversion.")
	end
end
function figure_2_1(s2_seq)
    (; zs, cyc, start, m, a, c) = s2_seq

    L = length(zs) - 1

    f = fig(
        w = 330, h = 330,
        xr = (-1.3, 1.3), yr = (-1.3, 1.3),
        axes = false, l = 4, r = 4, t = 22, b = 4,
        title = "a clock with m = $(m) positions"
    )

    ang(z) = π / 2 - 2π * z / m
    rad = (330 - 26) / 2.6

    circle!(f, 0, 0, rad; fill = "none", stroke = "#e0e0dc", sw = 1)

    lstep = m <= 32 ? 1 : 4

    for z in 0:(m - 1)
        θ = ang(z)

        dots!(f, [cos(θ)], [sin(θ)]; c = "#bdbdb7", r = 2.6)

        if z % lstep == 0
            text!(
                f, 1.15 * cos(θ), 1.15 * sin(θ), string(z);
                anchor = "middle", size = 11, c = "#666"
            )
        end
    end

    for i in 2:length(zs)
        θ0, θ1 = ang(zs[i - 1]), ang(zs[i])
        rep = cyc > 0 && i - 1 == L

        arrow!(
            f,
            0.95 * cos(θ0), 0.95 * sin(θ0),
            0.95 * cos(θ1), 0.95 * sin(θ1);
            c = rep ? COL.red : COL.blue,
            w = 1.2, head = 6, op = 0.45, s = i - 1
        )

        arrow!(
            f,
            0.95 * cos(θ0), 0.95 * sin(θ0),
            0.95 * cos(θ1), 0.95 * sin(θ1);
            c = rep ? COL.red : COL.blue,
            w = 2.6, head = 9, op = 1, o = i - 1
        )

        dots!(
            f, [cos(θ1)], [sin(θ1)];
            c = COL.red, r = 6, o = i - 1, over = true
        )
    end

    θstart = ang(zs[1])

    dots!(
        f, [cos(θstart)], [sin(θstart)];
        c = COL.red, r = 6, o = 0, over = true
    )

    circle!(
        f, cos(θstart), sin(θstart), 0.055;
        fill = "none", stroke = COL.ink, sw = 1.4
    )

    nrow = 13
    ncol = max(5, cld(length(zs), nrow))

    g = fig(
        w = 330, h = 330,
        xr = (0, ncol), yr = (nrow + 0.2, -0.9),
        axes = false, l = 4, r = 4, t = 22, b = 4,
        title = "the sequence (i: zᵢ)"
    )

    for (i, z) in enumerate(zs)
        k = i - 1
        col = k ÷ nrow
        row = k % nrow
        rep = cyc > 0 && k == L

        rect!(
            g, col, row + 0.05, col + 0.95, row + 0.95;
            fill = COL.light, op = 1, rx = 3, o = k
        )

        text!(
            g, col + 0.08, row + 0.5, "$(k): $(z)";
            size = 11, c = rep ? COL.red : COL.ink,
            bold = rep, s = k
        )
    end

    h = fig(
        w = 680, h = 92,
        xr = (0, 1), yr = (0, 1),
        xlabel = "uᵢ = zᵢ / m",
        yticks = Float64[],
        l = 16, b = 46, t = 6
    )

    for (i, z) in enumerate(zs)
        seg!(
            h, z / m, 0.1, z / m, 0.9;
            c = COL.blue, w = 1.6, s = i - 1
        )

        seg!(
            h, z / m, 0.02, z / m, 0.98;
            c = COL.red, w = 3, o = i - 1
        )
    end

    pf(n) = [
        p for p in 2:n
        if n % p == 0 && all(p % q != 0 for q in 2:(p - 1))
    ]

    hd = c != 0 &&
         gcd(c, m) == 1 &&
         all((a - 1) % p == 0 for p in pf(m)) &&
         (m % 4 != 0 || (a - 1) % 4 == 0)

    expl = if cyc == m
        "This is the **full period**: every value ``0, \\dots, $(m - 1)`` occurs once per cycle."
    elseif cyc == m - 1 && c == 0
        ""
    else
        ""
    end

    rule = c == 0 ? "" : (hd ? "" : "")

    txt = Markdown.parse(
        "Seed ``z_0 = $(zs[1])``: " *
        (start > 0 ?
            "after $(start) step(s) the sequence enters a cycle of length **$(cyc)**. " :
            "cycle length **$(cyc)**. ") *
        expl * " " * rule
    )

    reads = [
        "Step <b>$(k)</b>: " *
        (
            k == 0 ?
            "z₀ = $(zs[1]) (the seed)" :
            "z = ($(a)·$(zs[k]) + $(c)) mod $(m) = <b>$(zs[k + 1])</b>, u = $(fmt(zs[k + 1] / m; d = 3))"
        ) *
        (
            cyc > 0 && k == L ?
            " · <span style='color:$(COL.red)'>this value occurred before: the sequence repeats from here</span>" :
            ""
        )
        for k in 0:L
    ]

    player(
        f, g, h, txt;
        n = L, start = L, fps = 2,
        reads = reads, label = "step",
        caption = md"**Figure 2.1.** The linear congruential generator. Try ``a = 5`` (cycle length 3), ``a = 2`` (cycle length 5), or ``m = 32``, ``a = 5``."
    )
end

	function _sweep(x, us)
		n = length(x); K = length(us) - 1
		Fh(u) = count(<=(u), x) / n
		best = 0.0; bu = 0.0; blo = 0.0; bhi = 0.0
		out = NamedTuple[]
		for k in 0:K
			u = us[k+1]
			cands = Tuple{Float64,Float64}[(u, Fh(u))]
			if k > 0
				for (i, xi) in enumerate(x)
					us[k] < xi <= u && (push!(cands, (xi, (i - 1) / n)); push!(cands, (xi, i / n)))
				end
			end
			for (a, v) in cands
				g = abs(v - a)
				g > best + 1e-12 && (best = g; bu = a; blo = min(a, v); bhi = max(a, v))
			end
			push!(out, (u = u, F = Fh(u), best = best, bu = bu, blo = blo, bhi = bhi))
		end
		out
	end

	const DARKRED = "#7a1f12"

	function figure_3_2(s3_dn, s3_dseed)
		n = s3_dn; α = (sqrt(5) - 1) / 2
		xr = sort(rand(MersenneTwister(30 + 1009s3_dseed), n)); xq = sort([mod(k * α, 1) for k in 1:n])
		K = 30; us = collect(range(0, 1, length = K + 1))
		W, H = 680, 344
		f = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
		Yt, Yb = 84.0, 290.0; NL = 50.0
		sw = [_sweep(xr, us), _sweep(xq, us)]
		for (col, x, title, c) in ((1, xr, "pseudo-random points", COL.ink), (2, xq, "additive recurrence", COL.blue))
			ox = col == 1 ? 0.0 : 350.0
			L, R = ox + 14, ox + 232
			px(u) = L + u * (R - L); py(v) = Yb - v * (Yb - Yt)
			S = sw[col]
			text!(f, ox + 2, 14, title; size = 13.5, bold = true, c = c)
			# the number line with the points; [0, u] shaded; points left of u filled
			for k in 1:K
				rect!(f, L, NL - 11, px(us[k+1]), NL + 11; fill = COL.sky, op = 0.35, o = k)
			end
			seg!(f, L, NL, R, NL; c = "#555", w = 1.2)
			seg!(f, L, NL - 6, L, NL + 6; c = "#555", w = 1.2); seg!(f, R, NL - 6, R, NL + 6; c = "#555", w = 1.2)
			rp = n <= 20 ? 4.6 : n <= 50 ? 3.6 : 2.3
			passed(xi) = something(findfirst(u -> u >= xi, us), K + 1) - 1
			dots!(f, px.(x), fill(NL, n); c = c, r = rp, op = 0.25)
			dots!(f, px.(x), fill(NL, n); c = c, r = rp, op = 1.0, s = passed.(x))
			rect!(f, L, Yt, R, Yb; fill = "none", stroke = "#bbb", sw = 1, op = 1)
			seg!(f, px(0), py(0), px(1), py(1); c = "#888", w = 1.4, dash = "5 4")
			line!(f, px.(vcat(0.0, x, 1.0)), py.(vcat(0.0, (1:n) ./ n, 1.0)); c = c, w = 1.6, op = 0.25, steps = true)
			bp = sort(unique(vcat(0.0, x, us)))
			Fh(u) = count(<=(u), x) / n
			for j in 1:length(bp)-1
				a, b = bp[j], bp[j+1]
				seg!(f, px(a), py(Fh(a)), px(b), py(Fh(a)); c = c, w = 2.2, s = passed(b))
			end
			for (i, xi) in enumerate(x)
				seg!(f, px(xi), py((i - 1) / n), px(xi), py(i / n); c = c, w = 2.2, s = passed(xi))
			end
			text!(f, (L + R) / 2, Yb + 14, "position of the boundary u"; anchor = "middle", size = 11.5, c = "#444")
			# the two bars: share of the line, share of the points
			b1, b2 = R + 22, R + 60; bw = 26
			for b in (b1, b2)
				rect!(f, b, Yt, b + bw, Yb; fill = "#f1f1ee", op = 1)
			end
			text!(f, b1 + bw / 2, Yb + 14, "line"; anchor = "middle", size = 11.5, c = "#444")
			text!(f, b2 + bw / 2, Yb + 14, "points"; anchor = "middle", size = 11.5, c = "#444")
			for k in 0:K
				s = S[k+1]; u = s.u; F = s.F
				# the boundary through both panels
				seg!(f, px(u), NL - 18, px(u), Yb; c = COL.ink, w = 1.3, o = k)
				text!(f, px(u), NL - 25, "u"; anchor = "middle", size = 12.5, italic = true, bold = true, o = k)
				# guides from the diagonal and the staircase to the bars
				seg!(f, px(u), py(u), b1, py(u); c = "#999", w = 1, dash = "2 3", o = k)
				seg!(f, px(u), py(F), b2, py(F); c = c, w = 1, dash = "2 3", op = 0.7, o = k)
				rect!(f, b1, py(u), b1 + bw, Yb; fill = COL.sky, op = 0.85, o = k)
				rect!(f, b2, py(F), b2 + bw, Yb; fill = c, op = 0.75, o = k)
				lo, hi = minmax(u, F)
				if hi - lo > 1e-9
					bx = F > u ? b2 : b1
					rect!(f, bx - 2, py(hi), bx + bw + 2, py(lo); fill = COL.red, op = 0.9, o = k, over = true)
					seg!(f, b1, py(lo), b2 + bw, py(lo); c = COL.red, w = 1.4, o = k, over = true)
				end
				if s.best > 1e-9
					seg!(f, px(s.bu), py(s.blo), px(s.bu), py(s.bhi); c = DARKRED, w = 6, o = k, over = true)
					k == K && text!(f, px(s.bu) + (s.bu > 0.8 ? -9 : 9), py((s.blo + s.bhi) / 2), "Dₙ"; anchor = s.bu > 0.8 ? "end" : "start", size = 14, bold = true, c = DARKRED, bg = true, o = k)
				end
				hi - lo > 1e-9 && seg!(f, px(u), py(lo), px(u), py(hi); c = COL.red, w = 3, o = k, over = true)
			end
		end
		# legend
		ly = H - 12.0; lx = 14.0
		for (lab, c, w, d) in (("difference at the boundary", COL.red, 3, nothing), ("largest difference so far", DARKRED, 6, nothing), ("diagonal: share of the line", "#888", 1.4, "5 4"))
			seg!(f, lx, ly, lx + 22, ly; c = c, w = w, dash = d)
			text!(f, lx + 28, ly, lab; size = 11.5, c = "#333")
			lx += 40 + 6.3 * length(lab)
		end
		# the reads: words only
		thr = max(1.5 / n, 0.02)
		function phrase(s, prev)
			d = s.F - s.u
			p = d > thr ? "" :
				d < -thr ? "" : ""
			s.best > prev.best + 0.2 / n ? p * "; a new largest difference" : p
		end
		dr, dq = sw[1][end].best, sw[2][end].best
		reads = String[""]
		for k in 1:K-1
			push!(reads, "<b>Left:</b> " * phrase(sw[1][k+1], sw[1][k]) * ". <b>Right:</b> " * phrase(sw[2][k+1], sw[2][k]) * ".")
		end
		push!(reads, "Discrepancy Dₙ: " *
			(dr > 1.4dq ? "much larger for the pseudo-random points." : dq > 1.4dr ? "here larger for the additive recurrence." : "here similar on both sides."))
		ns = round.(Int, 10 .^ range(1, 4, length = 25))
		dp = [mean(discrepancy1(rand(MersenneTwister(k), m)) for k in 1:10) for m in ns]
		da = [discrepancy1([mod(k * α, 1) for k in 1:m]) for m in ns]
		Hf = fig(w = 680, h = 230, xr = (10, 1e4), yr = (1e-4, 0.6), xlog = true, ylog = true, xlabel = "number of points n (logarithmic scale)", ylabel = "Dₙ (logarithmic)",
			xticks = Float64[], yticks = Float64[], r = 150, title = "the discrepancy for more and more points")
		line!(Hf, ns, 0.9 ./ sqrt.(ns); c = COL.ink, w = 1.2, dash = "5 4", op = 0.7); line!(Hf, ns, 1.0 ./ ns; c = COL.blue, w = 1.2, dash = "5 4", op = 0.7)
		line!(Hf, ns, dp; c = COL.ink, w = 2.4); line!(Hf, ns, da; c = COL.blue, w = 2.4)
		text!(Hf, 1e4, dp[end], "pseudo-random"; dx = 8, size = 12, c = COL.ink, bold = true)
		text!(Hf, 1e4, da[end], "additive recurrence"; dx = 8, size = 12, c = COL.blue, bold = true)
		text!(Hf, 10^3.1, 0.9 / sqrt(10^3.1), "∝ 1/√n"; dy = -12, size = 12, c = COL.ink, bg = true)
		text!(Hf, 10^3.4, 1 / 10^3.4, "∝ 1/n"; dy = 14, anchor = "middle", size = 12, c = COL.blue, bg = true)
		seg!(Hf, n, 1e-4, n, 0.6; c = "#aaa", w = 1, dash = "2 3")
		text!(Hf, n, 0.45, "n chosen above"; dx = 4, size = 11, c = "#666")
		dots!(Hf, [n, n], [dr, dq]; c = [COL.ink, COL.blue], r = 5, op = 1, stroke = DARKRED, sw = 2)
		txt = Markdown.parse("For these $(n) points: ``D_n`` = $(fmt(dr; d = 3)) for the pseudo-random points and $(fmt(dq; d = 3)) for the additive recurrence.")
		player(f, Hf, txt; n = K, start = 0, fps = 1.2, reads = reads, label = "step",
			caption = md"**Figure 3.2.** The discrepancy.")
	end

	function figure_4_3(s4_bm, s4_nsum)
		rng = MersenneTwister(1958 + 7s4_bm); n = 2000; ns_ = s4_nsum
		u1 = rand(rng, n); u2 = rand(rng, n)
		r = sqrt.(-2 .* log.(u1)); x1 = r .* cos.(2π .* u2); x2 = r .* sin.(2π .* u2)
		bandc = [COL.red, COL.ochre, COL.green, COL.blue, COL.purple]
		band = min.(5, 1 .+ floor.(Int, 5 .* u1)); cols = bandc[band]
		f = fig(w = 330, h = 328, xr = (0, 1), yr = (0, 1), xlabel = "U₁", ylabel = "U₂", title = "uniform pairs (U₁, U₂)", frame = true, xticks = 0:0.2:1)
		dots!(f, u1, u2; c = cols, r = 1.8, op = 0.75)
		g = fig(w = 330, h = 328, xr = (-4, 4), yr = (-4, 4), xlabel = "X₁", ylabel = "X₂", title = "after the transform", frame = true)
		dots!(g, x1, x2; c = cols, r = 1.8, op = 0.75)
		pxu = (330 - 62 - 16) / 8
		for q in 0.2:0.2:0.8
			circle!(g, 0, 0, sqrt(-2log(q)) * pxu; fill = "none", stroke = "#888", sw = 0.8)
		end
		edges = collect(-4:0.25:4)
		hx = histdata(vcat(x1, x2), edges)
		h1 = fig(w = 330, h = 250, xr = (-4, 4), yr = (0, 0.5), xlabel = "x", ylabel = "density", title = "Box–Muller samples (X₁ and X₂)")
		hist!(h1, edges, hx; c = COL.blue, op = 0.55)
		zz = range(-4, 4, length = 200); line!(h1, zz, pdf.(Normal(), zz); c = COL.ink, w = 2)
		clt = [(sum(rand(rng, ns_)) - ns_ / 2) / sqrt(ns_ / 12) for _ in 1:200_000]
		bm = let a = rand(rng, 100_000), b = rand(rng, 100_000), rr = sqrt.(-2 .* log.(a)); vcat(rr .* cos.(2π .* b), rr .* sin.(2π .* b)) end
		zs = 2.0:0.25:6.5
		h2 = fig(w = 330, h = 250, xr = (2, 6.5), yr = (1e-10, 0.05), ylog = true, xlabel = "z", ylabel = "P(X > z)", title = "upper tail")
		line!(h2, zs, ccdf.(Normal(), zs); c = COL.ink, w = 2.2)
		line!(h2, zs, [(cc = count(>(z), clt) / length(clt); cc > 0 ? cc : NaN) for z in zs]; c = COL.red, w = 2)
		line!(h2, zs, [(cc = count(>(z), bm) / length(bm); cc > 0 ? cc : NaN) for z in zs]; c = COL.blue, w = 1.6, dash = "4 3")
		sqrt(3ns_) < 6.5 && vline!(h2, sqrt(3ns_); c = COL.red, dash = "2 3", w = 1)
		legend!(h2, [("exact normal", COL.ink), ("sum of $(ns_) uniforms", COL.red), ("Box–Muller (sample)", COL.blue, :dash)]; at = :bl, size = 10.5)
		figure(f, g, h1, h2; caption = md"**Figure 4.3.** The Box–Muller transform. Top left: uniform pairs, coloured by stripes of ``U_1``. Top right: the transformed pairs; each stripe becomes a ring. Bottom left: histogram of the transformed values and the standard normal density. Bottom right: the probability of exceeding ``z`` (logarithmic axis) for the exact normal distribution, the standardised sum of uniform numbers, and a Box–Muller sample of size ``2 \times 10^5``.")
	end

end;

# ╔═╡ 5444c240-a48f-ec8f-d971-a6521da0e78e
s2_seq = compute_s2_seq(s2_m, s2_a, s2_c, s2_z0);

# ╔═╡ 24cca1a6-d583-cbd0-012f-825f4257eddb
figure_2_1(s2_seq)

# ╔═╡ 28965f63-a7e8-c132-56e0-530b56e17918
s2_u = compute_s2_u(s2_n, s2_seq, s2_gen);

# ╔═╡ 1efbc5c8-0565-0447-bd14-3324ce777312
figure_2_2(s2_u)

# ╔═╡ 61e6a5b8-dc38-dc93-79dc-f370b285623b
figure_2_3(s2_g3, s2_yaw, s2_pitch)

# ╔═╡ d57f738c-6e7b-bcac-045d-e5deab103301
figure_2_4(s2_ms)

# ╔═╡ 78af998f-eec2-6a88-3ccb-5654c70446c4
# 
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

	faq(pairs::Pair...; head = "Questions and answers") = @htl("""$([@htl("<div style='margin-top:0.8em'><b>$(first(p))</b></div><div class='admonition hint'><p class='admonition-title'>Answer</p>$(last(p))</div>") for p in pairs])""")

	reveal(body; title = "Answer") = Markdown.MD(Markdown.Admonition("hint", title, Any[body]))

	proof(show::Bool, title, body) = @htl("""<details class="proof" open=$(show)><summary>$(title)</summary>$(body)</details>""")
end;

# ╔═╡ a8a87ed0-feb1-41d9-9907-6cc7a0a76748
mcq(md"Which statement about the function `rand()` in Julia, Python or C is correct?",
	md"It measures physical noise in the processor, so its numbers are truly random." => md"Hardware random number generators exist, but `rand()` uses an algorithm. Section 1 explains why simulations prefer algorithms.",
	md"It computes each number from the previous state by a fixed rule, so the whole sequence is determined by one starting value." => md"Right. This is a **pseudo-random number generator**; the starting value is the **seed**. The same seed reproduces the same sequence.",
	md"It reads the system clock for every number." => md"The clock is sometimes used once to choose a seed. Reading it for every number would be slow, and consecutive values would be strongly correlated.",
	md"It reads from a stored table of random digits." => md"Tables of random digits were used before computers, but they are very less for modern simulations.";
	correct = 2, head = "Try!")

# ╔═╡ 2c4cb982-6aef-9d22-2935-a5834f22c791
reveal(md"""No. A pseudo-random generator is deterministic: the seed fixes the whole sequence. Identical results with the same seed are what makes a simulation reproducible, so that a result can be checked. For independent repetitions, each run needs a different seed.""")

# ╔═╡ fb9e8659-2224-9e13-42d7-7bc05b5c108e
faq(md"Is it acceptable to use simple seeds such as 1, 2, 3?" =>
		md"For modern generators, yes: the seed is mixed into a large internal state, so that neighbouring seeds produce unrelated sequences. For simple generators, such as a multiplicative LCG, it is not.",
	md"Why are these generators unsuitable for cryptography?" =>
		md"Cryptography requires unpredictability. The state of an LCG or of Xoshiro can be reconstructed from a few outputs. Cryptographically secure generators use different constructions and are slower.";
	head = "Questions and answers")

# ╔═╡ a1d22a6e-25ee-29a9-e48f-6f69ed3d0569
mcq(md"The steps of a random walk are generated with the additive recurrence instead of a pseudo-random generator. What is the consequence?",
	md"None: the steps are uniformly distributed." => md"Uniformity of each step is not sufficient. Consecutive values differ by the fixed step ``\alpha``; the steps are strongly dependent.",
	md"The walk is too regular and does not have the statistical properties of a random walk." => md"Right. A random walk requires *independent* steps. Quasi-random sequences sacrifice independence for evenness.",
	md"The simulation becomes slower." => md"The recurrence is cheap to compute. The problem is statistical, not computational.";
	correct = 2, head = "Common pitfall")

# ╔═╡ 508bc972-b3b8-13dd-d970-f77257a6ce06
numq(md"``X`` is uniform on ``(0, 1)`` and ``Y = X^2``. Compute ``f_Y(0.25)``.", 1.0; rtol = 0.01,
	why = md"``x = g^{-1}(0.25) = 0.5`` and ``g'(0.5) = 1``, so ``f_Y(0.25) = 1/1 = 1``.", head = "Check your understanding")

# ╔═╡ f99f6bc0-62f9-0758-1931-52aa4c32d169
mcq(md"A student transforms a uniform ``X`` on ``(0,1)`` with ``Y = 2X`` and states: *the density of ``Y`` is ``f_Y(y) = f_X(y/2) = 1`` on ``(0, 2)``.* What is wrong?",
	md"Nothing." => md"The total probability would then be 2: the area under ``f_Y = 1`` on ``(0, 2)``.",
	md"The factor ``|d g^{-1}/dy| = 1/2`` is missing; the correct density is ``1/2`` on ``(0, 2)``." => md"Right. Stretching the interval by a factor 2 must halve the density, so that the total probability remains 1. Checking that a density integrates to 1 detects this error.",
	md"``Y`` is not uniform." => md"A linear transformation of a uniform variable is uniform, on the transformed interval.";
	correct = 2, head = "Common pitfall")

# ╔═╡ 3412ebf4-1b2a-9694-f05e-8091f194b638
reveal(md"""The inverse of the CDF ``F(x) = 1 - e^{-\lambda x}`` is ``x = -\ln(1 - u)/\lambda``, so the program should divide by ``\lambda``, not multiply. Its waiting times are 100 times too long, but it has the right *shape*. The check: compare the sample mean with the known expected value ``1/\lambda = 6`` minutes.""")

# ╔═╡ 36ae7acb-7533-1461-7905-65f2bf069dcd
reveal(md"""Each stripe has probability ``1/5``. Near the origin the normal density is high, so a thin ring already contains probability ``1/5``; far out the density is low, and a wide ring is needed. The radius is ``R = \sqrt{-2\ln U_1}``. If ``U_1 \ge 10^{-6}``, then ``R \le \sqrt{-2\ln 10^{-6}} = 5.26``: no normal number beyond ``5.26`` in absolute value is ever produced. The tails are determined by the smallest uniform numbers, so the resolution of the uniform generator near 0 matters.""")

# ╔═╡ 22d2a1d7-093f-69c1-af0b-7b93e2c687bf
reveal(md"""The output is not wrong: as long as the envelope ``\mu g`` is above the target everywhere. The method is merely wasteful, since on average ``\mu = 1000`` proposals are needed per sample. A proposal shaped more like the target reduces the waste. The dangerous error is when  an envelope that is too low somewhere, that is, ``\mu g(x) < f(x)`` for some ``x``. The algorithm then still runs and accepts many points, but the samples do not have density ``f``.
.""")

# ╔═╡ 7f62b5fe-9a6b-4628-9007-9ace2ab794fd
numq(md"The density of Beta(2, 5) has its maximum 2.4576 at ``x = 0.2``. With the box ``[0, 1] \times [0, 2.4576]``, what fraction of darts is accepted?", 1 / 2.4576; rtol = 0.01,
	why = md"The box has area 2.4576, the region under the density has area 1; the acceptance probability is ``1/2.4576 = 0.41``. On average, 2.46 darts are needed per sample.", head = "Check your understanding")

# ╔═╡ 6d72710d-0f60-e3c2-0bc9-c7940066efbe
mcq(md"An envelope is chosen too low, so that ``\mu g(x) < f(x)`` on some interval. What happens?",
	md"The program stops with an error." => md"Nothing in the algorithm checks the envelope condition; the program runs normally.",
	md"The samples follow ``\min(f, \mu g)`` (renormalised) instead of ``f``: the density is cut off where the envelope is too low." => md"Right. Where ``f > \mu g``, every proposal is accepted. The condition ``\mu g \ge f`` must be verified analytically.";	correct = 2, head = "Common pitfall")

# ╔═╡ 0ab47deb-7ad2-92d4-be52-3accfb8164f9
reveal(md"""The set-up cost is paid once. For ``10^9`` samples from a fixed density, the set-up cost is negligible and the higher acceptance rate dominates. For a density that changes after every sample, the boxes must be recomputed each time, and the set-up cost can exceed the saving. In the stochastic simulation of reaction networks (see the notebook on stochastic reaction networks), the distribution changes after every reaction.""")

# ╔═╡ 56132ccc-d75a-65ef-9807-9eb422af8ba5
mcq(md"A simulation needs ``10^6`` uniform numbers. An LCG with ``m = 2^{16}`` and full period is used. A student argues: *the generator produces every value exactly once per cycle, so the numbers are perfectly uniform; repeating the cycle about 15 times does not change that.* What is wrong with the argument?",
	md"Nothing; uniformity is all that matters." => md"Uniformity is not sufficient. The simulation must also behave as if the numbers were independent.",
	md"The simulation contains only 65 536 different numbers, each used about 15 times in the same order; the results are not those of ``10^6`` independent numbers." => md"Right. Repeating the cycle reproduces the same results 15 times; the effective sample size is ``65\,536``. The cycle length must be much larger than the number of draws.",
	md"An LCG with ``m = 2^{16}`` cannot have full period." => md"With ``c \neq 0`` it can. The problem lies elsewhere.";
	correct = 2, head = "Common pitfall")

# ╔═╡ 02e182b6-f19b-c32a-aa3d-2abbf67bea57
# Controls
begin
	Seed(default::Integer = 0; label = "New random numbers") = CounterButton(label)

	struct BarEditor
		values::Vector{Float64}
		labels::Vector{String}
		ymax::Float64
		width::Int
		color::String
	end

	BarEditor(v; labels = string.(1:length(v)), ymax = 2maximum(v), width = 480, color = COL.blue) =
		BarEditor(Float64.(v), string.(labels), Float64(ymax), width, color)

	AbstractPlutoDingetjes.Bonds.initial_value(b::BarEditor) = b.values

	function Base.show(io::IO, m::MIME"text/html", b::BarEditor)
		show(io, m, @htl("""
		<span class="bar-editor" style="display:inline-block;touch-action:none">
		<svg viewBox="0 0 $(b.width) 210" width="$(b.width)" style="max-width:100%;height:auto;background:#fff;border:1px solid #e3e3df;border-radius:4px;cursor:ns-resize;font-family:system-ui,sans-serif"></svg>
		<script>
		const span = currentScript.parentElement;
		const svg = span.querySelector("svg");
		const labels = $(b.labels), ymax = $(b.ymax), col = $(b.color);
		let vals = $(b.values);
		const W = $(b.width), H = 210, L = 14, R = 10, T = 22, Bm = 28;
		const n = vals.length, bw = (W - L - R) / n;
		const NS = "http://www.w3.org/2000/svg";
		function el(tag, attrs, text) {
			const e = document.createElementNS(NS, tag);
			for (const k in attrs) e.setAttribute(k, attrs[k]);
			if (text !== undefined) e.textContent = text;
			return e;
		}
		function yOf(v) { return T + (H - T - Bm) * (1 - v / ymax); }
		function draw() {
			while (svg.firstChild) svg.removeChild(svg.firstChild);
			svg.appendChild(el("line", {x1: L, x2: W - R, y1: H - Bm, y2: H - Bm, stroke: "#333", "stroke-width": 0.9}));
			for (let i = 0; i < n; i++) {
				const x = L + i * bw + 0.14 * bw, y = yOf(vals[i]);
				svg.appendChild(el("rect", {x: x, y: y, width: 0.72 * bw, height: H - Bm - y, fill: col, opacity: 0.8}));
				svg.appendChild(el("rect", {x: x, y: y - 3, width: 0.72 * bw, height: 6, fill: "#1b1b1b", opacity: 0.55, rx: 2}));
				svg.appendChild(el("text", {x: x + 0.36 * bw, y: H - Bm + 17, "text-anchor": "middle", "font-size": 12.5, fill: "#222"}, labels[i]));
				svg.appendChild(el("text", {x: x + 0.36 * bw, y: y - 9, "text-anchor": "middle", "font-size": 11, fill: "#555"}, vals[i].toFixed(1)));
			}
			svg.appendChild(el("text", {x: L, y: 12, "font-size": 11, fill: "#777"}, "drag the bars up or down"));
		}
		let dragging = false;
		function setFrom(ev) {
			const r = svg.getBoundingClientRect();
			const x = (ev.clientX - r.left) * W / r.width, y = (ev.clientY - r.top) * H / r.height;
			const i = Math.floor((x - L) / bw);
			if (i < 0 || i >= n) return;
			const v = Math.max(0, Math.min(ymax, ymax * (1 - (y - T) / (H - T - Bm))));
			vals[i] = Math.round(v * 10) / 10;
			draw();
		}
		function send() { span.value = vals.slice(); span.dispatchEvent(new CustomEvent("input")); }
		svg.addEventListener("pointerdown", (ev) => { dragging = true; svg.setPointerCapture(ev.pointerId); setFrom(ev); });
		svg.addEventListener("pointermove", (ev) => { if (dragging) setFrom(ev); });
		svg.addEventListener("pointerup", (ev) => { if (dragging) { dragging = false; send(); } });
		span.value = vals.slice();
		draw();
		</script>
		</span>"""))
	end
end;

# ╔═╡ 9a6a4aef-0e5c-4e7d-c318-c2eafc686c90
md"""
## 3. Quasi-random numbers

pseudo-random numbers are not evenly spread: they form clusters and leave gaps. For some applications, in particular numerical integration (see the notebook on Monte Carlo integration), points that cover the domain evenly are preferable. **Quasi-random** sequences are used for this purpose. A simple method is the **additive recurrence**

```math
u_{k+1} = (u_k + \alpha) \bmod 1, \qquad\text{that is,}\qquad u_k = k\alpha \bmod 1,
```

with an **irrational** step ``\alpha``, for example ``\alpha = (\sqrt5 - 1)/2 \approx 0.618``. Each new value is the previous one shifted by ``\alpha`` and wrapped around at 1. Because ``\alpha`` is irrational, no value is repeated, and each new value falls into one of the largest remaining gaps.

Points $(@bind s3_n Select([50, 200, 1000]; default = 200)) ``\qquad`` step $(@bind s3_al Select(["golden ratio: α = (√5 − 1)/2", "α = √2 − 1", "a rational α = 0.3", "α = 0.01 (small steps)"]; default = "golden ratio: α = (√5 − 1)/2")) ``\qquad`` $(@bind s3_new Seed(0; label = "New pseudo-random points"))
"""

# ╔═╡ 4be643c5-0792-ea8f-e5d6-d3987c8f5ff8
figure_3_1(s3_new, s3_n, s3_al)

# ╔═╡ 310e008c-3e96-fd0c-8391-ab26fcc75de0
md"""
### 3.1 Measuring evenness: the discrepancy

How evenly a set of points ``x_1, \dots, x_n`` in ``[0, 1]`` is distributed can be measured by comparing two quantities for every interval ``[0, u]``: the **fraction of points** in the interval, ``\hat F_n(u) = \frac1n \#\{i : x_i \le u\}``, and the **length** of the interval, ``u``. For perfectly even points these two should match. The **discrepancy** is the largest disagreement:

```math
D_n = \max_{0 \le u \le 1} \big|\hat F_n(u) - u\big| .
```

Geometrically, ``\hat F_n`` is a staircase that rises by ``1/n`` at every point, and ``D_n`` is the largest vertical distance between this staircase and the diagonal. Figure 3.2 shows both for a small number of points.

Number of points $(@bind s3_dn Select([10, 20, 50, 200]; default = 20)) ``\qquad`` $(@bind s3_dseed Seed(0; label = "New pseudo-random points"))
"""

# ╔═╡ 77aac2d7-d01d-4a1d-b236-fb6a0bcd27b7
figure_3_2(s3_dn, s3_dseed)

# ╔═╡ 9f590e8d-bfa7-eb57-a9e3-74de2caddaa2
md"""
## 4. Transformation of random variables

Above generators produce uniform numbers, however, simulations need random numbers from different distributions. They are obtained by **transforming** uniform random numbers. The basic question is: if ``Y = g(X)``, what is the distribution of ``Y``?

### 4.1 The change-of-variables formula

For a discrete random variable, ``p_Y(y)`` is the sum of the probabilities of all ``x`` with ``g(x) = y``. For a continuous random variable, the transformation **stretches** some regions and **compresses** others. In Figure 4.1, 20 000 uniform numbers ``x`` are mapped through ``g``; the histogram of the images ``y`` is drawn on the left. The orange strip has the same probability on both axes, but has a different width.

**What ``Y = g(X)`` means.** ``X`` is a random number, for example a uniform number from the generator. ``g`` is an ordinary function, a fixed rule such as ``g(x) = x^2``. Each time ``X`` is drawn, the rule is applied to the drawn value: ``x = 0.3`` gives ``y = g(0.3) = 0.09``, ``x = 0.8`` gives ``y = 0.64``. The results form a new random number ``Y``. 

Transformation ``g`` $(@bind s4_g Select(["y = 1 − x", "y = x²", "y = √x", "y = x³", "y = −ln(x)", "y = tan(π(x − ½))"]; default = "y = x²")) ``\qquad`` strip at ``x_0 =`` $(@bind s4_x0 Slider(0.02:0.02:0.98; default = 0.3, show_value = true)) ``\qquad`` $(@bind sd02_7 Seed(0))
"""

# ╔═╡ 271e7970-5cd5-bc07-6a0a-9dc22bf4d463
s4_T = compute_s4_T(s4_g);

# ╔═╡ 4864570d-ef47-783e-e9dd-813d496a1eaf
figure_4_1(sd02_7, s4_T, s4_x0)

# ╔═╡ 193cb405-461e-e7bb-dd24-16803eee65e4
md"""
### 4.2 The inversion method

The most important transformation uses the CDF of the target distribution.

!!! theorem "Inversion transform"
    Let ``F`` be a continuous, strictly increasing CDF. If ``U`` is uniform on ``(0, 1)``, then ``X = F^{-1}(U)`` has the CDF ``F``.

Proof: ``P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)``. Geometrically: draw the CDF, choose ``u`` uniformly on the vertical axis, go horizontally to the curve, and read off ``x``. Where the CDF is steep, a long stretch of ``u`` values is mapped onto a short stretch of ``x``, so that many samples fall there. For a discrete distribution (a step function), ``F^{-1}`` is replaced by the quantile function.

Distribution $(@bind s4_d Select(["exponential (λ = 1)", "normal (numerical inverse)", "a loaded die (discrete)", "triangular on [0, 2]"]; default = "exponential (λ = 1)")) ``\qquad`` samples $(@bind s4_N Slider(1:300; default = 40, show_value = true)) ``\qquad`` $(@bind s4_new Seed(0; label = "New samples"))
"""

# ╔═╡ f8322827-12dc-4a4d-a768-30de9492f047
	figure_4_2(s4_new, s4_N, s4_d)


# ╔═╡ 9603dd02-f29b-43e7-c146-ad54f169b0f0
md"""
### 4.3 Normally distributed random numbers

The CDF of the normal distribution, ``\Phi``, has no closed-form inverse, so a different method is needed.

**Sums of uniform numbers.** By the central limit theorem, a sum of many independent uniform numbers is approximately normal. Each uniform number has mean ``1/2`` and variance ``1/12``; the sum of ``n`` of them has mean ``n/2`` and variance ``n/12``. The common choice ``n = 12`` gives a variance of exactly 1, and

```math
X = U_1 + U_2 + \dots + U_{12} - 6
```

has mean 0 and variance 1. The method has a fundamental limitation: since each ``U_i`` lies between 0 and 1, the sum lies between 0 and 12, and **``X`` can never be smaller than ``-6`` or larger than ``+6``**. A standard normal variable exceeds ``6`` in absolute value with probability ``2 \cdot 10^{-9}``. This seems negligible, but a simulation that draws ``10^{10}`` normal numbers should contain about 20 such values, and a study of rare events (failures, extreme losses) depends on these values. **In addition, each normal number requires twelve uniform numbers.**

**The Box–Muller transform** (G. E. P. Box and M. E. Muller, 1958) instead uses two uniform numbers for two normal numbers. The basic idea is: a pair of independent normal numbers is a random point in the plane. Describe that point by its distance from the origin and its direction, and it turns out that both can be generated from uniform numbers by inversion.

It follows from four steps.

1. Two independent standard normal variables ``X_1, X_2`` have the joint density ``f(x_1, x_2) = \frac{1}{2\pi}\, e^{-(x_1^2 + x_2^2)/2}``. It depends only on the distance from the origin.
2. In polar coordinates ``x_1 = r\cos\theta``, ``x_2 = r\sin\theta`` (with the Jacobian ``r``), the density becomes
   ```math
   f(r, \theta) = \underbrace{\frac{1}{2\pi}}_{\text{angle}} \cdot \underbrace{r\,e^{-r^2/2}}_{\text{radius}}, \qquad r \ge 0,\; 0 \le \theta < 2\pi .
   ```
   It is a product of a function of ``\theta`` and a function of ``r``, so the angle ``\Theta`` and the radius ``R`` are independent, and ``\Theta`` is uniform on ``[0, 2\pi)``.
3. The radius has the CDF ``P(R \le r) = 1 - e^{-r^2/2}``. Inversion gives ``R = \sqrt{-2\ln(1 - U_1)}``, or equivalently ``R = \sqrt{-2 \ln U_1}``. The angle is ``\Theta = 2\pi U_2``.
4. Returning to Cartesian coordinates:
   ```math
   X_1 = \sqrt{-2 \ln U_1}\,\cos(2\pi U_2), \qquad X_2 = \sqrt{-2 \ln U_1}\,\sin(2\pi U_2).
   ```

In Figure 4.3, the unit square of ``(U_1, U_2)`` is divided into five vertical stripes of equal probability. Each stripe is mapped onto a ring: the stripe with the smallest ``U_1`` becomes the outermost ring, which produces the tails of the normal distribution.

| stripe of ``U_1`` | radius ``R = \sqrt{-2\ln U_1}`` |
|:---:|:---:|
| ``[0.8, 1)`` | ``0`` to ``0.67`` (inner disk) |
| ``[0.6, 0.8)`` | ``0.67`` to ``1.01`` |
| ``[0.4, 0.6)`` | ``1.01`` to ``1.35`` |
| ``[0.2, 0.4)`` | ``1.35`` to ``1.79`` |
| ``[0, 0.2)`` | ``1.79`` to ``\infty`` (outermost ring) |



Uniform numbers summed in the first method $(@bind s4_nsum Select([3, 6, 12, 48]; default = 12)) ``\qquad`` $(@bind s4_bm Seed(0; label = "New points"))
"""

# ╔═╡ ae941370-1645-0cf4-ad96-8e8293380611
figure_4_3(s4_bm, s4_nsum)

# ╔═╡ f842c511-8694-0756-2ac8-9b38e9a1f5c5
md"""
## 5. Accept–reject methods

Inversion requires the inverse CDF. Often only the density ``f`` is known, sometimes only up to a constant factor. The **accept–reject** method is used in such cases. It is based on geometry.

The probability that ``X`` is near a value ``x`` is proportional to the height ``f(x)`` of the graph there. So if we choose points uniformly under the graph of ``f``, their ``x``-coordinates have density ``f``.

To get such points, we first choose points uniformly in a larger region that contains the graph, and then reject the point that lies above the graph. For the larger region we take the area under ``\mu g``, where ``g`` is a density that is easy to sample and ``\mu \ge 1`` is a constant with ``\mu g(x) \ge f(x)`` for all ``x``. The function ``\mu g`` is called the **envelope**:

1. draw ``Y \sim g`` and ``U \sim \mathcal U(0, 1)``;
2. accept ``X = Y`` if ``U \mu g(Y) \le f(Y)``; otherwise reject and return to step 1.

Each attempt is accepted with probability ``1/\mu``, the area under ``f`` (equal to 1) divided by the area under the envelope (equal to ``\mu``).

Target density $(@bind s5_t Select(["Beta(2, 5): a skewed bump", "two bumps", "my drawing (bars below)"]; default = "Beta(2, 5): a skewed bump")) ``\qquad`` envelope $(@bind s5_env Select(["box (uniform proposal)", "Beta(2, 4) proposal", "box that is too low (invalid)"]; default = "box (uniform proposal)")) ``\qquad`` darts $(@bind s5_N Select([100, 300, 1000]; default = 300)) ``\qquad`` $(@bind s5_new Seed(0; label = "New darts"))
"""

# ╔═╡ 724fc1f2-1165-d86b-3bc2-0173701d6709
@bind s5_draw BarEditor([0.5, 1.5, 3, 3.5, 2, 1, 0.6, 1.2, 2.5, 1.5, 0.6, 0.3]; labels = fill("", 12), ymax = 5, width = 420, color = COL.ochre)

# ╔═╡ 4cb65688-cdc1-ca0e-b0e2-62ba60f0b859
s5_f = compute_s5_f(s5_t, s5_draw);

# ╔═╡ d3011dbd-20ec-86df-f69f-25c7799cade1
figure_5_1(s5_new, s5_f, s5_env, s5_N)

# ╔═╡ cda3220a-317c-33ee-6ca3-2356ff1fb136
md"""
### 5.1 Composition–rejection

A single box envelope works poorly when the density has a narrow peak. The box must be as tall as the peak, but it must also be as wide as the whole range of ``x``, so most of it is empty and most attempts are rejected.

**Composition–rejection** avoids this. We cut the region under the graph into horizontal layers whose height double: ``a,\ 2a,\ 4a,\ \dots``. Each layer is as wide as the part of the ``x``-axis where the density is non-zero for that layer. Sampling is done in two steps:

1. choose a box with probability proportional to its area;
2. apply accept–reject within that box.


Width of the peak $(@bind s5_pw Select([0.01, 0.03, 0.1, 0.3]; default = 0.03)) ``\qquad`` number of levels $(@bind s5_lv Slider(1:7; default = 5, show_value = true)) ``\qquad`` $(@bind s5_cr Seed(0; label = "New darts"))
"""

# ╔═╡ 7da4729c-5d6b-6dbc-0a8b-b561b13becc1
figure_5_3(s5_pw, s5_lv, s5_cr)

# ╔═╡ 44ce4893-4d52-54b6-0c56-b89fbb35016e
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
details.proof{border-left:3px solid #16427a;padding:2px 0 2px 12px;margin:8px 0}
details.proof summary{cursor:pointer;color:#16427a;font-weight:600}
.pc{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafaf7;border:1px solid #e0e0da;border-radius:4px;padding:6px 0;min-width:300px;max-width:440px;color:#1b1b1b}
.pc-head{font-family:system-ui,sans-serif;font-size:0.72rem;letter-spacing:0.09em;text-transform:uppercase;font-weight:700;color:#3d4a5c;padding:0 10px 4px}
.pc-ln{position:relative;padding:0 10px 0 34px;white-space:pre}
.pc-no{position:absolute;left:8px;color:#a0a09a;z-index:1}
.pc-ln code{position:relative;z-index:1;font:inherit;background:none;padding:0}
.pc-mark{position:absolute;left:0;right:0;top:0;bottom:0;background:#e3ecf8;border-left:3px solid #1f5aa6;z-index:0}
details.hint{margin-top:4px}
details.hint summary{cursor:pointer;font-weight:600}
.wk-state{font-variant-numeric:tabular-nums}
.wk-state .markdown,.wk-why .markdown,.wk-fail .markdown{display:inline}
.wk-state p,.wk-why p,.wk-fail p{display:inline;margin:0}
</style>"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractPlutoDingetjes = "~1.4.1"
Distributions = "~0.25.131"
HypertextLiteral = "~1.0.0"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.1"
manifest_format = "2.1"
project_hash = "1971cd5abe9029bbe8d73ca932e15f7f30076e59"

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
# ╟─a78f4338-148b-8ea3-b73d-1dbf9c1844a0
# ╟─13b07c56-4e67-1515-5f9f-9931dbd7766d
# ╟─a8a87ed0-feb1-41d9-9907-6cc7a0a76748
# ╟─05fbffb6-e4be-7ee3-061e-db4844f44486
# ╟─5444c240-a48f-ec8f-d971-a6521da0e78e
# ╟─24cca1a6-d583-cbd0-012f-825f4257eddb
# ╟─20ffbdf2-0dcd-6905-f64f-a4d729528b21
# ╠═8afaa8a6-31b3-5834-c6b1-e496f456fc51
# ╠═f8418f96-04c9-9794-25f3-94e145770c57
# ╟─4439b501-4289-1100-bd44-7e4864e9c7e3
# ╟─28965f63-a7e8-c132-56e0-530b56e17918
# ╟─1efbc5c8-0565-0447-bd14-3324ce777312
# ╟─3e086307-18a7-b5ad-500b-9a92ba0eeaac
# ╟─2c4cb982-6aef-9d22-2935-a5834f22c791
# ╟─f9ce07c7-8013-bb1b-f790-070ee8dfe900
# ╟─61e6a5b8-dc38-dc93-79dc-f370b285623b
# ╟─3864ded5-4243-970d-39f0-c2d1de65456f
# ╟─d57f738c-6e7b-bcac-045d-e5deab103301
# ╟─7edcde5d-3807-12c8-af71-b7143da81a4f
# ╟─fb9e8659-2224-9e13-42d7-7bc05b5c108e
# ╟─9a6a4aef-0e5c-4e7d-c318-c2eafc686c90
# ╠═4be643c5-0792-ea8f-e5d6-d3987c8f5ff8
# ╟─310e008c-3e96-fd0c-8391-ab26fcc75de0
# ╟─77aac2d7-d01d-4a1d-b236-fb6a0bcd27b7
# ╟─8d31b823-8dc3-9f89-7449-8326e98a7c13
# ╟─a1d22a6e-25ee-29a9-e48f-6f69ed3d0569
# ╟─9f590e8d-bfa7-eb57-a9e3-74de2caddaa2
# ╟─271e7970-5cd5-bc07-6a0a-9dc22bf4d463
# ╟─4864570d-ef47-783e-e9dd-813d496a1eaf
# ╟─72b1562a-d62c-4b5d-d6a7-f4ca54c2fcf0
# ╟─2f83d373-6aa5-cbf6-9485-bff5d7024957
# ╟─508bc972-b3b8-13dd-d970-f77257a6ce06
# ╟─f99f6bc0-62f9-0758-1931-52aa4c32d169
# ╟─193cb405-461e-e7bb-dd24-16803eee65e4
# ╟─f8322827-12dc-4a4d-a768-30de9492f047
# ╟─aba50892-b7dc-ac95-8273-18e596e2073b
# ╟─3412ebf4-1b2a-9694-f05e-8091f194b638
# ╟─97b6e3a0-e743-c7b8-19ee-0e9094eaf3d9
# ╠═184f7345-c519-5ed0-9322-bd0e9ad5a86c
# ╟─9603dd02-f29b-43e7-c146-ad54f169b0f0
# ╟─ae941370-1645-0cf4-ad96-8e8293380611
# ╟─b8e69f66-e5d7-5ddd-2d89-0a595ce225bf
# ╟─36ae7acb-7533-1461-7905-65f2bf069dcd
# ╟─f842c511-8694-0756-2ac8-9b38e9a1f5c5
# ╟─724fc1f2-1165-d86b-3bc2-0173701d6709
# ╟─4cb65688-cdc1-ca0e-b0e2-62ba60f0b859
# ╟─d3011dbd-20ec-86df-f69f-25c7799cade1
# ╟─1b7bd338-7795-38fc-ccb2-1b5cc4bd94f2
# ╟─22d2a1d7-093f-69c1-af0b-7b93e2c687bf
# ╟─cb3147ba-3d23-9081-f82d-68299f7decd6
# ╟─7f62b5fe-9a6b-4628-9007-9ace2ab794fd
# ╟─6d72710d-0f60-e3c2-0bc9-c7940066efbe
# ╟─cda3220a-317c-33ee-6ca3-2356ff1fb136
# ╟─7da4729c-5d6b-6dbc-0a8b-b561b13becc1
# ╟─9d409c93-6870-45ce-b6bd-5d2946233d32
# ╟─0ab47deb-7ad2-92d4-be52-3accfb8164f9
# ╟─a557a30b-8610-38da-c4c3-3f8fb19c04af
# ╟─56132ccc-d75a-65ef-9807-9eb422af8ba5
# ╟─7b60936c-bb3b-508f-3a07-326af90fe25f
# ╟─6a67d0d7-f6ca-4d0e-352a-a9b79c220dda
# ╟─3e02678a-dfa6-7e13-17f1-11258e9ebca7
# ╟─d873d3c8-a66f-1821-a9e5-842bdd89fa8e
# ╟─5472fa1d-a48c-d08d-cbf7-d185da690229
# ╟─9c4c7b32-e21e-c34c-1326-30e845b977d1
# ╟─42706196-7709-0faa-d7d6-1e4e3c90e190
# ╟─b19f5437-9612-dbb9-c293-c89f71af4333
# ╟─78af998f-eec2-6a88-3ccb-5654c70446c4
# ╟─02e182b6-f19b-c32a-aa3d-2abbf67bea57
# ╟─44ce4893-4d52-54b6-0c56-b89fbb35016e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
