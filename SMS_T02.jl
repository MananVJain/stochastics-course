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

# ╔═╡ 210e0daa-7ca9-59f3-8e78-237a825c9165
# Packages
begin
	using PlutoUI, Random, Statistics, Markdown, Printf, Distributions
	import AbstractPlutoDingetjes
end

# ╔═╡ 6d631793-d28b-5379-ba38-20f039ec3792
md"""
# Tutorial 02: Random Variate Generation
*Stochastic Modeling and Simulation*\
Date: 05.11.2026"""

# ╔═╡ 5788b98c-f3a8-5b34-9ae5-b20cbbd59e0e
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 1244948a-29c3-5fa9-96da-52588520174c
md"""
!!! info "From Tutorial 01"
    - A continuous random variable ``X`` has a density ``f_X`` and probability is the **area** under ``f_X``.
    - The CDF ``F_X(x) = P(X \le x)`` increases from 0 to 1; ``f_X = dF_X/dx``.
"""

# ╔═╡ 551cbe47-9b82-526e-80e0-f0437f99d1ca
md"""
## 1. Random numbers

**Question.** A simulation requires millions of random numbers. However, a computer follows a deterministic set of instructions, or algorithm. Then how can we get random numbers?

We don’t, strictly speaking. Computers do not generate truly random numbers. Instead, they use deterministic algorithms to produce sequences of numbers that are statistically indistinguishable from random numbers.

Run the cells below a few times (click into a cell and press Shift + Enter). Which results change, and which stay the same?
"""

# ╔═╡ 0d697b81-6a43-5bfb-9620-5b0781c78782
rand()

# ╔═╡ 451c1a90-3713-52d5-82a8-10a0fe049a58
rand(5)

# ╔═╡ 9e72d97b-ca16-5458-beea-55d7e1f078d6
begin
	Random.seed!(42)
	rand(3)
end

# ╔═╡ 917f96bf-e0eb-562c-9e8f-bffa47425eda
begin
	Random.seed!(42)
	rand(3)
end

# ╔═╡ 3905e070-8a75-570c-ae7c-43716de7bfa5
begin
	rng = Xoshiro(7)
	rand(rng, 3)
end

# ╔═╡ f0c6dc26-3753-5bf7-bb3b-59398dee3fa6
md"""
There are two ways to get random numbers.

- **Measure** something unpredictable/truly random event in the physical world, such as electronic noise. They are unpredictable, but they are slow and cannot be reproduced. They are used in cryptography.
- **Compute** them with a fixed set of instructions/algorithms. Each next number is computed from the previous state. The numbers are not random, but they *behave* like random numbers. They are fast and can be reproduced.

!!! definition "Pseudo-random number generator"
    A pseudorandom number generator has of a state z, a function which computes the next state from the current state, and an output function which maps each state to a number u ∈ [0, 1). The initial state is called the seed. As every step is deterministic, the same seed always produces the same sequence.
"""

# ╔═╡ 3ae224f2-861b-5d32-80d3-a2fbe3d44ae0
md"""
### 1.1 The linear congruential generator

The simplest recipe is the **linear congruential generator** (LCG):

```math
z_i = (a\,z_{i-1} + c) \bmod m, \qquad u_i = \frac{z_i}{m} \in [0, 1),
```

with a modulus ``m``, a multiplier ``a``, an increment ``c`` and a seed ``z_0``. "``\bmod m``" keeps only the remainder after division by ``m``, like a clock with ``m`` positions: each step jumps from one position to another.

!!! tasks "Tasks"
    1. Press ▶ and follow the jumps. When does the sequence start to repeat?
    2. Keep ``m = 31``, ``c = 0`` and try ``a = 3`` and ``a = 5``. How many positions are visited?
    3. Choose ``m = 32``, ``a = 5``, ``c = 1``. Is every position visited?

``m`` $(@bind s1_m NumberField(2:64; default = 31)) ``\qquad`` ``a`` $(@bind s1_a NumberField(0:200; default = 3)) ``\qquad`` ``c`` $(@bind s1_c NumberField(0:200; default = 0)) ``\qquad`` seed ``z_0`` $(@bind s1_z0 NumberField(0:200; default = 1))
"""

# ╔═╡ 46d98e61-0582-5067-819f-83826f341cc2
md"""
The state can take only ``m`` values. Once a state appears for the second time, the whole sequence repeats. The number of steps until this happens is the **period** (or cycle length). It is at most ``m``, and it depends strongly on ``a`` and ``c``.

!!! definition "Period"
    The **period** ``T`` of a generator is the number of steps after which the sequence repeats.

Real generators use very large parameters. A classic choice is ``a = 48\,271``, ``m = 2^{31} - 1`` (period about ``2.1 \cdot 10^9``). Modern generators follow the same idea with a much larger state: Julia's default generator, **Xoshiro256++**, has the period ``2^{256} - 1 \approx 10^{77}``.
"""

# ╔═╡ de272efe-a3b2-5ef2-991e-2b54cc7102c0
md"""
- Together, the processors can use at most ``T`` numbers: each one has only about ``T/P`` numbers before the stretches must meet.

!!! info "In practice"
    - Use a modern generator with a big period. Julia mixes the seed thoroughly, so `Xoshiro(1)`, `Xoshiro(2)`, … give unrelated sequences.
"""

# ╔═╡ 517bd77d-009a-50f9-aaf6-322ab5b07073
begin
	streams = [Xoshiro(i) for i in 1:4]    # one generator per processor
	[rand(s) for s in streams]
end

# ╔═╡ f802de82-955a-5106-a4d6-ac09eb9068f4
md"""
!!! situation "Situation"
    A research group runs 1000 jobs on a computer cluster. Each job uses an LCG with period ``2^{31} \approx 2.1 \cdot 10^9``, seeded with its job number, and draws ``10^7`` random numbers.
"""

# ╔═╡ c24791e9-3fef-5b27-9b26-01cfb783a26b
md"""
!!! info "Key points"
    - A pseudo-random generator is a fixed recipe started from a seed: fast, reproducible, but not random.
    - Every generator repeats after its period ``T``.
    - For parallel work, use a generator with a big period and separate streams per processor.
"""

# ╔═╡ 091fd133-48c0-5312-8bec-b8cfd837208b
md"""
!!! tasks "Tasks"
    1. Which methods leave empty cells? Which never do?
    2. Press **New points**. Which pictures change, and which stay the same?
    3. Choose the small LCG. What do you notice?
"""

# ╔═╡ 5851eac5-ecb4-5ddb-b266-54531e654ee7
md"""
Independent points do not avoid each other. Each new point can land anywhere, also next to an earlier one, so random points always form **clumps** and leave **holes**. This is because of independence.

Quasi-random points are designed to cover a space more evenly than pseudorandom points. The simplest example is the additive recurrence

```math
u_{k+1} = (u_k + \alpha) \bmod 1
```

with an irrational step ``\alpha``, for example ``\alpha = (\sqrt5 - 1)/2 \approx 0.618``. Each new point lands into one of the largest gaps left. The Halton and Sobol sequences use the digits of ``k`` in base 2 and 3 to fill the gaps in a similar way.
"""

# ╔═╡ 577c3021-dd02-5283-96a1-3ab6f9884beb
md"""
!!! definition "Pseudo-random and quasi-random numbers"
    - **Pseudo-random** numbers behave like independent uniform numbers. They form clumps and holes.
    - **Quasi-random** numbers cover the space **evenly**, BUT they are **not independent**.
    - Pseudo-random numbers mimic independence, while quasi-random numbers give it up in exchange for more uniform coverage.
"""

# ╔═╡ 820536ed-09a1-5363-9e1e-3cc65bebd79f
md"""
**When to use which.**

- To **simulate chance**, such as arrivals, decays, mutations, molecules moving: pseudo-random numbers, because the events must be independent.
- To **cover a space evenly**, such as test points for a parameter scan, or points for computing an integral: quasi-random numbers.
"""

# ╔═╡ 7c8b3aea-e97f-55c2-905c-8ef47d87f607
md"""
!!! situation "Situation"
    A colleague creates synthetic microscope images of molecules in a cell. To "cover the cell evenly", she places the molecules with a quasi-random sequence. She wants to now test a program that detects whether molecules form clusters.
"""

# ╔═╡ f18f88e3-e7d4-5fbd-a151-7a449164a618
md"""
## 3. From uniform to any distribution

So far we can produce uniform numbers on ``(0, 1)``. Simulations need much more: exponential, normal, and  many more. Can uniform numbers on ``(0, 1)`` be used to get random numbers from **any** other distribution?

The idea: change uniform numbers with a function, ``Y = g(X)``. So first we need to know what a function does to a distribution.

### 3.1 What a function does to a random variable

**Question.** Take uniform numbers ``X`` on ``(0, 1)`` and square them: ``Y = X^2``. Is ``Y`` still uniform? run the cell.
"""

# ╔═╡ 9bb57f37-3d52-5046-be70-1a8a2c17f674
md"""
The values of ``Y`` pile up near 0. To see how much, **count** with the CDF. For ``0 < y < 1``:

```math
F_Y(y) = P(Y \le y) = P(X^2 \le y) = P(X \le \sqrt{y}) = \sqrt{y}.
```
"""

# ╔═╡ 8454a302-2cd9-58f0-a11e-bee87e9ad4fe
md"""
The density is the derivative of the CDF:

```math
f_Y(y) = \frac{d}{dy}\sqrt{y} = \frac{1}{2\sqrt{y}}, \qquad 0 < y < 1.
```
"""

# ╔═╡ da34284f-5df2-5c65-a09a-d4974e7fae39
md"""
**The general rule.** The same two steps, count with the CDF and differentiate, for every **increasing** function ``g``:

```math
F_Y(y) = P(g(X) \le y) = P\big(X \le g^{-1}(y)\big) = F_X\big(g^{-1}(y)\big),
\qquad
f_Y(y) = f_X\big(g^{-1}(y)\big)\,\frac{d}{dy}g^{-1}(y).
```

In words: the factor ``\frac{d}{dy}g^{-1}(y) = \frac{dx}{dy}`` says how much the spacing of the lines changes. The probability between two lines does not change: ``f_Y(y)\,dy = f_X(x)\,dx``.
"""

# ╔═╡ 64e05f96-52de-509c-8ced-d0c09cb56c18
md"""
### 3.2 Decreasing functions

**Question.** What changes if ``g`` is decreasing, for example ``Y = 1/X``?

Choose ``Y = 1/X`` in Figure 3.1: large ``x`` give small ``y``. So the inequality turns around when we count:

```math
F_Y(y) = P(1/X \le y) = P(X \ge 1/y) = 1 - F_X(1/y).
```

Differentiating gives a minus sign: ``-f_X(g^{-1}(y))\,\frac{d}{dy}g^{-1}(y)``. Since ``g^{-1}`` is decreasing, its derivative is negative, and the two minus signs cancel. Both cases are covered by one formula with an absolute value.

!!! definition "Transformation of a continuous random variable"
    Let ``g`` be strictly monotone and ``Y = g(X)``.
    - ``g`` increasing: ``F_Y(y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))``.
    - ``g`` decreasing: ``F_Y(y) = P(X \ge g^{-1}(y)) = 1 - F_X(g^{-1}(y))``.
    In both cases, differentiating gives
    ```math
    f_Y(y) = f_X\big(g^{-1}(y)\big)\,\left|\frac{d}{dy} g^{-1}(y)\right|.
    ```
"""

# ╔═╡ 173493a5-996a-5dc2-9f24-b2ab377d1c75
md"""
Uniform numbers are, in this sense, the common raw material of all distributions.
"""

# ╔═╡ ee96ef03-e508-5656-a4ca-c17ee6cb8385
md"""
**Example: the exponential distribution.** Here ``u = F(x) = 1 - e^{-\lambda x}``. Solving for ``x``:

```math
x = F^{-1}(u) = -\frac{\ln(1 - u)}{\lambda}.
```

One uniform number gives one exact exponential random number.

When ``F^{-1}`` has no formula, it can be computed numerically (the normal distribution in Figure 3.4 does this). In Julia, `quantile(D, u)` computes ``F^{-1}(u)`` for every distribution of `Distributions.jl`:
"""

# ╔═╡ 1ea7eee7-d1dc-5eb7-8818-b0c9ee475341
quantile.(Weibull(1.5, 1.0), rand(5))       # five Weibull(1.5, 1) numbers by inversion

# ╔═╡ 594e9137-b6fa-5c41-8751-793e93e096a5
rand(Exponential(2.0), 5)                    # the same idea, done for you

# ╔═╡ 4287c3ae-ee13-5fd5-8fa8-a5a535ac1419
md"""
!!! situation "Situation"
    A simulation computes exponential random numbers as `-log(rand())`. After ``10^9`` samples, one value is `Inf` and ruins the average.
"""

# ╔═╡ 63212bb9-4635-50c8-b210-1689f5367e8e
md"""
!!! info "Key points"
    - To find the density of ``Y = g(X)``: count with the CDF, then differentiate. Result: ``f_Y(y) = f_X(x)\,|dx/dy|``; the absolute value covers decreasing ``g``.
    - Applying its own CDF makes every continuous distribution uniform.
    - Inversion goes back: ``X = F^{-1}(U)``. Exact, one uniform number per sample, but it needs ``F^{-1}``.
"""

# ╔═╡ e53cb7c7-5e9d-5f44-becf-356df7e54911
md"""
!!! tasks "Tasks"
    1. From which ``k`` does the histogram (left) look normal?
    2. Now look at the right panel. Where do the two curves agree, and where not?
"""

# ╔═╡ 2e27dfa5-2161-5c29-b569-93647b3d822b
md"""
!!! tasks "Tasks"
    1. Keep ``u_1`` fixed and move ``u_2``. What changes?
    2. Keep ``u_2`` fixed and move ``u_1`` from 0.01 to 0.99. Which values of ``u_1`` give points far from the centre?
"""

# ╔═╡ c18db550-9194-5bef-b9dc-e2c75f763718
md"""
!!! theorem "Box–Muller transform"
    If ``U_1, U_2`` are independent and uniform on ``(0, 1)``, then
    ```math
    X_1 = \sqrt{-2 \ln U_1}\,\cos(2\pi U_2), \qquad X_2 = \sqrt{-2 \ln U_1}\,\sin(2\pi U_2)
    ```
    are independent standard normal numbers. 
"""

# ╔═╡ 3d9093b6-10c4-5c08-b913-d4795e2949cb
md"""
In practice, `randn()` gives standard normal numbers and `rand(Normal(μ, σ))` any normal distribution.

!!! info "Key points"
    - Box–Muller: a round cloud has independent angle and distance; both can be generated from uniform numbers by inversion. One uniform number per normal number.
"""

# ╔═╡ eb689310-a305-5a42-b191-c7fe80fe139e
md"""
**How many darts are kept?** The darts fill the box evenly, so the fraction kept is a ratio of areas:

```math
\text{fraction kept} = \frac{\text{area under the curve}}{\text{area of the box}} = \frac{1}{\text{area of the box}} .
```
"""

# ╔═╡ 2e59d710-9932-51c1-aa45-c49ea36a2be1
md"""
### 5.2 Your own density

Draw any density in the pad below. Then choose an envelope: a **box**, or a **hill** ``\mu\,g(x)`` built from a normal density ``g``, which you can move and stretch. Darts are drawn from the envelope: ``x`` from ``g`` (or uniformly for the box) and a height uniformly between 0 and the envelope.

!!! tasks "Tasks"
    1. Draw a density with two hills. How many darts are kept with a box?
    2. Lower the box below the highest point. What goes wrong, and does the program warn you?
    3. Choose the hill and make it fit your density as closely as possible while staying above it. How many darts are kept now?
    4. Draw a narrow spike. Why is the box now so wasteful?
"""

# ╔═╡ ee77527c-e17a-58bf-b114-b9d9f0023b2c
md"""
!!! definition "Accept–reject with a proposal"
    Let ``g`` be a density that is easy to sample (the **proposal**) and ``\mu`` a constant with ``\mu\,g(x) \ge f(x)`` for all ``x`` (the **envelope**). Repeat:
    1. draw ``x`` from ``g`` and ``u`` uniform on ``(0, 1)``;
    2. if ``u\,\mu\,g(x) \le f(x)``, keep ``x``; otherwise throw it away.

    On average a fraction ``1/\mu`` of the darts is kept. The closer ``\mu g`` fits ``f``, the less is wasted.
"""

# ╔═╡ a44479fc-c5d7-5b24-a916-1a1fbfcd4c99
md"""
**Question.** What about more dimensions? To get uniform points in a ball, throw darts into the cube around it and keep the darts inside the ball.
"""

# ╔═╡ 6b74424c-5828-59ae-b2cc-30d2582785c8
md"""
!!! situation "Situation"
    A student needs uniform points in a ball in 30 dimensions. She throws darts into the cube ``[-1, 1]^{30}`` and keeps those inside the ball.
"""

# ╔═╡ 81a4026e-6407-544c-bd73-d87066f447bc
md"""
## 6. Which method when?

!!! info "Toolbox"
    - **Uniform numbers:** a modern pseudo-random generator, `rand()`. Record the seed; one stream per parallel task.
    - **Covering a space evenly:** quasi-random points. Not for simulating chance.
    - **``F^{-1}`` known (or computable):** inversion, ``X = F^{-1}(U)``; in Julia `quantile.(D, rand(n))`.
    - **Normal numbers:** Box–Muller; in practice `randn()`.
    - **Only the shape ``f`` known, few dimensions:** accept–reject with a proper envelope that stays above ``f``.
    - **Sharp peak, many samples from the same density:** composition–rejection.
    - **Many dimensions:** Markov chain Monte Carlo (We will learn this in upcoming classes).
"""

# ╔═╡ 48a9895e-e831-5d2d-b3ff-0d1ba88bd9aa
md"""
## Exercises

### Calculation problems
"""

# ╔═╡ bb3f11ac-88dc-57ef-8835-3c8cf08e3433
md"""
### Programming exercises

**Exercise 1 · Inversion.** The density ``f(x) = 2(1 - x)`` on ``[0, 1]`` has the CDF ``F(x) = 1 - (1 - x)^2``. Complete `tri_inv(u)` so that it returns ``F^{-1}(u)``.
"""

# ╔═╡ 40877568-8051-584c-8645-3cbfc8b4e1f0
function tri_inv(u)
	return missing   # your code here
end

# ╔═╡ ae01a976-cc53-5209-bd1e-8606b0a54977
md"""
**Exercise 2 · Box–Muller.** Complete `boxmuller(u1, u2)` so that it returns the pair `(x1, x2)` of the Box–Muller transform.
"""

# ╔═╡ 955720cb-49bd-5227-8a66-0b093d12fc3c
function boxmuller(u1, u2)
	return missing   # your code here
end

# ╔═╡ 66915c6c-d82f-545f-a83a-d37b8a6f61e6
md"""
**Exercise 3 · Accept–reject.** Complete `accept_reject(f, fmax, n)`: it returns a vector of `n` samples from a density `f` on ``[0, 1]`` with ``f(x) \le`` `fmax`, using darts in the box ``[0, 1] \times [0, \text{fmax}]``.
"""

# ╔═╡ 7713d1e9-e98d-51bd-8bca-56efbfc26b51
function accept_reject(f, fmax, n)
	out = Float64[]
	# your code here
	return out
end

# ╔═╡ f0263455-b054-556b-9b2b-923af1d43b58
md"""
## Further reading

- L. Devroye, *Non-Uniform Random Variate Generation*, Springer, 1986. Free online: <https://luc.devroye.org/rnbookindex.html>.
- Julia documentation, *Random Numbers*: <https://docs.julialang.org/en/v1/stdlib/Random/>.
- S. Vigna, *xoshiro / xoroshiro generators*: <https://prng.di.unimi.it/>.
- K. Karplus and A. Strong, "Digital synthesis of plucked-string and drum timbres", *Computer Music Journal* 7(2) (1983) 43–55.
- S. A. Hedges, "Dice music in the eighteenth century", *Music & Letters* 59 (1978) 180–187.
- P. Chordia and A. Rae, "Raag recognition using pitch-class and pitch-class dyad distributions", *ISMIR 2007*, 431–436.

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 2.
"""

# ╔═╡ 497add3d-fecd-58d5-b022-f3ddf2491203
# Page style: figures, players, questions and boxes
html"""<style>
.fg{margin:10px 0 6px}
.fg svg.fig{display:block;max-width:100%;height:auto;background:#fff}
.fg-row{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:flex-start}
.fg-cap{font-size:.9rem;color:#333;line-height:1.4;margin:6px 2px 0;max-width:720px}
.fg-cap p{margin:0}
.pl{border:1px solid #e3e3df;border-radius:4px;padding:8px 4px 6px;background:#fff;margin:8px 0}
.pl-bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:6px;font:13px system-ui,sans-serif}
.pl-bar button,.pn-b{font:inherit;padding:2px 10px;border:1px solid #cfcfca;border-radius:3px;background:#fafaf8;cursor:pointer}
.pl-bar input[type=range]{flex:1;min-width:160px;accent-color:#1f5aa6}
.pl-read{font:13px system-ui,sans-serif;color:#333;min-height:1.4em;margin-top:4px}
.quiz{border:1px solid #e3e3df;border-radius:4px;padding:10px 14px;margin:14px 0;background:#fdfdfc}
.quiz-head{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:#6a4c93;margin-bottom:4px}
.quiz-q p{margin:2px 0 6px}
.quiz-opt{display:flex;gap:10px;align-items:baseline;border:1px solid #e3e3df;border-radius:4px;padding:5px 10px;margin:6px 0;cursor:pointer;background:#fff}
.quiz-opt:hover{border-color:#1f5aa6}
.quiz-opt p,.quiz-fb p{margin:0}
.quiz-key{font-weight:700;color:#8c8c8c;min-width:1.2em}
.quiz-opt.right{border-color:#2f7d55;background:#eef6f0}
.quiz-opt.wrong{border-color:#c2462e;background:#fbf1ee}
.quiz-fb{border-radius:3px;padding:7px 12px;margin:6px 0 2px;font-size:.95rem}
.quiz-fb.yes{background:#eef6f0;border-left:3px solid #2f7d55}
.quiz-fb.no{background:#fbf1ee;border-left:3px solid #c2462e}
.quiz-num input{font:inherit;padding:3px 8px;border:1px solid #cfcfca;border-radius:3px}
.quiz-num button{padding:2px 10px;cursor:pointer}
.admonition.tasks{border-color:#6a4c93;background:#f4f1f8}
.admonition.tasks>.admonition-title{background:#6a4c93;color:#fff}
.admonition.situation{border-color:#b0651a;background:#fdf6ee}
.admonition.situation>.admonition-title{background:#b0651a;color:#fff}
.pluto-docs-binding{display:none}
</style>"""

# ╔═╡ c88f31de-390c-5032-9041-39c132990ca2
# Figures: a small SVG plotting library (blue = exact distribution, orange = samples)
begin
	const COL = (blue = "#1f5aa6", orange = "#e1871d", red = "#c2462e", green = "#2f7d55", purple = "#6a4c93",
		ochre = "#b8860b", grey = "#8c8c8c", ink = "#1b1b1b")

	# short number format
	function fmt(x::Real; d = 3)
		x isa Integer && return string(x)
		isfinite(x) || return x > 0 ? "∞" : "−∞"
		x == 0 && return "0"
		e = floor(Int, log10(abs(x)))
		(e < -3 || e >= 6) && return string(round(x / 10.0^e, sigdigits = d)) * " × 10" * sup(e)
		s = string(round(x, sigdigits = d)); endswith(s, ".0") ? s[1:end-2] : s
	end
	sup(e) = join(Dict(zip("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))[c] for c in string(e))
	xesc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

	# a figure collects SVG elements in data coordinates; `over` holds elements drawn on top of the axes
	mutable struct Fig
		w::Int; h::Int; l::Int; r::Int; t::Int; b::Int
		xr::NTuple{2, Float64}; yr::NTuple{2, Float64}; xlog::Bool; ylog::Bool
		axes::Bool; xlabel::String; ylabel::String; title::String
		xticks::Any; yticks::Any; xtl::Any; ytl::Any; grid::Bool
		body::IOBuffer; over::IOBuffer; id::String
	end
	fig(; w = 640, h = 300, xr = (0, 1), yr = (0, 1), xlabel = "", ylabel = "", title = "", xlog = false, ylog = false, axes = true,
		xticks = nothing, yticks = nothing, xticklabels = nothing, yticklabels = nothing, grid = false, l = nothing, r = 16, t = nothing, b = nothing) =
		Fig(w, h, something(l, axes ? 62 : 10), r, something(t, title == "" ? 14 : 30), something(b, axes ? (xlabel == "" ? 28 : 46) : 10),
			Float64.(xr), Float64.(yr), xlog, ylog, axes, xlabel, ylabel, title, xticks, yticks, xticklabels, yticklabels, grid,
			IOBuffer(), IOBuffer(), string(rand(UInt32), base = 36))

	# data to pixel coordinates
	_s(v, lg) = lg ? log10(max(v, 1e-300)) : v
	X(f, x) = ((a, b) = _s.(f.xr, f.xlog); f.l + (_s(x, f.xlog) - a) / (b - a) * (f.w - f.l - f.r))
	Y(f, y) = ((a, b) = _s.(f.yr, f.ylog); f.h - f.b - (_s(y, f.ylog) - a) / (b - a) * (f.h - f.t - f.b))
	_n(v) = @sprintf("%.1f", v)
	# s: visible from player step s on; o: visible only at step o
	_st(s, o) = (s === nothing ? "" : " data-s=\"$(s)\"") * (o === nothing ? "" : " data-o=\"$(o)\"")
	_dash(d) = d === nothing ? "" : " stroke-dasharray=\"$(d)\""
	_io(f, over) = over ? f.over : f.body
	raw!(f, s; over = false) = (print(_io(f, over), s); f)

	function line!(f, x, y; c = COL.blue, w = 1.8, dash = nothing, op = 1.0, s = nothing)
		d = join((i == 1 || !isfinite(y[i-1]) ? "M" : "L") * _n(X(f, x[i])) * "," * _n(Y(f, y[i])) for i in eachindex(x) if isfinite(y[i]))
		raw!(f, "<path d=\"$(d)\" fill=\"none\" stroke=\"$(c)\" stroke-width=\"$(w)\" stroke-linejoin=\"round\" opacity=\"$(op)\"$(_dash(dash))$(_st(s, nothing))/>")
	end
	function dots!(f, x, y; c = COL.blue, r = 3.2, op = 0.9, stroke = nothing, sw = 1, s = nothing, o = nothing, over = false)
		for i in eachindex(x)
			raw!(f, "<circle cx=\"$(_n(X(f, x[i])))\" cy=\"$(_n(Y(f, y[i])))\" r=\"$(r)\" fill=\"$(c)\" opacity=\"$(op)\"" *
				(stroke === nothing ? "" : " stroke=\"$(stroke)\" stroke-width=\"$(sw)\"") * _st(s, o) * "/>"; over)
		end
		f
	end
	function rect!(f, x0, y0, x1, y1; fill = COL.blue, op = 0.15, stroke = "none", sw = 1, s = nothing, o = nothing, over = false)
		a0, a1 = minmax(X(f, x0), X(f, x1)); b0, b1 = minmax(Y(f, y0), Y(f, y1))
		raw!(f, "<rect x=\"$(_n(a0))\" y=\"$(_n(b0))\" width=\"$(_n(a1 - a0))\" height=\"$(_n(b1 - b0))\" fill=\"$(fill)\" opacity=\"$(op)\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\"$(_st(s, o))/>"; over)
	end
	bars!(f, x, h; width = 0.8, c = COL.blue, op = 0.85) = (foreach(i -> rect!(f, x[i] - width / 2, 0, x[i] + width / 2, h[i]; fill = c, op, stroke = "white", sw = 0.8), eachindex(x)); f)
	hist!(f, edges, h; c = COL.orange, op = 0.75) = (foreach(i -> rect!(f, edges[i], 0, edges[i+1], h[i]; fill = c, op, stroke = "white", sw = 0.6), eachindex(h)); f)
	area!(f, x, y; c = COL.blue, op = 0.15) = raw!(f, "<path d=\"M$(_n(X(f, x[1]))),$(_n(Y(f, 0)))" * join("L$(_n(X(f, a))),$(_n(Y(f, b)))" for (a, b) in zip(x, y)) * "L$(_n(X(f, x[end]))),$(_n(Y(f, 0)))Z\" fill=\"$(c)\" opacity=\"$(op)\"/>")
	seg!(f, x0, y0, x1, y1; c = COL.grey, w = 1.2, dash = nothing, op = 1.0, s = nothing, o = nothing) =
		raw!(f, "<line x1=\"$(_n(X(f, x0)))\" y1=\"$(_n(Y(f, y0)))\" x2=\"$(_n(X(f, x1)))\" y2=\"$(_n(Y(f, y1)))\" stroke=\"$(c)\" stroke-width=\"$(w)\" opacity=\"$(op)\"$(_dash(dash))$(_st(s, o))/>")
	vline!(f, x; c = "#555", w = 1.1, dash = "4 3") = seg!(f, x, f.yr[1], x, f.yr[2]; c, w, dash)
	hline!(f, y; c = "#555", w = 1.1, dash = "4 3") = seg!(f, f.xr[1], y, f.xr[2], y; c, w, dash)
	function arrow!(f, x0, y0, x1, y1; c = COL.ink, w = 1.6, head = 8, op = 1.0, dash = nothing, s = nothing, o = nothing)
		a0, b0, a1, b1 = X(f, x0), Y(f, y0), X(f, x1), Y(f, y1); L = hypot(a1 - a0, b1 - b0)
		L < 0.5 && return f
		ux, uy = (a1 - a0) / L, (b1 - b0) / L; h = min(head, 0.45L)
		tip(k) = "$(_n(a1 - ux * h + k * uy * 0.42h)),$(_n(b1 - uy * h - k * ux * 0.42h))"
		raw!(f, "<g opacity=\"$(op)\"$(_st(s, o))><line x1=\"$(_n(a0))\" y1=\"$(_n(b0))\" x2=\"$(_n(a1 - 0.7ux * h))\" y2=\"$(_n(b1 - 0.7uy * h))\" stroke=\"$(c)\" stroke-width=\"$(w)\"$(_dash(dash))/>" *
			"<polygon points=\"$(_n(a1)),$(_n(b1)) $(tip(1)) $(tip(-1))\" fill=\"$(c)\"/></g>"; over = true)
	end
	text!(f, x, y, str; c = COL.ink, size = 12, anchor = "start", bold = false, dx = 0, dy = 0, bg = false, s = nothing) =
		raw!(f, "<text x=\"$(_n(X(f, x) + dx))\" y=\"$(_n(Y(f, y) + dy))\" font-size=\"$(size)\" fill=\"$(c)\" text-anchor=\"$(anchor)\" dominant-baseline=\"middle\"" *
			(bold ? " font-weight=\"600\"" : "") * (bg ? " paint-order=\"stroke\" stroke=\"white\" stroke-width=\"3.5\"" : "") * _st(s, nothing) * ">$(xesc(str))</text>"; over = true)
	circle!(f, x, y, rpx; stroke = COL.ink, sw = 1.5) = raw!(f, "<circle cx=\"$(_n(X(f, x)))\" cy=\"$(_n(Y(f, y)))\" r=\"$(rpx)\" fill=\"none\" stroke=\"$(stroke)\" stroke-width=\"$(sw)\"/>")

	# legend: items are (label, colour) or (label, colour, :line/:dash/:bar)
	function legend!(f, items; at = :tr, size = 12)
		n = length(items); wd = maximum(length(first(it)) for it in items) * 6.6 + 34
		x0 = at in (:tr, :br) ? f.w - f.r - wd - 4 : f.l + 10; y0 = at in (:tr, :tl) ? f.t + 8 : f.h - f.b - 18n - 4
		raw!(f, "<rect x=\"$(_n(x0 - 6))\" y=\"$(_n(y0 - 4))\" width=\"$(_n(wd + 8))\" height=\"$(18n + 4)\" fill=\"white\" opacity=\"0.85\"/>"; over = true)
		for (i, it) in enumerate(items)
			y = y0 + 18(i - 0.5); k = get(it, 3, :line)
			raw!(f, (k == :bar ? "<rect x=\"$(_n(x0 + 3))\" y=\"$(_n(y - 6))\" width=\"14\" height=\"12\" fill=\"$(it[2])\" opacity=\"0.8\"/>" :
				"<line x1=\"$(_n(x0))\" x2=\"$(_n(x0 + 20))\" y1=\"$(_n(y))\" y2=\"$(_n(y))\" stroke=\"$(it[2])\" stroke-width=\"2.2\"$(_dash(k == :dash ? "4 3" : nothing))/>") *
				"<text x=\"$(_n(x0 + 28))\" y=\"$(_n(y))\" font-size=\"$(size)\" dominant-baseline=\"middle\">$(xesc(it[1]))</text>"; over = true)
		end
		f
	end

	# about n round tick values between lo and hi
	function niceticks(lo, hi, n)
		step = 10.0^floor(log10((hi - lo) / n)); m = (hi - lo) / n / step
		step *= m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10
		[abs(t) < 1e-12 ? 0.0 : t for t in ceil(lo / step) * step:step:hi + 1e-9]
	end
	ticklabel(t) = abs(t) >= 1e5 || 0 < abs(t) < 1e-3 ? fmt(t) : (isinteger(t) ? string(Int(t)) : string(round(t, digits = 4)))
	logticks(r) = [10.0^k for k in ceil(Int, log10(r[1])):floor(Int, log10(r[2]))]

	function _axes(io, f)
		x0, x1, y0, y1 = f.l, f.w - f.r, f.t, f.h - f.b
		xt = something(f.xticks, f.xlog ? logticks(f.xr) : niceticks(f.xr..., clamp(round(Int, (x1 - x0) / 80), 2, 7)))
		yt = something(f.yticks, f.ylog ? logticks(f.yr)[1:(length(logticks(f.yr)) > 7 ? 2 : 1):end] : niceticks(f.yr..., 4))
		xl = something(f.xtl, [f.xlog ? "10" * sup(round(Int, log10(t))) : ticklabel(t) for t in xt])
		yl = something(f.ytl, [f.ylog ? "10" * sup(round(Int, log10(t))) : ticklabel(t) for t in yt])
		f.grid && foreach(t -> print(io, "<line x1=\"$(x0)\" x2=\"$(x1)\" y1=\"$(_n(Y(f, t)))\" y2=\"$(_n(Y(f, t)))\" stroke=\"#ececE8\"/>"), yt)
		print(io, "<path d=\"M$(x0),$(y0)V$(y1)H$(x1)\" fill=\"none\" stroke=\"#333\" stroke-width=\"0.9\"/>")
		for (t, s) in zip(xt, xl)
			x = X(f, t); x0 - 1 <= x <= x1 + 1 || continue
			print(io, "<line x1=\"$(_n(x))\" x2=\"$(_n(x))\" y1=\"$(y1)\" y2=\"$(y1 + 4)\" stroke=\"#333\"/><text x=\"$(_n(x))\" y=\"$(y1 + 16)\" font-size=\"11.5\" fill=\"#555\" text-anchor=\"middle\">$(xesc(s))</text>")
		end
		for (t, s) in zip(yt, yl)
			y = Y(f, t); y0 - 1 <= y <= y1 + 1 || continue
			print(io, "<line x1=\"$(x0 - 4)\" x2=\"$(x0)\" y1=\"$(_n(y))\" y2=\"$(_n(y))\" stroke=\"#333\"/><text x=\"$(x0 - 7)\" y=\"$(_n(y + 4))\" font-size=\"11.5\" fill=\"#555\" text-anchor=\"end\">$(xesc(s))</text>")
		end
		f.xlabel == "" || print(io, "<text x=\"$(_n((x0 + x1) / 2))\" y=\"$(f.h - 8)\" font-size=\"13\" text-anchor=\"middle\">$(xesc(f.xlabel))</text>")
		f.ylabel == "" || print(io, "<text transform=\"translate(14,$(_n((y0 + y1) / 2))) rotate(-90)\" font-size=\"13\" text-anchor=\"middle\">$(xesc(f.ylabel))</text>")
	end

	function svgstring(f::Fig)
		io = IOBuffer()
		print(io, "<svg class=\"fig\" xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $(f.w) $(f.h)\" width=\"$(round(Int, 0.97f.w))\" font-family=\"system-ui,sans-serif\">",
			"<clipPath id=\"c$(f.id)\"><rect x=\"$(f.l)\" y=\"$(f.t - 2)\" width=\"$(f.w - f.l - f.r + 2)\" height=\"$(f.h - f.t - f.b + 4)\"/></clipPath>")
		f.title == "" || print(io, "<text x=\"$(f.l)\" y=\"16\" font-size=\"13.5\" font-weight=\"600\">$(xesc(f.title))</text>")
		print(io, "<g clip-path=\"url(#c$(f.id))\">", String(f.body.data[1:f.body.size]), "</g>")
		f.axes && _axes(io, f)
		print(io, String(f.over.data[1:f.over.size]), "</svg>")
		String(take!(io))
	end
	Base.show(io::IO, ::MIME"text/html", f::Fig) = print(io, "<div class=\"fg\">", svgstring(f), "</div>")

	# several figures side by side (figure) or below each other (vstack), with an optional caption
	struct Panels
		items::Vector{Any}; caption::Any; vertical::Bool
	end
	figure(items...; caption = nothing) = Panels(collect(items), caption, false)
	vstack(items...; caption = nothing) = Panels(collect(items), caption, true)
	tohtml(x) = x isa Fig ? svgstring(x) : sprint(show, MIME"text/html"(), x)
	Base.show(io::IO, ::MIME"text/html", p::Panels) = print(io, "<div class=\"fg\"><div", p.vertical ? ">" : " class=\"fg-row\">",
		join("<div>" * tohtml(it) * "</div>" for it in p.items), "</div>", p.caption === nothing ? "" : "<div class=\"fg-cap\">" * tohtml(p.caption) * "</div>", "</div>")

	# histogram heights, normalised to a density
	function histdata(xs, edges)
		c = zeros(length(edges) - 1)
		for x in xs
			i = x == edges[end] ? length(c) : searchsortedlast(edges, x)
			1 <= i <= length(c) && (c[i] += 1)
		end
		c ./ (length(xs) .* diff(edges))
	end
end;

# ╔═╡ 9dcbfb89-2e67-56d1-b982-86d29c80faf5
# Interaction: seed button, step player, canvas animations, drawing pad
begin
	Seed(default = 0; label = "New random numbers") = CounterButton(label)

	# Julia values as a JavaScript
	js(x::Real) = isfinite(x) ? string(round(Float64(x); sigdigits = 7)) : "null"
	js(x::Union{Integer, Bool}) = string(x)
	js(x::AbstractString) = repr(x)
	js(x::Union{AbstractVector, Tuple}) = "[" * join(js.(x), ",") * "]"
	js(x::NamedTuple) = "{" * join(("\"$(k)\":" * js(v) for (k, v) in pairs(x)), ",") * "}"

	# A step player: SVG elements with data-s (shown from step s on) or data-o (only at step o)
	struct Player
		items::Vector{Any}; n::Int; start::Int; fps::Float64; reads::Vector{String}; label::String; caption::Any
	end
	player(items...; n, start = 0, fps = 12, reads = String[], label = "step", caption = nothing) =
		Player(collect(items), n, start, fps, reads, label, caption)
	function Base.show(io::IO, m::MIME"text/html", p::Player)
		print(io, """<div class="pl"><div class="fg-row">$(join("<div>" * tohtml(it) * "</div>" for it in p.items))</div>
		<div class="pl-bar"><button class="b0">⏮</button><button class="bp">▶</button><button class="b1">⏭</button><button class="bn">⏭⏭</button>
		<input type="range" min="0" max="$(p.n)" value="$(p.start)"><span class="lab"></span>
		<label>speed <select><option>1</option><option>2</option><option>5</option><option>20</option></select>×</label></div>
		<div class="pl-read"></div>$(p.caption === nothing ? "" : "<div class=\"fg-cap\">" * tohtml(p.caption) * "</div>")
		<script>
		const root = currentScript.parentElement, n = $(p.n), reads = $(js(p.reads));
		const els = Array.from(root.querySelectorAll("[data-s],[data-o]"));
		const range = root.querySelector("input"), play = root.querySelector(".bp");
		let k = +range.value, timer = null;
		function show(j) {
			k = Math.max(0, Math.min(n, j)); range.value = k;
			for (const e of els) e.style.display = (!e.dataset.s || +e.dataset.s <= k) && (!e.dataset.o || +e.dataset.o === k) ? "" : "none";
			root.querySelector(".lab").textContent = $(js(p.label)) + " " + k + " / " + n;
			if (reads.length) root.querySelector(".pl-read").innerHTML = reads[Math.min(k, reads.length - 1)];
		}
		function stop() { clearInterval(timer); timer = null; play.textContent = "▶"; }
		play.onclick = () => {
			if (timer) return stop();
			if (k >= n) show(0);
			play.textContent = "⏸";
			timer = setInterval(() => k >= n ? stop() : show(k + +root.querySelector("select").value), 1000 / $(p.fps));
		};
		root.querySelector(".b0").onclick = () => { stop(); show(0); };
		root.querySelector(".b1").onclick = () => { stop(); show(k + 1); };
		root.querySelector(".bn").onclick = () => { stop(); show(n); };
		range.oninput = () => { stop(); show(+range.value); };
		invalidation.then(stop);
		show(k);
		</script></div>""")
	end

	# Canvas animation over SVG figures. `code` defines draw(k) for frame k and may define info = k => text.
	# Helpers inside: P[i] (panel i with X, Y, ctx, clip), dot, box, ease, and the data D.
	function animate(figs, data, code; nframes = 100, fps = 30, start = nframes, caption = nothing)
		panels = join("<div style=\"position:relative;display:inline-block;line-height:0\">" * svgstring(f) *
			"<canvas style=\"position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none\"></canvas></div>" for f in figs)
		maps = [(l = f.l, r = f.r, t = f.t, b = f.b, w = f.w, h = f.h, x0 = f.xr[1], x1 = f.xr[2], y0 = f.yr[1], y1 = f.yr[2]) for f in figs]
		HTML("""<div class="pl"><div class="fg-row">$(panels)</div>
		<div class="pl-bar"><button class="b0">⏮</button><button class="bp">▶</button><button class="bn">⏭⏭</button>
		<input type="range" min="0" max="$(nframes)" value="$(start)">
		<label>speed <select><option>0.5</option><option selected>1</option><option>3</option><option>10</option></select>×</label></div>
		<div class="pl-read"></div>$(caption === nothing ? "" : "<div class=\"fg-cap\">" * tohtml(caption) * "</div>")
		<script>
		(() => {
		const root = currentScript.parentElement, D = $(js(data)), NF = $(nframes);
		const P = Array.from(root.querySelectorAll("canvas")).map((cv, i) => {
			const M = $(js(maps))[i], ctx = cv.getContext("2d");
			cv.width = 2 * M.w; cv.height = 2 * M.h; ctx.setTransform(2, 0, 0, 2, 0, 0);
			return {ctx, X: x => M.l + (x - M.x0) / (M.x1 - M.x0) * (M.w - M.l - M.r), Y: y => M.h - M.b - (y - M.y0) / (M.y1 - M.y0) * (M.h - M.t - M.b),
				clear: () => ctx.clearRect(0, 0, M.w, M.h),
				clip: () => { ctx.save(); ctx.beginPath(); ctx.rect(M.l, M.t - 2, M.w - M.l - M.r + 2, M.h - M.t - M.b + 4); ctx.clip(); }};
		});
		const dot = (p, x, y, r, c, a = 0.9) => { p.ctx.globalAlpha = a; p.ctx.fillStyle = c; p.ctx.beginPath(); p.ctx.arc(p.X(x), p.Y(y), r, 0, 6.2832); p.ctx.fill(); p.ctx.globalAlpha = 1; };
		const box = (p, x0, y0, x1, y1, c, a = 0.85) => { p.ctx.globalAlpha = a; p.ctx.fillStyle = c; p.ctx.fillRect(Math.min(p.X(x0), p.X(x1)), Math.min(p.Y(y0), p.Y(y1)), Math.abs(p.X(x1) - p.X(x0)), Math.abs(p.Y(y1) - p.Y(y0))); p.ctx.globalAlpha = 1; };
		const ease = t => t < 0 ? 0 : t > 1 ? 1 : t * t * (3 - 2 * t);
		let info = null;
		$(code)
		const range = root.querySelector("input"), play = root.querySelector(".bp"), read = root.querySelector(".pl-read");
		let k = +range.value, timer = null, acc = 0, last = 0;
		function show() { P.forEach(p => p.clear()); draw(k); range.value = k; if (info) read.innerHTML = info(k); }
		function stop() { cancelAnimationFrame(timer); timer = null; play.textContent = "▶"; }
		function tick(ts) {
			acc += last ? (ts - last) / 1000 * $(fps) * root.querySelector("select").value : 0; last = ts;
			if (acc >= 1) { k = Math.min(NF, k + Math.floor(acc)); acc %= 1; show(); }
			timer = k < NF ? requestAnimationFrame(tick) : (stop(), null);
		}
		play.onclick = () => { if (timer) return stop(); if (k >= NF) k = 0; play.textContent = "⏸"; last = 0; acc = 0; timer = requestAnimationFrame(tick); };
		root.querySelector(".b0").onclick = () => { stop(); k = 0; show(); };
		root.querySelector(".bn").onclick = () => { stop(); k = NF; show(); };
		range.oninput = () => { stop(); k = +range.value; show(); };
		invalidation.then(stop);
		show();
		})();
		</script></div>""")
	end

	# A pad to draw a density on [0, 1] with the mouse; its value is the drawn height at PAD_N points
	struct DrawPad
		values::Vector{Float64}
	end
	const PAD_N = 120
	function pad_preset(name)
		x = range(0, 1, length = PAD_N); g(m, s) = exp.(-((x .- m) ./ s) .^ 2 ./ 2)
		name == "two hills" ? 0.95g(0.28, 0.09) .+ 0.55g(0.72, 0.12) : name == "spike" ? 0.05 .+ 0.95g(0.4, 0.02) :
		name == "slope" ? 0.05 .+ 0.9x : 0.1 .+ 0.85g(0.5, 0.18)
	end
	DrawPad(; preset = "two hills") = DrawPad(pad_preset(preset))
	AbstractPlutoDingetjes.Bonds.initial_value(p::DrawPad) = p.values
	function Base.show(io::IO, ::MIME"text/html", p::DrawPad)
		presets = Dict(k => pad_preset(k) for k in ["one hill", "two hills", "spike", "slope"])
		print(io, """<span class="drawpad" style="display:inline-block;touch-action:none;font:13px system-ui,sans-serif">
		<div style="margin-bottom:4px">Draw with the mouse, or start from:
		$(join("<button data-p=\"$(k)\">$(k)</button> " for k in ["one hill", "two hills", "spike", "slope", "clear"]))</div>
		<svg viewBox="0 0 560 230" width="560" style="max-width:100%;border:1px solid #cfcfca;border-radius:4px;cursor:crosshair"></svg>
		<script>
		const span = currentScript.parentElement, svg = span.querySelector("svg"), presets = $(js((; (Symbol(k) => v for (k, v) in presets)...)));
		const N = $(PAD_N), W = 560, H = 230, L = 34, R = 12, T = 14, B = 26;
		let v = $(js(p.values)), down = false, last = null;
		const X = i => L + i / (N - 1) * (W - L - R), Yv = y => H - B - y * (H - T - B);
		function draw() {
			let d = "";
			for (let i = 0; i < N; i++) d += (i ? "L" : "M") + X(i).toFixed(1) + "," + Yv(v[i]).toFixed(1);
			svg.innerHTML = `<path d="M\${L},\${T}V\${H - B}H\${W - R}" fill="none" stroke="#333"/>` +
				[0, 0.5, 1].map(t => `<text x="\${L + t * (W - L - R)}" y="\${H - 8}" font-size="11" text-anchor="middle">\${t}</text>`).join("") +
				`<path d="\${d}L\${X(N - 1)},\${Yv(0)}L\${X(0)},\${Yv(0)}Z" fill="#1f5aa6" opacity="0.18"/><path d="\${d}" fill="none" stroke="#1f5aa6" stroke-width="2.4"/>`;
		}
		function pos(ev) {
			const r = svg.getBoundingClientRect(), x = (ev.clientX - r.left) * W / r.width, y = (ev.clientY - r.top) * H / r.height;
			return [Math.max(0, Math.min(N - 1, Math.round((x - L) / (W - L - R) * (N - 1)))), Math.max(0, Math.min(1, (H - B - y) / (H - T - B)))];
		}
		function paint([i1, y1]) {   // fill in all points between the last and the current mouse position
			const [i0, y0] = last || [i1, y1], n = Math.abs(i1 - i0);
			for (let k = 0; k <= n; k++) v[i0 + Math.sign(i1 - i0) * k] = n ? y0 + (y1 - y0) * k / n : y1;
			last = [i1, y1]; draw();
		}
		const send = () => { span.value = v.slice(); span.dispatchEvent(new CustomEvent("input")); };
		svg.onpointerdown = ev => { down = true; last = null; svg.setPointerCapture(ev.pointerId); paint(pos(ev)); };
		svg.onpointermove = ev => { if (down) paint(pos(ev)); };
		svg.onpointerup = () => { if (down) { down = false; send(); } };
		span.querySelectorAll("button").forEach(b => b.onclick = () => { v = b.dataset.p == "clear" ? new Array(N).fill(0) : presets[b.dataset.p].slice(); draw(); send(); });
		span.value = v.slice(); draw();
		</script></span>""")
	end

	# the drawn heights as a density f on [0, 1] (linear interpolation, area 1), and its maximum
	function pad_density(v)
		y = max.(Float64.(v), 0.0); A = sum(y[1:end-1] .+ y[2:end]) / 2 / (PAD_N - 1)
		A <= 1e-9 ? (y .= 1; A = 1.0) : nothing; y ./= A
		f(x) = 0 <= x <= 1 ? (t = x * (PAD_N - 1); i = min(floor(Int, t), PAD_N - 2); (1 - t + i) * y[i+1] + (t - i) * y[i+2]) : 0.0
		f, maximum(y)
	end
end;

# ╔═╡ 89c53ecc-9c5e-5ca9-a179-122d04f3857b
md"""
Example: Firework. Each firework spark is assigned a random direction and speed. Pseudo-random numbers generate values that appear unrelated, quasi-random numbers follow a fixed mathematical sequence (the next number is ``u_{k+1} = (u_k + 0.618) \bmod 1``). These both create different firework sparks.

$(@bind s2_launch Seed(0; label = "New firework"))
"""

# ╔═╡ 17475df6-7511-527c-b48c-adef23ff79e9
md"""
## 4. Normal random numbers

**Question.** The normal distribution is the most used distribution of all, but its CDF ``\Phi`` has no formula, and neither does ``\Phi^{-1}``. How can we produce normal numbers?

### 4.1 Adding uniform numbers

**Idea.** By the central limit theorem, a sum of many independent numbers is approximately normal. So add ``k`` uniform numbers. But how many?

Uniform numbers added ``k`` $(@bind s4_k Select([1, 2, 3, 6, 12, 48]; default = 12)) ``\qquad`` $(@bind s4_seed Seed(0; label = "New sums"))
"""

# ╔═╡ 20c9b80e-140b-502c-80aa-d86184f78466
md"""
### 4.2 The Box–Muller transform

**Question.** ``\Phi^{-1}`` has no formula. Can we still use inversion?

**Idea.** Take a *pair* of independent normal numbers ``(X_1, X_2)`` and look at it as a point in the plane. Its density

```math
f(x_1, x_2) = \frac{1}{2\pi}\, e^{-(x_1^2 + x_2^2)/2}
```

depends only on the distance ``r = \sqrt{x_1^2 + x_2^2}`` from the centre: the cloud of points is round. So describe the point by its **angle** ``\theta`` and its **distance** ``R`` instead of ``x_1`` and ``x_2``:

- **angle:** the cloud is round, so all directions are equally likely: ``\theta = 2\pi U_2``;
- **distance:** ``R`` has the CDF ``F_R(r) = 1 - e^{-r^2/2}``. Unlike ``\Phi``, this CDF can be inverted: ``R = \sqrt{-2\ln(1 - U_1)}``;
- **back to the plane:** ``X_1 = R\cos\theta``, ``X_2 = R\sin\theta``.

Choose the two uniform numbers yourself:

``u_1`` $(@bind s4_u1 Slider(0.01:0.01:0.99; default = 0.6, show_value = true)) ``\qquad`` ``u_2`` $(@bind s4_u2 Slider(0.01:0.01:0.99; default = 0.3, show_value = true)) ``\qquad`` show 1500 points $(@bind s4_cloud CheckBox(default = true)) ``\qquad`` $(@bind s4_bm Seed(0; label = "New points"))
"""

# ╔═╡ cef8c045-6446-5e38-9112-194dfb1dab6e
md"""
## 5. Accept–reject

**Question.** Sometimes we know only the **shape** of a density, for example a curve measured in an experiment, or a formula without a usable ``F^{-1}``. Can we still draw samples from it?

### 5.1 Darts under a curve

**Idea.** Throw darts uniformly into a box around the curve. Keep only the darts **below** the curve. More darts are kept where the curve is high, so the ``x``-values of the kept darts have exactly the density of the curve.

Darts $(@bind s5_n Select([200, 1000, 5000]; default = 1000)) ``\qquad`` $(@bind s5_seed Seed(0; label = "New darts"))
"""

# ╔═╡ 8a160746-b8f0-5ad7-85ec-a561ce566552
@bind s5_pad DrawPad(preset = "two hills")

# ╔═╡ 4d264b32-7d36-5589-b10f-de79b73da398
md"""
## (Optional) Bonus B. Randomness in music

Random numbers can also make sound and music: a plucked string, a waltz composed with dice, and a computer that recognises a rāga.

### B.1 A plucked string made of random numbers

Karplus and Strong (1983) found a simple method to produce the sound of a plucked string. The method starts with a short sequence of uniform random numbers. The length of the sequence is one period of the tone; for 220 Hz at 22 050 samples per second, it is approximately 100 numbers. The sequence is played repeatedly, and at each repetition each number is replaced by the average of itself and its neighbour. The averaging smooths the noise, and the result sounds like a plucked string.

**Caution:** Decrease the volume before you play the sound. The initial noise can be loud.

Tone $(@bind sB_note Select(["A3 (220 Hz)", "E4 (330 Hz)", "A4 (440 Hz)"])) ``\qquad`` $(@bind sB_pluck Seed(0; label = "New pluck"))
"""

# ╔═╡ abe71179-92e7-5e15-a1b7-1f306b79e5fd
md"""
**Repetition** makes the signal periodic with period ``P``. A periodic signal contains only the frequency ``f_0 = 1/P`` and its multiples ``2f_0, 3f_0, \dots``, and the ear hears a tone of pitch ``f_0``. **Averaging** of neighbouring values reduces the noise. The random start values sets the strengths of the frequencies, so each pluck sounds slightly different.

### B.2 Mozart's musical dice game

In 1792, Nikolaus Simrock in Berlin printed a *Musikalisches Würfelspiel* ("musical dice game") attributed to Wolfgang Amadeus Mozart. It composes waltzes with two dice. For each of the 16 bars, the player rolls two dice and adds the results. The sum, from 2 to 12, selects one of eleven precomposed bars from a table. The bars below are played with the plucked string of Section B.1. The sums do not have equal probabilities: a sum of 7 is six times as probable as a sum of 2.

$(@bind sB_dice Seed(0; label = "Roll the dice"))
"""

# ╔═╡ 7205d08e-52ae-51bf-a8f8-24bf95e18fac
md"""
### B.3 Can probability recognise a rāga?

In Indian classical music, a **rāga** is a melodic framework with its own notes and typical movements. Each rāga uses the twelve notes with its own probabilities. A computer can recognise a rāga from these probabilities alone, using **Bayes' theorem** (Tutorial 01): after each note, it multiplies its belief in each rāga by the probability that this rāga plays the note, and rescales.

The four rāgas below are similar. Bhupali and Deshkar even use the same five notes. Listen, then let the computer guess a mystery rāga.

$(@bind sB_raga Seed(0; label = "New melodies"))
"""

# ╔═╡ e02c61e9-8b5c-5ce4-9891-b6becbf05325
md"""
$(@bind sB_myst Seed(0; label = "New mystery rāga"))
"""

# ╔═╡ 55951fa3-a153-5c79-a275-4fe5d3d74120
# Questions: 
begin
	function mcq(question, options::Pair...; correct::Int, head = "Question")
		opts = join("<div class=\"quiz-opt\" data-i=\"$(i)\"><span class=\"quiz-key\">$('A' + i - 1)</span><span>$(tohtml(o[1]))</span></div>" for (i, o) in enumerate(options))
		fbs = join("<div class=\"quiz-fb $(i == correct ? "yes" : "no")\" data-i=\"$(i)\" style=\"display:none\"><b>$(i == correct ? "Correct." : "Not quite.")</b> $(tohtml(o[2]))</div>" for (i, o) in enumerate(options))
		HTML("""<div class="quiz"><div class="quiz-head">$(head)</div><div class="quiz-q">$(tohtml(question))</div>$(opts)$(fbs)
		<script>
		const root = currentScript.parentElement;
		root.querySelectorAll(".quiz-opt").forEach(o => o.onclick = () => {
			root.querySelectorAll(".quiz-fb").forEach(f => f.style.display = f.dataset.i === o.dataset.i ? "block" : "none");
			root.querySelectorAll(".quiz-opt").forEach(x => x.classList.remove("right", "wrong"));
			o.classList.add(+o.dataset.i === $(correct) ? "right" : "wrong");
		});
		</script></div>""")
	end

	# numeric answer: accepts expressions such as 3/4, 2*pi, sqrt(2), exp(-1)
	function numq(question, answer::Real; rtol = 0.02, why = md"", head = "Question")
		HTML("""<div class="quiz"><div class="quiz-head">$(head)</div><div class="quiz-q">$(tohtml(question))</div>
		<div class="quiz-num"><input placeholder="your answer" size="14"> <button class="chk">Check</button> <button class="ans" style="display:none">Show the answer</button></div>
		<div class="quiz-fb no" style="display:none"></div><div class="quiz-fb yes" style="display:none"><b>Correct.</b> $(tohtml(why))</div>
		<script>
		const root = currentScript.parentElement, ans = $(Float64(answer)), inp = root.querySelector("input");
		const [no, yes] = root.querySelectorAll(".quiz-fb"), show = root.querySelector(".ans");
		let tries = 0;
		function value(s) {
			s = s.trim().replace(/−/g, "-").replace(/π/g, "pi").replace(/\\^/g, "**").replace(/,/g, ".");
			if (/[^0-9+\\-*/().eE\\s]/.test(s.replace(/pi|sqrt|exp|ln|log/g, ""))) return NaN;
			try { return Function("return " + s.replace(/pi/g, "Math.PI").replace(/sqrt|exp/g, "Math.\$&").replace(/ln|log/g, "Math.log"))(); } catch (e) { return NaN; }
		}
		function check() {
			const v = value(inp.value); tries++;
			const ok = Math.abs(v - ans) <= $(Float64(rtol)) * Math.abs(ans);
			no.style.display = ok ? "none" : "block"; yes.style.display = ok ? "block" : "none";
			no.textContent = !isFinite(v) ? "This is not a number. Examples: 0.75, 3/4, 2*pi." : (v > ans ? "Too high." : "Too low.") + (tries > 1 ? " Check the reasoning." : " Try again.");
			show.style.display = ok || tries < 2 ? "none" : "inline-block";
		}
		root.querySelector(".chk").onclick = check;
		inp.onkeydown = e => e.key === "Enter" && check();
		show.onclick = () => { no.style.display = "none"; yes.style.display = "block"; yes.querySelector("b").textContent = "Answer: " + +ans.toPrecision(6) + "."; };
		</script></div>""")
	end

	reveal(body; title = "Answer") = Markdown.MD(Markdown.Admonition("hint", title, Any[body]))

	# run the student's code: missing if not done yet, the error text if it fails
	tryrun(f) = try
		r = f(); any(ismissing, r isa Union{Tuple, AbstractArray} ? r : (r,)) ? missing : r
	catch e
		ErrorException(first(sprint(showerror, e), 300))
	end
	function check(result, good, ok_msg, bad_msg; hint = md"")
		box(kind, title, body) = Markdown.MD(Markdown.Admonition(kind, title, Any[body]))
		result === missing && return box("warning", "No answer yet", md"Complete the function above; the check runs as soon as it returns a value.")
		result isa ErrorException && return box("danger", "Not yet: the code stops with an error", Markdown.Code(result.msg))
		good ? box("correct", "Correct", ok_msg) : Markdown.MD(box("danger", "Not yet", bad_msg), box("hint", "Hint", hint))
	end
end;

# ╔═╡ 90e77356-6b0c-5df6-a400-482447d289c9
mcq(md"Which statement about `rand()` is correct?",
	md"It measures noise in the processor, so its numbers are truly random." => md"Physical generators exist, but `rand()` uses a deterministic algorithm.",
	md"It computes each number from the previous state with a fixed recipe, so the seed determines the whole sequence." => md"Right. This is why `Random.seed!(42)` gives the same three numbers every time.",
	md"It reads the clock for every new number." => md"A clock may be used once to choose a seed, but not for every number: that would be slow and the numbers would be strongly related.";
	correct = 2, head = "Question 1.1")

# ╔═╡ 2f7020ae-4ab6-562b-adb6-cb65021184a4
mcq(md"A simulation needs ``10^6`` numbers. It uses an LCG with period ``65\,536``. A student argues: *every value appears exactly once per period, so the numbers are perfectly uniform.* What is wrong?",
	md"Nothing; uniformity is all that matters." => md"The numbers must also behave like independent draws. Here the same 65 536 numbers are repeated about 15 times in the same order.",
	md"The simulation uses only 65 536 different numbers, each about 15 times in the same order: the result is not that of ``10^6`` independent numbers." => md"Right. The period must be far larger than the number of draws.",
	md"An LCG cannot have the period 65 536." => md"With ``m = 2^{16}`` and good ``a, c`` it can. The problem is elsewhere.";
	correct = 2, head = "Question 1.2")

# ╔═╡ 9b4a22d7-4cf7-50ec-bb12-9d367c8d1902
mcq(md"Is this safe?",
	md"Yes: every job has a different seed." => md"Different seeds only give different starting points on the same cycle.",
	md"No: together the jobs draw ``10^{10}`` numbers, more than the period, so the jobs must reuse each other's numbers." => md"Right. ``1000 \cdot 10^7 = 10^{10} > 2.1 \cdot 10^9``: overlaps are certain, and consecutive seeds of an LCG can also give related sequences.",
	md"No, because 1000 jobs cannot run at the same time." => md"The problem is not timing but the numbers the jobs use.";
	correct = 2, head = "Question 1.3")

# ╔═╡ ad7d2f3e-7ede-5764-a3bc-47697c3cee6f
mcq(md"Why do the quasi-random sparks lie on a spiral instead of filling a round burst?",
	md"The speed is fixed by the direction: the next number is always ``u + 0.618`` (mod 1), so each direction has exactly one speed." => md"Right. Direction and speed must be independent. Quasi-random numbers are not.",
	md"Quasi-random numbers do not cover all directions." => md"All directions appear; look at the speeds.",
	md"There are too few sparks." => md"More sparks would only make the spiral sharper.";
	correct = 1, head = "Question 2.2")

# ╔═╡ 85055384-1e52-566e-b49c-ecf70de269fd
mcq(md"You simulate the arrival times of customers in a shop. Which numbers do you use?",
	md"Quasi-random: the arrivals are then spread evenly over the day." => md"Real arrivals are independent and do clump. Quasi-random arrivals would be far too regular.",
	md"Pseudo-random: the arrivals must be independent." => md"Right.";
	correct = 2, head = "Question 2.3")

# ╔═╡ dd337eb5-e171-5a94-89fb-a922a9ca1437
mcq(md"You want to test a model at 100 parameter combinations spread over a square of possible values. Which points do you use?",
	md"Pseudo-random points." => md"They work, but they leave holes and clumps: some regions are never tested.",
	md"Quasi-random points." => md"Right: the goal is even coverage, not independence.";
	correct = 2, head = "Question 2.4")

# ╔═╡ 18054f38-79ed-5262-aab7-cdd6343642a4
mcq(md"What will the test show?",
	md"Nothing special: the molecules are still uniformly spread." => md"Evenly spread is not the same as random: real independent molecules clump by chance.",
	md"The molecules look more regular than independent molecules ever are; the program will report that they avoid each other, a wrong conclusion about biology." => md"Right.",
	md"The program will crash." => md"The program runs normally; the result is wrong.";
	correct = 2, head = "Question 2.5")

# ╔═╡ 3d3c8553-033a-5e22-8932-a55dadc9f949
numq(md"Compute ``f_Y(0.25)``.", 1.0; rtol = 0.01,
	why = md"``f_Y(0.25) = \frac{1}{2\sqrt{0.25}} = \frac{1}{2 \cdot 0.5} = 1``.", head = "Question 3.2")

# ╔═╡ 28d74357-791f-59ab-9f58-47cada7e736d
numq(md"Check the result: compute the area ``\int_0^1 \frac{1}{2\sqrt{y}}\,dy``.", 1.0; rtol = 0.01,
	why = md"``\int_0^1 \frac{1}{2\sqrt{y}}\,dy = \big[\sqrt{y}\big]_0^1 = 1``. Every density should have area 1; this is a quick test of any transformation.", head = "Question 3.3")

# ╔═╡ e1e1dbbd-ec2d-57c8-a9c3-011375a2a3bb
mcq(md"A student transforms a uniform ``X`` on ``(0, 1)`` with ``Y = 2X`` and writes: *the density of ``Y`` is 1 on ``(0, 2)``.* What is wrong?",
	md"Nothing." => md"Then the total area would be 2.",
	md"The factor ``dx/dy = 1/2`` is missing: the density is ``1/2`` on ``(0, 2)``." => md"Right. See Figure 3.1. check if the area is 1.",
	md"``Y`` is not uniform." => md"See Figure 3.1. ``Y`` is uniform, on ``(0, 2)``.";
	correct = 2, head = "Question 3.4")

# ╔═╡ dca0d1b0-8755-5bfa-8650-b342cfd4306e
numq(md"``X`` is uniform on ``(0, 1)`` and ``Y = 1/X``. Compute ``f_Y(2)``.", 0.25; rtol = 0.01,
	why = md"``x = g^{-1}(y) = 1/y``, ``\left|\frac{d}{dy}\frac{1}{y}\right| = \frac{1}{y^2}``, so ``f_Y(y) = 1/y^2`` for ``y > 1`` and ``f_Y(2) = 0.25``.", head = "Question 3.5")

# ╔═╡ 9dee6ff4-69aa-5b92-b949-a77c05e3f489
mcq(md"For ``Y = 1/X`` a student forgets the absolute value and gets ``f_Y(y) = -1/y^2``. What went wrong?",
	md"``g`` is decreasing, so ``F_Y = 1 - F_X(g^{-1}(y))``; the minus sign of this case cancels the negative derivative of ``g^{-1}``." => md"Right. A density can never be negative: the absolute value takes care of the decreasing case.",
	md"``1/X`` has no density." => md"It has the density ``1/y^2`` on ``(1, \infty)``.",
	md"The formula only works for ``X`` uniform." => md"The formula works for every continuous ``X`` and strictly monotone ``g``.";
	correct = 1, head = "Question 3.6")

# ╔═╡ 1b59f513-f31c-588b-a4db-66ef5928a11f
mcq(md"What happened?",
	md"`rand()` returned exactly 0, and ``-\ln 0 = \infty``." => md"Right. `rand()` returns values in ``[0, 1)``: 0 is possible, 1 is not. Use `-log(1 - rand())`, where ``1 - U`` is in ``(0, 1]``, or Julia's `randexp()`.",
	md"`rand()` returned a negative number." => md"`rand()` never returns negative numbers.",
	md"The exponential distribution sometimes produces infinite values." => md"An exponential random number is always finite.";
	correct = 1, head = "Question 3.8")

# ╔═╡ 95d3ed2d-0e38-596b-a2df-8b081b432580
mcq(md"The sum of 12 uniform numbers looks perfectly normal in the histogram. What is different in the right panel?",
	md"Nothing: the curves agree everywhere." => md"Look far from the middle, where the density is very small.",
	md"In the tails the sum is less likely than a normal number, and it can never be smaller than 0 or larger than 12." => md"Right. The tails do not agree. Rare, extreme values are described by the tails.",
	correct = 2, head = "Question 4.1")

# ╔═╡ b2761858-949e-55b9-a406-0730522e7fa3
mcq(md"Why is it allowed to choose the angle and the distance separately, each from its own uniform number?",
	md"Because the density depends only on the distance: the angle tells nothing about the distance, so the two are independent." => md"Right. This splits one hard two-dimensional problem into two easy one-dimensional ones.",
	md"Because ``X_1`` and ``X_2`` are uniform." => md"``X_1`` and ``X_2`` are normal; ``U_1`` and ``U_2`` are uniform.",
	md"Because every pair of random numbers can be split this way." => md"Only a round cloud has an angle that is independent of the distance.";
	correct = 1, head = "Question 4.4")

# ╔═╡ 809641dd-b073-52e8-bec6-71e5ee792124
numq(md"How many uniform numbers does Box–Muller need for ``10^6`` normal numbers?", 1e6; rtol = 0.001,
	why = md"Two uniform numbers give two normal numbers: one uniform number per normal number.", head = "Question 4.5")

# ╔═╡ c616c604-383e-59db-9c51-9a9e96223b3e
numq(md"The density of Beta(2, 5) has its maximum ``2.4576``. With the box ``[0, 1] \times [0, 2.4576]``, what fraction of the darts is kept?", 1 / 2.4576; rtol = 0.01,
	why = md"Area under the density 1, area of the box ``2.4576``: ``1/2.4576 = 0.407``. On average ``2.46`` darts per sample.", head = "Question 5.1")

# ╔═╡ 181d2899-b8e4-53af-97ff-a688f914b04a
numq(md"The semicircle density ``f(x) = \frac{2}{\pi}\sqrt{1 - x^2}`` on ``[-1, 1]`` has its maximum ``2/\pi`` at 0. With the box ``[-1, 1] \times [0, 2/\pi]``, what fraction of the darts is kept?", π / 4; rtol = 0.01,
	why = md"Area of the box ``2 \cdot \frac{2}{\pi} = \frac{4}{\pi}``, area under the curve 1: the fraction is ``\pi/4 \approx 0.785``.", head = "Question 5.2")

# ╔═╡ d06eb53c-7fc9-52bd-b378-5ed24df48462
mcq(md"The envelope is below the density on some interval. What happens?",
	md"The program stops with an error." => md"Nothing in the algorithm checks the envelope; it runs normally.",
	md"On that interval every dart is kept, so the samples follow the envelope there instead of ``f``." => md"Right. The condition ``\mu g \ge f`` must be checked by the programmer.",
	md"Only the speed changes." => md"The samples themselves are wrong.";
	correct = 2, head = "Question 5.3")

# ╔═╡ f4249eb6-ab43-55fa-97ac-e32d45eef240
numq(md"**Case A.** About how many darts are thrown for one kept sample?", 41; rtol = 0.15,
	why = md"The box has the height ``1.02 \cdot f(0.4) \approx 40.7`` and the width 1, the area under the curve is 1: about 41 darts per sample. Almost all of the box is empty.", head = "Question 5.4")

# ╔═╡ d3e14753-8a81-51d6-aedc-4dfe1d5114d2
mcq(md"**Case B.** Compare the histogram of the kept values with the density. What is wrong?",
	md"Nothing: the histogram matches the density." => md"Look at ``x > 5``.",
	md"No value above 5 is ever produced, although ``P(X > 5) = 1/6``." => md"Right. A box has a finite width, but this density goes on forever. Every box loses part of the distribution.",
	md"Too many darts are rejected, but the samples are right." => md"The samples themselves are wrong.";
	correct = 2, head = "Question 5.5")

# ╔═╡ 7cef87e8-2181-5e3d-a10c-5500c4cd89cf
numq(md"In 3 dimensions, what fraction of uniform darts in the cube ``[-1, 1]^3`` land in the ball of radius 1?", π / 6; rtol = 0.01,
	why = md"Volume of the ball ``\frac{4}{3}\pi``, volume of the cube ``2^3 = 8``: ``\frac{4\pi/3}{8} = \frac{\pi}{6} \approx 0.52``. In 2 dimensions it was ``\pi/4 \approx 0.79``. With increasing dimensions, it gets worse.", head = "Question 5.7")

# ╔═╡ 97fa04c9-22c5-5b24-b088-11c53be9075c
mcq(md"Is this a good plan?",
	md"Yes: accept–reject always works." => md"It works in principle. But the fraction kept was 0.79 in 2 dimensions and 0.52 in 3: what happens in 30?",
	md"No: only about ``2 \cdot 10^{-14}`` of the darts land in the ball, so a single point needs about ``5 \cdot 10^{13}`` darts." => md"Right. In high dimensions almost all of the cube lies outside the ball. Other methods are needed (for example Markov chain Monte Carlo, We will learn this in a few weeks).",
	md"No, because a ball has no density." => md"The uniform distribution on a ball has a density.";
	correct = 2, head = "Question 5.8")

# ╔═╡ fd9d7dae-d6bf-5e1e-9988-204c749a4cb0
numq(md"**Problem 1.** ``X`` is uniform on ``(0, 1)`` and ``Y = 3X + 2``. Compute ``f_Y(3)``.", 1 / 3; rtol = 0.01,
	why = md"``Y`` is uniform on ``(2, 5)``: the axis is stretched by 3, so ``f_Y = 1/3``.", head = "Problem 1")

# ╔═╡ 7cdf7df5-0aff-568a-b807-b2669803d649
numq(md"**Problem 2.** ``X`` is uniform on ``(0, 1)`` and ``Y = -\ln X``. Compute ``f_Y(1)``.", exp(-1); rtol = 0.01,
	why = md"``x = e^{-y}``, ``\left|\frac{dx}{dy}\right| = e^{-y}``, so ``f_Y(y) = e^{-y}`` and ``f_Y(1) = e^{-1} = 0.368``: ``Y`` is exponential.", head = "Problem 2")

# ╔═╡ c93775d1-47c1-599a-90f9-8ed4f356a247
numq(md"**Problem 3.** ``X`` is uniform on ``(0, 1)`` and ``Y = X^3``. Compute ``f_Y(0.125)``.", 4 / 3; rtol = 0.01,
	why = md"``x = 0.125^{1/3} = 0.5`` and ``g'(0.5) = 3 \cdot 0.25 = 0.75``, so ``f_Y = 1/0.75 = 1.333``.", head = "Problem 3")

# ╔═╡ bff3f007-46ba-54e7-a165-2c1d7cdbddb4
numq(md"**Problem 4a.** The density ``f(x) = c\,x`` on ``[0, 2]``. Compute ``c``.", 0.5; rtol = 0.01,
	why = md"``\int_0^2 c\,x\,dx = 2c = 1``, so ``c = 1/2``.", head = "Problem 4")

# ╔═╡ 8213971a-ba13-51fe-8e37-fe31c2b05dcb
numq(md"**Problem 6.** The density ``f(x) = 3x^2`` on ``[0, 1]`` is sampled by darts in the box ``[0, 1] \times [0, 3]``. What fraction of the darts is kept?", 1 / 3; rtol = 0.01,
	why = md"Area under ``f``: 1. Area of the box: 3. Fraction ``1/3``: three darts per sample on average.", head = "Problem 6")

# ╔═╡ db38b8d7-fbdc-5f81-b280-d30d26ab7fba
mcq(md"**Problem 7.** ``U`` is uniform on ``(0, 1)``. Which of these is **not** uniform on ``(0, 1)``?",
	md"``1 - U``" => md"1 - U is uniform.",
	md"``U^2``" => md"Right: squaring squeezes the axis near 0, so the values pile up there (density ``1/(2\sqrt y)``).",
	md"``F(X)`` for an exponential ``X``" => md"The CDF values of any continuous distribution are uniform.";
	correct = 2, head = "Problem 7")

# ╔═╡ f8277d29-7dfe-5f2c-a832-d4397bc5aafa
# Sections 1 and 2: uniform random numbers
begin
	sub(n) = join(Dict(zip("0123456789", "₀₁₂₃₄₅₆₇₈₉"))[c] for c in string(n))

	# Figure 1.1: the LCG drawn as jumps on a clock with m positions
	function fig_lcg(m, a, c, z0)
		zs = [mod(z0, m)]
		while true                                  # run until a state comes back
			z = mod(a * zs[end] + c, m); push!(zs, z)
			count(==(z), zs) == 2 && break
		end
		L = length(zs) - 1; period = L + 1 - findfirst(==(zs[end]), zs)
		ang(z) = π / 2 - 2π * z / m
		f = fig(w = 360, h = 360, xr = (-1.3, 1.3), yr = (-1.3, 1.3), axes = false, l = 4, r = 4, t = 22, b = 4, title = "m = $(m) positions")
		for z in 0:m-1
			dots!(f, [cos(ang(z))], [sin(ang(z))]; c = "#bdbdb7", r = 2.6)
			z % (m <= 16 ? 1 : 5) == 0 && text!(f, 1.15cos(ang(z)), 1.15sin(ang(z)), string(z); anchor = "middle", size = 11, c = "#666")
		end
		dots!(f, [cos(ang(zs[1]))], [sin(ang(zs[1]))]; c = COL.red, r = 6, o = 0, over = true)
		for i in 1:L                                # jump i: thin from step i on, bold only at step i
			p, q = 0.95 .* (cos(ang(zs[i])), sin(ang(zs[i]))), 0.95 .* (cos(ang(zs[i+1])), sin(ang(zs[i+1])))
			col = i == L ? COL.red : COL.blue
			arrow!(f, p..., q...; c = col, w = 1.2, head = 6, op = 0.45, s = i)
			arrow!(f, p..., q...; c = col, w = 2.6, head = 9, o = i)
			dots!(f, [q[1] / 0.95], [q[2] / 0.95]; c = COL.red, r = 6, o = i, over = true)
		end
		h = fig(w = 680, h = 64, yticks = [], xlabel = "uᵢ = zᵢ / m", l = 20, b = 34, t = 6)
		foreach(i -> seg!(h, zs[i+1] / m, 0.1, zs[i+1] / m, 0.9; c = COL.orange, w = 2, s = i), 1:L)
		reads = ["z₀ = $(zs[1])"; ["z$(sub(i)) = ($(a) · $(zs[i]) + $(c)) mod $(m) = $(zs[i+1])" * (i == L ? ": repeated, the cycle closes" : "") for i in 1:L]]
		player(figure(f), h, Markdown.parse("Period: **$(period)**"); n = L, start = L, fps = 3, reads,
			caption = md"**Figure 1.1.** The LCG as a clock with ``m`` positions. Orange ticks: the numbers ``u_i``.")
	end

	# Figure 1.2: P processors start at random points of a cycle of length T and use n numbers each
	const PERIODS = [2.0^31 => "2³¹ ≈ 2.1·10⁹ (a classic LCG)", 2.0^64 => "2⁶⁴ ≈ 1.8·10¹⁹", 2.0^256 => "2²⁵⁶ − 1 ≈ 10⁷⁷ (Julia's Xoshiro256++)"]
	function fig_streams(P, n, T, seed)
		st = sort(rand(Xoshiro(91 + seed), P)); a = n / T              # starts and stretch length as fractions of the cycle
		gap = [mod(st[mod1(i + 1, P)] - st[i], 1.0) for i in 1:P]
		over = [gap[i] < a || gap[mod1(i - 1, P)] < a for i in 1:P]
		i1 = argmin(gap); i2 = mod1(i1 + 1, P)                        # the two closest starts
		shown = P <= 10 ? (1:P) : first(sort(unique([i1; i2; round.(Int, range(1, P, length = 8))])), 10)
		name = Dict(k => "P$(j)" for (j, k) in enumerate(shown))
		ang(t) = π / 2 - 2π * t
		arc(t0, t1) = (θ = range(ang(t0), ang(t1), length = 40); (cos.(θ), sin.(θ)))
		f = fig(w = 330, h = 330, xr = (-1.55, 1.55), yr = (-1.55, 1.55), axes = false, l = 4, r = 4, t = 26, b = 4,
			title = P <= 10 ? "the cycle, $(P) processors" : "the cycle, 10 of $(P) processors")
		line!(f, arc(0, 1)...; c = "#dcdcd6", w = 9)
		prev = -1.0
		for k in shown
			line!(f, arc(st[k], st[k] + max(a, 0.02))...; c = over[k] ? COL.red : COL.blue, w = 9)   # stretch, drawn at least 2 % long
			ρ = st[k] - prev < 0.05 || (k == last(shown) && st[k] > 0.95 + st[first(shown)] - 1) ? 1.42 : 1.2   # move close labels outwards
			prev = ρ == 1.2 ? st[k] : -1.0
			dots!(f, [cos(ang(st[k]))], [sin(ang(st[k]))]; c = COL.ink, r = 3)
			text!(f, ρ * cos(ang(st[k])), ρ * sin(ang(st[k])), name[k]; anchor = "middle", size = 11.5, bold = true, c = over[k] ? COL.red : COL.ink)
		end
		d = gap[i1] * T                                               # zoom on the closest pair, in numbers
		z = fig(w = 340, h = 200, xr = (-0.15n, 1.08max(d + n, 2n)), yr = (0, 3), yticks = [1, 2], yticklabels = [name[i2], name[i1]],
			xlabel = "position on the cycle (numbers)", title = "zoom: the two closest starts", l = 44, t = 30)
		rect!(z, 0, 1.7, n, 2.3; op = 0.55); rect!(z, d, 0.7, d + n, 1.3; op = 0.55)
		d < n && (rect!(z, d, 1.7, n, 2.3; fill = COL.red, op = 0.85); rect!(z, d, 0.7, n, 1.3; fill = COL.red, op = 0.85))
		p = P * a >= 1 ? 1.0 : P^2 * a < 1e-6 ? P * (P - 1) * a : -expm1((P - 1) * log1p(-P * a))   # chance of any overlap
		figure(f, z, Markdown.parse("");
			caption = md"**Figure 1.2.** Left: the cycle of the generator as a ring. Each processor starts where its seed points and uses the next numbers (drawn enlarged). Red: overlapping stretches. Right: the two closest starts; red: numbers that both use.")
	end

	# Section 2: point sets in the unit square
	const PHI2 = 1.324717957244746                                    # plastic number, for the 2D additive recurrence
	const SEQS = ["pseudo-random: Julia's Xoshiro", "pseudo-random: small LCG (m = 2048)",
		"quasi-random: additive recurrence", "quasi-random: Halton", "quasi-random: Sobol"]
	vdc(k, b) = (v = 0.0; f = 1 / b; while k > 0; v += f * (k % b); k ÷= b; f /= b; end; v)   # digits of k in base b, mirrored
	function sobol2(n)                                                # 2D Sobol points
		m = UInt32[1]; for _ in 2:32; push!(m, (m[end] << 1) ⊻ m[end]); end
		V1 = [UInt32(1) << (32 - j) for j in 1:32]; V2 = [m[j] << (32 - j) for j in 1:32]
		x = y = UInt32(0)
		map(1:n) do i
			c = trailing_ones(i - 1) + 1; x ⊻= V1[c]; y ⊻= V2[c]
			(x / 2.0^32, y / 2.0^32)
		end
	end
	function points(name, n, seed)
		r = Xoshiro(17 + 101seed)
		name == SEQS[1] && return [(rand(r), rand(r)) for _ in 1:n]
		name == SEQS[2] && (z = accumulate((z, _) -> mod(65z + 1, 2048), 1:2n; init = mod(1 + 37seed, 2048)) ./ 2048; return [(z[2k-1], z[2k]) for k in 1:n])
		name == SEQS[3] && return [(mod(0.5 + k / PHI2, 1), mod(0.5 + k / PHI2^2, 1)) for k in 1:n]
		name == SEQS[4] && return [(vdc(k, 2), vdc(k, 3)) for k in 1:n]
		sobol2(n)
	end

	# Figure 2.1: points land one by one; shaded cells of a 10 × 10 grid are still empty
	function fig_pq(left, right, n, seed)
		panel(name) = begin
			pts = points(name, n, seed); g = fig(w = 330, h = 345, title = split(name, ": ")[2], xticks = [0, 0.5, 1], yticks = [0, 0.5, 1])
			full = falses(10, 10); foreach(p -> full[min.(10, 1 .+ floor.(Int, 10 .* p))...] = true, pts)
			foreach(I -> full[I] || rect!(g, (I[1] - 1) / 10, (I[2] - 1) / 10, I[1] / 10, I[2] / 10; fill = COL.ochre, op = 0.16, s = n), CartesianIndices(full))
			foreach(k -> dots!(g, [pts[k][1]], [pts[k][2]]; c = startswith(name, "pseudo") ? COL.orange : COL.purple, r = n > 500 ? 2.2 : 3.2, s = k), 1:n)
			g, count(!, full)
		end
		(a, ea), (b, eb) = panel(left), panel(right)
		player(figure(a, b); n, start = n, fps = n <= 200 ? 20 : 120, reads = [fill("", n); "Empty cells: left $(ea), right $(eb)"], label = "point",
			caption = md"**Figure 2.1.** Points in the unit square. Orange: pseudo-random; purple: quasi-random. Shaded: empty cells of a ``10 \times 10`` grid.")
	end

	# draw a grey image (values 0…1) on a canvas
	function canvas_image(img; w = 210, label = "", note = "")
		H, W = size(img); data = round.(Int, 255 .* vec(permutedims(img[end:-1:1, :])))
		"""<div><div style="font:600 13px system-ui;margin:0 0 3px 2px">$(label)</div><canvas width="$(W)" height="$(H)" style="width:$(w)px;image-rendering:pixelated;border-radius:3px"></canvas>
		<div style="font:12px system-ui;color:#555">$(note)</div><script>
		const cv = currentScript.parentElement.querySelector("canvas"), v = $(js(data)), c = cv.getContext("2d"), im = c.createImageData($(W), $(H));
		v.forEach((a, i) => im.data.set([40 + 0.84 * a, 36 + 0.82 * a, 30 + 0.74 * a, 255], 4 * i));
		c.putImageData(im, 0, 0);
		</script></div>"""
	end

	# Figure 2.3: sparks with direction and speed from the next number
	function fig_firework(launch)
		n = 260; φ = (sqrt(5) - 1) / 2
		up = rand(Xoshiro(700 + launch), 2n); uq = [mod(0.13launch + k * φ, 1) for k in 1:2n]
		sparks(u) = (s = 0.35 .+ 0.65 .* u[2:2:end]; θ = 2π .* u[1:2:end]; (x = round.(s .* cos.(θ), digits = 3), y = round.(s .* sin.(θ), digits = 3)))
		cv(t) = """<div><div style="font:600 13px system-ui;margin:0 0 3px 2px">$(t)</div><canvas width="640" height="480" style="width:320px;max-width:100%;border-radius:4px;background:#070814"></canvas></div>"""
		HTML("""<div class="fg"><div class="fg-row">$(cv("pseudo-random"))$(cv("quasi-random"))</div><button>▶ launch again</button>
		<div class="fg-cap">$(tohtml(md"**Figure 2.3.** A firework with 260 sparks: direction from one random number, speed from another."))</div>
		<script>
		(() => {
		const root = currentScript.parentElement, cvs = root.querySelectorAll("canvas"), T = 2.6;
		const sets = [[$(js(sparks(up))), "255,170,60"], [$(js(sparks(uq))), "190,140,255"]];
		let raf = null, t0 = null;
		function frame(ts) {                       // positions move outwards, fall slowly and fade
			t0 = t0 ?? ts; const t = Math.min(T, (ts - t0) / 1000), s = 1 - Math.exp(-2.2 * t);
			sets.forEach(([p, col], k) => {
				const c = cvs[k].getContext("2d");
				c.fillStyle = "rgba(7,8,20,0.28)"; c.fillRect(0, 0, 640, 480);
				c.fillStyle = "rgba(" + col + "," + Math.max(0.35, 1 - t / T) + ")";
				p.x.forEach((x, i) => { c.beginPath(); c.arc(320 + 190 * s * x, 200 - 190 * s * p.y[i] + 17 * t * t, 2.6, 0, 6.2832); c.fill(); });
			});
			if (t < T) raf = requestAnimationFrame(frame);
		}
		function go() { cancelAnimationFrame(raf); t0 = null; cvs.forEach(c => { const x = c.getContext("2d"); x.fillStyle = "#070814"; x.fillRect(0, 0, 640, 480); }); raf = requestAnimationFrame(frame); }
		root.querySelector("button").onclick = go; go();
		invalidation.then(() => cancelAnimationFrame(raf));
		})();
		</script></div>""")
	end
end;

# ╔═╡ 093c1425-a05d-5573-b6b7-be1f2ff50a51
fig_lcg(s1_m, s1_a, s1_c, s1_z0)

# ╔═╡ 83489c02-78f1-574a-9dfc-eaaea8eddc08
md"""
### 1.2 Many processors

**Question.** A large simulation runs on 100 processors at the same time. Each processor needs its own random numbers. Can they all use the same generator with different seeds? Do we need to worry about the period?

Think of the whole cycle of the generator as a ring of ``T`` numbers. A seed picks a starting point on the ring; a processor then uses the next ``n`` numbers, a stretch of the ring. Different seeds give different starting points, but the stretches can **overlap**: then two processors compute with the same numbers, and their results are not independent.

!!! tasks "Tasks"
    1. Choose the classic LCG (period ``2^{31}``), 100 processors and ``10^7`` numbers each. What happens?
    2. Keep everything and switch to Julia's generator.

Processors $(@bind s1_P Select([2, 10, 100, 1000]; default = 100)) ``\qquad`` numbers per processor $(@bind s1_n Select([10^6 => "10⁶", 10^7 => "10⁷", 10^9 => "10⁹", 10^12 => "10¹²"]; default = 10^7)) ``\qquad`` period $(@bind s1_T Select(PERIODS; default = 2.0^31)) ``\qquad`` $(@bind s1_ps Seed(0; label = "New seeds"))
"""

# ╔═╡ bc82d9b4-8a8a-5561-a6e7-052c79e293c7
fig_streams(s1_P, s1_n, s1_T, s1_ps)

# ╔═╡ d425a268-6583-5bb9-9072-ca59ed4a9e0d
md"""
## 2. Pseudo-random or quasi-random?

**Question.** Throw 200 random points into a square. Do they cover it evenly?

Press ▶ in Figure 2.1 and watch where the points land. Then compare other generators.

Left $(@bind s2_left Select(SEQS; default = SEQS[1])) ``\qquad`` right $(@bind s2_right Select(SEQS; default = SEQS[3]))

Points $(@bind s2_n Select([50, 200, 1000]; default = 200)) ``\qquad`` $(@bind s2_seed Seed(0; label = "New points"))
"""

# ╔═╡ bc99d869-1a81-57b5-bd0b-84c9560f3ed7
fig_pq(s2_left, s2_right, s2_n, s2_seed)

# ╔═╡ 05045e3e-2732-5912-b226-e3472546f874
fig_firework(s2_launch)

# ╔═╡ f6959cf2-a6b3-555e-990c-d3a2341965a0
# Section 3: from uniform to any distribution
begin
	pick(list, name) = last(list[findfirst(==(name), first.(list))])   # value of a (name => value) option

	# histogram for code cells
	function quick_hist(v; bins = 40)
		e = collect(range(extrema(v)..., length = bins + 1)); h = histdata(v, e)
		hist!(fig(w = 640, h = 220, xr = extrema(v), yr = (0, 1.15maximum(h)), xlabel = "value", ylabel = "density", l = 56), e, h)
	end

	# Figure 3.1/3.2: Y = g(X); X, the function g, and the density of Y with fmax as the axis limit
	const TRANS = [
		"X ~ Uniform(0, 1), Y = X²" => (X = Uniform(0, 1), g = x -> x^2, fy = y -> 1 / 2sqrt(y), yr = (0, 1), fmax = 3.0),
		"X ~ Uniform(0, 1), Y = √X" => (X = Uniform(0, 1), g = sqrt, fy = y -> 2y, yr = (0, 1), fmax = 2.4),
		"X ~ Uniform(0, 1), Y = 2X" => (X = Uniform(0, 1), g = x -> 2x, fy = y -> 0.5, yr = (0, 2), fmax = 1.3),
		"X ~ Uniform(0, 1), Y = 1 − X" => (X = Uniform(0, 1), g = x -> 1 - x, fy = y -> 1.0, yr = (0, 1), fmax = 1.6),
		"X ~ Uniform(0, 1), Y = 1/X" => (X = Uniform(0, 1), g = x -> 1 / x, fy = y -> y > 1 ? 1 / y^2 : 0.0, yr = (0, 6), fmax = 1.3),
		"X ~ Exponential(1), Y = F(X) = 1 − e⁻ˣ" => (X = Exponential(1), g = x -> 1 - exp(-x), fy = y -> 1.0, yr = (0, 1), fmax = 1.6),
		"X ~ Normal(0, 1), Y = F(X) = Φ(X)" => (X = Normal(0, 1), g = x -> cdf(Normal(), x), fy = y -> 1.0, yr = (0, 1), fmax = 1.6)]
	const TRANS_NAMES = first.(TRANS)
	function fig_lines(name; fignum = "3.1")
		S = pick(TRANS, name); xr = (max(-3.5, minimum(S.X)), min(3.5, maximum(S.X))); yr = S.yr
		xk = quantile.(S.X, ((1:25) .- 0.5) ./ 25); yk = S.g.(xk)                 # 25 equally likely values of X
		y = S.g.(rand(Xoshiro(11), S.X, 20_000))
		e = collect(range(yr..., length = 31)); h = histdata(filter(v -> yr[1] <= v < yr[2], y), e) .* mean(v -> yr[1] <= v < yr[2], y)
		tx = range(xr[1] + 1e-4, xr[2], length = 300); ty = range(yr[1] + 1e-3, yr[2] - 1e-3, length = 300)
		purple(f, x0, y0, x1, y1) = seg!(f, x0, y0, x1, y1; c = COL.purple, op = 0.55)
		L = fig(w = 210, h = 300, xr = (0, S.fmax), yr = yr, xlabel = "density", ylabel = "y", title = "Y = g(X)", l = 46, b = 46, t = 30)
		foreach(j -> rect!(L, 0, e[j], min(h[j], S.fmax), e[j+1]; fill = COL.orange, op = 0.55, stroke = "white", sw = 0.6), eachindex(h))
		foreach(v -> purple(L, 0, v, S.fmax, v), yk)
		line!(L, min.(S.fy.(ty), S.fmax), ty; w = 2.2)
		G = fig(w = 420, h = 300, xr = xr, yr = yr, title = "the function g", l = 40, b = 46, t = 30)
		foreach((x, v) -> (purple(G, x, yr[1], x, v); purple(G, xr[1], v, x, v)), xk, yk)
		line!(G, tx, S.g.(tx); c = COL.ink, w = 2.4)
		B = fig(w = 420, h = 170, xr = xr, yr = (0, 1.25maximum(pdf.(S.X, tx))), xlabel = "x", title = "X", l = 40, b = 40, t = 26)
		foreach(x -> purple(B, x, 0, x, B.yr[2]), xk)
		line!(B, tx, pdf.(S.X, tx); w = 2.2)
		vstack(figure(L, G), figure(fig(w = 210, h = 170, axes = false), B);
			caption = Markdown.parse("**Figure $(fignum).** 25 values of ``X`` with equal probability (purple lines, bottom) go up to the curve ``g`` and across to the ``y``-axis. Left: histogram of 20 000 values of ``Y`` (orange) and its density (blue)."))
	end

	# Figure 3.3: samples of X, the CDF, and the values F(X)
	const FLAT_DISTS = ["Exponential(1)" => Exponential(1), "Normal(0, 1)" => Normal(0, 1), "Gamma(2, 1.5)" => Gamma(2, 1.5), "Beta(2, 5)" => Beta(2, 5),
		"mixture of Normal(−2, 0.7) and Normal(1.5, 1)" => MixtureModel([Normal(-2, 0.7), Normal(1.5, 1)], [0.4, 0.6])]
	function fig_flat(name, n, seed)
		D = pick(FLAT_DISTS, name); x = rand(Xoshiro(5 + 13seed), D, n)
		lo, hi = quantile(D, 0.001), quantile(D, 0.999); t = range(lo, hi, length = 300)
		e = collect(range(lo, hi, length = 31)); h = histdata(clamp.(x, lo, hi), e)
		a = fig(w = 225, h = 230, xr = (lo, hi), yr = (0, 1.3max(maximum(pdf.(D, t)), maximum(h))), xlabel = "x", ylabel = "density", title = "samples of X", l = 50)
		hist!(a, e, h); line!(a, t, pdf.(D, t); w = 2.2)
		b = fig(w = 225, h = 230, xr = (lo, hi), yr = (0, 1.05), xlabel = "x", ylabel = "F(x)", title = "CDF", l = 50)
		line!(b, t, cdf.(D, t); w = 2.4)
		for xi in x[1:min(12, n)]                                         # a few samples sent through the CDF
			seg!(b, xi, 0, xi, cdf(D, xi); c = COL.orange, op = 0.7); seg!(b, lo, cdf(D, xi), xi, cdf(D, xi); c = COL.purple, op = 0.7)
		end
		c = fig(w = 225, h = 230, yr = (0, 1.8), xlabel = "u = F(x)", ylabel = "density", title = "values of F(X)", l = 50, xticks = [0, 0.5, 1])
		hist!(c, 0:0.05:1, histdata(cdf.(D, x), 0:0.05:1); c = COL.purple, op = 0.7); hline!(c, 1.0; c = COL.ink, dash = "5 4", w = 1.6)
		figure(a, b, c; caption = md"**Figure 3.3.** Left: samples (orange) and density (blue). Middle: each sample is mapped through the CDF (12 shown). Right: the histogram of ``F(X)`` is flat for every distribution.")
	end

	# Figure 3.4: inversion, animated. Uniform heights u move across to the CDF and down to x
	const INV = ["Exponential(1)" => Exponential(1), "Weibull(1.5, 1)" => Weibull(1.5, 1), "TriangularDist(0, 2, 1)" => TriangularDist(0, 2, 1), "Normal(0, 1)" => Normal(0, 1)]
	function fig_inv(name, u0, n, seed)
		D = pick(INV, name); lo, hi = max(-3.5, minimum(D)), D isa Normal ? 3.5 : min(quantile(D, 0.995), maximum(D))
		u = rand(Xoshiro(3 + 31seed), n); t = range(lo, hi, length = 300); x0 = quantile(D, u0)
		top = fig(w = 640, h = 260, xr = (lo, hi), yr = (0, 1.02), ylabel = "u = F(x)", title = "CDF", l = 56, b = 26)
		line!(top, t, cdf.(D, t); w = 2.6)
		seg!(top, lo, u0, x0, u0; c = COL.purple, w = 1.8, dash = "5 3"); arrow!(top, x0, u0, x0, 0; c = COL.purple, w = 1.8, dash = "5 3")
		text!(top, lo, u0, "u"; dx = 6, dy = -6, c = COL.purple, bold = true, bg = true); text!(top, x0, 0.04, "F⁻¹(u)"; dx = 6, c = COL.purple, bold = true, bg = true)
		bot = fig(w = 640, h = 190, xr = (lo, hi), yr = (0, 1.35maximum(pdf.(D, t))), xlabel = "x", ylabel = "density", title = "samples x", l = 56)
		line!(bot, t, pdf.(D, t); w = 2.2)
		gap = max(0.15, 220 / n); nf = ceil(Int, 50 + gap * n)                    # sample i starts at frame i·gap and takes 50 frames
		code = """
		function draw(k) {
			const cnt = new Array(40).fill(0), bw = (D.hi - D.lo) / 40;
			P[0].clip();
			D.u.forEach((u, i) => {
				const t = (k - i * $(gap)) / 50, x = D.x[i];
				if (t <= 0) return;
				if (t < 0.45) dot(P[0], D.lo + ease(t / 0.45) * (x - D.lo), u, 2.6, "$(COL.purple)");
				else if (t < 1) dot(P[0], x, u * (1 - ease((t - 0.45) / 0.55)), 2.6, "$(COL.orange)");
				else { const j = Math.floor((x - D.lo) / bw); if (j >= 0 && j < 40) cnt[j]++; }
			});
			P[0].ctx.restore(); P[1].clip();
			cnt.forEach((c, j) => c && box(P[1], D.lo + j * bw, 0, D.lo + (j + 1) * bw, c / (D.u.length * bw), "$(COL.orange)", 0.75));
			P[1].ctx.restore();
		}
		"""
		animate([top, bot], (u = u, x = quantile.(D, u), lo = lo, hi = hi), code; nframes = nf, fps = 40,
			caption = md"**Figure 3.4.** Inversion. Each uniform height ``u`` goes across to the CDF and down to ``x = F^{-1}(u)``. Bottom: histogram of ``x`` and the density (blue).")
	end
end;

# ╔═╡ e38a18dc-2ca9-5085-8d51-5ca9e153060a
begin
	x = rand(10_000)
	y = x .^ 2
	quick_hist(y)
end

# ╔═╡ f2ac9463-6698-549d-81ab-4aade62fa9f9
md"""
**Why the values pile up.** In Figure 3.1, 25 values of ``X`` with equal probability (purple lines) go up to the curve ``g`` and across to the ``y``-axis. Where ``g`` is **flat**, the points cluster together: the density of ``Y`` is high. Where ``g`` is **steep**, they spread out: the density is low.

Function $(@bind s3_g Select(TRANS_NAMES[1:5]))
"""

# ╔═╡ 9246ff6c-2244-514a-95db-c5515fb10734
fig_lines(s3_g)

# ╔═╡ 2586d80d-48ae-5ff8-8f67-1fce22946920
md"""
### 3.3 A special function: the CDF itself

**Question.** What happens if the function ``g`` is the CDF ``F_X`` of ``X`` itself?

The CDF is steep exactly where the density is high, since ``F_X' = f_X``. Points that are close together (where ``X`` is likely) are spread apart, points far apart are moved together. With ``g = F_X`` we have ``dy = f_X(x)\,dx``, so

```math
f_Y(y) = f_X(x)\,\frac{dx}{dy} = \frac{f_X(x)}{f_X(x)} = 1 \quad \text{on } (0, 1).
```

!!! theorem "The values of the CDF are uniform"
    If ``X`` is continuous with CDF ``F_X``, then ``F_X(X)`` is uniform on ``(0, 1)``.

Function $(@bind s3_gF Select(TRANS_NAMES[6:7]))
"""

# ╔═╡ 4d50de29-a7d9-55c5-b13c-7cc056f6d0f3
fig_lines(s3_gF; fignum = "3.2")

# ╔═╡ 042615f4-60e1-51fb-b7dc-d7626d0c43de
md"""
Check the theorem with other distributions

Distribution $(@bind s3_d Select(first.(FLAT_DISTS))) ``\qquad`` samples $(@bind s3_n Select([200, 2000, 20_000]; default = 2000)) ``\qquad`` $(@bind s3_seed Seed(0; label = "New samples"))
"""

# ╔═╡ b5661853-aab5-5cd7-bb5d-60a2ad42867b
fig_flat(s3_d, s3_n, s3_seed)

# ╔═╡ e1c6463f-5489-5a80-9fe3-1c8c3cf3e58e
md"""
### 3.4 Inversion: from uniform to any distribution

**Question.** ``F`` turns ``X`` into a uniform number. How can we turn a uniform number into ``X``?

**Idea.** Go the other way. If ``U = F(X)``, then ``X = F^{-1}(U)``. Take a uniform height ``u`` on the vertical axis of the CDF and read off the ``x`` below it. Where the CDF is steep, many points land on a short piece of the ``x``-axis: the values pile up where the density is high.

!!! theorem "Inversion"
    Let ``F`` be a continuous, strictly increasing CDF. If ``U`` is uniform on ``(0, 1)``, then ``X = F^{-1}(U)`` has the CDF ``F``.

    *Reason:* ``P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)``.

Distribution $(@bind s3_inv Select(first.(INV))) ``\qquad`` height ``u`` $(@bind s3_u Slider(0.02:0.02:0.98; default = 0.7, show_value = true)) ``\qquad`` samples $(@bind s3_in Select([50, 300, 2000]; default = 300)) ``\qquad`` $(@bind s3_iseed Seed(0; label = "New heights"))
"""

# ╔═╡ fe273e31-0453-5b4b-8ce3-437b21ff2953
fig_inv(s3_inv, s3_u, s3_in, s3_iseed)

# ╔═╡ 4d17b13c-8ef3-5efd-99e8-1b526bfa5709
# Section 4: normal random numbers
begin
	ih_pdf(k, x) = 0 < x < k ? setprecision(BigFloat, 1024) do
		Float64(sum((-1)^j * binomial(big(k), j) * (big(x) - j)^(k - 1) for j in 0:floor(Int, x)) / factorial(big(k - 1)))
	end : 0.0

	# Figure 4.1: histogram of the sum of k uniforms, and both densities on a log scale
	function fig_clt(k, seed)
		rng = Xoshiro(41 + 7seed); S = [sum(rand(rng, k)) for _ in 1:50_000]
		μ, σ = k / 2, sqrt(k / 12); N = Normal(μ, σ)
		lo, hi = min(0.0, μ - 6σ), max(k, μ + 6σ); e = collect(range(lo, hi, length = 61)); t = range(lo, hi, length = 400)
		a = fig(w = 330, h = 260, xr = (lo, hi), yr = (0, 1.3pdf(N, μ)), xlabel = "sum of $(k) uniform numbers", ylabel = "density", title = "histogram", l = 50)
		b = fig(w = 330, h = 260, xr = (lo, hi), yr = (1e-12, 2pdf(N, μ)), ylog = true, xlabel = "sum of $(k) uniform numbers", ylabel = "density (log scale)", title = "the same densities, log scale", l = 58)
		foreach(g -> rect!(g, μ - 2σ, g.yr[1], μ + 2σ, g.yr[2]; fill = "#888", op = 0.08), (a, b))   # the core: within 2 standard deviations
		hist!(a, e, histdata(S, e)); line!(a, t, pdf.(N, t); w = 2.2)
		line!(b, t, max.(pdf.(N, t), 1e-13); w = 2.2); line!(b, t, [(p = ih_pdf(k, x)) > 1e-13 ? p : NaN for x in t]; c = COL.orange, w = 2.4)
		vline!(b, 0; c = "#777"); vline!(b, k; c = "#777")
		foreach(((x, s),) -> text!(b, x, 3e-12, s; anchor = "middle", size = 11, c = "#666"), ((μ, "core"), (μ - 4σ, "tail"), (μ + 4σ, "tail")))
		legend!(b, [("normal", COL.blue), ("sum of $(k)", COL.orange)]; at = :tl)
		figure(a, b; caption = md"**Figure 4.1.** The sum of ``k`` uniform numbers (orange) and the normal distribution with the same mean and variance (blue). Grey band: the core, within two standard deviations of the mean.")
	end

	# Figure 4.2: Box–Muller with the angle θ = 2πu₂ and the distance R = F_R⁻¹(u₁), F_R(r) = 1 − e^{−r²/2}
	function fig_bm(u1, u2, cloud, seed)
		θ = 2π * u2; R = sqrt(-2log(1 - u1)); x1, x2 = R * cos(θ), R * sin(θ)
		A = fig(w = 190, h = 240, xr = (-1.3, 1.3), yr = (-1.5, 1.5), axes = false, title = "angle θ = 2π u₂", l = 6, r = 6, t = 30, b = 6)
		circle!(A, 0, 0, 68; stroke = "#bbb", sw = 1.2); seg!(A, 0, 0, 1, 0; c = "#999"); arrow!(A, 0, 0, cos(θ), sin(θ); c = COL.red, w = 2.2)
		line!(A, 0.35cos.(range(0, θ, length = 60)), 0.35sin.(range(0, θ, length = 60)); c = COL.purple, w = 2)
		text!(A, 0.45cos(θ / 2), 0.45sin(θ / 2), "θ"; anchor = "middle", c = COL.purple, size = 13, bold = true)
		B = fig(w = 230, h = 240, xr = (0, 4), xlabel = "r", ylabel = "P(R ≤ r)", title = "distance R", l = 46, t = 30)
		line!(B, 0:0.02:4, [1 - exp(-r^2 / 2) for r in 0:0.02:4]; w = 2.4)
		seg!(B, 0, u1, min(R, 4), u1; c = COL.purple, w = 1.6, dash = "5 3"); R < 4 && arrow!(B, R, u1, R, 0; c = COL.red, w = 1.8)
		text!(B, 0, u1, "u₁"; dx = 5, dy = -5, c = COL.purple, bold = true)
		C = fig(w = 240, h = 240, xr = (-4, 4), yr = (-4, 4), xlabel = "x₁", ylabel = "x₂", title = "the point (X₁, X₂)", l = 40, t = 30, xticks = -4:2:4, yticks = -4:2:4)
		if cloud                                                             # 1500 points made the same way
			r = Xoshiro(73 + 11seed); a, b = rand(r, 1500), rand(r, 1500); ρ = sqrt.(-2log.(1 .- a))
			dots!(C, ρ .* cos.(2π .* b), ρ .* sin.(2π .* b); c = COL.orange, r = 1.6, op = 0.3)
		end
		line!(C, R .* cos.(0:0.07:6.3), R .* sin.(0:0.07:6.3); c = "#777", w = 1, dash = "4 3")
		arrow!(C, 0, 0, x1, x2; c = COL.red, w = 2); dots!(C, [x1], [x2]; c = COL.red, r = 6, stroke = "white", sw = 1.5, over = true)
		figure(A, B, C, Markdown.parse("``θ = $(fmt(θ; d = 2))``, ``R = $(fmt(R; d = 2))``, ``(X_1, X_2) = ($(fmt(x1; d = 2)), $(fmt(x2; d = 2)))``");
			caption = md"**Figure 4.2.** Box–Muller with ``R = F_R^{-1}(u_1) = \sqrt{-2\ln(1 - u_1)}``.")
	end
end;

# ╔═╡ d0dc658e-b0af-5da3-a7c3-1f68a933253e
fig_clt(s4_k, s4_seed)

# ╔═╡ d53aa502-ef50-5e15-b6a9-c2704f1fdaac
fig_bm(s4_u1, s4_u2, s4_cloud, s4_bm)

# ╔═╡ 3a5cca40-4805-51bb-bd91-e075a4305f5b
# Section 5: accept–reject and composition–rejection
begin
	# Darts (x from `propose`, height uniform under `env`) kept below f, animated. Returns the animation.
	# Red bands mark where the envelope is below the density.
	function darts(f, env, propose, xr, ymax, n, seed; caption, lowmark = true, bins = 40)
		rng = Xoshiro(seed); x = [propose(rng) for _ in 1:n]; y = rand(rng, n) .* env.(x)
		t = range(xr..., length = 400)
		top = fig(w = 640, h = 270, xr = xr, yr = (0, ymax), ylabel = "height", l = 56, b = 26)
		area!(top, t, f.(t); op = 0.12); line!(top, t, f.(t); w = 2.4); line!(top, t, env.(t); c = COL.ink, w = 1.6, dash = "6 4")
		if lowmark
			low = env.(t) .< f.(t) .- 1e-9
			for i in findall(low)
				rect!(top, t[i], 0, t[min(i + 1, end)], ymax; fill = COL.red, op = 0.12, stroke = COL.red, sw = 0.01)
			end
		end
		Z = sum(f, t) * step(t)
		bot = fig(w = 640, h = 180, xr = xr, yr = (0, 1.35maximum(f, t) / Z), xlabel = "x", ylabel = "density", title = "kept values", l = 56)
		line!(bot, t, f.(t) ./ Z; w = 2.2)
		code = """
		const rate = Math.max(1, D.x.length / 150), bw = (D.x1 - D.x0) / $(bins);
		const kept = k => D.a.slice(0, Math.min(D.x.length, Math.round(k * rate))).filter(Boolean).length;
		function draw(k) {
			const m = Math.min(D.x.length, Math.round(k * rate)), cnt = new Array($(bins)).fill(0);
			P[0].clip();
			for (let i = 0; i < m; i++) {
				dot(P[0], D.x[i], D.y[i], D.x.length > 1500 ? 1.7 : 2.4, D.a[i] ? "$(COL.green)" : "$(COL.red)", 0.75);
				const j = Math.floor((D.x[i] - D.x0) / bw); if (D.a[i] && j >= 0 && j < $(bins)) cnt[j]++;
			}
			P[0].ctx.restore(); P[1].clip();
			cnt.forEach((c, j) => c && box(P[1], D.x0 + j * bw, 0, D.x0 + (j + 1) * bw, c / (kept(k) * bw), "$(COL.orange)", 0.75));
			P[1].ctx.restore();
		}
		info = k => "kept " + kept(k) + " of " + Math.min(D.x.length, Math.round(k * rate));
		"""
		animate([top, bot], (x = x, y = y, a = y .<= f.(x), x0 = xr[1], x1 = xr[2]), code; nframes = 150, caption)
	end

	# Figure 5.1: Beta(2, 5) in the box [0, 1] × [0, 2.6]
	fig_area(n, seed) = vstack(darts(x -> pdf(Beta(2, 5), x), x -> 2.6, rand, (0, 1), 2.8, n, 101 + seed;
			caption = md"**Figure 5.1.** Darts in a box (dashed) around the Beta(2, 5) density. Green: kept; red: rejected. Bottom: histogram of the kept values."),
		md"Expected fraction kept: ``1 / 2.6 = 0.38``.")

	# Figure 5.2: the drawn density with a box (height hb × maximum) or a hill μ·Normal(m, s)
	const ENVS = ["a box", "a hill (normal proposal)"]
	function fig_pad(v, kind, hb, m, s, μ, n, seed)
		f, fmax = pad_density(v)
		env, propose, area = kind == ENVS[1] ? (x -> 0 <= x <= 1 ? hb * fmax : 0.0, rand, hb * fmax) : (x -> μ * pdf(Normal(m, s), x), r -> m + s * randn(r), μ)
		ok = all(x -> env(x) >= f(x) - 1e-9, 0:0.0025:1)
		anim = darts(f, env, propose, (0, 1), 1.15max(fmax, maximum(env, 0:0.005:1)), n, 500 + seed;
			caption = md"**Figure 5.2.** Accept–reject for your density (blue) with the chosen envelope (dashed). Red band: envelope below the density.")
		ok ? vstack(anim, Markdown.parse("Area under the envelope: ``$(fmt(area; d = 2))``.")) : anim
	end

	# Figure 5.3: a sharp peak, a long tail, a proposal with thin tails
	const FAILS = ["A: Normal(0.4, 0.01) with a box", "B: f(x) = 1/(1 + x)² on [0, ∞) with a box on [0, 5]", "C: Laplace(0, 1) with the envelope 1.6 × Normal(0, 1)"]
	function fig_fail(case, seed)
		caption = md"**Figure 5.3.** Darts under three densities (blue). Green: kept; red: rejected. Red band: where the envelope (dashed) lies below the density."
		case == FAILS[1] && return darts(x -> pdf(Normal(0.4, 0.01), x), x -> 40.7, rand, (0, 1), 44.8, 2000, 900 + seed; caption, lowmark = false, bins = 200)
		case == FAILS[2] && return darts(x -> 1 / (1 + x)^2, x -> x <= 5 ? 1.02 : 0.0, r -> 5rand(r), (0, 12), 1.12, 2000, 910 + seed; caption)
		darts(x -> exp(-abs(x)) / 2, x -> 1.6pdf(Normal(), x), randn, (-6, 6), 0.75, 3000, 920 + seed; caption)
	end

	# Figure 5.4: composition–rejection. 
	const COMP_DENS = ["a sharp peak on a flat background", "Beta(2, 5)", "two peaks", "your drawing from Section 5.2"]
	comp_density(name, v) = name == COMP_DENS[1] ? (x -> 0.3 + 0.7pdf(Normal(0.3, 0.015), x)) : name == COMP_DENS[2] ? (x -> pdf(Beta(2, 5), x)) :
		name == COMP_DENS[3] ? (x -> 0.5pdf(Normal(0.25, 0.04), x) + 0.5pdf(Normal(0.7, 0.08), x)) : first(pad_density(v))
	levshade(k, L) = (t = L == 1 ? 1.0 : (k - 1) / (L - 1); "rgb($(round(Int, 220 - 130t)),$(round(Int, 232 - 102t)),$(round(Int, 245 - 49t)))")
	function fig_comp(name, v, L, seed; nb = 20)
		f = comp_density(name, v); edges = range(0, 1, length = nb + 1)
		top = [maximum(f, range(edges[i], edges[i+1], length = 25)) for i in 1:nb]      # highest point of the curve in each block
		fmax = maximum(top); h = [fmax / 2.0^(L - k) for k in 1:L]                       # level heights a, 2a, …, fmax
		lev = [findfirst(>=(m - 1e-12), h) for m in top]
		area = [count(==(k), lev) * h[k] / nb for k in 1:L]
		# sampling: level ∝ area, block of that level uniformly, then a dart in the block
		rng = Xoshiro(77 + seed); xs = Float64[]; shown = []; ndarts = 0
		while length(xs) < 20_000
			k = findfirst(>=(rand(rng) * sum(area)), cumsum(area)); i = rand(rng, findall(==(k), lev))
			x = edges[i] + rand(rng) / nb; y = rand(rng) * h[k]; ndarts += 1
			y <= f(x) && push!(xs, x); length(shown) < 150 && push!(shown, (x, y, y <= f(x)))
		end
		t = range(0, 1, length = 600)
		g = fig(w = 640, h = 270, yr = (0, 1.12fmax), ylabel = "height", l = 56, r = 64, b = 26)
		rect!(g, 0, 0, 1, fmax; fill = "none", op = 1, stroke = COL.red, sw = 1.4)
		foreach(i -> rect!(g, edges[i], 0, edges[i+1], h[lev[i]]; fill = levshade(lev[i], L), op = 0.85, stroke = "white"), 1:nb)
		for k in 1:L
			hline!(g, h[k]; c = "#888", dash = "5 4"); text!(g, 1, h[k], "level $(k)"; dx = 6, size = 11, c = "#444")
		end
		line!(g, t, f.(t); w = 2.4)
		foreach(((x, y, ok),) -> dots!(g, [x], [y]; c = ok ? COL.green : COL.red, r = 2.2, op = 0.85), shown)
		hg = fig(w = 640, h = 180, yr = (0, 1.2fmax), xlabel = "x", ylabel = "density", title = "20 000 samples", l = 56)
		hist!(hg, 0:0.0125:1, histdata(xs, 0:0.0125:1)); line!(hg, t, f.(t); w = 2.2)
		vstack(g, hg, Markdown.parse("Darts per kept sample: one box (red outline) **$(fmt(fmax; d = 3))**, blocks **$(fmt(ndarts / 20_000; d = 3))**.");
			caption = md"**Figure 5.4.** Composition–rejection. Blocks of one level have the same height and colour, from light (level 1) to dark. Dots: the first 150 darts (green: kept, red: rejected). Bottom: histogram of the samples (orange) and the density (blue).")
	end
end;

# ╔═╡ 4fa16e4d-7156-58ff-ab1e-8a805c215e2d
fig_area(s5_n, s5_seed)

# ╔═╡ b02b9ad4-305f-5056-ac28-e204a67e65ca
md"""
Envelope $(@bind s5_env Select(ENVS)) ``\qquad`` box height (× highest point) $(@bind s5_hb Slider(0.5:0.05:2.0; default = 1.05, show_value = true)) ``\qquad`` darts $(@bind s5_pn Select([500, 2000, 6000]; default = 2000)) ``\qquad`` $(@bind s5_pseed Seed(0; label = "New darts"))

Hill: centre ``m`` $(@bind s5_mu Slider(0:0.02:1; default = 0.45, show_value = true)) ``\qquad`` width ``s`` $(@bind s5_sig Slider(0.05:0.01:0.6; default = 0.25, show_value = true)) ``\qquad`` factor ``\mu`` $(@bind s5_M Slider(0.5:0.05:4; default = 2.0, show_value = true))
"""

# ╔═╡ 1c0a2425-8080-5b4c-8fbb-09636c55f6a8
fig_pad(s5_pad, s5_env, s5_hb, s5_mu, s5_sig, s5_M, s5_pn, s5_pseed)

# ╔═╡ a6643369-9aa1-5f04-a0df-15baa3e27339
md"""
### 5.3 Three more densities

**Question.** Does throwing darts work equally well for every density? Try three cases.

Case $(@bind s5_fail Select(FAILS)) ``\qquad`` $(@bind s5_fseed Seed(0; label = "New darts"))
"""

# ╔═╡ 71c14254-c5e4-5a78-95fe-f5563828bbea
fig_fail(s5_fail, s5_fseed)

# ╔═╡ bd5677de-8ecf-5111-8b13-ac4e711f03af
md"""
### 5.4 Composition–rejection

**Question.** In case A most of the box is empty. Can we aim the darts better?

Replace the one big box by small boxes that follow the curve. Cut the ``x``-axis into blocks. The block heights is of ``a, 2a, 4a, \dots`` (levels); each block takes the lowest level above the curve. To sample, choose a level with probability proportional to its area, then a block of that level, then throw a dart into the block. 

Density $(@bind s5_cd Select(COMP_DENS)) ``\qquad`` levels $(@bind s5_L Slider(1:7; default = 5, show_value = true)) ``\qquad`` $(@bind s5_cseed Seed(0; label = "New darts"))
"""

# ╔═╡ b78e1f13-ea6f-5a5a-9969-3cddfeb81f01
fig_comp(s5_cd, s5_pad, s5_L, s5_cseed)

# ╔═╡ e1344783-2356-5df3-aedd-3fdf01ccd35e
# Bonus A: a spiral galaxy from uniform random numbers
begin
	const NUMBERS = ["pseudo-random", "quasi-random (additive recurrence)"]
	function fig_galaxy(arms, wind, numbers, seed; n = 12_000)
		rng = Xoshiro(2024 + seed); k = 0
		u() = numbers == NUMBERS[1] ? rand(rng) : mod(0.5 + (k += 1) * 0.6180339887, 1)
		normal2() = (R = sqrt(-2log(1 - u())); θ = 2π * u(); (R * cos(θ), R * sin(θ)))   # Box–Muller
		stars = map(1:n) do _
			bulge = u() < 0.15
			if bulge                                                    # round cloud in the centre
				x, y = 0.45 .* normal2()
			else                                                        # on a spiral arm, scattered around it
				r = min(-log(1 - u()), 6.0)                               # Exponential(1) by inversion
				θ = 2π * floor(arms * u()) / arms + wind * log(r + 0.2)
				dx, dy = normal2(); s = 0.2 * (0.25 + 0.35r)
				x, y = r * cos(θ) + s * dx, r * sin(θ) + s * dy
			end
			(round(x, digits = 2), round(y, digits = 2), round((1 - u())^(-1 / 2), digits = 2), bulge)   # brightness: Pareto with α = 2
		end
		HTML("""<div class="fg"><canvas width="1280" height="760" style="width:640px;max-width:100%;border-radius:4px;background:#05060d"></canvas>
		<div class="fg-cap">$(tohtml(md"**Figure A.1.** 12 000 stars. Yellow: the bulge; the size and whiteness of a star show its brightness."))</div>
		<script>
		const c = currentScript.parentElement.querySelector("canvas").getContext("2d"), S = $(js([s[1:3] for s in stars])), B = $(js(last.(stars)));
		c.fillStyle = "#05060d"; c.fillRect(0, 0, 1280, 760); c.globalCompositeOperation = "lighter";
		S.forEach(([x, y, L], i) => {
			L = Math.min(L, 60); const a = Math.min(0.9, 0.12 + 0.08 * L);
			c.fillStyle = B[i] ? `rgba(255,214,150,\${a})` : L > 4 ? `rgba(255,255,255,\${a})` : `rgba(160,190,255,\${a})`;
			c.beginPath(); c.arc(640 + 100 * x, 380 - 62 * y, 0.8 + 0.7 * Math.sqrt(L), 0, 6.2832); c.fill();
		});
		</script></div>""")
	end
end;

# ╔═╡ e4174005-1de3-5fd0-badb-f8a66b795e30
md"""
## (Optional) Bonus A. A Galaxy from `rand()`

**Creating a Galaxy:** movies use random numbers to generate galaxy images with millions of stars. Different probability distributions determine the position, arrangement, and brightness of each star, creating a natural galaxy-like appearance.

For each star, uniform random numbers are drawn and transformed:
- **Distance: Exponential distribution, by inversion (Section 3.4).** The CDF ``F(r) = 1 - e^{-r}`` is inverted: ``r = F^{-1}(U) = -\ln(1 - U)``. Most stars lie near the centre, and few lie far.
- **Arm: discrete uniform distribution.** With ``K`` arms, the star joins arm ``\lfloor K U \rfloor``, so each arm has probability ``1/K``. Along its arm the angle winds as a spiral: ``\theta = 2\pi \cdot \text{arm}/K + w \ln r``.
- **Scatter and bulge: Normal distribution, by Box–Muller (Section 4.2).** Two uniform numbers give an angle ``2\pi U_2`` and a distance ``\sqrt{-2\ln U_1}``, and these give two normal numbers. They spread the stars around the arms and form the round bulge in the centre.
- **Bulge or arm: Bernoulli distribution.** One uniform number decides where the star belongs: if ``U < 0.15``, it goes to the central bulge (15 % of the stars), otherwise to a spiral arm.
- **Brightness: Pareto distribution, by inversion (Section 3.4).** The CDF ``F(L) = 1 - L^{-2}`` (``L \ge 1``) is inverted: ``L = (1 - U)^{-1/2}``. Most stars are faint and a few are very bright.
- **Pseudo or quasi (Section 2).** Each star uses several **consecutive** random numbers, so these numbers must be independent. Switch to quasi-random numbers and see what happens.

Arms ``K`` $(@bind sA_arms Select([2, 3, 4, 5]; default = 2)) ``\qquad`` winding ``w`` $(@bind sA_w Slider(0:0.25:4; default = 1.5, show_value = true)) ``\qquad`` numbers $(@bind sA_num Select(NUMBERS)) ``\qquad`` $(@bind sA_seed Seed(0; label = "New galaxy"))
"""

# ╔═╡ 314c2546-9098-516d-ace4-1bf91b8dbbad
fig_galaxy(sA_arms, sA_w, sA_num, sA_seed)

# ╔═╡ f3c97bd5-d7ef-5e96-8c4a-2fbe5957a78b
# Exercises: checks of the programming exercises, with a histogram of the student's samples
begin
	function sample_plot(x, f, xr)
		e = collect(range(xr..., length = 33)); t = range(xr..., length = 200)
		g = fig(w = 640, h = 200, xr = xr, yr = (0, 1.25maximum(f, t)), xlabel = "x", ylabel = "density", l = 50)
		hist!(g, e, histdata(x, e)); line!(g, t, f.(t); w = 2.2)
	end

	function check_tri(tri_inv)
		us = [0.0, 0.19, 0.36, 0.75, 0.99]; r = tryrun(() -> tri_inv.(us))
		ok = r isa AbstractVector && r ≈ 1 .- sqrt.(1 .- us)
		res = check(r, ok, md"All five values are right: `tri_inv(rand())` samples the density ``2(1 - x)``.",
			md"Solve ``u = 1 - (1 - x)^2`` for ``x``."; hint = md"``(1 - x)^2 = 1 - u``, so ``x = 1 - \sqrt{1 - u}``.")
		ok ? vstack(res, sample_plot(tri_inv.(rand(20_000)), x -> 2(1 - x), (0, 1))) : res
	end

	function check_bm(boxmuller)
		r = tryrun(() -> [boxmuller(rand(), rand()) for _ in 1:20_000])
		r isa AbstractVector && all(p -> p isa Tuple && length(p) == 2, r) || return check(r, false, md"", md"`boxmuller(u1, u2)` must return a pair `(x1, x2)` of two numbers.")
		x1, x2 = first.(r), last.(r); v = [x1; x2]; p = boxmuller(0.3, 0.2)
		ok = abs(p[1]^2 + p[2]^2 + 2log(0.3)) < 1e-9 && abs(mean(v)) < 0.03 && abs(var(v) - 1) < 0.05 && abs(cor(x1, x2)) < 0.03
		res = check(r, ok, Markdown.parse("Mean $(fmt(mean(v); d = 2)), variance $(fmt(var(v); d = 3)), correlation of ``X_1`` and ``X_2`` $(fmt(cor(x1, x2); d = 2))."),
			md"The distance must be ``R = \sqrt{-2\ln u_1}`` and the angle ``2\pi u_2``."; hint = md"`R = sqrt(-2log(u1)); θ = 2π * u2; return (R * cos(θ), R * sin(θ))`")
		ok ? vstack(res, sample_plot(v, x -> pdf(Normal(), x), (-4, 4))) : res
	end

	function check_ar(accept_reject)
		f(x) = 6x * (1 - x); r = tryrun(() -> accept_reject(f, 1.5, 20_000))
		ok = r isa AbstractVector && length(r) == 20_000 && all(x -> 0 <= x <= 1, r) && abs(mean(r) - 0.5) < 0.01 && abs(var(r) - 0.05) < 0.004
		res = check(r, ok, Markdown.parse("20 000 samples with mean $(fmt(mean(r); d = 3)) and variance $(fmt(var(r); d = 3)) (exact: 0.5 and 0.05)."),
			md"The function must return exactly `n` kept values in ``[0, 1]``."; hint = md"Repeat until `n` values are kept: `x = rand(); y = fmax * rand(); y <= f(x) && push!(out, x)`.")
		ok ? vstack(res, sample_plot(r, f, (0, 1))) : res
	end
end;

# ╔═╡ cedb851f-d1fd-58b9-bbae-62e1a85f5e95
check_tri(tri_inv)

# ╔═╡ 9ac186d3-bea1-52c2-adcf-cb2c16ef2eaa
check_bm(boxmuller)

# ╔═╡ cb61c9cf-7765-5993-8371-860d42f015f1
check_ar(accept_reject)

# ╔═╡ a6ba0801-431b-5714-834e-789110ee48e4
# Bonus B: sound and music (from the introduction notebook)
begin
	# --- sound in the browser ---
	const B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	function base64(v)
		pad = mod(-length(v), 3); v = [v; zeros(UInt8, pad)]
		s = join(B64[(x >> k) & 63 + 1] for i in 1:3:length(v) for x in (UInt32(v[i]) << 16 | UInt32(v[i+1]) << 8 | v[i+2],) for k in (18, 12, 6, 0))
		s[1:end-pad] * "="^pad
	end
	# an audio player for samples x in [-1, 1] (16-bit WAV)
	function audio(x; fs = 22050, label = "")
		pcm = round.(Int16, 0.9 .* x ./ maximum(abs, x) .* 32767); nb = 2length(pcm)
		io = IOBuffer(); write(io, b"RIFF", UInt32(36 + nb), b"WAVEfmt ", UInt32(16), UInt16(1), UInt16(1), UInt32(fs), UInt32(2fs), UInt16(2), UInt16(16), b"data", UInt32(nb), htol.(pcm))
		wav = take!(io)
		HTML("<div style=\"display:flex;align-items:center;gap:8px\">$(label)<audio controls src=\"data:audio/wav;base64,$(base64(wav))\" style=\"height:32px\"></audio></div>")
	end

	# Karplus–Strong: a burst of P random numbers, repeated with period P = fs/f0 and averaged with the neighbour
	function pluck(f0, dur, rng; fs = 22050)
		P = round(Int, fs / f0); y = zeros(round(Int, dur * fs)); y[1:P] = 2rand(rng, P) .- 1
		for i in P+2:length(y); y[i] = 0.998 * (y[i-P] + y[i-P-1]) / 2; end
		y
	end
	spectrum(y, fs, fr) = (w = 0.5 .- 0.5cos.(2π .* (0:length(y)-1) ./ (length(y) - 1)); [abs(sum(w .* y .* cis.(-2π * f .* (0:length(y)-1) ./ fs))) for f in fr])

	# Play notes (start s, frequency Hz, duration s) as plucked strings, synthesised in the browser.
	# Elements with class pn-hl and data-g = group, data-a/data-b = times are shown only while the music is between a and b.
	function play_notes(notes; label = "Play", group = "")
		ev = [round.((t, f, d), digits = 3) for (t, f, d) in notes]
		HTML("""<span><button class="pn-b">▶ $(label)</button><script>
		const btn = currentScript.parentElement.querySelector("button"), ev = $(js(ev));
		const hl = Array.from((btn.closest("pluto-output") || document).querySelectorAll(".pn-hl")).filter(e => e.dataset.g === $(js(group)));
		const mark = t => hl.forEach(e => e.style.display = t >= +e.dataset.a && t < +e.dataset.b ? "" : "none");
		let ctx, src = null, raf;
		function render(sr) {
			const N = Math.ceil((Math.max(...ev.map(e => e[0] + e[2])) + 1.6) * sr), y = new Float32Array(N);
			for (const [t, f, d] of ev) {
				const a = Math.floor(t * sr), P = Math.round(sr / f), L = Math.min(N - a, Math.floor((d + 1.2) * sr)), b = new Float32Array(L);
				for (let i = 0; i < P; i++) b[i] = (2 * Math.random() - 1) * 0.4 + (i ? 0.6 * b[i - 1] : 0);
				for (let i = P + 1; i < L; i++) b[i] = (i < d * sr ? 0.998 : 0.985) * 0.5 * (b[i - P] + b[i - P - 1]);
				for (let i = 0; i < L; i++) y[a + i] += 0.6 * b[i];
			}
			const m = y.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
			return y.map(v => 0.85 * v / m);
		}
		function stop() { if (src) src.stop(); src = null; cancelAnimationFrame(raf); btn.textContent = "▶ $(label)"; mark(-1); }
		btn.onclick = () => {
			if (src) return stop();
			ctx = ctx || new AudioContext();
			const y = render(ctx.sampleRate), buf = ctx.createBuffer(1, y.length, ctx.sampleRate);
			buf.copyToChannel(y, 0); src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination); src.onended = stop;
			const t0 = ctx.currentTime + 0.05; src.start(t0); btn.textContent = "■ stop";
			const tick = () => { mark(ctx.currentTime - t0); raf = requestAnimationFrame(tick); }; tick();
		};
		mark(-1); invalidation.then(stop);
		</script></span>""")
	end

	# Figure B.1: the burst, the wave after 0.5 s, and the spectra of noise and string
	function fig_pluck(note, seed; fs = 22050)
		f0 = Dict("A3 (220 Hz)" => 220.0, "E4 (330 Hz)" => 329.63, "A4 (440 Hz)" => 440.0)[note]
		y = pluck(f0, 1.6, Xoshiro(91 + seed)); P = round(Int, fs / f0)
		noise = (2rand(Xoshiro(92 + seed), length(y)) .- 1) .* exp.(-(0:length(y)-1) ./ 0.35fs)
		ms = (0:3P) ./ fs .* 1000; t0 = fs ÷ 2
		F = fig(w = 330, h = 210, xr = extrema(ms), yr = (-1.1, 1.1), xlabel = "time (ms)", ylabel = "amplitude", title = "start: the repeated burst")
		G = fig(w = 330, h = 210, xr = extrema(ms), yr = (-1.1, 1.1), xlabel = "time (ms) after 0.5 s", title = "after 0.5 s (rescaled)")
		line!(F, ms, y[1:3P+1]; c = COL.red, w = 1.1); line!(G, ms, y[t0:t0+3P] ./ maximum(abs, y[t0:t0+3P]); w = 1.6)
		foreach(g -> foreach(j -> vline!(g, 1000j * P / fs; c = "#999", dash = "3 3", w = 1), 1:2), (F, G))
		fr = range(20, 10.5 * f0, length = 600); w = fs ÷ 20 .+ (0:4095)
		H = fig(w = 680, h = 250, xr = (0, 10.5 * f0), yr = (0, 1.42), yticks = [0, 0.5, 1], xlabel = "frequency (Hz)", ylabel = "relative magnitude",
			title = "spectrum of 0.19 s of sound", xticks = f0 .* (1:10), xticklabels = ["f₀ = $(round(Int, f0))"; ["$(k)f₀" for k in 2:10]])
		S0, S1 = spectrum(noise[w], fs, fr), spectrum(y[w], fs, fr)
		line!(H, fr, S0 ./ maximum(S0); c = COL.grey, w = 1); line!(H, fr, S1 ./ maximum(S1); w = 1.6)
		legend!(H, [("random numbers, not repeated", COL.grey), ("Karplus–Strong string", COL.blue)])
		figure(F, G, H, vstack(audio(noise; label = "random numbers, not repeated:"), audio(y; label = "Karplus–Strong string at $(round(Int, f0)) Hz:"));
			caption = Markdown.parse("**Figure B.1.** The Karplus–Strong algorithm. Top left: a burst of $(P) random numbers, repeated with period $(fmt(1000P / fs; d = 3)) ms (dashed). Top right: after repeated averaging, a smooth periodic wave. Bottom: the spectrum of the string consists of peaks at the multiples of ``f_0``."))
	end

	# --- Mozart's dice game (1792): row = bar 1…16, column = dice sum 2…12; entries are bar numbers 1…176 ---
	const MOZART_TABLE = [96 32 69 40 148 104 152 119 98 3 54;
		22 6 95 17 74 157 60 84 142 87 130;
		141 128 158 113 163 27 171 114 42 165 10;
		41 63 13 85 45 167 53 50 156 61 103;
		105 146 153 161 80 154 99 140 75 135 28;
		122 46 55 2 97 68 133 86 129 47 37;
		11 134 110 159 36 118 21 169 62 147 106;
		30 81 24 100 107 91 127 94 123 33 5;
		70 117 66 90 25 138 16 120 65 102 35;
		121 39 139 176 143 71 155 88 77 4 20;
		26 126 15 7 64 150 57 48 19 31 108;
		9 56 132 34 125 29 175 166 82 164 92;
		112 174 73 67 76 101 43 51 137 144 12;
		49 18 58 160 136 162 168 115 38 59 124;
		109 116 145 52 1 23 89 72 149 173 44;
		14 83 79 170 93 151 172 111 8 78 131]   # row = bar 1…16, column = dice sum 2…12
	# each bar: "midi pitch, onset, duration" in sixteenths, separated by ";" (right hand and left hand)
	const MOZART_RH = [
		"77,0,2;74,2,2;79,4,2", "69,0,2;66,2,1;67,3,1;71,4,1;79,5,1", "79,0,2;72,2,2;76,4,2", "79,0,2;74,2,4", "67,0,4;71,0,4;74,0,4;79,0,4", "67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;76,2,1;79,3,1;84,4,1;79,5,1", "72,0,4", "72,0,2;76,0,2;71,2,2;74,2,2", "71,0,1;69,1,1;71,2,1;72,3,1;74,4,1;71,5,1", "76,0,1;72,1,1;71,2,1;69,3,1;67,4,1;66,5,1", "64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"72,0,2;67,2,2;64,4,2", "72,0,4", "76,0,2;79,2,1;76,3,1;72,4,2", "81,0,2;78,2,2;74,4,2", "72,0,1;67,1,1;72,2,1;76,3,1;67,4,1;72,5,1", "67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;76,2,2;79,4,2", "79,0,2;83,2,1;86,3,1;74,4,2", "72,0,1;76,1,1;79,2,1;74,3,1;69,4,1;78,5,1", "76,0,2;72,2,2;67,4,2", "77,0,1;76,1,1;74,2,1;76,3,1;77,4,1;79,5,1", "67,0,4;71,0,4;74,0,4;79,0,4",
		"62,0,1;66,1,1;69,2,1;74,3,1;78,4,1;81,5,1", "72,0,2;76,0,2;72,2,2;76,2,2;72,4,2;76,4,2", "77,0,1;76,1,1;77,2,1;74,3,1;72,4,1;71,5,1", "78,0,1;74,1,1;69,2,1;81,3,1;78,4,1;74,5,1", "71,0,1;74,1,1;79,2,1;74,3,1;71,4,2", "67,0,4;71,0,4;74,0,4;79,0,4",
		"76,0,1;72,1,1;67,2,2;76,4,2", "67,0,2;72,2,2;76,4,2", "67,0,4;71,0,4;74,0,4;79,0,4", "76,0,1;72,1,1;74,2,1;71,3,1;67,4,2", "69,0,2;74,2,2;78,4,2", "69,0,1;76,1,1;74,2,1;79,3,1;78,4,1;81,5,1",
		"79,0,1;83,1,1;79,2,1;74,3,1;71,4,2", "72,0,2;67,2,2;76,4,2", "79,0,2;67,2,2;67,4,2", "72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1", "72,0,1;71,1,1;72,2,1;76,3,1;67,4,2", "71,0,1;72,1,1;74,2,1;71,3,1;69,4,1;67,5,1",
		"79,0,2;77,2,1;76,3,1;74,4,1;72,5,1", "69,0,2;77,2,1;74,3,1;69,4,1;71,5,1", "72,0,1;71,1,1;72,2,1;67,3,1;64,4,1;60,5,1", "79,0,2;83,2,1;79,3,1;74,4,1;71,5,1", "79,0,2;79,2,1;74,3,1;83,4,2", "76,0,2;72,2,1;76,3,1;79,4,1;84,5,1",
		"76,0,2;72,2,2;67,4,2", "72,0,2;76,2,1;72,3,1;67,4,2", "72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1", "74,0,1;73,1,1;74,2,1;77,3,1;67,4,1;71,5,1", "72,0,2;76,0,2;72,2,1;76,2,1;74,3,1;77,3,1;76,4,2;79,4,2", "64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"79,0,2;83,2,2;74,4,2", "74,0,1;71,1,1;67,2,2", "76,0,2;72,2,2;67,4,2", "79,0,2;76,2,2;72,4,2", "79,0,2;72,2,2;76,4,2", "79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"72,0,2;76,2,1;72,3,1;79,4,2", "76,0,1;72,1,1;71,2,1;67,3,1;69,4,1;66,5,1", "76,0,1;72,1,1;71,2,1;72,3,1;67,4,2", "76,0,1;79,1,1;84,2,1;79,3,1;76,4,1;72,5,1", "74,0,1;69,1,1;74,2,2;78,4,2", "78,0,2;81,2,2;78,4,2",
		"72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1", "79,0,2;83,2,1;79,3,1;74,4,1;79,5,1", "79,0,2;76,2,2;72,4,2", "78,0,2;81,2,1;78,3,1;74,4,1;78,5,1", "79,0,1;83,1,1;86,2,1;83,3,1;79,4,2", "77,0,1;76,1,1;74,2,1;72,3,1;71,4,1;74,5,1",
		"79,0,2;76,2,2;72,4,2", "84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1", "74,0,2;78,0,2;74,2,2;78,2,2;74,4,2;78,4,2", "84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1", "79,0,1;83,1,1;79,2,2;74,4,2", "72,0,2;60,2,2",
		"72,0,4", "74,0,2;69,2,2;78,4,2", "67,0,4;71,0,4;74,0,4;79,0,4", "74,0,1;71,1,1;67,2,2;79,4,2", "72,0,4", "72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1",
		"72,0,2;76,2,2;67,4,2", "74,0,2;74,2,1;79,3,1;83,4,2", "79,0,2;72,2,2;76,4,2", "79,0,1;74,1,1;79,2,1;83,3,1;79,4,1;74,5,1", "77,0,1;76,1,1;74,2,2;79,4,2", "78,0,1;81,1,1;86,2,1;81,3,1;78,4,1;81,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4", "71,0,2;74,0,2;79,2,1;83,3,1;74,4,2", "72,0,4", "67,0,4;71,0,4;74,0,4;79,0,4", "79,0,2;76,2,2;72,4,2", "76,0,2;72,2,2;67,4,2",
		"79,0,1;78,1,1;79,2,1;74,3,1;71,4,1;67,5,1", "72,0,2;67,2,2;76,4,2", "78,0,2;81,2,2;74,4,2", "67,0,4;71,0,4;74,0,4;79,0,4", "76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1", "78,0,1;74,1,1;69,2,2;78,4,2",
		"72,0,1;76,1,1;72,2,1;67,3,1;64,4,2", "76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1", "78,0,2;81,2,1;78,3,1;74,4,1;78,5,1", "69,0,2;74,2,1;72,3,1;71,4,1;69,5,1", "67,0,4;71,0,4;74,0,4;79,0,4", "76,0,2;79,2,2;84,4,2",
		"74,0,1;77,1,1;74,2,1;77,3,1;71,4,1;74,5,1", "71,0,1;74,0,1;69,1,1;72,1,1;69,2,1;72,2,1;67,3,1;71,3,1;67,4,1;71,4,1;66,5,1;69,5,1", "72,0,4", "76,0,2;72,2,2;67,4,2", "77,0,2;74,2,2;71,4,2", "71,0,2;74,0,2;71,2,2;74,2,2;71,4,2;74,4,2",
		"72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1", "74,0,1;77,1,1;81,2,1;77,3,1;74,4,1;71,5,1", "74,0,1;69,1,1;74,2,1;78,3,1;81,4,1;78,5,1", "76,0,1;81,1,1;79,2,1;83,3,1;78,4,1;81,5,1", "76,0,1;72,1,1;79,2,1;76,3,1;84,4,1;79,5,1", "86,0,2;81,2,1;78,3,1;74,4,1;69,5,1",
		"79,0,2;83,2,1;79,3,1;74,4,2", "79,0,1;78,1,1;79,2,1;83,3,1;74,4,2", "67,0,4;71,0,4;74,0,4;79,0,4", "64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2", "79,0,1;76,1,1;74,2,1;71,3,1;67,4,2", "72,0,1;67,1,1;72,2,1;76,3,1;79,4,1;72,5,1;76,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4", "71,0,2;74,2,2;79,4,2", "81,0,1;79,1,1;78,2,1;79,3,1;74,4,2", "64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2", "72,0,4", "72,0,2;76,0,2;71,2,1;74,2,1;67,3,1;71,3,1;67,4,2",
		"74,0,2;79,2,1;74,3,1;71,4,1;74,5,1", "69,0,1;76,1,1;71,2,1;74,2,1;69,3,1;72,3,1;67,4,1;71,4,1;66,5,1;69,5,1", "78,0,2;78,2,1;74,3,1;81,4,2", "84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1", "72,0,2;67,2,2;76,4,2", "69,0,2;74,0,2;78,0,2;78,2,4",
		"79,0,1;83,1,1;79,2,1;83,3,1;74,4,2", "69,0,2;69,2,1;74,3,1;78,4,2", "74,0,1;76,1,1;77,2,1;74,3,1;72,4,1;71,5,1", "72,0,2;67,2,2;76,4,2", "79,0,2;74,2,1;71,3,1;67,4,2", "79,0,2;72,2,2;76,4,2",
		"74,0,1;77,1,1;69,2,1;74,3,1;71,4,1;74,5,1", "66,0,2;74,0,2;74,2,2;78,2,2;78,4,2;81,4,2", "76,0,1;84,1,1;83,2,1;79,3,1;81,4,1;78,5,1", "84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1", "77,0,1;74,1,1;69,2,2;71,4,2", "67,0,2;72,0,2;76,0,2;76,2,4",
		"72,0,4", "79,0,2;77,2,1;76,3,1;74,4,1;72,5,1", "74,0,1;69,1,1;78,2,1;74,3,1;81,4,1;78,5,1", "74,0,1;73,1,1;74,2,1;78,3,1;81,4,1;78,5,1", "79,0,1;83,1,1;79,2,1;74,3,1;71,4,1;67,5,1", "72,0,1;67,1,1;76,2,1;72,3,1;79,4,2",
		"76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1", "71,0,2;74,2,1;71,3,1;69,4,1;67,5,1", "76,0,1;79,1,1;74,2,1;72,3,1;71,4,1;69,5,1", "72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1", "66,0,2;74,0,2;66,2,2;74,2,2;66,4,2;74,4,2", "76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"79,0,1;78,1,1;79,2,1;74,3,1;71,4,1;67,5,1", "74,0,2;67,2,4", "74,0,2;71,2,2;67,4,2", "74,0,1;83,1,1;79,2,1;74,3,1;71,4,2", "72,0,2;72,2,1;74,3,1;76,4,2", "79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"76,0,1;79,1,1;74,2,1;79,3,1;69,4,1;78,5,1", "72,0,4", "71,0,1;72,1,1;74,2,1;76,3,1;77,4,1;74,5,1", "72,0,4", "77,0,1;81,1,1;69,2,2;71,4,1;74,5,1", "67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;71,2,1;74,3,1;79,4,2", "81,0,1;79,1,1;83,2,1;79,3,1;74,4,1;79,5,1"]
	const MOZART_LH = [
		"53,0,2;50,2,2;55,4,2", "47,0,4;55,0,4", "48,0,4;52,0,4", "43,0,1;47,1,1;55,2,2;47,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "48,0,4;52,0,4",
		"48,0,4;55,0,4", "48,0,2;43,2,2;36,4,2", "55,0,4;43,4,2", "55,0,4", "48,0,2;50,2,2;38,4,2", "48,0,2;48,2,2;48,4,2",
		"52,0,4;55,0,4;52,4,2;55,4,2", "48,0,2;43,2,2;36,4,2", "48,0,4;55,0,4;48,4,2;52,4,2", "50,0,4;54,0,4;48,4,2;54,4,2", "52,0,4;55,0,4", "48,0,4;52,0,4;48,4,2;55,4,2",
		"48,0,4;55,0,4;48,4,2;52,4,2", "47,0,4", "48,0,2;50,2,2;38,4,2", "48,0,4", "53,0,1;52,1,1;50,2,1;52,3,1;53,4,1;55,5,1", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"50,0,4;48,4,2", "48,0,1;52,1,1;55,2,1;52,3,1;60,4,1;48,5,1", "55,0,4;59,0,4", "48,0,4;57,0,4", "55,0,4;43,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;55,0,4;48,4,2;55,4,2", "48,0,4;52,0,4", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "55,0,4", "50,0,4;54,0,4;48,4,2;57,4,2", "48,0,2;50,2,2;38,4,2",
		"47,0,4;50,0,4", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "47,0,1;50,1,1;55,2,1;50,3,1;47,4,1;43,5,1", "48,0,4;52,0,4", "48,0,4;52,0,4", "43,0,4",
		"48,0,4;52,0,4", "53,0,4;55,4,2", "52,0,4;55,0,4", "47,0,4;50,0,4", "47,0,4;50,0,4", "48,0,4;55,0,4;48,4,2;52,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "52,0,4;55,0,4", "48,0,4;52,0,4", "53,0,4;55,4,2", "48,0,4", "48,0,2;48,2,2;48,4,2",
		"47,0,4;50,0,4", "43,0,4;55,0,4;55,4,2", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "48,0,4;52,0,4",
		"52,0,4;55,0,4", "48,0,2;50,2,2;38,4,2", "48,0,4", "48,0,4;55,0,4;48,4,2;55,4,2", "50,0,4;54,0,4", "50,0,2;57,0,2;50,2,2;54,2,2;48,4,2;50,4,2",
		"48,0,4;52,0,4;52,4,2;55,4,2", "47,0,4", "48,0,4;52,0,4", "50,0,4;48,4,2", "47,0,4;50,0,4;47,4,2;50,4,2", "53,0,4;55,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "48,0,4;52,0,4", "48,0,2;48,2,2;48,4,2", "48,0,4;52,0,4;48,4,2;55,4,2", "47,0,4;50,0,4;47,4,2;55,4,2", "48,0,4;36,4,2",
		"48,0,2;43,2,2;36,4,2", "48,0,4", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "47,0,4;55,0,4;47,4,2;50,4,2", "48,0,2;43,2,2;36,4,2", "48,0,4;52,0,4",
		"52,0,4;55,0,4", "47,0,4;55,0,4", "48,0,4;52,0,4;48,4,2;55,4,2", "47,0,4;50,0,4;47,4,2;50,4,2", "53,0,1;52,1,1;50,2,2;55,4,2", "48,0,4;57,0,4;48,4,2;57,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "43,0,4;55,0,4;55,4,2", "48,0,2;43,2,2;36,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "48,0,4;52,0,4", "48,0,4",
		"47,0,4;50,0,4;47,4,2;55,4,2", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "48,0,4;57,0,4;48,4,2;57,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "48,0,4;55,0,4;48,4,2;52,4,2", "48,0,4;57,0,4;48,4,2;57,4,2",
		"52,0,4;55,0,4", "48,0,4", "48,0,4", "48,0,2;50,2,2;38,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "48,0,4;55,0,4;48,4,2;52,4,2",
		"53,0,4;57,0,4;55,4,2;62,4,2", "48,0,2;50,2,2;38,4,2", "48,0,2;43,2,2;36,4,2", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "55,0,4;59,0,4", "55,0,2;55,2,2;55,4,2",
		"48,0,4;52,0,4", "53,0,4;55,4,2", "50,0,4;54,0,4", "48,0,2;50,2,2;38,4,2", "48,0,4;52,0,4", "50,0,4;54,0,4;48,4,2;54,4,2",
		"47,0,4;55,0,4", "47,0,2;50,0,2;47,2,2;50,2,2;47,4,2;55,4,2", "43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "48,0,2;48,2,2;48,4,2", "55,0,2;43,2,2", "52,0,4;52,4,1;48,5,1",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1", "43,0,4", "47,0,2;50,0,2;47,2,2;50,2,2;47,4,2;55,4,2", "48,0,2;48,2,2;48,4,2", "48,0,2;43,2,2;36,4,2", "55,0,2;43,2,2",
		"47,0,4;55,0,4", "48,0,2;50,2,2;38,4,2", "48,0,2;50,0,2;48,2,2;50,2,2;48,4,2;50,4,2", "48,0,4;52,0,4", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1", "38,0,1;50,1,1;49,2,1;50,3,1;48,4,1;50,5,1",
		"47,0,4", "48,0,2;54,0,2;48,2,2;54,2,2;48,4,2;57,4,2", "47,0,4;55,0,4;43,4,2", "48,0,4;52,0,4", "47,0,4;50,0,4;47,4,2;50,4,2", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"53,0,4;55,4,2", "48,0,2;48,2,2;48,4,2", "48,0,2;50,2,2;38,4,2", "48,0,4;52,0,4", "53,0,4;55,4,2", "48,0,1;47,1,1;48,2,1;50,3,1;52,4,1;54,5,1",
		"48,0,2;43,2,2;36,4,2", "48,0,4;52,0,4", "48,0,4", "48,0,4", "47,0,4;50,0,4", "52,0,4;55,0,4",
		"48,0,4", "43,0,4", "48,0,2;50,2,2;38,4,2", "48,0,4;52,0,4;48,4,2;52,4,2", "48,0,2;48,2,2;48,4,2", "48,0,4;55,0,4;48,4,2;52,4,2",
		"47,0,4;50,0,4", "55,0,1;54,1,1;55,2,1;50,3,1;47,4,1;43,5,1", "47,0,4", "55,0,4;59,0,4", "48,0,4;52,0,4", "48,0,4;52,0,4;52,4,2;55,4,2",
		"48,0,2;50,2,2;38,4,2", "48,0,2;43,2,2;36,4,2", "43,0,4;55,0,4;47,4,2;55,4,2", "48,0,2;43,2,2;36,4,2", "53,0,4;55,4,2", "48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"55,0,2;43,2,2", "47,0,4;50,0,4;47,4,2;50,4,2"]
	notes_of(bar, t0) = [(t0 + o / 6, 440 * 2^((p - 69) / 12), d / 6) for (p, o, d) in (parse.(Int, split(e, ",")) for e in split(bar, ";") if !isempty(e))]

	# Figure B.2: roll 16 pairs of dice, look up the bars, play the waltz (one bar = 1 s)
	function fig_waltz(seed)
		rng = Xoshiro(110 + seed); sums = rand(rng, 1:6, 16) .+ rand(rng, 1:6, 16)
		bars = [MOZART_TABLE[i, sums[i] - 1] for i in 1:16]
		notes = vcat([[notes_of(MOZART_RH[b], i - 1); notes_of(MOZART_LH[b], i - 1)] for (i, b) in enumerate(bars)]...)
		f = fig(w = 680, h = 330, xr = (0.5, 16.5), yr = (1, 13.5), axes = false, l = 44, r = 6, t = 34, b = 8, title = "table of bars: row = sum of the dice, column = bar of the waltz")
		text!(f, 0.5, 13, "bar"; anchor = "end", size = 11, c = "#555", dx = -4)
		for i in 1:16
			text!(f, i, 13, "$(i)"; anchor = "middle", size = 11, bold = true, c = "#555")
			raw!(f, "<rect class='pn-hl' data-g='waltz' data-a='$(i - 1)' data-b='$(i)' x='$(_n(X(f, i - 0.5)))' y='$(_n(Y(f, 12.6)))' width='$(_n(X(f, 1) - X(f, 0)))' height='$(_n(Y(f, 1.4) - Y(f, 12.6)))' fill='$(COL.red)' opacity='0.18'/>"; over = true)
			for s in 2:12
				i == 1 && text!(f, 0.5, 14 - s, "$(s)"; anchor = "end", size = 11, bold = true, c = "#555", dx = -4)
				sums[i] == s && rect!(f, i - 0.47, 14 - s - 0.45, i + 0.47, 14 - s + 0.45; op = 0.85)
				text!(f, i, 14 - s, string(MOZART_TABLE[i, s-1]); anchor = "middle", size = 10.5, c = sums[i] == s ? "white" : "#666")
			end
		end
		g = fig(w = 330, h = 230, xr = (1.4, 12.6), yr = (0, 0.27), xlabel = "sum of two dice", ylabel = "probability", title = "how often each row is used", xticks = 2:12)
		bars!(g, 2:12, [(6 - abs(s - 7)) / 36 for s in 2:12]; width = 0.7, op = 0.6); hline!(g, 1 / 11; c = COL.red, dash = "5 3", w = 1.5)
		legend!(g, [("two dice", COL.blue, :bar), ("uniform 1/11", COL.red, :dash)]; size = 10.5)
		vstack(f, figure(g, play_notes(notes; label = "Play this waltz", group = "waltz"));
			caption = md"**Figure B.2.** Mozart's dice game. Top: the table of the 1792 edition; each number refers to one of 176 printed bars.")
	end

	# notes in semitones from Sa; a Markov chain of small steps that follows each rāga's rules
	const RJ = [1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8]   # just intonation
	const RPC = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma′", "Pa", "dha", "Dha", "ni", "Ni"]
	const RNAMES = ["Bhupali", "Deshkar", "Shuddh Kalyan", "Yaman"]
	const RCOL = [COL.ochre, COL.red, COL.green, COL.blue]
	const RDEF = Dict(
		"Bhupali" => (notes = [0, 2, 4, 7, 9], up = [0, 2, 4, 7, 9], down = [0, 2, 4, 7, 9], vadi = 4, samvadi = 9,
			w = Dict(0 => 1.4, 2 => 1.1, 4 => 2.4, 7 => 1.0, 9 => 1.5),
			pakad = [[4, 2, 0, -3, 0, 2, 4], [7, 4, 9, 7, 4, 2, 0], [0, 2, 4, 7, 4, 2, 0]]),
		"Deshkar" => (notes = [0, 2, 4, 7, 9], up = [0, 2, 4, 7, 9], down = [0, 4, 7, 9], vadi = 9, samvadi = 4,
			w = Dict(0 => 1.0, 2 => 0.5, 4 => 1.5, 7 => 2.0, 9 => 2.6),
			pakad = [[7, 9, 12, 9, 7], [9, 4, 7, 4, 2, 0], [12, 9, 7, 9, 4, 7]]),
		"Shuddh Kalyan" => (notes = [0, 2, 4, 6, 7, 9, 11], up = [0, 2, 4, 7, 9], down = [0, 2, 4, 6, 7, 9, 11], vadi = 4, samvadi = 9,
			w = Dict(0 => 1.4, 2 => 1.2, 4 => 2.2, 6 => 0.6, 7 => 1.0, 9 => 1.5, 11 => 0.7),
			pakad = [[4, 2, 4, 2, 0], [-1, -3, -5, 4, 2, 0], [4, 2, 4, 7, 2, 0]]),
		"Yaman" => (notes = [0, 2, 4, 6, 7, 9, 11], up = [0, 2, 4, 6, 9, 11], down = [0, 2, 4, 6, 7, 9, 11], vadi = 4, samvadi = 11,
			w = Dict(0 => 1.3, 2 => 1.2, 4 => 2.2, 6 => 1.3, 7 => 0.8, 9 => 1.1, 11 => 1.7),
			pakad = [[-1, 2, 4, 2, 0], [7, 6, 4, 2, 0], [6, 9, 11, 12]]))
	rnotes(rg) = [h for oct in -1:1 for h in RDEF[rg].notes .+ 12oct if -5 <= h <= 17]
	rfreq(h) = 220 * 2.0^fld(h, 12) * RJ[mod(h, 12) + 1]
	revents(mel; beat = 0.42) = (t = cumsum([0; last.(mel)]) .* beat .+ 0.4; [(t[i], rfreq(h), d * beat) for (i, (h, d)) in enumerate(mel)])

	function rmelody(rg, n, rng)
		R = RDEF[rg]; N = rnotes(rg); K = length(N)
		i = findfirst(==(0), N); dir = 1
		out = Tuple{Int, Float64}[]                       # (semitone, duration in beats)
		while length(out) < n
			if rand(rng) < 0.22
				for h in rand(rng, R.pakad)
					push!(out, (h, 0.5))
				end
				out[end] = (out[end][1], 1.5); i = something(findfirst(==(out[end][1]), N), i); dir = -1
				continue
			end
			w = zeros(K)
			for j in 1:K
				Δ = j - i; pc = mod(N[j], 12)
				(Δ > 0 && !(pc in R.up)) && continue
				(Δ < 0 && !(pc in R.down)) && continue
				Δ == 0 && continue
				w[j] = exp(-(abs(Δ) - 1) / 0.7) * (sign(Δ) == dir ? 2.0 : 1.0) * R.w[pc] * exp(-((N[j] - 5) / 8)^2 / 2)
			end
			j = something(findfirst(>(rand(rng) * sum(w)), cumsum(w)), K)
			dir = sign(j - i); i = j
			pc = mod(N[i], 12); nyas = pc in (0, R.vadi, R.samvadi)
			push!(out, (N[i], nyas && rand(rng) < 0.35 ? 1.5 : rand(rng) < 0.5 ? 0.5 : 1.0))
		end
		out = out[1:n]; out[end] = (0, 2.5)
		out
	end

	# how often each rāga plays each of the 12 notes, from a long simulated melody
	const RPROB = [(p = fill(1e-3, 12); foreach(((h, _),) -> p[mod(h, 12) + 1] += 1, rmelody(rg, 200_000, Xoshiro(2007))); p ./ sum(p)) for rg in RNAMES]

	# Figure B.3: the note probabilities of the four rāgas, with a melody of each
	function fig_ragas(seed)
		sev = [0, 2, 4, 6, 7, 9, 11]
		figure(map(1:4) do r
			f = fig(w = 330, h = 190, xr = (0.4, 7.6), yr = (0, 0.34), xticks = 1:7, xticklabels = RPC[sev .+ 1], ylabel = "probability", title = RNAMES[r], l = 52, r = 6)
			bars!(f, 1:7, RPROB[r][sev .+ 1]; width = 0.72, c = RCOL[r], op = 0.8)
			vstack(f, play_notes(revents(rmelody(RNAMES[r], 22, Xoshiro(3000 + 97r + 1009seed))); label = "Listen to $(RNAMES[r])"))
		end...; caption = md"**Figure B.3.** The four rāgas: how often each note occurs.")
	end

	# Figure B.4: a melody of a secret rāga; during playback the bars show P(rāga | notes so far) by Bayes' theorem
	function fig_mystery(seed)
		rng = Xoshiro(4000 + 1009seed); r = rand(rng, 1:4); mel = rmelody(RNAMES[r], 28, rng)
		post = accumulate((q, (h, _)) -> (q = q .* [RPROB[j][mod(h, 12) + 1] for j in 1:4]; q ./ sum(q)), mel; init = fill(0.25, 4))
		ev = revents(mel)
		B = fig(w = 680, h = 190, yr = (4.6, 0.4), axes = false, l = 130, r = 70, t = 26, b = 10, title = "the computer's belief after each note (press play)")
		for j in 1:4
			text!(B, 0, j, RNAMES[j]; anchor = "end", dx = -8, size = 13, bold = true, c = RCOL[j]); rect!(B, 0, j - 0.32, 1, j + 0.32; fill = "#f1f1ee", op = 1)
			for (k, q) in enumerate([[fill(0.25, 4)]; post])                 # one bar per moment, shown while it is current
				raw!(B, "<g class='pn-hl' data-g='myst' data-a='$(k == 1 ? -1 : ev[k-1][1])' data-b='$(k <= length(ev) ? ev[k][1] : 99)'>" *
					"<rect x='$(_n(X(B, 0)))' y='$(_n(Y(B, j - 0.32)))' width='$(_n(X(B, q[j]) - X(B, 0)))' height='$(_n(Y(B, j + 0.32) - Y(B, j - 0.32)))' fill='$(RCOL[j])' opacity='0.85'/>" *
					"<text x='$(_n(X(B, 1) + 8))' y='$(_n(Y(B, j) + 4))' font-size='13'>$(round(Int, 100q[j])) %</text></g>"; over = true)
			end
		end
		n90 = something(findfirst(q -> q[r] > 0.9, post), 0)
		vstack(play_notes(ev; label = "Play the mystery melody", group = "myst"), B,
			reveal(Markdown.parse("The melody was in rāga **$(RNAMES[r])**. " * (n90 > 0 ? "The computer was more than 90 % sure after **$(n90)** notes." : "The computer never became 90 % sure.")); title = "Which rāga was it?");
			caption = md"**Figure B.4.** A melody in one of the four rāgas, chosen at random. All four start at 25 %.")
	end
end;

# ╔═╡ e14ae8f9-37ec-5fc4-9d28-108f40924b45
fig_pluck(sB_note, sB_pluck)

# ╔═╡ 3fe7394a-e921-5fa3-bfdf-a8fc91385fc9
fig_waltz(sB_dice)

# ╔═╡ f3fb9779-1f85-5312-ad56-c844b5ff9b7c
fig_ragas(sB_raga)

# ╔═╡ b458b585-e73e-5915-8e4c-36116455a89c
fig_mystery(sB_myst)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractPlutoDingetjes = "~1.4.1"
Distributions = "~0.25.131"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "f0d95f568ca9e7dfc7c5fbd16abb8361a116ad6b"

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
version = "8.22.0+0"

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
version = "2026.10.20"

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
version = "3.5.8+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
registries = "General"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

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
version = "1.3.2+0"

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
version = "1.70.0+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─6d631793-d28b-5379-ba38-20f039ec3792
# ╟─5788b98c-f3a8-5b34-9ae5-b20cbbd59e0e
# ╟─1244948a-29c3-5fa9-96da-52588520174c
# ╟─551cbe47-9b82-526e-80e0-f0437f99d1ca
# ╠═0d697b81-6a43-5bfb-9620-5b0781c78782
# ╠═451c1a90-3713-52d5-82a8-10a0fe049a58
# ╠═9e72d97b-ca16-5458-beea-55d7e1f078d6
# ╠═917f96bf-e0eb-562c-9e8f-bffa47425eda
# ╠═3905e070-8a75-570c-ae7c-43716de7bfa5
# ╟─f0c6dc26-3753-5bf7-bb3b-59398dee3fa6
# ╟─90e77356-6b0c-5df6-a400-482447d289c9
# ╟─3ae224f2-861b-5d32-80d3-a2fbe3d44ae0
# ╟─093c1425-a05d-5573-b6b7-be1f2ff50a51
# ╟─46d98e61-0582-5067-819f-83826f341cc2
# ╟─2f7020ae-4ab6-562b-adb6-cb65021184a4
# ╟─83489c02-78f1-574a-9dfc-eaaea8eddc08
# ╟─bc82d9b4-8a8a-5561-a6e7-052c79e293c7
# ╟─de272efe-a3b2-5ef2-991e-2b54cc7102c0
# ╠═517bd77d-009a-50f9-aaf6-322ab5b07073
# ╟─f802de82-955a-5106-a4d6-ac09eb9068f4
# ╟─9b4a22d7-4cf7-50ec-bb12-9d367c8d1902
# ╟─c24791e9-3fef-5b27-9b26-01cfb783a26b
# ╟─d425a268-6583-5bb9-9072-ca59ed4a9e0d
# ╟─bc99d869-1a81-57b5-bd0b-84c9560f3ed7
# ╟─091fd133-48c0-5312-8bec-b8cfd837208b
# ╟─5851eac5-ecb4-5ddb-b266-54531e654ee7
# ╟─577c3021-dd02-5283-96a1-3ab6f9884beb
# ╟─89c53ecc-9c5e-5ca9-a179-122d04f3857b
# ╟─05045e3e-2732-5912-b226-e3472546f874
# ╟─ad7d2f3e-7ede-5764-a3bc-47697c3cee6f
# ╟─820536ed-09a1-5363-9e1e-3cc65bebd79f
# ╟─85055384-1e52-566e-b49c-ecf70de269fd
# ╟─dd337eb5-e171-5a94-89fb-a922a9ca1437
# ╟─7c8b3aea-e97f-55c2-905c-8ef47d87f607
# ╟─18054f38-79ed-5262-aab7-cdd6343642a4
# ╟─f18f88e3-e7d4-5fbd-a151-7a449164a618
# ╠═e38a18dc-2ca9-5085-8d51-5ca9e153060a
# ╟─9bb57f37-3d52-5046-be70-1a8a2c17f674
# ╟─8454a302-2cd9-58f0-a11e-bee87e9ad4fe
# ╟─3d3c8553-033a-5e22-8932-a55dadc9f949
# ╟─28d74357-791f-59ab-9f58-47cada7e736d
# ╟─f2ac9463-6698-549d-81ab-4aade62fa9f9
# ╟─9246ff6c-2244-514a-95db-c5515fb10734
# ╟─da34284f-5df2-5c65-a09a-d4974e7fae39
# ╟─e1e1dbbd-ec2d-57c8-a9c3-011375a2a3bb
# ╟─64e05f96-52de-509c-8ced-d0c09cb56c18
# ╟─dca0d1b0-8755-5bfa-8650-b342cfd4306e
# ╟─9dee6ff4-69aa-5b92-b949-a77c05e3f489
# ╟─2586d80d-48ae-5ff8-8f67-1fce22946920
# ╟─4d50de29-a7d9-55c5-b13c-7cc056f6d0f3
# ╟─042615f4-60e1-51fb-b7dc-d7626d0c43de
# ╟─b5661853-aab5-5cd7-bb5d-60a2ad42867b
# ╟─173493a5-996a-5dc2-9f24-b2ab377d1c75
# ╟─e1c6463f-5489-5a80-9fe3-1c8c3cf3e58e
# ╟─fe273e31-0453-5b4b-8ce3-437b21ff2953
# ╟─ee96ef03-e508-5656-a4ca-c17ee6cb8385
# ╠═1ea7eee7-d1dc-5eb7-8818-b0c9ee475341
# ╠═594e9137-b6fa-5c41-8751-793e93e096a5
# ╟─4287c3ae-ee13-5fd5-8fa8-a5a535ac1419
# ╟─1b59f513-f31c-588b-a4db-66ef5928a11f
# ╟─63212bb9-4635-50c8-b210-1689f5367e8e
# ╟─17475df6-7511-527c-b48c-adef23ff79e9
# ╟─d0dc658e-b0af-5da3-a7c3-1f68a933253e
# ╟─e53cb7c7-5e9d-5f44-becf-356df7e54911
# ╟─95d3ed2d-0e38-596b-a2df-8b081b432580
# ╟─20c9b80e-140b-502c-80aa-d86184f78466
# ╟─d53aa502-ef50-5e15-b6a9-c2704f1fdaac
# ╟─2e27dfa5-2161-5c29-b569-93647b3d822b
# ╟─b2761858-949e-55b9-a406-0730522e7fa3
# ╟─c18db550-9194-5bef-b9dc-e2c75f763718
# ╟─809641dd-b073-52e8-bec6-71e5ee792124
# ╟─3d9093b6-10c4-5c08-b913-d4795e2949cb
# ╟─cef8c045-6446-5e38-9112-194dfb1dab6e
# ╟─4fa16e4d-7156-58ff-ab1e-8a805c215e2d
# ╟─eb689310-a305-5a42-b191-c7fe80fe139e
# ╟─c616c604-383e-59db-9c51-9a9e96223b3e
# ╟─181d2899-b8e4-53af-97ff-a688f914b04a
# ╟─2e59d710-9932-51c1-aa45-c49ea36a2be1
# ╟─8a160746-b8f0-5ad7-85ec-a561ce566552
# ╟─b02b9ad4-305f-5056-ac28-e204a67e65ca
# ╟─1c0a2425-8080-5b4c-8fbb-09636c55f6a8
# ╟─ee77527c-e17a-58bf-b114-b9d9f0023b2c
# ╟─d06eb53c-7fc9-52bd-b378-5ed24df48462
# ╟─a6643369-9aa1-5f04-a0df-15baa3e27339
# ╟─71c14254-c5e4-5a78-95fe-f5563828bbea
# ╟─f4249eb6-ab43-55fa-97ac-e32d45eef240
# ╟─d3e14753-8a81-51d6-aedc-4dfe1d5114d2
# ╟─a44479fc-c5d7-5b24-a916-1a1fbfcd4c99
# ╟─7cef87e8-2181-5e3d-a10c-5500c4cd89cf
# ╟─6b74424c-5828-59ae-b2cc-30d2582785c8
# ╟─97fa04c9-22c5-5b24-b088-11c53be9075c
# ╟─bd5677de-8ecf-5111-8b13-ac4e711f03af
# ╟─b78e1f13-ea6f-5a5a-9969-3cddfeb81f01
# ╟─81a4026e-6407-544c-bd73-d87066f447bc
# ╟─48a9895e-e831-5d2d-b3ff-0d1ba88bd9aa
# ╟─fd9d7dae-d6bf-5e1e-9988-204c749a4cb0
# ╟─7cdf7df5-0aff-568a-b807-b2669803d649
# ╟─c93775d1-47c1-599a-90f9-8ed4f356a247
# ╟─bff3f007-46ba-54e7-a165-2c1d7cdbddb4
# ╟─8213971a-ba13-51fe-8e37-fe31c2b05dcb
# ╟─db38b8d7-fbdc-5f81-b280-d30d26ab7fba
# ╟─bb3f11ac-88dc-57ef-8835-3c8cf08e3433
# ╠═40877568-8051-584c-8645-3cbfc8b4e1f0
# ╟─cedb851f-d1fd-58b9-bbae-62e1a85f5e95
# ╟─ae01a976-cc53-5209-bd1e-8606b0a54977
# ╠═955720cb-49bd-5227-8a66-0b093d12fc3c
# ╟─9ac186d3-bea1-52c2-adcf-cb2c16ef2eaa
# ╟─66915c6c-d82f-545f-a83a-d37b8a6f61e6
# ╠═7713d1e9-e98d-51bd-8bca-56efbfc26b51
# ╟─cb61c9cf-7765-5993-8371-860d42f015f1
# ╟─e4174005-1de3-5fd0-badb-f8a66b795e30
# ╟─314c2546-9098-516d-ace4-1bf91b8dbbad
# ╟─4d264b32-7d36-5589-b10f-de79b73da398
# ╟─e14ae8f9-37ec-5fc4-9d28-108f40924b45
# ╟─abe71179-92e7-5e15-a1b7-1f306b79e5fd
# ╟─3fe7394a-e921-5fa3-bfdf-a8fc91385fc9
# ╟─7205d08e-52ae-51bf-a8f8-24bf95e18fac
# ╟─f3fb9779-1f85-5312-ad56-c844b5ff9b7c
# ╟─e02c61e9-8b5c-5ce4-9891-b6becbf05325
# ╟─b458b585-e73e-5915-8e4c-36116455a89c
# ╟─f0263455-b054-556b-9b2b-923af1d43b58
# ╟─210e0daa-7ca9-59f3-8e78-237a825c9165
# ╟─497add3d-fecd-58d5-b022-f3ddf2491203
# ╟─c88f31de-390c-5032-9041-39c132990ca2
# ╟─9dcbfb89-2e67-56d1-b982-86d29c80faf5
# ╟─55951fa3-a153-5c79-a275-4fe5d3d74120
# ╟─f8277d29-7dfe-5f2c-a832-d4397bc5aafa
# ╟─f6959cf2-a6b3-555e-990c-d3a2341965a0
# ╟─4d17b13c-8ef3-5efd-99e8-1b526bfa5709
# ╟─3a5cca40-4805-51bb-bd91-e075a4305f5b
# ╟─e1344783-2356-5df3-aedd-3fdf01ccd35e
# ╟─f3c97bd5-d7ef-5e96-8c4a-2fbe5957a78b
# ╟─a6ba0801-431b-5714-834e-789110ee48e4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
