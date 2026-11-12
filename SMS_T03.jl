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

# ╔═╡ 1a1fb627-3ddb-58ca-8702-671a29f5e45e
# Packages
begin
	using PlutoUI, LinearAlgebra, Random, Statistics, Markdown, Printf
end

# ╔═╡ fbcd1505-6d97-57ec-8d50-890c01db1ee0
md"""
# Tutorial 03: Discrete-Time Markov Chains
*Stochastic Modeling and Simulation*\
Date: 12.11.2026"""

# ╔═╡ 6c09197f-3d98-54a1-94f9-9047b2005a46
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 7c5135dd-3b5e-5c9a-a1ad-7eb3b034f30b
md"""
!!! info "From Tutorials 01 and 02"
    - A continuous random variable ``X`` has a density ``f_X``; probabilities are areas under the density, ``P(a \le X \le b) = \int_a^b f_X(x)\,dx``. The CDF ``F_X(x) = P(X \le x)`` increases from 0 to 1, and ``f_X = dF_X/dx``.
    - The mean ``E[X]`` and the variance ``\mathrm{Var}(X) = E[X^2] - E[X]^2`` summarise a distribution. By the law of large numbers, ``\bar X_N \to E[X]``; by the central limit theorem, sums of many i.i.d. variables are approximately normal.
    - **Inversion:** if ``U`` is uniform on ``(0, 1)``, then ``X = F^{-1}(U)`` can be used to get samples from any distribution function. In Julia: `quantile.(D, rand(n))`.
    - **Box–Muller:** two independent uniform numbers give two independent standard normal numbers (in practice `randn()`).
    - **Accept–reject:** if only the shape of a density ``f`` is known, points are drawn uniformly under an envelope (proposal distribution ``g`` frm which we can sample), and only the points that fall within ``f`` are kept.

**Colours in the equations.** ``\textcolor{#EE7733}{\text{Orange}}``: the next state. ``\textcolor{#4477AA}{\text{blue}}``: the present state. ``\textcolor{#999999}{\text{grey}}``: the earlier past.
"""

# ╔═╡ 9a2fa3ab-a6ff-560b-af72-7b3de5b77ac4
md"""
## 1. Stochastic processes and memory

Many quantities change randomly over time: the population of a country in year ``n``, the amount of rain on day ``n``, the result of the ``n``-th roll of a die.

!!! definition "Discrete-time stochastic process"
    A sequence of random variables ``X_0, X_1, X_2, \dots``, written ``\{X_n : n \ge 0\}``. The value ``x_n`` of ``X_n`` is the **state** at time ``n``, and ``x_0`` is the **initial state**. The set ``S`` of all possible values is the **state space**. It is **discrete** (die: ``S = \{1, \dots, 6\}``) or **continuous** (amount of rain: ``S = [0, \infty)``).
"""

# ╔═╡ 15132a2d-72e9-5520-b5e6-d871850b96f0
md"""
Dependence on the whole past is the most general case, but it is costly: for ``m`` states there are ``m^n`` possible histories of length ``n``, and each may lead to a different distribution of the next state.
"""

# ╔═╡ 0c3f7523-4219-5afc-84b0-c9f0de7bbc9c
md"""
For a Markov chain, every factor of the product above is a transition probability:

**(4) Probability of a path**
```math
P(X_0 = x_0, X_1 = x_1, \dots, X_n = x_n) = P(X_0 = x_0)\; P_{x_0 x_1} P_{x_1 x_2} \cdots P_{x_{n-1} x_n}.
```
"""

# ╔═╡ 4347d96d-7e4e-53e9-a12f-8c0b09d1217d
md"""
## 2. The transition matrix

For ``m`` states, the transition probabilities ``P_{ij}`` form an ``m \times m`` matrix. From now on, We will use an example of a rental bicycle that moves between three stations, Altmarkt, Campus and Neustadt, once per day.

!!! definition "Transition matrix"
    ``P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} = P(\textcolor{#EE7733}{X_{n+1} = j} \mid \textcolor{#4477AA}{X_n = i})``. The **row** index is the present state, the **column** index the next state. Every entry lies in ``[0, 1]``, and

    **(5) Every row sums to 1**
    ```math
    \sum_{\textcolor{#EE7733}{j}} P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} = 1 \quad \text{for every } \textcolor{#4477AA}{i}.
    ```
    From state ``i`` the chain moves to some state, possibly ``i`` itself (the diagonal entry). Row ``i`` is the distribution of the next state, given the present state ``i``.

Station $(@bind s2_k Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 3)) ``\qquad`` show $(@bind s2_mode Select(["a row: where does a bicycle go from here?", "a column: where do the bicycles arriving here come from?"]))
"""

# ╔═╡ 6a6fc898-9105-5dc6-b2e9-244b339d743a
md"""
!!! situation "Situation"
    A student estimates the transition matrix of a two-state chain and reports
    ```math
    P = \begin{pmatrix} 0.9 & 0.3 \\ 0.1 & 0.7 \end{pmatrix}.
    ```
"""

# ╔═╡ ae8d3fc6-7f2b-5171-81ac-ee0490aee440
md"""
## 3. Markov chains as recursions: simulation

**Question.** How does a computer generate a path of a Markov chain?

The next state is drawn from the row of the present state. This row is a discrete distribution, so the inversion method of Tutorial 02 applies: the interval ``[0, 1)`` is divided into pieces of lengths ``P_{i1}, P_{i2}, \dots``, and the next state is the state whose piece contains a uniform number ``u``.

Present state $(@bind s3_i Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1)) ``\qquad`` ``u`` $(@bind s3_u Slider(0.01:0.01:0.99; default = 0.73, show_value = true))
"""

# ╔═╡ a11433f6-b90f-4df3-bb25-8e62382093a0
md"""
**Does the order of the pieces matter?** Line 4 of the algorithm divides ``(0, 1)`` into pieces in a fixed order. There is no reason why the piece of the Altmarkt should be ``[0, 0.6)`` rather than ``[0.4, 1)``. Figure 3.3 divides the same row in two orders.

Row of $(@bind s3_ostart Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1)) ``\qquad`` ``U`` $(@bind s3_ou Slider(0.01:0.01:0.99; default = 0.5, show_value = true))
"""

# ╔═╡ a6c3bfb6-ac6d-596a-9248-91e3c3e4c76e
md"""
!!! theorem "Markov chains as recursions"
    **(6) Recursion**
    ```math
    \textcolor{#EE7733}{X_{n+1}} = f(\textcolor{#4477AA}{X_n}, U_n), \qquad U_0, U_1, \dots \text{ i.i.d. and independent of } X_0 .
    ```
    Every Markov chain can be written in this form with ``U_n`` uniform on ``(0, 1)``; ``f(i, u)`` is the inversion of row ``i``. A generator of uniform random numbers is therefore sufficient to simulate any Markov chain.

In this notebook, `simulate(P, i0, n, rng)` implements this recursion:
"""

# ╔═╡ 47a76164-e6ea-5f7d-833d-8a18949c1beb
md"""
!!! question "Question 4.2"
    In a short text the letter *z* occurs three times, each time followed by *e*, so that ``\hat P_{ze} = 1``. How reliable is this estimate?
"""

# ╔═╡ 013b4e5b-979e-56b0-a2f7-302d6993eefb
md"""
## 5. More than one step

**Question.** A bicycle is at Neustadt today. What is the probability that it is at the Altmarkt in **two** days?

The bicycle is at some station ``k`` tomorrow. Conditioning on this station (law of total probability) and applying the Markov property gives

```math
P(\textcolor{#EE7733}{X_2 = j} \mid X_0 = i) = \sum_k P(\textcolor{#4477AA}{X_1 = k} \mid X_0 = i)\, P(\textcolor{#EE7733}{X_2 = j} \mid \textcolor{#4477AA}{X_1 = k}) = \sum_k P_{ik} P_{kj} = (P^2)_{ij}.
```

The same argument applied repeatedly gives the ``n``-step transition probabilities ``P^{(n)}_{ij} = P(X_n = j \mid X_0 = i)``:

**(8) Chapman–Kolmogorov equation**
```math
P^{(m+n)} = P^{(m)} P^{(n)}, \qquad P^{(n)} = P^n .
```

From $(@bind s5_i Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 3)) ``\qquad`` to $(@bind s5_j Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1))
"""

# ╔═╡ d7bbb98f-7532-539e-9060-00f31cf55aa7
md"""
### The distribution after ``n`` steps

Often the initial state is not known exactly but given by a distribution. The distribution on day ``n`` is written as a **row vector** ``\pi_n = \big(P(X_n = 1), \dots, P(X_n = m)\big)``. The same argument as above gives

**(9) Distribution update**
```math
\textcolor{#EE7733}{\pi_{n+1}} = \textcolor{#4477AA}{\pi_n} P, \qquad \pi_n = \pi_0 P^n.
```

Start $(@bind s5_start Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 3))
"""

# ╔═╡ 2dc2b02c-584e-426a-a455-9630abebf04d
md"""
## 6. The long run: stationary and limiting distributions

Many questions about a chain concern its behaviour after a long time. How many bicycles should each station be able to hold? What fraction of days is rainy in a typical summer? What fraction of time does a molecule spend in each of its shapes? These questions concern the distribution ``\pi_n`` for large ``n``. Since ``\pi_{n+1} = \pi_n P`` by (9), two questions arise:

1. Is there a distribution that does not change from one step to the next?
2. Does ``\pi_n`` approach the same distribution for every initial state?

!!! definition "Stationary distribution"
    A row vector ``\pi`` with ``\pi_i \ge 0`` is a **stationary distribution** if

    **(10) Stationarity**
    ```math
    \textcolor{#4477AA}{\pi} P = \textcolor{#EE7733}{\pi}, \qquad \sum_i \pi_i = 1 .
    ```
    A chain that starts in ``\pi`` remains in ``\pi`` at every later step.

!!! definition "Limiting distribution"
    If the limit

    **(11) Limiting distribution**
    ```math
    \Pi_k = \lim_{n \to \infty} (P^n)_{jk}
    ```
    exists and does not depend on the initial state ``j``, then ``\Pi`` is the **limiting distribution**. From every initial state, the distribution of the chain converges to ``\Pi``.

The two definitions answer different questions. A stationary distribution is preserved by the chain; a limiting distribution is reached by the chain from any initial state.

### 6.1 The bicycle chain forgets its initial state

Day ``n`` $(@bind s6_n Slider(0:1:20; default = 3, show_value = true))
"""

# ╔═╡ 2924eb10-59ad-4572-82d4-5b5c25af5a59
md"""
For every initial station, the three probabilities approach the same values (dashed), and all rows of ``P^n`` become equal. The limiting distribution (11) therefore exists, and it is the stationary distribution of the chain.
"""

# ╔═╡ 9c7d85da-3170-4fcc-a138-f610cffee1cf
md"""
**Consequence for the long run.** In the chain with two separate parts, a start in ``\{1, 2\}`` and a start in ``\{3, 4\}`` lead to **different** distributions, and the chain has more than one stationary distribution. An irreducible chain has exactly one stationary distribution.
"""

# ╔═╡ 6bdd1741-0959-4ad7-86d2-86c35ecf3161
md"""
- **Ergodic** (chain D): one stationary distribution, which every initial state approaches. The stationary and the limiting distribution coincide.
- **Irreducible and periodic** (chain C): one stationary distribution, ``(0.25, 0.5, 0.25)``, but ``\pi_n`` oscillates. A stationary distribution exists; a limiting distribution does not.
- **Not irreducible, aperiodic** (chain B): each closed part has its own stationary distribution, and the long-run behaviour depends on the initial state. There are infinitely many stationary distributions and no limiting distribution.
- **Not irreducible, periodic** (chain E): infinitely many stationary distributions and no convergence.
"""

# ╔═╡ 2a76ae63-c44c-410c-b498-784caa67c8e6
md"""
### 7.5 Which kind of chain is it?

Each round shows a chain as a state diagram and as a transition matrix. Classify the chain.
"""

# ╔═╡ 8448bc74-05eb-5312-b41a-30cc5ac050a4
md"""
## Summary

!!! info "Equations of this tutorial"
    ``\textcolor{#EE7733}{\text{Orange}}``: next state; ``\textcolor{#4477AA}{\text{blue}}``: present state; ``\textcolor{#999999}{\text{grey}}``: earlier past.

    **(1) i.i.d. process** ``\quad P(\textcolor{#EE7733}{X_{n+1} = x} \mid \textcolor{#4477AA}{X_n}, \textcolor{#999999}{\dots, X_0}) = p(x)``

    **(2) Whole past** ``\quad P(\textcolor{#EE7733}{X_{n+1}} \mid \textcolor{#4477AA}{X_n}, \textcolor{#999999}{X_{n-1}, \dots, X_0})`` depends on every earlier state

    **(3) Markov property** ``\quad P(\textcolor{#EE7733}{X_{n+1} = j} \mid \textcolor{#4477AA}{X_n = i}, \textcolor{#999999}{\dots, X_0}) = P(\textcolor{#EE7733}{X_{n+1} = j} \mid \textcolor{#4477AA}{X_n = i}) = P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}}``

    **(4) Probability of a path** ``\quad P(X_0 = x_0, \dots, X_n = x_n) = P(X_0 = x_0)\, P_{x_0 x_1} \cdots P_{x_{n-1} x_n}``

    **(5) Rows sum to 1** ``\quad \sum_{\textcolor{#EE7733}{j}} P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} = 1``

    **(6) Recursion** ``\quad \textcolor{#EE7733}{X_{n+1}} = f(\textcolor{#4477AA}{X_n}, U_n)``, with ``f`` the inversion of the row of ``X_n``

    **(7) Estimate from counts** ``\quad \hat P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} = C_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} / \sum_k C_{\textcolor{#4477AA}{i}k}``

    **(8) Chapman–Kolmogorov** ``\quad P^{(m+n)} = P^{(m)} P^{(n)}, \quad P^{(n)} = P^n``

    **(9) Distribution update** ``\quad \textcolor{#EE7733}{\pi_{n+1}} = \textcolor{#4477AA}{\pi_n} P, \quad \pi_n = \pi_0 P^n``

    **(10) Stationary distribution** ``\quad \pi P = \pi, \quad \sum_i \pi_i = 1``

    **(11) Limiting distribution** ``\quad \Pi_k = \lim_{n \to \infty} (P^n)_{jk}``, independent of the initial state ``j``; it exists for every ergodic chain

!!! info "Terms of Section 7"
    **Closed set:** a set that no arrow leaves. **Absorbing state:** ``P_{ii} = 1``. **Irreducible:** ``S`` is the only closed set; every state can be reached from every other state. **Period:** the largest ``d`` such that returns are possible only after multiples of ``d`` steps. **Aperiodic:** period 1. **Ergodic:** irreducible and aperiodic.

## Exercises

### Calculation problems
"""

# ╔═╡ 4a09ac4a-45a6-53dd-94b2-8f143c042c94
md"""
### Programming exercises

**Exercise 1. From counts to a transition matrix.** `C[i, j]` is the number of observed transitions ``i \to j``. Complete `normalise_rows(C)` so that it returns the estimated transition matrix (7).
"""

# ╔═╡ 8836a53d-2c65-5a2b-8a65-5915655488fb
function normalise_rows(C)
	return missing   # your code here
end

# ╔═╡ 756c7eef-e0ef-5f2f-959b-a0160ecd591f
md"""
**Exercise 2. Correcting a simulation.** `simulate_chain(P, x0, n, rng)` should return a path ``x_0, \dots, x_n``. The function contains one error: depending on the matrix, it either fails or returns paths with incorrect transition frequencies. Find and correct the error.
"""

# ╔═╡ c13cda56-0424-553d-bbb2-9304cc243e93
function simulate_chain(P, x0, n, rng)
	x = zeros(Int, n + 1)
	x[1] = x0
	for t in 1:n
		x[t+1] = findfirst(>(rand(rng)), cumsum(P[:, x[t]]))
	end
	return x
end

# ╔═╡ 9c1bf37e-d165-58ab-a7ff-e7cfff82ce2a
md"""
**Exercise 3. Correcting the distribution after ``n`` steps.** `distribution_after(P, π0, n)` should return ``\pi_0 P^n`` for a row vector `π0`. Predict how the incorrect result depends on ``n``, then correct the function.
"""

# ╔═╡ 0ae8b55a-8e0c-569e-8ad1-f023348cf176
function distribution_after(P, π0, n)
	π = π0
	for _ in 1:n
		π = π0 * P
	end
	return π
end

# ╔═╡ d66c5fca-a098-5ead-be72-e0f55bef477e
md"""
**Exercise 4. The stationary distribution by iteration.** Complete `stationary_power(P)`: start from the uniform distribution and apply ``\pi \leftarrow \pi P`` until ``\pi`` no longer changes, for example until no entry changes by more than ``10^{-12}``. Return ``\pi``.
"""

# ╔═╡ d7d594a3-8dd8-583d-8ffb-1218a9650f1d
function stationary_power(P)
	return missing   # your code here
end

# ╔═╡ b81388ba-addc-5606-b0ff-490c2648e13d
md"""
## Further reading

- V. Powell and L. Lehe, *Markov Chains, Explained Visually*: <https://setosa.io/ev/markov-chains/>.
- J. R. Norris, *Markov Chains*, Cambridge University Press, 1997, Chapter 1.
- D. A. Levin, Y. Peres and E. L. Wilmer, *Markov Chains and Mixing Times*, 2nd ed., AMS, 2017, Chapters 1 and 4.
- C. E. Shannon, "A mathematical theory of communication", *Bell System Technical Journal* 27 (1948) 379–423.
- A. A. Markov, "An example of statistical investigation of the text *Eugene Onegin* concerning the connection of samples in chains" (1913), English translation in *Science in Context* 19 (2006) 591–600.

## Data

- Weather chain of Section 1.3: daily precipitation, summers 1960–2025, Dresden-Klotzsche (station 1048), Deutscher Wetterdienst, Climate Data Center, <https://opendata.dwd.de>. Rain means at least 1 mm.
- Books of Section 4 (Project Gutenberg, public domain): L. Carroll, *Alice's Adventures in Wonderland* (eBook 11); J. Austen, *Pride and Prejudice* (eBook 1342); *Grimms' Fairy Tales*, transl. E. Taylor and M. Edwardes (eBook 2591).
- Melodies of the bonus section: traditional tunes and the main theme of the fourth movement of Beethoven's Ninth Symphony, all in the public domain.

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 3.
"""

# ╔═╡ 676699ad-6d5a-5785-b0cf-36422c57c8ab
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
.pl-bar input[type=range]{flex:1;min-width:160px;accent-color:#4477AA}
.pl-read{font:13px system-ui,sans-serif;color:#333;min-height:1.4em;margin-top:4px}
.quiz{border:1px solid #e3e3df;border-radius:4px;padding:10px 14px;margin:14px 0;background:#fdfdfc}
.quiz-head{font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;font-weight:700;color:#AA3377;margin-bottom:4px}
.quiz-q p{margin:2px 0 6px}
.quiz-opt{display:flex;gap:10px;align-items:baseline;border:1px solid #e3e3df;border-radius:4px;padding:5px 10px;margin:6px 0;cursor:pointer;background:#fff}
.quiz-opt:hover{border-color:#4477AA}
.quiz-opt p,.quiz-fb p{margin:0}
.quiz-key{font-weight:700;color:#8c8c8c;min-width:1.2em}
.quiz-opt.right{border-color:#228833;background:#eef6f0}
.quiz-opt.wrong{border-color:#CC3311;background:#fbf1ee}
.quiz-fb{border-radius:3px;padding:7px 12px;margin:6px 0 2px;font-size:.95rem}
.quiz-fb.yes{background:#eef6f0;border-left:3px solid #228833}
.quiz-fb.no{background:#fbf1ee;border-left:3px solid #CC3311}
.quiz-num input{font:inherit;padding:3px 8px;border:1px solid #cfcfca;border-radius:3px}
.quiz-num button{padding:2px 10px;cursor:pointer}
.admonition.tasks{border-color:#AA3377;background:#f8f1f5}
.admonition.tasks>.admonition-title{background:#AA3377;color:#fff}
.admonition.situation{border-color:#EE7733;background:#fdf4ec}
.admonition.situation>.admonition-title{background:#EE7733;color:#fff}
.pc{font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:#fafaf7;border:1px solid #e0e0da;border-radius:4px;padding:6px 0;min-width:300px;max-width:470px;color:#1b1b1b}
.pc-head{font:700 .72rem system-ui,sans-serif;letter-spacing:.09em;text-transform:uppercase;color:#3d4a5c;padding:0 10px 4px}
.pc-ln{position:relative;padding:0 10px 0 34px;white-space:pre}
.pc-no{position:absolute;left:8px;color:#a0a09a;z-index:1}
.pc-ln code{position:relative;z-index:1;font:inherit;background:none;padding:0}
.pc-mark{position:absolute;left:0;right:0;top:0;bottom:0;background:#e3ecf8;border-left:3px solid #4477AA;z-index:0}
.eqs{border-left:3px solid #4477AA;padding:2px 0 2px 14px;margin:10px 0}
.review{border:1px solid #e3e3df;border-left:5px solid #999;border-radius:4px;background:#fcfcfa;padding:8px 14px 4px;margin:12px 0;font-size:.95rem}
.review-head{font:700 .78rem system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:#333;margin-bottom:4px}
.review ol{margin:4px 0 6px;padding-left:20px}
.review li{margin:4px 0}
.review li p{margin:0;display:inline}
.review-wait{font-size:.9rem;color:#777;font-style:italic;margin:6px 0}
.pluto-docs-binding{display:none}
.og{border:1px solid #e3e3df;border-radius:6px;padding:12px 14px;margin:10px 0;font-family:system-ui,sans-serif}
.og-top{display:flex;justify-content:space-between;font-size:14px}
.og-opts{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
.og-opts button{padding:5px 10px;border:1px solid #cfcfca;border-radius:4px;background:#fff;cursor:pointer;font:inherit;font-size:13.5px}
.og-opts button.ok{border-color:#228833;background:#eef6f0}.og-opts button.no{border-color:#CC3311;background:#fbf1ee}
.og-fb{min-height:1.6em;font-weight:600;font-size:14px;margin-bottom:6px}
</style>"""

# ╔═╡ 1416ee70-0bd9-5f50-a1d2-64ff3e152e45
# Figures: a small SVG plotting library (blue = exact distribution, orange = samples)
begin
	const COL = (blue = "#4477AA", orange = "#EE7733", red = "#CC3311", green = "#228833", purple = "#AA3377",
		ochre = "#CCBB44", grey = "#999999", ink = "#222222")

	# short number format: 3 significant digits, scientific notation for very small or large values
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

	# data → pixel coordinates
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
	seg!(f, x0, y0, x1, y1; c = COL.grey, w = 1.2, dash = nothing, op = 1.0, s = nothing, o = nothing, over = false) =
		raw!(f, "<line x1=\"$(_n(X(f, x0)))\" y1=\"$(_n(Y(f, y0)))\" x2=\"$(_n(X(f, x1)))\" y2=\"$(_n(Y(f, y1)))\" stroke=\"$(c)\" stroke-width=\"$(w)\" opacity=\"$(op)\"$(_dash(dash))$(_st(s, o))/>"; over)
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
	text!(f, x, y, str; c = COL.ink, size = 12, anchor = "start", bold = false, dx = 0, dy = 0, bg = false, s = nothing, o = nothing) =
		raw!(f, "<text x=\"$(_n(X(f, x) + dx))\" y=\"$(_n(Y(f, y) + dy))\" font-size=\"$(size)\" fill=\"$(c)\" text-anchor=\"$(anchor)\" dominant-baseline=\"middle\"" *
			(bold ? " font-weight=\"600\"" : "") * (bg ? " paint-order=\"stroke\" stroke=\"white\" stroke-width=\"3.5\"" : "") * _st(s, o) * ">$(xesc(str))</text>"; over = true)

	# legend: items are (label, colour) or (label, colour, :line/:dash/:bar)
	function legend!(f, items; at = :tr, size = 12)
		n = length(items); wd = maximum(length(first(it)) for it in items) * 6.6 + 34
		x0 = at in (:tr, :br) ? f.w - f.r - wd - 4 : f.l + 10; y0 = at in (:tr, :tl) ? f.t + 8 : f.h - f.b - 18n - 4
		raw!(f, "<rect x=\"$(_n(x0 - 6))\" y=\"$(_n(y0 - 4))\" width=\"$(_n(wd + 8))\" height=\"$(18n + 4)\" fill=\"white\" opacity=\"0.85\"/>"; over = true)
		for (i, it) in enumerate(items)
			y = y0 + 18(i - 0.5); k = get(it, 3, :line)
			raw!(f, (k == :bar ? "<rect x=\"$(_n(x0 + 3))\" y=\"$(_n(y - 6))\" width=\"14\" height=\"12\" fill=\"$(it[2])\" opacity=\"0.8\"/>" :
				k == :dot ? "<circle cx=\"$(_n(x0 + 10))\" cy=\"$(_n(y))\" r=\"3.5\" fill=\"$(it[2])\"/>" :
				"<line x1=\"$(_n(x0))\" x2=\"$(_n(x0 + 20))\" y1=\"$(_n(y))\" y2=\"$(_n(y))\" stroke=\"$(it[2])\" stroke-width=\"2.2\"$(_dash(k == :dash ? "4 3" : nothing))/>") *
				"<text x=\"$(_n(x0 + 28))\" y=\"$(_n(y))\" font-size=\"$(size)\" dominant-baseline=\"middle\">$(xesc(it[1]))</text>"; over = true)
		end
		f
	end

	# about n round tick values between lo and hi
	function niceticks(lo, hi, n)
		lo, hi = minmax(lo, hi); hi > lo || return [lo]
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
end;

# ╔═╡ 700553cc-975f-57b9-bfb1-ab8cf122591d
# Interaction: seed button, pseudocode, step player and canvas animations
begin
	Seed(default = 0; label = "New random numbers") = CounterButton(label)

	# Julia values as a JavaScript literal (for data passed to scripts)
	js(x::Real) = isfinite(x) ? string(round(Float64(x); sigdigits = 7)) : "null"
	js(x::Union{Integer, Bool}) = string(x)
	js(x::AbstractString) = repr(x)
	js(x::Union{AbstractVector, Tuple}) = "[" * join(js.(x), ",") * "]"
	js(x::NamedTuple) = "{" * join(("\"$(k)\":" * js(v) for (k, v) in pairs(x)), ",") * "}"

	# Pseudocode with numbered lines; active[k] lists the lines shaded at step k - 1 of a player
	function pseudo(lines; title = "", active = Vector{Int}[])
		marks(i) = join("<span class=\"pc-mark\" data-o=\"$(k - 1)\"></span>" for (k, a) in enumerate(active) if i in a)
		HTML("<div class=\"pc\">" * (title == "" ? "" : "<div class=\"pc-head\">$(xesc(title))</div>") *
			join("<div class=\"pc-ln\">$(marks(i))<span class=\"pc-no\">$(i)</span><code>$(xesc(l))</code></div>" for (i, l) in enumerate(lines)) * "</div>")
	end

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
end;

# ╔═╡ 12d6ecd7-5a90-5d99-ac27-ea8d2e79e8f7
md"""
Suppose the states ``x_0, \dots, x_n`` have been observed. The distribution of the next state ``X_{n+1}`` may depend on

- none of the past states (rolls of a die),
- the present state ``x_n`` only,
- all past states ``x_0, \dots, x_n``.

How much of the past the next state depends on is called the **memory** of the process.

The memory determines how the probability of a whole sequence is computed. For every process, the chain rule of Tutorial 01 writes this probability as a product with one factor per time step:

```math
P(X_0 = x_0, \dots, X_n = x_n) = P(X_0 = x_0) \prod_{k=1}^{n} P\big(\textcolor{#EE7733}{X_k = x_k} \mid \textcolor{#4477AA}{X_{k-1}}, \textcolor{#999999}{X_{k-2}, \dots, X_0}\big).
```

Each factor is the probability of the next state (orange) given the present state (blue) and the earlier states (grey).

### 1.1 No memory: i.i.d. processes

The tosses of a fair coin are independent and have the same distribution. Such a process is called **independent and identically distributed** (i.i.d.):

**(1) i.i.d. process**
```math
P\big(\textcolor{#EE7733}{X_{n+1} = x} \mid \textcolor{#4477AA}{X_n}, \textcolor{#999999}{X_{n-1}, \dots, X_0}\big) = P(\textcolor{#EE7733}{X_{n+1} = x}) = p(x) \quad \text{for all } n.
```

A single distribution ``p`` describes the process, and the probability of a path is the product ``p(x_0)\,p(x_1) \cdots p(x_n)``.

$(@bind s1_iid Seed(0; label = "New tosses"))
"""

# ╔═╡ dab72e64-ba25-5367-9248-00db2044fef9
md"""
### 1.2 Memory of the whole past

A deck contains 10 red and 10 black cards. The cards are drawn one at a time **without** replacement. The probability that the next card is red depends on the number of red cards drawn so far, and therefore on all earlier draws:

**(2) Dependence on the whole past**
```math
P\big(\textcolor{#EE7733}{X_{n+1} = \text{red}} \mid \textcolor{#4477AA}{X_n}, \textcolor{#999999}{X_{n-1}, \dots, X_1}\big) = \frac{10 - (\text{number of red cards among } X_1, \dots, X_n)}{20 - n}.
```

$(@bind s1_cards Seed(0; label = "Shuffle again"))
"""

# ╔═╡ dbf76961-8cb4-5cbd-9fc8-83a637e782ff
md"""
### 1.3 One step of memory: Markov chains

Between these two cases lies the most widely used model. Tomorrow's weather depends strongly on today's weather; once today is known, the weather of the previous week adds little information.

!!! definition "Markov chain"
    ``\{X_n\}`` is a **Markov chain** if, for all ``n`` and all states,

    **(3) Markov property**
    ```math
    P\big(\textcolor{#EE7733}{X_{n+1} = j} \mid \textcolor{#4477AA}{X_n = i}, \textcolor{#999999}{X_{n-1} = x_{n-1}, \dots, X_0 = x_0}\big) = P\big(\textcolor{#EE7733}{X_{n+1} = j} \mid \textcolor{#4477AA}{X_n = i}\big) =: P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}}.
    ```
    The distribution of the next state depends only on the present state and not on the path by which it was reached. ``P_{ij}`` is the **transition probability** from ``i`` to ``j``.

As an example, consider a weather chain with the two states sun (S) and rain (R), with ``P(\text{S} \to \text{R}) = 0.26`` and ``P(\text{R} \to \text{R}) = 0.45``.

$(@bind s1_markov Seed(0; label = "New weather"))
"""

# ╔═╡ d5132e9a-4831-569f-a096-62737eaaf5e5
md"""
Figure 1.4 shows the same chain as a **state diagram**: one circle for each state and one arrow for each possible transition, labelled with its probability.

Start $(@bind s1_wstart Select([1 => "sun", 2 => "rain"])) ``\qquad`` $(@bind s1_wseed Seed(0; label = "New path"))
"""

# ╔═╡ 3eb00177-1a76-58f6-be1f-cf09baed9850
md"""
**All two-state chains.** A chain with two states is determined by two numbers, ``a = P_{12}`` and ``b = P_{21}``. Figure 2.2 shows how ``a`` and ``b`` change the path.

``a = P(\text{sun} \to \text{rain})`` $(@bind s2_a Slider(0.02:0.02:0.98; default = 0.26, show_value = true)) ``\qquad`` ``b = P(\text{rain} \to \text{sun})`` $(@bind s2_b Slider(0.02:0.02:0.98; default = 0.56, show_value = true)) ``\qquad`` start $(@bind s2_start Select([1 => "sun", 2 => "rain"])) ``\qquad`` $(@bind s2_sseed Seed(0))
"""

# ╔═╡ 5bc81e14-9e62-5843-9bd1-327876aae4c7
md"""
### Playground

Enter a matrix with one row per line and the entries separated by spaces. The notebook checks condition (5), draws the state diagram and simulates the chain.

$(@bind s2_text TextField((36, 4); default = "0.0 0.7 0.3\n0.5 0.0 0.5\n0.6 0.4 0.0")) ``\qquad`` start in state $(@bind s2_pstart Select(1:6)) ``\qquad`` $(@bind s2_pseed Seed(0; label = "New path"))
"""

# ╔═╡ 0c797968-7e82-57cd-98a2-b5902e4f8ef1
md"""
Repeating this step produces a path. Figure 3.2 executes the algorithm; each step uses the row of the **current** state and a new uniform number.

Start $(@bind s3_i0 Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 3)) ``\qquad`` $(@bind s3_seed Seed(0; label = "New uniform numbers"))
"""

# ╔═╡ 0a87c857-79be-5e51-bbdc-11e81c4764f4
md"""
### 4.2 Generating text

Generating text means simulating the chain as in Figure 3.2: start with a character, draw the next character from its row with one uniform number, and repeat.

$(@bind s4_gseed Seed(0; label = "Generate again"))
"""

# ╔═╡ 0caf768a-0e16-528f-9d82-7337aab2c233
md"""
### 4.3 Longer memory

Let the state be the last ``k`` characters (or words). As in Question 1.8, this enlarged state again forms a Markov chain: the next state depends only on the present one. Its rows are estimated by counting which character follows each context in the book.

Unit $(@bind s4_unit Select(["characters", "words"])) ``\qquad`` memory ``k`` $(@bind s4_k Slider(1:8; default = 3, show_value = true)) ``\qquad`` $(@bind s4_tseed Seed(0; label = "Generate again"))
"""

# ╔═╡ 5de3b9d1-4a34-5aed-ae11-deaa1e4b4994
md"""
Each bicycle of a fleet follows its own path of the same chain. The number of bicycles on day 0 at each station defines the starting distribution ``\pi_0``.

Bicycles on day 0: Altmarkt $(@bind s5_a NumberField(0:300; default = 0)) ``\quad`` Campus $(@bind s5_c NumberField(0:300; default = 0)) ``\quad`` Neustadt $(@bind s5_n NumberField(0:300; default = 300)) ``\qquad`` days $(@bind s5_D Select([10, 30, 60]; default = 30)) ``\qquad`` $(@bind s5_seed Seed(0; label = "New random paths"))
"""

# ╔═╡ b99d244d-c8ff-49ad-81af-b66fbed9e35a
md"""
## 8. Mixing time: how long is "long enough"?

The limit (11) concerns ``n \to \infty``. In practice an ergodic chain reaches its limiting distribution to any required accuracy after a finite number of steps, the **mixing time**. After the mixing time, the states of the chain follow the limiting distribution; the steps before it still depend on the initial state and are discarded (the **burn-in**).

The mixing time is rarely known in advance. It is determined by simulation: many chains are run from the same initial state, and one observes when their distribution stops changing. Figure 8.1 compares two ergodic chains: Chain 1 is the bicycle chain, and Chain 2 has six states.

Chain $(@bind s8_chain Select(["Chain 1: the bicycle chain", "Chain 2"])) ``\qquad`` start $(@bind s8_start Select(1:6)) ``\qquad`` $(@bind s8_seed Seed(0; label = "New simulation"))
"""

# ╔═╡ f4cb8951-f5e1-452d-b24c-70eb2ebad9e8
md"""
A melody is generated by simulating the note chain, as in Figure 3.2.

$(@bind sb_gseed Seed(0; label = "Generate again"))
"""

# ╔═╡ f33a5b05-dae3-47f7-9c91-416ee1222614
md"""
With the last ``k`` notes as the state, the generated melody contains longer passages of the original melody.

Memory ``k`` $(@bind sb_k Slider(1:4; default = 2, show_value = true)) ``\qquad`` $(@bind sb_tseed Seed(0; label = "Generate again"))
"""

# ╔═╡ 6fedd8f9-68da-5565-ad6e-09a8ddd985bb
# Questions: multiple choice, numeric answers, hidden answers and checks of code exercises
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
		inp.addEventListener("keydown", e => { e.stopPropagation(); if (e.key === "Enter") check(); });
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

# ╔═╡ 173b4138-5a7a-5c9b-adca-9e44dc21dfb7
mcq(md"Which of these processes has a **continuous** state space?",
	md"The result of the ``n``-th roll of a die." => md"``S = \{1, \dots, 6\}`` is discrete.",
	md"The time the ``n``-th patient spends in hospital." => md"Correct. A duration can take any value in ``[0, \infty)``.",
	md"The number of cars in a car park at hour ``n``." => md"``S = \{0, 1, 2, \dots\}`` is discrete.";
	correct = 2, head = "Question 1.1")

# ╔═╡ 69cd095c-74f5-5d53-8cc6-7c2b577c1e31
numq(md"A fair coin is tossed three times. What is the probability of the sequence head, head, tail?", 1 / 8; rtol = 0.001,
	why = md"By (1) the factors are independent: ``\tfrac12 \cdot \tfrac12 \cdot \tfrac12 = \tfrac18``.", head = "Question 1.2")

# ╔═╡ 54308917-b0c7-532d-a6ec-03728a6a5b0a
mcq(md"A fair coin has shown head five times in a row. What is the probability of head on the next toss?",
	md"Less than 0.5, because tail is due." => md"The coin has no memory. By (1) the past does not change the probability.",
	md"0.5." => md"Correct. In an i.i.d. process the past carries no information about the future.",
	md"More than 0.5, because the coin is on a run." => md"The coin has no memory. By (1) the past does not change the probability.";
	correct = 2, head = "Question 1.3")

# ╔═╡ 08fc849e-be46-5664-af48-e0d0e43bb63f
numq(md"Today is sunny, and ``P(\text{S} \to \text{S}) = 0.74``, ``P(\text{S} \to \text{R}) = 0.26``, ``P(\text{R} \to \text{R}) = 0.45``. What is the probability that the next three days are sun, rain, rain?", 0.74 * 0.26 * 0.45; rtol = 0.01,
	why = md"By (4): ``P_{SS} P_{SR} P_{RR} = 0.74 \cdot 0.26 \cdot 0.45 \approx 0.087``. Each factor depends only on the preceding day.", head = "Question 1.4")

# ╔═╡ 024bb141-72c1-50ee-bded-0ab124965f51
mcq(md"Let ``\{X_n\}`` be a Markov chain. Which statement is correct?",
	md"``X_{n+1}`` is independent of ``X_{n-1}``." => md"Not in general. Rain yesterday makes rain today likely, and rain today makes rain tomorrow likely; yesterday is therefore informative about tomorrow.",
	md"``X_{n+1}`` is independent of ``X_{n-1}`` **given** ``X_n``." => md"Correct. Given the present, the future is independent of the past.",
	md"All ``X_n`` are independent." => md"This describes an i.i.d. process.";
	correct = 2, head = "Question 1.5")

# ╔═╡ 48884c22-0e49-5773-b8c3-2a90f010b51a
mcq(md"Is an i.i.d. process a Markov chain?",
	md"No, because it has no memory." => md"Condition (3) still holds: both sides equal ``P(X_{n+1} = j)``.",
	md"Yes. Condition (3) holds, and every row of the transition matrix is the same distribution ``p``." => md"Correct (script, Example 3.4). An i.i.d. process is the special Markov chain in which the present state has no influence.";
	correct = 2, head = "Question 1.6")

# ╔═╡ d912443f-9594-5a79-9902-abeda451aa67
mcq(md"Which of these processes is **not** a Markov chain with the given state?",
	md"The capital of a player after each bet of 1 euro on a fair coin." => md"Tomorrow's capital is today's capital ``\pm 1``; it depends on today only.",
	md"The colour of the last card drawn from the deck of Section 1.2." => md"Correct. The next card depends on all earlier cards, not only on the last one.",
	md"The result of a die rolled once a day." => md"An i.i.d. process is a Markov chain.";
	correct = 2, head = "Question 1.7")

# ╔═╡ 89e1eec7-bae1-5a50-a15b-581c444f2b69
mcq(md"Tomorrow's weather depends on today **and** on yesterday. Can it be modelled by a Markov chain?",
	md"No, because a Markov chain has only one step of memory." => md"Not with the weather of a single day as the state. The state can, however, be redefined.",
	md"Yes, with the **pair** (yesterday, today) as the state, that is, with the four states SS, SR, RS, RR." => md"Correct. The pair at time ``n + 1`` depends only on the pair at time ``n``. Any finite memory becomes a Markov chain when the state is enlarged.",
	md"Yes, by adding two transition matrices." => md"The rows of the sum add up to 2; it is not a transition matrix.";
	correct = 2, head = "Question 1.8")

# ╔═╡ 67add958-bbb4-5765-952f-e76af480fb8d
numq(md"A bicycle is at the Campus today. What is the probability that it is at Neustadt tomorrow?", 0.15; rtol = 0.001,
	why = md"Row Campus, column Neustadt: ``P_{23} = 0.15``.", head = "Question 2.1")

# ╔═╡ 7b1ebe0a-4337-5f04-8f87-953e0089b7b3
mcq(md"Why must every row of ``P`` sum to 1, but not every column?",
	md"Row ``i`` is the distribution of the next state given the present state ``i``. Column ``j`` collects probabilities from different present states and is not a distribution." => md"Correct. The column of Neustadt, ``(0.15, 0.15, 0.60)``, sums to 0.9; nothing requires it to sum to 1.",
	md"This is a convention; a matrix whose columns sum to 1 describes the same chain." => md"In this course and in the script, rows correspond to the present state, and the rows must sum to 1.",
	md"Both rows and columns must sum to 1." => md"See the column of Neustadt in Figure 2.1.";
	correct = 1, head = "Question 2.2")

# ╔═╡ ff9c1ef1-7949-5bdb-803c-32f0a6001ca7
numq(md"A chain with three states has the first row ``(0.2,\ ?,\ 0.5)``. What is the missing entry?", 0.3; rtol = 0.001,
	why = md"By (5): ``1 - 0.2 - 0.5 = 0.3``.", head = "Question 2.3")

# ╔═╡ ec1eb4fe-b24c-59e4-80eb-34ffad90c957
mcq(md"What is wrong with this matrix?",
	md"Nothing; all entries lie between 0 and 1." => md"The rows sum to 1.2 and 0.8.",
	md"The columns sum to 1 and the rows do not: the matrix is transposed." => md"Correct. The counts were divided by the column totals instead of the row totals. The correct matrix is ``\begin{pmatrix} 0.9 & 0.1 \\ 0.3 & 0.7 \end{pmatrix}``.",
	md"A transition matrix must be symmetric." => md"Transition matrices are in general not symmetric; ``P_{12} \ne P_{21}`` is the usual case.";
	correct = 2, head = "Question 2.4")

# ╔═╡ 106a1fbd-14a1-56e8-bc76-9bbf9c742dd0
mcq(md"For ``a = 0.3`` and ``b = 0.7``, what kind of process is the chain?",
	md"An i.i.d. process: both rows of ``P`` equal ``(0.7, 0.3)``, so the next day does not depend on today." => md"Correct. For ``a + b = 1`` the two rows coincide. The process is still a Markov chain, the special case of Question 1.6.",
	md"A process with memory of the whole past." => md"The next day depends at most on the present day.";
	correct = 1, head = "Question 2.5")

# ╔═╡ e4a357c0-ca5c-5ee0-9acc-609a2a24c607
mcq(md"The bicycle is at the Campus and ``u = 0.80``. Where is it tomorrow?",
	md"Altmarkt" => md"In the row of the Campus, the piece of the Altmarkt is ``[0, 0.35)``.",
	md"Campus" => md"Correct. The piece of the Campus is ``[0.35, 0.85)``, which contains 0.80.",
	md"Neustadt" => md"The piece of Neustadt is ``[0.85, 1)``.";
	correct = 2, head = "Question 3.1")

# ╔═╡ 7b344dd1-555d-48b8-821d-171033e436aa
mcq(md"The same uniform number selects different stations in the two orders. Do the two orders simulate the same Markov chain?",
	md"No, because they produce different paths." => md"A path is a single random outcome. The percentages in Figure 3.3 show that each station is selected equally often in both orders.",
	md"Yes. A uniform number falls into a piece with probability equal to the length of the piece, ``P_{ij}``, wherever the piece lies." => md"Correct. The order only determines which uniform numbers lead to which station. Every order gives the same transition probabilities and hence the same Markov chain.",
	md"Only the order Altmarkt, Campus, Neustadt is correct." => md"Any fixed order can be used.";
	correct = 2, head = "Question 3.2")

# ╔═╡ 403db6b6-983d-5c1b-93ea-267cda5d8af0
mcq(md"In Figure 3.2, which part of the matrix determines the next state?",
	md"The row of the current state." => md"Correct. Only this row is used.",
	md"The column of the current state." => md"A column contains probabilities from different present states.",
	md"The whole matrix." => md"One row is sufficient for each step.";
	correct = 1, head = "Question 3.3")

# ╔═╡ c685eb82-0a7b-53a3-bbf0-e52cb3d828e2
mcq(md"A student simulates the next state with the following loop. What is wrong?

```julia
for j in 1:3
    if rand() < P[i, j]
        return j
    end
end
```",	correct = 2, head = "Question 3.4")

# ╔═╡ d0bd4d33-3397-541a-8d47-df03d93588b9
mcq(md"Most cells of the matrix in Figure 4.1 are empty. What does an empty cell imply for the text generated by the chain?",
	md"The chain never produces this pair of characters." => md"Correct. ``\hat P_{ij} = 0`` excludes the transition. Some of these pairs do not occur in English (``qz``); others merely do not occur in this book.",
	md"The pair is produced with a small probability." => md"An empty cell means ``\hat P_{ij} = 0``.",
	md"Nothing; the chain fills empty cells itself." => md"The chain uses only the estimated probabilities.";
	correct = 1, head = "Question 4.1")

# ╔═╡ c4ee1c46-0f46-5ebf-8d55-1e7e7d7a8410
reveal(md"""Not reliable. Three observations are too few to estimate a distribution over 27 characters; another text would give a different row. Moreover, ``\hat P_{ze} = 1`` excludes every other continuation, so the generated text never contains them. An estimate should therefore be reported together with the counts on which it is based.""")

# ╔═╡ e0ea0af5-dff7-550e-87d6-8dfb420feb0b
mcq(md"Why does the generated text contain so few real words?",
	md"The matrix is estimated incorrectly." => md"The matrix reproduces the frequencies of letter pairs in the book.",
	md"The state is a single character; the chain knows the last letter, but not the word it belongs to." => md"Correct. Each pair of consecutive letters is plausible, but nothing relates letters that are more than one step apart.",
	md"The uniform numbers are not random enough." => md"Better random numbers do not add memory.";
	correct = 2, head = "Question 4.3")

# ╔═╡ d5fb6239-c0aa-52a9-9b36-b0216f9cc453
numq(md"A bicycle is at Neustadt today. Compute the probability that it is at the Altmarkt in two days.", 0.31; rtol = 0.005,
	why = md"``(P^2)_{31} = 0.20 \cdot 0.60 + 0.20 \cdot 0.35 + 0.60 \cdot 0.20 = 0.12 + 0.07 + 0.12 = 0.31``.", head = "Question 5.1")

# ╔═╡ 4bf301e9-1340-57e5-b0a4-5dd90098f5be
mcq(md"Is ``(P^2)_{ij}`` equal to ``(P_{ij})^2``?",
	md"Yes; both square the transition probability." => md"``(P_{ij})^2`` is the square of a single number; it ignores all paths through other states.",
	md"No. ``(P^2)_{ij}`` sums over all intermediate states, whereas ``(P_{ij})^2`` is the square of a single entry." => md"Correct. For Neustadt → Altmarkt, ``(P^2)_{31} = 0.31``, but ``(P_{31})^2 = 0.04``.";
	correct = 2, head = "Question 5.2")

# ╔═╡ 1dad2f06-0327-5c17-a470-c23466178964
numq(md"A bicycle starts at the Campus. What is the probability of the **path** Campus → Campus → Altmarkt → Neustadt?", 0.02625; rtol = 0.005,
	why = md"By (4), the probabilities along the path are multiplied: ``P_{22} P_{21} P_{13} = 0.50 \cdot 0.35 \cdot 0.15 = 0.02625``. This is not ``(P^3)_{23}``, which sums over all paths of length 3 from the Campus to Neustadt.", head = "Question 5.3")

# ╔═╡ 6bec7e8e-7bf2-542a-877b-b0a2115dd87b
numq(md"On day 0 a bicycle is at each station with probability ``1/3``. What is the probability that it is at the Altmarkt on day 1?", (0.6 + 0.35 + 0.2) / 3; rtol = 0.005,
	why = md"By (9), the first entry of ``\pi_0 P`` is ``\pi_0`` multiplied by the **column** of the Altmarkt: ``\tfrac13 (0.60 + 0.35 + 0.20) \approx 0.383``.", head = "Question 5.4")

# ╔═╡ fa0107ec-66d4-5ad8-9040-27d786245879
mcq(md"In the chain with a closed set of two states, which set is closed?",
	md"``\{1\}``" => md"From state 1, arrows lead to states 2 and 3.",
	md"``\{2, 3\}``" => md"Correct. From states 2 and 3, arrows lead only to states 2 and 3. Once the chain leaves state 1, it never returns.",
	md"``\{1, 2\}``" => md"From state 2 the chain can move to state 3, which lies outside the set.";
	correct = 2, head = "Question 7.1")

# ╔═╡ 38f02ca2-6dd2-5bb6-8323-b2c165841643
mcq(md"A state has ``P_{ii} = 0.99``. Is it absorbing?",
	md"Yes, because the chain almost never leaves it." => md"An absorbing state is never left: ``P_{ii} = 1``.",
	md"No. The chain leaves it with probability 0.01 in every step." => md"Correct.";
	correct = 2, head = "Question 7.2")

# ╔═╡ 526c7cda-1e09-566d-b386-9c92a951727b
mcq(md"The chain with two separate parts is started in state 1. Which statement is correct?",
	md"It eventually visits all four states." => md"No arrow leads from ``\{1, 2\}`` to ``\{3, 4\}``.",
	md"It remains in ``\{1, 2\}`` forever. Both ``\{1, 2\}`` and ``\{3, 4\}`` are closed, so the chain is not irreducible." => md"Correct. The long-run behaviour depends on the initial state; compare the two lower panels of Figure 7.2.",
	md"State 1 is absorbing." => md"An absorbing state has ``P_{11} = 1``; here ``P_{11} = 0.7``.";
	correct = 2, head = "Question 7.3")

# ╔═╡ 55bfec8c-bb5c-5886-8c90-346673c23e24
mcq(md"Is the bicycle chain irreducible?",
	md"Yes. Every entry of ``P`` is positive, so every station can be reached from every station in one step." => md"Correct. The only closed set is ``S``.",
	md"No, because the bicycles return to the same stations." => md"Returning does not create a closed set; a closed set is a set that cannot be **left**.";
	correct = 1, head = "Question 7.4")

# ╔═╡ f86b8b72-25c2-5347-9e9b-ff277850ee9d
numq(md"**Problem 1.** A bicycle starts at the Campus. Compute the probability that it is at the Campus again after two days.", 0.3675; rtol = 0.005,
	why = md"``(P^2)_{22} = 0.35 \cdot 0.25 + 0.50 \cdot 0.50 + 0.15 \cdot 0.20 = 0.0875 + 0.25 + 0.03 = 0.3675``.", head = "Problem 1")

# ╔═╡ e06f8c76-580a-5403-be52-143a181045b4
numq(md"**Problem 2.** A chain has ``P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}``. Compute the first entry ``\pi_1`` of its stationary distribution.", 5 / 6; rtol = 0.005,
	why = md"``\pi P = \pi`` gives ``0.1\,\pi_1 = 0.5\,\pi_2``. Together with ``\pi_1 + \pi_2 = 1``: ``\pi_1 = 0.5 / 0.6 = 5/6 \approx 0.833``.", head = "Problem 2")

# ╔═╡ 1e5a2cf0-64a1-5e0d-9022-ee2566a71f13
numq(md"**Problem 3.** Same chain as in Problem 2. Compute ``(P^2)_{21}``.", 0.7; rtol = 0.005,
	why = md"``(P^2)_{21} = 0.5 \cdot 0.9 + 0.5 \cdot 0.5 = 0.45 + 0.25 = 0.7``.", head = "Problem 3")

# ╔═╡ 71d608b1-4677-5061-875b-6a676b8f28fc
mcq(md"**Problem 5.** A process takes the values 0 and 1. ``X_{n+1}`` equals ``X_{n-1}`` with probability 0.8 and the opposite value otherwise. Which statement is correct?",
	md"``X_n`` is a Markov chain." => md"The distribution of ``X_{n+1}`` depends on ``X_{n-1}``, also when ``X_n`` is known.",
	md"``X_n`` is not a Markov chain, but the pairs ``Y_n = (X_{n-1}, X_n)`` form a Markov chain with four states." => md"Correct. Enlarging the state turns a finite memory into a Markov chain (Question 1.8).",
	md"``X_n`` is i.i.d." => md"Consecutive values are dependent.";
	correct = 2, head = "Problem 5")

# ╔═╡ bc555a3a-7d39-5ea1-b11a-ab81035d3096
# Data: weather counts, three public-domain books and the bicycle model
begin
	# transition counts from all summer days (June–August) 1960–2025 at the same station; states 1 = Sun, 2 = Rain
	const WEATHER_P = [2993 1042; 1043 837] ./ [4035, 1880]
	const WEATHER_NAMES = ["Sun", "Rain"]

	# a model of one rental bicycle moving between three stations, one step per day
	const BIKE_P = [0.60 0.25 0.15; 0.35 0.50 0.15; 0.20 0.20 0.60]
	const BIKE_NAMES = ["Altmarkt", "Campus", "Neustadt"]


	# Jane Austen, Pride and Prejudice (1813), chapters 1–3, Project Gutenberg eBook 1342
	const AUSTEN = "It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters. \"My dear Mr. Bennet,\" said his lady to him one day, \"have you heard that Netherfield Park is let at last?\" Mr. Bennet replied that he had not. \"But it is,\" returned she; \"for Mrs. Long has just been here, and she told me all about it.\" Mr. Bennet made no answer. \"Do not you want to know who has taken it?\" cried his wife, impatiently. \"You want to tell me, and I have no objection to hearing it.\" [Illustration: \"He came down to see the place\" [Copyright 1894 by George Allen.]] This was invitation enough. \"Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week.\" \"What is his name?\" \"Bingley.\" \"Is he married or single?\" \"Oh, single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!\" \"How so? how can it affect them?\" \"My dear Mr. Bennet,\" replied his wife, \"how can you be so tiresome? You must know that I am thinking of his marrying one of them.\" \"Is that his design in settling here?\" \"Design? Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes.\" \"I see no occasion for that. You and the girls may go--or you may send them by themselves, which perhaps will be still better; for as you are as handsome as any of them, Mr. Bingley might like you the best of the party.\" \"My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty.\" \"In such cases, a woman has not often much beauty to think of.\" \"But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood.\" \"It is more than I engage for, I assure you.\" \"But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account; for in general, you know, they visit no new comers. Indeed you must go, for it will be impossible for us to visit him, if you do not.\" \"You are over scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls--though I must throw in a good word for my little Lizzy.\" \"I desire you will do no such thing. Lizzy is not a bit better than the others: and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference.\" \"They have none of them much to recommend them,\" replied he: \"they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters.\" \"Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion on my poor nerves.\" \"You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these twenty years at least.\" \"Ah, you do not know what I suffer.\" \"But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood.\" \"It will be no use to us, if twenty such should come, since you will not visit them.\" \"Depend upon it, my dear, that when there are twenty, I will visit them all.\" Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develope. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married: its solace was visiting and news. [Illustration: M^{r.} & M^{rs.} Bennet [Copyright 1894 by George Allen.]] [Illustration: \"I hope Mr. Bingley will like it\" [Copyright 1894 by George Allen.]] CHAPTER II. [Illustration] Mr. Bennet was among the earliest of those who waited on Mr. Bingley. He had always intended to visit him, though to the last always assuring his wife that he should not go; and till the evening after the visit was paid she had no knowledge of it. It was then disclosed in the following manner. Observing his second daughter employed in trimming a hat, he suddenly addressed her with,-- \"I hope Mr. Bingley will like it, Lizzy.\" \"We are not in a way to know what Mr. Bingley likes,\" said her mother, resentfully, \"since we are not to visit.\" \"But you forget, mamma,\" said Elizabeth, \"that we shall meet him at the assemblies, and that Mrs. Long has promised to introduce him.\" \"I do not believe Mrs. Long will do any such thing. She has two nieces of her own. She is a selfish, hypocritical woman, and I have no opinion of her.\" \"No more have I,\" said Mr. Bennet; \"and I am glad to find that you do not depend on her serving you.\" Mrs. Bennet deigned not to make any reply; but, unable to contain herself, began scolding one of her daughters. \"Don't keep coughing so, Kitty, for heaven's sake! Have a little compassion on my nerves. You tear them to pieces.\" \"Kitty has no discretion in her coughs,\" said her father; \"she times them ill.\" \"I do not cough for my own amusement,\" replied Kitty, fretfully. \"When is your next ball to be, Lizzy?\" \"To-morrow fortnight.\" \"Ay, so it is,\" cried her mother, \"and Mrs. Long does not come back till the day before; so, it will be impossible for her to introduce him, for she will not know him herself.\" \"Then, my dear, you may have the advantage of your friend, and introduce Mr. Bingley to her.\" \"Impossible, Mr. Bennet, impossible, when I am not acquainted with him myself; how can you be so teasing?\" \"I honour your circumspection. A fortnight's acquaintance is certainly very little. One cannot know what a man really is by the end of a fortnight. But if we do not venture, somebody else will; and after all, Mrs. Long and her nieces must stand their chance; and, therefore, as she will think it an act of kindness, if you decline the office, I will take it on myself.\" The girls stared at their father. Mrs. Bennet said only, \"Nonsense, nonsense!\" \"What can be the meaning of that emphatic exclamation?\" cried he. \"Do you consider the forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite agree with you there. What say you, Mary? For you are a young lady of deep reflection, I know, and read great books, and make extracts.\" Mary wished to say something very sensible, but knew not how. \"While Mary is adjusting her ideas,\" he continued, \"let us return to Mr. Bingley.\" \"I am sick of Mr. Bingley,\" cried his wife. \"I am sorry to hear that; but why did you not tell me so before? If I had known as much this morning, I certainly would not have called on him. It is very unlucky; but as I have actually paid the visit, we cannot escape the acquaintance now.\" The astonishment of the ladies was just what he wished--that of Mrs. Bennet perhaps surpassing the rest; though when the first tumult of joy was over, she began to declare that it was what she had expected all the while. \"How good it was in you, my dear Mr. Bennet! But I knew I should persuade you at last. I was sure you loved your girls too well to neglect such an acquaintance. Well, how pleased I am! And it is such a good joke, too, that you should have gone this morning, and never said a word about it till now.\" \"Now, Kitty, you may cough as much as you choose,\" said Mr. Bennet; and, as he spoke, he left the room, fatigued with the raptures of his wife. \"What an excellent father you have, girls,\" said she, when the door was shut. \"I do not know how you will ever make him amends for his kindness; or me either, for that matter. At our time of life, it is not so pleasant, I can tell you, to be making new acquaintances every day; but for your sakes we would do anything. Lydia, my love, though you are the youngest, I dare say Mr. Bingley will dance with you at the next ball.\" \"Oh,\" said Lydia, stoutly, \"I am not afraid; for though I am the youngest, I'm the tallest.\" The rest of the evening was spent in conjecturing how soon he would return Mr. Bennet's visit, and determining when they should ask him to dinner. [Illustration: \"I'm the tallest\"] [Illustration: \"He rode a black horse\" ] CHAPTER III. [Illustration] Not all that Mrs. Bennet, however, with the assistance of her five daughters, could ask on the subject, was sufficient to draw from her husband any satisfactory description of Mr. Bingley. They attacked him in various ways, with barefaced questions, ingenious suppositions, and distant surmises; but he eluded the skill of them all; and they were at last obliged to accept the second-hand intelligence of their neighbour, Lady Lucas. Her report was highly favourable. Sir William had been delighted with him. He was quite young, wonderfully handsome, extremely agreeable, and, to crown the whole, he meant to be at the next assembly with a large party. Nothing could be more delightful! To be fond of dancing was a certain step towards falling in love; and very lively hopes of Mr. Bingley's heart were entertained. \"If I can but see one of my daughters happily settled at Netherfield,\" said Mrs. Bennet to her husband, \"and all the others equally well married, I shall have nothing to wish for.\" In a few days Mr. Bingley returned Mr. Bennet's visit, and sat about ten minutes with him in his library. He had entertained hopes of being admitted to a sight of the young ladies, of whose beauty he had heard much; but he saw only the father. The ladies were somewhat more fortunate, for they had the advantage of ascertaining, from an upper window, that he wore a blue coat and rode a black horse. An invitation to dinner was soon afterwards despatched; and already had Mrs. Bennet planned the courses that were to do credit to her housekeeping, when an answer arrived which deferred it all. Mr. Bingley was obliged to be in town the following day, and consequently unable to accept the honour of their invitation, etc. Mrs. Bennet was quite disconcerted. She could not imagine what business he could have in town so soon after his arrival in Hertfordshire; and she began to fear that he might always be flying about from one place to another, and never settled at Netherfield as he ought to be. Lady Lucas quieted her fears a little by starting the idea of his [Illustration: \"When the Party entered\" [Copyright 1894 by George Allen.]] being gone to London only to get a large party for the ball; and a report soon followed that Mr. Bingley was to bring twelve ladies and seven gentlemen with him to the assembly. The girls grieved over such a number of ladies; but were comforted the day before the ball by hearing that, instead of twelve, he had brought only six with him from London, his five sisters and a cousin. And when the party entered the assembly-room, it consisted of only five altogether: Mr. Bingley, his two sisters, the husband of the eldest, and another young man. Mr. Bingley was good-looking and gentlemanlike: he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien, and the report, which was in general circulation within five minutes after his entrance, of his having ten thousand a year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration for about half the evening, till his manners gave a disgust which turned the tide of his popularity; for he was discovered to be proud, to be above his company, and above being pleased; and not all his large estate in Derbyshire could save him from having a most forbidding, disagreeable countenance, and being unworthy to be compared with his friend. Mr. Bingley had soon made himself acquainted with all the principal people in the room: he was lively and unreserved, danced every dance, was angry that the ball closed so early, and talked of giving one himself at Netherfield. Such amiable qualities must speak for themselves. What a contrast between him and his friend! Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley, declined being introduced to any other lady, and spent the rest of the evening in walking about the room, speaking occasionally to one of his own party. His character was decided. He was the proudest, most disagreeable man in the world, and everybody hoped that he would never come there again. Amongst the most violent against him was Mrs. Bennet, whose dislike of his general behaviour was sharpened into particular resentment by his having slighted one of her daughters. Elizabeth Bennet had been obliged, by the scarcity of gentlemen, to sit down for two dances; and during part of that time, Mr. Darcy had been standing near enough for her to overhear a conversation between him and Mr. Bingley, who came from the dance for a few minutes to press his friend to join it. \"Come, Darcy,\" said he, \"I must have you dance. I hate to see you standing about by yourself in this stupid manner. You had much better dance.\" \"I certainly shall not. You know how I detest it, unless I am particularly acquainted with my partner. At such an assembly as this, it would be insupportable. Your sisters are engaged, and there is not another woman in the room whom it would not be a punishment to me to stand up with.\" \"I would not be so fastidious as you are,\" cried Bingley, \"for a kingdom! Upon my honour, I never met with so many pleasant girls in my life as I have this evening; and there are several of them, you see, uncommonly pretty.\" \"You are dancing with the only handsome girl in the room,\" said Mr. Darcy, looking at the eldest Miss Bennet. \"Oh, she is the most beautiful creature I ever beheld! But there is one of her sisters sitting down just behind you, who is very pretty, and I dare say very agreeable. Do let me ask my partner to introduce you.\" [Illustration: \"She is tolerable\" [Copyright 1894 by George Allen.]] \"Which do you mean?\" and turning round, he looked for a moment at Elizabeth, till, catching her eye, he withdrew his own, and coldly said, \"She is tolerable: but not handsome enough to tempt me; and I am in no humour at present to give consequence to young ladies who are slighted by other men. You had better return to your partner and enjoy her smiles, for you are wasting your time with me.\" Mr. Bingley followed his advice. Mr. Darcy walked off; and Elizabeth remained with no very cordial feelings towards him. She told the story, however, with great spirit among her friends; for she had a lively, playful disposition, which delighted in anything ridiculous. The evening altogether passed off pleasantly to the whole family. Mrs. Bennet had seen her eldest daughter much admired by the Netherfield party. Mr. Bingley had danced with her twice, and she had been distinguished by his sisters. Jane was as much gratified by this as her mother could be, though in a quieter way. Elizabeth felt Jane's pleasure. Mary had heard herself mentioned to Miss Bingley as the most accomplished girl in the neighbourhood; and Catherine and Lydia had been fortunate enough to be never without partners, which was all that they had yet learnt to care for at a ball. They returned, therefore, in good spirits to Longbourn, the village where they lived, and of which they were the principal inhabitants. They found Mr. Bennet still up. With a book, he was regardless of time; and on the present occasion he had a good deal of curiosity as to the event of an evening which had raised such splendid expectations. He had rather hoped that all his wife's views on the stranger would be disappointed; but he soon found that he had a very different story to hear. \"Oh, my dear Mr. Bennet,\" as she entered the room, \"we have had a most delightful evening, a most excellent ball. I wish you had been there. Jane was so admired, nothing could be like it. Everybody said how well she looked; and Mr. Bingley thought her quite beautiful, and danced with her twice. Only think of that, my dear: he actually danced with her twice; and she was the only creature in the room that he asked a second time. First of all, he asked Miss Lucas. I was so vexed to see him stand up with her; but, however, he did not admire her at all; indeed, nobody can, you know; and he seemed quite struck with Jane as she was going down the dance. So he inquired who she was, and got introduced, and asked her for the two next. Then, the two third he danced with Miss King, and the two fourth with Maria Lucas, and the two fifth with Jane again, and the two sixth with Lizzy, and the Boulanger----\" \"If he had had any compassion for me,\" cried her husband impatiently, \"he would not have danced half so much! For God's sake, say no more of his partners. O that he had sprained his ancle in the first dance!\" \"Oh, my dear,\" continued Mrs. Bennet, \"I am quite delighted with him. He is so excessively handsome! and his sisters are charming women. I never in my life saw anything more elegant than their dresses. I dare say the lace upon Mrs. Hurst's gown----\" Here she was interrupted again. Mr. Bennet protested against any description of finery. She was therefore obliged to seek another branch of the subject, and related, with much bitterness of spirit, and some exaggeration, the shocking rudeness of Mr. Darcy. \"But I can assure you,\" she added, \"that Lizzy does not lose much by not suiting his fancy; for he is a most disagreeable, horrid man, not at all worth pleasing. So high and so conceited, that there was no enduring him! He walked here, and he walked there, fancying himself so very great! Not handsome enough to dance with! I wish you had been there, my dear, to have given him one of your set-downs. I quite detest the man.\" [Illustration] CHAPTER IV. [Illustration] When Jane and Elizabeth were alone, the former, who had been cautious in her praise of Mr. Bingley before, expressed to her sister how very much she admired him. \"He is just what a young man ought to be,\" said she, \"sensible, good-humoured, lively; and I never saw such happy manners! so much ease, with such perfect good breeding!\" \"He is also handsome,\" replied Elizabeth, \"which a young man ought likewise to be if he possibly can. His character is thereby complete.\" \"I was very much flattered by his asking me to dance a second time. I did not expect such a compliment.\" \"Did not you? I did for you. But that is one great difference between us. Compliments always take you by surprise, and me never. What could be more natural than his asking you again? He could not help seeing that you were about five times as pretty as every other woman in the room. No thanks to his gallantry for that. Well, he certainly is very agreeable, and I give you leave to like him. You have liked many a stupider person.\" \"Dear Lizzy!\" \"Oh, you are a great deal too apt, you know, to like people in general. You never see a fault in anybody. All the world are good and agreeable in your eyes. I never heard you speak ill of a human being in my life.\" \"I would wish not to be hasty in censuring anyone; but I always speak what I think.\" \"I know you do: and it is that which makes the wonder. With your good sense, to be so honestly blind to the follies and nonsense of others! Affectation of candour is common enough; one meets with it everywhere. But to be candid without ostentation or design,--to take the good of everybody's character and make it still better, and say nothing of the bad,--belongs to you alone. And so, you like this man's sisters, too, do you? Their manners are not equal to his.\" \"Certainly not, at first; but they are very pleasing women when you converse with them. Miss Bingley is to live with her brother, and keep his house; and I am much mistaken if we shall not find a very charming neighbour in her.\" Elizabeth listened in silence, but was not convinced: their behaviour at the assembly had not been calculated to please in general; and with more quickness of observation and less pliancy of temper than her sister, and with a judgment, too, unassailed by any attention to herself, she was very little disposed to approve them. They were, in fact, very fine ladies; not deficient in good-humour when they were pleased, nor in the power of being agreeable where they chose it; but proud and conceited. They were rather handsome; had been educated in one of the first private seminaries in town; had a fortune of twenty thousand pounds; were in the habit of spending more than they ought, and of associating with people of rank; and were, therefore, in every respect entitled to think well of themselves and meanly of others. They were of a respectable family in the north of England; a circumstance more deeply impressed on their memories than that their brother's fortune and their own had been acquired by trade. Mr. Bingley inherited property to the amount of nearly a hundred thousand pounds from his father, who had intended to purchase an estate, but did not live to do it. Mr. Bingley intended it likewise, and sometimes made choice of his county; but, as he was now provided with a good house and the liberty of a manor, it was doubtful to many of those who best knew the easiness of his temper, whether he might not spend the remainder of his days at Netherfield, and leave the next generation to purchase. His sisters were very anxious for his having an estate of his own; but though he was now established only as a tenant, Miss Bingley was by no means unwilling to preside at his table; nor was Mrs. Hurst, who had married a man of more fashion than fortune, less disposed to consider his house as her home when it suited her. Mr. Bingley had not been of age two years when he was tempted, by an accidental recommendation, to look at Netherfield House. He did look at it, and into it, for half an hour; was pleased with the situation and the principal rooms, satisfied with what the owner said in its praise, and took it immediately. Between him and Darcy there was a very steady friendship, in spite of a great opposition of character. Bingley was endeared to Darcy by the easiness, openness, and ductility of his temper, though no disposition could offer a greater contrast to his own, and though with his own he never appeared dissatisfied. On the strength of Darcy's regard, Bingley had the firmest reliance, and of his judgment the highest opinion. In understanding, Darcy was the superior. Bingley was by no means deficient; but Darcy was clever. He was at the same time haughty, reserved, and fastidious; and his manners, though well bred, were not inviting. In that respect his friend had greatly the advantage. Bingley was sure of being liked wherever he appeared; Darcy was continually giving offence. The manner in which they spoke of the Meryton assembly was sufficiently characteristic."

	# Source texts (Project Gutenberg): first chapters of Alice in Wonderland (eBook 11) and Grimms' Fairy Tales (eBook 2591)
	const ALICE = raw"""Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?" So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge. In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it. "Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!" (Which was very likely true.) Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think , " (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) " , yes, that's about the right distance , but then I wonder what Latitude or Longitude I've got to?" (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.) Presently she began again. "I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think , " (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) " , but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?" (and she tried to curtsey as she spoke , fancy curtseying as you're falling through the air! Do you think you could manage it?) "And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere." Down, down, down. There was nothing else to do, so Alice soon began talking again. "Dinah'll miss me very much to-night, I should think!" (Dinah was the cat.) "I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?" And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, "Do cats eat bats? Do cats eat bats?" and sometimes, "Do bats eat cats?" for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, "Now, Dinah, tell me the truth: did you ever eat a bat?" when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over. Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, "Oh my ears and whiskers, how late it's getting!" She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof. There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again. Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted! Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; "and even if my head would go through," thought poor Alice, "it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin." For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible. There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ("which certainly was not here before," said Alice,) and round the neck of the bottle was a paper label, with the words "DRINK ME," beautifully printed on it in large letters. It was all very well to say "Drink me," but the wise little Alice was not going to do that in a hurry. "No, I'll look first," she said, "and see whether it's marked 'poison' or not"; for she had read several nice little histories about children who had got burnt, and eaten up by wild beasts and other unpleasant things, all because they would not remember the simple rules their friends had taught them: such as, that a red-hot poker will burn you if you hold it too long; and that if you cut your finger very deeply with a knife, it usually bleeds; and she had never forgotten that, if you drink much from a bottle marked "poison," it is almost certain to disagree with you, sooner or later. However, this bottle was not marked "poison," so Alice ventured to taste it, and finding it very nice, (it had, in fact, a sort of mixed flavour of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast,) she very soon finished it off. * * * * * * * * * * * * * * * * * * * * "What a curious feeling!" said Alice; "I must be shutting up like a telescope." And so it was indeed: she was now only ten inches high, and her face brightened up at the thought that she was now the right size for going through the little door into that lovely garden. First, however, she waited for a few minutes to see if she was going to shrink any further: she felt a little nervous about this; "for it might end, you know," said Alice to herself, "in my going out altogether, like a candle. I wonder what I should be like then?" And she tried to fancy what the flame of a candle is like after the candle is blown out, for she could not remember ever having seen such a thing. After a while, finding that nothing more happened, she decided on going into the garden at once; but, alas for poor Alice! when she got to the door, she found she had forgotten the little golden key, and when she went back to the table for it, she found she could not possibly reach it: she could see it quite plainly through the glass, and she tried her best to climb up one of the legs of the table, but it was too slippery; and when she had tired herself out with trying, the poor little thing sat down and cried. "Come, there's no use in crying like that!" said Alice to herself, rather sharply; "I advise you to leave off this minute!" She generally gave herself very good advice, (though she very seldom followed it), and sometimes she scolded herself so severely as to bring tears into her eyes; and once she remembered trying to box her own ears for having cheated herself in a game of croquet she was playing against herself, for this curious child was very fond of pretending to be two people. "But it's no use now," thought poor Alice, "to pretend to be two people! Why, there's hardly enough of me left to make one respectable person!" Soon her eye fell on a little glass box that was lying under the table: she opened it, and found in it a very small cake, on which the words "EAT ME" were beautifully marked in currants. "Well, I'll eat it," said Alice, "and if it makes me grow larger, I can reach the key; and if it makes me grow smaller, I can creep under the door; so either way I'll get into the garden, and I don't care which happens!" She ate a little bit, and said anxiously to herself, "Which way? Which way?", holding her hand on the top of her head to feel which way it was growing, and she was quite surprised to find that she remained the same size: to be sure, this generally happens when one eats cake, but Alice had got so much into the way of expecting nothing but out-of-the-way things to happen, that it seemed quite dull and stupid for life to go on in the common way. So she set to work, and very soon finished off the cake. * * * * * * * * * * * * * * * * * * * * The Pool of Tears "Curiouser and curiouser!" cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English); "now I'm opening out like the largest telescope that ever was! Good-bye, feet!" (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far off). "Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best way you can; , but I must be kind to them," thought Alice, "or perhaps they won't walk the way I want to go! Let me see: I'll give them a new pair of boots every Christmas." And she went on planning to herself how she would manage it. "They must go by the carrier," she thought; "and how funny it'll seem, sending presents to one's own feet! And how odd the directions will look! Alice's Right Foot, Esq., Hearthrug, near the Fender, (with Alice's love). Oh dear, what nonsense I'm talking!" Just then her head struck against the roof of the hall: in fact she was now more than nine feet high, and she at once took up the little golden key and hurried off to the garden door. Poor Alice! It was as much as she could do, lying down on one side, to look through into the garden with one eye; but to get through was more hopeless than ever: she sat down and began to cry again. "You ought to be ashamed of yourself," said Alice, "a great girl like you," (she might well say this), "to go on crying in this way! Stop this moment, I tell you!" But she went on all the same, shedding gallons of tears, until there was a large pool all round her, about four inches deep and reaching half down the hall. After a time she heard a little pattering of feet in the distance, and she hastily dried her eyes to see what was coming. It was the White Rabbit returning, splendidly dressed, with a pair of white kid gloves in one hand and a large fan in the other: he came trotting along in a great hurry, muttering to himself as he came, "Oh! the Duchess, the Duchess! Oh! won't she be savage if I've kept her waiting!" Alice felt so desperate that she was ready to ask help of any one; so, when the Rabbit came near her, she began, in a low, timid voice, "If you please, sir , " The Rabbit started violently, dropped the white kid gloves and the fan, and skurried away into the darkness as hard as he could go. Alice took up the fan and gloves, and, as the hall was very hot, she kept fanning herself all the time she went on talking: "Dear, dear! How queer everything is to-day! And yesterday things went on just as usual. I wonder if I've been changed in the night? Let me think: was I the same when I got up this morning? I almost think I can remember feeling a little different. But if I'm not the same, the next question is, Who in the world am I? Ah, that's the great puzzle!" And she began thinking over all the children she knew that were of the same age as herself, to see if she could have been changed for any of them. "I'm sure I'm not Ada," she said, "for her hair goes in such long ringlets, and mine doesn't go in ringlets at all; and I'm sure I can't be Mabel, for I know all sorts of things, and she, oh! she knows such a very little! Besides, she's she, and I'm I, and , oh dear, how puzzling it all is! I'll try if I know all the things I used to know. Let me see: four times five is twelve, and four times six is thirteen, and four times seven is , oh dear! I shall never get to twenty at that rate! However, the Multiplication Table doesn't signify: let's try Geography. London is the capital of Paris, and Paris is the capital of Rome, and Rome , no, that's all wrong, I'm certain! I must have been changed for Mabel! I'll try and say 'How doth the little , '" and she crossed her hands on her lap as if she were saying lessons, and began to repeat it, but her voice sounded hoarse and strange, and the words did not come the same as they used to do: , "How doth the little crocodile Improve his shining tail, And pour the waters of the Nile On every golden scale! "How cheerfully he seems to grin, How neatly spread his claws, And welcome little fishes in With gently smiling jaws!" "I'm sure those are not the right words," said poor Alice, and her eyes filled with tears again as she went on, "I must be Mabel after all, and I shall have to go and live in that poky little house, and have next to no toys to play with, and oh! ever so many lessons to learn! No, I've made up my mind about it; if I'm Mabel, I'll stay down here! It'll be no use their putting their heads down and saying 'Come up again, dear!' I shall only look up and say 'Who am I then? Tell me that first, and then, if I like being that person, I'll come up: if not, I'll stay down here till I'm somebody else' , but, oh dear!" cried Alice, with a sudden burst of tears, "I do wish they would put their heads down! I am so very tired of being all alone here!" As she said this she looked down at her hands, and was surprised to see that she had put on one of the Rabbit's little white kid gloves while she was talking. "How can I have done that?" she thought. "I must be growing small again." She got up and went to the table to measure herself by it, and found that, as nearly as she could guess, she was now about two feet high, and was going on shrinking rapidly: she soon found out that the cause of this was the fan she was holding, and she dropped it hastily, just in time to avoid shrinking away altogether. "That was a narrow escape!" said Alice, a good deal frightened at the sudden change, but very glad to find herself still in existence; "and now for the garden!" and she ran with all speed back to the little door: but, alas! the little door was shut again, and the little golden key was lying on the glass table as before, "and things are worse than ever," thought the poor child, "for I never was so small as this before, never! And I declare it's too bad, that it is!" As she said these words her foot slipped, and in another moment, splash! she was up to her chin in salt water. Her first idea was that she had somehow fallen into the sea, "and in that case I can go back by railway," she said to herself. (Alice had been to the seaside once in her life, and had come to the general conclusion, that wherever you go to on the English coast you find a number of bathing machines in the sea, some children digging in the sand with wooden spades, then a row of lodging houses, and behind them a railway station.) However, she soon made out that she was in the pool of tears which she had wept when she was nine feet high. "I wish I hadn't cried so much!" said Alice, as she swam about, trying to find her way out. "I shall be punished for it now, I suppose, by being drowned in my own tears! That will be a queer thing, to be sure! However, everything is queer to-day." Just then she heard something splashing about in the pool a little way off, and she swam nearer to make out what it was: at first she thought it must be a walrus or hippopotamus, but then she remembered how small she was now, and she soon made out that it was only a mouse that had slipped in like herself. "Would it be of any use, now," thought Alice, "to speak to this mouse? Everything is so out-of-the-way down here, that I should think very likely it can talk: at any rate, there's no harm in trying." So she began: "O Mouse, do you know the way out of this pool? I am very tired of swimming about here, O Mouse!" (Alice thought this must be the right way of speaking to a mouse: she had never done such a thing before, but she remembered having seen in her brother's Latin Grammar, "A mouse , of a mouse , to a mouse , a mouse , O mouse!") The Mouse looked at her rather inquisitively, and seemed to her to wink with one of its little eyes, but it said nothing. "Perhaps it doesn't understand English," thought Alice; "I daresay it's a French mouse, come over with William the Conqueror." (For, with all her knowledge of history, Alice had no very clear notion how long ago anything had happened.) So she began again: "Où est ma chatte?" which was the first sentence in her French lesson-book. The Mouse gave a sudden leap out of the water, and seemed to quiver all over with fright. "Oh, I beg your pardon!" cried Alice hastily, afraid that she had hurt the poor animal's feelings. "I quite forgot you didn't like cats." "Not like cats!" cried the Mouse, in a shrill, passionate voice. "Would you like cats if you were me?" "Well, perhaps not," said Alice in a soothing tone: "don't be angry about it. And yet I wish I could show you our cat Dinah: I think you'd take a fancy to cats if you could only see her. She is such a dear quiet thing," Alice went on, half to herself, as she swam lazily about in the pool, "and she sits purring so nicely by the fire, licking her paws and washing her face , and she is such a nice soft thing to nurse , and she's such a capital one for catching mice , oh, I beg your pardon!" cried Alice again, for this time the Mouse was bristling all over, and she felt certain it must be really offended. "We won't talk about her any more if you'd rather not." "We indeed!" cried the Mouse, who was trembling down to the end of his tail. "As if I would talk on such a subject! Our family always hated cats: nasty, low, vulgar things! Don't let me hear the name again!" "I won't indeed!" said Alice, in a great hurry to change the subject of conversation. "Are you , are you fond , of , of dogs?" The Mouse did not answer, so Alice went on eagerly: "There is such a nice little dog near our house I should like to show you! A little bright-eyed terrier, you know, with oh, such long curly brown hair! And it'll fetch things when you throw them, and it'll sit up and beg for its dinner, and all sorts of things , I can't remember half of them , and it belongs to a farmer, you know, and he says it's so useful, it's worth a hundred pounds! He says it kills all the rats and , oh dear!" cried Alice in a sorrowful tone, "I'm afraid I've offended it again!" For the Mouse was swimming away from her as hard as it could go, and making quite a commotion in the pool as it went. So she called softly after it, "Mouse dear! Do come back again, and we won't talk about cats or dogs either, if you don't like them!" When the Mouse heard this, it turned round and swam slowly back to her: its face was quite pale (with passion, Alice thought), and it said in a low trembling voice, "Let us get to the shore, and then I'll tell you my history, and you'll understand why it is I hate cats and dogs." It was high time to go, for the pool was getting quite crowded with the birds and animals that had fallen into it: there were a Duck and a Dodo, a Lory and an Eaglet, and several other curious creatures. Alice led the way, and the whole party swam to the shore. A Caucus-Race and a Long Tale They were indeed a queer-looking party that assembled on the bank , the birds with draggled feathers, the animals with their fur clinging close to them, and all dripping wet, cross, and uncomfortable. The first question of course was, how to get dry again: they had a consultation about this, and after a few minutes it seemed quite natural to Alice to find herself talking familiarly with them, as if she had known them all her life. Indeed, she had quite a long argument with the Lory, who at last turned sulky, and would only say, "I am older than you, and must know better;" and this Alice would not allow without knowing how old it was, and, as the Lory positively refused to tell its age, there was no more to be said. At last the Mouse, who seemed to be a person of authority among them, called out, "Sit down, all of you, and listen to me! I'll soon make you dry enough!" They all sat down at once, in a large ring, with the Mouse in the middle. Alice kept her eyes anxiously fixed on it, for she felt sure she would catch a bad cold if she did not get dry very soon. "Ahem!" said the Mouse with an important air, "are you all ready? This is the driest thing I know. Silence all round, if you please! 'William the Conqueror, whose cause was favoured by the pope, was soon submitted to by the English, who wanted leaders, and had been of late much accustomed to usurpation and conquest. Edwin and Morcar, the earls of Mercia and Northumbria , '" "Ugh!" said the Lory, with a shiver. "I beg your pardon!" said the Mouse, frowning, but very politely: "Did you speak?" "Not I!" said the Lory hastily. "I thought you did," said the Mouse. " , I proceed. 'Edwin and Morcar, the earls of Mercia and Northumbria, declared for him: and even Stigand, the patriotic archbishop of Canterbury, found it advisable , '" "Found what?" said the Duck. "Found it," the Mouse replied rather crossly: "of course you know what 'it' means." "I know what 'it' means well enough, when I find a thing," said the Duck: "it's generally a frog or a worm. The question is, what did the archbishop find?" The Mouse did not notice this question, but hurriedly went on, "' , found it advisable to go with Edgar Atheling to meet William and offer him the crown. William's conduct at first was moderate. But the insolence of his Normans , ' How are you getting on now, my dear?" it continued, turning to Alice as it spoke. "As wet as ever," said Alice in a melancholy tone: "it doesn't seem to dry me at all." "In that case," said the Dodo solemnly, rising to its feet, "I move that the meeting adjourn, for the immediate adoption of more energetic remedies , " "Speak English!" said the Eaglet. "I don't know the meaning of half those long words, and, what's more, I don't believe you do either!" And the Eaglet bent down its head to hide a smile: some of the other birds tittered audibly. "What I was going to say," said the Dodo in an offended tone, "was, that the best thing to get us dry would be a Caucus-race." "What is a Caucus-race?" said Alice; not that she wanted much to know, but the Dodo had paused as if it thought that somebody ought to speak, and no one else seemed inclined to say anything. "Why," said the Dodo, "the best way to explain it is to do it." (And, as you might like to try the thing yourself, some winter day, I will tell you how the Dodo managed it.) First it marked out a race-course, in a sort of circle, ("the exact shape doesn't matter," it said,) and then all the party were placed along the course, here and there. There was no "One, two, three, and away," but they began running when they liked, and left off when they liked, so that it was not easy to know when the race was over. However, when they had been running half an hour or so, and were quite dry again, the Dodo suddenly called out "The race is over!" and they all crowded round it, panting, and asking, "But who has won?" This question the Dodo could not answer without a great deal of thought, and it sat for a long time with one finger pressed upon its forehead (the position in which you usually see Shakespeare, in the pictures of him), while the rest waited in silence. At last the Dodo said, "Everybody has won, and all must have prizes." "But who is to give the prizes?" quite a chorus of voices asked. "Why, she, of course," said the Dodo, pointing to Alice with one finger; and the whole party at once crowded round her, calling out in a confused way, "Prizes! Prizes!" Alice had no idea what to do, and in despair she put her hand in her pocket, and pulled out a box of comfits, (luckily the salt water had not got into it), and handed them round as prizes. There was exactly one a-piece, all round. "But she must have a prize herself, you know," said the Mouse. "Of course," the Dodo replied very gravely. "What else have you got in your pocket?" he went on, turning to Alice. "Only a thimble," said Alice sadly. "Hand it over here," said the Dodo. Then they all crowded round her once more, while the Dodo solemnly presented the thimble, saying "We beg your acceptance of this elegant thimble;" and, when it had finished this short speech, they all cheered. Alice thought the whole thing very absurd, but they all looked so grave that she did not dare to laugh; and, as she could not think of anything to say, she simply bowed, and took the thimble, looking as solemn as she could. The next thing was to eat the comfits: this caused some noise and confusion, as the large birds complained that they could not taste theirs, and the small ones choked and had to be patted on the back. However, it was over at last, and they sat down again in a ring, and begged the Mouse to tell them something more."""
	const GRIMM = raw"""Hard by a great forest dwelt a poor wood-cutter with his wife and his two children. The boy was called Hansel and the girl Gretel. He had little to bite and to break, and once when great dearth fell on the land, he could no longer procure even daily bread. Now when he thought over this by night in his bed, and tossed about in his anxiety, he groaned and said to his wife: 'What is to become of us? How are we to feed our poor children, when we no longer have anything even for ourselves?' 'I'll tell you what, husband,' answered the woman, 'early tomorrow morning we will take the children out into the forest to where it is the thickest; there we will light a fire for them, and give each of them one more piece of bread, and then we will go to our work and leave them alone. They will not find the way home again, and we shall be rid of them.' 'No, wife,' said the man, 'I will not do that; how can I bear to leave my children alone in the forest?--the wild animals would soon come and tear them to pieces.' 'O, you fool!' said she, 'then we must all four die of hunger, you may as well plane the planks for our coffins,' and she left him no peace until he consented. 'But I feel very sorry for the poor children, all the same,' said the man. The two children had also not been able to sleep for hunger, and had heard what their stepmother had said to their father. Gretel wept bitter tears, and said to Hansel: 'Now all is over with us.' 'Be quiet, Gretel,' said Hansel, 'do not distress yourself, I will soon find a way to help us.' And when the old folks had fallen asleep, he got up, put on his little coat, opened the door below, and crept outside. The moon shone brightly, and the white pebbles which lay in front of the house glittered like real silver pennies. Hansel stooped and stuffed the little pocket of his coat with as many as he could get in. Then he went back and said to Gretel: 'Be comforted, dear little sister, and sleep in peace, God will not forsake us,' and he lay down again in his bed. When day dawned, but before the sun had risen, the woman came and awoke the two children, saying: 'Get up, you sluggards! we are going into the forest to fetch wood.' She gave each a little piece of bread, and said: 'There is something for your dinner, but do not eat it up before then, for you will get nothing else.' Gretel took the bread under her apron, as Hansel had the pebbles in his pocket. Then they all set out together on the way to the forest. When they had walked a short time, Hansel stood still and peeped back at the house, and did so again and again. His father said: 'Hansel, what are you looking at there and staying behind for? Pay attention, and do not forget how to use your legs.' 'Ah, father,' said Hansel, 'I am looking at my little white cat, which is sitting up on the roof, and wants to say goodbye to me.' The wife said: 'Fool, that is not your little cat, that is the morning sun which is shining on the chimneys.' Hansel, however, had not been looking back at the cat, but had been constantly throwing one of the white pebble-stones out of his pocket on the road. When they had reached the middle of the forest, the father said: 'Now, children, pile up some wood, and I will light a fire that you may not be cold.' Hansel and Gretel gathered brushwood together, as high as a little hill. The brushwood was lighted, and when the flames were burning very high, the woman said: 'Now, children, lay yourselves down by the fire and rest, we will go into the forest and cut some wood. When we have done, we will come back and fetch you away.' Hansel and Gretel sat by the fire, and when noon came, each ate a little piece of bread, and as they heard the strokes of the wood-axe they believed that their father was near. It was not the axe, however, but a branch which he had fastened to a withered tree which the wind was blowing backwards and forwards. And as they had been sitting such a long time, their eyes closed with fatigue, and they fell fast asleep. When at last they awoke, it was already dark night. Gretel began to cry and said: 'How are we to get out of the forest now?' But Hansel comforted her and said: 'Just wait a little, until the moon has risen, and then we will soon find the way.' And when the full moon had risen, Hansel took his little sister by the hand, and followed the pebbles which shone like newly-coined silver pieces, and showed them the way. They walked the whole night long, and by break of day came once more to their father's house. They knocked at the door, and when the woman opened it and saw that it was Hansel and Gretel, she said: 'You naughty children, why have you slept so long in the forest?--we thought you were never coming back at all!' The father, however, rejoiced, for it had cut him to the heart to leave them behind alone. Not long afterwards, there was once more great dearth throughout the land, and the children heard their mother saying at night to their father: 'Everything is eaten again, we have one half loaf left, and that is the end. The children must go, we will take them farther into the wood, so that they will not find their way out again; there is no other means of saving ourselves!' The man's heart was heavy, and he thought: 'It would be better for you to share the last mouthful with your children.' The woman, however, would listen to nothing that he had to say, but scolded and reproached him. He who says A must say B, likewise, and as he had yielded the first time, he had to do so a second time also. The children, however, were still awake and had heard the conversation. When the old folks were asleep, Hansel again got up, and wanted to go out and pick up pebbles as he had done before, but the woman had locked the door, and Hansel could not get out. Nevertheless he comforted his little sister, and said: 'Do not cry, Gretel, go to sleep quietly, the good God will help us.' Early in the morning came the woman, and took the children out of their beds. Their piece of bread was given to them, but it was still smaller than the time before. On the way into the forest Hansel crumbled his in his pocket, and often stood still and threw a morsel on the ground. 'Hansel, why do you stop and look round?' said the father, 'go on.' 'I am looking back at my little pigeon which is sitting on the roof, and wants to say goodbye to me,' answered Hansel. 'Fool!' said the woman, 'that is not your little pigeon, that is the morning sun that is shining on the chimney.' Hansel, however little by little, threw all the crumbs on the path. The woman led the children still deeper into the forest, where they had never in their lives been before. Then a great fire was again made, and the mother said: 'Just sit there, you children, and when you are tired you may sleep a little; we are going into the forest to cut wood, and in the evening when we are done, we will come and fetch you away.' When it was noon, Gretel shared her piece of bread with Hansel, who had scattered his by the way. Then they fell asleep and evening passed, but no one came to the poor children. They did not awake until it was dark night, and Hansel comforted his little sister and said: 'Just wait, Gretel, until the moon rises, and then we shall see the crumbs of bread which I have strewn about, they will show us our way home again.' When the moon came they set out, but they found no crumbs, for the many thousands of birds which fly about in the woods and fields had picked them all up. Hansel said to Gretel: 'We shall soon find the way,' but they did not find it. They walked the whole night and all the next day too from morning till evening, but they did not get out of the forest, and were very hungry, for they had nothing to eat but two or three berries, which grew on the ground. And as they were so weary that their legs would carry them no longer, they lay down beneath a tree and fell asleep. It was now three mornings since they had left their father's house. They began to walk again, but they always came deeper into the forest, and if help did not come soon, they must die of hunger and weariness. When it was mid-day, they saw a beautiful snow-white bird sitting on a bough, which sang so delightfully that they stood still and listened to it. And when its song was over, it spread its wings and flew away before them, and they followed it until they reached a little house, on the roof of which it alighted; and when they approached the little house they saw that it was built of bread and covered with cakes, but that the windows were of clear sugar. 'We will set to work on that,' said Hansel, 'and have a good meal. I will eat a bit of the roof, and you Gretel, can eat some of the window, it will taste sweet.' Hansel reached up above, and broke off a little of the roof to try how it tasted, and Gretel leant against the window and nibbled at the panes. Then a soft voice cried from the parlour: 'Nibble, nibble, gnaw, Who is nibbling at my little house?' The children answered: 'The wind, the wind, The heaven-born wind,' and went on eating without disturbing themselves. Hansel, who liked the taste of the roof, tore down a great piece of it, and Gretel pushed out the whole of one round window-pane, sat down, and enjoyed herself with it. Suddenly the door opened, and a woman as old as the hills, who supported herself on crutches, came creeping out. Hansel and Gretel were so terribly frightened that they let fall what they had in their hands. The old woman, however, nodded her head, and said: 'Oh, you dear children, who has brought you here? do come in, and stay with me. No harm shall happen to you.' She took them both by the hand, and led them into her little house. Then good food was set before them, milk and pancakes, with sugar, apples, and nuts. Afterwards two pretty little beds were covered with clean white linen, and Hansel and Gretel lay down in them, and thought they were in heaven. The old woman had only pretended to be so kind; she was in reality a wicked witch, who lay in wait for children, and had only built the little house of bread in order to entice them there. When a child fell into her power, she killed it, cooked and ate it, and that was a feast day with her. Witches have red eyes, and cannot see far, but they have a keen scent like the beasts, and are aware when human beings draw near. When Hansel and Gretel came into her neighbourhood, she laughed with malice, and said mockingly: 'I have them, they shall not escape me again!' Early in the morning before the children were awake, she was already up, and when she saw both of them sleeping and looking so pretty, with their plump and rosy cheeks she muttered to herself: 'That will be a dainty mouthful!' Then she seized Hansel with her shrivelled hand, carried him into a little stable, and locked him in behind a grated door. Scream as he might, it would not help him. Then she went to Gretel, shook her till she awoke, and cried: 'Get up, lazy thing, fetch some water, and cook something good for your brother, he is in the stable outside, and is to be made fat. When he is fat, I will eat him.' Gretel began to weep bitterly, but it was all in vain, for she was forced to do what the wicked witch commanded. And now the best food was cooked for poor Hansel, but Gretel got nothing but crab-shells. Every morning the woman crept to the little stable, and cried: 'Hansel, stretch out your finger that I may feel if you will soon be fat.' Hansel, however, stretched out a little bone to her, and the old woman, who had dim eyes, could not see it, and thought it was Hansel's finger, and was astonished that there was no way of fattening him. When four weeks had gone by, and Hansel still remained thin, she was seized with impatience and would not wait any longer. 'Now, then, Gretel,' she cried to the girl, 'stir yourself, and bring some water. Let Hansel be fat or lean, tomorrow I will kill him, and cook him.' Ah, how the poor little sister did lament when she had to fetch the water, and how her tears did flow down her cheeks! 'Dear God, do help us,' she cried. 'If the wild beasts in the forest had but devoured us, we should at any rate have died together.' 'Just keep your noise to yourself,' said the old woman, 'it won't help you at all.' Early in the morning, Gretel had to go out and hang up the cauldron with the water, and light the fire. 'We will bake first,' said the old woman, 'I have already heated the oven, and kneaded the dough.' She pushed poor Gretel out to the oven, from which flames of fire were already darting. 'Creep in,' said the witch, 'and see if it is properly heated, so that we can put the bread in.' And once Gretel was inside, she intended to shut the oven and let her bake in it, and then she would eat her, too. But Gretel saw what she had in mind, and said: 'I do not know how I am to do it; how do I get in?' 'Silly goose,' said the old woman. 'The door is big enough; just look, I can get in myself!' and she crept up and thrust her head into the oven. Then Gretel gave her a push that drove her far into it, and shut the iron door, and fastened the bolt. Oh! then she began to howl quite horribly, but Gretel ran away and the godless witch was miserably burnt to death. Gretel, however, ran like lightning to Hansel, opened his little stable, and cried: 'Hansel, we are saved! The old witch is dead!' Then Hansel sprang like a bird from its cage when the door is opened. How they did rejoice and embrace each other, and dance about and kiss each other! And as they had no longer any need to fear her, they went into the witch's house, and in every corner there stood chests full of pearls and jewels. 'These are far better than pebbles!' said Hansel, and thrust into his pockets whatever could be got in, and Gretel said: 'I, too, will take something home with me,' and filled her pinafore full. 'But now we must be off,' said Hansel, 'that we may get out of the witch's forest.' When they had walked for two hours, they came to a great stretch of water. 'We cannot cross,' said Hansel, 'I see no foot-plank, and no bridge.' 'And there is also no ferry,' answered Gretel, 'but a white duck is swimming there: if I ask her, she will help us over.' Then she cried: 'Little duck, little duck, dost thou see, Hansel and Gretel are waiting for thee? There's never a plank, or bridge in sight, Take us across on thy back so white.' The duck came to them, and Hansel seated himself on its back, and told his sister to sit by him. 'No,' replied Gretel, 'that will be too heavy for the little duck; she shall take us across, one after the other.' The good little duck did so, and when they were once safely across and had walked for a short time, the forest seemed to be more and more familiar to them, and at length they saw from afar their father's house. Then they began to run, rushed into the parlour, and threw themselves round their father's neck. The man had not known one happy hour since he had left the children in the forest; the woman, however, was dead. Gretel emptied her pinafore until pearls and precious stones ran about the room, and Hansel threw one handful after another out of his pocket to add to them. Then all anxiety was at an end, and they lived together in perfect happiness. My tale is done, there runs a mouse; whosoever catches it, may make himself a big fur cap out of it. THE MOUSE, THE BIRD, Once upon a time, a mouse, a bird, and a sausage, entered into partnership and set up house together. For a long time all went well; they lived in great comfort, and prospered so far as to be able to add considerably to their stores. The bird's duty was to fly daily into the wood and bring in fuel; the mouse fetched the water, and the sausage saw to the cooking. When people are too well off they always begin to long for something new. And so it came to pass, that the bird, while out one day, met a fellow bird, to whom he boastfully expatiated on the excellence of his household arrangements. But the other bird sneered at him for being a poor simpleton, who did all the hard work, while the other two stayed at home and had a good time of it. For, when the mouse had made the fire and fetched in the water, she could retire into her little room and rest until it was time to set the table. The sausage had only to watch the pot to see that the food was properly cooked, and when it was near dinner-time, he just threw himself into the broth, or rolled in and out among the vegetables three or four times, and there they were, buttered, and salted, and ready to be served. Then, when the bird came home and had laid aside his burden, they sat down to table, and when they had finished their meal, they could sleep their fill till the following morning: and that was really a very delightful life. Influenced by those remarks, the bird next morning refused to bring in the wood, telling the others that he had been their servant long enough, and had been a fool into the bargain, and that it was now time to make a change, and to try some other way of arranging the work. Beg and pray as the mouse and the sausage might, it was of no use; the bird remained master of the situation, and the venture had to be made. They therefore drew lots, and it fell to the sausage to bring in the wood, to the mouse to cook, and to the bird to fetch the water. And now what happened? The sausage started in search of wood, the bird made the fire, and the mouse put on the pot, and then these two waited till the sausage returned with the fuel for the following day. But the sausage remained so long away, that they became uneasy, and the bird flew out to meet him. He had not flown far, however, when he came across a dog who, having met the sausage, had regarded him as his legitimate booty, and so seized and swallowed him. The bird complained to the dog of this bare-faced robbery, but nothing he said was of any avail, for the dog answered that he found false credentials on the sausage, and that was the reason his life had been forfeited. He picked up the wood, and flew sadly home, and told the mouse all he had seen and heard. They were both very unhappy, but agreed to make the best of things and to remain with one another. So now the bird set the table, and the mouse looked after the food and, wishing to prepare it in the same way as the sausage, by rolling in and out among the vegetables to salt and butter them, she jumped into the pot; but she stopped short long before she reached the bottom, having already parted not only with her skin and hair, but also with life. Presently the bird came in and wanted to serve up the dinner, but he could nowhere see the cook. In his alarm and flurry, he threw the wood here and there about the floor, called and searched, but no cook was to be found. Then some of the wood that had been carelessly thrown down, caught fire and began to blaze. The bird hastened to fetch some water, but his pail fell into the well, and he after it, and as he was unable to recover himself, he was drowned. Once upon a time there was a widow who had two daughters; one of them was beautiful and industrious, the other ugly and lazy. The mother, however, loved the ugly and lazy one best, because she was her own daughter, and so the other, who was only her stepdaughter, was made to do all the work of the house, and was quite the Cinderella of the family. Her stepmother sent her out every day to sit by the well in the high road, there to spin until she made her fingers bleed. Now it chanced one day that some blood fell on to the spindle, and as the girl stopped over the well to wash it off, the spindle suddenly sprang out of her hand and fell into the well. She ran home crying to tell of her misfortune, but her stepmother spoke harshly to her, and after giving her a violent scolding, said unkindly, 'As you have let the spindle fall into the well you may go yourself and fetch it out.' The girl went back to the well not knowing what to do, and at last in her distress she jumped into the water after the spindle. She remembered nothing more until she awoke and found herself in a beautiful meadow, full of sunshine, and with countless flowers blooming in every direction. She walked over the meadow, and presently she came upon a baker's oven full of bread, and the loaves cried out to her, 'Take us out, take us out, or alas! we shall be burnt to a cinder; we were baked through long ago.' So she took the bread-shovel and drew them all out. She went on a little farther, till she came to a tree full of apples. 'Shake me, shake me, I pray,' cried the tree; 'my apples, one and all, are ripe.' So she shook the tree, and the apples came falling down upon her like rain; but she continued shaking until there was not a single apple left upon it. Then she carefully gathered the apples together in a heap and walked on again. The next thing she came to was a little house, and there she saw an old woman looking out, with such large teeth, that she was terrified, and turned to run away. But the old woman called after her, 'What are you afraid of, dear child? Stay with me; if you will do the work of my house properly for me, I will make you very happy. You must be very careful, however, to make my bed in the right way, for I wish you always to shake it thoroughly, so that the feathers fly about; then they say, down there in the world, that it is snowing; for I am Mother Holle.' The old woman spoke so kindly, that the girl summoned up courage and agreed to enter into her service. She took care to do everything according to the old woman's bidding and every time she made the bed she shook it with all her might, so that the feathers flew about like so many snowflakes. The old woman was as good as her word: she never spoke angrily to her, and gave her roast and boiled meats every day. So she stayed on with Mother Holle for some time, and then she began to grow unhappy. She could not at first tell why she felt sad, but she became conscious at last of great longing to go home; then she knew she was homesick, although she was a thousand times better off with Mother Holle than with her mother and sister. After waiting awhile, she went to Mother Holle and said, 'I am so homesick, that I cannot stay with you any longer, for although I am so happy here, I must return to my own people.' Then Mother Holle said, 'I am pleased that you should want to go back to your own people, and as you have served me so well and faithfully, I will take you home myself.' Thereupon she led the girl by the hand up to a broad gateway. The gate was opened, and as the girl passed through, a shower of gold fell upon her, and the gold clung to her, so that she was covered with it from head to foot. 'That is a reward for your industry,' said Mother Holle, and as she spoke she handed her the spindle which she had dropped into the well. The gate was then closed, and the girl found herself back in the old world close to her mother's house. As she entered the courtyard, the cock who was perched on the well, called out: 'Cock-a-doodle-doo! Your golden daughter's come back to you.' Then she went in to her mother and sister, and as she was so richly covered with gold, they gave her a warm welcome. She related to them all that had happened, and when the mother heard how she had come by her great riches, she thought she should like her ugly, lazy daughter to go and try her fortune. So she made the sister go and sit by the well and spin, and the girl pricked her finger and thrust her hand into a thorn-bush, so that she might drop some blood on to the spindle; then she threw it into the well, and jumped in herself. Like her sister she awoke in the beautiful meadow, and walked over it till she came to the oven. 'Take us out, take us out, or alas! we shall be burnt to a cinder; we were baked through long ago,' cried the loaves as before. But the lazy girl answered, 'Do you think I am going to dirty my hands for you?' and walked on. Presently she came to the apple-tree. 'Shake me, shake me, I pray; my apples, one and all, are ripe,' it cried. But she only answered, 'A nice thing to ask me to do, one of the apples might fall on my head,' and passed on. At last she came to Mother Holle's house, and as she had heard all about the large teeth from her sister, she was not afraid of them, and engaged herself without delay to the old woman. The first day she was very obedient and industrious, and exerted herself to please Mother Holle, for she thought of the gold she should get in return. The next day, however, she began to dawdle over her work, and the third day she was more idle still; then she began to lie in bed in the mornings and refused to get up. Worse still, she neglected to make the old woman's bed properly, and forgot to shake it so that the feathers might fly about. So Mother Holle very soon got tired of her, and told her she might go. The lazy girl was delighted at this, and thought to herself, 'The gold will soon be mine.' Mother Holle led her, as she had led her sister, to the broad gateway; but as she was passing through, instead of the shower of gold, a great bucketful of pitch came pouring over her. 'That is in return for your services,' said the old woman, and she shut the gate. So the lazy girl had to go home covered with pitch, and the cock on the well called out as she saw her: 'Cock-a-doodle-doo! Your dirty daughter's come back to you.' But, try what she would, she could not get the pitch off and it stuck to her as long as she lived. Once upon a time there was a dear little girl who was loved by everyone who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child. Once she gave her a little cap of red velvet, which suited her so well that she would never wear anything else; so she was always called 'Little Red-Cap.' One day her mother said to her: 'Come, Little Red-Cap, here is a piece of cake and a bottle of wine; take them to your grandmother, she is ill and weak, and they will do her good. Set out before it gets hot, and when you are going, walk nicely and quietly and do not run off the path, or you may fall and break the bottle, and then your grandmother will get nothing; and when you go into her room, don't forget to say, "Good morning", and don't peep into every corner before you do it.' 'I will take great care,' said Little Red-Cap to her mother, and gave her hand on it. The grandmother lived out in the wood, half a league from the village, and just as Little Red-Cap entered the wood, a wolf met her. Red-Cap did not know what a wicked creature he was, and was not at all afraid of him. 'Good day, Little Red-Cap,' said he. 'Thank you kindly, wolf.' 'Whither away so early, Little Red-Cap?' 'To my grandmother's.' 'What have you got in your apron?' 'Cake and wine; yesterday was baking-day, so poor sick grandmother is to have something good, to make her stronger.' 'Where does your grandmother live, Little Red-Cap?' 'A good quarter of a league farther on in the wood; her house stands under the three large oak-trees, the nut-trees are just below; you surely must know it,' replied Little Red-Cap. The wolf thought to himself: 'What a tender young creature! what a nice plump mouthful--she will be better to eat than the old woman. I must act craftily, so as to catch both.' So he walked for a short time by the side of Little Red-Cap, and then he said: 'See, Little Red-Cap, how pretty the flowers are about here--why do you not look round? I believe, too, that you do not hear how sweetly the little birds are singing; you walk gravely along as if you were going to school, while everything else out here in the wood is merry.' Little Red-Cap raised her eyes, and when she saw the sunbeams dancing here and there through the trees, and pretty flowers growing everywhere, she thought: 'Suppose I take grandmother a fresh nosegay; that would please her too."""
end;

# ╔═╡ fb9f5687-b340-51f4-91f4-f0e87ce9482c
# Markov chains: computations, state diagrams, the transition matrix, and an animated walker
begin
	const SCOL = [COL.ochre, COL.blue, COL.green, COL.purple, "#EE6677", "#66CCEE", "#BBBBBB"]
	scol(i) = SCOL[mod1(i, length(SCOL))]
	eye(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

	# one step by inversion: the row is cut into pieces of [0, 1), u picks a piece
	next_state(P, i, u) = something(findfirst(>(u), cumsum(P[i, :])), size(P, 1))
	simulate(P, i0, n, rng) = accumulate((i, u) -> next_state(P, i, u), rand(rng, n); init = i0) |> x -> [i0; x]

	# stationary distribution: solve π(P − I) = 0 together with Σπ = 1
	stationary(P) = (n = size(P, 1); [P' - eye(n); ones(1, n)] \ [zeros(n); 1.0])

	# communicating classes, closed classes and their periods
	function classify(P)
		n = size(P, 1); R = (P .> 0) .| (eye(n) .> 0)
		for k in 1:n, i in 1:n, j in 1:n                                  # reachability (Warshall)
			R[i, j] |= R[i, k] & R[k, j]
		end
		classes = unique([findall(j -> R[i, j] && R[j, i], 1:n) for i in 1:n])
		closed = [all(P[i, j] == 0 for i in c for j in setdiff(1:n, c)) for c in classes]
		function period(c)                                                  # gcd of (level[u] + 1 − level[v]) over edges in the class
			lev = Dict(c[1] => 0); queue = [c[1]]; g = 0
			while !isempty(queue)
				u = popfirst!(queue)
				for v in c
					P[u, v] > 0 || continue
					haskey(lev, v) ? (g = gcd(g, lev[u] + 1 - lev[v])) : (lev[v] = lev[u] + 1; push!(queue, v))
				end
			end
			g
		end
		per = period.(classes)
		(classes = classes, closed = closed, period = per, absorbing = [i for i in 1:n if P[i, i] == 1],
			irreducible = length(classes) == 1, aperiodic = all(==(1), per), ergodic = length(classes) == 1 && per[1] == 1)
	end

	circle_pos(n; r = 0.31) = [(0.5 + r * sin(2π * (k - 1) / n), 0.5 + r * cos(2π * (k - 1) / n)) for k in 1:n]

	function edge_curve(f, pos, i, j, R)
		p = [X(f, pos[i][1]), Y(f, pos[i][2])]
		if i == j
			cx = sum(X(f, q[1]) for q in pos) / length(pos); cy = sum(Y(f, q[2]) for q in pos) / length(pos)
			φ = hypot(p[1] - cx, p[2] - cy) < 1 ? -π / 2 : atan(p[2] - cy, p[1] - cx)
			d(a, s) = p .+ s .* [cos(a), sin(a)]
			return [d(φ - 0.5, R), d(φ - 0.8, 2.4R), d(φ + 0.8, 2.4R), d(φ + 0.5, R + 3)]
		end
		q = [X(f, pos[j][1]), Y(f, pos[j][2])]; L = hypot((q - p)...); u = (q - p) / L; nrm = [u[2], -u[1]]
		c = (p + q) / 2 + 0.16L * nrm
		s = p + R * (c - p) / hypot((c - p)...); e = q + (R + 3) * (c - q) / hypot((c - q)...)
		[s, s + 2 / 3 * (c - s), e + 2 / 3 * (c - e), e]
	end
	bez(b, t) = (1 - t)^3 * b[1] + 3(1 - t)^2 * t * b[2] + 3(1 - t) * t^2 * b[3] + t^3 * b[4]

	# The state diagram: one circle per state, one arrow per positive transition probability.
	# hl = (:row, i) or (:col, j) highlights the arrows of one row or one column; width = :flow scales arrows by π_i P_ij.
	function diagram(P, names; pos = circle_pos(size(P, 1)), w = 380, h = 300, R = 27, hl = nothing, width = :prob, labels = true, title = "")
		n = size(P, 1); f = fig(; w, h, axes = false, l = 8, r = 8, t = title == "" ? 22 : 34, b = 8, title)
		π = width == :flow ? stationary(P) : ones(n)
		raw!(f, "<defs><marker id=\"ah$(f.id)\" viewBox=\"0 0 10 10\" refX=\"8\" refY=\"5\" markerUnits=\"userSpaceOnUse\" markerWidth=\"9\" markerHeight=\"9\" orient=\"auto-start-reverse\"><path d=\"M0,0L10,5L0,10z\" fill=\"#555\"/></marker></defs>"; over = true)
		for i in 1:n, j in 1:n
			P[i, j] > 0 && !(width == :flow && i == j) || continue   # self-loops carry no flow between states
			b = edge_curve(f, pos, i, j, R)
			on = hl === nothing || (hl[1] == :row ? hl[2] == i : hl[2] == j)
			wd = width == :flow ? 0.8 + 40π[i] * P[i, j] : 0.9 + 3.2P[i, j]
			raw!(f, "<path d=\"M$(_n(b[1][1])),$(_n(b[1][2]))C$(join((_n(q[1]) * "," * _n(q[2]) for q in b[2:4]), " "))\" fill=\"none\" stroke=\"$(on ? "#555" : "#ddd")\" stroke-width=\"$(_n(wd))\" marker-end=\"url(#ah$(f.id))\"/>"; over = true)
			if labels && on
				m = bez(b, 0.5); v = width == :flow ? π[i] * P[i, j] : P[i, j]
				raw!(f, "<text x=\"$(_n(m[1]))\" y=\"$(_n(m[2] + 4))\" font-size=\"11.5\" text-anchor=\"middle\" paint-order=\"stroke\" stroke=\"white\" stroke-width=\"4\" fill=\"#333\">$(fmt(v; d = 2))</text>"; over = true)
			end
		end
		for i in 1:n
			x, y = X(f, pos[i][1]), Y(f, pos[i][2])
			raw!(f, "<circle cx=\"$(_n(x))\" cy=\"$(_n(y))\" r=\"$(R)\" fill=\"$(scol(i))\" fill-opacity=\"0.22\" stroke=\"$(scol(i))\" stroke-width=\"2\"/>" *
				"<text x=\"$(_n(x))\" y=\"$(_n(y + 4))\" font-size=\"$(length(names[i]) > 6 ? 10.5 : 12.5)\" font-weight=\"600\" text-anchor=\"middle\">$(xesc(names[i]))</text>"; over = true)
		end
		f
	end

	# The transition matrix as a table; hl = (:row, i), (:col, j) or (:cell, i, j)
	function matrix_fig(P, names; hl = nothing, digits = 2, w = 70 + 64size(P, 1), title = "")
		n = size(P, 1); f = fig(; w, h = 46 + 30n, xr = (0, n + 1.2), yr = (0, n + 1), axes = false, l = 4, r = 4, t = title == "" ? 4 : 24, b = 4, title)
		abbr(s) = length(s) > 5 ? s[1:4] * "." : s
		if hl !== nothing
			hl[1] == :row && rect!(f, 1.15, n - hl[2] + 0.05, n + 1.15, n - hl[2] + 0.95; fill = scol(hl[2]), op = 0.18)
			hl[1] == :col && rect!(f, hl[2] + 0.15, 0.05, hl[2] + 1.15, n - 0.05 + 0.0; fill = scol(hl[2]), op = 0.18)
		end
		text!(f, 0.6, n + 0.5, "from \\ to"; anchor = "middle", size = 10, c = "#777")
		for j in 1:n; text!(f, j + 0.65, n + 0.5, abbr(names[j]); anchor = "middle", size = 11, bold = true, c = scol(j)); end
		for i in 1:n
			text!(f, 0.6, n - i + 0.5, abbr(names[i]); anchor = "middle", size = 11, bold = true, c = scol(i))
			for j in 1:n; text!(f, j + 0.65, n - i + 0.5, string(round(P[i, j]; digits)); anchor = "middle", size = 12, c = P[i, j] == 0 ? "#bbb" : COL.ink); end
		end
		raw!(f, "<path d=\"M$(_n(X(f, 1.15))),$(_n(Y(f, 0)))V$(_n(Y(f, n)))M$(_n(X(f, n + 1.15))),$(_n(Y(f, 0)))V$(_n(Y(f, n)))\" stroke=\"#333\" fill=\"none\"/>"; over = true)
		f
	end

	function chain_anim(P, names, path; pos = circle_pos(size(P, 1)), w = 380, h = 300, F = 24, matrix = true, caption = nothing)
		n = size(P, 1); L = length(path) - 1
		D = diagram(P, names; pos, w, h)
		edges = Dict("$(i)-$(j)" => vcat(edge_curve(D, pos, i, j, 27)...) for i in 1:n, j in 1:n if P[i, j] > 0)
		S = fig(w = 680, h = 70, xr = (0, max(L + 1, 30)), yr = (0, 1), yticks = [], xlabel = "step n", l = 20, b = 34, t = 6)
		M = matrix_fig(P, names)
		data = (path = path .- 1, L = L, nodes = [[X(D, p[1]), Y(D, p[2])] for p in pos], cols = scol.(1:n), n = n, F = F,
			keys = collect(keys(edges)), curves = collect(values(edges)), names = names)
		code = """
		const E = {}; D.keys.forEach((k, i) => E[k] = D.curves[i]);
		const bz = (b, t) => [0, 1].map(c => (1 - t) ** 3 * b[c] + 3 * (1 - t) ** 2 * t * b[2 + c] + 3 * (1 - t) * t * t * b[4 + c] + t ** 3 * b[6 + c]);
		function draw(k) {
			const m = Math.min(D.L, Math.floor(k / D.F)), t = k / D.F - m, i = D.path[m], c = P[0].ctx;
			let xy = [D.nodes[i][0], D.nodes[i][1] + 13];          // at rest: below the name of the state
			if (m < D.L && t > 0.3) {                         // moving along the edge i → j
				const b = E[(i + 1) + "-" + (D.path[m + 1] + 1)]; xy = bz(b, ease((t - 0.3) / 0.7));
				c.strokeStyle = "$(COL.red)"; c.lineWidth = 3; c.globalAlpha = 0.55; c.beginPath(); c.moveTo(b[0], b[1]); c.bezierCurveTo(b[2], b[3], b[4], b[5], b[6], b[7]); c.stroke(); c.globalAlpha = 1;
			}
			c.fillStyle = "$(COL.ink)"; c.strokeStyle = "white"; c.lineWidth = 2; c.beginPath(); c.arc(xy[0], xy[1], 7, 0, 6.2832); c.fill(); c.stroke();
			for (let s = 0; s <= m; s++) box(P[1], s + 0.06, 0.1, s + 0.94, 0.9, D.cols[D.path[s]], 0.85);
			if (P.length > 2) {                                // the row of the current state
				const r = P[2]; box(r, 1.15, D.n - i - 0.95, D.n + 1.15, D.n - i - 0.05, D.cols[i], 0.25);
				if (m < D.L && t > 0.3) box(r, D.path[m + 1] + 1.15, D.n - i - 0.95, D.path[m + 1] + 2.15, D.n - i - 0.05, "$(COL.red)", 0.3);
			}
		}
		info = k => { const m = Math.min(D.L, Math.floor(k / D.F)); return "step " + m + ": state <b>" + D.names[D.path[m]] + "</b>" +
			(m < D.L ? " → next state drawn from row <b>" + D.names[D.path[m]] + "</b>" : ""); };
		"""
		animate(matrix ? [D, S, M] : [D, S], data, code; nframes = L * F + F, fps = 30, start = 0, caption)
	end

	# a sequence of states as a strip of coloured squares
	function strip_fig(states, cols; w = 680, label = "", letters = nothing, l = 92)
		L = length(states); f = fig(; w, h = 52, xr = (0, L), yr = (0, 1), yticks = [], xticks = [], l, b = 6, t = 6, r = 6)
		foreach(k -> rect!(f, k - 0.94, 0.1, k - 0.06, 0.9; fill = cols[states[k]], op = 0.85), 1:L)
		text!(f, 0, 0.5, label; anchor = "end", dx = -8, size = 12, bold = true)
		f
	end
end;

# ╔═╡ c27ca38d-d585-5585-aad9-49a9e817e79b
simulate(BIKE_P, 3, 10, Xoshiro(1))

# ╔═╡ 112744d4-9679-5a61-bdb4-4466114d8658
# Sections 1–5: kinds of memory, the bicycle chain, simulation by inversion, n-step probabilities
begin
	const WPOS = [(0.25, 0.5), (0.75, 0.5)]

	# Figures 1.1–1.3: what each kind of process uses to predict the next outcome
	const CASE = Dict(
		:iid => (names = ("H", "T"), cols = (COL.ochre, COL.blue), what = "head", fig = "1.1",
			uses = n -> "depends on no earlier outcome: P(head) = 0.5"),
		:cards => (names = ("R", "B"), cols = (COL.red, COL.ink), what = "red", fig = "1.2",
			uses = n -> "depends on all $(n) cards drawn so far"),
		:markov => (names = ("S", "R"), cols = (COL.ochre, COL.blue), what = "rain", fig = "1.3",
			uses = n -> "depends on the present day only"))
	function fig_case(kind, seed; N = 20)
		rng = Xoshiro(13 + seed); c = CASE[kind]
		x, p = if kind == :iid
			x = [rand(rng) < 0.5 ? 1 : 2 for _ in 1:N]; (x, fill(0.5, N - 1))
		elseif kind == :cards
			x = shuffle(rng, [fill(1, 10); fill(2, 10)]); (x, [(10 - count(==(1), x[1:n])) / (20 - n) for n in 1:N-1])
		else
			x = simulate(WEATHER_P, 1, N - 1, rng); (x, [WEATHER_P[x[n], 2] for n in 1:N-1])
		end
		f = fig(w = 680, h = 120, xr = (0.3, N + 0.7), yr = (0, 2.2), axes = false, l = 8, r = 8, t = 6, b = 4)
		for k in 1:N
			rect!(f, k - 0.42, 1.2, k + 0.42, 2.0; fill = c.cols[x[k]], op = 0.85, s = k)
			text!(f, k, 1.6, c.names[x[k]]; anchor = "middle", size = 12, bold = true, c = "white", s = k)
		end
		for n in 1:N-1
			rect!(f, n + 0.58, 1.2, n + 1.42, 2.0; fill = "none", op = 1, stroke = COL.orange, sw = 2.5, over = true, o = n)
			text!(f, n + 1, 1.6, "?"; anchor = "middle", size = 15, bold = true, c = COL.orange, o = n)
			kind == :cards && rect!(f, 0.52, 1.1, n + 0.48, 2.1; fill = "none", op = 1, stroke = "#999", sw = 2, over = true, o = n)
			kind == :markov && rect!(f, n - 0.48, 1.1, n + 0.48, 2.1; fill = "none", op = 1, stroke = COL.blue, sw = 2.5, over = true, o = n)
			text!(f, 0.5, 0.55, "next = outcome $(n + 1): " * c.uses(n); size = 12.5, o = n)
		end
		g = fig(w = 680, h = 130, xr = (0.3, N + 0.7), yr = (0, 1), yticks = [0, 0.5, 1], xticks = 2:2:N, xlabel = "outcome number", ylabel = "P(next = $(c.what))", l = 70, t = 8)
		for n in 1:N-1; dots!(g, [n + 1], [p[n]]; c = COL.orange, r = 4, s = n); n > 1 && seg!(g, n, p[n-1], n + 1, p[n]; c = COL.orange, w = 1.5, s = n); end
		reads = ["After outcome $(n): P(next = $(c.what)) = <b>$(fmt(p[n]; d = 3))</b>" for n in 1:N-1]
		player(f, g; n = N - 1, start = 1, fps = 1.5, reads = ["", reads...], label = "outcome",
			caption = Markdown.parse("**Figure $(c.fig).** Top: the outcomes in order. Orange: the outcome to be predicted; framed: the part of the past on which its distribution depends. Bottom: the probability of $(c.what) for the next outcome."))
	end

	# Figure 1.4: the weather chain in motion
	fig_weather(start, seed) = chain_anim(WEATHER_P, WEATHER_NAMES, simulate(WEATHER_P, start, 30, Xoshiro(7 + seed)); pos = WPOS, h = 180,
		caption = md"**Figure 1.4.** The weather chain. The black dot marks the current state; in each step the next state is drawn from the row of the current state.")

	# Figure 2.2: the two-state chain for any a and b
	function fig_twostate(a, b, start, seed)
		P = [1-a a; b 1-b]; x = simulate(P, start, 179, Xoshiro(21 + seed))
		vstack(figure(diagram(P, WEATHER_NAMES; pos = WPOS, w = 300, h = 170)),
			strip_fig(x, SCOL; label = "180 days");
			caption = md"**Figure 2.2.** A two-state chain with ``P(\text{sun} \to \text{rain}) = a`` and ``P(\text{rain} \to \text{sun}) = b``.")
	end

	# Figure 2.1: one row or one column of the bicycle matrix
	function fig_bike(k, mode)
		hl = (startswith(mode, "a row") ? :row : :col, k)
		s = hl[1] == :row ? sum(BIKE_P[k, :]) : sum(BIKE_P[:, k])
		figure(diagram(BIKE_P, BIKE_NAMES; hl), matrix_fig(BIKE_P, BIKE_NAMES; hl),
			Markdown.parse(hl[1] == :row ? "Row **$(BIKE_NAMES[k])** sums to **$(fmt(s))**: a bicycle at $(BIKE_NAMES[k]) today is at one of the three stations tomorrow." :
				"");
			caption = md"**Figure 2.1.** The bicycle chain as a state diagram and as a transition matrix. Rows correspond to the present station, columns to the next station.")
	end

	# Figure 2.3: a matrix typed by the reader
	function parse_matrix(s)
		rows = filter(!isempty, strip.(split(s, r"[;\n]")))
		M = [parse.(Float64, split(replace(r, "," => " "))) for r in rows]
		allequal(length.(M)) && length(M) == length(M[1]) || error("the matrix must be square")
		permutedims(hcat(M...))
	end
	function fig_playground(s, start, seed)
		P = try parse_matrix(s) catch e
			return Markdown.parse("**This is not a square matrix of numbers.** Write one row per line or separate rows with `;`.")
		end
		n = size(P, 1); bad = [i for i in 1:n if any(<(0), P[i, :]) || abs(sum(P[i, :]) - 1) > 1e-9]
		if !isempty(bad)
			return Markdown.parse("**Not a transition matrix.** " * join(["row $(i) sums to $(fmt(sum(P[i, :]); d = 4))" * (any(<(0), P[i, :]) ? " and has a negative entry" : "") for i in bad], "; ") * ". Every row must be a probability distribution.")
		end
		start > n && return Markdown.parse("**Your chain has only $(n) states;** choose a start between 1 and $(n).")
		chain_anim(P, string.(1:n), simulate(P, start, 40, Xoshiro(5 + seed)); F = 18, caption = Markdown.parse("**Figure 2.3.** The entered chain, started in state $(start)."))
	end

	# Figure 3.1: the row of state i as pieces of [0, 1), and the piece hit by u
	function fig_row(i, u)
		f = fig(w = 680, h = 150, xr = (0, 1), yr = (0, 1), yticks = [], xlabel = "u", title = "row of $(BIKE_NAMES[i])", l = 20)
		c = [0; cumsum(BIKE_P[i, :])]; j = next_state(BIKE_P, i, u)
		for k in 1:3
			rect!(f, c[k], 0.22, c[k+1], 0.78; fill = scol(k), op = k == j ? 0.75 : 0.25, stroke = "white", sw = 2)
			text!(f, (c[k] + c[k+1]) / 2, 0.5, "$(BIKE_NAMES[k])  $(BIKE_P[i, k])"; anchor = "middle", size = 12, bold = k == j)
		end
		seg!(f, u, 0.08, u, 0.92; c = COL.red, w = 2.4, over = true)
		figure(f, Markdown.parse("``u = $(u)`` lies in the piece of **$(BIKE_NAMES[j])**; the next state is $(BIKE_NAMES[j])."); caption = md"**Figure 3.1.** One step of the chain. The row of the current state divides ``[0, 1)`` into pieces whose lengths are the transition probabilities; the uniform number ``u`` selects one piece.")
	end

	# Figure 3.2: the algorithm, the rows of P and the path, two frames per day
	const SIM_CODE = ["x₀ ← start station", "for n = 0, 1, …, N − 1", "    draw U, uniform on (0, 1)",
		"    cut (0, 1) into pieces of lengths P[xₙ, 1], P[xₙ, 2], P[xₙ, 3]", "    xₙ₊₁ ← the station whose piece contains U", "return x₀, x₁, …, x_N"]
	function fig_recursion(i0, seed; D = 30)
		U = rand(Xoshiro(23 + seed), D); x = accumulate((i, u) -> next_state(BIKE_P, i, u), U; init = i0); x = [i0; x]
		f = fig(w = 680, h = 190, xr = (0, 1), yr = (3.6, 0.4), xlabel = "U", l = 112, yticks = [], title = "the rows of P")
		g = fig(w = 680, h = 170, xr = (-0.4, D + 0.4), yr = (0.5, 3.5), xlabel = "day n", yticks = 1:3, yticklabels = BIKE_NAMES, l = 76, title = "the path")
		piece(r, k, c, frame, strong) = (rect!(f, c[k], r - 0.32, c[k+1], r + 0.32; fill = scol(k), op = strong ? 0.85 : 0.25, stroke = "white", sw = 2, o = frame, over = strong);
			text!(f, (c[k] + c[k+1]) / 2, r, BIKE_NAMES[k]; anchor = "middle", size = 11, c = strong ? "white" : "#555", bold = strong, o = frame))
		for r in 1:3
			c = [0; cumsum(BIKE_P[r, :])]; foreach(k -> piece(r, k, c, nothing, false), 1:3)
			text!(f, -0.01, r, "from " * BIKE_NAMES[r]; anchor = "end", size = 11, c = "#555")
		end
		dots!(g, [0], [i0]; c = COL.red, r = 4)
		for n in 1:D, frame in (2n - 1, 2n)
			i, j = x[n], x[n+1]; c = [0; cumsum(BIKE_P[i, :])]
			foreach(k -> piece(i, k, c, frame, true), 1:3)
			seg!(f, U[n], 0.45, U[n], 3.55; c = COL.red, w = 2.4, o = frame, over = true)
			if frame == 2n
				rect!(f, c[j], i - 0.36, c[j+1], i + 0.36; fill = "none", op = 1, stroke = COL.ink, sw = 2.5, o = frame, over = true)
				seg!(g, n - 1, i, n, j; c = COL.red, w = 1.8, s = frame); dots!(g, [n], [j]; c = COL.red, r = 4, s = frame)
			end
		end
		active = [[[1]]; [isodd(k) ? [3, 4] : [5] for k in 1:2D]]; push!(active[end], 6)
		reads = ["Day 0: the bicycle starts at <b>$(BIKE_NAMES[i0])</b>."]
		for n in 1:D
			push!(reads, "Day $(n - 1) → $(n): the bicycle is at <b>$(BIKE_NAMES[x[n]])</b>, so the row of $(BIKE_NAMES[x[n]]) is used. U = <b>$(fmt(U[n]; d = 3))</b>.",
				"U = $(fmt(U[n]; d = 3)) lies in the piece of <b>$(BIKE_NAMES[x[n+1]])</b>: x$(n) = $(BIKE_NAMES[x[n+1]]).")
		end
		player(pseudo(SIM_CODE; title = "Simulating a chain", active), f, g; n = 2D, start = 0, fps = 2, reads, label = "frame",
			caption = md"**Figure 3.2.** Simulation of the bicycle chain for 30 days. Top: the algorithm. Middle: the rows of ``P``; the row of the current station is emphasised, the uniform number ``U`` is shown in red, and the selected piece is framed. Bottom: the path.")
	end

	# Figure 3.3: one row cut in two orders; the same U selects different stations, every station keeps its share
	const ORDERS = [[1, 2, 3], [3, 2, 1]]
	function fig_order(i, u; M = 10_000)
		v = rand(Xoshiro(7), M)
		f = fig(w = 680, h = 200, xr = (0, 1), yr = (2.7, 0.3), xlabel = "U", l = 150, yticks = [], title = "row of $(BIKE_NAMES[i]), cut in two orders")
		for (r, o) in enumerate(ORDERS)
			c = [0; cumsum(BIKE_P[i, o])]; pick(w) = o[something(findfirst(>(w), c[2:end]), 3)]
			for (q, k) in enumerate(o)
				rect!(f, c[q], r - 0.3, c[q+1], r + 0.3; fill = scol(k), op = k == pick(u) ? 0.85 : 0.3, stroke = "white", sw = 2)
				text!(f, (c[q] + c[q+1]) / 2, r - 0.08, "$(BIKE_NAMES[k])  $(BIKE_P[i, k])"; anchor = "middle", size = 11.5, bold = k == pick(u))
				text!(f, (c[q] + c[q+1]) / 2, r + 0.16, fmt(100count(w -> pick(w) == k, v) / M; d = 3) * " %"; anchor = "middle", size = 10.5, c = "#444")
			end
			text!(f, -0.01, r, "order " * join((BIKE_NAMES[k][1] for k in o), ", "); anchor = "end", size = 11.5)
		end
		seg!(f, u, 0.6, u, 2.4; c = COL.red, w = 2.4, over = true)
		figure(f; caption = Markdown.parse("**Figure 3.3.** The row of $(BIKE_NAMES[i]) divided in two different orders. The same uniform number ``U`` (red) selects a different station in each order. The percentages give the fraction of $(M) uniform numbers that fall into each piece."))
	end

	# Figure 5.1: all two-step paths from i to j
	function fig_paths(i, j)
		f = fig(w = 680, h = 300, xr = (0, 1), yr = (0, 1), axes = false, l = 8, r = 8, t = 12, b = 12)
		node(x, y, k) = (raw!(f, "<circle cx=\"$(_n(X(f, x)))\" cy=\"$(_n(Y(f, y)))\" r=\"31\" fill=\"$(scol(k))\" fill-opacity=\"0.22\" stroke=\"$(scol(k))\" stroke-width=\"2\"/>"; over = true);
			text!(f, x, y, BIKE_NAMES[k]; anchor = "middle", size = 10, bold = true))
		terms = String[]
		for k in 1:3
			y = 0.85 - 0.35(k - 1); a, b = BIKE_P[i, k], BIKE_P[k, j]
			arrow!(f, 0.08, 0.5, 0.25, y; c = "#666", w = 1 + 3a); arrow!(f, 0.33, y, 0.5, 0.5; c = "#666", w = 1 + 3b)
			text!(f, 0.165, (0.5 + y) / 2, string(a); dy = -9, anchor = "middle", size = 11, bg = true); text!(f, 0.415, (0.5 + y) / 2, string(b); dy = -9, anchor = "middle", size = 11, bg = true)
			node(0.29, y, k)
			text!(f, 0.6, y, "via $(BIKE_NAMES[k]):  $(a) · $(b) = $(round(a * b, digits = 4))"; size = 12.5)
			push!(terms, string(round(a * b, digits = 4)))
		end
		node(0.05, 0.5, i); node(0.53, 0.5, j)
		text!(f, 0.6, 0.06, "sum = " * join(terms, " + ") * " = $(round((BIKE_P^2)[i, j], digits = 4))"; size = 13, bold = true, c = COL.blue)
		figure(f; caption = Markdown.parse("**Figure 5.1.** All paths of length 2 from $(BIKE_NAMES[i]) to $(BIKE_NAMES[j]). The probabilities along each path are multiplied, and the products are added: ``(P^2)_{ij} = \\sum_k P_{ik} P_{kj}``."))
	end

	# Figure 5.2: the distribution π_n = π_0 Pⁿ, step by step
	function fig_flow(start; N = 15)
		π = [[k == start ? 1.0 : 0.0 for k in 1:3]]; for _ in 1:N; push!(π, π[end]' * BIKE_P |> vec); end
		b = fig(w = 260, h = 260, xr = (0.4, 3.6), yr = (0, 1.05), xticks = 1:3, xticklabels = BIKE_NAMES, ylabel = "P(Xₙ = station)", title = "distribution on day n", l = 50)
		l = fig(w = 400, h = 260, xr = (0, N), yr = (0, 1.05), xlabel = "day n", title = "the three probabilities over time", l = 40)
		for n in 0:N, k in 1:3
			rect!(b, k - 0.35, 0, k + 0.35, π[n+1][k]; fill = scol(k), op = 0.8, o = n)
			text!(b, k, π[n+1][k], string(round(π[n+1][k], digits = 3)); anchor = "middle", dy = -9, size = 11, o = n)
			n > 0 && seg!(l, n - 1, π[n][k], n, π[n+1][k]; c = scol(k), w = 2.2, s = n)
		end
		player(figure(b, l); n = N, start = 0, fps = 2, reads = ["day $(n): π = (" * join(round.(π[n+1], digits = 3), ", ") * ")" for n in 0:N],
			caption = md"**Figure 5.2.** The distribution ``\pi_n`` on day ``n``, computed with ``\pi_{n+1} = \pi_n P``.")
	end

	# Figure 5.3: a fleet of bicycles, each following its own path
	function fig_many(n0, D, seed)
		n0 = sum(n0) == 0 ? [0, 0, 300] : sum(n0) > 400 ? round.(Int, n0 .* 400 / sum(n0)) : collect(n0)
		home = vcat([fill(i, n0[i]) for i in 1:3]...); N = length(home); rng = Xoshiro(30 + seed)
		X = [home]; for _ in 1:D; push!(X, [next_state(BIKE_P, i, rand(rng)) for i in X[end]]); end
		cnt = [count(==(i), x) for x in X, i in 1:3]
		f = fig(w = 680, h = 250, xr = (0, 3), yr = (0, 1), axes = false, l = 6, r = 6, t = 26, b = 30)
		for i in 1:3
			rect!(f, i - 0.97, 0.02, i - 0.03, 0.98; fill = "#f7f7f4", op = 1, stroke = "#ccc")
			text!(f, i - 0.5, -0.07, BIKE_NAMES[i]; anchor = "middle", size = 12.5, bold = true)
		end
		for d in 0:D, i in 1:3
			for (k, b) in enumerate(findall(==(i), X[d+1]))
				dots!(f, [i - 0.94 + mod(k - 1, 20) * 0.0465], [0.05 + div(k - 1, 20) * 0.045]; c = scol(home[b]), r = 2.6, o = d)
			end
			text!(f, i - 0.5, 1.07, "$(cnt[d+1, i]) bicycles"; anchor = "middle", size = 12, o = d)
		end
		g = fig(w = 680, h = 230, xr = (0, D), yr = (0, 1.04N), xlabel = "day n", ylabel = "bicycles at the station")
		π = stationary(BIKE_P)
		for i in 1:3
			hline!(g, N * π[i]; c = scol(i)); dots!(g, [0], [cnt[1, i]]; c = scol(i), r = 3.5)
			foreach(d -> seg!(g, d - 1, cnt[d, i], d, cnt[d+1, i]; c = scol(i), w = 2, s = d), 1:D)
		end
		legend!(g, [(BIKE_NAMES[i], scol(i)) for i in 1:3]; size = 11)
		E = [N * vec((n0 ./ N)' * BIKE_P^d) for d in 0:D]
		reads = ["Day <b>$(d)</b>: Altmarkt <b>$(cnt[d+1, 1])</b>, Campus <b>$(cnt[d+1, 2])</b>, Neustadt <b>$(cnt[d+1, 3])</b>; expected by (9): " * join(round.(Int, E[d+1]), ", ") for d in 0:D]
		player(f, g; n = D, start = 0, fps = 3, reads, label = "day",
			caption = md"**Figure 5.3.** A fleet of bicycles, each following its own path of the chain. The colour of a dot is the station where the bicycle started. Bottom: the number of bicycles at each station.")
	end
end;

# ╔═╡ 67599d4a-1700-529b-af44-50887299ac69
fig_case(:iid, s1_iid)

# ╔═╡ 067c0ccc-53dd-5c22-b980-506f821ca184
fig_case(:cards, s1_cards)

# ╔═╡ d4feb79b-dfa0-54af-84b2-6212d3d955fd
fig_case(:markov, s1_markov)

# ╔═╡ f502a389-a09c-5bb5-bdd0-f7e8afbc8d18
fig_weather(s1_wstart, s1_wseed)

# ╔═╡ 82c59329-3378-5a62-ad9a-c8ce00139217
fig_bike(s2_k, s2_mode)

# ╔═╡ 6ec8f671-49bf-5fde-8c87-b3de4a7e2966
fig_twostate(s2_a, s2_b, s2_start, s2_sseed)

# ╔═╡ 5934928c-510f-55e0-834f-5b644341515d
fig_playground(s2_text, s2_pstart, s2_pseed)

# ╔═╡ 5c6c8894-ee37-5fa4-8334-6f331a991f72
fig_row(s3_i, s3_u)

# ╔═╡ efb02241-76cc-5fe6-9e93-1d9bf58ec76f
fig_recursion(s3_i0, s3_seed)

# ╔═╡ 39aa462a-5e19-408c-ae76-ee735f29611c
fig_order(s3_ostart, s3_ou)

# ╔═╡ 9e697e7d-e084-597c-88cb-3198deab1401
fig_paths(s5_i, s5_j)

# ╔═╡ ca4edfce-0c25-53c3-8f05-b7eab93e9e82
fig_flow(s5_start)

# ╔═╡ 011683d4-c23b-53d1-aaa6-88da20b45103
fig_many((s5_a, s5_c, s5_n), s5_D, s5_seed)

# ╔═╡ 4c9e90da-4039-542e-8155-ce7ddb8d8758
# Section 4 and the bonus: Markov chains that write text and melodies
begin
	const BOOKS = ["Alice's Adventures in Wonderland (Carroll, 1865)" => ALICE, "Pride and Prejudice (Austen, 1813)" => AUSTEN,
		"Grimms' Fairy Tales (transl. Taylor and Edwardes)" => GRIMM]
	const ALPH = [' '; collect('a':'z')]
	# counts of consecutive pairs: C[i, j] = number of times alphabet[i] is followed by alphabet[j]
	function pair_counts(s, alphabet)
		idx = Dict(c => k for (k, c) in enumerate(alphabet)); C = zeros(Int, length(alphabet), length(alphabet))
		for (a, b) in zip(s, Iterators.drop(s, 1))
			haskey(idx, a) && haskey(idx, b) && (C[idx[a], idx[b]] += 1)
		end
		C
	end
	chlabel(c) = c == ' ' ? "␣" : string(c)
	book(name) = strip(replace(lowercase(last(BOOKS[findfirst(==(name), first.(BOOKS))])), r"[^a-z]+" => " "))
	letter_P(t) = (C = pair_counts(t, ALPH); foreach(i -> sum(C[i, :]) == 0 && (C[i, 1] = 1), 1:27); C ./ sum(C, dims = 2))   # a letter never followed by anything goes to the space

	# Figure 4.1: the 27 × 27 matrix of a book and one of its rows
	function fig_letters(name, ch)
		t = book(name); P = letter_P(t); sel = findfirst(==(ch == "space" ? ' ' : ch[1]), ALPH)
		f = fig(w = 400, h = 410, xr = (0.5, 27.5), yr = (27.5, 0.5), xlabel = "next character", ylabel = "current character",
			xticks = 1:27, yticks = 1:27, xticklabels = chlabel.(ALPH), yticklabels = chlabel.(ALPH), l = 46, b = 40, t = 8, r = 8)
		for i in 1:27, j in 1:27
			P[i, j] > 0 && rect!(f, j - 0.5, i - 0.5, j + 0.5, i + 0.5; fill = COL.purple, op = 0.1 + 0.9sqrt(P[i, j]))
		end
		rect!(f, 0.5, sel - 0.5, 27.5, sel + 0.5; fill = "none", op = 1, stroke = COL.red, sw = 2, over = true)
		row = P[sel, :]; ord = sortperm(row; rev = true)[1:min(10, count(>(0), row))]
		g = fig(w = 260, h = 410, xr = (0, 1.2maximum(row)), yr = (length(ord) + 0.6, 0.4), xlabel = "probability", yticks = collect(1:length(ord)),
			yticklabels = ["→ " * chlabel(ALPH[k]) for k in ord], l = 46, title = "row of '$(chlabel(ALPH[sel]))'")
		for (r, k) in enumerate(ord)
			rect!(g, 0, r - 0.35, row[k], r + 0.35; fill = COL.purple, op = 0.75)
			text!(g, row[k], r, fmt(row[k]; d = 2); dx = 4, size = 10.5)
		end
		figure(f, g; caption = Markdown.parse("**Figure 4.1.** The letter chain estimated from $(length(t)) characters. Left: the ``27 \\times 27`` matrix ``\\hat P``; darker cells mark larger entries; ␣ denotes the space. Right: the ten largest entries of the row of '$(chlabel(ALPH[sel]))'."))
	end

	# Figure 4.2: generating text = simulating the chain (Section 3), one character per step
	function fig_generate(name, seed; T = 40)
		t = book(name); P = letter_P(t); rng = Xoshiro(33 + seed)
		x = [findfirst(==('t'), ALPH)]; us = rand(rng, T)
		for k in 1:T; push!(x, next_state(P, x[k], us[k])); end
		g = fig(w = 680, h = 220, xr = (0.4, 8.6), yr = (0, 1.08), ylabel = "probability", xticks = Float64[], b = 30,
			title = "row of the current character: the 8 most probable next characters")
		for s in 1:T
			row = P[x[s], :]; ord = sortperm(row; rev = true)[1:8]
			for (r, k) in enumerate(ord)
				hit = k == x[s+1]
				rect!(g, r - 0.36, 0, r + 0.36, row[k]; fill = hit ? COL.red : COL.purple, op = hit ? 0.9 : 0.5, o = s)
				text!(g, r, -0.05, "→ " * chlabel(ALPH[k]); anchor = "middle", size = 12, bold = hit, c = hit ? COL.red : COL.ink, o = s)
				text!(g, r, row[k], fmt(row[k]; d = 2); anchor = "middle", dy = -8, size = 10.5, o = s)
			end
		end
		gen(s) = join(chlabel.(ALPH[x[1:s+1]]))
		reads = ["Step <b>$(s)</b>" * (s == 0 ? ": start with 't'" : ": current '$(chlabel(ALPH[x[s]]))', U = $(fmt(us[s]; d = 3)) → '<b>$(chlabel(ALPH[x[s+1]]))</b>'") *
			"<br><span style='font:1.05rem Georgia,serif'>" * replace(gen(s), "␣" => " ") * "</span>" for s in 0:T]
		player(g; n = T, start = 0, fps = 3, reads, caption = md"**Figure 4.2.** Text generated by simulating the chain. Each step uses the row of the current character and one uniform number ``U``; the drawn character is shown in red.")
	end

	# Figure 4.3: the state is the last k characters or words
	function fig_memory_text(name, unit, k, seed; n = 400)
		raw_ = last(BOOKS[findfirst(==(name), first.(BOOKS))]); rng = Xoshiro(30 + seed); words = unit == "words"
		toks = words ? [m.match for m in eachmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]", raw_)] : collect(book(name))
		k = words ? min(k, 4) : k
		nxt = Dict{Vector{eltype(toks)}, Vector{eltype(toks)}}()
		for i in 1:length(toks)-k; push!(get!(nxt, toks[i:i+k-1], eltype(toks)[]), toks[i+k]); end
		out = toks[1:k]
		while length(out) < (words ? 120 : n)
			c = get(nxt, out[end-k+1:end], nothing)
			c === nothing ? append!(out, toks[1:k]) : push!(out, rand(rng, c))
		end
		L = words ? 6 : 20; src = Set(toks[i:i+L-1] for i in 1:length(toks)-L+1)
		copied = mean(out[i:i+L-1] in src for i in 1:length(out)-L+1)
		single = count(v -> allequal(v), values(nxt)) / length(nxt)
		txt = words ? replace(join(out, " "), r" ([.,;:!?])" => s"\1") : String(out)
		vstack(HTML("<div style=\"font:1.02rem/1.55 Georgia,serif;border-left:4px solid $(COL.purple);padding:6px 14px;max-width:680px\">$(xesc(txt))</div>"),
			Markdown.parse("States observed: **$(length(nxt))**. States with only one observed continuation: **$(round(Int, 100single)) %**. Pieces of $(L) $(words ? "words" : "characters") copied word for word from the book: **$(round(Int, 100copied)) %**.");
			caption = Markdown.parse("**Figure 4.3.** Text from a chain whose state is the last ``k = $(k)`` $(words ? "words" : "characters")."))
	end

	# six public-domain melodies as (pitch, duration) pairs; pitch as MIDI number, duration in eighth notes
	const MELODIES = [
		"Ode to Joy (Beethoven)" => [(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (64, 3), (62, 1), (62, 4),
			(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (62, 3), (60, 1), (60, 4),
			(62, 2), (62, 2), (64, 2), (60, 2), (62, 2), (64, 1), (65, 1), (64, 2), (60, 2), (62, 2), (64, 1), (65, 1), (64, 2), (62, 2), (60, 2), (62, 2), (55, 4),
			(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (62, 3), (60, 1), (60, 4)],
		"Twinkle, twinkle, little star" => [(60, 2), (60, 2), (67, 2), (67, 2), (69, 2), (69, 2), (67, 4), (65, 2), (65, 2), (64, 2), (64, 2), (62, 2), (62, 2), (60, 4),
			(67, 2), (67, 2), (65, 2), (65, 2), (64, 2), (64, 2), (62, 4), (67, 2), (67, 2), (65, 2), (65, 2), (64, 2), (64, 2), (62, 4),
			(60, 2), (60, 2), (67, 2), (67, 2), (69, 2), (69, 2), (67, 4), (65, 2), (65, 2), (64, 2), (64, 2), (62, 2), (62, 2), (60, 4)],
		"Frère Jacques" => [(60, 2), (62, 2), (64, 2), (60, 2), (60, 2), (62, 2), (64, 2), (60, 2), (64, 2), (65, 2), (67, 4), (64, 2), (65, 2), (67, 4),
			(67, 1), (69, 1), (67, 1), (65, 1), (64, 2), (60, 2), (67, 1), (69, 1), (67, 1), (65, 1), (64, 2), (60, 2), (60, 2), (55, 2), (60, 4), (60, 2), (55, 2), (60, 4)],
		"Alle meine Entchen" => [(60, 2), (62, 2), (64, 2), (65, 2), (67, 4), (67, 4), (69, 2), (69, 2), (69, 2), (69, 2), (67, 8), (69, 2), (69, 2), (69, 2), (69, 2), (67, 8),
			(65, 2), (65, 2), (65, 2), (65, 2), (64, 4), (64, 4), (62, 2), (62, 2), (62, 2), (62, 2), (60, 8)],
		"Hänschen klein" => [(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (62, 2), (64, 2), (65, 2), (67, 2), (67, 2), (67, 4),
			(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (64, 2), (67, 2), (67, 2), (60, 8),
			(62, 2), (62, 2), (62, 2), (62, 2), (62, 2), (64, 2), (65, 4), (64, 2), (64, 2), (64, 2), (64, 2), (64, 2), (65, 2), (67, 4),
			(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (64, 2), (67, 2), (67, 2), (60, 8)],
		"Mary had a little lamb" => [(64, 3), (62, 1), (60, 2), (62, 2), (64, 2), (64, 2), (64, 4), (62, 2), (62, 2), (62, 4), (64, 2), (67, 2), (67, 4),
			(64, 3), (62, 1), (60, 2), (62, 2), (64, 2), (64, 2), (64, 2), (64, 2), (62, 2), (62, 2), (64, 2), (62, 2), (60, 8)]]
	melody(name) = last(MELODIES[findfirst(==(name), first.(MELODIES))])
	tune(name) = first.(melody(name))
	pname(m) = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"][mod(m, 12) + 1] * string(m ÷ 12 - 1)
	midi_hz(m) = 440.0 * 2.0^((m - 69) / 12)
	const EIGHTH = 0.24                     # seconds per eighth note

	# Play note events (t, f, d, v, k) in the browser: start time and duration in s, frequency in Hz, loudness, instrument.
	# k = 0 is a plucked string synthesised from random numbers (Karplus–Strong). Elements with class pn-hl are lit while their notes sound.
	function play_notes(events; label = "Play", seed = 1)
		ev = [[e[1], e[2], e[3], e[4], e[5]] for e in events]
		HTML("""<span class="pn"><button class="pn-b">▶ $(label)</button><script>
		const root = currentScript.parentElement, btn = root.querySelector(".pn-b"), ev = $(js(ev)), lab = $(js(label));
		let ctx = null, src = null, raf = null, t0 = 0;
		function rng(a) { return function() { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
		function render(sr) {
			const R = rng($(seed));
			let T = 0; for (const e of ev) T = Math.max(T, e[0] + e[2]);
			const N = Math.ceil((T + 1.6) * sr), y = new Float32Array(N);
			for (const [t, f, d, v] of ev) {
				const a = Math.floor(t * sr), P = Math.max(2, Math.round(sr / f)), L = Math.min(N - a, Math.floor((d + 1.2) * sr)), buf = new Float32Array(L);
				let lp = 0; for (let i = 0; i < Math.min(P, L); i++) { lp = 0.6 * lp + 0.4 * (2 * R() - 1); buf[i] = lp; }
				const off = Math.floor(d * sr);
				for (let i = P; i < L; i++) { const s = i < off ? 0.998 : 0.985; buf[i] = s * 0.5 * (buf[i - P] + buf[i - P - 1 >= 0 ? i - P - 1 : i - P]); }
				for (let i = 0; i < L; i++) y[a + i] += v * buf[i];
			}
			let m = 0; for (let i = 0; i < N; i++) m = Math.max(m, Math.abs(y[i]));
			if (m > 0) for (let i = 0; i < N; i++) y[i] *= 0.85 / m;
			return y;
		}
		const scope = root.closest("pluto-output") || root.parentElement, hl = Array.from(scope.querySelectorAll(".pn-hl"));
		function mark(t) { for (const e of hl) e.style.display = (t >= +e.dataset.a && t < +e.dataset.b) ? "" : "none"; }
		function stop() { if (src) { try { src.stop(); } catch (e) {} src = null; } if (raf) cancelAnimationFrame(raf); raf = null; btn.textContent = "▶ " + lab; mark(-1); }
		mark(-1);
		btn.addEventListener("click", () => {
			if (src) { stop(); return; }
			ctx = ctx || new (window.AudioContext || window.webkitAudioContext)();
			const sr = ctx.sampleRate, y = render(sr), b = ctx.createBuffer(1, y.length, sr);
			b.copyToChannel(y, 0); src = ctx.createBufferSource(); src.buffer = b; src.connect(ctx.destination);
			src.onended = stop; t0 = ctx.currentTime + 0.05; src.start(t0); btn.textContent = "■ stop";
			const tick = () => { mark(ctx.currentTime - t0); raf = requestAnimationFrame(tick); }; tick();
		});
		invalidation.then(stop);
		</script></span>""")
	end

	# chords for the accompaniment: name, pitch classes, root
	const CHORDS = [("C", [0, 4, 7], 48), ("F", [5, 9, 0], 53), ("G", [7, 11, 2], 55), ("Am", [9, 0, 4], 57), ("Dm", [2, 5, 9], 50)]
	# one chord per bar (8 eighths): the chord that contains most of the bar's notes, ending on C
	function harmonise(notes)
		nb = cld(sum(last.(notes)), 8); score = zeros(nb, length(CHORDS)); t = 0
		for (p, d) in notes
			for q in 0:d-1
				b = (t + q) ÷ 8 + 1; b > nb && continue
				for (c, (_, pcs, _)) in enumerate(CHORDS); mod(p, 12) in pcs && (score[b, c] += (t + q) % 8 == 0 ? 2.0 : 1.0); end
			end
			t += d
		end
		ch = [argmax(score[b, :] .+ [0.3, 0.1, 0.2, 0, 0]) for b in 1:nb]
		ch[end] = 1; nb > 1 && score[nb-1, 3] > 0 && (ch[nb-1] = 3)
		ch
	end
	function accompaniment(ch)
		ev = Tuple{Float64, Float64, Float64, Float64, Int}[]
		for (b, c) in enumerate(ch)
			_, pcs, root = CHORDS[c]; t0 = (b - 1) * 8 * EIGHTH
			push!(ev, (t0, midi_hz(root - 12), 2EIGHTH, 0.5, 0), (t0 + 4EIGHTH, midi_hz(root - 5), 2EIGHTH, 0.42, 0))
			for q in (2, 6), pc in pcs[2:3]; push!(ev, (t0 + q * EIGHTH, midi_hz(52 + mod(pc - 52, 12)), 1.6EIGHTH, 0.22, 0)); end
		end
		ev
	end
	function events(notes)
		t = 0; ev = Tuple{Float64, Float64, Float64, Float64, Int}[]
		for (p, d) in notes; push!(ev, (t * EIGHTH, midi_hz(p), d * EIGHTH, 0.8, 0)); t += d; end
		ev
	end
	play_both(notes; seed = 2) = HTML(tohtml(play_notes(events(notes); label = "Play", seed)) * " " *
		tohtml(play_notes(vcat(events(notes), accompaniment(harmonise(notes))); label = "Play with chords", seed)))

	# a melody as a piano roll; one box per note, lit while it sounds
	function pianoroll(notes, title; pitches = sort(unique(first.(notes))))
		T = sum(last.(notes))
		f = fig(w = 680, h = 60 + 16length(pitches), xr = (0, T), yr = (0.5, length(pitches) + 0.5), xlabel = "time (eighth notes)", yticks = 1:length(pitches), yticklabels = pname.(pitches), l = 40, title = title)
		foreach(b -> vline!(f, b; c = "#d8d8d8", dash = nothing, w = 1), 0:8:T)
		t = 0
		for (p, d) in notes
			k = findfirst(==(p), pitches)
			rect!(f, t + 0.08, k - 0.36, t + d - 0.08, k + 0.36; fill = COL.purple, op = 0.85)
			raw!(f, "<rect class=\"pn-hl\" data-a=\"$(t * EIGHTH)\" data-b=\"$((t + d) * EIGHTH)\" x=\"$(_n(X(f, t)))\" y=\"$(_n(Y(f, k + 0.48)))\" width=\"$(_n(X(f, t + d) - X(f, t)))\" height=\"$(_n(Y(f, k - 0.48) - Y(f, k + 0.48)))\" fill=\"none\" stroke=\"$(COL.red)\" stroke-width=\"2.5\" style=\"display:none\"/>"; over = true)
			t += d
		end
		f
	end

	# generate a melody of 8 bars: the state is the last k notes (pitch, duration); a note never crosses a bar line;
	# if the last k notes were never seen, the chain uses the last k − 1 notes; the last bar is a cadence on C
	function generate(m, k, rng; bars = 8)
		nxt = [Dict{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}() for _ in 1:k]
		for j in 1:k, i in 1:length(m)-j; push!(get!(nxt[j], m[i:i+j-1], Tuple{Int, Int}[]), m[i+j]); end
		gen = [m[1]]; t = m[1][2]; T = 8bars
		while t < T - 8
			room = 8 - t % 8; c = Tuple{Int, Int}[]
			for j in min(k, length(gen)):-1:1
				c = [x for x in get(nxt[j], gen[end-j+1:end], Tuple{Int, Int}[]) if x[2] <= room]
				isempty(c) || break
			end
			isempty(c) && (c = [(gen[end][1], room)])
			x = rand(rng, c); push!(gen, x); t += x[2]
		end
		push!(gen, (rand(rng, [62, 64, 67]), 4), (60, 4))
		gen, length(nxt[k])
	end

	# Figure B.1: the selected melody
	fig_tune(name) = vstack(pianoroll(melody(name), name), play_both(melody(name));
		caption = Markdown.parse("**Figure B.1.** *$(name)* as a piano roll: each box is one note, plotted at its pitch, with its length in eighth notes. Grey lines separate the bars. With chords, a simple accompaniment of one chord per bar is added."))

	# Figure B.2: the pitch transitions of the selected melody, and one row
	function fig_notes(name, sel)
		m = tune(name); ps = sort(unique(m)); n = length(ps); k = Dict(p => i for (i, p) in enumerate(ps)); C = zeros(n, n)
		for (a, b) in zip(m, m[2:end]); C[k[a], k[b]] += 1; end
		P = C ./ max.(sum(C, dims = 2), 1); i = something(findfirst(==(sel), pname.(ps)), 1)
		f = fig(w = 380, h = 380, xr = (0.5, n + 0.5), yr = (n + 0.5, 0.5), xlabel = "next note", ylabel = "current note",
			xticks = 1:n, yticks = 1:n, xticklabels = pname.(ps), yticklabels = pname.(ps), l = 46, b = 40, t = 8, r = 8)
		for a in 1:n, b in 1:n; P[a, b] > 0 && rect!(f, b - 0.5, a - 0.5, b + 0.5, a + 0.5; fill = COL.purple, op = 0.1 + 0.9sqrt(P[a, b])); end
		rect!(f, 0.5, i - 0.5, n + 0.5, i + 0.5; fill = "none", op = 1, stroke = COL.red, sw = 2, over = true)
		g = fig(w = 280, h = 380, xr = (0.4, n + 0.6), yr = (0, 1.05), xticks = 1:n, xticklabels = pname.(ps), ylabel = "probability", l = 46, title = "row of $(pname(ps[i]))")
		for b in 1:n; P[i, b] > 0 && (rect!(g, b - 0.35, 0, b + 0.35, P[i, b]; fill = COL.purple, op = 0.75); text!(g, b, P[i, b], fmt(P[i, b]; d = 2); anchor = "middle", dy = -8, size = 10)); end
		figure(f, g; caption = Markdown.parse("**Figure B.2.** The pitch transitions estimated with (7) from *$(name)* ($(length(m)) notes). Left: the matrix ``\\hat P``; darker cells mark larger entries. Right: the row of the selected note."))
	end

	# Figure B.3: a melody generated by the chain whose state is the last note
	function fig_melody(name, seed)
		gen, _ = generate(melody(name), 1, Xoshiro(31 + seed))
		vstack(pianoroll(gen, "generated melody"), play_both(gen; seed = 3);
			caption = Markdown.parse("**Figure B.3.** A melody of eight bars generated from *$(name)*. The state is the last note, a pair (pitch, duration); each next note is drawn from the notes that follow the current one in the original. The last bar ends on C."))
	end

	# Figure B.4: a melody generated by the chain whose state is the last k notes
	function fig_memory_music(name, k, seed; L = 6)
		m = melody(name); gen, ns = generate(m, k, Xoshiro(30 + seed))
		src = Set(m[i:i+L-1] for i in 1:length(m)-L+1); copied = mean(gen[i:i+L-1] in src for i in 1:length(gen)-L)
		vstack(pianoroll(gen, "generated melody, memory k = $(k)"), play_both(gen; seed = 4),
			Markdown.parse("States observed: **$(ns)**. Pieces of $(L) notes copied from the original: **$(round(Int, 100copied)) %**.");
			caption = Markdown.parse("**Figure B.4.** A melody generated from *$(name)* by a chain whose state is the last ``k = $(k)`` notes."))
	end
end;

# ╔═╡ 400c407d-2c73-52e3-a9ea-e865377e916c
md"""
## 4. (Optional) Bonus Example: a Markov chain that writes text

The state of a Markov chain can be any object from a finite list, for example the current letter of a text. In 1913, A. A. Markov counted the letters of Pushkin's *Eugene Onegin* to study such dependence; in 1948, C. E. Shannon used chains of letters in the founding paper of information theory. Two steps are required: the matrix is **estimated** from a text, and the chain is then **simulated**.

### 4.1 Estimating the transition matrix

The state is the current character, one of the 26 letters or the space. ``C_{ij}`` denotes the number of times character ``i`` is followed by character ``j`` in a book. Dividing each row by its total gives

**(7) Estimate from counts**
```math
\hat P_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}} = \frac{C_{\textcolor{#4477AA}{i}\textcolor{#EE7733}{j}}}{\sum_k C_{\textcolor{#4477AA}{i}k}} .
```

Book $(@bind s4_book Select(first.(BOOKS))) ``\qquad`` character $(@bind s4_ch Select(["space", "a", "e", "h", "o", "q", "t", "w"]; default = "t"))
"""

# ╔═╡ 2f30f87f-4672-5209-a2e2-e2796119a911
fig_letters(s4_book, s4_ch)

# ╔═╡ 04532a1c-d900-5ee9-be18-dab9d329b891
fig_generate(s4_book, s4_gseed)

# ╔═╡ b3de8c6a-34ca-51ee-b232-e4af621d1b05
fig_memory_text(s4_book, s4_unit, s4_k, s4_tseed)

# ╔═╡ a9060a31-c192-4373-9628-56d3fef27c09
md"""
Constructing a chain for a given distribution, instead of computing the distribution of a given chain, is the basis of the sampling methods treated later in this course. The properties of Sections 6–8 determine whether, and how fast, such a chain produces correct samples.

## Bonus (optional): a Markov chain that plays melodies

The state of a chain can also be the current note of a melody. The six melodies below are short public-domain tunes. Each figure of this section uses only the selected melody.

Melody $(@bind sb_tune Select(first.(MELODIES)))
"""

# ╔═╡ 695db1a8-9a03-4092-87eb-f934b144c79d
fig_tune(sb_tune)

# ╔═╡ e8ac34ca-7c63-4816-9ca0-e856c1658e8b
md"""
The note chain of the selected melody is estimated with (7).

Note $(@bind sb_note Select(pname.(sort(unique(tune(sb_tune))))))
"""

# ╔═╡ 6e8aecd6-6f44-45c3-b7df-7d119dc29086
fig_notes(sb_tune, sb_note)

# ╔═╡ 81153baf-bd45-4ac6-98b3-b5cd3b2e64a6
fig_melody(sb_tune, sb_gseed)

# ╔═╡ bd8fdefe-47b8-447a-8434-d3fcdfaf74ca
fig_memory_music(sb_tune, sb_k, sb_tseed)

# ╔═╡ a67909c4-72a1-57f7-9ccd-d1b5a9f58bb6
# Sections 6–8: stationary and limiting distributions, properties of chains, mixing time
begin
	# the small chains of Sections 6 and 7
	const CHAIN_A = [0.5 0.3 0.2; 0.3 0.5 0.2; 0 0 1]                         # state 3 absorbing
	const CHAIN_A2 = [0.4 0.3 0.3; 0 0.5 0.5; 0 0.6 0.4]                       # {2, 3} closed
	const CHAIN_B = [0.7 0.3 0 0; 0.4 0.6 0 0; 0 0 0.5 0.5; 0 0 0.2 0.8]        # two islands
	const CHAIN_C = [0 1 0; 0.5 0 0.5; 0 1 0]                                  # period 2
	const CHAIN_D = [0 0.7 0.3; 0.5 0 0.5; 0.6 0.4 0]                          # ergodic, no self-loops
	const CHAIN_E = [0 1 0 0; 1 0 0 0; 0 0 0 1; 0 0 1 0]                       # two islands, each periodic
	const SWAP = [0.0 1; 1 0]
	const CYCLE3 = [0 1 0; 0 0 1; 1.0 0 0]

	# P(X_n = j) for n = 0…N, one line per state j
	function prob_lines(P, start; N = 20, w = 330, title = "P(Xₙ = j), start in state $(start)")
		n = size(P, 1); π = [[k == start ? 1.0 : 0.0 for k in 1:n]]; for _ in 1:N; push!(π, vec(π[end]' * P)); end
		f = fig(; w, h = 220, xr = (0, N), yr = (0, 1.05), xlabel = "step n", title, l = 40)
		for k in 1:n; line!(f, 0:N, [p[k] for p in π]; c = scol(k), w = 2); dots!(f, 0:N, [p[k] for p in π]; c = scol(k), r = 2.4); end
		f
	end

	# a chain in motion from one start, and π_n from that start and from a second one
	function fig_chain(P, start, other, seed, caption; F = 16)
		n = size(P, 1); start = min(start, n); other = min(other, n)
		vstack(chain_anim(P, string.(1:n), simulate(P, start, 30, Xoshiro(17 + seed)); F, w = 320, h = 250),
			figure(prob_lines(P, start), prob_lines(P, other)); caption)
	end
	pick(list, name) = last(list[findfirst(==(name), first.(list))])

	# Figure 7.1: closed sets and absorbing states
	const TRAPS = ["an absorbing state" => CHAIN_A, "a closed set of two states" => CHAIN_A2]
	fig_trap(name, start, other, seed) = fig_chain(pick(TRAPS, name), start, other, seed,
		md"**Figure 7.1.** Top: one path from the chosen start. Bottom: the probability of each state after ``n`` steps, from the chosen start and from the second start. Once the chain enters the closed set, it never leaves it.")

	# Figure 7.2: irreducible or not
	const ISLANDS = ["the bicycle chain" => BIKE_P, "two islands" => CHAIN_B]
	fig_islands(name, start, other, seed) = fig_chain(pick(ISLANDS, name), start, other, seed,
		md"**Figure 7.2.** Top: one path from the chosen start. Bottom: the probability of each state after ``n`` steps, from the two chosen starts.")

	# Figure 7.3: periodic and aperiodic chains
	const PERIODS = ["swap: 1 ⇄ 2" => SWAP, "cycle of three: 1 → 2 → 3 → 1" => CYCLE3, "chain C: 1 ⇄ 2 ⇄ 3" => CHAIN_C,
		"chain D: no self-loops" => CHAIN_D, "the bicycle chain" => BIKE_P]
	function fig_period(name, start, seed)
		P = pick(PERIODS, name); start = min(start, size(P, 1))
		vstack(chain_anim(P, string.(1:size(P, 1)), simulate(P, start, 30, Xoshiro(17 + seed)); F = 16, w = 320, h = 250), prob_lines(P, start; w = 680);
			caption = md"**Figure 7.3.** Top: one path from the chosen start. Bottom: the probability of each state after ``n`` steps.")
	end

	# stationary distributions of a chain: one per closed class
	function closed_stationary(P)
		C = classify(P); n = size(P, 1)
		map(C.classes[C.closed]) do c
			v = zeros(n); v[c] = stationary(P[c, c]); v
		end
	end
	vfmt(v) = "(" * join(fmt.(v; d = 3), ", ") * ")"

	# Figure 7.4: the four combinations of irreducible and aperiodic
	const CASES = ["ergodic (chain D)" => CHAIN_D, "irreducible, periodic (chain C)" => CHAIN_C,
		"not irreducible, aperiodic (chain B)" => CHAIN_B, "not irreducible, periodic (chain E)" => CHAIN_E]
	function fig_cases(name, start, other, seed)
		P = pick(CASES, name); S = closed_stationary(P)
		txt = length(S) == 1 ? "One stationary distribution: π = $(vfmt(S[1]))." :
			"Each closed set has its own stationary distribution: " * join(vfmt.(S), " and ") * ""
		vstack(fig_chain(P, start, other, seed, nothing), Markdown.parse(txt);
			caption = md"**Figure 7.4.** Top: one path. Bottom: ``\pi_n`` from two initial states. A limiting distribution exists only if both lower panels converge to the same constant values.")
	end

	# Figure 6.1: from every start the bicycle approaches the same distribution
	function fig_converge(n; N = 20)
		π = stationary(BIKE_P); Q = [BIKE_P^m for m in 0:N]
		panels = map(1:3) do i
			f = fig(w = 225, h = 220, xr = (0, N), yr = (0, 1.02), xlabel = "day n", ylabel = i == 1 ? "probability" : "", title = "start: $(BIKE_NAMES[i])", l = i == 1 ? 50 : 30)
			for j in 1:3; hline!(f, π[j]; c = scol(j)); line!(f, 0:N, [q[i, j] for q in Q]; c = scol(j), w = 2.2); end
			vline!(f, n; c = "#999", dash = "2 3"); f
		end
		figure(panels..., legend_html(), matrix_fig(BIKE_P^n, BIKE_NAMES; digits = 3, title = "Pⁿ for n = $(n): row = start", w = 300);
			caption = md"**Figure 6.1.** The distribution of one bicycle on day ``n``, for each start (solid), and the stationary distribution (dashed). The rows of ``P^n`` become equal, so the initial state is forgotten.")
	end
	legend_html() = HTML("<div style=\"font:13px system-ui;width:100%\">" * join(["<span style=\"color:$(scol(k))\">■</span> $(BIKE_NAMES[k])" for k in 1:3], " &nbsp; ") * "</div>")

	# mixing time: first n at which every entry of πₙ is within tol of π
	mixing_time(P, start; tol = 0.01) = (π = stationary(P); p = [k == start ? 1.0 : 0.0 for k in 1:size(P, 1)]; n = 0;
		while maximum(abs, p - π) >= tol && n < 10^6; p = vec(p' * P); n += 1; end; n)

	# Chain 2 of Section 8: two groups of three states, joined only by the edge 3 ⇄ 4
	const CHAIN2 = let P = zeros(6, 6)
		for g in (0, 3), i in 1:3, j in 1:3; P[g+i, g+j] = 1 / 3; end
		P[3, 3] -= 0.01; P[3, 4] = 0.01; P[4, 4] -= 0.01; P[4, 3] = 0.01; P
	end
	const CHAIN2_POS = [(0.08, 0.78), (0.08, 0.22), (0.36, 0.5), (0.64, 0.5), (0.92, 0.78), (0.92, 0.22)]

	# Figure 8.1: many simulated chains from one start
	function fig_mixsim(chain, start, seed; R = 1000)
		two = chain == "Chain 2"; P = two ? CHAIN2 : BIKE_P; N = two ? 600 : 20; n = size(P, 1); start = min(start, n)
		names = two ? string.(1:n) : BIKE_NAMES; π = stationary(P); X = [simulate(P, start, N, Xoshiro(41 + seed + r)) for r in 1:R]
		d = two ? diagram(P, names; pos = CHAIN2_POS, w = 680, h = 220) : diagram(P, names; w = 360, h = 260)
		f = fig(w = 680, h = 250, xr = (0, N), yr = (0, 1), xlabel = "step n", ylabel = "fraction of the chains", l = 56)
		for j in 1:n; hline!(f, π[j]; c = scol(j)); line!(f, 0:N, [count(x -> x[k+1] == j, X) / R for k in 0:N]; c = scol(j), w = 2); end
		legend!(f, [(names[j], scol(j)) for j in 1:n])
		figure(d, f, Markdown.parse("From state $(names[start]), every entry of ``\\pi_n`` is within 0.01 of the limiting distribution after **$(mixing_time(P, start))** steps.");
			caption = Markdown.parse("**Figure 8.1.** $(chain). Bottom: the fraction of $(R) simulated chains in each state (solid) and the limiting distribution (dashed)."))
	end

	# Figure 7.5: which kind of chain is it?
	const GAME_KINDS = ["ergodic", "irreducible and periodic", "not irreducible: several closed sets", "not irreducible: one absorbing state"]
	const GAME_CHAINS = [BIKE_P => 1, WEATHER_P => 1, CHAIN_D => 1, [0.5 0.5 0; 0 0.5 0.5; 0.5 0 0.5] => 1,
		SWAP => 2, CYCLE3 => 2, CHAIN_C => 2, [0 0.5 0 0.5; 0.5 0 0.5 0; 0 0.5 0 0.5; 0.5 0 0.5 0] => 2,
		CHAIN_B => 3, CHAIN_E => 3, [1 0 0; 0.3 0.4 0.3; 0 0 1] => 3,
		CHAIN_A => 4, [0.6 0.4 0; 0 0.5 0.5; 0 0 1] => 4]
	function chain_game(; rounds = 6)
		show = [tohtml(figure(diagram(P, string.(1:size(P, 1)); w = 300, h = 230, R = 20), matrix_fig(P, string.(1:size(P, 1))))) for (P, _) in GAME_CHAINS]
		HTML("""<div class="og"><div class="og-top"><b>Which kind of chain is it?</b><span class="og-score"></span></div>
		<div class="og-dia"></div><div class="og-opts"></div><div class="og-fb"></div>
		<div><button class="og-next pn-b">next chain</button> <button class="og-new pn-b">new game</button></div>
		<script>
		const root = currentScript.parentElement, D = $(js(show)), K = $(js(last.(GAME_CHAINS))), NAMES = $(js(GAME_KINDS)), R = $(rounds);
		const opts = root.querySelector(".og-opts"), fb = root.querySelector(".og-fb"), sc = root.querySelector(".og-score");
		let order = [], round = 0, score = 0, done = false;
		function newGame() { order = [...K.keys()].sort(() => Math.random() - 0.5).slice(0, R); round = 0; score = 0; show(); }
		function show() {
			opts.innerHTML = ""; fb.textContent = "";
			if (round >= R) { root.querySelector(".og-dia").innerHTML = ""; sc.textContent = ""; fb.textContent = "Result: " + score + " of " + R + " correct."; return; }
			const k = order[round]; done = false;
			root.querySelector(".og-dia").innerHTML = D[k];
			sc.textContent = "chain " + (round + 1) + " of " + R + " · score " + score;
			NAMES.forEach((name, i) => {
				const b = document.createElement("button"); b.textContent = name;
				b.onclick = () => {
					if (done) return; done = true;
					const ok = i + 1 === K[k]; score += ok;
					b.className = ok ? "ok" : "no"; fb.textContent = ok ? "Correct." : "Not correct.";
					sc.textContent = "chain " + (round + 1) + " of " + R + " · score " + score;
				};
				opts.appendChild(b);
			});
		}
		root.querySelector(".og-next").onclick = () => { round++; show(); };
		root.querySelector(".og-new").onclick = newGame;
		newGame();
		</script></div>""")
	end
end;

# ╔═╡ a764371f-6172-4770-8fc2-69e9aed97b24
fig_converge(s6_n)

# ╔═╡ d62ea4c0-59f4-4a75-b077-77b629f008c5
md"""
## 7. When does the long run exist? Properties of chains

For the bicycle chain, the stationary distribution and the limiting distribution coincide. This is not the case for every chain, for three reasons:

1. the chain can be **trapped** in a set of states;
2. the state diagram can consist of **separate parts** that do not reach each other;
3. the chain can move in a fixed **rhythm**.

### 7.1 Traps: closed sets and absorbing states

!!! definition "Closed set, absorbing state"
    A set ``C`` of states is **closed** if no arrow leaves ``C``, that is, if no state outside ``C`` can be reached from ``C``. A closed set that consists of a single state ``i`` is an **absorbing** state: ``P_{ii} = 1``. Once the chain enters a closed set, it never leaves it.

Chain $(@bind s7_trap Select(first.(TRAPS))) ``\qquad`` start $(@bind s7_tstart Select(1:3)) ``\qquad`` second start $(@bind s7_tother Select(1:3; default = 2)) ``\qquad`` $(@bind s7_tseed Seed(0; label = "New path"))
"""

# ╔═╡ 5586aa8e-bc3a-4145-8051-2cdb51bc932e
fig_trap(s7_trap, s7_tstart, s7_tother, s7_tseed)

# ╔═╡ c42d9641-d5fb-489b-a79b-52da695ccf65
md"""
### 7.2 Separate parts: irreducible chains

!!! definition "Irreducible"
    A chain is **irreducible** if ``S`` is its only closed set. Equivalently, every state can be reached from every other state, possibly in several steps.

Chain $(@bind s7_isl Select(first.(ISLANDS))) ``\qquad`` start $(@bind s7_istart Select(1:4)) ``\qquad`` second start $(@bind s7_iother Select(1:4; default = 3)) ``\qquad`` $(@bind s7_iseed Seed(0; label = "New path"))
"""

# ╔═╡ d135773c-f229-4b8f-b26c-bbfb79a79298
fig_islands(s7_isl, s7_istart, s7_iother, s7_iseed)

# ╔═╡ c0effd89-d534-4cb4-9dcf-b3619f3c5ab9
md"""
### 7.3 Periodic and aperiodic states

!!! definition "Period"
    The **period** of a state ``i`` is the largest integer ``d`` such that the chain can return to ``i`` only after a multiple of ``d`` steps. A state with period ``d > 1`` is **periodic**; a state with period 1 is **aperiodic**.

In a periodic chain the probabilities ``P(X_n = j)`` oscillate and do not converge. A self-loop (``P_{ii} > 0``) makes a state aperiodic, since a return is then possible after one step.

Chain $(@bind s7_per Select(first.(PERIODS))) ``\qquad`` start $(@bind s7_pstart Select(1:3)) ``\qquad`` $(@bind s7_pseed Seed(0; label = "New path"))
"""

# ╔═╡ fe0d493f-fb6e-4ac1-9f47-cdbe46d886c6
fig_period(s7_per, s7_pstart, s7_pseed)

# ╔═╡ 50aa9253-3632-480e-969a-acf68d727cc4
md"""
### 7.4 Ergodic chains

!!! definition "Ergodic"
    A chain is **ergodic** if it is irreducible **and** aperiodic.

!!! theorem "Limiting and stationary distribution"
    An ergodic chain has a limiting distribution. It is the **unique** stationary distribution, and the chain converges to it from every initial state.

In particular, an ergodic chain cannot converge to different distributions from different initial states. Figure 7.4 shows the four combinations of the two properties.

Case $(@bind s7_case Select(first.(CASES))) ``\qquad`` start $(@bind s7_cstart Select(1:4)) ``\qquad`` second start $(@bind s7_cother Select(1:4; default = 3)) ``\qquad`` $(@bind s7_cseed Seed(0; label = "New path"))
"""

# ╔═╡ b6ce307d-ec85-4e4c-af74-759aad7cc68f
fig_cases(s7_case, s7_cstart, s7_cother, s7_cseed)

# ╔═╡ 54cc2434-21b7-44f7-9c95-108504b924e4
figure(chain_game(); caption = md"**Figure 7.5.** Six rounds with chains drawn at random from a set of thirteen.")

# ╔═╡ 604e0caa-fc9c-4368-a1e4-ae801f3eb199
fig_mixsim(s8_chain, s8_start, s8_seed)

# ╔═╡ 78b7470a-7c29-5c5b-a480-e6b7553031aa
# Exercises: checks of the programming exercises
begin
	function check_normalise(normalise_rows)
		C = [2 6 2; 0 5 5; 9 0 3]; r = tryrun(() -> normalise_rows(C))
		ok = r isa AbstractMatrix && size(r) == (3, 3) && r ≈ C ./ sum(C, dims = 2)
		check(r, ok, md"Every row is the count of each next state divided by the row total, so every row sums to 1.",
			md"Divide each row by its own sum, not by the total of the matrix or by the column sums."; hint = md"`C ./ sum(C, dims = 2)`: `dims = 2` sums along each row.")
	end

	function check_simulate(simulate_chain)
		P = [0.1 0.9 0.0; 0.0 0.2 0.8; 0.7 0.0 0.3]; r = tryrun(() -> simulate_chain(P, 1, 100_000, Xoshiro(1)))
		r isa AbstractVector && length(r) == 100_001 && r[1] == 1 || return check(r, false, md"", md"The path must have ``n + 1`` entries and start in `x0`.")
		C = pair_counts(r, 1:3); F = C ./ sum(C, dims = 2); ok = maximum(abs, F - P) < 0.01
		check(r, ok, Markdown.parse("The observed transition frequencies agree with ``P`` (largest difference $(fmt(maximum(abs, F - P); d = 2)))."),
			Markdown.parse("The observed transition frequencies differ from ``P`` by up to $(fmt(maximum(abs, F - P); d = 2)). For example, the observed frequency of 1 → 3 is $(fmt(F[1, 3]; d = 2)), but ``P_{13} = 0``.");
			hint = md"The next state is drawn from the **row** of the current state: `P[x[t], :]`.")
	end

	function check_after(distribution_after)
		π0 = [0.0, 0.0, 1.0]; r = tryrun(() -> [distribution_after(BIKE_P, π0', n) for n in (1, 2, 10)])
		ok = r isa AbstractVector && all(((x, n),) -> vec(x) ≈ vec(π0' * BIKE_P^n), zip(r, (1, 2, 10)))
		check(r, ok, md"``\pi_n = \pi_0 P^n`` for ``n = 1, 2, 10``.",
			md"For ``n = 2`` and ``n = 10`` the result is still ``\pi_0 P``: the update multiplies the start distribution again instead of the current one."; hint = md"`π = π * P` inside the loop.")
	end

	function check_power(stationary_power)
		r = tryrun(() -> [stationary_power(P) for P in (BIKE_P, WEATHER_P, [0.0 1.0; 0.5 0.5])])
		ok = r isa AbstractVector && all(((x, P),) -> vec(x) ≈ stationary(P) && abs(sum(x) - 1) < 1e-8, zip(r, (BIKE_P, WEATHER_P, [0.0 1.0; 0.5 0.5])))
		check(r, ok, md"The three stationary distributions are correct, and each sums to 1.",
			md"Start from any distribution and repeat ``\pi \leftarrow \pi P`` until ``\pi`` stops changing."; hint = md"`π = fill(1 / n, 1, n)` and then `π = π * P` until `maximum(abs, π * P - π) < tol`.")
	end
end;

# ╔═╡ 6997c198-aa84-54be-a811-a4fc37a3ffed
check_normalise(normalise_rows)

# ╔═╡ 028cdd65-2e34-574e-81e2-fdff8ff4e78d
check_simulate(simulate_chain)

# ╔═╡ 229acdda-115f-51f8-8e4c-f8b908902db1
check_after(distribution_after)

# ╔═╡ f70e9532-2c5f-579c-a9c8-d0f6a5512952
check_power(stationary_power)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "6ad4d1938234426f2b0e5d9acc6813a90d928bb0"

[[deps.AbstractPlutoDingetjes]]
git-tree-sha1 = "e71ee7b4aa06b045259a7d6101e1cb45ad140bce"
registries = "General"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.4.1"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.5.5+2"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
registries = "General"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
registries = "General"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.8+0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "e189d0623e7ce9c37389bac17e80aac3b0302e75"
registries = "General"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.83"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
registries = "General"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "1.0.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
registries = "General"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

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
# ╟─fbcd1505-6d97-57ec-8d50-890c01db1ee0
# ╟─6c09197f-3d98-54a1-94f9-9047b2005a46
# ╟─7c5135dd-3b5e-5c9a-a1ad-7eb3b034f30b
# ╟─9a2fa3ab-a6ff-560b-af72-7b3de5b77ac4
# ╟─173b4138-5a7a-5c9b-adca-9e44dc21dfb7
# ╟─12d6ecd7-5a90-5d99-ac27-ea8d2e79e8f7
# ╟─67599d4a-1700-529b-af44-50887299ac69
# ╟─69cd095c-74f5-5d53-8cc6-7c2b577c1e31
# ╟─54308917-b0c7-532d-a6ec-03728a6a5b0a
# ╟─dab72e64-ba25-5367-9248-00db2044fef9
# ╟─067c0ccc-53dd-5c22-b980-506f821ca184
# ╟─15132a2d-72e9-5520-b5e6-d871850b96f0
# ╟─dbf76961-8cb4-5cbd-9fc8-83a637e782ff
# ╟─d4feb79b-dfa0-54af-84b2-6212d3d955fd
# ╟─d5132e9a-4831-569f-a096-62737eaaf5e5
# ╟─f502a389-a09c-5bb5-bdd0-f7e8afbc8d18
# ╟─0c3f7523-4219-5afc-84b0-c9f0de7bbc9c
# ╟─08fc849e-be46-5664-af48-e0d0e43bb63f
# ╟─024bb141-72c1-50ee-bded-0ab124965f51
# ╟─48884c22-0e49-5773-b8c3-2a90f010b51a
# ╟─d912443f-9594-5a79-9902-abeda451aa67
# ╟─89e1eec7-bae1-5a50-a15b-581c444f2b69
# ╟─4347d96d-7e4e-53e9-a12f-8c0b09d1217d
# ╟─82c59329-3378-5a62-ad9a-c8ce00139217
# ╟─67add958-bbb4-5765-952f-e76af480fb8d
# ╟─7b1ebe0a-4337-5f04-8f87-953e0089b7b3
# ╟─ff9c1ef1-7949-5bdb-803c-32f0a6001ca7
# ╟─6a6fc898-9105-5dc6-b2e9-244b339d743a
# ╟─ec1eb4fe-b24c-59e4-80eb-34ffad90c957
# ╟─3eb00177-1a76-58f6-be1f-cf09baed9850
# ╟─6ec8f671-49bf-5fde-8c87-b3de4a7e2966
# ╟─106a1fbd-14a1-56e8-bc76-9bbf9c742dd0
# ╟─5bc81e14-9e62-5843-9bd1-327876aae4c7
# ╟─5934928c-510f-55e0-834f-5b644341515d
# ╟─ae8d3fc6-7f2b-5171-81ac-ee0490aee440
# ╟─5c6c8894-ee37-5fa4-8334-6f331a991f72
# ╟─e4a357c0-ca5c-5ee0-9acc-609a2a24c607
# ╟─0c797968-7e82-57cd-98a2-b5902e4f8ef1
# ╟─efb02241-76cc-5fe6-9e93-1d9bf58ec76f
# ╟─a11433f6-b90f-4df3-bb25-8e62382093a0
# ╟─39aa462a-5e19-408c-ae76-ee735f29611c
# ╟─7b344dd1-555d-48b8-821d-171033e436aa
# ╟─a6c3bfb6-ac6d-596a-9248-91e3c3e4c76e
# ╠═c27ca38d-d585-5585-aad9-49a9e817e79b
# ╟─403db6b6-983d-5c1b-93ea-267cda5d8af0
# ╟─c685eb82-0a7b-53a3-bbf0-e52cb3d828e2
# ╟─400c407d-2c73-52e3-a9ea-e865377e916c
# ╟─2f30f87f-4672-5209-a2e2-e2796119a911
# ╟─d0bd4d33-3397-541a-8d47-df03d93588b9
# ╟─47a76164-e6ea-5f7d-833d-8a18949c1beb
# ╟─c4ee1c46-0f46-5ebf-8d55-1e7e7d7a8410
# ╟─0a87c857-79be-5e51-bbdc-11e81c4764f4
# ╟─04532a1c-d900-5ee9-be18-dab9d329b891
# ╟─e0ea0af5-dff7-550e-87d6-8dfb420feb0b
# ╟─0caf768a-0e16-528f-9d82-7337aab2c233
# ╟─b3de8c6a-34ca-51ee-b232-e4af621d1b05
# ╟─013b4e5b-979e-56b0-a2f7-302d6993eefb
# ╟─9e697e7d-e084-597c-88cb-3198deab1401
# ╟─d5fb6239-c0aa-52a9-9b36-b0216f9cc453
# ╟─4bf301e9-1340-57e5-b0a4-5dd90098f5be
# ╟─1dad2f06-0327-5c17-a470-c23466178964
# ╟─d7bbb98f-7532-539e-9060-00f31cf55aa7
# ╟─ca4edfce-0c25-53c3-8f05-b7eab93e9e82
# ╟─6bec7e8e-7bf2-542a-877b-b0a2115dd87b
# ╟─5de3b9d1-4a34-5aed-ae11-deaa1e4b4994
# ╟─011683d4-c23b-53d1-aaa6-88da20b45103
# ╟─2dc2b02c-584e-426a-a455-9630abebf04d
# ╟─a764371f-6172-4770-8fc2-69e9aed97b24
# ╟─2924eb10-59ad-4572-82d4-5b5c25af5a59
# ╟─d62ea4c0-59f4-4a75-b077-77b629f008c5
# ╟─5586aa8e-bc3a-4145-8051-2cdb51bc932e
# ╟─fa0107ec-66d4-5ad8-9040-27d786245879
# ╟─38f02ca2-6dd2-5bb6-8323-b2c165841643
# ╟─c42d9641-d5fb-489b-a79b-52da695ccf65
# ╟─d135773c-f229-4b8f-b26c-bbfb79a79298
# ╟─9c7d85da-3170-4fcc-a138-f610cffee1cf
# ╟─526c7cda-1e09-566d-b386-9c92a951727b
# ╟─55bfec8c-bb5c-5886-8c90-346673c23e24
# ╟─c0effd89-d534-4cb4-9dcf-b3619f3c5ab9
# ╟─fe0d493f-fb6e-4ac1-9f47-cdbe46d886c6
# ╟─50aa9253-3632-480e-969a-acf68d727cc4
# ╟─b6ce307d-ec85-4e4c-af74-759aad7cc68f
# ╟─6bdd1741-0959-4ad7-86d2-86c35ecf3161
# ╟─2a76ae63-c44c-410c-b498-784caa67c8e6
# ╟─54cc2434-21b7-44f7-9c95-108504b924e4
# ╟─b99d244d-c8ff-49ad-81af-b66fbed9e35a
# ╟─604e0caa-fc9c-4368-a1e4-ae801f3eb199
# ╟─a9060a31-c192-4373-9628-56d3fef27c09
# ╟─695db1a8-9a03-4092-87eb-f934b144c79d
# ╟─e8ac34ca-7c63-4816-9ca0-e856c1658e8b
# ╟─6e8aecd6-6f44-45c3-b7df-7d119dc29086
# ╟─f4cb8951-f5e1-452d-b24c-70eb2ebad9e8
# ╟─81153baf-bd45-4ac6-98b3-b5cd3b2e64a6
# ╟─f33a5b05-dae3-47f7-9c91-416ee1222614
# ╟─bd8fdefe-47b8-447a-8434-d3fcdfaf74ca
# ╟─8448bc74-05eb-5312-b41a-30cc5ac050a4
# ╟─f86b8b72-25c2-5347-9e9b-ff277850ee9d
# ╟─e06f8c76-580a-5403-be52-143a181045b4
# ╟─1e5a2cf0-64a1-5e0d-9022-ee2566a71f13
# ╟─71d608b1-4677-5061-875b-6a676b8f28fc
# ╟─4a09ac4a-45a6-53dd-94b2-8f143c042c94
# ╠═8836a53d-2c65-5a2b-8a65-5915655488fb
# ╟─6997c198-aa84-54be-a811-a4fc37a3ffed
# ╟─756c7eef-e0ef-5f2f-959b-a0160ecd591f
# ╠═c13cda56-0424-553d-bbb2-9304cc243e93
# ╟─028cdd65-2e34-574e-81e2-fdff8ff4e78d
# ╟─9c1bf37e-d165-58ab-a7ff-e7cfff82ce2a
# ╠═0ae8b55a-8e0c-569e-8ad1-f023348cf176
# ╟─229acdda-115f-51f8-8e4c-f8b908902db1
# ╟─d66c5fca-a098-5ead-be72-e0f55bef477e
# ╠═d7d594a3-8dd8-583d-8ffb-1218a9650f1d
# ╟─f70e9532-2c5f-579c-a9c8-d0f6a5512952
# ╟─b81388ba-addc-5606-b0ff-490c2648e13d
# ╟─1a1fb627-3ddb-58ca-8702-671a29f5e45e
# ╟─676699ad-6d5a-5785-b0cf-36422c57c8ab
# ╟─1416ee70-0bd9-5f50-a1d2-64ff3e152e45
# ╟─700553cc-975f-57b9-bfb1-ab8cf122591d
# ╟─6fedd8f9-68da-5565-ad6e-09a8ddd985bb
# ╟─bc555a3a-7d39-5ea1-b11a-ab81035d3096
# ╟─fb9f5687-b340-51f4-91f4-f0e87ce9482c
# ╟─112744d4-9679-5a61-bdb4-4466114d8658
# ╟─4c9e90da-4039-542e-8155-ce7ddb8d8758
# ╟─a67909c4-72a1-57f7-9ccd-d1b5a9f58bb6
# ╟─78b7470a-7c29-5c5b-a480-e6b7553031aa
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
