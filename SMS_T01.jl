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

# ╔═╡ b3f096c7-22ad-55a4-9eeb-ef56384fd1b9
begin
	using PlutoUI, PlutoPlotly, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions
	import AbstractPlutoDingetjes
end

# ╔═╡ 589b67a3-92c5-523e-8160-a6dd640b2a79
md"""
# Tutorial 01: Probability Refresher 2 - Random Variables and Distributions
*Stochastic Modeling and Simulation*\
Date: 29.10.2026"""

# ╔═╡ cba88b0d-ce76-5bdd-afd5-bed6512f3f1e
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ df5332f8-c4dc-5f62-aa4c-be2bb5dadfb1
md"""

!!! info "From lecture 1"
    Conditional probability: ``P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}``, for ``P(B) > 0``.

    "``A`` and ``B``": ``P(A \cap B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)``.

    Law of total probability: ``P(A) = \sum_i P(A \mid B_i)\,P(B_i)``, for disjoint ``B_1, \dots, B_n`` that together cover ``\Omega``.

    Bayes' theorem: ``P(A \mid B) = \dfrac{P(B \mid A)\,P(A)}{P(B)}``.

    Independent events: ``P(A \cap B) = P(A)\,P(B)``, equivalently for indepedent events: ``P(A \mid B) = P(A)``.
    
    Disjoint events: ``P(A \cap B) = 0``.
"""

# ╔═╡ d35ae31a-df12-52be-9e35-95cf1ced49f4
md"""

!!! definition "And, given, and sequences"

    - ``P(A \cap B)``, "``A`` **and** ``B``": both events happen. The order doesn't matter: ``P(A \cap B) = P(B \cap A)``.
    - ``P(A \mid B)``, "``A`` **given** ``B``": ``B`` is known to have happened, and we ask how likely ``A`` is in that case. The order matters: in general ``P(A \mid B) \ne P(B \mid A)``.
    - ``P(X_1 = a,\ X_2 = b)``, a **sequence**: "``a`` on day 1, **then** ``b`` on day 2". The comma lists the values of ``X_1, X_2`` in order, so swapping the **values** gives a different sequence: in general ``P(X_1 = a,\ X_2 = b) \ne P(X_1 = b,\ X_2 = a)``.

!!! tasks "Tasks"
    Choose two different values ``a \ne b``. Write down ``P(X_1 = a,\ X_2 = b)`` and ``P(X_1 = b,\ X_2 = a)``. Are they equal in general? Explain.
"""

# ╔═╡ fca0d34d-333c-502e-957f-4891fbe8892b
md"""
!!! definition "Chain rule"
    For any events ``A_1, A_2, \dots, A_n``,
    ```math
    \begin{aligned} P(A_1 \cap A_2 \cap \dots \cap A_n) = {} & P(A_1)\,P(A_2 \mid A_1)\,P(A_3 \mid A_1 \cap A_2) \\ & \cdots P(A_n \mid A_1 \cap \dots \cap A_{n-1}). \end{aligned}
    ```
    Each factor is the probability of the next event, given **all** events before it.
"""

# ╔═╡ 4298582e-7132-53cd-b782-4f623baf532b
md"""
### 1.4 Memory

The last factor of Figure 1.3 depends on both earlier days. A model can also depend on many or no previous values:

```math
P(X_3 \mid {\color{#6a4c93}X_1, X_2}) = \begin{cases} P(X_3) & \text{no memory: the past does not matter,} \\ P(X_3 \mid {\color{#6a4c93}X_2}) & \text{memory of the last day,} \\ P(X_3 \mid {\color{#6a4c93}X_1, X_2}) & \text{memory of all earlier days.} \end{cases}
```
"""

# ╔═╡ 2f4aea61-4ef5-57bf-bcbc-795449cbf4e9
# ╠═╡ disabled = true
#=╠═╡
mcq(md"A model describes 10 days, each sun or rain. With memory of the last day, how many different conditional distributions are needed?",
	md"3: one for day 1 and one for each possible weather of the day before." => md"Right. The same two conditional distributions are used for every day from day 2 on.",
	md"10: one per day." => md"With memory of the last day, day 5 uses the same numbers as day 2: only the previous weather matters.",
	md"``2^{9}``" => md"That many would be needed for day 10 alone with memory of **all** earlier days.";
	correct = 1, head = "Question 1.14")
  ╠═╡ =#

# ╔═╡ 355e9c9a-3643-5e4e-92a7-fee9a638ae15
md"""
!!! question "Question 1.15"
    With no memory, the chain rule becomes ``P(A_1 \cap A_2 \cap A_3) = P(A_1)\,P(A_2)\,P(A_3)``. Which property of events does this describe?
"""

# ╔═╡ 847fa071-8823-4007-a5c1-5e5be69cd5e6
md"""
Suppose the first ``i`` people all have different birthdays. They occupy ``i`` days of the calendar, so ``365 - i`` days are still free. Person ``i + 1`` has 365 equally likely birthdays; the probability to land on a free day is

```math
P(\text{person } i + 1 \text{ avoids the first } i \mid \text{the first } i \text{ are all different}) = \frac{365 - i}{365}.
```

Person 1 has nothing to avoid: ``365/365 = 1``. Person 2 avoids one day: ``364/365``. Person 3 avoids two days: ``363/365``. The chain rule (Section 1.3) multiplies these conditional probabilities:

```math
P(\text{all } n \text{ different}) = \frac{365}{365} \cdot \frac{364}{365} \cdot \frac{363}{365} \cdots \frac{365 - (n - 1)}{365},
```
```math
P(\text{at least two share}) = 1 - P(\text{all } n \text{ different}).
```

Each factor is a **conditional** probability: it is computed knowing that all earlier birthdays are different. 
"""

# ╔═╡ 1c380d45-b7f0-5021-bfd5-c708e95b07f3
md"""
!!! info "Key points"
    - ``P(A \cap B) = P(B \cap A)`` always. ``P(A \mid B)`` and ``P(B \mid A)`` differ in general. A sequence ``P(X_1 = a, X_2 = b)`` changes when the values are swapped.
    - Chain rule: ``P(A_1 \cap \dots \cap A_n) = P(A_1)\,P(A_2 \mid A_1) \cdots P(A_n \mid A_1 \cap \dots \cap A_{n-1})``.
"""

# ╔═╡ 013acf06-3812-56d3-be62-b643b5fbf03d
md"""
!!! definition "Random variable"
    A random variable is a number assigned to each possible outcome of an experiment: a function
    ```math
    X : \Omega \to \mathbb{R}, \qquad \omega \mapsto X(\omega).
    ```
    Random variables are written with capital letters (``X``), their values with lower-case letters (``x``). ``\{X = x\}`` is the event that ``X`` takes the value ``x``.

The function is fixed; the randomness lies in the outcome ``\omega``. Figure 2.2 shows two of the functions of Figure 2.1 for the six faces.
"""

# ╔═╡ 151e32de-e8d0-56a0-b035-642c0229267b
md"""
Classify the following quantities.
"""

# ╔═╡ fa0ee360-7d3d-55a1-8540-c002da7a2647
md"""
### 2.3 Reading a discrete distribution

For a discrete random variable, a distribution plot has the **possible values** ``x`` on the horizontal axis and the **probability** ``P(X = x)`` as the height of a bar.
"""

# ╔═╡ c4d90ed1-fc54-515e-bd4d-f684cd6ae159
md"""
!!! info "Key points"
    - A random variable is a function ``X : \Omega \to \mathbb{R}``; ``x`` is one of its values.
    - One experiment can carry many random variables, each with its own distribution.
    - Binary: values 0 and 1. Discrete: values that can be listed. Continuous: values in an interval.
"""

# ╔═╡ a7bb2f07-c717-596a-845e-2ee6b9492ad1
md"""
## 3. Discrete distributions

Let ``X`` be the sum of two fair dice. How likely is each sum?

### 3.1 The PMF: counting outcomes

The 36 ordered pairs of faces are equally likely. Choose a sum ``x`` and count the cells.

Sum ``x`` $(@bind s3_s Slider(2:12; default = 4, show_value = true))
"""

# ╔═╡ 9916a4af-0b65-5ada-babd-e07ab261f83f
md"""
!!! definition "Probability mass function (PMF)"
    For a discrete random variable ``X`` with values ``x_1, x_2, \dots``,
    ```math
    p_X(x_i) = P(X = x_i), \qquad 0 \le p_X(x_i) \le 1, \qquad \sum_i p_X(x_i) = 1.
    ```

### 3.2 The CDF

How likely is a sum of **at most** ``x``? 

``x`` $(@bind s3_x Slider(1:0.5:12.5; default = 4, show_value = true))
"""

# ╔═╡ d348ba24-657c-53fb-b4c3-0012ca400150
md"""
!!! definition "Cumulative distribution function (CDF)"
    ```math
    F_X(x) = P(X \le x) = \sum_{x_i \le x} p_X(x_i).
    ```
    Properties: ``0 \le F_X(x) \le 1``; ``F_X`` never decreases; ``F_X(x) = 0`` below the smallest value and ``1`` above the largest; at each value ``x_i`` it jumps by ``p_X(x_i)``, and between the values it is flat.
"""

# ╔═╡ a3b233b2-495b-5638-9d03-b4750091ea0d
md"""
### 3.3 Probabilities of intervals

With strict (``<``) and non-strict (``\le``) inequalities, different bars are counted. Choose the inequalities and the limits: the bars counted turn dark, and the same probability is a rise of the CDF.

``a`` $(@bind s3_a Slider(2:12; default = 3, show_value = true)) $(@bind s3_lop Select(["<", "≤"]; default = "<")) ``X`` $(@bind s3_rop Select(["<", "≤"]; default = "≤")) ``b`` $(@bind s3_b Slider(2:12; default = 6, show_value = true))
"""

# ╔═╡ 4680ba24-b816-5bc5-9c1a-87142fd24bb0
md"""
!!! definition "Intervals from the CDF"
    For integer values,
    ```math
    P(a < X \le b) = F_X(b) - F_X(a), \qquad P(a \le X \le b) = F_X(b) - F_X(a - 1),
    ```
    ```math
    P(X > b) = 1 - F_X(b), \qquad P(X < b) = F_X(b - 1).
    ```
"""

# ╔═╡ 12c6e558-cf7f-533a-bc85-ee9164181175
md"""
!!! info "Key points"
    - PMF ``p_X(x) = P(X = x)``: count or compute the probability of each value.
    - CDF ``F_X(x) = P(X \le x)``: add up the bars.
    - Intervals: ``P(a < X \le b) = F_X(b) - F_X(a)``.
    - Joint table, marginals as sums, conditionals as rescaled rows.
"""

# ╔═╡ 8bdecce6-d833-5a8f-a0fa-eee6427adfce
md"""
## 4. Continuous distributions

### 4.1 From counts to a density

What will the temperature be tomorrow at noon? The temperature is a random variable ``X``, with values in an interval of °C. To learn its distribution, the experiment is repeated: the noon temperature is recorded on many days.
"""

# ╔═╡ 4f11fe3e-6eb3-56f2-82f4-56673834d95a
md"""
The fraction of days per °C is a **density**. For narrow bins of width ``\Delta x`` around ``x``, the fraction of days in the bin is close to ``f_X(x)\,\Delta x``.

!!! definition "Probability density function (PDF)"
    A continuous random variable ``X`` has the density ``f_X`` if, for all ``a \le b``,
    ```math
    P(a \le X \le b) = \int_a^b f_X(x)\,dx .
    ```
    Then ``f_X(x) \ge 0``, ``\int_{-\infty}^{\infty} f_X(x)\,dx = 1``, the CDF is ``F_X(x) = P(X \le x) = \int_{-\infty}^{x} f_X(\tilde x)\,d\tilde x``, and ``f_X(x) = \dfrac{dF_X(x)}{dx}``.
"""

# ╔═╡ 0db53489-210c-4fb1-a1cd-ec1e47e97d38
md"""
!!! definition "Reading a distribution plot"
    - **Discrete:** the **height** of a bar is a probability, ``p_X(x) = P(X = x)``. 
    - **Continuous:** the height ``f_X(x)`` is **not** a probability. A probability is an **area** under the curve, ``P(a \le X \le b) = \int_a^b f_X(x)\,dx``. For a narrow interval, area ``\approx`` height × width: ``P(x \le X \le x + \Delta x) \approx f_X(x)\,\Delta x``.
    - A single value has width 0, so ``P(X = x) = 0`` for a continuous random variable.
"""

# ╔═╡ 997f9358-5c4c-523c-a346-21694667b207
md"""
### 4.5 Two continuous random variables

A penalty kick crosses the goal line at a point ``(X, Y)``: sideways position and height. Press ▶ and watch the kicks of a kicker pile up.
"""

# ╔═╡ 9881f340-a35a-5984-a10b-5f9b49cc0348
md"""
Where the kicks are dense, the probability is high. A function of **two** variables ,the joint density, describes this. In 3D it is a mountain over the goal (rotate it with the mouse).
"""

# ╔═╡ 430216b4-a145-5ed3-b688-de7d9a783a45
md"""
**Marginals.** The marginal density squashes the joint density onto one axis: it integrates over the other variable.

Show the CDFs instead $(@bind s4_mcdf CheckBox(default = false))
"""

# ╔═╡ e9415266-6f6a-53b8-ae3b-49f9d46b28ac
md"""
**Conditionals.** The conditional density of ``X`` given ``Y = y`` is a slice of the joint density at height ``y``, divided by the area of the slice, ``f_Y(y)``.

Position of the slice $(@bind s4_t Slider(0.02:0.02:0.98; default = 0.7)) ``\qquad`` 3D view of the slice $(@bind s4_s3d CheckBox(default = false))
"""

# ╔═╡ 3b6d0a8e-842c-5c13-bc36-42369de45a42
md"""
**The joint CDF.** ``F_{X,Y}(x, y)`` is the volume below and to the left of the point ``(x, y)``. Moving ``y`` to the top gives the marginal CDF: ``F_X(x) = F_{X,Y}(x, \infty)``.

Position of the point: ``x`` $(@bind s4_cx Slider(0:0.02:1; default = 0.6)) ``\qquad`` ``y`` $(@bind s4_cy Slider(0:0.02:1; default = 0.6))
"""

# ╔═╡ 54c262ee-e7ba-51fd-b464-0e5767201186
md"""
!!! info "Key points"
    - ``P(a \le X \le b) = \int_a^b f_X(x)\,dx = F_X(b) - F_X(a)``. ``P(X = x) = 0``; ``f_X`` can exceed 1.
    - Reading the CDF backwards, ``x = F_X^{-1}(u)``, turns uniform heights into values with the distribution of ``X``.
"""

# ╔═╡ 0baab108-be3a-5488-b65c-212518240d4e
md"""
### 5.3 Matching moments one by one

The mean and the variance are the moments of order 1 and 2. Each order measures one more property of the shape:

!!! definition "What each moment measures"
    - Order 0: ``\int f_X(x)\,dx = 1``, the **area**: the total probability.
    - Order 1: ``E[X]``, the **centroid**: the point where the area balances.
    - Order 2: ``\mathrm{Var}(X) = E\big[(X - E[X])^2\big]``, the **spread** around the centroid (in mechanics: the moment of inertia).
    - Order 3: the skewness ``E\big[(X - E[X])^3\big]/\sigma^3``, the **asymmetry**: which side has the longer tail.
    - Order 4: the kurtosis ``E\big[(X - E[X])^4\big]/\sigma^4``, the **tails**: how much probability lies far from the centroid.
"""

# ╔═╡ a6df4399-8a0d-42c4-adba-f1c5b206c05d
md"""
!!! definition "Covariance"
    ```math
    \mathrm{Cov}(X, Y) = E\big[(X - E[X])(Y - E[Y])\big]
    ```
    - **Positive:** when ``X`` is above its mean, ``Y`` tends to be above its mean too. They increase together (more green points).
    - **Negative:** when ``X`` is above its mean, ``Y`` tends to be below its mean. One increases while the other decreases (more red points).
    - **Zero:** no such tendency, for example two separate dice. Independent random variables have ``\mathrm{Cov}(X, Y) = 0``.

    The covariance appears in the variance of a sum: ``\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)``.
"""

# ╔═╡ e634a810-f0bf-5e00-a472-113b0e3d3ecb
md"""
!!! info "Key points"
    - Moments summarise a distribution: area (order 0), centroid (1), spread (2), asymmetry (3), tails (4). The more moments two distributions share, the more alike they are.
    - ``\mathrm{Cov}(X, Y) > 0``: ``X`` and ``Y`` increase together. ``\mathrm{Cov}(X, Y) < 0``: one increases while the other decreases.
    - ``E[X]`` belongs to the distribution, ``\bar x`` to the data; ``\bar X_N \to E[X]``.
"""

# ╔═╡ 7a4fb405-604d-532d-b02d-0c2b56850c11
md"""
!!! info "Key points"
    - Binomial: the number of successes in ``n`` independent trials with two results. Uniform: all values in an interval equally likely. Normal: a symmetric bell curve around ``\mu``. Poisson: the number of events at a constant rate. Exponential: the waiting time until the next event.
    - Central limit theorem: the sum of many independent random variables with the same distribution is approximately normal, whatever that distribution is.
"""

# ╔═╡ 5db34b3e-5299-5ed3-8db1-d1b31332c573
md"""

### Case 2: the density can be evaluated, but not sampled directly

A hiker is lost in a 10 km × 10 km area. From the last phone signal, a map gives the probability density of the hiker's position at every point. A search simulation needs samples of possible positions.

Show 200 points spread uniformly over the area $(@bind s7_pts CheckBox(default = false))
"""

# ╔═╡ 1c590021-d415-5426-94ca-3af67bd1c293
md"""

### Case 3: the distribution is unknown

An archery target has a hidden score map: some regions give many points, but nobody knows where. Each arrow lands near its aim point and shows only its own score. You have 15 arrows. Type aim points as `x, y` pairs separated by semicolons, with ``x`` and ``y`` between ``-1`` and ``1``.

Aim points $(@bind s7_aims TextField(60; default = "0, 0; 0.5, 0.5; -0.5, -0.5")) ``\qquad`` reveal the score map $(@bind s7_rev CheckBox(default = false))
"""

# ╔═╡ 4b2ab322-b660-59d8-b738-77351d2036ac
md"""
!!! question "Question 7.3"
    Where should the next arrow go? How can the high-scoring regions be found, and how can the total score of 15 arrows be made as large as possible, when the map is unknown?
"""

# ╔═╡ 77b8c106-a6a1-532a-a1d4-b9c16d2b86be
md"""
## 9. Exercises

**Exercise 1 · Chain rule.** Three cards are drawn from a deck of 52 without replacement. Compute the probability that all three are hearts.
"""

# ╔═╡ ea8ce6c1-9f7b-5dbb-ac57-adb72842e1ea
md"""
**Exercise 2 · Sequences.** Using the weather model of Section 1.2, compute ``P(X_2 = \text{rain})`` and then ``P(X_1 = \text{sun} \mid X_2 = \text{rain})``.
"""

# ╔═╡ 75efa6e0-5409-5089-aadc-1c288cfc3df3
md"""
**Exercise 3 · Memory.** In the weather model with memory of the last day, compute ``P(X_1 = \text{sun}, X_2 = \text{sun}, X_3 = \text{sun})``.
"""

# ╔═╡ f095f16b-1941-5682-a9f2-84953cc1858d
md"""
**Exercise 4 · PMF and CDF.** A random variable takes the values 0, 1, 2, 3, 4, 5 with probabilities 0.1, 0.2, 0.3, 0.2, 0.1, 0.1. (a) Give ``F_X`` at 0, …, 5. (b) Compute ``P(2 \le X \le 4)`` and ``P(2 < X < 4)``.
"""

# ╔═╡ d6ecec15-b609-5af2-89a6-96abe072f1da
md"""
**Exercise 5 · A joint table.** Two binary random variables have ``P(0,0) = 0.3``, ``P(0,1) = 0.2``, ``P(1,0) = 0.1``, ``P(1,1) = 0.4`` (first argument ``X``, second ``Y``). Compute the marginals and ``P(Y = 1 \mid X = 1)``. Are ``X`` and ``Y`` independent?
"""

# ╔═╡ 659d2feb-a802-506c-8db3-d721d3bd3329
md"""
**Exercise 6 · A density.** ``f(x) = c\,x^2`` on ``[0, 3]``. (a) Find ``c``. (b) Give ``F_X``. (c) Compute ``P(X \le 1)``.
"""

# ╔═╡ 649fa6c2-8da8-5351-86c4-b273e198eda8
md"""
**Exercise 7 · A joint density.** ``f(x, y) = 4xy`` on the unit square. Compute ``f_X(x)`` and ``f_Y(y)``. Are ``X`` and ``Y`` independent? Compute ``P(X \le \tfrac12, Y \le \tfrac12)``.
"""

# ╔═╡ 029ba212-2bc6-5aa3-a21b-499672fb4baa
md"""
**Exercise 8 · Moments.** For the random variable of Exercise 4, compute ``E[X]``, ``M_2[X]`` and ``\mathrm{Var}(X)``. Then give the mean and the variance of ``Y = 3X - 2``.
"""

# ╔═╡ 479fca6c-815f-5079-a99a-dea660da9b09
md"""
**Exercise 10 · A normal random number from uniform ones (programming).** Let ``U_1, \dots, U_{12}`` be independent and uniform on ``[0, 1]``. (a) Show that ``Z = U_1 + \dots + U_{12} - 6`` has mean 0 and variance 1. (b) Complete `z12` so that it returns one value of ``Z``.
"""

# ╔═╡ b07d2299-bcd2-5f07-90ba-d02a6a054b4a
function z12(rng)
	u = rand(rng, 12)
	return missing   # your code here
end

# ╔═╡ 163717d0-f375-5456-bcbc-0f74af8185f8
md"""
## Further reading

- *Seeing Theory*, Brown University: interactive chapters on probability and distributions, <https://seeing-theory.brown.edu/>.
- J. K. Blitzstein and J. Hwang, *Introduction to Probability*, 2nd ed., CRC Press, 2019.

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 1.
"""

# ╔═╡ 16fc0065-e8b4-5a0d-8791-748b7d612f19
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
table.tbl{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:0.92rem;margin:6px 0}
table.tbl th{border-bottom:1px solid #1b1b1b;padding:2px 12px;text-align:right;font-weight:600}
table.tbl td{padding:2px 12px;text-align:right}
table.tbl td:first-child,table.tbl th:first-child{text-align:left}
table.tbl tr:nth-child(even) td{background:#f6f6f3}
details.hint{margin-top:4px}
details.hint summary{cursor:pointer;font-weight:600}
.admonition.tasks{border-color:#6a4c93;background:#f4f1f8}
.admonition.notation{border-color:#1f5aa6;background:#f3f6fb}
.admonition.notation>.admonition-title{background:#1f5aa6;color:#fff;font-family:monospace}
.admonition.tasks>.admonition-title{background:#6a4c93;color:#fff}
.pluto-docs-binding{display:none}
.admonition.tasks ol{margin:4px 0 2px 0}
ol.rvq li{margin:3px 0}
ol.rvq li span{display:inline-block;min-width:390px}
</style>"""

# ╔═╡ f585d0a7-289a-5c17-977c-31f12b029dbd
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

	row(items...; gap = "18px") = @htl("""<div class="fg-row" style="gap:6px $(gap)">$([@htl("<div>$(it)</div>") for it in items])</div>""")

	function histdata(xs, edges; density = true)
		c = zeros(length(edges) - 1)
		for x in xs
			i = searchsortedlast(edges, x)
			(1 <= i <= length(c)) && (c[i] += 1)
			x == edges[end] && (c[end] += 1)
		end
		density ? c ./ (length(xs) .* diff(edges)) : c
	end

	parse_numbers(s) = [parse(Float64, t) for t in split(replace(s, "," => " ")) if tryparse(Float64, t) !== nothing]
end;

# ╔═╡ 2b123d71-3924-5781-b9cd-d096dac34b6e
# Animations
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

# ╔═╡ 42476b37-1ce4-52ca-bc63-b775b9d44892
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

	hint(body) = reveal(body; title = "Hint")

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
end;

# ╔═╡ 4d9ded5f-68b5-5e3b-961e-9415756f7537
numq(md"What is ``P(\text{ill} \cap \text{positive})``, the probability that a person is ill **and** tests positive?", 0.045;
	why = md"45 of 1000 people are in this cell: ``45/1000 = 0.045``.", head = "Question 1.1")

# ╔═╡ e97f2520-fd0d-57c5-867b-58cc879a5629
mcq(md"Compare ``P(\text{ill} \cap \text{positive})`` and ``P(\text{positive} \cap \text{ill})``.",
	md"They are equal." => md"Right. Both describe the same people: those who are ill and test positive. The order of writing an intersection does not matter.",
	md"The first is larger." => md"Both are the same cell of the table, 45 people.",
	md"The second is larger." => md"Both are the same cell of the table, 45 people.";
	correct = 1, head = "Question 1.2")

# ╔═╡ 7459f9bc-1021-5dca-ba44-64150f91cfcb
numq(md"What is ``P(\text{positive} \mid \text{ill})``? Use the condition *ill* in Figure 1.1.", 0.9;
	why = md"Among the 50 ill people, 45 test positive: ``45/50 = 0.9``.", head = "Question 1.3")

# ╔═╡ 6e22c2bf-f19e-59f7-8ee1-eb79b9222ffd
numq(md"What is ``P(\text{ill} \mid \text{positive})``?", 45 / 140; rtol = 0.01,
	why = md"Among the 140 people who test positive, 45 are ill: ``45/140 = 0.321``. ", head = "Question 1.4")

# ╔═╡ 51a50d1a-f72c-571a-8601-14a14d7e0a17
numq(md"Compute ``P(\text{positive})`` with the law of total probability, from ``P(\text{positive} \mid \text{ill}) = 0.9``, ``P(\text{positive} \mid \text{healthy}) = 0.1`` and ``P(\text{ill}) = 0.05``.", 0.14; rtol = 0.01,
	why = md"``0.9 \cdot 0.05 + 0.1 \cdot 0.95 = 0.045 + 0.095 = 0.14``, which is ``140/1000``.", head = "Question 1.5")

# ╔═╡ bad79ded-b0c1-5d66-94a1-99375f0fb585
mcq(md"Are the events *ill* and *positive* independent?",
	md"Yes, because the test is applied to everybody." => md"Independence is a statement about probabilities, not about how the test is applied.",
	md"No, because ``P(\text{positive} \mid \text{ill}) = 0.9 \ne P(\text{positive}) = 0.14``." => md"Right. Knowing that a person is ill changes the probability of a positive test.",
	md"Yes, because ``P(\text{ill} \cap \text{positive}) = P(\text{positive} \cap \text{ill})``." => md"That equality holds for any two events. Independence requires ``P(\text{ill} \cap \text{positive}) = P(\text{ill})\,P(\text{positive}) = 0.007``, but the value is 0.045.";
	correct = 2, head = "Question 1.6")

# ╔═╡ eb526ce8-32a9-4cf5-a1c4-966ffce1617f
begin
	q8 = numq(md"What is ``P(X_1 = \text{rain},\ X_2 = \text{sun})``?", 0.12; rtol = 0.01,
		why = md"``0.3 \cdot 0.4 = 0.12``. The comma asks for the whole sequence: day 1 is rain **and** day 2 is sun.", head = "Question 1.8")

	q9 = numq(md"What is ``P(X_2 = \text{sun} \mid X_1 = \text{rain})``?", 0.4; rtol = 0.01,
		why = md"``0.4``, given directly. Day 1 is **known** to be rain, so the chance ``0.3`` of rain is not counted. This is why it differs from Question 1.8: ``0.12 = 0.3 \cdot 0.4``.", head = "Question 1.9")

	q10 = numq(md"What is ``P(X_1 = \text{sun},\ X_2 = \text{rain})``?", 0.14; rtol = 0.01,
		why = md"``0.7 \cdot 0.2 = 0.14``. The two sequences contain the same two weathers in a different order, and they have different probabilities (``0.14 \ne 0.12``).", head = "Question 1.10")

	q11 = numq(md"What is ``P(X_1 = \text{rain} \mid X_2 = \text{sun})``?", 0.12 / 0.68; rtol = 0.01,
		why = md"``P(X_2 = \text{sun}) = 0.56 + 0.12 = 0.68``, so ``P(X_1 = \text{rain} \mid X_2 = \text{sun}) = 0.12 / 0.68 \approx 0.18``. This is not ``P(X_2 = \text{sun} \mid X_1 = \text{rain}) = 0.4``: in general ``P(A \mid B) \ne P(B \mid A)``.", head = "Question 1.11")

	q12 = numq(md"What do you get when you add the probabilities of **all four** sequences?", 1.0; rtol = 0.01,
		why = md"``0.56 + 0.14 + 0.12 + 0.18 = 1``. Every outcome of the two days is exactly one sequence.", head = "Question 1.12")

	PlutoUI.ExperimentalLayout.vbox([q8, q9, q10, q11, q12])
end

# ╔═╡ 081d0c9d-1053-50ad-aba1-4c04e7097370
mcq(md"Which pair of probabilities is **always** equal?",
	md"``P(X_1 = \text{sun},\ X_2 = \text{rain})`` and ``P(X_1 = \text{rain},\ X_2 = \text{sun})``" => md"These are two different sequences: 0.14 and 0.12.",
	md"``P(\{X_1 = \text{sun}\} \cap \{X_2 = \text{rain}\})`` and ``P(\{X_2 = \text{rain}\} \cap \{X_1 = \text{sun}\})``" => md"Right. Both describe the same days: sun on day 1 and rain on day 2. Only the order of writing differs.",
	md"``P(X_2 = \text{rain} \mid X_1 = \text{sun})`` and ``P(X_1 = \text{sun} \mid X_2 = \text{rain})``" => md"The first is 0.2. The second asks about day 1 given day 2: ``0.14 / P(X_2 = \text{rain}) = 0.14/0.32 = 0.44``.";
	correct = 2, head = "Question 1.9")

# ╔═╡ eef9e620-0813-540e-9425-fb091c5f61c1
numq(md"Use Figure 1.3: compute ``P(X_1 = \text{sun},\ X_2 = \text{sun},\ X_3 = \text{rain})``.", 0.7 * 0.8 * 0.15; rtol = 0.01,
	why = md"``0.7 \cdot 0.8 \cdot 0.15 = 0.084``.", head = "Question 1.11")

# ╔═╡ c95211eb-34df-5779-b4ec-8d718d95ee9a
numq(md"Add the probabilities of all eight sequences of Figure 1.3.", 1.0; rtol = 0.001,
	why = md"The eight sequences are all possible outcomes of three days and exclude each other, so their probabilities add up to 1.", head = "Question 1.12")

# ╔═╡ 612bca3d-1a69-5202-a88b-69bd79c5a3f2
mcq(md"How many factors does the chain rule have for a sequence of 10 days?",
	md"2" => md"Each day contributes one factor.",
	md"10" => md"Right: one factor per day; the last one is conditioned on the 9 days before it.",
	md"``2^{10}``" => md"``2^{10} = 1024`` is the number of possible **sequences**, not the number of factors in one product.";
	correct = 2, head = "Question 1.13")

# ╔═╡ 8f04976e-5182-56b6-93ed-44a4e4518728
reveal(md"Independence: ``P(A_2 \mid A_1) = P(A_2)``, so the probability of all events is the product of their probabilities.")

# ╔═╡ 049d987c-e962-450d-b212-ff6fc028ffc3
mcq(md"In Figure 1.5, the blue curve has the value 0.71 at ``n = 30``. What does this mean?",
	md"Person 30 shares a birthday with someone with probability 0.71." => md"No. Person 30 alone shares a birthday with one of the 29 others with probability ``1 - (364/365)^{29} = 0.077``.",
	md"In a room of exactly 30 people, at least two share a birthday with probability 0.71." => md"Right. In about 71 % of all rooms of 30 people there is a shared birthday.",
	md"In one room, the probability rises to 0.71 when the 30th person enters." => md"The curve compares rooms of different sizes; it does not follow one room in time.";
	correct = 2, head = "Question 1.17")

# ╔═╡ 72e186ee-b9bf-5fcc-a99a-f3fa45211168
numq(md"From Figure 1.5: for which room size ``n`` does ``P(\text{at least two share})`` first exceed 1/2?", 23; rtol = 0.0, atol = 0.5,
	why = md"For ``n = 23``, ``P(\text{at least two share}) = 0.507``.", head = "Question 1.18")

# ╔═╡ e12356a5-a1e1-546c-af82-72df1bc98576
mcq(md"Which statement is correct?",
	md"``X`` is a probability." => md"``X`` assigns numbers to outcomes. ``P(X = x)`` is a probability.",
	md"``X`` is a function of the outcome; ``x`` is one of its possible values." => md"Right. For example ``X(\omega) = 4`` if face 4 is on top, and ``P(X = 4) = 1/6``.",
	md"``X`` and ``x`` are the same." => md"``X`` is random; ``x`` is a fixed number such as 4.";
	correct = 2, head = "Question 2.1")

# ╔═╡ 717322d6-99a0-54ba-9ae0-a0abe273d37a
mcq(md"The number of cars in a car park with 500 places can take 501 values. What type of random variable is it?",
	md"Continuous, because there are very many values." => md"The number of values is not the criterion. The values 0, 1, …, 500 can be listed.",
	md"Discrete." => md"Right. Its values can be listed one by one.",
	md"Binary." => md"A binary random variable has only two values.";
	correct = 2, head = "Question 2.2")

# ╔═╡ 6896481b-4336-5c6c-b60b-d8ffea83f478
mcq(md"A thermometer shows body temperatures rounded to 0.1 °C. Is the body temperature discrete?",
	md"Yes, because the thermometer shows only a list of values." => md"The display is rounded; the quantity itself can take any value in an interval.",
	md"No, it is continuous; only its display is rounded." => md"Right. Rounding is a property of the measurement, not of the quantity.";
	correct = 2, head = "Question 2.3")

# ╔═╡ 8273806c-9697-5d2c-a18f-a9d25399f84e
numq(md"Read Figure 2.4: what is ``P(X = 8)``?", 5 / 36; rtol = 0.01,
	why = md"The bar at 8 has the height ``5/36 = 0.139``.", head = "Question 2.4")

# ╔═╡ 69d12213-09d4-5398-963e-7b895a9f83c6
numq(md"Add the heights of all bars of Figure 2.4.", 1.0; rtol = 0.001,
	why = md"The values 2, …, 12 are all possible sums and exclude each other, so their probabilities add up to 1.", head = "Question 2.5")

# ╔═╡ 7d7abd6e-6f5d-5822-8fde-1b9e113ea208
numq(md"How many of the 36 outcomes give the sum 7? Enter ``P(X = 7)``.", 6 / 36; rtol = 0.01,
	why = md"``(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)``: 6 outcomes, so ``P(X = 7) = 6/36 = 0.167``.", head = "Question 3.1")

# ╔═╡ c31bb482-89df-5b63-8dc5-7d4cb534ac09
numq(md"Compute ``F_X(4)``.", 6 / 36; rtol = 0.01,
	why = md"``p_X(2) + p_X(3) + p_X(4) = (1 + 2 + 3)/36 = 6/36``.", head = "Question 3.2")

# ╔═╡ 508070f9-631e-587f-b7da-689134abb430
numq(md"Compute ``F_X(4.5)``.", 6 / 36; rtol = 0.01,
	why = md"``F_X(4.5) = F_X(4) = 6/36``.", head = "Question 3.3")

# ╔═╡ d4b7046f-bc26-5ec0-be4e-0702c45c5500
mcq(md"How large is the jump of ``F_X`` at ``x = 7``?",
	md"``F_X(7)``" => md"``F_X(7) = 21/36`` is the height after the jump, not the jump.",
	md"``p_X(7) = 6/36``" => md"Right. The jump at ``x_i`` is the bar ``p_X(x_i)``.",
	md"``1/11``, because there are 11 values." => md"The 11 values are not equally likely.";
	correct = 2, head = "Question 3.4")

# ╔═╡ 8c73d868-8574-5372-8f97-f7ed0a03369d
numq(md"Compute ``P(3 < X \le 6)``.", 12 / 36; rtol = 0.01,
	why = md"``F_X(6) - F_X(3) = 15/36 - 3/36 = 12/36``: the sums 4, 5, 6.", head = "Question 3.5")

# ╔═╡ 4fde056f-95bc-561b-a5e0-aa9289576627
numq(md"Compute ``P(3 \le X < 6)``.", 9 / 36; rtol = 0.01,
	why = md"The sums 3, 4, 5: ``F_X(5) - F_X(2) = 10/36 - 1/36 = 9/36``.", head = "Question 3.6")

# ╔═╡ d9a548b3-0a54-587c-aa8d-8c62e390acb1
numq(md"Compute ``P(X > 9)``.", 6 / 36; rtol = 0.01,
	why = md"``1 - F_X(9) = 1 - 30/36 = 6/36``: the sums 10, 11, 12.", head = "Question 3.7")

# ╔═╡ e2045581-bb21-586f-9003-8a2473921f83
mcq(md"Which expression equals ``P(4 \le X \le 7)``?",
	md"``F_X(7) - F_X(4)``" => md"This is ``P(4 < X \le 7)``: the bar at 4 is missing, because ``F_X(4)`` already contains it.",
	md"``F_X(7) - F_X(3)``" => md"Right. ``F_X(3)`` removes the sums 2 and 3 and keeps 4.",
	md"``p_X(4) + p_X(7)``" => md"This leaves out 5 and 6.";
	correct = 2, head = "Question 3.8")

# ╔═╡ 018e7f5e-df50-5d95-a88c-b5694adfb4d0
numq(md"``X`` = first die, ``Y`` = sum of both dice. Compute ``P(Y = 9 \mid X = 4)``.", 1 / 6; rtol = 0.01,
	why = md"Given ``X = 4``, the sum is 9 exactly if the second die shows 5.", head = "Question 3.9")

# ╔═╡ 769e597a-1fa2-555e-ae84-157035fa3afa
numq(md"Same pair: compute ``P_Y(9)``, the marginal probability of the sum 9.", 4 / 36; rtol = 0.01,
	why = md"Sum of the column ``y = 9``: four cells of ``1/36``.", head = "Question 3.10")

# ╔═╡ ea499cfc-9462-5b30-a99b-0dc5fae42230
mcq(md"Are the first die and the sum independent?",
	md"Yes, the dice do not influence each other." => md"The two **dice** are independent, but the sum contains the first die.",
	md"No: ``P(Y = 9 \mid X = 4) = 1/6 \ne P_Y(9) = 1/9``." => md"Right. Knowing ``X`` changes the distribution of ``Y``.";
	correct = 2, head = "Question 3.11")

# ╔═╡ d36163ad-52fe-5e2b-8191-ae91ded8c137
numq(md"``X`` is uniform on ``[0, 10]``, so ``f_X(x) = 1/10``. Compute ``P(2 \le X \le 5)``.", 0.3;
	why = md"``\int_2^5 \frac{1}{10}\,dx = \frac{5 - 2}{10} = 0.3``: a rectangle of width 3 and height 0.1.", head = "Question 4.4")

# ╔═╡ ac2dfb9b-ed01-5d18-8a56-49eb4c69825c
numq(md"``f(x) = 2x`` on ``[0, 1]``. First check that the total area is 1. Then compute ``P(0.2 \le X \le 0.5)``.", 0.21;
	why = md"``\int_0^1 2x\,dx = [x^2]_0^1 = 1``. ``\int_{0.2}^{0.5} 2x\,dx = 0.5^2 - 0.2^2 = 0.25 - 0.04 = 0.21``.", head = "Question 4.5")

# ╔═╡ c1887141-0ff8-5d23-a23a-e4b03efd4c64
mcq(md"For ``f(x) = 2x`` on ``[0, 1]``, what is the CDF ``F_X(x)`` for ``0 \le x \le 1``?",
	md"``2x``" => md"That is the density itself.",
	md"``x^2``" => md"Right. ``F_X(x) = \int_0^x 2\tilde x\,d\tilde x = x^2``, and ``dF_X/dx = 2x``.",
	md"``2``" => md"``2`` is the derivative of the density.";
	correct = 2, head = "Question 4.6")

# ╔═╡ 30b28689-279e-56c1-9d72-43b9c1bcbadd
numq(md"``X`` is exponential with ``f(x) = e^{-x}`` for ``x \ge 0``. Compute ``P(X > 1)``.", exp(-1); rtol = 0.01,
	why = md"``\int_1^\infty e^{-x}\,dx = [-e^{-x}]_1^\infty = e^{-1} = 0.368``.", head = "Question 4.7")

# ╔═╡ b1bb018a-fbd8-5d98-a19e-4598c7201b89
mcq(md"For ``f(x) = 2x`` on ``[0, 1]``, ``f(1) = 2``. What follows?",
	md"``P(X = 1) = 2``, which is impossible, so ``f`` is not a density." => md"``f(1)`` is not ``P(X = 1)``. A density is a probability **per unit** of ``x``.",
	md"Nothing is wrong: densities can exceed 1; only areas under ``f`` are probabilities." => md"Right. For example ``P(0.99 \le X \le 1) \approx 2 \cdot 0.01 = 0.02``.",
	md"``P(X \le 1) = 2``." => md"``P(X \le 1) = F_X(1) = 1``.";
	correct = 2, head = "Question 4.8")

# ╔═╡ 8646e367-46dd-503b-98e8-88c7b0e0e7b8
mcq(md"For a continuous random variable, what is ``P(X = 0.5)``?",
	md"``f_X(0.5)``" => md"A single point has no width, so it has no area.",
	md"0, but this does not mean that the value 0.5 is impossible." => md"Right. ``\int_{0.5}^{0.5} f_X(x)\,dx = 0``. Probabilities of continuous random variables are asked for intervals.",
	md"It depends on the density." => md"For every density, the integral over a single point is 0.";
	correct = 2, head = "Question 4.9")

# ╔═╡ b9ff50f4-4fa2-556a-b3cd-f2a504b68d10
mcq(md"Where is the CDF steepest?",
	md"Where the density is largest." => md"Right. ``f_X = dF_X/dx``: a steep CDF means a large density, and a wide range of heights ``u`` maps to a short range of ``x``.",
	md"Where the density is smallest." => md"Where the density is small, the CDF is almost flat.",
	md"At the median." => md"Only for symmetric single-peaked densities do the two coincide.";
	correct = 1, head = "Question 4.10")

# ╔═╡ 8bb43df8-53f4-5098-9d96-29a7ba856aff
numq(md"``f(x, y) = x + y`` on the unit square. Check: ``\int_0^1\!\!\int_0^1 (x + y)\,dx\,dy = ?``", 1.0;
	why = md"``\int_0^1 \big(\tfrac12 + y\big)\,dy = \tfrac12 + \tfrac12 = 1``.", head = "Question 4.11")

# ╔═╡ 83d360b6-fc86-51f6-8a5d-42e0e7682311
numq(md"Same density: compute ``P(X \le \tfrac12,\ Y \le \tfrac12) = \int_0^{1/2}\!\!\int_0^{1/2} (x + y)\,dx\,dy``.", 0.125;
	why = md"Inner integral: ``\int_0^{1/2}(x + y)\,dx = \tfrac18 + \tfrac{y}{2}``. Outer: ``\int_0^{1/2}\big(\tfrac18 + \tfrac{y}{2}\big)dy = \tfrac1{16} + \tfrac1{16} = \tfrac18``. Check with Figure 4.6.", head = "Question 4.12")

# ╔═╡ cbf18a63-f866-5285-94d9-4a5ccabb5ca9
mcq(md"``f(x, y) = x + y`` on the unit square. What is the marginal density ``f_X(x)``?",
	md"``x``" => md"The integral over ``y`` also contributes ``\int_0^1 y\,dy = 1/2``.",
	md"``x + \tfrac12``" => md"Right. ``f_X(x) = \int_0^1 (x + y)\,dy = x + \tfrac12``.",
	md"``x + y``" => md"A marginal density depends on ``x`` only: ``y`` is integrated out.";
	correct = 2, head = "Question 4.13")

# ╔═╡ 2f4adad5-d5d8-5ae1-81ce-0dac63211717
mcq(md"In pair 1, what differs between A and B?",
	md"The location: B is shifted to the right." => md"Right. A number that measures the location, such as the mean, tells A and B apart.",
	md"The spread." => md"Both have the same width.",
	md"The shape." => md"Both are bell curves of the same shape.";
	correct = 1, head = "Question 5.1")

# ╔═╡ 1842d8d5-d704-5c06-bcba-04a2c496e29c
mcq(md"In pairs 3 and 4, the centre and the width look alike. What differs?",
	md"Nothing: they are the same distribution." => md"Look at the left and right sides of B in pair 3, and at the tails of B in pair 4.",
	md"The shape: B is asymmetric in pair 3, and has heavier tails in pair 4." => md"Right. Location and spread are not enough to describe these differences; more numbers are needed.",
	md"Only the height of the peak." => md"The peak height is a consequence of the shape; the whole shape differs.";
	correct = 2, head = "Question 5.2")

# ╔═╡ e9e430c3-6e96-56bd-8a22-72b14fbfbea2
numq(md"A fair die: compute ``E[X] = \sum_x x\,p_X(x)``.", 3.5;
	why = md"``\frac16(1 + 2 + 3 + 4 + 5 + 6) = 3.5``.", head = "Question 5.3")

# ╔═╡ bbe7079a-9112-5066-8ab5-8a866a4d0b17
numq(md"A fair die: compute ``M_2[X] = \sum_x x^2\,p_X(x)``.", 91 / 6; rtol = 0.005,
	why = md"``\frac16(1 + 4 + 9 + 16 + 25 + 36) = 91/6 = 15.17``.", head = "Question 5.4")

# ╔═╡ 490c2de4-1aaf-5ff1-ad2b-d082bb00b1e3
numq(md"A fair die: compute ``\mathrm{Var}(X) = M_2 - M_1^2``.", 35 / 12; rtol = 0.005,
	why = md"``15.17 - 3.5^2 = 2.917``.", head = "Question 5.5")

# ╔═╡ 0d16d1f6-3365-5060-bc35-b2c5234b4664
numq(md"``X`` is uniform on ``[0, 10]``. Compute ``E[X] = \int_0^{10} x \cdot \frac{1}{10}\,dx``.", 5.0;
	why = md"``\frac{1}{10}\big[\frac{x^2}{2}\big]_0^{10} = \frac{1}{10}\cdot 50 = 5``.", head = "Question 5.6")

# ╔═╡ 0705e823-2568-5ae0-b400-21b1a42132b8
numq(md"``X`` is uniform on ``[0, 10]``. Compute ``\mathrm{Var}(X)`` from ``M_2 = \int_0^{10} x^2 \cdot \frac{1}{10}\,dx``.", 100 / 12; rtol = 0.005,
	why = md"``M_2 = \frac{1}{10}\cdot\frac{1000}{3} = 33.33``, so ``\mathrm{Var}(X) = 33.33 - 25 = 8.33 = 10^2/12``.", head = "Question 5.7")

# ╔═╡ f94f7cf2-271b-4990-b62f-e86336c93a70
mcq(md"Two distributions have the same area, the same centroid and the same spread. Are they the same distribution?",
	md"Yes: location and spread describe a distribution completely." => md"Look at pairs 3 and 4 of Figure 5.1: same mean and variance, different shapes.",
	md"Not necessarily: higher moments, such as the asymmetry and the tails, can still differ." => md"Right. Only for some families, such as the normal distribution, the first two moments fix everything.",
	md"Yes, if both are continuous." => md"Pair 3 of Figure 5.1 shows two continuous densities with the same mean and variance and different shapes.";
	correct = 2, head = "Question 5.8")

# ╔═╡ c7c49e86-99d6-53ae-83f5-77829abcfc2b
mcq(md"What is the sign of ``\mathrm{Cov}(\text{temperature}, \text{hot-drink sales})``?",
	md"Positive" => md"High temperatures come with **low** sales.",
	md"Negative" => md"Right. Most points lie in the upper-left and lower-right quadrants.",
	md"Zero" => md"The points follow a falling line.";
	correct = 2, head = "Question 5.9")

# ╔═╡ 04006285-331e-4baf-926b-aad05d2eefc8
mcq(md"``X`` = hours of sunshine on a day, ``Y`` = number of umbrellas sold that day. Which sign do you expect for ``\mathrm{Cov}(X, Y)``?",
	md"Positive" => md"On sunny days few umbrellas are sold: when ``X`` increases, ``Y`` tends to decrease.",
	md"Negative" => md"Right. More sunshine comes with fewer umbrellas.",
	md"Zero" => md"Sunshine and umbrella sales are related.";
	correct = 2, head = "Question 5.10")

# ╔═╡ 2f1758ea-8a11-4ae8-97a5-a3f8c7aaf72f
numq(md"**Binomial.** Ten students each choose tea or coffee. Each one chooses tea with probability 0.6, independently of the others. What is the probability that exactly 7 choose tea?", binomial(10, 7) * 0.6^7 * 0.4^3; rtol = 0.01,
	why = md"Success = tea, ``n = 10``, ``p = 0.6``: ``\binom{10}{7}\,0.6^7\,0.4^3 = 120 \cdot 0.0280 \cdot 0.064 = 0.215``.", head = "Question 6.1")

# ╔═╡ 0b79e12f-b5d8-4692-bc1a-8f9c97ead410
numq(md"**Binomial.** Same students: how many are expected to choose tea?", 6.0;
	why = md"``E[X] = np = 10 \cdot 0.6 = 6``.", head = "Question 6.2")

# ╔═╡ 7d2ee499-8854-4915-8268-b027d7ffee1d
numq(md"**Binomial.** A fair coin is tossed 4 times. What is the probability of exactly 2 heads?", 6 / 16; rtol = 0.01,
	why = md"``\binom42\,0.5^2\,0.5^2 = 6/16 = 0.375``. The six sequences are HHTT, HTHT, HTTH, THHT, THTH, TTHH.", head = "Question 6.3")

# ╔═╡ 93f87c1e-3120-4c90-9dd3-c4ae23a8518f
numq(md"**Binomial.** A die is rolled 5 times. What is the probability that no six appears?", (5 / 6)^5; rtol = 0.01,
	why = md"Success = six, ``p = 1/6``, ``k = 0``: ``\binom50 (1/6)^0 (5/6)^5 = 0.402``.", head = "Question 6.4")

# ╔═╡ f1d93e91-bfc2-5724-bf44-cbff2dba7e5f
numq(md"**Exponential.** A machine fails on average once every 10 hours, at random times (``\lambda = 0.1`` per hour). What is the probability that it runs for more than 20 hours?", exp(-2); rtol = 0.01,
	why = md"``P(X > 20) = 1 - F_X(20) = e^{-0.1 \cdot 20} = e^{-2} = 0.135``.", head = "Question 6.7")

# ╔═╡ ff238df2-0486-5e65-ab87-eb070765046f
hint(md"``P(H_1 \cap H_2 \cap H_3) = P(H_1)\,P(H_2 \mid H_1)\,P(H_3 \mid H_1 \cap H_2)``.")

# ╔═╡ 1bc312e7-e7e1-5ab7-8f48-2b8d722dae6f
reveal(md"``\frac{13}{52}\cdot\frac{12}{51}\cdot\frac{11}{50} = 0.0129``.")

# ╔═╡ e18549a6-4faf-5f58-9b26-15a8b9482dc8
hint(md"Add the two sequences that end with rain. Then divide the probability of *sun, rain* by the result.")

# ╔═╡ b23bd7f1-ec12-53f4-8227-8eae27f570f9
reveal(md"``P(X_2 = \text{rain}) = 0.7 \cdot 0.2 + 0.3 \cdot 0.6 = 0.32``. ``P(X_1 = \text{sun} \mid X_2 = \text{rain}) = 0.14/0.32 = 0.4375``.")

# ╔═╡ 45b67360-313b-5a90-99dc-1b0e3b78216f
hint(md"``P(X_1 = \text{sun})\,P(X_2 = \text{sun} \mid X_1 = \text{sun})\,P(X_3 = \text{sun} \mid X_2 = \text{sun})``.")

# ╔═╡ 9fadbdca-a6a0-55a4-b4be-2ea3451116ac
reveal(md"``0.7 \cdot 0.8 \cdot 0.8 = 0.448``.")

# ╔═╡ a7e96f90-7d92-5f91-9252-c15adc8adb2e
hint(md"For (b), use ``F_X(4) - F_X(1)`` and ``F_X(3) - F_X(2)``.")

# ╔═╡ 938bceb7-cbde-5e9d-978b-558de3df1904
reveal(md"(a) ``0.1, 0.3, 0.6, 0.8, 0.9, 1.0``. (b) ``0.9 - 0.3 = 0.6`` and ``0.8 - 0.6 = 0.2``.")

# ╔═╡ a3a0d09c-8069-5fa7-994a-a73ec3e631ac
hint(md"Marginals are row and column sums. Compare ``P(Y = 1 \mid X = 1)`` with ``P(Y = 1)``.")

# ╔═╡ dc30c49d-64f8-5a49-a80a-11a21d6bb535
reveal(md"``P_X(0) = P_X(1) = 0.5``; ``P_Y(0) = 0.4``, ``P_Y(1) = 0.6``. ``P(Y = 1 \mid X = 1) = 0.4/0.5 = 0.8 \ne 0.6``: not independent.")

# ╔═╡ e50fee24-1ab1-58f5-a34f-9f68ebb7d4a8
hint(md"``\int_0^3 c\,x^2\,dx = 9c`` must be 1.")

# ╔═╡ 83284896-bc1d-560d-8427-31d9d536f897
reveal(md"(a) ``c = 1/9``. (b) ``F_X(x) = x^3/27`` on ``[0, 3]``. (c) ``1/27 = 0.037``.")

# ╔═╡ 75242b03-29f9-5dfd-8d36-3a9e05cae436
hint(md"``f_X(x) = \int_0^1 4xy\,dy``. Compare ``f_X(x)\,f_Y(y)`` with ``f(x, y)``.")

# ╔═╡ e6ae10b6-c536-5f29-8a08-d97f8a14e6e6
reveal(md"``f_X(x) = 2x``, ``f_Y(y) = 2y``, and ``f_X f_Y = 4xy = f``: independent. ``P = \big(\int_0^{1/2} 2x\,dx\big)^2 = (1/4)^2 = 1/16``.")

# ╔═╡ 154aaf9d-6f2b-58f7-8fa0-40ed6b5ff4df
hint(md"``E[aX + b] = aE[X] + b`` and ``\mathrm{Var}(aX + b) = a^2\mathrm{Var}(X)``.")

# ╔═╡ a754f17c-1a78-58ac-be97-6beee0d4a615
reveal(md"``E[X] = 2.3``, ``M_2 = 7.3``, ``\mathrm{Var}(X) = 2.01``. ``E[Y] = 4.9``, ``\mathrm{Var}(Y) = 18.09``.")

# ╔═╡ 6b13e745-10ff-5ddd-8438-242d12587133
hint(md"(a) ``E[Z] = 12 \cdot \tfrac12 - 6 = 0``; for independent terms the variances add: ``\mathrm{Var}(Z) = 12 \cdot \tfrac{1}{12} = 1``.")

# ╔═╡ 1abe641f-f83d-5de5-867f-0a302b18d184
# Captions for figures
begin
	cap(x, c) = x isa Panels ? Panels(x.items, c, x.vertical) : x isa Player ? Player(x.items, x.n, x.start, x.fps, x.reads, x.label, c, x.loop) : vstack(x; caption = c)
end;

# ╔═╡ 370bc577-21cc-509c-9b41-5fe39739f6d8
# Section 1
begin
	const ORANGE = "#e1871d"
	Seed(default::Integer = 0; label = "New random numbers") = CounterButton(label)
	sub(s) = replace(string(s), "0" => "₀", "1" => "₁", "2" => "₂", "3" => "₃", "4" => "₄", "5" => "₅", "6" => "₆", "7" => "₇", "8" => "₈", "9" => "₉")

	# 1000 people: 50 ill (45 test positive), 950 healthy (95 test positive)
	const SCREEN = [45 5; 95 855]                 # rows: ill, healthy; columns: positive, negative
	const SCREEN_CONDS = ["everybody", "ill", "healthy", "positive", "negative"]

	function fig_screen(cond)
		f = fig(w = 560, h = 210, xr = (0, 560), yr = (210, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
		x0 = 150; cw = 130; y0 = 40; ch = 52
		rows = ["ill", "healthy"]; cols = ["positive", "negative"]
		keep(i, j) = cond == "everybody" || (cond == "ill" && i == 1) || (cond == "healthy" && i == 2) || (cond == "positive" && j == 1) || (cond == "negative" && j == 2)
		for (j, c) in enumerate(cols)
			text!(f, x0 + (j - 0.5) * cw, y0 - 14, "test " * c; anchor = "middle", size = 13, c = "#333", bold = true)
		end
		text!(f, x0 + 2.5cw, y0 - 14, "total"; anchor = "middle", size = 13, c = "#333", bold = true)
		for (i, r) in enumerate(rows)
			text!(f, x0 - 12, y0 + (i - 0.5) * ch, r; anchor = "end", size = 13, c = "#333", bold = true)
			for j in 1:2
				k = keep(i, j)
				rect!(f, x0 + (j - 1) * cw + 2, y0 + (i - 1) * ch + 2, x0 + j * cw - 2, y0 + i * ch - 2; fill = k ? (i == 1 ? "#f6d2c9" : "#d9e4f2") : "#f4f4f1", op = 1, stroke = k ? COL.ink : "#ddd", sw = k ? 1.6 : 0.8, rx = 4)
				text!(f, x0 + (j - 0.5) * cw, y0 + (i - 0.5) * ch, string(SCREEN[i, j]); anchor = "middle", size = 17, bold = k, c = k ? COL.ink : "#bbb")
			end
			text!(f, x0 + 2.5cw, y0 + (i - 0.5) * ch, string(sum(SCREEN[i, :])); anchor = "middle", size = 15, c = "#555")
		end
		for j in 1:2
			text!(f, x0 + (j - 0.5) * cw, y0 + 2.5ch, string(sum(SCREEN[:, j])); anchor = "middle", size = 15, c = "#555")
		end
		text!(f, x0 - 12, y0 + 2.5ch, "total"; anchor = "end", size = 13, c = "#333", bold = true)
		text!(f, x0 + 2.5cw, y0 + 2.5ch, "1000"; anchor = "middle", size = 15, c = "#555")
		f
	end

	function screen_text(cond)
		cond == "everybody" && return md"All 1000 people are possible. A probability is a fraction of 1000."
		n = cond == "ill" ? 50 : cond == "healthy" ? 950 : cond == "positive" ? 140 : 860
		Markdown.parse("Given **$(cond)**, only the $(n) people remain possible. A conditional probability given $(cond) is a fraction of these $(n) people.")
	end

	const P1 = Dict(:S => 0.7, :R => 0.3)                                            # day 1
	const P2 = Dict((:S, :S) => 0.8, (:S, :R) => 0.2, (:R, :S) => 0.4, (:R, :R) => 0.6) # day 2 given day 1
	const P3 = Dict((:S, :S, :S) => 0.85, (:S, :S, :R) => 0.15, (:R, :S, :S) => 0.7, (:R, :S, :R) => 0.3,
		(:S, :R, :S) => 0.45, (:S, :R, :R) => 0.55, (:R, :R, :S) => 0.3, (:R, :R, :R) => 0.7)    # day 3 given days 1 and 2
	wname(c) = c == :S ? "sun" : "rain"
	wcol(c) = c == :S ? "#f2c14e" : "#7fa8d8"

	function wnode!(f, x, y, c; r = 14)
		circle!(f, x, y, r; fill = wcol(c), stroke = "white", sw = 1.5)
		text!(f, x, y, c == :S ? "sun" : "rain"; anchor = "middle", size = 10.5, c = c == :S ? "#5a4300" : "#0d2a52", bold = true)
	end

	function fig_weather_tree(days, hot; model = :full, colourby = nothing)
		seqs = allseqs(days)
		nl = length(seqs); H = days == 2 ? 300 : 420
		f = fig(w = 680, h = H, xr = (0, 10), yr = (0, nl + 1), axes = false, l = 4, r = 4, t = 8, b = 8)
		xs = days == 2 ? [0.5, 2.6, 4.9] : [0.4, 2.0, 3.6, 5.2]
		yof(prefix) = begin
			idx = [i for (i, s) in enumerate(seqs) if s[1:length(prefix)] == prefix]
			nl + 1 - (first(idx) + last(idx)) / 2
		end
		circle!(f, xs[1], yof(Symbol[]), 9; fill = "#f2f2ef", stroke = "#999", sw = 1.2)
		prob(prefix, c) = length(prefix) == 0 ? P1[c] : condp(prefix, c, model)
		for d in 1:days
			prefixes = unique([s[1:d-1] for s in seqs])
			for p in prefixes, c in (:S, :R)
				q = vcat(p, c); ishot = hot !== nothing && hot[1:d] == q
				ya = yof(p); yb = yof(q)
				seg!(f, xs[d] + 0.25, ya, xs[d+1] - 0.3, yb; c = ishot ? COL.ink : "#c4c4c0", w = ishot ? 2.4 : 1.2)
				pv = prob(p, c)
				lc = colourby === nothing ? (ishot ? COL.ink : "#555") : colourby(d, p, c)
				text!(f, (xs[d] + xs[d+1]) / 2, (ya + yb) / 2 + (c == :S ? 0.28 : -0.28) * (days == 2 ? 1 : 0.8), fmt(pv; d = 2); anchor = "middle", size = 12, c = lc, bold = ishot || colourby !== nothing, bg = true)
				wnode!(f, xs[d+1], yb, c; r = days == 2 ? 15 : 12)
			end
		end
		for (i, s) in enumerate(seqs)
			y = nl + 1 - i; p = seqprob(s, model); ishot = hot !== nothing && s == hot
			x = xs[end] + 0.55
			ishot && rect!(f, x - 0.1, y - 0.42, 9.95, y + 0.42; fill = "#e8eef7", op = 1, stroke = COL.blue, sw = 1.2, rx = 4)
			txt = join(["$(wname(c))$(sub(k))" for (k, c) in enumerate(s)], " → ")
			factors = join([fmt(k == 1 ? P1[c] : condp(s[1:k-1], c, model); d = 2) for (k, c) in enumerate(s)], " · ")
			text!(f, x, y, "$(txt):  $(factors) = $(fmt(p; d = 3))"; size = days == 2 ? 12.5 : 11.5, c = ishot ? COL.ink : "#444", bold = ishot)
		end
		f
	end

	allseqs(d) = d == 0 ? [Symbol[]] : [vcat(c, s) for c in (:S, :R) for s in allseqs(d - 1)]

	function condp(prefix, c, model)
		model == :none && return P1[c]
		length(prefix) == 1 && return P2[(prefix[1], c)]
		model == :last && return P2[(prefix[end], c)]
		P3[(prefix[1], prefix[2], c)]
	end
	seqprob(s, model) = prod(k == 1 ? P1[c] : condp(s[1:k-1], c, model) for (k, c) in enumerate(s))

	const SEQ2 = [[:S, :S], [:S, :R], [:R, :S], [:R, :R]]
	const SEQ2_NAMES = ["sun, sun", "sun, rain", "rain, sun", "rain, rain"]
	const SEQ3 = allseqs(3)
	const SEQ3_NAMES = [join(wname.(s), ", ") for s in SEQ3]

	function seq_text(s)
		n = length(s)
		lhs = "P(" * join(["X_$(k) = \\text{$(wname(c))}" for (k, c) in enumerate(s)], ",\\ ") * ")"
		terms = String[]; nums = String[]
		for (k, c) in enumerate(s)
			cond = join(["X_$(j) = \\text{$(wname(s[j]))}" for j in 1:k-1], ",\\ ")
			push!(terms, k == 1 ? "P(X_1 = \\text{$(wname(c))})" : "P(X_$(k) = \\text{$(wname(c))} \\mid {\\color{$(COL.purple)}$(cond)})")
			push!(nums, fmt(k == 1 ? P1[c] : condp(s[1:k-1], c, :full); d = 2))
		end
		Markdown.parse("```math\n\\begin{aligned} & $(lhs) \\\\ &= " * join(terms, "\\,") * " \\\\ &= " * join(nums, " \\cdot ") * " = $(fmt(seqprob(s, :full); d = 3)) \\end{aligned}\n```")
	end

	const MEMORY = ["no memory", "memory of the last day", "memory of all earlier days"]
	const MEMKEY = Dict(MEMORY[1] => :none, MEMORY[2] => :last, MEMORY[3] => :full)

	function fig_memory_tree(case)
		model = MEMKEY[case]
		palette = [COL.blue, COL.red, COL.green, COL.purple, COL.ochre, COL.teal, "#8c564b", "#d62728"]
		keyof(d, p) = model == :none ? :all : d == 1 ? :start : model == :last ? (d, p[end]) : (d, Tuple(p))
		keys = Any[]
		colourby = (d, p, c) -> begin
			k = keyof(d, p)
			k in keys || push!(keys, k)
			palette[mod1(findfirst(==(k), keys), length(palette))]
		end
		fig_weather_tree(3, nothing; model = model, colourby = colourby)
	end

	memory_count(case) = case == MEMORY[1] ? (1, "") :
		case == MEMORY[2] ? (3, "") :
		(7, "")

	# 365 equally likely days, no leap day
	p_shared(n) = n < 2 ? 0.0 : 1 - prod((365 - i) / 365 for i in 1:n-1)
	# probability that the first shared birthday comes with person n
	p_first(n) = n < 2 ? 0.0 : prod((365 - i) / 365 for i in 1:n-2; init = 1.0) * (n - 1) / 365

	const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
	const MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	const MSTART = cumsum([0; MDAYS[1:end-1]])
	daymd(d) = (m = findlast(<=(d - 1), MSTART); (m, d - MSTART[m]))
	dayname(d) = (md_ = daymd(d); "$(md_[2]) $(MONTHS[md_[1]])")

	# people enter one by one until the first shared birthday
	function bday_room(rng)
		b = Int[]
		while true
			d = rand(rng, 1:365)
			push!(b, d)
			count(==(d), b) == 2 && return b
		end
	end

	function fig_bday_room(seed)
		b = bday_room(MersenneTwister(2365 + 17seed)); K = length(b)
		f = fig(w = 680, h = 290, xr = (-2.2, 31.6), yr = (13.2, -0.3), axes = false, l = 4, r = 4, t = 4, b = 4)
		for m in 1:12
			text!(f, -0.5, m, MONTHS[m]; anchor = "end", size = 11, c = "#555")
			for k in 1:MDAYS[m]
				rect!(f, k - 0.42, m - 0.4, k + 0.42, m + 0.4; fill = "#f3f3f0", op = 1, stroke = "#dcdcd6", sw = 0.6)
			end
		end
		for k in (1, 10, 20, 31)
			text!(f, k, 0.15, string(k); anchor = "middle", size = 10, c = "#888")
		end
		first_of = Dict{Int,Int}()
		reads = ["Press ▶. The room is empty. People enter one at a time and mark their birthday."]
		for (i, d) in enumerate(b)
			m, k = daymd(d)
			if haskey(first_of, d)
				j = first_of[d]
				rect!(f, k - 0.5, m - 0.48, k + 0.5, m + 0.48; fill = COL.red, op = 0.25, stroke = COL.red, sw = 2.2, s = i, over = true)
				push!(reads, "person <b>$(i)</b>: born on $(dayname(d)), the <b>same day as person $(j)</b>. First shared birthday with person $(i).")
			else
				first_of[d] = i
				circle!(f, k, m, 5.2; fill = ORANGE, stroke = "white", sw = 1, s = i)
				push!(reads, i == 1 ? "person <b>1</b>: born on $(dayname(d)). No one to share with: 365 of 365 days are free." :
					"person <b>$(i)</b>: born on $(dayname(d)). $(i - 1) days were taken, $(366 - i) of 365 were free, and this birthday landed on a free day (probability $(366 - i)/365 = $(fmt((366 - i) / 365; d = 4))).")
			end
		end
		player(f; n = K, start = 0, fps = 2, reads = reads, label = "person",
			caption = md"**Figure 1.4.** One room. Each person marks a birthday on the calendar (orange). The room stops at the first shared birthday (red).")
	end

	function fig_bday_many(T, seed)
		rng = MersenneTwister(4711 + 31seed)
		N = [length(bday_room(rng)) for _ in 1:T]
		nmax = 80; ns = 2:nmax
		a = fig(w = 340, h = 270, xr = (0, nmax + 1), yr = (0, 0.06), xlabel = "person n with the first shared birthday", ylabel = "fraction of rooms", title = "where the first match came", l = 56)
		bars!(a, collect(ns), [count(==(n), N) / T for n in ns]; width = 0.9, c = ORANGE, op = 0.85)
		line!(a, collect(ns), p_first.(ns); c = COL.blue, w = 2.2)
		b = fig(w = 340, h = 270, xr = (0, nmax + 1), yr = (0, 1.05), xlabel = "number of people n in a room", ylabel = "probability", title = "rooms of n people with a match", l = 56)
		hline!(b, 0.5; c = "#999")
		line!(b, collect(1:nmax), p_shared.(1:nmax); c = COL.blue, w = 2.0)
		line!(b, collect(1:nmax), [count(<=(n), N) / T for n in 1:nmax]; c = ORANGE, w = 2.0, steps = true)
		vline!(b, 23; c = COL.purple)
		c23 = count(<=(23), N)
		figure(a, b, Markdown.parse("Orange: $(T) simulated rooms. Blue: the exact probabilities. In **$(c23) of $(T)** rooms ($(fmt(100c23 / T; d = 3)) %) the first shared birthday came with person 23 or earlier; the exact probability for a room of 23 people is $(fmt(p_shared(23); d = 3))."))
	end
end;

# ╔═╡ 707daa4a-e946-538b-b15d-08ccaada4f3e
md"""
## 1. Several events

### 1.1 A screening test

A screening test for a disease is applied to 1000 people. The table counts how many are ill or healthy, and how many test positive or negative. 

Condition $(@bind s1_cond Select(SCREEN_CONDS))
"""

# ╔═╡ f9756f28-86f4-5f7d-b686-537a34240106
cap(vstack(fig_screen(s1_cond), screen_text(s1_cond)), md"**Figure 1.1.** A screening test applied to 1000 people. Rows: ill or healthy. Columns: test positive or negative.")

# ╔═╡ c09633bf-90df-592e-9dd0-66218754ebe0
md"""
### 1.2 Events in a sequence: does the order matter?

The weather is observed on consecutive days. Let ``X_1`` be the weather on day 1 and ``X_2`` the weather on day 2, each *sun* or *rain*. A simple model gives

- ``P(X_1 = \text{sun}) = 0.7``, ``P(X_1 = \text{rain}) = 0.3``;
- ``P(X_2 = \text{sun} \mid X_1 = \text{sun}) = 0.8``, ``P(X_2 = \text{rain} \mid X_1 = \text{sun}) = 0.2``;
- ``P(X_2 = \text{sun} \mid X_1 = \text{rain}) = 0.4``, ``P(X_2 = \text{rain} \mid X_1 = \text{rain}) = 0.6``.

The tree lists every **sequence** of two days. Each path is one sequence; its probability is the product along the path.

Sequence $(@bind s1_seq2 Select(SEQ2 .=> SEQ2_NAMES; default = SEQ2[2]))
"""

# ╔═╡ d7242770-0c2f-5dd7-89b2-229e3e4149a3
cap(vstack(fig_weather_tree(2, s1_seq2), seq_text(s1_seq2)), md"**Figure 1.2.** All sequences of two days. Branch labels: the probability of day 1, then the conditional probability of day 2 given day 1. Right: each sequence and its probability.")

# ╔═╡ 1a8eb0ba-d334-541c-97ad-5fc919eaf91c
md"""
### 1.3 The chain rule

The calculation of Figure 1.2 is the chain rule for two events: first the probability of the first event, then the probability of the second **given** the first,

```math
P(A_1 \cap A_2) = P(A_1)\,P(A_2 \mid A_1).
```

What changes for **three** days? A third factor is needed: the probability of day 3 given **both** earlier days. 

Sequence of three days $(@bind s1_seq3 Select(SEQ3 .=> SEQ3_NAMES; default = SEQ3[2]))
"""

# ╔═╡ 434dd7a5-a6a4-5e05-a157-db86d5168ff9
cap(vstack(fig_weather_tree(3, s1_seq3), seq_text(s1_seq3)), md"**Figure 1.3.** All eight sequences of three days. Each path multiplies three factors: day 1, day 2 given day 1, and day 3 given days 1 and 2.")

# ╔═╡ 1700ba25-c976-5f5a-951d-0ce4970db4c6
md"""
### 1.5 The birthday problem

In a room of ``n`` people, what is the probability that at least two of them share a birthday? Assume 365 equally likely days (no 29 February, no twins).

**One room.** People enter an empty room one at a time and mark their birthday on a calendar. The room stops at the first shared birthday.

$(@bind s1_room Seed(0; label = "New room"))
"""

# ╔═╡ d9461806-1548-5163-977d-c597a0e6227c
fig_bday_room(s1_room)

# ╔═╡ b8ae1dfc-0c0e-484f-af9e-80141a944239
md"""
**Many rooms.** Repeat the experiment of Figure 1.4 with many new rooms. For every room, note the person ``n`` who had the first shared birthday.

Rooms $(@bind s1_T Select([10, 100, 1000, 10_000]; default = 1000)) ``\qquad`` $(@bind s1_Tseed Seed(0; label = "New rooms"))
"""

# ╔═╡ 82f1bdfe-32fe-4249-8edd-92558bd63782
cap(fig_bday_many(s1_T, s1_Tseed), md"**Figure 1.5.** Left: in how many rooms the first shared birthday came with person ``n``. Right: the fraction of rooms whose first shared birthday came with person ``n`` or earlier (orange), and ``P(\text{at least two share})`` for a room of ``n`` people (blue). The purple line marks ``n = 23``.")

# ╔═╡ efd4e479-53dc-5fe8-8e73-7b80e98679ee
md"""
## 2. Random variables

### 2.1 One experiment, three questions

A die is rolled on a table. Three questions can be asked:

1. Was it a six?
2. Which number is on top?
3. How long did the die roll before it stopped?


$(@bind s2_seed Seed(0; label = "Roll the die"))
"""

# ╔═╡ ad1327ad-b0b5-582c-8fb6-09289c695cbc
md"""
### 2.2 Three types of random variables


- ``I`` takes only the values 0 and 1: a **binary** (indicator) random variable, ``I_A(\omega) = 1`` if ``\omega \in A`` and 0 otherwise;
- ``X`` takes values from a set that can be listed, ``\{1, \dots, 6\}``: a **discrete** random variable;
- ``T`` takes any value in an interval of times: a **continuous** random variable.

Rolling the die many times shows how the values of each random variable are spread. The same rolls give three different pictures.

Number of rolls $(@bind s2_N Select([10, 100, 1000, 10_000]; default = 1000)) ``\qquad`` $(@bind s2_seed2 Seed(0; label = "Roll again"))
"""

# ╔═╡ 52f2fff4-2b8d-5302-8529-2a7f327c365b
md"""
### 5.5 Distribution and data

The moments above are computed from a **distribution**. When only **data** ``x_1, \dots, x_n`` are available, the same quantities are estimated from the data.

!!! definition "From the distribution, and from data"
    From the distribution: ``E[X] = \sum_x x\,p_X(x)`` and ``\mathrm{Var}(X) = \sum_x (x - E[X])^2\,p_X(x)``. These are fixed numbers.

    From data: ``\bar x = \dfrac1n \sum_{i=1}^n x_i`` and ``s^2 = \dfrac{1}{n-1}\sum_{i=1}^n (x_i - \bar x)^2``.

Rolls $(@bind s5_n Select([5, 20, 100, 1000]; default = 20)) ``\qquad`` $(@bind s5_dseed Seed(0; label = "New rolls"))
"""

# ╔═╡ 8146bd55-84ec-58f5-9792-ec5e95c093ba
md"""
!!! theorem "Law of large numbers"
    For i.i.d. random variables ``X_1, \dots, X_N`` with ``E[X_i] = \mu``, the sample mean ``\bar X_N = \frac1N \sum_{i=1}^N X_i`` converges to ``\mu`` as ``N \to \infty``.

$(@bind s5_lseed Seed(0; label = "New sequences"))
"""

# ╔═╡ 317c73b1-4016-571d-8e9d-2546ff355d9d
md"""
Samples $(@bind s6_n Select([20, 200, 2000]; default = 200)) ``\qquad`` position of ``x`` $(@bind s6_x Slider(0:0.01:1; default = 0.4)) ``\qquad`` $(@bind s6_seed Seed(0; label = "New samples"))
"""

# ╔═╡ 796fa7e9-723a-5773-ba36-0f2ae0d25cbe
md"""
### 6.1 The central limit theorem

A Galton board drops a ball through ``n`` rows of pins. At each row the ball bounces left or right: ``X_i = 1`` if it bounces right at row ``i`` (probability ``p``), and ``X_i = 0`` otherwise. The slot where the ball lands is the **sum**

```math
S_n = X_1 + X_2 + \dots + X_n ,
```

so it follows ``\mathrm{Bin}(n, p)``.

Rows ``n`` $(@bind s6_rows Slider(2:20; default = 10, show_value = true)) ``\qquad`` ``p`` $(@bind s6_p Slider(0.1:0.05:0.9; default = 0.5, show_value = true)) ``\qquad`` balls $(@bind s6_balls Select([50, 200, 500]; default = 200)) ``\qquad`` $(@bind s6_gseed Seed(0; label = "New balls"))
"""

# ╔═╡ b91e6ef3-c0b8-5405-b2f2-1d673d919819
md"""
## 7. Why sampling?

Simulations need **samples**: many values drawn from a distribution. 

### Case 1: the distribution is known

A weather simulation needs 1000 temperatures from ``\mathcal N(17, 4^2)``. How do you get random numbers form this distribution for your simulation?

Samples $(@bind s7_m Select([100, 1000, 10_000]; default = 1000)) ``\qquad`` $(@bind s7_seed Seed(0; label = "New samples"))
"""

# ╔═╡ d9a597f1-de5b-5a08-a2fb-f2ccaabab9c9
let r = tryrun(() -> [z12(MersenneTwister(k)) for k in 1:20_000])
	ok = r isa AbstractVector && abs(mean(r)) < 0.03 && abs(var(r) - 1) < 0.05
	good = ok ? Markdown.parse("The 20 000 values have mean $(fmt(mean(r); d = 2)) and variance $(fmt(var(r); d = 3)), as for ``\\mathcal N(0, 1)``.") : md""
	res = check(r, ok, good, md"The values do not have mean 0 and variance 1. Check that the function returns the **sum** of the twelve numbers minus 6."; hint = md"`sum(u) - 6`")
	if ok
		e = collect(-4:0.25:4); f = fig(w = 680, h = 230, xr = (-4, 4), yr = (0, 0.5), xlabel = "z", ylabel = "density", grid = true)
		hist!(f, e, histdata(r, e); c = ORANGE, op = 0.8); line!(f, -4:0.02:4, pdf.(Normal(), -4:0.02:4); c = COL.blue, w = 2.2)
		vstack(res, f)
	else
		res
	end
end

# ╔═╡ 9220b219-748f-51f0-b3ca-1acf6a9ab0bc
# Section 2: random variables
begin
	function die!(f, x, y, v; s = 0.3, hot = false, col = COL.ink, op = 1.0)
		kx = (f.w - f.l - f.r) / (f.xr[2] - f.xr[1]); ky = (f.h - f.t - f.b) / abs(f.yr[2] - f.yr[1])
		sy = s * kx / ky
		rect!(f, x - s, y - sy, x + s, y + sy; fill = hot ? "#fff4e5" : "white", op = op, stroke = hot ? ORANGE : "#888", sw = hot ? 2.4 : 1.1, rx = 5)
		P = Dict(1 => [(0, 0)], 2 => [(-1, -1), (1, 1)], 3 => [(-1, -1), (0, 0), (1, 1)], 4 => [(-1, -1), (1, -1), (-1, 1), (1, 1)],
			5 => [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6 => [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)])
		for (a, b) in P[v]
			circle!(f, x + 0.55s * a, y + 0.55sy * b, max(2.6, 0.13 * 2s * kx); fill = col, stroke = "none", op = op)
		end
		f
	end

	rolltime(rng) = round(exp(log(1.6) + 0.25randn(rng)); digits = 2)

	function fig_roll(seed)
		rng = MersenneTwister(606 + 13seed); v = rand(rng, 1:6); t = rolltime(rng)
		f = fig(w = 680, h = 230, xr = (0, 10), yr = (0, 10), axes = false, l = 4, r = 4, t = 6, b = 6)
		text!(f, 0.2, 9.4, "one outcome ω of the experiment"; size = 12.5, c = "#333")
		die!(f, 1.4, 6.0, v; s = 0.55)
		text!(f, 1.4, 2.6, "rolled for $(t) s"; anchor = "middle", size = 12.5, c = "#333")
		qs = [("Was it a six?", "I(ω) = $(Int(v == 6))", "binary: 0 or 1"),
			("Which number is on top?", "X(ω) = $(v)", "discrete: 1, 2, …, 6"),
			("How long did the die roll?", "T(ω) = $(t) s", "continuous: any time > 0")]
		for (k, (q, val, kind)) in enumerate(qs)
			y = 8.6 - 3.0 * (k - 1)
			arrow!(f, 2.3, 6.0, 4.0, y - 0.3; c = ORANGE, w = 1.5, head = 7)
			text!(f, 4.2, y, q; size = 12.5, c = "#333")
			rect!(f, 4.15, y - 1.75, 7.0, y - 0.55; fill = "#fff4e5", op = 1, stroke = ORANGE, sw = 1.4, rx = 5)
			text!(f, 5.575, y - 1.15, val; anchor = "middle", size = 14, bold = true, c = COL.ink)
			text!(f, 7.25, y - 1.15, kind; size = 12, c = "#555")
		end
		f
	end

	function fig_mapping()
		f = fig(w = 680, h = 300, xr = (0, 10), yr = (0, 10), axes = false, l = 4, r = 4, t = 6, b = 6)
		text!(f, 1.0, 9.6, "Ω"; anchor = "middle", size = 15, bold = true)
		text!(f, 5.0, 9.6, "X(ω): number on top"; anchor = "middle", size = 12.5, c = COL.blue, bold = true)
		text!(f, 8.6, 9.6, "I(ω): was it a six?"; anchor = "middle", size = 12.5, c = COL.purple, bold = true)
		seg!(f, 4.3, 0.6, 4.3, 8.9; c = "#333", w = 1.2); seg!(f, 8.0, 3.0, 8.0, 7.0; c = "#333", w = 1.2)
		for v in 1:6
			y = 9.0 - 1.4v
			die!(f, 1.0, y, v; s = 0.32)
			arrow!(f, 1.5, y, 4.2, y; c = COL.blue, w = 1.3, head = 7, op = 0.8)
			text!(f, 4.5, y, string(v); size = 13, c = COL.blue, bold = true)
			yi = v == 6 ? 6.5 : 3.5
			arrow!(f, 4.9, y, 7.9, yi; c = COL.purple, w = 1.1, head = 6, op = 0.6, dash = "4 3")
		end
		text!(f, 8.2, 6.5, "1"; size = 13, c = COL.purple, bold = true); text!(f, 8.2, 3.5, "0"; size = 13, c = COL.purple, bold = true)
		f
	end

	function fig_three(N, seed)
		rng = MersenneTwister(707 + 17seed)
		v = rand(rng, 1:6, N); t = [rolltime(rng) for _ in 1:N]
		I = Int.(v .== 6)
		a = fig(w = 222, h = 230, xr = (-0.6, 1.6), yr = (0, 1), xlabel = "value of I", ylabel = "relative frequency", title = "I: was it a six?", l = 52, xticks = [0, 1])
		bars!(a, [0, 1], [count(==(0), I), count(==(1), I)] ./ N; width = 0.5, c = ORANGE)
		dots!(a, [0, 1], [5 / 6, 1 / 6]; c = COL.blue, r = 4)
		b = fig(w = 222, h = 230, xr = (0.3, 6.7), yr = (0, 0.3), xlabel = "value of X", ylabel = "relative frequency", title = "X: number on top", l = 52, xticks = collect(1:6))
		bars!(b, collect(1:6), [count(==(k), v) for k in 1:6] ./ N; width = 0.6, c = ORANGE)
		dots!(b, collect(1:6), fill(1 / 6, 6); c = COL.blue, r = 4)
		e = collect(0.6:0.1:3.2)
		c = fig(w = 222, h = 230, xr = (0.6, 3.2), yr = (0, 0.12), xlabel = "value of T (s)", ylabel = "relative frequency", title = "T: rolling time", l = 52)
		hist!(c, e, histdata(t, e; density = false) ./ N; c = ORANGE, op = 0.85)
		figure(a, b, c)
	end

	function fig_read_pmf()
		ks = collect(2:12); p = [(6 - abs(k - 7)) / 36 for k in ks]
		f = fig(w = 680, h = 250, xr = (1.3, 12.7), yr = (0, 0.2), xlabel = "value x of the sum of two dice", ylabel = "P(X = x)", grid = true, xticks = ks)
		bars!(f, ks, p; width = 0.6, c = COL.blue, labels = ["$(round(Int, 36q))/36" for q in p], lsize = 10.5)
		f
	end

	const RV_ITEMS = [
		("whether a medical test comes back positive", "binary", "Two values only: positive (1) or negative (0)."),
		("the number of emails that arrive in one hour", "discrete", "A count: 0, 1, 2, … The values can be listed, even if there is no upper limit."),
		("the time until the next earthquake in a region", "continuous", "A time can take any value in an interval."),
		("the number of heads in 20 coin tosses", "discrete", "A count with the 21 possible values 0, 1, …, 20."),
		("the body temperature of a patient", "continuous", "A measured temperature is a point of an interval. A thermometer rounds it, but the quantity itself is continuous."),
		("whether a photon is detected in a given microsecond", "binary", "Detected (1) or not detected (0)."),
		("the number of cars in a car park with 500 places", "discrete", "Many possible values (0 to 500), but they can be listed: the variable is discrete."),
		("the angle at which a spinning pointer stops", "continuous", "Any angle between 0° and 360° is possible.")]

	function rv_feedback(ans)
		lines = String[]; nok = 0; nans = 0
		for (i, (item, sol, why)) in enumerate(RV_ITEMS)
			a = ans[i]; a == "choose" && continue
			nans += 1; ok = a == sol; nok += ok
			push!(lines, ok ? "- ✓ *$(item)*: **$(sol)**. $(why)" : "- ✗ *$(item)*: not $(a). Think again about the set of possible values.")
		end
		nans == 0 && return md"Choose a type for each quantity; the feedback appears here."
		Markdown.parse(join(lines, "\n") * "\n\n**$(nok) of $(nans)** answers correct" * (nans < length(RV_ITEMS) ? " ($(length(RV_ITEMS) - nans) not answered yet)." : "."))
	end

	rv_quiz() = PlutoUI.combine() do Child
		@htl("""<ol class="rvq">$([@htl("<li><span>$(it[1])</span> $(Child(Select(["choose", "binary", "discrete", "continuous"])))</li>") for it in RV_ITEMS])</ol>""")
	end
end;

# ╔═╡ 9604bb6e-f09a-50b5-8d09-9ddbed4a4e78
cap(fig_roll(s2_seed), md"**Figure 2.1.** One outcome ``\omega`` of the experiment.")

# ╔═╡ 5150232d-151b-5a73-99c9-712af15d577e
cap(fig_mapping(), md"**Figure 2.2.** Two random variables on the same sample space ``\Omega = \{1, \dots, 6\}``. ``X`` maps each face to its number; ``I`` maps the six to 1 and every other face to 0.")

# ╔═╡ 08dddad4-f342-5079-8fe0-acf6d1f0cbd9
cap(fig_three(s2_N, s2_seed2), md"**Figure 2.3.** The same rolls, three random variables. Orange: relative frequencies in the rolls. Blue dots: the exact probabilities.")

# ╔═╡ 2af944a3-6c96-5bd1-960e-67eb48926450
@bind s2_cls rv_quiz()

# ╔═╡ a7ac41bc-388c-5f4a-bbeb-4aa30a77b6a8
rv_feedback(s2_cls)

# ╔═╡ c2721eec-bcba-5876-8c32-f68498bfd697
cap(fig_read_pmf(), md"**Figure 2.4.** The distribution of the sum of two dice.")

# ╔═╡ 7fa726bc-5f9b-5d5e-87f5-ea5a5860ceb3
# Section 3: discrete distributions, the sum of two dice
begin
	const SKS = collect(2:12)
	const SP = [(6 - abs(k - 7)) / 36 for k in SKS]
	const SF = cumsum(SP)
	Fsum(x) = x < 2 ? 0.0 : x >= 12 ? 1.0 : SF[floor(Int, x) - 1]
	frac36(p) = "$(round(Int, 36p))/36"

	function fig_grid_pmf(s)
		f = fig(w = 300, h = 300, xr = (0.5, 6.5), yr = (6.6, 0.0), axes = false, l = 40, r = 6, t = 34, b = 6)
		for i in 1:6, j in 1:6
			hot = i + j == s
			rect!(f, j - 0.46, i - 0.46, j + 0.46, i + 0.46; fill = hot ? COL.blue : "#f4f4f1", op = hot ? 0.8 : 1, stroke = "#d0d0cb", sw = 0.8, rx = 3)
			text!(f, j, i, string(i + j); anchor = "middle", size = 12, c = hot ? "white" : "#666", bold = hot)
		end
		for k in 1:6
			text!(f, k, 0.12, string(k); anchor = "middle", size = 12, c = "#333")
			text!(f, 0.25, k, string(k); anchor = "middle", size = 12, c = "#333")
		end
		text!(f, 3.5, -0.32, "second die"; anchor = "middle", size = 11.5, c = "#555")
		text!(f, -0.25, 3.5, "first die"; anchor = "middle", size = 11.5, c = "#555", rot = -90)
		g = fig(w = 360, h = 300, xr = (1.3, 12.7), yr = (0, 0.21), xlabel = "x", ylabel = "p_X(x) = P(X = x)", xticks = SKS, t = 34)
		bars!(g, SKS, SP; width = 0.65, c = [k == s ? COL.blue : "#c9d6ea" for k in SKS], op = 1)
		text!(g, s, SP[s-1], frac36(SP[s-1]); anchor = "middle", dy = -10, size = 12, c = COL.ink, bold = true)
		figure(f, g, Markdown.parse("The sum ``x = $(s)`` occurs in **$(round(Int, 36SP[s-1]))** of the 36 equally likely outcomes, so ``p_X($(s)) = P(X = $(s)) = $(frac36(SP[s-1])) = $(fmt(SP[s-1]; d = 3))``."))
	end

	function stair!(g, upto; c = COL.green)
		xs = [1.3; SKS; 12.7]; Fs = [0.0; SF; 1.0]
		for k in 1:length(xs)-1
			xs[k] > upto && break
			seg!(g, xs[k], Fs[k], min(xs[k+1], max(upto, xs[k])), Fs[k]; c = c, w = 2.4)
		end
		ks = filter(<=(upto), SKS)
		isempty(ks) || dots!(g, ks, SF[ks .- 1]; c = c, r = 3.4)
		isempty(ks) || dots!(g, ks, [k == 2 ? 0.0 : SF[k-2] for k in ks]; c = "white", r = 3.0, stroke = c, sw = 1.4)
		g
	end

	function fig_build_cdf(x)
		f = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 0.21), xlabel = "x", ylabel = "p_X(x)", title = "PMF", xticks = SKS)
		bars!(f, SKS, SP; width = 0.65, c = [k <= x ? COL.blue : "#c9d6ea" for k in SKS], op = 1)
		vline!(f, x + 0.001; c = COL.purple)
		g = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 1.05), xlabel = "x", ylabel = "F_X(x) = P(X ≤ x)", title = "CDF", xticks = SKS)
		stair!(g, x)
		dots!(g, [x], [Fsum(x)]; c = COL.purple, r = 5.5, stroke = "white", sw = 1.2)
		ks = filter(<=(x), SKS)
		expr = isempty(ks) ? "0" : join(["p_X($(k))" for k in ks], " + ")
		figure(f, g, Markdown.parse("``F_X($(fmt(x; d = 3))) = $(expr) = $(frac36(Fsum(x))) = $(fmt(Fsum(x); d = 3))`` (dark bars on the left, purple point on the right)"))
	end

	function fig_interval(a, lop, b, rop)
		lo = lop == "<" ? a + 1 : a; hi = rop == "<" ? b - 1 : b
		sel = [lo <= k <= hi for k in SKS]
		Flo = Fsum(lo - 1); Fhi = Fsum(hi); val = max(Fhi - Flo, 0.0)
		f = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 0.21), xlabel = "x", ylabel = "p_X(x)", title = "PMF", xticks = SKS)
		bars!(f, SKS, SP; width = 0.65, c = [s ? COL.blue : "#c9d6ea" for s in sel], op = 1)
		g = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 1.05), xlabel = "x", ylabel = "F_X(x)", title = "CDF", xticks = SKS)
		stair!(g, 12.7)
		if hi >= lo
			xb = clamp(hi, 2, 12)
			rect!(g, xb - 0.2, Flo, xb + 0.2, Fhi; fill = COL.blue, op = 0.4, stroke = COL.blue, sw = 1)
			hline!(g, Flo; c = COL.purple); hline!(g, Fhi; c = COL.purple)
		end
		lhs = "P($(a) $(lop == "<" ? "<" : "\\le") X $(rop == "<" ? "<" : "\\le") $(b))"
		rhs = hi < lo ? "0" : "F_X($(hi)) - F_X($(lo - 1)) = $(frac36(Fhi)) - $(frac36(Flo))"
		figure(f, g, Markdown.parse("``$(lhs) = $(rhs) = $(frac36(val)) = $(fmt(val; d = 3))``. The values counted are ``$(hi < lo ? "none" : lo == hi ? string(lo) : "$(lo), \\dots, $(hi)")``."))
	end

	const DPAIRS = ["X = first die, Y = second die", "X = first die, Y = sum of both dice"]

	function dice_joint(pair)
		if pair == DPAIRS[1]
			ys = collect(1:6); J = fill(1 / 36, 6, 6)
		else
			ys = collect(2:12); J = [((y - x) in 1:6) / 36 for x in 1:6, y in ys]
		end
		ys, J
	end

	function fig_joint_dice(pair, x0)
		ys, J = dice_joint(pair); xs = 1:6; nY = length(ys)
		pX = vec(sum(J, dims = 2)); pY = vec(sum(J, dims = 1))
		cw = pair == DPAIRS[1] ? 36 : 26; ch = 30
		x0g = 52; y0g = 44
		W = x0g + cw * nY + 74; H = y0g + 6 * ch + 70
		f = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
		for (i, x) in enumerate(xs), (j, y) in enumerate(ys)
			v = J[i, j]; hot = x == x0
			a = x0g + (j - 1) * cw; b = a + cw; c = y0g + (i - 1) * ch; d = c + ch
			rect!(f, a, c, b, d; fill = v == 0 ? "#fafaf8" : COL.blue, op = v == 0 ? 1 : 0.75, stroke = "white", sw = 1)
			v > 0 && text!(f, (a + b) / 2, (c + d) / 2, "$(round(Int, 36v))/36"; anchor = "middle", size = cw > 30 ? 10.5 : 9, c = "white")
		end
		rect!(f, x0g, y0g + (x0 - 1) * ch, x0g + nY * cw, y0g + x0 * ch; fill = "none", op = 1, stroke = COL.purple, sw = 2.6, over = true)
		for (i, x) in enumerate(xs)
			text!(f, x0g - 10, y0g + (i - 0.5) * ch, string(x); anchor = "end", size = 12, c = x == x0 ? COL.purple : "#333", bold = x == x0)
			bw = 56 * pX[i] / maximum(pX)
			rect!(f, x0g + nY * cw + 8, y0g + (i - 0.82) * ch, x0g + nY * cw + 8 + bw, y0g + (i - 0.18) * ch; fill = COL.green, op = 0.75)
		end
		base = y0g + 6 * ch + 52
		for (j, y) in enumerate(ys)
			text!(f, x0g + (j - 0.5) * cw, y0g - 9, string(y); anchor = "middle", size = 12, c = "#333")
			bh = 44 * pY[j] / maximum(pY)
			rect!(f, x0g + (j - 0.82) * cw, base - bh, x0g + (j - 0.18) * cw, base; fill = COL.green, op = 0.75)
		end
		seg!(f, x0g, base, x0g + nY * cw, base; c = "#999", w = 0.8)
		text!(f, x0g + nY * cw / 2, 12, pair == DPAIRS[1] ? "Y = second die" : "Y = sum of both dice"; anchor = "middle", size = 12.5, c = "#222")
		text!(f, 14, y0g + 3 * ch, "X = first die"; anchor = "middle", size = 12.5, c = "#222", rot = -90)
		text!(f, x0g + nY * cw + 8, y0g - 9, "p_X(x)"; size = 11.5, c = COL.green)
		text!(f, x0g - 6, base - 8, "p_Y(y)"; anchor = "end", size = 11.5, c = COL.green)
		cond = J[x0, :] ./ pX[x0]
		g = fig(w = 268, h = H, xr = (first(ys) - 0.7, last(ys) + 0.7), yr = (0, 1.3 * max(maximum(cond), maximum(pY))), xlabel = "y", ylabel = "probability", t = 44, xticks = nY > 6 ? collect(2:2:12) : ys)
		bars!(g, ys, pY; width = 0.8, c = "white", op = 1, stroke = COL.green, sw = 1.6)
		bars!(g, ys, cond; width = 0.5, c = COL.purple, op = 0.85)
		legend!(g, [("p_{Y|X}(y | $(x0))", COL.purple, :bar), ("p_Y(y)", COL.green, :line)]; at = :top, size = 11.5)
		indep = all(abs.(cond .- pY) .< 1e-12)
		txt = Markdown.parse("Row ``x = $(x0)`` of the table, divided by its sum ``p_X($(x0)) = 6/36``, gives the conditional distribution ``p_{Y \\mid X}(y \\mid $(x0))`` (purple bars, right). " *
			(indep ? "It coincides with the marginal ``p_Y`` (green outline) for **every** row: knowing ``X`` says nothing about ``Y``, so ``X`` and ``Y`` are **independent**, ``p_{X,Y}(x, y) = p_X(x)\\,p_Y(y)``." :
			"It differs from the marginal ``p_Y`` (green outline): knowing ``X`` changes the distribution of ``Y``, so ``X`` and ``Y`` are **dependent**. For example ``p_{X,Y}($(x0), $(x0 + 1)) = 1/36``, but ``p_X($(x0))\\,p_Y($(x0 + 1)) = $(fmt(pX[x0] * pY[x0 + 1 - first(ys) + 1]; d = 3))``."))
		figure(f, g, txt)
	end

	# a 3D bar view of the joint table (PlutoPlotly)
	function joint3d_dice(pair)
		ys, J = dice_joint(pair)
		X = Float64[]; Y = Float64[]; Z = Float64[]; I = Int[]; Jj = Int[]; K = Int[]; col = Float64[]
		for (a, x) in enumerate(1:6), (b, y) in enumerate(ys)
			v = J[a, b]; v == 0 && continue
			o = length(X)
			for (dx, dy, z) in ((-0.35, -0.35, 0), (0.35, -0.35, 0), (0.35, 0.35, 0), (-0.35, 0.35, 0), (-0.35, -0.35, v), (0.35, -0.35, v), (0.35, 0.35, v), (-0.35, 0.35, v))
				push!(X, x + dx); push!(Y, y + dy); push!(Z, z); push!(col, v)
			end
			for (i, j, k) in ((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7))
				push!(I, o + i); push!(Jj, o + j); push!(K, o + k)
			end
		end
		tr = mesh3d(x = X, y = Y, z = Z, i = I, j = Jj, k = K, intensity = col, colorscale = [[0, "#c9d6ea"], [1, COL.blue]], showscale = false, flatshading = true)
		PlutoPlotly.plot(tr, Layout(height = 380, margin = attr(l = 0, r = 0, t = 10, b = 0),
			scene = attr(xaxis = attr(title = "x (first die)"), yaxis = attr(title = pair == DPAIRS[1] ? "y (second die)" : "y (sum)"), zaxis = attr(title = "p(x, y)"),
				camera = attr(eye = attr(x = 1.6, y = -1.6, z = 1.1)))))
	end
end;

# ╔═╡ 2517909b-0fa2-5c3e-af6b-5554cc9967a2
cap(fig_grid_pmf(s3_s), md"**Figure 3.1.** Left: the 36 outcomes, labelled with their sum; the cells with the chosen sum are blue. Right: the probability of each sum.")

# ╔═╡ 3e291457-a06b-536d-a450-893063d3df27
cap(fig_build_cdf(s3_x), md"**Figure 3.2.** Left: the bars added up to ``x``. Right: the CDF; filled dots mark its value at each jump, open dots the value just before.")

# ╔═╡ 751fad30-f343-50eb-b6b9-8c3a7b932284
cap(fig_interval(s3_a, s3_lop, s3_b, s3_rop), md"**Figure 3.3.** Left: the bars counted. Right: the same probability as the difference of two values of the CDF (between the purple lines).")

# ╔═╡ 44da24f9-33f2-5d57-a96c-f6b114724d0b
md"""
### 3.4 Two discrete random variables

Two random variables of the same roll, for example the first die ``X`` and a second quantity ``Y``, are described together by a table.

!!! definition "Joint, marginal and conditional PMF"
    Joint: ``P_{X,Y}(x, y) = P(\{X = x\} \cap \{Y = y\})``. Marginal: ``P_X(x) = \sum_y P_{X,Y}(x, y)``. Conditional:
    ```math
    P_{Y \mid X}(y \mid x) = \frac{P_{X,Y}(x, y)}{P_X(x)}.
    ```
    ``X`` and ``Y`` are independent if ``P_{X,Y}(x, y) = P_X(x)\,P_Y(y)`` for all ``x, y``.


Pair $(@bind s3_pair Select(DPAIRS; default = DPAIRS[2])) ``\qquad`` condition ``X = x``: $(@bind s3_xc Slider(1:6; default = 3, show_value = true)) ``\qquad`` 3D view $(@bind s3_3d CheckBox(default = false))
"""

# ╔═╡ 0ad92dc2-5559-59a9-9a9b-0d84eb40f079
cap(fig_joint_dice(s3_pair, s3_xc), md"**Figure 3.4.** Left: the joint PMF as a table, with the marginals in green. Right: the conditional PMF of ``Y`` given ``X = x`` (purple) and the marginal ``P_Y`` (green outline).")

# ╔═╡ 1eafee27-1bf2-5b79-8a9b-332f8c0a9ba6
s3_3d ? joint3d_dice(s3_pair) : md""

# ╔═╡ 172c387f-8025-5133-a600-4e42c20970b1
# Section 4.1: 
begin
	const TMODEL = Normal(17, 4)                                      # noon temperature in May, °C (synthetic)
	const TEMPS = round.(rand(MersenneTwister(2026), TMODEL, 100_000); digits = 2)
	const TLO, THI = 2.0, 32.0

	function fig_temp_player()
		v = TEMPS[1:300]; edges = collect(TLO:1.0:THI)
		f = fig(w = 680, h = 330, xr = (TLO, THI), yr = (0, 34), xlabel = "noon temperature x (°C)", ylabel = "number of days", grid = true)
		cnt = zeros(Int, length(edges) - 1)
		reads = ["Press ▶. Each dot is one day; it is placed in the 1 °C bin."]
		for (k, t) in enumerate(v)
			i = clamp(searchsortedlast(edges, t), 1, length(cnt)); cnt[i] += 1
			circle!(f, edges[i] + 0.5, cnt[i] - 0.5, 3.6; fill = ORANGE, stroke = "none", s = k)
			push!(reads, "day <b>$(k)</b>: $(fmt(t; d = 3)) °C")
		end
		player(f; n = length(v), start = length(v), fps = 20, reads = reads, label = "day",
			caption = md"**Figure 4.1.** Temperatures of 300 days. Each new day adds one dot to the bin of its temperature.")
	end

	const YSCALES = ["count", "relative frequency", "density"]
	function fig_temp_scale(N, scale, w)
		v = TEMPS[1:N]; edges = collect(TLO:w:THI + w); c = histdata(v, edges; density = false)
		y = scale == YSCALES[1] ? c : scale == YSCALES[2] ? c ./ N : c ./ (N * w)
		lab = scale == YSCALES[1] ? "number of days in the bin" : scale == YSCALES[2] ? "fraction of days in the bin" : "fraction of days per °C"
		f = fig(w = 680, h = 280, xr = (TLO, THI), yr = (0, 1.15 * maximum(y)), xlabel = "temperature x (°C)", ylabel = lab, grid = true)
		hist!(f, edges, y; c = ORANGE, op = 0.85)
		if scale == YSCALES[3]
			xs = range(TLO, THI, length = 300); line!(f, xs, pdf.(TMODEL, xs); c = COL.blue, w = 2.2)
		end
		tot = scale == YSCALES[1] ? "the heights add up to N = $(N)" : scale == YSCALES[2] ? "the heights add up to 1" : "the **areas** (height × $(w) °C) add up to 1"
		figure(f, Markdown.parse("``N = $(N)`` days, bin width $(w) °C: tallest bar **$(fmt(maximum(y); d = 3))**; $(tot)." * (scale == YSCALES[3] ? "" : "")))
	end

	# ---------------- Figure 4.5: probability is area, written as an integral ----------------
	const DENS_EX = ["uniform on [0, 10]", "f(x) = 2x on [0, 1]", "exponential: f(x) = e^(−x), x ≥ 0", "normal: temperature, N(17, 4²)"]
	function dens_ex(name)
		name == DENS_EX[1] ? (Uniform(0, 10), -1.0, 11.0) : name == DENS_EX[2] ? (TriangularDist(0, 1, 1), -0.1, 1.1) :
		name == DENS_EX[3] ? (Exponential(1), 0.0, 5.0) : (TMODEL, 2.0, 32.0)
	end

	function integral_text(name, a, b, P)
		A, Bb = fmt(a; d = 3), fmt(b; d = 3)
		name == DENS_EX[1] ? "\\int_{$(A)}^{$(Bb)} \\frac{1}{10}\\,dx = \\frac{$(Bb) - $(A)}{10}" :
		name == DENS_EX[2] ? "\\int_{$(A)}^{$(Bb)} 2x\\,dx = \\big[x^2\\big]_{$(A)}^{$(Bb)} = $(Bb)^2 - $(A)^2" :
		name == DENS_EX[3] ? "\\int_{$(A)}^{$(Bb)} e^{-x}\\,dx = \\big[-e^{-x}\\big]_{$(A)}^{$(Bb)} = e^{-$(A)} - e^{-$(Bb)}" :
		"\\int_{$(A)}^{$(Bb)} \\frac{1}{\\sqrt{2\\pi}\\,4}\\,e^{-(x-17)^2/32}\\,dx = F_X($(Bb)) - F_X($(A))"
	end

	function fig_area(name, ta, tb)
		d, lo, hi = dens_ex(name)
		s0, s1 = name == DENS_EX[1] ? (0.0, 10.0) : name == DENS_EX[2] ? (0.0, 1.0) : name == DENS_EX[3] ? (0.0, 5.0) : (2.0, 32.0)
		a = round(s0 + min(ta, tb) * (s1 - s0); digits = 2); b = round(s0 + max(ta, tb) * (s1 - s0); digits = 2)
		xs = collect(range(lo, hi, length = 500)); y = pdf.(d, xs)
		P = cdf(d, b) - cdf(d, a)
		f = fig(w = 330, h = 270, xr = (lo, hi), yr = (0, 1.2 * maximum(y)), xlabel = "x", ylabel = "f_X(x)", title = "PDF: P(a ≤ X ≤ b) is the area")
		inside = filter(x -> a <= x <= b, xs); isempty(inside) || area!(f, [a; inside; b], pdf.(d, [a; inside; b]); c = COL.blue, op = 0.4)
		line!(f, xs, y; c = COL.blue, w = 2.2)
		g = fig(w = 330, h = 270, xr = (lo, hi), yr = (0, 1.05), xlabel = "x", ylabel = "F_X(x)", title = "CDF")
		line!(g, xs, cdf.(d, xs); c = COL.green, w = 2.2)
		dots!(g, [a, b], cdf.(d, [a, b]); c = COL.green, r = 4)
		rect!(g, b - 0.012(hi - lo), cdf(d, a), b + 0.012(hi - lo), cdf(d, b); fill = COL.blue, op = 0.4, stroke = COL.blue)
		hline!(g, cdf(d, a); c = COL.purple); hline!(g, cdf(d, b); c = COL.purple)
		figure(f, g, Markdown.parse("```math\nP($(fmt(a; d = 3)) \\le X \\le $(fmt(b; d = 3))) = $(integral_text(name, a, b, P)) = $(fmt(P; d = 3))\n```"))
	end

	# ---------------- Figure 4.6: from the CDF back to values ----------------
	const INV_DISTS = ["normal N(0, 1)", "uniform U(0, 1)", "exponential Exp(1)", "f(x) = 2x on [0, 1]", "sum of two dice (discrete)"]
	function fig_inverse(name, u, m, seed)
		rng = MersenneTwister(99 + 5seed); us = rand(rng, m)
		if name == INV_DISTS[5]
			inv(uu) = SKS[min(searchsortedfirst(SF, uu - 1e-12), 11)]
			x = inv(u); xsmp = inv.(us)
			g = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 1.05), xlabel = "x", ylabel = "u = F_X(x)", title = "CDF", xticks = SKS)
			stair!(g, 12.7)
			for k in 1:11
				seg!(g, SKS[k], k == 1 ? 0.0 : SF[k-1], SKS[k], SF[k]; c = COL.green, w = 1, dash = "2 2")
			end
			f = fig(w = 330, h = 280, xr = (1.3, 12.7), yr = (0, 0.25), xlabel = "x", ylabel = "probability", title = "PMF and the values obtained", xticks = SKS)
			bars!(f, SKS, SP; width = 0.75, c = COL.blue, op = 0.25, stroke = COL.blue)
			bars!(f, SKS, [count(==(k), xsmp) / m for k in SKS]; width = 0.45, c = ORANGE, op = 0.9)
			lo = 1.3
		else
			d, lo, hi = name == INV_DISTS[1] ? (Normal(), -3.5, 3.5) : name == INV_DISTS[2] ? (Uniform(0, 1), -0.2, 1.2) : name == INV_DISTS[3] ? (Exponential(1), 0.0, 5.0) : (TriangularDist(0, 1, 1), -0.1, 1.1)
			x = quantile(d, u); xsmp = quantile.(d, us)
			xs = collect(range(lo, hi, length = 400))
			g = fig(w = 330, h = 280, xr = (lo, hi), yr = (0, 1.05), xlabel = "x", ylabel = "u = F_X(x)", title = "CDF")
			line!(g, xs, cdf.(d, xs); c = COL.green, w = 2.4)
			e = collect(range(lo, hi, length = 36)); h = histdata(xsmp, e)
			f = fig(w = 330, h = 280, xr = (lo, hi), yr = (0, 1.25 * max(maximum(pdf.(d, xs)), maximum(h))), xlabel = "x", ylabel = "density", title = "PDF and the values obtained")
			hist!(f, e, h; c = ORANGE, op = 0.75)
			line!(f, xs, pdf.(d, xs); c = COL.blue, w = 2.2)
		end
		for uu in us[1:min(m, 60)]
			seg!(g, lo, uu, lo + 0.04 * (g.xr[2] - g.xr[1]), uu; c = ORANGE, w = 1.2, op = 0.8)
		end
		arrow!(g, g.xr[1], u, x, u; c = COL.purple, w = 1.8); arrow!(g, x, u, x, 0; c = COL.purple, w = 1.8)
		text!(g, x, 0.06, " x = $(fmt(x; d = 3))"; size = 12, c = COL.purple, bold = true, bg = true)
		vline!(f, x; c = COL.purple, dash = nothing, w = 1.6)
		figure(g, f)
	end

	function pipe_probe()
		@htl("""<div class="probe" style="font-family:system-ui,sans-serif;font-size:12px;color:#1b1b1b">
		<svg viewBox="0 0 680 300" style="width:100%;max-width:680px;display:block;background:white">
		<g class="ax"></g><path class="sig" fill="none" stroke="#e1871d" stroke-width="1.2"/><circle class="dot" r="4" fill="#e1871d"/>
		<g class="bars"></g><path class="pdf" fill="none" stroke="#1f5aa6" stroke-width="2.2"/>
		<line class="mean" stroke="#555" stroke-dasharray="4 3" stroke-width="1.1"/><text class="cnt" x="440" y="16" font-size="12" fill="#333"></text></svg>
		<div style="display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin:6px 0">
		<button class="pp">pause</button><button class="rs">restart</button>
		<label>samples per frame <select class="spf"><option>1</option><option selected>4</option><option>16</option><option>64</option></select></label>
		<label>bin width <select class="bw"><option value="0.004">0.004 m/s</option><option value="0.008" selected>0.008 m/s</option><option value="0.016">0.016 m/s</option></select></label>
		<label><input type="checkbox" class="sp"> show the density of a long record</label></div></div>
		<script>
		const root = currentScript.previousElementSibling, q = s => root.querySelector(s), NS = "http://www.w3.org/2000/svg";
		const M = 1.18, S1 = 0.035, T1 = 0.02, S2 = 0.012, T2 = 0.001, dt = 1e-4, W = 1000, SD = Math.sqrt(S1*S1 + S2*S2);
		const y0 = 1.03, y1 = 1.33, L = 62, R = 420, H0 = 20, H1 = 262, HL = 440, HR = 670;
		const Y = v => H1 - (v - y0) / (y1 - y0) * (H1 - H0), Xs = i => L + i / (W - 1) * (R - L);
		const gauss = () => { let u = 0, v = 0; while (u === 0) u = Math.random(); v = Math.random(); return Math.sqrt(-2*Math.log(u)) * Math.cos(2*Math.PI*v); };
		const a1 = Math.exp(-dt/T1), a2 = Math.exp(-dt/T2), b1 = S1*Math.sqrt(1-a1*a1), b2 = S2*Math.sqrt(1-a2*a2);
		let z1, z2, buf, all, playing = true, raf;
		const mk = (t, at) => { const e = document.createElementNS(NS, t); for (const k in at) e.setAttribute(k, at[k]); return e; };
		const ax = q(".ax");
		ax.appendChild(mk("line", {x1: L, x2: R, y1: H1, y2: H1, stroke: "#333"})); ax.appendChild(mk("line", {x1: L, x2: L, y1: H0, y2: H1, stroke: "#333"}));
		ax.appendChild(mk("line", {x1: HL, x2: HR, y1: H1, y2: H1, stroke: "#333"}));
		for (let v = 1.05; v < 1.31; v += 0.05) { ax.appendChild(mk("line", {x1: L-4, x2: L, y1: Y(v), y2: Y(v), stroke: "#333"}));
			const t = mk("text", {x: L-7, y: Y(v)+4, "text-anchor": "end", "font-size": 11, fill: "#333"}); t.textContent = v.toFixed(2); ax.appendChild(t); }
		const lab = (x, y, s, rot) => { const t = mk("text", {x, y, "text-anchor": "middle", "font-size": 12, fill: "#333"}); if (rot) t.setAttribute("transform", `rotate(-90 \${x} \${y})`); t.textContent = s; ax.appendChild(t); };
		lab(240, 285, "time (the last 0.1 s) →"); lab(555, 285, "number of samples in each bin →"); lab(16, 141, "axial velocity u (m/s)", true);
		const tt = mk("text", {x: L, y: 13, "font-size": 12, "font-weight": "bold", fill: "#333"}); tt.textContent = "probe signal"; ax.appendChild(tt);
		const mean = q(".mean"); mean.setAttribute("x1", L); mean.setAttribute("x2", HR); mean.setAttribute("y1", Y(M)); mean.setAttribute("y2", Y(M));
		function reset() { z1 = S1*gauss(); z2 = S2*gauss(); buf = []; all = []; for (let i = 0; i < W; i++) step(false); all = []; }
		function step(keep = true) { z1 = a1*z1 + b1*gauss(); z2 = a2*z2 + b2*gauss(); const v = M + z1 + z2; buf.push(v); if (buf.length > W) buf.shift(); if (keep) all.push(v); return v; }
		function draw() {
			q(".sig").setAttribute("d", "M" + buf.map((v, i) => Xs(i).toFixed(1) + "," + Y(v).toFixed(1)).join("L"));
			const last = buf[buf.length-1]; q(".dot").setAttribute("cx", R); q(".dot").setAttribute("cy", Y(last));
			const bw = +q(".bw").value, nb = Math.round((y1 - y0) / bw), c = new Array(nb).fill(0);
			for (const v of all) { const k = Math.floor((v - y0) / bw); if (k >= 0 && k < nb) c[k]++; }
			const pk = Math.max(10, ...c, all.length * bw / (Math.sqrt(2*Math.PI)*SD) * 1.05), sx = n => HL + n / pk * (HR - HL - 10);
			const g = q(".bars"); g.innerHTML = ""; const kl = Math.floor((last - y0) / bw);
			c.forEach((n, k) => { if (n > 0) g.appendChild(mk("rect", {x: HL, y: Y(y0 + (k+1)*bw) + 0.5, width: sx(n) - HL, height: Math.max(0.5, Y(y0 + k*bw) - Y(y0 + (k+1)*bw) - 1), fill: k === kl ? "#b85c00" : "#e1871d", opacity: 0.85})); });
			const pdf = q(".pdf");
			if (q(".sp").checked) { let d = ""; for (let v = y0; v <= y1; v += 0.002) { const n = all.length * bw * Math.exp(-0.5*((v-M)/SD)**2) / (Math.sqrt(2*Math.PI)*SD); d += (d ? "L" : "M") + sx(n).toFixed(1) + "," + Y(v).toFixed(1); } pdf.setAttribute("d", d); } else pdf.setAttribute("d", "");
			q(".cnt").textContent = "histogram: " + all.length + " samples";
		}
		function loop() { if (playing) { const k = +q(".spf").value; for (let i = 0; i < k; i++) step(); draw(); } raf = requestAnimationFrame(loop); }
		q(".pp").onclick = () => { playing = !playing; q(".pp").textContent = playing ? "pause" : "play"; };
		q(".rs").onclick = () => { reset(); draw(); };
		q(".bw").onchange = draw; q(".sp").onchange = draw;
		reset(); draw(); raf = requestAnimationFrame(loop);
		invalidation.then(() => cancelAnimationFrame(raf));
		</script>""")
	end

end;

# ╔═╡ aea33500-c9c8-5d28-aae8-5872949d5552
fig_temp_player()

# ╔═╡ 2b4c119d-d181-5796-b52e-3f585dc70cd2
md"""
Every day gets a different value, so the values cannot be counted one by one as for the dice. Figure 4.1 therefore groups them into **bins** of 1 °C. How should the vertical axis be chosen? Try all three choices, and change the bin width and the number of days.

!!! tasks "Tasks"
    1. Choose *count*. Make the bins narrower. What happens to the tallest bar?
    2. Choose *relative frequency*. Repeat. Does the height still depend on the bin width?
    3. Choose *density*. Repeat with 100 000 days. What do the bars approach?

Days $(@bind s4_N Select([100, 1000, 100_000]; default = 1000)) ``\qquad`` vertical axis $(@bind s4_scale Select(YSCALES)) ``\qquad`` bin width $(@bind s4_w Select([4.0 => "4 °C", 2.0 => "2 °C", 1.0 => "1 °C", 0.5 => "0.5 °C", 0.25 => "0.25 °C"]; default = 2.0))
"""

# ╔═╡ e8fb2d0c-e641-5157-a8c4-148a83648dd4
cap(fig_temp_scale(s4_N, s4_scale, s4_w), md"**Figure 4.2.** Histogram of the temperatures (orange) with the selected vertical axis. On the density scale, the blue curve is the density of the model.")

# ╔═╡ 074cb7b7-22db-4303-8b94-d8c4c82ae710
cap(pipe_probe(), md"**Figure 4.3.** Axial velocity in a pipe, measured by a probe (synthetic). Left: the signal of the last 0.1 s. Right: all samples so far, sorted into bins; the newest bin is dark. With the check box: the density of a long record (blue), scaled to the counts.")

# ╔═╡ c17db78a-2f6e-44f5-b7b5-53c2e647f53e
md"""
### 4.3 Probabilities as integrals

The integral is the area under the density.

Density $(@bind s4_ex Select(DENS_EX)) ``\qquad`` ``a`` $(@bind s4_ta Slider(0:0.01:1; default = 0.2)) ``\qquad`` ``b`` $(@bind s4_tb Slider(0:0.01:1; default = 0.5))
"""

# ╔═╡ 6aa339ec-330f-529d-b125-67ef6c086d7b
cap(fig_area(s4_ex, s4_ta, s4_tb), md"**Figure 4.5.** Left: the density; the shaded area is ``P(a \le X \le b)``. Right: the CDF.")

# ╔═╡ bfcd3871-3340-53f9-ab5f-bc02a64e4356
md"""
### 4.4 From the CDF back to values

The CDF goes from a value ``x`` to a height ``u = F_X(x)``. In the other direction, it goes from a height ``u`` to a value:

```math
x = F_X^{-1}(u).
```

Choose a distribution and a height ``u``. Then let many random heights between 0 and 1 run through the CDF, and compare the values obtained with the distribution.

!!! tasks "Tasks"
    1. For each distribution, compare the shape of the CDF with the shape of the density.
    2. Where is the CDF steep? Where do most of the random heights land on the ``x``-axis?

Distribution $(@bind s4_inv Select(INV_DISTS)) ``\qquad`` height ``u`` $(@bind s4_u Slider(0.01:0.01:0.99; default = 0.7, show_value = true))

Random heights $(@bind s4_m Select([20, 200, 2000]; default = 2000)) ``\qquad`` $(@bind s4_useed Seed(0; label = "New heights"))
"""

# ╔═╡ 458816be-fc49-545f-a426-d0a116bab369
cap(fig_inverse(s4_inv, s4_u, s4_m, s4_useed), md"**Figure 4.6.** Left: the CDF, read from a height ``u`` (purple) to the value ``x = F_X^{-1}(u)``; orange ticks are random heights. Right: the density (or PMF) and the values obtained from all random heights (orange).")

# ╔═╡ 694accf7-7fe4-597b-8a26-700c588d1318
# Section 4.2: two continuous random variables, penalty kicks (synthetic kickers)
begin
	const POST = 3.66; const BAR = 2.44                      # goal: 7.32 m wide, 2.44 m high
	const KX = collect(-5.5:0.05:5.5); const KY = collect(0.0:0.025:3.5)
	const DKX = 0.05; const DKY = 0.025

	const KICKERS = [
		"precise striker" => (ax = 2.3, ay = 0.8, sx = 0.45, sy = 0.3, ρ = 0.0),
		"wild striker" => (ax = 0.0, ay = 1.2, sx = 1.6, sy = 0.7, ρ = 0.0),
		"corner specialist" => (ax = -2.9, ay = 1.9, sx = 0.6, sy = 0.4, ρ = 0.0),
		"tilted striker" => (ax = 0.0, ay = 1.1, sx = 1.4, sy = 0.55, ρ = 0.75)]
	const KICKER_NAMES = first.(KICKERS)



	struct Kicker
		p::NamedTuple
		f::Matrix{Float64}          # f[i, j] = joint density at (KX[i], KY[j]), normalised on the grid
		fx::Vector{Float64}; fy::Vector{Float64}
	end

	function make_kicker(name, custom)
		p = name == "my own kicker" ? (ax = custom[1], ay = custom[2], sx = custom[3], sy = custom[4], ρ = custom[5]) : last(KICKERS[findfirst(==(name), first.(KICKERS))])
		f = [begin
				u = (x - p.ax) / p.sx; v = (y - p.ay) / p.sy
				exp(-(u^2 - 2p.ρ * u * v + v^2) / (2 * (1 - p.ρ^2)))
			end for x in KX, y in KY]
		f ./= sum(f) * DKX * DKY
		Kicker(p, f, vec(sum(f, dims = 2)) .* DKY, vec(sum(f, dims = 1)) .* DKX)
	end

	function shoot(K::Kicker, n, seed)
		rng = MersenneTwister(2026 + 101seed); p = K.p
		xs = Float64[]; ys = Float64[]
		while length(xs) < n
			u = randn(rng); w = randn(rng)
			x = p.ax + p.sx * u; y = p.ay + p.sy * (p.ρ * u + sqrt(1 - p.ρ^2) * w)
			(y >= 0 && abs(x) <= 5.5 && y <= 3.5) || continue
			push!(xs, x); push!(ys, y)
		end
		xs, ys
	end

	ontarget(x, y) = abs(x) <= POST && y <= BAR

	function goal!(f; lw = 5)
		for xn in -POST:0.61:POST
			seg!(f, xn, 0, xn, BAR; c = "#d8d8d4", w = 0.6)
		end
		for yn in 0:0.305:BAR
			seg!(f, -POST, yn, POST, yn; c = "#d8d8d4", w = 0.6)
		end
		seg!(f, -5.5, 0, 5.5, 0; c = "#6b8f4e", w = 3)
		seg!(f, -POST, 0, -POST, BAR; c = "#333", w = lw, over = true); seg!(f, POST, 0, POST, BAR; c = "#333", w = lw, over = true)
		seg!(f, -POST, BAR, POST, BAR; c = "#333", w = lw, over = true)
		f
	end

	goalfig(; w = 680, h = 270, title = "", kw...) = fig(; w = w, h = h, xr = (-5.5, 5.5), yr = (0, 3.5), xlabel = "sideways position x (m)", ylabel = "height y (m)", title = title, kw...)

	function fig_shots_player(K, seed)
		n = 300; xs, ys = shoot(K, n, seed)
		f = goalfig(h = 280); goal!(f)
		reads = ["Press ▶. Each dot is one penalty kick; the point is where the ball crosses the goal line."]; hits = 0
		for k in 1:n
			on = ontarget(xs[k], ys[k]); hits += on
			circle!(f, xs[k], ys[k], 4; fill = on ? ORANGE : "white", stroke = on ? "white" : ORANGE, sw = 1.4, s = k, over = true)
			push!(reads, "kick <b>$(k)</b>: x = $(fmt(xs[k]; d = 2)) m, y = $(fmt(ys[k]; d = 2)) m · on target so far: $(hits) of $(k) ($(round(Int, 100hits / k)) %)")
		end
		player(f; n = n, start = n, fps = 10, reads = reads, label = "kick",
			caption = md"**Figure 4.7.** Front view of the goal (7.32 m × 2.44 m). Filled dots: kicks inside the frame; open dots: kicks that miss. ")
	end

	# colour scale for densities: white → blue
	function heatcol(t)
		t = clamp(t, 0, 1)
		r = round(Int, 255 - t * (255 - 31)); g = round(Int, 255 - t * (255 - 90)); b = round(Int, 255 - t * (255 - 166))
		"rgb($(r),$(g),$(b))"
	end

	function heat!(f, K; step = 3, op = 1.0)
		m = maximum(K.f)
		print(f.body, "<defs><filter id=\"bl$(f.id)\" x=\"-5%\" y=\"-5%\" width=\"110%\" height=\"110%\"><feGaussianBlur stdDeviation=\"2.2\"/></filter></defs><g filter=\"url(#bl$(f.id))\">")
		for i in 1:step:length(KX)-step, j in 1:step:length(KY)-step
			v = K.f[i, j]; v < 0.02m && continue
			c = heatcol(v / m)
			rect!(f, KX[i], KY[j], KX[i+step], KY[j+step]; fill = c, op = op, stroke = c, sw = 0.6)
		end
		print(f.body, "</g>")
		f
	end

	# 3D surface with an optional slice at height y0 (PlutoPlotly)
	function surface3d(K; y0 = nothing)
		st = 4; xi = 1:st:length(KX); yi = 1:st:length(KY)
		tr = [surface(x = KX[xi], y = KY[yi], z = K.f[xi, yi], colorscale = [[0, "#ffffff"], [1, COL.blue]], showscale = false, opacity = 0.95,
			contours = attr(z = attr(show = true, usecolormap = true, project = attr(z = true))))]
		push!(tr, scatter3d(x = [-POST, -POST, POST, POST], y = [0, BAR, BAR, 0], z = zeros(4), mode = "lines", line = attr(color = "#333", width = 6), showlegend = false))
		if y0 !== nothing
			j = argmin(abs.(KY .- y0))
			push!(tr, scatter3d(x = KX, y = fill(KY[j], length(KX)), z = K.f[:, j], mode = "lines", line = attr(color = COL.purple, width = 7), showlegend = false))
			zmax = maximum(K.f)
			push!(tr, mesh3d(x = [-5.5, 5.5, 5.5, -5.5], y = fill(KY[j], 4), z = [0, 0, zmax, zmax], i = [0, 0], j = [1, 2], k = [2, 3], color = COL.purple, opacity = 0.18))
		end
		PlutoPlotly.plot(tr, Layout(height = 420, margin = attr(l = 0, r = 0, t = 10, b = 0),
			scene = attr(xaxis = attr(title = "x (m)"), yaxis = attr(title = "y (m)"), zaxis = attr(title = "f(x, y)"),
				aspectratio = attr(x = 1.6, y = 0.8, z = 0.6), camera = attr(eye = attr(x = 0.0, y = -2.1, z = 1.0)))))
	end
end;

# ╔═╡ c63e6f69-6c9f-5b05-b607-47d85be10609
fig_shots_player(make_kicker("tilted striker", nothing), 0)

# ╔═╡ 75ca6b65-c931-5805-abf8-16817a0d336b
surface3d(make_kicker("tilted striker", nothing))

# ╔═╡ ed28dc41-d9d5-5aac-bb20-9ee103cf7a96
# Section 4.4: two continuous random variables, standard joint densities
begin
	struct D2
		name::String
		xs::Vector{Float64}; ys::Vector{Float64}; dx::Float64; dy::Float64
		f::Matrix{Float64}; fx::Vector{Float64}; fy::Vector{Float64}
	end

	const D2_NAMES = ["bivariate normal", "f(x, y) = x + y on the unit square", "two independent exponentials"]

	function make_d2(name, ρ)
		if name == D2_NAMES[1]
			xs = collect(-3.5:0.05:3.5); ys = xs
			f = [exp(-(x^2 - 2ρ * x * y + y^2) / (2 * (1 - ρ^2))) / (2π * sqrt(1 - ρ^2)) for x in xs, y in ys]
		elseif name == D2_NAMES[2]
			xs = collect(0.0025:0.005:0.9975); ys = xs
			f = [x + y for x in xs, y in ys]
		else
			xs = collect(0.0:0.04:5.0); ys = xs
			f = [exp(-x - y) for x in xs, y in ys]
		end
		dx = xs[2] - xs[1]; dy = ys[2] - ys[1]
		f ./= sum(f) * dx * dy
		D2(name, xs, ys, dx, dy, f, vec(sum(f, dims = 2)) .* dy, vec(sum(f, dims = 1)) .* dx)
	end

	function heat2!(g, D; op = 1.0)
		m = maximum(D.f); n = length(D.xs); st = max(1, n ÷ 45)
		print(g.body, "<defs><filter id=\"b2$(g.id)\" x=\"-5%\" y=\"-5%\" width=\"110%\" height=\"110%\"><feGaussianBlur stdDeviation=\"1.6\"/></filter></defs><g filter=\"url(#b2$(g.id))\">")
		for i in 1:st:n-st, j in 1:st:length(D.ys)-st
			v = D.f[i, j]; v < 0.015m && continue
			c = heatcol(v / m)
			rect!(g, D.xs[i], D.ys[j], D.xs[i+st], D.ys[j+st]; fill = c, op = op, stroke = c, sw = 0.6)
		end
		print(g.body, "</g>")
		g
	end

	rng2(D) = (D.xs[1], D.xs[end]), (D.ys[1], D.ys[end])
	sq(D; w = 330, h = 300, kw...) = (r = rng2(D); fig(; w = w, h = h, xr = r[1], yr = r[2], xlabel = "x", ylabel = "y", kw...))

	# ---------------- regions and their probabilities ----------------
	function regions(D)
		D.name == D2_NAMES[2] ? ["X ≤ 1/2 and Y ≤ 1/2", "X < Y", "X + Y ≤ 1"] :
		D.name == D2_NAMES[1] ? ["X ≤ 0 and Y ≤ 0", "X < Y", "X > 1"] : ["X ≤ 1 and Y ≤ 1", "X < Y", "X + Y ≤ 2"]
	end
	function inreg(D, r, x, y)
		r == "X ≤ 1/2 and Y ≤ 1/2" ? (x <= 0.5 && y <= 0.5) : r == "X < Y" ? x < y : r == "X + Y ≤ 1" ? x + y <= 1 :
		r == "X ≤ 0 and Y ≤ 0" ? (x <= 0 && y <= 0) : r == "X > 1" ? x > 1 : r == "X ≤ 1 and Y ≤ 1" ? (x <= 1 && y <= 1) : x + y <= 2
	end
	regprob(D, r) = sum(D.f[i, j] for i in eachindex(D.xs), j in eachindex(D.ys) if inreg(D, r, D.xs[i], D.ys[j])) * D.dx * D.dy

	function regpoly(D, r)
		(x0, x1), (y0, y1) = rng2(D)
		r == "X ≤ 1/2 and Y ≤ 1/2" ? ([x0, 0.5, 0.5, x0], [y0, y0, 0.5, 0.5]) :
		r == "X ≤ 0 and Y ≤ 0" ? ([x0, 0, 0, x0], [y0, y0, 0, 0]) :
		r == "X ≤ 1 and Y ≤ 1" ? ([x0, 1, 1, x0], [y0, y0, 1, 1]) :
		r == "X < Y" ? ([x0, x1, x0], [y0, y1, y1]) :
		r == "X > 1" ? ([1, x1, x1, 1], [y0, y0, y1, y1]) :
		r == "X + Y ≤ 1" ? ([x0, 1, x0], [y0, y0, 1]) : ([x0, 2, x0], [y0, y0, 2])
	end

	function fig_region2(D, r)
		g = sq(D; w = 330, h = 300); heat2!(g, D)
		px, py = regpoly(D, r); poly!(g, px, py; fill = COL.purple, op = 0.18, stroke = COL.purple, sw = 2)
		g
	end

	# ---------------- marginals ----------------
	function fig_marg2(D, showcdf)
		(x0, x1), (y0, y1) = rng2(D)
		c = sq(D; w = 470, h = 300, l = 50, r = 6); heat2!(c, D)
		top = fig(w = 470, h = 150, xr = (x0, x1), yr = (0, showcdf ? 1.05 : 1.2 * maximum(D.fx)), ylabel = showcdf ? "F_X(x)" : "f_X(x)", l = 50, r = 6, b = 22, t = 10)
		side = fig(w = 170, h = 300, xr = (0, showcdf ? 1.05 : 1.2 * maximum(D.fy)), yr = (y0, y1), xlabel = showcdf ? "F_Y(y)" : "f_Y(y)", l = 34, r = 10, b = 46, t = 14)
		if showcdf
			line!(top, D.xs, cumsum(D.fx) .* D.dx; c = COL.green, w = 2.2); line!(side, cumsum(D.fy) .* D.dy, D.ys; c = COL.green, w = 2.2)
		else
			area!(top, D.xs, D.fx; c = COL.blue, op = 0.15); line!(top, D.xs, D.fx; c = COL.blue, w = 2.2)
			line!(side, D.fy, D.ys; c = COL.blue, w = 2.2)
		end
		vstack(top, figure(c, side))
	end

	# ---------------- conditional slices ----------------
	function fig_slice2(D, t)
		(x0, x1), (y0, y1) = rng2(D)
		j = clamp(round(Int, 1 + t * (length(D.ys) - 1)), 1, length(D.ys)); y = D.ys[j]
		raw = D.f[:, j]; fyv = D.fy[j]; cond = raw ./ fyv
		g = sq(D; w = 230, h = 230); heat2!(g, D); hline!(g, y; c = COL.purple, dash = nothing, w = 2.4)
		a = fig(w = 220, h = 230, xr = (x0, x1), yr = (0, 1.15 * maximum(D.f)), xlabel = "x", ylabel = "f_{X,Y}(x, y)", title = "slice as cut", l = 50)
		area!(a, D.xs, raw; c = COL.purple, op = 0.2); line!(a, D.xs, raw; c = COL.purple, w = 2.2)
		b = fig(w = 220, h = 230, xr = (x0, x1), yr = (0, 1.15 * max(maximum(cond), maximum(D.fx))), xlabel = "x", ylabel = "f_{X|Y}(x | y)", title = "divided by its area", l = 50)
		line!(b, D.xs, D.fx; c = COL.green, w = 1.8, dash = "5 4")
		area!(b, D.xs, cond; c = COL.purple, op = 0.2); line!(b, D.xs, cond; c = COL.purple, w = 2.4)
		mx = sum(D.xs .* cond) * D.dx
		figure(g, a, b, Markdown.parse("Slice at ``y = $(fmt(y; d = 3))``. Its area is ``f_Y($(fmt(y; d = 3))) = $(fmt(fyv; d = 3))``. The conditional density (purple, right) has the mean ``E[X \\mid Y = $(fmt(y; d = 3))] = $(fmt(mx; d = 3))``; the marginal ``f_X`` is dashed green."))
	end

	# ---------------- the joint CDF ----------------
	function fig_jcdf2(D, tx, ty)
		(x0, x1), (y0, y1) = rng2(D)
		i = clamp(round(Int, 1 + tx * (length(D.xs) - 1)), 1, length(D.xs)); j = clamp(round(Int, 1 + ty * (length(D.ys) - 1)), 1, length(D.ys))
		g = sq(D; w = 400, h = 300); heat2!(g, D)
		rect!(g, x0, y0, D.xs[i], D.ys[j]; fill = COL.purple, op = 0.2, stroke = COL.purple, sw = 2, over = true)
		dots!(g, [D.xs[i]], [D.ys[j]]; c = COL.purple, r = 6, stroke = "white", sw = 1.5, over = true)
		F = sum(D.f[1:i, 1:j]) * D.dx * D.dy; FX = sum(D.fx[1:i]) * D.dx
		figure(g, Markdown.parse("``F_{X,Y}($(fmt(D.xs[i]; d = 3)), $(fmt(D.ys[j]; d = 3))) = P(X \\le $(fmt(D.xs[i]; d = 3)),\\ Y \\le $(fmt(D.ys[j]; d = 3))) =`` **$(fmt(F; d = 3))**. For comparison ``F_X($(fmt(D.xs[i]; d = 3))) = $(fmt(FX; d = 3))``" *
			(ty >= 0.999 ? ": with ``y`` at the top they agree, ``F_X(x) = F_{X,Y}(x, \\infty)``." : ".")))
	end

	# ---------------- 3D view ----------------
	function surface3d2(D; t = nothing)
		st = max(1, length(D.xs) ÷ 60); xi = 1:st:length(D.xs); yi = 1:st:length(D.ys)
		tr = [surface(x = D.xs[xi], y = D.ys[yi], z = D.f[xi, yi], colorscale = [[0, "#ffffff"], [1, COL.blue]], showscale = false, opacity = 0.95)]
		if t !== nothing
			j = clamp(round(Int, 1 + t * (length(D.ys) - 1)), 1, length(D.ys)); zm = maximum(D.f)
			push!(tr, scatter3d(x = D.xs, y = fill(D.ys[j], length(D.xs)), z = D.f[:, j], mode = "lines", line = attr(color = COL.purple, width = 8), showlegend = false))
			push!(tr, mesh3d(x = [D.xs[1], D.xs[end], D.xs[end], D.xs[1]], y = fill(D.ys[j], 4), z = [0, 0, zm, zm], i = [0, 0], j = [1, 2], k = [2, 3], color = COL.purple, opacity = 0.15))
		end
		PlutoPlotly.plot(tr, Layout(height = 400, margin = attr(l = 0, r = 0, t = 10, b = 0),
			scene = attr(xaxis = attr(title = "x"), yaxis = attr(title = "y"), zaxis = attr(title = "f(x, y)"), camera = attr(eye = attr(x = 1.5, y = -1.6, z = 1.1)))))
	end
end;

# ╔═╡ 5d73f290-a06d-5ce0-b3d9-caafd26e3e12
md"""

!!! definition "Joint, marginal and conditional densities"
    Joint density: ``P\big((X, Y) \in A\big) = \iint_A f_{X,Y}(x, y)\,dx\,dy``. Joint CDF: ``F_{X,Y}(x, y) = P(X \le x,\ Y \le y)``.

    Marginal densities: ``f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dy``, ``\;f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y)\,dx``.

    Conditional density: ``f_{X \mid Y}(x \mid y) = \dfrac{f_{X,Y}(x, y)}{f_Y(y)}``. Independence: ``f_{X,Y}(x, y) = f_X(x)\,f_Y(y)``.

Density $(@bind s4_d2 Select(D2_NAMES)) ``\qquad`` correlation ``\rho`` (bivariate normal only) $(@bind s4_rho Slider(-0.9:0.1:0.9; default = 0.6, show_value = true)) ``\qquad`` 3D view $(@bind s4_3d CheckBox(default = false))
"""

# ╔═╡ 394f5ac6-eaf2-5c37-83d6-2984b2bf1f22
D = make_d2(s4_d2, s4_rho);

# ╔═╡ 5b3dff57-e4e6-59a0-a192-09408988719d
s4_3d ? surface3d2(D) : md""

# ╔═╡ cc96137c-6b21-562e-8db3-4a5580fc878b
md"""
**Volumes.** The probability of a region is the volume under the density over it.

Region $(@bind s4_reg Select(regions(D)))
"""

# ╔═╡ 9d147992-9b72-507b-8673-865ac8e16f9d
cap(vstack(fig_region2(D, s4_reg), Markdown.parse("``P(" * replace(s4_reg, " and " => ",\\ ") * ") =`` **$(fmt(regprob(D, s4_reg); d = 3))**")), md"**Figure 4.8.** The joint density (blue, darker is larger) and the selected region (purple).")

# ╔═╡ 74dfa6ff-24d1-563d-b024-944667c2f6e7
cap(fig_marg2(D, s4_mcdf), md"**Figure 4.9.** The joint density (centre) and its marginal densities ``f_X`` (top) and ``f_Y`` (right), or the marginal CDFs.")

# ╔═╡ 851491e7-b3a3-5bab-ad0a-65ee4c9f955d
cap(fig_slice2(D, s4_t), md"**Figure 4.10.** Left: the joint density with the slice (purple). Middle: the slice as cut. Right: the slice divided by its area, the conditional density ``f_{X \mid Y}(x \mid y)``, with the marginal ``f_X`` (dashed).")

# ╔═╡ 832b90a7-b6fc-590f-9416-26eb5fe47448
s4_s3d ? surface3d2(D; t = s4_t) : md""

# ╔═╡ d35f2b09-3eed-518e-8c9c-664bd66cfbf4
cap(fig_jcdf2(D, s4_cx, s4_cy), md"**Figure 4.11.** The joint CDF is the volume under the density over the shaded region.")

# ╔═╡ 94226bca-c7c5-52f6-8ce5-ed60e560a6c4
# Section 5: moments
begin
	fmt0(x; d = 3) = abs(x) < 1e-9 ? "0" : fmt(x; d = d)
	fulcrum!(f, m; c = COL.ink) = (h = 0.07 * (f.yr[2] - f.yr[1]); w = 0.025 * (f.xr[2] - f.xr[1]); poly!(f, [m - w, m + w, m], [-h, -h, 0]; fill = c, op = 1); f)
	function sdbar!(f, m, s, y; c = COL.teal, lab = "")
		seg!(f, m - s, y, m + s, y; c = c, w = 2.6)
		seg!(f, m - s, 0.96y, m - s, 1.04y; c = c, w = 2); seg!(f, m + s, 0.96y, m + s, 1.04y; c = c, w = 2)
		lab == "" || text!(f, m + s + 0.02 * (f.xr[2] - f.xr[1]), y, lab; c = c, size = 12, bold = true, bg = true)
		f
	end

	# ---------------- Figure 5.1: are these two distributions the same? ----------------
	const SG2 = Gamma(2, 1 / sqrt(2))
	const PAIRS = [
		"pair 1" => (x -> pdf(Normal(0, 1), x), x -> pdf(Normal(1.5, 1), x)),
		"pair 2" => (x -> pdf(Normal(0, 1), x), x -> pdf(Normal(0, 2), x)),
		"pair 3" => (x -> pdf(Normal(0, 1), x), x -> pdf(SG2, x + mean(SG2))),
		"pair 4" => (x -> pdf(Normal(0, 1), x), x -> pdf(TDist(5), x * sqrt(5 / 3)) * sqrt(5 / 3))]
	function fig_pair(name)
		f1, f2 = last(PAIRS[findfirst(==(name), first.(PAIRS))])
		xs = range(-5, 6, length = 400)
		mk(fun, t) = (g = fig(w = 330, h = 220, xr = (-5, 6), yr = (0, 0.6), xlabel = "x", ylabel = "density", title = t); area!(g, xs, fun.(xs); c = COL.blue, op = 0.18); line!(g, xs, fun.(xs); c = COL.blue, w = 2.2); g)
		figure(mk(f1, "distribution A"), mk(f2, "distribution B"))
	end

	# ---------------- Figure 5.2: mean and spread of standard distributions ----------------
	const STD_DISTS = ["fair die", "uniform U(0, 10)", "exponential Exp(1)", "normal N(0, 1)", "normal N(17, 4²)"]
	function std_grid(name)
		name == "fair die" && return (collect(1.0:6.0), fill(1 / 6, 6), true)
		d = name == STD_DISTS[2] ? Uniform(0, 10) : name == STD_DISTS[3] ? Exponential(1) : name == STD_DISTS[4] ? Normal(0, 1) : Normal(17, 4)
		lo, hi = name == STD_DISTS[2] ? (-1.0, 11.0) : name == STD_DISTS[3] ? (0.0, 6.0) : name == STD_DISTS[4] ? (-4.0, 4.0) : (2.0, 32.0)
		xs = collect(range(lo, hi, length = 2401)); w = pdf.(d, xs); w ./= sum(w)
		(xs, w, false)
	end
	function fig_balance(name)
		xs, w, disc = std_grid(name)
		M1 = sum(xs .* w); M2 = sum(xs .^ 2 .* w); V = M2 - M1^2; σ = sqrt(V)
		y = disc ? w : w ./ (xs[2] - xs[1]); ymax = 1.3 * maximum(y)
		f = fig(w = 680, h = 260, xr = (minimum(xs) - (disc ? 0.8 : 0), maximum(xs) + (disc ? 0.8 : 0)), yr = (-0.17ymax, ymax), xlabel = "x", ylabel = disc ? "p_X(x)" : "f_X(x)", grid = true)
		disc ? bars!(f, xs, y; width = 0.6, c = COL.blue) : (area!(f, xs, y; c = COL.blue, op = 0.2); line!(f, xs, y; c = COL.blue, w = 2.2))
		seg!(f, f.xr[1], 0, f.xr[2], 0; c = "#333", w = 1.4); fulcrum!(f, M1)
		sdbar!(f, M1, σ, 0.88ymax; lab = "E[X] ± σ")
		text!(f, M1, -0.12ymax, "E[X] = $(fmt0(M1; d = 4))"; dx = 18, size = 12, bold = true, bg = true)
		figure(f, Markdown.parse("``M_1[X] = E[X] = $(fmt0(M1; d = 4))``, ``\\;M_2[X] = E[X^2] = $(fmt0(M2; d = 4))``, ``\\;\\mathrm{Var}(X) = M_2 - M_1^2 = $(fmt0(V; d = 4))``, ``\\;\\sigma = $(fmt0(σ; d = 3))``."))
	end

	# ---------------- Figure 5.3: matching moments one by one ----------------
	const MG = collect(range(0, 10, length = 1001)); const MDX = MG[2] - MG[1]
	const MTARGETS = [
		"skewed" => x -> pdf(Gamma(2.5, 1.2), x),
		"two peaks" => x -> 0.65pdf(Normal(3, 0.75), x) + 0.35pdf(Normal(6.5, 0.75), x),
		"normal" => x -> pdf(Normal(6, 1.2), x),
		"sharp peak, heavy tails" => x -> pdf(Laplace(4.5, 0.8), x)]
	const MNAMES = ["area", "centroid (mean)", "spread (variance)", "asymmetry (skewness)", "tails (kurtosis)"]

	# area, mean, variance, skewness, kurtosis of a density on the grid MG
	function mstats(f)
		A = sum(f) * MDX; w = f .* MDX ./ A
		m = sum(w .* MG); v = sum(w .* (MG .- m) .^ 2); s = sqrt(v)
		[A, m, v, sum(w .* ((MG .- m) ./ s) .^ 3), sum(w .* ((MG .- m) ./ s) .^ 4)]
	end

	# the flattest density on [0, 10] with the same first k moments as the target (maximum entropy)
	function mflat(tf, k)
		k == 0 && return fill(0.1, length(MG))
		u = (MG .- 5) ./ 5; U = hcat([u .^ j for j in 1:k]...)
		t = tf.(MG); t ./= sum(t); m = U' * t
		dual(λ) = (e = U * λ; c = maximum(e); c + log(sum(exp.(e .- c))) - dot(λ, m))
		λ = zeros(k)
		for _ in 1:300
			e = U * λ; p = exp.(e .- maximum(e)); p ./= sum(p)
			g = U' * p .- m
			maximum(abs.(g)) < 1e-10 && break
			H = U' * (U .* p) .- (U' * p) * (U' * p)'
			st = (H + 1e-12I) \ g; τ = 1.0; D0 = dual(λ)
			while dual(λ .- τ .* st) > D0 - 1e-4τ * dot(g, st) && τ > 1e-8
				τ /= 2
			end
			λ .-= τ .* st
		end
		e = U * λ; p = exp.(e .- maximum(e)); p ./ (sum(p) * MDX)
	end

	function fig_match(tname, k)
		tf = last(MTARGETS[findfirst(==(tname), first.(MTARGETS))])
		t = tf.(MG); t ./= sum(t) * MDX
		q = mflat(tf, k)
		ts = mstats(t); qs = mstats(q)
		ym = 1.25 * max(maximum(t), maximum(q))
		f = fig(w = 680, h = 310, xr = (0, 10), yr = (-0.12ym, ym), xlabel = "x", ylabel = "density", grid = true, t = 34)
		area!(f, MG, t; c = COL.blue, op = 0.18); line!(f, MG, t; c = COL.blue, w = 2.2)
		line!(f, MG, q; c = ORANGE, w = 2.8)
		seg!(f, 0, 0, 10, 0; c = "#333", w = 1.2)
		k >= 1 && fulcrum!(f, ts[2])
		k >= 1 && text!(f, ts[2], -0.09ym, "centroid"; dx = 14, size = 11.5, bold = true, bg = true)
		k >= 2 && sdbar!(f, ts[2], sqrt(ts[3]), 0.9ym; c = COL.purple, lab = "mean ± σ")
		legend!(f, [("target", COL.blue, :line), ("flattest density with the same moments up to order $(k)", ORANGE, :line)]; at = :top)
		rows = ["- order $(j - 1), **$(MNAMES[j])**: target $(fmt0(ts[j]; d = 3)), orange $(fmt0(qs[j]; d = 3))" * (j - 1 <= k ? " ✓ matched" : "") for j in 1:5]
		figure(f, Markdown.parse(join(rows, "\n")))
	end

	# ---------------- Figure 5.4: the sign of the covariance ----------------
	const COV_CASES = ["temperature and ice-cream sales", "temperature and hot-drink sales", "two separate dice"]
	function fig_cov(case, seed)
		rng = MersenneTwister(919 + 7seed); n = 160
		x, y, xl, yl = case == COV_CASES[1] ? (t = 8 .+ 22 .* rand(rng, n); (t, 40 .+ 9 .* t .+ 25 .* randn(rng, n), "temperature (°C)", "ice creams sold")) :
			case == COV_CASES[2] ? (t = 8 .+ 22 .* rand(rng, n); (t, 420 .- 9 .* t .+ 25 .* randn(rng, n), "temperature (°C)", "hot drinks sold")) :
			(rand(rng, 1:6, n) .+ 0.25 .* (rand(rng, n) .- 0.5), rand(rng, 1:6, n) .+ 0.25 .* (rand(rng, n) .- 0.5), "first die (jittered)", "second die (jittered)")
		mx, my = mean(x), mean(y); pr = (x .- mx) .* (y .- my)
		lo, hi = extrema(x); ly, hy = extrema(y); px = 0.06 * (hi - lo); py = 0.08 * (hy - ly)
		f = fig(w = 560, h = 300, xr = (lo - px, hi + px), yr = (ly - py, hy + py), xlabel = xl, ylabel = yl, l = 62)
		rect!(f, mx, my, hi + px, hy + py; fill = COL.green, op = 0.06); rect!(f, lo - px, ly - py, mx, my; fill = COL.green, op = 0.06)
		rect!(f, lo - px, my, mx, hy + py; fill = COL.red, op = 0.06); rect!(f, mx, ly - py, hi + px, my; fill = COL.red, op = 0.06)
		vline!(f, mx; c = "#555"); hline!(f, my; c = "#555")
		dots!(f, x, y; c = [p >= 0 ? COL.green : COL.red for p in pr], r = 3.3, op = 0.85)
		figure(f, Markdown.parse("$(count(>=(0), pr)) of $(n) products ``(x_i - \\bar x)(y_i - \\bar y)`` are positive (green). Sample covariance: **$(fmt(sum(pr) / (n - 1); d = 3))**."))
	end

	# ---------------- Figure 5.5: distribution and data side by side ----------------
	function fig_dist_data(n, seed)
		rng = MersenneTwister(4321 + 23seed); v = rand(rng, 1:6, n)
		a = fig(w = 330, h = 240, xr = (0.3, 6.7), yr = (0, 0.45), xlabel = "x", ylabel = "p_X(x)", title = "the distribution", xticks = collect(1:6))
		bars!(a, collect(1:6), fill(1 / 6, 6); width = 0.6, c = COL.blue); vline!(a, 3.5; c = COL.blue, dash = nothing, w = 2.4)
		text!(a, 3.5, 0.41, "E[X] = 3.5"; dx = 6, size = 12, c = COL.blue, bold = true, bg = true)
		b = fig(w = 330, h = 240, xr = (0.3, 6.7), yr = (0, 0.45), xlabel = "x", ylabel = "relative frequency", title = "the data: $(n) rolls", xticks = collect(1:6))
		bars!(b, collect(1:6), [count(==(k), v) / n for k in 1:6]; width = 0.6, c = ORANGE); vline!(b, mean(v); c = ORANGE, dash = nothing, w = 2.4)
		text!(b, mean(v), 0.41, "x̄ = $(fmt(mean(v); d = 3))"; dx = 6, size = 12, c = ORANGE, bold = true, bg = true)
		figure(a, b, Markdown.parse("Distribution: ``E[X] = 3.5``, ``\\mathrm{Var}(X) = 2.917``. Data: ``\\bar x = $(fmt(mean(v); d = 3))``, ``s^2 = $(n > 1 ? fmt(var(v); d = 3) : "–")``."))
	end

	function fig_lln(seed)
		rng = MersenneTwister(77 + seed); N = 5000
		f = fig(w = 680, h = 280, xr = (1, N), yr = (1, 6), xlog = true, xlabel = "number of rolls N (log scale)", ylabel = "running mean", grid = true)
		for k in 1:8
			v = rand(rng, 1:6, N); idx = unique(round.(Int, exp.(range(0, log(N), length = 300))))
			line!(f, idx, (cumsum(v) ./ (1:N))[idx]; c = ORANGE, w = 1.3, op = 0.75)
		end
		hline!(f, 3.5; c = COL.blue, dash = nothing, w = 2.4)
		text!(f, N, 3.5, "E[X] = 3.5"; anchor = "end", dy = -10, size = 12, c = COL.blue, bold = true, bg = true)
		f
	end
end;

# ╔═╡ 98b45c5f-498c-59b2-9010-25dbe9a98850
md"""
## 5. Moments

### 5.1 Comparing distributions

Are the two distributions of each pair the same? If not, how do they differ, and how could the difference be measured?

Pair $(@bind s5_pair Select(first.(PAIRS)))
"""

# ╔═╡ 1b1650f9-1660-5065-bdeb-be226dff3ee1
cap(fig_pair(s5_pair), md"**Figure 5.1.** Two distributions, A and B.")

# ╔═╡ fd58d426-e175-5762-891d-baa3b756a663
md"""
To compare distributions, each is first summarised by numbers: one for the location, one for the spread, then more for the shape. These numbers are the **moments**.

!!! definition "Moments"
    The moment of order ``\mu`` of a random variable ``X``:
    ```math
    M_\mu[X] = \sum_x x^\mu\,p_X(x) \quad \text{(discrete)}, \qquad M_\mu[X] = \int_{-\infty}^{\infty} x^\mu\,f_X(x)\,dx \quad \text{(continuous)}.
    ```
    ``M_0 = 1``. ``M_1[X] = E[X]`` is the expectation (mean). ``M_2[X] = \mathrm{Var}(X) + E[X]^2``, so
    ```math
    \mathrm{Var}(X) = M_2 - M_1^2 = E\big[(X - E[X])^2\big], \qquad \sigma = \sqrt{\mathrm{Var}(X)}.
    ```

### 5.2 Mean and spread of standard distributions

Distribution $(@bind s5_dist Select(STD_DISTS))
"""

# ╔═╡ 630be91b-51e5-5227-a7c4-f0193c12e85e
cap(fig_balance(s5_dist), md"**Figure 5.2.** The distribution balances at ``E[X]`` (black triangle). The bar shows ``E[X] \pm \sigma``.")

# ╔═╡ 4a640ff4-c9e3-5073-ad51-c801b4394b54
md"""
### 5.4 Do two quantities vary together?

On a hot day, are more ice creams sold? More hot drinks? Does the first die say anything about the second die?

!!! tasks "Tasks"
    1. For each data set: when ``x`` increases, does ``y`` tend to increase, to decrease, or neither?
    2. Compare the numbers of green and red points.

Data $(@bind s5_cov Select(COV_CASES)) ``\qquad`` $(@bind s5_cseed Seed(0; label = "New data"))
"""

# ╔═╡ 07709070-ab1a-54f0-b181-690bc289173b
cap(fig_cov(s5_cov, s5_cseed), md"**Figure 5.4.** Pairs of values. Dashed lines: the means ``\bar x`` and ``\bar y``. Green: both values above or both below their means, so ``(x - \bar x)(y - \bar y) > 0``. Red: one above and one below, so the product is negative. The temperature data are synthetic.")

# ╔═╡ c8012528-97cf-53e3-8b48-a33de90c55a8
cap(fig_dist_data(s5_n, s5_dseed), md"**Figure 5.5.** Left: the distribution of a fair die and its mean. Right: data from ``n`` rolls and their mean.")

# ╔═╡ 1ddc53a0-aaaf-5793-bdc2-363f4cfb03cd
cap(fig_lln(s5_lseed), md"**Figure 5.6.** Running means of eight sequences of die rolls (orange) and ``E[X] = 3.5`` (blue).")

# ╔═╡ f12502d3-0daf-50cf-b4f2-dfa3f6088620
# Section 6: common distributions, the Galton board, a scale-free distribution
begin
	const FAMILY = ["binomial: Bin(n, p)", "uniform: U(a, b)", "normal: N(μ, σ²)", "Poisson: Poiss(λ)", "exponential: Exp(λ)"]

	# controls for each family (shown below the selector)
	function family_controls(name)
		name == FAMILY[1] ? PlutoUI.combine(Ch -> md"``n`` $(Ch(Slider(1:40; default = 20, show_value = true))) ``\qquad p`` $(Ch(Slider(0.05:0.05:0.95; default = 0.5, show_value = true)))") :
		name == FAMILY[2] ? PlutoUI.combine(Ch -> md"``a`` $(Ch(Slider(-3:0.5:2; default = 0, show_value = true))) ``\qquad b`` $(Ch(Slider(2.5:0.5:6; default = 4, show_value = true)))") :
		name == FAMILY[3] ? PlutoUI.combine(Ch -> md"``\mu`` $(Ch(Slider(-2:0.5:2; default = 0, show_value = true))) ``\qquad \sigma`` $(Ch(Slider(0.25:0.25:3; default = 1, show_value = true)))") :
		name == FAMILY[4] ? PlutoUI.combine(Ch -> md"``\lambda`` $(Ch(Slider(0.5:0.5:12; default = 2.5, show_value = true)))") :
		PlutoUI.combine(Ch -> md"``\lambda`` $(Ch(Slider(0.25:0.25:3; default = 1, show_value = true)))")
	end

	function family_dist(name, par)
		name == FAMILY[1] ? Binomial(par[1], par[2]) : name == FAMILY[2] ? Uniform(par[1], par[2]) :
		name == FAMILY[3] ? Normal(par[1], par[2]) : name == FAMILY[4] ? Poisson(par[1]) : Exponential(1 / par[1])
	end

	const FAMILY_TEXT = Dict(
		FAMILY[1] => md"``X`` counts the successes in ``n`` independent trials. Each trial has **two** results, success or failure, with the same success probability ``p``. ``P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}`` for ``k = 0, \dots, n``. Parameters ``n, p``; ``E[X] = np``, ``\mathrm{Var}(X) = np(1-p)``.",
		FAMILY[2] => md"All values in the interval ``[a, b]`` are equally likely: the density is flat. ``f_X(x) = 1/(b-a)`` on ``[a, b]``. Parameters ``a, b``; ``E[X] = (a+b)/2``, ``\mathrm{Var}(X) = (b-a)^2/12``.",
		FAMILY[3] => md"Bell-shaped and symmetric around ``\mu``. About 68 % of the values lie within ``\mu \pm \sigma``, about 95 % within ``\mu \pm 2\sigma``. ``f_X(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/(2\sigma^2)}``. Parameters ``\mu, \sigma``; ``E[X] = \mu``, ``\mathrm{Var}(X) = \sigma^2``.",
		FAMILY[4] => md"``X`` counts events in a fixed time interval; the events happen independently at a constant average rate ``\lambda``. ``P(X = k) = e^{-\lambda}\lambda^k/k!``. Parameter ``\lambda``; ``E[X] = \mathrm{Var}(X) = \lambda``.",
		FAMILY[5] => md"The waiting time until the next event, when events happen at a constant rate ``\lambda``. ``f_X(x) = \lambda e^{-\lambda x}``, ``F_X(x) = 1 - e^{-\lambda x}`` for ``x \ge 0``. Parameter ``\lambda``; ``E[X] = 1/\lambda``, ``\mathrm{Var}(X) = 1/\lambda^2``.")

	function fig_family(name, par, n, x0, seed)
		d = family_dist(name, par); disc = d isa DiscreteDistribution
		rng = MersenneTwister(66 + 9seed); smp = rand(rng, d, n)
		lo, hi = disc ? (minimum(d), max(maximum(d) == Inf ? quantile(d, 0.9995) : maximum(d), 1)) : (min(quantile(d, 0.0005), -1), max(quantile(d, 0.9995), 1))
		name == FAMILY[2] && ((lo, hi) = (-3.5, 6.5))
		name == FAMILY[5] && ((lo, hi) = (0.0, 6.0))
		name == FAMILY[3] && ((lo, hi) = (-8.0, 8.0))
		x0 = lo + x0 * (hi - lo); disc && (x0 = floor(x0))
		if disc
			ks = collect(Int(lo):Int(ceil(hi))); p = pdf.(d, ks); fr = [count(==(k), smp) / n for k in ks]
			f = fig(w = 330, h = 260, xr = (lo - 0.7, hi + 0.7), yr = (0, 1.3 * max(maximum(p), maximum(fr))), xlabel = "k", ylabel = "P(X = k)", title = "PMF and $(n) samples", l = 54)
			bars!(f, ks, p; width = 0.8, c = [k <= x0 ? COL.blue : "#c9d6ea" for k in ks], op = 1)
			bars!(f, ks, fr; width = 0.4, c = ORANGE, op = 0.9)
			g = fig(w = 330, h = 260, xr = (lo - 0.7, hi + 0.7), yr = (0, 1.05), xlabel = "k", ylabel = "F_X(k)", title = "CDF", l = 54)
			xs = [lo - 0.7; ks; hi + 0.7]; Fs = [0.0; cdf.(d, ks); 1.0]
			for k in 1:length(xs)-1
				seg!(g, xs[k], Fs[k], xs[k+1], Fs[k]; c = COL.green, w = 2.4)
			end
			length(ks) <= 41 && dots!(g, ks, cdf.(d, ks); c = COL.green, r = 3)
		else
			xs = collect(range(lo, hi, length = 400)); y = pdf.(d, xs)
			e = collect(range(lo, hi, length = 41)); h = histdata(smp, e)
			f = fig(w = 330, h = 260, xr = (lo, hi), yr = (0, 1.3 * max(maximum(y), maximum(h))), xlabel = "x", ylabel = "f_X(x)", title = "PDF and $(n) samples", l = 54)
			hist!(f, e, h; c = ORANGE, op = 0.7)
			inside = filter(<=(x0), xs); isempty(inside) || area!(f, inside, pdf.(d, inside); c = COL.blue, op = 0.35)
			line!(f, xs, y; c = COL.blue, w = 2.2)
			g = fig(w = 330, h = 260, xr = (lo, hi), yr = (0, 1.05), xlabel = "x", ylabel = "F_X(x)", title = "CDF", l = 54)
			line!(g, xs, cdf.(d, xs); c = COL.green, w = 2.4)
		end
		F0 = cdf(d, x0); vline!(g, x0; c = COL.blue); dots!(g, [x0], [F0]; c = COL.blue, r = 5)
		vline!(f, x0; c = COL.blue)
		figure(f, g, Markdown.parse("``P(X \\le $(fmt(x0; d = 3))) = F_X($(fmt(x0; d = 3))) =`` **$(fmt(F0; d = 3))** (shaded on the left). ``E[X] = $(fmt(mean(d); d = 3))``, ``\\mathrm{Var}(X) = $(fmt(var(d); d = 3))``; from the samples: ``\\bar x = $(fmt(mean(smp); d = 3))``, ``s^2 = $(fmt(var(smp); d = 3))``."))
	end

	# ---------------- Figure 6.2: the Galton board ----------------
	function fig_galton(nrows, p, nballs, seed)
		rng = MersenneTwister(1877 + 1009seed)
		R = rand(rng, nballs, nrows) .< p
		slot = vec(sum(R, dims = 2))
		n = nrows; xl = n / 2 + 0.9
		cnt = zeros(Int, n + 1); hgt = zeros(Int, nballs)
		for b in 1:nballs
			cnt[slot[b]+1] += 1; hgt[b] = cnt[slot[b]+1]
		end
		μ = n * p; σ = sqrt(n * p * (1 - p))
		expmax = nballs * maximum(pdf.(Binomial(n, p), 0:n))
		top = 1.15max(maximum(cnt), expmax)
		pileH = 0.95n + 2; u = pileH / top; base = -(n + 1.2) - pileH
		f = fig(w = 640, h = 540, xr = (-xl, xl), yr = (base - 0.3, 0.8), axes = false, l = 6, r = 6, t = 6, b = 22)
		for i in 1:n, j in 0:i-1
			circle!(f, j - (i - 1) / 2, -i + 0.5, n > 14 ? 2.6 : 3.4; fill = "#555", stroke = "none")
		end
		for k in 0:n+1
			seg!(f, k - n / 2 - 0.5, -(n + 0.6), k - n / 2 - 0.5, base; c = "#c8c8c2", w = 1)
		end
		seg!(f, -xl, base, xl, base; c = "#333", w = 1.2)
		for k in 0:n
			text!(f, k - n / 2, base - 0.25, string(k); anchor = "middle", size = n > 14 ? 9 : 11, c = "#444", dy = 8)
		end
		for b in 1:nballs
			x = 0.0; px = [0.0]; py = [0.8]
			for i in 1:n
				push!(px, x); push!(py, -i + 0.5 + 0.25)
				x += R[b, i] ? 0.5 : -0.5
			end
			push!(px, x); push!(py, -(n + 0.6))
			line!(f, px, py; c = COL.purple, w = 2.2, o = b, over = true)
			k = slot[b]
			rect!(f, k - n / 2 - 0.4, base + (hgt[b] - 1) * u, k - n / 2 + 0.4, base + hgt[b] * u; fill = ORANGE, op = 0.85, s = b)
		end
		ks = 0:n; pk = pdf.(Binomial(n, p), ks)
		dots!(f, collect(ks) .- n / 2, base .+ nballs .* pk .* u; c = COL.blue, r = 3.8, s = nballs, over = true)
		zz = range(-0.5, n + 0.5, length = 200)
		line!(f, zz .- n / 2, base .+ nballs .* pdf.(Normal(μ, σ), zz) .* u; c = COL.blue, w = 2.2, s = nballs, over = true)
		reads = ["ball <b>$(b)</b>: " * join([R[b, i] ? "R" : "L" for i in 1:n], " ") * " → $(slot[b]) bounces to the right → slot $(slot[b])" for b in 1:nballs]
		reads = vcat(["Press ▶. At every row the ball bounces left (L) or right (R)."], reads)
		reads[end] *= " · all balls dropped: blue dots = binomial PMF × $(nballs); blue curve = normal density with μ = np = $(fmt(μ)) and σ = √(np(1−p)) = $(fmt(σ; d = 3)), times $(nballs)"
		player(f; n = nballs, start = nballs, fps = 25, reads = reads, label = "ball",
			caption = md"**Figure 6.2.** The Galton board. At every row the ball goes right with probability ``p``, independently of the other rows. The slot is the sum of the ``n`` values ``X_i`` (0 or 1), so it follows ``\mathrm{Bin}(n, p)``.")
	end

	# ---------------- Figure 6.3: sums of random variables ----------------
	const CLT_TERMS = ["uniform on [0, 1]", "exponential (skewed)", "coin: 0 or 1", "die: 1 to 6"]
	function fig_clt(term, k, seed)
		rng = MersenneTwister(12 + 1009seed)
		draw, μ1, σ1, disc = Dict(CLT_TERMS[1] => (() -> rand(rng), 0.5, sqrt(1 / 12), false), CLT_TERMS[2] => (() -> randexp(rng), 1.0, 1.0, false),
			CLT_TERMS[3] => (() -> Float64(rand(rng, Bool)), 0.5, 0.5, true), CLT_TERMS[4] => (() -> Float64(rand(rng, 1:6)), 3.5, sqrt(35 / 12), true))[term]
		one = [draw() for _ in 1:20000]
		a = if disc
			ks = sort(unique(one)); g = fig(w = 260, h = 250, xr = (minimum(ks) - 0.7, maximum(ks) + 0.7), yr = (0, 0.6), xlabel = "value of X_i", ylabel = "relative frequency", title = "one term X_i", l = 52)
			bars!(g, ks, [count(==(v), one) / length(one) for v in ks]; width = 0.5, c = ORANGE); g
		else
			e = collect(range(0, maximum(one) < 1.01 ? 1 : 6, length = 31)); h = histdata(one, e)
			g = fig(w = 260, h = 250, xr = (e[1], e[end]), yr = (0, 1.2maximum(h)), xlabel = "value of X_i", ylabel = "density", title = "one term X_i", l = 52)
			hist!(g, e, h; c = ORANGE, op = 0.8); g
		end
		s = [sum(draw() for _ in 1:k) for _ in 1:20000]; z = (s .- k * μ1) ./ (σ1 * sqrt(k))
		edges = collect(-4:0.2:4); hz = histdata(z, edges)
		b = fig(w = 410, h = 250, xr = (-4, 4), yr = (0, max(0.6, 1.1maximum(hz))), xlabel = "(S_k − kμ) / (σ√k)", ylabel = "density", title = "sum S_k of k = $(k) terms, standardised")
		hist!(b, edges, hz; c = ORANGE, op = 0.75)
		xs = range(-4, 4, length = 300); line!(b, xs, pdf.(Normal(), xs); c = COL.blue, w = 2.2)
		figure(a, b)
	end

	# ---------------- Figure 6.4: zooming out ----------------
	const ZOOMS = ["power law", "exponential", "normal"]
	function fig_zoom(name)
		fx = name == "power law" ? (t -> 1.5 * t^(-2.5)) : name == "exponential" ? (t -> exp(-t / 3) / 3) : (t -> pdf(Normal(0, 30), t))
		panels = Any[]
		for a in (1.0, 10.0, 100.0)
			uu = range(1, 10, length = 160); dens = fx.(a .* uu)
			g = fig(w = 222, h = 210, xr = (1, 10), yr = (0, 1.12), xlabel = "x / $(round(Int, a))", ylabel = a == 1 ? "density, rescaled" : "", title = "window $(round(Int, a)) to $(round(Int, 10a))", l = a == 1 ? 50 : 30, xticks = [1, 4, 7, 10])
			m = maximum(dens)
			if m > 1e-300
				area!(g, uu, dens ./ m; c = COL.blue, op = 0.15); line!(g, uu, dens ./ m; c = COL.blue, w = 2.2)
			else
				text!(g, 5.5, 0.5, "practically 0 here"; anchor = "middle", size = 12, c = "#777")
			end
			push!(panels, g)
		end
		figure(panels...)
	end

end;

# ╔═╡ 8eeeaaa8-93de-5388-b7d8-e51183ec8d9c
md"""
## 6. Common distributions

Family $(@bind s6_fam Select(FAMILY; default = FAMILY[1]))
"""

# ╔═╡ ae2f29c6-0af2-50fa-9fcb-c5e87a06c254
@bind s6_par family_controls(s6_fam)

# ╔═╡ 53f2a6d6-ea97-5a96-bc55-9bfa732b7636
cap(vstack(FAMILY_TEXT[s6_fam], fig_family(s6_fam, s6_par, s6_n, s6_x, s6_seed)), md"**Figure 6.1.** Left: the PMF or PDF (blue) and samples (orange); the shading is ``P(X \le x)``. Right: the CDF.")

# ╔═╡ db618c05-95e8-5dbb-babb-8de2780772bb
fig_galton(s6_rows, s6_p, s6_balls, s6_gseed)

# ╔═╡ e0503400-df36-58d1-9937-2ed7489d4df3
md"""
The pile of balls looks like a normal curve. Does this need values 0 and 1? In Figure 6.3, choose the distribution of one random variable ``X_i`` and add ``k`` independent copies of it: ``S_k = X_1 + \dots + X_k``.


Distribution of each ``X_i`` $(@bind s6_term Select(CLT_TERMS)) ``\qquad`` number of terms ``k`` $(@bind s6_k Select([1, 2, 3, 5, 10, 30, 100]; default = 2)) ``\qquad`` $(@bind s6_cseed Seed(0; label = "New sums"))
"""

# ╔═╡ 9c388a7d-9c4f-5896-bd90-70ce6398c55c
cap(fig_clt(s6_term, s6_k, s6_cseed), md"**Figure 6.3.** Left: the distribution of one term ``X_i``. Right: the sum ``S_k`` of ``k`` independent terms, shifted and scaled to mean 0 and standard deviation 1 (orange), and the standard normal density (blue).")

# ╔═╡ 588e4463-82a0-5b71-aeec-ed90848ede23
md"""
!!! theorem "Central limit theorem"
    Let ``X_1, X_2, \dots, X_k`` be independent random variables with the same distribution, with mean ``\mu`` and finite variance ``\sigma^2``. For large ``k``, the sum
    ```math
    S_k = X_1 + X_2 + \dots + X_k \quad \text{is approximately} \quad \mathcal N(k\mu,\ k\sigma^2),
    ```
    whatever the distribution of each ``X_i``. This is why the normal distribution appears so often: many measured quantities are sums of many small random parts.

### 6.2 Distributions without a scale

Zoom out on a distribution: look at the windows ``[1, 10]``, ``[10, 100]`` and ``[100, 1000]``, each rescaled to the same axes. Does the picture change?

Distribution $(@bind s6_zoom Select(ZOOMS))
"""

# ╔═╡ 771e7984-cccf-50cc-adbd-8689714b3ac9
cap(fig_zoom(s6_zoom), md"**Figure 6.4.** The same density in three windows, each rescaled to ``[1, 10]`` and to a maximum of 1. For the power law ``f(x) \propto x^{-2.5}``, all three windows look the same: there is no characteristic scale. For the exponential and the normal distribution, they do not.")

# ╔═╡ 161d8b2d-967b-5f49-a2d2-0e223b7aa286
# Section 7: why sampling
begin
	function fig_case1(m, seed)
		rng = MersenneTwister(31 + 3seed); us = rand(rng, m); xsmp = quantile.(TMODEL, us)
		xs = collect(range(TLO, THI, length = 400))
		g = fig(w = 330, h = 270, xr = (TLO, THI), yr = (0, 1.05), xlabel = "temperature x (°C)", ylabel = "u = F_X(x)", title = "uniform numbers u → CDF")
		line!(g, xs, cdf.(TMODEL, xs); c = COL.green, w = 2.4)
		for (u, x) in zip(us[1:min(m, 25)], xsmp[1:min(m, 25)])
			seg!(g, TLO, u, x, u; c = ORANGE, w = 0.9, op = 0.7); seg!(g, x, u, x, 0; c = ORANGE, w = 0.9, op = 0.7)
		end
		e = collect(TLO:1.0:THI); h = histdata(xsmp, e)
		f = fig(w = 330, h = 270, xr = (TLO, THI), yr = (0, 1.3 * max(maximum(h), pdf(TMODEL, 17))), xlabel = "temperature x (°C)", ylabel = "density", title = "$(m) simulated noon temperatures")
		hist!(f, e, h; c = ORANGE, op = 0.8); line!(f, xs, pdf.(TMODEL, xs); c = COL.blue, w = 2.2)
		figure(g, f)
	end

	hiker(x, y) = 0.6exp(-((x - 3)^2 + (y - 6.5)^2) / 1.2) + 0.3exp(-((x - 7)^2 + (y - 3)^2) / 0.6) + 0.1exp(-((x - 6.5)^2 + (y - 7.5)^2) / 3)
	function fig_case2(showpts, seed)
		f = fig(w = 400, h = 380, xr = (0, 10), yr = (0, 10), xlabel = "east (km)", ylabel = "north (km)")
		m = hiker(3, 6.5); st = 0.2
		print(f.body, "<defs><filter id=\"bh$(f.id)\"><feGaussianBlur stdDeviation=\"1.5\"/></filter></defs><g filter=\"url(#bh$(f.id))\">")
		for x in 0:st:10-st, y in 0:st:10-st
			v = hiker(x + st / 2, y + st / 2) / m; v < 0.02 && continue
			c = heatcol(v); rect!(f, x, y, x + st, y + st; fill = c, op = 1, stroke = c, sw = 0.5)
		end
		print(f.body, "</g>")
		if showpts
			rng = MersenneTwister(5 + seed); px = 10 .* rand(rng, 200); py = 10 .* rand(rng, 200)
			dots!(f, px, py; c = ORANGE, r = 2.6, op = 0.9)
		end
		f
	end

	score(x, y) = round(Int, 10 * exp(-((x - 0.45)^2 + (y + 0.3)^2) / 0.02) + 6 * exp(-((x + 0.5)^2 + (y - 0.4)^2) / 0.08) + 2 * exp(-(x^2 + y^2) / 0.5))
	function parse_aims(s)
		out = Tuple{Float64, Float64}[]
		for part in split(s, ';')
			v = [tryparse(Float64, strip(t)) for t in split(part, ',')]
			length(v) == 2 && all(!isnothing, v) && push!(out, (clamp(v[1], -1, 1), clamp(v[2], -1, 1)))
		end
		out[1:min(end, 15)]
	end
	function fig_case3(aims, reveal)
		f = fig(w = 400, h = 400, xr = (-1, 1), yr = (-1, 1), xlabel = "x", ylabel = "y")
		if reveal
			st = 0.025
			for x in -1:st:1-st, y in -1:st:1-st
				v = score(x + st / 2, y + st / 2) / 10; v <= 0 && continue
				c = heatcol(v); rect!(f, x, y, x + st, y + st; fill = c, op = 1, stroke = c, sw = 0.4)
			end
		end
		for r in (0.25, 0.5, 0.75, 1.0)
			θ = range(0, 2π, length = 100); line!(f, r .* cos.(θ), r .* sin.(θ); c = "#bbb", w = 1)
		end
		rng = MersenneTwister(2718); tot = 0; pts = String[]
		for (k, (ax, ay)) in enumerate(aims)
			x = ax + 0.06randn(rng); y = ay + 0.06randn(rng); sc = score(x, y); tot += sc
			dots!(f, [x], [y]; c = ORANGE, r = 5, stroke = COL.ink, sw = 1)
			text!(f, x, y, string(sc); dx = 8, size = 12, bold = true, bg = true)
			push!(pts, "$(k): $(sc)")
		end
		figure(f, Markdown.parse(isempty(aims) ? "No arrows yet. Type aim points above, for example `0, 0; 0.5, -0.5`." :
			"Arrows used: $(length(aims)) of 15. Scores: " * join(pts, ", ") * ". **Total: $(tot)**."))
	end
end;

# ╔═╡ d7c1b120-6198-5168-843d-e16057a82d6a
cap(fig_case1(s7_m, s7_seed), md"**Figure 7.1.** Left: uniform random heights ``u`` mapped through the CDF of ``\mathcal N(17, 4^2)``. Right: the temperatures obtained, against the density.")

# ╔═╡ b14f55d6-7b6c-5628-b395-abb92aec9a9d
cap(fig_case2(s7_pts, 0), md"**Figure 7.2.** The density of the hiker's position (blue, darker is larger).")

# ╔═╡ a84ed3ba-8704-5438-9cbc-8e2fc3182a51
cap(fig_case3(parse_aims(s7_aims), s7_rev), md"**Figure 7.3.** Where the arrows landed and their scores. With the checkbox, the hidden score map is shown (blue, darker is more points).")

# ╔═╡ 92f42b3b-9842-583f-8e6d-6e207b3bdbb7
# Section 8: notation check
begin
	# colours used in the formulas: random variable, value, event, condition
	const NC = (rv = "#1f5aa6", val = "#d9730d", ev = "#2f7d55", cond = "#6a4c93")
	const NOTATION = [
		"P(A ∩ B)" => (raw"P({\color{#2f7d55}A} \cap {\color{#2f7d55}B}) = P({\color{#2f7d55}B} \cap {\color{#2f7d55}A})",
			"the probability of A and B",
			"Both events happen. The order of writing does not matter: it is the same event."),
		"P(A | B)" => (raw"P({\color{#2f7d55}A} \mid {\color{#6a4c93}B}) = \dfrac{P({\color{#2f7d55}A} \cap {\color{#6a4c93}B})}{P({\color{#6a4c93}B})}",
			"the probability of A given B",
			"The condition ``B`` is known to have happened, so only the outcomes in ``B`` count. In general ``P(A \\mid B) \\ne P(B \\mid A)``."),
		"P(X₁ = a, X₂ = b)" => (raw"P({\color{#1f5aa6}X_1} = {\color{#d9730d}a},\ {\color{#1f5aa6}X_2} = {\color{#d9730d}b}) = P({\color{#1f5aa6}X_1} = {\color{#d9730d}a})\, P({\color{#1f5aa6}X_2} = {\color{#d9730d}b} \mid {\color{#6a4c93}X_1 = a})",
			"the probability that X₁ is a and then X₂ is b",
			"A **sequence**: the index gives the position in time. Swapping the values gives another sequence, in general with another probability."),
		"X and x" => (raw"P({\color{#1f5aa6}X} = {\color{#d9730d}x})",
			"the probability that the random variable X takes the value x",
			"Capital ``X``: the random variable, a rule that assigns a number to every outcome. Small ``x``: one possible value. ``\\{X = x\\}`` is an event."),
		"p_X(x)" => (raw"p_{\color{#1f5aa6}X}({\color{#d9730d}x}) = P({\color{#1f5aa6}X} = {\color{#d9730d}x})",
			"the probability mass function (PMF) of X at x",
			"For a **discrete** random variable. Each value is a probability between 0 and 1, and all values add up to 1."),
		"f_X(x)" => (raw"P({\color{#d9730d}a} \le {\color{#1f5aa6}X} \le {\color{#d9730d}b}) = \int_{\color{#d9730d}a}^{\color{#d9730d}b} f_{\color{#1f5aa6}X}({\color{#d9730d}x})\,d{\color{#d9730d}x}",
			"the probability density function (PDF) of X at x",
			"For a **continuous** random variable. Probabilities are areas under ``f_X``."),
		"F_X(x)" => (raw"F_{\color{#1f5aa6}X}({\color{#d9730d}x}) = P({\color{#1f5aa6}X} \le {\color{#d9730d}x})",
			"the cumulative distribution function (CDF) of X at x",
			"For discrete and continuous random variables alike. It rises from 0 to 1 and never decreases."),
		"f_{X,Y}, f_X, f_{X|Y}" => (raw"f_{{\color{#1f5aa6}X} \mid {\color{#6a4c93}Y}}({\color{#d9730d}x} \mid {\color{#6a4c93}y}) = \dfrac{f_{{\color{#1f5aa6}X},{\color{#6a4c93}Y}}({\color{#d9730d}x}, {\color{#6a4c93}y})}{f_{\color{#6a4c93}Y}({\color{#6a4c93}y})}",
			"the conditional density of X given Y = y",
			"``f_{X,Y}`` is the joint density. The marginal density ``f_X(x) = \\int f_{X,Y}(x, y)\\,dy`` integrates the other variable out. The conditional density is a slice of the joint density at ``Y = y``, divided by its area."),
		"X ~ N(μ, σ²)" => (raw"{\color{#1f5aa6}X} \sim \mathcal{N}(\mu, \sigma^2)",
			"X is distributed as a normal distribution with mean μ and variance σ²",
			""),
		"M_μ[X], E[X], Var(X)" => (raw"M_\mu[{\color{#1f5aa6}X}] = \sum_{\color{#d9730d}x} {\color{#d9730d}x}^\mu\, p_{\color{#1f5aa6}X}({\color{#d9730d}x}), \qquad E[{\color{#1f5aa6}X}] = M_1, \qquad \mathrm{Var}({\color{#1f5aa6}X}) = M_2 - M_1^2",
			"the moment of order μ; the expectation; the variance",
			""),
		"Cov(X, Y)" => (raw"\mathrm{Cov}({\color{#1f5aa6}X}, {\color{#1f5aa6}Y}) = E\big[({\color{#1f5aa6}X} - E[{\color{#1f5aa6}X}])({\color{#1f5aa6}Y} - E[{\color{#1f5aa6}Y}])\big]",
			"the covariance of X and Y",
			"Positive if ``X`` and ``Y`` tend to increase together; negative if one tends to increase while the other decreases; 0 for independent random variables."),
		"x̄ and X̄_N" => (raw"\bar{\color{#1f5aa6}X}_N = \frac{1}{N}\sum_{i=1}^N {\color{#1f5aa6}X_i} \;\longrightarrow\; E[{\color{#1f5aa6}X}]",
			"X bar: the sample mean of N samples",
			""),
		"i.i.d." => (raw"{\color{#1f5aa6}X_1}, {\color{#1f5aa6}X_2}, \dots, {\color{#1f5aa6}X_N} \;\text{i.i.d.}",
			"independent and identically distributed",
			"Separate draws from the same distribution, none influencing another.")]
	const NOTATION_KEYS = first.(NOTATION)

	function notation_card(key)
		f, read, say = last(NOTATION[findfirst(==(key), NOTATION_KEYS)])
		Markdown.MD(Markdown.Admonition("notation", key, Any[
			Markdown.parse("```math\n" * f * "\n```"),
			Markdown.parse("*Read:* " * read * "."),
			Markdown.parse(say)]))
	end
end;

# ╔═╡ 583c818f-35c5-51b2-ab3e-844c5aad4f90
md"""
## 8. Notation check

Colours in the formulas: ``{\color{#1f5aa6}X}`` a random variable, ``{\color{#d9730d}x}`` a value, ``{\color{#2f7d55}A}`` an event, ``{\color{#6a4c93}B}`` a condition (what is known).

Symbol $(@bind s8_key Select(NOTATION_KEYS))
"""

# ╔═╡ 5f072d1c-5554-5a3f-8c4a-557f943cf73e
notation_card(s8_key)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoPlotly = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
AbstractPlutoDingetjes = "~1.4.1"
Distributions = "~0.25.131"
HypertextLiteral = "~1.0.0"
PlutoPlotly = "~0.6.6"
PlutoUI = "~0.7.83"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.13.3"
manifest_format = "2.1"
project_hash = "30fe68203181d28d0bc7de850e5651cd0c057424"

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

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
registries = "General"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "61761f58648aa7217445f24f841839b78c712232"
registries = "General"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.3"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
registries = "General"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "291665b547f137df070e4dd83e432b5fee8cc4a0"
registries = "General"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.2"

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

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
registries = "General"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
registries = "General"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

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

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "4657a834b01ce00f2b94546095c4350617b9af2c"
registries = "General"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.10.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
registries = "General"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "1.0.0"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "Zstd_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.22.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "PCRE2_jll", "Zlib_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.7+0"

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

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
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
git-tree-sha1 = "94ba93778373a53bfd5a0caaf7d809c445292ff4"
registries = "General"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.2"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.47.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
registries = "General"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
registries = "General"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
registries = "General"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "Zstd_jll", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.13.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Colors", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "6256ab3ee24ef079b3afa310593817e069925eeb"
registries = "General"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.23"

    [deps.PlotlyBase.extensions]
    DataFramesExt = "DataFrames"
    DistributionsExt = "Distributions"
    IJuliaExt = "IJulia"
    JSON3Ext = "JSON3"

    [deps.PlotlyBase.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"

[[deps.PlutoPlotly]]
deps = ["AbstractPlutoDingetjes", "Artifacts", "ColorSchemes", "Colors", "Dates", "Downloads", "HypertextLiteral", "InteractiveUtils", "LaTeXStrings", "Markdown", "Pkg", "PlotlyBase", "PrecompileTools", "Reexport", "ScopedValues", "Scratch", "TOML"]
git-tree-sha1 = "2b9e3d771adfe535a4fdda855f4741fdaacd3f7f"
registries = "General"
uuid = "8e989ff0-3d88-8e9f-f020-2b208a939ff0"
version = "0.6.6"

    [deps.PlutoPlotly.extensions]
    PlotlyKaleidoExt = "PlotlyKaleido"
    UnitfulExt = "Unitful"

    [deps.PlutoPlotly.weakdeps]
    PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.REPL]]
deps = ["Base64", "Dates", "FileWatching", "InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
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

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
registries = "General"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

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

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "67a144433c4ce877ee6d1ada69a124d6b1ecf7be"
registries = "General"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
registries = "General"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
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

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "b814d5005d6a529d740ffe06f8a86396f6501138"
registries = "General"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.9.2"

    [deps.StructUtils.extensions]
    StructUtilsLazilyInitializedFieldsExt = ["LazilyInitializedFields"]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    LazilyInitializedFields = "0e77f7df-68c5-4e49-93ce-4cd80f5598bf"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

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

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
registries = "General"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
registries = "General"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.8.3+0"

[registries.General]
url = "https://github.com/JuliaRegistries/General.git"
uuid = "23338594-aafe-5451-b93e-139f81909106"
"""

# ╔═╡ Cell order:
# ╟─589b67a3-92c5-523e-8160-a6dd640b2a79
# ╟─cba88b0d-ce76-5bdd-afd5-bed6512f3f1e
# ╟─df5332f8-c4dc-5f62-aa4c-be2bb5dadfb1
# ╟─707daa4a-e946-538b-b15d-08ccaada4f3e
# ╟─f9756f28-86f4-5f7d-b686-537a34240106
# ╟─4d9ded5f-68b5-5e3b-961e-9415756f7537
# ╟─e97f2520-fd0d-57c5-867b-58cc879a5629
# ╟─7459f9bc-1021-5dca-ba44-64150f91cfcb
# ╟─6e22c2bf-f19e-59f7-8ee1-eb79b9222ffd
# ╟─51a50d1a-f72c-571a-8601-14a14d7e0a17
# ╟─bad79ded-b0c1-5d66-94a1-99375f0fb585
# ╟─c09633bf-90df-592e-9dd0-66218754ebe0
# ╟─d7242770-0c2f-5dd7-89b2-229e3e4149a3
# ╟─d35ae31a-df12-52be-9e35-95cf1ced49f4
# ╟─eb526ce8-32a9-4cf5-a1c4-966ffce1617f
# ╟─081d0c9d-1053-50ad-aba1-4c04e7097370
# ╟─1a8eb0ba-d334-541c-97ad-5fc919eaf91c
# ╟─434dd7a5-a6a4-5e05-a157-db86d5168ff9
# ╟─fca0d34d-333c-502e-957f-4891fbe8892b
# ╟─eef9e620-0813-540e-9425-fb091c5f61c1
# ╟─c95211eb-34df-5779-b4ec-8d718d95ee9a
# ╟─612bca3d-1a69-5202-a88b-69bd79c5a3f2
# ╟─4298582e-7132-53cd-b782-4f623baf532b
# ╟─2f4aea61-4ef5-57bf-bcbc-795449cbf4e9
# ╟─355e9c9a-3643-5e4e-92a7-fee9a638ae15
# ╟─8f04976e-5182-56b6-93ed-44a4e4518728
# ╟─1700ba25-c976-5f5a-951d-0ce4970db4c6
# ╟─d9461806-1548-5163-977d-c597a0e6227c
# ╟─847fa071-8823-4007-a5c1-5e5be69cd5e6
# ╟─b8ae1dfc-0c0e-484f-af9e-80141a944239
# ╟─82f1bdfe-32fe-4249-8edd-92558bd63782
# ╟─049d987c-e962-450d-b212-ff6fc028ffc3
# ╟─72e186ee-b9bf-5fcc-a99a-f3fa45211168
# ╟─1c380d45-b7f0-5021-bfd5-c708e95b07f3
# ╟─efd4e479-53dc-5fe8-8e73-7b80e98679ee
# ╟─9604bb6e-f09a-50b5-8d09-9ddbed4a4e78
# ╟─013acf06-3812-56d3-be62-b643b5fbf03d
# ╟─5150232d-151b-5a73-99c9-712af15d577e
# ╟─e12356a5-a1e1-546c-af82-72df1bc98576
# ╟─ad1327ad-b0b5-582c-8fb6-09289c695cbc
# ╟─08dddad4-f342-5079-8fe0-acf6d1f0cbd9
# ╟─151e32de-e8d0-56a0-b035-642c0229267b
# ╟─2af944a3-6c96-5bd1-960e-67eb48926450
# ╟─a7ac41bc-388c-5f4a-bbeb-4aa30a77b6a8
# ╟─717322d6-99a0-54ba-9ae0-a0abe273d37a
# ╟─6896481b-4336-5c6c-b60b-d8ffea83f478
# ╟─fa0ee360-7d3d-55a1-8540-c002da7a2647
# ╟─c2721eec-bcba-5876-8c32-f68498bfd697
# ╟─8273806c-9697-5d2c-a18f-a9d25399f84e
# ╟─69d12213-09d4-5398-963e-7b895a9f83c6
# ╟─c4d90ed1-fc54-515e-bd4d-f684cd6ae159
# ╟─a7bb2f07-c717-596a-845e-2ee6b9492ad1
# ╟─2517909b-0fa2-5c3e-af6b-5554cc9967a2
# ╟─7d7abd6e-6f5d-5822-8fde-1b9e113ea208
# ╟─9916a4af-0b65-5ada-babd-e07ab261f83f
# ╟─3e291457-a06b-536d-a450-893063d3df27
# ╟─d348ba24-657c-53fb-b4c3-0012ca400150
# ╟─c31bb482-89df-5b63-8dc5-7d4cb534ac09
# ╟─508070f9-631e-587f-b7da-689134abb430
# ╟─d4b7046f-bc26-5ec0-be4e-0702c45c5500
# ╟─a3b233b2-495b-5638-9d03-b4750091ea0d
# ╟─751fad30-f343-50eb-b6b9-8c3a7b932284
# ╟─4680ba24-b816-5bc5-9c1a-87142fd24bb0
# ╟─8c73d868-8574-5372-8f97-f7ed0a03369d
# ╟─4fde056f-95bc-561b-a5e0-aa9289576627
# ╟─d9a548b3-0a54-587c-aa8d-8c62e390acb1
# ╟─e2045581-bb21-586f-9003-8a2473921f83
# ╟─44da24f9-33f2-5d57-a96c-f6b114724d0b
# ╟─0ad92dc2-5559-59a9-9a9b-0d84eb40f079
# ╟─1eafee27-1bf2-5b79-8a9b-332f8c0a9ba6
# ╟─018e7f5e-df50-5d95-a88c-b5694adfb4d0
# ╟─769e597a-1fa2-555e-ae84-157035fa3afa
# ╟─ea499cfc-9462-5b30-a99b-0dc5fae42230
# ╟─12c6e558-cf7f-533a-bc85-ee9164181175
# ╟─8bdecce6-d833-5a8f-a0fa-eee6427adfce
# ╟─aea33500-c9c8-5d28-aae8-5872949d5552
# ╟─2b4c119d-d181-5796-b52e-3f585dc70cd2
# ╟─e8fb2d0c-e641-5157-a8c4-148a83648dd4
# ╟─074cb7b7-22db-4303-8b94-d8c4c82ae710
# ╟─4f11fe3e-6eb3-56f2-82f4-56673834d95a
# ╟─0db53489-210c-4fb1-a1cd-ec1e47e97d38
# ╟─c17db78a-2f6e-44f5-b7b5-53c2e647f53e
# ╟─6aa339ec-330f-529d-b125-67ef6c086d7b
# ╟─d36163ad-52fe-5e2b-8191-ae91ded8c137
# ╟─ac2dfb9b-ed01-5d18-8a56-49eb4c69825c
# ╟─c1887141-0ff8-5d23-a23a-e4b03efd4c64
# ╟─30b28689-279e-56c1-9d72-43b9c1bcbadd
# ╟─b1bb018a-fbd8-5d98-a19e-4598c7201b89
# ╟─8646e367-46dd-503b-98e8-88c7b0e0e7b8
# ╟─bfcd3871-3340-53f9-ab5f-bc02a64e4356
# ╟─458816be-fc49-545f-a426-d0a116bab369
# ╟─b9ff50f4-4fa2-556a-b3cd-f2a504b68d10
# ╟─997f9358-5c4c-523c-a346-21694667b207
# ╟─c63e6f69-6c9f-5b05-b607-47d85be10609
# ╟─9881f340-a35a-5984-a10b-5f9b49cc0348
# ╟─75ca6b65-c931-5805-abf8-16817a0d336b
# ╟─5d73f290-a06d-5ce0-b3d9-caafd26e3e12
# ╟─394f5ac6-eaf2-5c37-83d6-2984b2bf1f22
# ╟─5b3dff57-e4e6-59a0-a192-09408988719d
# ╟─cc96137c-6b21-562e-8db3-4a5580fc878b
# ╟─9d147992-9b72-507b-8673-865ac8e16f9d
# ╟─8bb43df8-53f4-5098-9d96-29a7ba856aff
# ╟─83d360b6-fc86-51f6-8a5d-42e0e7682311
# ╟─430216b4-a145-5ed3-b688-de7d9a783a45
# ╟─74dfa6ff-24d1-563d-b024-944667c2f6e7
# ╟─cbf18a63-f866-5285-94d9-4a5ccabb5ca9
# ╟─e9415266-6f6a-53b8-ae3b-49f9d46b28ac
# ╟─851491e7-b3a3-5bab-ad0a-65ee4c9f955d
# ╟─832b90a7-b6fc-590f-9416-26eb5fe47448
# ╟─3b6d0a8e-842c-5c13-bc36-42369de45a42
# ╟─d35f2b09-3eed-518e-8c9c-664bd66cfbf4
# ╟─54c262ee-e7ba-51fd-b464-0e5767201186
# ╟─98b45c5f-498c-59b2-9010-25dbe9a98850
# ╟─1b1650f9-1660-5065-bdeb-be226dff3ee1
# ╟─2f4adad5-d5d8-5ae1-81ce-0dac63211717
# ╟─1842d8d5-d704-5c06-bcba-04a2c496e29c
# ╟─fd58d426-e175-5762-891d-baa3b756a663
# ╟─630be91b-51e5-5227-a7c4-f0193c12e85e
# ╟─e9e430c3-6e96-56bd-8a22-72b14fbfbea2
# ╟─bbe7079a-9112-5066-8ab5-8a866a4d0b17
# ╟─490c2de4-1aaf-5ff1-ad2b-d082bb00b1e3
# ╟─0d16d1f6-3365-5060-bc35-b2c5234b4664
# ╟─0705e823-2568-5ae0-b400-21b1a42132b8
# ╟─0baab108-be3a-5488-b65c-212518240d4e
# ╟─f94f7cf2-271b-4990-b62f-e86336c93a70
# ╟─4a640ff4-c9e3-5073-ad51-c801b4394b54
# ╟─07709070-ab1a-54f0-b181-690bc289173b
# ╟─a6df4399-8a0d-42c4-adba-f1c5b206c05d
# ╟─c7c49e86-99d6-53ae-83f5-77829abcfc2b
# ╟─04006285-331e-4baf-926b-aad05d2eefc8
# ╟─52f2fff4-2b8d-5302-8529-2a7f327c365b
# ╟─c8012528-97cf-53e3-8b48-a33de90c55a8
# ╟─8146bd55-84ec-58f5-9792-ec5e95c093ba
# ╟─1ddc53a0-aaaf-5793-bdc2-363f4cfb03cd
# ╟─e634a810-f0bf-5e00-a472-113b0e3d3ecb
# ╟─8eeeaaa8-93de-5388-b7d8-e51183ec8d9c
# ╟─ae2f29c6-0af2-50fa-9fcb-c5e87a06c254
# ╟─317c73b1-4016-571d-8e9d-2546ff355d9d
# ╟─53f2a6d6-ea97-5a96-bc55-9bfa732b7636
# ╟─2f1758ea-8a11-4ae8-97a5-a3f8c7aaf72f
# ╟─0b79e12f-b5d8-4692-bc1a-8f9c97ead410
# ╟─7d2ee499-8854-4915-8268-b027d7ffee1d
# ╟─93f87c1e-3120-4c90-9dd3-c4ae23a8518f
# ╟─f1d93e91-bfc2-5724-bf44-cbff2dba7e5f
# ╟─796fa7e9-723a-5773-ba36-0f2ae0d25cbe
# ╟─db618c05-95e8-5dbb-babb-8de2780772bb
# ╟─e0503400-df36-58d1-9937-2ed7489d4df3
# ╟─9c388a7d-9c4f-5896-bd90-70ce6398c55c
# ╟─588e4463-82a0-5b71-aeec-ed90848ede23
# ╟─771e7984-cccf-50cc-adbd-8689714b3ac9
# ╟─7a4fb405-604d-532d-b02d-0c2b56850c11
# ╟─b91e6ef3-c0b8-5405-b2f2-1d673d919819
# ╟─d7c1b120-6198-5168-843d-e16057a82d6a
# ╟─5db34b3e-5299-5ed3-8db1-d1b31332c573
# ╟─b14f55d6-7b6c-5628-b395-abb92aec9a9d
# ╟─1c590021-d415-5426-94ca-3af67bd1c293
# ╟─a84ed3ba-8704-5438-9cbc-8e2fc3182a51
# ╟─4b2ab322-b660-59d8-b738-77351d2036ac
# ╟─583c818f-35c5-51b2-ab3e-844c5aad4f90
# ╟─5f072d1c-5554-5a3f-8c4a-557f943cf73e
# ╟─77b8c106-a6a1-532a-a1d4-b9c16d2b86be
# ╟─ff238df2-0486-5e65-ab87-eb070765046f
# ╟─1bc312e7-e7e1-5ab7-8f48-2b8d722dae6f
# ╟─ea8ce6c1-9f7b-5dbb-ac57-adb72842e1ea
# ╟─e18549a6-4faf-5f58-9b26-15a8b9482dc8
# ╟─b23bd7f1-ec12-53f4-8227-8eae27f570f9
# ╟─75efa6e0-5409-5089-aadc-1c288cfc3df3
# ╟─45b67360-313b-5a90-99dc-1b0e3b78216f
# ╟─9fadbdca-a6a0-55a4-b4be-2ea3451116ac
# ╟─f095f16b-1941-5682-a9f2-84953cc1858d
# ╟─a7e96f90-7d92-5f91-9252-c15adc8adb2e
# ╟─938bceb7-cbde-5e9d-978b-558de3df1904
# ╟─d6ecec15-b609-5af2-89a6-96abe072f1da
# ╟─a3a0d09c-8069-5fa7-994a-a73ec3e631ac
# ╟─dc30c49d-64f8-5a49-a80a-11a21d6bb535
# ╟─659d2feb-a802-506c-8db3-d721d3bd3329
# ╟─e50fee24-1ab1-58f5-a34f-9f68ebb7d4a8
# ╟─83284896-bc1d-560d-8427-31d9d536f897
# ╟─649fa6c2-8da8-5351-86c4-b273e198eda8
# ╟─75242b03-29f9-5dfd-8d36-3a9e05cae436
# ╟─e6ae10b6-c536-5f29-8a08-d97f8a14e6e6
# ╟─029ba212-2bc6-5aa3-a21b-499672fb4baa
# ╟─154aaf9d-6f2b-58f7-8fa0-40ed6b5ff4df
# ╟─a754f17c-1a78-58ac-be97-6beee0d4a615
# ╟─479fca6c-815f-5079-a99a-dea660da9b09
# ╠═b07d2299-bcd2-5f07-90ba-d02a6a054b4a
# ╟─d9a597f1-de5b-5a08-a2fb-f2ccaabab9c9
# ╟─6b13e745-10ff-5ddd-8438-242d12587133
# ╟─163717d0-f375-5456-bcbc-0f74af8185f8
# ╟─b3f096c7-22ad-55a4-9eeb-ef56384fd1b9
# ╟─16fc0065-e8b4-5a0d-8791-748b7d612f19
# ╟─f585d0a7-289a-5c17-977c-31f12b029dbd
# ╟─2b123d71-3924-5781-b9cd-d096dac34b6e
# ╟─42476b37-1ce4-52ca-bc63-b775b9d44892
# ╟─1abe641f-f83d-5de5-867f-0a302b18d184
# ╟─370bc577-21cc-509c-9b41-5fe39739f6d8
# ╟─9220b219-748f-51f0-b3ca-1acf6a9ab0bc
# ╟─7fa726bc-5f9b-5d5e-87f5-ea5a5860ceb3
# ╟─172c387f-8025-5133-a600-4e42c20970b1
# ╟─694accf7-7fe4-597b-8a26-700c588d1318
# ╟─ed28dc41-d9d5-5aac-bb20-9ee103cf7a96
# ╟─94226bca-c7c5-52f6-8ce5-ed60e560a6c4
# ╟─f12502d3-0daf-50cf-b4f2-dfa3f6088620
# ╟─161d8b2d-967b-5f49-a2d2-0e223b7aa286
# ╟─92f42b3b-9842-583f-8e6d-6e207b3bdbb7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
