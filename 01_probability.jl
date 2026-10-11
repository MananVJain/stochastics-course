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

# ╔═╡ 7a7f1404-b2f9-cb94-8a1f-f3ff0a9a9450
begin
	using PlutoUI, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions, SHA, Base64
	import AbstractPlutoDingetjes
end

# ╔═╡ 026886f6-938e-eee9-cc02-d3a2a0f87b0b
md"""
# 1 · Elementary Probability, Random Variables and Distributions
*Stochastic Modeling and Simulation*
"""

# ╔═╡ 1568e639-5a1d-0073-eafa-1e935a26a6ee
md"""
Show the mathematics $(@bind show_math CheckBox(default = false)) ``\quad`` Solutions password $(@bind pw PasswordField(default = ""))
"""

# ╔═╡ 08130b13-b9b0-2a1b-956f-700872faa63d
md"""
This notebook explores the basic concepts: events and their probabilities, conditional probabilities and Bayes' theorem, random variables, their distributions and moments, and the common distributions. 

!!! hook "Guiding problem"
    A service center receives calls at random times, on average **3 calls per hour**. Callers call independently of each other.

    **(a)** What is the probability that no call is received for the next 20 minutes?

    **(b)** The operator has already waited 40 minutes without a call. What is now the probability that no call is received for the next 20 minutes?

    **(c)** Which distribution does the waiting time until the next call follow, and why?

    Your estimate for (a): $(@bind guess_a NumberField(0:0.01:1; default = 0)) ``\qquad`` your estimate for (b): $(@bind guess_b NumberField(0:0.01:1; default = 0))

Many people expect that in (b) a call is "due", so that the probability in (b) should be smaller than in (a), is this correct?

## 1. Elementary probabilities

### 1.1 Basic terms

Example $(@bind s1_ex Select(["rolling two dice", "one hour at a service hotline", "quality control of phones"]; default = "rolling two dice"))
"""

# ╔═╡ 46f07831-286d-701a-b204-5b281db99bd6
md"""
Probability theory gives a mathematical model for experiments whose results cannot be predicted with certainty, but whose possible results are known in advance. 

The **population** is the totality of objects or outcome. An **experiment** is a well-defined procedure that gives an observation, and a **sample** is the collection of observations obtained.

In the model, the set of all possible results of the experiment is the **sample space** ``\Omega``. An **event** is a subset ``A \subseteq \Omega``, and ``A`` is said to occur if the observed result belongs to ``A``. The **probability** ``P`` gives to every event ``A`` a number ``P(A)`` with ``0 \le P(A) \le 1``, and ``P(\Omega) = 1``. 
"""

# ╔═╡ 8bf0688d-c883-af15-104b-524a9d32ce5b
md"""
The relative frequency fluctuates at first and approaches the probability slowly.
"""

# ╔═╡ 9a651b5f-214c-5a92-07bb-4c01d8f96244
md"""
## 2. Events, axioms and combinatorics

### 2.1 Events are sets

Because events are sets, they can be combined with set operations. For events ``A`` and ``B``:

- ``A \cap B`` (*A and B*) occurs if both occur;
- ``A \cup B`` (*A or B*) occurs if at least one of them occurs;
- ``\neg A`` (*not A*) occurs if ``A`` does not occur.

The 36 ordered results ``(i, j)`` of two dice are equally likely and can be drawn as a grid. Choose two events and count the cells.

Event ``A`` $(@bind s2_A Select(["sum is 7", "doubles", "first die even", "sum at least 10", "at least one six", "second die larger", "first die shows 1"]; default = "at least one six")) ``\qquad`` Event ``B`` $(@bind s2_B Select(["sum is 7", "doubles", "first die even", "sum at least 10", "at least one six", "second die larger", "first die shows 1"]; default = "sum is 7"))
"""

# ╔═╡ f72df3d9-38db-37ba-4d42-c02cb9038fc7
md"""
If the cells of ``A`` and those of ``B`` are simply added, the cells in ``A \cap B`` are counted twice, so they must be subtracted once.

Any two events divide the sample space into four disjoint pieces: outcomes in ``A`` only, in both, in ``B`` only, and in neither. 

Operation $(@bind s2_op Select(["A", "B", "A and B  (A ∩ B)", "A or B  (A ∪ B)", "not A  (¬A)", "A but not B  (A ∩ ¬B)", "exactly one of A and B", "neither A nor B  (¬A ∩ ¬B)"]; default = "A or B  (A ∪ B)")) ``\qquad`` ``P(A)`` $(@bind s2_pa Slider(0.1:0.05:0.8; default = 0.5, show_value = true)) ``\qquad`` ``P(B)`` $(@bind s2_pb Slider(0.1:0.05:0.8; default = 0.4, show_value = true)) ``\qquad`` ``P(A \cap B)`` $(@bind s2_pab Slider(0.0:0.05:0.8; default = 0.2, show_value = true))
"""

# ╔═╡ ae5a928d-6312-97a6-47c8-38004d91b5a6
md"""
The addition rule and all other rules follow from three axioms.

!!! definition "Axioms of probability"
    For all events ``A, B \subseteq \Omega``:
    1. ``P(A) \ge 0``;
    2. ``P(\Omega) = 1``;
    3. ``P(A \cup B) = P(A) + P(B)`` if ``A`` and ``B`` are mutually exclusive (``A \cap B = \emptyset``).


```math
P(\neg A) = 1 - P(A), \qquad P(A \cup B) = P(A) + P(B) - P(A \cap B).
```
"""

# ╔═╡ a5579cb0-86f7-3ab6-bfb8-0391685088f8
# ╠═╡ disabled = true
#=╠═╡
proof(show_math, "Derivation of the two rules", md"""
**NOT.** ``A`` and ``\neg A`` are mutually exclusive and together form ``\Omega``. Axioms 3 and 2 give ``P(A) + P(\neg A) = P(\Omega) = 1``.

**OR.** Write ``A \cup B`` as the union of three mutually exclusive sets: ``A`` without ``B``, ``B`` without ``A``, and ``A \cap B``. Likewise, ``A`` is the union of (``A`` without ``B``) and ``A \cap B``, and similarly for ``B``. Axiom 3 then gives
```math
P(A \cup B) = \big(P(A) - P(A \cap B)\big) + \big(P(B) - P(A \cap B)\big) + P(A \cap B) = P(A) + P(B) - P(A \cap B).
```
""")
  ╠═╡ =#

# ╔═╡ 42928f2b-b87a-081f-c4b1-ca697ac06ae4
md"""
### 2.2 Combinatorics

If all elementary events are equally likely, a probability is a ratio of two counts. For large sample spaces, these counts can be computed instead of listing. The basis is the **multiplication principle**: if a first experiment has ``m`` possible results and a second one has ``n``, the two experiments together have ``mn`` results. 

Selecting ``r`` objects from ``n`` distinct objects, there are two cases:

- **Permutations** (order matters, e.g. first, second and third place in a race): ``n`` choices for the first position, ``n - 1`` for the second, and so on, so ``P_{n,r} = n!/(n-r)!``.
- **Combinations** (order does not matter, e.g. a committee): each group of ``r`` objects appears ``r!`` times among the permutations, once for each order, so
  ```math
  C_{n,r} = \binom{n}{r} = \frac{n!}{r!\,(n-r)!}.
  ```

The figure lists all ordered selections and groups those that contain the same objects.

``n`` objects $(@bind s2_n Slider(3:5; default = 4, show_value = true)) ``\qquad`` choose ``r`` $(@bind s2_r Slider(1:3; default = 2, show_value = true))
"""

# ╔═╡ fc198618-db34-ee07-ad5b-aea65703ad06
md"""
## 3. Conditional probabilities

### 3.1 Definition and independence

Information about an experiment changes the probability of an event by excluding impossible outcomes. Knowing that an event has occurred changes the set of possible outcomes and may therefore change the probability of another event. For example, knowing that a randomly drawn card from a standard card deck is red excludes all black cards. This idea is formalised by the **conditional probability**. 

A **conditional probability** ``P(A \mid B)`` is the probability of ``A`` given that ``B`` is known to have occurred. In the figure, a card is drawn from a deck of 52 cards, and a hint ``B`` about the card is given. Cards excluded by the hint are faded.

Hint (condition ``B``) $(@bind s3_B Select(["no hint", "the card is red", "the card is a heart", "the card is a face card (J, Q, K)", "the card is an ace"]; default = "the card is red")) ``\qquad`` event ``A`` $(@bind s3_A Select(["the ace of hearts", "a king", "an ace", "a heart", "a red card"]; default = "the ace of hearts"))
"""

# ╔═╡ f7846c5a-e442-c0e6-3a2c-3af2a2717bf8
md"""
Given ``B``, only the cards in ``B`` are possible. The conditional probability is the fraction of these cards that belong to ``A``. 

!!! definition "Definition (conditional probability)"
    For events ``A`` and ``B`` with ``P(B) > 0``,
    ```math
    P(A \mid B) = \frac{P(A \cap B)}{P(B)} .
    ```

The knowledge that the card is red doubles the probability of the ace of hearts, from ``1/52`` to ``1/26``. It does not change the probability of a king: ``P(\text{king} \mid \text{red}) = \frac{2/52}{1/2} = \frac{1}{13} = P(\text{king})``. In the second case the information is irrelevant, and the events are called **independent**.

!!! definition "Definition (independence)"
    ``A`` and ``B`` are independent if ``P(A \cap B) = P(A)\,P(B)``, equivalently ``P(A \mid B) = P(A)`` if ``P(B) > 0``.

Conditioning on ``B`` discards every outcome outside ``B`` and rescales the probabilities of the outcomes inside ``B`` so that they sum to 1 again. The two conditional probabilities ``P(A \mid B)`` and ``P(B \mid A)`` have the same numerator, ``P(A \cap B)``, but different denominators, and are therefore different in general.

Quantity $(@bind s3_cop Select(["P(A | B)", "P(B | A)", "P(A | not B)", "P(A ∩ B) for comparison"]; default = "P(A | B)")) ``\qquad`` ``P(A)`` $(@bind s3_cpa Slider(0.1:0.05:0.8; default = 0.3, show_value = true)) ``\qquad`` ``P(B)`` $(@bind s3_cpb Slider(0.1:0.05:0.8; default = 0.4, show_value = true)) ``\qquad`` ``P(A \cap B)`` $(@bind s3_cpab Slider(0.0:0.05:0.8; default = 0.2, show_value = true))
"""

# ╔═╡ 5a1368ba-5f15-b685-223d-3e9facb62b66
md"""
### 3.2 The law of total probability

Often, the probability of an event is easiest to compute by distinguishing cases. Let ``B_1, \dots, B_n`` be a **partition** of ``\Omega``: mutually exclusive events that together cover ``\Omega``. Then

```math
P(A) = \sum_{i=1}^{n} P(A \mid B_i)\, P(B_i).
```

**Example.** A student travels to the university by bicycle on 60 % of the days, by tram on 30 % and by bus on 10 %. The probability of arriving late is 0.15 by bicycle (weather), 0.10 by tram and 0.40 by bus (traffic). The probability of arriving late on a random day is ``0.15 \cdot 0.6 + 0.10 \cdot 0.3 + 0.40 \cdot 0.1 = 0.16``. Figure 3.3 represents this calculation as areas: the unit square is divided into columns whose widths are the probabilities ``P(B_i)`` of the means of transport; in each column, the shaded part has the height ``P(\text{late} \mid B_i)``. The shaded area is therefore ``P(\text{late})``.

``P(\text{late} \mid \text{bicycle})`` $(@bind s3_pb Slider(0.0:0.05:1; default = 0.15, show_value = true)) ``\qquad`` ``P(\text{late} \mid \text{tram})`` $(@bind s3_pt Slider(0.0:0.05:1; default = 0.1, show_value = true)) ``\qquad`` ``P(\text{late} \mid \text{bus})`` $(@bind s3_pu Slider(0.0:0.05:1; default = 0.4, show_value = true))
"""

# ╔═╡ b968816c-a225-d26f-f5f2-d37574af771c
md"""
!!! think "Think"
    A test for a disease that affects 1 in 1000 people gives the correct result for 99 % of the ill and for 99 % of the healthy. A report states: *a person with a positive test is ill with probability 99 %*. Judge the statement.
"""

# ╔═╡ 2cc4fbec-c384-b84d-71ac-933fa13cf56d
md"""
### 3.3 Bayes' theorem

The calculation can also be reversed. Suppose the student arrived late. What is the probability that the student travelled by bus? This is ``P(\text{bus} \mid \text{late})``, while the given data are ``P(\text{late} \mid \text{bus})``. The two conditional probabilities answer different questions:

- ``P(\text{late} \mid \text{bus}) = 0.40``: **among the days on which the student takes the bus**, the student is late on 40 %.
- ``P(\text{bus} \mid \text{late})``: **among the days on which the student is late**, the fraction of bus days.

In the figure, the first is the fraction of the bus column that is shaded; the second is the fraction of the total shaded area that is in the bus column. Writing the joint probability in both orders, ``P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)``, and dividing by ``P(B)``, gives Bayes' theorem.

!!! theorem "Bayes' theorem"
    ```math
    P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}, \qquad P(B) = \sum_i P(B \mid A_i)\,P(A_i).
    ```
    ``P(A)`` is the **prior** probability (before the observation ``B``), ``P(B \mid A)`` the **likelihood** of the observation under ``A``, ``P(B)`` the **evidence** (total probability of the observation), and ``P(A \mid B)`` the **posterior** probability (after the observation).

For the example, ``P(\text{bus} \mid \text{late}) = 0.40 \cdot 0.1 / 0.16 = 0.25``. The right panel of Figure 3.3 compares prior and posterior probabilities.
"""

# ╔═╡ bac78fcd-f484-a235-8cf9-7ba8516939d9
md"""
!!! think "Think"
    In the spam example, what happens to ``P(\text{spam} \mid \text{offer})`` if the fraction of spam among all messages drops from 10 % to 1 %, while the two likelihoods stay the same? What does this imply for a filter that was trained in one company and is used in another?
"""

# ╔═╡ 9941d0df-e52f-5d35-4a99-74bcd848fbc1
md"""
### 3.4 An application of Bayes' theorem: the three doors

The following problem is known as the Monty Hall problem. A car is behind one of three doors, goats are behind the other two. You choose a door. The host, who knows where the car is, opens one of the other two doors and shows a goat. You may then keep your door or switch to the other closed door. Play several rounds with both strategies before reading the calculation.
"""

# ╔═╡ a442dcab-e71a-9da8-2a8c-30fa1ba0589e
md"""
Suppose you choose door 1 and the host opens door 3. Let ``C_i`` be the event that the car is behind door ``i``, with prior ``P(C_i) = 1/3``, and let ``H_3`` be the event that the host opens door 3. The likelihoods follow from the rules of the host: if the car is behind door 1, he chooses between doors 2 and 3 at random, ``P(H_3 \mid C_1) = 1/2``; if it is behind door 2, he must open door 3, ``P(H_3 \mid C_2) = 1``; if it is behind door 3, he cannot open it, ``P(H_3 \mid C_3) = 0``. Bayes' theorem gives

```math
P(C_2 \mid H_3) = \frac{1 \cdot \tfrac13}{\tfrac12 \cdot \tfrac13 + 1 \cdot \tfrac13 + 0 \cdot \tfrac13} = \frac{2}{3}, \qquad P(C_1 \mid H_3) = \frac{1}{3}.
```

Switching wins with probability ``2/3``. The intuitive answer "two doors remain, so each has probability ``1/2``" ignores the likelihoods: the host's action is not random, it is constrained by the position of the car, and therefore it carries information.
"""

# ╔═╡ c33bb7f0-a34b-1c4c-ac63-b482f6fc113f
md"""
!!! think "Think"
    Consider two variations. (a) The host does not know where the car is and opens one of the two other doors at random; in this round he happens to show a goat. (b) There are 100 doors, and the host, who knows where the car is, opens 98 doors with goats. What is the probability of winning by switching in each case?
"""

# ╔═╡ e95ac4e1-1538-23f7-f1c1-97a1633599ac
md"""
Let ``A_i`` be the event that person ``i + 1`` has a birthday different from the first ``i`` persons (ignoring 29 February and assuming all days equally likely). If the first ``i`` birthdays are different, ``365 - i`` days remain, so ``P(A_i \mid A_1 \cap \dots \cap A_{i-1}) = (365 - i)/365``. The expansion gives

```math
P(\text{shared birthday}) = 1 - \prod_{i=1}^{n-1} \frac{365 - i}{365},
```

which exceeds ``1/2`` for ``n = 23``.

## 4. Random variables

!!! definition "Definition (random variable)"
    A random variable ``X`` is a function that assigns a real number ``X(\omega)`` to every elementary event ``\omega \in \Omega``. Random variables are written with capital letters, their values with lower-case letters; ``\{X = x\}`` denotes the event that ``X`` takes the value ``x``.

The function ``X`` itself is fixed; the randomness lies in the elementary event to which it is applied. Many random variables can be defined on the same experiment. The figure shows several random variables on the experiment "roll two dice".

Random variable ``X`` $(@bind s4_rv Select(["sum of the two dice", "larger of the two dice", "number of sixes", "indicator of doubles", "first minus second"]; default = "sum of the two dice"))
"""

# ╔═╡ 0cb406b1-397b-0f4c-2356-c93f0a1df1a5
md"""
Kinds of random variables:

- an **indicator** (binary) random variable takes only the values 0 and 1, for example ``I_A = 1`` if event ``A`` occurs;
- a **discrete** random variable takes values in a finite or countable set, for example the number of calls in an hour;
- a **continuous** random variable takes values in an interval, for example the time between two calls.
"""

# ╔═╡ 863281d2-2bc7-6ff5-0d19-adfafa2a7dab
md"""
## 5. Probability distributions

### 5.1 Discrete random variables: PMF and CDF

!!! definition "Probability mass function and cumulative distribution function"
    For a discrete random variable ``X``, the **probability mass function** (PMF) is ``p_X(x) = P(X = x)``. The **cumulative distribution function** (CDF) is
    ```math
    F_X(x) = P(X \le x) = \sum_{x_i \le x} p_X(x_i).
    ```

For a fair die, the PMF is ``1/6`` for each face, and the CDF is a step function that increases by ``1/6`` at each face. The bars below can be dragged to change the probabilities of the faces. The slider ``x`` shows that ``F_X(x)`` is the sum of the dark bars.
"""

# ╔═╡ 0866ac05-80f0-1ca3-a833-38680d1e54b8
md"""
``x`` $(@bind s5_x Slider(0:0.5:7; default = 3.5, show_value = true)) ``\qquad`` ``q`` $(@bind s5_q Slider(0.05:0.05:0.95; default = 0.5, show_value = true))
"""

# ╔═╡ 21e91b1b-8f9a-5261-a343-c6647f23c5d7
md"""
Counts and fractions per bin decrease with the bin width, but the *fraction per unit of time* converges to a curve, the probability density: probability per unit of ``x``, in the same way as mass density is mass per unit of volume.

!!! definition "Probability density function"
    A continuous random variable ``X`` has the density ``f_X`` if, for all ``a \le b``,
    ```math
    P(a \le X \le b) = \int_a^b f_X(x)\,dx, \qquad F_X(x) = \int_{-\infty}^{x} f_X(\tilde x)\,d\tilde x, \qquad f_X(x) = \frac{dF_X(x)}{dx}.
    ```

**Probabilities are areas under the density, not values of the density.** The area between ``a`` and ``b`` equals the increase of the CDF from ``a`` to ``b``.

Distribution $(@bind s5_dist Select(["waiting time between calls (exponential)", "adult height (normal)", "position of a raindrop on a 1 m stick (uniform)", "two groups mixed (bimodal)"]; default = "waiting time between calls (exponential)")) ``\qquad`` ``a`` $(@bind s5_a Slider(0:0.02:1; default = 0.2, show_value = false)) ``\qquad`` ``b`` $(@bind s5_b Slider(0:0.02:1; default = 0.5, show_value = false))
"""

# ╔═╡ dfa4c187-1ca8-d1ee-7d64-36b95ba8b9db
md"""
## 6. Moments of distributions

### 6.1 Location: mean, median and mode

The **expected value** or **mean** is the probability-weighted average of the values,

```math
E[X] = \sum_x x\,p_X(x) \qquad\text{or}\qquad E[X] = \int x\,f_X(x)\,dx .
```

Physically, it is the centre of mass of the distribution. The **median** divides the probability into two halves, and the **mode** is the location of the maximum of the PMF or density. For symmetric unimodal distributions the three coincide; for skewed distributions they differ.

Distribution $(@bind s6_shape Select(["symmetric: adult height", "right-skewed: household income", "right-skewed: waiting time", "bimodal: two groups"]; default = "right-skewed: household income"))
"""

# ╔═╡ f42faca5-ab5b-edb7-7f22-56607ec13c66
md"""
### 6.2 Spread: variance and standard deviation

The expected value of a random variable summarizes its weighted average, but it says nothing about how its values are scattered. Two random variables can have the same mean and still behave very differently. 

Suppose two bus lines, A and B, both have an average waiting time of ``\mu = 10`` minutes. On line A, the waiting time is almost always close to 10 minutes. On line B, it can be as short as 1 minute or as long as 19 minutes. Passengers would experience these two lines very differently, but the average is the same for both. The average alone therefore does not tell us how much the waiting time varies from one trip to the next. To distinguish them, we need a measure of **spread** (also called dispersion).

A natural idea is to measure how far ``X`` is from its mean, using the deviation ``X - \mu``. This however  fails, because positive and negative deviations cancel each other:

```math
E[X - \mu] = E[X] - \mu = 0.
```

The average deviation is therefore zero for every random variable, whatever its spread, and has no information. The remedy is to make all deviations nonnegative before averaging. Squaring works, and it also penalizes large deviations more heavily than small ones.

**Definition (variance).** Let ``X`` be a random variable with finite mean ``\mu = E[X]``. The *variance* of ``X`` is the expected squared deviation from the mean:

```math
\mathrm{Var}(X) = E\big[(X - \mu)^2\big].
```

Because squaring changes the units (minutes become minutes²), we return to the original units by taking the square root.

**Definition (standard deviation).** The *standard deviation* of ``X`` is

```math
\sigma = \sqrt{\mathrm{Var}(X)}.
```

Applied to the bus example, line A has a small variance and line B a higher variance, even though both have ``\mu = 10``. The standard deviation ``\sigma`` can be read as a distance of the waiting time from its mean, expressed in minutes.
The figure draws each squared deviation as a rectangle; the variance is the mean area of these rectangles, and the standard deviation is the side length of the red square of mean area.

Data $(@bind s6_set Select(["bus line A", "bus line B", "my data"]; default = "bus line A")) ``\qquad`` my data: $(@bind s6_my TextField(40; default = "2 3 3 4 5 9"))
"""

# ╔═╡ b6f9b7ca-ec9b-9a8b-0883-ac95b6a25b01
md"""
The mean and the variance are special cases of the **moments** of a distribution. The ``k``-th moment is ``M_k = E[X^k]``; the mean is ``M_1``. Expanding the square in the definition gives the formula that is usually used for calculations:

```math
\mathrm{Var}(X) = E[X^2] - \big(E[X]\big)^2 = M_2 - M_1^2 .
```
"""

# ╔═╡ a9372474-8601-2306-eb74-6b6760461817
md"""
### 6.5 Higher moments and the comparison of distributions

More generally, the expected value of a function ``g`` of a random variable is
```math
E[g(X)] = \sum_x g(x)\,p_X(x) \qquad\text{or}\qquad E[g(X)] = \int g(x)\,f_X(x)\,dx .
```
With ``g(x) = x^k`` this gives the **moment of order** ``k``,``M_k = E[X^k]``: 

``M_0 = 1`` because the total probability is 1, 

``M_1 = E[X]`` is the mean, and ``M_2 = \mathrm{Var}(X) + E[X]^2``. 

For a fair die, ``M_1 = 3.5`` and ``M_2 = \frac16(1 + 4 + \dots + 36) = 15.17``, so ``\mathrm{Var}(X) = 15.17 - 3.5^2 = 2.92``.

Moments about the mean, the **central moments** ``m_k = E\big[(X - \mu)^k\big]``, describe the shape independently of the location: ``m_1 = 0`` and ``m_2 = \mathrm{Var}(X)``. 

| quantity | definition | describes |
|:--|:--|:--|
| mean | ``\mu = E[X]`` | location |
| standard deviation | ``\sigma = \sqrt{E[(X - \mu)^2]}`` | spread |
| skewness | ``E[(X - \mu)^3]/\sigma^3`` | asymmetry: positive for a long right tail |
| kurtosis | ``E[(X - \mu)^4]/\sigma^4`` | weight of the tails |

So, a distribution can be described by a sequence of numbers called its **moments**. Each moment captures a different feature of the distribution:

1. the **mean** describes where the values are centered (location);
2. the **variance** describes the spread about the mean (spread);
3. the **skewness** describes whether the distribution leans to one side (asymmetry);
4. the **kurtosis** describes how much probability lies far from the center (tails).

Two distributions can therefore be compared moment by moment, starting with the mean and moving on to the higher ones. In most practical cases, knowing all the moments determines the distribution completely.

In practice, however, we rarely use them all. We often use mean and variance alone, which is only an approximation. Two distributions with the same mean and the same variance can still behave differently, especially in how likely they produce rare, extreme values.

Pair of distributions $(@bind s6_cmp Select(["same mean, different spread", "same mean and variance, different symmetry", "same mean and variance, different tails", "same mean and variance, one peak or two", "same mean and variance: uniform and normal"]; default = "same mean and variance, different symmetry"))
"""

# ╔═╡ 3dc905eb-90b3-cbc8-ffe4-3074c926ca49
md"""
## 7. Joint and marginal distributions

Models usually have several random variables: the numbers of calls in two consecutive hours, the height and weight of a person, the pixel values of an image. For two discrete random variables, the **joint** PMF ``p_{X,Y}(x, y) = P(X = x, Y = y)`` is a table. Its row and column sums are the **marginal** distributions, ``p_X(x) = \sum_y p_{X,Y}(x, y)``. A row of the table, divided by its sum, is a **conditional** distribution:

```math
p_{X \mid Y}(x \mid y) = \frac{p_{X,Y}(x, y)}{p_Y(y)} .
```

For continuous random variables, densities replace the table and integrals replace the sums.

Pair ``(X, Y)`` $(@bind s7_pair Select(["X = first die, Y = second die", "X = larger, Y = smaller", "X = first die, Y = sum"]; default = "X = larger, Y = smaller")) ``\qquad`` condition on ``Y = `` $(@bind s7_y Slider(1:12; default = 2, show_value = true))
"""

# ╔═╡ f326e49a-dfcb-8f65-697a-81ca2dd93311
md"""
## 8. Common distributions

### 8.0 Uniform Distibution
The simplest continuous distribution is the one in which no value is preferred. A random variable ``X`` has the **uniform distribution** on the interval ``[a, b]`` if all values in the interval are equally likely. Its density is therefore constant on ``[a, b]`` and zero elsewhere:

```math
f_X(x) = \begin{cases} \dfrac{1}{b - a}, & a \le x \le b, \\[4pt] 0, & \text{otherwise.} \end{cases}
```


```math
E[X] = \frac{a + b}{2}, \qquad \mathrm{Var}(X) = \frac{(b - a)^2}{12}.
```

For example, the uniform distribution on ``[0, 1]`` has mean ``1/2`` and variance ``1/12 \approx 0.083``.

This distribution is the starting point of all random number generation: computers produce numbers that behave like independent uniform values on ``[0, 1]``, and every other distribution is obtained from them by a transformation (see the notebook on random numbers).
	
### 8.1 Bernoulli and binomial distribution

A single trial with two results, success with probability ``p`` and failure with probability ``1 - p``, is a **Bernoulli** random variable, with ``E[X] = p`` and ``\mathrm{Var}(X) = p(1-p)``. 

The number of successes in ``n`` independent Bernoulli trials is **binomially** distributed, ``X \sim \mathrm{Bin}(n, p)``. 
A particular sequence with ``k`` successes and ``n - k`` failures has probability ``p^k(1-p)^{n-k}``, and there are ``\binom{n}{k}`` such sequences, so

```math
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \qquad E[X] = np, \qquad \mathrm{Var}(X) = np(1-p).
```

The mean and variance follow from the Bernoulli case, because ``X`` is a sum of ``n`` independent Bernoulli variables, whose means and variances add.

``n`` $(@bind s8_n Slider(1:60; default = 20, show_value = true)) ``\qquad`` ``p`` $(@bind s8_p Slider(0.01:0.01:0.99; default = 0.5, show_value = true))
"""

# ╔═╡ 8ae7349a-e9bc-1721-2d25-f6bbc27db6fa
md"""
**Example 8.1.** A component is defective with probability ``p = 0.02``, independently of other components. A box contains ``n = 50`` components. (a) What is the probability that the box contains no defective component? (b) What is the probability of at most one? (c) What are the mean and standard deviation of the number of defective components?
"""

# ╔═╡ 07207311-dadf-fa61-94ec-2dd8626b33a9
md"""
### 8.2 Poisson distribution

The binomial distribution counts successes in a fixed number ``n`` of trials. Many processes have no such ``n``. A hotline receives calls at random moments during an hour, and nobody can say how many "trials" took place. All we usually know is the **average** number of calls per hour, ``\lambda``. How likely is it that exactly ``k`` calls arrive in a given hour?

We can reduce this problem. Divide the hour into ``n`` short intervals of equal length, short enough that at most one call falls into each interval. Let a call arrive in an interval with probability ``\lambda/n``, independently of the other intervals. The expected number of calls is then ``n \cdot \lambda/n = \lambda``, as required, and the number of calls follows ``\mathrm{Bin}(n, \lambda/n)``.

The choice of ``n`` is arbitrary, and the answer should not depend on it. Use the controls below to change ``n`` and watch what happens: for small ``n`` the model is crude (at most ``n`` calls are possible), but from some point on the distribution no longer changes visibly.

Number of intervals ``n`` $(@bind s8_slots Select([5, 8, 12, 20, 50, 200, 2000]; default = 8)) ``\qquad`` mean number of calls per hour ``\lambda`` $(@bind s8_lam Slider(0.5:0.5:6; default = 3, show_value = true))
"""

# ╔═╡ f8289551-d765-b681-52c3-11f487af1f0a
md"""
The limit ``n \to \infty`` is the **Poisson distribution**:

```math
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \dots, \qquad E[X] = \mathrm{Var}(X) = \lambda .
```

It describes the number of events in a fixed interval of time or space when the events occur independently at a constant rate: calls, radioactive decays, typing errors on a page, mutations in a genome. 
"""

# ╔═╡ db787f2d-3129-c9f5-8412-513bb1fd558a
md"""
**Example 8.2.** The hotline receives on average ``\lambda = 3`` calls per hour. (a) What is the probability of no call in an hour? (b) Of more than 5 calls? (c) The hotline can handle at most 6 calls per hour. In what fraction of hours is it overloaded?
"""

# ╔═╡ 636bcacc-2d74-83b7-2fcb-61d831dc48a2
md"""
**Example 8.3.** The lifetime of an electronic component is exponentially distributed with mean 5 years. (a) What is the probability that it fails within the first year? (b) What is the median lifetime? (c) A component has worked for 5 years. What is the probability that it works for another 5 years?
"""

# ╔═╡ 6596d7fa-bfb7-d365-316c-fbd07644a72a
md"""
Whatever the distribution of the single effects (provided their variance is finite), the standardised sum approaches the **normal distribution** ``\mathcal N(\mu, \sigma^2)``. This is the **central limit theorem**. The normal density and CDF are

```math
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right), \qquad P(X \le x) = \Phi\left(\frac{x - \mu}{\sigma}\right),
```

where ``\Phi`` is the CDF of the standard normal distribution ``\mathcal N(0, 1)`` (in Julia, `cdf(Normal(), z)`). About 68 % of the probability lies within ``\mu \pm \sigma``, 95 % within ``\mu \pm 2\sigma`` and 99.7 % within ``\mu \pm 3\sigma``.

**Example 8.4.** The filling weight of coffee packages is normally distributed with ``\mu = 502`` g and ``\sigma = 4`` g. (a) What fraction of packages weighs less than the declared 500 g? (b) Below which weight lie the lightest 1 % of the packages? Use ``\Phi(0.5) = 0.6915`` and ``\Phi^{-1}(0.01) = -2.326``.
"""

# ╔═╡ a97689aa-309c-a508-c44e-4df55f2e793a
md"""
### 8.6 Choosing a distribution

| mechanism | distribution |
|:--|:--|
| one trial with two results | Bernoulli(``p``) |
| number of successes in ``n`` independent trials | binomial(``n, p``) |
| number of independent events at a constant rate | Poisson(``\lambda``); check that variance ≈ mean |
| waiting time between such events (no ageing) | exponential(``\lambda``) |
| sum of many small independent effects | normal(``\mu, \sigma^2``) |
| product of many independent effects, proportional growth | log-normal, power law |
| only lower and upper limits known, all values equally plausible | uniform(``a, b``) |


## 9. Solution of the guiding problem

We return to the initial question. The assumptions are: calls arrive independently of each other and of the past, and the rate ``\lambda = 3`` per hour is constant. We show that the waiting time ``T`` until the next call must be exponentially distributed, by these arguments.

**Argument 1: discretisation.** Divide time into short intervals of length ``\Delta t``. In each interval, a call arrives with probability ``\lambda\,\Delta t``, independently of all other intervals. The waiting time exceeds ``t`` exactly if none of the first ``n = t/\Delta t`` intervals contains a call. Each interval is empty with probability ``1 - \lambda\,\Delta t``, so

```math
P(T > t) = (1 - \lambda\,\Delta t)^{n} = \Big(1 - \frac{\lambda t}{n}\Big)^{n} \;\xrightarrow{\;n \to \infty\;}\; e^{-\lambda t}.
```

Letting ``\Delta t \to 0`` is the same as letting ``n \to \infty``.
For finite ``\Delta t``, the number of intervals until the first call is **geometrically** distributed. The exponential distribution is its continuous limit. Figure 9.1 shows the convergence.

Interval length ``\Delta t`` (minutes) $(@bind s9_dt Select([10.0, 5.0, 2.0, 1.0, 0.1]; default = 5.0))
"""

# ╔═╡ 02ec734f-1c74-01f5-0d4c-821475fd87b8
md"""
**Argument 2: memorylessness determines the distribution.** Independence of the past means that the time already waited carries no information about future time: ``P(T > s + t \mid T > s) = P(T > t)``. Let ``G(t) = P(T > t)`` be the probability of waiting longer than ``t``. By the definition of conditional probability, the equation above says ``G(s + t)/G(s) = G(t)``, that is,

```math
G(s + t) = G(s)\,G(t) .
```

In words: to wait longer than ``s + t``, we must first wait longer than ``s`` and then, independently, longer than ``t``. Let ``a = G(1)`` be the probability of waiting longer than one hour. Then:

- waiting longer than 2 hours has probability ``G(2) = G(1)\,G(1) = a^2``, and in the same way ``G(3) = a^3``;
- waiting longer than half an hour has probability ``G(1/2)``, and ``G(1/2)^2 = G(1) = a``, so ``G(1/2) = a^{1/2}``.

The same reasoning works for every time ``t``, so ``G(t) = a^{t}``. Since ``0 < a < 1``, we can write ``a = e^{-\lambda}`` with ``\lambda > 0``, and therefore

```math
G(t) = e^{-\lambda t} .
```

Hence **memorylessness alone forces the exponential distribution**. The parameter ``\lambda = -\ln G(1)`` is the rate, and ``E[T] = 1/\lambda``.

The answers to the guiding problem are then: With ``\lambda = 3`` per hour and ``t = 1/3`` hour:

- **(a)** ``P(T > 20\,\text{min}) = e^{-3 \cdot 1/3} = e^{-1} = 0.368``.
- **(b)** By memorylessness, the 40 minutes already waited are irrelevant: ``P(T > 60 \mid T > 40) = P(T > 20) = 0.368``. A call is not "due".
- **(c)** ``T`` is exponentially distributed with rate ``\lambda = 3`` per hour, mean 20 minutes. The number of calls per hour is Poisson distributed with mean 3.

The same reasoning is used in stochastic reaction networks: if each of several possible events occurs at a constant rate, independently of the past, the waiting time until the next event is exponential, with the sum of the rates as its parameter. This is the basis of the stochastic simulation algorithm.
"""

# ╔═╡ 04d2b444-081a-4087-b6dd-4b923b70213a
md"""
!!! think "Think"
    The derivation relies on two assumptions: independence of the past and a constant rate. For each of the following situations, state which assumption fails and whether the waiting time is still exponential: (a) the hotline receives more calls in the morning than in the evening; (b) after an advertisement on television, many people call within a few minutes; (c) the time until a car tyre needs replacement.
"""

# ╔═╡ 5526c079-4650-74ef-8bd1-2db943843d48
md"""
## 11. Exercises

### 11.1 Calculation problems

**Problem 1.** For two events, ``P(A) = 0.5``, ``P(B) = 0.4`` and ``P(A \cup B) = 0.7``. (a) Compute ``P(A \cap B)``. (b) Are ``A`` and ``B`` independent? (c) Compute ``P(A \mid B)`` and ``P(B \mid A)``.
"""

# ╔═╡ 04c2e89a-0b93-f00b-cae9-b1319dc64958
md"""
**Problem 2.** A committee of 3 persons is chosen at random from 5 women and 4 men. (a) How many different committees are possible? (b) What is the probability that the committee consists of 2 women and 1 man? (c) What is the probability that it contains at least one man?
"""

# ╔═╡ 486619e5-4bc3-8708-19a0-cacaf3d026f6
md"""
**Problem 3.** A factory has three production lines that produce 50 %, 30 % and 20 % of the output; their defect rates are 1 %, 2 % and 5 %. (a) Compute the probability that a randomly chosen product is defective. (b) A product is defective. Compute the probability that it comes from each line. (c) Explain in one sentence why the answer to (b) differs from the defect rates.
"""

# ╔═╡ a7796982-5923-905a-140e-9e79aa3d65cb
md"""
**Problem 4.** A discrete random variable takes the values 0, 1, 2 with probabilities 0.5, 0.3, 0.2. (a) Compute ``E[X]``, ``E[X^2]`` and ``\mathrm{Var}(X)``. (b) Compute mean and variance of ``Y = 3X - 2``. (c) Write down the CDF of ``X``.
"""

# ╔═╡ ee550866-c25c-abfa-268f-4eff9ef34e87
md"""
**Problem 5.** A continuous random variable has the density ``f(x) = c\,x^2`` on ``[0, 2]`` and 0 elsewhere. (a) Determine ``c``. (b) Compute ``P(X \le 1)``. (c) Compute ``E[X]`` and ``\mathrm{Var}(X)``. (d) Determine the median.
"""

# ╔═╡ a6f81dee-9307-de84-1014-084ff001980a
md"""
### 11.2 Programming exercises

**Exercise 1 · Expectation and variance.** Complete `mean_var(values, probs)`, which returns the mean and the variance of a discrete distribution with ``\mathrm{Var}(X) = E[X^2] - E[X]^2``.
"""

# ╔═╡ 4903d880-e38e-5532-b10c-e09010ea5c04
function mean_var(values, probs)
	m = sum(values .* probs)
	m2 = missing   # your code here: E[X^2]
	return (m, m2 - m^2)
end;

# ╔═╡ 1a33514e-bd6d-bc9d-894a-439623a56699
md"""
**Exercise 2 · The binomial PMF.** Complete `binom_pmf(k, n, p)` using the formula of Section 8.1. The function `binomial(n, k)` computes ``\binom{n}{k}``.
"""

# ╔═╡ f62efebd-11a0-0262-22b0-ea39bc2c48d9
function binom_pmf(k, n, p)
	return missing   # your code here
end;

# ╔═╡ 6171131b-4213-62e2-1d02-82fc4b22087a
md"""
**Exercise 3 · Debugging: counting outcomes.** `prob_two_dice(event)` should return the probability of an event for two fair dice by counting the ordered outcomes ``(i, j)`` for which `event(i, j)` is true. Predict first what the function returns for the sum 7, then correct it.
"""

# ╔═╡ 94a8c162-9409-ee58-2f1d-dc4aac34b133
function prob_two_dice(event)
	hits = 0
	for i in 1:6, j in i:6
		event(i, j) && (hits += 1)
	end
	return hits / 36
end;

# ╔═╡ c8663cd6-de2e-828f-415a-f558a3f4c86e
md"""
**Exercise 4 · Debugging: the birthday problem.** `birthday(n)` should return the probability that at least two of ``n`` persons share a birthday. Evaluate `birthday(1)` first: what must it return?
"""

# ╔═╡ 71e4ee82-f666-e018-0680-f30b9a5c0dc2
function birthday(n)
	p_all_different = 1.0
	for i in 1:n
		p_all_different *= (365 - i) / 365
	end
	return 1 - p_all_different
end;

# ╔═╡ c7ffbfab-6ffd-8da9-ad67-d85bb262ac46
md"""
**Exercise 6 · Simulation of the three doors.** Complete `monty_round(rng; switch)`, which plays one round and returns `true` if the player wins the car. Only the line for switching is missing: the new choice is the door that is neither the old choice nor the opened door.
"""

# ╔═╡ bc4a2f8f-ed58-f19e-7e31-a7f50b1512c7
function monty_round(rng; switch)
	prize = rand(rng, 1:3)
	pick = rand(rng, 1:3)
	opened = rand(rng, [d for d in 1:3 if d != pick && d != prize])
	if switch
		pick = missing   # your code here
	end
	return pick == prize
end;

# ╔═╡ 1738ff3e-0692-eb6b-1ea4-7f461e99d8e6
md"""
## Further reading

| resource | content |
|:--|:--|
| [Harvard Stat 110 (J. Blitzstein)](https://stat110.hsites.harvard.edu/) | lectures |
| [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) | interactive chapters on probability, distributions and Bayesian inference |
| [Probability Distribution Explorer (J. Bois, Caltech)](https://distribution-explorer.github.io/) |  |

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 1.
- A. Tversky and D. Kahneman, "Extensional versus intuitive reasoning: the conjunction fallacy in probability judgment", *Psychological Review* 90 (1983) 293–315.
- M. E. J. Newman, "Power laws, Pareto distributions and Zipf's law", *Contemporary Physics* 46 (2005) 323–351.
- S. Selvin, "A problem in probability" (letter to the editor), *The American Statistician* 29(1) (1975) 67 (the Monty Hall problem).
"""

# ╔═╡ 25d01d86-4470-d901-7c7f-2f5143870d2f
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 2732e001-4996-f98a-ec98-bae0a1b086a9
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

# ╔═╡ 99564cdb-737c-f1be-0c75-2c0d6b0e1dff
# helpers and data
begin
	s2_events = Dict(
		"sum is 7" => (i, j) -> i + j == 7,
		"doubles" => (i, j) -> i == j,
		"first die even" => (i, j) -> iseven(i),
		"sum at least 10" => (i, j) -> i + j >= 10,
		"at least one six" => (i, j) -> i == 6 || j == 6,
		"second die larger" => (i, j) -> j > i,
		"first die shows 1" => (i, j) -> i == 1)

	function dicegrid(inA, inB; w = 330, title = "", labA = "A", labB = "B")
		f = fig(w = w, h = w + 26, xr = (0.5, 6.5), yr = (6.5, 0.2), axes = false, l = 44, t = 40, b = 8, r = 8)
		for i in 1:6, j in 1:6
			a = inA(i, j); bb = inB(i, j)
			rect!(f, j - 0.46, i - 0.46, j + 0.46, i + 0.46; fill = a ? COL.blue : "#f2f2ef", op = a ? 0.55 : 1, stroke = "#d0d0cb", sw = 0.8, rx = 3)
			bb && rect!(f, j - 0.44, i - 0.44, j + 0.44, i + 0.44; fill = "none", op = 1, stroke = COL.red, sw = 2.6, rx = 3)
			text!(f, j, i, "$(i),$(j)"; anchor = "middle", size = 11, c = a ? "#0d2a52" : "#555")
		end
		for k in 1:6
			text!(f, k, 0.25, string(k); anchor = "middle", size = 12, c = "#333")
			text!(f, 0.3, k, string(k); anchor = "middle", size = 12, c = "#333")
		end
		text!(f, 3.5, -0.15, "second die"; anchor = "middle", size = 11.5, c = "#555")
		text!(f, -0.15, 3.5, "first die"; anchor = "middle", size = 11.5, c = "#555", rot = -90)
		f
	end

	s3_deck = [(rank = r, suit = s) for s in ["♠", "♥", "♦", "♣"] for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]]

	s3_pred = Dict(
		"no hint" => c -> true,
		"the card is red" => c -> c.suit in ("♥", "♦"),
		"the card is a heart" => c -> c.suit == "♥",
		"the card is a face card (J, Q, K)" => c -> c.rank in ("J", "Q", "K"),
		"the card is an ace" => c -> c.rank == "A",
		"the ace of hearts" => c -> c.rank == "A" && c.suit == "♥",
		"a king" => c -> c.rank == "K",
		"an ace" => c -> c.rank == "A",
		"a heart" => c -> c.suit == "♥",
		"a red card" => c -> c.suit in ("♥", "♦"))
end;

# ╔═╡ 95604cc8-0e65-2aef-fa54-84f6f2a613ee
begin
	const MODES = ["bicycle", "tram", "bus"]

	const MODE_P = [0.6, 0.3, 0.1]

	function figure_3_3(s3_pb, s3_pt, s3_pu)
		pl = [s3_pb, s3_pt, s3_pu]; pm = MODE_P
		joint = pl .* pm; PL = sum(joint)
		cols = [COL.blue, COL.green, COL.purple]
		f = fig(w = 330, h = 290, xr = (0, 1), yr = (0, 1), xlabel = "means of transport (width = P(mode))", ylabel = "P(late | mode)",
			xticks = cumsum(pm) .- pm ./ 2, xticklabels = MODES, title = "shaded area = P(late) = $(fmt(PL; d = 3))")
		x0 = 0.0
		for k in 1:3
			rect!(f, x0, 0, x0 + pm[k], 1; fill = cols[k], op = 0.08, stroke = "white", sw = 2)
			rect!(f, x0, 0, x0 + pm[k], pl[k]; fill = cols[k], op = 0.75, stroke = "white", sw = 2)
			text!(f, x0 + pm[k] / 2, pl[k] > 0.88 ? pl[k] - 0.06 : pl[k] + 0.05, fmt(pl[k]; d = 2); anchor = "middle", size = 11, c = pl[k] > 0.88 ? "white" : cols[k], bold = true)
			x0 += pm[k]
		end
		if PL > 0
			post = joint ./ PL
			g = fig(w = 330, h = 290, xr = (0.4, 3.6), yr = (0, 1.08), xticks = 1:3, xticklabels = MODES, ylabel = "probability", title = "before and after observing \"late\"")
			bars!(g, (1:3) .- 0.17, pm; width = 0.32, c = "#bdbdb7", op = 1, stroke = "none", labels = [fmt(v; d = 2) for v in pm], lsize = 10.5)
			bars!(g, (1:3) .+ 0.17, post; width = 0.32, c = cols, op = 0.85, stroke = "none", labels = [fmt(v; d = 2) for v in post], lsize = 10.5)
			legend!(g, [("prior P(mode)", "#bdbdb7", :bar), ("posterior P(mode | late)", COL.ink, :bar)]; at = :tr, size = 10.5)
			rows = [[MODES[k], fmt(pm[k]; d = 2), fmt(pl[k]; d = 2), fmt(joint[k]; d = 3), fmt(post[k]; d = 3)] for k in 1:3]
			push!(rows, ["sum", "1", "", fmt(PL; d = 3) * " = P(late)", "1"])
			t = tbl(["mode", "P(mode)", "P(late | mode)", "P(mode and late)", "P(mode | late)"], rows)
			vstack(figure(f, g), t; caption = md"**Figure 3.3.** Left: the law of total probability as areas. The column widths are the probabilities of the three means of transport, the shaded heights the conditional probabilities of being late; the total shaded area is ``P(\text{late})``. Right: Bayes' theorem. Each posterior probability is the shaded area of one column divided by the total shaded area.")
		else
			figure(f, md"With all three probabilities equal to 0, the student is never late, and ``P(\text{mode} \mid \text{late})`` is undefined.")
		end
	end
end;

# ╔═╡ 8bd196fa-1329-e24c-9625-088a0796155b
figure_3_3(s3_pb, s3_pt, s3_pu)

# ╔═╡ 5721bf8d-183a-2e54-3234-1272b4c392dd
begin
	function figure_8_10(s8_zd, sd01_8)
		rng = MersenneTwister(80 + 1009sd01_8); N = 400_000
		if s8_zd == "power law"
			α = 1.5; x = (1 .- rand(rng, N)) .^ (-1 / α); fx = t -> α * t^(-α - 1); lab = "Pareto density f(x) = 1.5 x^(−2.5), x ≥ 1"
		elseif s8_zd == "exponential"
			x = randexp(rng, N) .* 3; fx = t -> exp(-t / 3) / 3; lab = "exponential density, mean 3"
		else
			x = exp.(2 .* randn(rng, N)); fx = t -> exp(-log(t)^2 / 8) / (t * sqrt(8π)); lab = "log-normal density, μ = 0, σ = 2"
		end
		panels = Any[]
		for a in (1.0, 10.0, 100.0)
			edges = collect(range(a, 10a, length = 19)); w = edges[2] - edges[1]
			cnt = [count(v -> edges[k] <= v < edges[k+1], x) for k in 1:18]
			uu = range(1, 10, length = 120); dens = fx.(a .* uu)
			g = fig(w = 216, h = 210, xr = (1, 10), yr = (0, 1.12), xlabel = "x / $(round(Int, a))", ylabel = a == 1 ? "rescaled density" : "", title = "window $(round(Int, a))–$(round(Int, 10a))", l = a == 1 ? 50 : 30, xticks = [1, 4, 7, 10])
			if sum(cnt) >= 30
				h = cnt ./ maximum(cnt)
				hist!(g, edges ./ a, h; c = COL.blue, op = 0.45)
			end
			line!(g, uu, dens ./ maximum(dens); c = COL.red, w = 2)
			push!(panels, g)
		end
		cnts = [count(v -> a <= v < 10a, x) for a in (1, 10, 100)]
		txt = Markdown.parse("Number of the $(N) simulated values in the three windows: $(cnts[1]), $(cnts[2]) and $(cnts[3]). In each panel, the horizontal axis is rescaled to ``[1, 10]`` and the vertical axis to a maximum of 1. " *
			(s8_zd == "power law" ? "" :
			 s8_zd == "exponential" ? "" :
			 "The shape changes slowly from window to window: over a limited range the log-normal distribution with a large ``\\sigma`` resembles a power law, but it is not exactly scale-free."))
		figure(panels..., txt; caption = Markdown.parse("**Figure 8.7.** Zooming out on one distribution ($(lab)). Each panel shows the window ``[a, 10a]`` for ``a`` = 1, 10, 100, rescaled to the same axes: histogram of the simulated values (blue) and the exact density (red)."))
	end
end;

# ╔═╡ 926b8e8a-bec6-9a47-6997-236c0c4de539
begin
	function figure_sets(op, pa, pb, pab, num)
		let
			a, b = pa, pb
			c = clamp(pab, max(0.0, a + b - 1), min(a, b))
			pr = (Aonly = a - c, AB = c, Bonly = b - c, none = 1 - a - b + c)
			ptot(rs) = sum(getfield(pr, r) for r in rs)
			f2(x) = fmt(x; d = 2); f3(x) = fmt(x; d = 3)

			# conditional probabilities: selected pieces, condition, formula parts, circle to outline
			conds = Dict(
				"P(A | B)" => (sel = (:AB,), uni = (:AB, :Bonly), name = "P(A \\mid B)", num = "P(A \\cap B)", den = "P(B)", prior = "P(A)", pv = a, ring = 240),
				"P(B | A)" => (sel = (:AB,), uni = (:Aonly, :AB), name = "P(B \\mid A)", num = "P(A \\cap B)", den = "P(A)", prior = "P(B)", pv = b, ring = 150),
				"P(A | not B)" => (sel = (:Aonly,), uni = (:Aonly, :none), name = "P(A \\mid \\neg B)", num = "P(A \\cap \\neg B)", den = "P(\\neg B)", prior = "P(A)", pv = a, ring = nothing))
			cnd = get(conds, op, nothing)

			# plain events: selected pieces and formula
			ev_sel, formula =
				if op == "A"; ((:Aonly, :AB), "P(A)")
				elseif op == "B"; ((:AB, :Bonly), "P(B)")
				elseif startswith(op, "A or B"); ((:Aonly, :AB, :Bonly), "P(A \\cup B) = P(A) + P(B) - P(A \\cap B) = $(f2(a)) + $(f2(b)) - $(f2(c))")
				elseif startswith(op, "not A"); ((:Bonly, :none), "P(\\neg A) = 1 - P(A) = 1 - $(f2(a))")
				elseif startswith(op, "A but not"); ((:Aonly,), "P(A \\cap \\neg B) = P(A) - P(A \\cap B) = $(f2(a)) - $(f2(c))")
				elseif startswith(op, "exactly"); ((:Aonly, :Bonly), "P(A) + P(B) - 2P(A \\cap B) = $(f2(a)) + $(f2(b)) - 2 \\cdot $(f2(c))")
				elseif startswith(op, "neither"); ((:none,), "P(\\neg A \\cap \\neg B) = 1 - P(A \\cup B) = 1 - $(f2(a + b - c))")
				else; ((:AB,), "P(A \\cap B)") end
			sel, uni = cnd === nothing ? (ev_sel, nothing) : (cnd.sel, cnd.uni)

			order = (:Aonly, :AB, :Bonly, :none)
			col = Dict(:Aonly => COL.blue, :AB => COL.purple, :Bonly => COL.red, :none => "#9a9a9a")
			lab = Dict(:Aonly => "A only", :AB => "A ∩ B", :Bonly => "B only", :none => "neither")

			# ── Venn diagram (schematic)
			id = string(rand(UInt32), base = 36)
			circ(cx, color) = "<circle cx='$cx' cy='105' r='72' fill='$color'/>"
			full = "<rect x='0' y='0' width='390' height='210' fill='black'/>"
			masks = Dict(
				:Aonly => full * circ(150, "white") * circ(240, "black"),
				:AB => full * "<g clip-path='url(#cA$id)'>" * circ(240, "white") * "</g>",
				:Bonly => full * circ(240, "white") * circ(150, "black"),
				:none => "<rect x='15' y='15' width='360' height='180' fill='white'/>" * circ(150, "black") * circ(240, "black"))
			svg = IOBuffer()
			print(svg, "<svg viewBox='0 0 390 210' width='390' style='max-width:100%;height:auto;font-family:system-ui,sans-serif'><defs>")
			print(svg, "<clipPath id='cA$id'>", circ(150, "black"), "</clipPath>")
			for r in order
				print(svg, "<mask id='m$(r)$id'>", masks[r], "</mask>")
			end
			print(svg, "<pattern id='h$id' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'><rect width='6' height='6' fill='#eeeeee'/><line x1='0' y1='0' x2='0' y2='6' stroke='#bbbbbb' stroke-width='2'/></pattern></defs>")
			print(svg, "<rect x='15' y='15' width='360' height='180' fill='#ffffff' stroke='#555' stroke-width='1.5'/>")
			for r in order
				excluded = uni !== nothing && !(r in uni)
				fill = excluded ? "url(#h$id)" : r in sel ? col[r] : "#ffffff"
				opac = !excluded && r in sel ? 0.75 : 1.0
				print(svg, "<rect x='15' y='15' width='360' height='180' fill='$fill' fill-opacity='$opac' mask='url(#m$(r)$id)'/>")
			end
			print(svg, "<circle cx='150' cy='105' r='72' fill='none' stroke='$(COL.blue)' stroke-width='2.5'/><circle cx='240' cy='105' r='72' fill='none' stroke='$(COL.red)' stroke-width='2.5'/>")
			cnd !== nothing && cnd.ring !== nothing && print(svg, "<circle cx='$(cnd.ring)' cy='105' r='72' fill='none' stroke='#1b1b1b' stroke-width='4.5'/>")
			print(svg, "<text x='92' y='42' font-size='16' font-weight='bold' fill='$(COL.blue)'>A</text><text x='290' y='42' font-size='16' font-weight='bold' fill='$(COL.red)'>B</text><text x='24' y='34' font-size='14' fill='#333'>Ω</text></svg>")

			# ── strips with widths proportional to probabilities
			function strip(parts, total, title)
				W = 390; x = 10.0; io = IOBuffer()
				print(io, "<svg viewBox='0 0 $W 62' width='$W' style='max-width:100%;height:auto;font-family:system-ui,sans-serif'><text x='10' y='12' font-size='11.5' fill='#333'>$title</text>")
				for r in parts
					w = (W - 20) * getfield(pr, r) / total
					w <= 0 && continue
					print(io, "<rect x='$(round(x, digits = 1))' y='18' width='$(round(w, digits = 1))' height='24' fill='$(col[r])' fill-opacity='$(r in sel ? 0.85 : 0.18)' stroke='white' stroke-width='1.5'/>")
					w > 34 && print(io, "<text x='$(round(x + w / 2, digits = 1))' y='55' font-size='10.5' text-anchor='middle' fill='#333'>$(lab[r]) $(f2(getfield(pr, r) / total))</text>")
					x += w
				end
				print(io, "</svg>"); String(take!(io))
			end
			strips = strip(order, 1.0, "Ω (total probability 1)")
			uni === nothing || (strips *= strip(Tuple(r for r in order if r in uni), ptot(uni), "condition, rescaled to total probability 1"))

			# ── result line and caption
			if cnd === nothing
				txt = Markdown.parse("``$formula = `` **$(f3(ptot(sel)))**")
				caption = "**Figure $num.** Set operations. Left: Venn diagram of ``\\Omega`` with the selected event coloured (schematic, areas not to scale). Right: the four disjoint pieces of ``\\Omega`` as a strip whose widths are the probabilities."
			else
				p_num, p_den = ptot(sel), ptot(uni)
				txt = Markdown.parse("``$(cnd.name) = \\dfrac{$(cnd.num)}{$(cnd.den)} = \\dfrac{$(f3(p_num))}{$(f3(p_den))} = `` **$(f3(p_den > 0 ? p_num / p_den : NaN))**, compared with ``$(cnd.prior) = `` $(f3(cnd.pv)).")
				caption = "**Figure $num.** Conditioning. Left: hatched areas are excluded by the condition. Right: all pieces of ``\\Omega``, and below them the pieces that remain, rescaled to total probability 1."
			end
			note = pab == c ? md"" : Markdown.parse("*``P(A \\cap B) = $pab`` is impossible for these ``P(A)`` and ``P(B)``; ``$(f2(c))`` is used instead.*")

			row = @htl("""<div style="display:flex;gap:14px;align-items:flex-start"><div style="flex:0 1 330px;min-width:0">$(HTML(String(take!(svg))))</div><div style="flex:1 1 0;min-width:0">$(vstack(HTML(strips), txt, note))</div></div>""")
			vstack(row; caption = Markdown.parse(caption))
		end
	end
end;

# ╔═╡ 34dcce91-1771-b5c9-d0da-0ed7c71d2cb2
figure_sets(s2_op, s2_pa, s2_pb, s2_pab, "2.2")

# ╔═╡ 37f62532-951e-0161-8fa1-c8192e6e4c26
figure_sets(s3_cop, s3_cpa, s3_cpb, s3_cpab, "3.2")

# ╔═╡ a86a9f28-4268-6a70-3602-19f92906f0bc
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

# ╔═╡ 340cdc26-d96f-0212-edfc-f5b65ae22efb
begin
	const MOMENT_PAIRS = Dict(
		"same mean, different spread" =>
			(("normal, σ = 1", x -> pdf(Normal(0, 1), x)), ("normal, σ = 2", x -> pdf(Normal(0, 2), x)), (-7.0, 7.0),
			 "The means agree, the standard deviations do not: the second distribution spreads its probability twice as wide."),
		"same mean and variance, different symmetry" =>
			(("normal", x -> pdf(Normal(1, 1), x)), ("exponential", x -> pdf(Exponential(1), x)), (-3.0, 6.0),
			 "Mean and variance agree. The exponential distribution has a long tail to the right and none to the left: its skewness is positive. Its kurtosis is also larger, because of the long tail."),
		"same mean and variance, different tails" =>
			(("normal", x -> pdf(Normal(0, 1), x)), ("Laplace", x -> pdf(Laplace(0, 1 / sqrt(2)), x)), (-5.0, 5.0),
			 "Mean, variance and skewness agree. The Laplace distribution is more peaked and has heavier tails."),
		"same mean and variance, one peak or two" =>
			(("normal", x -> pdf(Normal(0, 1), x)), ("two peaks", x -> 0.5pdf(Normal(-0.9, sqrt(0.19)), x) + 0.5pdf(Normal(0.9, sqrt(0.19)), x)), (-4.0, 4.0),
			 "Mean, variance and skewness agree, but one distribution has two peaks. Its kurtosis is small. The first moments do not reveal the second peak directly."),
		"same mean and variance: uniform and normal" =>
			(("normal", x -> pdf(Normal(0, 1), x)), ("uniform", x -> abs(x) <= sqrt(3) ? 1 / (2sqrt(3)) : 0.0), (-4.0, 4.0),
			 "Mean, variance and skewness agree. The uniform distribution has no tails, so its kurtosis is smaller than that of the normal distribution."))

	function _moments(f, lo, hi)
		x = range(lo - 6, hi + 6; length = 24001); w = f.(x); w = w ./ sum(w)
		μ = sum(w .* x); m(k) = sum(w .* (x .- μ) .^ k)
		σ = sqrt(m(2))
		(mean = μ, sd = σ, skew = m(3) / σ^3, kurt = m(4) / σ^4)
	end

	function figure_6_5(s6_cmp)
		let
			(la, fa), (lb, fb), (lo, hi), story = MOMENT_PAIRS[s6_cmp]
			x = range(lo, hi; length = 600); ya = fa.(x); yb = fb.(x)
			F = fig(w = 380, h = 250, xr = (lo, hi), yr = (0, 1.12 * max(maximum(ya), maximum(yb))), xlabel = "x", ylabel = "density", title = "two distributions")
			area!(F, x, ya; c = COL.blue, op = 0.15); line!(F, x, ya; c = COL.blue, w = 2.2)
			area!(F, x, yb; c = COL.red, op = 0.12); line!(F, x, yb; c = COL.red, w = 2.2, dash = "6 3")
			legend!(F, [(la, COL.blue), (lb, COL.red, :dash)]; at = :tr, size = 11)
			ma = _moments(fa, lo, hi); mb = _moments(fb, lo, hi)
			names = ["mean", "standard deviation", "skewness", "kurtosis"]
			clean(v) = abs(v) < 5e-3 ? 0.0 : v
			va = clean.([ma.mean, ma.sd, ma.skew, ma.kurt]); vb = clean.([mb.mean, mb.sd, mb.skew, mb.kurt])
			same = [abs(va[k] - vb[k]) < 0.02 for k in 1:4]
			rows = [[names[k], fmt(va[k]; d = 2), fmt(vb[k]; d = 2), same[k] ? "equal" : "**differ**"] for k in 1:4]
			first_diff = findfirst(!, same)
			txt = Markdown.parse((first_diff === nothing ? "" : "The first moment in which the two distributions differ is the **$(names[first_diff])**. ") * story)
			vstack(F, tbl(["", la, lb, ""], rows), txt;
				caption = md"**Figure 6.5.** Comparing two distributions. Top: the two densities. Below: their mean, standard deviation, skewness and kurtosis, computed from the densities.")
		end
	end

	function figure_6_4(sd01_4, s6_a, s6_b)
		let
			x = 18 .+ 5 .* randn(MersenneTwister(31 + 1009sd01_4), 3000)
			y = s6_a .* x .+ s6_b
			lo = min(minimum(x), minimum(y)) - 3; hi = max(maximum(x), maximum(y)) + 3
			lo = max(lo, -120); hi = min(hi, 160)
			edges = collect(range(lo, hi, length = 90))
			hx = histdata(x, edges); hy = histdata(y, edges)
			f = fig(w = 680, h = 280, xr = (lo, hi), yr = (0, 1.25max(maximum(hx), maximum(hy))), xlabel = "value", ylabel = "density")
			hist!(f, edges, hx; c = COL.blue, op = 0.55)
			hist!(f, edges, hy; c = COL.red, op = 0.45)
			for (v, c, lab) in ((x, COL.blue, "X: mean $(fmt(mean(x))), σ $(fmt(std(x)))"), (y, COL.red, "Y = aX + b: mean $(fmt(mean(y))), σ $(fmt(std(y)))"))
				vline!(f, mean(v); c = c, dash = nothing, w = 2)
			end
			legend!(f, [("X: daily maximum in °C, mean $(fmt(mean(x))), σ $(fmt(std(x)))", COL.blue, :bar), ("Y = aX + b: mean $(fmt(mean(y))), σ $(fmt(std(y)))", COL.red, :bar)]; at = :tl)
			figure(f, md"Check: ``a\,E[X] + b =`` $(fmt(s6_a * mean(x) + s6_b)) and ``|a|\,\sigma_X =`` $(fmt(abs(s6_a) * std(x))).";
				caption = md"**Figure 6.4.** A sample of daily maximum temperatures ``X`` (°C, blue) and the transformed values ``Y = aX + b`` (red). With ``a = 1.8``, ``b = 32`` the red histogram is the same data in °F: shifted and stretched, with its standard deviation multiplied by 1.8.")
		end
	end

	function figure_7_1(s7_pair, s7_y)
		let
			fx, fy = Dict("X = first die, Y = second die" => ((i, j) -> i, (i, j) -> j), "X = larger, Y = smaller" => ((i, j) -> max(i, j), (i, j) -> min(i, j)),
				"X = first die, Y = sum" => ((i, j) -> i, (i, j) -> i + j))[s7_pair]
			xs = sort(unique([fx(i, j) for i in 1:6 for j in 1:6])); ys = sort(unique([fy(i, j) for i in 1:6 for j in 1:6]))
			J = zeros(length(ys), length(xs))
			for i in 1:6, j in 1:6
				J[findfirst(==(fy(i, j)), ys), findfirst(==(fx(i, j)), xs)] += 1 / 36
			end
			px = vec(sum(J, dims = 1)); py = vec(sum(J, dims = 2))
			y0 = clamp(s7_y, ys[1], ys[end]); iy = findfirst(==(y0), ys)
			nx, ny = length(xs), length(ys)
			f = fig(w = 400, h = 160 + 26ny, xr = (0.5, nx + 3.2), yr = (ny + 0.5, -2.6), axes = false, l = 40, r = 8, t = 6, b = 50)
			pmax = maximum(J)
			for a in 1:ny, b in 1:nx
				v = J[a, b]
				sh = v == 0 ? "#f6f6f3" : "rgb($(round(Int, 235 - 190v / pmax)),$(round(Int, 240 - 140v / pmax)),$(round(Int, 248 - 60v / pmax)))"
				rect!(f, b - 0.47, a - 0.45, b + 0.47, a + 0.45; fill = sh, op = 1, stroke = a == iy ? COL.red : "#ffffff", sw = a == iy ? 2.2 : 1)
				v > 0 && text!(f, b, a, string(round(Int, 36v)); anchor = "middle", size = 11, c = v / pmax > 0.6 ? "white" : "#222")
			end
			for b in 1:nx
				h = px[b] / maximum(px) * 1.9
				rect!(f, b - 0.35, -0.6, b + 0.35, -0.6 - h; fill = COL.blue, op = 0.7)
				text!(f, b, ny + 0.95, string(xs[b]); anchor = "middle", size = 11.5)
			end
			for a in 1:ny
				w = py[a] / maximum(py) * 2.3
				rect!(f, nx + 0.7, a - 0.33, nx + 0.7 + w, a + 0.33; fill = COL.ochre, op = 0.75)
				text!(f, 0.15, a, string(ys[a]); anchor = "middle", size = 11.5)
			end
			text!(f, (nx + 1) / 2, ny + 1.5, "X"; anchor = "middle", size = 12.5, italic = true)
			text!(f, -0.35, ny / 2 + 0.5, "Y"; anchor = "middle", size = 12.5, italic = true)
			text!(f, nx + 0.7, -1.6, "marginals:"; size = 11.5, c = "#555")
			text!(f, nx + 0.7, -0.9, "top p(x), right p(y)"; size = 11.5, c = "#555")
			cond = J[iy, :] ./ py[iy]
			g = fig(w = 280, h = 260, xr = (xs[1] - 0.7, xs[end] + 0.7), yr = (0, 1.05), xlabel = "x", ylabel = "p(x | Y = $(y0))", xticks = xs, title = "conditional of X given Y = $(y0)")
			bars!(g, xs, cond; width = 0.65, c = COL.red, op = 0.75, labels = [c > 0 ? fmt(c; d = 2) : "" for c in cond], lsize = 10.5)
			indep = maximum(abs.(cond .- px)) < 1e-9
			EXc = sum(xs .* cond); EX = sum(xs .* px)
			figure(f, g; caption = md"**Figure 7.1.** Left: the joint distribution as a table of counts out of 36, shaded by probability; the bars along the top and right are the marginals (row and column totals). The red row is the condition ``Y = y``. Right: the conditional distribution of ``X``, which is the red row divided by its total.")
		end
	end

	function figure_7_2(s7_new, s7_rho, s7_sy)
		let
			ρ = s7_rho; σy = s7_sy; rng = MersenneTwister(7 + 11s7_new); n = 2000
			z1 = randn(rng, n); z2 = randn(rng, n)
			x = z1; y = σy .* (ρ .* z1 .+ sqrt(1 - ρ^2) .* z2)
			vs = 1 + σy^2 + 2ρ * σy
			f = fig(w = 330, h = 300, xr = (-3.5, 3.5), yr = (-3.5max(1, σy), 3.5max(1, σy)), xlabel = "X", ylabel = "Y")
			dots!(f, x[1:800], y[1:800]; c = COL.blue, r = 2.2, op = 0.45)
			s = x .+ y
			edges = collect(-7:0.35:7)
			g = fig(w = 350, h = 300, xr = (-7, 7), yr = (0, 1.25 / sqrt(2π * max(vs, 0.05))), xlabel = "X + Y", ylabel = "density")
			hist!(g, edges, histdata(s, edges); c = COL.purple, op = 0.6)
			zz = range(-7, 7, length = 300)
			line!(g, zz, pdf.(Normal(0, sqrt(vs)), zz); c = COL.ink, w = 2)
			line!(g, zz, pdf.(Normal(0, sqrt(1 + σy^2)), zz); c = COL.grey, w = 1.5, dash = "5 3")
			legend!(g, [("Var = 1 + $(fmt(σy^2)) + 2ρσ_Y = $(fmt(vs))", COL.ink), ("if independent: $(fmt(1 + σy^2))", COL.grey, :dash)]; at = :tr, size = 11)
			figure(f, g; caption = md"**Figure 7.2.** Left: 800 pairs ``(X, Y)`` with correlation ``\rho``. Right: the distribution of ``X + Y`` (histogram) against the normal curve with variance ``1 + \sigma_Y^2 + 2\rho\sigma_Y`` (black). Positive correlation widens the sum, negative correlation narrows it: deviations of ``X`` are then partly cancelled by deviations of ``Y``.")
		end
	end

	function figure_8_2(s8_n, s8_p)
		let
			n, p = s8_n, s8_p; d = Binomial(n, p); ks = 0:n; pm = pdf.(d, ks)
			m = n * p; s = sqrt(n * p * (1 - p))
			f = fig(w = 680, h = 270, xr = (-0.8, n + 0.8), yr = (0, 1.2maximum(pm)), xlabel = "number of successes k", ylabel = "P(X = k)")
			bars!(f, ks, pm; width = n > 30 ? 0.85 : 0.7, c = COL.blue, op = 0.8)
			vline!(f, m; c = COL.red, dash = nothing, w = 1.8)
			seg!(f, m - s, 1.1maximum(pm), m + s, 1.1maximum(pm); c = COL.purple, w = 2.5)
			text!(f, m > n / 2 ? m - s - 0.3 : m + s + 0.3, 1.1maximum(pm), "mean ± σ: $(fmt(m)) ± $(fmt(s))"; c = COL.purple, size = 11.5, anchor = m > n / 2 ? "end" : "start", bg = true)
			kmode = ks[argmax(pm)]
			figure(f; caption = md"**Figure 8.1.** The binomial PMF.")
		end
	end

	function figure_8_3(s8_slots, s8_lam)
		let
			n = s8_slots; λ = Float64(s8_lam); ks = 0:14
			b = pdf.(Binomial(n, λ / n), ks); po = pdf.(Poisson(λ), ks)
			f = fig(w = 680, h = 280, xr = (-0.8, 14.8), yr = (0, 1.15max(maximum(b), maximum(po))), xlabel = "number of calls in one hour k", ylabel = "probability")
			bars!(f, ks .- 0.18, b; width = 0.34, c = COL.sky, op = 1)
			bars!(f, ks .+ 0.18, po; width = 0.34, c = COL.blue, op = 1)
			legend!(f, [("binomial, $(n) intervals, p = $(fmt(λ))/$(n)", COL.sky, :bar), ("Poisson, λ = $(fmt(λ))", COL.blue, :bar)]; at = :tr)
tv = 0.5 * sum(abs.(b .- po))
figure(f, md"Distance between the two distributions: ``\tfrac12\sum_k |b_k - p_k|`` = **$(fmt(tv; d = 3))**.";
	caption = md"**Figure 8.2.** The number of calls when an hour is divided into ``n`` intervals with probability ``\lambda/n`` each (light) and the Poisson distribution with mean ``\lambda`` (dark).")
		end
	end

	function figure_8_4(s8_arr, s8_rate)
		let
			rng = MersenneTwister(404 + 9s8_arr); λ = s8_rate / 60; T = 240
			ts = Float64[]; t = 0.0
			while true
				t += randexp(rng) / λ
				t > T && break
				push!(ts, t)
			end
			steps = 0:2:T
			f = fig(w = 680, h = 150, xr = (0, T), yr = (0, 1), xlabel = "time (minutes)", yticks = Float64[], l = 16)
			for (i, a) in enumerate(ts)
				k = findfirst(>=(a), steps) - 1
				seg!(f, a, 0.15, a, 0.85; c = COL.blue, w = 2.2, s = k)
			end
			for (k, tt) in enumerate(steps)
				last = filter(<=(tt), ts)
				l0 = isempty(last) ? 0.0 : last[end]
				rect!(f, l0, 0.42, tt, 0.58; fill = COL.ochre, op = 0.5, o = k - 1)
				seg!(f, tt, 0, tt, 1; c = COL.red, w = 1.4, o = k - 1)
			end
			gaps = diff(vcat(0.0, ts))
			reads = [let tt = steps[k+1], last = filter(<=(tt), ts)
				"Time: <b>$(round(Int, tt)) min</b> · calls so far: <b>$(length(last))</b> · time since the last call: <b>$(round(Int, tt - (isempty(last) ? 0 : last[end]))) min</b>"
			end for k in 0:length(steps)-1]
			mg = 1 / λ
			player(f; n = length(steps) - 1, start = 0, fps = 8, reads = reads, label = "frame",
				caption = md"**Figure 8.3.** Four hours of simulated calls (blue ticks) at the chosen rate. The orange bar is the time since the last call.")
		end
	end

	function figure_8_5(sd01_5, s8_s)
		let
			g = randexp(MersenneTwister(9 + 1009sd01_5), 40000) .* 20
			rest = g[g .> s8_s] .- s8_s
			edges = collect(0:4:100)
			f = fig(w = 680, h = 270, xr = (0, 100), yr = (0, 0.07), xlabel = "remaining waiting time (min)", ylabel = "density")
			hist!(f, edges, histdata(rest, edges); c = COL.ochre, op = 0.6)
			xs = range(0, 100, length = 300)
			line!(f, xs, pdf.(Exponential(20), xs); c = COL.blue, w = 2.2)
			legend!(f, [("remaining time after $(s8_s) min ($(length(rest)) waiting times)", COL.ochre, :bar), ("new waiting time: exponential, mean 20 min", COL.blue)]; at = :tr)
			figure(f, md"Mean remaining waiting time: **$(fmt(mean(rest); d = 3)) min** (mean of a new waiting time: 20 min).";
				caption = md"**Figure 8.4.** The distribution of the *remaining* waiting time.")
		end
	end

	function figure_8_6(sd01_6, s8_k, s8_eff)
		let
			k = s8_k; rng = MersenneTwister(12 + 1009sd01_6)
			draw, μ1, σ1 = Dict("uniform on [0, 1]" => (() -> rand(rng), 0.5, sqrt(1 / 12)), "exponential (skewed)" => (() -> randexp(rng), 1.0, 1.0),
				"coin: 0 or 1" => (() -> Float64(rand(rng, Bool)), 0.5, 0.5), "die: 1 to 6" => (() -> Float64(rand(rng, 1:6)), 3.5, sqrt(35 / 12)))[s8_eff]
			s = [sum(draw() for _ in 1:k) for _ in 1:20000]
			z = (s .- k * μ1) ./ (σ1 * sqrt(k))
			edges = collect(-4:0.2:4); hz = histdata(z, edges)
			f = fig(w = 680, h = 270, xr = (-4, 4), yr = (0, max(0.6, 1.1maximum(hz))), xlabel = "standardised sum (sum − mean) / σ", ylabel = "density")
			hist!(f, edges, hz; c = COL.blue, op = 0.6)
			xs = range(-4, 4, length = 300)
			line!(f, xs, pdf.(Normal(), xs); c = COL.red, w = 2.2)
			dev = maximum(abs.(cdf.(Normal(), sort(z)) .- ((1:length(z)) .- 0.5) ./ length(z)))
			figure(f; caption = md"**Figure 8.5.** The standardised sum of ``k`` independent effects (histogram) against the standard normal density (red).")
		end
	end

	function figure_8_7(s8_tw, s8_sa, s8_sm)
		let
			rng = MersenneTwister(1906 + 5s8_tw); N = 400; Y = 80; frames = 4:4:Y
			add = fill(1000.0, N); mul = fill(1000.0, N)
			HA = Vector{Vector{Float64}}(); HM = Vector{Vector{Float64}}(); SA = Vector{Vector{Float64}}(); SM = Vector{Vector{Float64}}()
			ea = collect(0:250:12000); em = collect(range(1, 6, length = 41))
			for y in 0:Y
				if y in frames
					push!(HA, histdata(add, ea; density = false) ./ N)
					push!(HM, histdata(log10.(max.(mul, 10.0)), em; density = false) ./ N)
				end
				y in frames && (push!(SA, copy(add)); push!(SM, copy(mul)))
				add .= max.(add .+ s8_sa .* randn(rng, N), 10.0)
				mul .= mul .* exp.(s8_sm .* randn(rng, N))
			end
			ymx = 1.1max(maximum(maximum.(HA)), maximum(maximum.(HM)))
			f = fig(w = 335, h = 270, xr = (0, 12000), yr = (0, ymx), xlabel = "inhabitants", ylabel = "fraction of towns", title = "additive growth", xticks = 0:4000:12000, r = 26)
			g = fig(w = 335, h = 270, xr = (1, 6), yr = (0, ymx), xlabel = "inhabitants (log scale)", ylabel = "fraction of towns", title = "multiplicative growth",
				xticks = 1:6, xticklabels = ["10", "100", "10³", "10⁴", "10⁵", "10⁶"])
			for (k, _) in enumerate(frames)
				hist!(f, ea, HA[k]; c = COL.blue, op = 0.75, o = k - 1)
				hist!(g, em, HM[k]; c = COL.red, op = 0.7, o = k - 1)
			end
			reads = ["Year <b>$(frames[k+1])</b> · additive: largest town $(round(Int, maximum(SA[k+1]))), " *
				"largest/median = $(fmt(maximum(SA[k+1]) / median(SA[k+1]))) · multiplicative: largest town $(round(Int, maximum(SM[k+1]))), largest/median = <b>$(fmt(maximum(SM[k+1]) / median(SM[k+1])))</b>" for k in 0:length(frames)-1]
			top10(v) = sum(sort(v; rev = true)[1:40]) / sum(v)
			player(f, g; n = length(frames) - 1, start = 0, fps = 4, reads = reads, label = "frame",
				caption = md"**Figure 8.6.** Four hundred towns starting with 1000 inhabitants. Left: random yearly *additions*. Right: random yearly *percentage* changes; note the logarithmic axis. After 80 years, town sizes span several orders of magnitude.")
		end
	end
end;

# ╔═╡ 8184cf8c-4033-0f80-3230-0a4e96227082
figure_6_5(s6_cmp)

# ╔═╡ bded43a7-ebd9-f3aa-1606-d3343694b59b
figure_7_1(s7_pair, s7_y)

# ╔═╡ f74f5c16-e323-9d55-8b1e-b5f7a7cfb7ad
figure_8_2(s8_n, s8_p)

# ╔═╡ 2ed74b5d-98cd-b360-2181-16950f65e07c
figure_8_3(s8_slots, s8_lam)

# ╔═╡ 643c3c88-8139-195c-2b0b-e817d9d7c3b0
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

# ╔═╡ b7fedc90-a101-262c-d46c-a204480eed7b
mcq(md"Two dice are rolled and the sum is recorded. A student writes the sample space as ``\Omega = \{2, 3, \dots, 12\}`` and concludes that ``P(\text{sum} = 7) = 1/11``. Where is the error?",
	md"The sample space is wrong: the sum 1 is also possible." => md"With two dice the smallest sum is 2.",
	md"The sample space is correct, but its 11 elements are not equally likely, so the counting rule cannot be used." => md"Right. ``\{2, \dots, 12\}`` is a valid sample space, but the sum 7 comes from 6 of the 36 equally likely ordered pairs. The counting rule ``P(A) = |A| / |\Omega|`` requires equally likely elementary events.",
	md"There is no error." => md"``P(\text{sum} = 7) = 6/36 = 1/6``, not ``1/11``.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 88bace24-5b37-7d3d-3bf0-a710d0e27b1d
numq(md"A lottery draws 6 numbers from 1 to 49; the order of drawing does not matter. How many different tickets are possible?", 13983816,
	why = md"``\binom{49}{6} = 13\,983\,816``. A single ticket wins with probability ``7.2 \cdot 10^{-8}``.", head = "Check your understanding")

# ╔═╡ 0e99d99e-00b4-9ebd-4e02-820bbc7466cf
mcq(md"In Figure 3.2, choose ``P(A) = 0.5``, ``P(B) = 0.4`` and ``P(A \cap B) = 0.2``. Which statement is correct?",
	md"``P(A \mid B) = 0.2``, the size of the overlap." => md"``0.2`` is the unconditional probability ``P(A \cap B)``.",
	md"``P(A \mid B) = P(A) = 0.5``: knowing ``B`` does not change the probability of ``A``." => md"Correct. Here ``P(A \cap B) = P(A)P(B)``: the events are independent.",
	md"``P(A \mid B) = P(B \mid A)``." => md"``P(B \mid A) = 0.2/0.5 = 0.4 \ne 0.5``. The numerators are same; the denominators are not.";
	correct = 2, head = "Pitfall")

# ╔═╡ 9c9fc6f6-caaa-dcd1-581f-74f00d772e30
mcq(md"Events ``A`` and ``B`` both have positive probability and are **mutually exclusive**. Are they independent?",
	md"Yes, because they have nothing in common." => md"If ``B`` occurs, ``A`` certainly does not. The occurrence of ``B`` carries complete information about ``A``.",
	md"No." => md"Right. ``P(A \cap B) = 0``, but ``P(A)P(B) > 0``. Mutually exclusive events with positive probabilities are always dependent.",
	md"It depends on the values of ``P(A)`` and ``P(B)``." => md"For positive probabilities the answer does not depend on the values: ``P(A \mid B) = 0 \neq P(A)``.";
	correct = 2, head = "Common pitfall")

# ╔═╡ eea37975-9707-9623-4d93-982dd172a2c3
reveal(md"""The report confuses ``P(+ \mid \text{ill}) = 0.99`` with ``P(\text{ill} \mid +)``. By Bayes' theorem
```math
P(\text{ill} \mid +) = \frac{0.99 \cdot 0.001}{0.99 \cdot 0.001 + 0.01 \cdot 0.999} \approx 0.09 .
```
Among 100 000 people, about 99 ill people and about 999 healthy people test positive; most positive tests come from the much larger healthy group.""")

# ╔═╡ d3572545-ac4d-1bd2-ec48-61cccabf7f2c
mcq(md"A spam filter is described by two numbers: 80 % of all spam messages contain the word *offer*, and 4 % of all regular messages contain it. Of all incoming messages, 10 % are spam. A message contains the word *offer*. What is the probability that it is spam?",
	md"80 %" => md"80 % is ``P(\text{offer} \mid \text{spam})``. The question asks for ``P(\text{spam} \mid \text{offer})``.",
	md"About 69 %" => md"Right. Of 1000 messages, 100 are spam and 80 of them contain *offer*; of the 900 regular messages, 36 contain it. ``P(\text{spam} \mid \text{offer}) = 80/116 \approx 0.69``.",
	md"About 96 %" => md"96 % would be the probability that a regular message does *not* contain the word. It does not answer the question.",
	md"10 %" => md"10 % is the prior probability. The observation of the word is evidence and changes it.";
	correct = 2, head = "Common pitfall")

# ╔═╡ 7720a263-fb68-581b-c5e6-e47949db7eb9
reveal(md"""With prior 0.01: ``P(\text{spam} \mid \text{offer}) = \frac{0.8 \cdot 0.01}{0.8 \cdot 0.01 + 0.04 \cdot 0.99} = \frac{0.008}{0.0476} \approx 0.17``. The posterior depends strongly on the prior. A classifier whose decision threshold was set for one base rate produces many false alarms where the base rate is lower. In machine learning this is known as **prior shift** or label shift; the likelihoods may transfer, the priors usually do not.""")

# ╔═╡ 63562cee-ee88-ed3e-620d-a3c2c9386d1d
reveal(md"""(a) Switching does not help, because the host's choice carries no information. (b) The first choice is correct with probability ``1/100``; switching wins with probability ``99/100``. The information lies in the constraint on the host, not in the number of remaining doors.""")

# ╔═╡ 955a0e99-07ab-85bf-2d11-01b1e3fb7103
mcq(md"Which of the following is a **continuous** random variable?",
	md"The number of cells in a tissue sample." => md"A count takes the values 0, 1, 2, …: it is discrete, however large the values are.",
	md"The duration of a phone call." => md"Right. A duration can take any value in an interval.",
	md"Whether a seed germinates." => md"Two outcomes, coded 0 and 1: an indicator variable.",
	md"The number shown by a die." => md"Six possible values: a discrete random variable.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 7ee8367e-9c26-a992-aaef-49578ecbbdcd
mcq(md"A density takes the value ``f(x) = 2`` at some point ``x``. What follows?",
	md"Nothing is wrong: a density can exceed 1." => md"Right. The uniform distribution on ``[0, 0.5]`` has density 2 everywhere, and its total area is ``2 \cdot 0.5 = 1``. Only areas are probabilities.",
	md"The density is invalid, because probabilities cannot exceed 1." => md"``f(x)`` is not a probability but a probability per unit length.",
	md"``P(X = x) = 2``." => md"For a continuous random variable, ``P(X = x) = 0`` for every single value ``x``.";
	correct = 1, head = "Common pitfall")

# ╔═╡ eda8da94-ec21-1142-4232-8757dc73195b
mcq(md"A report states that the mean annual salary in a company is 70 000 and the median 48 000. What does this indicate?",
	md"Most employees earn about 70 000." => md"Half of the employees earn less than 48 000. The mean is increased by a small number of high salaries.",
	md"The distribution of salaries has a long right tail: a few employees earn much more than the others." => md"Right. A mean far above the median indicates a right-skewed distribution.",
	md"The figures are inconsistent." => md"Mean and median are equal only for symmetric distributions.";
	correct = 2, head = "Check your understanding")

# ╔═╡ d87d14dd-b78c-3750-0687-c18fb12ea648
proof(show_math, "Derivation", md"""
With ``\mu = E[X]`` and linearity of the expectation:
```math
E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2 = E[X^2] - \mu^2 .
```
""")

# ╔═╡ ebdad859-4c9c-0919-a468-593d9bbd3bad
mcq(md"The daily maximum temperature in Dresden in July has mean 25 °C and standard deviation 4 °C. What are the mean and standard deviation in °F (``F = 1.8\,C + 32``)?",
	md"Mean 77 °F, standard deviation 39.2 °F" => md"The shift by 32 does not change the spread: ``\sigma_F = 1.8 \cdot 4``.",
	md"Mean 77 °F, standard deviation 7.2 °F" => md"Right. ``E[F] = 1.8 \cdot 25 + 32 = 77`` and ``\sigma_F = 1.8 \cdot 4 = 7.2``. The variance is multiplied by ``1.8^2 = 3.24``: from 16 to 51.84.",
	md"Mean 77 °F, standard deviation 12.96 °F" => md"12.96 would result from multiplying the standard deviation by ``1.8^2``. The square applies to the variance, not to the standard deviation.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 92dfee96-d49d-1c20-0507-50314e9d547c
mcq(md"Two random variables have the same mean and the same variance. Which statement is correct?",
	md"They have the same distribution." => md"No. Figure 6.5 shows pairs with equal mean and variance but different shapes: an exponential and a normal distribution, or a distribution with one peak and one with two.",
	md"They have the same probability of exceeding a large threshold." => md"No. Tail probabilities depend on the shape of the tails, which the first two moments do not fix; compare the normal and the Laplace distribution.",
	md"Their distributions can still differ, for example in skewness, kurtosis or the number of peaks." => md"Right. The mean and the variance summarise location and spread only.";
	correct = 3, head = "Pitfall")

# ╔═╡ bbcecfa8-ff8f-f470-3194-48e9f6966722
mcq(md"``X`` is uniformly distributed on ``\{-1, 0, 1\}`` and ``Y = X^2``. Which statement is correct?",
	md"``X`` and ``Y`` are independent, because their covariance is zero." => md"The covariance is indeed zero: ``E[XY] = E[X^3] = 0`` and ``E[X] = 0``. But ``Y`` is a function of ``X``: knowing ``X`` determines ``Y`` completely.",
	md"``X`` and ``Y`` are uncorrelated but not independent." => md"Right. ``\mathrm{Cov}(X, Y) = E[X^3] - E[X]E[X^2] = 0``, but ``P(Y = 0 \mid X = 0) = 1 \neq P(Y = 0) = 1/3``. Independence implies zero covariance; the converse is false. Covariance measures only *linear* dependence.",
	md"``X`` and ``Y`` are positively correlated." => md"For negative ``X``, ``Y`` is positive as well; the positive and negative contributions to the covariance cancel.";
	correct = 2, head = "Common pitfall")

# ╔═╡ 83778b1e-6f8b-7145-f7f2-48f42467e36e
solution(pw, "0YSRPo0BjoRz7xqdR936ZnhrnduP/67F3hx5wxMiaac/voT0BgvWnB4yAJT64yEu8L/XBhWrEWEQW4HQFPA2MuoGpLG9iQNhfnuhzQJN2l0OW+iSpqvuqyVOZpWgPsYHZ0r5EH21C8P+GSZj/LYF5pDRWeItpHPF73GseKdbUudi/srcAjC//FhyBiXLPHGemzniNqZ6k0AD5sUiFj05Uq58mH3tOnKViptFc9xq1JAz5GgFfklYWs4Z+9RDg16NgQgYWtdE++KE5KXKZL72sCZpr5pMwsth3ZPEIw9sW6qSWNXd6M4Fyme+3nYisaISpS/8bVyptKUttSkVjjsI3hVyKvRn"; title = "Solution to Example 8.1")

# ╔═╡ 55c537f7-a73f-6e6c-6e09-67d95814ad1d
proof(show_math, "Derivation of the Poisson limit", md"""
With ``p = \lambda/n``,
```math
\binom{n}{k}\left(\frac{\lambda}{n}\right)^k\left(1 - \frac{\lambda}{n}\right)^{n-k} = \frac{\lambda^k}{k!}\cdot\underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to 1}\cdot\underbrace{\left(1 - \frac{\lambda}{n}\right)^{n}}_{\to e^{-\lambda}}\cdot\underbrace{\left(1 - \frac{\lambda}{n}\right)^{-k}}_{\to 1}.
```
""")

# ╔═╡ 94c3aadf-28c8-51db-a8ea-9496913dbe8d
solution(pw, "Q8sKfPBWgRaB5b8tSxKuo7QIVx8704z8MgYN7RQrVZR/oCsc74rAYoVEM9f03nSq+vTH8OFMVjr9W2bG2cuWmVCyVZiW5P2ANnqR0SO1wzj5zmfNdMOmm2JK7kSL/kkvdOOcncIo2QL1oYr2PXYt5L+6f9ZKPOB4ib053N62z8y6XzB3EkU9A9SXZOPzgSzc3UnGjiiHpk/ADzmseba95aajCGi0qOOgvQSOP6F5//u6eV/O3lShf4nkO7v1yKwP7e/2ZgQ39uWqmU7qT5ytQmFpY0yA0bARFoUYldi3szi1tZmYdb3xeqVUgi2macABCbcAXzDT4VJgw7E7cMI4+FZRFaEmsmh1RTvM9Yi1p7K+OIiaCQR6wgi185ggQwItIWI02wkqebGRjqbF+3Bf9mACv6n2acKrAUDwSupj8HGP3ZFU9Q0="; title = "Solution to Example 8.2")

# ╔═╡ cd7c651f-63ac-391c-50ac-1353c65aaa3c
solution(pw, "Qx0xcU3WrA0kr3p+YNGjcIyGSUPU8lkHHwdS0uIukgxKvu4xo6FJqoVw6GLs/ZJlBDRwMCilP+VBrMr8/sHVZ7wOMQmGvoKbzEIW//UX4YyX36NO4O6FZ7q/BkwAqj/qnyEkuaAJbtD/kmQL92Iuan+7y+6OJ/Y3p2iDdUm/nDFzRG+68dvz9dWhdWOTj0ozM9WpGyFOom1yJVDLTrFN9nU/8rOoFdgsz3tBlEHjpxcLiJYcyusZf75Cu3oyEsW0JLzZQqbia5lF5NrMUcXkYWnpPnPQhJSJQ9fb6+Qz5sAD85wIVelFmK6HsGN2vM2T5GpJJ0auihKwCYpV6ERwl9R1rh1PoNLuHRRRslzgp0oVeAagnGVWjl++gmim2fLj/u0pXda0SND+tvAe7EskyOOh2hvIjWkq20UBFKakJSPoNLYIR3Co1DQ+OcPBmqEZ62Bvtg5AXJZ7cWqK7I74WLlPAUS/iuwLEU2e+LKBuhy+/DO1U2pFJFc4fs5GDGAk0NaltzSHgtg="; title = "Solution to Example 8.3")

# ╔═╡ 72335459-d175-30f5-6101-758984005c2e
solution(pw, "771arBdjakRe45v8mlrZxWt6OXaF44LxQRllgw+n+BDBlV/zxh5MhQ+SSXn8AsLE8RzMUoBTJqgBSMlnfYaMR0qaBfuf9XgdSfN3vvkshcu+w0Ck03JhtMQ9eT1nhCE8IzY+qfE2H4OAX357s5IyCxd4ikzOQz3zePsi72ivUWOBpEd+/NjMYtF4Kl21EPFRzx8jAKp775rx+althdAGcF2dr665a4JQ8htb04YNFE2RgDjAnsS7J7+9oftTF+L3hnBAMvMrV7zcMCkszjA+RiA8CFloqJiqHtHnXZNHtNaUkG8zcmDsHbY/Ppwdmbt+kql/"; title = "Solution to Example 8.4")

# ╔═╡ 9dfb719a-f728-2979-3298-ccb19b7fde3c
reveal(md"""(a) The rate is not constant. Locally, over short periods, the waiting time is approximately exponential, but over a whole day it is not. (b) The calls are not independent: one event (the advertisement) triggers many calls, which arrive in clusters. Waiting times are then more often very short and very long than in the exponential model. (c) A tyre ages: the probability of failure in the next month increases with the distance driven. The waiting time has memory and is not exponential; models with an increasing failure rate (for example the Weibull distribution) are used instead.""")

# ╔═╡ 7f8cd34d-982e-775d-da30-01956444611d
solution(pw, "Urzf/4jlqq0kS0Pezh604BI3vq1HRhx1BcMBbpntcRfkY4proo35tAQHzK+KqMGRirxsSHJTG1tOqTMOext6t1nY9v3z6tFx4M9LAjtZgLDsFVVh33rgurutDnOc8kR1rhQYU31vqkNp3+xOtCTAYucDdytnM2CpWpHj2a3PvuXYY400T/2JkOFYQkHSdvrr26lC6rIqxYRN/qLS3YkMWSbq91YdvjozqwTk6imaxheQn7fotyvWpEyeioYnHlbn6ILQj0JZ4zQM0bI4CKH6xbND5UCM/4LMk0r8ddkFEi28UB4SgBIFmMISbHkNw6JHfepMYhqX2dmno6m9nITsmQJPED2oZXyF4VIk35RDW2zOX2zLABgyD+gILi+Qlk+a4btGoQILcYxsG7XbdU4gYCxuerSXcQ=="; title = "Solution to Problem 1")

# ╔═╡ b0788314-9702-bdc3-a02f-b19c69dc62e2
solution(pw, "AA6rFrfBHd0SCa1l2MmpMGI7+Qo6rg2Aw3NVvC6P5XFcZ2TOUJt2Y84RLZfeEIMNgEabovLbrZgUPU7Ni5ku7pIDbDPPl1LuKphGYPOPb7E10iWArvEX+fLoViAVvEVW+7TbYxhw6T1if3tjuC8DCOoahpSWGQlfnsetPDECrlHeV0kpWbpmNPFxJnw7c389EMAKwmuv5I+BJfHPjk0uqDdz4A3RDhc8gMSxlXVKIQ4jmEzZNtRXPTyumVTx7MvmT3aqOyqB+I/gI3UNtKJficMVB7ZK6GK9CWRi/MbEl4vsesMJ1TsSSz+wQJPtbAtNtg=="; title = "Solution to Problem 2")

# ╔═╡ 1bfd04a5-4240-d653-ab9b-cd710024917b
solution(pw, "t/EG+FNktHT00E3fFKbGqV5hngtRuGl+B6Vz1ejGdu9Ke1/T7IrDACLv12uddGAGyclgr2PDUI1d6P6XpducPZEXqJvWjbhSvssE5pguCjBfa+9Q465JR0PVfgaRsbWcQpBQLAe1BXguew3Kw/BCTDanmFCfS5OWYYkxNz+YlXoJOmjGDtkt80nVjyQxFNG6+kx2HcnxkuZECaAlrtYcu0jDB7jy79o/B+JbHGOBg0EKxQUlt8Dhk1vxx0bSIoE1R4C3q3eLh8Uiv/pJIED7GQbv8Qfk8ke48YNThYDIOYRZj2MXriyzsE25eyiMJLM20bgcpwhbb9Xp4ly81TvVrjpeOJk+gv3q+ql4FFSu19HLGknQap0d6mQ2o88dVZxMRz0d8KOldtXgiqQFVj/7xd4oVByHreYbNjdnBV+DTc20vGclfbEuimW2FFIsQ0Rjv94lE8epDwlcvjuO6D0LlTkdfCnc9Te6g6HfgL/f0QyozArexjBdcztx5RUGUNqZUyvmyoB6v5AevvTD29JCOSEUkhJvBdgo2+Z0fCuyzgprKEPrvbfDKVWxkPwVzhyygolerIxC6aIOyR3KbwptQdGZuZnXwZ2W3t9+/IdZibTz1w=="; title = "Solution to Problem 3")

# ╔═╡ 495f4e1a-279b-264a-f204-c246cccce481
solution(pw, "FcxoirbfdcbduGuFb253VDWT/ISwGKp46SAz0b0T8sGZ6mtuYi5mc3DUE5txknz4kg8ERK/8w9Tv31PNr0yxt7K9GkihD6yKgNcRb49BM3B+Vb/Lkorz1ASF4ZiVz65nbGZYO2CMCAXBcGAHWOQN/vCrVkgBQqxuwqLnFXdY8GRJ4PUKFj+kLcT89Nc+LBNr/OqPQHNUnyyfzilF/FNE+1E/VqJsalKmqGZoqjysP06l661/x5OdO1Ha610FT+X2XluGusqL8nbPZUMc6euXq8YniolD8lfGcVBU040kyLHK6EuQ+6xplp4R6P/AlJ7z2U48TGRwDntQxTFe0S7wLeuuY5D7iOjZVseIjeDbtgakrniXYK1iOFmQSDRy7F1O4zuhoeE//v61XlB2DDprcQ93UsFsXr0OWqdR4Uw8Pmjt+51NZlbFHANTvyzkMnlg6OtowTp8YkufFQxRcQIz7RII"; title = "Solution to Problem 4")

# ╔═╡ e617cbea-70a6-998e-7c50-39228342a128
solution(pw, "/usWiDjgHg3cLGkrn3h+Hh09JuSBg1Dsw5agEsZmApS+OrfuoMzfrTL6AEKiBQaEFFIGm/m0prdCgWlvjJt7E/LSno2hQ1U85mIQUc+M7hhmpoX4gXb9dVG1xPUqXYjPCKdrHCibRl9thvc5CK43cGIax4dJqroP30eD+0ukX+iWLkfFQjfVk8gpPiPCVctp9ff0EK8RrgxIpEEUkL4h7hGUgGFVOqy8Yi8Tpm7fYbpuQB/kR92amOBCML/bEtGwfChe/NFlH/6f4k2MV8BBOaODuOiZ7SvRg+MFY4O7jbYKXxe3Z+8UielDfuJRGUx6mULI1lxTgtcRrE/fs9kZ4tppfvvXqGM2v38KicsO6gXgl9eQ1DFDMTGIsAeQmUBWwKLq0AFuHEqoTkPzad4ZzdMIvrtP16xuglDI09SvUBoIDTe2/uvE6ReJUF+9D+J79aGiNgESpEW41Xfed5cyk5+NbxaMYoskptyAso16uigIHUfsx7FYQVZLFBz9FeHtqxZvoN6OPAn5lheNsAlV"; title = "Solution to Problem 5")

# ╔═╡ bdb8f72d-117c-e5ca-3a52-6b4f41181f63
solution(pw, "RYckhQhiZZpPq8xAEKBpRLIjcTwo40g2pu3cja77D/ocb1WwhmdfRN2kQXyDr3KAxG0iZm9/KkmAoDeAPYFvcvG3IRV0ZcU8DYhMR2U+tu/gy0gbyixjmjB6KJcQx/UeZUcgkQDzXs7XRxbUQXNAZSi2XHlNmimwJK3rWhjXKOMmXKQ1ml0ypgpLDBrS/Cq7jlL22WCYY6Bva6wUKFbDY++ucSXuWDf9n3Kr/D1mk4TvkXHhFCxpedORoa9bkHsPa6g81NeWL1VTfDXp1l4RAsYmMOC5UB51U8Az5fM2CLG8k1r2tuKTy890L7MVTFnEvo1b6+71ONPUinVfgy8gnn/jZ6U="; title = "Solution to Exercise 1")

# ╔═╡ 85dc7ada-c6f5-47b1-fdf6-70e53d467a5b
solution(pw, "je3etaFRUjfqr7681Y1kxAkCb7c+9QFPyBEG4PA45gDxfGOjJMGkIqzs5YC75qneoB3IjeGEaT4KEwrKg/GKBt2VDFaARBSJIO30DB9V8TmEMUSTHYnFhAFC6w/8PXYbkDPJCKlk1CDEA6ME6cxwgoSIEUUSVghQbFuirLKoxz5eLR2LKClwCsZ7R9UV0dLDrAQCFXLHdGqBvRNCE5MpW+yGtSjIx9iiRN1O+GHnaJox2Zvwp2Uaz8Qm+bpEQpWh9L7sQgYh0JaA3ukI7WvowM/iuQgKMRvaN8bTDT84ASa/GPpGEksmfPshveJbB3dKlGxRjzbD/p+QZh1aAie7dIU="; title = "Solution to Exercise 2")

# ╔═╡ 7147ee06-1085-ae4b-a8da-ad633a402b73
solution(pw, "mxRXfn8n7OgQ4RLlsV4mH/3Dfjs7DeWjY2xhqhiKDY7jGBw28aJPhflvS5Cc5hMsWtlmZZSXeBA+jgrrlqqoUX3RJ9EmtXSYGYN8MWRfquM5YcBfkG6efaSpl5r/oa71e3ZpJ73Tzb6gxcj7ckrkFEZNN5z6ZolyOiPXj3oi8dy0K3udfz/kvCgac7rH/BZjBqNL9TxySAbMRleBVPsp+9rKPojY0rAs4utbW/iE6ImcQU9Kq4Q9gzPM6ttioQlgyFFY8ogQcEcdPriYg42pugZ1eOwUVtT8s+CgnjyIqDU3yuJbUligS/MP5UT8QBhqcoC9lse+/8a/kfrvDQ5BR3L38JAs6hWGH9Ta+0gNemeZL02r+6gPU4RnK4VXvuU4hAT9y1I4OfzubSYmhHc/QQhnDxQ+ke6/DyNj9lSfD781Q4Qk53qCM+kInAEbRI1DblSFUGP7rrS6qzCjhSnhFsF0K/ZEZVS8ker5KUOCHQ=="; title = "Solution to Exercise 3")

# ╔═╡ 84c27ad5-7d39-5fb5-5b1d-1432fef699b8
solution(pw, "1YYIGZ+mRGMjsV0NmvyExlBQdNOMuhI3vRhATLhzopkOcJwEL72sA0sTMJQ4AW3XMFJU4UqgV0zeXsYvUVbniVXCDruvDWXUvv9od8JptSe9z4bJ/KYMJQk5/4CyCNvdkU/JjJ4IYHXBLJjE+MXg8NxU58INEydfmrjtVlFpon7zP2JF+qr+GLFU/Rj27slybHYRNNDxV4IDeV/EJ+j5gzpmyM3MbSrJacDQierO/A/023ocR/tuJi+YwxIBHS9AioL2nIIYEC1td/Jy8pPb5iQbji6ka3r4Ctm8WoQyxdvvUiJBgvNy5UQKRJNc8ZL0yhDEQE31f7BBtP2DIdCWCbOi9aeBHTdEbfX8knEobLBaa3NJ6zCx9eGAHmoBaLp1+xaEyhGSsDH471/eH6KeLgLVXJVpsyJkLfi+Ukn/2iAuZy97cBIw9SsVgYhWThvFsQvdHss6O3uQR2nRhSEIyk2wlneXbNUf3h6WgWlZESs="; title = "Solution to Exercise 4")

# ╔═╡ 9c1000fb-901d-7cd5-c36c-278fa306809d
solution(pw, "U2IwShKEkn5AF4rElmp1a8wif9ia+FHobcNs7myxQSkxVLK0xHZibRaEZc2fGoBiNhFxvK5KoJXALz+4taqHK1CBldWiF1cHXLTdjKUA6qwHgEbFe76kQE1XvFpcb23Ecgmv1VO0xtEAeFncL9lEAnsGGuF/gCCguz+3kCpkKkSYB5dMbR+MjfGqu+/JWqzXCGlaAZlLIRdBIE+pnub89LfNjQ4blnoifJ7e3abyDGCS8vdLig7nb/7uI/BPtjEfXbvXjNAlovAqAcr7q45wWuyI9zJe+tAiBtnMPxaYI5gzyST+A/YvEwqZM5i7gd8WAoXce0GO3Mn5wsT6a/FcKgLQAp2HdxKID6MHNqX5My+fpQkm7j8xbN6Q7h5gvI7x+TG/0+Mk5rkdMcsTn+xgam4AfK50Xbb8/g=="; title = "Solution to Exercise 6")

# ╔═╡ 3ad6f6c2-5d1d-58e1-408c-bc3006ae1b4b
# figures
begin
	function figure_1_2(s1_roll, s1_n, s1_ev)
		let
			rng = MersenneTwister(2024 + 13s1_roll); n = s1_n
			ev, p, plab = Dict("a six" => (x -> x == 6, 1 / 6, "1/6"), "an even number" => (x -> iseven(x), 1 / 2, "1/2"), "at least 5" => (x -> x >= 5, 1 / 3, "1/3"), "not a one" => (x -> x != 1, 5 / 6, "5/6"))[s1_ev]
			r = rand(rng, 1:6, n); hit = ev.(r)
			fr = cumsum(hit) ./ (1:n)
			f = fig(w = 680, h = 150, xr = (0, n), yr = (0.5, 6.5), ylabel = "face", yticks = 1:6, xticks = Float64[], b = 10)
			dots!(f, collect(1:n), r; c = [h ? COL.red : COL.grey for h in hit], r = n > 1000 ? 1.5 : 2.3, op = [h ? 0.95 : 0.45 for h in hit], s = collect(1:n))
			g = fig(w = 680, h = 260, xr = (0, n), yr = (0, 1), xlabel = "number of rolls n", ylabel = "fraction with the event")
			k = 1:n
			band!(g, collect(k), max.(p .- 2 .* sqrt.(p * (1 - p) ./ k), 0), min.(p .+ 2 .* sqrt.(p * (1 - p) ./ k), 1); c = COL.grey, op = 0.15)
			hline!(g, p; c = COL.red, dash = nothing, w = 1.4)
			text!(g, n * 0.99, p + 0.035, plab; anchor = "end", c = COL.red)
			for i in 2:n
				seg!(g, i - 1, fr[i-1], i, fr[i]; c = COL.blue, w = 1.8, s = i)
			end
			dots!(g, collect(1:n), fr; c = COL.blue, r = 4, o = collect(1:n))
			text!(g, n * 0.99, 0.95, ""; anchor = "end", c = "#555", size = 11.5)
			zsc(j) = (fr[j] - p) / sqrt(p * (1 - p) / j)
			reads = ["Rolls: <b>$(j)</b> · rolls with $(s1_ev): <b>$(j == 0 ? 0 : count(hit[1:j]))</b> · fraction: <b>$(j == 0 ? "–" : fmt(fr[j]))</b>" * (j == 0 ? "" : " · deviation from $(plab): $(fmt(fr[j] - p; d = 2)), " * (abs(zsc(j)) <= 2 ? "inside the band" : "<b>outside the band</b>")) for j in 0:n]
			player(f, g; n = n, start = 0, fps = 30, reads = reads, label = "roll",
				caption = md"**Figure 1.2.** Top: the faces rolled; rolls with the event in red. Bottom: the fraction of such rolls after ``n`` rolls.")
		end
	end

	function figure_2_1(s2_A, s2_B)
		let
			fA = s2_events[s2_A]; fB = s2_events[s2_B]
			nA = count(fA(i, j) for i in 1:6, j in 1:6); nB = count(fB(i, j) for i in 1:6, j in 1:6)
			nAB = count(fA(i, j) && fB(i, j) for i in 1:6, j in 1:6); nAoB = count(fA(i, j) || fB(i, j) for i in 1:6, j in 1:6)
			g = dicegrid(fA, fB)
			t = tbl(["event", "cells", "probability"], [
				["A: " * s2_A, nA, "$(nA)/36 = $(fmt(nA / 36))"],
				["B: " * s2_B, nB, "$(nB)/36 = $(fmt(nB / 36))"],
				["A and B", nAB, "$(nAB)/36 = $(fmt(nAB / 36))"],
				["A or B", nAoB, "$(nAoB)/36 = $(fmt(nAoB / 36))"],
				["not A", 36 - nA, "$(36 - nA)/36 = $(fmt((36 - nA) / 36))"]])
			chk = md"``P(A) + P(B) - P(A \cap B) = `` $(nA)/36 + $(nB)/36 − $(nAB)/36 = $(nA + nB - nAB)/36 ``= P(A \cup B)``."
			figure(g, @htl("<div style='padding-top:40px'>$(t)$(chk)</div>"); caption = md"**Figure 2.1.** The 36 results of two dice. Cells of ``A`` are filled blue; cells of ``B`` have a red border. Cells that are both blue and red-bordered are ``A \cap B``.")
		end
	end

	function figure_2_3(s2_n, s2_r)
		let
			n = s2_n; r = min(s2_r, n)
			letters = ["A", "B", "C", "D", "E"][1:n]
			cols = [COL.blue, COL.red, COL.ochre, COL.green, COL.purple]
			combos = Vector{Vector{Int}}()
			function rec(start, cur)
				length(cur) == r && (push!(combos, copy(cur)); return)
				for k in start:n
					push!(cur, k); rec(k + 1, cur); pop!(cur)
				end
			end
			rec(1, Int[])
			perms_of(v) = length(v) <= 1 ? [v] : [[v[i]; p] for i in eachindex(v) for p in perms_of(deleteat!(copy(v), i))]
			nc = length(combos); nr = factorial(r)
			tw = 16; gapc = 14
			W = max(600, nc * (r * tw + gapc) + 40); H = 60 + nr * 22
			f = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
			for (c, cmb) in enumerate(combos)
				x0 = 20 + (c - 1) * (r * tw + gapc)
				rect!(f, x0 - 4, 30, x0 + r * tw + 2, 34 + nr * 22; fill = "#f4f4f1", op = 1, stroke = "#d6d6d0", rx = 3)
				for (k, p) in enumerate(perms_of(cmb))
					for (m, idx) in enumerate(p)
						xx = x0 + (m - 1) * tw; yy = 36 + (k - 1) * 22
						rect!(f, xx, yy, xx + tw - 2, yy + 18; fill = cols[idx], op = 0.85, rx = 2)
						text!(f, xx + (tw - 2) / 2, yy + 9.5, letters[idx]; anchor = "middle", c = "white", size = 11, bold = true)
					end
				end
			end
			text!(f, 20, 14, "$(nc) groups (combinations) × $(nr) orderings each = $(nc * nr) ordered selections (permutations)"; size = 12.5)
			figure(f; caption = md"**Figure 2.3.** Every ordered selection of ``r`` of the ``n`` lettered objects. Each grey box has one *combination*; inside it are all ``r!`` orderings of the same letters. Hence ``P_{n,r} = C_{n,r}\, r!``.")
		end
	end

	function _check_prob_two_dice(prob_two_dice)
		let
			r = tryrun(() -> prob_two_dice((i, j) -> i + j >= 10))
			ok = r !== missing && isapprox(r, 1 / 6) && isapprox(prob_two_dice((i, j) -> i == j), 1 / 6) && isapprox(prob_two_dice((i, j) -> i == 6 || j == 6), 11 / 36)
			check(r, ok, () -> md"``P(\text{sum} \ge 10) = 6/36``, ``P(\text{doubles}) = 6/36``, ``P(\text{at least one six}) = 11/36``.",
				() -> md"Got $(r) for ``P(\text{sum} \ge 10)``; expected ``6/36 \approx 0.1667``. Divide the count by 36.";
				hint = md"Are the outcomes (1, 6) and (6, 1) both visited by the loop?")
		end
	end

	function figure_3_1(s3_B, s3_A)
		let
			inB = s3_pred[s3_B]; inA = s3_pred[s3_A]
			f = fig(w = 680, h = 250, xr = (0, 13), yr = (4, 0), axes = false, l = 8, r = 8, t = 8, b = 8)
			for (k, c) in enumerate(s3_deck)
				row = div(k - 1, 13); col = mod(k - 1, 13)
				red = c.suit in ("♥", "♦")
				alive = inB(c); hit = inA(c)
				rect!(f, col + 0.06, row + 0.06, col + 0.94, row + 0.94; fill = alive ? "#ffffff" : "#efefec", op = 1, stroke = hit ? COL.blue : "#bdbdb7", sw = hit ? 3 : 0.8, rx = 4)
				text!(f, col + 0.5, row + 0.36, c.rank; anchor = "middle", size = 13, bold = true, c = alive ? (red ? COL.red : COL.ink) : "#c8c8c3")
				text!(f, col + 0.5, row + 0.7, c.suit; anchor = "middle", size = 14, c = alive ? (red ? COL.red : COL.ink) : "#c8c8c3")
			end
			nB = count(inB, s3_deck); nA = count(inA, s3_deck); nAB = count(c -> inA(c) && inB(c), s3_deck)
			t = tbl(["quantity", "count", "probability"], [
				["P(A), without the hint", "$(nA) of 52", fmt(nA / 52)],
				["P(B)", "$(nB) of 52", fmt(nB / 52)],
				["P(A and B)", "$(nAB) of 52", fmt(nAB / 52)],
				["P(A | B) = P(A and B) / P(B)", "$(nAB) of $(nB)", fmt(nAB / nB)]])
			verdict = isapprox(nAB / nB, nA / 52) ? md"Here ``P(A \mid B) = P(A)``: the hint carries no information about ``A``. The events are **independent**." :
				(nAB / nB > nA / 52 ? md"The hint makes ``A`` **more** likely." : md"The hint makes ``A`` **less** likely.")
			figure(f, @htl("<div>$(t)$(verdict)</div>"); caption = md"**Figure 3.1.** The deck. Faded cards are excluded by the hint ``B``; outlined cards belong to ``A``. Given the hint, the remaining cards are equally likely, so ``P(A \mid B)`` is the fraction of the remaining cards that are outlined.")
		end
	end

	function figure_3_4(s3_room, s3_D)
		let
			D = s3_D; rng = MersenneTwister(23 + 17s3_room)
			n = D == 365 ? 60 : D == 52 ? 25 : 10
			bd = rand(rng, 1:D, n)
			first_rep = findfirst(k -> bd[k] in bd[1:k-1], 2:n)
			first_rep = first_rep === nothing ? n + 1 : first_rep + 1
			unit = D == 365 ? "day of the year" : D == 52 ? "week of the year" : "month"
			xt = D == 365 ? [1, 60, 121, 182, 244, 305, 365] : D == 52 ? [1, 13, 26, 39, 52] : collect(1:12)
			xtl = D == 365 ? ["1 Jan", "1 Mar", "1 May", "1 Jul", "1 Sep", "1 Nov", "31 Dec"] : D == 52 ? ["1", "13", "26", "39", "52"] : ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
			f = fig(w = 680, h = 120, xr = (0.5, D + 0.5), yr = (0, 1), axes = true, xlabel = unit, yticks = Float64[], l = 14, r = 30, xticks = xt, xticklabels = xtl)
			for k in 1:n
				dup = any(bd[j] == bd[k] for j in 1:k-1)
				jit = D <= 52 ? 0.6 * count(==(bd[k]), bd[1:k-1]) / 4 : 0.0
				seg!(f, bd[k], 0.15 + jit, bd[k], 0.45 + jit; c = dup ? COL.red : COL.blue, w = D == 365 ? (dup ? 3 : 1.8) : 5, s = k)
			end
			g = fig(w = 680, h = 250, xr = (0, n), yr = (0, 1), xlabel = "number of people n", ylabel = "P(at least one shared value)")
			pb(m) = m > D ? 1.0 : 1 - prod((D - i) / D for i in 0:m-1)
			line!(g, 1:n, pb.(1:n); c = COL.ink, w = 2)
			hline!(g, 0.5; c = COL.grey)
			dots!(g, collect(1:n), pb.(1:n); c = COL.red, r = 5, o = collect(1:n))
			half = findfirst(m -> pb(m) >= 0.5, 1:n)
			reads = [k == 0 ? "Nobody in the room yet." : "People: <b>$(k)</b> · P(shared) = <b>$(fmt(pb(k)))</b>" * (k >= first_rep ? " · <span style='color:$(COL.red)'>this room already has a shared value (person $(first_rep))</span>" : "") for k in 0:n]
			player(f, g; n = n, start = 0, fps = 4, reads = reads, label = "people",
				caption = md"**Figure 3.5.** Top: the values (birthdays, weeks or months) of the people who have entered so far. Bottom: the probability that a room of ``n`` people contains at least one shared value.")
		end
	end

	function _check_birthday(birthday)
		let
			r = tryrun(() -> birthday(23))
			ok = r !== missing && isapprox(r, 0.5072972; atol = 1e-5) && isapprox(birthday(50), 0.9703736; atol = 1e-5)
			check(r, ok, () -> md"``P = `` $(fmt(r; d = 4)) for 23 people; ``P > 0.99`` from ``n = 57``.",
				() -> md"Got $(r); expected 0.5073 for ``n = 23``.";
				hint = md"Write out the product for ``n = 2``. How many factors should it have?")
		end
	end

	function figure_4_1(s4_rv)
		let
			rv = Dict("sum of the two dice" => (i, j) -> i + j, "larger of the two dice" => (i, j) -> max(i, j), "number of sixes" => (i, j) -> (i == 6) + (j == 6),
				"indicator of doubles" => (i, j) -> Int(i == j), "first minus second" => (i, j) -> i - j)[s4_rv]
			vals = sort(unique([rv(i, j) for i in 1:6 for j in 1:6]))
			nv = length(vals)
			ramp(t) = "rgb($(round(Int, 230 - 180t)),$(round(Int, 236 - 130t)),$(round(Int, 246 - 70t)))"
			colof(v) = ramp(nv == 1 ? 0.5 : (findfirst(==(v), vals) - 1) / (nv - 1))
			f = fig(w = 300, h = 320, xr = (0.5, 6.5), yr = (6.5, 0.3), axes = false, l = 40, t = 36, b = 8, r = 8)
			for i in 1:6, j in 1:6
				v = rv(i, j)
				rect!(f, j - 0.46, i - 0.46, j + 0.46, i + 0.46; fill = colof(v), op = 1, stroke = "#ffffff", sw = 1, rx = 3)
				text!(f, j, i, string(v); anchor = "middle", size = 12.5, c = findfirst(==(v), vals) > nv / 2 ? "white" : "#1b1b1b", bold = true)
			end
			for k in 1:6
				text!(f, k, 0.15, string(k); anchor = "middle", size = 11.5, c = "#555")
				text!(f, 0.2, k, string(k); anchor = "middle", size = 11.5, c = "#555")
			end
			text!(f, 3.5, -0.3, "second die"; anchor = "middle", size = 11.5, c = "#555")
			text!(f, -0.3, 3.5, "first die"; anchor = "middle", size = 11.5, c = "#555", rot = -90)
			p = [count(rv(i, j) == v for i in 1:6 for j in 1:6) / 36 for v in vals]
			g = fig(w = 370, h = 320, xr = (vals[1] - 0.7, vals[end] + 0.7), yr = (0, max(0.45, 1.15maximum(p))), xlabel = "value x", ylabel = "P(X = x)", xticks = vals)
			bars!(g, vals, p; width = 0.7, c = [colof(v) for v in vals], op = 1, stroke = "#888", labels = ["$(round(Int, 36q))/36" for q in p], lsize = 10.5)
			imax = argmax(p); EX = sum(vals .* p)
			figure(f, g; caption = md"**Figure 4.1.** Left: the random variable as a rule that writes a number into every cell of ``\Omega``. Right: its distribution, obtained by counting the cells with each value.")
		end
	end

	function compute_s5_die(s5_w)
		let
			w = max.(Float64.(s5_w), 0.0)
			sum(w) > 0 ? w ./ sum(w) : fill(1 / 6, 6)
		end
	end

	function figure_5_1(s5_die, s5_x, s5_q)
		let
			p = s5_die; F = cumsum(p); x = s5_x; q = s5_q
			Fx = x < 1 ? 0.0 : F[min(floor(Int, x), 6)]
			xq = findfirst(>=(q - 1e-12), F)
			f = fig(w = 330, h = 290, xr = (0.3, 6.7), yr = (0, max(0.4, 1.15maximum(p))), xlabel = "face x", ylabel = "p(x) = P(X = x)", xticks = 1:6)
			bars!(f, 1:6, p; width = 0.7, c = [k <= x ? COL.blue : "#c9d7ea" for k in 1:6], op = 1, labels = [fmt(v; d = 2) for v in p], lsize = 10.5)
			vline!(f, x; c = COL.red)
			g = fig(w = 360, h = 290, xr = (0, 7), yr = (0, 1.05), xlabel = "x", ylabel = "F(x) = P(X ≤ x)", xticks = 0:7)
			xs = [0.0; 1:6; 7.0]; Fs = [0.0; F; 1.0]
			for k in 1:7
				seg!(g, xs[k], Fs[k], xs[k+1], Fs[k]; c = COL.blue, w = 2.4)
			end
			dots!(g, 1:6, F; c = COL.blue, r = 4)
			dots!(g, 1:6, [0.0; F[1:5]]; c = "white", r = 3.5, stroke = COL.blue, sw = 1.5)
			seg!(g, x, 0, x, Fx; c = COL.red, dash = "4 3")
			seg!(g, 0, Fx, x, Fx; c = COL.red, dash = "4 3")
			dots!(g, [x], [Fx]; c = COL.red, r = 5)
			text!(g, 0.15, Fx > 0.9 ? Fx - 0.06 : Fx + 0.05, "F($(fmt(x))) = $(fmt(Fx))"; c = COL.red, size = 11.5, bg = true)
			seg!(g, 0, q, xq, q; c = COL.green, dash = "2 3", w = 1.6)
			seg!(g, xq, q, xq, 0; c = COL.green, dash = "2 3", w = 1.6)
			text!(g, xq + (xq > 4 ? -0.12 : 0.12), 0.06, "q-quantile = $(xq)"; c = COL.green, size = 11.5, anchor = xq > 4 ? "end" : "start")
			EX = sum((1:6) .* p)
			figure(f, g; caption = md"**Figure 5.1.** Left: the PMF of the loaded die; the bars with ``k \le x`` are dark. Right: the CDF, a staircase that jumps by ``p(k)`` at each face. Red: ``F(x)`` is the total height of the dark bars. Green: the ``q``-quantile is where the staircase first reaches height ``q``.")
		end
	end

	function _check_cdf_from_pmf(cdf_from_pmf, quantile_from_pmf)
		let
			p = [0.1, 0.2, 0.3, 0.15, 0.15, 0.1]
			F = tryrun(() -> cdf_from_pmf(p)); Q = tryrun(() -> quantile_from_pmf(p, 0.5))
			r = (F === missing || Q === missing) ? missing : (F, Q)
			ok = r !== missing && isapprox(collect(F), cumsum(p)) && Q == 3 && quantile_from_pmf(p, 0.95) == 6 && quantile_from_pmf(p, 0.1) == 1
			check(r, ok, () -> md"CDF $(round.(collect(F), digits = 2)); median at face 3; 0.95-quantile at face 6.",
				() -> md"Got $(r); expected the cumulative sums and the median 3.";
				hint = md"The quantile is ``\min\{x : F(x) \ge q\}``. Does the code look for the first value at or above ``q``?")
		end
	end

	function compute_s5_gaps(sd01_3, s5_ng)
		randexp(MersenneTwister(5 + 1009sd01_3), s5_ng) .* 20
	end

	function figure_5_2(s5_bw, s5_gaps)
		let
			bw = s5_bw; edges = collect(0:bw:80 + 1e-9)
			cnt = histdata(s5_gaps, edges; density = false)
			fr = cnt ./ length(s5_gaps); den = fr ./ bw
			mk(ylab, h, ymax, title) = (f = fig(w = 226, h = 240, xr = (0, 80), yr = (0, ymax), xlabel = "gap (min)", ylabel = ylab, title = title, l = 52); hist!(f, edges, h; c = COL.blue, op = 0.75); f)
			f1 = mk("count", cnt, 1.1maximum(cnt), "counts per bin")
			f2 = mk("fraction", fr, max(0.05, 1.1maximum(fr)), "fraction per bin")
			f3 = mk("fraction per minute", den, 0.07, "fraction ÷ bin width")
			xs = range(0, 80, length = 300)
			line!(f3, xs, pdf.(Exponential(20), xs); c = COL.red, w = 2.2)
			legend!(f3, [("density f(t)", COL.red)]; at = :tr, size = 11)
			n = length(s5_gaps)
			txt = Markdown.parse("With $(n) gaps and bins of $(fmt(bw)) min, the first bin has $(round(Int, cnt[1])) gaps, a fraction $(fmt(fr[1])); divided by the width this is **$(fmt(den[1]))** per minute, against the density ``f(0) = 1/20 = 0.05``. " * (n * bw < 1500 ? "With few gaps per bin the right panel becomes noisy: narrow bins need many data." : "The finer the bins, the closer the right panel follows the red curve, provided there are enough data in each bin."))
			figure(f1, f2, f3, txt; caption = md"**Figure 5.2.** Simulated waiting times between calls (exponential, mean 20 minutes), binned in three ways.")
		end
	end

	function compute_s5_D(s5_dist)
		Dict(
			"waiting time between calls (exponential)" => (d = Exponential(20), lo = 0.0, hi = 100.0, unit = "min"),
			"adult height (normal)" => (d = Normal(172, 9), lo = 140.0, hi = 205.0, unit = "cm"),
			"position of a raindrop on a 1 m stick (uniform)" => (d = Uniform(0, 1), lo = -0.2, hi = 1.2, unit = "m"),
			"two groups mixed (bimodal)" => (d = MixtureModel([Normal(-1.6, 0.6), Normal(1.4, 0.8)], [0.45, 0.55]), lo = -4.0, hi = 4.5, unit = ""))[s5_dist]
	end

	function figure_5_3(s5_D, s5_a, s5_b)
		let
			D = s5_D; lo, hi = D.lo, D.hi
			a = lo + min(s5_a, s5_b) * (hi - lo); b = lo + max(s5_a, s5_b) * (hi - lo)
			xs = range(lo, hi, length = 400); fx = pdf.(D.d, xs); Fx = cdf.(D.d, xs)
			ymax = 1.15maximum(fx)
			f = fig(w = 340, h = 280, xr = (lo, hi), yr = (0, ymax), xlabel = "x" * (D.unit == "" ? "" : " ($(D.unit))"), ylabel = "density f(x)")
			sel = [a <= x <= b for x in xs]
			area!(f, xs[sel], fx[sel]; c = COL.blue, op = 0.35)
			line!(f, xs, fx; c = COL.blue, w = 2.2)
			vline!(f, a; c = COL.grey); vline!(f, b; c = COL.grey)
			pab = cdf(D.d, b) - cdf(D.d, a)
			text!(f, (a + b) / 2, ymax * 0.93, "area = " * fmt(pab); anchor = "middle", c = COL.blue, size = 12, bold = true, bg = true)
			g = fig(w = 340, h = 280, xr = (lo, hi), yr = (0, 1.05), xlabel = "x" * (D.unit == "" ? "" : " ($(D.unit))"), ylabel = "F(x) = P(X ≤ x)")
			line!(g, xs, Fx; c = COL.ink, w = 2.2)
			Fa, Fb = cdf(D.d, a), cdf(D.d, b)
			seg!(g, lo, Fa, a, Fa; c = COL.grey, dash = "3 3"); seg!(g, lo, Fb, b, Fb; c = COL.grey, dash = "3 3")
			seg!(g, a, 0, a, Fa; c = COL.grey, dash = "3 3"); seg!(g, b, 0, b, Fb; c = COL.grey, dash = "3 3")
			xm = lo + 0.06 * (hi - lo)
			seg!(g, xm, Fa, xm, Fb; c = COL.blue, w = 5)
			text!(g, xm + 0.02 * (hi - lo), (Fa + Fb) / 2, "F(b) − F(a) = " * fmt(pab); c = COL.blue, size = 12, bold = true, bg = true)
			figure(f, g; caption = md"**Figure 5.3.** Left: ``P(a \le X \le b)`` is the shaded area under the density. Right: the same number is the increase of the CDF from ``a`` to ``b`` (blue bar). Where the density is high, the CDF is steep; where the density is zero, the CDF is flat.")
		end
	end

	function figure_6_1(s6_shape)
		let
			D = Dict("symmetric: adult height" => (Normal(172, 9), 140.0, 205.0, "height (cm)"),
				"right-skewed: household income" => (LogNormal(log(42), 0.6), 0.0, 180.0, "income (thousand per year)"),
				"right-skewed: waiting time" => (Exponential(15), 0.0, 75.0, "waiting time (min)"),
				"bimodal: two groups" => (MixtureModel([Normal(4, 1.0), Normal(10, 1.4)], [0.6, 0.4]), 0.0, 15.0, "value"))[s6_shape]
			d, lo, hi, lab = D
			xs = range(lo, hi, length = 500); fx = pdf.(d, xs)
			m = mean(d); s = std(d); med = quantile(d, 0.5); mo = xs[argmax(fx)]
			ymax = 1.7maximum(fx)
			f = fig(w = 680, h = 340, xr = (lo, hi), yr = (-0.12ymax, ymax), xlabel = lab, ylabel = "density", yticks = Float64[])
			left = xs .<= med
			area!(f, xs[left], fx[left]; c = COL.ochre, op = 0.18)
			area!(f, xs[.!left], fx[.!left]; c = COL.blue, op = 0.12)
			line!(f, xs, fx; c = COL.ink, w = 2)
			seg!(f, lo, 0, hi, 0; c = "#333", w = 1)
			tri = [m - 0.018(hi - lo), m + 0.018(hi - lo), m]
			poly!(f, tri, [-0.1ymax, -0.1ymax, 0]; fill = COL.red, op = 0.9)
			seg!(f, med, 0, med, pdf(d, med); c = COL.ochre, w = 2.4)
			dots!(f, [mo], [maximum(fx)]; c = COL.green, r = 5.5)
			seg!(f, m - s, 0.66ymax, m + s, 0.66ymax; c = COL.purple, w = 2.5)
			seg!(f, m - s, 0.63ymax, m - s, 0.69ymax; c = COL.purple, w = 2); seg!(f, m + s, 0.63ymax, m + s, 0.69ymax; c = COL.purple, w = 2)
			text!(f, m, 0.72ymax, "mean ± 1 standard deviation (σ = $(fmt(s)))"; c = COL.purple, size = 11.5, anchor = "middle", bg = true)
			legend!(f, [("mean (balance point) = $(fmt(m))", COL.red, :bar), ("median (halves the area) = $(fmt(med))", COL.ochre), ("mode (peak) = $(fmt(mo))", COL.green, :dot)]; at = :tr)
			rel = abs(m - med) < 0.02s ? "Mean and median almost coincide: the distribution is close to symmetric." : m > med ? "The mean exceeds the median by $(fmt(m - med)): the long right tail pulls the balance point to the right, while the median, which only counts how much area lies on each side, does not move. Most values lie below the mean." : "The mean lies below the median by $(fmt(med - m)): a long left tail pulls it down."
			txt = Markdown.parse("Mean **$(fmt(m))**, median **$(fmt(med))**, mode **$(fmt(mo))**, standard deviation **$(fmt(s))**. " * rel * (s6_shape == "bimodal: two groups" ? "" : ""))
			figure(f, txt; caption = md"**Figure 6.1.** Mean, median and mode of the selected distribution.")
		end
	end

	function compute_s6_data(s6_my, s6_set)
		Dict("bus line A" => [7.0, 8, 9, 10, 10, 11, 12, 13], "bus line B" => [1.0, 3, 6, 9, 11, 14, 17, 19], "my data" => parse_numbers(s6_my))[s6_set]
	end

	function figure_6_2(s6_data)
		let
			x = s6_data
			if length(x) < 2
				md"*Enter at least two numbers separated by spaces.*"
			else
				n = length(x); xb = mean(x); d = x .- xb; v = mean(d .^ 2); s = sqrt(v)
				lo = min(minimum(x), xb - s) - 1; hi = max(maximum(x), xb + s) + 1
				W = 680; l = 30; r = 16
				sx = (W - l - r) / (hi - lo)
				k = min(1.0, 56 / (max(maximum(abs.(d)), s) * sx))
				hts = abs.(d) .* sx .* k
				order = sortperm(x)
				Hp = sum(hts) + 6n + s * sx * k + 40
				H = round(Int, Hp + 10 + 46)
				f = fig(w = W, h = H, xr = (lo, hi), yr = (0, Hp), xlabel = "value", yticks = Float64[], l = l, r = r, t = 10, b = 46)
				vline!(f, xb; c = COL.ink, dash = "5 3", w = 1.4)
				y = 4.0
				for i in order
					x0, x1 = minmax(xb, x[i])
					rect!(f, x0, y, x1, y + hts[i]; fill = d[i] >= 0 ? COL.blue : COL.ochre, op = 0.28, stroke = d[i] >= 0 ? COL.blue : COL.ochre, sw = 1)
					dots!(f, [x[i]], [y]; c = COL.ink, r = 3.5)
					y += hts[i] + 6
				end
				hs = s * sx * k
				rect!(f, xb, y + 6, xb + s, y + 6 + hs; fill = COL.red, op = 0.35, stroke = COL.red, sw = 1.6)
				lab = "variance = $(fmt(v)) (average area), σ = $(fmt(s))"
				if xb - lo > hi - xb - s
					text!(f, xb - 0.01(hi - lo), y + 6 + hs / 2, lab; c = COL.red, size = 12, anchor = "end")
				else
					text!(f, xb + s + 0.01(hi - lo), y + 6 + hs / 2, lab; c = COL.red, size = 12)
				end
				text!(f, xb, Hp - 8, "mean = $(fmt(xb))"; anchor = "middle", size = 12, bg = true)
				rows = [[fmt(x[i]), fmt(d[i]), fmt(d[i]^2)] for i in order]
				push!(rows, ["mean", "0", fmt(v)])
				figure(f, tbl(["value xᵢ", "deviation xᵢ − x̄", "square (xᵢ − x̄)²"], rows);
					caption = md"**Figure 6.2.** Each value (dot) and its deviation from the mean (dashed line), drawn as a rectangle of width ``|x_i - \bar x|``.")
			end
		end
	end

	function _check_mean_var(mean_var)
		let
			r = tryrun(() -> mean_var(1:6, fill(1 / 6, 6)))
			ok = r !== missing && isapprox(r[1], 3.5) && isapprox(r[2], 35 / 12) && all(isapprox.(mean_var([0, 1], [0.7, 0.3]), (0.3, 0.21)))
			check(r, ok, () -> md"Fair die: mean 3.5, variance 2.917. A Bernoulli variable with ``p = 0.3``: mean 0.3, variance ``0.3 \cdot 0.7 = 0.21``.",
				() -> md"Got $(r); expected (3.5, 2.917) for the fair die.";
				hint = md"The mean is `sum(values .* probs)`. The second moment ``E[X^2]`` weights the *squared* values with the probabilities. Then use ``\operatorname{Var} = E[X^2] - (E[X])^2``.")
		end
	end

	function figure_6_3(s6_roll, s5_die, s6_n)
		let
			p = s5_die; vals = 1:6
			m = sum(vals .* p); v = sum((vals .- m) .^ 2 .* p)
			rng = MersenneTwister(66 + 31s6_roll); n = s6_n
			cp = cumsum(p); r = [findfirst(>(rand(rng)), cp) for _ in 1:n]
			r = [x === nothing ? 6 : x for x in r]
			rm = cumsum(r) ./ (1:n)
			rv = [k == 1 ? 0.0 : mean((r[1:k] .- rm[k]) .^ 2) for k in 1:n]
			f = fig(w = 335, h = 260, xr = (0, n), yr = (1, 6), xlabel = "rolls", ylabel = "running mean", title = "mean")
			hline!(f, m; c = COL.red, dash = nothing, w = 1.5)
			text!(f, n, m + 0.25, "E[X] = $(fmt(m))"; anchor = "end", c = COL.red, size = 11.5)
			g = fig(w = 335, h = 260, xr = (0, n), yr = (0, max(6, 1.6v)), xlabel = "rolls", ylabel = "running variance", title = "variance")
			hline!(g, v; c = COL.red, dash = nothing, w = 1.5)
			text!(g, n, v + 0.3, "Var(X) = $(fmt(v))"; anchor = "end", c = COL.red, size = 11.5)
			for k in 2:n
				seg!(f, k - 1, rm[k-1], k, rm[k]; c = COL.blue, w = 1.8, s = k)
				seg!(g, k - 1, rv[k-1], k, rv[k]; c = COL.green, w = 1.8, s = k)
			end
			reads = ["Rolls: <b>$(k)</b>" * (k == 0 ? "" : " · last roll: <b>$(r[k])</b> · sample mean <b>$(fmt(rm[k]))</b> · average squared deviation <b>$(fmt(rv[k]))</b>") for k in 0:n]
			note = Markdown.parse("After all $(n) rolls the sample mean is $(fmt(rm[n])) (exact $(fmt(m)), off by $(fmt(abs(rm[n] - m); d = 2))) and the average squared deviation is $(fmt(rv[n])) (exact $(fmt(v))). The typical error of the mean after ``n`` rolls is ``\\sigma/\\sqrt n``, here $(fmt(sqrt(v / n); d = 2)); roll again to see how much the end points move. Change the die in Figure 5.1 to change the targets.")
			player(f, g, note; n = n, start = 0, fps = n > 1000 ? 60 : 25, reads = reads, label = "roll",
				caption = md"**Figure 6.3.** Running sample mean (left) and running average squared deviation (right) for rolls of the loaded die of Figure 5.1, against the values computed from its distribution (red).")
		end
	end
end;

# ╔═╡ 7303ec21-89b8-7a3f-495c-9a33cb8d75e1
figure_2_1(s2_A, s2_B)

# ╔═╡ 9f77330a-df6f-702c-ae64-9c0a1e5ca609
figure_2_3(s2_n, s2_r)

# ╔═╡ 1dd0247f-65c1-fbbb-3ff7-b77a146d2cbf
figure_3_1(s3_B, s3_A)

# ╔═╡ c2f97557-3d83-8d6d-7532-a6de6ccca995
figure_4_1(s4_rv)

# ╔═╡ 97e88008-96ed-5038-df09-4db0a2e1b27c
s5_D = compute_s5_D(s5_dist);

# ╔═╡ a5e29ac8-8b21-e8f1-521f-3aca135df3ca
figure_5_3(s5_D, s5_a, s5_b)

# ╔═╡ e201e3f9-249d-602d-e8d1-d83ff1a4ae3e
figure_6_1(s6_shape)

# ╔═╡ e089f15e-f8e2-1505-1ad0-488b06ae9cc1
s6_data = compute_s6_data(s6_my, s6_set);

# ╔═╡ b8300184-a115-f7c8-c3fb-291cabcd93fe
figure_6_2(s6_data)

# ╔═╡ 4ffa8386-28e9-519f-d266-af124bd86b05
# the guiding problem
begin
	function figure_9_1(s9_dt)
		λ = 3 / 60; Δ = s9_dt; p = λ * Δ; tmax = 100.0
		ks = 1:ceil(Int, tmax / Δ)
		pk = [(1 - p)^(k - 1) * p for k in ks]
		tt = range(0, tmax, length = 300)
		f = fig(w = 330, h = 260, xr = (0, tmax), yr = (0, 1.25λ), xlabel = "waiting time t (min)", ylabel = "probability per minute", title = "geometric versus exponential")
		if Δ >= 1
			for (k, q) in zip(ks, pk)
				rect!(f, (k - 1) * Δ, 0, k * Δ, q / Δ; fill = COL.blue, op = 0.45, stroke = "white", sw = Δ >= 2 ? 1 : 0.3)
			end
		else
			area!(f, ks .* Δ, pk ./ Δ; c = COL.blue, op = 0.45)
		end
		line!(f, tt, λ .* exp.(-λ .* tt); c = COL.red, w = 2)
		legend!(f, [("geometric, Δt = $(fmt(Δ)) min", COL.blue, :bar), ("exponential density λe^(−λt)", COL.red)]; at = :tr, size = 10.5)
		g = fig(w = 330, h = 260, xr = (0, tmax), yr = (0, 1.05), xlabel = "t (min)", ylabel = "P(T > t)", title = "survival function")
		surv = [(1 - p)^floor(t / Δ) for t in tt]
		line!(g, tt, surv; c = COL.blue, w = 2, steps = true)
		line!(g, tt, exp.(-λ .* tt); c = COL.red, w = 2, dash = "5 3")
		vline!(g, 20; c = "#999", dash = "3 3", w = 1)
		pd = (1 - p)^(20 / Δ)
		figure(f, g; caption = md"**Figure 9.1.** Discretised time. Left: the distribution of the waiting time when time is divided into intervals of length ``\Delta t`` (geometric distribution) and the exponential density (red). Right: the probability that the waiting time exceeds ``t``. As ``\Delta t \to 0``, the geometric distribution converges to the exponential distribution.")
	end

	function answer(guess_a, guess_b)
		p = exp(-1)
		g = guess_a > 0 || guess_b > 0 ? "Your estimates: $(guess_a) for (a) and $(guess_b) for (b). " : ""
		adm("key", "Result", Markdown.parse(g * "Both probabilities are ``e^{-1} \\approx`` **$(fmt(p; d = 3))**. The time already waited does not change the distribution of the remaining waiting time."))
	end
end;

# ╔═╡ 90fe57dd-f17b-bad0-b419-caba2185e6ab
figure_9_1(s9_dt)

# ╔═╡ 703cfafb-cedd-df0e-9b19-e53c36c5f728
answer(guess_a, guess_b)

# ╔═╡ a51b76ee-b700-52d3-8eb9-e01c001b496e
begin
	function _check_binom_pmf(binom_pmf)
		r = tryrun(() -> binom_pmf(3, 10, 0.2))
		ok = r !== missing && r isa Real && isapprox(r, pdf(Binomial(10, 0.2), 3); atol = 1e-10) && isapprox(sum(binom_pmf(k, 7, 0.35) for k in 0:7), 1; atol = 1e-10)
		check(r, ok, () -> md"``P(X = 3)`` for ``\mathrm{Bin}(10, 0.2)`` is $(fmt(r; d = 4)), and the probabilities for ``n = 7`` sum to 1.",
			() -> md"For ``k = 3``, ``n = 10``, ``p = 0.2`` the function returns $(r); the correct value is 0.2013.";
			hint = md"``\binom{n}{k} p^k (1-p)^{n-k}`` translates directly: `binomial(n, k) * p^k * (1 - p)^(n - k)`.")
	end

	check_binom_pmf(f) = guarded(() -> _check_binom_pmf(f))
end;

# ╔═╡ b1a64b16-e722-9012-073e-ff05a0615c5f
check_binom_pmf(binom_pmf)

# ╔═╡ b96e7f21-a157-96fb-baea-4b9ab8657284
begin
	const VOCAB = Dict(
		"rolling two dice" => ["all possible rolls of two dice", "the 20 rolls you make today", "roll both dice once and note the two faces", "Ω = all 36 ordered pairs (1,1), (1,2), …, (6,6)", "A = \"the sum is 7\" = {(1,6), (2,5), …, (6,1)}", "P(A) = 6/36 = 1/6"],
		"one hour at a service hotline" => ["all hours in which the hotline is open", "the 200 hours recorded last month", "count the calls that arrive in one hour", "Ω = {0, 1, 2, 3, …}", "A = \"more than 6 calls\" = {7, 8, 9, …}", "P(A) = the long-run fraction of hours with more than 6 calls"],
		"quality control of phones" => ["all phones made in the factory this month", "10 phones taken at random from the line", "test the 10 phones and count the faulty ones", "Ω = {0, 1, 2, …, 10}", "A = \"at least one faulty phone\" = {1, 2, …, 10}", "P(A) = how often a batch of 10 contains a faulty phone"])

	# the possible results of each experiment, drawn as cells; inA marks the event A
	const OUT = Dict(
		"rolling two dice" => (cols = 6, labels = fill("", 36), inA = [i + j == 7 for j in 1:6 for i in 1:6]),
		"one hour at a service hotline" => (cols = 7, labels = [string.(0:11); "…"], inA = [n > 6 for n in 0:11] |> x -> [x; true]),
		"quality control of phones" => (cols = 6, labels = string.(0:10), inA = [n >= 1 for n in 0:10]))

	const INK = "#1a1a1a"   # all body text: near-black, never grey

function figure_1_1(s1_ex)
	v = VOCAB[s1_ex]; o = OUT[s1_ex]
	W, H = 680, 590
	f = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
	wrap(t, k) = begin
		ls = String[]; cur = ""
		for w in split(t)
			if cur != "" && length(cur) + length(w) + 1 > k
				push!(ls, cur); cur = String(w)
			else
				cur = cur == "" ? String(w) : cur * " " * w
			end
		end
		cur != "" && push!(ls, cur); ls
	end
	sides = [
		(x0 = 15, col = COL.blue, bg = "#eef3fa", head = "THE REAL WORLD",
			terms = ["population", "sample", "experiment"],
			defs = ["everything we want to know about", "the part we actually look at", "what we do to get data"], idx = 1:3),
		(x0 = 385, col = COL.purple, bg = "#f6f4fa", head = "THE MATHEMATICAL MODEL",
			terms = ["sample space Ω", "event A", "probability P(A)"],
			defs = ["the set of every possible result", "a set of results we care about", "a number saying how likely A is"], idx = 4:6)]
	for (side, s) in enumerate(sides)
		x0 = s.x0; x1 = x0 + 280
		text!(f, x0, 22, s.head; size = 15, bold = true, c = s.col)
		rect!(f, x0, 34, x1, 234; fill = s.bg, op = 1, stroke = s.col, sw = 1.8, rx = 14)
		if side == 1
			# population: light blue objects, with the sample (dark blue) inside it
			text!(f, x0 + 16, 56, "population"; size = 14, bold = true, c = s.col)
			sx0 = x0 + 120; sx1 = x1 - 14; sy0 = 112; sy1 = 220
			gx = Float64[]; gy = Float64[]
			for i in 0:9, j in 0:5
				x = x0 + 26 + 24i; y = 78 + 24j
				(sx0 - 12 <= x <= sx1 + 12 && sy0 - 12 <= y <= sy1 + 12) && continue
				push!(gx, x); push!(gy, y)
			end
			dots!(f, gx, gy; c = "#9db6dd", r = 3.5)
			rect!(f, sx0, sy0, sx1, sy1; fill = "white", op = 1, stroke = COL.blue, sw = 2.4, rx = 10)
			text!(f, sx0 + 14, sy0 + 20, "sample"; size = 13, bold = true, c = COL.blue)
			bx = Float64[]; by = Float64[]
			for i in 0:4, j in 0:2
				push!(bx, sx0 + 22 + 24i); push!(by, sy0 + 44 + 24j)
			end
			dots!(f, bx, by; c = COL.blue, r = 4)
		else
			# sample space: the real outcomes as cells, event A highlighted
			text!(f, x0 + 16, 56, "sample space Ω"; size = 14, bold = true, c = s.col)
			rect!(f, x1 - 92, 45, x1 - 78, 59; fill = "#fbe3df", op = 1, stroke = COL.red, sw = 1.6, rx = 2)
			text!(f, x1 - 72, 57, "event A"; size = 12, bold = true, c = COL.red)
			n = length(o.labels); rows = cld(n, o.cols); cs = min(34, 150 ÷ rows)
			gx0 = x0 + (280 - o.cols * cs) / 2; gy0 = 70 + (160 - rows * cs) / 2
			for k in 1:n
				c = (k - 1) % o.cols; r = (k - 1) ÷ o.cols
				cx = gx0 + c * cs; cy = gy0 + r * cs; a = o.inA[k]
				rect!(f, cx + 1, cy + 1, cx + cs - 1, cy + cs - 1; fill = a ? "#fbe3df" : "white", op = 1, stroke = a ? COL.red : "#5b6b8c", sw = a ? 2 : 1.2, rx = 4)
				text!(f, cx + cs / 2, cy + cs / 2 + 4, o.labels[k]; anchor = "middle", size = 12, bold = a, c = a ? COL.red : INK)
			end
		end
		# three aligned blocks: TERM (coloured) -> what it means -> Example
		y = 268.0
		for (k, term) in enumerate(s.terms)
			rect!(f, x0, y - 14, x0 + 280, y + 6; fill = s.bg, op = 1, rx = 4)
			text!(f, x0 + 8, y, term; size = 14, bold = true, c = s.col)
			text!(f, x0 + 8, y + 22, s.defs[k]; size = 12.5, c = INK)
			text!(f, x0 + 8, y + 42, "Example:"; size = 12, bold = true, c = s.col)
			ls = wrap(v[s.idx[k]], 38)
			for (i, l) in enumerate(ls)
				text!(f, x0 + 70, y + 42 + 16 * (i - 1), l; size = 12.5, c = INK)
			end
			# arrow from each real-world term to its model counterpart
			side == 1 && arrow!(f, 300, y - 4, 380, y - 4; c = INK, w = 2, head = 8)
			side == 1 && text!(f, 340, y - 12, "modelled by"; anchor = "middle", size = 11.5, bold = true, c = INK)
			y += 100
		end
	end
	figure(f; caption = md"**Figure 1.1.** Basic terms in probability.")
end

	"""Monty Hall."""
	function montyhall_game()
		@htl("""<div class="mh" style="font-family:system-ui,sans-serif;max-width:680px">
		<div class="mh-doors" style="display:flex;gap:18px;justify-content:center;margin:8px 0"></div>
		<div class="mh-msg" style="text-align:center;min-height:2.6em;font-size:1.02rem"></div>
		<div style="text-align:center;margin:6px 0" class="mh-act"></div>
		<table class="tbl" style="margin:10px auto"><thead><tr><th></th><th>rounds</th><th>won the car</th><th>share</th></tr></thead>
		<tbody><tr><td>stayed</td><td class="s-n">0</td><td class="s-w">0</td><td class="s-p">–</td></tr>
		<tr><td>switched</td><td class="w-n">0</td><td class="w-w">0</td><td class="w-p">–</td></tr></tbody></table>
		<div style="text-align:center;font-size:0.9rem;color:#555">Impatient? <button class="auto-s">stay 100 times</button> <button class="auto-w">switch 100 times</button> <button class="reset">reset the score</button></div>
		<script>
		const root = currentScript.parentElement;
		const doorsEl = root.querySelector(".mh-doors"), msg = root.querySelector(".mh-msg"), act = root.querySelector(".mh-act");
		let prize, pick, opened, stage, sc = {sn: 0, sw: 0, wn: 0, ww: 0};
		const emoji = {car: "🚗", goat: "🐐"};
		function upd() {
			const f = (w, n) => n ? (100 * w / n).toFixed(1) + " %" : "–";
			root.querySelector(".s-n").textContent = sc.sn; root.querySelector(".s-w").textContent = sc.sw; root.querySelector(".s-p").textContent = f(sc.sw, sc.sn);
			root.querySelector(".w-n").textContent = sc.wn; root.querySelector(".w-w").textContent = sc.ww; root.querySelector(".w-p").textContent = f(sc.ww, sc.wn);
		}
		function draw() {
			doorsEl.innerHTML = "";
			for (let d = 0; d < 3; d++) {
				const b = document.createElement("button");
				const open = (stage === "decide" && d === opened) || stage === "done";
				b.style.cssText = "width:120px;height:150px;border-radius:8px 8px 2px 2px;font-size:" + (open ? "54px" : "26px") + ";cursor:pointer;border:3px solid " + (d === pick ? "#c2462e" : "#7a5c3a") + ";background:" + (open ? "#fffdf5" : "#b9874f") + ";color:#fff";
				b.textContent = open ? (d === prize ? emoji.car : emoji.goat) : "Door " + (d + 1);
				if (!open) b.style.color = "#fff";
				b.addEventListener("click", () => choose(d));
				doorsEl.appendChild(b);
			}
		}
		function newRound() {
			prize = Math.floor(Math.random() * 3); pick = -1; opened = -1; stage = "pick";
			msg.innerHTML = "A car is behind one door, goats behind the other two. <b>Pick a door.</b>"; act.innerHTML = ""; draw();
		}
		function choose(d) {
			if (stage !== "pick") return;
			pick = d; const cand = [0, 1, 2].filter(x => x !== pick && x !== prize);
			opened = cand[Math.floor(Math.random() * cand.length)]; stage = "decide";
			msg.innerHTML = "You picked door " + (pick + 1) + ". The host, who knows where the car is, opens door " + (opened + 1) + ": a goat. <b>Stay or switch?</b>";
			act.innerHTML = "<button class='st'>stay with door " + (pick + 1) + "</button> <button class='sw'>switch</button>";
			act.querySelector(".st").addEventListener("click", () => finish(false));
			act.querySelector(".sw").addEventListener("click", () => finish(true));
			draw();
		}
		function finish(sw) {
			if (sw) pick = [0, 1, 2].find(x => x !== pick && x !== opened);
			const win = pick === prize; stage = "done";
			if (sw) { sc.wn++; if (win) sc.ww++; } else { sc.sn++; if (win) sc.sw++; }
			msg.innerHTML = (win ? "🎉 <b>You win the car!</b>" : "<b>A goat.</b> Better luck next time.") + " You " + (sw ? "switched" : "stayed") + ".";
			act.innerHTML = "<button class='again'>play again</button>"; act.querySelector(".again").addEventListener("click", newRound);
			draw(); upd();
		}
		function auto(sw) {
			for (let k = 0; k < 100; k++) {
				const pr = Math.floor(Math.random() * 3), pk = Math.floor(Math.random() * 3);
				const op = [0, 1, 2].filter(x => x !== pk && x !== pr)[0];
				const fin = sw ? [0, 1, 2].find(x => x !== pk && x !== op) : pk;
				if (sw) { sc.wn++; if (fin === pr) sc.ww++; } else { sc.sn++; if (fin === pr) sc.sw++; }
			}
			upd();
		}
		root.querySelector(".auto-s").addEventListener("click", () => auto(false));
		root.querySelector(".auto-w").addEventListener("click", () => auto(true));
		root.querySelector(".reset").addEventListener("click", () => { sc = {sn: 0, sw: 0, wn: 0, ww: 0}; upd(); });
		newRound(); upd();
		</script></div>""")
	end

	function figure_3_6()
		f = fig(w = 680, h = 250, xr = (0, 10), yr = (0, 6.2), axes = false, l = 4, r = 4, t = 24, b = 4, title = "you pick door 1; where is the car, and what does the host do?")
		rows = [("car behind 1", "host opens 2 or 3", "stay wins", "switch loses", COL.blue), ("car behind 2", "host must open 3", "stay loses", "switch wins", COL.red), ("car behind 3", "host must open 2", "stay loses", "switch wins", COL.red)]
		for (k, (a, b, c, d, col)) in enumerate(rows)
			y = 5.2 - 1.8(k - 1)
			arrow!(f, 0.6, 3.4, 2.0, y; c = "#888", w = 1.3, head = 6); text!(f, 1.1, (3.4 + y) / 2 + 0.25, "1/3"; size = 12, bold = true, c = "#555")
			rect!(f, 2.05, y - 0.38, 4.3, y + 0.38; fill = "#f4f4f1", op = 1, stroke = "#bbb", rx = 4); text!(f, 3.17, y, a; anchor = "middle", size = 12)
			arrow!(f, 4.35, y, 5.05, y; c = "#888", w = 1.3, head = 6)
			rect!(f, 5.1, y - 0.38, 7.25, y + 0.38; fill = "#f4f4f1", op = 1, stroke = "#bbb", rx = 4); text!(f, 6.17, y, b; anchor = "middle", size = 12)
			text!(f, 7.5, y + 0.2, c; size = 12, c = col == COL.blue ? COL.blue : "#777", bold = col == COL.blue)
			text!(f, 7.5, y - 0.25, d; size = 12, c = col == COL.red ? COL.red : "#777", bold = col == COL.red)
		end
		circle!(f, 0.45, 3.4, 13; fill = "#fff", stroke = COL.ink)
		text!(f, 0.45, 3.4, "start"; anchor = "middle", size = 9.5)
		figure(f; caption = md"**Figure 3.4.** The three equally likely places of the car, after you picked door 1. Staying wins only in the first branch (probability 1/3); switching wins in the other two (probability 2/3). The host's knowledge is what makes this work: he never opens the door with the car.")
	end

	function _check_monty_round(monty_round)
		let
			r = tryrun(() -> (monty_round(MersenneTwister(1); switch = true), monty_round(MersenneTwister(1); switch = false)))
			ok = r !== missing && let
				sw = mean(monty_round(MersenneTwister(k); switch = true) for k in 1:30_000)
				st = mean(monty_round(MersenneTwister(k); switch = false) for k in 1:30_000)
				abs(sw - 2 / 3) < 0.015 && abs(st - 1 / 3) < 0.015
			end
			check(r, ok, () -> md"Over 30 000 rounds switching won $(pct(mean(monty_round(MersenneTwister(k); switch = true) for k in 1:30_000))) and staying $(pct(mean(monty_round(MersenneTwister(k); switch = false) for k in 1:30_000))) of the time: 2/3 against 1/3.",
				md"Switching should win about 2/3 of the rounds and staying about 1/3.";
				hint = md"After the switch, `pick` must be the only door that is neither the old `pick` nor `opened`. A list comprehension such as `[d for d in 1:3 if …]` finds it, and `only(…)` takes its single element.")
		end
	end

	check_monty_round(f) = guarded(() -> _check_monty_round(f))

	check_prob_two_dice(prob_two_dice) = guarded(() -> _check_prob_two_dice(prob_two_dice))

	check_birthday(birthday) = guarded(() -> _check_birthday(birthday))

	check_cdf_from_pmf(cdf_from_pmf, quantile_from_pmf) = guarded(() -> _check_cdf_from_pmf(cdf_from_pmf, quantile_from_pmf))

	check_mean_var(mean_var) = guarded(() -> _check_mean_var(mean_var))
end;

# ╔═╡ 006f021e-087e-40cd-e940-f311ddb26ca4
figure_1_1(s1_ex)

# ╔═╡ 814b0094-53a9-137f-aaf1-c3cea64dbdc9
montyhall_game()

# ╔═╡ 7e2d1664-bbd8-5829-6739-c16fadca983d
figure_3_6()

# ╔═╡ e2e171ad-0331-00d9-da7c-d0f4cc166d2c
check_mean_var(mean_var)

# ╔═╡ 941cdcee-df7c-a45d-1592-d2802c8739d8
check_prob_two_dice(prob_two_dice)

# ╔═╡ 9f81f1ff-233a-ed94-d1aa-c02a369d607c
check_birthday(birthday)

# ╔═╡ 2febad3a-cd42-3540-d23c-341e2e7e00e9
check_monty_round(monty_round)

# ╔═╡ 117bdd63-7da7-0d16-32bc-9a1f4e270dda
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

# ╔═╡ eeadeb56-cc29-a153-6de8-d81cdba978cd
md"""
### 1.2 The frequentist interpretation

In the frequentist interpretation, ``P(A)`` is the fraction of repetitions of the experiment in which ``A`` occurs, after many repetitions. If all elementary events are equally likely, this fraction can be computed by counting:

```math
P(A) = \frac{\#\text{elementary events in } A}{\#\text{elementary events in } \Omega} .
```

For a fair die and ``A = \{2, 4, 6\}`` this gives ``P(A) = 3/6 = 1/2``. Figure 1.2 shows how the relative frequency of an event approaches its probability when a die is rolled repeatedly.

Event $(@bind s1_ev Select(["a six", "an even number", "at least 5", "not a one"]; default = "a six")) ``\qquad`` rolls $(@bind s1_n Select([100, 600, 2000]; default = 600)) ``\qquad`` $(@bind s1_roll Seed(0; label = "Roll again"))
"""

# ╔═╡ a170994f-dee4-3124-f06b-45d89ad2041b
figure_1_2(s1_roll, s1_n, s1_ev)

# ╔═╡ f8bc7c83-9cfb-b9df-1b35-03e854f60aa5
md"""
### 3.5 Probability expansion and the birthday problem

Applying the multiplication rule repeatedly gives the **probability expansion**:

```math
P(A_1 \cap A_2 \cap \dots \cap A_n) = P(A_1)\,P(A_2 \mid A_1)\,P(A_3 \mid A_1 \cap A_2) \cdots P(A_n \mid A_1 \cap \dots \cap A_{n-1}).
```

It decomposes a complicated probability into simple factors and is the starting point of Markov chains. A classical application is the birthday problem: how many people must be in a room for the probability that two of them share a birthday to exceed ``1/2``? Let the people enter one at a time.

Shared $(@bind s3_D Select([365 => "birthday (365 days)", 52 => "birth week (52 weeks)", 12 => "birth month (12 months)"]; default = 365)) ``\qquad`` $(@bind s3_room Seed(0; label = "New room"))
"""

# ╔═╡ 3e775b6f-d070-32dc-247f-8560d03bc236
figure_3_4(s3_room, s3_D)

# ╔═╡ ad3d3961-5c43-3887-362d-7c59e18c9665
@bind s5_w BarEditor([1, 1, 1, 1, 1, 1]; labels = ["1", "2", "3", "4", "5", "6"], ymax = 5, width = 420)

# ╔═╡ 745d2791-e877-7788-c1a9-de93a435c2c0
s5_die = compute_s5_die(s5_w);

# ╔═╡ 4d7e011d-aa56-e523-adcf-dd5023002587
figure_5_1(s5_die, s5_x, s5_q)

# ╔═╡ c1f6512a-9cdd-f49a-6755-3eccd42374fb
md"""
A CDF takes values between 0 and 1, is non-decreasing, and increases from 0 (far left) to 1 (far right). Probabilities of ranges: ``P(X > x) = 1 - F_X(x)`` and ``P(a < X \le b) = F_X(b) - F_X(a)``. The inverse question, which value is not exceeded with probability ``q``, is answered by the **``q``-quantile**, the smallest ``x`` with ``F_X(x) \ge q``. The ``0.5``-quantile is the **median**.

### 5.2 Continuous random variables: densities
A random variable that can take any value in an interval, such as a waiting time, is called **continuous**. For such a variable, every single value has probability zero, ``P(X = x) = 0``, and only intervals have positive probability. Figure 5.2 sorts simulated waiting times into bins and shows the histogram in three ways. Reduce the bin width and observe which version converges.

Bin width (minutes) $(@bind s5_bw Select([10.0, 5.0, 2.0, 1.0, 0.5, 0.2]; default = 5.0)) ``\qquad`` number of waiting times $(@bind s5_ng Select([200, 5000, 100_000]; default = 5000)) ``\qquad`` $(@bind sd01_3 Seed(0))
"""

# ╔═╡ 81a05c35-293a-66f3-e433-9ada9ed86767
s5_gaps = compute_s5_gaps(sd01_3, s5_ng);

# ╔═╡ 0c33f013-ed16-3d70-ab63-b975e823af49
figure_5_2(s5_bw, s5_gaps)

# ╔═╡ 5f484f1d-f62f-b71d-08fc-88f49195ef32
md"""
### 6.3 From data to distribution

The mean and variance of a *distribution* are fixed numbers. The mean and variance of a *sample* are random: they vary from sample to sample and approach the values of the distribution as the sample size increases. The figure rolls the loaded die of Section 5.1; changing its bars changes the experiment.

Rolls $(@bind s6_n Select([100, 400, 2000]; default = 400)) ``\qquad`` $(@bind s6_roll Seed(0; label = "Roll again"))
"""

# ╔═╡ 9ed2e8a0-4dc7-4343-698d-d7afc11fb90c
figure_6_3(s6_roll, s5_die, s6_n)

# ╔═╡ c852a524-4a6d-a701-ddd0-a952055a5b5d
md"""
### 6.4 Linear transformations

A change of units is a linear transformation ``Y = aX + b``, for example from degrees Celsius to degrees Fahrenheit with ``a = 1.8`` and ``b = 32``. Its effect on mean and variance is

```math
E[aX + b] = a\,E[X] + b, \qquad \mathrm{Var}(aX + b) = a^2\, \mathrm{Var}(X), \qquad \sigma_{aX+b} = |a|\,\sigma_X .
```

Because, the mean is a weighted average of the values, so it is transformed exactly like each value: multiplied by ``a`` and shifted by ``b``. The variance is the mean squared **deviation from the mean**. The shift ``b`` moves each value and the mean by the same amount, so the deviations ``(aX + b) - (aE[X] + b) = a(X - E[X])`` do not contain ``b`` (got cancelled). The factor ``a`` multiplies every deviation, and since the deviations are squared, the variance is multiplied by ``a^2``. The standard deviation, the square root, is multiplied by ``|a|``; a negative ``a`` mirrors the distribution, but a spread cannot be negative. In short, ``b`` changes the location but not the spread, and ``a`` changes both.

``a`` $(@bind s6_a Slider([-2:0.1:-0.1; 0.1:0.1:3]; default = 1.8, show_value = true)) ``\qquad`` ``b`` $(@bind s6_b Slider(-40:2:40; default = 32, show_value = true)) ``\qquad`` $(@bind sd01_4 Seed(0))
"""

# ╔═╡ 2321bf60-f343-55c0-d6f8-8656ba45225e
figure_6_4(sd01_4, s6_a, s6_b)

# ╔═╡ d3148875-30e4-14f9-9c0c-a3bf25834cdf
md"""
The Gibbs sampler samples from a joint distribution by drawing repeatedly from such conditional distributions.

### Covariance and the variance of a sum

Expectations always add: ``E[X + Y] = E[X] + E[Y]``. Variances add only if the variables are uncorrelated. The joint variation is measured by the **covariance**,

```math
\mathrm{Cov}(X, Y) = E\big[(X - \mu_X)(Y - \mu_Y)\big] = E[XY] - E[X]E[Y],
```

and the **correlation** ``\rho = \mathrm{Cov}(X, Y)/(\sigma_X \sigma_Y) \in [-1, 1]``. Then

```math
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y).
```

Correlation ``\rho`` $(@bind s7_rho Slider(-0.95:0.05:0.95; default = 0.6, show_value = true)) ``\qquad`` standard deviation of ``Y`` $(@bind s7_sy Slider(0.5:0.25:2; default = 1, show_value = true)) ``\qquad`` $(@bind s7_new Seed(0; label = "New pairs"))
"""

# ╔═╡ 40984a25-56d7-6be6-dccc-8c608f74f872
figure_7_2(s7_new, s7_rho, s7_sy)

# ╔═╡ 2710f376-dccc-51c7-c151-1b0f63989cbb
md"""
### 8.3 Exponential distribution

The Poisson distribution counts how many calls arrive in a fixed period. We can also ask the reverse question: how long do we wait for the next call? Let ``T`` be the time from now until the next call arrives, measured in hours, when calls arrive at an average rate of ``\lambda`` per hour. Unlike a count, ``T`` can take any positive real value, so it is a continuous random variable.

The key observation is that the both questions are linked. The waiting time ``T`` is longer than ``t`` exactly when no call arrives in the interval ``[0, t]``. The number of calls in this interval follows a Poisson distribution with mean ``\lambda t``, and the probability of zero calls is ``e^{-\lambda t}``. Hence

```math
P(T > t) = e^{-\lambda t}, \qquad t \ge 0 .
```

Use the slider to change the rate, and the button to simulate a new day of calls. Observe how the gaps between consecutive calls change with ``\lambda``.

Calls per hour $(@bind s8_rate Slider(1:0.5:12; default = 3, show_value = true)) ``\qquad`` $(@bind s8_arr Seed(0; label = "New day"))
"""

# ╔═╡ d6175989-e5a1-c5a6-aaff-8b9fa976f80f
figure_8_4(s8_arr, s8_rate)

# ╔═╡ 931d6e51-bf53-8378-215d-6eb253e22d28
md"""
From ``P(T > t)`` we obtain the distribution function and, by differentiation, the density:

```math
F_T(t) = 1 - e^{-\lambda t}, \qquad f_T(t) = \lambda e^{-\lambda t}, \qquad t \ge 0 .
```

A random variable with this density is said to have the **exponential distribution** with rate ``\lambda``. Its mean and variance are

```math
E[T] = \frac{1}{\lambda}, \qquad \mathrm{Var}(T) = \frac{1}{\lambda^2} .
```

The mean has a direct interpretation. If calls arrive at ``\lambda = 3`` per hour, the average wait for the next call is ``1/3`` hour, that is, 20 minutes. A higher rate means shorter waits, and the standard deviation ``1/\lambda`` equals the mean, so the waiting times are widely spread: very short gaps and occasional long ones are both common.

**Memorylessness.** The exponential distribution has a notable property: it has no memory. Suppose we have already waited ``s`` without a call. Then the remaining wait has the same distribution as the wait at the start:

```math
P(T > s + t \mid T > s) = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(T > t) .
```

This follows from the independence of the calls: the past gives no information about the future. Figure 8.4 tests this. It takes many simulated waiting times, keeps only those that lasted longer than ``s`` minutes, and plots the time that remained after minute ``s``, that is, ``T - s``. The histogram has the same shape for every ``s``, and it is the same as the distribution of the original waiting time. Move the slider and check that the shape does not change. Only the number of waiting times that survive to minute ``s`` becomes smaller, so the histogram becomes noisier.

The exponential distribution is used for waiting times between events that occur independently at a constant rate, such as calls, the lifetime of a component that does not wear out, or the time between radioactive decays.

Time already waited ``s`` (minutes) $(@bind s8_s Slider(0:5:60; default = 40, show_value = true)) ``\qquad`` $(@bind sd01_5 Seed(0))
"""

# ╔═╡ 4b2e63d6-8514-80bc-aacf-2071b061ff12
figure_8_5(sd01_5, s8_s)

# ╔═╡ 62cc3839-fa91-c85f-2cb3-81ec40192e1c
md"""
### 8.4 Normal distribution

Many quantities result from the combination of many small, independent factors. A measured length is the true length plus many small errors from the instrument, the temperature, and the observer. The height of a person results from many genetic and environmental effects, each of which contributes only a little. What does the distribution of such a total look like?

Figure 8.5 shows this. It adds ``k`` independent effects, each with the same distribution, and shows the distribution of the sum. The sum is standardised, that is, shifted to mean ``0`` and rescaled to standard deviation ``1``, so that the shape can be compared for different ``k``. For ``k = 1`` we see the distribution of a single effect. Increase ``k`` and observe how the shape changes. Then change the distribution of the effects and check whether the result depends on it.

Number of effects ``k`` $(@bind s8_k Select([1, 2, 3, 5, 10, 30, 100]; default = 2)) ``\qquad`` distribution of each effect $(@bind s8_eff Select(["uniform on [0, 1]", "exponential (skewed)", "coin: 0 or 1", "die: 1 to 6"]; default = "uniform on [0, 1]")) ``\qquad`` $(@bind sd01_6 Seed(0))
"""

# ╔═╡ 6c40671c-7e87-bdb9-90e6-0161375d762f
figure_8_6(sd01_6, s8_k, s8_eff)

# ╔═╡ 52aaae87-ac22-f145-6295-4b08e672a21a
md"""
### 8.5 Scale-free distributions

Human body heights are concentrated around 170 cm, and nobody is ten times taller than the average person. Many other quantities behave very differently. The populations of towns, the sizes of files, the strengths of earthquakes, the numbers of followers of social media accounts. They range over many orders of magnitude, and extremely large values are not rare. What results in such distributions?

A common explanation is growth in proportion to the current size. A large town gains or loses more inhabitants per year than a small one, so changes are percentages instead of fixed amounts. Such effects multiply instead of adding up, and the central limit theorem of Section 8.4, which deals with sums, no longer applies.

Figure 8.6 compares the two mechanisms. It follows 400 towns that start with the same size for 80 years. In the **additive model**, each town gains or loses a random *number* of inhabitants per year. In the **multiplicative model**, it gains or loses a random *percentage*. Change the size of the yearly fluctuation in each model and compare the resulting distributions of town sizes. 

Yearly change, additive model $(@bind s8_sa Slider(50:50:500; default = 250, show_value = true)) inhabitants ``\qquad`` multiplicative model $(@bind s8_sm Slider(0.05:0.01:0.3; default = 0.18, show_value = true)) ``\qquad`` $(@bind s8_tw Seed(0; label = "New towns"))
"""

# ╔═╡ 51c5c75b-15fe-3e56-d901-1717d38185aa
figure_8_7(s8_tw, s8_sa, s8_sm)

# ╔═╡ e5247aee-387e-bd1d-9820-6cd66cb018a6
md"""
The additive model gives a symmetric bell shape: the size of a town is its initial size plus a sum of many small independent changes, which is approximately normal. The multiplicative model gives a strongly skewed distribution with a long right tail, in which a few towns become many times larger than the typical one.

The normal and the exponential distribution have a **characteristic scale** (``\sigma`` and ``1/\lambda``). A **scale-free** distribution has no such scale: rescaling the variable by any factor ``c > 0`` changes the density only by a constant factor, ``f(c\,x) = g(c)\,f(x)``. The densities with this property are the power laws

```math
f(x) \propto x^{-\alpha}, \qquad x \ge x_{\min}.
```

The log-normal distribution is only approximately scale-free, over a limited range of scales. 

Figure 8.7 shows the same data in three windows. For a power law, the three panels have the same shape, so the picture does not reveal the scale. For an exponential distribution, the shape changes from window to window, because its mean sets a scale.

Distribution $(@bind s8_zd Select(["power law", "exponential", "log-normal"]; default = "power law")) ``\qquad`` $(@bind sd01_8 Seed(0))
"""

# ╔═╡ 56c895c3-0cde-27a9-eac3-7d6fabe42ff3
figure_8_10(s8_zd, sd01_8)

# ╔═╡ 65f31e20-fc4c-a93e-50b7-c30a7ef64deb
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
</style>"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AbstractPlutoDingetjes = "6e696c72-6542-2067-7265-42206c756150"
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
project_hash = "e8b5d387a25c5b86869c82b5e1e49370be4d0a19"

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
# ╟─026886f6-938e-eee9-cc02-d3a2a0f87b0b
# ╟─1568e639-5a1d-0073-eafa-1e935a26a6ee
# ╟─08130b13-b9b0-2a1b-956f-700872faa63d
# ╟─006f021e-087e-40cd-e940-f311ddb26ca4
# ╟─46f07831-286d-701a-b204-5b281db99bd6
# ╟─b7fedc90-a101-262c-d46c-a204480eed7b
# ╟─eeadeb56-cc29-a153-6de8-d81cdba978cd
# ╟─a170994f-dee4-3124-f06b-45d89ad2041b
# ╟─8bf0688d-c883-af15-104b-524a9d32ce5b
# ╟─9a651b5f-214c-5a92-07bb-4c01d8f96244
# ╟─7303ec21-89b8-7a3f-495c-9a33cb8d75e1
# ╟─f72df3d9-38db-37ba-4d42-c02cb9038fc7
# ╟─34dcce91-1771-b5c9-d0da-0ed7c71d2cb2
# ╟─ae5a928d-6312-97a6-47c8-38004d91b5a6
# ╟─a5579cb0-86f7-3ab6-bfb8-0391685088f8
# ╟─42928f2b-b87a-081f-c4b1-ca697ac06ae4
# ╟─9f77330a-df6f-702c-ae64-9c0a1e5ca609
# ╟─88bace24-5b37-7d3d-3bf0-a710d0e27b1d
# ╟─fc198618-db34-ee07-ad5b-aea65703ad06
# ╟─1dd0247f-65c1-fbbb-3ff7-b77a146d2cbf
# ╟─f7846c5a-e442-c0e6-3a2c-3af2a2717bf8
# ╟─37f62532-951e-0161-8fa1-c8192e6e4c26
# ╟─0e99d99e-00b4-9ebd-4e02-820bbc7466cf
# ╟─9c9fc6f6-caaa-dcd1-581f-74f00d772e30
# ╟─5a1368ba-5f15-b685-223d-3e9facb62b66
# ╟─8bd196fa-1329-e24c-9625-088a0796155b
# ╟─b968816c-a225-d26f-f5f2-d37574af771c
# ╟─eea37975-9707-9623-4d93-982dd172a2c3
# ╟─2cc4fbec-c384-b84d-71ac-933fa13cf56d
# ╟─d3572545-ac4d-1bd2-ec48-61cccabf7f2c
# ╟─bac78fcd-f484-a235-8cf9-7ba8516939d9
# ╟─7720a263-fb68-581b-c5e6-e47949db7eb9
# ╟─9941d0df-e52f-5d35-4a99-74bcd848fbc1
# ╟─814b0094-53a9-137f-aaf1-c3cea64dbdc9
# ╟─a442dcab-e71a-9da8-2a8c-30fa1ba0589e
# ╟─7e2d1664-bbd8-5829-6739-c16fadca983d
# ╟─c33bb7f0-a34b-1c4c-ac63-b482f6fc113f
# ╟─63562cee-ee88-ed3e-620d-a3c2c9386d1d
# ╟─f8bc7c83-9cfb-b9df-1b35-03e854f60aa5
# ╟─3e775b6f-d070-32dc-247f-8560d03bc236
# ╟─e95ac4e1-1538-23f7-f1c1-97a1633599ac
# ╟─c2f97557-3d83-8d6d-7532-a6de6ccca995
# ╟─0cb406b1-397b-0f4c-2356-c93f0a1df1a5
# ╟─955a0e99-07ab-85bf-2d11-01b1e3fb7103
# ╟─863281d2-2bc7-6ff5-0d19-adfafa2a7dab
# ╟─ad3d3961-5c43-3887-362d-7c59e18c9665
# ╟─0866ac05-80f0-1ca3-a833-38680d1e54b8
# ╟─745d2791-e877-7788-c1a9-de93a435c2c0
# ╟─4d7e011d-aa56-e523-adcf-dd5023002587
# ╟─c1f6512a-9cdd-f49a-6755-3eccd42374fb
# ╟─81a05c35-293a-66f3-e433-9ada9ed86767
# ╟─0c33f013-ed16-3d70-ab63-b975e823af49
# ╟─21e91b1b-8f9a-5261-a343-c6647f23c5d7
# ╟─97e88008-96ed-5038-df09-4db0a2e1b27c
# ╟─a5e29ac8-8b21-e8f1-521f-3aca135df3ca
# ╟─7ee8367e-9c26-a992-aaef-49578ecbbdcd
# ╟─dfa4c187-1ca8-d1ee-7d64-36b95ba8b9db
# ╟─e201e3f9-249d-602d-e8d1-d83ff1a4ae3e
# ╟─eda8da94-ec21-1142-4232-8757dc73195b
# ╟─f42faca5-ab5b-edb7-7f22-56607ec13c66
# ╟─e089f15e-f8e2-1505-1ad0-488b06ae9cc1
# ╟─b8300184-a115-f7c8-c3fb-291cabcd93fe
# ╟─b6f9b7ca-ec9b-9a8b-0883-ac95b6a25b01
# ╟─d87d14dd-b78c-3750-0687-c18fb12ea648
# ╟─5f484f1d-f62f-b71d-08fc-88f49195ef32
# ╟─9ed2e8a0-4dc7-4343-698d-d7afc11fb90c
# ╟─c852a524-4a6d-a701-ddd0-a952055a5b5d
# ╟─2321bf60-f343-55c0-d6f8-8656ba45225e
# ╟─ebdad859-4c9c-0919-a468-593d9bbd3bad
# ╟─a9372474-8601-2306-eb74-6b6760461817
# ╟─8184cf8c-4033-0f80-3230-0a4e96227082
# ╟─92dfee96-d49d-1c20-0507-50314e9d547c
# ╟─3dc905eb-90b3-cbc8-ffe4-3074c926ca49
# ╟─bded43a7-ebd9-f3aa-1606-d3343694b59b
# ╟─d3148875-30e4-14f9-9c0c-a3bf25834cdf
# ╟─40984a25-56d7-6be6-dccc-8c608f74f872
# ╟─bbcecfa8-ff8f-f470-3194-48e9f6966722
# ╟─f326e49a-dfcb-8f65-697a-81ca2dd93311
# ╟─f74f5c16-e323-9d55-8b1e-b5f7a7cfb7ad
# ╟─8ae7349a-e9bc-1721-2d25-f6bbc27db6fa
# ╟─83778b1e-6f8b-7145-f7f2-48f42467e36e
# ╟─07207311-dadf-fa61-94ec-2dd8626b33a9
# ╟─2ed74b5d-98cd-b360-2181-16950f65e07c
# ╟─f8289551-d765-b681-52c3-11f487af1f0a
# ╟─55c537f7-a73f-6e6c-6e09-67d95814ad1d
# ╟─db787f2d-3129-c9f5-8412-513bb1fd558a
# ╟─94c3aadf-28c8-51db-a8ea-9496913dbe8d
# ╟─2710f376-dccc-51c7-c151-1b0f63989cbb
# ╟─d6175989-e5a1-c5a6-aaff-8b9fa976f80f
# ╟─931d6e51-bf53-8378-215d-6eb253e22d28
# ╟─4b2e63d6-8514-80bc-aacf-2071b061ff12
# ╟─636bcacc-2d74-83b7-2fcb-61d831dc48a2
# ╟─cd7c651f-63ac-391c-50ac-1353c65aaa3c
# ╟─62cc3839-fa91-c85f-2cb3-81ec40192e1c
# ╟─6c40671c-7e87-bdb9-90e6-0161375d762f
# ╟─6596d7fa-bfb7-d365-316c-fbd07644a72a
# ╟─72335459-d175-30f5-6101-758984005c2e
# ╟─52aaae87-ac22-f145-6295-4b08e672a21a
# ╟─51c5c75b-15fe-3e56-d901-1717d38185aa
# ╟─e5247aee-387e-bd1d-9820-6cd66cb018a6
# ╟─56c895c3-0cde-27a9-eac3-7d6fabe42ff3
# ╟─a97689aa-309c-a508-c44e-4df55f2e793a
# ╟─90fe57dd-f17b-bad0-b419-caba2185e6ab
# ╟─02ec734f-1c74-01f5-0d4c-821475fd87b8
# ╟─703cfafb-cedd-df0e-9b19-e53c36c5f728
# ╟─04d2b444-081a-4087-b6dd-4b923b70213a
# ╟─9dfb719a-f728-2979-3298-ccb19b7fde3c
# ╟─5526c079-4650-74ef-8bd1-2db943843d48
# ╟─7f8cd34d-982e-775d-da30-01956444611d
# ╟─04c2e89a-0b93-f00b-cae9-b1319dc64958
# ╟─b0788314-9702-bdc3-a02f-b19c69dc62e2
# ╟─486619e5-4bc3-8708-19a0-cacaf3d026f6
# ╟─1bfd04a5-4240-d653-ab9b-cd710024917b
# ╟─a7796982-5923-905a-140e-9e79aa3d65cb
# ╟─495f4e1a-279b-264a-f204-c246cccce481
# ╟─ee550866-c25c-abfa-268f-4eff9ef34e87
# ╟─e617cbea-70a6-998e-7c50-39228342a128
# ╟─a6f81dee-9307-de84-1014-084ff001980a
# ╠═4903d880-e38e-5532-b10c-e09010ea5c04
# ╟─e2e171ad-0331-00d9-da7c-d0f4cc166d2c
# ╟─bdb8f72d-117c-e5ca-3a52-6b4f41181f63
# ╟─1a33514e-bd6d-bc9d-894a-439623a56699
# ╠═f62efebd-11a0-0262-22b0-ea39bc2c48d9
# ╟─b1a64b16-e722-9012-073e-ff05a0615c5f
# ╟─85dc7ada-c6f5-47b1-fdf6-70e53d467a5b
# ╟─6171131b-4213-62e2-1d02-82fc4b22087a
# ╠═94a8c162-9409-ee58-2f1d-dc4aac34b133
# ╟─941cdcee-df7c-a45d-1592-d2802c8739d8
# ╟─7147ee06-1085-ae4b-a8da-ad633a402b73
# ╟─c8663cd6-de2e-828f-415a-f558a3f4c86e
# ╠═71e4ee82-f666-e018-0680-f30b9a5c0dc2
# ╟─9f81f1ff-233a-ed94-d1aa-c02a369d607c
# ╟─84c27ad5-7d39-5fb5-5b1d-1432fef699b8
# ╟─c7ffbfab-6ffd-8da9-ad67-d85bb262ac46
# ╠═bc4a2f8f-ed58-f19e-7e31-a7f50b1512c7
# ╟─2febad3a-cd42-3540-d23c-341e2e7e00e9
# ╟─9c1000fb-901d-7cd5-c36c-278fa306809d
# ╟─1738ff3e-0692-eb6b-1ea4-7f461e99d8e6
# ╟─7a7f1404-b2f9-cb94-8a1f-f3ff0a9a9450
# ╟─25d01d86-4470-d901-7c7f-2f5143870d2f
# ╟─99564cdb-737c-f1be-0c75-2c0d6b0e1dff
# ╟─3ad6f6c2-5d1d-58e1-408c-bc3006ae1b4b
# ╟─340cdc26-d96f-0212-edfc-f5b65ae22efb
# ╟─95604cc8-0e65-2aef-fa54-84f6f2a613ee
# ╟─5721bf8d-183a-2e54-3234-1272b4c392dd
# ╟─4ffa8386-28e9-519f-d266-af124bd86b05
# ╟─a51b76ee-b700-52d3-8eb9-e01c001b496e
# ╟─b96e7f21-a157-96fb-baea-4b9ab8657284
# ╟─926b8e8a-bec6-9a47-6997-236c0c4de539
# ╟─2732e001-4996-f98a-ec98-bae0a1b086a9
# ╟─a86a9f28-4268-6a70-3602-19f92906f0bc
# ╟─643c3c88-8139-195c-2b0b-e817d9d7c3b0
# ╟─117bdd63-7da7-0d16-32bc-9a1f4e270dda
# ╟─65f31e20-fc4c-a93e-50b7-c30a7ef64deb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
