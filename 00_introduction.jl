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

# ╔═╡ a3e07761-83cc-ccd0-23d7-1bf78ced6ca2
# packages
begin
	using PlutoUI, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions, Base64
end

# ╔═╡ f1c82cb2-1756-b057-4a10-2eb3b0fc6686
md"""
# 0 · Introduction to Stochastic Modeling and Simulation

*Stochastic Modeling and Simulation*
"""

# ╔═╡ 6193d783-e07f-1669-7e17-28aeeb5e7982
md"""
## 1. Randomness, models and simulation

Many processes in nature can be predicted precisely. With a few laws and a known initial state, we can calculate the position of a planet next year, the time a stone takes to fall from a tower, or the current in an electrical circuit. Other processes cannot be predicted with this **certainty**, for example, whether tossing a coin give heads, when the next customer enters a shop, which molecule in a cell reacts next, or how much will it snow in Dresden next Tuesday. Such processes are called **random** or **stochastic**.

To describe randomness clearly, we need a few terms. A random process (or *experiment*) has several possible results, called **outcomes**. The set of all outcomes is the **sample space**, e.g. {heads, tails} for a coin. An **event** is a set of outcomes we are interested in, e.g. "an even number" = {2, 4, 6} when rolling a die. A **probability** is a number between 0 (impossible) and 1 (certain) that measures how likely an event is. A **probability distribution** gives a probability to every possible outcome, and these probabilities sum to 1. A **random variable** is a quantity determined by the outcome, such as the number shown by a die. A **realisation** is one result, such as the sequence H, T, T, H from four tosses.

Randomness has different causes. Sometimes the laws are deterministic, but the outcome depends so sensitively on details we cannot measure, such as the exact speed and spin of a coin, that it is unpredictable in practice. Sometimes a system has very many small parts that interact with each other, such as traffic in a city or molecules in a gas, and nobody can follow all of them. Some processes, such as radioactive decay, are random in principle, as far as physics knows today.

Although single outcomes are unpredictable, random processes still follow some patterns/distributions. A fair coin, for example, shows heads in about half of a large number of tosses. **Probability theory** is the branch of mathematics that describes such randomness by assigning probabilities to outcomes and studying their patterns. Randomness appears in almost all field of science and engineering:

- **physics and chemistry:** diffusion, radioactive decay, chemical reactions between small numbers of molecules;
- **biology and medicine:** mutations, gene expression in single cells, epidemics;
- **engineering:** noise in signals, the reliability of machines, queues in networks;
- **economics:** prices, insurance claims, the demand for a product;
- **data science and machine learning:** random sampling of data, random initialisation of neural networks, stochastic gradient descent, generative models.

A **deterministic model** predicts exactly which state a system will be in, whereas a **stochastic model** predicts only the **probabilities** of states. A weather model, for example, does not say "it will rain tomorrow" but "the probability of rain is 60 %." A **stochastic simulation** then generates many **realisations** of the model on a computer, from which we can estimate quantities that we cannot calculate by hand.

## 2. Intuition about randomness

Random choices are often required in real world situations: a survey selects participants at random, a password should be unpredictable, a clinical trial assigns patients to treatments at random. It is tempting to make such choices "by hand". Before we study random processes, we therefore see how well a person can produce randomness. 

!!! game "Experiment · a sequence of coin tosses"
    Write down a sequence of coin tosses **without using a coin**, as a string of the letters H (heads) and T (tails). Try to make the sequence as random as possible. Any length between 20 and 400 letters is accepted; 50 to 100 letters give a meaningful result. Other characters are ignored. **Can a single sequence be "more random" than another?**

    Your sequence: $(@bind s0_seq TextField((70, 3); default = "HTHHTHTTHTHHTHTTHTHTHHTTHTHTHHTHTTHTHHTHTHTTHTHHTH"))
"""

# ╔═╡ af5fe0f4-e492-bcb5-d1ac-3f04d91b9ddc
md"""
But how can a sequence of letters be judged? Every sequence of ``n`` tosses has the same probability ``2^{-n}`` for a fair coin, so no single sequence is "more random" than another. However, what can be judged are **summary statistics** of the sequence. We used two of them: the number of switches between H and T, and the length of the longest run of same letters. For each statistic, the computer simulates 20 000 sequences of the same length with a fair coin and draws the distribution of the statistic. A sequence that lies far in the tail of this distribution is unlikely to have been produced by a coin.

Sequences written by people usually switch too often and avoid long runs. A continuous five or six equal letters appears regular to us and is avoided, although a fair coin produces such runs in most sequences of 50 tosses. This effect is well documented in psychology (Wagenaar 1972; Bar-Hillel and Wagenaar 1991), and it is used to detect fabricated data.

The experiment shows following points:

1. **Randomness is a property of the process, not of a single outcome.** We can only ask whether an outcome is *plausible* under a given random process!
2. **Intuition about randomness is unreliable.**
3. **Simulation gives a reference.** We did not compute the distribution of the longest run by hand; we simulated many realisations of the model "fair coin" and compared. This is the basic method of stochastic simulation.
"""

# ╔═╡ 241e69c2-fe6d-130a-29c2-5629bdcb90bf
md"""
!!! think "Think"
    In the left panel of Figure 2.1, the simulated numbers of switches form a symmetric, bell-shaped histogram centred at about half the sequence length. Why is the centre at ``(n-1)/2``? And why is the histogram of the longest run in the right panel not symmetric?
"""

# ╔═╡ 78a5c5f0-4f7e-4b7f-b86e-0a92b7786012
md"""
In the stochastic model, customers sometimes arrive in quick succession. The clerk cannot serve them all at once, and queue increases. At other times when nobody comes, the clerk is idle. Idle time cannot be saved and used later, so the waiting caused by the busy periods is never compensated by the free periods. The closer the load is to 1, the less free time is available to clear a line, and the average line grows rapidly (right panel).

The deterministic model is correct about the averages, but it does not describe the waiting at all. This is the general reason for stochastic models: whenever **fluctuations** around the average influence the quantity of interest, a model of averages is insufficient. Examples are waiting lines in traffic and computer networks, the extinction or outbreak of an epidemic that starts with a few cases, and chemical reactions in a cell in which only a few molecules of a kind are present.
"""

# ╔═╡ 1b5b4b1a-4f65-b2d2-dfab-638f119d67fc
md"""
!!! think "Think"
    A deterministic model of the counter (every customer exactly on schedule) predicts that nobody waits at a load of 0.9. Under which conditions would you trust a deterministic model of this kind, and when would you insist on a stochastic one?
"""

# ╔═╡ 2e2cb45a-21e2-a2af-edea-c4659f9aa114
md"""
Two things to notice. First, the shape of a distribution follows from the **mechanism** that produces the values: a sum of many small independent effects produces a bell-shaped (normal) distribution, a count of rare events produces a Poisson distribution, a waiting time without memory produces an exponential distribution, and growth processes in which effects multiply produce heavy-tailed distributions. Second, not every distribution has a "typical" value. For body heights the mean is a useful summary; for the sizes of cities it is not, because a few very large cities dominate it.

!!! think "Think"
    You receive 5000 measured values without any information about their origin, and you are asked to estimate the probability that a new value will be **extremely large**, that is, larger than almost all the values you have measured. Why is it risky to assume a normal distribution?
"""

# ╔═╡ 9c38abe1-e7e9-37c0-c159-36d93a15dca4
md"""
For small ``n`` the averages scatter widely; for large ``n`` they concentrate around 3.5. This is the **law of large numbers**, the reason why stochastic simulation works at all (see the notebook on Monte Carlo integration). The width of the shaded band decreases like ``1/\sqrt{n}``. 

To see why, we need a number that measures how much a random quantity **varies** around its mean. This is the **variance**: the average of the squared distances from the mean. For ``n`` values ``x_1, \dots, x_n`` with mean ``\bar x``, it is estimated by

```math
s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar x)^2 .
```

Each term ``x_i - \bar x`` is the distance of one value from the mean. We square it so that positive and negative distances do not cancel, add everything up and divide by ``n-1``. (We divide by ``n-1`` instead of ``n`` because the mean ``\bar x`` was itself computed from the same data. For large ``n`` the difference is negligible.) The square root ``s`` is the **standard deviation**. It has the same unit as the data and tells us how far a value lies from the mean.

For a die, we know the exact mean ``\mu = 3.5`` and all six faces are equally likely, so we can average over the faces directly:

```math
\sigma^2 = \frac{(1-3.5)^2 + (2-3.5)^2 + \dots + (6-3.5)^2}{6} = \frac{35}{12} \approx 2.9, \qquad \sigma \approx 1.7 .
```

Two simple rules:

1. For independent rolls, **variances add**: the variance of the sum of two rolls is ``\sigma^2 + \sigma^2``.
2. Dividing by a number ``n`` divides the variance by ``n^2``.

The sum of ``n`` rolls therefore has variance ``n\sigma^2``, and the average (the sum divided by ``n``) has variance

```math
\mathrm{Var}\Big(\frac{1}{n}\sum_{i=1}^n X_i\Big) = \frac{1}{n^2} \sum_{i=1}^n \mathrm{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n},
\qquad
\text{standard deviation} = \frac{\sigma}{\sqrt n}.
```

The **typical error** of an average of ``n`` independent samples is therefore proportional to ``1/\sqrt n``. To reduce the error by a factor of 10, one needs 100 times as many samples. Every simulation result should therefore be reported together with an estimate of its error.
"""

# ╔═╡ 46107205-57a4-76cf-314d-440c6e968207
md"""
!!! think "Think"
    A colleague simulates the above counter model once and reports a mean waiting time of 3.8 minutes. A second run with another seed gives 5.1 minutes. Which value is correct, and what should be reported?
"""

# ╔═╡ ac8a5206-b005-db56-4160-e8061ec2a977
md"""
## 7. Random numbers on a deterministic machine

A computer executes its program exactly. How can it produce random numbers? The answer is that it does not. A **pseudo-random number generator** is a deterministic recipe that turns a starting value, the **seed**, into a long sequence of numbers that pass statistical tests of randomness. A consequence is that the same seed always produces the same sequence. Run the cell below twice, then change the seed (42).
"""

# ╔═╡ 35a3d731-d1ce-b451-27eb-960fb19082b0
rand(MersenneTwister(42), 1:6, 10)

# ╔═╡ 75d055de-b272-66cb-d589-17dd51321fed
md"""
This property makes simulations **reproducible**: another person can repeat a simulation and obtain exactly the same numbers. In machine learning, the seed determines the random split into training and test data, the initial weights of a network and the order of the training examples. 

## 8. Julia and Pluto

These notebooks are **Pluto notebooks** in the programming language **Julia**. Julia code is close to mathematical notation and fast enough for large simulations.

**Running a notebook.** Install Julia from [julialang.org](https://julialang.org/downloads/), start it, and type `import Pluto; Pluto.run()`. A browser window opens in which the file can be opened. The required packages are installed automatically the first time; this takes a few minutes.

**Reactivity.** A notebook consists of cells. When a cell is changed, all cells that depend on it are executed again automatically. Change `n` below and observe the sentence.
"""

# ╔═╡ 617dc83e-3505-235e-7d62-f8737a779b17
n = 11

# ╔═╡ dee8c1e2-2c42-e34d-fad9-42417702f3eb
md"The first ``n = `` $(n) odd numbers add up to $(sum(1:2:2n - 1)), which equals ``n^2``."

# ╔═╡ c2c53e4e-ce74-a24f-595e-c74bb526dd5c
md"""

| purpose | Julia | remark |
|:--|:--|:--|
| vector | `x = [3, 1, 4]` | indices start at **1**: `x[1]` is 3, `x[end]` is 4 |
| range | `1:6`, `0:0.1:1` | |
| element-wise operation | `x .^ 2`, `exp.(x)` |  |
| function | `f(x) = x^2 + 1` | or `function f(x) … end` for longer functions |
| loop | `for k in 1:10 … end` | loops are fast in Julia |
| random numbers | `rand()`, `rand(1:6)`, `randn()` | uniform on ``[0,1)``, a die, a standard normal variable |
| reproducibility | `rng = MersenneTwister(1); rand(rng)` | same seed, same numbers |
| summaries | `sum`, `mean`, `var`, `count(==(6), x)` | |
| plots | `histplot(x)`, `lineplot(x, y)` |  |

The following cell can be edited freely.
"""

# ╔═╡ 51ef3675-db22-42b7-5dda-870d228da0cd
md"""
With a fair coin, HH requires about 6 tosses on average and HT about 4. But doesn't both have equal probability of 1/4?
The difference becomes clear when it is drawn as a diagram in which each circle is a state and each arrow is one toss.
"""

# ╔═╡ 3341557a-d516-9b57-3a71-a1c5c4fa804f
md"""
For HH, if we get a tail after one head, we start again. For HT, if we get another head after one head, we can keep the head as the start of a new sequence. These diagrams can be used to get the exact answer. Let ``E_s`` be the expected number of remaining tosses in the state *start* and ``E_h`` in the state *H*. One toss is always made, and each arrow is followed with probability ``\tfrac12``. For HH:

```math
E_s = 1 + \tfrac12 E_h + \tfrac12 E_s, \qquad E_h = 1 + \tfrac12 \cdot 0 + \tfrac12 E_s .
```

Solving gives ``E_h = 4`` and ``E_s = 6``. This method, conditioning on the outcome of the first step, is called **first-step analysis**, and the diagrams are **Markov chains**.
"""

# ╔═╡ b960034c-9a69-0296-24db-226d53c5a3b7
md"""
## 10. Games of chance and strategy

### 10.1 Penney's game

The waiting problem has an interesting consequence, published by Walter Penney in 1969. Two players each choose a pattern of three tosses, for example HHT and THH. A fair coin is tossed until one of the two patterns appears; the player whose pattern comes first wins. All eight patterns have the same probability ``\tfrac18`` at every position, so the game appears fair. You choose first; the computer chooses second.
"""

# ╔═╡ 712b24fb-f85e-48b4-a08a-cf043a04370b
md"""
The computer uses a fixed rule: if your pattern is ``A\,B\,C``, it chooses ``\bar B\,A\,B``, where ``\bar B`` is the opposite of your second letter. The reason is that the computer's pattern ends with the first two letters of yours. In most races, your pattern can only be completed after the computer's pattern has appeared one toss earlier.

To understand this, suppose you choose HHH and the computer chooses THH. If the first three tosses are HHH, you win. This happens with probability ``\tfrac18``.

In every other case, a T appears before the first run of three heads. The last T before this run, followed by the first two H's of the run, forms THH. Therefore, the computer wins with probability ``\tfrac78``.

The table gives the winning probabilities of the computer's reply for every choice.
"""

# ╔═╡ acdd423f-5442-a003-9c14-37b1f8f057b1
md"""
For every pattern there is another pattern that beats it. THH beats HHT, HHT beats HTT, HTT beats TTH, and TTH beats THH. There is no best pattern, and the player who chooses second has the advantage. Games with this property are called **non-transitive**; the best known example is rock–paper–scissors.

### 10.2 Rock–paper–scissors against a predictor

In rock–paper–scissors, rock beats scissors, scissors beats paper, and paper beats rock. If one player chooses each move with probability ``\tfrac13``, independently of the past, then no strategy of the other player can win more than one third of the rounds on average. Uniform random play is the optimal strategy in this game.

The computer below does not play randomly. It records which move you played after each of your previous moves, estimates the probability of your next move from these counts, and plays the move that beats the most probable one. Play at least 30 rounds.
"""

# ╔═╡ e58560b7-c56c-100a-8a62-d8a68b4a5f2a
md"""
If the computer wins clearly more than one third of the rounds, your moves were predictable from your previous move. The table of counts that the computer learns is an estimate of the **transition probabilities** of your play, the same quantity that defines a Markov chain. The button *let a random player play* replaces you by a pseudo-random generator; against it, the predictor cannot do better than one third.

!!! think "Think"
    Why can no strategy of the computer, however clever, win more than one third of the rounds against a player who chooses each move independently with probability ``\tfrac13``?
"""

# ╔═╡ f2984ab0-b30b-cca5-f920-9a6f9a252282
md"""
Two operations turn noise into a tone. **Repetition** makes the signal periodic with period ``P``. A periodic signal contains only the frequencies ``f_0 = 1/P`` and its integer multiples ``2f_0, 3f_0, \dots`` (the harmonics), and the ear perceives such a signal as a tone of pitch ``f_0``, whatever the shape of one period. **Averaging** neighbouring values is a low-pass filter. It damps high harmonics. The randomness of the initial burst sets the relative strengths of the harmonics, so every pluck sounds slightly different.

### 11.2 Mozart's musical dice game

In 1792, Nikolaus Simrock in Berlin printed a *Musikalisches Würfelspiel*, a "musical dice game" attributed to Wolfgang Amadeus Mozart. It has instructions to compose as many waltzes as one wishes with two dice, "without being musical or understanding anything of composition". For each of the 16 measures, the player rolls two dice and adds their values. The resulting sum, from 2 to 12, determines which precomposed musical passage (bar) is selected from a table. The selected passages are then combined to create a waltz. Below, the selected bars are played on the plucked string of Section 11.1.
"""

# ╔═╡ c21f605e-1ffe-5b60-bc82-7d789af525aa
md"""
Notes that occur in only some of the rāgas (Ma′ and Ni) are strong evidence: one such note rules out Bhupali and Deshkar at once. Notes shared by all four shift the belief only slightly. Listening to **pairs** of notes, that is, to the transition probabilities of a Markov chain, usually decides faster, because the rāgas differ more in how they transition than in which notes they use.

#### Part 3 · You play, the computer listens

Click the notes to play a melody of your own; the computer guesses the rāga after every note. Try to play a Deshkar phrase such as *Pa Dha Ṡa Dha Pa · Dha Ga Pa*, or a Yaman phrase such as *Ni Re Ga Ma′ Dha Ni Ṡa*.
"""

# ╔═╡ 2619603e-a21e-719f-d925-686f72a498c9
md"""
!!! think "Think"
    Play a melody that belongs to none of the four rāgas, for example a phrase of a Western song. The computer may become almost certain of one rāga. Why is a probability of 99 % not a guarantee that the answer is right?
"""

# ╔═╡ b6f153f1-0f52-ae9a-6447-d9b93b26a9ff
md"""
!!! think "Think"
    The computer becomes certain of a rāga without any notion of melody, mood or beauty. What exactly does it compute, and why does a single note sometimes change its belief completely? In what sense is its certainty misleading?
"""

# ╔═╡ 203fdbfd-57b7-2723-2a7a-8771f410ea18
md"""

## Further reading

| resource | content |
|:--|:--|
| [Seeing Theory (Brown University)](https://seeing-theory.brown.edu/) | interactive visualisations of probability and statistics |
| [J. K. Blitzstein and J. Hwang, *Introduction to Probability*](https://projects.iq.harvard.edu/stat110/home) | Textbook |
| [Julia documentation](https://docs.julialang.org/) and [Pluto](https://plutojl.org/) |  |
| [K. Karplus and A. Strong, *Computer Music Journal* 7 (1983)](https://doi.org/10.2307/3680062) | the plucked-string algorithm of Section 11.1 |
| [Musikalisches Würfelspiel, score (IMSLP)](https://imslp.org/wiki/Musikalisches_W%C3%BCrfelspiel,_K.516f_(Mozart,_Wolfgang_Amadeus)) | the 1792 print of the dice game |
| [P. Chordia and A. Rae, *Raag recognition using pitch-class and pitch-class dyad distributions*, ISMIR 2007](https://archives.ismir.net/ismir2007/paper/000431.pdf) | recognition of rāgas from note statistics |
| [tanarang.com, rāga descriptions](https://tanarang.com/) | descriptions of rāgas and characteristic phrases |

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden.
- F. Galton, *Natural Inheritance*, Macmillan, 1889.
- W. Penney, "Problem 95: Penney-ante", *Journal of Recreational Mathematics* 2 (1969) 241.
- J. von Neumann and O. Morgenstern, *Theory of Games and Economic Behavior*, Princeton University Press, 1944.
- S. A. Hedges, "Dice music in the eighteenth century", *Music & Letters* 59 (1978) 180–187.
- N. A. Jairazbhoy, *The Rāgs of North Indian Music: Their Structure and Evolution*, Faber and Faber, London, 1971.
- Bar data of the dice game: transcription of the 1792 edition from the LilyPond community wiki, "Mozart's Musikalisches Würfelspiel".
- M. Gardner, "Mathematical games: on the paradoxical situations that arise from nontransitive relations", *Scientific American* 231(4) (1974) 120–125 (J. H. Conway's formula for Penney's game, Table 10.1).
- L. J. Guibas and A. M. Odlyzko, "String overlaps, pattern matching, and nontransitive games", *Journal of Combinatorial Theory, Series A* 30 (1981) 183–208.
- K. Karplus and A. Strong, "Digital synthesis of plucked-string and drum timbres", *Computer Music Journal* 7(2) (1983) 43–55.
- P. Chordia and A. Rae, "Raag recognition using pitch-class and pitch-class dyad distributions", *Proceedings of the 8th International Conference on Music Information Retrieval (ISMIR)*, Vienna, 2007, 431–436.
- Rāga descriptions (scale, āroha, avaroha, vādī, samvādī and characteristic phrases of Bhupali, Deshkar, Shuddh Kalyan and Yaman): tanarang.com and raag-hindustani.com, and Jairazbhoy (1971). The note weights and the melody generator of Section 11.3 are a simplified synthetic model constructed for this tutorial; they are not from actual performances.
"""

# ╔═╡ 249842c4-56ec-a847-415a-6b05b433f905
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 55d4eb1e-f704-d206-8e72-27c6deae61c8
#figures
begin
	function runstats(seq::AbstractVector{Bool})
		n = length(seq); n == 0 && return (switches = 0, longest = 0)
		sw = count(i -> seq[i] != seq[i-1], 2:n)
		best = 1; cur = 1
		for i in 2:n
			cur = seq[i] == seq[i-1] ? cur + 1 : 1
			best = max(best, cur)
		end
		(switches = sw, longest = best)
	end

	parse_pattern(s) = [c == 'H' for c in uppercase(s) if c in "HT"]

	patstr(p) = join(b ? "H" : "T" for b in p)

	function pattern_automaton(pat)
		k = length(pat); nxt = zeros(Int, k, 2)
		for st in 0:k-1, (j, b) in enumerate((true, false))
			w = vcat(pat[1:st], b); m = min(length(w), k)
			while m > 0 && w[end-m+1:end] != pat[1:m]
				m -= 1
			end
			nxt[st+1, j] = m
		end
		nxt
	end

	function expected_wait(pat, p)
		k = length(pat); nxt = pattern_automaton(pat); Q = zeros(k, k)
		for st in 1:k, j in 1:2
			t = nxt[st, j]
			t < k && (Q[st, t+1] += j == 1 ? p : 1 - p)
		end
		((I - Q) \ ones(k))[1]
	end

	function wait_sim(rng, pat, p; cap = 1_000_000)
		k = length(pat); nxt = pattern_automaton(pat); st = 0; n = 0
		while st < k && n < cap
			n += 1; st = nxt[st+1, rand(rng) < p ? 1 : 2]
		end
		n
	end

	"""Single counter, service time exactly 1 minute."""
	function counter(arr::Vector{Float64})
		start = similar(arr); free = 0.0
		for (i, a) in enumerate(arr)
			start[i] = max(a, free); free = start[i] + 1.0
		end
		start
	end

	poisson_arrivals(rng, λ, T) = (t = Float64[]; s = randexp(rng) / λ; while s < T; push!(t, s); s += randexp(rng) / λ; end; t)

	function waiting_curve(arr, start, T)
		ev = sort(vcat([(a, 1) for a in arr], [(s, -1) for s in start]))
		ts = [0.0]; q = [0]; n = 0
		for (t, d) in ev
			t > T && break
			n += d; push!(ts, t); push!(q, n)
		end
		push!(ts, T); push!(q, q[end]); ts, q
	end

	"""Karplus–Strong plucked string... a burst of random numbers, repeated with period 1/f0 and averaged."""
	function pluck(f0, dur, rng; fs = 22050, decay = 0.998)
		P = max(round(Int, fs / f0), 2); n = round(Int, dur * fs)
		y = zeros(n); y[1:min(P, n)] .= 2 .* rand(rng, min(P, n)) .- 1
		for i in P+1:n
			y[i] = decay * 0.5 * (y[i-P] + (i - P - 1 >= 1 ? y[i-P-1] : 0.0))
		end
		y
	end

	"""Magnitude of the discrete Fourier transform of `y` at the frequencies `fr` (Hz)."""
	function spectrum(y, fs, fr)
		n = length(y); w = 0.5 .- 0.5 .* cos.(2π .* (0:n-1) ./ (n - 1))
		[abs(sum(w[j] * y[j] * cis(-2π * f * (j - 1) / fs) for j in 1:n)) for f in fr]
	end
end;

# ╔═╡ 665c5cbe-b1b3-5670-7cfe-c50804a43840
# Drawing figures (SVG)
begin
	const COL = (blue = "#1f5aa6", red = "#c2462e", ochre = "#b8860b", green = "#2f7d55", purple = "#6a4c93",
		grey = "#8c8c8c", sky = "#7fa8d8", ink = "#1b1b1b", light = "#d9e4f2", pink = "#e7a493", teal = "#2b6f74")

	const PALETTE = [COL.blue, COL.red, COL.ochre, COL.green, COL.purple, COL.grey, COL.teal, COL.sky]

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

	raw!(f::Fig, s::AbstractString; over = false) = (print(over ? f.over : f.body, s); f)

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

	function lineplot(x, ys...; labels = nothing, xlabel = "", ylabel = "", title = "", w = 520, h = 240, xlog = false, ylog = false)
		allv = filter(isfinite, reduce(vcat, [collect(Float64, y) for y in ys]))
		xv = collect(Float64, x)
		yr = ylog ? (minimum(filter(>(0), allv)) / 1.5, 1.5maximum(allv)) : _pad(minimum(allv), maximum(allv))
		f = fig(; w, h, xr = xlog ? (minimum(xv), maximum(xv)) : (minimum(xv), maximum(xv)), yr, xlabel, ylabel, title, xlog, ylog)
		for (i, y) in enumerate(ys)
			line!(f, xv, collect(Float64, y); c = PALETTE[mod1(i, length(PALETTE))])
		end
		labels === nothing || legend!(f, [(string(l), PALETTE[mod1(i, length(PALETTE))]) for (i, l) in enumerate(labels)])
		f
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

# ╔═╡ c57862d7-c138-379c-7437-dac2aae5b938
let
	rolls = rand(1:6, 1000)
	histplot(rolls; bins = 6, xlabel = "number of eyes", title = "1000 rolls, average $(round(mean(rolls), digits = 2))")
end

# ╔═╡ 1c721cb6-1518-5f08-14e2-3715f9e19718
# Code for the figures: Section 2: the coin experiment
begin
	function figure_0_1(sd00_0, s0_seq)
		u = [c == 'H' for c in uppercase(s0_seq) if c in "HT"]
		nbad = count(c -> !(c in "HTht") && !isspace(c), s0_seq)
		length(u) < 20 && return md"*Enter at least 20 letters H or T.*"
		cut = length(u) > 400; cut && (u = u[1:400])
		n = length(u); rng = MersenneTwister(50 + 1009sd00_0)
		sims = [runstats(rand(rng, Bool, n)) for _ in 1:20000]
		me = runstats(u)
		sw = [s.switches for s in sims]; lg = [s.longest for s in sims]
		lo, hi = min(minimum(sw), me.switches) - 1, max(maximum(sw), me.switches) + 1
		ks = lo:hi; fsw = [count(==(k), sw) / length(sw) for k in ks]; ytop = 1.25maximum(fsw)
		f = fig(w = 330, h = 250, xr = (lo - 0.5, hi + 0.5), yr = (0, ytop), xlabel = "number of switches H↔T", ylabel = "relative frequency", title = "switches")
		bars!(f, ks, fsw; width = 0.85, c = COL.blue, op = 0.6, stroke = "none")
		vline!(f, me.switches; c = COL.red, dash = nothing, w = 2.5)
		text!(f, me.switches, 0.93ytop, "yours: $(me.switches)"; anchor = me.switches > (lo + hi) / 2 ? "end" : "start", dx = me.switches > (lo + hi) / 2 ? -5 : 5, c = COL.red, size = 12, bg = true)
		L = max(maximum(lg), me.longest) + 1
		flg = [count(==(k), lg) / length(lg) for k in 1:L]; ytop2 = 1.25maximum(flg)
		g = fig(w = 330, h = 250, xr = (0.4, L + 0.6), yr = (0, ytop2), xlabel = "longest run of equal letters", ylabel = "relative frequency", title = "longest run")
		bars!(g, 1:L, flg; width = 0.8, c = COL.blue, op = 0.6, stroke = "none")
		vline!(g, me.longest; c = COL.red, dash = nothing, w = 2.5)
		text!(g, me.longest, 0.93ytop2, "yours: $(me.longest)"; anchor = me.longest > L / 2 ? "end" : "start", dx = me.longest > L / 2 ? -5 : 5, c = COL.red, size = 12, bg = true)
		psw = count(>=(me.switches), sw) / length(sw); psw_lo = count(<=(me.switches), sw) / length(sw)
		prun = count(<=(me.longest), lg) / length(lg); prun_hi = count(>=(me.longest), lg) / length(lg)
		nh = count(u)
		info = "Your sequence: **$(n) tosses** ($(nh) H, $(n - nh) T), **$(me.switches) switches**, longest run **$(me.longest)**." *
			(cut ? " Only the first 400 letters are used." : "") * (nbad > 0 ? " $(nbad) other characters were ignored." : "")
		stats = "Among 20 000 simulated fair-coin sequences of length $(n), $(pct(psw)) have at least as many switches as yours; $(pct(prun)) have a longest run at most as long as yours."
		unusual = min(psw, psw_lo, prun, prun_hi) < 0.025
		verdict = unusual ? "In at least one statistic, your sequence lies in the outer 2.5 % of the distribution for a fair coin. Such a sequence is possible, but unusual for a coin." :
			" In both statistics, your sequence lies within the range that is typical for a fair coin."
		figure(f, g, Markdown.parse(info * "\n\n" * stats * "\n\n" * verdict);
			caption = Markdown.parse("**Figure 2.1.** Distribution of the number of switches (left) and of the longest run (right) in 20 000 simulated sequences of a fair coin of length $(n) (blue), and the values of your sequence (red). For a fair coin, about $(fmt((n - 1) / 2; d = 3)) switches are expected, and the longest run is typically close to ``\\log_2 n`` = $(fmt(log2(n); d = 2))."))
	end
end;

# ╔═╡ 0df99053-9c4f-8048-c719-1068fd38abc0
# Code for the figures: Section 3: the counter
begin
	function figure_1_1(s1_load, sd00_1)
		λ = s1_load; T = 480.0; rng = MersenneTwister(11 + 1009sd00_1)
		arr_r = poisson_arrivals(rng, λ, T); st_r = counter(arr_r)
		arr_d = collect(0.0:1/λ:T); st_d = counter(arr_d)
		W = 40.0
		tl = fig(w = 680, h = 150, l = 70, xr = (0, W), yr = (0, 2.2), xlabel = "time (minutes)", ylabel = "", yticks = [0.6, 1.6], yticklabels = ["random", "regular"],
			title = "the first $(Int(W)) minutes: waiting (light) and service (dark) of each customer")
		for (row, arr, st) in ((1.6, arr_d, st_d), (0.6, arr_r, st_r))
			for i in eachindex(arr)
				arr[i] > W && break
				k = isodd(i) ? 0.12 : -0.12
				st[i] > arr[i] + 1e-9 && rect!(tl, arr[i], row + k - 0.1, st[i], row + k + 0.1; fill = COL.red, op = 0.35)
				rect!(tl, st[i], row + k - 0.1, st[i] + 1, row + k + 0.1; fill = COL.blue, op = 0.85)
				seg!(tl, arr[i], row + k - 0.2, arr[i], row + k + 0.2; c = COL.ink, w = 1)
			end
		end
		ts, q = waiting_curve(arr_r, st_r, T)
		qn = max.(q .- 1, 0)
		ymax = max(maximum(qn) + 1, 4)
		f = fig(w = 330, h = 240, xr = (0, T), yr = (0, ymax), xlabel = "time (minutes)", ylabel = "people waiting", xticks = 0:120:480, title = "one day, random arrivals")
		area!(f, ts, qn; c = COL.blue, op = 0.12); line!(f, ts, qn; c = COL.blue, w = 1.3, steps = true)
		hline!(f, 0; c = COL.red, dash = "6 4", w = 2.2)
		legend!(f, [("random arrivals", COL.blue), ("regular arrivals", COL.red, :dash)]; at = :tl)
		loads = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
		wr = map(loads) do ℓ
			r = MersenneTwister(77); a = poisson_arrivals(r, ℓ, 200_000.0); s = counter(a); mean(s .- a)
		end
		g = fig(w = 330, h = 240, xr = (0.45, 1.0), yr = (0, 1.1maximum(wr)), xlabel = "load λ", ylabel = "mean waiting time (min)", title = "mean wait in the long run")
		line!(g, loads, wr; c = COL.blue, w = 2); dots!(g, loads, wr; c = COL.blue, r = 3.5)
		hline!(g, 0; c = COL.red, dash = "6 4", w = 2.2)
		k = findfirst(==(λ), loads); k === nothing || circle!(g, λ, wr[k], 6; fill = "none", stroke = COL.red, sw = 2, over = true)
		waits = st_r .- arr_r
		txt = Markdown.parse("Load ``\\lambda`` = $(λ): $(length(arr_r)) customers arrived during the simulated day of 8 hours. With random arrivals, $(pct(mean(waits .> 1e-9); d = 0)) of them had to wait, on average $(fmt(mean(waits); d = 2)) minutes, and at most $(maximum(qn)) people were waiting at the same time. With regular arrivals, nobody waits.")
		vstack(tl, figure(f, g), txt; caption = md"**Figure 3.1.** A counter that serves each customer in exactly 1 minute. Top: each customer is drawn as a bar that starts at the arrival (black tick); the light part is the waiting time, the dark part the service. Regular and random arrivals have the same average rate. Bottom left: the number of people waiting over one simulated day. Bottom right: the mean waiting time per customer as a function of the load, estimated from 200 000 minutes of simulation; the chosen load is circled.")
	end
end;

# ╔═╡ 869f7884-b749-642b-ef06-cebefe174a7b
# figures: distributions
begin
	const DISTS = ["heads in 20 coin tosses", "typing errors on a page", "waiting time for the next bus", "adult body height", "population of a town"]

	function figure_3_1(s3_kind, sd00_3)
		rng = MersenneTwister(31 + 1009sd00_3); n = 5000
		if s3_kind == DISTS[1]
			x = rand(rng, Binomial(20, 0.5), n); D = Binomial(20, 0.5); unit = "number of heads"
			story = "Each toss adds 0 or 1 to the count. The number of heads is **binomially** distributed with ``n = 20`` and ``p = 0.5``."
		elseif s3_kind == DISTS[2]
			x = rand(rng, Poisson(1.5), n); D = Poisson(1.5); unit = "number of errors"
			story = "A page has many characters, each with a small probability of an error. The number of rare events in a fixed region is approximately **Poisson** distributed (here with mean 1.5)."
		elseif s3_kind == DISTS[3]
			x = rand(rng, Exponential(10), n); D = Exponential(10); unit = "minutes"
			story = "If buses come at random moments without any schedule, the time already waited carries no information about the remaining waiting time. This property, absence of memory, characterises the **exponential** distribution (here with mean 10 minutes)."
		elseif s3_kind == DISTS[4]
			x = rand(rng, Normal(170, 8), n); D = Normal(170, 8); unit = "cm"
			story = "Body height depends on many small genetic and environmental effects that add up. Sums of many small independent effects are approximately **normally** distributed (central limit theorem)."
		else
			x = 10_000 .* (1 .- rand(rng, n)) .^ (-1 / 1.1); D = nothing; unit = "inhabitants"
			story = "If towns grow in proportion to their current size, effects multiply instead of adding up. Products of random numbers give a log-normal distribution, because taking the logarithm turns a product into a sum. It behaves like a **power law** (a scale-free distribution): most towns are small, very few are very large, and there is no typical size."
		end
		m, md_, sd = mean(x), median(x), std(x)
		if D isa DiscreteDistribution
			ks = minimum(x):maximum(x)
			f = fig(w = 440, h = 270, xr = (minimum(ks) - 0.7, maximum(ks) + 0.7), yr = (0, 1.2maximum(count(==(k), x) / n for k in ks)), xlabel = unit, ylabel = "relative frequency", title = s3_kind)
			bars!(f, ks, [count(==(k), x) / n for k in ks]; width = 0.8, c = COL.blue, op = 0.7)
			dots!(f, collect(ks), pdf.(D, ks); c = COL.ink, r = 3.5)
		elseif D !== nothing
			lo, hi = quantile(x, 0.001), quantile(x, 0.999); edges = collect(range(lo, hi, length = 36))
			hd = histdata(x, edges)
			f = fig(w = 440, h = 270, xr = (lo, hi), yr = (0, 1.2maximum(hd)), xlabel = unit, ylabel = "density", title = s3_kind)
			hist!(f, edges, hd; c = COL.blue, op = 0.6)
			xx = range(lo, hi, length = 200); line!(f, xx, pdf.(D, xx); c = COL.ink, w = 2)
		else
			edges = 10 .^ collect(range(4, log10(maximum(x)) + 0.01, length = 30))
			hd = [count(v -> edges[k] <= v < edges[k+1], x) / n for k in 1:length(edges)-1]
			f = fig(w = 440, h = 270, xr = (1e4, edges[end]), yr = (1e-4, 1), xlog = true, ylog = true, xlabel = "inhabitants (logarithmic axis)", ylabel = "fraction of towns (log)", title = s3_kind)
			for k in 1:length(hd)
				hd[k] > 0 && rect!(f, edges[k], 1e-4, edges[k+1], hd[k]; fill = COL.blue, op = 0.6, stroke = "white")
			end
		end
		vline!(f, m; c = COL.red, w = 2, dash = "5 3")
		tbl_ = tbl(["statistic", "value"], [["mean (red line)", fmt(m; d = 3)], ["median", fmt(md_; d = 3)], ["standard deviation", fmt(sd; d = 3)]])
		figure(f, vstack(tbl_, Markdown.parse(story)); caption = md"**Figure 5.1.** 5000 simulated values of the chosen quantity (blue) and the distribution of the model which generated them (black). The red line is the sample mean.")
	end

	function figure_3_2()
		rng = MersenneTwister(5); x = rand(rng, LogNormal(log(200), 0.8), 5000)
		m, s = mean(x), std(x); N = Normal(m, s)
		edges = collect(range(0, quantile(x, 0.995), length = 45)); hd = histdata(x, edges)
		f = fig(w = 330, h = 250, xr = (-300, edges[end]), yr = (0, 1.2maximum(hd)), xlabel = "response time (ms)", ylabel = "density", title = "data and fitted normal")
		hist!(f, edges, hd; c = COL.blue, op = 0.6)
		xx = range(-300, edges[end], length = 200); line!(f, xx, pdf.(N, xx); c = COL.red, w = 2)
		vline!(f, 0; c = "#999", dash = "3 3", w = 1)
		ths = range(m, quantile(x, 0.9995), length = 60)
		pe = [max(count(>(t), x) / length(x), 1e-5) for t in ths]; pn = [max(ccdf(N, t), 1e-7) for t in ths]
		g = fig(w = 330, h = 250, xr = (ths[1], ths[end]), yr = (1e-6, 1), ylog = true, xlabel = "threshold t (ms)", ylabel = "P(X > t)", title = "probability of exceeding t")
		line!(g, ths, pe; c = COL.blue, w = 2); line!(g, ths, pn; c = COL.red, w = 2)
		legend!(g, [("data", COL.blue), ("normal fit", COL.red)]; at = :tr)
		t3 = m + 3s; pd = count(>(t3), x) / length(x)
		txt = Markdown.parse("""A normal distribution is described by its mean and standard deviation only, and its tails decrease extremely fast. Many quantities do not behave in this way. The figure uses 5000 response times of a web server, generated from a log-normal model (a common model for durations and sizes). A normal distribution with the same mean ($(fmt(m; d = 3)) ms) and standard deviation ($(fmt(s; d = 3)) ms) is fitted to them.

	- The fitted normal gives a probability of $(pct(cdf(N, 0); d = 1)) to **negative** response times, which are impossible.
	- For the threshold ``t = \\bar x + 3s.d.`` = $(fmt(t3; d = 3)) ms, the normal distribution gives ``P(X > t)`` = $(pct(ccdf(N, t3); d = 2)), while $(pct(pd; d = 2)) of the data exceed it. On the right, the difference increases to several orders of magnitude for higher thresholds.

	The appropriate procedure is to look at the data before choosing a model: a histogram, a histogram on logarithmic axes if the values are of several orders of magnitude, and a plot of the tail ``P(X > t)``. The choice of a model should also be supported by the mechanism that produced the data. Heavy tails are typical for response times, file sizes, insurance claims and financial losses.""")
		figure(f, g, txt; caption = md"**Figure 5.2.** Left: histogram of the data (blue) and the fitted normal density (red). Right: the probability of exceeding a threshold, from the data and from the normal fit.")
	end
end;

# ╔═╡ 66d3f869-827a-73d7-3c8f-76e3eda1f123
# running averages
begin
	function figure_4_1(s4_N, sd00_4)
		rng = MersenneTwister(41 + 1009sd00_4); R = 20; N = s4_N
		runs = [cumsum(rand(rng, 1:6, N)) ./ (1:N) for _ in 1:R]
		f = fig(w = 440, h = 280, xr = (1, N), yr = (1, 6), xlog = true, xlabel = "number of rolls n (logarithmic axis)", ylabel = "running average", title = "20 experiments with $(N) rolls each")
		nn = 10 .^ range(0, log10(N), length = 120); σ = sqrt(35 / 12)
		band!(f, nn, clamp.(3.5 .- 2σ ./ sqrt.(nn), 1, 6), clamp.(3.5 .+ 2σ ./ sqrt.(nn), 1, 6); c = COL.blue, op = 0.12)
		for r in runs
			idx = unique(round.(Int, 10 .^ range(0, log10(N), length = 150)))
			line!(f, idx, r[idx]; c = COL.blue, w = 1, op = 0.55)
		end
		hline!(f, 3.5; c = COL.ink, dash = "5 3", w = 1.5)
		legend!(f, [("3.5 ± 2σ/√n", COL.blue, :bar), ("expected value 3.5", COL.ink, :dash)]; at = :tr)
		finals = [r[end] for r in runs]
		txt = Markdown.parse("After n = $(N) rolls, the 20 averages lie between **$(fmt(minimum(finals); d = 3))** and **$(fmt(maximum(finals); d = 3))**.")
		figure(f, txt; caption = md"**Figure 6.1.** Each line is the running average of one experiment in which a fair die is rolled repeatedly.")
	end
end;

# ╔═╡ 2c84d8e1-9402-34f3-cf41-dacdaf409664
# Penney's game and rock–paper–scissors
begin
	const PATS3 = ["HHH", "HHT", "HTH", "HTT", "THH", "THT", "TTH", "TTT"]

	penney_reply(a::String) = (a[2] == 'H' ? "T" : "H") * a[1:2]

	"""Conway's correlation.... sum of 2^(k-1) over all k for which the last k letters of a equal the first k letters of b."""
	corr(a, b) = sum((2^(k - 1) for k in 1:min(length(a), length(b)) if a[end-k+1:end] == b[1:k]); init = 0)

	"""Probability that pattern b appears before pattern a in fair coin tosses (Conway's formula)."""
	p_beats(b, a) = (corr(a, a) - corr(a, b)) / ((corr(a, a) - corr(a, b)) + (corr(b, b) - corr(b, a)))

	function penney_table()
		rows = [[a, penney_reply(a), pct(p_beats(penney_reply(a), a); d = 1), fmt(1 / (1 - p_beats(penney_reply(a), a)) - 1; d = 3) * " : 1"] for a in PATS3]
		vstack(tbl(["your pattern", "computer's reply", "P(computer wins)", "odds"], rows);
			caption = md"**Table 10.1.** Exact probabilities that the computer's reply appears before your pattern, from J. H. Conway's formula.")
	end

	function penney_game()
		@htl("""<div class="pg" style="font-family:system-ui,sans-serif;max-width:680px">
		<div style="margin:4px 0">Your pattern: <span class="pg-pats"></span></div>
		<div class="pg-msg" style="margin:8px 0;min-height:1.4em"></div>
		<div class="pg-seq" style="display:flex;flex-wrap:wrap;gap:3px;min-height:34px;padding:6px;border:1px solid #e3e3df;border-radius:4px;background:#fafaf7"></div>
		<div style="margin:8px 0;display:flex;gap:8px;flex-wrap:wrap;align-items:center"><button class="pg-go">Toss until a pattern appears</button><button class="pg-many">Play 100 races at once</button><button class="pg-reset">Reset the score</button></div>
		<table class="tbl" style="margin:6px 0"><thead><tr><th></th><th>races</th><th>won by you</th><th>won by the computer</th><th>share won by the computer</th></tr></thead>
		<tbody><tr><td class="pg-pair">–</td><td class="pg-n">0</td><td class="pg-y">0</td><td class="pg-c">0</td><td class="pg-p">–</td></tr></tbody></table>
		<div style="font-size:0.85rem;color:#666">Figure 10.1. Penney's game. Each circle is one toss of a fair coin. The race stops as soon as the last three tosses form one of the two patterns.</div>
		<script>
		const root = currentScript.parentElement, P = ["HHH","HHT","HTH","HTT","THH","THT","TTH","TTT"];
		const pats = root.querySelector(".pg-pats"), msg = root.querySelector(".pg-msg"), seq = root.querySelector(".pg-seq");
		let mine = "HHT", comp = "", sc = {n: 0, y: 0, c: 0}, timer = null;
		const reply = a => (a[1] === "H" ? "T" : "H") + a.slice(0, 2);
		function upd() {
			root.querySelector(".pg-pair").textContent = mine + " vs " + comp;
			root.querySelector(".pg-n").textContent = sc.n; root.querySelector(".pg-y").textContent = sc.y; root.querySelector(".pg-c").textContent = sc.c;
			root.querySelector(".pg-p").textContent = sc.n ? (100 * sc.c / sc.n).toFixed(1) + " %" : "–";
		}
		function choose(a) {
			if (timer) { clearInterval(timer); timer = null; }
			mine = a; comp = reply(a); sc = {n: 0, y: 0, c: 0}; seq.innerHTML = "";
			pats.querySelectorAll("button").forEach(b => { b.style.background = b.textContent === a ? "#1f5aa6" : ""; b.style.color = b.textContent === a ? "#fff" : ""; });
			msg.innerHTML = "You chose <b style='color:#1f5aa6'>" + mine + "</b>. The computer chooses <b style='color:#c2462e'>" + comp + "</b>.";
			upd();
		}
		P.forEach(a => { const b = document.createElement("button"); b.textContent = a; b.style.cssText = "margin:0 3px;font-family:monospace;font-size:0.95rem"; b.addEventListener("click", () => choose(a)); pats.appendChild(b); });
		function coin(c, hl) {
			const d = document.createElement("span");
			d.textContent = c;
			d.style.cssText = "display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:50%;font:bold 13px monospace;border:2px solid " + (hl || "#999") + ";background:" + (c === "H" ? "#e8c547" : "#fff");
			return d;
		}
		function race() { let s = ""; while (true) { s += Math.random() < 0.5 ? "H" : "T"; const t = s.slice(-3); if (t === mine) return [s, 1]; if (t === comp) return [s, 2]; } }
		root.querySelector(".pg-go").addEventListener("click", () => {
			if (timer) return;
			const [s, w] = race(); seq.innerHTML = ""; let i = 0;
			msg.innerHTML = "Tossing …";
			timer = setInterval(() => {
				seq.appendChild(coin(s[i], i >= s.length - 3 ? (w === 1 ? "#1f5aa6" : "#c2462e") : null)); i++;
				if (i >= s.length) {
					clearInterval(timer); timer = null; sc.n++; if (w === 1) sc.y++; else sc.c++; upd();
					msg.innerHTML = (w === 1 ? "<b style='color:#1f5aa6'>" + mine + "</b> appeared first: you win" : "<b style='color:#c2462e'>" + comp + "</b> appeared first: the computer wins") + " after " + s.length + " tosses.";
				}
			}, 260);
		});
		root.querySelector(".pg-many").addEventListener("click", () => { for (let k = 0; k < 100; k++) { const w = race()[1]; sc.n++; if (w === 1) sc.y++; else sc.c++; } upd(); msg.innerHTML = "100 races played."; });
		root.querySelector(".pg-reset").addEventListener("click", () => choose(mine));
		invalidation.then(() => { if (timer) clearInterval(timer); });
		choose("HHT");
		</script></div>""")
	end

	function rps_game()
		@htl("""<div class="rps" style="font-family:system-ui,sans-serif;max-width:680px">
		<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:6px 0">Your move:
		<button data-m="0" style="font-size:1rem">✊ rock</button><button data-m="1" style="font-size:1rem">✋ paper</button><button data-m="2" style="font-size:1rem">✌ scissors</button></div>
		<div class="rps-msg" style="margin:8px 0;min-height:1.4em">Choose a move. The computer decides before it sees your move.</div>
		<div style="display:flex;gap:18px;flex-wrap:wrap">
		<table class="tbl"><thead><tr><th>rounds</th><th>you win</th><th>draw</th><th>computer wins</th></tr></thead>
		<tbody><tr><td class="r-n">0</td><td class="r-y">0</td><td class="r-d">0</td><td class="r-c">0</td></tr></tbody></table>
		<table class="tbl r-T"></table></div>
		<div style="margin:6px 0;display:flex;gap:8px;flex-wrap:wrap"><button class="r-rand">Let a random player play 300 rounds</button><button class="r-reset">Reset</button></div>
		<div style="font-size:0.85rem;color:#666">Figure 10.2. Rock–paper–scissors against a predictor. Right table: how often each of your moves followed each of your previous moves, as counted by the computer.</div>
		<script>
		const root = currentScript.parentElement, names = ["rock", "paper", "scissors"];
		let C, prev, sc;
		function reset() { C = [[0,0,0],[0,0,0],[0,0,0]]; prev = -1; sc = {n: 0, y: 0, d: 0, c: 0}; upd(); root.querySelector(".rps-msg").textContent = "Choose a move. The computer decides before it sees your move."; }
		function predict() {
			if (prev < 0) return Math.floor(Math.random() * 3);
			const r = C[prev], m = Math.max(...r); const best = [0,1,2].filter(k => r[k] === m);
			return best[Math.floor(Math.random() * best.length)];
		}
		function play(m) {
			const guess = predict(), cm = (guess + 1) % 3;
			const res = m === cm ? 0 : ((m - cm + 3) % 3 === 1 ? 1 : -1);
			sc.n++; if (res === 1) sc.y++; else if (res === 0) sc.d++; else sc.c++;
			if (prev >= 0) C[prev][m]++; prev = m; return [cm, res];
		}
		function upd() {
			const q = k => sc.n ? " (" + (100 * k / sc.n).toFixed(0) + " %)" : "";
			root.querySelector(".r-n").textContent = sc.n; root.querySelector(".r-y").textContent = sc.y + q(sc.y);
			root.querySelector(".r-d").textContent = sc.d + q(sc.d); root.querySelector(".r-c").textContent = sc.c + q(sc.c);
			let h = "<thead><tr><th>previous ↓ / next →</th>" + names.map(x => "<th>" + x + "</th>").join("") + "</tr></thead><tbody>";
			for (let i = 0; i < 3; i++) h += "<tr><td>" + names[i] + "</td>" + C[i].map(x => "<td>" + x + "</td>").join("") + "</tr>";
			root.querySelector(".r-T").innerHTML = h + "</tbody>";
		}
		root.querySelectorAll("button[data-m]").forEach(b => b.addEventListener("click", () => {
			const m = +b.dataset.m, [cm, res] = play(m);
			root.querySelector(".rps-msg").innerHTML = "You: <b>" + names[m] + "</b>, computer: <b>" + names[cm] + "</b>. " + (res === 1 ? "You win this round." : res === 0 ? "Draw." : "The computer wins this round.");
			upd();
		}));
		root.querySelector(".r-rand").addEventListener("click", () => { reset(); for (let k = 0; k < 300; k++) play(Math.floor(Math.random() * 3)); upd(); root.querySelector(".rps-msg").textContent = "300 rounds of a player who chooses each move with probability 1/3, independently of the past."; });
		root.querySelector(".r-reset").addEventListener("click", reset);
		reset();
		</script></div>""")
	end
end;

# ╔═╡ f913a25f-68e6-6c6e-6755-a025e40c1086
penney_game()

# ╔═╡ 4437b792-6c97-96aa-cd53-913ac13cc38b
penney_table()

# ╔═╡ 312d8ed8-f90d-6f0f-ba41-a9826728f9a1
rps_game()

# ╔═╡ d31f2275-5a0a-0e59-1256-393f6cf1990e
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

	"""Play a sound: `x` are samples in [-1, 1] at rate `fs`; returns an HTML audio player (16-bit WAV, embedded)."""
	function audio(x::AbstractVector{<:Real}; fs::Integer = 16000, label = "")
		m = maximum(abs, x; init = 0.0); y = m > 0 ? x ./ m .* 0.9 : x
		pcm = round.(Int16, clamp.(y, -1, 1) .* 32767)
		io = IOBuffer(); nb = 2length(pcm)
		write(io, b"RIFF", UInt32(36 + nb), b"WAVEfmt ", UInt32(16), UInt16(1), UInt16(1), UInt32(fs), UInt32(2fs), UInt16(2), UInt16(16), b"data", UInt32(nb))
		for v in pcm; write(io, htol(v)); end
		src = "data:audio/wav;base64," * base64encode(take!(io))
		@htl("""<span class="aud" style="display:inline-flex;align-items:center;gap:8px">$(label == "" ? "" : @htl("<span style='color:#444'>$(label)</span>"))<audio controls preload="auto" src="$(src)" style="height:32px;vertical-align:middle"></audio></span>""")
	end

	"""
	    play_notes(events; label, seed)

	Play a list of note events in the browser (no audio file is embedded). Each event is a tuple
	`(t, f, d, v, k)`: start time in s, frequency in Hz, duration in s, loudness in 0…1 and the
	instrument `k` (0 a plucked string synthesised by the Karplus–Strong method from random numbers,
	1 a low drum, 2 a high drum). Elements with class `pn-hl` and attributes `data-a`, `data-b`
	(times in s) in the same figure are shown only while the music is between these times.
	"""
	function play_notes(events; label = "Play", seed::Integer = 1, group::AbstractString = "")
		ev = [[round(Float64(e[1]), digits = 4), round(Float64(e[2]), digits = 3), round(Float64(e[3]), digits = 4), round(Float64(e[4]), digits = 3), Int(e[5])] for e in events]
		@htl("""<span class="pn"><button class="pn-b">▶ $(label)</button><script>
		const root = currentScript.parentElement, btn = root.querySelector(".pn-b"), ev = $(ev), lab = $(label), grp = $(group);
		let ctx = null, src = null, raf = null, t0 = 0;
		function rng(a) { return function() { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
		function render(sr) {
			const R = rng($(seed));
			let T = 0; for (const e of ev) T = Math.max(T, e[0] + e[2]);
			const N = Math.ceil((T + 1.6) * sr), y = new Float32Array(N);
			for (const [t, f, d, v, k] of ev) {
				const a = Math.floor(t * sr);
				if (k === 0) {
					const P = Math.max(2, Math.round(sr / f)), L = Math.min(N - a, Math.floor((d + 1.2) * sr)), buf = new Float32Array(L);
					let lp = 0; for (let i = 0; i < Math.min(P, L); i++) { lp = 0.6 * lp + 0.4 * (2 * R() - 1); buf[i] = lp; }
					const off = Math.floor(d * sr);
					for (let i = P; i < L; i++) { const s = i < off ? 0.998 : 0.985; buf[i] = s * 0.5 * (buf[i - P] + buf[i - P - 1 >= 0 ? i - P - 1 : i - P]); }
					for (let i = 0; i < L; i++) y[a + i] += v * buf[i];
				} else if (k === 3) {
					// a struck bowl: three partials with exponential decay (jal tarang, chimes)
					const L = Math.min(N - a, Math.floor((d + 1.6) * sr)), parts = [[1, 1, 2.0], [2.76, 0.18, 0.7], [5.4, 0.06, 0.3]];
					for (const [m, g, tau] of parts) { const w = 2 * Math.PI * f * m / sr; if (f * m > 0.45 * sr) continue;
						for (let i = 0; i < L; i++) { const tt = i / sr; y[a + i] += v * g * Math.min(1, tt / 0.003) * Math.exp(-tt / tau) * Math.sin(w * i); } }
				} else if (k === 4) {
					// a drone tone: harmonic spectrum with a slow swell and slow beating
					const L = Math.min(N - a, Math.floor(d * sr));
					for (let n = 1; n <= 6; n++) { const w = 2 * Math.PI * f * n * (1 + 0.0004 * n) / sr, g = Math.pow(n, -1.1), ph = R() * 6.28, rt = 0.07 + 0.04 * n;
						for (let i = 0; i < L; i++) { const tt = i / sr, env = Math.min(1, tt / 1.5) * Math.min(1, (L - i) / sr / 1.5);
							y[a + i] += v * g * env * (0.6 + 0.4 * Math.sin(2 * Math.PI * rt * tt + ph)) * Math.sin(w * i + ph); } }
				} else {
					const low = k === 1, L = Math.min(N - a, Math.floor((low ? 0.7 : 0.35) * sr));
					let ph = 0;
					for (let i = 0; i < L; i++) {
						const tt = i / sr, fr = low ? 70 + 60 * Math.exp(-tt * 18) : 330 + 40 * Math.exp(-tt * 40);
						ph += 2 * Math.PI * fr / sr;
						const env = Math.exp(-tt * (low ? 5 : 12));
						let s = Math.sin(ph) + (low ? 0.2 * Math.sin(2 * ph) : 0.45 * Math.sin(2.02 * ph) + 0.25 * Math.sin(3.01 * ph));
						if (i < 0.006 * sr) s += 0.6 * (2 * R() - 1);
						y[a + i] += v * 0.6 * env * s;
					}
				}
			}
			let m = 0; for (let i = 0; i < N; i++) m = Math.max(m, Math.abs(y[i]));
			if (m > 0) for (let i = 0; i < N; i++) y[i] *= 0.85 / m;
			return y;
		}
		const scope = root.closest("pluto-output") || root.closest(".fg") || root.parentElement;
		const hl = Array.from(scope.querySelectorAll(".pn-hl")).filter(e => !grp || e.dataset.g === grp);
		function mark(t) { for (const e of hl) { const a = +e.dataset.a, b = +e.dataset.b; e.style.display = (t >= a && t < b) ? "" : "none"; } }
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
end;

# ╔═╡ 8ee00361-b280-9c3e-d72b-cab8912974ad
# Code for the figures: Section 4: the Galton board
begin
	function figure_2_1(s2_rows, sd00_2)
		R = s2_rows; N = 150; rng = MersenneTwister(21 + 1009sd00_2)
		paths = [cumsum(rand(rng, Bool, R)) for _ in 1:N]
		bins = [p[end] for p in paths]
		H = 0.62; top = R + 0.6 + H * R; ylo = -0.1; yhi = top + 0.8
		f = fig(w = 440, h = 430, xr = (-1, R + 1), yr = (ylo, yhi), axes = false, l = 6, r = 6, t = 26, b = 6, title = "Galton board with $(R) rows of pegs")
		pxu = (430 - 26 - 6) / (yhi - ylo)
		for r in 1:R, j in 0:r-1
			circle!(f, R / 2 - (r - 1) / 2 + j, top - r * H, 3; fill = "#555", stroke = "#555", sw = 0.5)
		end
		for k in 0:R+1
			seg!(f, k - 0.5, 0, k - 0.5, R + 0.3; c = "#ccc", w = 1)
		end
		seg!(f, -0.5, 0, R + 0.5, 0; c = "#555", w = 1.5)
		e = [N * binomial(R, k) / 2^R for k in 0:R]
		cmax = max(maximum(count(==(b), bins) for b in 0:R), maximum(e)); sc = min((R + 0.1) / max(cmax, 1), 0.9)
		cnt = zeros(Int, R + 1)
		for (i, p) in enumerate(paths)
			pp = vcat(0, p)
			xs = Float64[R / 2]; ys = Float64[top + 0.5]
			for r in 1:R
				push!(xs, R / 2 - (r - 1) / 2 + pp[r]); push!(ys, top - r * H + 0.18)
			end
			k = p[end]; cnt[k+1] += 1
			push!(xs, k); push!(ys, sc * (cnt[k+1] - 0.5))
			line!(f, xs, ys; c = COL.red, w = 2, o = i)
			circle!(f, k, sc * (cnt[k+1] - 0.5), max(1.5, 0.45 * sc * pxu); fill = COL.blue, stroke = "white", sw = 0.5, s = i)
		end
		line!(f, collect(0:R), sc .* e; c = COL.ink, w = 2, dash = "5 3", s = N)
		reads = ["Ball $(i) was deflected to the right $(bins[i]) times out of $(R) and landed in bin $(bins[i])." for i in 1:N]
		pushfirst!(reads, "At every peg, the ball is deflected to the left or to the right with probability ½. Press ▶.")
		player(f; n = N, start = 0, fps = 8, reads = reads, label = "ball",
			caption = md"**Figure 4.1.** Each ball (red path) is deflected to the left or right at every peg with probability ½. The position of a single ball cannot be predicted, but after many balls the histogram approaches the binomial distribution (dashed line, shown after the last ball).")
	end
end;

# ╔═╡ f4fe5c11-2b9e-9980-bc5f-bc30e2351409
# music
begin
	function figure_9_1(s9_note, sd00_9)
		f0 = Dict("A3 (220 Hz)" => 220.0, "E4 (330 Hz)" => 329.63, "A4 (440 Hz)" => 440.0)[s9_note]
		fs = 22050; rng = MersenneTwister(91 + 1009sd00_9)
		y = pluck(f0, 1.6, rng; fs = fs)
		P = round(Int, fs / f0)
		noise = 2 .* rand(MersenneTwister(92 + 1009sd00_9), length(y)) .- 1
		noise .*= exp.(-(0:length(y)-1) ./ (0.35fs))
		w1 = 1:3P; t0 = round(Int, 0.5fs); w2 = t0:t0+3P
		F = fig(w = 330, h = 210, xr = (0, 3P / fs * 1000), yr = (-1.1, 1.1), xlabel = "time (ms)", ylabel = "amplitude", title = "start: the repeated burst")
		line!(F, (collect(w1) .- 1) ./ fs .* 1000, y[w1]; c = COL.red, w = 1.1)
		for j in 1:2; vline!(F, j * P / fs * 1000; c = "#999", dash = "3 3", w = 1); end
		G = fig(w = 330, h = 210, xr = (0, 3P / fs * 1000), yr = (-1.1, 1.1), xlabel = "time (ms) after 0.5 s", ylabel = "", title = "after 0.5 s (rescaled)")
		m = maximum(abs, y[w2]) > 0 ? 1 / maximum(abs, y[w2]) : 1.0
		line!(G, (collect(w2) .- t0) ./ fs .* 1000, m .* y[w2]; c = COL.blue, w = 1.6)
		for j in 1:2; vline!(G, j * P / fs * 1000; c = "#999", dash = "3 3", w = 1); end
		seg_ = round(Int, 0.05fs):round(Int, 0.05fs)+4095
		fr = range(20, 10.5 * f0, length = 600)
		S1 = spectrum(y[seg_], fs, fr); S0 = spectrum(noise[seg_], fs, fr)
		S1 ./= maximum(S1); S0 ./= maximum(S0)
		H = fig(w = 680, h = 250, xr = (0, 10.5 * f0), yr = (0, 1.42), yticks = [0, 0.5, 1], xlabel = "frequency (Hz)", ylabel = "relative magnitude", title = "spectrum of 0.19 s of sound starting at 50 ms",
			xticks = [k * f0 for k in 1:10], xticklabels = [k == 1 ? "f₀ = $(round(Int, f0))" : "$(k)f₀" for k in 1:10])
		line!(H, fr, S0; c = COL.grey, w = 1); line!(H, fr, S1; c = COL.blue, w = 1.6)
		legend!(H, [("random numbers, not repeated", COL.grey), ("Karplus–Strong string", COL.blue)]; at = :tr)
		figure(F, G, H, vstack(audio(noise; fs = fs, label = "random numbers, not repeated:"), audio(y; fs = fs, label = "Karplus–Strong string at $(round(Int, f0)) Hz:"));
			caption = Markdown.parse("**Figure 11.1.** The Karplus–Strong algorithm at $(fs) samples per second. Top left: the signal starts as a burst of P = $(P) random numbers, repeated with period $(fmt(1000P / fs; d = 3)) ms (dashed lines). Top right: after repeated averaging, the random components have decayed and a smooth periodic wave is obtained. Bottom: the spectrum of the plucked string (blue) consists of peaks at the multiples of ``f_0``."))
	end

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

	const MOZART_RH = ["77,0,2;74,2,2;79,4,2",
		"69,0,2;66,2,1;67,3,1;71,4,1;79,5,1",
		"79,0,2;72,2,2;76,4,2",
		"79,0,2;74,2,4",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"72,0,4",
		"72,0,2;76,0,2;71,2,2;74,2,2",
		"71,0,1;69,1,1;71,2,1;72,3,1;74,4,1;71,5,1",
		"76,0,1;72,1,1;71,2,1;69,3,1;67,4,1;66,5,1",
		"64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"72,0,2;67,2,2;64,4,2",
		"72,0,4",
		"76,0,2;79,2,1;76,3,1;72,4,2",
		"81,0,2;78,2,2;74,4,2",
		"72,0,1;67,1,1;72,2,1;76,3,1;67,4,1;72,5,1",
		"67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;76,2,2;79,4,2",
		"79,0,2;83,2,1;86,3,1;74,4,2",
		"72,0,1;76,1,1;79,2,1;74,3,1;69,4,1;78,5,1",
		"76,0,2;72,2,2;67,4,2",
		"77,0,1;76,1,1;74,2,1;76,3,1;77,4,1;79,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"62,0,1;66,1,1;69,2,1;74,3,1;78,4,1;81,5,1",
		"72,0,2;76,0,2;72,2,2;76,2,2;72,4,2;76,4,2",
		"77,0,1;76,1,1;77,2,1;74,3,1;72,4,1;71,5,1",
		"78,0,1;74,1,1;69,2,1;81,3,1;78,4,1;74,5,1",
		"71,0,1;74,1,1;79,2,1;74,3,1;71,4,2",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"76,0,1;72,1,1;67,2,2;76,4,2",
		"67,0,2;72,2,2;76,4,2",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"76,0,1;72,1,1;74,2,1;71,3,1;67,4,2",
		"69,0,2;74,2,2;78,4,2",
		"69,0,1;76,1,1;74,2,1;79,3,1;78,4,1;81,5,1",
		"79,0,1;83,1,1;79,2,1;74,3,1;71,4,2",
		"72,0,2;67,2,2;76,4,2",
		"79,0,2;67,2,2;67,4,2",
		"72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1",
		"72,0,1;71,1,1;72,2,1;76,3,1;67,4,2",
		"71,0,1;72,1,1;74,2,1;71,3,1;69,4,1;67,5,1",
		"79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"69,0,2;77,2,1;74,3,1;69,4,1;71,5,1",
		"72,0,1;71,1,1;72,2,1;67,3,1;64,4,1;60,5,1",
		"79,0,2;83,2,1;79,3,1;74,4,1;71,5,1",
		"79,0,2;79,2,1;74,3,1;83,4,2",
		"76,0,2;72,2,1;76,3,1;79,4,1;84,5,1",
		"76,0,2;72,2,2;67,4,2",
		"72,0,2;76,2,1;72,3,1;67,4,2",
		"72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1",
		"74,0,1;73,1,1;74,2,1;77,3,1;67,4,1;71,5,1",
		"72,0,2;76,0,2;72,2,1;76,2,1;74,3,1;77,3,1;76,4,2;79,4,2",
		"64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"79,0,2;83,2,2;74,4,2",
		"74,0,1;71,1,1;67,2,2",
		"76,0,2;72,2,2;67,4,2",
		"79,0,2;76,2,2;72,4,2",
		"79,0,2;72,2,2;76,4,2",
		"79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"72,0,2;76,2,1;72,3,1;79,4,2",
		"76,0,1;72,1,1;71,2,1;67,3,1;69,4,1;66,5,1",
		"76,0,1;72,1,1;71,2,1;72,3,1;67,4,2",
		"76,0,1;79,1,1;84,2,1;79,3,1;76,4,1;72,5,1",
		"74,0,1;69,1,1;74,2,2;78,4,2",
		"78,0,2;81,2,2;78,4,2",
		"72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1",
		"79,0,2;83,2,1;79,3,1;74,4,1;79,5,1",
		"79,0,2;76,2,2;72,4,2",
		"78,0,2;81,2,1;78,3,1;74,4,1;78,5,1",
		"79,0,1;83,1,1;86,2,1;83,3,1;79,4,2",
		"77,0,1;76,1,1;74,2,1;72,3,1;71,4,1;74,5,1",
		"79,0,2;76,2,2;72,4,2",
		"84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1",
		"74,0,2;78,0,2;74,2,2;78,2,2;74,4,2;78,4,2",
		"84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1",
		"79,0,1;83,1,1;79,2,2;74,4,2",
		"72,0,2;60,2,2",
		"72,0,4",
		"74,0,2;69,2,2;78,4,2",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"74,0,1;71,1,1;67,2,2;79,4,2",
		"72,0,4",
		"72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1",
		"72,0,2;76,2,2;67,4,2",
		"74,0,2;74,2,1;79,3,1;83,4,2",
		"79,0,2;72,2,2;76,4,2",
		"79,0,1;74,1,1;79,2,1;83,3,1;79,4,1;74,5,1",
		"77,0,1;76,1,1;74,2,2;79,4,2",
		"78,0,1;81,1,1;86,2,1;81,3,1;78,4,1;81,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"71,0,2;74,0,2;79,2,1;83,3,1;74,4,2",
		"72,0,4",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"79,0,2;76,2,2;72,4,2",
		"76,0,2;72,2,2;67,4,2",
		"79,0,1;78,1,1;79,2,1;74,3,1;71,4,1;67,5,1",
		"72,0,2;67,2,2;76,4,2",
		"78,0,2;81,2,2;74,4,2",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"78,0,1;74,1,1;69,2,2;78,4,2",
		"72,0,1;76,1,1;72,2,1;67,3,1;64,4,2",
		"76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"78,0,2;81,2,1;78,3,1;74,4,1;78,5,1",
		"69,0,2;74,2,1;72,3,1;71,4,1;69,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"76,0,2;79,2,2;84,4,2",
		"74,0,1;77,1,1;74,2,1;77,3,1;71,4,1;74,5,1",
		"71,0,1;74,0,1;69,1,1;72,1,1;69,2,1;72,2,1;67,3,1;71,3,1;67,4,1;71,4,1;66,5,1;69,5,1",
		"72,0,4",
		"76,0,2;72,2,2;67,4,2",
		"77,0,2;74,2,2;71,4,2",
		"71,0,2;74,0,2;71,2,2;74,2,2;71,4,2;74,4,2",
		"72,0,1;67,1,1;76,2,1;72,3,1;79,4,1;76,5,1",
		"74,0,1;77,1,1;81,2,1;77,3,1;74,4,1;71,5,1",
		"74,0,1;69,1,1;74,2,1;78,3,1;81,4,1;78,5,1",
		"76,0,1;81,1,1;79,2,1;83,3,1;78,4,1;81,5,1",
		"76,0,1;72,1,1;79,2,1;76,3,1;84,4,1;79,5,1",
		"86,0,2;81,2,1;78,3,1;74,4,1;69,5,1",
		"79,0,2;83,2,1;79,3,1;74,4,2",
		"79,0,1;78,1,1;79,2,1;83,3,1;74,4,2",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"79,0,1;76,1,1;74,2,1;71,3,1;67,4,2",
		"72,0,1;67,1,1;72,2,1;76,3,1;79,4,1;72,5,1;76,5,1",
		"67,0,4;71,0,4;74,0,4;79,0,4",
		"71,0,2;74,2,2;79,4,2",
		"81,0,1;79,1,1;78,2,1;79,3,1;74,4,2",
		"64,0,2;72,0,2;64,2,2;72,2,2;64,4,2;72,4,2",
		"72,0,4",
		"72,0,2;76,0,2;71,2,1;74,2,1;67,3,1;71,3,1;67,4,2",
		"74,0,2;79,2,1;74,3,1;71,4,1;74,5,1",
		"69,0,1;76,1,1;71,2,1;74,2,1;69,3,1;72,3,1;67,4,1;71,4,1;66,5,1;69,5,1",
		"78,0,2;78,2,1;74,3,1;81,4,2",
		"84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1",
		"72,0,2;67,2,2;76,4,2",
		"69,0,2;74,0,2;78,0,2;78,2,4",
		"79,0,1;83,1,1;79,2,1;83,3,1;74,4,2",
		"69,0,2;69,2,1;74,3,1;78,4,2",
		"74,0,1;76,1,1;77,2,1;74,3,1;72,4,1;71,5,1",
		"72,0,2;67,2,2;76,4,2",
		"79,0,2;74,2,1;71,3,1;67,4,2",
		"79,0,2;72,2,2;76,4,2",
		"74,0,1;77,1,1;69,2,1;74,3,1;71,4,1;74,5,1",
		"66,0,2;74,0,2;74,2,2;78,2,2;78,4,2;81,4,2",
		"76,0,1;84,1,1;83,2,1;79,3,1;81,4,1;78,5,1",
		"84,0,1;83,1,1;84,2,1;79,3,1;76,4,1;72,5,1",
		"77,0,1;74,1,1;69,2,2;71,4,2",
		"67,0,2;72,0,2;76,0,2;76,2,4",
		"72,0,4",
		"79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"74,0,1;69,1,1;78,2,1;74,3,1;81,4,1;78,5,1",
		"74,0,1;73,1,1;74,2,1;78,3,1;81,4,1;78,5,1",
		"79,0,1;83,1,1;79,2,1;74,3,1;71,4,1;67,5,1",
		"72,0,1;67,1,1;76,2,1;72,3,1;79,4,2",
		"76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"71,0,2;74,2,1;71,3,1;69,4,1;67,5,1",
		"76,0,1;79,1,1;74,2,1;72,3,1;71,4,1;69,5,1",
		"72,0,1;71,1,1;72,2,1;76,3,1;67,4,1;72,5,1",
		"66,0,2;74,0,2;66,2,2;74,2,2;66,4,2;74,4,2",
		"76,0,1;74,1,1;76,2,1;79,3,1;84,4,1;79,5,1",
		"79,0,1;78,1,1;79,2,1;74,3,1;71,4,1;67,5,1",
		"74,0,2;67,2,4",
		"74,0,2;71,2,2;67,4,2",
		"74,0,1;83,1,1;79,2,1;74,3,1;71,4,2",
		"72,0,2;72,2,1;74,3,1;76,4,2",
		"79,0,2;77,2,1;76,3,1;74,4,1;72,5,1",
		"76,0,1;79,1,1;74,2,1;79,3,1;69,4,1;78,5,1",
		"72,0,4",
		"71,0,1;72,1,1;74,2,1;76,3,1;77,4,1;74,5,1",
		"72,0,4",
		"77,0,1;81,1,1;69,2,2;71,4,1;74,5,1",
		"67,0,2;72,2,2;76,4,2",
		"76,0,1;72,1,1;71,2,1;74,3,1;79,4,2",
		"81,0,1;79,1,1;83,2,1;79,3,1;74,4,1;79,5,1"]

	const MOZART_LH = ["53,0,2;50,2,2;55,4,2",
		"47,0,4;55,0,4",
		"48,0,4;52,0,4",
		"43,0,1;47,1,1;55,2,2;47,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;52,0,4",
		"48,0,4;55,0,4",
		"48,0,2;43,2,2;36,4,2",
		"55,0,4;43,4,2",
		"55,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,2;48,2,2;48,4,2",
		"52,0,4;55,0,4;52,4,2;55,4,2",
		"48,0,2;43,2,2;36,4,2",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"50,0,4;54,0,4;48,4,2;54,4,2",
		"52,0,4;55,0,4",
		"48,0,4;52,0,4;48,4,2;55,4,2",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"47,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,4",
		"53,0,1;52,1,1;50,2,1;52,3,1;53,4,1;55,5,1",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"50,0,4;48,4,2",
		"48,0,1;52,1,1;55,2,1;52,3,1;60,4,1;48,5,1",
		"55,0,4;59,0,4",
		"48,0,4;57,0,4",
		"55,0,4;43,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;55,0,4;48,4,2;55,4,2",
		"48,0,4;52,0,4",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"55,0,4",
		"50,0,4;54,0,4;48,4,2;57,4,2",
		"48,0,2;50,2,2;38,4,2",
		"47,0,4;50,0,4",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"47,0,1;50,1,1;55,2,1;50,3,1;47,4,1;43,5,1",
		"48,0,4;52,0,4",
		"48,0,4;52,0,4",
		"43,0,4",
		"48,0,4;52,0,4",
		"53,0,4;55,4,2",
		"52,0,4;55,0,4",
		"47,0,4;50,0,4",
		"47,0,4;50,0,4",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"52,0,4;55,0,4",
		"48,0,4;52,0,4",
		"53,0,4;55,4,2",
		"48,0,4",
		"48,0,2;48,2,2;48,4,2",
		"47,0,4;50,0,4",
		"43,0,4;55,0,4;55,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"48,0,4;52,0,4",
		"52,0,4;55,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,4",
		"48,0,4;55,0,4;48,4,2;55,4,2",
		"50,0,4;54,0,4",
		"50,0,2;57,0,2;50,2,2;54,2,2;48,4,2;50,4,2",
		"48,0,4;52,0,4;52,4,2;55,4,2",
		"47,0,4",
		"48,0,4;52,0,4",
		"50,0,4;48,4,2",
		"47,0,4;50,0,4;47,4,2;50,4,2",
		"53,0,4;55,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"48,0,4;52,0,4",
		"48,0,2;48,2,2;48,4,2",
		"48,0,4;52,0,4;48,4,2;55,4,2",
		"47,0,4;50,0,4;47,4,2;55,4,2",
		"48,0,4;36,4,2",
		"48,0,2;43,2,2;36,4,2",
		"48,0,4",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"47,0,4;55,0,4;47,4,2;50,4,2",
		"48,0,2;43,2,2;36,4,2",
		"48,0,4;52,0,4",
		"52,0,4;55,0,4",
		"47,0,4;55,0,4",
		"48,0,4;52,0,4;48,4,2;55,4,2",
		"47,0,4;50,0,4;47,4,2;50,4,2",
		"53,0,1;52,1,1;50,2,2;55,4,2",
		"48,0,4;57,0,4;48,4,2;57,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"43,0,4;55,0,4;55,4,2",
		"48,0,2;43,2,2;36,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;52,0,4",
		"48,0,4",
		"47,0,4;50,0,4;47,4,2;55,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"48,0,4;57,0,4;48,4,2;57,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"48,0,4;57,0,4;48,4,2;57,4,2",
		"52,0,4;55,0,4",
		"48,0,4",
		"48,0,4",
		"48,0,2;50,2,2;38,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"53,0,4;57,0,4;55,4,2;62,4,2",
		"48,0,2;50,2,2;38,4,2",
		"48,0,2;43,2,2;36,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"55,0,4;59,0,4",
		"55,0,2;55,2,2;55,4,2",
		"48,0,4;52,0,4",
		"53,0,4;55,4,2",
		"50,0,4;54,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,4;52,0,4",
		"50,0,4;54,0,4;48,4,2;54,4,2",
		"47,0,4;55,0,4",
		"47,0,2;50,0,2;47,2,2;50,2,2;47,4,2;55,4,2",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"48,0,2;48,2,2;48,4,2",
		"55,0,2;43,2,2",
		"52,0,4;52,4,1;48,5,1",
		"43,0,2;59,2,1;55,3,1;54,4,1;52,5,1",
		"43,0,4",
		"47,0,2;50,0,2;47,2,2;50,2,2;47,4,2;55,4,2",
		"48,0,2;48,2,2;48,4,2",
		"48,0,2;43,2,2;36,4,2",
		"55,0,2;43,2,2",
		"47,0,4;55,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,2;50,0,2;48,2,2;50,2,2;48,4,2;50,4,2",
		"48,0,4;52,0,4",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"38,0,1;50,1,1;49,2,1;50,3,1;48,4,1;50,5,1",
		"47,0,4",
		"48,0,2;54,0,2;48,2,2;54,2,2;48,4,2;57,4,2",
		"47,0,4;55,0,4;43,4,2",
		"48,0,4;52,0,4",
		"47,0,4;50,0,4;47,4,2;50,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"53,0,4;55,4,2",
		"48,0,2;48,2,2;48,4,2",
		"48,0,2;50,2,2;38,4,2",
		"48,0,4;52,0,4",
		"53,0,4;55,4,2",
		"48,0,1;47,1,1;48,2,1;50,3,1;52,4,1;54,5,1",
		"48,0,2;43,2,2;36,4,2",
		"48,0,4;52,0,4",
		"48,0,4",
		"48,0,4",
		"47,0,4;50,0,4",
		"52,0,4;55,0,4",
		"48,0,4",
		"43,0,4",
		"48,0,2;50,2,2;38,4,2",
		"48,0,4;52,0,4;48,4,2;52,4,2",
		"48,0,2;48,2,2;48,4,2",
		"48,0,4;55,0,4;48,4,2;52,4,2",
		"47,0,4;50,0,4",
		"55,0,1;54,1,1;55,2,1;50,3,1;47,4,1;43,5,1",
		"47,0,4",
		"55,0,4;59,0,4",
		"48,0,4;52,0,4",
		"48,0,4;52,0,4;52,4,2;55,4,2",
		"48,0,2;50,2,2;38,4,2",
		"48,0,2;43,2,2;36,4,2",
		"43,0,4;55,0,4;47,4,2;55,4,2",
		"48,0,2;43,2,2;36,4,2",
		"53,0,4;55,4,2",
		"48,0,1;52,0,1;55,1,1;48,2,1;52,2,1;55,3,1;48,4,1;52,4,1;55,5,1",
		"55,0,2;43,2,2",
		"47,0,4;50,0,4;47,4,2;50,4,2"]

	"""1 bar is a vector of (midi pitch, onset in sixteenths, duration in sixteenths)."""
	decode_bar(s) = isempty(s) ? Tuple{Int,Int,Int}[] : [Tuple(parse.(Int, split(e, ","))) for e in split(s, ";")]

	midi_hz(n) = 440.0 * 2.0^((n - 69) / 12)

	const BAR_S = 1.0     # duration of one bar (3/8) in seconds

	function figure_11_2(sd00_11)
		rng = MersenneTwister(110 + 1009sd00_11)
		d1 = rand(rng, 1:6, 16); d2 = rand(rng, 1:6, 16); sums = d1 .+ d2
		bars = [MOZART_TABLE[i, sums[i]-1] for i in 1:16]
		ev = Tuple{Float64,Float64,Float64,Float64,Int}[]
		six = BAR_S / 6
		for (i, b) in enumerate(bars)
			t0 = (i - 1) * BAR_S
			for (n, o, d) in decode_bar(MOZART_RH[b]); push!(ev, (t0 + o * six, midi_hz(n), d * six, 0.55, 0)); end
			for (n, o, d) in decode_bar(MOZART_LH[b]); push!(ev, (t0 + o * six, midi_hz(n), d * six, 0.4, 0)); end
		end
		# the table
		f = fig(w = 680, h = 330, xr = (0.5, 16.5), yr = (1.0, 13.5), axes = false, l = 44, r = 6, t = 34, b = 8, title = "table of bars: row = sum of the dice, column = bar of the waltz")
		for i in 1:16
			text!(f, i, 13.0, "$(i)"; anchor = "middle", size = 11, bold = true, c = "#555")
		end
		for s in 2:12
			text!(f, 0.5, 14 - s, "$(s)"; anchor = "end", size = 11, bold = true, c = "#555", dx = -4)
			for i in 1:16
				chosen = sums[i] == s
				chosen && rect!(f, i - 0.47, 14 - s - 0.45, i + 0.47, 14 - s + 0.45; fill = COL.blue, op = 0.85, rx = 3)
				text!(f, i, 14 - s, string(MOZART_TABLE[i, s-1]); anchor = "middle", size = 10.5, c = chosen ? "white" : "#666")
			end
		end
		for i in 1:16
			raw!(f, "<rect class=\"pn-hl\" data-a=\"$((i - 1) * BAR_S)\" data-b=\"$(i * BAR_S)\" x=\"$(_n(X(f, i - 0.5)))\" y=\"$(_n(Y(f, 12.6)))\" width=\"$(_n(X(f, i + 0.5) - X(f, i - 0.5)))\" height=\"$(_n(Y(f, 1.4) - Y(f, 12.6)))\" fill=\"$(COL.red)\" opacity=\"0.18\"/>"; over = true)
		end
		text!(f, 0.5, 13.0, "bar"; anchor = "end", size = 11, c = "#555", dx = -4)
		dice = join(["$(i): $(d1[i])+$(d2[i])=$(sums[i])" for i in 1:16], ", ")
		# frequencies of the sums
		r2 = MersenneTwister(4); M = 2000
		cnt = zeros(Int, 11)
		for _ in 1:16M; cnt[rand(r2, 1:6)+rand(r2, 1:6)-1] += 1; end
		ex = [(6 - abs(s - 7)) / 36 for s in 2:12]
		g = fig(w = 330, h = 230, xr = (1.4, 12.6), yr = (0, 0.27), xlabel = "sum of two dice", ylabel = "probability", title = "how often each row is used", xticks = 2:12)
		bars!(g, 2:12, cnt ./ sum(cnt); width = 0.7, c = COL.blue, op = 0.6, stroke = "none")
		dots!(g, collect(2:12), ex; c = COL.ink, r = 3.5)
		hline!(g, 1 / 11; c = COL.red, dash = "5 3", w = 1.5)
		legend!(g, [("simulated, $(M) waltzes", COL.blue, :bar), ("uniform 1/11", COL.red, :dash)]; at = :tr, size = 10.5)
		p = prod((6 - abs(s - 7)) / 36 for s in sums[[1:7; 9:15]])
		info = Markdown.parse("**Your rolls** (bar: dice = sum): $(dice).")
		vstack(f, figure(g, vstack(play_notes(ev; label = "Play this waltz", seed = 7 + sd00_11), info));
			caption = md"**Figure 11.2.** Mozart's dice game. Top: the table of the 1792 edition; each number refers to one of 176 printed bars.")
	end
end;

# ╔═╡ 5d272b7c-9498-97d6-581f-14f29a4a7613
# rāga?
begin
	const RJ = [1, 16 / 15, 9 / 8, 6 / 5, 5 / 4, 4 / 3, 45 / 32, 3 / 2, 8 / 5, 5 / 3, 9 / 5, 15 / 8]   # just intonation

	const RPC = ["Sa", "re", "Re", "ga", "Ga", "Ma", "Ma′", "Pa", "dha", "Dha", "ni", "Ni"]               # lower case: komal

	const RSA = 220.0

	const RNAMES = ["Bhupali", "Deshkar", "Shuddh Kalyan", "Yaman"]

	const RCOL = [COL.ochre, COL.red, COL.green, COL.blue]

	const RDEF = Dict(
		"Bhupali" => (notes = [0, 2, 4, 7, 9], up = [0, 2, 4, 7, 9], down = [0, 2, 4, 7, 9], vadi = 4, samvadi = 9,
			w = Dict(0 => 1.4, 2 => 1.1, 4 => 2.4, 7 => 1.0, 9 => 1.5),
			pakad = [[4, 2, 0, -3, 0, 2, 4], [7, 4, 9, 7, 4, 2, 0], [0, 2, 4, 7, 4, 2, 0]], mood = "evening"),
		"Deshkar" => (notes = [0, 2, 4, 7, 9], up = [0, 2, 4, 7, 9], down = [0, 4, 7, 9], vadi = 9, samvadi = 4,
			w = Dict(0 => 1.0, 2 => 0.5, 4 => 1.5, 7 => 2.0, 9 => 2.6),
			pakad = [[7, 9, 12, 9, 7], [9, 4, 7, 4, 2, 0], [12, 9, 7, 9, 4, 7]], mood = "morning"),
		"Shuddh Kalyan" => (notes = [0, 2, 4, 6, 7, 9, 11], up = [0, 2, 4, 7, 9], down = [0, 2, 4, 6, 7, 9, 11], vadi = 4, samvadi = 9,
			w = Dict(0 => 1.4, 2 => 1.2, 4 => 2.2, 6 => 0.6, 7 => 1.0, 9 => 1.5, 11 => 0.7),
			pakad = [[4, 2, 4, 2, 0], [-1, -3, -5, 4, 2, 0], [4, 2, 4, 7, 2, 0]], mood = "evening"),
		"Yaman" => (notes = [0, 2, 4, 6, 7, 9, 11], up = [0, 2, 4, 6, 9, 11], down = [0, 2, 4, 6, 7, 9, 11], vadi = 4, samvadi = 11,
			w = Dict(0 => 1.3, 2 => 1.2, 4 => 2.2, 6 => 1.3, 7 => 0.8, 9 => 1.1, 11 => 1.7),
			pakad = [[-1, 2, 4, 2, 0], [7, 6, 4, 2, 0], [6, 9, 11, 12]], mood = "evening"))

	rnotes(rg) = [h for oct in -1:1 for h in RDEF[rg].notes .+ 12oct if -5 <= h <= 17]

	rfreq(h) = RSA * 2.0^fld(h, 12) * RJ[mod(h, 12)+1]

	"""Markov chain on the notes (small steps, rules of ascent and descent, emphasis on the important notes), with characteristic phrases and phrase endings on Sa, the vādī or the saṃvādī."""
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

	"""note frequencies and note-pair frequencies, estimated from a long simulation."""
	const RSTATS = let
		rng = MersenneTwister(2007)
		map(RNAMES) do rg
			m = rmelody(rg, 200_000, rng); pcs = [mod(h, 12) + 1 for (h, _) in m]
			p = fill(1e-3, 12); T = fill(1e-3, 12, 12)
			for (k, c) in enumerate(pcs)
				p[c] += 1; k > 1 && (T[pcs[k-1], c] += 1)
			end
			(p = p ./ sum(p), T = T ./ sum(T, dims = 2))
		end
	end

	function rposterior(pcs, model)
		lp = zeros(4); out = [fill(0.25, 4)]
		for (k, c) in enumerate(pcs)
			for r in 1:4
				lp[r] += model == "pairs" && k > 1 ? log(RSTATS[r].T[pcs[k-1], c]) : log(RSTATS[r].p[c])
			end
			q = exp.(lp .- maximum(lp)); push!(out, q ./ sum(q))
		end
		out
	end

	function revents(mel; beat = 0.42, drone = true)
		ev = Tuple{Float64,Float64,Float64,Float64,Int}[]; t = 0.4
		for (h, d) in mel
			push!(ev, (t, rfreq(h), d * beat, 0.8, 0)); t += d * beat
		end
		drone && append!(ev, [(0.0, RSA / 2, t + 2.5, 0.2, 4), (0.0, RSA * 3 / 8, t + 2.5, 0.13, 4)])
		ev
	end

	function rroll(mel, title; group = "", beat = 0.42, w = 680, h = 200, col = COL.blue)
		hs = [x[1] for x in mel]; lo = minimum(hs) - 1; hi = maximum(hs) + 1
		T = sum(last.(mel))
		rows = [x for x in lo:hi if mod(x, 12) in (0, 2, 4, 6, 7, 9, 11)]; row(x) = findfirst(==(x), rows)
		f = fig(w = w, h = h, xr = (0, T), yr = (0.5, length(rows) + 0.5), yticks = 1:length(rows), yticklabels = [RPC[mod(x, 12)+1] * (x >= 12 ? "\u0307" : x < 0 ? "\u0323" : "") for x in rows], xlabel = "beats", title = title, l = 44, r = 10)
		for x in rows
			mod(x, 12) == 0 && rect!(f, 0, row(x) - 0.5, T, row(x) + 0.5; fill = "#efebe0", op = 0.8)
		end
		t = 0.0
		for (x0, d) in mel
			x = row(x0)
			rect!(f, t + 0.04, x - 0.4, t + d - 0.04, x + 0.4; fill = col, op = 0.85, rx = 2)
			raw!(f, "<rect class=\"pn-hl\" data-g=\"$(group)\" data-a=\"$(0.4 + t * beat)\" data-b=\"$(0.4 + (t + d) * beat)\" x=\"$(_n(X(f, t)))\" y=\"$(_n(Y(f, x + 0.55)))\" width=\"$(_n(X(f, t + d) - X(f, t)))\" height=\"$(_n(Y(f, x - 0.55) - Y(f, x + 0.55)))\" fill=\"none\" stroke=\"$(COL.ink)\" stroke-width=\"2.2\"/>"; over = true)
			t += d
		end
		f
	end

	function figure_11_3(sd00_12)
		let
			panels = map(1:4) do r
				rg = RNAMES[r]; p = RSTATS[r].p
				sev = [0, 2, 4, 6, 7, 9, 11]
				f = fig(w = 330, h = 190, xr = (0.4, 7.6), yr = (0, 0.34), xticks = 1:7, xticklabels = RPC[sev .+ 1], ylabel = "probability", title = "$(rg) · $(RDEF[rg].mood)", l = 52, r = 6)
				for (k, c) in enumerate(sev .+ 1)
					p[c] > 0.002 && rect!(f, k - 0.36, 0, k + 0.36, p[c]; fill = RCOL[r], op = c - 1 in (RDEF[rg].vadi, RDEF[rg].samvadi) ? 0.95 : 0.55)
					p[c] <= 0.002 && text!(f, k, 0.01, "–"; anchor = "middle", size = 12, c = "#999")
				end
				mel = rmelody(rg, 22, MersenneTwister(3000 + 97r + 1009sd00_12))
				vstack(f, play_notes(revents(mel); label = "Listen to $(rg)", seed = r + sd00_12))
			end
			figure(panels...; caption = md"**Figure 11.3.** The four rāgas: how often each note occurs.")
		end
	end

	const RHOW = [4, 2, 0, 7, 9, 4, 2, 6, 4, 2, 11, 12]     # Ga Re Sa Pa Dha Ga Re Ma′ Ga Re Ni Ṡa

	function figure_11_how()
		let
			pcs = [mod(h, 12) + 1 for h in RHOW]; N = length(pcs)
			post = rposterior(pcs, "notes")
			W, H = 680, 470
			f = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)
			#  melody
			text!(f, 6, 18, "1 · the notes heard"; size = 13, bold = true)
			cw = (W - 12) / N
			for (i, h) in enumerate(RHOW)
				x0 = 6 + (i - 1) * cw; nm = RPC[mod(h, 12)+1] * (h >= 12 ? "̇" : "")
				rect!(f, x0 + 2, 28, x0 + cw - 2, 58; fill = "#ecebe6", op = 1, rx = 6)
				rect!(f, x0 + 2, 28, x0 + cw - 2, 58; fill = COL.ink, op = 1, rx = 6, s = 2i - 1)
				text!(f, x0 + cw / 2, 48, nm; anchor = "middle", size = 13, c = "#888")
				text!(f, x0 + cw / 2, 48, nm; anchor = "middle", size = 13, bold = true, c = "white", s = 2i - 1)
				rect!(f, x0, 25, x0 + cw, 61; fill = "none", stroke = COL.red, sw = 3, op = 1, rx = 7, o = (2i - 1, 2i))
			end
			# ... fingerprints
			text!(f, 6, 90, "2 · how often each rāga plays this note"; size = 13, bold = true)
			sev = [0, 2, 4, 6, 7, 9, 11]; pw = (W - 12 - 30) / 4; top = 104; bot = 222
			pmax = maximum(maximum(RSTATS[r].p) for r in 1:4)
			for r in 1:4
				x0 = 6 + (r - 1) * (pw + 10); bw = pw / 7
				text!(f, x0 + pw / 2, top + 6, RNAMES[r]; anchor = "middle", size = 12, bold = true, c = RCOL[r])
				seg!(f, x0, bot, x0 + pw, bot; c = "#999", w = 1)
				for (k, c) in enumerate(sev .+ 1)
					bx = x0 + (k - 1) * bw; v = RSTATS[r].p[c]
					hgt = v / pmax * (bot - top - 22)
					v > 0.002 ? rect!(f, bx + 2, bot - hgt, bx + bw - 2, bot; fill = RCOL[r], op = 0.45) :
						text!(f, bx + bw / 2, bot - 4, "–"; anchor = "middle", size = 12, c = "#aaa")
					text!(f, bx + bw / 2, bot + 13, RPC[c]; anchor = "middle", size = 9.5, c = "#666")
					for i in 1:N
						pcs[i] == c || continue
						v > 0.002 ? rect!(f, bx + 1, bot - hgt - 1, bx + bw - 1, bot; fill = RCOL[r], op = 1, stroke = COL.ink, sw = 2.2, o = (2i - 1, 2i)) :
							text!(f, bx + bw / 2, bot - 16, "✕"; anchor = "middle", size = 16, bold = true, c = COL.red, o = (2i - 1, 2i))
					end
				end
			end
			# ── band 3: the belief
			text!(f, 6, 262, "3 · the computer's belief: old belief × bar height, then rescaled"; size = 13, bold = true)
			bl = 120; br = W - 70; rowy(r) = 270 + 44 * (r - 1)
			for r in 1:4
				y = rowy(r)
				text!(f, bl - 8, y + 23, RNAMES[r]; anchor = "end", size = 12.5, bold = true, c = RCOL[r])
				rect!(f, bl, y + 8, br, y + 32; fill = "#f1f1ee", op = 1)
			end
			for k in 0:N
				fr = k == N ? (2k, 2k) : (2k, 2k + 1)
				for r in 1:4
					y = rowy(r); v = post[k+1][r]; old = k == 0 ? v : post[k][r]
					k > 0 && rect!(f, bl, y + 8, bl + old * (br - bl), y + 32; fill = "none", stroke = "#555", sw = 1.2, op = 1, o = 2k)
					rect!(f, bl, y + 8, bl + v * (br - bl), y + 32; fill = RCOL[r], op = 0.85, o = fr)
					if k > 0
						mark = v < 0.004 ? "✕ ruled out" : v > old * 1.03 ? "▲" : v < old * 0.97 ? "▼" : "="
						text!(f, max(bl + v * (br - bl), bl) + 8, y + 25, mark; size = 13, bold = true, c = mark == "▲" ? COL.green : mark == "=" ? "#777" : COL.red, o = 2k)
					end
				end
			end
			# ── step texts (words only)
			reads = String["Before any note, the computer has no preference: all four rāgas are equally likely."]
			for i in 1:N
				c = pcs[i]; nm = "<b>" * RPC[c] * "</b>"
				zero = [RNAMES[r] for r in 1:4 if RSTATS[r].p[c] <= 0.002]
				hi = argmax([RSTATS[r].p[c] for r in 1:4])
				push!(reads, "Note $(i): $(nm). Look up its bar in each distribution. " * (isempty(zero) ? "All four rāgas use this note; it is most typical of <b>$(RNAMES[hi])</b>." : "<b>" * join(zero, "</b> and <b>") * "</b> never play$(length(zero) == 1 ? "s" : "") it."))
				push!(reads, "Each belief is multiplied by its bar and rescaled: a tall bar makes the belief grow (▲), a short bar makes it shrink (▼)" * (isempty(zero) ? "." : ", and a missing note rules the rāga out (✕)."))
			end
			player(f; n = 2N, start = 0, fps = 0.8, reads = reads, label = "step",
				caption = md"**How the computer decides.** A short phrase is heard note by note. For each note, the computer looks up how often each rāga plays it (Figure 11.3) and multiplies its belief in that rāga by this bar; then it rescales the four beliefssuch that the sum is 1.")
		end
	end

	function rmystery(sd00_13)
		rng = MersenneTwister(4000 + 1009sd00_13); r = rand(rng, 1:4)
		(r = r, mel = rmelody(RNAMES[r], 28, rng))
	end

	function figure_11_4(s11_myst, s11_model)
		let
			mel = s11_myst.mel; pcs = [mod(h, 12) + 1 for (h, _) in mel]
			post = rposterior(pcs, s11_model == "which note follows which (pairs)" ? "pairs" : "notes")
			R = rroll(mel, "the mystery melody"; group = "myst", col = COL.grey)
			B = fig(w = 680, h = 190, xr = (0, 1), yr = (4.6, 0.4), axes = false, l = 130, r = 70, t = 26, b = 10, title = "the computer's belief after each note (press play)")
			for r in 1:4
				text!(B, 0, r, RNAMES[r]; anchor = "end", dx = -8, size = 13, bold = true, c = RCOL[r])
				rect!(B, 0, r - 0.32, 1, r + 0.32; fill = "#f1f1ee", op = 1)
				seg!(B, 0.25, r - 0.38, 0.25, r + 0.38; c = "#bbb", w = 1)
			end
			beat = 0.42; starts = cumsum(vcat(0.0, last.(mel)))
			for k in 0:length(mel)
				a = k == 0 ? 0.0 : 0.4 + starts[k] * beat
				b_ = k == 0 ? 0.4 : k < length(mel) ? 0.4 + starts[k+1] * beat : 0.4 + starts[end] * beat + 2.0
				for r in 1:4
					v = post[k+1][r]
					raw!(B, "<g class=\"pn-hl\" data-g=\"myst\" data-a=\"$(a)\" data-b=\"$(b_)\"><rect x=\"$(_n(X(B, 0)))\" y=\"$(_n(Y(B, r - 0.32)))\" width=\"$(_n(X(B, v) - X(B, 0)))\" height=\"$(_n(Y(B, r + 0.32) - Y(B, r - 0.32)))\" fill=\"$(RCOL[r])\" fill-opacity=\"0.85\"/><text x=\"$(_n(X(B, 1) + 8))\" y=\"$(_n(Y(B, r) + 4))\" font-size=\"13\" fill=\"#222\">$(round(Int, 100v)) %</text></g>"; over = true)
				end
			end
			vstack(R, play_notes(revents(mel); label = "Play the mystery melody", seed = 7, group = "myst"), B;
				caption = md"**Figure 11.4.** A melody in one of the four rāgas, chosen at random. During playback, the bars show the probability of each rāga given the notes heard so far, computed with Bayes' theorem; all four start at 25 %.")
		end
	end

	function figure_11_5(s11_myst, s11_model)
		let
			mel = s11_myst.mel; pcs = [mod(h, 12) + 1 for (h, _) in mel]
			f = fig(w = 680, h = 230, xr = (0, length(mel)), yr = (0, 1.02), xlabel = "number of notes heard", ylabel = "probability", title = "belief in each rāga, note by note", l = 52)
			for (m, sty) in (("notes", nothing), ("pairs", "5 3"))
				post = rposterior(pcs, m)
				for r in 1:4
					line!(f, 0:length(mel), [q[r] for q in post]; c = RCOL[r], w = m == "notes" ? 2.2 : 1.6, dash = sty)
				end
			end
			legend!(f, vcat([(RNAMES[r], RCOL[r]) for r in 1:4], [("dashed: note pairs", "#666", :dash)]); at = :br, size = 10)
			pn = rposterior(pcs, "notes"); pp = rposterior(pcs, "pairs")
			first90(post) = something(findfirst(q -> q[s11_myst.r] > 0.9, post), length(post) + 1) - 1
			n1 = first90(pn); n2 = first90(pp)
			vstack(Markdown.parse("The melody was in rāga **$(RNAMES[s11_myst.r])**. Listening to note frequencies, the computer was more than 90 % sure after " * (n1 <= length(mel) ? "**$(n1)** notes" : "the whole melody did not reach 90 %") * "; listening to note pairs, after " * (n2 <= length(mel) ? "**$(n2)** notes." : "the whole melody did not reach 90 %.")), f)
		end
	end

	function raga_keyboard()
		# probabilities below 1e-6 are set to 1e-6...
		# then has the same tiny probability under all four and leaves the beliefs unchanged
		P = [max.(round.(RSTATS[r].p, digits = 6), 1e-6) for r in 1:4]
		T = [[max.(round.(RSTATS[r].T[i, :], digits = 6), 1e-6) for i in 1:12] for r in 1:4]
		@htl("""<div class="rk" style="font-family:system-ui,sans-serif;max-width:680px">
		<div class="rk-keys" style="display:flex;gap:4px;flex-wrap:wrap;margin:6px 0"></div>
		<div style="display:flex;gap:8px;align-items:center;margin:6px 0;flex-wrap:wrap">
		<button class="rk-drone">drone: off</button><button class="rk-reset">start again</button>
		<label>the computer listens to <select class="rk-model"><option value="notes">how often each note occurs</option><option value="pairs">which note follows which</option></select></label></div>
		<div class="rk-played" style="min-height:1.4em;color:#555;font-size:0.95em"></div>
		<div class="rk-note" style="min-height:1.3em;color:#9a3b2e;font-size:0.92em"></div>
		<svg class="rk-bars" viewBox="0 0 680 170" width="680" style="max-width:100%;height:auto"></svg>
		<script>
		const root = currentScript.parentElement;
		const P = $(P), T = $(T), names = $(RNAMES), cols = $(RCOL), pcs = $(RPC), J = $(RJ), SA = $(RSA);
		let ctx = null, drone = null, played = [], lp = [0, 0, 0, 0];
		const keys = root.querySelector(".rk-keys"), svg = root.querySelector(".rk-bars"), model = root.querySelector(".rk-model"), shown = root.querySelector(".rk-played");
		function audio() { ctx = ctx || new (window.AudioContext || window.webkitAudioContext)(); return ctx; }
		function pluck(f) {
			const c = audio(), sr = c.sampleRate, L = Math.floor(2.2 * sr), P = Math.max(2, Math.round(sr / f)), y = new Float32Array(L);
			let l = 0; for (let i = 0; i < P; i++) { l = 0.5 * l + 0.5 * (2 * Math.random() - 1); y[i] = l; }
			for (let i = P; i < L; i++) y[i] = 0.996 * 0.5 * (y[i - P] + y[i - P - 1 >= 0 ? i - P - 1 : i - P]);
			for (let i = 0; i < L; i++) y[i] = 0.5 * Math.tanh(2 * y[i]);
			const b = c.createBuffer(1, L, sr); b.copyToChannel(y, 0); const s = c.createBufferSource(); s.buffer = b; s.connect(c.destination); s.start();
		}
		function toggleDrone(btn) {
			const c = audio();
			if (drone) { drone.forEach(o => o.stop()); drone = null; btn.textContent = "drone: off"; return; }
			drone = []; const g = c.createGain(); g.gain.value = 0.05; g.connect(c.destination);
			const lpf = c.createBiquadFilter(); lpf.type = "lowpass"; lpf.frequency.value = 900; lpf.connect(g);
			for (const [f, det] of [[SA / 2, 0], [SA / 2, 3], [SA * 3 / 8, 0], [SA / 4, -2]]) { const o = c.createOscillator(); o.type = "sawtooth"; o.frequency.value = f; o.detune.value = det; o.connect(lpf); o.start(); drone.push(o); }
			btn.textContent = "drone: on";
		}
		function draw() {
			const m = Math.max(...lp), q = lp.map(v => Math.exp(v - m)), s = q.reduce((a, b) => a + b, 0);
			let h = "";
			for (let r = 0; r < 4; r++) {
				const v = q[r] / s, y = 10 + r * 40;
				h += `<text x="125" y="\${y + 20}" text-anchor="end" font-size="14" font-weight="bold" fill="\${cols[r]}">\${names[r]}</text>`;
				h += `<rect x="135" y="\${y}" width="460" height="28" fill="#f1f1ee"/><rect x="135" y="\${y}" width="\${(460 * v).toFixed(1)}" height="28" fill="\${cols[r]}" fill-opacity="0.85"/>`;
				h += `<text x="605" y="\${y + 19}" font-size="14" fill="#222">\${Math.round(100 * v)} %</text>`;
			}
			svg.innerHTML = h;
			const last = played.length ? played[played.length - 1] % 12 : -1;
			root.querySelector(".rk-note").textContent = last >= 0 && P.every(p => p[last] <= 1e-6) ? "The note " + pcs[last] + " is used by none of the four rāgas. It is equally unlikely under all four, so it has no information: the beliefs do not change." : "";
			shown.textContent = played.length ? "notes played: " + played.map(c => c === 12 ? "Ṡa" : pcs[c]).join(" · ") : "Click the notes below. The computer starts at 25 % for each rāga.";
		}
		function recompute() {
			lp = [0, 0, 0, 0];
			played.forEach((c0, k) => { const c = c0 % 12; for (let r = 0; r < 4; r++) lp[r] += Math.log(model.value === "pairs" && k > 0 ? T[r][played[k - 1] % 12][c] : P[r][c]); });
			draw();
		}
		for (let c = 0; c <= 12; c++) {
			const b = document.createElement("button"), pc = c % 12;
			b.textContent = c === 12 ? "Ṡa" : pcs[pc];
			b.style.cssText = "min-width:46px;padding:10px 4px;border-radius:6px;border:1px solid #bbb;font-size:14px;cursor:pointer;" + ([1, 3, 6, 8, 10].includes(pc) ? "background:#3a3a3a;color:white" : "background:#fbfaf6");
			b.addEventListener("click", () => { pluck(SA * (c === 12 ? 2 : J[pc])); played.push(c); recompute(); });
			keys.appendChild(b);
		}
		root.querySelector(".rk-drone").addEventListener("click", e => toggleDrone(e.target));
		root.querySelector(".rk-reset").addEventListener("click", () => { played = []; recompute(); });
		model.addEventListener("change", recompute);
		invalidation.then(() => { if (drone) drone.forEach(o => o.stop()); });
		draw();
		</script></div>""")
	end
end;

# ╔═╡ f0e20a59-7d49-73d1-8d02-3367ea065c32
figure_11_how()

# ╔═╡ df771203-9b78-d191-63cf-6eaed2431be7
raga_keyboard()

# ╔═╡ f83a5449-e79c-b196-a31e-868728129769
# Questions, answers
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

	reveal(body; title = "Answer") = Markdown.MD(Markdown.Admonition("hint", title, Any[body]))
end;

# ╔═╡ 3554be30-3ea7-6e49-89b8-281b6268e7ac
reveal(md"""For ``n`` coin tosses, there are ``n-1`` pairs of neighboring tosses. Each toss has a probability of ``\tfrac12`` of being different from the previous toss.

Therefore, the number of switches (for example, Heads to Tails or Tails to Heads) is the sum of ``n-1`` independent outcomes. Its average value is ``(n-1)/2``, and its distribution is **binomial**, which becomes approximately **bell-shaped** for large ``n`` (See Notebook 1)

The longest run is different because it is a *maximum*, and not a sum. It cannot be smaller than ``1`` and, for a large number of tosses, is usually around ``\log_2 n``. Sometimes an unusually long run can be encountered, so its distribution has a **longer tail to the right**.""")


# ╔═╡ 3e0cceb1-9da8-0f3c-d0cd-3e04fc3e55a5
reveal(md"""A deterministic model describes averages. It is okay when the quantity of interest depends only on average behaviour and the random fluctuations about it are small. A stochastic model is needed when the quantity of interest depends on the fluctuations themselves, because an average does not show them. Whenever the answer depends on fluctuations (queues, extinction, failures, extremes), the model must contain the randomness.""")

# ╔═╡ bb94335b-965b-1b71-4625-dde2945f33b0
mcq(md"A shop manager plans with a deterministic model and finds that the clerk is busy 80 % of the time. The manager concludes that a second clerk is unnecessary because there is spare capacity. Which statement is correct?",
	md"The conclusion is correct: at a load below 100 %, no waiting line can form." => md"A line forms whenever customers arrive faster than they can be served, even if only for a short time.",
	md"The conclusion may be wrong: random arrivals create waiting lines even at a load of 80 %." => md"Right. The average load says nothing about the fluctuations, and the fluctuations cause the waiting.",
	md"The conclusion is wrong because the deterministic model overestimates the load." => md"Both models have exactly the same load; they differ only in the timing of the arrivals.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 62cdaefa-7628-542d-6d7b-881a3f64cd2c
reveal(figure_3_2(); title = "Answer")

# ╔═╡ 24a9ef40-c328-932c-ca92-21806dc4f9bd
reveal(md"""Neither value alone. Each simulation is one random outcome; the difference between them shows how large the random variation is. One reports the average over many independent simulations together with its uncertainty, for example the standard error.""")

# ╔═╡ d18f1c8e-3a84-c14f-346f-8f2c9e091951
mcq(md"A simulation with 10 000 runs gives a result with a standard error of 0.02. How many runs are needed for a standard error of 0.01?",
	md"20 000" => md"Doubling the number of runs reduces the error only by a factor ``\sqrt 2``, to about 0.014.",
	md"40 000" => md"Right. The error is proportional to ``1/\sqrt n``; halving it requires four times as many runs.",
	md"100 000" => md"Ten times as many runs reduce the error by ``\sqrt{10} \approx 3.2``, to about 0.006.";
	correct = 2, head = "Check your understanding")

# ╔═╡ e15957d2-199a-5a42-a3ab-a9028a430407
mcq(md"A fair die has been rolled 600 times, and so far a six appeared only 80 times instead of the expected 100. What is the expected number of sixes in the next 600 rolls?",
	md"More than 100, because the die has to make up for the missing sixes." => md"The die has no memory of earlier rolls. Each roll is a six with probability ``1/6``, whatever happened before.",
	md"100" => md"Right. The law of large numbers does not work by compensation. The early deficit of 20 sixes is not corrected; it becomes negligible *relative* to the increasing total.",
	md"Fewer than 100, because the die appears to be biased against sixes." => md"For a fair die, 80 sixes in 600 rolls are unusual but possible. If the die is known to be fair, the expectation for the next rolls is 100.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 28fb3e9a-86d0-49d4-3672-4d502fc92561
reveal(md"""Whatever move the computer chooses in a round, the random player's move is independent of it and equally likely to be rock, paper or scissors. The computer therefore wins, draws and loses each with probability ``\tfrac13`` in every round, independently of its strategy. Information about the past is useless because the past contains no information about the next move.""")

# ╔═╡ 0e2b2d47-a23c-b51f-ef72-e4df50bdbb4a
reveal(md"""The computer only compares the four rāgas it knows; it cannot answer *none of them*. Its probabilities are correct only if the model is correct: the four rāgas are the only possibilities, the notes follow the estimated frequencies, and successive notes are independent. When these assumptions fail, Bayes' theorem still produces a confident number, but the confidence is about the model, not about the music. A probability computed from a model must always be read together with the assumptions of the model.""")

# ╔═╡ 1722f956-b420-818a-e485-4bdb66637944
reveal(md"""For every rāga ``r`` it computes ``P(r \mid \text{notes}) \propto P(r) \prod_k P(\text{note}_k \mid r)``, the prior times the likelihood of the notes heard (with pairs: ``\prod_k P(\text{note}_k \mid \text{note}_{k-1}, r)``). A note that a rāga never uses has probability (almost) zero under that rāga, so one such note multiplies its probability by (almost) zero. The certainty is only as good as the model: the computer assumes that the melody comes from one of exactly four rāgas and that its notes follow the estimated probabilities. A melody in a fifth rāga, or a deliberately misleading one, will also be assigned to one of the four, possibly with high confidence. **Probabilities computed from a model describe the model, not necessarily the world.**""")

# ╔═╡ 405e9e8a-2097-4bb6-076a-df5dd0da1d35
# waiting for a pattern
begin
	function figure_7_1(s7_res, s7_p)
		a, b = s7_res.a, s7_res.b; N = length(a); A, B_ = patstr(s7_res.pa), patstr(s7_res.pb)
		ea, eb = expected_wait(s7_res.pa, s7_p), expected_wait(s7_res.pb, s7_p)
		kmax = clamp(round(Int, quantile(vcat(a, b), 0.97)), 8, 120)
		ks = 1:kmax; fa = [count(==(k), a) / N for k in ks]; fb = [count(==(k), b) / N for k in ks]
		f = fig(w = 680, h = 280, xr = (0.3, kmax + 0.7), yr = (0, 1.2max(maximum(fa), maximum(fb))), xlabel = "number of tosses until the pattern appears", ylabel = "relative frequency")
		wd = 0.4
		bars!(f, ks .- wd / 2, fa; width = wd, c = COL.blue, op = 0.85, stroke = "none")
		bars!(f, ks .+ wd / 2, fb; width = wd, c = COL.red, op = 0.85, stroke = "none")
		vline!(f, mean(a); c = COL.blue, dash = "5 3", w = 1.8); vline!(f, mean(b); c = COL.red, dash = "5 3", w = 1.8)
		legend!(f, [("$(A): mean $(fmt(mean(a); d = 3))", COL.blue, :bar), ("$(B_): mean $(fmt(mean(b); d = 3))", COL.red, :bar)]; at = :tr)
		txt = Markdown.parse("Simulated means: **$(fmt(mean(a); d = 3))** tosses for $(A) and **$(fmt(mean(b); d = 3))** for $(B_).")
		figure(f, txt; caption = md"**Figure 9.1.** Distribution of the number of tosses until each pattern appears for the first time, over many simulated experiments. Dashed lines: simulated means.")
	end

	function compute_s7_res(sd00_7, s7_pa, s7_pb, s7_p, s7_N)
		pa, pb = parse_pattern(s7_pa), parse_pattern(s7_pb)
		pa = isempty(pa) || length(pa) > 6 ? [true, true] : pa
		pb = isempty(pb) || length(pb) > 6 ? [true, false] : pb
		rng = MersenneTwister(2 + 1000sd00_7)
		(pa = pa, pb = pb, a = [wait_sim(rng, pa, s7_p) for _ in 1:s7_N], b = [wait_sim(rng, pb, s7_p) for _ in 1:s7_N])
	end

	function figure_7_2(s7_p)
		hp = "H " * fmt(s7_p); tp = "T " * fmt(1 - s7_p)
		mk(title, back) = begin
			f = fig(w = 330, h = 200, xr = (0, 10), yr = (0, 6), axes = false, l = 6, r = 6, t = 26, b = 6, title = title)
			pos = Dict(:s => (1.5, 3.0), :h => (5.0, 3.0), :d => (8.6, 3.0))
			for (k, (x, y)) in pos
				circle!(f, x, y, 22; fill = k == :d ? "#e6f0e9" : "#ffffff", stroke = COL.ink, sw = 1.6, over = true)
				text!(f, x, y, Dict(:s => "start", :h => "H", :d => "done")[k]; anchor = "middle", size = 12, bold = true)
			end
			arrow!(f, 2.3, 3.3, 4.2, 3.3; c = COL.ochre, w = 2); text!(f, 3.25, 3.9, hp; anchor = "middle", size = 11.5, c = COL.ochre)
			arrow!(f, 5.8, 3.0, 7.8, 3.0; c = COL.green, w = 2.4); text!(f, 6.8, 3.6, back ? hp : tp; anchor = "middle", size = 11.5, c = COL.green)
			raw!(f, "<path d=\"M $(_n(X(f, 1.1))) $(_n(Y(f, 3.75))) C $(_n(X(f, 0.2))) $(_n(Y(f, 5.6))), $(_n(X(f, 2.4))) $(_n(Y(f, 5.6))), $(_n(X(f, 1.85))) $(_n(Y(f, 3.8)))\" fill=\"none\" stroke=\"$(COL.grey)\" stroke-width=\"1.6\"/>"; over = true)
			text!(f, 1.4, 5.55, tp; anchor = "middle", size = 11.5, c = COL.grey)
			if back
				raw!(f, "<path d=\"M $(_n(X(f, 4.6))) $(_n(Y(f, 2.25))) C $(_n(X(f, 4.0))) $(_n(Y(f, 0.6))), $(_n(X(f, 2.4))) $(_n(Y(f, 0.6))), $(_n(X(f, 1.9))) $(_n(Y(f, 2.2)))\" fill=\"none\" stroke=\"$(COL.red)\" stroke-width=\"2\"/>"; over = true)
				text!(f, 3.25, 0.75, tp * ": back to start"; anchor = "middle", size = 11.5, c = COL.red)
			else
				raw!(f, "<path d=\"M $(_n(X(f, 4.6))) $(_n(Y(f, 3.75))) C $(_n(X(f, 3.9))) $(_n(Y(f, 5.6))), $(_n(X(f, 6.1))) $(_n(Y(f, 5.6))), $(_n(X(f, 5.4))) $(_n(Y(f, 3.8)))\" fill=\"none\" stroke=\"$(COL.blue)\" stroke-width=\"2\"/>"; over = true)
				text!(f, 5.0, 5.55, hp * ": stay at H"; anchor = "middle", size = 11.5, c = COL.blue)
			end
			f
		end
		figure(mk("waiting for HH", true), mk("waiting for HT", false); caption = md"**Figure 9.2.** The two waiting problems as diagrams.")
	end

	function answer(guess_hh, guess_ht)
		g1 = guess_hh > 0 ? "Your estimates: $(guess_hh) tosses for HH and $(guess_ht) for HT. " : ""
		adm("key", "Result", Markdown.parse(g1 * "The expected numbers of tosses are **6 for HH** and **4 for HT**. Equal probability does not imply equal expected waiting time."))
	end
end;

# ╔═╡ 06cb0b9a-a27b-6917-2138-dc57684c09e0
# Controls
begin
	Seed(default::Integer = 0; label = "New random numbers") = CounterButton(label)
end;

# ╔═╡ 9f151c03-40e3-ef9c-0dd6-139e4dedb40f
md"""
$(@bind sd00_0 Seed(0))
"""

# ╔═╡ c0115aa5-2236-3e1b-d41d-682d7e9ee1b7
figure_0_1(sd00_0, s0_seq)

# ╔═╡ f25326a1-6984-c1ae-f099-d42c496d98b0
md"""
## 3. Why models need randomness?

**Why should a model contain randomness at all, if a deterministic model with average values is simpler?** We consider a small example: a ticket counter where one clerk serves customers.

- Serving one customer takes exactly **1 minute**.
- Customers arrive at an average rate ``\lambda`` per minute, for example 0.9 per minute. The ratio ``\lambda / 1`` is called the **load** of the counter.

In the **deterministic model**, a customer arrives exactly every ``1/\lambda`` minutes, for example every 1.11 minutes. Each customer is served before the next one comes, and nobody waits. In the **stochastic model**, the customers arrive on average at the same rate, but at random moments, *independently* of each other. Both models have the same averages. Figure 3.1 compares these models.

Load ``\lambda`` $(@bind s1_load Select([0.5, 0.7, 0.8, 0.9, 0.95]; default = 0.9)) ``\qquad`` $(@bind sd00_1 Seed(0))
"""

# ╔═╡ 8fb24130-1d7d-ae68-5d73-ca356cbebbe1
figure_1_1(s1_load, sd00_1)

# ╔═╡ ead3929e-2d78-545a-6a58-3843d3a8be95
md"""
## 4. Unpredictable individuals, predictable ensembles

If every single event is random, how can a stochastic model predict anything? A device built by Francis Galton in 1889 describes this. Balls fall through rows of pegs, and at every peg a ball is deflected to the left or to the right with equal probability.

Rows of pegs $(@bind s2_rows Select([4, 6, 10, 14]; default = 10)) ``\qquad`` $(@bind sd00_2 Seed(0))
"""

# ╔═╡ 8587e28e-2bd3-de2e-c978-38c7bc1537d4
figure_2_1(s2_rows, sd00_2)

# ╔═╡ c1956916-7c2e-8f9d-0599-16de4fd8f01e
md"""
The path of a single ball cannot be predicted, but the distribution of many balls over the bins can. A ball lands in bin ``k`` if it is deflected to the right exactly ``k`` times in ``n`` rows. There are ``\binom{n}{k}`` sequences of deflections with this property, and each has probability ``2^{-n}``. The fraction of balls in bin ``k`` therefore approaches

```math
P(K = k) = \binom{n}{k}\, 2^{-n}, \qquad k = 0, 1, \dots, n,
```

the **binomial distribution**. More on this in Notebook 1.

The Galton board illustrates the central idea: individual events are **random**, but when we look at many events together, a clear distribution appears. A **stochastic model** describes the ensemble (the whole collection of possible results), and a **stochastic simulation** produces individual realisations, from which properties of the ensemble can be estimated.

## 5. Probability distributions

Random quantities differ in how their values are spread. The number of heads in 20 tosses, the waiting time for a bus, and the body height of adults are all random, but their values are distributed in different ways. A rule that assigns a probability to each possible value, or to each range of values, is called a **probability distribution**. The phenomena below are simulated (synthetic data).

Phenomenon $(@bind s3_kind Select(DISTS; default = "waiting time for the next bus")) ``\qquad`` $(@bind sd00_3 Seed(0))
"""

# ╔═╡ 994ca157-4fb4-c28e-cccb-92af816d9f6e
figure_3_1(s3_kind, sd00_3)

# ╔═╡ d96b6def-77c6-1458-5853-56d13fe81f0a
md"""
## 6. One experiment is not enough

A simulation is an experiment, and its result is random. We roll a fair die ``n`` times and compute the average of the rolls. The average approaches the expected value 3.5, but how fast? Figure 6.1 shows the running averages of 20 independent experiments.

Number of rolls per experiment $(@bind s4_N Select([10, 100, 1000, 10_000]; default = 1000)) ``\qquad`` $(@bind sd00_4 Seed(0))
"""

# ╔═╡ d34fe2af-aea7-e9cf-c8bc-3aab1b16a26a
figure_4_1(s4_N, sd00_4)

# ╔═╡ 4f5e8f33-86be-dac5-d277-5d51be38253d
md"""
## 9. Waiting for a pattern

!!! game "Predict"
    A fair coin is tossed repeatedly. **What is the expected number of tosses until two heads in a row (HH) appear for the first time?** What is the expected number until HT appears? At any given position, both patterns have probability ``\tfrac12 \cdot \tfrac12 = \tfrac14``.

    Your estimate for HH: $(@bind guess_hh NumberField(0:0.5:50; default = 0)) tosses ``\qquad`` for HT: $(@bind guess_ht NumberField(0:0.5:50; default = 0)) tosses

Enter your estimates, then run the simulation. Other patterns and a biased coin can be selected as well.

Pattern A $(@bind s7_pa TextField(6; default = "HH")) ``\quad`` pattern B $(@bind s7_pb TextField(6; default = "HT")) ``\quad`` probability of heads $(@bind s7_p Slider(0.1:0.05:0.9; default = 0.5, show_value = true)) ``\quad`` experiments $(@bind s7_N Select([100, 1000, 10_000, 100_000]; default = 10_000)) ``\quad`` $(@bind sd00_7 Seed(0))
"""

# ╔═╡ 86ed671b-417f-8b6a-4406-8e6572f30d29
s7_res = compute_s7_res(sd00_7, s7_pa, s7_pb, s7_p, s7_N);

# ╔═╡ d3404b31-2fa1-999d-52d0-8629c6a71d74
figure_7_1(s7_res, s7_p)

# ╔═╡ 39a2cb16-bcfd-f007-12be-bfa89744492e
figure_7_2(s7_p)

# ╔═╡ 032f8fc9-78a8-b617-5c19-f7f4f73d409d
answer(guess_hh, guess_ht)

# ╔═╡ 6f320240-f45b-1d85-ef5f-4ebd66ba8da8
md"""
## 11. Randomness in music

Randomness is not only a property of natural systems; it can also be used to create. We consider three examples from music: a sound produced from random numbers, a composition produced with dice, and a computer that recognises a rāga of Indian classical music using probability.

### 11.1 A plucked string made of random numbers

In 1983 Kevin Karplus and Alex Strong described a very simple method to produce the sound of a plucked string on a computer. It begins with a short sequence of random numbers whose length corresponds to one period of the desired tone. For example, a frequency of 220 Hz at a sampling rate of 22,050 samples per second requires approximately 100 numbers. This sequence is played repeatedly, with each number replaced by the average of its current value and that of its neighbor. Over time, this averaging smooths the random noise, producing a sound similar to a plucked string.

**Warning:** Turn down your volume before playing the sound, as the initial random noise can be loud.

Tone $(@bind s9_note Select(["A3 (220 Hz)", "E4 (330 Hz)", "A4 (440 Hz)"]; default = "A3 (220 Hz)")) ``\qquad`` $(@bind sd00_9 Seed(0))
"""

# ╔═╡ a3aa1cc7-64ec-0fc1-b886-3352e585f61c
figure_9_1(s9_note, sd00_9)

# ╔═╡ b462da24-a94f-88ef-0842-57d8d683da9a
md"""
$(@bind sd00_11 Seed(0; label = "Roll the dice"))
"""

# ╔═╡ 5736c25f-3e32-77e6-f722-34ad01e82c63
figure_11_2(sd00_11)

# ╔═╡ 2e5bbffb-25ca-d20c-6871-4452706ffcc3
md"""
### 11.3 Can probability recognise a rāga?

In Indian classical music, a **rāga** is a melodic framework, defined by its notes, ascending and descending patterns, hierarchy of important notes, and phrases. An experienced listener recognises a rāga after a few phrases. Remarkably, a computer can do the same from note statistics. P. Chordia and A. Rae (2007) analysed about 20 hours of recordings of 31 rāgas with statistical classifiers. Counting **how often each note occurs** identified the rāga of a one-minute excerpt in about three cases out of four; adding **which note follows which** increased the accuracy to 97–99 % in their tests, and about 75 % of excerpts from performances that the system had never heard were still recognised. Here the computer uses the simple concept of **Bayes' theorem.**

This section uses four rāgas that are hard to distinguish, because they have most of their notes in common. **Bhupali** and **Deshkar** use the same five notes, Sa Re Ga Pa Dha; they differ only in which notes are emphasised more and how the melody changes. **Shuddh Kalyan** is like Bhupali in ascent but adds Ma′ and Ni in descent, and **Yaman** uses all seven notes. The melodies are generated by a Markov chain that follows the rules of each rāga; the note statistics are estimated from a long simulation of this chain.

#### Part 1 · Four rāga

Every rāga has its own probability distribution over the twelve notes of the octave. Listen to each and compare the bars.

$(@bind sd00_12 Seed(0; label = "New melodies"))
"""

# ╔═╡ d91c4e3a-a01f-ac4f-6fe5-a9e7023d308e
figure_11_3(sd00_12)

# ╔═╡ 7e9c31f0-373d-d625-bf28-32ef5c045fb9
md"""
#### Part 2 · The mystery rāga

The computer secretly chooses one of the four rāgas and plays a melody in it. With every note it hears, it updates its belief with **Bayes' theorem**: the probability of each rāga is multiplied by the probability that this rāga produces the note just heard, and the four results are rescaled to sum to 1. Before pressing play, decide which rāga you think it is, and watch whether you or the computer is faster.

The computer listens to $(@bind s11_model Select(["how often each note occurs", "which note follows which (pairs)"]; default = "how often each note occurs")) ``\qquad`` $(@bind sd00_13 Seed(0; label = "New mystery"))
"""

# ╔═╡ 56f347c0-28b7-e27a-8de6-325ac15b6142
s11_myst = rmystery(sd00_13);

# ╔═╡ 3191d2c5-dcaf-1f57-f8f5-9b7ce602e1c1
figure_11_4(s11_myst, s11_model)

# ╔═╡ f3519896-92af-dbc4-269c-f66b578fa607
reveal(figure_11_5(s11_myst, s11_model); title = "Which rāga was it, and how fast did the computer know?")

# ╔═╡ f2d18cc3-9cf9-d2c5-4e6e-7f740c20840b
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
table.tbl{border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:0.92rem;margin:6px 0}
table.tbl th{border-bottom:1px solid #1b1b1b;padding:2px 12px;text-align:right;font-weight:600}
table.tbl td{padding:2px 12px;text-align:right}
table.tbl td:first-child,table.tbl th:first-child{text-align:left}
table.tbl tr:nth-child(even) td{background:#f6f6f3}
.pn{display:inline-block;margin:4px 0}.pn-b{font-size:0.95rem;padding:3px 12px;cursor:pointer}
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
project_hash = "c8794b78a8ebf0e2fccad160bc858a50a8463900"

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
# ╟─f1c82cb2-1756-b057-4a10-2eb3b0fc6686
# ╟─6193d783-e07f-1669-7e17-28aeeb5e7982
# ╟─9f151c03-40e3-ef9c-0dd6-139e4dedb40f
# ╟─c0115aa5-2236-3e1b-d41d-682d7e9ee1b7
# ╟─af5fe0f4-e492-bcb5-d1ac-3f04d91b9ddc
# ╟─241e69c2-fe6d-130a-29c2-5629bdcb90bf
# ╟─3554be30-3ea7-6e49-89b8-281b6268e7ac
# ╟─f25326a1-6984-c1ae-f099-d42c496d98b0
# ╟─8fb24130-1d7d-ae68-5d73-ca356cbebbe1
# ╟─78a5c5f0-4f7e-4b7f-b86e-0a92b7786012
# ╟─1b5b4b1a-4f65-b2d2-dfab-638f119d67fc
# ╟─3e0cceb1-9da8-0f3c-d0cd-3e04fc3e55a5
# ╟─bb94335b-965b-1b71-4625-dde2945f33b0
# ╟─ead3929e-2d78-545a-6a58-3843d3a8be95
# ╟─8587e28e-2bd3-de2e-c978-38c7bc1537d4
# ╟─c1956916-7c2e-8f9d-0599-16de4fd8f01e
# ╟─994ca157-4fb4-c28e-cccb-92af816d9f6e
# ╟─2e2cb45a-21e2-a2af-edea-c4659f9aa114
# ╟─62cdaefa-7628-542d-6d7b-881a3f64cd2c
# ╟─d96b6def-77c6-1458-5853-56d13fe81f0a
# ╟─d34fe2af-aea7-e9cf-c8bc-3aab1b16a26a
# ╟─9c38abe1-e7e9-37c0-c159-36d93a15dca4
# ╟─46107205-57a4-76cf-314d-440c6e968207
# ╟─24a9ef40-c328-932c-ca92-21806dc4f9bd
# ╟─d18f1c8e-3a84-c14f-346f-8f2c9e091951
# ╟─e15957d2-199a-5a42-a3ab-a9028a430407
# ╟─ac8a5206-b005-db56-4160-e8061ec2a977
# ╠═35a3d731-d1ce-b451-27eb-960fb19082b0
# ╟─75d055de-b272-66cb-d589-17dd51321fed
# ╠═617dc83e-3505-235e-7d62-f8737a779b17
# ╟─dee8c1e2-2c42-e34d-fad9-42417702f3eb
# ╟─c2c53e4e-ce74-a24f-595e-c74bb526dd5c
# ╠═c57862d7-c138-379c-7437-dac2aae5b938
# ╟─4f5e8f33-86be-dac5-d277-5d51be38253d
# ╟─86ed671b-417f-8b6a-4406-8e6572f30d29
# ╟─d3404b31-2fa1-999d-52d0-8629c6a71d74
# ╟─51ef3675-db22-42b7-5dda-870d228da0cd
# ╟─39a2cb16-bcfd-f007-12be-bfa89744492e
# ╟─3341557a-d516-9b57-3a71-a1c5c4fa804f
# ╟─032f8fc9-78a8-b617-5c19-f7f4f73d409d
# ╟─b960034c-9a69-0296-24db-226d53c5a3b7
# ╟─f913a25f-68e6-6c6e-6755-a025e40c1086
# ╟─712b24fb-f85e-48b4-a08a-cf043a04370b
# ╟─4437b792-6c97-96aa-cd53-913ac13cc38b
# ╟─acdd423f-5442-a003-9c14-37b1f8f057b1
# ╟─312d8ed8-f90d-6f0f-ba41-a9826728f9a1
# ╟─e58560b7-c56c-100a-8a62-d8a68b4a5f2a
# ╟─28fb3e9a-86d0-49d4-3672-4d502fc92561
# ╟─6f320240-f45b-1d85-ef5f-4ebd66ba8da8
# ╟─a3aa1cc7-64ec-0fc1-b886-3352e585f61c
# ╟─f2984ab0-b30b-cca5-f920-9a6f9a252282
# ╟─b462da24-a94f-88ef-0842-57d8d683da9a
# ╟─5736c25f-3e32-77e6-f722-34ad01e82c63
# ╟─2e5bbffb-25ca-d20c-6871-4452706ffcc3
# ╟─d91c4e3a-a01f-ac4f-6fe5-a9e7023d308e
# ╟─7e9c31f0-373d-d625-bf28-32ef5c045fb9
# ╟─56f347c0-28b7-e27a-8de6-325ac15b6142
# ╟─3191d2c5-dcaf-1f57-f8f5-9b7ce602e1c1
# ╟─f3519896-92af-dbc4-269c-f66b578fa607
# ╟─f0e20a59-7d49-73d1-8d02-3367ea065c32
# ╟─c21f605e-1ffe-5b60-bc82-7d789af525aa
# ╟─df771203-9b78-d191-63cf-6eaed2431be7
# ╟─2619603e-a21e-719f-d925-686f72a498c9
# ╟─0e2b2d47-a23c-b51f-ef72-e4df50bdbb4a
# ╟─b6f153f1-0f52-ae9a-6447-d9b93b26a9ff
# ╟─1722f956-b420-818a-e485-4bdb66637944
# ╟─203fdbfd-57b7-2723-2a7a-8771f410ea18
# ╟─a3e07761-83cc-ccd0-23d7-1bf78ced6ca2
# ╟─249842c4-56ec-a847-415a-6b05b433f905
# ╟─55d4eb1e-f704-d206-8e72-27c6deae61c8
# ╠═1c721cb6-1518-5f08-14e2-3715f9e19718
# ╟─0df99053-9c4f-8048-c719-1068fd38abc0
# ╟─8ee00361-b280-9c3e-d72b-cab8912974ad
# ╟─869f7884-b749-642b-ef06-cebefe174a7b
# ╟─66d3f869-827a-73d7-3c8f-76e3eda1f123
# ╟─405e9e8a-2097-4bb6-076a-df5dd0da1d35
# ╟─2c84d8e1-9402-34f3-cf41-dacdaf409664
# ╟─f4fe5c11-2b9e-9980-bc5f-bc30e2351409
# ╟─5d272b7c-9498-97d6-581f-14f29a4a7613
# ╟─665c5cbe-b1b3-5670-7cfe-c50804a43840
# ╟─d31f2275-5a0a-0e59-1256-393f6cf1990e
# ╟─f83a5449-e79c-b196-a31e-868728129769
# ╟─06cb0b9a-a27b-6917-2138-dc57684c09e0
# ╟─f2d18cc3-9cf9-d2c5-4e6e-7f740c20840b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
