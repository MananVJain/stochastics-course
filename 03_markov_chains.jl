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

# ╔═╡ 40d2386f-b6ef-0f47-64c7-877e08ddc9e8
begin
	using PlutoUI, HypertextLiteral, Random, Statistics, LinearAlgebra, Markdown, Printf, Distributions, SHA, Base64
end

# ╔═╡ 6b97923e-96a0-5c85-7a8d-c87d675a9cd0
md"""
# 3 · Discrete-Time Markov Chains

*Stochastic Modeling and Simulation*
"""

# ╔═╡ 8c28c86c-d249-08a7-a333-1937c09a4777
md"""
Solutions password $(@bind pw PasswordField(default = ""))
"""

# ╔═╡ 2e1a3eaa-123f-de5a-aa60-2579de80a678
md"""
How can a process with memory be described? By the probability expansion, the probability of a path is always

```math
P(X_0 = x_0, \dots, X_n = x_n) = P(X_0 = x_0) \prod_{i=1}^{n} P\big(X_i = x_i \,\big|\, \textcolor{#c2462e}{X_{i-1} = x_{i-1}, \dots, X_0 = x_0}\big).
```

What each factor is conditioned on decides how much memory the process has:

- **Full history.** The factor depends on the entire past, as written above. With ``k`` states, the last factor alone needs a table with ``k^{n+1}`` entries: it grows exponentially with ``n`` and cannot be used.
- **i.i.d.** All conditions are dropped, ``P(X_i = x_i \mid \textcolor{#c2462e}{X_{i-1}, \dots, X_0}) = P(X_i = x_i)``. No dependence.
- **Markov chain.** Only the present matters for the next state, ``P(X_i = x_i \mid \textcolor{#c2462e}{X_{i-1}, \dots, X_0}) = P(X_i = x_i \mid \textcolor{#1f5aa6}{X_{i-1} = x_{i-1}})``, so the path probability is ``P(X_0 = x_0) \prod_{i=1}^{n} P(X_i = x_i \mid \textcolor{#1f5aa6}{X_{i-1} = x_{i-1}})``. Each factor is a ``k \times k`` table.

Markov chains are the useful compromise between the two extremes. Consecutive values can depend on each other, but only through one step.

## 2. Markov chains

### 2.1 The Markov property

!!! definition "Definition (Markov chain)"
    A stochastic process ``\{X_n\}`` with a discrete state space is a **Markov chain** if, for all ``n`` and all states,
    ```math
    P(X_{n+1} = j \mid X_n = i, X_{n-1} = x_{n-1}, \dots, X_0 = x_0) = P(X_{n+1} = j \mid X_n = i).
    ```
    The chain is **time-homogeneous** if these probabilities do not depend on ``n``; they are then denoted ``P_{ij}`` and called **transition probabilities**.

In words: **the conditional distribution of the next state, given the entire history, depends only on the present state.** Equivalently, the future and the past are conditionally independent given the present. The present state contains all the information about the past that is relevant for the future.

For a Markov chain, every factor of the probability expansion reduces to a transition probability, and the probability of a path becomes a product:

```math
P(X_0 = x_0, X_1 = x_1, \dots, X_n = x_n) = \pi_0(x_0)\, P_{x_0 x_1}\, P_{x_1 x_2} \cdots P_{x_{n-1} x_n},
```

where ``\pi_0`` is the distribution of the initial state. The whole process is therefore described by the initial distribution and the matrix ``P``. For example, a bicycle that starts at the Altmarkt follows the path Altmarkt → Campus → Campus → Neustadt with probability ``1 \cdot 0.25 \cdot 0.50 \cdot 0.15 = 0.01875``.
"""

# ╔═╡ 4b25057e-204a-820f-0d5e-fcf6ed8a1089
md"""
### 2.2 The transition matrix

The transition probabilities of a time-homogeneous chain with ``k`` states form the ``k \times k`` **transition matrix** ``P``. The figure shows the matrix of the bicycle chain and its **state diagram**, a graph with one node per state and one arrow per possible transition. Select a station and a row or a column.

Station $(@bind s2_sel Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1)) ``\qquad`` view $(@bind s2_mode Select(["a row: where does a bicycle go from here?", "a column: where do the bicycles arriving here come from?"]; default = "a row: where does a bicycle go from here?"))
"""

# ╔═╡ e2a970c2-42da-d032-1a40-445009c8f4c6
md"""
!!! think "Think"
    In a simulated path of 30 days the bicycle never visits one of the stations. Does this show that the probability of moving there is zero?
"""

# ╔═╡ 3837421e-3e08-bb9c-2319-1c76aff449bf
md"""
### 2.3 Markov chains as recursions

How is a path computed? Given the current state ``i``, the next state is drawn from **row ``i``** by the discrete inversion method: the interval ``[0, 1)`` is divided into pieces of lengths ``P_{i1}, P_{i2}, \dots``, a uniform random number ``U`` is drawn, and the chain moves to the state whose piece contains ``U``. The general result is: every Markov chain can be written as a recursion

```math
X_{n+1} = f(X_n, U_n)
```

with i.i.d. uniform random numbers ``U_n``. A uniform random number generator is therefore all that is needed to simulate any Markov chain.

Current station $(@bind s2_i Select(["Altmarkt" => "Altmarkt", "Campus" => "Campus", "Neustadt" => "Neustadt"]; default = "Altmarkt")) ``\qquad`` uniform number ``u`` $(@bind s2_u Slider(0:0.01:0.99; default = 0.7, show_value = true))
"""

# ╔═╡ 97d49dc2-4ee3-9c0c-b272-64d905ad243e
md"""
!!! info "Key points"
    - A Markov chain forgets everything except the current state: the next state depends only on the present one.
    - It is described by a transition matrix ``P`` with ``P_{ij} = P(X_{n+1} = j \mid X_n = i)``; each row is a probability distribution and sums to 1.
    - A path is simulated by drawing the next state from the row of the current state with one uniform number.
"""

# ╔═╡ f69b1902-ef36-8034-3fad-5a385c884dd6
md"""
## 3. Example: a Markov chain that writes text

The state of a Markov chain can be anything that can be listed, for example the last letter of a text. In 1913, A. A. Markov counted the letters of 20 000 characters of Pushkin's novel in verse *Eugene Onegin* to study such dependencies, and in 1948 C. E. Shannon used letter chains to found information theory. 

This requires two steps: **estimating** the transition matrix from a text, and **generating** new text with it.

### 3.1 Estimating the transition matrix

The state is the current character: one of the 26 letters or the space. For each pair of consecutive characters in a source text, the count ``C_{ij}`` (character ``i`` followed by character ``j``) is increased by one. Dividing each row by its total gives the estimated transition probabilities ``\hat P_{ij} = C_{ij} / \sum_k C_{ik}``. Figure 3.1 shows the resulting ``27 \times 27`` matrix for the first chapters of a book (about 28 000 characters; texts from Project Gutenberg). Select a character to see its row.

Source text $(@bind s3_corpus Select(["Alice's Adventures in Wonderland (Lewis Carroll, 1865)", "Pride and Prejudice (Jane Austen, 1813)", "Grimms' Fairy Tales (transl. E. Taylor and M. Edwardes)", "my own text"])) ``\qquad`` character $(@bind s3_ch Select(["space", "a", "e", "h", "o", "q", "t", "w"]; default = "t"))

My own text: $(@bind s3_my TextField((70, 3); default = "paste a paragraph of at least 200 characters in any language here"))
"""

# ╔═╡ b52f8abb-db65-40c2-b6b7-08b8c56e9d14
md"""The matrix is **sparse**: most of the ``27^2 = 729`` possible transitions never occur in the source. Each row is a probability distribution; for example, after ``t`` the most frequent continuations in English are ``h`` and the space."""


# ╔═╡ 5ce0c0fb-aa20-96c9-9236-8edf4dcdfb26
md"""
!!! think "Think"
    In a short text, the letter *z* is followed by *e* in all 3 of its occurrences, so the estimated probability ``\hat P_{ze}`` equals 1. How much would you trust this estimate, and how should such estimates be reported?
"""

# ╔═╡ 0996dcb8-2191-b9fd-aeb2-0744ebe3be4e
md"""
!!! think "Think"
    For characters, increase ``k`` from 1 to 8; for words, from 1 to 4. Observe the last column of the table below the text. Which memory gives readable text? At which memory does the chain begin to copy long passages of the source? Why does a longer memory make the chain less able to produce new text?
"""

# ╔═╡ 9d8f4a66-5575-5d76-3db0-625edb9894b8
md"""
Markov's own experiment used only two states: vowel and consonant. He found that the letters are clearly **dependent** (a vowel is much more likely after a consonant than after a vowel), and that the fraction of vowels nevertheless converges. The same experiment can be repeated with any text.

$(@bind s5_text TextField((70, 4); default = "It was a bright cold day in spring, and the river ran high between the stones. The ferryman waited at the landing for the last travellers of the evening, counting the coins in his pocket and watching the light fade over the hills. Nobody came. At last he untied the boat, pushed off from the bank and let the current carry him slowly towards the village, where the windows were already lit and the smell of bread drifted across the water."))
"""

# ╔═╡ d8ac51ee-c6d4-2d1c-3442-def3ae629472
md"""
!!! info "Key points"
    - Transition probabilities can be estimated from data by counting transitions and normalising each row.
    - A longer memory reproduces the data better but generalises worse: with few observations per state the chain copies its training text.
"""

# ╔═╡ 1f5e7ae5-68bf-a819-71da-5a1c5f271027
md"""
The generated melody uses only transitions that occur in the training melodies, with the observed frequencies. With a memory of one note, each step is plausible, but the melody wanders. With a memory of two notes, the chain reproduces short motifs of the training melodies and combines them in new orders; pooling several melodies gives it more possibilities at each step, and therefore more new combinations. The rhythm is part of the state, so the generated melody inherits the rhythmic patterns of the training melodies.

!!! think "Think"
    Train the chain on *Ode to Joy* alone with a memory of two notes, and then on all five melodies. Which version produces melodies that are closer to copies of the training data, and why? What happens when the chain reaches a state that was observed only once, or never as a current state?
"""

# ╔═╡ eca6d5d4-2592-10f3-1232-d9f09b3f25d2
md"""
The expected counts can be computed without simulation. Write the distribution of one bicycle on day ``n`` as a **row vector** ``\pi_n = \big(P(X_n = 1), P(X_n = 2), P(X_n = 3)\big)``. To be at station ``j`` tomorrow, the bicycle must be at some station ``i`` today and then move from ``i`` to ``j``. The law of total probability gives

```math
\pi_{n+1}(j) = \sum_i \pi_n(i)\,P_{ij}, \qquad\text{that is}\qquad \pi_{n+1} = \pi_n P, \qquad\text{and therefore}\qquad \pi_n = \pi_0 P^n .
```

The matrix power ``P^n`` is the **``n``-step transition matrix**: ``(P^n)_{ij}`` is the probability of moving from ``i`` to ``j`` in ``n`` steps. Moving from ``i`` to ``j`` in ``n + m`` steps means passing through some state ``k`` after ``n`` steps, which gives the **Chapman–Kolmogorov equation**

```math
P^{(n+m)}_{ij} = \sum_k P^{(n)}_{ik}\, P^{(m)}_{kj},
```

which is matrix multiplication. Figure 5.2 shows the rows of ``P^n``.

Steps ``n`` $(@bind s3_n Slider(0:1:25; default = 1, show_value = true))
"""

# ╔═╡ 206fefd1-c733-b9b6-7785-5e3f29cdf0c2
md"""
!!! info "Key points"
    - The distribution after one step is ``\pi_{n+1} = \pi_n P``, and after ``n`` steps ``\pi_n = \pi_0 P^n`` (Chapman–Kolmogorov).
    - The entry ``(P^n)_{ij}`` is the probability of being in ``j`` after ``n`` steps when starting in ``i``.
"""

# ╔═╡ d2337916-8ae9-6b2f-929e-076c5cbf0156
md"""
## 6. Properties of Markov chains

In Figure 5.2, the three rows of ``P^n`` become identical: after some time, the starting station no longer matters. This does not happen for every chain. Whether it happens depends on the structure of the state diagram. Figure 6.1 shows four typical cases.
"""

# ╔═╡ 71881e83-5f57-a444-d69a-6765a7423d5c
md"""
!!! info "Key points"
    - Whether a chain forgets its start depends on the structure of its state diagram: closed sets, irreducibility and periodicity.
    - A self-loop makes a state aperiodic; an absorbing state traps the chain forever.
"""

# ╔═╡ 35599b1e-d402-939f-07d9-e28e1868dc1d
md"""
## 7. The stationary distribution

### 7.1 Definition and convergence

A distribution ``\pi`` that does not change under one step of the chain,

```math
\pi P = \pi, \qquad \sum_i \pi_i = 1, \qquad \pi_i \ge 0,
```

is a **stationary distribution**. If moreover ``\pi_n \to \pi`` for every initial distribution, ``\pi`` is the **limiting distribution**. Figure 7.1 shows ``\pi_n`` for the bicycle chain from the three possible initial stations: all three converge to the same values, the stationary distribution.

Day ``n`` for the right panel $(@bind s7_n Slider(0:1:20; default = 0, show_value = true)) ``\qquad`` initial station for the right panel $(@bind s7_st Select([3 => "Neustadt", 1 => "Altmarkt", 2 => "Campus"]; default = 3))
"""

# ╔═╡ 4c82addf-9498-4582-7fd4-d0b0ef59a34d
md"""
Stationarity does not mean that the bicycles stop moving. It means that the **distribution** no longer changes: at every station, the expected number of bicycles arriving per day equals the expected number leaving. Figure 7.2 shows one step ``\pi_{n+1} = \pi_n P`` as a vector–matrix product. The entry of ``\pi_{n+1}`` for a station is the product of the row vector ``\pi_n`` with the **column** of that station: the sum, over all stations, of the share that is there today times the probability of moving to the selected station.

Station $(@bind s7_j Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1))
"""

# ╔═╡ 183d0998-0476-b6e9-95fd-dc93d7e8813c
md"""
!!! think "Think"
    A simulation of a chain is run for 50 steps from one initial state, and the final frequencies are reported as the stationary distribution. What could go wrong, and how would you check the result?
"""

# ╔═╡ 193c1a08-df48-f72d-956b-e6f4e587ae02
md"""
### 7.2 Computing the stationary distribution by hand

The condition ``\pi P = \pi`` is a system of linear equations. With ``\pi = (a, c, n)`` for Altmarkt, Campus and Neustadt, the product ``\pi P`` is evaluated column by column:

```math
\begin{aligned}
0.60\,a + 0.35\,c + 0.20\,n &= a \\
0.25\,a + 0.50\,c + 0.20\,n &= c \\
0.15\,a + 0.15\,c + 0.60\,n &= n
\end{aligned}
```

Each equation states: *expected inflow into a station = current share of the station*. The three equations are linearly dependent (adding them gives ``a + c + n = a + c + n``, because the rows of ``P`` sum to 1), so one of them is replaced by the normalisation ``a + c + n = 1``. The third equation gives ``0.4\,n = 0.15\,(a + c)``, so ``n = \tfrac38 (a + c)``. The first gives ``0.4\,a = 0.35\,c + 0.2\,n``. Substituting ``n`` yields ``0.4\,a = 0.35\,c + 0.075\,(a + c)``, so ``0.325\,a = 0.425\,c`` and ``c = \tfrac{13}{17}\,a``. Then ``n = \tfrac38 \cdot \tfrac{30}{17}\,a = \tfrac{45}{68}\,a``, and the normalisation gives

```math
\pi = \left(\tfrac{68}{165},\ \tfrac{52}{165},\ \tfrac{45}{165}\right) \approx (0.412,\ 0.315,\ 0.273).
```

For a chain with two states and ``P = \begin{pmatrix} 1-a & a \\ b & 1-b\end{pmatrix}``, the balance of the flows between the two states, ``\pi_1 a = \pi_2 b``, gives ``\pi = \left(\frac{b}{a+b}, \frac{a}{a+b}\right)`` directly.
"""

# ╔═╡ e54c6465-068f-8cc4-1aae-3ec637a1677f
md"""
### 7.3 Existence, uniqueness and convergence

!!! theorem "Theorem"
    A finite, irreducible Markov chain has a unique stationary distribution ``\pi``, with ``\pi_i > 0`` for all states. It has a limiting distribution if and only if it is also aperiodic, that is, ergodic; the limiting distribution is then ``\pi``.

The conditions for a chain to have a stationary distribution and to converge to it from any start: the chain must be **irreducible**, **aperiodic** and **positive recurrent** (the expected time to return to a state is finite; this is satisfied automatically for finite irreducible chains). If one of these condition fail:

- **Irreducible.** In the chain *two islands* of Figure 6.1, a walker never leaves the island on which it starts. Each island has its own stationary distribution, and the long-run behaviour depends on the start.
- **Aperiodic.** In the cycle of three, the walker is in state 1 only at times ``0, 3, 6, \dots``. The distribution ``\pi_n`` keeps rotating and never settles, although a stationary distribution exists.

The periodic chain of Figure 6.1 illustrates the difference between stationary and limiting distributions: ``(\tfrac13, \tfrac13, \tfrac13)`` is its unique stationary distribution, but a chain started in state 1 cycles through ``(1, 0, 0), (0, 1, 0), (0, 0, 1)`` and does not converge.

A second interpretation of ``\pi`` is important for simulation. For an ergodic chain, the **fraction of time** that one long chain spends in state ``j`` converges to ``\pi_j``. A single bicycle, followed for years, spends about 41 % of its days at the Altmarkt (compare Figure 2.2 with 365 days). Sampling from ``\pi`` is therefore possible by simulating one long chain: this is the principle of the Markov chain Monte Carlo methods.
"""

# ╔═╡ 1d6b3874-3f3b-0ce4-e954-72e305c504c3
md"""
!!! info "Key points"
    - A stationary distribution solves ``\pi P = \pi`` with ``\sum_i \pi_i = 1``: inflow equals outflow at every state.
    - A finite chain that is irreducible and aperiodic (ergodic) converges to its unique stationary distribution from every start.
    - For an ergodic chain the fraction of time spent in each state converges to ``\pi``; this is the basis of Markov chain Monte Carlo.
"""

# ╔═╡ ebc565f5-3b1e-bbb8-f77d-623b8a50d999
md"""
## 8. Solution of the guiding problem

**(a, b)** The bicycle chain is ergodic, so the distribution of every bicycle converges to the unique stationary distribution ``\pi \approx (0.412, 0.315, 0.273)``, independently of the initial station. The expected numbers are about **124 bicycles at the Altmarkt, 95 at the Campus and 82 at Neustadt**, independently of the initial placement.

**(c)** The expected number is not sufficient, because the actual number fluctuates. In the long run, each bicycle is at the Altmarkt with probability ``\pi_A = 0.412``, independently of the others, so the number there is binomially distributed with ``n = 300`` and ``p = 0.412``. The required capacity is its ``0.99``-quantile.

Accepted risk of overflow $(@bind s7_risk Select([0.05, 0.01, 0.001]; default = 0.01)) ``\qquad`` fleet size $(@bind s7_fleet Select([100, 300, 1000, 3000]; default = 300)) bicycles
"""

# ╔═╡ 15844db3-797f-f4e5-8bd1-2a9e6351c4d2
md"""
## 9. Application: absorbing chains and the duration of a game

Board games of the type *snakes and ladders* are Markov chains: the state is the square, a die moves the token, ladders and snakes are fixed jumps, and the last square is absorbing. Let ``Q`` be the transition matrix restricted to the non-absorbing squares and ``E_s`` the expected number of turns until the end, starting from square ``s``. First-step analysis gives ``E = \mathbf 1 + QE``, that is, ``(I - Q)E = \mathbf 1``. The powers ``P^n`` give the probability that the game has ended after ``n`` turns. The board can be edited: each jump is written as *from-to*.

Snakes and ladders $(@bind s8_jumps TextField(50; default = "3-16 8-21 24-29 17-4 19-7 27-9"))
"""

# ╔═╡ a7941415-0ec9-1224-1b07-ec4c654ef512
md"""
!!! game "Experiment · design a board"
    Design a board with at most three snakes and three ladders whose expected duration is as close as possible to **10 turns**. Then design the board with the longest expected duration. Which single jump has the largest effect, and why does its position matter?

## 10. Exercises

### 10.1 Calculation problems

**Problem 1.** For the bicycle chain, a bicycle starts at the Campus. (a) Compute the probability of the path Campus → Campus → Altmarkt → Neustadt. (b) Compute the distribution of the bicycle after two days. (c) Compute ``P(X_2 = \text{Campus} \mid X_0 = \text{Campus})``.
"""

# ╔═╡ ef43fac3-bee5-bd5a-4a36-22703979a08f
md"""
**Problem 2.** A chain has the transition matrix ``P = \begin{pmatrix} 0.9 & 0.1 \\ 0.5 & 0.5 \end{pmatrix}``. (a) Draw the state diagram. (b) Is the chain ergodic? (c) Compute the stationary distribution. (d) Compute ``P^2``.
"""

# ╔═╡ 8aea2228-5660-2a9f-f19b-be7345292b8e
md"""
**Problem 3.** Classify the chains with the following transition matrices (irreducible or reducible, periodic or aperiodic, absorbing states), and state whether a unique stationary distribution and a limiting distribution exist.

```math
P_1 = \begin{pmatrix} 0 & 1 & 0 \\ 0.5 & 0 & 0.5 \\ 0 & 1 & 0 \end{pmatrix}, \qquad
P_2 = \begin{pmatrix} 1 & 0 & 0 \\ 0.3 & 0.4 & 0.3 \\ 0 & 0 & 1 \end{pmatrix}, \qquad
P_3 = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0 & 0.5 & 0.5 \\ 0.5 & 0 & 0.5 \end{pmatrix}.
```
"""

# ╔═╡ bebfecf6-fd82-e30c-2633-62d0e4c6f4c4
md"""
**Problem 4.** A process takes the values 0 and 1. The value at time ``n + 1`` equals the value at time ``n - 1`` with probability 0.8 and is otherwise the opposite. (a) Is ``X_n`` a Markov chain?
"""

# ╔═╡ 4efc2fc9-f75b-9367-1af1-d7cffbddfb18
md"""
### 10.2 Programming exercises

**Exercise 1 · From counts to probabilities.** Complete `normalise_rows(C)`, which turns a matrix of transition counts into a transition matrix, as in Section 3.1.
"""

# ╔═╡ a1500e90-3e0b-b0d9-149b-bb2f2dcd4273
function normalise_rows(C)
	return missing   # your code here
end;

# ╔═╡ 0e09964e-b4fc-5277-8da0-8d9713a2ab21
md"""
**Exercise 2 · Debugging: simulating a chain.** `simulate_chain(P, x0, n, rng)` should return a path ``x_0, \dots, x_n``. The code reads a column of ``P`` instead of a row. Correct it.
"""

# ╔═╡ 22cbdf8e-ad9e-c486-a912-e426e469a245
function simulate_chain(P, x0, n, rng)
	x = zeros(Int, n + 1)
	x[1] = x0
	for t in 1:n
		x[t+1] = findfirst(>(rand(rng)), cumsum(P[:, x[t]]))
	end
	return x
end;

# ╔═╡ ea8f76f1-0ce7-8f6a-60e0-f2d325c02c98
md"""
**Exercise 3 · Debugging: the distribution after ``n`` steps.** `distribution_after(P, π0, n)` should return ``\pi_0 P^n``. Predict how the faulty result depends on ``n``, then correct the update.
"""

# ╔═╡ e7c22bb8-04fd-8b97-e48b-f71e2a480859
function distribution_after(P, π0, n)
	π = π0
	for _ in 1:n
		π = π0 * P
	end
	return π
end;

# ╔═╡ 5ae1ef4a-5819-edc1-bb9b-93de0f4649d8
md"""
### 10.3 Practice

!!! exam "Practice"
    **a)** What is the difference between an independent and identically distributed (i.i.d.) process and a Markov process? Explain in 1–2 sentences.

    **b)** Formally state the Markov property in words and as an equation. 

    **c)** What are the three properties a Markov process must possess in order to have a stationary distribution and to converge to it? Name them and describe each in words or equations.

    **d)** Write down the Chapman–Kolmogorov equation and explain its meaning in one sentence. 

    **e)** Compute the stationary distribution of the chain with ``P = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}``. 

    **f)** Explain why the rows, but not the columns, of a transition matrix sum to 1. 

## Further reading

| resource | content |
|:--|:--|
| [V. Powell and L. Lehe, *Markov chains explained visually*](https://setosa.io/ev/markov-chains/) | an animated chain with an editable transition matrix |
| [B. Hayes, *First links in the Markov chain*, American Scientist (2013)](https://www.americanscientist.org/sites/americanscientist.org/files/201321152149545-2013-03Hayes.pdf) | Markov's vowels in *Eugene Onegin* and the history of the idea |
| [C. M. Grinstead and J. L. Snell, *Introduction to Probability*, Chapter 11](https://math.dartmouth.edu/~prob/prob/prob.pdf) | absorbing and ergodic chains with worked examples |

## References

- I. F. Sbalzarini and C. Zechner, *Stochastic Modeling and Simulation*, lecture notes, TU Dresden, Chapter 3.
- A. A. Markov, "An example of statistical investigation of the text *Eugene Onegin* concerning the connection of samples in chains" (1913), English translation in *Science in Context* 19 (2006) 591–600.
- C. E. Shannon, "A mathematical theory of communication", *Bell System Technical Journal* 27 (1948) 379–423.
- Source texts: L. Carroll, *Alice's Adventures in Wonderland* (1865) and J. Austen, *Pride and Prejudice* (1813), from Project Gutenberg (public domain).
- Source text: *Grimms' Fairy Tales*, translated by E. Taylor and M. Edwardes, Project Gutenberg eBook #2591.
- Melodies of Section 4 (public domain, transcribed in C major for this notebook): L. van Beethoven, Symphony No. 9, Op. 125, fourth movement (1824), main theme; traditional folk songs *Alle meine Entchen*, *Hänschen klein*, *Ah vous dirai-je, maman* and *Frère Jacques*.
"""

# ╔═╡ 20fc9273-54c2-6f44-fb9f-7fd89862872d
TableOfContents(title = "Contents", aside = true, depth = 2)

# ╔═╡ 33e5ba05-da49-4cfd-17c1-9a8dc5c7a3ee
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

	parse_numbers(s) = [parse(Float64, t) for t in split(replace(s, "," => " ")) if tryparse(Float64, t) !== nothing]
end;

# ╔═╡ 3fa32784-82dd-fdf2-30a2-5d71a588950e
begin
	const BIKE_P = [0.6 0.25 0.15; 0.35 0.5 0.15; 0.2 0.2 0.6]

	const BIKE_NAMES = ["Altmarkt", "Campus", "Neustadt"]

	function chain_graph(P; names = string.(1:size(P, 1)), w = 380, h = 340, pos = nothing, rnode = 26, hl = 0)
		k = size(P, 1)
		if pos === nothing
			θ = [π / 2 - 2π * (i - 1) / k for i in 1:k]
			pos = [(cos(t), sin(t)) for t in θ]
		end
		f = fig(w = w, h = h, xr = (-1.95, 1.95), yr = (-1.75, 1.85), axes = false, l = 4, r = 4, t = 4, b = 4)
		sc = (w - 8) / 3.9
		L = maximum(length.(names)); lsz = L > 6 ? 11 : 12.5
		rnode = max(rnode, ceil(Int, 0.31 * lsz * L + 3))
		for i in 1:k, j in 1:k
			(i == j || P[i, j] <= 0) && continue
			(x0, y0), (x1, y1) = pos[i], pos[j]
			dx, dy = x1 - x0, y1 - y0; L = hypot(dx, dy); ux, uy = dx / L, dy / L
			ox, oy = 0.06uy, -0.06ux
			gap = (rnode + 4) / sc
			arrow!(f, x0 + gap * ux + ox, y0 + gap * uy + oy, x1 - gap * ux + ox, y1 - gap * uy + oy; c = COL.blue, w = 0.8 + 4P[i, j], head = 9, op = 0.8)
			text!(f, (x0 + x1) / 2 + 2.3ox, (y0 + y1) / 2 + 2.3oy, fmt(P[i, j]; d = 2); anchor = "middle", size = 11.5, c = "#1b3f73", bg = true)
		end
		for i in 1:k
			x, y = pos[i]
			if P[i, i] > 0
				r = hypot(x, y); ux, uy = r == 0 ? (0.0, 1.0) : (x / r, y / r)
				scy = (h - 8) / 3.6; d1 = rnode + 6; d2 = rnode + 6 + 0.17sc + 12
				cx, cy = x + d1 * ux / sc, y + d1 * uy / scy
				circle!(f, cx, cy, 0.17sc; fill = "none", stroke = COL.blue, sw = 0.8 + 4P[i, i], op = 0.6, over = true)
				text!(f, x + d2 * ux / sc, y + d2 * uy / scy, fmt(P[i, i]; d = 2); anchor = "middle", size = 11.5, c = "#1b3f73", bg = true)
			end
		end
		for i in 1:k
			x, y = pos[i]
			circle!(f, x, y, rnode; fill = i == hl ? COL.ochre : "#ffffff", stroke = COL.ink, sw = 1.6, over = true)
			text!(f, x, y, names[i]; anchor = "middle", size = lsz, bold = true, c = i == hl ? "white" : COL.ink)
		end
		f
	end

	function parse_matrix(str)
		rows = [parse_numbers(r) for r in split(str, r"[;\n]") if !isempty(strip(r))]
		k = length(rows)
		(k < 2 || k > 5 || any(length(r) != k for r in rows)) && return nothing
		P = permutedims(reduce(hcat, rows))
		any(P .< 0) && return nothing
		any(sum(P, dims = 2) .== 0) && return nothing
		P ./ sum(P, dims = 2)
	end

	function classify(P)
		k = size(P, 1); R = (P .> 0) .| Matrix{Bool}(I, k, k)
		for m in 1:k, i in 1:k, j in 1:k
			R[i, j] |= R[i, m] && R[m, j]
		end
		cls = Vector{Vector{Int}}(); seen = falses(k)
		for i in 1:k
			seen[i] && continue
			c = [j for j in 1:k if R[i, j] && R[j, i]]; seen[c] .= true; push!(cls, c)
		end
		closed = [all(!R[i, j] for i in c for j in setdiff(1:k, c)) for c in cls]
		per = map(cls) do c
			i = c[1]; Q = Matrix{Float64}(I, k, k); g = 0
			for n in 1:2k^2
				Q = Q * P
				Q[i, i] > 1e-12 && (g = gcd(g, n))
			end
			g
		end
		(classes = cls, closed = closed, period = per, irreducible = length(cls) == 1)
	end
end;

# ╔═╡ 06424fb2-40ef-acc3-a0b8-dedf4107751e
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

# ╔═╡ a2e99ae6-04b2-5c25-b415-68ca954ed85a
begin
	const MELODIES = Dict(
		"Ode to Joy (Beethoven, 1824)" => [(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (64, 3), (62, 1), (62, 4),
			(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (62, 3), (60, 1), (60, 4),
			(62, 2), (62, 2), (64, 2), (60, 2), (62, 2), (64, 1), (65, 1), (64, 2), (60, 2), (62, 2), (64, 1), (65, 1), (64, 2), (62, 2), (60, 2), (62, 2), (55, 4),
			(64, 2), (64, 2), (65, 2), (67, 2), (67, 2), (65, 2), (64, 2), (62, 2), (60, 2), (60, 2), (62, 2), (64, 2), (62, 3), (60, 1), (60, 4)],
		"Alle meine Entchen (German folk song)" => [(60, 2), (62, 2), (64, 2), (65, 2), (67, 4), (67, 4), (69, 2), (69, 2), (69, 2), (69, 2), (67, 8), (69, 2), (69, 2), (69, 2), (69, 2), (67, 8),
			(65, 2), (65, 2), (65, 2), (65, 2), (64, 4), (64, 4), (62, 2), (62, 2), (62, 2), (62, 2), (60, 8)],
		"Ah vous dirai-je, maman (French folk song)" => [(60, 2), (60, 2), (67, 2), (67, 2), (69, 2), (69, 2), (67, 4), (65, 2), (65, 2), (64, 2), (64, 2), (62, 2), (62, 2), (60, 4),
			(67, 2), (67, 2), (65, 2), (65, 2), (64, 2), (64, 2), (62, 4), (67, 2), (67, 2), (65, 2), (65, 2), (64, 2), (64, 2), (62, 4),
			(60, 2), (60, 2), (67, 2), (67, 2), (69, 2), (69, 2), (67, 4), (65, 2), (65, 2), (64, 2), (64, 2), (62, 2), (62, 2), (60, 4)],
		"Frère Jacques (French folk song)" => [(60, 2), (62, 2), (64, 2), (60, 2), (60, 2), (62, 2), (64, 2), (60, 2), (64, 2), (65, 2), (67, 4), (64, 2), (65, 2), (67, 4),
			(67, 1), (69, 1), (67, 1), (65, 1), (64, 2), (60, 2), (67, 1), (69, 1), (67, 1), (65, 1), (64, 2), (60, 2), (60, 2), (55, 2), (60, 4), (60, 2), (55, 2), (60, 4)],
		"Hänschen klein (German folk song)" => [(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (62, 2), (64, 2), (65, 2), (67, 2), (67, 2), (67, 4),
			(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (64, 2), (67, 2), (67, 2), (60, 8),
			(62, 2), (62, 2), (62, 2), (62, 2), (62, 2), (64, 2), (65, 4), (64, 2), (64, 2), (64, 2), (64, 2), (64, 2), (65, 2), (67, 4),
			(67, 2), (64, 2), (64, 4), (65, 2), (62, 2), (62, 4), (60, 2), (64, 2), (67, 2), (67, 2), (60, 8)])

	const MEL_ALL = "all five melodies"

	const NOTE_NAMES = Dict(48 => "C3", 50 => "D3", 52 => "E3", 53 => "F3", 55 => "G3", 57 => "A3", 59 => "B3", 60 => "C4", 62 => "D4", 64 => "E4", 65 => "F4", 67 => "G4", 69 => "A4", 71 => "B4", 72 => "C5")

	midi_hz(n) = 440.0 * 2.0^((n - 69) / 12)

	const CHORDS = [("C", [0, 4, 7], 48), ("F", [5, 9, 0], 53), ("G", [7, 11, 2], 55), ("Am", [9, 0, 4], 57), ("Dm", [2, 5, 9], 50)]

	function pianoroll(notes, title; w = 680, h = 190, pitches, group = "", eighth = 0.22, chords = nothing)
		T = sum(last.(notes))
		f = fig(w = w, h = h + (chords === nothing ? 0 : 14), xr = (0, T), yr = (0.5, length(pitches) + 0.5), xlabel = "time (eighth notes)", yticks = 1:length(pitches), yticklabels = [NOTE_NAMES[p] for p in pitches], l = 40, title = title, t = chords === nothing ? 30 : 44)
		for b in 0:8:T
			vline!(f, b; c = "#d8d8d8", dash = nothing, w = 1)
		end
		if chords !== nothing
			for (b, c) in enumerate(chords)
				text!(f, (b - 1) * 8 + 0.3, length(pitches) + 0.5, c; dy = -6, size = 11, bold = true, c = COL.ochre)
			end
		end
		t = 0
		for (p, d) in notes
			k = findfirst(==(p), pitches)
			rect!(f, t + 0.08, k - 0.36, t + d - 0.08, k + 0.36; fill = COL.purple, op = 0.85, rx = 2)
			raw!(f, "<rect class=\"pn-hl\" data-g=\"$(group)\" data-a=\"$(t * eighth)\" data-b=\"$((t + d) * eighth)\" x=\"$(_n(X(f, t)))\" y=\"$(_n(Y(f, k + 0.48)))\" width=\"$(_n(X(f, t + d) - X(f, t)))\" height=\"$(_n(Y(f, k - 0.48) - Y(f, k + 0.48)))\" fill=\"none\" stroke=\"$(COL.red)\" stroke-width=\"2.5\"/>"; over = true)
			t += d
		end
		f
	end

	function harmonise(notes)
		T = sum(last.(notes)); nb = cld(T, 8); score = zeros(nb, length(CHORDS)); t = 0
		for (p, d) in notes
			for q in 0:d-1
				b = (t + q) ÷ 8 + 1; b > nb && continue
				for (c, (_, pcs, _)) in enumerate(CHORDS)
					mod(p, 12) in pcs && (score[b, c] += (t + q) % 8 == 0 ? 2.0 : 1.0)
				end
			end
			t += d
		end
		ch = [argmax(score[b, :] .+ [0.3, 0.1, 0.2, 0, 0]) for b in 1:nb]
		ch[end] = 1
		nb > 1 && score[nb-1, 3] > 0 && (ch[nb-1] = 3)
		ch
	end

	function accompaniment(ch, eighth)
		ev = Tuple{Float64,Float64,Float64,Float64,Int}[]
		for (b, c) in enumerate(ch)
			_, pcs, root = CHORDS[c]; t0 = (b - 1) * 8 * eighth
			fifth = root - 5
			push!(ev, (t0, midi_hz(root - 12), 2eighth, 0.5, 0))
			push!(ev, (t0 + 4eighth, midi_hz(fifth), 2eighth, 0.42, 0))
			for (q, off) in ((2, 0), (6, 0))
				for pc in pcs[2:3]
					m = 52 + mod(pc - 52, 12)
					push!(ev, (t0 + q * eighth, midi_hz(m), 1.6eighth, 0.22, 0))
				end
			end
		end
		ev
	end

	function figure_3_2m(s4_mel, s4_ord, s4_acc, sd03_m)
		train = s4_mel == MEL_ALL ? [MELODIES[k] for k in sort(collect(keys(MELODIES)))] : [MELODIES[s4_mel]]
		mel = s4_mel == MEL_ALL ? MELODIES["Ode to Joy (Beethoven, 1824)"] : MELODIES[s4_mel]
		allnotes = reduce(vcat, train)
		pitches = sort(unique(vcat(first.(allnotes), 60))); K = length(pitches)
		ix = Dict(p => i for (i, p) in enumerate(pitches))
		ord2 = startswith(s4_ord, "2")
		# states are (pitch, duration) pairs; transitions are counted within each training melody
		f1 = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}(); f2 = Dict{NTuple{2,Tuple{Int,Int}}, Vector{Tuple{Int,Int}}}()
		C = zeros(K, K)
		for m in train
			for t in 1:length(m)-1
				push!(get!(f1, m[t], Tuple{Int,Int}[]), m[t+1]); C[ix[m[t][1]], ix[m[t+1][1]]] += 1
				t <= length(m) - 2 && push!(get!(f2, (m[t], m[t+1]), Tuple{Int,Int}[]), m[t+2])
			end
		end
		P = C ./ max.(sum(C, dims = 2), 1)
		rng = MersenneTwister(31 + 1009sd03_m); bars = 8; Ttot = 8bars
		gen = [first(rand(rng, train))]; t = gen[1][2]; restarts = 0
		while t < Ttot - 8
			cands = ord2 && length(gen) >= 2 ? get(f2, (gen[end-1], gen[end]), Tuple{Int,Int}[]) : get(f1, gen[end], Tuple{Int,Int}[])
			room = 8 - t % 8
			cands = [c for c in cands if c[2] <= room]
			if isempty(cands)
				cands = [c for c in get(f1, gen[end], Tuple{Int,Int}[]) if c[2] <= room]
				isempty(cands) && (cands = [(gen[end][1], room)]; restarts += 1)
			end
			nx = rand(rng, cands); push!(gen, nx); t += nx[2]
		end
		# the last bar: a cadence that ends on C
		rest = Ttot - t
		if rest > 4
			push!(gen, (rand(rng, [62, 67, 64]), rest - 4)); push!(gen, (60, 4))
		else
			push!(gen, (60, rest))
		end
		for (p, _) in gen
			haskey(ix, p) || (push!(pitches, p))
		end
		pitches = sort(unique(pitches))
		eighth = 0.24
		ev(notes) = (tt = 0.0; out = Tuple{Float64,Float64,Float64,Float64,Int}[]; for (p, d) in notes; push!(out, (tt * eighth, midi_hz(p), d * eighth, 0.8, 0)); tt += d; end; out)
		acc = s4_acc == "with accompaniment"
		chm = harmonise(mel); chg = harmonise(gen)
		R1 = pianoroll(mel, "training melody" * (s4_mel == MEL_ALL ? " (one of five): Ode to Joy" : ": $(s4_mel)"); pitches = pitches, group = "train", eighth = eighth, chords = acc ? first.(CHORDS[chm]) : nothing)
		R2 = pianoroll(gen, "generated melody (memory: $(s4_ord))"; pitches = pitches, group = "gen", eighth = eighth, chords = acc ? first.(CHORDS[chg]) : nothing)
		pk = sort(unique(first.(allnotes))); Kk = length(pk)
		H = fig(w = 330, h = 300, xr = (0.5, Kk + 0.5), yr = (Kk + 0.5, 0.5), xticks = 1:Kk, yticks = 1:Kk, xticklabels = [NOTE_NAMES[p] for p in pk], yticklabels = [NOTE_NAMES[p] for p in pk],
			xlabel = "next pitch", ylabel = "current pitch", title = "estimated pitch transitions", l = 46)
		for (i, p) in enumerate(pk), (j, q) in enumerate(pk)
			v = P[ix[p], ix[q]]; v > 0 || continue
			rect!(H, j - 0.48, i - 0.48, j + 0.48, i + 0.48; fill = COL.purple, op = 0.15 + 0.75v)
			Kk <= 9 && text!(H, j, i, fmt(v; d = 2); anchor = "middle", size = 10, c = v > 0.55 ? "white" : "#222")
		end
		nstates = length(f1)
		info = Markdown.parse("Training data: $(length(train)) melod$(length(train) == 1 ? "y" : "ies") with $(length(allnotes)) notes; $(nstates) different states (pitch, duration). " *
			"" * (restarts > 0 ? "" : "") *
			(acc ? "" : ""))
		evm(notes, ch) = acc ? vcat(ev(notes), accompaniment(ch, eighth)) : ev(notes)
		vstack(R1, play_notes(evm(mel, chm); label = "Play the training melody", seed = 2, group = "train"), figure(H, info), R2, play_notes(evm(gen, chg); label = "Play the generated melody", seed = 3, group = "gen");
			caption = md"**Figure 4.1.** Top: a training melody as a piano roll. Middle: the pitch transitions estimated from the training data (row = current pitch, column = next pitch). Bottom: a melody generated by the chain, whose state is the pitch and the duration of the last one or two notes. Chord symbols above the bars give the accompaniment.")
	end
end;

# ╔═╡ 994cbe9a-9b5c-5b9c-8847-970c28ba92a5
begin
	const SCOL = [COL.blue, COL.ochre, COL.green]

	_how_fig(W, H) = fig(w = W, h = H, xr = (0, W), yr = (H, 0), axes = false, l = 0, r = 0, t = 0, b = 0)

	_bw(s) = "<b>" * s * "</b>"

	const HOW_U = [0.68, 0.30, 0.88, 0.50]

	function figure_2_how(s2_i)
		i0 = something(findfirst(==(s2_i), BIKE_NAMES), 1)
		xs = [i0]
		for u in HOW_U
			push!(xs, findfirst(>(u), cumsum(BIKE_P[xs[end], :])))
		end
		S = length(HOW_U); NF = 3S + 1
		W, H = 680, 312
		f = _how_fig(W, H)
		text!(f, 10, 16, "the stations"; size = 13, bold = true)
		pos = [(140.0, 82.0), (222.0, 222.0), (58.0, 222.0)]; cen = (140.0, 175.0); rn = 31
		for i in 1:3, j in 1:3
			i == j && continue
			(x0, y0), (x1, y1) = pos[i], pos[j]
			dx, dy = x1 - x0, y1 - y0; L = hypot(dx, dy); ux, uy = dx / L, dy / L
			ox, oy = -6uy, 6ux
			a0 = (x0 + (rn + 4) * ux + ox, y0 + (rn + 4) * uy + oy); a1 = (x1 - (rn + 4) * ux + ox, y1 - (rn + 4) * uy + oy)
			arrow!(f, a0..., a1...; c = "#a9bcd9", w = 0.8 + 6BIKE_P[i, j], head = 9)
			for k in 1:S
				xs[k] == i && xs[k+1] == j && arrow!(f, a0..., a1...; c = COL.red, w = 2 + 6BIKE_P[i, j], head = 12, o = 3k)
			end
		end
		for i in 1:3
			x, y = pos[i]; d = hypot(x - cen[1], y - cen[2]); ux, uy = (x - cen[1]) / d, (y - cen[2]) / d
			lx, ly = x + (rn + 8) * ux, y + (rn + 8) * uy
			circle!(f, lx, ly, 13; fill = "none", stroke = "#a9bcd9", sw = 0.8 + 6BIKE_P[i, i])
			for k in 1:S
				xs[k] == i && xs[k+1] == i && circle!(f, lx, ly, 13; fill = "none", stroke = COL.red, sw = 2 + 6BIKE_P[i, i], o = 3k)
			end
		end
		for i in 1:3
			x, y = pos[i]
			circle!(f, x, y, rn; fill = SCOL[i], stroke = "white", sw = 2, over = true)
			text!(f, x, y, BIKE_NAMES[i]; anchor = "middle", size = 11, bold = true, c = "white")
		end
		for k in 0:S   # the black ring marks the bicycle; it moves in frame 3k
			x, y = pos[xs[k+1]]
			circle!(f, x, y, rn + 6; fill = "none", stroke = COL.ink, sw = 3.5, over = true, o = (3k, k == S ? NF : 3k + 2))
		end
		# ── right: the rows of P as sticks cut into pieces
		x0 = 380.0; wid = 280.0
		text!(f, 300, 16, "the rows of P, each cut into pieces"; size = 13, bold = true)
		rowy = [62.0, 110.0, 158.0]; hh = 15
		for r in 1:3
			c = [0.0; cumsum(BIKE_P[r, :])]; y = rowy[r]
			text!(f, x0 - 8, y, "from " * BIKE_NAMES[r]; anchor = "end", size = 11.5, c = "#666")
			for j in 1:3
				rect!(f, x0 + c[j] * wid, y - hh, x0 + c[j+1] * wid, y + hh; fill = SCOL[j], op = 0.22, stroke = "white", sw = 2)
			end
		end
		text!(f, x0, 186, "colour of a piece = station the bicycle goes to"; size = 11, c = "#666", italic = true)
		# the uniform number U: a ruler below the rows
		ry = 214.0
		text!(f, x0 - 8, ry, "random number U"; anchor = "end", size = 11.5, c = COL.red)
		seg!(f, x0, ry, x0 + wid, ry; c = "#999", w = 1.5)
		seg!(f, x0, ry - 5, x0, ry + 5; c = "#999", w = 1.5); seg!(f, x0 + wid, ry - 5, x0 + wid, ry + 5; c = "#999", w = 1.5)
		for k in 1:S
			r = xs[k]; c = [0.0; cumsum(BIKE_P[r, :])]; y = rowy[r]; j = xs[k+1]
			for jj in 1:3
				rect!(f, x0 + c[jj] * wid, y - hh, x0 + c[jj+1] * wid, y + hh; fill = SCOL[jj], op = 0.92, stroke = "white", sw = 2, o = (3k - 2, 3k))
			end
			rect!(f, x0 - 112, y - 10, x0 - 2, y + 10; fill = "white", op = 1, over = true, o = (3k - 2, 3k))
			text!(f, x0 - 8, y, "from " * BIKE_NAMES[r]; anchor = "end", size = 11.5, bold = true, c = COL.ink, o = (3k - 2, 3k))
			ux = x0 + HOW_U[k] * wid
			seg!(f, ux, ry - 6, ux, y + hh + 1; c = COL.red, w = 2.4, over = true, o = (3k - 1, 3k))
			text!(f, ux, ry + 1, "▲"; anchor = "middle", size = 15, c = COL.red, o = (3k - 1, 3k))
			rect!(f, x0 + c[j] * wid, y - hh - 3, x0 + c[j+1] * wid, y + hh + 3; fill = "none", stroke = COL.ink, sw = 3, op = 1, over = true, o = (3k - 1, 3k))
		end
		# ── bottom: the path so far
		py = 276.0
		text!(f, 292, py, "the path"; anchor = "end", size = 12, bold = true)
		for k in 0:S
			bx = 300 + 72k
			rect!(f, bx, py - 15, bx + 64, py + 15; fill = SCOL[xs[k+1]], op = 0.9, rx = 5, s = 3k)
			text!(f, bx + 32, py, BIKE_NAMES[xs[k+1]]; anchor = "middle", size = 11, bold = true, c = "white", s = 3k)
			k > 0 && text!(f, bx - 4, py, "›"; anchor = "middle", size = 14, c = "#888", s = 3k)
		end
		# ── step texts (words only)
		nm(i) = _bw(BIKE_NAMES[i])
		reads = ["The bicycle starts at $(nm(i0)) (black ring)."]
		for k in 1:S
			i, j = xs[k], xs[k+1]
			push!(reads, k == 1 ? "The bicycle is at $(nm(i)), so only the row from $(nm(i)) is used. Its pieces are as long as the transition probabilities." :
				"The bicycle is now at $(nm(i)): the row from $(nm(i)) is used. Where the bicycle was before plays no role.")
			push!(reads, i == j ? "The bicycle stays at $(nm(i)) (red loop)." : "The bicycle moves from $(nm(i)) to $(nm(j)) (red arrow).")
		end
		player(f; n = NF, start = 0, fps = 1, reads = reads, label = "step",
			caption = md"**How it works.** One day of the bicycle chain, repeated for a few days, starting at the station selected above.")
	end
end;

# ╔═╡ ba5d51f6-8c2a-09db-42c3-c615de9cc41d
figure_2_how(s2_i)

# ╔═╡ 9e20789c-f283-9cb2-9715-5b3edbf7232e
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

# ╔═╡ 183d13f9-2344-c1da-6c8a-acd99a55998c
mcq(md"Let ``\{X_n\}`` be a Markov chain. Which statement is correct?",
	md"``X_{n+1}`` is independent of ``X_{n-1}``." => md"In general it is not. The past influences the future, but only *through* the present.",
	md"``X_{n+1}`` is conditionally independent of ``X_{n-1}``, given ``X_n``." => md"Right. Once ``X_n`` is known, ``X_{n-1}`` has no additional information about ``X_{n+1}``. Without the condition, ``X_{n-1}`` and ``X_{n+1}`` are dependent.",
	md"The states ``X_0, X_1, \dots`` are independent." => md"That is an i.i.d. process, which is a very special Markov chain (all rows of ``P`` equal).";
	correct = 2, head = "Common pitfall")

# ╔═╡ 4ec681fb-1420-2683-a858-41d8b9436f2f
mcq(md"Which of these processes is **not** a Markov chain with the given state?",
	md"The capital of a player after each bet of 1 euro at even odds." => md"Tomorrow's capital is today's capital ``\pm 1``: it depends only on today. This random walk is a Markov chain.",
	md"The weather today, in a region where the probability of rain depends on the weather of the last two days." => md"Right. A memory of two days violates the Markov property for the state *weather today*.",
	md"The number shown by a die that is rolled once a day." => md"An i.i.d. process is a Markov chain whose transition matrix has identical rows.";
	correct = 2, head = "Check your understanding")

# ╔═╡ 4a9b637b-31f2-6055-e33e-c1a2c7ff0232
reveal(md"""No. A transition with a small probability may simply not occur in a short path. Absence in one simulated path is a weak evidence. Many simulations, longer paths, or a seeing the transition matrix itself answer the question.""")

# ╔═╡ 83c43687-d0d4-2021-055f-196fe8174a8e
reveal(md"""Very little. An estimate from 3 observations can be far from the true probability; another text would give a different row. A probability of exactly 1 also forbids all other continuations, which a generator then never produces. Estimated probabilities should be reported together with the counts they are based on, and rows with few counts can be smoothed with pseudo-counts""")

# ╔═╡ 7ee1306d-a6da-86b8-f840-640051025bdb
reveal(md"""For characters, ``k = 1`` or 2 reproduces letter frequencies and pairs; ``k = 4`` to 6 produces mostly real words. For words, ``k = 1`` gives real words in locally plausible order but without sentence structure; ``k = 2`` gives grammatical fragments that change topic frequently; ``k = 3`` or 4 gives long, correct sentences, most of which are copied from the source. With a longer memory, most contexts occur only once in a source of about 28 000 characters (5 000 words), so they have exactly one observed continuation, and the chain reproduces the source verbatim. The number of possible states grows much faster than the amount of data. The model then fits its training data exactly instead of capturing its regularities: this is **overfitting**, the central problem of statistical learning. Modern language models also predict the next word from the previous ones, but they share information between similar contexts and are trained on very large corpora.""")

# ╔═╡ 14969370-baf6-55f0-bc57-e14baf92ec69
reveal(md"""With a single training melody, most pairs of notes occur once and have a single observed continuation: the chain then follows the training melody note by note and differs from it only where a pair occurs at several places. Pooling five melodies creates more options, so the chain changes between melodies at shared motifs (for example the descending line E–D–C, which occurs in several of them). A state observed once has a row with a single entry 1, and the chain repeats what it saw. A state never observed as a current state has no row at all, and the chain cannot continue; in this implementation the current pitch is then held to the end of the bar. Rows estimated from few observations are unreliable: an unobserved transition receives probability 0, although its true probability may be positive.""")

# ╔═╡ 40e04f0e-1ecc-da30-1ca0-5c051b4f855a
numq(md"A bicycle is at Neustadt today. What is the probability that it is at the Altmarkt in **two** days?", (BIKE_P^2)[3, 1]; rtol = 0.005,
	why = md"``(P^2)_{NA} = 0.2 \cdot 0.6 + 0.2 \cdot 0.35 + 0.6 \cdot 0.2 = 0.31``: one term for each station where the bicycle can be tomorrow.", head = "Check your understanding")

# ╔═╡ 83f2505d-c074-ddbc-9a65-55508c05d710
mcq(md"Is the bicycle chain ergodic?",
	md"Yes: every station can be reached from every other, and the self-loops make it aperiodic." => md"Right. All transition probabilities are positive, so the chain is irreducible; ``P_{ii} > 0`` makes every state aperiodic.",
	md"No: it is periodic, because bicycles keep returning to the same stations." => md"Periodicity means that returns are possible *only* at multiples of some ``t > 1``. A bicycle can return after 1, 2, 3, … days.",
	md"No: the Altmarkt is absorbing, because most bicycles end up there." => md"A state is absorbing only if ``P_{ii} = 1``; here ``P_{AA} = 0.6``.";
	correct = 1, head = "Check your understanding")

# ╔═╡ 41dd7fa7-0a22-07e1-d2f8-c9d15f965201
reveal(md"""Fifty steps may be too few to forget the initial state, so the frequencies might still depend on the start. If the chain is periodic it never converges, and if it is reducible the result depends on where it started. Checks: start from several different initial states and compare; run longer and see whether the frequencies still change; and, for a small chain, verify ``\pi P = \pi`` directly.""")

# ╔═╡ 2ba8ad4c-8908-06e9-16c7-f78e6f027bb4
numq(md"A two-state weather chain: after a dry day, it rains with probability ``a = 0.2``; after a rainy day, it is dry with probability ``b = 0.4``. What fraction of days is rainy in the long run?", 0.2 / 0.6; rtol = 0.01,
	why = md"``\pi_{\text{rain}} = a/(a + b) = 0.2/0.6 = 1/3``.", head = "Check your understanding")

# ╔═╡ b4d99478-1cf5-6e3d-0335-e1d0b713ee96
solution(pw, "L/qtwVyZZoY3AO9af5s/KSg92S08E5NUxAPQ4zDHzPaJGSHqWzZeAGJ5hETWjhJHfP/hWHvmF2eSLAdJcIddFrqduQ0BirGxUJppGPjcNZqLeVHcx02w8BnOh3kzTLAaLcRtQ/dmDKwV5ZGgFDjNmw8d9bfYw9R5lekKb3METyC9SBFo2PwlP215uKlwe9Ag6gKQK1u8opeDEIfNqkgbV9U+lDJN0+hE8+D4tOjmlaFnx7IOoKcq3OmId86mOJfC+n30ZGHge3veThJBHQ203zUhyWV7MNEcYJLPxBnvjxOAbRkVWPpBMWMr8MaBBII5S/G488dw9haNGKA0M7GGExnDFVEgpB9o6sYjJ/ojv3Od4MR1iBJT+RPycSKxgrLNBDIAmdxGgmA9XKaE//t4OQfv3Y5Ek0+Z5hj4Y5QZsSXHhQlovT3uyK4j0MZENqc9xYV6bWs9n+Kws4n8Ba+w8PqSR1vxSV5EybgcH/IAVrABLOd/jW47fWBW1Q4HlUD1tgp9jo0lkeY+l4P8RmHO+3tsNawEMkMA0Mo5rTrum8HxAnJp6S/n3Qlz3jbE8Vjl4CPkJn+a4/I0eXY9sE9Z86q6WF9K"; title = "Solution to Problem 1")

# ╔═╡ 9afc3fbd-c863-ad7a-0201-731163f20e9f
solution(pw, "aEeH45qwBPP4P6DfjzQmgkhw2MjZ3pv6wlB+GNLnglJcXeqWG99yjhHcaC8cfXviDGbrJOK4gYle74CLqzylz4w0sjsVQ+EoVpyIJwDXgp29fizqF4evUauK/RxuCbKpOqVaKQlhc+QSB/askNcy8NOjDfE7t6s3XdUhaBGXg9eUXgi+6g2o0JbbWZaDSpkXPtJfhuiFU0IafEG+Z9EVO6Oc2fVus/zSVUzon3818hjTakvjxA/BEM78hXF7H2g4Y3fuFcAdwi241a+W5q5yrg7eXiqdj5EQRtrIj+29/AdcMVEHUDrcbXo2wEuKIkJA8j2aOXkszgJ6ucW/VU6Dj8IB6phb6y6xQ82VdeOW7R8cgbswUIWYaHEmlqS2cYDD2eJvLqWdo/PlSF1ZLHrSN0YHkXZtBvLYMoZqz2IvG5aaz2qh6D5bqI8jC8whhchOQlQlJqjAqa+AVzldPBAfNBH1nxMEYynJ6NDTBaPV7ZqUeQtGNNDndCYMuD24z1KYCiPZ08LJS5wZ+6be5Al7YL7s5iKecxZVqHRw/Q=="; title = "Solution to Problem 2")

# ╔═╡ 1a7c250b-d6bf-d28a-1fa3-304edfe2c38a
solution(pw, "isseyVNOQC9K9EYfdix7bcmscRvMuRZRQ9DV45L67OV857huYVhI6g+BGd0aKs+/aGW92iSt1zeKyh5Bz/rzeq4hL4CMde+yXGtWrRZIVohyKD7ui3bul6WvLTrueKiU9N/y8KbKcvWtKlEsfYG13vfHZpkiPVsAZk1q+BzqMjxtC0R/GBzbRT8OKZlxWP1hNdNwmhC84bBHXz0KnHMaEg5vaYXs18RFnzeqemKjR7fWeHaXF30eES2oa7/WujBzhENVmgueS4Qc9NBuB6YP+GGqdWNyWjD269ZMzuaBpxGAj2MszXvRZx6ZsPQu97Mp34XB7qbli+5PsoJcBPC0NcgTk8f3YtjR7siUfQSmzBYBRprZAJexq757UO1JP86YXnpB/V2tYRWZy92DPkv1+3uw/WaAAtUaSuUQw107G46Kt7rDZu0WEs6awe4LF+HzCP4tvKjzRcW7MFZAbYvAeew/gXB61+BcqboxMk3Rj5EQxAeIdVBo0ajP0AQK6BQ0OmSocdEhnd14Ck+nLvRJY+OZ4M03lZ5opbL2NvzhVVOi3B2QxFI4u6nM6d5YcX4doxSKRfQXxvxhkP3vHwSyW1xgnNhT8TxKarvqRcBNjGb1erEXYod0RZzfWgymNm5o5pKb52lb9i2IIXRFBUavgg=="; title = "Solution to Problem 3")

# ╔═╡ ebc7f26e-1018-9f03-13c9-10785abf9ea4
solution(pw, "AVZTkUoO3lFezeeXCFcNrmg/LVd2FRrXUUXMbaYlH6+ORkcj0ebeo9OiUisSke9pCNGY6Ue5NiD5yvc6ySi8xz9We8BR9D3etqUWRKoDQodLMo9p10wWRlK6OTU0Myl5F7z6g1oz2ED+gcimneXn1LiYDbhTgyXp6z8hK3ngVqh8UKWKAQ=="; title = "Solution to Problem 4")

# ╔═╡ a3bfd27c-6ee0-11c0-d28b-b7b3ef1f3e91
solution(pw, "Yzjx2DnpDuRQ0/cpaAAs5dpAr6A++JBfpe5CoGy0xJsbBvSMNSZty6LLVkdct4LjbsaXQKO478vTb7GSOcuyof0QNq+aszm0SmtIUFWPvqI8KTmFXLQbqx69ZkLMldoeZ77DJIp7jQ+n3nfD44lVW1iVkI9yApcX67LFyX5w4Bb9V95cld5GUyZ5Yqh8vt8x5aoAXDqlQD0QjYMdN9iL6zLX1iy+s5CMeIRxYoqJyqPQZiYXmIt/c4Qdtie1e1LshvN2dk4g27vDMJxkprhtGVONni4pqk5Cp/YBrDfpa6ageLG2PzQfX0069aaDS2wGEdDrGR6kLiBGrYYroHpmreFtCkO/FY5AT2vc3A+2QiBGi8iwcErIJKkPhu85"; title = "Solution to Exercise 1")

# ╔═╡ 69ff4c82-7cbe-af9c-8a8a-c73bb2d20eb9
solution(pw, "6hRg578QQ11jDV9fZY/K7ptShBaJyHyJI5TV3SWW00+O8wcxa+Z9sS8dU/8sxFZhGq9FA4XfFOj2TwdUnKJw9/dsml5+5e8kbFYAAF0xnB/8fMKLhVRtSsHYANpWo14nODL+FVnFq/+wlCdGpHgKIKQ2NeSDR/FCqf2UHrw0Od7w5J4v9Uq88EAv6wjVKQYT8L6a3/SsQLcReSDBeFMORDngDvVUCH6AsXRBYgxpHjGmoLoDnPSxi/HHrQmFinJV1ODWcqQVC0v/TcxH36JcizyubJNyWgxwxLhrW50V+u6YZDmqd6QVEWwkw7ajEwKPeqwgYfWVHitavyTOZCt13BhdISsf4iQKnuvjcsTpONm/HjBfYwqJhlx+PhxirMR0cXpuW/2xyAAVS7QGkBDrkrUL8OJJvOBhj0q2SyBHc+jvadyP+3PQeM3xa153XY/Er+qnHby19XNEIuywsDEI"; title = "Solution to Exercise 2")

# ╔═╡ 3c9dc54c-f230-b1b2-0b01-0df6751982b4
solution(pw, "qFhTkdkvGqgv8ykknz/lGUyFMD3K4BfRH9XmSv6oL591tckqGxcW6f4QhVbGscsQShCtfbuIuNR8nuU3lf0hLvPHEH2t2+1yqG8n09uqNxOk29xiRwnucu8qG6cF2RR6ZpXnVtxRNPQeNBNS7JjbnImqBbNFCDlb+NCpkCOoYI7GB7kKPGv3Mm6iFTwEm3480s5WwNAOHm/3HUFzhNvvo2MKa2Cug2+D/osruC188e5OBREOawmepfqBZwGYrvbT5Ij5U9PbBkpprN6esIKrJrH0l8f691uj7DU3kZsPnArULk0/VvuQkTsNM1QYNhQHc7jv"; title = "Solution to Exercise 3")

# ╔═╡ dc02448f-28ea-527e-d834-d383c6c8b5bb
begin
	function figure_1_1(s1_new, s1_r)
		let
			rng = MersenneTwister(31 + s1_new); n = 100
			coin = rand(rng, Bool, n)
			w = Vector{Bool}(undef, n); w[1] = rand(rng, Bool)
			for i in 2:n
				w[i] = rand(rng) < s1_r ? w[i-1] : rand(rng, Bool)
			end
			walk = cumsum(rand(rng, (-1, 1), n))
			strip(v, title, c1, c0, l1, l0) = begin
				f = fig(w = 680, h = 92, xr = (0.5, n + 0.5), yr = (0, 1), axes = false, l = 100, r = 6, t = 20, b = 6, title = title)
				for i in 1:n
					rect!(f, i - 0.45, 0.1, i + 0.45, 0.9; fill = v[i] ? c1 : c0, op = 0.9)
				end
				text!(f, 0.2, 0.5, "$(l1) / $(l0)"; anchor = "end", size = 11.5, c = "#555")
				f
			end
			runs(v) = (b = 1; c = 1; for i in 2:length(v); c = v[i] == v[i-1] ? c + 1 : 1; b = max(b, c); end; b)
			g = fig(w = 680, h = 180, xr = (0.5, n + 0.5), yr = (-20, 20), xlabel = "time n", ylabel = "position", l = 100, title = "random walk: position = previous position ± 1")
			hline!(g, 0; c = "#bbb", dash = nothing)
			line!(g, 1:n, walk; c = COL.purple, w = 1.8, steps = true)
			figure(strip(coin, "coin tosses (i.i.d.): longest run $(runs(coin))", COL.ochre, "#e8e8e3", "H", "T"),
				strip(w, "weather with memory: longest run $(runs(w)), changes of weather $(count(i -> w[i] != w[i-1], 2:n))", COL.blue, "#f2d38a", "rain", "dry"), g;
				caption = md"**Figure 1.1.** Three processes in discrete time. Coin tosses are independent. The weather copies yesterday's state with probability ``r`` and is otherwise drawn afresh, producing long spells of rain and of dry weather; at ``r = 0`` it is i.i.d. The random walk adds ``\pm 1`` to its previous position.")
		end
	end

	function figure_2_1(s2_newpath, s2_days, s2_start)
		let
			rng = MersenneTwister(8 + 23s2_newpath); n = s2_days
			x = zeros(Int, n + 1); x[1] = s2_start
			for t in 1:n
				x[t+1] = findfirst(>(rand(rng)), cumsum(BIKE_P[x[t], :]))
			end
			short = ["A", "C", "N"]
			g = chain_graph(BIKE_P; names = BIKE_NAMES, w = 360, h = 330)
			θ = [π / 2 - 2π * (i - 1) / 3 for i in 1:3]
			for t in 0:n
				i = x[t+1]
				circle!(g, cos(θ[i]), sin(θ[i]), 37; fill = "none", stroke = COL.red, sw = 4, o = t, over = true)
			end
			h = fig(w = 320, h = 330, xr = (-0.5, n + 0.5), yr = (0.5, 3.5), xlabel = "day n", yticks = 1:3, yticklabels = BIKE_NAMES, l = 76)
			for t in 1:n
				seg!(h, t - 1, x[t], t, x[t+1]; c = COL.red, w = 1.8, s = t)
			end
			dots!(h, collect(0:n), x; c = COL.red, r = n > 100 ? 1.6 : 3, s = collect(0:n))
			share(t, j) = count(==(j), x[1:t+1]) / (t + 1)
			reads = ["Day <b>$(t)</b>: the bicycle is at <b>$(BIKE_NAMES[x[t+1]])</b>" * (t == 0 ? "" : " (yesterday: $(BIKE_NAMES[x[t]]); probability of this move $(fmt(BIKE_P[x[t], x[t+1]]; d = 2)))") *
				" · fraction of days so far: Altmarkt $(pct(share(t, 1); d = 0)), Campus $(pct(share(t, 2); d = 0)), Neustadt $(pct(share(t, 3); d = 0))" for t in 0:n]
			stays = [count(t -> x[t] == j && x[t+1] == j, 1:n) / max(1, count(t -> x[t] == j, 1:n)) for j in 1:3]
			note = Markdown.parse("Over $(n) days, this bicycle spent $(pct(share(n, 1); d = 0)) of the days at the Altmarkt, $(pct(share(n, 2); d = 0)) at the Campus and $(pct(share(n, 3); d = 0)) at Neustadt. Observed fractions of days on which it stayed at its station: Altmarkt $(fmt(stays[1]; d = 2)), Campus $(fmt(stays[2]; d = 2)), Neustadt $(fmt(stays[3]; d = 2)); the diagonal of ``P`` is 0.6, 0.5, 0.6. " * (n >= 365 ? "Over a year, the fractions approach the stationary distribution, (0.412, 0.315, 0.273), whatever the initial station." : ""))
			player(vstack(figure(g, h), note); n = n, start = 0, fps = n > 100 ? 12 : 2, reads = reads, label = "day",
				caption = md"**Figure 2.2.** Left: the state diagram of the bicycle chain. Right: its path. Each move depends only on the current station.")
		end
	end

	function figure_2_2(s2_i, s2_u)
		let
			i = findfirst(==(s2_i), BIKE_NAMES)
			f = fig(w = 680, h = 200, xr = (0, 1), yr = (3.6, 0.4), xlabel = "u", l = 130, yticks = Float64[])
			cols = [COL.blue, COL.ochre, COL.green]
			for r in 1:3
				c = [0.0; cumsum(BIKE_P[r, :])]
				for j in 1:3
					rect!(f, c[j], r - 0.32, c[j+1], r + 0.32; fill = cols[j], op = r == i ? 0.85 : 0.25, stroke = "white", sw = 2)
					text!(f, (c[j] + c[j+1]) / 2, r, BIKE_NAMES[j]; anchor = "middle", size = 11.5, c = r == i ? "white" : "#555", bold = r == i)
				end
				text!(f, -0.01, r, "from " * BIKE_NAMES[r]; anchor = "end", size = 11.5, c = r == i ? COL.ink : "#888")
			end
			seg!(f, s2_u, 0.45, s2_u, 3.55; c = COL.red, w = 2.4, over = true)
			j = findfirst(>(s2_u), cumsum(BIKE_P[i, :]))
			figure(f, Markdown.parse("From **$(s2_i)** with ``u = `` $(s2_u): the bicycle moves to **$(BIKE_NAMES[j])**.");
				caption = md"**Figure 2.3.** The recursion ``X_{n+1} = f(X_n, U_n)`` for the bicycle chain.")
		end
	end

	function figure_2_3(s2_ws, s2_i)
		let
			i0 = findfirst(==(s2_i), BIKE_NAMES); D = 30
			rng = MersenneTwister(23 + 1009s2_ws)
			cols = [COL.blue, COL.ochre, COL.green]
			code = ["x₀ ← start station",
				"for n = 0, 1, …, N − 1",
				"    draw U, uniform on (0, 1)",
				"    cut (0, 1) into pieces of lengths P[xₙ, 1], P[xₙ, 2], P[xₙ, 3]",
				"    xₙ₊₁ ← the station whose piece contains U",
				"return x₀, x₁, …, x_N"]
			f = fig(w = 680, h = 190, xr = (0, 1), yr = (3.6, 0.4), xlabel = "U", l = 112, yticks = Float64[], title = "the rows of P")
			pre = [i0]; let r2 = MersenneTwister(23 + 1009s2_ws)
				for n in 1:D
					push!(pre, findfirst(>(rand(r2)), cumsum(BIKE_P[pre[end], :])))
				end
			end
			active(k) = k == 0 ? 0 : pre[div(k + 1, 2)]
			for r in 1:3
				c = [0.0; cumsum(BIKE_P[r, :])]
				for j in 1:3
					rect!(f, c[j], r - 0.32, c[j+1], r + 0.32; fill = cols[j], op = 0.25, stroke = "white", sw = 2)
					for k in 0:2D
						active(k) == r || text!(f, (c[j] + c[j+1]) / 2, r, BIKE_NAMES[j]; anchor = "middle", size = 11, c = "#555", o = k)
					end
				end
				text!(f, -0.01, r, "from " * BIKE_NAMES[r]; anchor = "end", size = 11, c = "#555")
			end
			g = fig(w = 680, h = 170, xr = (-0.4, D + 0.4), yr = (0.5, 3.5), xlabel = "day n", yticks = 1:3, yticklabels = BIKE_NAMES, l = 76, title = "the path")
			x = [i0]; steps = Any[(line = 1, state = "Day 0: the bicycle starts at **$(BIKE_NAMES[i0])**.")]
			dots!(g, [0], [i0]; c = COL.red, r = 4, s = 0)
			for n in 1:D
				i = x[end]; U = rand(rng); cs = cumsum(BIKE_P[i, :]); j = findfirst(>(U), cs); push!(x, j)
				b = 2(n - 1)
				c = [0.0; cs]
				for jj in 1:3
					rect!(f, c[jj], i - 0.32, c[jj+1], i + 0.32; fill = cols[jj], op = 0.85, stroke = "white", sw = 2, o = (b + 1, b + 2))
					text!(f, (c[jj] + c[jj+1]) / 2, i, BIKE_NAMES[jj]; anchor = "middle", size = 11, c = "white", bold = true, o = (b + 1, b + 2))
				end
				seg!(f, U, 0.45, U, 3.55; c = COL.red, w = 2.4, over = true, o = (b + 1, b + 2))
				rect!(f, c[j], i - 0.36, c[j+1], i + 0.36; fill = "none", op = 1, stroke = COL.ink, sw = 2.5, over = true, o = b + 2)
				seg!(g, n - 1, i, n, j; c = COL.red, w = 1.8, s = b + 2)
				dots!(g, [n], [j]; c = COL.red, r = 4, s = b + 2)
				push!(steps, (line = [3, 4], state = "Day $(n - 1) → $(n): the bicycle is at **$(BIKE_NAMES[i])**, so row $(BIKE_NAMES[i]) is used: pieces of lengths $(join(fmt.(BIKE_P[i, :]; d = 2), ", ")). U = **$(fmt(U; d = 3))**.",
					why = n == 1 ? "U is uniform, so it lands in a piece with probability equal to the piece's length: P[x, j]." : n == 2 ? "" : ""))
				push!(steps, (line = 5, state = "U = $(fmt(U; d = 3)) lies in the piece from $(fmt(c[j]; d = 2)) to $(fmt(c[j+1]; d = 2)): the bicycle moves to **$(BIKE_NAMES[j])**" * (j == i ? " (it stays)." : "."),
					why = n == 1 ? "" : "",
					fail = n == 1 ? "" : n == 3 ? "" : ""))
			end
			walk(code, steps, f, g; title = "simulating a chain", label = "frame",
				caption = md"**Figure 2.4.** Simulating the bicycle chain for 30 days.")
		end
	end

	function _check_simulate_chain(simulate_chain)
		let
			r = tryrun(() -> simulate_chain(BIKE_P, 1, 10, MersenneTwister(1)))
			ok = r !== missing && length(r) == 11 && let
				x = simulate_chain(BIKE_P, 1, 300_000, MersenneTwister(2))
				T = zeros(3, 3)
				for t in 1:length(x)-1
					T[x[t], x[t+1]] += 1
				end
				maximum(abs.(T ./ sum(T, dims = 2) .- BIKE_P)) < 0.01
			end
			check(r, ok, md"Over 300 000 days the observed frequencies of each move match the transition matrix.",
				md"The observed transition frequencies do not match ``P``.";
				hint = md"``P_{ij}`` is the probability of moving from ``i`` to ``j``. Which index is the current state?")
		end
	end

	function compute_s3_x0(s3_c0, s3_u0, s3_p0)
		let
			v = [s3_c0, s3_u0, s3_p0]
			sum(v) == 0 ? [0, 0, 300] : sum(v) > 400 ? round.(Int, v .* 400 / sum(v)) : v
		end
	end

	function figure_3_1(s3_newbikes, s3_D, s3_x0)
		let
			rng = MersenneTwister(30 + 41s3_newbikes); D = s3_D
			pos = vcat([fill(i, s3_x0[i]) for i in 1:3]...)
			N = length(pos)
			cols = [COL.blue, COL.ochre, COL.green]
			home = copy(pos)
			f = fig(w = 680, h = 250, xr = (0, 3), yr = (0, 1), axes = false, l = 6, r = 6, t = 26, b = 30)
			for i in 1:3
				rect!(f, i - 0.97, 0.02, i - 0.03, 0.98; fill = "#f7f7f4", op = 1, stroke = "#ccc", rx = 6)
				text!(f, i - 0.5, -0.07, BIKE_NAMES[i]; anchor = "middle", size = 12.5, bold = true)
			end
			counts = zeros(Int, D + 1, 3)
			for d in 0:D
				d > 0 && (pos = [findfirst(>(rand(rng)), cumsum(BIKE_P[p, :])) for p in pos])
				for i in 1:3
					idx = findall(==(i), pos); counts[d+1, i] = length(idx)
					for (k, b) in enumerate(idx)
						col = mod(k - 1, 20); row = div(k - 1, 20)
						dots!(f, [i - 0.94 + col * 0.0465], [0.05 + row * 0.045]; c = cols[home[b]], r = 2.6, o = d)
					end
					text!(f, i - 0.5, 1.07, "$(length(idx)) bicycles"; anchor = "middle", size = 12, o = d)
				end
			end
			g = fig(w = 680, h = 230, xr = (0, D), yr = (0, 1.04N), xlabel = "day n", ylabel = "bicycles at station")
			π = [68, 52, 45] ./ 165
			expd = [N .* ((s3_x0' ./ N) * BIKE_P^d) for d in 0:D]
			for i in 1:3
				dots!(g, [0], [counts[1, i]]; c = cols[i], r = 3.5)
				hline!(g, N * π[i]; c = cols[i], dash = "4 3", w = 1)
				for d in 1:D
					seg!(g, d - 1, counts[d, i], d, counts[d+1, i]; c = cols[i], w = 2, s = d)
				end
						end
			legend!(g, [(BIKE_NAMES[i], cols[i]) for i in 1:3]; at = :tr, size = 11)
			reads = ["Day <b>$(d)</b>: Altmarkt <b>$(counts[d+1, 1])</b>, Campus <b>$(counts[d+1, 2])</b>, Neustadt <b>$(counts[d+1, 3])</b> · expected $(round(Int, expd[d+1][1])), $(round(Int, expd[d+1][2])), $(round(Int, expd[d+1][3]))" for d in 0:D]
			player(
    f, g;
    n = D, start = 0, fps = 3,
    reads = reads, label = "day",
    caption = md"**Figure 5.1.** A fleet of bicycles, each following its own random path."
)
end
end

	function figure_3_2(s3_n)
		let
			Pn = BIKE_P^s3_n
			cols = [COL.blue, COL.ochre, COL.green]
			f = fig(w = 680, h = 220, xr = (0, 1), yr = (3.6, 0.4), axes = true, xlabel = "probability", l = 130, t = 34, yticks = Float64[])
			for r in 1:3
				c = [0.0; cumsum(Pn[r, :])]
				for j in 1:3
					rect!(f, c[j], r - 0.3, c[j+1], r + 0.3; fill = cols[j], op = 0.85, stroke = "white", sw = 1.5)
					Pn[r, j] > 0.08 && text!(f, (c[j] + c[j+1]) / 2, r, fmt(Pn[r, j]; d = 3); anchor = "middle", size = 11, c = "white", bold = true)
				end
				text!(f, -0.01, r, "start: " * BIKE_NAMES[r]; anchor = "end", size = 11.5)
			end
			legend!(f, [(BIKE_NAMES[j], cols[j], :bar) for j in 1:3]; at = :top, size = 11.5)
			figure(f; caption = md"**Figure 5.2.** The rows of ``P^n``: the distribution of the station after ``n`` days, for each initial station.")	
		end
	end

	function _check_distribution_after(distribution_after)
		let
			r = tryrun(() -> distribution_after(BIKE_P, [0 0 1.0], 3))
			ok = r !== missing && isapprox(vec(r), vec([0 0 1.0] * BIKE_P^3)) && isapprox(vec(distribution_after(BIKE_P, [1.0 0 0], 0)), [1, 0, 0])
			check(r, ok, () -> md"After 3 days from Neustadt: ``\pi_3 = `` $(round.(vec(r), digits = 3)), i.e. about $(round.(Int, 300 .* vec(r))) bicycles.",
				() -> md"Got $(r); expected ``\pi_0 P^3 = `` $(round.(vec([0 0 1.0] * BIKE_P^3), digits = 3)).";
				hint = md"Does each pass of the loop start from the previous day's distribution?")
		end
	end

	function compute_s4_P(s4_c, s4_my)
		s4_c == "my chain (matrix below)" ? something(parse_matrix(s4_my), BIKE_P) :
			Dict("ergodic: the bicycle chain" => BIKE_P, "reducible: two islands" => [0.5 0.5 0; 0.3 0.7 0; 0 0 1.0],
			"periodic: a cycle of three" => [0 1.0 0; 0 0 1; 1 0 0], "absorbing: a trap" => [0.6 0.3 0.1; 0.3 0.5 0.2; 0 0 1.0],
			"periodic with period 2" => [0 0.5 0.5; 1 0 0; 1 0 0])[s4_c]
	end

	function figure_4_1(s4_new, s4_P, s4_s, s4_c)
		let
			P = s4_P; n = 60; k = size(P, 1); st = min(s4_s, k)
			π0 = zeros(1, k); π0[st] = 1
			path = zeros(n + 1, k); πt = π0
			for t in 0:n
				path[t+1, :] = πt; πt = πt * P
			end
			rng = MersenneTwister(4 + 7s4_new); x = zeros(Int, n + 1); x[1] = st
			for t in 1:n
				x[t+1] = something(findfirst(>(rand(rng)), cumsum(P[x[t], :])), k)
			end
			names = s4_c == "ergodic: the bicycle chain" ? ["A", "C", "N"] : string.(1:k)
			g = chain_graph(P; names = names, w = 300, h = 290, rnode = 20, hl = st)
			θ = [π / 2 - 2π * (i - 1) / k for i in 1:k]
			for t in 0:n
				circle!(g, cos(θ[x[t+1]]), sin(θ[x[t+1]]), 25; fill = "none", stroke = COL.red, sw = 3.5, o = t, over = true)
			end
			cols = PALETTE[1:k]
			f = fig(w = 380, h = 290, xr = (0, n), yr = (-0.02, 1.02), xlabel = "step n", ylabel = "P(Xₙ = state)", t = 30)
			for j in 1:k, t in 1:n
				seg!(f, t - 1, path[t, j], t, path[t+1, j]; c = cols[j], w = 2.2, s = t)
			end
			legend!(f, [("state $(names[j])", cols[j]) for j in 1:k]; at = :top, size = 11)
			C = classify(P)
			cname(c) = "{" * join(names[c], ", ") * "}"
			desc = C.irreducible ? "The chain is **irreducible**: all states communicate. " : "The chain is **reducible**, with communicating classes " * join(cname.(C.classes), ", ") * "; closed: " * join([cname(c) for (c, cl) in zip(C.classes, C.closed) if cl], ", ") * ". "
			abs_ = [names[c[1]] for (c, cl) in zip(C.classes, C.closed) if cl && length(c) == 1 && P[c[1], c[1]] > 0.999]
			desc *= isempty(abs_) ? "" : "Absorbing state" * (length(abs_) > 1 ? "s " : " ") * join(abs_, ", ") * ". "
			desc *= "Periods of the classes: " * join(["$(cname(c)): $(p)" for (c, p) in zip(C.classes, C.period)], ", ") * ". "
			desc *= C.irreducible && C.period[1] == 1 ? "Irreducible and aperiodic: **ergodic**, so ``\\pi_n`` converges to the same limit from every start." :
				C.irreducible ? "Irreducible but periodic: a unique stationary distribution exists, but ``\\pi_n`` oscillates and does not converge." :
				"Reducible: where the chain ends up depends on where it starts."
			reads = ["Step <b>$(t)</b>: the walker is in state <b>$(names[x[t+1]])</b> · P(Xₙ = each state) = " * join([fmt(path[t+1, j]; d = 2) for j in 1:k], ", ") for t in 0:n]
			player(vstack(figure(g, f), Markdown.parse(desc)); n = n, start = n, fps = 3, reads = reads, label = "step",
				caption = md"**Figure 6.2.** The ergodic chain converges to the same distribution from every start. The reducible chain converges too, but the limit depends on the start. A periodic chain cycles forever without converging.")
		end
	end

	function figure_5_5(s5_text)
		let
			letters = [c for c in lowercase(s5_text) if isletter(c) && isascii(c)]
			n = length(letters)
			if n < 50
				md"*Enter a text with at least 50 letters.*"
			else
				v = [c in "aeiou" for c in letters]
				C = zeros(2, 2)
				for t in 1:n-1
					C[v[t] ? 1 : 2, v[t+1] ? 1 : 2] += 1
				end
				P = C ./ sum(C, dims = 2)
				a, b = P[1, 2], P[2, 1]
				πv = b / (a + b)
				run = cumsum(v) ./ (1:n)
				f = fig(w = 370, h = 260, xr = (1, n), yr = (0, 1), xlabel = "letters read", ylabel = "fraction of vowels so far")
				line!(f, 1:n, run; c = COL.purple, w = 1.8)
				hline!(f, πv; c = COL.ink, dash = "4 3")
				G = fig(w = 300, h = 260, xr = (0.3, 2.7), yr = (0, 1), xticks = [1, 2], xticklabels = ["after vowel", "after consonant"], ylabel = "P(next letter is a vowel)")
				bars!(G, [1, 2], [P[1, 1], P[2, 1]]; width = 0.55, c = [COL.red, COL.blue], op = 0.8, labels = [fmt(P[1, 1]; d = 2), fmt(P[2, 1]; d = 2)])
				hline!(G, mean(v); c = COL.ink, dash = "4 3")
				figure(f, G; caption = md"**Figure 3.4.** Left: the running fraction of vowels in the text, approaching the stationary distribution of the vowel–consonant chain. Right: the transition probabilities to a vowel; if letters were independent, both bars would equal the overall vowel fraction (dashed).")
			end
		end
	end

	function figure_7_1(s7_fleet, s7_risk, guess_c)
		let
			π = [68, 52, 45] ./ 165; cols = [COL.blue, COL.ochre, COL.green]
			Nb = s7_fleet
			caps = [quantile(Binomial(Nb, π[i]), 1 - s7_risk) for i in 1:3]
			lo = floor(Int, Nb * π[3] - 4.5sqrt(Nb * π[3])); hi = ceil(Int, Nb * π[1] + 4.5sqrt(Nb * π[1]))
			ymax = 1.15maximum(pdf(Binomial(Nb, π[i]), round(Int, Nb * π[i])) for i in 1:3)
			f = fig(w = 680, h = 280, xr = (lo, hi), yr = (0, ymax), xlabel = "bicycles at the station on a given day", ylabel = "probability")
			for i in 1:3
				ks = lo:hi; pm = pdf.(Binomial(Nb, π[i]), ks)
				line!(f, ks, pm; c = cols[i], w = 2.2)
				vline!(f, caps[i] + 0.5; c = cols[i], dash = "4 3", w = 1.5)
				text!(f, lo + 0.02(hi - lo), ymax * (0.96 - 0.1i), "$(BIKE_NAMES[i]): $(caps[i]) docks"; size = 11.5, c = cols[i])
			end
ans = Markdown.parse(
    (guess_c > 0 && Nb == 300 ?
        "Your estimate for the Altmarkt was $(guess_c) bicycles; the expected number is 124. " :
        "") *
    "With $(Nb) bicycles and an accepted risk of $(pct(s7_risk; d = 1)) per station and day, the required numbers of docks are **$(caps[1])** (Altmarkt), **$(caps[2])** (Campus) and **$(caps[3])** (Neustadt)."
)
			figure(f, ans; caption = md"**Figure 8.1.** Long-run distribution of the number of bicycles at each station (binomial, with ``n`` equal to the fleet size) and the capacities that are exceeded with a probability below the chosen risk (dashed).")
		end
	end

	check_simulate_chain(simulate_chain) = guarded(() -> _check_simulate_chain(simulate_chain))

	check_distribution_after(distribution_after) = guarded(() -> _check_distribution_after(distribution_after))

	check_stationary(stationary) = guarded(() -> _check_stationary(stationary))
end;

# ╔═╡ c31c2f38-403b-51a7-0327-204572af510a
figure_2_2(s2_i, s2_u)

# ╔═╡ 6adef4cb-318b-c06b-e388-e8bb87e2eeae
figure_5_5(s5_text)

# ╔═╡ 12d3b05f-6427-4f74-9efa-c148ad3650f3
figure_3_2(s3_n)

# ╔═╡ 7bf55512-9b57-69c8-db32-f85b8a084fda
check_simulate_chain(simulate_chain)

# ╔═╡ ac7dca37-1584-3edd-af3e-c85fc55da5dd
check_distribution_after(distribution_after)

# ╔═╡ 5c8d6357-9a80-a750-aa96-a9d30aba4b1a
begin
	function figure_2_0(s2_sel, s2_mode)
		P = BIKE_P; k = s2_sel; row = startswith(s2_mode, "a row")
		f = fig(w = 400, h = 270, xr = (0, 5.7), yr = (0, 5), axes = false, l = 4, r = 4, t = 4, b = 4)
		text!(f, 3.0, 4.75, "tomorrow (to)"; anchor = "middle", size = 12.5, bold = true, c = COL.blue)
		text!(f, 0.25, 2.0, "today (from)"; anchor = "middle", size = 12.5, bold = true, c = COL.red, rot = -90)
		for j in 1:3
			text!(f, 1.5 + j, 4.3, BIKE_NAMES[j]; anchor = "middle", size = 11.5)
			text!(f, 1.1, 3.5 - (j - 1), BIKE_NAMES[j]; anchor = "middle", size = 11.5)
		end
		for i in 1:3, j in 1:3
			x = 1.5 + j; y = 3.5 - (i - 1)
			hl = row ? i == k : j == k
			rect!(f, x - 0.48, y - 0.42, x + 0.48, y + 0.42; fill = hl ? (row ? "#fbe9e4" : "#e6eef9") : "#f6f6f3", op = 1, stroke = hl ? (row ? COL.red : COL.blue) : "#ddd", sw = hl ? 1.6 : 0.8, rx = 3)
			text!(f, x, y, fmt(P[i, j]; d = 2); anchor = "middle", size = 13, bold = hl)
		end
		if row
			text!(f, 5.68, 3.5 - (k - 1), "sum = $(fmt(sum(P[k, :]); d = 2))"; anchor = "end", size = 11.5, c = COL.red, bold = true)
		else
			text!(f, 1.5 + k, 0.55, "sum = $(fmt(sum(P[:, k]); d = 2))"; anchor = "middle", size = 11.5, c = COL.blue, bold = true)
		end
		g = chain_graph(P; names = BIKE_NAMES, w = 270, h = 310, hl = k)
		n = BIKE_NAMES[k]
		txt = row ? Markdown.parse("**Row $(n)** answers the question *where is a bicycle that is at $(n) today tomorrow?*") :
			Markdown.parse("**Column $(n)** answers the question *with which probability does a bicycle from each station arrive at $(n)?*")
		figure(f, g, txt; caption = md"**Figure 2.1.** The transition matrix of the bicycle chain and its state diagram. Entry ``P_{ij}`` in row ``i``, column ``j`` is the probability of moving from ``i`` today to ``j`` tomorrow.")
	end

	const CORPORA = Dict(
		"Alice's Adventures in Wonderland (Lewis Carroll, 1865)" => raw"""Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, "and what is the use of a book," thought Alice "without pictures or conversations?" So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her. There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, "Oh dear! Oh dear! I shall be late!" (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge. In another moment down went Alice after it, never once considering how in the world she was to get out again. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well. Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled "ORANGE MARMALADE", but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it. "Well!" thought Alice to herself, "after such a fall as this, I shall think nothing of tumbling down stairs! How brave they'll all think me at home! Why, I wouldn't say anything about it, even if I fell off the top of the house!" (Which was very likely true.) Down, down, down. Would the fall never come to an end? "I wonder how many miles I've fallen by this time?" she said aloud. "I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think , " (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a very good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) " , yes, that's about the right distance , but then I wonder what Latitude or Longitude I've got to?" (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.) Presently she began again. "I wonder if I shall fall right through the earth! How funny it'll seem to come out among the people that walk with their heads downward! The Antipathies, I think , " (she was rather glad there was no one listening, this time, as it didn't sound at all the right word) " , but I shall have to ask them what the name of the country is, you know. Please, Ma'am, is this New Zealand or Australia?" (and she tried to curtsey as she spoke , fancy curtseying as you're falling through the air! Do you think you could manage it?) "And what an ignorant little girl she'll think me for asking! No, it'll never do to ask: perhaps I shall see it written up somewhere." Down, down, down. There was nothing else to do, so Alice soon began talking again. "Dinah'll miss me very much to-night, I should think!" (Dinah was the cat.) "I hope they'll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I'm afraid, but you might catch a bat, and that's very like a mouse, you know. But do cats eat bats, I wonder?" And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, "Do cats eat bats? Do cats eat bats?" and sometimes, "Do bats eat cats?" for, you see, as she couldn't answer either question, it didn't much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, "Now, Dinah, tell me the truth: did you ever eat a bat?" when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over. Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, "Oh my ears and whiskers, how late it's getting!" She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof. There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again. Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice's first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted! Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; "and even if my head would go through," thought poor Alice, "it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin." For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible. There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, ("which certainly was not here before," said Alice,) and round the neck of the bottle was a paper label, with the words "DRINK ME," beautifully printed on it in large letters. It was all very well to say "Drink me," but the wise little Alice was not going to do that in a hurry. "No, I'll look first," she said, "and see whether it's marked 'poison' or not"; for she had read several nice little histories about children who had got burnt, and eaten up by wild beasts and other unpleasant things, all because they would not remember the simple rules their friends had taught them: such as, that a red-hot poker will burn you if you hold it too long; and that if you cut your finger very deeply with a knife, it usually bleeds; and she had never forgotten that, if you drink much from a bottle marked "poison," it is almost certain to disagree with you, sooner or later. However, this bottle was not marked "poison," so Alice ventured to taste it, and finding it very nice, (it had, in fact, a sort of mixed flavour of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast,) she very soon finished it off. * * * * * * * * * * * * * * * * * * * * "What a curious feeling!" said Alice; "I must be shutting up like a telescope." And so it was indeed: she was now only ten inches high, and her face brightened up at the thought that she was now the right size for going through the little door into that lovely garden. First, however, she waited for a few minutes to see if she was going to shrink any further: she felt a little nervous about this; "for it might end, you know," said Alice to herself, "in my going out altogether, like a candle. I wonder what I should be like then?" And she tried to fancy what the flame of a candle is like after the candle is blown out, for she could not remember ever having seen such a thing. After a while, finding that nothing more happened, she decided on going into the garden at once; but, alas for poor Alice! when she got to the door, she found she had forgotten the little golden key, and when she went back to the table for it, she found she could not possibly reach it: she could see it quite plainly through the glass, and she tried her best to climb up one of the legs of the table, but it was too slippery; and when she had tired herself out with trying, the poor little thing sat down and cried. "Come, there's no use in crying like that!" said Alice to herself, rather sharply; "I advise you to leave off this minute!" She generally gave herself very good advice, (though she very seldom followed it), and sometimes she scolded herself so severely as to bring tears into her eyes; and once she remembered trying to box her own ears for having cheated herself in a game of croquet she was playing against herself, for this curious child was very fond of pretending to be two people. "But it's no use now," thought poor Alice, "to pretend to be two people! Why, there's hardly enough of me left to make one respectable person!" Soon her eye fell on a little glass box that was lying under the table: she opened it, and found in it a very small cake, on which the words "EAT ME" were beautifully marked in currants. "Well, I'll eat it," said Alice, "and if it makes me grow larger, I can reach the key; and if it makes me grow smaller, I can creep under the door; so either way I'll get into the garden, and I don't care which happens!" She ate a little bit, and said anxiously to herself, "Which way? Which way?", holding her hand on the top of her head to feel which way it was growing, and she was quite surprised to find that she remained the same size: to be sure, this generally happens when one eats cake, but Alice had got so much into the way of expecting nothing but out-of-the-way things to happen, that it seemed quite dull and stupid for life to go on in the common way. So she set to work, and very soon finished off the cake. * * * * * * * * * * * * * * * * * * * * The Pool of Tears "Curiouser and curiouser!" cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English); "now I'm opening out like the largest telescope that ever was! Good-bye, feet!" (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far off). "Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I'm sure I shan't be able! I shall be a great deal too far off to trouble myself about you: you must manage the best way you can; , but I must be kind to them," thought Alice, "or perhaps they won't walk the way I want to go! Let me see: I'll give them a new pair of boots every Christmas." And she went on planning to herself how she would manage it. "They must go by the carrier," she thought; "and how funny it'll seem, sending presents to one's own feet! And how odd the directions will look! Alice's Right Foot, Esq., Hearthrug, near the Fender, (with Alice's love). Oh dear, what nonsense I'm talking!" Just then her head struck against the roof of the hall: in fact she was now more than nine feet high, and she at once took up the little golden key and hurried off to the garden door. Poor Alice! It was as much as she could do, lying down on one side, to look through into the garden with one eye; but to get through was more hopeless than ever: she sat down and began to cry again. "You ought to be ashamed of yourself," said Alice, "a great girl like you," (she might well say this), "to go on crying in this way! Stop this moment, I tell you!" But she went on all the same, shedding gallons of tears, until there was a large pool all round her, about four inches deep and reaching half down the hall. After a time she heard a little pattering of feet in the distance, and she hastily dried her eyes to see what was coming. It was the White Rabbit returning, splendidly dressed, with a pair of white kid gloves in one hand and a large fan in the other: he came trotting along in a great hurry, muttering to himself as he came, "Oh! the Duchess, the Duchess! Oh! won't she be savage if I've kept her waiting!" Alice felt so desperate that she was ready to ask help of any one; so, when the Rabbit came near her, she began, in a low, timid voice, "If you please, sir , " The Rabbit started violently, dropped the white kid gloves and the fan, and skurried away into the darkness as hard as he could go. Alice took up the fan and gloves, and, as the hall was very hot, she kept fanning herself all the time she went on talking: "Dear, dear! How queer everything is to-day! And yesterday things went on just as usual. I wonder if I've been changed in the night? Let me think: was I the same when I got up this morning? I almost think I can remember feeling a little different. But if I'm not the same, the next question is, Who in the world am I? Ah, that's the great puzzle!" And she began thinking over all the children she knew that were of the same age as herself, to see if she could have been changed for any of them. "I'm sure I'm not Ada," she said, "for her hair goes in such long ringlets, and mine doesn't go in ringlets at all; and I'm sure I can't be Mabel, for I know all sorts of things, and she, oh! she knows such a very little! Besides, she's she, and I'm I, and , oh dear, how puzzling it all is! I'll try if I know all the things I used to know. Let me see: four times five is twelve, and four times six is thirteen, and four times seven is , oh dear! I shall never get to twenty at that rate! However, the Multiplication Table doesn't signify: let's try Geography. London is the capital of Paris, and Paris is the capital of Rome, and Rome , no, that's all wrong, I'm certain! I must have been changed for Mabel! I'll try and say 'How doth the little , '" and she crossed her hands on her lap as if she were saying lessons, and began to repeat it, but her voice sounded hoarse and strange, and the words did not come the same as they used to do: , "How doth the little crocodile Improve his shining tail, And pour the waters of the Nile On every golden scale! "How cheerfully he seems to grin, How neatly spread his claws, And welcome little fishes in With gently smiling jaws!" "I'm sure those are not the right words," said poor Alice, and her eyes filled with tears again as she went on, "I must be Mabel after all, and I shall have to go and live in that poky little house, and have next to no toys to play with, and oh! ever so many lessons to learn! No, I've made up my mind about it; if I'm Mabel, I'll stay down here! It'll be no use their putting their heads down and saying 'Come up again, dear!' I shall only look up and say 'Who am I then? Tell me that first, and then, if I like being that person, I'll come up: if not, I'll stay down here till I'm somebody else' , but, oh dear!" cried Alice, with a sudden burst of tears, "I do wish they would put their heads down! I am so very tired of being all alone here!" As she said this she looked down at her hands, and was surprised to see that she had put on one of the Rabbit's little white kid gloves while she was talking. "How can I have done that?" she thought. "I must be growing small again." She got up and went to the table to measure herself by it, and found that, as nearly as she could guess, she was now about two feet high, and was going on shrinking rapidly: she soon found out that the cause of this was the fan she was holding, and she dropped it hastily, just in time to avoid shrinking away altogether. "That was a narrow escape!" said Alice, a good deal frightened at the sudden change, but very glad to find herself still in existence; "and now for the garden!" and she ran with all speed back to the little door: but, alas! the little door was shut again, and the little golden key was lying on the glass table as before, "and things are worse than ever," thought the poor child, "for I never was so small as this before, never! And I declare it's too bad, that it is!" As she said these words her foot slipped, and in another moment, splash! she was up to her chin in salt water. Her first idea was that she had somehow fallen into the sea, "and in that case I can go back by railway," she said to herself. (Alice had been to the seaside once in her life, and had come to the general conclusion, that wherever you go to on the English coast you find a number of bathing machines in the sea, some children digging in the sand with wooden spades, then a row of lodging houses, and behind them a railway station.) However, she soon made out that she was in the pool of tears which she had wept when she was nine feet high. "I wish I hadn't cried so much!" said Alice, as she swam about, trying to find her way out. "I shall be punished for it now, I suppose, by being drowned in my own tears! That will be a queer thing, to be sure! However, everything is queer to-day." Just then she heard something splashing about in the pool a little way off, and she swam nearer to make out what it was: at first she thought it must be a walrus or hippopotamus, but then she remembered how small she was now, and she soon made out that it was only a mouse that had slipped in like herself. "Would it be of any use, now," thought Alice, "to speak to this mouse? Everything is so out-of-the-way down here, that I should think very likely it can talk: at any rate, there's no harm in trying." So she began: "O Mouse, do you know the way out of this pool? I am very tired of swimming about here, O Mouse!" (Alice thought this must be the right way of speaking to a mouse: she had never done such a thing before, but she remembered having seen in her brother's Latin Grammar, "A mouse , of a mouse , to a mouse , a mouse , O mouse!") The Mouse looked at her rather inquisitively, and seemed to her to wink with one of its little eyes, but it said nothing. "Perhaps it doesn't understand English," thought Alice; "I daresay it's a French mouse, come over with William the Conqueror." (For, with all her knowledge of history, Alice had no very clear notion how long ago anything had happened.) So she began again: "Où est ma chatte?" which was the first sentence in her French lesson-book. The Mouse gave a sudden leap out of the water, and seemed to quiver all over with fright. "Oh, I beg your pardon!" cried Alice hastily, afraid that she had hurt the poor animal's feelings. "I quite forgot you didn't like cats." "Not like cats!" cried the Mouse, in a shrill, passionate voice. "Would you like cats if you were me?" "Well, perhaps not," said Alice in a soothing tone: "don't be angry about it. And yet I wish I could show you our cat Dinah: I think you'd take a fancy to cats if you could only see her. She is such a dear quiet thing," Alice went on, half to herself, as she swam lazily about in the pool, "and she sits purring so nicely by the fire, licking her paws and washing her face , and she is such a nice soft thing to nurse , and she's such a capital one for catching mice , oh, I beg your pardon!" cried Alice again, for this time the Mouse was bristling all over, and she felt certain it must be really offended. "We won't talk about her any more if you'd rather not." "We indeed!" cried the Mouse, who was trembling down to the end of his tail. "As if I would talk on such a subject! Our family always hated cats: nasty, low, vulgar things! Don't let me hear the name again!" "I won't indeed!" said Alice, in a great hurry to change the subject of conversation. "Are you , are you fond , of , of dogs?" The Mouse did not answer, so Alice went on eagerly: "There is such a nice little dog near our house I should like to show you! A little bright-eyed terrier, you know, with oh, such long curly brown hair! And it'll fetch things when you throw them, and it'll sit up and beg for its dinner, and all sorts of things , I can't remember half of them , and it belongs to a farmer, you know, and he says it's so useful, it's worth a hundred pounds! He says it kills all the rats and , oh dear!" cried Alice in a sorrowful tone, "I'm afraid I've offended it again!" For the Mouse was swimming away from her as hard as it could go, and making quite a commotion in the pool as it went. So she called softly after it, "Mouse dear! Do come back again, and we won't talk about cats or dogs either, if you don't like them!" When the Mouse heard this, it turned round and swam slowly back to her: its face was quite pale (with passion, Alice thought), and it said in a low trembling voice, "Let us get to the shore, and then I'll tell you my history, and you'll understand why it is I hate cats and dogs." It was high time to go, for the pool was getting quite crowded with the birds and animals that had fallen into it: there were a Duck and a Dodo, a Lory and an Eaglet, and several other curious creatures. Alice led the way, and the whole party swam to the shore. A Caucus-Race and a Long Tale They were indeed a queer-looking party that assembled on the bank , the birds with draggled feathers, the animals with their fur clinging close to them, and all dripping wet, cross, and uncomfortable. The first question of course was, how to get dry again: they had a consultation about this, and after a few minutes it seemed quite natural to Alice to find herself talking familiarly with them, as if she had known them all her life. Indeed, she had quite a long argument with the Lory, who at last turned sulky, and would only say, "I am older than you, and must know better;" and this Alice would not allow without knowing how old it was, and, as the Lory positively refused to tell its age, there was no more to be said. At last the Mouse, who seemed to be a person of authority among them, called out, "Sit down, all of you, and listen to me! I'll soon make you dry enough!" They all sat down at once, in a large ring, with the Mouse in the middle. Alice kept her eyes anxiously fixed on it, for she felt sure she would catch a bad cold if she did not get dry very soon. "Ahem!" said the Mouse with an important air, "are you all ready? This is the driest thing I know. Silence all round, if you please! 'William the Conqueror, whose cause was favoured by the pope, was soon submitted to by the English, who wanted leaders, and had been of late much accustomed to usurpation and conquest. Edwin and Morcar, the earls of Mercia and Northumbria , '" "Ugh!" said the Lory, with a shiver. "I beg your pardon!" said the Mouse, frowning, but very politely: "Did you speak?" "Not I!" said the Lory hastily. "I thought you did," said the Mouse. " , I proceed. 'Edwin and Morcar, the earls of Mercia and Northumbria, declared for him: and even Stigand, the patriotic archbishop of Canterbury, found it advisable , '" "Found what?" said the Duck. "Found it," the Mouse replied rather crossly: "of course you know what 'it' means." "I know what 'it' means well enough, when I find a thing," said the Duck: "it's generally a frog or a worm. The question is, what did the archbishop find?" The Mouse did not notice this question, but hurriedly went on, "' , found it advisable to go with Edgar Atheling to meet William and offer him the crown. William's conduct at first was moderate. But the insolence of his Normans , ' How are you getting on now, my dear?" it continued, turning to Alice as it spoke. "As wet as ever," said Alice in a melancholy tone: "it doesn't seem to dry me at all." "In that case," said the Dodo solemnly, rising to its feet, "I move that the meeting adjourn, for the immediate adoption of more energetic remedies , " "Speak English!" said the Eaglet. "I don't know the meaning of half those long words, and, what's more, I don't believe you do either!" And the Eaglet bent down its head to hide a smile: some of the other birds tittered audibly. "What I was going to say," said the Dodo in an offended tone, "was, that the best thing to get us dry would be a Caucus-race." "What is a Caucus-race?" said Alice; not that she wanted much to know, but the Dodo had paused as if it thought that somebody ought to speak, and no one else seemed inclined to say anything. "Why," said the Dodo, "the best way to explain it is to do it." (And, as you might like to try the thing yourself, some winter day, I will tell you how the Dodo managed it.) First it marked out a race-course, in a sort of circle, ("the exact shape doesn't matter," it said,) and then all the party were placed along the course, here and there. There was no "One, two, three, and away," but they began running when they liked, and left off when they liked, so that it was not easy to know when the race was over. However, when they had been running half an hour or so, and were quite dry again, the Dodo suddenly called out "The race is over!" and they all crowded round it, panting, and asking, "But who has won?" This question the Dodo could not answer without a great deal of thought, and it sat for a long time with one finger pressed upon its forehead (the position in which you usually see Shakespeare, in the pictures of him), while the rest waited in silence. At last the Dodo said, "Everybody has won, and all must have prizes." "But who is to give the prizes?" quite a chorus of voices asked. "Why, she, of course," said the Dodo, pointing to Alice with one finger; and the whole party at once crowded round her, calling out in a confused way, "Prizes! Prizes!" Alice had no idea what to do, and in despair she put her hand in her pocket, and pulled out a box of comfits, (luckily the salt water had not got into it), and handed them round as prizes. There was exactly one a-piece, all round. "But she must have a prize herself, you know," said the Mouse. "Of course," the Dodo replied very gravely. "What else have you got in your pocket?" he went on, turning to Alice. "Only a thimble," said Alice sadly. "Hand it over here," said the Dodo. Then they all crowded round her once more, while the Dodo solemnly presented the thimble, saying "We beg your acceptance of this elegant thimble;" and, when it had finished this short speech, they all cheered. Alice thought the whole thing very absurd, but they all looked so grave that she did not dare to laugh; and, as she could not think of anything to say, she simply bowed, and took the thimble, looking as solemn as she could. The next thing was to eat the comfits: this caused some noise and confusion, as the large birds complained that they could not taste theirs, and the small ones choked and had to be patted on the back. However, it was over at last, and they sat down again in a ring, and begged the Mouse to tell them something more.""",
		"Pride and Prejudice (Jane Austen, 1813)" => raw"""It is a truth universally acknowledged, that a single man in possession of a good fortune must be in want of a wife. However little known the feelings or views of such a man may be on his first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, that he is considered as the rightful property of some one or other of their daughters. "My dear Mr. Bennet," said his lady to him one day, "have you heard that Netherfield Park is let at last?" Mr. Bennet replied that he had not. "But it is," returned she; "for Mrs. Long has just been here, and she told me all about it." Mr. Bennet made no answer. "Do not you want to know who has taken it?" cried his wife, impatiently. "You want to tell me, and I have no objection to hearing it." ] This was invitation enough. "Why, my dear, you must know, Mrs. Long says that Netherfield is taken by a young man of large fortune from the north of England; that he came down on Monday in a chaise and four to see the place, and was so much delighted with it that he agreed with Mr. Morris immediately; that he is to take possession before Michaelmas, and some of his servants are to be in the house by the end of next week." "What is his name?" "Bingley." "Is he married or single?" "Oh, single, my dear, to be sure! A single man of large fortune; four or five thousand a year. What a fine thing for our girls!" "How so? how can it affect them?" "My dear Mr. Bennet," replied his wife, "how can you be so tiresome? You must know that I am thinking of his marrying one of them." "Is that his design in settling here?" "Design? Nonsense, how can you talk so! But it is very likely that he may fall in love with one of them, and therefore you must visit him as soon as he comes." "I see no occasion for that. You and the girls may go--or you may send them by themselves, which perhaps will be still better; for as you are as handsome as any of them, Mr. Bingley might like you the best of the party." "My dear, you flatter me. I certainly have had my share of beauty, but I do not pretend to be anything extraordinary now. When a woman has five grown-up daughters, she ought to give over thinking of her own beauty." "In such cases, a woman has not often much beauty to think of." "But, my dear, you must indeed go and see Mr. Bingley when he comes into the neighbourhood." "It is more than I engage for, I assure you." "But consider your daughters. Only think what an establishment it would be for one of them. Sir William and Lady Lucas are determined to go, merely on that account; for in general, you know, they visit no new comers. Indeed you must go, for it will be impossible for us to visit him, if you do not." "You are over scrupulous, surely. I dare say Mr. Bingley will be very glad to see you; and I will send a few lines by you to assure him of my hearty consent to his marrying whichever he chooses of the girls--though I must throw in a good word for my little Lizzy." "I desire you will do no such thing. Lizzy is not a bit better than the others: and I am sure she is not half so handsome as Jane, nor half so good-humoured as Lydia. But you are always giving her the preference." "They have none of them much to recommend them," replied he: "they are all silly and ignorant like other girls; but Lizzy has something more of quickness than her sisters." "Mr. Bennet, how can you abuse your own children in such a way? You take delight in vexing me. You have no compassion on my poor nerves." "You mistake me, my dear. I have a high respect for your nerves. They are my old friends. I have heard you mention them with consideration these twenty years at least." "Ah, you do not know what I suffer." "But I hope you will get over it, and live to see many young men of four thousand a year come into the neighbourhood." "It will be no use to us, if twenty such should come, since you will not visit them." "Depend upon it, my dear, that when there are twenty, I will visit them all." Mr. Bennet was so odd a mixture of quick parts, sarcastic humour, reserve, and caprice, that the experience of three-and-twenty years had been insufficient to make his wife understand his character. Her mind was less difficult to develope. She was a woman of mean understanding, little information, and uncertain temper. When she was discontented, she fancied herself nervous. The business of her life was to get her daughters married: its solace was visiting and news. ] ] Mr. Bennet was among the earliest of those who waited on Mr. Bingley. He had always intended to visit him, though to the last always assuring his wife that he should not go; and till the evening after the visit was paid she had no knowledge of it. It was then disclosed in the following manner. Observing his second daughter employed in trimming a hat, he suddenly addressed her with,-- "I hope Mr. Bingley will like it, Lizzy." "We are not in a way to know what Mr. Bingley likes," said her mother, resentfully, "since we are not to visit." "But you forget, mamma," said Elizabeth, "that we shall meet him at the assemblies, and that Mrs. Long has promised to introduce him." "I do not believe Mrs. Long will do any such thing. She has two nieces of her own. She is a selfish, hypocritical woman, and I have no opinion of her." "No more have I," said Mr. Bennet; "and I am glad to find that you do not depend on her serving you." Mrs. Bennet deigned not to make any reply; but, unable to contain herself, began scolding one of her daughters. "Don't keep coughing so, Kitty, for heaven's sake! Have a little compassion on my nerves. You tear them to pieces." "Kitty has no discretion in her coughs," said her father; "she times them ill." "I do not cough for my own amusement," replied Kitty, fretfully. "When is your next ball to be, Lizzy?" "To-morrow fortnight." "Ay, so it is," cried her mother, "and Mrs. Long does not come back till the day before; so, it will be impossible for her to introduce him, for she will not know him herself." "Then, my dear, you may have the advantage of your friend, and introduce Mr. Bingley to her." "Impossible, Mr. Bennet, impossible, when I am not acquainted with him myself; how can you be so teasing?" "I honour your circumspection. A fortnight's acquaintance is certainly very little. One cannot know what a man really is by the end of a fortnight. But if we do not venture, somebody else will; and after all, Mrs. Long and her nieces must stand their chance; and, therefore, as she will think it an act of kindness, if you decline the office, I will take it on myself." The girls stared at their father. Mrs. Bennet said only, "Nonsense, nonsense!" "What can be the meaning of that emphatic exclamation?" cried he. "Do you consider the forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite agree with you there. What say you, Mary? For you are a young lady of deep reflection, I know, and read great books, and make extracts." Mary wished to say something very sensible, but knew not how. "While Mary is adjusting her ideas," he continued, "let us return to Mr. Bingley." "I am sick of Mr. Bingley," cried his wife. "I am sorry to hear that; but why did you not tell me so before? If I had known as much this morning, I certainly would not have called on him. It is very unlucky; but as I have actually paid the visit, we cannot escape the acquaintance now." The astonishment of the ladies was just what he wished--that of Mrs. Bennet perhaps surpassing the rest; though when the first tumult of joy was over, she began to declare that it was what she had expected all the while. "How good it was in you, my dear Mr. Bennet! But I knew I should persuade you at last. I was sure you loved your girls too well to neglect such an acquaintance. Well, how pleased I am! And it is such a good joke, too, that you should have gone this morning, and never said a word about it till now." "Now, Kitty, you may cough as much as you choose," said Mr. Bennet; and, as he spoke, he left the room, fatigued with the raptures of his wife. "What an excellent father you have, girls," said she, when the door was shut. "I do not know how you will ever make him amends for his kindness; or me either, for that matter. At our time of life, it is not so pleasant, I can tell you, to be making new acquaintances every day; but for your sakes we would do anything. Lydia, my love, though you are the youngest, I dare say Mr. Bingley will dance with you at the next ball." "Oh," said Lydia, stoutly, "I am not afraid; for though I am the youngest, I'm the tallest." The rest of the evening was spent in conjecturing how soon he would return Mr. Bennet's visit, and determining when they should ask him to dinner. Not all that Mrs. Bennet, however, with the assistance of her five daughters, could ask on the subject, was sufficient to draw from her husband any satisfactory description of Mr. Bingley. They attacked him in various ways, with barefaced questions, ingenious suppositions, and distant surmises; but he eluded the skill of them all; and they were at last obliged to accept the second-hand intelligence of their neighbour, Lady Lucas. Her report was highly favourable. Sir William had been delighted with him. He was quite young, wonderfully handsome, extremely agreeable, and, to crown the whole, he meant to be at the next assembly with a large party. Nothing could be more delightful! To be fond of dancing was a certain step towards falling in love; and very lively hopes of Mr. Bingley's heart were entertained. "If I can but see one of my daughters happily settled at Netherfield," said Mrs. Bennet to her husband, "and all the others equally well married, I shall have nothing to wish for." In a few days Mr. Bingley returned Mr. Bennet's visit, and sat about ten minutes with him in his library. He had entertained hopes of being admitted to a sight of the young ladies, of whose beauty he had heard much; but he saw only the father. The ladies were somewhat more fortunate, for they had the advantage of ascertaining, from an upper window, that he wore a blue coat and rode a black horse. An invitation to dinner was soon afterwards despatched; and already had Mrs. Bennet planned the courses that were to do credit to her housekeeping, when an answer arrived which deferred it all. Mr. Bingley was obliged to be in town the following day, and consequently unable to accept the honour of their invitation, etc. Mrs. Bennet was quite disconcerted. She could not imagine what business he could have in town so soon after his arrival in Hertfordshire; and she began to fear that he might always be flying about from one place to another, and never settled at Netherfield as he ought to be. Lady Lucas quieted her fears a little by starting the idea of his ] being gone to London only to get a large party for the ball; and a report soon followed that Mr. Bingley was to bring twelve ladies and seven gentlemen with him to the assembly. The girls grieved over such a number of ladies; but were comforted the day before the ball by hearing that, instead of twelve, he had brought only six with him from London, his five sisters and a cousin. And when the party entered the assembly-room, it consisted of only five altogether: Mr. Bingley, his two sisters, the husband of the eldest, and another young man. Mr. Bingley was good-looking and gentlemanlike: he had a pleasant countenance, and easy, unaffected manners. His sisters were fine women, with an air of decided fashion. His brother-in-law, Mr. Hurst, merely looked the gentleman; but his friend Mr. Darcy soon drew the attention of the room by his fine, tall person, handsome features, noble mien, and the report, which was in general circulation within five minutes after his entrance, of his having ten thousand a year. The gentlemen pronounced him to be a fine figure of a man, the ladies declared he was much handsomer than Mr. Bingley, and he was looked at with great admiration for about half the evening, till his manners gave a disgust which turned the tide of his popularity; for he was discovered to be proud, to be above his company, and above being pleased; and not all his large estate in Derbyshire could save him from having a most forbidding, disagreeable countenance, and being unworthy to be compared with his friend. Mr. Bingley had soon made himself acquainted with all the principal people in the room: he was lively and unreserved, danced every dance, was angry that the ball closed so early, and talked of giving one himself at Netherfield. Such amiable qualities must speak for themselves. What a contrast between him and his friend! Mr. Darcy danced only once with Mrs. Hurst and once with Miss Bingley, declined being introduced to any other lady, and spent the rest of the evening in walking about the room, speaking occasionally to one of his own party. His character was decided. He was the proudest, most disagreeable man in the world, and everybody hoped that he would never come there again. Amongst the most violent against him was Mrs. Bennet, whose dislike of his general behaviour was sharpened into particular resentment by his having slighted one of her daughters. Elizabeth Bennet had been obliged, by the scarcity of gentlemen, to sit down for two dances; and during part of that time, Mr. Darcy had been standing near enough for her to overhear a conversation between him and Mr. Bingley, who came from the dance for a few minutes to press his friend to join it. "Come, Darcy," said he, "I must have you dance. I hate to see you standing about by yourself in this stupid manner. You had much better dance." "I certainly shall not. You know how I detest it, unless I am particularly acquainted with my partner. At such an assembly as this, it would be insupportable. Your sisters are engaged, and there is not another woman in the room whom it would not be a punishment to me to stand up with." "I would not be so fastidious as you are," cried Bingley, "for a kingdom! Upon my honour, I never met with so many pleasant girls in my life as I have this evening; and there are several of them, you see, uncommonly pretty." "You are dancing with the only handsome girl in the room," said Mr. Darcy, looking at the eldest Miss Bennet. "Oh, she is the most beautiful creature I ever beheld! But there is one of her sisters sitting down just behind you, who is very pretty, and I dare say very agreeable. Do let me ask my partner to introduce you." ] "Which do you mean?" and turning round, he looked for a moment at Elizabeth, till, catching her eye, he withdrew his own, and coldly said, "She is tolerable: but not handsome enough to tempt me; and I am in no humour at present to give consequence to young ladies who are slighted by other men. You had better return to your partner and enjoy her smiles, for you are wasting your time with me." Mr. Bingley followed his advice. Mr. Darcy walked off; and Elizabeth remained with no very cordial feelings towards him. She told the story, however, with great spirit among her friends; for she had a lively, playful disposition, which delighted in anything ridiculous. The evening altogether passed off pleasantly to the whole family. Mrs. Bennet had seen her eldest daughter much admired by the Netherfield party. Mr. Bingley had danced with her twice, and she had been distinguished by his sisters. Jane was as much gratified by this as her mother could be, though in a quieter way. Elizabeth felt Jane's pleasure. Mary had heard herself mentioned to Miss Bingley as the most accomplished girl in the neighbourhood; and Catherine and Lydia had been fortunate enough to be never without partners, which was all that they had yet learnt to care for at a ball. They returned, therefore, in good spirits to Longbourn, the village where they lived, and of which they were the principal inhabitants. They found Mr. Bennet still up. With a book, he was regardless of time; and on the present occasion he had a good deal of curiosity as to the event of an evening which had raised such splendid expectations. He had rather hoped that all his wife's views on the stranger would be disappointed; but he soon found that he had a very different story to hear. "Oh, my dear Mr. Bennet," as she entered the room, "we have had a most delightful evening, a most excellent ball. I wish you had been there. Jane was so admired, nothing could be like it. Everybody said how well she looked; and Mr. Bingley thought her quite beautiful, and danced with her twice. Only think of that, my dear: he actually danced with her twice; and she was the only creature in the room that he asked a second time. First of all, he asked Miss Lucas. I was so vexed to see him stand up with her; but, however, he did not admire her at all; indeed, nobody can, you know; and he seemed quite struck with Jane as she was going down the dance. So he inquired who she was, and got introduced, and asked her for the two next. Then, the two third he danced with Miss King, and the two fourth with Maria Lucas, and the two fifth with Jane again, and the two sixth with Lizzy, and the Boulanger----" "If he had had any compassion for me," cried her husband impatiently, "he would not have danced half so much! For God's sake, say no more of his partners. O that he had sprained his ancle in the first dance!" "Oh, my dear," continued Mrs. Bennet, "I am quite delighted with him. He is so excessively handsome! and his sisters are charming women. I never in my life saw anything more elegant than their dresses. I dare say the lace upon Mrs. Hurst's gown----" Here she was interrupted again. Mr. Bennet protested against any description of finery. She was therefore obliged to seek another branch of the subject, and related, with much bitterness of spirit, and some exaggeration, the shocking rudeness of Mr. Darcy. "But I can assure you," she added, "that Lizzy does not lose much by not suiting his fancy; for he is a most disagreeable, horrid man, not at all worth pleasing. So high and so conceited, that there was no enduring him! He walked here, and he walked there, fancying himself so very great! Not handsome enough to dance with! I wish you had been there, my dear, to have given him one of your set-downs. I quite detest the man." When Jane and Elizabeth were alone, the former, who had been cautious in her praise of Mr. Bingley before, expressed to her sister how very much she admired him. "He is just what a young man ought to be," said she, "sensible, good-humoured, lively; and I never saw such happy manners! so much ease, with such perfect good breeding!" "He is also handsome," replied Elizabeth, "which a young man ought likewise to be if he possibly can. His character is thereby complete." "I was very much flattered by his asking me to dance a second time. I did not expect such a compliment." "Did not you? I did for you. But that is one great difference between us. Compliments always take you by surprise, and me never. What could be more natural than his asking you again? He could not help seeing that you were about five times as pretty as every other woman in the room. No thanks to his gallantry for that. Well, he certainly is very agreeable, and I give you leave to like him. You have liked many a stupider person." "Dear Lizzy!" "Oh, you are a great deal too apt, you know, to like people in general. You never see a fault in anybody. All the world are good and agreeable in your eyes. I never heard you speak ill of a human being in my life." "I would wish not to be hasty in censuring anyone; but I always speak what I think." "I know you do: and it is that which makes the wonder. With your good sense, to be so honestly blind to the follies and nonsense of others! Affectation of candour is common enough; one meets with it everywhere. But to be candid without ostentation or design,--to take the good of everybody's character and make it still better, and say nothing of the bad,--belongs to you alone. And so, you like this man's sisters, too, do you? Their manners are not equal to his." "Certainly not, at first; but they are very pleasing women when you converse with them. Miss Bingley is to live with her brother, and keep his house; and I am much mistaken if we shall not find a very charming neighbour in her." Elizabeth listened in silence, but was not convinced: their behaviour at the assembly had not been calculated to please in general; and with more quickness of observation and less pliancy of temper than her sister, and with a judgment, too, unassailed by any attention to herself, she was very little disposed to approve them. They were, in fact, very fine ladies; not deficient in good-humour when they were pleased, nor in the power of being agreeable where they chose it; but proud and conceited. They were rather handsome; had been educated in one of the first private seminaries in town; had a fortune of twenty thousand pounds; were in the habit of spending more than they ought, and of associating with people of rank; and were, therefore, in every respect entitled to think well of themselves and meanly of others. They were of a respectable family in the north of England; a circumstance more deeply impressed on their memories than that their brother's fortune and their own had been acquired by trade. Mr. Bingley inherited property to the amount of nearly a hundred thousand pounds from his father, who had intended to purchase an estate, but did not live to do it. Mr. Bingley intended it likewise, and sometimes made choice of his county; but, as he was now provided with a good house and the liberty of a manor, it was doubtful to many of those who best knew the easiness of his temper, whether he might not spend the remainder of his days at Netherfield, and leave the next generation to purchase. His sisters were very anxious for his having an estate of his own; but though he was now established only as a tenant, Miss Bingley was by no means unwilling to preside at his table; nor was Mrs. Hurst, who had married a man of more fashion than fortune, less disposed to consider his house as her home when it suited her. Mr. Bingley had not been of age two years when he was tempted, by an accidental recommendation, to look at Netherfield House. He did look at it, and into it, for half an hour; was pleased with the situation and the principal rooms, satisfied with what the owner said in its praise, and took it immediately. Between him and Darcy there was a very steady friendship, in spite of a great opposition of character. Bingley was endeared to Darcy by the easiness, openness, and ductility of his temper, though no disposition could offer a greater contrast to his own, and though with his own he never appeared dissatisfied. On the strength of Darcy's regard, Bingley had the firmest reliance, and of his judgment the highest opinion. In understanding, Darcy was the superior. Bingley was by no means deficient; but Darcy was clever. He was at the same time haughty, reserved, and fastidious; and his manners, though well bred, were not inviting. In that respect his friend had greatly the advantage. Bingley was sure of being liked wherever he appeared; Darcy was continually giving offence. The manner in which they spoke of the Meryton assembly was sufficiently characteristic. Bingley had never met with pleasanter people or prettier girls in his life; everybody had been most kind and attentive to him; there had been no formality, no stiffness; he had soon felt acquainted with all the room; and as to Miss Bennet, he could not conceive an angel more beautiful. Darcy, on the contrary, had seen a collection of people in whom there was little beauty and no fashion, for none of whom he had felt the smallest interest, and from none received either attention or pleasure. Miss Bennet he acknowledged to be pretty; but she smiled too much. Mrs. Hurst and her sister allowed it to be so; but still they admired her and liked her, and pronounced her to be a sweet girl, and one whom they should not object to know more of. Miss Bennet was therefore established as a sweet girl; and their brother felt authorized by such commendation to think of her as he chose. ] Within a short walk of Longbourn lived a family with whom the Bennets were particularly intimate. Sir William Lucas had been formerly in trade in Meryton, where he had made a tolerable fortune, and risen to the honour of knighthood by an address to the king during his mayoralty. The distinction had, perhaps, been felt too strongly. It had given him a disgust to his business and to his residence in a small market town; and, quitting them both, he had removed with his family to a house about a mile from Meryton, denominated from that period Lucas Lodge; where he could think with pleasure of his own importance, and, unshackled by business, occupy himself solely in being civil to all the world. For, though elated by his rank, it did not render him supercilious; on the contrary, he was all attention to everybody. By nature inoffensive, friendly, and obliging, his presentation at St. James's had made him courteous. Lady Lucas was a very good kind of woman, not too clever to be a valuable neighbour to Mrs. Bennet. They had several children. The eldest of them, a sensible, intelligent young woman, about twenty-seven, was Elizabeth's intimate friend. That the Miss Lucases and the Miss Bennets should meet to talk over a ball was absolutely necessary; and the morning after the assembly brought the former to Longbourn to hear and to communicate. "You began the evening well, Charlotte," said Mrs. Bennet, with civil self-command, to Miss Lucas. "You were Mr. Bingley's first choice." "Yes; but he seemed to like his second better." "Oh, you mean Jane, I suppose, because he danced with her twice. To be sure that did seem as if he admired her--indeed, I rather believe he did--I heard something about it--but I hardly know what--something about Mr. Robinson." "Perhaps you mean what I overheard between him and Mr. Robinson: did not I mention it to you? Mr. Robinson's asking him how he liked our Meryton assemblies, and whether he did not think there were a great many pretty women in the room, and which he thought the prettiest? and his answering immediately to the last question, 'Oh, the eldest Miss Bennet, beyond a doubt: there cannot be two opinions on that point.'" "Upon my word! Well, that was very decided, indeed--that does seem as if--but, however, it may all come to nothing, you know." "My overhearings were more to the purpose than yours, Eliza," said Charlotte. "Mr. Darcy is not so well worth listening to as his friend, is he? Poor Eliza! to be only just tolerable." "I beg you will not put it into Lizzy's head to be vexed by his ill-treatment, for he is such a disagreeable man that it would be quite a misfortune to be liked by him. Mrs. Long told me last night that he sat close to her for half an hour without once opening his lips." ] "Are you quite sure, ma'am? Is not there a little mistake?" said Jane. "I certainly saw Mr. Darcy speaking to her." "Ay, because she asked him at last how he liked Netherfield, and he could not help answering her; but she said he seemed very angry at being spoke to." "Miss Bingley told me," said Jane, "that he never speaks much unless among his intimate acquaintance. With them he is remarkably agreeable." "I do not believe a word of it, my dear. If he had been so very agreeable, he would have talked to Mrs. Long. But I can guess how it was; everybody says that he is eat up with pride, and I dare say he had heard somehow that Mrs. Long does not keep a carriage, and had to come to the ball in a hack chaise." "I do not mind his not talking to Mrs. Long," said Miss Lucas, "but I wish he had danced with Eliza." "Another time, Lizzy," said her mother, "I would not dance with him, if I were you." "I believe, ma'am, I may safely promise you never to dance with him.""",
		"Grimms' Fairy Tales (transl. E. Taylor and M. Edwardes)" => raw"""Hard by a great forest dwelt a poor wood-cutter with his wife and his two children. The boy was called Hansel and the girl Gretel. He had little to bite and to break, and once when great dearth fell on the land, he could no longer procure even daily bread. Now when he thought over this by night in his bed, and tossed about in his anxiety, he groaned and said to his wife: 'What is to become of us? How are we to feed our poor children, when we no longer have anything even for ourselves?' 'I'll tell you what, husband,' answered the woman, 'early tomorrow morning we will take the children out into the forest to where it is the thickest; there we will light a fire for them, and give each of them one more piece of bread, and then we will go to our work and leave them alone. They will not find the way home again, and we shall be rid of them.' 'No, wife,' said the man, 'I will not do that; how can I bear to leave my children alone in the forest?--the wild animals would soon come and tear them to pieces.' 'O, you fool!' said she, 'then we must all four die of hunger, you may as well plane the planks for our coffins,' and she left him no peace until he consented. 'But I feel very sorry for the poor children, all the same,' said the man. The two children had also not been able to sleep for hunger, and had heard what their stepmother had said to their father. Gretel wept bitter tears, and said to Hansel: 'Now all is over with us.' 'Be quiet, Gretel,' said Hansel, 'do not distress yourself, I will soon find a way to help us.' And when the old folks had fallen asleep, he got up, put on his little coat, opened the door below, and crept outside. The moon shone brightly, and the white pebbles which lay in front of the house glittered like real silver pennies. Hansel stooped and stuffed the little pocket of his coat with as many as he could get in. Then he went back and said to Gretel: 'Be comforted, dear little sister, and sleep in peace, God will not forsake us,' and he lay down again in his bed. When day dawned, but before the sun had risen, the woman came and awoke the two children, saying: 'Get up, you sluggards! we are going into the forest to fetch wood.' She gave each a little piece of bread, and said: 'There is something for your dinner, but do not eat it up before then, for you will get nothing else.' Gretel took the bread under her apron, as Hansel had the pebbles in his pocket. Then they all set out together on the way to the forest. When they had walked a short time, Hansel stood still and peeped back at the house, and did so again and again. His father said: 'Hansel, what are you looking at there and staying behind for? Pay attention, and do not forget how to use your legs.' 'Ah, father,' said Hansel, 'I am looking at my little white cat, which is sitting up on the roof, and wants to say goodbye to me.' The wife said: 'Fool, that is not your little cat, that is the morning sun which is shining on the chimneys.' Hansel, however, had not been looking back at the cat, but had been constantly throwing one of the white pebble-stones out of his pocket on the road. When they had reached the middle of the forest, the father said: 'Now, children, pile up some wood, and I will light a fire that you may not be cold.' Hansel and Gretel gathered brushwood together, as high as a little hill. The brushwood was lighted, and when the flames were burning very high, the woman said: 'Now, children, lay yourselves down by the fire and rest, we will go into the forest and cut some wood. When we have done, we will come back and fetch you away.' Hansel and Gretel sat by the fire, and when noon came, each ate a little piece of bread, and as they heard the strokes of the wood-axe they believed that their father was near. It was not the axe, however, but a branch which he had fastened to a withered tree which the wind was blowing backwards and forwards. And as they had been sitting such a long time, their eyes closed with fatigue, and they fell fast asleep. When at last they awoke, it was already dark night. Gretel began to cry and said: 'How are we to get out of the forest now?' But Hansel comforted her and said: 'Just wait a little, until the moon has risen, and then we will soon find the way.' And when the full moon had risen, Hansel took his little sister by the hand, and followed the pebbles which shone like newly-coined silver pieces, and showed them the way. They walked the whole night long, and by break of day came once more to their father's house. They knocked at the door, and when the woman opened it and saw that it was Hansel and Gretel, she said: 'You naughty children, why have you slept so long in the forest?--we thought you were never coming back at all!' The father, however, rejoiced, for it had cut him to the heart to leave them behind alone. Not long afterwards, there was once more great dearth throughout the land, and the children heard their mother saying at night to their father: 'Everything is eaten again, we have one half loaf left, and that is the end. The children must go, we will take them farther into the wood, so that they will not find their way out again; there is no other means of saving ourselves!' The man's heart was heavy, and he thought: 'It would be better for you to share the last mouthful with your children.' The woman, however, would listen to nothing that he had to say, but scolded and reproached him. He who says A must say B, likewise, and as he had yielded the first time, he had to do so a second time also. The children, however, were still awake and had heard the conversation. When the old folks were asleep, Hansel again got up, and wanted to go out and pick up pebbles as he had done before, but the woman had locked the door, and Hansel could not get out. Nevertheless he comforted his little sister, and said: 'Do not cry, Gretel, go to sleep quietly, the good God will help us.' Early in the morning came the woman, and took the children out of their beds. Their piece of bread was given to them, but it was still smaller than the time before. On the way into the forest Hansel crumbled his in his pocket, and often stood still and threw a morsel on the ground. 'Hansel, why do you stop and look round?' said the father, 'go on.' 'I am looking back at my little pigeon which is sitting on the roof, and wants to say goodbye to me,' answered Hansel. 'Fool!' said the woman, 'that is not your little pigeon, that is the morning sun that is shining on the chimney.' Hansel, however little by little, threw all the crumbs on the path. The woman led the children still deeper into the forest, where they had never in their lives been before. Then a great fire was again made, and the mother said: 'Just sit there, you children, and when you are tired you may sleep a little; we are going into the forest to cut wood, and in the evening when we are done, we will come and fetch you away.' When it was noon, Gretel shared her piece of bread with Hansel, who had scattered his by the way. Then they fell asleep and evening passed, but no one came to the poor children. They did not awake until it was dark night, and Hansel comforted his little sister and said: 'Just wait, Gretel, until the moon rises, and then we shall see the crumbs of bread which I have strewn about, they will show us our way home again.' When the moon came they set out, but they found no crumbs, for the many thousands of birds which fly about in the woods and fields had picked them all up. Hansel said to Gretel: 'We shall soon find the way,' but they did not find it. They walked the whole night and all the next day too from morning till evening, but they did not get out of the forest, and were very hungry, for they had nothing to eat but two or three berries, which grew on the ground. And as they were so weary that their legs would carry them no longer, they lay down beneath a tree and fell asleep. It was now three mornings since they had left their father's house. They began to walk again, but they always came deeper into the forest, and if help did not come soon, they must die of hunger and weariness. When it was mid-day, they saw a beautiful snow-white bird sitting on a bough, which sang so delightfully that they stood still and listened to it. And when its song was over, it spread its wings and flew away before them, and they followed it until they reached a little house, on the roof of which it alighted; and when they approached the little house they saw that it was built of bread and covered with cakes, but that the windows were of clear sugar. 'We will set to work on that,' said Hansel, 'and have a good meal. I will eat a bit of the roof, and you Gretel, can eat some of the window, it will taste sweet.' Hansel reached up above, and broke off a little of the roof to try how it tasted, and Gretel leant against the window and nibbled at the panes. Then a soft voice cried from the parlour: 'Nibble, nibble, gnaw, Who is nibbling at my little house?' The children answered: 'The wind, the wind, The heaven-born wind,' and went on eating without disturbing themselves. Hansel, who liked the taste of the roof, tore down a great piece of it, and Gretel pushed out the whole of one round window-pane, sat down, and enjoyed herself with it. Suddenly the door opened, and a woman as old as the hills, who supported herself on crutches, came creeping out. Hansel and Gretel were so terribly frightened that they let fall what they had in their hands. The old woman, however, nodded her head, and said: 'Oh, you dear children, who has brought you here? do come in, and stay with me. No harm shall happen to you.' She took them both by the hand, and led them into her little house. Then good food was set before them, milk and pancakes, with sugar, apples, and nuts. Afterwards two pretty little beds were covered with clean white linen, and Hansel and Gretel lay down in them, and thought they were in heaven. The old woman had only pretended to be so kind; she was in reality a wicked witch, who lay in wait for children, and had only built the little house of bread in order to entice them there. When a child fell into her power, she killed it, cooked and ate it, and that was a feast day with her. Witches have red eyes, and cannot see far, but they have a keen scent like the beasts, and are aware when human beings draw near. When Hansel and Gretel came into her neighbourhood, she laughed with malice, and said mockingly: 'I have them, they shall not escape me again!' Early in the morning before the children were awake, she was already up, and when she saw both of them sleeping and looking so pretty, with their plump and rosy cheeks she muttered to herself: 'That will be a dainty mouthful!' Then she seized Hansel with her shrivelled hand, carried him into a little stable, and locked him in behind a grated door. Scream as he might, it would not help him. Then she went to Gretel, shook her till she awoke, and cried: 'Get up, lazy thing, fetch some water, and cook something good for your brother, he is in the stable outside, and is to be made fat. When he is fat, I will eat him.' Gretel began to weep bitterly, but it was all in vain, for she was forced to do what the wicked witch commanded. And now the best food was cooked for poor Hansel, but Gretel got nothing but crab-shells. Every morning the woman crept to the little stable, and cried: 'Hansel, stretch out your finger that I may feel if you will soon be fat.' Hansel, however, stretched out a little bone to her, and the old woman, who had dim eyes, could not see it, and thought it was Hansel's finger, and was astonished that there was no way of fattening him. When four weeks had gone by, and Hansel still remained thin, she was seized with impatience and would not wait any longer. 'Now, then, Gretel,' she cried to the girl, 'stir yourself, and bring some water. Let Hansel be fat or lean, tomorrow I will kill him, and cook him.' Ah, how the poor little sister did lament when she had to fetch the water, and how her tears did flow down her cheeks! 'Dear God, do help us,' she cried. 'If the wild beasts in the forest had but devoured us, we should at any rate have died together.' 'Just keep your noise to yourself,' said the old woman, 'it won't help you at all.' Early in the morning, Gretel had to go out and hang up the cauldron with the water, and light the fire. 'We will bake first,' said the old woman, 'I have already heated the oven, and kneaded the dough.' She pushed poor Gretel out to the oven, from which flames of fire were already darting. 'Creep in,' said the witch, 'and see if it is properly heated, so that we can put the bread in.' And once Gretel was inside, she intended to shut the oven and let her bake in it, and then she would eat her, too. But Gretel saw what she had in mind, and said: 'I do not know how I am to do it; how do I get in?' 'Silly goose,' said the old woman. 'The door is big enough; just look, I can get in myself!' and she crept up and thrust her head into the oven. Then Gretel gave her a push that drove her far into it, and shut the iron door, and fastened the bolt. Oh! then she began to howl quite horribly, but Gretel ran away and the godless witch was miserably burnt to death. Gretel, however, ran like lightning to Hansel, opened his little stable, and cried: 'Hansel, we are saved! The old witch is dead!' Then Hansel sprang like a bird from its cage when the door is opened. How they did rejoice and embrace each other, and dance about and kiss each other! And as they had no longer any need to fear her, they went into the witch's house, and in every corner there stood chests full of pearls and jewels. 'These are far better than pebbles!' said Hansel, and thrust into his pockets whatever could be got in, and Gretel said: 'I, too, will take something home with me,' and filled her pinafore full. 'But now we must be off,' said Hansel, 'that we may get out of the witch's forest.' When they had walked for two hours, they came to a great stretch of water. 'We cannot cross,' said Hansel, 'I see no foot-plank, and no bridge.' 'And there is also no ferry,' answered Gretel, 'but a white duck is swimming there: if I ask her, she will help us over.' Then she cried: 'Little duck, little duck, dost thou see, Hansel and Gretel are waiting for thee? There's never a plank, or bridge in sight, Take us across on thy back so white.' The duck came to them, and Hansel seated himself on its back, and told his sister to sit by him. 'No,' replied Gretel, 'that will be too heavy for the little duck; she shall take us across, one after the other.' The good little duck did so, and when they were once safely across and had walked for a short time, the forest seemed to be more and more familiar to them, and at length they saw from afar their father's house. Then they began to run, rushed into the parlour, and threw themselves round their father's neck. The man had not known one happy hour since he had left the children in the forest; the woman, however, was dead. Gretel emptied her pinafore until pearls and precious stones ran about the room, and Hansel threw one handful after another out of his pocket to add to them. Then all anxiety was at an end, and they lived together in perfect happiness. My tale is done, there runs a mouse; whosoever catches it, may make himself a big fur cap out of it. THE MOUSE, THE BIRD, Once upon a time, a mouse, a bird, and a sausage, entered into partnership and set up house together. For a long time all went well; they lived in great comfort, and prospered so far as to be able to add considerably to their stores. The bird's duty was to fly daily into the wood and bring in fuel; the mouse fetched the water, and the sausage saw to the cooking. When people are too well off they always begin to long for something new. And so it came to pass, that the bird, while out one day, met a fellow bird, to whom he boastfully expatiated on the excellence of his household arrangements. But the other bird sneered at him for being a poor simpleton, who did all the hard work, while the other two stayed at home and had a good time of it. For, when the mouse had made the fire and fetched in the water, she could retire into her little room and rest until it was time to set the table. The sausage had only to watch the pot to see that the food was properly cooked, and when it was near dinner-time, he just threw himself into the broth, or rolled in and out among the vegetables three or four times, and there they were, buttered, and salted, and ready to be served. Then, when the bird came home and had laid aside his burden, they sat down to table, and when they had finished their meal, they could sleep their fill till the following morning: and that was really a very delightful life. Influenced by those remarks, the bird next morning refused to bring in the wood, telling the others that he had been their servant long enough, and had been a fool into the bargain, and that it was now time to make a change, and to try some other way of arranging the work. Beg and pray as the mouse and the sausage might, it was of no use; the bird remained master of the situation, and the venture had to be made. They therefore drew lots, and it fell to the sausage to bring in the wood, to the mouse to cook, and to the bird to fetch the water. And now what happened? The sausage started in search of wood, the bird made the fire, and the mouse put on the pot, and then these two waited till the sausage returned with the fuel for the following day. But the sausage remained so long away, that they became uneasy, and the bird flew out to meet him. He had not flown far, however, when he came across a dog who, having met the sausage, had regarded him as his legitimate booty, and so seized and swallowed him. The bird complained to the dog of this bare-faced robbery, but nothing he said was of any avail, for the dog answered that he found false credentials on the sausage, and that was the reason his life had been forfeited. He picked up the wood, and flew sadly home, and told the mouse all he had seen and heard. They were both very unhappy, but agreed to make the best of things and to remain with one another. So now the bird set the table, and the mouse looked after the food and, wishing to prepare it in the same way as the sausage, by rolling in and out among the vegetables to salt and butter them, she jumped into the pot; but she stopped short long before she reached the bottom, having already parted not only with her skin and hair, but also with life. Presently the bird came in and wanted to serve up the dinner, but he could nowhere see the cook. In his alarm and flurry, he threw the wood here and there about the floor, called and searched, but no cook was to be found. Then some of the wood that had been carelessly thrown down, caught fire and began to blaze. The bird hastened to fetch some water, but his pail fell into the well, and he after it, and as he was unable to recover himself, he was drowned. Once upon a time there was a widow who had two daughters; one of them was beautiful and industrious, the other ugly and lazy. The mother, however, loved the ugly and lazy one best, because she was her own daughter, and so the other, who was only her stepdaughter, was made to do all the work of the house, and was quite the Cinderella of the family. Her stepmother sent her out every day to sit by the well in the high road, there to spin until she made her fingers bleed. Now it chanced one day that some blood fell on to the spindle, and as the girl stopped over the well to wash it off, the spindle suddenly sprang out of her hand and fell into the well. She ran home crying to tell of her misfortune, but her stepmother spoke harshly to her, and after giving her a violent scolding, said unkindly, 'As you have let the spindle fall into the well you may go yourself and fetch it out.' The girl went back to the well not knowing what to do, and at last in her distress she jumped into the water after the spindle. She remembered nothing more until she awoke and found herself in a beautiful meadow, full of sunshine, and with countless flowers blooming in every direction. She walked over the meadow, and presently she came upon a baker's oven full of bread, and the loaves cried out to her, 'Take us out, take us out, or alas! we shall be burnt to a cinder; we were baked through long ago.' So she took the bread-shovel and drew them all out. She went on a little farther, till she came to a tree full of apples. 'Shake me, shake me, I pray,' cried the tree; 'my apples, one and all, are ripe.' So she shook the tree, and the apples came falling down upon her like rain; but she continued shaking until there was not a single apple left upon it. Then she carefully gathered the apples together in a heap and walked on again. The next thing she came to was a little house, and there she saw an old woman looking out, with such large teeth, that she was terrified, and turned to run away. But the old woman called after her, 'What are you afraid of, dear child? Stay with me; if you will do the work of my house properly for me, I will make you very happy. You must be very careful, however, to make my bed in the right way, for I wish you always to shake it thoroughly, so that the feathers fly about; then they say, down there in the world, that it is snowing; for I am Mother Holle.' The old woman spoke so kindly, that the girl summoned up courage and agreed to enter into her service. She took care to do everything according to the old woman's bidding and every time she made the bed she shook it with all her might, so that the feathers flew about like so many snowflakes. The old woman was as good as her word: she never spoke angrily to her, and gave her roast and boiled meats every day. So she stayed on with Mother Holle for some time, and then she began to grow unhappy. She could not at first tell why she felt sad, but she became conscious at last of great longing to go home; then she knew she was homesick, although she was a thousand times better off with Mother Holle than with her mother and sister. After waiting awhile, she went to Mother Holle and said, 'I am so homesick, that I cannot stay with you any longer, for although I am so happy here, I must return to my own people.' Then Mother Holle said, 'I am pleased that you should want to go back to your own people, and as you have served me so well and faithfully, I will take you home myself.' Thereupon she led the girl by the hand up to a broad gateway. The gate was opened, and as the girl passed through, a shower of gold fell upon her, and the gold clung to her, so that she was covered with it from head to foot. 'That is a reward for your industry,' said Mother Holle, and as she spoke she handed her the spindle which she had dropped into the well. The gate was then closed, and the girl found herself back in the old world close to her mother's house. As she entered the courtyard, the cock who was perched on the well, called out: 'Cock-a-doodle-doo! Your golden daughter's come back to you.' Then she went in to her mother and sister, and as she was so richly covered with gold, they gave her a warm welcome. She related to them all that had happened, and when the mother heard how she had come by her great riches, she thought she should like her ugly, lazy daughter to go and try her fortune. So she made the sister go and sit by the well and spin, and the girl pricked her finger and thrust her hand into a thorn-bush, so that she might drop some blood on to the spindle; then she threw it into the well, and jumped in herself. Like her sister she awoke in the beautiful meadow, and walked over it till she came to the oven. 'Take us out, take us out, or alas! we shall be burnt to a cinder; we were baked through long ago,' cried the loaves as before. But the lazy girl answered, 'Do you think I am going to dirty my hands for you?' and walked on. Presently she came to the apple-tree. 'Shake me, shake me, I pray; my apples, one and all, are ripe,' it cried. But she only answered, 'A nice thing to ask me to do, one of the apples might fall on my head,' and passed on. At last she came to Mother Holle's house, and as she had heard all about the large teeth from her sister, she was not afraid of them, and engaged herself without delay to the old woman. The first day she was very obedient and industrious, and exerted herself to please Mother Holle, for she thought of the gold she should get in return. The next day, however, she began to dawdle over her work, and the third day she was more idle still; then she began to lie in bed in the mornings and refused to get up. Worse still, she neglected to make the old woman's bed properly, and forgot to shake it so that the feathers might fly about. So Mother Holle very soon got tired of her, and told her she might go. The lazy girl was delighted at this, and thought to herself, 'The gold will soon be mine.' Mother Holle led her, as she had led her sister, to the broad gateway; but as she was passing through, instead of the shower of gold, a great bucketful of pitch came pouring over her. 'That is in return for your services,' said the old woman, and she shut the gate. So the lazy girl had to go home covered with pitch, and the cock on the well called out as she saw her: 'Cock-a-doodle-doo! Your dirty daughter's come back to you.' But, try what she would, she could not get the pitch off and it stuck to her as long as she lived. Once upon a time there was a dear little girl who was loved by everyone who looked at her, but most of all by her grandmother, and there was nothing that she would not have given to the child. Once she gave her a little cap of red velvet, which suited her so well that she would never wear anything else; so she was always called 'Little Red-Cap.' One day her mother said to her: 'Come, Little Red-Cap, here is a piece of cake and a bottle of wine; take them to your grandmother, she is ill and weak, and they will do her good. Set out before it gets hot, and when you are going, walk nicely and quietly and do not run off the path, or you may fall and break the bottle, and then your grandmother will get nothing; and when you go into her room, don't forget to say, "Good morning", and don't peep into every corner before you do it.' 'I will take great care,' said Little Red-Cap to her mother, and gave her hand on it. The grandmother lived out in the wood, half a league from the village, and just as Little Red-Cap entered the wood, a wolf met her. Red-Cap did not know what a wicked creature he was, and was not at all afraid of him. 'Good day, Little Red-Cap,' said he. 'Thank you kindly, wolf.' 'Whither away so early, Little Red-Cap?' 'To my grandmother's.' 'What have you got in your apron?' 'Cake and wine; yesterday was baking-day, so poor sick grandmother is to have something good, to make her stronger.' 'Where does your grandmother live, Little Red-Cap?' 'A good quarter of a league farther on in the wood; her house stands under the three large oak-trees, the nut-trees are just below; you surely must know it,' replied Little Red-Cap. The wolf thought to himself: 'What a tender young creature! what a nice plump mouthful--she will be better to eat than the old woman. I must act craftily, so as to catch both.' So he walked for a short time by the side of Little Red-Cap, and then he said: 'See, Little Red-Cap, how pretty the flowers are about here--why do you not look round? I believe, too, that you do not hear how sweetly the little birds are singing; you walk gravely along as if you were going to school, while everything else out here in the wood is merry.' Little Red-Cap raised her eyes, and when she saw the sunbeams dancing here and there through the trees, and pretty flowers growing everywhere, she thought: 'Suppose I take grandmother a fresh nosegay; that would please her too.""",
	)

	function _check_normalise_rows(normalise_rows)
		r = tryrun(() -> normalise_rows([2.0 2.0; 1.0 3.0]))
		ok = r !== missing && size(r) == (2, 2) && isapprox(r, [0.5 0.5; 0.25 0.75]) && let
			C = rand(MersenneTwister(4), 0:5, 6, 6) .+ 1.0; Q = normalise_rows(C)
			all(isapprox.(sum(Q, dims = 2), 1)) && isapprox(Q[2, 3], C[2, 3] / sum(C[2, :]))
		end
		check(r, ok, md"Every row now sums to 1: each row is the distribution of the next state given the current one.",
			md"For the counts `[2 2; 1 3]` the result should be `[0.5 0.5; 0.25 0.75]`: each row divided by its own total.";
			hint = md"`sum(C, dims = 2)` is a column of row totals. Dividing with a dot, `C ./ …`, divides every row by its own total.")
	end

	function figure_5_0()
		cases = [
			("absorbing state", [0.5 0.5 0; 0.3 0.4 0.3; 0 0 1.0], "state 3 has P₃₃ = 1: once there, never leaves", "everything ends in state 3"),
			("two islands", [0.5 0.5 0 0; 0.4 0.6 0 0; 0 0 0.5 0.5; 0 0 0.7 0.3], "no arrows between {1, 2} and {3, 4}: reducible", "the long run depends on where you start"),
			("periodic", [0 1.0 0; 0 0 1.0; 1.0 0 0], "no self-loops; returns only every 3 steps", ""),
			("ergodic", [0.3 0.7 0; 0 0 1.0; 1.0 0 0], "irreducible, and one self-loop makes it aperiodic", "a unique limit, from every start")]
		panels = map(cases) do (title, P, cond, run)
			g = chain_graph(P; names = string.(1:size(P, 1)), w = 330, h = 230, rnode = 18)
			@htl("""<div style="width:330px">$(vstack(@htl("<div class='qa-head'>$(title)</div>"), g, Markdown.parse("**Condition:** $(cond).  \n**Long run:** $(run).")))</div>""")
		end
		figure(panels...; caption = md"**Figure 6.1.** Four small chains. Only the last one is ergodic.")
	end

	function parse_jumps(str)
		d = Dict{Int,Int}()
		for m in eachmatch(r"(\d+)\s*(?:-|→|->|>)\s*(\d+)", str)
			a, b = parse(Int, m.captures[1]), parse(Int, m.captures[2])
			(1 <= a < 30 && 0 <= b <= 30 && a != b) && (d[a] = b)
		end
		d
	end

	function ladder_P(jumps; n = 30)
		P = zeros(n + 1, n + 1)
		for sq in 0:n-1, roll in 1:6
			t = min(sq + roll, n); t = get(jumps, t, t)
			P[sq+1, t+1] += 1 / 6
		end
		P[n+1, n+1] = 1
		P
	end

	expected_turns(P; n = 30) = ((I - P[1:n, 1:n]) \ ones(n))[1]

	function figure_8_1(s8_jumps)
		jumps = parse_jumps(s8_jumps); n = 30
		P = ladder_P(jumps); E = expected_turns(P)
		pos(s) = begin r = (s - 1) ÷ 6; c = (s - 1) % 6; ((iseven(r) ? c : 5 - c) + 0.5, r + 0.5) end
		B = fig(w = 330, h = 290, xr = (0, 6), yr = (0, 5), axes = false, l = 4, r = 4, t = 4, b = 4)
		for s in 1:n
			x, y = pos(s)
			rect!(B, x - 0.5, y - 0.5, x + 0.5, y + 0.5; fill = isodd(s) ? "#f6f2e8" : "#fffdf7", op = 1, stroke = "#ccc")
			text!(B, x - 0.42, y + 0.32, string(s); size = 10, c = "#666", bg = true)
		end
		for (a, b) in jumps
			b == 0 && continue
			(x0, y0), (x1, y1) = pos(a), pos(b)
			arrow!(B, x0, y0, x1, y1; c = b > a ? COL.green : COL.red, w = 3, head = 10, op = 0.85, over = false)
		end
		ks = 0:60
		done = [(P^k)[1, end] for k in ks]
		G = fig(w = 330, h = 290, xr = (0, 60), yr = (0, 0.12), xlabel = "number of turns", ylabel = "probability the game ends then", title = "length of the game")
		bars!(G, collect(ks[2:end]), diff(done); width = 0.8, c = COL.blue, op = 0.7)
		vline!(G, E; c = COL.red, w = 2, dash = "5 3")
		text!(G, E > 35 ? E - 1 : E + 1, 0.11, "mean $(fmt(E; d = 3)) turns"; size = 11.5, c = COL.red, bg = true, anchor = E > 35 ? "end" : "start")
		rows = []
		for (a, b) in sort(collect(jumps))
			j2 = copy(jumps); delete!(j2, a)
			push!(rows, ["$(a) → $(b)", b > a ? "ladder" : "snake", fmt(expected_turns(ladder_P(j2)) - E; d = 1)])
		end
		t = isempty(rows) ? md"*No snakes or ladders.*" : tbl(["jump", "kind", "removing it changes the average by (turns)"], rows)
		figure(B, G, t; caption = md"**Figure 9.1.** A board with 30 squares, ladders (green) and snakes (red).")
	end
end;

# ╔═╡ 516cc0d6-be63-cffa-0d43-ceda160e9016
figure_2_0(s2_sel, s2_mode)

# ╔═╡ de5242bc-b988-b284-8239-c6b1c383a1e1
figure_5_0()

# ╔═╡ 9fefaa5e-33e4-73b8-f541-95534bfe68b9
figure_8_1(s8_jumps)

# ╔═╡ 35e19898-7503-9a6d-e737-e96bc2f79d5f
begin
	const ALPH = [' '; collect('a':'z')]

	chlabel(c) = c == ' ' ? "␣" : string(c)

	function clean_text(s3_corpus, s3_my)
		t = s3_corpus == "my own text" ? lowercase(s3_my) : CORPORA[s3_corpus]
		t = replace(lowercase(t), r"[^a-z]+" => " ")
		strip(t)
	end

	function letter_counts(t)
		C = zeros(27, 27); idx = Dict(c => i for (i, c) in enumerate(ALPH))
		ch = collect(t)
		for i in 1:length(ch)-1
			C[idx[ch[i]], idx[ch[i+1]]] += 1
		end
		C
	end

	function figure_3_0(s3_corpus, s3_my, s3_ch)
		t = clean_text(s3_corpus, s3_my)
		length(t) < 200 && return md"*Enter at least 200 characters of text.*"
		C = letter_counts(t); rs = sum(C, dims = 2); P = C ./ max.(rs, 1)
		sel = s3_ch == "space" ? 1 : findfirst(==(s3_ch[1]), ALPH)
		f = fig(w = 400, h = 410, xr = (0.5, 27.5), yr = (27.5, 0.5), xlabel = "next character j", ylabel = "current character i",
			xticks = 1:27, yticks = 1:27, xticklabels = chlabel.(ALPH), yticklabels = chlabel.(ALPH), l = 46, b = 40, t = 8, r = 8)
		for i in 1:27, j in 1:27
			P[i, j] > 0 && rect!(f, j - 0.5, i - 0.5, j + 0.5, i + 0.5; fill = COL.purple, op = 0.12 + 0.88sqrt(P[i, j]))
		end
		rect!(f, 0.5, sel - 0.5, 27.5, sel + 0.5; fill = "none", op = 1, stroke = COL.red, sw = 2, over = true)
		row = P[sel, :]; ord = sortperm(row; rev = true)[1:min(10, count(>(0), row))]
		g = fig(w = 270, h = 410, xr = (0, 1.15maximum(row; init = 0.1)), yr = (length(ord) + 0.6, 0.4), xlabel = "probability", yticks = collect(1:length(ord)), yticklabels = ["→ " * chlabel(ALPH[k]) for k in ord], l = 46, title = "row of '$(chlabel(ALPH[sel]))'")
		for (r, k) in enumerate(ord)
			rect!(g, 0, r - 0.35, row[k], r + 0.35; fill = COL.purple, op = 0.75)
			text!(g, row[k], r, fmt(row[k]; d = 2); dx = 4, size = 10.5, c = "#333")
		end
		nz = count(>(0), C)
		txt = Markdown.parse("Source: $(length(t)) characters. The character '$(chlabel(ALPH[sel]))' occurs $(Int(rs[sel])) times as a current state; its row has $(count(>(0), row)) non-zero entries. Of the 729 possible transitions, $(nz) occur in the source.")
		figure(f, g, txt; caption = md"**Figure 3.1.** Left: the estimated transition matrix of the letter chain; row ``i`` (current character) and column ``j`` (next character), with darker cells for larger ``\hat P_{ij}``; ␣ denotes the space. Right: the ten most probable continuations of the selected character.")
	end

	function figure_3_0g(s3_corpus, s3_my, sd03_g)
		t = clean_text(s3_corpus, s3_my)
		length(t) < 200 && return md"*Enter at least 200 characters of text.*"
		C = letter_counts(t); P = C ./ max.(sum(C, dims = 2), 1)
		rng = MersenneTwister(33 + 1009sd03_g); T = 60
		cur = findfirst(==(t[rand(rng, 1:length(t))]), ALPH); out = [cur]; us = Float64[]
		for _ in 1:T
			row = P[cur, :]; sum(row) == 0 && (row = fill(1 / 27, 27))
			u = rand(rng); push!(us, u)
			cur = something(findfirst(>(u), cumsum(row)), 27); push!(out, cur)
		end
		g = fig(w = 680, h = 230, xr = (0.4, 8.6), yr = (0, 1.08), ylabel = "probability", xticks = Float64[], b = 30, title = "the row of the current character: probabilities of the next character")
		for s in 1:T
			row = P[out[s], :]; ord = sortperm(row; rev = true)[1:8]
			for (r, k) in enumerate(ord)
				row[k] > 0 || continue
				if s == 1
					# frame 0: the row of the initial character, before anything is drawn
					rect!(g, r - 0.36, 0, r + 0.36, row[k]; fill = COL.purple, op = 0.55, o = 0)
					text!(g, r, -0.05, "→ " * chlabel(ALPH[k]); anchor = "middle", size = 12, c = "#333", o = 0)
					text!(g, r, row[k], fmt(row[k]; d = 2); anchor = "middle", dy = -8, size = 10.5, c = "#333", o = 0)
				end
				hit = k == out[s+1]
				rect!(g, r - 0.36, 0, r + 0.36, row[k]; fill = hit ? COL.red : COL.purple, op = hit ? 0.9 : 0.55, o = s)
				text!(g, r, -0.05, "→ " * chlabel(ALPH[k]); anchor = "middle", size = 12, bold = hit, c = hit ? COL.red : "#333", o = s, px = false)
				text!(g, r, row[k], fmt(row[k]; d = 2); anchor = "middle", dy = -8, size = 10.5, c = "#333", o = s)
			end
			hitrank = findfirst(==(out[s+1]), ord)
			hitrank === nothing && text!(g, 8.5, 0.95, "drawn: '$(chlabel(ALPH[out[s+1]]))' (outside the 8 most probable)"; anchor = "end", size = 11, c = COL.red, o = s)
		end
		gen(s) = join(chlabel.(ALPH[out[1:s+1]]))
		reads = ["Step <b>$(s)</b>" * (s == 0 ? ": initial character '$(chlabel(ALPH[out[1]]))'" : ": current '$(chlabel(ALPH[out[s]]))', u = $(fmt(us[s]; d = 3)), next '<b>$(chlabel(ALPH[out[s+1]]))</b>'") *
			"<br><span style='font-family:Georgia,serif;font-size:1.05rem'>" * replace(gen(s), "␣" => " ") * "</span>" for s in 0:T]
		player(g; n = T, start = 0, fps = 3, reads = reads, label = "step",
			caption = md"**Figure 3.2.** Generating text with the letter chain (memory of one character). In each step, the row of the current character is used, a uniform number ``u`` selects the next character (red).")
	end

	function word_tokens(text)
		[m.match for m in eachmatch(r"[A-Za-z]+(?:'[A-Za-z]+)?|[.,;:!?]", text)]
	end

	function join_tokens(toks)
		io = IOBuffer(); cap = true
		for (i, t) in enumerate(toks)
			ispunct = occursin(r"^[.,;:!?]$", t)
			i > 1 && !ispunct && print(io, " ")
			print(io, cap && !ispunct ? uppercasefirst(t) : t)
			cap = t in (".", "!", "?") ? true : ispunct ? cap : false
		end
		String(take!(io))
	end

	function figure_3_1t(s3_corpus, s3_unit, s3_order, s3_my, sd03_t)
		raw_ = s3_corpus == "my own text" ? s3_my : CORPORA[s3_corpus]
		words = startswith(s3_unit, "words")
		order = words ? min(s3_order, 4) : s3_order
		rng = MersenneTwister(30 + 1009sd03_t)
		if words
			toks = word_tokens(raw_)
			length(toks) < 60 && return md"*Enter at least 60 words of text.*"
			follow = Dict{Vector{String}, Vector{String}}()
			for i in 1:length(toks) - order
				push!(get!(follow, toks[i:i+order-1], String[]), toks[i+order])
			end
			starts = [i + 1 for i in 1:length(toks) - order - 1 if toks[i] == "."]
			s0 = isempty(starts) ? 1 : rand(rng, starts)
			out = toks[s0:s0+order-1]
			for _ in 1:130
				nx = get(follow, out[end-order+1:end], String[])
				isempty(nx) && (s1 = isempty(starts) ? 1 : rand(rng, starts); append!(out, ["."]); append!(out, toks[s1:s1+order-1]); continue)
				push!(out, rand(rng, nx))
			end
			gen = join_tokens(out)
			L = 6; srcs = Set(toks[i:i+L-1] for i in 1:length(toks)-L+1)
			copied = mean(out[i:i+L-1] in srcs for i in 1:length(out)-L+1)
			piece = "6-word sequences found verbatim in the source"
			ntok = "$(length(toks)) words and punctuation marks"
		else
			text = clean_text(s3_corpus, s3_my)
			length(text) < 200 && return md"*Enter at least 200 characters of text.*"
			chars = collect(text)
			follow = Dict{String, Vector{Char}}()
			for i in 1:length(chars) - order
				push!(get!(follow, String(chars[i:i+order-1]), Char[]), chars[i+order])
			end
			start = rand(rng, 1:length(chars) - order)
			outc = chars[start:start+order-1]
			for _ in 1:400
				push!(outc, rand(rng, get(follow, String(outc[end-order+1:end]), [' '])))
			end
			gen = String(outc)
			L = 25; subs = [gen[nextind(gen, 0, i):nextind(gen, 0, i + L - 1)] for i in 1:5:length(outc)-L]
			copied = mean(occursin(s_, text) for s_ in subs)
			piece = "25-character pieces found verbatim in the source"
			ntok = "$(length(chars)) characters"
		end
		nst = length(follow); choices = mean(length(unique(v)) for v in values(follow))
		single = count(v -> length(unique(v)) == 1, values(follow)) / nst
		t = tbl(["unit", "memory k", "distinct states observed", "mean number of different continuations per state", "states with a single continuation", piece],
			[[words ? "words" : "characters", string(order), string(nst), fmt(choices; d = 3), pct(single; d = 0), pct(copied; d = 0)]])
		note = words && s3_order > 4 ? md"*For words the memory is limited to 4.*" : md""
		@htl("""<div class="fg"><div style="font-family:Georgia,serif;font-size:1.02rem;line-height:1.55;padding:10px 14px;border-left:4px solid #6a4c93;background:#f8f6fb;white-space:pre-wrap">$(gen)</div>$(t)$(note)<div class="fg-cap">$(Markdown.parse("**Figure 3.3.** Text generated by a Markov chain whose state is the last ``k`` $(words ? "words" : "characters"), with transition probabilities estimated from the source text ($(ntok)). When a state has no observed continuation, the chain restarts at the beginning of a random sentence of the source."))</div></div>""")
	end
end;

# ╔═╡ 3d4e1488-caab-c2ba-255a-b412018c6837
figure_3_0(s3_corpus, s3_my, s3_ch)

# ╔═╡ 323311a2-70e3-005b-678d-50e9b2fe2588
begin
	const BIKE_PI = [68, 52, 45] ./ 165

	function figure_st_1(s7_st, s7_n)
		cols = [COL.blue, COL.ochre, COL.green]; N = 20
		panels = map(1:3) do s
			e = zeros(1, 3); e[s] = 1
			tr = vcat([vec(e * BIKE_P^n)' for n in 0:N]...)
			f = fig(w = 216, h = 230, xr = (0, N), yr = (0, 1.02), xlabel = "day n", ylabel = s == 1 ? "P(Xₙ = station)" : "", title = "start: $(BIKE_NAMES[s])", l = s == 1 ? 52 : 30)
			for j in 1:3
				hline!(f, BIKE_PI[j]; c = cols[j], dash = "4 3", w = 1)
				line!(f, 0:N, tr[:, j]; c = cols[j], w = 2)
			end
			vline!(f, s7_n; c = "#999", dash = "2 3", w = 1)
			f
		end
		rows = [[BIKE_NAMES[s], [fmt((e = zeros(1, 3); e[s] = 1; vec(e * BIKE_P^s7_n))[j]; d = 3) for j in 1:3]...] for s in 1:3]
		push!(rows, ["stationary π", [fmt(BIKE_PI[j]; d = 3) for j in 1:3]...])
		leg = HTML("<p>Colours: " * join(["<span style='color:$(cols[j])'>■</span> $(BIKE_NAMES[j])" for j in 1:3], " · ") * "; dashed lines: the stationary distribution.</p>")
		figure(panels..., leg, tbl(["initial station", "P(Altmarkt) on day $(s7_n)", "P(Campus)", "P(Neustadt)"], rows);
			caption = md"**Figure 7.1.** The distribution ``\pi_n = \pi_0 P^n`` of one bicycle over 20 days, for the three possible initial stations. ")
	end

	function figure_st_2(s7_st, s7_n, s7_j)
		e = zeros(1, 3); e[s7_st] = 1
		πn = vec(e * BIKE_P^s7_n); πn1 = vec(permutedims(πn) * BIKE_P); j = s7_j
		f = fig(w = 680, h = 230, xr = (0, 17), yr = (0, 6), axes = false, l = 4, r = 4, t = 6, b = 6)
		box(x, y, v, hl; w = 1.3) = (rect!(f, x - w / 2, y - 0.45, x + w / 2, y + 0.45; fill = hl ? "#fbe9e4" : "#f6f6f3", op = 1, stroke = hl ? COL.red : "#bbb", sw = hl ? 2 : 1, rx = 3); text!(f, x, y, fmt(v; d = 3); anchor = "middle", size = 12, bold = hl))
		ab = ["A", "C", "N"]
		text!(f, 2.0, 5.3, "πₙ (day $(s7_n))"; anchor = "middle", size = 12.5, bold = true)
		for i in 1:3
			box(0.7 + 1.3(i - 1), 3.0, πn[i], true); text!(f, 0.7 + 1.3(i - 1), 4.0, ab[i]; anchor = "middle", size = 11, c = "#555")
		end
		text!(f, 4.65, 3.0, "×"; anchor = "middle", size = 18)
		text!(f, 7.6, 5.3, "P"; anchor = "middle", size = 12.5, bold = true)
		for r in 1:3
			text!(f, 5.45, 4.0 - (r - 1), ab[r]; anchor = "middle", size = 11, c = "#555")
			for c in 1:3
				box(6.3 + 1.3(c - 1), 4.0 - (r - 1), BIKE_P[r, c], c == j)
			end
		end
		for c in 1:3; text!(f, 6.3 + 1.3(c - 1), 4.85, ab[c]; anchor = "middle", size = 11, c = "#555"); end
		text!(f, 10.7, 3.0, "="; anchor = "middle", size = 18)
		text!(f, 13.4, 5.3, "πₙ₊₁ (day $(s7_n + 1))"; anchor = "middle", size = 12.5, bold = true)
		for i in 1:3
			box(12.1 + 1.3(i - 1), 3.0, πn1[i], i == j); text!(f, 12.1 + 1.3(i - 1), 4.0, ab[i]; anchor = "middle", size = 11, c = "#555")
		end
		terms = join(["$(fmt(πn[i]; d = 3)) · $(fmt(BIKE_P[i, j]; d = 2))" for i in 1:3], " + ")
		text!(f, 8.5, 0.6, "πₙ₊₁($(BIKE_NAMES[j])) = $(terms) = $(fmt(πn1[j]; d = 3))"; anchor = "middle", size = 12.5, c = COL.red)
		inflow = sum(πn[i] * BIKE_P[i, j] for i in 1:3 if i != j); outflow = πn[j] * (1 - BIKE_P[j, j])
		figure(f, txt; caption = md"**Figure 7.2.** One step of the distribution, ``\pi_{n+1} = \pi_n P``. The selected entry of ``\pi_{n+1}`` (red) is the product of the row vector ``\pi_n`` with the selected column of ``P``: the share at each station today times the probability of moving from there to the selected station, summed over all stations. A = Altmarkt, C = Campus, N = Neustadt.")
	end

	check_normalise_rows(f) = guarded(() -> _check_normalise_rows(f))
end;

# ╔═╡ e8596949-de58-f526-2555-cbd445d5a10e
figure_st_1(s7_st, s7_n)

# ╔═╡ 74d5a361-a27b-c26b-9705-71b0aeddd908
figure_st_2(s7_st, s7_n, s7_j)

# ╔═╡ 934be011-f089-ed27-7454-abd58d8f17f0
check_normalise_rows(normalise_rows)

# ╔═╡ 8bd8e61c-1b1a-9121-dce5-b156a7656d21
# Controls
begin
	Seed(default::Integer = 0; label = "New random numbers") = CounterButton(label)
end;

# ╔═╡ 5e424a41-2118-9fbb-52db-0c01848b3df4
md"""
The notebooks on probability and random numbers focussed on single random experiments and sequences of independent random numbers. Most systems of interest, however, **evolve in time**, and their future depends on their present state: the weather tomorrow depends on the weather today, the next letter of a text on the previous letters, the number of bicycles at a station tomorrow on the number today. This notebook introduces the simplest class of models with such dependence, the **Markov chains**. They are used to model queues, populations, texts and molecules, and they are also the basis of the most important sampling algorithms, which we will see in the later notebooks.

!!! hook "Guiding problem"
    A bicycle-sharing scheme in Dresden has three stations: **Altmarkt** (city centre), **Campus** and **Neustadt**, with 100 bicycles each at the start. Every bicycle is ridden at most once per day. Records show that the probability that a bicycle that is at one station in the morning is at another station the next morning as:

    | from \ to | Altmarkt | Campus | Neustadt |
    |:--|:--:|:--:|:--:|
    | Altmarkt | 0.60 | 0.25 | 0.15 |
    | Campus | 0.35 | 0.50 | 0.15 |
    | Neustadt | 0.20 | 0.20 | 0.60 |

    **(a)** Where are the 300 bicycles after one month? **(b)** Does the answer depend on the initial placement? **(c)** How many docks does each station need to ensure that it overflows on fewer than 1% of days?

    Your estimate of the number of bicycles at the Altmarkt after one month: $(@bind guess_c NumberField(0:1:300; default = 0))

## 1. Stochastic processes in discrete time

A **stochastic process in discrete time** is a sequence of random variables ``X_0, X_1, X_2, \dots``. The index ``n`` is the time (days, generations, positions in a text), the value ``x_n`` is the **state** at time ``n``, and the set of all possible states is the **state space**. A sequence of observed values ``x_0, x_1, \dots, x_n`` is a **realisation** or **path** of the process.

The simplest processes consist of **independent and identically distributed** (i.i.d.) random variables, such as repeated coin tosses: each ``X_n`` has the same distribution and is independent of all others. Such a process has no memory. Most real processes have memory. Figure 1.1 compares three processes: i.i.d. coin tosses; a weather model that repeats the weather of the previous day with probability ``r`` and otherwise draws new; and a random walk, which adds ``\pm 1`` to its previous position.

Persistence of the weather ``r`` $(@bind s1_r Slider(0:0.05:0.95; default = 0.75, show_value = true)) ``\qquad`` $(@bind s1_new Seed(0; label = "New realisations"))
"""

# ╔═╡ f0a2f08e-495e-99bf-28be-d6e2b7362516
figure_1_1(s1_new, s1_r)

# ╔═╡ cdcd5046-2b2a-a193-af85-076adfca8b08
figure_7_1(s7_fleet, s7_risk, guess_c)

# ╔═╡ 5c9ea641-ef69-1ced-fc14-34de2ec792c3
md"""

- **Row = from, column = to.** ``P_{ij}`` is the probability of moving from state ``i`` to state ``j`` in one step.
- **Every row sums to 1.** A bicycle that is at ``i`` today must be somewhere tomorrow, so row ``i`` is a probability distribution, the conditional distribution of ``X_{n+1}`` given ``X_n = i``. 
- **Columns need not sum to 1.** Column ``j`` collects probabilities from different starting states; they are not one distribution.

The diagonal entry ``P_{ii}`` is the probability of remaining in state ``i``; in the diagram it is a loop. Figure 2.2 follows a single bicycle.

Start at $(@bind s2_start Select([1 => "Altmarkt", 2 => "Campus", 3 => "Neustadt"]; default = 1)) ``\qquad`` days $(@bind s2_days Select([40, 100, 365]; default = 100)) ``\qquad`` $(@bind s2_newpath Seed(0; label = "New path"))
"""

# ╔═╡ aea6556b-d0e6-5421-53ac-171764803dd7
figure_2_1(s2_newpath, s2_days, s2_start)

# ╔═╡ 878465c5-ac6f-a943-f106-a575acd8924c
md"""
Applied repeatedly, starting from the station selected above:

$(@bind s2_ws Seed(0))
"""

# ╔═╡ 95274cdc-6153-e30c-d2dc-93b6db9181f4
figure_2_3(s2_ws, s2_i)

# ╔═╡ f1837180-af8c-3529-65ff-9d7bc8c12c65
md"""

### 3.2 Generating text

Generation is the simulation of the chain (Section 2.3): starting from a character, the next character is drawn from the row of the current character, and so on. Figure 3.2 shows the first 60 steps.

$(@bind sd03_g Seed(0; label = "Generate again"))
"""

# ╔═╡ bed2b08b-18d3-c4ba-8189-7bc12087077c
figure_3_0g(s3_corpus, s3_my, sd03_g)

# ╔═╡ 651a86f8-f8a8-372a-8545-8a7a37aefb78
md"""
### 3.3 Longer memory, and words instead of letters

With one character of memory, the generated text has the correct letter frequencies and plausible letter pairs, but it contains few real words. Two changes can be introduced to improve it. The memory can be increased by defining the state as the last ``k`` characters; the process is then still a Markov chain, on a much larger state space (up to ``27^k`` states). Alternatively, the unit can be changed from characters to **words**: the state is the last ``k`` words (punctuation marks count as words), and the next word is drawn from the words that followed this context in the source. A word chain produces only real words by construction, and with ``k = 2`` it produces mostly grammatical fragments of sentences.

!!! note "Longer memory is still a Markov chain"
    The Markov property requires that the future depend only on the **current state**. It does not prescribe what the state is. If the next character depends on the last two characters ``c_{n-1}, c_n``, the sequence of single characters is *not* a Markov chain. But define the state as the pair ``X_n = (c_{n-1}, c_n)``. The next state is ``X_{n+1} = (c_n, c_{n+1})``; it shares ``c_n`` with ``X_n``, and ``c_{n+1}`` is drawn from a distribution that depends only on ``X_n``. Hence ``X_0, X_1, \dots`` is an ordinary Markov chain on the set of pairs. For example, with the state ``(t, h)``, the next state is ``(h, e)``, ``(h, a)``, ``(h, i)``, … with the frequencies observed after "th". In the same way, a memory of ``k`` characters or words is a Markov chain whose states are blocks of ``k`` symbols. The downside is the size of the state space: ``27^k`` blocks of characters, and the transition matrix must be estimated for each of them.

Unit $(@bind s3_unit Select(["words", "characters"]; default = "words")) ``\qquad`` memory ``k`` $(@bind s3_order Slider(1:8; default = 2, show_value = true)) ``\qquad`` $(@bind sd03_t Seed(0; label = "Generate again"))
"""

# ╔═╡ 7018545f-11e7-03e4-b776-e000ae469f81
figure_3_1t(s3_corpus, s3_unit, s3_order, s3_my, sd03_t)

# ╔═╡ aec16143-ace7-4a7e-a642-fb69348cd66e
md"""
## 4. Example: a Markov chain that composes a melody

The same construction applies to music. A melody is a sequence of notes, each with a **pitch** and a **duration**, and the state of the chain is the pair (pitch, duration) of the last one or two notes. The transition probabilities are estimated from training melodies by counting, exactly as for text. Five melodies in C major are available: the main theme of the last movement of Beethoven's Ninth Symphony (*Ode to Joy*, 1824) and four folk songs (*Alle meine Entchen*, *Hänschen klein*, *Frère Jacques*, and *Ah vous dirai-je, maman*, the tune Mozart varied in K. 265 and known in English as *Twinkle, Twinkle, Little Star*). They can be used individually or pooled.

Two rules are added that a pure Markov chain does not have. First, music is organised in **bars** (here of four beats): a note is drawn only among the observed continuations that fit into the remainder of the current bar, which is a conditional distribution (see the notebook on probability). Second, the melody is 8 bars long and its last bar is a **cadence** that ends on the tonic C. An optional accompaniment chooses one chord per bar (C, F, G, A minor or D minor) by a deterministic rule: the chord whose notes cover most of the melody in that bar.

Training melodies $(@bind s4_mel Select(["all five melodies", "Ode to Joy (Beethoven, 1824)", "Alle meine Entchen (German folk song)", "Hänschen klein (German folk song)", "Frère Jacques (French folk song)", "Ah vous dirai-je, maman (French folk song)"]; default = "all five melodies")) ``\qquad`` memory $(@bind s4_ord Select(["1 note", "2 notes"]; default = "2 notes")) ``\qquad`` $(@bind s4_acc Select(["with accompaniment", "melody only"]; default = "with accompaniment")) ``\qquad`` $(@bind sd03_m Seed(0; label = "Compose again"))
"""

# ╔═╡ 64e1e135-2408-fe80-7742-1c41cb5db27f
figure_3_2m(s4_mel, s4_ord, s4_acc, sd03_m)

# ╔═╡ 66a26e2d-fe7e-eede-1f32-e278cd355994
md"""
## 5. The distribution after ``n`` steps

A single chain is random. A large number of independent bicycles, however, behaves almost deterministically, because their individual fluctuations average out. Figure 5.1 moves all bicycles for up to 60 days.

Bicycles on day 0: Altmarkt $(@bind s3_c0 NumberField(0:300; default = 0)) ``\quad`` Campus $(@bind s3_u0 NumberField(0:300; default = 0)) ``\quad`` Neustadt $(@bind s3_p0 NumberField(0:300; default = 300)) ``\qquad`` days $(@bind s3_D Select([10, 30, 60]; default = 30)) ``\qquad`` $(@bind s3_newbikes Seed(0; label = "New random paths"))
"""

# ╔═╡ bdb2f209-e5ec-0447-50a7-bbe4ea8bf6ae
s3_x0 = compute_s3_x0(s3_c0, s3_u0, s3_p0);

# ╔═╡ fe5193e0-693b-b871-9c1a-dac6953a0cff
figure_3_1(s3_newbikes, s3_D, s3_x0)

# ╔═╡ e9c888bc-91b4-b3eb-66c2-8bcfc8a75975
md"""
These properties are defined as follows.

!!! definition "Definitions"
    - A set ``C`` of states is **closed** if no state outside ``C`` can be reached from a state in ``C``. A closed set consisting of a single state is an **absorbing** state.
    - A chain is **irreducible** if every state can be reached from every other state; equivalently, the only closed set is the whole state space.
    - A state ``i`` is **periodic** with period ``t > 1`` if returns to ``i`` are possible only after multiples of ``t`` steps; otherwise it is **aperiodic**. A self-loop ``P_{ii} > 0`` makes a state aperiodic.
    - A chain that is irreducible and aperiodic is called **ergodic**.

Select one of the cases, or enter a transition matrix, and follow the distribution ``\pi_0 P^n`` from a single initial state.

Chain $(@bind s4_c Select(["ergodic: the bicycle chain", "reducible: two islands", "periodic: a cycle of three", "absorbing: a trap", "periodic with period 2", "my chain (matrix below)"]; default = "periodic: a cycle of three")) ``\qquad`` initial state $(@bind s4_s Select(1:5; default = 1)) ``\qquad`` $(@bind s4_new Seed(0; label = "New walker"))

My chain (rows separated by semicolons, up to 5 states): $(@bind s4_my TextField(60; default = "0.5 0.5 0 0; 0.5 0.5 0 0; 0 0.2 0.3 0.5; 0 0 0.5 0.5"))
"""

# ╔═╡ c6de412e-b6e6-5168-f54d-88bdf68e7d75
s4_P = compute_s4_P(s4_c, s4_my);

# ╔═╡ 93f047b0-1e26-03b5-4ecb-9264e9ffa11e
figure_4_1(s4_new, s4_P, s4_s, s4_c)

# ╔═╡ 1b1f83ae-889d-31d3-7ead-5dbc444ef254
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
.pc{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12.5px;line-height:1.55;background:#fafaf7;border:1px solid #e0e0da;border-radius:4px;padding:6px 0;min-width:300px;max-width:440px;color:#1b1b1b}
.pc-head{font-family:system-ui,sans-serif;font-size:0.72rem;letter-spacing:0.09em;text-transform:uppercase;font-weight:700;color:#3d4a5c;padding:0 10px 4px}
.pc-ln{position:relative;padding:0 10px 0 34px;white-space:pre}
.pc-no{position:absolute;left:8px;color:#a0a09a;z-index:1}
.pc-ln code{position:relative;z-index:1;font:inherit;background:none;padding:0}
.pc-mark{position:absolute;left:0;right:0;top:0;bottom:0;background:#e3ecf8;border-left:3px solid #1f5aa6;z-index:0}
details.hint{margin-top:4px}
details.hint summary{cursor:pointer;font-weight:600}
.wk-state{font-variant-numeric:tabular-nums}
.pn{display:inline-block;margin:4px 0}.pn-b{font-size:0.95rem;padding:3px 12px;cursor:pointer}
.wk-state .markdown,.wk-why .markdown,.wk-fail .markdown{display:inline}
.wk-state p,.wk-why p,.wk-fail p{display:inline;margin:0}
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
# ╟─6b97923e-96a0-5c85-7a8d-c87d675a9cd0
# ╟─8c28c86c-d249-08a7-a333-1937c09a4777
# ╟─5e424a41-2118-9fbb-52db-0c01848b3df4
# ╟─f0a2f08e-495e-99bf-28be-d6e2b7362516
# ╟─2e1a3eaa-123f-de5a-aa60-2579de80a678
# ╟─183d13f9-2344-c1da-6c8a-acd99a55998c
# ╟─4ec681fb-1420-2683-a858-41d8b9436f2f
# ╟─4b25057e-204a-820f-0d5e-fcf6ed8a1089
# ╟─516cc0d6-be63-cffa-0d43-ceda160e9016
# ╟─5c9ea641-ef69-1ced-fc14-34de2ec792c3
# ╟─aea6556b-d0e6-5421-53ac-171764803dd7
# ╟─e2a970c2-42da-d032-1a40-445009c8f4c6
# ╟─4a9b637b-31f2-6055-e33e-c1a2c7ff0232
# ╟─3837421e-3e08-bb9c-2319-1c76aff449bf
# ╟─c31c2f38-403b-51a7-0327-204572af510a
# ╟─ba5d51f6-8c2a-09db-42c3-c615de9cc41d
# ╟─878465c5-ac6f-a943-f106-a575acd8924c
# ╟─95274cdc-6153-e30c-d2dc-93b6db9181f4
# ╟─97d49dc2-4ee3-9c0c-b272-64d905ad243e
# ╟─f69b1902-ef36-8034-3fad-5a385c884dd6
# ╟─3d4e1488-caab-c2ba-255a-b412018c6837
# ╟─b52f8abb-db65-40c2-b6b7-08b8c56e9d14
# ╟─5ce0c0fb-aa20-96c9-9236-8edf4dcdfb26
# ╟─83c43687-d0d4-2021-055f-196fe8174a8e
# ╟─f1837180-af8c-3529-65ff-9d7bc8c12c65
# ╟─bed2b08b-18d3-c4ba-8189-7bc12087077c
# ╟─651a86f8-f8a8-372a-8545-8a7a37aefb78
# ╟─7018545f-11e7-03e4-b776-e000ae469f81
# ╟─0996dcb8-2191-b9fd-aeb2-0744ebe3be4e
# ╟─7ee1306d-a6da-86b8-f840-640051025bdb
# ╟─9d8f4a66-5575-5d76-3db0-625edb9894b8
# ╟─6adef4cb-318b-c06b-e388-e8bb87e2eeae
# ╟─d8ac51ee-c6d4-2d1c-3442-def3ae629472
# ╟─aec16143-ace7-4a7e-a642-fb69348cd66e
# ╟─64e1e135-2408-fe80-7742-1c41cb5db27f
# ╟─1f5e7ae5-68bf-a819-71da-5a1c5f271027
# ╟─14969370-baf6-55f0-bc57-e14baf92ec69
# ╟─66a26e2d-fe7e-eede-1f32-e278cd355994
# ╟─bdb2f209-e5ec-0447-50a7-bbe4ea8bf6ae
# ╟─fe5193e0-693b-b871-9c1a-dac6953a0cff
# ╟─eca6d5d4-2592-10f3-1232-d9f09b3f25d2
# ╟─12d3b05f-6427-4f74-9efa-c148ad3650f3
# ╟─40e04f0e-1ecc-da30-1ca0-5c051b4f855a
# ╟─206fefd1-c733-b9b6-7785-5e3f29cdf0c2
# ╟─d2337916-8ae9-6b2f-929e-076c5cbf0156
# ╟─de5242bc-b988-b284-8239-c6b1c383a1e1
# ╟─e9c888bc-91b4-b3eb-66c2-8bcfc8a75975
# ╟─c6de412e-b6e6-5168-f54d-88bdf68e7d75
# ╟─93f047b0-1e26-03b5-4ecb-9264e9ffa11e
# ╟─83f2505d-c074-ddbc-9a65-55508c05d710
# ╟─71881e83-5f57-a444-d69a-6765a7423d5c
# ╟─35599b1e-d402-939f-07d9-e28e1868dc1d
# ╟─e8596949-de58-f526-2555-cbd445d5a10e
# ╟─4c82addf-9498-4582-7fd4-d0b0ef59a34d
# ╟─74d5a361-a27b-c26b-9705-71b0aeddd908
# ╟─183d0998-0476-b6e9-95fd-dc93d7e8813c
# ╟─41dd7fa7-0a22-07e1-d2f8-c9d15f965201
# ╟─193c1a08-df48-f72d-956b-e6f4e587ae02
# ╟─2ba8ad4c-8908-06e9-16c7-f78e6f027bb4
# ╟─e54c6465-068f-8cc4-1aae-3ec637a1677f
# ╟─1d6b3874-3f3b-0ce4-e954-72e305c504c3
# ╟─ebc565f5-3b1e-bbb8-f77d-623b8a50d999
# ╟─cdcd5046-2b2a-a193-af85-076adfca8b08
# ╟─15844db3-797f-f4e5-8bd1-2a9e6351c4d2
# ╟─9fefaa5e-33e4-73b8-f541-95534bfe68b9
# ╟─a7941415-0ec9-1224-1b07-ec4c654ef512
# ╟─b4d99478-1cf5-6e3d-0335-e1d0b713ee96
# ╟─ef43fac3-bee5-bd5a-4a36-22703979a08f
# ╟─9afc3fbd-c863-ad7a-0201-731163f20e9f
# ╟─8aea2228-5660-2a9f-f19b-be7345292b8e
# ╟─1a7c250b-d6bf-d28a-1fa3-304edfe2c38a
# ╟─bebfecf6-fd82-e30c-2633-62d0e4c6f4c4
# ╟─ebc7f26e-1018-9f03-13c9-10785abf9ea4
# ╟─4efc2fc9-f75b-9367-1af1-d7cffbddfb18
# ╠═a1500e90-3e0b-b0d9-149b-bb2f2dcd4273
# ╟─934be011-f089-ed27-7454-abd58d8f17f0
# ╟─a3bfd27c-6ee0-11c0-d28b-b7b3ef1f3e91
# ╟─0e09964e-b4fc-5277-8da0-8d9713a2ab21
# ╠═22cbdf8e-ad9e-c486-a912-e426e469a245
# ╟─7bf55512-9b57-69c8-db32-f85b8a084fda
# ╟─69ff4c82-7cbe-af9c-8a8a-c73bb2d20eb9
# ╟─ea8f76f1-0ce7-8f6a-60e0-f2d325c02c98
# ╠═e7c22bb8-04fd-8b97-e48b-f71e2a480859
# ╟─ac7dca37-1584-3edd-af3e-c85fc55da5dd
# ╟─3c9dc54c-f230-b1b2-0b01-0df6751982b4
# ╟─5ae1ef4a-5819-edc1-bb9b-93de0f4649d8
# ╟─40d2386f-b6ef-0f47-64c7-877e08ddc9e8
# ╟─20fc9273-54c2-6f44-fb9f-7fd89862872d
# ╟─3fa32784-82dd-fdf2-30a2-5d71a588950e
# ╟─dc02448f-28ea-527e-d834-d383c6c8b5bb
# ╟─5c8d6357-9a80-a750-aa96-a9d30aba4b1a
# ╟─35e19898-7503-9a6d-e737-e96bc2f79d5f
# ╟─a2e99ae6-04b2-5c25-b415-68ca954ed85a
# ╟─323311a2-70e3-005b-678d-50e9b2fe2588
# ╟─994cbe9a-9b5c-5b9c-8847-970c28ba92a5
# ╟─33e5ba05-da49-4cfd-17c1-9a8dc5c7a3ee
# ╟─06424fb2-40ef-acc3-a0b8-dedf4107751e
# ╟─9e20789c-f283-9cb2-9715-5b3edbf7232e
# ╟─8bd8e61c-1b1a-9121-dce5-b156a7656d21
# ╟─1b1f83ae-889d-31d3-7ead-5dbc444ef254
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
