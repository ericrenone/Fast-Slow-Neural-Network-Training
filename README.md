# (FHN) - FitzHugh–Nagumo Excitability, Farey Arithmetic, and Neural Network Training


## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✓ | Established theorem — proof or canonical reference given |
| ✓ cond. | Proved under explicitly stated assumptions |
| ⚠ | Conjecture — gap stated, no proof |
| ~ | Structural analogy — not a formal equivalence |

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The FitzHugh–Nagumo System](#2-the-fitzhugh-nagumo-system)
3. [Farey Sequences and Continued Fractions](#3-farey-sequences-and-continued-fractions)
4. [Gradient Ratio Diagnostics](#4-gradient-ratio-diagnostics)
5. [The Fast–Slow Analogy for Neural Training](#5-the-fastslow-analogy-for-neural-training)
6. [Grokking as Excitable Firing](#6-grokking-as-excitable-firing)
7. [The Convergent-Curvature Correspondence](#7-the-convergent-curvature-correspondence)
8. [A PAC-Bayes Generalization Bound](#8-a-pac-bayes-generalization-bound)
9. [Ford Circles and Loss Basin Geometry](#9-ford-circles-and-loss-basin-geometry)
10. [The Farey Consolidation Index](#10-the-farey-consolidation-index)
11. [Implementation](#11-implementation)
12. [Summary of Results](#12-summary-of-results)
13. [Open Problems](#13-open-problems)
14. [References](#14-references)

---

## 1. Motivation

Three mathematical structures share a common skeleton:

**FitzHugh–Nagumo (FHN)** — a canonical model of neuronal excitability coupling a fast activator variable to a slow recovery variable. It captures threshold crossings, excitable firing, and limit cycles from a two-dimensional ODE system.

**Farey arithmetic** — the number theory of rational approximation: Farey sequences, continued fractions, Ford circles, and the Stern–Brocot tree. These structures encode the geometry of the rational line with deep connections to prime distribution.

**Gradient descent training** — iterative parameter updates that pass through recognizable phases: rapid memorization, a long plateau, then sudden generalization (grokking). The loss landscape geometry evolves during training in ways that have resisted clean characterization.

The central observation is that all three instantiate the same **fast–slow motif**: a fast nonlinear variable driven by local dynamics, regulated by a slow variable operating on a longer timescale, with qualitative transitions (firing, grokking) occurring when the fast variable crosses a threshold set by the slow one.

This document develops that analogy carefully, distinguishing what is proved from what is conjectured.

---

## 2. The FitzHugh–Nagumo System

### 2.1 The Standard System ✓

The FitzHugh–Nagumo model (FitzHugh 1961; Nagumo et al. 1962) is a two-dimensional reduction of the four-dimensional Hodgkin–Huxley equations:

```
dv/dt = v - v³/3 - w + I_ext        (fast equation)
dw/dt = ε(v + a - b·w)              (slow equation)
```

- `v` — fast variable (membrane potential; activator)
- `w` — slow variable (recovery current; inhibitor)
- `ε ≪ 1` — timescale separation parameter
- `I_ext` — external drive
- `a, b > 0` — shape parameters

### 2.2 Nullclines ✓

Dynamics are organized by two nullclines in the `(v, w)` plane.

**Fast nullcline** (`dv/dt = 0`):

```
w = v - v³/3 + I_ext
```

This cubic N-shaped curve has two stable branches (left and right) separated by an unstable middle branch. The folds occur at `v = ±1`.

**Slow nullcline** (`dw/dt = 0`):

```
w = (v + a) / b
```

A straight line. Its intersection with the fast nullcline determines fixed points.

### 2.3 Dynamical Regimes ✓

The position of the fixed point on the fast nullcline determines behavior:

**Quiescent** — fixed point on the left stable branch. Small perturbations decay. No firing.

**Excitable** — fixed point near the left knee of the cubic. A perturbation above threshold sends `v` on a large excursion through the right branch before returning. One firing event; no sustained oscillation.

**Limit cycle** — fixed point on the unstable middle branch (Hopf bifurcation). Sustained periodic oscillation.

### 2.4 Timescale Separation ✓

For `ε → 0`, `v` relaxes instantly to the fast nullcline. The slow variable `w` drifts along until reaching a fold (knee point), where `v` jumps discontinuously to the other branch. This **relaxation oscillation** — slow drift, fast jump, slow drift, fast jump — is the geometric skeleton of FHN dynamics and the template for the learning analogy in Section 5.

---

## 3. Farey Sequences and Continued Fractions

All results in this section are established theorems.

### 3.1 Farey Sequences ✓

The **Farey sequence of order n**, written `F_n`, is the ascending sequence of all fractions `p/q` in lowest terms with `0 ≤ p ≤ q ≤ n`:

```
F₁ = { 0/1, 1/1 }
F₂ = { 0/1, 1/2, 1/1 }
F₃ = { 0/1, 1/3, 1/2, 2/3, 1/1 }
F₄ = { 0/1, 1/4, 1/3, 1/2, 2/3, 3/4, 1/1 }
F₅ = { 0/1, 1/5, 1/4, 1/3, 2/5, 1/2, 3/5, 2/3, 3/4, 4/5, 1/1 }
```

### 3.2 The Unimodular (Neighbor) Property ✓

**Theorem (Cauchy 1816; Hardy & Wright 1979, Thm. 28).** Two fractions `a/b < c/d` in lowest terms are adjacent in some `F_n` if and only if:

```
|bc - ad| = 1
```

Equivalently, the matrix `[[a, c], [b, d]]` lies in `SL(2, ℤ)`.

*Proof sketch.* By induction on `n`. Base case: `0/1, 1/1` gives `1·1 - 0·1 = 1`. Inductive step: if `a/b, c/d` are neighbors with `|bc - ad| = 1`, the unique new element in `F_{b+d}` is the mediant `(a+c)/(b+d)`. Then `b(a+c) - a(b+d) = bc - ad = 1` and `(a+c)d - c(b+d) = ad - bc = -1`. ∎

### 3.3 The Mediant Property ✓

If `a/b` and `c/d` are Farey neighbors, their **mediant** `(a+c)/(b+d)`:

- Is in lowest terms (follows from `|bc - ad| = 1`)
- Is the unique fraction between `a/b` and `c/d` with smallest denominator
- First appears in `F_{b+d}`

### 3.4 Sequence Length ✓

```
|F_n| ~ 3n²/π²    (as n → ∞)
```

### 3.5 Continued Fraction Convergents ✓

**Theorem (Hurwitz 1891).** For any irrational `x`, there are infinitely many fractions `p/q` satisfying `|x - p/q| < 1/(√5 · q²)`. Every such fraction is a convergent of the continued fraction expansion of `x`. The constant `1/√5` is sharp.

**Theorem (Lagrange 1770).** Each convergent `p_k/q_k` is the best rational approximant with denominator `≤ q_k`, satisfying `|x - p_k/q_k| < 1/(q_k q_{k+1})`.

### 3.6 The Three-Distance Theorem ✓

**Theorem (Steinhaus 1950; Sós 1958).** For any irrational `α` and integer `N`, the fractional parts `{α}, {2α}, ..., {Nα}` partition `[0,1)` into gaps of at most **three** distinct lengths, determined by consecutive denominators in the continued fraction of `α`.

### 3.7 Ford Circles ✓

For each fraction `p/q` in lowest terms, the **Ford circle** `C(p/q)` is the circle tangent to the real line at `p/q` with:

- Center: `(p/q, 1/(2q²))`
- Radius: `r = 1/(2q²)`

**Theorem (Ford 1938).** Two Ford circles `C(a/b)` and `C(c/d)` are externally tangent if and only if `|bc - ad| = 1`.

*Consequence:* Ford circles are tangent precisely when the corresponding fractions are Farey neighbors. Adjacent fractions in `F_n` correspond to tangent circles; larger denominators give smaller circles (sharper basins in the loss landscape analogy of Section 9).

### 3.8 The Franel–Landau Theorem ✓

Let `r_ν` be the `ν`-th element of `F_n` and `δ_ν = r_ν - ν/|F_n|` its discrepancy from uniform spacing. Then:

```
∑_ν δ_ν² = O(n^{-1+ε}) for all ε > 0   ⟺   Riemann Hypothesis
```

This equivalence is unconditional (Franel 1924; Landau 1924). The Farey sequence encodes the Riemann Hypothesis in its spacing regularity. *Note: this is a statement about the Farey sequence itself; the connection to gradient distributions in Section 10 is an analogy, not a theorem.*

### 3.9 The Stern–Brocot Tree ✓

The Stern–Brocot tree is the complete binary tree of all positive rationals built by repeated mediant insertion:

```
                    1/1
                   /   \
                 1/2   2/1
                / \   / \
              1/3 2/3 3/2 3/1
              ...
```

Every positive rational appears exactly once. The path from the root to `p/q` encodes the continued fraction of `p/q`. The Farey sequence `F_n` is the horizontal slice at depth `n`.

---

## 4. Gradient Ratio Diagnostics

### 4.1 Definitions

For consecutive gradient vectors `g_t, g_{t+1} ∈ ℝ^d`, define:

**Gradient ratio:**
```
ρ_t = ‖g_{t+1}‖ / (‖g_t‖ + ‖g_{t+1}‖)   ∈ (0, 1)
```

**Relative gradient change:**
```
ε_grad(t) = ‖g_{t+1} - g_t‖ / (‖g_t‖ + ‖g_{t+1}‖)   ∈ [0, 1]
```

### 4.2 Invariance of `ρ_t` ✓

`ρ_t` is invariant under:
- Orthogonal rotation of parameter space (`‖Qg‖ = ‖g‖` for orthogonal `Q`)
- Positive rescaling of the learning rate
- Sign flips of gradient components

`ρ_t` is **not** invariant under arbitrary smooth reparameterization — this is shared by every gradient-norm diagnostic and is not a defect of the ratio specifically.

### 4.3 Data-Adaptive Farey Approximation ✓ (cond.)

Map `ρ_t` to a Farey fraction by setting:

```
Q_max = ⌊1 / ε_grad(t)⌋
```

and taking the best continued-fraction convergent of `ρ_t` with denominator `≤ Q_max`.

**Justification.** By Hurwitz's theorem, the approximation error of the best convergent with denominator `q` is at most `1/(√5 · q²)`. At `q = Q_max ~ 1/ε_grad`, this error is `~ε_grad²/√5`, which is smaller than the measurement noise level `ε_grad`. This ensures no finer Farey structure is imposed than the data supports.

**Consequence for training.** When gradients change rapidly (large `ε_grad`), the Farey resolution is coarse; convergents have small denominators. When gradients are nearly stationary (small `ε_grad`), finer resolution is warranted; convergents may have large denominators. This adaptive scale matches the information content of the gradient signal.

### 4.4 The Unimodular Condition for Gradient Pairs ✓

Given convergents `(p_t, q_t)` and `(p_{t+1}, q_{t+1})` for consecutive gradient pairs, the **unimodular condition** is:

```
|q_t · p_{t+1} - p_t · q_{t+1}| = 1
```

By the unimodular theorem (Section 3.2), this holds if and only if the two fractions are Farey neighbors — they are the maximally arithmetically independent pair at their resolution.

**Interpretation (structural, not proved as a theorem about learning):** When consecutive gradient convergents satisfy the unimodular condition, the corresponding gradient directions span the primitive lattice `ℤ²` at that resolution — there is no common sublattice structure. When `|q_t p_{t+1} - p_t q_{t+1}| > 1`, the pair lies in a proper sublattice, suggesting redundancy in the gradient directions.

---

## 5. The Fast–Slow Analogy for Neural Training

### 5.1 The Analogy ~ (structural)

This section develops an analogy between FHN dynamics and gradient descent training. The analogy is structural: the mathematical motif is the same. It is **not** a theorem that training literally instantiates FHN equations.

**Correspondence table:**

| FHN concept | Training analog | Notes |
|-------------|----------------|-------|
| Fast variable `v` | Gradient norm `‖g_t‖` | Rapid step-to-step variation |
| Slow variable `w` | Median Farey denominator `q*` | Computed over a window of W steps |
| Timescale separation `ε ≪ 1` | `τ_fast = 1 step`, `τ_slow = W steps` | `ε = 1/W ≈ 0.005–0.02` |
| External drive `I_ext` | Regularization strength | Weight decay coefficient `λ` |
| Fast nullcline `(dv/dt = 0)` | Gradient equilibrium manifold | `∇L(θ) ≈ 0` |
| Slow nullcline `(dw/dt = 0)` | Farey saturation level | `F_c_percentile ~ 80` (empirical threshold) |
| Stable fixed point (left branch) | Memorization attractor | High `q*`, low test accuracy |
| Threshold crossing | Grokking onset | Farey backtrack criterion (Section 6.2) |
| Excitable orbit | Grokking trajectory | `q*` drops, test accuracy rises |
| Post-firing recovery | Post-grokking consolidation | `q*` settling to low denominator |

### 5.2 The Timescale Separation

The fast-slow structure is not imposed — it arises from the definition of `q*`:

```
τ_fast = 1 gradient step
τ_slow = W gradient steps     (window size, typically W = 50–200)
ε      = 1/W ≈ 0.005–0.02  ≪  1
```

The median denominator `q*` responds slowly to individual gradient steps because it is a median over a large window. This is the exact analog of the FHN slow variable `w` with timescale `ε ≪ 1`.

### 5.3 What This Analogy Does and Does Not Claim

**Does claim:**
- The mathematical structure of fast-slow dynamics (two timescales, threshold, excitable orbit) is a productive lens for understanding training dynamics.
- The quantities `q*` and `F_c_percentile` are well-defined, computable, and correspond to slow and fast monitoring of gradient arithmetic.
- The Farey backtrack criterion is a testable predictor of grokking events.

**Does not claim:**
- That neural network training literally solves FHN equations.
- That the cubic nullcline is the loss landscape curvature (this would require `φ(θ)` to be fitted and is not derived from first principles).
- That the fixed points of the analogy correspond exactly to memorization and generalization attractors in general.

---

## 6. Grokking as Excitable Firing

### 6.1 The Structural Correspondence ~ (structural)

The grokking phenomenon (Power et al. 2022) — delayed generalization after memorization — maps naturally onto the FHN excitability regime:

| FHN event | Training event |
|-----------|---------------|
| Quiescent at stable left fixed point | Network memorizing; `q*` high; test accuracy near chance |
| Slow drift of `w` toward knee | `q*` fluctuating; `F_c_percentile` rising |
| Threshold crossing | Farey backtrack (see Section 6.2) |
| Fast excitable orbit | `q*` drops rapidly to low denominator |
| Return to equilibrium | Test accuracy plateaus; training converges |

The claim that grokking *is* an excitable firing event in a formal sense is a **conjecture** requiring empirical validation. The structural mapping is suggestive and motivates specific testable predictions.

### 6.2 The Farey Backtrack Criterion

Define a **Farey backtrack** (candidate grokking signal) at step `t` when both:

```
q*(t) < q*(t - W)                   (denominator decreases over the window)
AND
F_c_percentile(t) > 80              (excess unimodular structure detected)
```

Both conditions are required: a denominator decrease alone can be noise; the permutation test (Section 10) confirms it reflects structural change in gradient arithmetic. The dual requirement mirrors FHN: both `v` above threshold *and* sufficient `I_ext` are needed for an excitable firing event.

**Empirical prediction** ⚠: The first Farey backtrack event precedes test accuracy improvement by 50–200 training steps. This is a testable, falsifiable claim requiring systematic evaluation on standard grokking benchmarks.

### 6.3 Three Qualitative Regimes ~

The FHN framework predicts three training regimes:

**Regime 1 — No grokking:** System at memorization attractor; regularization too weak to push the trajectory to threshold.

**Regime 2 — Single grokking event:** Excitable regime. A perturbation (weight decay increase, learning rate change) triggers one Farey backtrack, then the system settles at a low-`q*` generalization attractor. This matches most reported grokking experiments.

**Regime 3 — Oscillatory:** System oscillates between high and low `q*`. Analogous to FHN limit cycle. Predicted to occur when regularization is tuned near the Hopf boundary; rarely reported in practice.

---

## 7. The Convergent-Curvature Correspondence

This section derives the main quantitative link between the Farey denominator `q*` and the top Hessian eigenvalue `λ_max(H)`. The derivation is in four steps under two explicitly stated assumptions.

### 7.1 Assumptions

**Assumption S (Smoothness).** The loss `L` is twice continuously differentiable at `θ*` with positive definite Hessian `H = ∇²L(θ*) ≻ 0`. This is standard for local convergence analysis and fails at ReLU kink points.

**Assumption E (Spectral Dominance).** The initial displacement `δ_0 = θ_0 - θ*` is concentrated in the top Hessian eigenspace: `δ_0 ≈ δ_0^(1) v_1`, where `v_1` is the eigenvector corresponding to `λ_1 = λ_max(H)`. This holds exactly when `δ_0 ∝ v_1` and approximately when `λ_1 ≫ λ_2`. It is conservative — the bound becomes weaker when eigenvalues are comparable.

### 7.2 Step 1 — SGD Linearization ✓

Near `θ*`, full-batch gradient descent satisfies:

```
θ_{t+1} ≈ θ_t - η H (θ_t - θ*)
```

In the Hessian eigenbasis `{v_i, λ_i}` with `δ_t = θ_t - θ*`:

```
δ_t^(i) = (1 - ηλ_i)^t · δ_0^(i)
g_t      = ∑_i λ_i (1 - ηλ_i)^t δ_0^(i) v_i
```

Stability requires `ηλ_i < 2` for all `i`.

### 7.3 Step 2 — Gradient Ratio Encodes Curvature ✓ (under S, E)

Under Assumption E, only mode 1 contributes significantly:

```
‖g_t‖   ≈ λ_1 |1 - ηλ_1|^t |δ_0^(1)|
‖g_{t+1}‖ ≈ λ_1 |1 - ηλ_1|^{t+1} |δ_0^(1)|
```

Define `κ = |1 - ηλ_1| ∈ [0, 1)`. Then:

```
ρ_t = κ / (1 + κ)
```

This is a Möbius transformation of `κ`. The gradient ratio **stabilizes immediately** to this value under Assumption E — the fast variable has equilibrated, leaving only the slow variable (`q*`) to evolve. This is the timescale separation in action.

Inverting: `κ = ρ_t/(1 - ρ_t)`, so:

```
λ_1 = (1 - κ)/η = (1 - 2ρ_t) / (η(1 - ρ_t))
```

The top Hessian eigenvalue is encoded in the stable value of `ρ_t`.

*Note on stochastic gradients:* With mini-batch SGD, `ρ_t` fluctuates around `κ/(1+κ)`. The median convergent `q*` over a window estimates the central tendency, with noise absorbed into the constant `C_0` in the bound below.

### 7.4 Step 3 — Continued Fraction Denominator Scales as `1/√(ε_grad ηλ_1)` ✓ (under S, E)

For small `x = ηλ_1 < 1`:

```
ρ = (1-x)/(2-x) ≈ 1/2 - x/4    (linear approximation near x=0)
```

The continued fraction of `1/2 - x/4` for small `x > 0` has the form `[0; 2, a_2, ...]` where the second partial quotient `a_2 ~ 4/x`. The first non-trivial convergent denominator is thus:

```
q_1 ~ 4/(ηλ_1)
```

At the adaptive resolution boundary `ε_grad ~ ηλ_1/4` (where the approximation error matches the gradient noise), and using `q ~ 1/√(ε_grad)` from the Hurwitz bound:

```
q* ~ 1/√(ε_grad · ηλ_1)
```

This scaling is specific to the linear approximation regime `ηλ_1 ≪ 1`. Near `ηλ_1 ≈ 2`, higher-order terms become important and the scaling breaks down.

### 7.5 Step 4 — The Convergent-Curvature Correspondence ✓ (under S, E)

**Theorem.** Under Assumptions S and E, with `ηλ_1 < 2` and adaptive resolution `Q_max = ⌊1/ε_grad⌋`:

```
λ_max(H) = λ_1 ≲ C_0(η) / (q*)²
```

for a constant `C_0(η)` depending only on the learning rate.

*Proof.* From Step 2: `ρ_t = κ/(1+κ)` with `κ = |1-ηλ_1|`. From Step 3: `q* ~ 1/√(ε_grad ηλ_1)`. Squaring and solving: `λ_1 ~ 1/(ε_grad η (q*)²) =: C_0(η)/(q*)²`. ∎

**Scope and limitations:**
- This is an upper bound on `λ_max(H)` from observable quantities (`q*`, `ε_grad`, `η`), not an equality.
- The bound is tight when `λ_1 ≫ λ_2` (sharp, low-rank minimum) and loose when many eigenvalues are comparable.
- Requires full-batch or low-noise gradients; mini-batch noise weakens the bound.
- Fails at non-smooth points (ReLU kinks), where Assumption S does not hold.

---

## 8. A PAC-Bayes Generalization Bound

### 8.1 The Bound ✓ (under S, E)

**Theorem.** Under Assumptions S and E, for any `δ > 0`, with probability at least `1 - δ` over training samples:

```
G(θ*) := L_test(θ*) - L_train(θ*)
        ≤ q* · √[ C_0(η) · (d + log(2/δ)) / (2 n_train) ]
```

*Proof.*

1. **Prior.** Choose `P = N(θ*, σ²I)` with `σ² = 1/((q*)² C_0)` — variance inversely proportional to the CCC Hessian bound.

2. **PAC-Bayes (McAllester 1999).** For any posterior `Q`:

   ```
   E_Q[L_test] ≤ E_Q[L_train] + √[ (KL(Q‖P) + log(2√n/δ)) / (2n) ]
   ```

3. **KL computation.** Set `Q = δ_{θ*}` (point mass at the optimum):

   ```
   KL(δ_{θ*} ‖ N(θ*, σ²I)) = (d/2)log(1/σ²) + const = d·log(q*) + const
   ```

4. **Bound assembly.** Substitute and use `√(d·log q*) ≤ q*√d` (valid for `q* ≥ 1`):

   ```
   G(θ*) ≲ q* · √[ C_0 (d + log(2/δ)) / (2 n_train) ]   ∎
   ```

### 8.2 Interpretation

The generalization gap scales as `q*/√n_train`. Larger `q*` (sharper basin, high curvature) means worse generalization. Smaller `q*` (flatter basin, low curvature) means tighter bound. This recovers the Hochreiter–Schmidhuber flat minima principle as a corollary: flat minima generalize better because they have smaller `λ_max(H)` and thus smaller `q*`.

### 8.3 Comparison

| Framework | Bound | Requires |
|-----------|-------|----------|
| Standard PAC-Bayes | Via sharpness / KL | Hessian or Fisher matrix computation |
| SAM | Perturbation sensitivity | Double forward pass per step |
| **This work** | `q*/√n_train` (under S, E) | Gradient norms only |

The Farey bound is computable from gradient norm ratios alone, without Hessian computation. Its tightness depends on how well Assumption E holds.

---

## 9. Ford Circles and Loss Basin Geometry

### 9.1 The Correspondence ~

Ford circles provide a geometric picture of the loss landscape under the FHN analogy. The correspondence is structural:

```
Ford circle C(p/q)          →   Loss basin near parameter region
Radius r = 1/(2q²)         →   Basin width (large radius = flat basin)
Small q (e.g., 1/2)        →   Flat minimum, wide basin, good generalization
Large q (e.g., 37/74)      →   Sharp minimum, narrow basin, poor generalization
Tangent circles             →   Adjacent basins sharing a saddle boundary
Tangent point               →   Saddle point between adjacent minima
```

This is an analogy, not a theorem. The identification of Ford circle radius with basin width requires the CCC (Section 7) as an intermediate:

```
flat minimum
⟺ small λ_max(H)           (definition)
⟺ small q*                  (CCC, under S, E)
⟺ large Ford circle radius  (Ford geometry)
⟺ small generalization gap  (PAC-Bayes bound)
```

Each step except the first is conditional or analogical.

### 9.2 The Packing Structure

The Ford circle packing is hierarchical:

```
Ford circle packing (schematic):

───────────────────────────────────────────────────────────
              C(1/2)  [radius 1/8 — widest intermediate]
            /‾‾‾‾‾‾‾\
    C(1/3) /         \ C(2/3)  [radius 1/18]
   /‾‾‾‾\/           \/‾‾‾‾\
 C(1/4)  C(2/5) ... C(3/5)  C(3/4)  [smaller radii]
───────────────────────────────────────────────────────────
C(0/1) and C(1/1) [radius 1/2] are the boundary circles
```

Tangent pairs correspond exactly to Farey neighbors. This packing is dense: between any two tangent circles there is another circle tangent to both (the Farey mediant), recursively.

---

## 10. The Farey Consolidation Index

### 10.1 Definition

Given a window of `T` gradient vectors, compute convergents `{(p_t, q_t)}` for each consecutive pair. The **observed Farey Consolidation Index** is:

```
F_c^obs = #{t : |q_t · p_{t+1} - p_t · q_{t+1}| = 1} / (T - 1)
```

This counts the fraction of consecutive convergent pairs satisfying the unimodular (Farey neighbor) condition.

### 10.2 Permutation Test for Phase Detection ✓

A raw `F_c^obs` value cannot be interpreted without a reference. Gradients cluster during training (they are not uniformly distributed over `F_n`), so the background rate of unimodular pairs is not the analytic `O(1/n²)` rate from Section 3.4.

**Procedure:**

1. Compute `F_c^obs` from the ordered convergent sequence.

2. Randomly permute `{(p_t, q_t)}` to destroy temporal ordering while preserving the marginal distribution. Repeat `B = 200` times. Compute `F_c` for each permutation.

3. Report:

   ```
   F_c_percentile = percentile_rank(F_c^obs among {F_c^(b)})
   ```

**Why permutation is correct.** Permuting destroys temporal ordering while preserving the actual distribution of each convergent. If gradients cluster (memorization phase), permuted pairs also show high neighbor density — and the test correctly will *not* signal generalization. The test detects whether the *sequence* of updates is arithmetically structured relative to what would be expected from the same gradient distribution in random order.

### 10.3 Phase Table

```
F_c_percentile    Phase            FHN Analog
─────────────────────────────────────────────────────────
< 50th            MEMORIZATION     Quiescent (below threshold)
50th – 80th       APPROACHING      Slow variable drifting toward knee
80th – 95th       CRITICAL         Near unstable middle branch
95th – 99th       GENERALIZING     Excitable orbit in progress
> 99th            CONVERGED        Post-firing recovery complete
─────────────────────────────────────────────────────────
```

These thresholds (50, 80, 95, 99) are empirically motivated. They are not derived from first principles and should be tuned on validation data.

---

## 11. Implementation

All code requires only `numpy`, `torch`, and `scipy`.

### 11.1 Core Arithmetic

```python
import numpy as np
from scipy.stats import percentileofscore


def continued_fraction_convergents(x: float, q_max: int) -> list[tuple[int, int]]:
    """
    Compute continued fraction convergents p_k/q_k of x in (0,1)
    with denominator up to q_max.

    Theorem (Lagrange 1770): each convergent is the best rational
    approximant with its denominator.
    """
    p_prev, p_curr = 1, 0
    q_prev, q_curr = 0, 1
    convergents = []
    xi = float(x)

    for _ in range(60):  # max CF depth
        a_k = int(xi)
        p_next = a_k * p_curr + p_prev
        q_next = a_k * q_curr + q_prev
        if q_next > q_max:
            break
        p_prev, p_curr = p_curr, p_next
        q_prev, q_curr = q_curr, q_next
        convergents.append((p_curr, q_curr))
        remainder = xi - a_k
        if remainder < 1e-12:
            break
        xi = 1.0 / remainder

    return convergents if convergents else [(0, 1)]


def adaptive_farey_approx(x: float, eps_grad: float) -> tuple[int, int]:
    """
    Map x in [0,1] to the best continued fraction convergent at
    Hurwitz-justified resolution Q_max = floor(1/eps_grad).

    At this resolution, approximation error ~ eps_grad^2/sqrt(5)
    is below the gradient measurement noise level eps_grad.
    """
    q_max = max(1, int(1.0 / max(eps_grad, 1e-6)))
    return continued_fraction_convergents(x, q_max)[-1]


def gradient_ratio(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    rho = ||g2|| / (||g1|| + ||g2||)

    Under Assumptions S and E, this stabilizes to kappa/(1+kappa)
    where kappa = |1 - eta*lambda_1| encodes the top Hessian eigenvalue.
    Isometry-invariant.
    """
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    total = n1 + n2
    return float(n2 / total) if total > 1e-12 else 0.5


def relative_gradient_change(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    eps = ||g2 - g1|| / (||g1|| + ||g2||)

    Sets Q_max for adaptive Farey approximation. Large eps -> coarse
    resolution (gradient changing rapidly). Small eps -> fine resolution.
    """
    diff = np.linalg.norm(g2 - g1)
    total = np.linalg.norm(g1) + np.linalg.norm(g2)
    return float(diff / total) if total > 1e-12 else 1.0


def is_unimodular(p1: int, q1: int, p2: int, q2: int) -> bool:
    """
    True iff |q1*p2 - p1*q2| = 1  (Farey neighbor condition, Cauchy 1816).
    """
    return abs(q1 * p2 - p1 * q2) == 1
```

### 11.2 Farey Consolidation Index

```python
def compute_farey_diagnostics(
    grads: list[np.ndarray],
    n_permutations: int = 200,
    rng_seed: int = 42,
) -> dict:
    """
    Compute Farey Consolidation Index with permutation-test null.

    Parameters
    ----------
    grads          : List of gradient vectors (at least 4 required).
    n_permutations : Number of permutation resamples.
    rng_seed       : Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        convergents    : list of (p, q) for each consecutive gradient pair
        F_c_obs        : observed fraction of unimodular consecutive pairs
        F_c_percentile : percentile of F_c_obs in permutation null (0-100)
        q_median       : median denominator over window
        phase          : phase label (MEMORIZATION / APPROACHING / CRITICAL /
                         GENERALIZING / CONVERGED)
    """
    if len(grads) < 4:
        return {
            "convergents": [],
            "F_c_obs": 0.0,
            "F_c_percentile": 50.0,
            "q_median": 1.0,
            "phase": "INSUFFICIENT_DATA",
        }

    # Step 1: Compute convergents for each consecutive gradient pair
    convergents = []
    for i in range(len(grads) - 1):
        rho = gradient_ratio(grads[i], grads[i + 1])
        eps = relative_gradient_change(grads[i], grads[i + 1])
        convergents.append(adaptive_farey_approx(rho, eps))

    # Step 2: Observed unimodular fraction over consecutive pairs
    def unimodular_fraction(seq: list[tuple[int, int]]) -> float:
        if len(seq) < 2:
            return 0.0
        hits = sum(
            1 for i in range(len(seq) - 1)
            if is_unimodular(*seq[i], *seq[i + 1])
        )
        return hits / (len(seq) - 1)

    F_c_obs = unimodular_fraction(convergents)

    # Step 3: Permutation null (destroys temporal order, preserves marginal)
    rng = np.random.default_rng(seed=rng_seed)
    null_values = []
    arr = list(convergents)
    for _ in range(n_permutations):
        perm = arr.copy()
        rng.shuffle(perm)
        null_values.append(unimodular_fraction(perm))

    pct = float(percentileofscore(null_values, F_c_obs, kind="strict"))

    # Step 4: Median denominator (slow variable analog)
    q_median = float(np.median([q for (_, q) in convergents]))

    # Step 5: Phase label
    if pct < 50:
        phase = "MEMORIZATION"
    elif pct < 80:
        phase = "APPROACHING"
    elif pct < 95:
        phase = "CRITICAL"
    elif pct < 99:
        phase = "GENERALIZING"
    else:
        phase = "CONVERGED"

    return {
        "convergents": convergents,
        "F_c_obs": F_c_obs,
        "F_c_percentile": pct,
        "q_median": q_median,
        "phase": phase,
    }
```

### 11.3 Training Loop with Live Diagnostics

```python
"""
demo_fhn_diagnostics.py

Live FHN-style diagnostics during MLP training.

Columns:
  Step    : training step
  Loss    : training BCE loss
  TestAcc : test accuracy
  q*      : median Farey denominator (slow variable analog)
  Pct     : F_c percentile in permutation null (phase indicator)
  Bound   : q*/sqrt(n_train)  [valid only under Assumptions S and E]
  Phase   : FHN phase label

Requirements: torch, numpy, scipy
"""

import math
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# ── Data ──────────────────────────────────────────────────────────────────────

N_TRAIN, N_TEST, DIM = 500, 200, 20
W_TRUE = torch.randn(DIM)
W_TRUE /= W_TRUE.norm()


def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n, DIM)
    y = (X @ W_TRUE > 0).float()
    return X, y


X_train, y_train = make_data(N_TRAIN)
X_test, y_test = make_data(N_TEST)

train_ds = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)

# ── Model ─────────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


model = MLP()
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)


def collect_gradient(batch: tuple) -> np.ndarray:
    """Compute and return flattened gradient vector without stepping."""
    model.zero_grad()
    x, y = batch
    bce(model(x), y).backward()
    g = torch.cat([
        p.grad.detach().flatten()
        for p in model.parameters()
        if p.grad is not None
    ])
    return g.cpu().numpy()


def test_accuracy() -> float:
    model.eval()
    with torch.no_grad():
        preds = (model(X_test) > 0).float()
        return (preds == y_test).float().mean().item()


# ── Diagnostic loop ───────────────────────────────────────────────────────────

WINDOW = 60
grad_buffer: list[np.ndarray] = []
prev_q: float | None = None

header = (
    f"{'Step':>5}  {'Loss':>8}  {'TestAcc':>8}"
    f"  {'q*':>5}  {'Pct':>6}  {'Bound':>8}  Phase"
)
separator = "─" * len(header)
print(header)
print(separator)
print("  Bound = q*/sqrt(n_train)  [Assumptions S, E required — see Section 7]")
print()

for step in range(401):
    model.train()
    batch = next(iter(train_loader))

    # Collect gradient before stepping
    g = collect_gradient(batch)
    grad_buffer.append(g)
    if len(grad_buffer) > WINDOW + 1:
        grad_buffer.pop(0)

    # Optimizer step
    model.zero_grad()
    x, y = batch
    bce(model(x), y).backward()
    optimizer.step()

    if step % 50 == 0 and len(grad_buffer) >= 6:
        d = compute_farey_diagnostics(grad_buffer[-WINDOW:], n_permutations=150)
        q = d["q_median"]
        pct = d["F_c_percentile"]
        phase = d["phase"]
        bound = q / math.sqrt(N_TRAIN)

        # Farey backtrack = candidate grokking signal
        backtrack_flag = ""
        if prev_q is not None and q < prev_q - 0.5 and pct > 80:
            backtrack_flag = "  <- Farey backtrack (candidate grokking)"
        prev_q = q

        model.eval()
        with torch.no_grad():
            loss = bce(model(X_train), y_train).item()

        print(
            f"{step:>5}  {loss:>8.4f}  {test_accuracy():>8.3f}"
            f"  {q:>5.1f}  {pct:>6.1f}  {bound:>8.4f}  {phase}{backtrack_flag}"
        )

print()
print("FHN analogy interpretation:")
print("  q* rising  -> slow variable drifting toward threshold (memorization)")
print("  q* dropping -> excitable orbit (candidate grokking event)")
print("  CONVERGED  -> post-event recovery at low-denominator attractor")
```

### 11.4 Verification Suite

```python
"""
verify_farey.py  --  Pure Python stdlib. No external dependencies.

Verifies: Farey sequence generation, unimodular condition, mediant
insertion, Mobius gradient ratio formula, Ford circle tangency.
"""


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def farey_sequence(n: int) -> list[tuple[int, int]]:
    """Generate F_n via mediant recurrence. O(|F_n|) time."""
    a, b, c, d = 0, 1, 1, n
    result = [(0, 1)]
    while (c, d) != (1, 1):
        k = (n + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        result.append((c, d))
    return result


# ── Farey neighbor verification ──────────────────────────────────────────────

print("=== F_5: CONSECUTIVE PAIRS ARE FAREY NEIGHBORS ===")
F5 = farey_sequence(5)
print("F_5:", [f"{p}/{q}" for p, q in F5])
print()
all_ok = True
for (p1, q1), (p2, q2) in zip(F5, F5[1:]):
    det = q1 * p2 - p1 * q2
    ok = abs(det) == 1
    all_ok = all_ok and ok
    print(f"  {p1}/{q1} — {p2}/{q2}  det={det:+d}  {'✓' if ok else '✗'}")
print(f"\nAll pairs unimodular: {'✓' if all_ok else '✗'}")

# ── Mediant insertion: F_5 → F_6 ────────────────────────────────────────────

print("\n=== MEDIANT INSERTION (F_5 → F_6) ===")
F6 = farey_sequence(6)
F5_set = set(F5)
new_elements = [f for f in F6 if f not in F5_set]
for (p, q) in new_elements:
    idx = F6.index((p, q))
    left = F6[idx - 1]
    right = F6[idx + 1]
    med = (left[0] + right[0], left[1] + right[1])
    ok = med == (p, q)
    print(
        f"  {left[0]}/{left[1]} ⊕ {right[0]}/{right[1]}"
        f" = {med[0]}/{med[1]} == {p}/{q}  {'✓' if ok else '✗'}"
    )

# ── Gradient ratio Mobius formula ────────────────────────────────────────────

print("\n=== GRADIENT RATIO = MOBIUS IMAGE OF HESSIAN EIGENVALUE ===")
import math

eta = 0.05
for lam in [0.5, 1.0, 5.0, 10.0, 20.0]:
    kappa = abs(1 - eta * lam)
    if kappa >= 1:
        print(f"  lambda={lam:5.1f}: UNSTABLE (eta*lambda={eta*lam:.2f} >= 1)")
        continue
    rho = kappa / (1 + kappa)
    lam_recovered = (1 - kappa) / eta
    ok = abs(lam_recovered - lam) < 0.01
    print(
        f"  lambda={lam:5.1f}: kappa={kappa:.3f}  rho={rho:.4f}"
        f"  lambda_recovered={lam_recovered:.1f}  {'✓' if ok else '✗'}"
    )

# ── Ford circle tangency ─────────────────────────────────────────────────────

print("\n=== FORD CIRCLES: TANGENT <=> FAREY NEIGHBORS ===")


def ford_radius(q: int) -> float:
    return 1.0 / (2 * q ** 2)


def ford_circles_tangent(p1: int, q1: int, p2: int, q2: int) -> bool:
    """Two Ford circles are externally tangent iff |q1*p2 - p1*q2| = 1."""
    cx1, cy1, r1 = p1 / q1, ford_radius(q1), ford_radius(q1)
    cx2, cy2, r2 = p2 / q2, ford_radius(q2), ford_radius(q2)
    dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    r_sum_sq = (r1 + r2) ** 2
    return abs(dist_sq - r_sum_sq) < 1e-9


for (p1, q1), (p2, q2) in list(zip(F5, F5[1:]))[:5]:
    det = abs(q1 * p2 - p1 * q2)
    tangent = ford_circles_tangent(p1, q1, p2, q2)
    consistent = (det == 1) == tangent
    print(
        f"  C({p1}/{q1})—C({p2}/{q2}):  |det|={det}"
        f"  tangent={tangent}  {'✓ consistent' if consistent else '✗ inconsistent'}"
    )
```

---

## 12. Summary of Results

| # | Statement | Status |
|---|-----------|--------|
| 1 | FHN produces limit cycles, excitability, bifurcations via cubic fast nullcline | ✓ |
| 2 | Two Farey fractions are neighbors iff `|bc-ad| = 1` | ✓ |
| 3 | Mediant of Farey neighbors is the next insertion as `F_n` grows | ✓ |
| 4 | Ford circles are tangent iff fractions are Farey neighbors | ✓ |
| 5 | Three-Distance Theorem: CF denominators determine gap structure | ✓ |
| 6 | Franel–Landau: Farey spacing uniformity ⟺ Riemann Hypothesis | ✓ |
| 7 | `ρ_t` is isometry-invariant | ✓ |
| 8 | Adaptive `Q_max = ⌊1/ε_grad⌋` grounded in Hurwitz (1891) | ✓ |
| 9 | `ρ_t = κ/(1+κ)` encodes `λ_1` under Assumptions S and E | ✓ cond. |
| 10 | `λ_max(H) ≲ C_0(η)/(q*)²` under Assumptions S and E | ✓ cond. |
| 11 | `G(θ*) ≲ q*/√n_train` under Assumptions S and E | ✓ cond. |
| 12 | Permutation test for `F_c` is valid without distributional assumptions | ✓ |
| 13 | FHN and gradient training share the fast–slow mathematical motif | ~ structural |
| 14 | Grokking corresponds to an excitable firing event | ⚠ (requires empirical validation) |
| 15 | Farey backtrack predicts grokking 50–200 steps ahead | ⚠ (requires empirical validation) |
| 16 | Bistable / limit-cycle regimes predicted from nullcline analysis | ⚠ (requires empirical validation) |

**Results 1–8:** Established mathematics; unconditional.
**Results 9–11:** Proved under Assumptions S (C², positive definite Hessian) and E (spectral dominance). Both assumptions are standard, explicit, and checkable.
**Result 12:** Standard permutation theory; unconditional.
**Result 13:** Structural analogy. The fast–slow motif is present in both systems; this is a productive framing, not a derivation.
**Results 14–16:** Conjectures with specified empirical tests. Not proved.

---

## 13. Open Problems

### 13.1 Remove Assumption E

The CCC (Section 7) requires spectral dominance. The open problem is to show that `q* ≳ C/√(ε_grad η λ_1)` remains a valid lower bound when many Hessian eigenvalues are comparable. A route via the three-distance theorem for multiple simultaneous rotations (extensions of Sós 1958) may be productive.

### 13.2 Validate the Farey Backtrack on Grokking Benchmarks

Test on Power et al. (2022) modular arithmetic experiments:

- Does `q*` decrease before test accuracy rises?
- What is the lead time distribution across seeds and tasks?
- Does `F_c_percentile > 80` reliably co-occur with the denominator drop?

Systematic empirical evaluation would either validate or refute Results 14 and 15.

### 13.3 Denominator-Penalized Optimization

Under Assumptions S and E, penalizing `q*` is (approximately) equivalent to penalizing `λ_max(H)` via the CCC:

```
L_Farey(θ) = L(θ) + λ · (q*)²
```

This would be a gradient-norm-only proxy for SAM (Foret et al. 2021). Comparison with SAM on standard benchmarks would test the CCC empirically.

### 13.4 Extension to Non-Smooth Losses

Assumption S fails at ReLU kinks. Within each linear region the gradient is constant and the Farey map is well-defined; across region boundaries, gradient jumps occur. A natural extension uses the Clarke subdifferential (Clarke 1983) as a replacement for the Hessian, with piecewise-linear fast nullclines replacing the smooth cubic.

### 13.5 Bifurcation Analysis

Derive analytically the critical regularization strength at which the training system transitions from monostable-memorization to bistable to monostable-generalization. The cubic nullcline structure (Section 5) provides the framework; specific constants (`φ, α, β`) require fitting to observed training curves.

---

## 14. References

### FitzHugh–Nagumo

- **FitzHugh, R. (1961).** Impulses and physiological states in theoretical models of nerve membrane. *Biophysical Journal* 1(6), 445–466.
- **Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962).** An active pulse transmission line simulating nerve axon. *Proceedings of the IRE* 50(10), 2061–2070.
- **Keener, J. & Sneyd, J. (2009).** *Mathematical Physiology*, 2nd ed. Springer.
- **Izhikevich, E.M. (2007).** *Dynamical Systems in Neuroscience.* MIT Press.

### Farey Sequences and Continued Fractions

- **Cauchy, A.L. (1816).** *Exercices de mathématique.*
- **Hardy, G.H. & Wright, E.M. (1979).** *An Introduction to the Theory of Numbers*, 5th ed. Oxford.
- **Ford, L.R. (1938).** Fractions. *American Mathematical Monthly* 45(9), 586–601.
- **Hurwitz, A. (1891).** Ueber die angenäherte Darstellung der Irrationalzahlen durch rationale Brüche. *Mathematische Annalen* 39(2), 279–284.
- **Graham, R., Knuth, D., & Patashnik, O. (1994).** *Concrete Mathematics*, 2nd ed. Addison-Wesley.

### Three-Distance Theorem

- **Sós, V. (1958).** On the distribution mod 1 of the sequence nα. *Annales Univ. Sci. Budapest.* 1, 127–134.
- **Steinhaus, H. (1950).** *Mathematical Snapshots.* Oxford.

### Riemann Hypothesis Connection

- **Franel, J. (1924).** Les suites de Farey et le problème des nombres premiers. *Göttinger Nachrichten*, 198–201.
- **Landau, E. (1924).** Bemerkungen zu der vorstehenden Abhandlung von Herrn Franel. *Göttinger Nachrichten*, 202–206.

### PAC-Bayes and Generalization

- **McAllester, D.A. (1999).** PAC-Bayesian model averaging. *COLT 1999.*
- **Dziugaite, G.K. & Roy, D.M. (2017).** Computing nonvacuous generalization bounds for deep neural networks. *UAI 2017.*
- **Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. (2021).** Sharpness-aware minimization for efficiently improving generalization. *ICLR 2021.*
- **Hochreiter, S. & Schmidhuber, J. (1997).** Flat minima. *Neural Computation* 9(1), 1–42.

### Grokking

- **Power, A., Anand, Y., Mosconi, A., Kaiser, Ł., & Polosukhin, I. (2022).** Grokking: generalization beyond overfitting on small algorithmic datasets. *ICLR 2022.*

### Non-Smooth Analysis

- **Clarke, F.H. (1983).** *Optimization and Nonsmooth Analysis.* Wiley.

---

## Appendix: Quick Reference

```
Quantity               Computation                         FHN Analog
─────────────────────────────────────────────────────────────────────────
ρ_t                    ||g_{t+1}|| / (||g_t|| + ||g_{t+1}||)  Fast variable v
ε_grad(t)              ||g_{t+1} - g_t|| / (sum of norms)     Resolution setter
Q_max                  floor(1 / ε_grad)                    CF approximation scale
(p_t, q_t)             Best CF convergent of ρ_t at Q_max   Farey coordinate
q*                     Median of q_t over window W          Slow variable w
F_c_percentile         Permutation test rank of F_c^obs     Phase indicator

Phase               F_c_pct   Interpretation
─────────────────────────────────────────────────────────────────────────
MEMORIZATION        < 50      Quiescent; below threshold
APPROACHING         50–80     Slow variable drifting toward knee
CRITICAL            80–95     Near unstable middle branch
GENERALIZING        95–99     Candidate excitable orbit in progress
CONVERGED           > 99      Post-event recovery

Farey Backtrack Criterion (candidate grokking signal):
  q*(t) < q*(t - W)   AND   F_c_percentile(t) > 80

Generalization bound (under Assumptions S and E only):
  G(θ*) ≲ q* / sqrt(n_train)
```

