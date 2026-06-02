# free-probability-c

**Xavier initialization. He initialization. Kaiming initialization.**

You've used these. You know they work. But *why* do they work?

Because they match the Marchenko-Pastur distribution. Free probability explains what your weight matrices are doing when you're not looking.

---

## The R-Transform Is the Fourier Transform for Matrices

You know how the Fourier transform turns convolution into multiplication? The R-transform does the same trick for random matrix addition.

For two **freely independent** random matrices A and B:

```
R_{A+B}(z) = R_A(z) + R_B(z)
```

That's it. The entire eigenvalue distribution of the *sum of two matrices* — something that would normally require an O(n³) eigendecomposition — reduces to adding two power series. No matrix arithmetic. No spectral decomposition. Just add the free cumulants term by term.

And for matrix *multiplication*, the S-transform does the analogous thing:

```
S_{AB}(z) = S_A(z) · S_B(z)
```

Multiply two polynomials. That's the eigenvalue distribution of the product.

This library implements both transforms, plus the Marchenko-Pastur law, moment-cumulant conversion, and practical gradient analysis tools — in ~600 lines of C with zero dependencies beyond libm.

---

## Why Initialization Matters: The Marchenko-Pastur Law

Take a random weight matrix W of shape d×d with entries drawn from N(0, σ²). Form the covariance WᵀW/d. What do its eigenvalues look like?

Not Gaussian. Not uniform. They follow the **Marchenko-Pastur distribution**:

```
ρ(x) = √((b−x)(x−a)) / (2πσ²λx)
```

where λ = p/n is the feature-to-sample ratio, a = σ²(1−√λ)², b = σ²(1+√λ)².

This distribution has **hard edges**. The eigenvalues are packed into a finite interval [a, b]. They *cannot* escape — unless your initialization violates the assumptions that make the MP law hold.

**Every training failure that looked random was a Marchenko-Pastur violation.** The variance was too large, so eigenvalues spilled past the upper edge. Gradients exploded. The "randomness" was structured — and free probability reads the structure.

Xavier init uses σ = √(2/(fan_in + fan_out)). He init uses σ = √(2/fan_in). These aren't magic numbers someone found by trial and error. They're the exact scales that keep the MP bulk bounded — that keep your eigenvalue spectrum well-conditioned. The `1/√d` isn't arbitrary; it's the free probability condition for spectral stability.

### The Semicircle Law Shows Up Too

The MP moments for λ=1 are the Catalan numbers: 1, 2, 5, 14, 42, 132... These are also the moments of Wigner's semicircle law. That's not a coincidence — it's the free probability version of the central limit theorem. Stack enough freely independent random matrices and their eigenvalue distribution converges to a semicircle, exactly like summing classically independent random variables converges to a Gaussian.

The free cumulants of MP(λ=1) are the cleanest objects in all of free probability: κ_n = 1 for all n. The Catalan numbers emerge from the non-crossing partition formula: m_n = Σ_{π ∈ NC(n)} 1, and that sum *is* the Catalan number C_n.

---

## Concrete Example: Predict the Eigenvalue Distribution of Combined Layers

You're building a 12-layer transformer. Hidden dim 768. You want to know: what happens to the eigenvalue distribution when all those weight matrices interact through residual connections?

You *could* multiply twelve 768×768 matrices and compute eigenvalues. That's expensive and tells you about one specific initialization.

Or you could use free probability:

```c
#include "free_prob.h"

// Your single layer's free cumulants (from weight matrix eigenvalues)
double layer_cumulants[] = {1.0, 0.5, 0.1, 0.01};

// R-transform additivity: combined cumulants = n × single-layer cumulants
double combined[4];
predict_combined_distribution(layer_cumulants, 4, 4, combined);
// combined = {4.0, 2.0, 0.4, 0.04}

// Ask Marchenko-Pastur theory: what init scale keeps this stable?
TransformerConfig config = {
    .n_layers    = 12,
    .hidden_dim  = 768.0,
    .n_samples   = 10000.0,
    .weight_std  = 0.02,
    .learning_rate = 0.001
};
InitSuggestion sug = suggest_initialization(&config);
// sug.suggested_std    — init scale that keeps MP eigenvalues bounded
// sug.condition_number — predicted condition number from MP edges
// sug.suggested_lr_scale — learning rate scaled to spectral radius

// Detect if your gradients have outlier eigenvalues
double moments[] = {0.0, 1.0, 0.0, 3.0, 0.0, 15.0}; // gradient covariance moments
RegularizerSuggestion reg = suggest_regularizer(moments, 6, &config);
// reg.outlier_fraction — eigenvalues outside the MP bulk (these cause instability)
// reg.lambda_reg       — regularization matched to tail behavior
// reg.stability_score  — 0–1, how well-behaved your spectrum is
```

You just predicted the eigenvalue structure of a 12-layer network **without computing any 768×768 matrices.** That's free probability.

---

## The Four Operations You Actually Need

| What you want | Free probability tool | Function |
|---|---|---|
| Eigenvalues of A + B | R-transform additivity | `r_transform_add()` |
| Eigenvalues of A × B | S-transform multiplication | `s_transform_multiply()` |
| Is this init stable? | Marchenko-Pastur bounds | `suggest_initialization()` |
| Why are my gradients exploding? | Tail mass outside MP bulk | `suggest_regularizer()` |

---

## Full API

### Moments ↔ Free Cumulants (`src/moments.c`)

The foundation. Free cumulants are to eigenvalue distributions what classical cumulants are to scalar distributions — but they capture **matrix structure**, not just scalar statistics.

```c
void compute_moments(const EmpiricalDist *dist, unsigned max_k, double *output);
void moment_to_cumulant(const double *moments, size_t n, double *cumulants);
void cumulant_to_moment(const double *cumulants, size_t n, double *moments);
bool validate_moments(const double *moments, size_t n);
```

The moment-to-cumulant transform uses the non-crossing partition formula:

> m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}

This is what makes free cumulants "free" — they sum over *non-crossing* partitions instead of all partitions. The combinatorial structure is simpler, and that simplicity is what makes R-transform additivity possible.

### R-Transform — Free Additivity (`src/r_transform.c`)

```c
double r_transform_from_cumulants(const double *cumulants, size_t n, double z);
void   r_transform_add(const double *a, const double *b, size_t n, double *sum);
double stieltjes_from_r(const double *cumulants, size_t n, double z,
                        unsigned max_iter, double tol);
double marchenko_pastur_density(double x, double lambda, double sigma);
void   marchenko_pastur_moments(double lambda, unsigned max_k, double *moments);
```

The MP free cumulants: κ_n = λ^(n−1). For λ=1, every free cumulant equals 1. This is why the moments are Catalan numbers — they count non-crossing partitions weighted by all-ones cumulants.

The Stieltjes transform is recovered from the R-transform via fixed-point iteration on G(z) = 1/(z − R(G(z))). From G(z), you get the eigenvalue density by the Sokhotski–Plemelj formula: ρ(x) = −(1/π) Im G(x + iε).

### S-Transform — Free Multiplicativity (`src/s_transform.c`)

```c
void s_transform_from_moments(const double *moments, size_t n, double *s_coeffs);
void s_transform_multiply(const double *sa, const double *sb,
                          size_t n, double *product);
```

The S-transform is computed via compositional inversion (series reversion) of the moment generating function ψ(z) = Σ m_n z^n, then S(z) = (1+z)/(z · χ(z)) where χ = ψ⁻¹. For MP(1), S(z) = 1/(1+z), which expands as 1 − z + z² − z³ + ⋯.

### Gradient Analysis (`src/gradient_analysis.c`)

```c
void predict_combined_distribution(const double *layer_cumulants,
                                   unsigned n_layers, size_t cumulant_order,
                                   double *combined_cumulants);
InitSuggestion suggest_initialization(const TransformerConfig *config);
RegularizerSuggestion suggest_regularizer(const double *moments, size_t n,
                                          const TransformerConfig *config);
```

---

## Applications

### 1. Weight Initialization

The init scale σ = 1/√d isn't folklore. It's the exact condition for the MP upper edge σ²(1+√λ)² to stay bounded when the matrix is d×d. `suggest_initialization()` computes this for your architecture and adjusts for depth — because R-transforms add up, deeper networks need smaller init to compensate.

### 2. Gradient Analysis

Compute moments of your gradient covariance eigenvalues. Compare against Marchenko-Pastur. Eigenvalues outside the MP support are outliers — they're the ones causing training instability. `outlier_fraction` tells you exactly what fraction of your gradient spectrum is pathological.

### 3. Regularization Design

L2 regularization is spectral shrinkage. The right strength depends on eigenvalue tail behavior. `suggest_regularizer()` computes this from free cumulant structure — heavier tails (higher free kurtosis) need stronger regularization, scaled by √(n_layers) because that's how the R-transforms compound across depth.

### 4. Layer Combination Prediction

Residual connections add matrices. Batch norm and layer fusion combine them. The R-transform tells you exactly what eigenvalue distribution results — in O(k) time where k is the cumulant order, not O(n³) for the full eigendecomposition.

---

## Building

```bash
make          # static library (libfreeprob.a)
make test     # build and run 30 tests
make clean
```

**Dependencies:** C11 compiler, libm. That's it. ~600 lines of C. One header.

---

## The Deeper Point

Free probability is what happens when you stop treating matrices like big numbers and start treating them like what they are — non-commutative random variables.

In classical probability, you use the Fourier transform because it turns convolution into multiplication. In free probability, you use the R-transform because it turns matrix addition into scalar addition. The analogy is exact:

| Classical probability | Free probability |
|---|---|
| Characteristic function φ(t) | Cauchy transform G(z) |
| Log of φ → classical cumulants | R-transform → free cumulants |
| Cumulants add for independent RVs | Free cumulants add for freely independent matrices |
| Central limit → Gaussian | Free central limit → Semicircle |
| Law of large numbers → deterministic mean | Marchenko-Pastur → deterministic spectrum |

The eigenvalue distribution isn't a scalar statistic. It's the entire shape of how your network transforms information. Free probability gives you the algebra to reason about that shape at scale.

This library makes that algebra available in the fewest lines possible, with the cleanest API, so you can use it in your training loop, your analysis pipeline, or your next paper.

---

## Math Appendix

### Non-Crossing Partitions

A partition π of {1, ..., n} is *non-crossing* if there are no blocks B₁, B₂ ∈ π and indices a < b < c < d with a, c ∈ B₁ and b, d ∈ B₂. The number of non-crossing partitions of {1, ..., n} is the Catalan number C_n = (1/(n+1))binom(2n, n).

### Free Cumulants

Free cumulants κ_n are defined implicitly by the moment-cumulant formula:

> m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}

This inverts recursively: κ_1 = m_1, and for n > 1:

> κ_n = m_n − Σ_{π ∈ NC(n), π ≠ {{1,...,n}}} Π_{B ∈ π} κ_{|B|}

### R-Transform

> R(z) = Σ_{n≥1} κ_n z^{n−1}

The R-transform linearizes free additive convolution. If A and B are freely independent, then R_{A+B} = R_A + R_B. This is the free analogue of how log-characteristic functions add for classically independent random variables.

### S-Transform

Define ψ(z) = Σ_{n≥1} m_n z^n (the moment generating series) and let χ be its compositional inverse: ψ(χ(z)) = z. Then:

> S(z) = (1+z) / (z · χ(z))

The S-transform linearizes free multiplicative convolution: S_{AB} = S_A · S_B for freely independent A, B.

### Marchenko-Pastur Distribution

For X = (1/n)WWᵀ where W is p×n with iid entries of variance σ²/p, the limiting eigenvalue density as p, n → ∞ with p/n → λ is:

> ρ(x) = √((b−x)(x−a)) / (2πσ²λx), &nbsp;&nbsp; a = σ²(1−√λ)², &nbsp;&nbsp; b = σ²(1+√λ)²

with a point mass max(0, 1−λ) at 0 when λ < 1. The free cumulants are κ_n = σ^{2n} · λ^{n−1}.

### Cauchy Transform and Density Recovery

The Cauchy transform G(z) = ∫ dμ(x)/(z−x) satisfies K(G(z)) = z where K(w) = R(w) + 1/w. The eigenvalue density is recovered via:

> ρ(x) = −(1/π) lim_{ε→0⁺} Im G(x + iε)

The implementation uses fixed-point iteration G_{k+1} = 1/(z − R(G_k)).

## License

MIT
