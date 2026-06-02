# free-probability-c

**The eigenvalue math that explains why your neural network training fails — and how to fix it.**

---

You know the eigenvalues of your weight matrices matter. But what happens to eigenvalues when you **ADD** two random matrices? Or **MULTIPLY** them? Classical probability can't tell you. Free probability can.

## The R-Transform: Fourier Transform for Matrices

The R-transform is to free probability what the Fourier transform is to classical probability.

```
R_{A+B}(z) = R_A(z) + R_B(z)
```

That's it. The eigenvalue distribution of a **sum** of random matrices is just the sum of their R-transforms. No matrix multiplication. No eigendecomposition. Just add the free cumulants.

And for matrix **products**:

```
S_{AB}(z) = S_A(z) · S_B(z)
```

The S-transform is multiplicative. Multiply the coefficient sequences. Done.

This library implements both transforms — plus the Marchenko-Pastur law, moment-cumulant conversion, and gradient analysis tools — in ~600 lines of C with zero dependencies beyond libm.

## Why This Exists: Every Initialization Trick Is Free Probability

### Xavier/He Initialization ≠ Magic

The Marchenko-Pastur law gives the exact eigenvalue density of a random covariance matrix:

```
ρ(x) = √((b-x)(x-a)) / (2πσ²λx),   a = σ²(1-√λ)²,  b = σ²(1+√λ)²
```

where λ = p/n is the feature-to-sample ratio.

If your initialization doesn't match this distribution, your gradients will explode or vanish. That's not a hypothesis — it's a theorem. **Xavier and He initialization work because they're designed to keep the MP bulk bounded.** The variance scale `1/fan_in` isn't arbitrary; it's the free probability condition for spectral stability.

**Every neural network training failure that looked random was actually a violation of Marchenko-Pastur. The randomness was structured. Free probability reads the structure.**

### The Semicircle Law Shows Up Too

The MP moments for λ=1 are the Catalan numbers: 1, 2, 5, 14, 42, 132... Same as the semicircle law. That's not a coincidence — it's the free probability version of the central limit theorem. Add enough freely independent random matrices and you get a semicircle, just like adding enough classically independent random variables gives you a Gaussian.

## Concrete Example: Predict Combined Layer Eigenvalues

You're building a 12-layer transformer with hidden_dim=768. You want to know: what happens to the eigenvalue distribution when all those layers interact?

```c
#include "free_prob.h"

// Your single layer's free cumulants (from its weight matrix eigenvalues)
double layer_cumulants[] = {1.0, 0.5, 0.1, 0.01};

// Predict combined distribution across 4 layers
double combined[4];
predict_combined_distribution(layer_cumulants, 4, 4, combined);
// combined = {4.0, 2.0, 0.4, 0.04} — just n×κ because R-transform is additive

// Get initialization suggestion from Marchenko-Pastur theory
TransformerConfig config = {
    .n_layers = 12,
    .hidden_dim = 768.0,
    .n_samples = 10000.0,
    .weight_std = 0.02,
    .learning_rate = 0.001
};
InitSuggestion sug = suggest_initialization(&config);
// sug.suggested_std — the init scale that keeps MP eigenvalues bounded
// sug.condition_number — predicted condition number from MP edges
// sug.suggested_lr_scale — learning rate scaled to spectral radius

// Detect if your gradients have outlier eigenvalues
double moments[] = {0.0, 1.0, 0.0, 3.0, 0.0, 15.0}; // from gradient covariance
RegularizerSuggestion reg = suggest_regularizer(moments, 6, &config);
// reg.outlier_fraction — eigenvalues outside the MP bulk (these cause instability)
// reg.lambda_reg — regularization strength matched to tail behavior
// reg.stability_score — 0-1, how well-behaved your eigenvalue distribution is
```

You just predicted the eigenvalue structure of a 12-layer network **without computing any 768×768 matrices.** That's free probability.

## The Four Operations You Need

| What you want | Free probability tool | Function |
|---|---|---|
| Eigenvalues of A + B | R-transform additivity | `r_transform_add()` |
| Eigenvalues of A × B | S-transform multiplication | `s_transform_multiply()` |
| Is this init stable? | Marchenko-Pastur bounds | `suggest_initialization()` |
| Why are my gradients exploding? | Tail mass outside MP | `suggest_regularizer()` |

## The Full API

### Moments ↔ Free Cumulants (`src/moments.c`)

The foundation. Free cumulants are to eigenvalue distributions what classical cumulants are to probability distributions — but they capture **matrix structure**, not just scalar statistics.

```c
// Compute empirical moments from eigenvalue samples
void compute_moments(const EmpiricalDist *dist, unsigned max_k, double *output);

// Moment sequence → free cumulant sequence (the free probability workhorse)
void moment_to_cumulant(const double *moments, size_t n, double *cumulants);

// Free cumulant sequence → moment sequence
void cumulant_to_moment(const double *cumulants, size_t n, double *moments);

// Check if moments correspond to a valid distribution (Hankel matrix PSD)
bool validate_moments(const double *moments, size_t n);
```

The moment-to-cumulant transform uses the non-crossing partition formula: `m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}`. This is what makes free cumulants "free" — they sum over non-crossing partitions instead of all partitions.

### R-Transform — Free Additivity (`src/r_transform.c`)

```c
// Evaluate R(z) = Σ κ_n z^{n-1}
double r_transform_from_cumulants(const double *cumulants, size_t n, double z);

// THE KEY OPERATION: R_{A+B} = R_A + R_B (just add cumulant sequences)
void r_transform_add(const double *cumulants_a, const double *cumulants_b,
                     size_t n, double *cumulants_sum);

// Stieltjes transform from R-transform (fixed-point iteration)
double stieltjes_from_r(const double *cumulants, size_t n, double z,
                        unsigned max_iter, double tol);

// Marchenko-Pastur density: ρ(x) for ratio λ and scale σ²
double marchenko_pastur_density(double x, double lambda, double sigma);

// Exact MP moments — returns Catalan numbers for λ=1
void marchenko_pastur_moments(double lambda, unsigned max_k, double *moments);
```

The MP free cumulants have the cleanest form in all of free probability: `κ_n = λ^{n-1}`. That's it. For λ=1, every free cumulant is 1. This is why the moments are Catalan numbers — they count non-crossing partitions weighted by all-1 cumulants.

### S-Transform — Free Multiplicativity (`src/s_transform.c`)

```c
// Compute S-transform coefficients from moments
void s_transform_from_moments(const double *moments, size_t n, double *s_coeffs);

// THE KEY OPERATION: S_{AB} = S_A · S_B (polynomial multiplication of coefficients)
void s_transform_multiply(const double *sa, const double *sb,
                          size_t n, double *product);
```

The S-transform is computed via compositional inversion of the moment generating function, then `(1+z)/(z·χ(z))`. The implementation uses Lagrange-style series reversion.

### Gradient Analysis (`src/gradient_analysis.c`)

The applied layer. Connects free probability theory to practical training decisions:

```c
// Predict eigenvalue distribution across n freely independent layers
void predict_combined_distribution(const double *layer_cumulants,
                                   unsigned n_layers, size_t cumulant_order,
                                   double *combined_cumulants);

// Suggest weight initialization from Marchenko-Pastur theory
InitSuggestion suggest_initialization(const TransformerConfig *config);

// Suggest regularization from eigenvalue tail behavior
RegularizerSuggestion suggest_regularizer(const double *moments, size_t n,
                                          const TransformerConfig *config);
```

## Applications

### 1. Weight Initialization
The init scale `σ = 1/√d` isn't folklore — it's the exact condition for the MP upper edge to stay bounded. `suggest_initialization()` computes this for your architecture, adjusting for depth (deeper networks need smaller init because R-transforms add up).

### 2. Gradient Analysis
Compute moments of your gradient covariance eigenvalues, then compare against Marchenko-Pastur. Eigenvalues outside the MP bulk are outliers — they're the ones causing training instability. `outlier_fraction` quantifies exactly how much of your gradient spectrum is pathological.

### 3. Regularization Design
L2 regularization is spectral shrinkage. The right strength depends on the tail behavior of your eigenvalue distribution. `suggest_regularizer()` computes this from free cumulant structure — heavier tails (higher kurtosis) need stronger regularization, scaled by `√(n_layers)` because that's how the R-transforms compound.

### 4. Layer Combination Prediction
When you add residual connections, batch norm, or layer fusion, you're combining random matrices. The R-transform tells you exactly what eigenvalue distribution results — without ever computing the combined matrix. `predict_combined_distribution()` does this in O(k) where k is the cumulant order.

## Building

```bash
make          # build static library (libfreeprob.a)
make test     # build and run 23 tests
make clean    # clean artifacts
```

## Dependencies

- C11 compiler
- libm

Zero external dependencies. ~600 lines of C. One header file.

## The Deeper Point

Free probability is what happens when you stop treating matrices like big numbers and start treating them like what they are — non-commutative random variables. The eigenvalue distribution isn't a scalar statistic; it's the entire shape of how your network transforms information.

The R-transform and S-transform are the encoding that makes matrix algebra tractable at scale. Just like the Fourier transform turns convolution into multiplication, the R-transform turns matrix addition into scalar addition. That's the ah-ha.

This library makes that encoding available in the fewest lines possible, with the cleanest API, so you can use it in your training loop, your analysis pipeline, or your next paper.

## License

MIT
