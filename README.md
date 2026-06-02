# free-probability-c

**Free probability calculus for transformer gradient dynamics.**

Free probability theory provides exact formulas for the eigenvalue distributions of sums and products of large random matrices — without computing the matrices themselves. This library implements the core transforms needed to analyze and predict gradient covariance structure in deep transformers.

## Why This Matters for Transformers

When training large transformers, the eigenvalue distribution of gradient covariance matrices determines:
- **Training stability** — outlier eigenvalues cause exploding/vanishing gradients
- **Optimal learning rate** — depends on spectral radius of the gradient covariance
- **Regularization strength** — should match tail behavior of eigenvalue distribution
- **Weight initialization** — Marchenko-Pastur law gives the exact baseline

The key insight from free probability: **random matrices from different layers are approximately freely independent**. This means:

1. **R-transform additivity**: `R_{A+B}(z) = R_A(z) + R_B(z)` — eigenvalue distribution of combined layers is computable from individual layers
2. **S-transform multiplicativity**: `S_{AB}(z) = S_A(z) · S_B(z)` — eigenvalue distribution of matrix products is computable
3. **Marchenko-Pastur law** gives the exact eigenvalue density of random covariance matrices — the "null hypothesis" for initialized networks

## API Overview

### Moments (`src/moments.c`)
- `compute_moments()` — empirical moments from samples
- `moment_to_cumulant()` — free cumulant sequence from moment sequence
- `cumulant_to_moment()` — inverse transform
- `validate_moments()` — check Hankel positivity (valid distribution)

### R-transform (`src/r_transform.c`)
- `r_transform_from_cumulants()` — evaluate R(z) = Σ κ_{n+1} z^n
- `r_transform_add()` — **free additivity**: R_{A+B} = R_A + R_B
- `stieltjes_from_r()` — Stieltjes transform via fixed-point iteration
- `marchenko_pastur_density()` — MP law eigenvalue density
- `marchenko_pastur_moments()` — exact MP moments (Catalan numbers)

### S-transform (`src/s_transform.c`)
- `s_transform_from_moments()` — compute S-transform coefficients
- `s_transform_multiply()` — **free multiplicativity**: S_{AB} = S_A · S_B

### Gradient Analysis (`src/gradient_analysis.c`)
- `predict_combined_distribution()` — predict eigenvalues when combining layers
- `suggest_regularizer()` — regularization from eigenvalue tail behavior
- `suggest_initialization()` — init scale from Marchenko-Pastur theory

## Building

```bash
make          # build static library
make test     # build and run tests
make clean    # clean artifacts
```

## Dependencies

- C11 compiler
- libm (math library)

## License

MIT
