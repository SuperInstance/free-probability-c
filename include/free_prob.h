#ifndef FREE_PROB_H
#define FREE_PROB_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of moments/cumulants supported */
#define FP_MAX_ORDER 64

/* ---------- Moments ---------- */

typedef struct {
    double support_min;   /* left edge of support */
    double support_max;   /* right edge of support */
    size_t n_points;      /* number of sample points */
    const double *points; /* array of sample points */
} EmpiricalDist;

/* Compute k-th raw moment of an empirical distribution (1-indexed: order=1 → mean) */
double compute_moment(const EmpiricalDist *dist, unsigned k);

/* Compute moments up to order max_k (output[0]=m1, output[1]=m2, ...) */
void compute_moments(const EmpiricalDist *dist, unsigned max_k, double *output);

/* Convert moment sequence to free cumulant sequence (moment-to-cumulant transform).
   moments[0..n-1] = m1..mn, cumulants[0..n-1] = κ1..κn. */
void moment_to_cumulant(const double *moments, size_t n, double *cumulants);

/* Convert free cumulant sequence to moment sequence (inverse). */
void cumulant_to_moment(const double *cumulants, size_t n, double *moments);

/* Validate that moments correspond to a valid probability distribution:
   - Hankel matrices of moments must be positive semi-definite */
bool validate_moments(const double *moments, size_t n);

/* ---------- R-transform ---------- */

/* Evaluate R(z) = Σ_{n=1..N} κ_n * z^{n-1} from free cumulants */
double r_transform_from_cumulants(const double *cumulants, size_t n, double z);

/* Free additivity: R_{A+B}(z) = R_A(z) + R_B(z).
   Adds two cumulant sequences; result_k = κ_k^{(A)} + κ_k^{(B)}.
   n is the max order; output must have space for n elements. */
void r_transform_add(const double *cumulants_a, const double *cumulants_b,
                     size_t n, double *cumulants_sum);

/* Compute Stieltjes transform S(z) = 1/z * (1 + R(S(z))) iteratively.
   Uses fixed-point iteration to solve S = (1 + R(S))/z. */
double stieltjes_from_r(const double *cumulants, size_t n, double z,
                        unsigned max_iter, double tol);

/* Inverse: recover R-transform from Stieltjes transform samples.
   Given S(z), compute R(z) = -S^{-1}(S(z)) - 1/S(z) ... simplified via:
   R(z) ≈ G^{-1}(z) - 1/z where G is Cauchy transform = -S. */
double r_from_stieltjes(double s, double z);

/* ---------- Marchenko-Pastur ---------- */

typedef struct {
    double ratio;       /* λ = p/n (features/samples ratio) */
    double sigma;       /* variance scale (default 1.0) */
} MPParams;

/* Compute Marchenko-Pastur density at point x for given ratio λ. */
double marchenko_pastur_density(double x, double lambda, double sigma);

/* Compute Marchenko-Pastur moments up to order max_k.
   For λ ≤ 1: closed-form Catalan-like formulas. */
void marchenko_pastur_moments(double lambda, unsigned max_k, double *moments);

/* ---------- S-transform ---------- */

/* Compute S-transform from moments.
   S(z) = (1+z) / (z * χ(z)) where χ is the inverse Cauchy transform.
   Implemented via moment series expansion. */
void s_transform_from_moments(const double *moments, size_t n,
                              double *s_coeffs);

/* Free multiplicativity: S_{AB}(z) = S_A(z) * S_B(z).
   Multiplies two S-transform coefficient sequences (polynomial multiplication).
   sa, sb have n coefficients each; output has 2n-1 elements. */
void s_transform_multiply(const double *sa, const double *sb,
                          size_t n, double *product);

/* ---------- Gradient Analysis ---------- */

typedef struct {
    unsigned n_layers;          /* number of transformer layers */
    double hidden_dim;          /* hidden dimension */
    double n_samples;           /* number of training samples */
    double weight_std;          /* weight initialization std */
    double learning_rate;       /* current learning rate */
} TransformerConfig;

typedef struct {
    double suggested_std;       /* suggested weight init std */
    double suggested_lr_scale;  /* suggested lr relative to base */
    double condition_number;    /* predicted condition number */
    double tail_mass;           /* fraction of eigenvalues in tail */
} InitSuggestion;

typedef struct {
    double lambda_reg;          /* suggested L2 regularization strength */
    double spectral_radius;    /* predicted spectral radius of grad cov */
    double outlier_fraction;    /* fraction of eigenvalues outside MP */
    double stability_score;     /* 0-1, higher = more stable */
} RegularizerSuggestion;

/* Predict eigenvalue distribution when combining layers via R-transform. */
void predict_combined_distribution(const double *layer_cumulants,
                                   unsigned n_layers,
                                   size_t cumulant_order,
                                   double *combined_cumulants);

/* Suggest regularization from eigenvalue tail behavior. */
RegularizerSuggestion suggest_regularizer(const double *moments, size_t n,
                                          const TransformerConfig *config);

/* Suggest weight initialization scale from Marchenko-Pastur. */
InitSuggestion suggest_initialization(const TransformerConfig *config);

#ifdef __cplusplus
}
#endif

#endif /* FREE_PROB_H */
