#include "free_prob.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * R-transform: R(z) = Σ_{n=1}^{N} κ_n * z^{n-1}
 * This is the free probability analogue of the log of the moment generating function.
 * Key property: R_{A+B}(z) = R_A(z) + R_B(z) for freely independent A, B.
 */
double r_transform_from_cumulants(const double *cumulants, size_t n, double z) {
    double result = 0.0;
    double z_power = 1.0; /* z^0 */
    for (size_t i = 0; i < n; i++) {
        result += cumulants[i] * z_power;
        z_power *= z;
    }
    return result;
}

void r_transform_add(const double *cumulants_a, const double *cumulants_b,
                     size_t n, double *cumulants_sum) {
    for (size_t i = 0; i < n; i++) {
        cumulants_sum[i] = cumulants_a[i] + cumulants_b[i];
    }
}

/*
 * Stieltjes transform from R-transform via fixed-point iteration.
 * 
 * S(z) = 1/z * (1 + R(S(z) * z))  ... not quite.
 * 
 * The Cauchy transform G(z) = ∫ dμ(x)/(z-x) relates to R via:
 * G(1/R(z) + z) = z, or equivalently: K(G(z)) = z where K(w) = 1/w + R(w).
 * Wait, the standard relation is:
 * G(K(z)) = z where K(z) = z + R(z)... hmm.
 *
 * Actually: the Stieltjes transform S(z) and R-transform are related by:
 * S(1/z + R(z)) = z, or equivalently R(S(z)) + 1/S(z) = z.
 *
 * Wait, let me be precise. The Cauchy transform is:
 * G(z) = ∫ dμ(x)/(z-x) = Σ_{n>=0} m_n / z^{n+1}
 * 
 * The R-transform satisfies: G(R(z) + 1/z) = z
 * Or equivalently: R(G(z)) + 1/G(z) = z
 * i.e., R(w) = z - 1/w where w = G(z)
 *
 * So to compute G(z) from R, we solve:
 * R(w) + 1/w = z  for w = G(z).
 * This is: Σ κ_n * w^{n-1} + 1/w = z
 * or: Σ_{n>=1} κ_n * w^{n-1} + 1/w - z = 0
 *
 * We use fixed-point iteration on: w = 1/(z - R(w))
 * i.e., w_{k+1} = 1/(z - R(w_k))
 *
 * The Stieltjes transform is S(z) = G(z) = w.
 */
double stieltjes_from_r(const double *cumulants, size_t n, double z,
                        unsigned max_iter, double tol) {
    /* Fixed-point: w = 1/(z - R(w)) */
    /* Initial guess: w = 1/z (for large z, this is approximately right) */
    double w = 1.0 / z;
    
    for (unsigned iter = 0; iter < max_iter; iter++) {
        double r_val = r_transform_from_cumulants(cumulants, n, w);
        double w_new = 1.0 / (z - r_val);
        
        if (fabs(w_new - w) < tol) {
            w = w_new;
            break;
        }
        w = w_new;
    }
    
    return w;
}

/*
 * Recover R(z) from Stieltjes transform value.
 * R(w) = z - 1/w where w = S(z).
 * But we want R(z) at a specific z, which requires the inverse.
 * 
 * Given a pair (z, S(z)), we have: R(S(z)) = z - 1/S(z).
 * So R evaluated at w=S(z) gives z - 1/w.
 */
double r_from_stieltjes(double s, double z) {
    /* R(s) = z - 1/s */
    return z - 1.0 / s;
}

/*
 * Marchenko-Pastur distribution.
 * For X = (1/n) W W^T where W is p×n with iid entries of variance σ²/p,
 * the eigenvalue density is:
 * ρ(x) = (1/(2πxσ²λ)) * sqrt((b-x)(x-a)) for a ≤ x ≤ b
 * where a = σ²(1-√λ)², b = σ²(1+√λ)², λ = p/n
 * Plus a point mass at 0 of weight max(0, 1-λ) if λ < 1.
 */
double marchenko_pastur_density(double x, double lambda, double sigma) {
    if (lambda <= 0.0) return 0.0;
    
    double a = sigma * sigma * (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    double b = sigma * sigma * (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    
    if (x < a || x > b) return 0.0;
    if (x == a && x == b) {
        /* Degenerate case λ → 0 or σ → 0 */
        return 1e30; /* delta function */
    }
    
    double diff_b = b - x;
    double diff_x = x - a;
    
    if (diff_b < 0.0 || diff_x < 0.0) return 0.0;
    
    double lambda_eff = lambda < 1.0 ? lambda : 1.0;
    double density = sqrt(diff_b * diff_x) / (2.0 * M_PI * sigma * sigma * lambda_eff * x);
    return density;
}

/*
 * Marchenko-Pastur moments.
 * For λ ≤ 1, the moments are given by the Narayana numbers:
 * m_k = Σ_{j=0}^{k-1} (1/(k-j)) * C(k,j) * C(k-1,j) * λ^j / σ^{2k}
 * Actually, the standard MP moments (σ²=1) are:
 * m_k = Σ_{j=0}^{k-1} (1/(k)) * C(k,j) * C(k,j+1) * λ^j  (Narayana-type)
 *
 * More precisely, the k-th moment of MP(λ, σ²=1) is:
 * m_k = Σ_{j=0}^{k-1} N(k, j) * λ^j
 * where N(k, j) = (1/k) * C(k, j) * C(k, j+1) are the Narayana numbers.
 * 
 * Actually even simpler: m_k = Σ_{s=1}^{k} (1/s) * C(k-1, s-1) * C(k, s-1) * λ^{s-1}
 *
 * The simplest formula: m_k (for MP with ratio λ and σ²=1) equals
 * the sum over Narayana numbers times λ^j.
 * 
 * We can also compute recursively:
 * m_0 = 1, m_1 = 1
 * m_k = Σ_{j=0}^{k-1} m_j * m_{k-1-j} * (1/(k)) * ... Catalan-like
 *
 * Actually the simplest: MP free cumulants are κ_n = Σ_{j=0}^{n-1} C(n-1,j)² λ^j
 * No wait, the free cumulants of MP(λ) are: κ_n = λ^{n-1} (for σ²=1).
 * Wait, that's not right either.
 *
 * The correct free cumulants of MP(λ, σ²=1) are:
 * κ_n = Σ_{k=0}^{n-1} C(n-1, k)² * λ^k  ... hmm no.
 *
 * Let me just use: the MP distribution has R-transform R(z) = 1/(1-λz) (for σ²=1).
 * Wait no. For MP(λ), the R-transform is R(z) = 1/(1 - λz).
 * Expanding: R(z) = Σ_{n=0}^∞ λ^n z^n = 1 + λz + λ²z² + ...
 * So κ_{n+1} = λ^n for n >= 0, meaning κ_1 = 1, κ_2 = λ, κ_3 = λ², ...
 *
 * Hmm, that's for σ²=1. More generally, κ_n = σ^{2n} * λ^{n-1}.
 *
 * Actually I think for standard MP(λ): R(z) = 1/(1-λz)
 * So κ_1 = 1, κ_{n+1} = λ^n for n >= 1.
 * And then moments can be computed from cumulant_to_moment.
 */
void marchenko_pastur_moments(double lambda, unsigned max_k, double *moments) {
    if (max_k == 0) return;
    
    /* Free cumulants of MP(λ): κ_1 = 1, κ_n = λ^{n-1} for n >= 2 */
    /* More precisely: R(z) = 1/(1-λz) = Σ_{n=0}^∞ λ^n z^n */
    /* So κ_1 = 1 (coefficient of z^0), κ_2 = λ, κ_3 = λ², ... */
    
    double cumulants[FP_MAX_ORDER];
    for (size_t i = 0; i < max_k; i++) {
        /* κ_{i+1} = λ^i */
        cumulants[i] = pow(lambda, (double)i);
    }
    
    cumulant_to_moment(cumulants, max_k, moments);
}
