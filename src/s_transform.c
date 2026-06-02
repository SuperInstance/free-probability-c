#include "free_prob.h"
#include <math.h>
#include <string.h>

/*
 * S-transform: the multiplicative analogue of the R-transform.
 * For freely independent A, B: S_{AB}(z) = S_A(z) * S_B(z)
 *
 * The S-transform is defined as:
 * S(z) = (1+z) / (z * χ(z))
 * where χ(z) is the inverse of the moment generating function ψ(z) = z*M(z) - 1.
 * More precisely, ψ(χ(z)) = z, and S(z) = (1+z)/(z * χ(z)).
 *
 * Practical computation via moment expansion:
 * S(z) = Σ s_n z^n can be computed from moments.
 *
 * For a distribution with moments m_1, m_2, ..., we first compute the
 * ψ-transform: ψ(z) = Σ_{n>=1} m_n z^n, then find its inverse χ,
 * then compute S.
 */

/*
 * Compute S-transform coefficients from moments.
 * 
 * The S-transform series: S(z) = Σ_{n>=0} s_n z^n
 * 
 * Algorithm:
 * 1. Build ψ(z) = Σ_{n>=1} m_n z^n (moment generating function minus 1)
 * 2. Compute χ(z) = ψ^{-1}(z) (compositional inverse) via series reversion
 * 3. S(z) = (1+z) / (z * χ(z))
 *    But since we work with formal power series:
 *    χ(z) = m_1 z + (m_2 - m_1²/m_1) z² + ...
 *    z * χ(z) = m_1 z² + ...
 *    (1+z)/(z*χ(z)) needs careful handling.
 *
 * Alternative formula:
 * The S-transform can be computed as:
 * S(z) = (1+z) * Σ_{n>=0} c_n z^n
 * where c_n are coefficients of 1/(z*χ(z)).
 *
 * Actually, let's use the direct moment-based formula.
 * Given free cumulants κ_n, the S-transform is:
 * S(z) = 1/κ_1 * Π_{n>=1} (1 + z * something)^... 
 * 
 * The cleanest approach: from the moment generating function.
 * Define ψ(z) = Σ_{n>=1} m_n z^n
 * Let χ be the compositional inverse: ψ(χ(z)) = z
 * Then S(z) = (1+z) / (z * χ(z))
 *
 * Step 1: Series reversion to get χ from ψ
 * If ψ(z) = a_1 z + a_2 z² + ... with a_1 ≠ 0,
 * then χ(z) = b_1 z + b_2 z² + ... where ψ(χ(z)) = z.
 *
 * Using Lagrange inversion:
 * b_1 = 1/a_1
 * b_n = (1/a_1^n) * (1/(n-1)!) * [d^{n-1}/dz^{n-1}] (z/ψ(z))^n |_{z=0}
 * 
 * For numerical computation, we use the recursive formula:
 * [z^n] χ(z) = (1/(n * a_1)) * [z^{n-1}] (z/ψ(z))^n... 
 *
 * Actually, let's use the practical algorithm via power series composition.
 * We need χ such that ψ(χ(z)) = z.
 * 
 * Given ψ(z) = Σ_{k>=1} a_k z^k, χ(z) = Σ_{k>=1} b_k z^k:
 * 
 * From ψ(χ(z)) = z:
 * a_1 b_1 = 1 → b_1 = 1/a_1
 * a_1 b_2 + a_2 b_1² = 0 → b_2 = -a_2 b_1² / a_1 = -a_2 / a_1³
 * etc.
 * 
 * General: [z^n] ψ(χ(z)) = Σ_{k>=1} a_k * [z^n] (χ(z))^k
 * [z^n] (χ(z))^k = Σ over compositions of n into k parts of Π b_{i_j}
 *
 * This can be computed using the same DP approach as in moments.c.
 */
void s_transform_from_moments(const double *moments, size_t n, double *s_coeffs) {
    if (n == 0) return;
    
    /* Step 1: Build ψ(z) coefficients: psi[k] = m_{k+1} = moments[k] */
    double psi[FP_MAX_ORDER + 1];
    for (size_t i = 0; i < n; i++) psi[i] = moments[i];
    
    /* Step 2: Compositional inverse χ via series reversion.
     * ψ(χ(z)) = z
     * χ(z) = Σ b_k z^k
     */
    double chi[FP_MAX_ORDER + 1];
    memset(chi, 0, sizeof(chi));
    
    if (fabs(psi[0]) < 1e-15) {
        /* Degenerate: zero mean */
        for (size_t i = 0; i <= n; i++) s_coeffs[i] = 0.0;
        return;
    }
    
    chi[0] = 1.0 / psi[0]; /* b_1 = 1/a_1 */
    
    /* Compute χ using series reversion via composition constraint:
     * For each order j from 2 to n:
     * [z^j] ψ(χ(z)) = 0 for j >= 2 (must equal [z^j] z = δ_{j,1})
     * 
     * ψ(χ(z)) = Σ_{k>=1} psi[k-1] * (χ(z))^k
     * [z^j] ψ(χ(z)) = Σ_{k=1}^{j} psi[k-1] * [z^j] χ^k
     * 
     * For j >= 2: 0 = Σ_{k=1}^{j} psi[k-1] * [z^j] χ^k
     * psi[0] * [z^j] χ^1 = -Σ_{k=2}^{j} psi[k-1] * [z^j] χ^k
     * But [z^j] χ^1 = chi[j-1] (since χ is 1-indexed in theory but 0-indexed here)
     * Actually: χ(z) = chi[0] z + chi[1] z² + ... (chi[k] = b_{k+1})
     * So [z^j] χ^1 = chi[j-1]
     */
    
    /* Power series for χ^k: chi_pow[k][j] = [z^j] χ(z)^k */
    /* chi_pow[1] = chi itself */
    double chi_pow_prev[FP_MAX_ORDER + 1];
    double chi_pow_new[FP_MAX_ORDER + 1];
    
    /* Initialize chi_pow_prev = χ^1 */
    memcpy(chi_pow_prev, chi, (n + 1) * sizeof(double));
    
    for (size_t j = 2; j <= n; j++) {
        /* Compute chi[j-1] (i.e., b_j) from the constraint:
         * 0 = Σ_{k=1}^{j} psi[k-1] * [z^j] χ^k
         * 
         * [z^j] χ^1 = chi[j-1] (the unknown)
         * [z^j] χ^k for k >= 2 can be computed from chi[0..j-2]
         */
        
        double rhs = 0.0;
        
        /* Update chi_pow to χ^2, χ^3, ... as needed */
        /* Start fresh for each j */
        
        /* chi_pow_prev holds χ^1 */
        memcpy(chi_pow_prev, chi, (n + 1) * sizeof(double));
        
        for (size_t k = 2; k <= j; k++) {
            /* χ^k = χ^{k-1} * χ */
            memset(chi_pow_new, 0, (n + 1) * sizeof(double));
            for (size_t a = 0; a < j; a++) {
                if (chi_pow_prev[a] == 0.0) continue;
                for (size_t b = 0; b < j - a; b++) {
                    chi_pow_new[a + b] += chi_pow_prev[a] * chi[b];
                }
            }
            rhs += psi[k - 1] * chi_pow_new[j - 1];
            memcpy(chi_pow_prev, chi_pow_new, (n + 1) * sizeof(double));
        }
        
        chi[j - 1] = -rhs / psi[0];
    }
    
    /* Step 3: Compute S(z) = (1+z) / (z * χ(z))
     * 
     * z * χ(z) = Σ_{k>=1} chi[k-1] z^k = chi[0] z + chi[1] z² + ...
     * 
     * 1/(z*χ(z)): we need the formal inverse of the series z*χ(z).
     * Let f(z) = chi[0] z + chi[1] z² + ... = z * χ(z)
     * f(z) has no constant term and starts at z^1.
     * 1/f(z) = (1/chi[0]) z^{-1} * 1/(1 + (chi[1]/chi[0]) z + ...)
     * 
     * Actually: S(z) = (1+z)/(z*χ(z))
     * 
     * z*χ(z) = Σ_{k>=1} chi[k-1] z^k
     * Let g(z) = Σ_{k>=0} chi[k] z^{k+1}
     * 
     * We want h(z) = 1/g(z). Since g(z) = z * χ(z) and χ(z) is a power series
     * starting at z^1, g(z) starts at chi[0]*z.
     * 
     * 1/g(z) has a z^{-1} term. Let's work with z*χ(z) properly.
     * 
     * Alternative: define φ(z) = χ(z)/z = chi[0] + chi[1]*z + ...
     * Wait, χ(z) = chi[0]*z + chi[1]*z² + ...
     * So χ(z)/z = chi[0] + chi[1]*z + chi[2]*z² + ... which is a regular series.
     * 
     * Then z*χ(z) = z² * (χ(z)/z) = z² * φ(z)
     * And S(z) = (1+z) / (z² * φ(z)) ... that still has z² in denominator.
     * 
     * Hmm, let me reconsider. The standard definition of S-transform uses:
     * χ(w) is the inverse of ψ(w) = w*M(w) - 1 where M is the moment generating function.
     * 
     * Wait, there are different conventions. Let me use the moment-based approach.
     * 
     * Define ψ(z) = z * M(z) - 1 where M(z) = 1 + m_1 z + m_2 z² + ...
     * So ψ(z) = m_1 z + m_2 z² + ... (this is what we have as psi above)
     * 
     * χ is the compositional inverse: ψ(χ(z)) = z.
     * 
     * S(z) = (1+z)/(z * χ(z))
     * 
     * Here χ(z) = chi[0]*z + chi[1]*z² + ...
     * So z * χ(z) = chi[0]*z² + chi[1]*z³ + ...
     * 
     * And (1+z) / (z*χ(z)): for this to be a power series, we need
     * (1+z) / (chi[0]*z² + ...) which has a z^{-2} singularity.
     * 
     * The S-transform is NOT a power series at z=0. It's typically expressed as:
     * S(z) = 1/(χ(z)) * (1+z)/z = (1+z)/(z*χ(z))
     * 
     * With χ(z) = chi[0]*z + chi[1]*z² + ...
     * S(z) = (1+z) / (chi[0]*z² + chi[1]*z³ + ...)
     * 
     * Hmm, this has a pole at 0. The S-transform is typically a series in z
     * that converges near 0 only if we consider it as:
     * S(z) = (1+z)/(z * χ(z))
     * 
     * With χ(z) starting at order z, z*χ(z) starts at z².
     * So S(z) ~ (1)/(chi[0]*z) near z=0, which diverges.
     * 
     * Actually I think the convention is different. Let me look at this more carefully.
     * 
     * The standard definition (e.g., Nica-Speicher):
     * ψ_μ(z) = Σ_{n>=1} m_n z^n
     * χ_μ = ψ_μ^{-1} (compositional inverse)
     * S_μ(z) = (1+z) / (z * χ_μ^{-1}(z))
     * 
     * Wait, there might be a typo in my reference. Let me use:
     * S_μ(z) = (1+z)/(z) * χ_μ(z)
     * where χ_μ satisfies ψ_μ(χ_μ(z)) = z.
     * 
     * So S(z) = (1+z)/z * χ(z) = (1+z) * χ(z) / z
     * 
     * With χ(z) = chi[0]*z + chi[1]*z² + ...
     * χ(z)/z = chi[0] + chi[1]*z + chi[2]*z² + ...
     * 
     * So S(z) = (1+z) * (chi[0] + chi[1]*z + chi[2]*z² + ...)
     *         = chi[0] + (chi[0] + chi[1])*z + (chi[1] + chi[2])*z² + ...
     * 
     * YES! This is a proper power series.
     */
    
    /* Compute S(z) = (1+z) * (χ(z)/z)
     * χ(z)/z = Σ_{k>=0} chi[k] z^k
     * S(z) = (1+z) * Σ_{k>=0} chi[k] z^k
     *       = Σ_{k>=0} chi[k] z^k + Σ_{k>=0} chi[k] z^{k+1}
     *       = chi[0] + Σ_{k>=1} (chi[k] + chi[k-1]) z^k
     */
    
    s_coeffs[0] = chi[0];
    for (size_t i = 1; i < n; i++) {
        s_coeffs[i] = chi[i] + chi[i - 1];
    }
}

void s_transform_multiply(const double *sa, const double *sb,
                          size_t n, double *product) {
    /* Polynomial multiplication: (Σ sa_i z^i) * (Σ sb_i z^i) */
    size_t result_len = 2 * n - 1;
    for (size_t i = 0; i < result_len; i++) {
        product[i] = 0.0;
    }
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            product[i + j] += sa[i] * sb[j];
        }
    }
}
