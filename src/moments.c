#include "free_prob.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* Compute k-th raw moment E[X^k] from empirical samples */
double compute_moment(const EmpiricalDist *dist, unsigned k) {
    if (!dist || dist->n_points == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < dist->n_points; i++) {
        double val = dist->points[i];
        double term = 1.0;
        for (unsigned j = 0; j < k; j++) term *= val;
        sum += term;
    }
    return sum / (double)dist->n_points;
}

void compute_moments(const EmpiricalDist *dist, unsigned max_k, double *output) {
    for (unsigned k = 1; k <= max_k; k++) {
        output[k - 1] = compute_moment(dist, k);
    }
}

/*
 * Free cumulant ↔ moment conversion.
 *
 * Uses the identity: m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
 * Where NC(n) = non-crossing partitions of {1,...,n}.
 *
 * Recursive formula:
 *   κ_1 = m_1
 *   κ_n = m_n - Σ_{s=1}^{n-1} κ_s * T(n, s)
 * where T(n, s) = [z^{n-s}] M̃(z)^s
 * and M̃(z) = 1 + m_1 z + m_2 z² + ... (moment generating series)
 */

void moment_to_cumulant(const double *moments, size_t n, double *cumulants) {
    if (n == 0) return;

    double m_ext[FP_MAX_ORDER + 1];
    m_ext[0] = 1.0;
    for (size_t i = 0; i < n; i++) m_ext[i + 1] = moments[i];

    double series[FP_MAX_ORDER + 1];   /* current M̃(z)^s */
    double new_series[FP_MAX_ORDER + 1];

    cumulants[0] = moments[0]; /* κ_1 = m_1 */

    for (size_t nn = 2; nn <= n; nn++) {
        /* κ_nn = m_nn - Σ_{s=1}^{nn-1} κ_s * T(nn, s)
         * T(nn, s) = [z^{nn-s}] M̃(z)^s = series[nn - s] */

        /* Initialize series = M̃(z)^1 */
        for (size_t j = 0; j < nn; j++) series[j] = m_ext[j];

        double val = moments[nn - 1]; /* m_nn */

        for (size_t s = 1; s <= nn - 1; s++) {
            double t_val = (nn >= s) ? series[nn - s] : 0.0;
            val -= cumulants[s - 1] * t_val;

            /* Update series to M̃(z)^{s+1} for next iteration */
            if (s < nn - 1) {
                memset(new_series, 0, nn * sizeof(double));
                for (size_t j = 0; j < nn; j++) {
                    for (size_t i = 0; i <= j; i++) {
                        new_series[j] += series[i] * m_ext[j - i];
                    }
                }
                memcpy(series, new_series, nn * sizeof(double));
            }
        }

        cumulants[nn - 1] = val;
    }
}

void cumulant_to_moment(const double *cumulants, size_t n, double *moments) {
    if (n == 0) return;

    double m_ext[FP_MAX_ORDER + 1];
    m_ext[0] = 1.0;

    double series[FP_MAX_ORDER + 1];
    double new_series[FP_MAX_ORDER + 1];

    for (size_t nn = 1; nn <= n; nn++) {
        if (nn == 1) {
            moments[0] = cumulants[0];
            m_ext[1] = moments[0];
            continue;
        }

        /* m_nn = κ_nn + Σ_{s=1}^{nn-1} κ_s * T(nn, s) */

        /* Initialize series = M̃(z)^1 */
        for (size_t j = 0; j < nn; j++) series[j] = m_ext[j];

        double val = cumulants[nn - 1]; /* κ_nn */

        for (size_t s = 1; s <= nn - 1; s++) {
            double t_val = (nn >= s) ? series[nn - s] : 0.0;
            val += cumulants[s - 1] * t_val;

            if (s < nn - 1) {
                memset(new_series, 0, nn * sizeof(double));
                for (size_t j = 0; j < nn; j++) {
                    for (size_t i = 0; i <= j; i++) {
                        new_series[j] += series[i] * m_ext[j - i];
                    }
                }
                memcpy(series, new_series, nn * sizeof(double));
            }
        }

        moments[nn - 1] = val;
        m_ext[nn] = val;
    }
}

bool validate_moments(const double *moments, size_t n) {
    if (n == 0) return true;

    /* Extended moment array: ext[0]=1 (m_0), ext[k]=moments[k-1] for k>=1 */
    double ext[FP_MAX_ORDER + 1];
    ext[0] = 1.0;
    for (size_t i = 0; i < n; i++) ext[i + 1] = moments[i];
    size_t ext_len = n + 1;

    /* Max Hankel size: k such that we need ext[2k-2] */
    size_t max_k = ext_len / 2 + 1;
    if (max_k > FP_MAX_ORDER / 2) max_k = FP_MAX_ORDER / 2;

    for (size_t k = 1; k <= max_k && 2 * k - 2 < ext_len; k++) {
        /* Build k×k Hankel matrix and compute determinant */
        double mat[FP_MAX_ORDER / 2][FP_MAX_ORDER / 2];

        for (size_t i = 0; i < k; i++)
            for (size_t j = 0; j < k; j++)
                mat[i][j] = (i + j < ext_len) ? ext[i + j] : 0.0;

        double det = 1.0;
        for (size_t i = 0; i < k; i++) {
            double pivot = mat[i][i];
            if (fabs(pivot) < 1e-12) {
                bool found = false;
                for (size_t j = i + 1; j < k; j++) {
                    if (fabs(mat[j][i]) > 1e-12) {
                        for (size_t l = 0; l < k; l++) {
                            double tmp = mat[i][l];
                            mat[i][l] = mat[j][l];
                            mat[j][l] = -tmp;
                        }
                        pivot = mat[i][i];
                        found = true;
                        break;
                    }
                }
                if (!found) { det = 0.0; break; }
            }
            det *= pivot;
            for (size_t j = i + 1; j < k; j++) {
                double factor = mat[j][i] / pivot;
                for (size_t l = i; l < k; l++)
                    mat[j][l] -= factor * mat[i][l];
            }
        }

        if (det < -1e-10) return false;
    }

    return true;
}
