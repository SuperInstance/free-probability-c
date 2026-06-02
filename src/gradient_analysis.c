#include "free_prob.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/*
 * Transformer gradient analysis using free probability.
 * 
 * Key insight: the eigenvalue distribution of gradient covariance matrices
 * in deep transformers can be predicted using free probability theory.
 * When layers are "freely independent" (a good approximation for randomly
 * initialized deep networks), we can use R-transforms and S-transforms
 * to predict combined distributions without computing the actual matrices.
 */

void predict_combined_distribution(const double *layer_cumulants,
                                   unsigned n_layers,
                                   size_t cumulant_order,
                                   double *combined_cumulants) {
    /*
     * When combining n_layers freely independent layers,
     * R_{combined}(z) = Σ_i R_i(z)
     * 
     * If all layers have the same eigenvalue distribution (shared cumulants),
     * then κ_combined = n_layers * κ_single.
     */
    for (size_t k = 0; k < cumulant_order; k++) {
        combined_cumulants[k] = (double)n_layers * layer_cumulants[k];
    }
}

RegularizerSuggestion suggest_regularizer(const double *moments, size_t n,
                                          const TransformerConfig *config) {
    RegularizerSuggestion result = {0};
    
    if (n == 0 || !config) {
        result.lambda_reg = 1e-4;
        result.spectral_radius = 1.0;
        result.outlier_fraction = 0.0;
        result.stability_score = 1.0;
        return result;
    }
    
    /* Compute free cumulants to understand eigenvalue structure */
    double cumulants[FP_MAX_ORDER];
    moment_to_cumulant(moments, n, cumulants);
    
    /* Spectral radius ≈ largest eigenvalue ≈ sqrt(m_2) for centered distribution */
    double variance = (n >= 2) ? moments[1] : 1.0;
    double mean = (n >= 1) ? moments[0] : 0.0;
    double centered_var = variance - mean * mean;
    if (centered_var < 0.0) centered_var = 0.0;
    
    result.spectral_radius = sqrt(centered_var) * 3.0; /* 3-sigma */
    
    /* Compare with Marchenko-Pastur to detect outliers */
    double lambda = config->hidden_dim / (config->n_samples > 0 ? config->n_samples : 1.0);
    if (lambda > 4.0) lambda = 4.0;
    if (lambda < 0.01) lambda = 0.01;
    
    double mp_b = (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda)); /* upper edge of MP */
    double ratio = result.spectral_radius / (sqrt(mp_b) > 0 ? sqrt(mp_b) : 1.0);
    
    result.outlier_fraction = (ratio > 1.0) ? (ratio - 1.0) / ratio : 0.0;
    if (result.outlier_fraction > 1.0) result.outlier_fraction = 1.0;
    
    /* Suggest regularization strength based on tail behavior.
     * Stronger regularization when eigenvalues have heavy tails. */
    double kurtosis = 1.0;
    if (n >= 4 && centered_var > 1e-12) {
        double m4 = moments[3];
        double m2 = centered_var;
        kurtosis = m4 / (m2 * m2);
    }
    
    /* Higher kurtosis → heavier tails → more regularization */
    double kurtosis_excess = kurtosis - 3.0; /* excess over Gaussian */
    if (kurtosis_excess < 0.0) kurtosis_excess = 0.0;
    
    result.lambda_reg = 1e-4 * (1.0 + kurtosis_excess * 0.5) * (1.0 + result.outlier_fraction);
    
    /* Scale by inverse of layers (deeper = more careful) */
    result.lambda_reg *= sqrt((double)config->n_layers);
    
    /* Stability score: higher when eigenvalue distribution is well-behaved */
    double cond_estimate = (centered_var > 1e-12) ? result.spectral_radius / sqrt(centered_var) : 1.0;
    (void)cond_estimate;
    result.stability_score = 1.0 / (1.0 + 0.1 * kurtosis_excess + result.outlier_fraction);
    if (result.stability_score > 1.0) result.stability_score = 1.0;
    if (result.stability_score < 0.0) result.stability_score = 0.0;
    
    return result;
}

InitSuggestion suggest_initialization(const TransformerConfig *config) {
    InitSuggestion result = {0};
    
    if (!config) {
        result.suggested_std = 0.02;
        result.suggested_lr_scale = 1.0;
        result.condition_number = 1.0;
        result.tail_mass = 0.0;
        return result;
    }
    
    /*
     * From Marchenko-Pastur theory:
     * For a random weight matrix W of size d_in × d_out with entries of std σ,
     * the eigenvalue distribution of W^T W / d_in has:
     * - Upper edge: b = σ² * (1 + √(d_out/d_in))²
     * - Lower edge: a = σ² * (1 - √(d_out/d_in))²
     * 
     * For stable training, we want:
     * 1. The spectral norm (≈ √b) to be O(1) → σ ~ 1/√d
     * 2. The condition number √(b/a) to be manageable
     * 
     * Xavier/He initialization: σ = √(2/(d_in + d_out)) or σ = √(2/d_in)
     * From MP theory: σ = 1/√d keeps the bulk eigenvalues in [0, 4] for square matrices.
     */
    
    double d = config->hidden_dim;
    double n = config->n_samples > 0 ? config->n_samples : d;
    
    /* MP ratio */
    double lambda = d / n;
    
    /* Standard initialization: 1/√d for unit spectral norm */
    double base_std = 1.0 / sqrt(d);
    
    /* Adjust for depth: deeper networks need smaller init.
     * Heuristic: divide by √(depth) to maintain signal propagation. */
    result.suggested_std = base_std / sqrt((double)config->n_layers);
    
    /* Ensure reasonable range */
    if (result.suggested_std > 1.0) result.suggested_std = 1.0;
    if (result.suggested_std < 1e-6) result.suggested_std = 1e-6;
    
    /* MP upper edge with suggested init */
    double mp_upper = result.suggested_std * result.suggested_std * d *
                      (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    double mp_lower = result.suggested_std * result.suggested_std * d *
                      (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    
    if (mp_lower > 1e-12) {
        result.condition_number = mp_upper / mp_lower;
    } else {
        result.condition_number = mp_upper / 1e-12;
    }
    
    /* Learning rate should be inversely proportional to spectral radius */
    result.suggested_lr_scale = 1.0 / (1.0 + 0.1 * sqrt(mp_upper));
    if (result.suggested_lr_scale > 1.0) result.suggested_lr_scale = 1.0;
    if (result.suggested_lr_scale < 0.01) result.suggested_lr_scale = 0.01;
    
    /* Tail mass: fraction of eigenvalues outside [0, 4σ²d] for MP */
    /* For MP, the support is bounded, so tail mass is 0 from MP theory.
     * But in practice, deviations from MP create tails. */
    result.tail_mass = 0.05; /* heuristic default */
    
    return result;
}
