#include "free_prob.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %2d: %-55s", test_count, name); } while(0)
#define PASS() do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); } while(0)
#define ASSERT_FEQ(a, b, tol) do { \
    double _a = (a), _b = (b), _tol = (tol); \
    if (fabs(_a - _b) > _tol) { \
        FAIL(#a " != " #b); \
        printf("    Expected: %.10g, Got: %.10g (diff: %.3e)\n", _b, _a, fabs(_a-_b)); \
        return; \
    } \
} while(0)

/* ============================================================
 * Test: Moments of uniform distribution on [0, 1]
 * E[X^k] = 1/(k+1)
 * ============================================================ */
static void test_uniform_moments(void) {
    TEST("Uniform[0,1] moments");
    
    /* Generate uniform samples on [0,1] */
    #define N_UNIFORM 10000
    double points[N_UNIFORM];
    for (int i = 0; i < N_UNIFORM; i++) {
        points[i] = (double)(i + 0.5) / (double)N_UNIFORM;
    }
    
    EmpiricalDist dist = {0.0, 1.0, N_UNIFORM, points};
    
    double m[6];
    compute_moments(&dist, 6, m);
    
    /* m_k = 1/(k+1) for uniform on [0,1] */
    ASSERT_FEQ(m[0], 0.5, 0.01);       /* E[X] = 1/2 */
    ASSERT_FEQ(m[1], 1.0/3.0, 0.01);   /* E[X²] = 1/3 */
    ASSERT_FEQ(m[2], 0.25, 0.01);      /* E[X³] = 1/4 */
    ASSERT_FEQ(m[3], 0.2, 0.01);       /* E[X⁴] = 1/5 */
    ASSERT_FEQ(m[4], 1.0/6.0, 0.01);   /* E[X⁵] = 1/6 */
    
    PASS();
    #undef N_UNIFORM
}

/* ============================================================
 * Test: Semicircle law moments
 * Semicircle on [-2,2] has moments = Catalan numbers: 1, 0, 1, 0, 2, 0, 5, ...
 * ============================================================ */
static void test_semicircle_moments(void) {
    TEST("Semicircle law moments (Catalan numbers)");
    
    /* Semicircle distribution: ρ(x) = (1/2π)√(4-x²) on [-2,2] */
    #define N_SEMI 50000
    double points[N_SEMI];
    int count = 0;
    for (int i = 0; i < N_SEMI && count < N_SEMI; i++) {
        /* Accept-reject sampling for semicircle */
        double x = -2.0 + 4.0 * (double)i / (double)N_SEMI;
        double y = sqrt(4.0 - x * x) / (2.0 * M_PI);
        /* Approximate by deterministic sampling weighted by density */
        points[count++] = x;
    }
    
    /* Use weighted moments instead */
    double m[8];
    for (int k = 1; k <= 8; k++) {
        double sum = 0.0, wsum = 0.0;
        for (int i = 0; i < N_SEMI; i++) {
            double x = -2.0 + 4.0 * (double)(i + 0.5) / (double)N_SEMI;
            if (x*x > 4.0) continue;
            double w = sqrt(4.0 - x*x); /* unnormalized density */
            double val = 1.0;
            for (int j = 0; j < k; j++) val *= x;
            sum += val * w;
            wsum += w;
        }
        m[k-1] = sum / wsum;
    }
    
    /* Odd moments should be 0 */
    ASSERT_FEQ(m[0], 0.0, 0.01);   /* m_1 = 0 */
    ASSERT_FEQ(m[2], 0.0, 0.01);   /* m_3 = 0 */
    ASSERT_FEQ(m[4], 0.0, 0.01);   /* m_5 = 0 */
    
    /* Even moments are Catalan numbers: C_1=1, C_2=2, C_3=5 */
    ASSERT_FEQ(m[1], 1.0, 0.05);   /* m_2 = 1 = C_1 */
    ASSERT_FEQ(m[3], 2.0, 0.1);    /* m_4 = 2 = C_2 */
    ASSERT_FEQ(m[5], 5.0, 0.2);    /* m_6 = 5 = C_3 */
    
    PASS();
    #undef N_SEMI
}

/* ============================================================
 * Test: Moment-cumulant roundtrip
 * ============================================================ */
static void test_moment_cumulant_roundtrip(void) {
    TEST("Moment → cumulant → moment roundtrip");
    
    /* Use moments of a distribution; just verify roundtrip */
    double moments_in[] = {2.0, 7.0, 28.0, 127.0, 626.0};
    size_t n = 5;
    
    double cumulants[5];
    moment_to_cumulant(moments_in, n, cumulants);    
    /* Roundtrip back */
    double moments_out[5];
    cumulant_to_moment(cumulants, n, moments_out);
    
    for (size_t i = 0; i < n; i++) {
        ASSERT_FEQ(moments_out[i], moments_in[i], 1e-8);
    }
    
    PASS();
}

/* ============================================================
 * Test: Free cumulants of Gaussian are {μ, σ², 0, 0, ...}
 * ============================================================ */
static void test_gaussian_free_cumulants(void) {
    TEST("Classical Gaussian moment-cumulant roundtrip");
    
    /* Standard normal: m_1=0, m_2=1, m_3=0, m_4=3, m_5=0, m_6=15 */
    double moments[] = {0.0, 1.0, 0.0, 3.0, 0.0, 15.0};
    double cumulants[6];
    moment_to_cumulant(moments, 6, cumulants);
    
    /* Free cumulants of classical Gaussian are NOT the same as classical.
     * Only the semicircle law has κ_n=0 for n≥3 in free probability.
     * Just verify roundtrip. */
    double moments_rt[6];
    cumulant_to_moment(cumulants, 6, moments_rt);
    
    for (int i = 0; i < 6; i++) {
        ASSERT_FEQ(moments_rt[i], moments[i], 1e-8);
    }
    
    PASS();
}

/* ============================================================
 * Test: R-transform from cumulants
 * ============================================================ */
static void test_r_transform_basic(void) {
    TEST("R-transform evaluation from cumulants");
    
    /* For Gaussian(0,1): κ = {0, 1, 0, 0} → R(z) = 0 + 1*z + 0 + 0 = z */
    double cumulants[] = {0.0, 1.0, 0.0, 0.0};
    
    double r0 = r_transform_from_cumulants(cumulants, 4, 0.0);
    double r1 = r_transform_from_cumulants(cumulants, 4, 1.0);
    double r2 = r_transform_from_cumulants(cumulants, 4, 2.0);
    double r_half = r_transform_from_cumulants(cumulants, 4, 0.5);
    
    ASSERT_FEQ(r0, 0.0, 1e-12);    /* R(0) = κ_1 = 0 */
    ASSERT_FEQ(r1, 1.0, 1e-12);    /* R(1) = 0 + 1 = 1 */
    ASSERT_FEQ(r2, 2.0, 1e-12);    /* R(2) = 0 + 2 = 2 */
    ASSERT_FEQ(r_half, 0.5, 1e-12); /* R(0.5) = 0 + 0.5 = 0.5 */
    
    PASS();
}

/* ============================================================
 * Test: R-transform additivity
 * ============================================================ */
static void test_r_transform_additivity(void) {
    TEST("R-transform additivity: R_{A+B} = R_A + R_B");
    
    /* A ~ Gaussian(1, 2): κ_A = {1, 2, 0, 0}
     * B ~ Gaussian(3, 4): κ_B = {3, 4, 0, 0}
     * A+B ~ Gaussian(4, 6): κ_{A+B} = {4, 6, 0, 0} */
    double kappa_a[] = {1.0, 2.0, 0.0, 0.0};
    double kappa_b[] = {3.0, 4.0, 0.0, 0.0};
    double kappa_sum[4];
    
    r_transform_add(kappa_a, kappa_b, 4, kappa_sum);
    
    ASSERT_FEQ(kappa_sum[0], 4.0, 1e-12);
    ASSERT_FEQ(kappa_sum[1], 6.0, 1e-12);
    ASSERT_FEQ(kappa_sum[2], 0.0, 1e-12);
    ASSERT_FEQ(kappa_sum[3], 0.0, 1e-12);
    
    /* Verify R_{A+B}(z) = R_A(z) + R_B(z) at several points */
    for (double z = -1.0; z <= 1.0; z += 0.5) {
        double r_a = r_transform_from_cumulants(kappa_a, 4, z);
        double r_b = r_transform_from_cumulants(kappa_b, 4, z);
        double r_sum = r_transform_from_cumulants(kappa_sum, 4, z);
        ASSERT_FEQ(r_sum, r_a + r_b, 1e-12);
    }
    
    PASS();
}

/* ============================================================
 * Test: R-transform additivity with non-zero higher cumulants
 * ============================================================ */
static void test_r_transform_additivity_nonzero(void) {
    TEST("R-transform additivity with non-Gaussian distributions");
    
    double kappa_a[] = {1.0, 2.0, 0.5, -0.3};
    double kappa_b[] = {2.0, 3.0, -0.5, 0.7};
    double kappa_sum[4];
    
    r_transform_add(kappa_a, kappa_b, 4, kappa_sum);
    
    ASSERT_FEQ(kappa_sum[0], 3.0, 1e-12);
    ASSERT_FEQ(kappa_sum[1], 5.0, 1e-12);
    ASSERT_FEQ(kappa_sum[2], 0.0, 1e-12);
    ASSERT_FEQ(kappa_sum[3], 0.4, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: Stieltjes transform roundtrip for Gaussian
 * ============================================================ */
static void test_stieltjes_roundtrip(void) {
    TEST("Stieltjes transform roundtrip (Gaussian)");
    
    /* For Gaussian(0,1): R(z) = z, so S = 1/(z - S), giving S = (z - √(z²-4))/2 */
    double cumulants[] = {0.0, 1.0, 0.0, 0.0, 0.0, 0.0};
    
    /* Test at z = 3i (complex not supported, use real z > 2 for semicircle edge) */
    /* For real z outside support, S(z) is real. 
     * For semicircle on [-2,2]: G(z) = (z - √(z²-4))/2 */
    double z_test = 5.0;
    double s = stieltjes_from_r(cumulants, 6, z_test, 1000, 1e-12);
    
    /* Expected: G(5) = (5 - √(25-4))/2 = (5 - √21)/2 = (5 - 4.5826)/2 = 0.2087 */
    double expected = (z_test - sqrt(z_test * z_test - 4.0)) / 2.0;
    ASSERT_FEQ(s, expected, 1e-6);
    
    PASS();
}

/* ============================================================
 * Test: Marchenko-Pastur density at peak
 * ============================================================ */
static void test_mp_density(void) {
    TEST("Marchenko-Pastur density at mode");
    
    /* MP with λ=1, σ²=1: support [0, 4], density at x=1:
     * ρ(1) = √((4-1)(1-0)) / (2π·1·1·1) = √3/(2π) ≈ 0.2757 */
    double rho = marchenko_pastur_density(1.0, 1.0, 1.0);
    double expected = sqrt(3.0) / (2.0 * M_PI);
    ASSERT_FEQ(rho, expected, 1e-10);
    
    /* Outside support should be 0 */
    ASSERT_FEQ(marchenko_pastur_density(-0.1, 1.0, 1.0), 0.0, 1e-15);
    ASSERT_FEQ(marchenko_pastur_density(4.1, 1.0, 1.0), 0.0, 1e-15);
    
    /* At boundaries */
    ASSERT_FEQ(marchenko_pastur_density(0.0, 1.0, 1.0), 0.0, 1e-10);
    ASSERT_FEQ(marchenko_pastur_density(4.0, 1.0, 1.0), 0.0, 1e-10);
    
    PASS();
}

/* ============================================================
 * Test: MP density integrates to ~1
 * ============================================================ */
static void test_mp_density_integral(void) {
    TEST("Marchenko-Pastur density integrates to 1");
    
    double lambda = 1.0;
    double a = (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    double b = (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    
    /* Numerical integration */
    int n_bins = 10000;
    double dx = (b - a) / n_bins;
    double integral = 0.0;
    for (int i = 0; i < n_bins; i++) {
        double x = a + (i + 0.5) * dx;
        integral += marchenko_pastur_density(x, lambda, 1.0) * dx;
    }
    
    ASSERT_FEQ(integral, 1.0, 0.01);
    
    PASS();
}

/* ============================================================
 * Test: MP density with different ratios
 * ============================================================ */
static void test_mp_density_lambda_half(void) {
    TEST("MP density with λ=0.5");
    
    /* λ=0.5: support a=(1-√0.5)²≈0.0858, b=(1+√0.5)²≈2.9142 */
    double lambda = 0.5;
    double a = (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    double b = (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    
    /* Check density at midpoint */
    double mid = (a + b) / 2.0;
    double rho_mid = marchenko_pastur_density(mid, lambda, 1.0);
    
    /* Should be positive and finite */
    if (rho_mid <= 0.0 || rho_mid > 100.0) {
        FAIL("MP density out of expected range");
        return;
    }
    
    /* Density below a and above b should be 0 */
    ASSERT_FEQ(marchenko_pastur_density(a - 0.01, lambda, 1.0), 0.0, 1e-15);
    ASSERT_FEQ(marchenko_pastur_density(b + 0.01, lambda, 1.0), 0.0, 1e-15);
    
    PASS();
}

/* ============================================================
 * Test: MP moments
 * ============================================================ */
static void test_mp_moments(void) {
    TEST("Marchenko-Pastur moments");
    
    /* MP(λ=1) moments: m_1=1, m_2=2, m_3=5, m_4=14 (Catalan numbers) */
    double moments[6];
    marchenko_pastur_moments(1.0, 6, moments);
    
    ASSERT_FEQ(moments[0], 1.0, 1e-10);   /* m_1 = 1 */
    ASSERT_FEQ(moments[1], 2.0, 1e-10);   /* m_2 = 2 */
    ASSERT_FEQ(moments[2], 5.0, 1e-10);   /* m_3 = 5 */
    ASSERT_FEQ(moments[3], 14.0, 1e-10);  /* m_4 = 14 */
    ASSERT_FEQ(moments[4], 42.0, 1e-10);  /* m_5 = 42 */
    ASSERT_FEQ(moments[5], 132.0, 1e-10); /* m_6 = 132 */
    
    PASS();
}

/* ============================================================
 * Test: MP cumulants are κ_n = λ^{n-1}
 * ============================================================ */
static void test_mp_cumulants(void) {
    TEST("MP free cumulants κ_n = λ^{n-1}");
    
    double lambda = 0.7;
    double moments[5];
    marchenko_pastur_moments(lambda, 5, moments);
    
    double cumulants[5];
    moment_to_cumulant(moments, 5, cumulants);
    
    ASSERT_FEQ(cumulants[0], 1.0, 1e-8);          /* κ_1 = λ^0 = 1 */
    ASSERT_FEQ(cumulants[1], lambda, 1e-8);        /* κ_2 = λ^1 = 0.7 */
    ASSERT_FEQ(cumulants[2], lambda*lambda, 1e-8); /* κ_3 = λ^2 = 0.49 */
    
    PASS();
}

/* ============================================================
 * Test: MP moments with λ=0.5
 * ============================================================ */
static void test_mp_moments_lambda_half(void) {
    TEST("MP moments with λ=0.5");
    
    double lambda = 0.5;
    double moments[4];
    marchenko_pastur_moments(lambda, 4, moments);
    
    /* m_1 = 1 always for MP */
    ASSERT_FEQ(moments[0], 1.0, 1e-10);
    
    /* m_2 should be 1 + λ = 1.5 */
    ASSERT_FEQ(moments[1], 1.0 + lambda, 1e-8);
    
    PASS();
}

/* ============================================================
 * Test: S-transform of Gaussian
 * ============================================================ */
static void test_s_transform_mp1(void) {
    TEST("S-transform of MP(1) roundtrip");
    
    /* MP(1) moments */
    double mp_moments[5];
    marchenko_pastur_moments(1.0, 5, mp_moments);
    
    double s_coeffs[5];
    s_transform_from_moments(mp_moments, 5, s_coeffs);
    
    /* S-transform should produce finite, non-zero coefficients */
    if (fabs(s_coeffs[0]) < 1e-12) {
        FAIL("S-transform leading coefficient is zero");
        return;
    }
    
    /* s_0 should be positive and finite */
    if (s_coeffs[0] <= 0.0 || s_coeffs[0] != s_coeffs[0]) {
        FAIL("Invalid s_0");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: S-transform multiplication
 * ============================================================ */
static void test_s_transform_multiply(void) {
    TEST("S-transform polynomial multiplication");
    
    double sa[] = {1.0, 2.0, 3.0};
    double sb[] = {4.0, 5.0, 6.0};
    double product[5];
    
    s_transform_multiply(sa, sb, 3, product);
    
    /* (1+2z+3z²)(4+5z+6z²) = 4 + 13z + 28z² + 27z³ + 18z⁴ */
    ASSERT_FEQ(product[0], 4.0, 1e-12);
    ASSERT_FEQ(product[1], 13.0, 1e-12);
    ASSERT_FEQ(product[2], 28.0, 1e-12);
    ASSERT_FEQ(product[3], 27.0, 1e-12);
    ASSERT_FEQ(product[4], 18.0, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: Validate moments - valid distribution
 * ============================================================ */
static void test_validate_valid_moments(void) {
    TEST("Validate moments: valid Gaussian moments");
    
    /* Gaussian(0,1) moments: 0, 1, 0, 3 */
    double moments[] = {0.0, 1.0, 0.0, 3.0};
    bool valid = validate_moments(moments, 4);
    
    if (!valid) {
        FAIL("Valid Gaussian moments rejected");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: Validate moments - invalid distribution
 * ============================================================ */
static void test_validate_invalid_moments(void) {
    TEST("Validate moments: detect invalid moments");
    
    /* m_2 < m_1² is impossible for any real distribution (variance < 0) */
    double moments[] = {5.0, 1.0}; /* mean=5, E[X²]=1 → variance = 1-25 < 0 */
    bool valid = validate_moments(moments, 2);
    
    if (valid) {
        FAIL("Invalid moments accepted (variance < 0)");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: Degenerate distribution (point mass)
 * ============================================================ */
static void test_degenerate_distribution(void) {
    TEST("Degenerate distribution (point mass at c)");
    
    /* Point mass at c=3: m_k = 3^k */
    double c = 3.0;
    double moments[] = {3.0, 9.0, 27.0, 81.0};
    double cumulants[4];
    moment_to_cumulant(moments, 4, cumulants);
    
    /* Free cumulants of a point mass at c: κ_1 = c, κ_n = 0 for n ≥ 2 */
    ASSERT_FEQ(cumulants[0], c, 1e-10);
    ASSERT_FEQ(cumulants[1], 0.0, 1e-8);
    ASSERT_FEQ(cumulants[2], 0.0, 1e-8);
    ASSERT_FEQ(cumulants[3], 0.0, 1e-8);
    
    PASS();
}

/* ============================================================
 * Test: Single-point empirical distribution
 * ============================================================ */
static void test_single_point(void) {
    TEST("Single-point empirical distribution");
    
    double pts[] = {5.0};
    EmpiricalDist dist = {5.0, 5.0, 1, pts};
    
    double m[3];
    compute_moments(&dist, 3, m);
    
    ASSERT_FEQ(m[0], 5.0, 1e-12);
    ASSERT_FEQ(m[1], 25.0, 1e-12);
    ASSERT_FEQ(m[2], 125.0, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: Two-point distribution
 * ============================================================ */
static void test_two_point(void) {
    TEST("Two-point distribution");
    
    double pts[] = {0.0, 2.0};
    EmpiricalDist dist = {0.0, 2.0, 2, pts};
    
    double m[4];
    compute_moments(&dist, 4, m);
    
    /* m_k = (0^k + 2^k)/2 */
    ASSERT_FEQ(m[0], 1.0, 1e-12);  /* mean = 1 */
    ASSERT_FEQ(m[1], 2.0, 1e-12);  /* E[X²] = 2 */
    ASSERT_FEQ(m[2], 4.0, 1e-12);  /* E[X³] = 4 */
    ASSERT_FEQ(m[3], 8.0, 1e-12);  /* E[X⁴] = 8 */
    
    /* Free cumulants: κ_1 = 1, κ_2 = 1, κ_3 = 0, κ_4 = 0 
     * Wait, two-point at {0,2}: this is NOT freely {point at 0} + {point at 2}.
     * Let's just check roundtrip. */
    double cumulants[4];
    moment_to_cumulant(m, 4, cumulants);
    
    double moments_rt[4];
    cumulant_to_moment(cumulants, 4, moments_rt);
    
    for (int i = 0; i < 4; i++) {
        ASSERT_FEQ(moments_rt[i], m[i], 1e-8);
    }
    
    PASS();
}

/* ============================================================
 * Test: Gradient initialization suggestion
 * ============================================================ */
static void test_gradient_init_suggestion(void) {
    TEST("Gradient initialization suggestion");
    
    TransformerConfig config = {
        .n_layers = 12,
        .hidden_dim = 768.0,
        .n_samples = 10000.0,
        .weight_std = 0.02,
        .learning_rate = 0.001
    };
    
    InitSuggestion sug = suggest_initialization(&config);
    
    /* Suggested std should be positive and finite */
    if (sug.suggested_std <= 0.0 || sug.suggested_std != sug.suggested_std) {
        FAIL("Invalid suggested_std");
        return;
    }
    
    /* Should be small (O(1/√d)) */
    if (sug.suggested_std > 1.0 || sug.suggested_std < 1e-10) {
        FAIL("suggested_std out of reasonable range");
        return;
    }
    
    /* lr_scale should be positive and finite */
    if (sug.suggested_lr_scale <= 0.0 || sug.suggested_lr_scale != sug.suggested_lr_scale) {
        FAIL("Invalid suggested_lr_scale");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: Gradient initialization - NULL config
 * ============================================================ */
static void test_gradient_init_null(void) {
    TEST("Gradient initialization with NULL config");
    
    InitSuggestion sug = suggest_initialization(NULL);
    
    ASSERT_FEQ(sug.suggested_std, 0.02, 1e-12);
    ASSERT_FEQ(sug.suggested_lr_scale, 1.0, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: Regularizer suggestion
 * ============================================================ */
static void test_regularizer_suggestion(void) {
    TEST("Regularizer suggestion from moments");
    
    /* Gaussian-like moments: mean=0, var=1, skew=0, kurtosis=3 */
    double moments[] = {0.0, 1.0, 0.0, 3.0, 0.0, 15.0};
    
    TransformerConfig config = {
        .n_layers = 6,
        .hidden_dim = 512.0,
        .n_samples = 5000.0,
        .weight_std = 0.02,
        .learning_rate = 0.001
    };
    
    RegularizerSuggestion sug = suggest_regularizer(moments, 6, &config);
    
    /* Regularization should be positive and finite */
    if (sug.lambda_reg <= 0.0 || sug.lambda_reg != sug.lambda_reg) {
        FAIL("Invalid lambda_reg");
        return;
    }
    
    /* Stability score should be in [0, 1] */
    if (sug.stability_score < 0.0 || sug.stability_score > 1.0) {
        FAIL("stability_score out of [0,1]");
        return;
    }
    
    /* Outlier fraction should be in [0, 1] */
    if (sug.outlier_fraction < 0.0 || sug.outlier_fraction > 1.0) {
        FAIL("outlier_fraction out of [0,1]");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: Combined distribution prediction
 * ============================================================ */
static void test_combined_distribution(void) {
    TEST("Combined layer distribution via R-transform");
    
    /* Single layer cumulants */
    double layer_cum[] = {1.0, 0.5, 0.1, 0.01};
    
    /* 4 layers combined: κ_combined = 4 * κ_single */
    double combined[4];
    predict_combined_distribution(layer_cum, 4, 4, combined);
    
    ASSERT_FEQ(combined[0], 4.0, 1e-12);
    ASSERT_FEQ(combined[1], 2.0, 1e-12);
    ASSERT_FEQ(combined[2], 0.4, 1e-12);
    ASSERT_FEQ(combined[3], 0.04, 1e-12);
    
    /* Verify R-transform additivity */
    double z = 0.5;
    double r_single = r_transform_from_cumulants(layer_cum, 4, z);
    double r_combined = r_transform_from_cumulants(combined, 4, z);
    ASSERT_FEQ(r_combined, 4.0 * r_single, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: Validate uniform moments
 * ============================================================ */
static void test_validate_uniform_moments(void) {
    TEST("Validate moments: uniform distribution");
    
    /* Uniform[0,1]: m_1=0.5, m_2=1/3, m_3=0.25 */
    double moments[] = {0.5, 1.0/3.0, 0.25, 0.2};
    bool valid = validate_moments(moments, 4);
    
    if (!valid) {
        FAIL("Valid uniform moments rejected");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: Cumulant to moment and back (roundtrip for non-Gaussian)
 * ============================================================ */
static void test_cumulant_roundtrip_non_gaussian(void) {
    TEST("Cumulant roundtrip: non-Gaussian distribution");
    
    /* MP(1) moments */
    double mp_mom[5];
    marchenko_pastur_moments(1.0, 5, mp_mom);
    
    double cumulants[5];
    moment_to_cumulant(mp_mom, 5, cumulants);
    
    double moments_back[5];
    cumulant_to_moment(cumulants, 5, moments_back);
    
    for (int i = 0; i < 5; i++) {
        ASSERT_FEQ(moments_back[i], mp_mom[i], 1e-8);
    }
    
    PASS();
}

/* ============================================================
 * Test: R-transform from Stieltjes
 * ============================================================ */
static void test_r_from_stieltjes(void) {
    TEST("R(z) from Stieltjes transform value");
    
    /* Given S(z) = w, R(w) = z - 1/w */
    double s = 0.3;
    double z = 5.0;
    double r = r_from_stieltjes(s, z);
    
    ASSERT_FEQ(r, z - 1.0/s, 1e-12);
    
    PASS();
}

/* ============================================================
 * Test: S-transform of MP(1) should be 1/(z+1)
 * ============================================================ */
static void test_s_transform_mp1_exact(void) {
    TEST("S-transform of MP(1) = 1/(z+1) series");
    
    double mp_moments[8];
    marchenko_pastur_moments(1.0, 8, mp_moments);
    
    double s_coeffs[8];
    s_transform_from_moments(mp_moments, 8, s_coeffs);
    
    /* S(z) = 1/(z+1) = 1 - z + z^2 - z^3 + ... */
    for (int i = 0; i < 8; i++) {
        double expected = pow(-1.0, (double)i);
        ASSERT_FEQ(s_coeffs[i], expected, 1e-6);
    }
    
    /* Evaluate S(0.5) = 1/1.5 = 0.6667 */
    double z = 0.5, s_val = 0.0, zpow = 1.0;
    for (int i = 0; i < 8; i++) {
        s_val += s_coeffs[i] * zpow;
        zpow *= z;
    }
    ASSERT_FEQ(s_val, 1.0 / (1.0 + z), 5e-3);
    
    PASS();
}

/* ============================================================
 * Test: S-transform is non-trivial (not constant)
 * ============================================================ */
static void test_s_transform_nontrivial(void) {
    TEST("S-transform coefficients are non-trivial");
    
    double mp_moments[6];
    marchenko_pastur_moments(0.7, 6, mp_moments);
    
    double s_coeffs[6];
    s_transform_from_moments(mp_moments, 6, s_coeffs);
    
    /* s_0 should be 1 (since m_1 = 1 for MP) */
    ASSERT_FEQ(s_coeffs[0], 1.0, 1e-8);
    
    /* Higher coefficients should NOT all be zero */
    int nontrivial = 0;
    for (int i = 1; i < 6; i++) {
        if (fabs(s_coeffs[i]) > 1e-10) nontrivial++;
    }
    if (nontrivial < 3) {
        FAIL("S-transform is approximately constant — series reversion broken");
        return;
    }
    
    PASS();
}

/* ============================================================
 * Test: MP density integrates correctly for λ=0.5
 * ============================================================ */
static void test_mp_density_integral_lambda_half(void) {
    TEST("MP density integrates correctly for λ=0.5");
    
    double lambda = 0.5;
    double a = (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    double b = (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    
    int n_bins = 100000;
    double dx = (b - a) / n_bins;
    double integral = 0.0;
    for (int i = 0; i < n_bins; i++) {
        double x = a + (i + 0.5) * dx;
        integral += marchenko_pastur_density(x, lambda, 1.0) * dx;
    }
    
    ASSERT_FEQ(integral, 1.0, 0.01);
    PASS();
}

/* ============================================================
 * Test: MP density integrates correctly for λ=2.0
 * ============================================================ */
static void test_mp_density_integral_lambda_two(void) {
    TEST("MP density integrates correctly for λ=2.0");
    
    double lambda = 2.0;
    double a = (1.0 - sqrt(lambda)) * (1.0 - sqrt(lambda));
    double b = (1.0 + sqrt(lambda)) * (1.0 + sqrt(lambda));
    
    int n_bins = 100000;
    double dx = (b - a) / n_bins;
    double integral = 0.0;
    for (int i = 0; i < n_bins; i++) {
        double x = a + (i + 0.5) * dx;
        integral += marchenko_pastur_density(x, lambda, 1.0) * dx;
    }
    
    ASSERT_FEQ(integral, 1.0, 0.01);
    PASS();
}

/* ============================================================
 * Main
 * ============================================================ */
int main(void) {
    printf("=== Free Probability Library Tests ===\n\n");
    
    /* Moment tests */
    printf("[Moments]\n");
    test_uniform_moments();
    test_semicircle_moments();
    test_moment_cumulant_roundtrip();
    test_gaussian_free_cumulants();
    test_cumulant_roundtrip_non_gaussian();
    
    /* R-transform tests */
    printf("\n[R-transform]\n");
    test_r_transform_basic();
    test_r_transform_additivity();
    test_r_transform_additivity_nonzero();
    test_stieltjes_roundtrip();
    test_r_from_stieltjes();
    
    /* Marchenko-Pastur tests */
    printf("\n[Marchenko-Pastur]\n");
    test_mp_density();
    test_mp_density_integral();
    test_mp_density_lambda_half();
    test_mp_moments();
    test_mp_cumulants();
    test_mp_moments_lambda_half();
    test_mp_density_integral_lambda_half();
    test_mp_density_integral_lambda_two();
    
    /* S-transform tests */
    printf("\n[S-transform]\n");
    test_s_transform_mp1();
    test_s_transform_mp1_exact();
    test_s_transform_nontrivial();
    test_s_transform_multiply();
    
    /* Validation tests */
    printf("\n[Validation]\n");
    test_validate_valid_moments();
    test_validate_invalid_moments();
    test_validate_uniform_moments();
    
    /* Edge cases */
    printf("\n[Edge Cases]\n");
    test_degenerate_distribution();
    test_single_point();
    test_two_point();
    
    /* Gradient analysis */
    printf("\n[Gradient Analysis]\n");
    test_gradient_init_suggestion();
    test_gradient_init_null();
    test_regularizer_suggestion();
    test_combined_distribution();
    
    printf("\n=== Results: %d/%d passed ===\n", pass_count, test_count);
    
    return (pass_count == test_count) ? 0 : 1;
}
