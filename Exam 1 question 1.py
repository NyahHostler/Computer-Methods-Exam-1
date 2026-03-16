from hw4a import makeSample, sampleStats, getFDMaxFDMin
from hw3b import t_cdf
import math

def main():
    '''
    Runs a one–tailed t-test to check whether gravel supplied by
    Supplier B has a statistically smaller average diameter than
    gravel supplied by Supplier A.

    Supplier A uses a screen with a 1"x1" opening.
    Supplier B uses a tighter screen with a 7/8"x7/8" opening.

    H0: mean_B >= mean_A  (B is not smaller on average)
    H1: mean_B < mean_A   (B produces smaller gravel)

    Sample generation uses the truncated lognormal distribution
    implemented in hw4a and the t distribution CDF from hw3b.
    Portions of this function were assisted by AI.
    '''

    print("Gravel Supplier Statistical Comparison\n")

    #region Step 1: Ask the user for input values (defaults are provided)
    mu_default = math.log(1)
    sigma_default = 0.2
    D_min_default = 0.375
    D_max_A_default = 1.0
    D_max_B_default = 7/8
    sample_size_default = 100
    N_samples_default = 11
    alpha = 0.05

    st_mu = input(f'Mean of ln(D) for pre-sieved rocks? ({mu_default:.3f}): ').strip()
    mu = mu_default if st_mu == '' else float(st_mu)

    st_sigma = input(f'Standard deviation of ln(D)? ({sigma_default:.3f}): ').strip()
    sigma = sigma_default if st_sigma == '' else float(st_sigma)

    st_D_min = input(f'Minimum rock diameter, D_min (inches)? ({D_min_default:.3f}): ').strip()
    D_min = D_min_default if st_D_min == '' else float(st_D_min)

    st_D_max_A = input(f'Supplier A large aperture size (inches)? ({D_max_A_default:.3f}): ').strip()
    D_max_A = D_max_A_default if st_D_max_A == '' else float(st_D_max_A)

    st_D_max_B = input(f'Supplier B large aperture size (inches)? ({D_max_B_default:.3f}): ').strip()
    D_max_B = D_max_B_default if st_D_max_B == '' else float(st_D_max_B)

    st_N = input(f'Number of samples? ({N_samples_default}): ').strip()
    N_samples = N_samples_default if st_N == '' else int(st_N)

    st_size = input(f'Sample size (rocks per sample)? ({sample_size_default}): ').strip()
    sample_size = sample_size_default if st_size == '' else int(st_size)
    #endregion

    #region Step 2: Determine truncation limits and create the random samples
    print("\nGenerating samples...")
    F_A_min, F_A_max = getFDMaxFDMin((mu, sigma, D_min, D_max_A))
    F_B_min, F_B_max = getFDMaxFDMin((mu, sigma, D_min, D_max_B))

    samples_A = [makeSample((mu, sigma, D_min, D_max_A, F_A_max, F_A_min), sample_size) for _ in range(N_samples)]
    samples_B = [makeSample((mu, sigma, D_min, D_max_B, F_B_max, F_B_min), sample_size) for _ in range(N_samples)]
    #endregion

    #region Step 3: Calculate statistics for each generated sample
    print("\nSupplier A samples:")
    means_A = []
    for i, s in enumerate(samples_A):
        mean, var = sampleStats(s)
        means_A.append(mean)
        print(f"  Sample {i+1}: mean = {mean:.3f}, var = {var:.6f}")

    print("\nSupplier B samples:")
    means_B = []
    for i, s in enumerate(samples_B):
        mean, var = sampleStats(s)
        means_B.append(mean)
        print(f"  Sample {i+1}: mean = {mean:.3f}, var = {var:.6f}")
    #endregion

    #region Step 4: Determine the mean and variance of the sample means
    mean_of_means_A = sum(means_A) / N_samples
    mean_of_means_B = sum(means_B) / N_samples
    var_of_means_A = sum((m - mean_of_means_A)**2 for m in means_A) / (N_samples - 1)
    var_of_means_B = sum((m - mean_of_means_B)**2 for m in means_B) / (N_samples - 1)

    print(f"\nSupplier A: mean of sampling means = {mean_of_means_A:.4f}, variance = {var_of_means_A:.6f}")
    print(f"Supplier B: mean of sampling means = {mean_of_means_B:.4f}, variance = {var_of_means_B:.6f}")
    #endregion

    #region Step 5: Conduct the one-sided t-test (testing if B < A)
    print("\n--- Hypothesis Test ---")
    print("H0: mean_B >= mean_A (Supplier B is NOT significantly smaller)")
    print("H1: mean_B < mean_A (Supplier B IS significantly smaller)")
    print(f"Alpha = {alpha}")

    mean_diff = mean_of_means_A - mean_of_means_B  # positive value indicates A has larger average diameter
    t_stat = mean_diff / math.sqrt(var_of_means_A / N_samples + var_of_means_B / N_samples)

    df_num = (var_of_means_A / N_samples + var_of_means_B / N_samples) ** 2
    df_den = (var_of_means_A**2) / (N_samples**2 * (N_samples - 1)) + (var_of_means_B**2) / (N_samples**2 * (N_samples - 1))
    df = df_num / df_den

    p_val = 1 - t_cdf(t_stat, df)

    print(f"\nt-statistic = {t_stat:.3f}")
    print(f"Degrees of freedom = {df:.3f}")
    print(f"p-value = {p_val:.4f}")

    if p_val < alpha:
        print("\nConclusion: Reject H0. Supplier B produces statistically significantly smaller gravel than Supplier A.")
    else:
        print("\nConclusion: Fail to reject H0. No statistically significant difference in gravel size.")
    #endregion

if __name__ == "__main__":
    main()