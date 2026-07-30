import numpy as np
from scipy import stats

def compute_statistics(values, confidence=0.95):
    """
    Computes mean, standard deviation, and 95% confidence interval for a list of values.
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1) if n > 1 else 0.0
    
    if n > 1 and std > 0:
        se = std / np.sqrt(n)
        h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    else:
        h = 0.0
        
    return {
        "mean": mean,
        "std": std,
        "ci_95": h,
        "n": n
    }

def welch_ttest(sample_a, sample_b):
    """
    Performs Welch's t-test (two-sample un-equal variance t-test).
    Returns t-statistic and p-value.
    """
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.statistic), float(res.pvalue)

def mann_whitney_test(sample_a, sample_b):
    """
    Performs Mann-Whitney U rank-sum test (non-parametric).
    Returns u-statistic and p-value.
    """
    a = np.array(sample_a, dtype=np.float64)
    b = np.array(sample_b, dtype=np.float64)
    res = stats.mannwhitneyu(a, b, alternative='two-sided')
    return float(res.statistic), float(res.pvalue)

def format_stat_string(mean, std, p_val=None, latex=False):
    """
    Formats stats as 'Mean ± Std (p=X.XXX)' or LaTeX equivalent.
    """
    if latex:
        base = f"${mean:.2f} \\pm {std:.2f}$"
        if p_val is not None:
            base += f" ($p={p_val:.3f}$)"
        return base
    else:
        base = f"{mean:.2f} ± {std:.2f}"
        if p_val is not None:
            base += f" (p={p_val:.3f})"
        return base

if __name__ == "__main__":
    # Quick test
    s1 = [82.1, 84.3, 81.5, 83.0, 82.8]
    s2 = [71.2, 74.0, 70.5, 72.8, 71.9]
    res1 = compute_statistics(s1)
    t_stat, p_val = welch_ttest(s1, s2)
    print("Sample 1:", format_stat_string(res1['mean'], res1['std']))
    print(f"Welch t-test p-value: {p_val:.4f}")
