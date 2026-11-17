import math


def wilson_confidence_interval(successes, trials, z=1.96):
    """Return a two-sided 95% Wilson interval for a binomial proportion."""
    if trials == 0:
        return float("nan"), float("nan")

    proportion = successes / trials
    denominator = 1 + z**2 / trials
    center = (proportion + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials
        + z**2 / (4 * trials**2)
    ) / denominator
    return center - margin, center + margin
