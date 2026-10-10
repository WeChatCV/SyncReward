"""Agreement metrics between predicted rewards and human sync ratings (the paper's definitions).

* Spearman / Pearson / Kendall tau-b against the target.
* Pair Acc: fraction of clip pairs with different targets that the prediction orders correctly
  (prediction ties count 0.5). `pair_acc_gap0.75` keeps only pairs whose targets differ by > 0.75.
* Top-1 (N=4): best-of-N selection accuracy. Clips are randomly partitioned into groups of N
  (groups with a single target value are dropped); a group counts as correct when the top-scored
  clip is among the human-best ones (prediction ties get fractional credit). Averaged over
  `draws` seeded partitions, so every method sees identical groups.
* 95% CIs: clip-level bootstrap with seeded resamples shared across methods.
"""

from collections import defaultdict

import numpy as np

TIE = 1e-12
SEED = 20260820


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) != len(x):
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    concordant = discordant = tied_x_only = tied_y_only = 0
    for i in range(len(x) - 1):
        dx = np.sign(x[i] - x[i + 1:])
        dy = np.sign(y[i] - y[i + 1:])
        s = dx * dy
        concordant += int((s > 0).sum())
        discordant += int((s < 0).sum())
        tied_x_only += int(((dx == 0) & (dy != 0)).sum())
        tied_y_only += int(((dy == 0) & (dx != 0)).sum())
    denom = np.sqrt(float(concordant + discordant + tied_x_only) * float(concordant + discordant + tied_y_only))
    return float((concordant - discordant) / denom) if denom > 0 else 0.0


def pairwise_accuracy(target, prediction, min_gap=0.0):
    """(accuracy, n_pairs) over pairs with |target gap| > min_gap; prediction ties count 0.5."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rows, cols = np.triu_indices(len(target), k=1)
    gap = target[rows] - target[cols]
    keep = np.abs(gap) > (min_gap if min_gap > 0 else TIE)
    if not keep.any():
        return float("nan"), 0
    product = np.sign(gap[keep]) * np.sign((prediction[rows] - prediction[cols])[keep])
    return (float((product > 0).sum()) + 0.5 * float((product == 0).sum())) / len(product), int(len(product))


def top1_on_groups(scores, targets, groups):
    """(accuracy, chance, groups used) of naming a human-best clip in each group."""
    credit = chance = 0.0
    used = 0
    for group in groups:
        t = targets[group]
        if t.max() - t.min() <= TIE:
            continue
        best = t >= t.max() - TIE
        s = scores[group]
        top = s >= s.max() - TIE
        credit += float((top & best).sum()) / float(top.sum())
        chance += float(best.sum()) / float(len(group))
        used += 1
    return (credit / used, chance / used, used) if used else (float("nan"), float("nan"), 0)


def random_groups(rng, size, group_size):
    order = rng.permutation(size)
    usable = (size // group_size) * group_size
    return order[:usable].reshape(-1, group_size)


def top1_accuracy(target, prediction, group_size=4, draws=1000, seed=SEED):
    """Best-of-N selection accuracy averaged over `draws` random partitions. Returns (acc, chance)."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    acc, chance = [], []
    for repeat in range(draws):
        rng = np.random.default_rng([seed, group_size, 0, repeat])
        a, c, _ = top1_on_groups(prediction, target, random_groups(rng, len(target), group_size))
        acc.append(a)
        chance.append(c)
    return float(np.mean(acc)), float(np.mean(chance))


def compute_metrics(target, prediction, group_size=4, draws=1000, seed=SEED):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(target) == 0 or len(target) != len(prediction):
        raise ValueError("target and prediction must have equal non-zero length")
    out = {"n": int(len(target)), "spearman": spearman(prediction, target), "pearson": pearson(prediction, target),
           "kendall": kendall_tau_b(prediction, target), "mae": float(np.abs(prediction - target).mean())}
    out["pair_acc"], out["pair_n"] = pairwise_accuracy(target, prediction)
    out["pair_acc_gap0.75"], out["pair_n_gap0.75"] = pairwise_accuracy(target, prediction, min_gap=0.75)
    if draws > 0:
        out[f"top1_n{group_size}"], out[f"top1_n{group_size}_chance"] = top1_accuracy(
            target, prediction, group_size, draws, seed)
    return out


def bootstrap_ci(target, prediction, num_samples=1000, group_size=4, seed=SEED):
    """95% percentile CIs for spearman, pair_acc and top1 (one random partition per resample)."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    values = defaultdict(list)
    for repeat in range(num_samples):
        pick = np.random.default_rng([seed, 999, repeat]).integers(0, len(target), len(target))
        t, p = target[pick], prediction[pick]
        if t.max() - t.min() <= TIE:
            continue
        groups = random_groups(np.random.default_rng([seed, 888, repeat]), len(pick), group_size)
        values["spearman"].append(spearman(t, p))
        values["pair_acc"].append(pairwise_accuracy(t, p)[0])
        values[f"top1_n{group_size}"].append(top1_on_groups(p, t, groups)[0])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in values.items()}
