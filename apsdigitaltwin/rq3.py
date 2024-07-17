import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from scipy.stats import entropy
import seaborn as sns


def euclidian(series, num, df):
    distances_for_series = [np.sqrt(np.sum(np.square(np.subtract(series_1, series_2)))) for series_1, (idx_2, series_2) in zip(np.tile(series, (len(df), 1)), df.iterrows()) if series.name != idx_2]
    return np.min(distances_for_series)

def min_distance(df: pd.DataFrame):
    distances_bg = df[["start_bg"]].apply(euclidian, axis=1, args=(1, df[["start_bg"]]))
    distances_cob = df[["start_cob"]].apply(euclidian, axis=1, args=(1, df[["start_cob"]]))
    distances_iob = df[["start_iob"]].apply(euclidian, axis=1, args=(1, df[["start_iob"]]))
    return {"start_bg": distances_bg, "start_cob": distances_cob, "start_iob": distances_iob}

def entropyf(df):
    entropies = dict()
    for name, col in df.items():
        dist = np.histogram(col, 30)[0]
        uniform = np.ones(len(dist)) / len(dist)
        entropies[name] = entropy(dist, uniform, 2)
    return entropies

def entropy_func(vals):
    df, trace = vals
    entropies = []
    for i in range(len(df)):
        entropies.append(entropyf(df[:i + 1]))
        # entropies.append(min_distance(df[:300 + i + 1]))
    return np.array(entropies), trace

def average_entropy(traces):
    print(f"Entropy for {traces}")
    count = 0
    pool_vals = []
    entropy_vals = []
    for trace in os.listdir(traces):
        if trace.endswith(".csv"):
            df = pd.read_csv(os.path.join(traces, trace))[["start_bg", "start_cob", "start_iob"]]
            if len(df) == 500:
                pool_vals.append((df, trace))
                count += 1

    with mp.Pool(23) as pool:
        res = pool.map(entropy_func, pool_vals)
        for d, trace in res:
            entropy_vals.append(d)

    return entropy_vals

def average_entropy_split(traces):
    print(f"Entropy for {traces}")
    count = 0
    pool_vals = []
    entropy_vals_success = []
    entropy_vals_failure = []
    for trace in os.listdir(traces):
        if trace.endswith(".csv"):
            df = pd.read_csv(os.path.join(traces, trace))[["start_bg", "start_cob", "start_iob"]]
            if len(df) == 500:
                pool_vals.append((df, trace))
                count += 1

    with mp.Pool(23) as pool:
        res = pool.map(entropy_func, pool_vals)
        for d, trace in res:
            file = open(os.path.join(traces, trace[:-9] + ".txt"))
            if file.read().split("\n")[1] == "200":
                entropy_vals_failure.append(d)
            else:
                entropy_vals_success.append(d)

    return entropy_vals_success, entropy_vals_failure

success_ensemble, failure_ensemble = average_entropy_split("output_2_ensemble")
success_hybrid, failure_hybrid = average_entropy_split("outputs_3")

success_ensemble_bg = []
success_ensemble_bg_lower = []
success_ensemble_bg_upper = []
for idx, series in pd.DataFrame(success_ensemble).items():
    if idx < 299:
        continue
    success_ensemble_bg.append(np.median([x["start_bg"] for x in series]))
    success_ensemble_bg_lower.append(np.quantile([x["start_bg"] for x in series], 0.25))
    success_ensemble_bg_upper.append(np.quantile([x["start_bg"] for x in series], 0.75))

success_ensemble_cob = []
success_ensemble_cob_lower = []
success_ensemble_cob_upper = []
for idx, series in pd.DataFrame(success_ensemble).items():
    if idx < 299:
        continue
    success_ensemble_cob.append(np.median([x["start_cob"] for x in series]))
    success_ensemble_cob_lower.append(np.quantile([x["start_cob"] for x in series], 0.25))
    success_ensemble_cob_upper.append(np.quantile([x["start_cob"] for x in series], 0.75))

success_ensemble_iob = []
success_ensemble_iob_lower = []
success_ensemble_iob_upper = []
for idx, series in pd.DataFrame(success_ensemble).items():
    if idx < 299:
        continue
    success_ensemble_iob.append(np.median([x["start_iob"] for x in series]))
    success_ensemble_iob_lower.append(np.quantile([x["start_iob"] for x in series], 0.25))
    success_ensemble_iob_upper.append(np.quantile([x["start_iob"] for x in series], 0.75))

failure_ensemble_bg = []
failure_ensemble_bg_lower = []
failure_ensemble_bg_upper = []
for idx, series in pd.DataFrame(failure_ensemble).items():
    if idx < 299:
        continue
    failure_ensemble_bg.append(np.median([x["start_bg"] for x in series]))
    failure_ensemble_bg_lower.append(np.quantile([x["start_bg"] for x in series], 0.25))
    failure_ensemble_bg_upper.append(np.quantile([x["start_bg"] for x in series], 0.75))

failure_ensemble_cob = []
failure_ensemble_cob_lower = []
failure_ensemble_cob_upper = []
for idx, series in pd.DataFrame(failure_ensemble).items():
    if idx < 299:
        continue
    failure_ensemble_cob.append(np.median([x["start_cob"] for x in series]))
    failure_ensemble_cob_lower.append(np.quantile([x["start_cob"] for x in series], 0.25))
    failure_ensemble_cob_upper.append(np.quantile([x["start_cob"] for x in series], 0.75))

failure_ensemble_iob = []
failure_ensemble_iob_lower = []
failure_ensemble_iob_upper = []
for idx, series in pd.DataFrame(failure_ensemble).items():
    if idx < 299:
        continue
    failure_ensemble_iob.append(np.median([x["start_iob"] for x in series]))
    failure_ensemble_iob_lower.append(np.quantile([x["start_iob"] for x in series], 0.25))
    failure_ensemble_iob_upper.append(np.quantile([x["start_iob"] for x in series], 0.75))

success_hybrid_bg = []
success_hybrid_bg_lower = []
success_hybrid_bg_upper = []
for idx, series in pd.DataFrame(success_hybrid).items():
    if idx < 299:
        continue
    success_hybrid_bg.append(np.median([x["start_bg"] for x in series]))
    success_hybrid_bg_lower.append(np.quantile([x["start_bg"] for x in series], 0.25))
    success_hybrid_bg_upper.append(np.quantile([x["start_bg"] for x in series], 0.75))

success_hybrid_cob = []
success_hybrid_cob_lower = []
success_hybrid_cob_upper = []
for idx, series in pd.DataFrame(success_hybrid).items():
    if idx < 299:
        continue
    success_hybrid_cob.append(np.median([x["start_cob"] for x in series]))
    success_hybrid_cob_lower.append(np.quantile([x["start_cob"] for x in series], 0.25))
    success_hybrid_cob_upper.append(np.quantile([x["start_cob"] for x in series], 0.75))

success_hybrid_iob = []
success_hybrid_iob_lower = []
success_hybrid_iob_upper = []
for idx, series in pd.DataFrame(success_hybrid).items():
    if idx < 299:
        continue
    success_hybrid_iob.append(np.median([x["start_iob"] for x in series]))
    success_hybrid_iob_lower.append(np.quantile([x["start_iob"] for x in series], 0.25))
    success_hybrid_iob_upper.append(np.quantile([x["start_iob"] for x in series], 0.75))

failure_hybrid_bg = []
failure_hybrid_bg_lower = []
failure_hybrid_bg_upper = []
for idx, series in pd.DataFrame(failure_hybrid).items():
    if idx < 299:
        continue
    failure_hybrid_bg.append(np.median([x["start_bg"] for x in series]))
    failure_hybrid_bg_lower.append(np.quantile([x["start_bg"] for x in series], 0.25))
    failure_hybrid_bg_upper.append(np.quantile([x["start_bg"] for x in series], 0.75))

failure_hybrid_cob = []
failure_hybrid_cob_lower = []
failure_hybrid_cob_upper = []
for idx, series in pd.DataFrame(failure_hybrid).items():
    if idx < 299:
        continue
    failure_hybrid_cob.append(np.median([x["start_cob"] for x in series]))
    failure_hybrid_cob_lower.append(np.quantile([x["start_cob"] for x in series], 0.25))
    failure_hybrid_cob_upper.append(np.quantile([x["start_cob"] for x in series], 0.75))

failure_hybrid_iob = []
failure_hybrid_iob_lower = []
failure_hybrid_iob_upper = []
for idx, series in pd.DataFrame(failure_hybrid).items():
    if idx < 299:
        continue
    failure_hybrid_iob.append(np.median([x["start_iob"] for x in series]))
    failure_hybrid_iob_lower.append(np.quantile([x["start_iob"] for x in series], 0.25))
    failure_hybrid_iob_upper.append(np.quantile([x["start_iob"] for x in series], 0.75))

pd.DataFrame({
    "success_ensemble_bg": success_ensemble_bg,
    "success_ensemble_bg_lower": success_ensemble_bg_lower,
    "success_ensemble_bg_upper": success_ensemble_bg_upper,

    "success_ensemble_cob": success_ensemble_cob,
    "success_ensemble_cob_lower": success_ensemble_cob_lower,
    "success_ensemble_cob_upper": success_ensemble_cob_upper,

    "success_ensemble_iob": success_ensemble_iob,
    "success_ensemble_iob_lower": success_ensemble_iob_lower,
    "success_ensemble_iob_upper": success_ensemble_iob_upper,

    "failure_ensemble_bg": failure_ensemble_bg,
    "failure_ensemble_bg_lower": failure_ensemble_bg_lower,
    "failure_ensemble_bg_upper": failure_ensemble_bg_upper,

    "failure_ensemble_cob": failure_ensemble_cob,
    "failure_ensemble_cob_lower": failure_ensemble_cob_lower,
    "failure_ensemble_cob_upper": failure_ensemble_cob_upper,

    "failure_ensemble_iob": failure_ensemble_iob,
    "failure_ensemble_iob_lower": failure_ensemble_iob_lower,
    "failure_ensemble_iob_upper": failure_ensemble_iob_upper,

    "success_hybrid_bg": success_hybrid_bg,
    "success_hybrid_bg_lower": success_hybrid_bg_lower,
    "success_hybrid_bg_upper": success_hybrid_bg_upper,

    "success_hybrid_cob": success_hybrid_cob,
    "success_hybrid_cob_lower": success_hybrid_cob_lower,
    "success_hybrid_cob_upper": success_hybrid_cob_upper,

    "success_hybrid_iob": success_hybrid_iob,
    "success_hybrid_iob_lower": success_hybrid_iob_lower,
    "success_hybrid_iob_upper": success_hybrid_iob_upper,

    "failure_hybrid_bg": failure_hybrid_bg,
    "failure_hybrid_bg_lower": failure_hybrid_bg_lower,
    "failure_hybrid_bg_upper": failure_hybrid_bg_upper,

    "failure_hybrid_cob": failure_hybrid_cob,
    "failure_hybrid_cob_lower": failure_hybrid_cob_lower,
    "failure_hybrid_cob_upper": failure_hybrid_cob_upper,

    "failure_hybrid_iob": failure_hybrid_iob,
    "failure_hybrid_iob_lower": failure_hybrid_iob_lower,
    "failure_hybrid_iob_upper": failure_hybrid_iob_upper,
}).to_csv("test_entropy_df.csv")
    
fig, axes = plt.subplots(1, 2, sharey=True)

sns.lineplot({"Initial BG": np.divide(np.array(success_ensemble_bg) + np.array(failure_ensemble_bg), 2),
              "Initial COB": np.divide(np.array(success_ensemble_cob) + np.array(failure_ensemble_cob), 2),
              "Initial IOB": np.divide(np.array(success_ensemble_iob) + np.array(failure_ensemble_iob), 2),
              }, ax=axes[0])

axes[0].fill_between(range(len(success_ensemble_bg)), 
                     np.divide(np.array(success_ensemble_bg_lower) + np.array(failure_ensemble_bg_lower), 2), 
                     np.divide(np.array(success_ensemble_bg_upper) + np.array(failure_ensemble_bg_upper), 2), alpha=0.2)

axes[0].fill_between(range(len(success_ensemble_cob)), 
                     np.divide(np.array(success_ensemble_cob_lower) + np.array(failure_ensemble_cob_lower), 2), 
                     np.divide(np.array(success_ensemble_cob_upper) + np.array(failure_ensemble_cob_upper), 2), alpha=0.2)
axes[0].fill_between(range(len(success_ensemble_iob)), 
                     np.divide(np.array(success_ensemble_iob_lower) + np.array(failure_ensemble_iob_lower), 2), 
                     np.divide(np.array(success_ensemble_iob_upper) + np.array(failure_ensemble_iob_upper), 2), alpha=0.2)

sns.lineplot({"Initial BG": np.divide(np.array(success_hybrid_bg) + np.array(failure_hybrid_bg), 2),
              "Initial COB": np.divide(np.array(success_hybrid_cob) + np.array(failure_hybrid_cob), 2),
              "Initial IOB": np.divide(np.array(success_hybrid_iob) + np.array(failure_hybrid_iob), 2)}, ax=axes[1])

axes[1].fill_between(range(len(success_hybrid_bg)), 
                     np.divide(np.array(success_hybrid_bg_lower) + np.array(failure_hybrid_bg_lower), 2), 
                     np.divide(np.array(success_hybrid_bg_upper) + np.array(failure_hybrid_bg_upper), 2), alpha=0.2)
axes[1].fill_between(range(len(success_hybrid_cob)), 
                     np.divide(np.array(success_hybrid_cob_lower) + np.array(failure_hybrid_cob_lower), 2), 
                     np.divide(np.array(success_hybrid_cob_upper) + np.array(failure_hybrid_cob_upper), 2), alpha=0.2)
axes[1].fill_between(range(len(success_hybrid_iob)), 
                     np.divide(np.array(success_hybrid_iob_lower) + np.array(failure_hybrid_iob_lower), 2), 
                     np.divide(np.array(success_hybrid_iob_upper) + np.array(failure_hybrid_iob_upper), 2), alpha=0.2)

fig.set_size_inches(20, 5)

for ax in axes:
    ax.set_ylim([0, 1.2])
    ax.set_ylabel("Test Set KL Divergence from Uniform")
    ax.set_xlabel("Search Iterations")
    ax.get_xaxis().label.set_fontsize(14)
    ax.get_yaxis().label.set_fontsize(14)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(12)

axes[0].set_title(f"Ensemble Test Set Divergence", fontsize=16)
axes[1].set_title(f"Causally-Assisted Test Set Divergence", fontsize=16)

fig.tight_layout()
fig.savefig("./figures/RQ3.png")

plt.show()