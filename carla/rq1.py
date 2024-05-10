import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import multiprocessing as mp
from scipy.stats import entropy
import seaborn as sns

# def euclidian(series, num, df):
#     distances_for_series = [np.sqrt(np.sum(np.square(np.subtract(series_1, series_2)))) for series_1, (idx_2, series_2) in zip(np.tile(series, (len(df), 1)), df.iterrows()) if series.name != idx_2]
#     return np.min(distances_for_series)

# def min_distance(df: pd.DataFrame):
#     distances = df.apply(euclidian, axis=1, args=(1, df))
#     return np.mean(distances)

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
            df = pd.read_csv(os.path.join(traces, trace))[["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]]
            if len(df) == 99:
                pool_vals.append((df, trace))
                count += 1

    with mp.Pool(23) as pool:
        res = pool.map(entropy_func, pool_vals)
        for d, trace in res:
            entropy_vals.append(d)
            # plt.plot(d, c="r", linewidth=1)
            # plt.gca().set_xlim([300, 500])
            # plt.gca().set_ylim([0, 1])
            # plt.title("Test Set Entropy")
            # plt.xlabel("Search Iterations")
            # plt.ylabel("Test Set Diameter")
            # plt.gca().title.set_fontsize(18)
            # plt.gca().xaxis.label.set_fontsize(14)
            # plt.gca().yaxis.label.set_fontsize(14)
            # for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
            #     item.set_fontsize(12)
            # plt.savefig(f"./graph_outputs/{trace[:-4]}.png")
            # plt.clf()

    return entropy_vals

# def average_entropy_split(traces):
#     count = 0
#     pool_vals = []
#     entropy_vals_success = []
#     entropy_vals_failure = []
#     for trace in os.listdir(traces):
#         if trace.endswith(".csv"):
#             df = pd.read_csv(os.path.join(traces, trace))[["start_bg", "start_cob", "start_iob"]]
#             if len(df) == 500:
#                 pool_vals.append((df, trace))
#                 count += 1

#     with mp.Pool(23) as pool:
#         res = pool.map(entropy_func, pool_vals)
#         for d, trace in res:
#             file = open(os.path.join(traces, trace[:-9] + ".txt"))
#             if file.read().split("\n")[1] == "200":
#                 entropy_vals_failure.append(d)
#             else:
#                 entropy_vals_success.append(d)

#     return entropy_vals_success, entropy_vals_failure

ensemble = average_entropy("outputs_ensemble")
hybrid = average_entropy("outputs_hybrid")
# failure_hybrid = average_entropy("outputs_3")

entropy_dict_ensemble = dict()

for input_var in ["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]:
    medians = []
    lowers = []
    uppers = []
    for idx, series in pd.DataFrame(ensemble).items():
        if idx < 39:
            continue
        medians.append(np.median([x[input_var] for x in series]))
        lowers.append(np.quantile([x[input_var] for x in series], 0.25))
        uppers.append(np.quantile([x[input_var] for x in series], 0.75))
    entropy_dict_ensemble[input_var] = {"medians": medians, "lowers": lowers, "uppers": uppers}

entropy_dict_hybrid = dict()

for input_var in ["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]:
    medians = []
    lowers = []
    uppers = []
    for idx, series in pd.DataFrame(hybrid).items():
        if idx < 39:
            continue
        medians.append(np.median([x[input_var] for x in series]))
        lowers.append(np.quantile([x[input_var] for x in series], 0.25))
        uppers.append(np.quantile([x[input_var] for x in series], 0.75))
    entropy_dict_hybrid[input_var] = {"medians": medians, "lowers": lowers, "uppers": uppers}

    
fig, axes = plt.subplots(1, 2, sharey=True)

sns.lineplot({"Ensemble road_type": entropy_dict_ensemble["road_type"]["medians"],
              "Ensemble road_id": entropy_dict_ensemble["road_id"]["medians"],
              "Ensemble scenario_length": entropy_dict_ensemble["scenario_length"]["medians"],
              "Ensemble vehicle_front": entropy_dict_ensemble["vehicle_front"]["medians"],
              "Ensemble vehicle_adjacent": entropy_dict_ensemble["vehicle_adjacent"]["medians"],
              "Ensemble vehicle_opposite": entropy_dict_ensemble["vehicle_opposite"]["medians"],
              "Ensemble vehicle_front_two_wheeled": entropy_dict_ensemble["vehicle_front_two_wheeled"]["medians"],
              "Ensemble vehicle_adjacent_two_wheeled": entropy_dict_ensemble["vehicle_adjacent_two_wheeled"]["medians"],
              "Ensemble vehicle_opposite_two_wheeled": entropy_dict_ensemble["vehicle_opposite_two_wheeled"]["medians"],
              "Ensemble time": entropy_dict_ensemble["time"]["medians"],
              "Ensemble weather": entropy_dict_ensemble["weather"]["medians"],
              "Ensemble pedestrian_density": entropy_dict_ensemble["pedestrian_density"]["medians"],
              "Ensemble target_speed": entropy_dict_ensemble["target_speed"]["medians"],
              "Ensemble trees": entropy_dict_ensemble["trees"]["medians"],
              "Ensemble buildings": entropy_dict_ensemble["buildings"]["medians"],
              "Ensemble task": entropy_dict_ensemble["task"]["medians"],
              }, ax=axes[0])

for input_var in ["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]:
    axes[0].fill_between(range(len(entropy_dict_ensemble[input_var]["medians"])),
                         entropy_dict_ensemble[input_var]["lowers"], entropy_dict_ensemble[input_var]["uppers"],
                         alpha = 0.2)

sns.lineplot({"Hybrid road_type": entropy_dict_hybrid["road_type"]["medians"],
              "Hybrid road_id": entropy_dict_hybrid["road_id"]["medians"],
              "Hybrid scenario_length": entropy_dict_hybrid["scenario_length"]["medians"],
              "Hybrid vehicle_front": entropy_dict_hybrid["vehicle_front"]["medians"],
              "Hybrid vehicle_adjacent": entropy_dict_hybrid["vehicle_adjacent"]["medians"],
              "Hybrid vehicle_opposite": entropy_dict_hybrid["vehicle_opposite"]["medians"],
              "Hybrid vehicle_front_two_wheeled": entropy_dict_hybrid["vehicle_front_two_wheeled"]["medians"],
              "Hybrid vehicle_adjacent_two_wheeled": entropy_dict_hybrid["vehicle_adjacent_two_wheeled"]["medians"],
              "Hybrid vehicle_opposite_two_wheeled": entropy_dict_hybrid["vehicle_opposite_two_wheeled"]["medians"],
              "Hybrid time": entropy_dict_hybrid["time"]["medians"],
              "Hybrid weather": entropy_dict_hybrid["weather"]["medians"],
              "Hybrid pedestrian_density": entropy_dict_hybrid["pedestrian_density"]["medians"],
              "Hybrid target_speed": entropy_dict_hybrid["target_speed"]["medians"],
              "Hybrid trees": entropy_dict_hybrid["trees"]["medians"],
              "Hybrid buildings": entropy_dict_hybrid["buildings"]["medians"],
              "Hybrid task": entropy_dict_hybrid["task"]["medians"],
              }, ax=axes[1])

for input_var in ["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]:
    axes[1].fill_between(range(len(entropy_dict_hybrid[input_var]["medians"])),
                         entropy_dict_hybrid[input_var]["lowers"], entropy_dict_hybrid[input_var]["uppers"],
                         alpha = 0.2)

fig.set_size_inches(20, 10)

for ax in axes:
    ax.set_title(f"Test Set Entropy throughout Search", fontsize=16)
    ax.set_ylabel("Test Set KL Divergence from Uniform")
    ax.set_xlabel("Search Iterations")
    ax.get_xaxis().label.set_fontsize(14)
    ax.get_yaxis().label.set_fontsize(14)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(12)

fig.tight_layout()
fig.savefig("./figures/RQ1.png")

plt.show()

fig, axes = plt.subplots(2, 3)

for idx, marker in enumerate(["road_id", "weather", "road_type", "time", "target_speed", "task"]):
    sns.lineplot({
                f"Ensemble": entropy_dict_ensemble[marker]["medians"],
                f"Hybrid": entropy_dict_hybrid[marker]["medians"]
                }, ax=axes[idx//3][idx%3])
                

    axes[idx//3][idx%3].fill_between(range(len(entropy_dict_ensemble[marker]["medians"])),
                        entropy_dict_ensemble[marker]["lowers"], entropy_dict_ensemble[marker]["uppers"],
                        alpha = 0.2)
    
    axes[idx//3][idx%3].fill_between(range(len(entropy_dict_hybrid[marker]["medians"])),
                        entropy_dict_hybrid[marker]["lowers"], entropy_dict_hybrid[marker]["uppers"],
                        alpha = 0.2)
    
    axes[idx//3][idx%3].set_title(f"{marker}", fontsize=14)
    
fig.set_size_inches(12, 8)
for axe in axes:
    for ax in axe:
        # ax.set_title(f"Test Set KL Divergence from Uniform throughout Search", fontsize=16)
        ax.set_ylabel("Test Set KL Divergence from Uniform")
        ax.set_xlabel("Search Iterations")
        ax.get_xaxis().label.set_fontsize(14)
        ax.get_yaxis().label.set_fontsize(14)
        ax.legend(loc="upper left")
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(12)

fig.tight_layout()
fig.savefig("./figures/RQ1_a.png")

plt.show()

fig, axes = plt.subplots(4, 4)

for idx, marker in enumerate(["road_type", "road_id", "scenario_length", "vehicle_front", "vehicle_adjacent", 
                                "vehicle_opposite", "vehicle_front_two_wheeled", "vehicle_adjacent_two_wheeled", 
                                "vehicle_opposite_two_wheeled", "time", "weather", "pedestrian_density", 
                                "target_speed", "trees", "buildings", "task"]):
    sns.lineplot({
                f"Ensemble {marker}": entropy_dict_ensemble[marker]["medians"],
                f"Hybrid {marker}": entropy_dict_hybrid[marker]["medians"]
                }, ax=axes[idx//4][idx%4])
                

    axes[idx//4][idx%4].fill_between(range(len(entropy_dict_ensemble[marker]["medians"])),
                        entropy_dict_ensemble[marker]["lowers"], entropy_dict_ensemble[marker]["uppers"],
                        alpha = 0.2)
    
    axes[idx//4][idx%4].fill_between(range(len(entropy_dict_hybrid[marker]["medians"])),
                        entropy_dict_hybrid[marker]["lowers"], entropy_dict_hybrid[marker]["uppers"],
                        alpha = 0.2)
    
fig.set_size_inches(20, 20)
for axe in axes:
    for ax in axe:
    # ax.set_title(f"Test Set KL Divergence from Uniform throughout Search", fontsize=16)
        ax.set_ylabel("Test Set KL Divergence from Uniform")
        ax.set_xlabel("Search Iterations")
        ax.get_xaxis().label.set_fontsize(14)
        ax.get_yaxis().label.set_fontsize(14)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(12)

fig.tight_layout()
fig.savefig("./figures/RQ1_suplimental.png")

plt.show()