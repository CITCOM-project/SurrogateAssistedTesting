import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import os

ensemble = []

for trace in os.listdir("./outputs_ensemble"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_ensemble", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            ensemble.append(int(num))

hybrid = []

for trace in os.listdir("./outputs_hybrid"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_hybrid", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            hybrid.append(int(num))

ensemble_dists = []
hybrid_dists = []
np.random.seed(123)
for i in range(9, 200, 10):
    ensemble_dist = [1 if x <= i else 0 for x in ensemble]
    hybrid_dist = [1 if x <= i else 0 for x in hybrid]

    np.random.shuffle(ensemble_dist)
    np.random.shuffle(hybrid_dist)
    ensemble_dists.append(ensemble_dist)
    hybrid_dists.append(hybrid_dist)

distribution_pairs = list(zip(ensemble_dists, hybrid_dists))

p_values = []

for e_dist, h_dist in distribution_pairs:
    ensemble_samples = []
    hybrid_samples = []

    for i in range(28):
        ensemble_samples.append(np.average(e_dist[i * 33: (i * 33) + 33]))
        hybrid_samples.append(np.average(h_dist[i * 33: (i * 33) + 33]))

    print(
        np.mean(ensemble_samples), 
        np.mean(hybrid_samples), 

        stats.shapiro(ensemble_samples).pvalue, 
        stats.shapiro(hybrid_samples).pvalue, 
          
        stats.mannwhitneyu(ensemble_samples, hybrid_samples).pvalue
        )