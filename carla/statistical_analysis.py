import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats

import os

def chi_square(observed, expected):
    return pow((observed - expected), 2) / expected

ensemble = []

for trace in os.listdir("./outputs_ensemble_formatted"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_ensemble_formatted", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            ensemble.append(int(num))

hybrid = []

for trace in os.listdir("./outputs_hybrid_formatted"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_hybrid_formatted", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            hybrid.append(int(num))
            
ensemble_dists = []
hybrid_dists = []
for i in range(1, 4):
    ensemble_dist = [1 if x <= i else 0 for x in ensemble]
    hybrid_dist = [1 if x <= i else 0 for x in hybrid]

    ensemble_dists.append(ensemble_dist)
    hybrid_dists.append(hybrid_dist)
    
distribution_pairs = list(zip(ensemble_dists, hybrid_dists))

for idx, (e_dist, h_dist) in enumerate(distribution_pairs):
    found_e = len([x for x in e_dist if x == 1])
    found_h = len([x for x in h_dist if x == 1])
    
    missed_e = 30 - found_e
    missed_h = 30 - found_h
    
    total_found = found_e + found_h
    total_missed = missed_e + missed_h
    
    overall_total = total_found + total_missed
    
    found_expected = (total_found * 30) / overall_total
    missed_expected = (total_missed * 30) / overall_total
    
    print(idx + 1, "&",
          found_e, "&",
          found_h, "&",
          "%.5f" % (chi_square(found_e, found_expected) + chi_square(found_h, found_expected) + chi_square(missed_e, missed_expected) + chi_square(missed_h, missed_expected)), "\\\\")