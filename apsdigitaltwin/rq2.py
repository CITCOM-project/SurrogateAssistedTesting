import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import numpy as np

random = []

for trace in os.listdir("./outputs_RS"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_RS", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            random.append(int(num))

ensemble = []

for trace in os.listdir("./outputs_2_ensemble"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_2_ensemble", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            ensemble.append(int(num))

hybrid = []

for trace in os.listdir("./outputs_3"):
    if trace.endswith(".txt"):
        with open(os.path.join("outputs_3", trace), "r") as file:
            output = file.read()
            num = output.split("\n")[1]
            hybrid.append(int(num))

random_cumulative = []
for i in range(200):
    random_cumulative.append(len([x for x in random if x <= i]))

ensemble_cumulative = []
for i in range(200):
    ensemble_cumulative.append(len([x for x in ensemble if x <= i]))

hybrid_cumulative = []
for i in range(200):
    hybrid_cumulative.append(len([x for x in hybrid if x <= i]))

sns.lineplot({"Causally-Assisted": hybrid_cumulative,
              "Ensemble": ensemble_cumulative,
              "Random": random_cumulative})

plt.gcf().set_size_inches(10, 4)

plt.title("Search Iterations to find a Violation")
plt.xlabel("Search Iterations")
plt.ylabel("Traces with Violations Found")
plt.legend()
plt.gca().xaxis.label.set_fontsize(14)
plt.gca().yaxis.label.set_fontsize(14)
plt.gca().title.set_fontsize(18)
for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    item.set_fontsize(12)

plt.tight_layout()
plt.savefig("./figures/RQ2.png")
plt.show()