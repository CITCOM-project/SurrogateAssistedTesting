import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

random = []

for trace in os.listdir("./output_random"):
    if trace.endswith(".csv"):
        df = pd.read_csv(os.path.join("output_random", trace))
        if len(df) < 500:
            random.append(len(df) - 300)

ensemble = []

for trace in os.listdir("./output_2_ensemble"):
    if trace.endswith(".csv"):
        df = pd.read_csv(os.path.join("output_2_ensemble", trace))
        if len(df) < 500:
            ensemble.append(len(df) - 300)

hybrid = []

for trace in os.listdir("./outputs_4"):
    if trace.endswith(".txt"):
        file = open(os.path.join("outputs_4", trace))
        num = int(file.read().split("\n")[1])
        if num < 200:
            hybrid.append(num)

random_cumulative = [0]
for i in range(200):
    random_cumulative.append(len([x for x in random if x <= i]))

ensemble_cumulative = [0]
for i in range(200):
    ensemble_cumulative.append(len([x for x in ensemble if x <= i]))

hybrid_cumulative = [0]
for i in range(200):
    hybrid_cumulative.append(len([x for x in hybrid if x <= i]))

sns.lineplot({"Hybrid": hybrid_cumulative,
              "Ensemble": ensemble_cumulative,
              "Random": random_cumulative})

plt.title("Search Iterations to find a Violation")
plt.xlabel("Search Iterations")
plt.ylabel("Traces with Violations Found")
plt.legend()
plt.gca().xaxis.label.set_fontsize(14)
plt.gca().yaxis.label.set_fontsize(14)
plt.gca().title.set_fontsize(18)
for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    item.set_fontsize(12)

plt.savefig("./figures/RQ2.png")
plt.show()