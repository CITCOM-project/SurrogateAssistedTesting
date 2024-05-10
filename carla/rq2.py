import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

random = []

for trace in os.listdir("./outputs_random_reinterpreted"):
    if trace.endswith(".txt"):
        with open(os.path.join("./outputs_random", trace)) as file:
            num = file.readlines()[1]
            if num != "200":
                random.append(int(num))

ensemble = []

for trace in os.listdir("./outputs_ensemble_reinterpreted"):
    if trace.endswith(".txt"):
        with open(os.path.join("./outputs_ensemble_reinterpreted", trace)) as file:
            num = file.readlines()[1]
            if num != "200":
                ensemble.append(int(num))

hybrid = []

for trace in os.listdir("./outputs_hybrid_reinterpreted"):
    if trace.endswith(".txt"):
        with open(os.path.join("./outputs_hybrid_reinterpreted", trace)) as file:
            num = file.readlines()[1]
            if num != "200":
                hybrid.append(int(num))

random_cumulative = [0]
for i in range(5):
    random_cumulative.append(len([x for x in random if x <= i]))

ensemble_cumulative = [0]
for i in range(5):
    ensemble_cumulative.append(len([x for x in ensemble if x <= i]))
    print(len([x for x in ensemble if x <= i]))

hybrid_cumulative = [0]
for i in range(5):
    hybrid_cumulative.append(len([x for x in hybrid if x <= i]))

sns.lineplot({"Causally-Assisted": hybrid_cumulative,
              "Ensemble": ensemble_cumulative,
              "Random": random_cumulative})

plt.gcf().set_size_inches(10, 4)

# plt.plot(random_cumulative, c="g", label="Random Search")
# plt.plot(ensemble_cumulative, c="r", label="Associative Search")
# plt.plot(hybrid_cumulative, c="b", label="Hybrid Causal Search")
plt.title("Search Iterations to find a Violation")
plt.xlabel("Search Iterations")
plt.ylabel("Traces with Violations Found")
plt.legend()
plt.gca().xaxis.label.set_fontsize(14)
plt.gca().yaxis.label.set_fontsize(14)
plt.gca().title.set_fontsize(18)
for item in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
    item.set_fontsize(12)

plt.gcf().tight_layout()
plt.savefig("./figures/RQ2.png")
plt.show()