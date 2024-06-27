from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.surrogate.causal_surrogate_assisted import CausalSurrogateAssistedTestCase, SimulationResult, Simulator
from causal_testing.surrogate.surrogate_search_algorithms import GeneticSearchAlgorithm
from util.model import Model, OpenAPS, i_label, g_label, s_label

import pandas as pd
import numpy as np
import os
import multiprocessing as mp

import random
from dotenv import load_dotenv


class APSDigitalTwinSimulator(Simulator):
    def __init__(self, constants, profile_path, output_file = "./openaps_temp") -> None:
        super().__init__()

        self.constants = constants
        self.profile_path = profile_path
        self.output_file = output_file

    def is_valid(self, configuration):
        min_bg = 200
        model_control = Model([configuration["start_cob"], 0, 0, configuration["start_bg"], configuration["start_iob"]], self.constants)
        for t in range(1, 121):
            model_control.update(t)
            min_bg = min(min_bg, model_control.history[-1][g_label])

            if min_bg < 50:
                return False

        return True

    def run_with_config(self, configuration) -> SimulationResult:
        if not self.is_valid(configuration):
            return SimulationResult(None, False, None)

        min_bg = 200
        max_bg = 0
        end_bg = 0
        end_cob = 0
        end_iob = 0
        open_aps_output = 0
        violation = False

        open_aps = OpenAPS(profile_path=self.profile_path)
        model_openaps = Model([configuration["start_cob"], 0, 0, configuration["start_bg"], configuration["start_iob"]], self.constants)
        for t in range(1, 121):
            if t % 5 == 1:
                rate = open_aps.run(model_openaps.history, output_file=self.output_file, faulty=False)
                # if rate == -1:
                #     violation = True
                open_aps_output += rate
                for j in range(5):
                    model_openaps.add_intervention(t + j, i_label, rate / 5.0)
            model_openaps.update(t)

            min_bg = min(min_bg, model_openaps.history[-1][g_label])
            max_bg = max(max_bg, model_openaps.history[-1][g_label])

            end_bg = model_openaps.history[-1][g_label]
            end_cob = model_openaps.history[-1][s_label]
            end_iob = model_openaps.history[-1][i_label]

        data = {
            "start_bg": configuration["start_bg"],
            "start_cob": configuration["start_cob"],
            "start_iob": configuration["start_iob"],
            "end_bg": end_bg,
            "end_cob": end_cob,
            "end_iob": end_iob,
            "hypo": min_bg,
            "hyper": max_bg,
            "open_aps_output": open_aps_output,
        }

        violation = max_bg > 200 or min_bg < 50

        return SimulationResult(data, violation, None)
    
    def startup(self, **kwargs):
        return super().startup(**kwargs)
    
    def shutdown(self, **kwargs):
        return super().shutdown(**kwargs)

def main(idx_file):
    idx, file = idx_file
    random.seed(idx)
    np.random.seed(idx)

    constants = []
    const_file_name = file.replace("datasets", "constants").replace("_np_random_random_nonfaulty_scenarios", ".txt")
    with open(const_file_name, "r") as const_file:
        constants = const_file.read().replace("[", "").replace("]", "").split(",")
        constants = [np.float64(const) for const in constants]
        constants[7] = int(constants[7])

    df = pd.read_csv(f"./{file}.csv")

    simulator = APSDigitalTwinSimulator(constants, "./util/profile.json", f"./{file}_openaps_temp")

    res = None
    iter = 0

    for i in range(200):
        candidate_test_case = {
            "start_bg": random.random() * (180 - 70) + 70,
            "start_cob": random.random() * (300 - 100) + 100,
            "start_iob": random.random() * (150 - 0) + 0,
        }

        simulator.startup()
        test_result = simulator.run_with_config(candidate_test_case)
        simulator.shutdown()

        df = df.append(test_result.data, ignore_index=True)

        if test_result.fault and res is None:

            res = test_result
            iter = i + 1
    
    if res == None:
        res = "No fault found"
        iter = 200

    with open(f"./outputs_RS/{file.replace('./datasets/', '')}.txt", "w") as out:
        out.write(str(res) + "\n" + str(iter))
    df.to_csv(f"./outputs_RS/{file.replace('./datasets/', '')}_full.csv")

    print(f"finished {file}")

if __name__ == "__main__":
    load_dotenv()

    all_traces = os.listdir("./datasets")
    all_finished = os.listdir("./outputs_RS")

    pool_vals = []

    for idx, data_trace in enumerate(all_traces):
        if data_trace[:-4] + ".txt" not in all_finished:
            if data_trace.endswith(".csv"):
                if len(pd.read_csv(os.path.join("./datasets", data_trace))) >= 300:
                    pool_vals.append((f"./datasets/{data_trace[:-4]}", idx))

    with mp.Pool(processes=20) as pool:
        pool.map(main, pool_vals)