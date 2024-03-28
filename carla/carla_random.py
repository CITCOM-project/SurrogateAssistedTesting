from util.carla_simulator import CarlaSimulator
from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from util.causal_surrogate_assisted_3 import CausalSurrogateAssistedTestCase, SimulationResult, Simulator
from util.surrogate_search_algorithms_new import GeneticEnembleSearchAlgorithm, GeneticMultiProcessSearchAlgorithm

from util.carla_interface import run_single_scenario

import random
import numpy as np
import os
import multiprocessing as mp
import tensorflow as tf
import pandas as pd


def main(seed):
    random.seed(seed)
    seeded_seed = random.randint(0, 9999999)
    random.seed(seeded_seed)
    np.random.seed(seeded_seed)

    simulator = CarlaSimulator()

    df = pd.read_csv(os.path.join("datasets", str(seed) + ".csv"))
    res = None

    for i in range(60):
        candidate_test_case = {
            "road_type": random.randint(0, 3),
            "road_id": random.randint(0, 3),
            "scenario_length": 0,
            "vehicle_front": random.randint(0, 1),
            "vehicle_adjacent": random.randint(0, 1),
            "vehicle_opposite": random.randint(0, 1),
            "vehicle_front_two_wheeled": random.randint(0, 1),
            "vehicle_adjacent_two_wheeled": random.randint(0, 1),
            "vehicle_opposite_two_wheeled": random.randint(0, 1),
            "time": random.randint(0, 2),
            "weather": random.randint(0, 6),
            "pedestrian_density": random.randint(0, 1),
            "target_speed": random.randint(2, 4),
            "trees": random.randint(0, 1),
            "buildings": random.randint(0, 1),
            "task": random.randint(0, 2),
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

    with open(f"./outputs_RS/{seed}.txt", "w") as out:
        out.write(str(res) + "\n" + str(iter))
    df.to_csv(f"./outputs_RS/{seed}_full.csv")

    print(f"finished {seed}")

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    all_seeds = range(1,31)
    all_finished = os.listdir("./outputs_RS")

    pool_vals = []

    for data_seed in all_seeds:
        if str(data_seed) + ".txt" not in all_finished:
            main(data_seed)