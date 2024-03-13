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


class CarlaSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def startup(self, **kwargs):
        pass

    def shutdown(self, **kwargs):
        pass

    def run_with_config(self, configuration) -> SimulationResult:
        threshold_criteria = [0,0,0,0,0.95,0]

        res = run_single_scenario([
            configuration["road_type"],
            configuration["road_id"],
            configuration["scenario_length"],
            configuration["vehicle_front"],
            configuration["vehicle_adjacent"],
            configuration["vehicle_opposite"],
            configuration["vehicle_front_two_wheeled"],
            configuration["vehicle_adjacent_two_wheeled"],
            configuration["vehicle_opposite_two_wheeled"],
            configuration["time"],
            configuration["weather"],
            configuration["pedestrian_density"],
            configuration["target_speed"],
            configuration["trees"],
            configuration["buildings"],
            configuration["task"],
        ])

        violations = len([i for i, j in zip(res, threshold_criteria) if i <= j])

        return SimulationResult({
            "road_type": configuration["road_type"],
            "road_id": configuration["road_id"],
            "scenario_length": configuration["scenario_length"],
            "vehicle_front": configuration["vehicle_front"],
            "vehicle_adjacent": configuration["vehicle_adjacent"],
            "vehicle_opposite": configuration["vehicle_opposite"],
            "vehicle_front_two_wheeled": configuration["vehicle_front_two_wheeled"],
            "vehicle_adjacent_two_wheeled": configuration["vehicle_adjacent_two_wheeled"],
            "vehicle_opposite_two_wheeled": configuration["vehicle_opposite_two_wheeled"],
            "time": configuration["time"],
            "weather": configuration["weather"],
            "pedestrian_density": configuration["pedestrian_density"],
            "target_speed": configuration["target_speed"],
            "trees": configuration["trees"],
            "buildings": configuration["buildings"],
            "task": configuration["task"],

            "follow_center": res[0],
            "avoid_vehicles": res[1],
            "avoid_pedestrians": res[2],
            "avoid_static": res[3],
            "abide_rules": res[5],
            "reach_destination": res[4],
        }, violations > 0, None)


def main(seed):
    random.seed(seed)
    np.random.seed(seed)

    simulator = CarlaSimulator()

    df = pd.DataFrame(columns=["road_type",
            "road_id",
            "scenario_length",
            "vehicle_front",
            "vehicle_adjacent",
            "vehicle_opposite",
            "vehicle_front_two_wheeled",
            "vehicle_adjacent_two_wheeled",
            "vehicle_opposite_two_wheeled",
            "time",
            "weather",
            "pedestrian_density",
            "target_speed",
            "trees",
            "buildings",
            "task",

            "follow_center",
            "avoid_vehicles",
            "avoid_pedestrians",
            "avoid_static",
            "abide_rules",
            "reach_destination",])
    res = None

    for i in range(39):

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
            "task": random.randint(0, 3),
        }

        simulator.startup()
        test_result = simulator.run_with_config(candidate_test_case)
        simulator.shutdown()

        df = df.append(test_result.data, ignore_index=True)

    df.to_csv(f"./datasets/{seed}.csv")

    print(f"finished {seed}")

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    all_seeds = range(1,31)
    all_finished = os.listdir("./datasets")

    pool_vals = []

    for data_seed in all_seeds:
        if str(data_seed) + ".csv" not in all_finished:
            main(data_seed)