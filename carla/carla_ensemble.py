from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_dag import CausalDAG
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.specification.scenario import Scenario
from causal_testing.specification.variable import Input, Output
from causal_testing.surrogate.causal_surrogate_assisted import SimulationResult, Simulator
from util.causal_surrogate_assisted_ensemble import CausalSurrogateAssistedTestCase
from util.surrogate_search_algorithms_new import GeneticEnembleSearchAlgorithm

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


def main(i):
    random.seed(i)
    np.random.seed(i)

    search_bias = Input("search_bias", float, hidden=True)
    
    road_type = Input("road_type", int)
    road_id = Input("road_id", int)
    scenario_length = Input("scenario_length", int)
    vehicle_front = Input("vehicle_front", int)
    vehicle_adjacent = Input("vehicle_adjacent", int)
    vehicle_opposite = Input("vehicle_opposite", int)
    vehicle_front_two_wheeled = Input("vehicle_front_two_wheeled", int)
    vehicle_adjacent_two_wheeled = Input("vehicle_adjacent_two_wheeled", int)
    vehicle_opposite_two_wheeled = Input("vehicle_opposite_two_wheeled", int)
    time = Input("time", int)
    weather = Input("weather", int)
    pedestrian_density = Input("pedestrian_density", int)
    target_speed = Input("target_speed", int)
    trees = Input("trees", int)
    buildings = Input("buildings", int)
    task = Input("task", int)

    follow_center = Output("follow_center", float)
    avoid_vehicles = Output("avoid_vehicles", float)
    avoid_pedestrians = Output("avoid_pedestrians", float)
    avoid_static = Output("avoid_static", float)
    abide_rules = Output("abide_rules", float)
    reach_destination = Output("reach_destination", float)

    constraints = {
        road_type >= 0, road_type <= 3,
        road_id >= 0, road_id <= 3,
        scenario_length >= 0, scenario_length <= 0,
        vehicle_front >= 0, vehicle_front <= 1,
        vehicle_adjacent >= 0, vehicle_adjacent <= 1,
        vehicle_opposite >= 0, vehicle_opposite <= 1,
        vehicle_front_two_wheeled >= 0, vehicle_front_two_wheeled <= 1,
        vehicle_adjacent_two_wheeled >= 0, vehicle_adjacent_two_wheeled <= 1,
        vehicle_opposite_two_wheeled >= 0, vehicle_opposite_two_wheeled <= 1,
        time >= 0, time <= 2,
        weather >= 0, weather <= 6,
        pedestrian_density >= 0, pedestrian_density <= 1,
        target_speed >= 2, target_speed <= 4,
        trees >= 0, trees <= 1,
        buildings >= 0, buildings <= 1,
        task >= 0, task <= 3,
    }

    scenario = Scenario(
        variables={
            search_bias,

            road_type,
            road_id,
            scenario_length,
            vehicle_front,
            vehicle_adjacent,
            vehicle_opposite,
            vehicle_front_two_wheeled,
            vehicle_adjacent_two_wheeled,
            vehicle_opposite_two_wheeled,
            time,
            weather,
            pedestrian_density,
            target_speed,
            trees,
            buildings,
            task,

            follow_center,
            avoid_vehicles,
            avoid_pedestrians,
            avoid_static,
            abide_rules,
            reach_destination,
        },
        constraints = constraints
    )

    dag = CausalDAG("./dag.dot")
    specification = CausalSpecification(scenario, dag)

    ga_config = {
        "parent_selection_type": "tournament",
        "K_tournament": 4,
        "mutation_type": "random",
        "mutation_percent_genes": 50,
        "mutation_by_replacement": True,
    }

    ga_search_ensemble = GeneticEnembleSearchAlgorithm(config=ga_config)

    simulator = CarlaSimulator()

    data_collector = ObservationalDataCollector(scenario, pd.read_csv(os.path.join("datasets", str(i) + ".csv")))
    test_case = CausalSurrogateAssistedTestCase(specification, ga_search_ensemble, simulator)

    res, iter, df = test_case.execute(data_collector, 60)
    with open(f"./outputs_ensemble/{i}.txt", "w") as out:
        out.write(str(res) + "\n" + str(iter))
    df.to_csv(f"./outputs_ensemble/{i}_full.csv")

    print(f"finished {i}")

if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    all_seeds = range(1,31)
    all_finished = os.listdir("./outputs_ensemble")

    pool_vals = []

    for data_seed in all_seeds:
        if str(data_seed) + ".txt" not in all_finished:
            main(data_seed)