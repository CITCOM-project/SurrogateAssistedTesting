from util.carla_interface import run_single_scenario
from util.causal_surrogate_assisted_3 import SimulationResult, Simulator


class CarlaSimulator(Simulator):
    def __init__(self) -> None:
        super().__init__()

    def startup(self, **kwargs):
        pass

    def shutdown(self, **kwargs):
        pass

    def run_with_config(self, configuration) -> SimulationResult:
        threshold_criteria = [0.95,0,0,0,0,0]

        res = (1000,1000,1000,1000,1000,1000)

        while res[0] == 1000:
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