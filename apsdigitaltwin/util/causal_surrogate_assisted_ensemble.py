from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm, Simulator

from typing import Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SimulationResult:
    """Data class holding the data and result metadata of a simulation"""

    data: dict
    fault: bool
    relationship: str


class Simulator(ABC):
    """Class to be inherited with Simulator specific functions to start, shutdown and run the simulation with the give
    config file"""

    @abstractmethod
    def startup(self, **kwargs):
        """Function that when run, initialises and opens the Simulator"""

    @abstractmethod
    def shutdown(self, **kwargs):
        """Function to safely exit and shutdown the Simulator"""

    @abstractmethod
    def run_with_config(self, configuration: dict) -> SimulationResult:
        """Run the simulator with the given configuration and return the results in the structure of a
        SimulationResult
        :param configuration: The configuration required to initialise the Simulation
        :return: Simulation results in the structure of the SimulationResult data class"""


class CausalSurrogateAssistedTestCase:
    def __init__(
        self,
        specification: CausalSpecification,
        search_algorithm: SearchAlgorithm,
        simulator: Simulator,
    ):
        self.specification = specification
        self.search_algorithm = search_algorithm
        self.simulator = simulator

    def execute(
        self,
        data_collector: ObservationalDataCollector,
        max_executions: int = 200,
        custom_data_aggregator: Callable[[dict, dict], dict] = None,
    ):
        data_collector.collect_data()

        res = None
        iter = 200

        for i in range(max_executions):
            surrogate_model = self.search_algorithm.generate_ensemble(data_collector.data)
            candidate_test_case, _fitness, surrogate = self.search_algorithm.search(
                surrogate_model, self.specification
            )

            self.simulator.startup()
            test_result = self.simulator.run_with_config(candidate_test_case)
            self.simulator.shutdown()

            if custom_data_aggregator is not None:
                if data_collector.data is not None:
                    data_collector.data = custom_data_aggregator(data_collector.data, test_result.data)
            else:
                data_collector.data = data_collector.data.append(test_result.data, ignore_index=True)

            # if test_result.fault:
            #     test_result.relationship = ""
            #     return test_result, i + 1, data_collector.data
                
            if test_result.fault and res is None:
                res = test_result
                iter = i + 1

        if res == None:
            res = "No fault found"
            iter = 200

        print(res)
        return res, iter, data_collector.data