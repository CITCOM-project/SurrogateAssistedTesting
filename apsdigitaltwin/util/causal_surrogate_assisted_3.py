from causal_testing.data_collection.data_collector import ObservationalDataCollector
from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.base_test_case import BaseTestCase
from causal_testing.testing.estimators import Estimator, PolynomialRegressionEstimator

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class SimulationResult:
    data: dict
    fault: bool
    relationship: str


@dataclass
class SearchFitnessFunction:
    fitness_function: Any
    surrogate_model: PolynomialRegressionEstimator


class SearchAlgorithm:
    def generate_fitness_functions(self, surrogate_models: list[Estimator]) -> list[SearchFitnessFunction]:
        pass

    def search(self, fitness_functions: list[SearchFitnessFunction], specification: CausalSpecification) -> list:
        pass


class Simulator:
    def startup(self, **kwargs):
        pass

    def shutdown(self, **kwargs):
        pass

    def run_with_config(self, configuration) -> SimulationResult:
        pass


class CausalSurrogateAssistedTestCase:
    def __init__(
        self,
        specification: CausalSpecification,
        search_algorithm: SearchAlgorithm,
        search_algorithm_ensemble: SearchAlgorithm,
        simulator: Simulator,
    ):
        self.specification = specification
        self.search_algorithm = search_algorithm
        self.search_algorithm_ensemble = search_algorithm_ensemble
        self.simulator = simulator

    def execute(
        self,
        data_collector: ObservationalDataCollector,
        max_executions: int = 200,
        custom_data_aggregator: Callable[[dict, dict], dict] = None,
    ):
        data_collector.collect_data()

        for i in range(max_executions):
            candidate_test_case = None
            if i % 2 == 0:
                surrogate_model = self.search_algorithm_ensemble.generate_ensemble(data_collector.data)
                fitness_functions = self.search_algorithm_ensemble.generate_fitness_functions(surrogate_model)
                candidate_test_case, _fitness, surrogate = self.search_algorithm_ensemble.search(
                    fitness_functions, self.specification
                )
            else:
                surrogate_models = self.generate_surrogates(self.specification, data_collector)
                fitness_functions = self.search_algorithm.generate_fitness_functions(surrogate_models)
                candidate_test_case, _fitness, surrogate = self.search_algorithm.search(
                    fitness_functions, self.specification
                )


            self.simulator.startup()
            test_result = self.simulator.run_with_config(candidate_test_case)
            self.simulator.shutdown()

            if custom_data_aggregator is not None:
                if data_collector.data is not None:
                    data_collector.data = custom_data_aggregator(data_collector.data, test_result.data)
            else:
                data_collector.data = data_collector.data.append(test_result.data, ignore_index=True)

            if test_result.fault:
                test_result.relationship = ""
                return test_result, i + 1, data_collector.data


        print("No fault found")
        return "No fault found", i + 1, data_collector.data

    def generate_surrogates(
        self, specification: CausalSpecification, data_collector: ObservationalDataCollector
    ) -> list[SearchFitnessFunction]:
        surrogate_models = []

        for u, v in specification.causal_dag.graph.edges:
            edge_metadata = specification.causal_dag.graph.adj[u][v]
            if "included" in edge_metadata:
                from_var = specification.scenario.variables.get(u)
                to_var = specification.scenario.variables.get(v)
                base_test_case = BaseTestCase(from_var, to_var)

                minimal_adjustment_set = specification.causal_dag.identification(base_test_case, specification.scenario)

                surrogate = PolynomialRegressionEstimator(
                    u,
                    0,
                    0,
                    minimal_adjustment_set,
                    v,
                    4,
                    df=data_collector.data,
                    expected_relationship=edge_metadata["expected"],
                )
                surrogate_models.append(surrogate)

        return surrogate_models