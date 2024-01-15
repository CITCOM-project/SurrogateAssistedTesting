from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import Estimator, PolynomialRegressionEstimator
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm, SearchFitnessFunction
from causal_testing.surrogate.surrogate_search_algorithms import GeneticSearchAlgorithm
from util.ensemble_util import Ensemble

from pygad import GA
from operator import itemgetter
import multiprocessing as mp
import warnings


class GeneticEnembleSearchAlgorithm(GeneticSearchAlgorithm):
    def __init__(self, delta=0.05, config: dict = None) -> None:
        super().__init__()

        self.delta = delta
        self.config = config

    def generate_ensemble(self, data):
        return Ensemble(data)

    def generate_fitness_functions(
        self, surrogate_model: Ensemble
    ) -> SearchFitnessFunction:

        def fitness_function(_ga, solution, idx):
            return surrogate_model.predict(solution)

        search_fitness_function = SearchFitnessFunction(fitness_function, surrogate_model)
        return search_fitness_function

    def search(self, fitness_function: SearchFitnessFunction, specification: CausalSpecification) -> list:
        var_space = dict()
        for adj in fitness_function.surrogate_model.vars:
            var_space[adj] = dict()

        for relationship in list(specification.scenario.constraints):
            rel_split = str(relationship).split(" ")

            if rel_split[1] == ">=":
                var_space[rel_split[0]]["low"] = int(rel_split[2])
            elif rel_split[1] == "<=":
                var_space[rel_split[0]]["high"] = int(rel_split[2])

        gene_space = []
        for adj in fitness_function.surrogate_model.vars:
            gene_space.append(var_space[adj])

        gene_types = []
        for adj in fitness_function.surrogate_model.vars:
            gene_types.append(specification.scenario.variables.get(adj).datatype)

        ga = GA(
            num_generations=200,
            num_parents_mating=4,
            fitness_func=fitness_function.fitness_function,
            sol_per_pop=10,
            num_genes=len(fitness_function.surrogate_model.vars),
            gene_space=gene_space,
            gene_type=gene_types,
        )

        if self.config is not None:
            for k, v in self.config.items():
                if k == "gene_space":
                    raise Exception(
                        "Gene space should not be set through config. This is generated from the causal specification"
                    )
                setattr(ga, k, v)

        ga.run()
        solution, fitness, _idx = ga.best_solution()

        solution_dict = dict()
        for idx, adj in enumerate(fitness_function.surrogate_model.vars):
            solution_dict[adj] = solution[idx]
        return (solution_dict, fitness, fitness_function.surrogate_model)