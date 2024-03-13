from causal_testing.specification.causal_specification import CausalSpecification
from causal_testing.testing.estimators import CubicSplineRegressionEstimator
from causal_testing.surrogate.causal_surrogate_assisted import SearchAlgorithm
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

    def search(self, surrogate_model: Ensemble, specification: CausalSpecification) -> list:

        def fitness_function(_ga, solution, idx):
            return surrogate_model.predict(solution)

        var_space = dict()
        for adj in surrogate_model.vars:
            var_space[adj] = dict()

        for relationship in list(specification.scenario.constraints):
            rel_split = str(relationship).split(" ")

            if rel_split[1] == ">=":
                var_space[rel_split[0]]["low"] = int(rel_split[2])
            elif rel_split[1] == "<=":
                var_space[rel_split[0]]["high"] = int(rel_split[2]) + 1

        gene_space = []
        for adj in surrogate_model.vars:
            gene_space.append(var_space[adj])

        gene_types = []
        for adj in surrogate_model.vars:
            gene_types.append(specification.scenario.variables.get(adj).datatype)

        ga = GA(
            num_generations=200,
            num_parents_mating=4,
            fitness_func=fitness_function,
            sol_per_pop=10,
            num_genes=len(surrogate_model.vars),
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
        for idx, adj in enumerate(surrogate_model.vars):
            solution_dict[adj] = solution[idx]
        return (solution_dict, fitness, surrogate_model)
    
pool_vals = []
    
def process(args):
    surrogate, specification, delta, contradiction_functions, config = pool_vals[args]

    contradiction_function = contradiction_functions[surrogate.expected_relationship]

    # The GA fitness function after including required variables into the function's scope
    # Unused arguments are required for pygad's fitness function signature
    def fitness_function(ga, solution, idx): # pylint: disable=unused-argument
        surrogate.control_value = solution[0] - delta
        surrogate.treatment_value = solution[0] + delta

        adjustment_dict = {}
        for i, adjustment in enumerate(surrogate.adjustment_set):
            adjustment_dict[adjustment] = solution[i + 1]

        ate = surrogate.estimate_ate_calculated(adjustment_dict)

        return contradiction_function(ate)

    gene_types, gene_space = GeneticSearchAlgorithm.create_gene_types(surrogate, specification)

    ga = GA(
        num_generations=200,
        num_parents_mating=4,
        fitness_func=fitness_function,
        sol_per_pop=10,
        num_genes=1 + len(surrogate.adjustment_set),
        gene_space=gene_space,
        gene_type=gene_types,
    )

    if config is not None:
        for k, v in config.items():
            if k == "gene_space":
                raise ValueError(
                    "Gene space should not be set through config. This is generated from the causal "
                    "specification"
                )
            setattr(ga, k, v)

    ga.run()
    solution, fitness, _ = ga.best_solution()

    solution_dict = {}
    solution_dict[surrogate.treatment] = solution[0]
    for idx, adj in enumerate(surrogate.adjustment_set):
        solution_dict[adj] = solution[idx + 1]

    return solution_dict, fitness, surrogate
    
class GeneticMultiProcessSearchAlgorithm(GeneticSearchAlgorithm):

    def __init__(self, processes=1, delta=0.05, config: dict = None) -> None:
        super().__init__(delta, config)

        self.processes = processes

    def search(
            self, surrogate_models: list[CubicSplineRegressionEstimator], specification: CausalSpecification
    ) -> list:
        solutions = []

        global pool_vals

        pool_vals.clear()

        for surrogate in surrogate_models:
            pool_vals.append((surrogate, specification, self.delta, self.contradiction_functions, self.config))
            
        with mp.Pool(processes=self.processes) as pool:
            indices = range(len(pool_vals))
            solutions = pool.map(process, indices)

        return max(solutions, key=itemgetter(1))  # This can be done better with fitness normalisation between edges