import numpy as np
from pygad import GA
import random

import aiida
from aiida import orm
from aiida.engine import WorkChain, while_

function_inputs = [4,-2,3.5,5,-11,-4.7] # Function inputs.
desired_output = 44 # Function output.

class GeneticAlgorithmWorkChain(WorkChain):
    """WorkChain to run demo GA """
    
    @classmethod
    def define(cls, spec):
        """Specify imputs and outputs"""
        super().define(spec)
        # spec.input('parameters', valid_type=orm.Dict)
        spec.outline(
            cls.start,  # prepare init population and parameters
            while_(cls.do_new_generation)(
                cls.fitness,    # calc fitness of current generation
                cls.parents,    # choice parents based on current generation
                cls.crossover,  # do crossover
                cls.mutation,   # do mutation
                cls.generation, # prepare new population for next generation
            ),
            cls.fitness, # calc fitness of last generation
            cls.stop,   # stop iteration and get results
        )
        spec.output('result', valid_type=orm.Dict)
        
    def start(self):
        self.ctx.num_generation = 50
        self.ctx.num_parents_mating = 12
        self.ctx.sol_per_pop = 20
        self.ctx.num_genes = 6
        self.ctx.keep_parents = 4
        self.ctx.mutation_num_genes = 6
        self.ctx.num_offspring = self.ctx.sol_per_pop - self.ctx.keep_parents
        self.ctx.crossover_probability = 0.8
        self.ctx.gene_space = [{'low': -10.0, 'high': 10.0}] * self.ctx.num_genes
        self.ctx.gene_type = [float, float, float, float, float, float]
        
        self.ctx.current_generation = 1
        
        # population
        np.random.seed(0)
        self.ctx.population = np.random.uniform(low=-10.0, high=10, size=(20, 6))
        self.ctx.last_generation_parents = None
        self.ctx.last_generation_parents_indices = None
        self.ctx.previous_generation_fitness = None
        
        # solution
        self.ctx.best_solution = None
        self.ctx.best_solution_fitness = None
        self.ctx.best_match_idx = None
    
    def do_new_generation(self):
        if self.ctx.current_generation > self.ctx.num_generation:
            return False
        else:
            return True

    def fitness(self):
        self.report(f'On fitness at generation: {self.ctx.current_generation}')
        
        self.ctx.last_generation_fitness = _cal_pop_fitness(self.ctx.population,
                                                self.ctx.last_generation_parents,
                                                self.ctx.last_generation_parents_indices,
                                                self.ctx.previous_generation_fitness,
                                                fitness_func)
        
        self.ctx.best_solution, self.ctx.best_solution_fitness, self.ctx.best_match_idx = _best_solution(pop_fitness=self.ctx.last_generation_fitness,
                                                                            population=self.ctx.population)
    
        prediction = np.sum(np.array(function_inputs)*self.ctx.best_solution)
        self.report(f'PRED SUM: {prediction}')
        self.report(f'BEST: sol={self.ctx.best_solution}, fitness={self.ctx.best_solution_fitness}')
    def parents(self):
        # self.report('on parents')
        
        self.ctx.last_generation_parents, self.ctx.last_generation_parents_indices = _rank_selection(
            self.ctx.population, 
            self.ctx.last_generation_fitness, 
            self.ctx.num_parents_mating)

        # import ipdb; ipdb.set_trace()
        # self.report(f'{self.ctx.last_generation_parents}')
    
    def crossover(self):
        self.report('on crossover')
        
        self.ctx.last_generation_offspring_crossover = _two_points_crossover(self.ctx.last_generation_parents,
                                                            offspring_size=(self.ctx.num_offspring, self.ctx.num_genes),
                                                            crossover_probability=self.ctx.crossover_probability)
        
    def mutation(self):
        # self.report('on mutation')
        
        self.ctx.last_generation_offspring_mutation = _mutation_by_space(self.ctx.last_generation_offspring_crossover, 
                                                                         self.ctx.num_genes, 
                                                                         self.ctx.mutation_num_genes, 
                                                                         self.ctx.gene_space, 
                                                                         self.ctx.gene_type)
    
    def generation(self):
        # self.report('on generation')
        # self.report(f'The current generation is: {self.ctx.current_generation}')
        
        # import ipdb; ipdb.set_trace()
        parents_to_keep, _ = _steady_state_selection(
            self.ctx.population, 
            self.ctx.last_generation_fitness, 
            self.ctx.keep_parents)
        self.ctx.population[0:parents_to_keep.shape[0], :] = parents_to_keep
        self.ctx.population[parents_to_keep.shape[0]:, :] = self.ctx.last_generation_offspring_mutation
        # self.report(f'POP: {self.ctx.population}')
        
        self.ctx.previous_generation_fitness = self.ctx.last_generation_fitness.copy()
        
        self.ctx.current_generation += 1
    
    def stop(self):
        # self.report('on stop')
        g = self.ctx.current_generation
        self.out('result', orm.Dict(dict={
                'current_generation': g,
            }).store())
     
     
def fitness_func(solution, solution_idx):
    output = np.sum(solution*function_inputs)
    fitness = 1.0 / (np.abs(output - desired_output) + 0.000001)
    return fitness   
        
def _cal_pop_fitness(population, 
                        last_generation_parents, 
                        last_generation_parents_indices,
                        previous_generation_fitness,
                        fitness_func):

    """
    Calculating the fitness values of all solutions in the current population. 
    It returns:
        -fitness: An array of the calculated fitness values.
    """

    pop_fitness = []
    # Calculating the fitness value of each solution in the current population.
    for sol_idx, sol in enumerate(population):

        # Check if this solution is a parent from the previous generation and its fitness value is already calculated. If so, use the fitness value instead of calling the fitness function.
        if (last_generation_parents is not None) and len(np.where(np.all(last_generation_parents == sol, axis=1))[0] > 0):
            # Index of the parent in the parents array (self.last_generation_parents). This is not its index within the population.
            parent_idx = np.where(np.all(last_generation_parents == sol, axis=1))[0][0]
            # Index of the parent in the population.
            parent_idx = last_generation_parents_indices[parent_idx]
            # Use the parent's index to return its pre-calculated fitness value.
            fitness = previous_generation_fitness[parent_idx]
        else:
            fitness = fitness_func(sol, sol_idx)
            if type(fitness) in GA.supported_int_float_types:
                pass
            else:
                raise ValueError("The fitness function should return a number but the value {fit_val} of type {fit_type} found.".format(fit_val=fitness, fit_type=type(fitness)))
        pop_fitness.append(fitness)

    pop_fitness = np.array(pop_fitness)

    return pop_fitness

def _best_solution(pop_fitness, population):
    # Then return the index of that solution corresponding to the best fitness.
    best_match_idx = np.where(pop_fitness == np.max(pop_fitness))[0][0]

    best_solution = population[best_match_idx, :].copy()
    best_solution_fitness = pop_fitness[best_match_idx]

    return best_solution, best_solution_fitness, best_match_idx

def _random_selection(population, fitness, num_parents):

    """
    Selects the parents randomly. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """
    parents = np.empty((num_parents, population.shape[1]), dtype=object)

    rand_indices = np.random.randint(low=0.0, high=fitness.shape[0], size=num_parents)

    for parent_num in range(num_parents):
        parents[parent_num, :] = population[rand_indices[parent_num], :].copy()

    return parents, rand_indices

def _rank_selection(population, fitness, num_parents):

    """
    Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

    parents = np.empty((num_parents, population.shape[1]), dtype=object)
    
    for parent_num in range(num_parents):
        parents[parent_num, :] = population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]

def _steady_state_selection(population, fitness, num_parents):

    """
    Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns an array of the selected parents.
    """
    
    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, population.shape[1]), dtype=object)
    for parent_num in range(num_parents):
        parents[parent_num, :] = population[fitness_sorted[parent_num], :].copy()

    return parents, fitness_sorted[:num_parents]


def _two_points_crossover(parents, offspring_size, crossover_probability):

    """
    Applies the 2 points crossover. It selects the 2 points randomly at which crossover takes place between the pairs of parents.
    It accepts 2 parameters:
        -parents: The parents to mate for producing the offspring.
        -offspring_size: The size of the offspring to produce.
    It returns an array the produced offspring.
    """

    offspring = np.empty(offspring_size, dtype=object)

    for k in range(offspring_size[0]):
        if (parents.shape[1] == 1): # If the chromosome has only a single gene. In this case, this gene is copied from the second parent.
            crossover_point1 = 0
        else:
            crossover_point1 = np.random.randint(low=0, high=np.ceil(parents.shape[1]/2 + 1), size=1)[0]

        crossover_point2 = crossover_point1 + int(parents.shape[1]/2) # The second point must always be greater than the first point.

        if not (crossover_probability is None):
            probs = np.random.random(size=parents.shape[0])
            indices = np.where(probs <= crossover_probability)[0]

            # If no parent satisfied the probability, no crossover is applied and a parent is selected.
            if len(indices) == 0:
                offspring[k, :] = parents[k % parents.shape[0], :]
                continue
            elif len(indices) == 1:
                parent1_idx = indices[0]
                parent2_idx = parent1_idx
            else:
                indices = random.sample(set(indices), 2)
                parent1_idx = indices[0]
                parent2_idx = indices[1]
        else:
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]

        # The genes from the beginning of the chromosome up to the first point are copied from the first parent.
        offspring[k, 0:crossover_point1] = parents[parent1_idx, 0:crossover_point1]
        # The genes from the second point up to the end of the chromosome are copied from the first parent.
        offspring[k, crossover_point2:] = parents[parent1_idx, crossover_point2:]
        # The genes between the 2 points are copied from the second parent.
        offspring[k, crossover_point1:crossover_point2] = parents[parent2_idx, crossover_point1:crossover_point2]
    return offspring

def _mutation_by_space(offspring, num_genes, mutation_num_genes, gene_space, gene_type):

    """
    Applies the random mutation using the mutation values' space.
    It accepts a single parameter:
        -offspring: The offspring to mutate.
    It returns an array of the mutated offspring using the mutation space.
    """

    # For each offspring, a value from the gene space is selected randomly and assigned to the selected mutated gene.
    for offspring_idx in range(offspring.shape[0]):
        mutation_indices = np.array(random.sample(range(0, num_genes), mutation_num_genes))
        for gene_idx in mutation_indices:

            # Returning the current gene space from the 'gene_space' attribute.
            curr_gene_space = gene_space[gene_idx]

            value_from_space = np.random.uniform(low=curr_gene_space['low'],
                                                    high=curr_gene_space['high'],
                                                    size=1)

            # Assinging the selected value from the space to the gene.
            offspring[offspring_idx, gene_idx] = gene_type[gene_idx](value_from_space)

            # if self.allow_duplicate_genes == False:
            #     offspring[offspring_idx], _, _ = self.solve_duplicate_genes_by_space(solution=offspring[offspring_idx],
            #                                                                             gene_type=self.gene_type,
            #                                                                             num_trials=10)
            
    return offspring