import numpy as np
import random
from pygments import highlight
import yaml

import aiida
from aiida import orm
from aiida.engine import WorkChain, while_
from aiida.engine.persistence import ObjectLoader
from functools import singledispatch
import math


# Generic process evaluation the code is ref from aiida-optimize
_YAML_IDENTIFIER = '!!YAML!!'

@singledispatch
def get_fullname(cls_obj):
    """
    Serializes an AiiDA process class / function to an AiiDA String.
    :param cls_obj: Object to be serialized
    :type cls_obj: Process
    """
    try:
        return orm.Str(ObjectLoader().identify_object(cls_obj))
    except ValueError:
        return orm.Str(_YAML_IDENTIFIER + yaml.dump(cls_obj))

#: Keyword arguments to be passed to ``spec.input`` for serializing an input which is a class / process into a string.
PROCESS_INPUT_KWARGS = {
    'valid_type': orm.Str,
    'serializer': get_fullname,
}

def load_object(cls_name):
    """
    Loads the process from the serialized string.
    """
    if isinstance(cls_name, orm.Str):
        cls_name_str = cls_name.value
    else:
        cls_name_str = str(cls_name)
    try:
        return ObjectLoader().load_object(cls_name_str)
    except ValueError as err:
        if cls_name_str.startswith(_YAML_IDENTIFIER):
            return yaml.load(cls_name_str[len(_YAML_IDENTIFIER):])
        raise ValueError(f"Could not load class name '{cls_name_str}'.") from err

class GeneticAlgorithmWorkChain(WorkChain):
    """WorkChain to run GA """
    
    _EVAL_PREFIX = 'eval_'
    
    @classmethod
    def define(cls, spec):
        """Specify imputs and outputs"""
        super().define(spec)
        spec.input('parameters', valid_type=orm.Dict)
        spec.input('evaluate_process', help='Process which produces the result to be optimized.',
            **PROCESS_INPUT_KWARGS)
        spec.input('input_mapping', valid_type=orm.List)    # map gene to input of evaluate process in oder
        spec.input('output_mapping', valid_type=orm.Str)    # name of output of evaluate process to be the result for fitness
        spec.outline(
            cls.start,  # prepare init population and parameters
            while_(cls.not_finished)(
                cls.launch_evaluation,    # calc fitness of current generation
                cls.get_results,
                cls.breed,    # breed new generation (parents, crossover, mutation, new_generation_breed)
            ),
            cls.finalize,   # stop iteration and get results
        )
        spec.output('result', valid_type=orm.Dict)
        
        spec.exit_code(
            201,
            'ERROR_EVALUATE_PROCESS_FAILED',
            message='GA optimization failed because one of the evaluate processes did not finish ok.'
        )
        
    @property
    def indices_to_retrieve(self):
        return self.ctx.setdefault('indices_to_retrieve', [])

    @indices_to_retrieve.setter
    def indices_to_retrieve(self, value):
        self.ctx.indices_to_retrieve = value
        
    def _init_population(self, num_pop, gene_space, gene_type, seed):
        """return numpy array"""
        random.seed(seed)
        
        # set an unassigned array
        pop = np.empty([num_pop, len(gene_space)], dtype=float)
        
        for i in range(num_pop):
            for j in range(len(gene_space)):
                space = gene_space[j]
                value = random.uniform(space['low'], space['high'])
                
                if gene_type[j] == 'int':
                    value = round(value)
                    
                pop[i][j] = round(value, 4)
                
        return pop
    
    def eval_key(self, index):
        """
        Returns the evaluation key corresponding to a given index.
        """
        return self._EVAL_PREFIX + str(index)
    
        
    def start(self):
        # to store const parameters in ctx over GA procedure
        parameters = self.ctx.const_parameters = self.inputs.parameters.get_dict()
        
        self.ctx.current_generation = 1
        
        # population
        seed = self.ctx.seed = parameters['seed']
        num_pop = parameters['num_pop_per_generation']
        gene_space = parameters['gene_space']
        gene_type = parameters['gene_type']
        self.ctx.population = self._init_population(num_pop, gene_space, gene_type, seed=seed)
        
        # initialize the ctx variable to update during GA
        self.ctx.fitness = None
        
        # solution
        self.ctx.best_solution = None
    
    def not_finished(self):
        """return a bool, whether create new generation"""
        if self.ctx.current_generation > self.ctx.const_parameters['num_generation']:
            return False
        else:
            return True
        
    def _inputs_from_mapping(self, input_mapping, input_values):
        """list of mapping key and value"""
        input_ret = {}
        
        for key, value in zip(input_mapping, input_values):
            # nest key separated by `.`
            input_ret[key] = value
            
        return {
            'parameters': orm.Dict(dict=input_ret),    
        }
            
        
    def launch_evaluation(self):
        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
            
        evaluate_process is the name of process
        """
        self.report(f'On fitness at generation: {self.ctx.current_generation}')

        evaluate_process = load_object(self.inputs.evaluate_process.value)
        
        # submit evaluation for the pop ind
        evals = {}
        for idx, ind in enumerate(self.ctx.population):
            inputs = self._inputs_from_mapping(self.inputs.input_mapping.get_list(), list(ind))
            node = self.submit(evaluate_process, **inputs)
            evals[self.eval_key(idx)] = node
            self.indices_to_retrieve.append(idx)

            
        return self.to_context(**evals)

    def get_results(self):
        """-> fitness parser, calc best solution"""
        self.report('Checking finished evaluations.')
        outputs = {}
        while self.indices_to_retrieve:
            idx = self.indices_to_retrieve.pop(0)
            key = self.eval_key(idx)
            self.report('Retrieving output for evaluation {}'.format(idx))
            eval_proc = self.ctx[key]
            if not eval_proc.is_finished_ok:
                # When evaluate process failed it can be
                # - the parameters are not proper, this should result the bad score for the GA input
                # - the evalute process failed for resoure reason, should raise and reported.
                # - TODO: Test configuration 0 is get but no other configuration results -> check what output look like
                if eval_proc.exit_status == 201: # ERROR_PSPOT_HAS_NODE
                    outputs[idx] = -math.inf
                else:
                    return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            else:
                outputs[idx] = eval_proc.outputs['result']
            
        self.ctx.fitness = outputs
        self.ctx.best_solution = self._get_best_solution(outputs)
        self.report(self.ctx.best_solution)
        
    def _get_best_solution(self, outputs):
        import operator
        
        # get the max value and idx of fitness
        idx, best_fitness =  max(outputs.items(), key=operator.itemgetter(1))
        key = self.eval_key(idx)
        eval_proc = self.ctx[key]
        process_uuid = eval_proc.id
        parameters = eval_proc.inputs.parameters.get_dict()
        
        # TODO store more than one best solutions
        return {
            'best_fitness': best_fitness,
            'process_uuid': process_uuid,
            'parameters': parameters,
        }
        
    def breed(self):
        """breed new generation"""
        self.ctx.current_generation += 1    # IMPORTANT, otherwise infinity loop
        
        # keep and mating parents selection
        keep_parents, mating_parents = _rank_selection(
            self.ctx.population, 
            self.ctx.fitness, 
            self.ctx.const_parameters['num_keep_parents'],
            self.ctx.const_parameters['num_mating_parents'],
        )
        
        # crossover
        num_offsprings = self.ctx.const_parameters['num_pop_per_generation'] - self.ctx.const_parameters['num_keep_parents']
        offspring = _crossover(mating_parents, num_offsprings, seed=self.ctx.seed)
        
        # mutation
        mut_offspring = _mutate(
            offspring, 
            mutate_probability=self.ctx.const_parameters['mutate_probability'], 
            gene_space=self.ctx.const_parameters['gene_space'],
            gene_type=self.ctx.const_parameters['gene_type'],
            seed=self.ctx.seed
        )
        
        # population generation: update ctx population for next generation
        self.ctx.population = np.vstack((keep_parents, mut_offspring))
        
    
    def finalize(self):
        self.report('on stop')
        self.out('result', orm.Dict(dict={
                'current_generation': self.ctx.current_generation,
            }).store())
        
def _rank_selection(population, fitness, num_keep_parents, num_mating_parents):
    """
    Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns two arrays of the selected keep parents and mating parents respectively.
    """

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    fitness_sorted.reverse()    # max fitness the best, reverse to put in front of list
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

    keep_parents = np.empty((num_keep_parents, population.shape[1]), dtype=object)
    for i in range(num_keep_parents):
        # set i-th best ind to i row
        keep_parents[i, :] = population[fitness_sorted[i], :].copy()
        
    mating_parents = np.empty((num_mating_parents, population.shape[1]), dtype=object)
    for i in range(num_mating_parents):
        # set i-th best ind to i row
        mating_parents[i, :] = population[fitness_sorted[i], :].copy()

    return keep_parents, mating_parents

def _crossover(parents, num_offsprings, seed):
    random.seed(seed)
    
    num_parents, num_genes = parents.shape
    offspring = np.empty((num_offsprings, num_genes), dtype=object)
    for i in range(num_offsprings):
        m_idx, f_idx = random.sample(range(num_parents), 2)
        mother = parents[m_idx] # mother from mother idx
        father = parents[f_idx] # father from father idx
        
        #mating
        # TODO: two points crossover if more gene
        k = 1   # TODO -> range(1:num_genes)
        child = np.hstack((mother[:k], father[k:]))
        offspring[i, :] = child
        
    return offspring

def _mutate(inds, mutate_probability, gene_space, gene_type, seed):
    random.seed(seed)
        
    mut_inds = np.empty_like(inds)
    num_inds, num_genes = inds.shape
    for i in range(num_inds):
        for j in range(num_genes):
            space = gene_space[j]
            # based of probability, keep original value if not being hit
            # TODO: if not fully random but a pertubation may better??
            if random.random() < mutate_probability:
                value = random.uniform(space['low'], space['high'])
                
                if gene_type[j] == 'int':
                    value = round(value)
                mut_inds[i, j] = value
            else:
                mut_inds[i, j] = inds[i, j]
                        
    return mut_inds