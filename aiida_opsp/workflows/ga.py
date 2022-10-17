import numpy as np
import random
from pygments import highlight
import yaml

import aiida
from aiida import orm
from aiida.engine import WorkChain, while_, append_
from aiida.engine.persistence import ObjectLoader
from functools import singledispatch
import math

from plumpy.utils import AttributesFrozendict

from aiida.orm.nodes.data.base import to_aiida_type

from aiida_opsp.workflows.ls import LocalSearchWorkChain


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
        spec.input('vars_info', valid_type=orm.Dict)    # map gene to input of evaluate process in order
        spec.input('result_key', valid_type=orm.Str)    # name of key to be the result for fitness
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)  # The fixed input parameters that will combined with change parameters
        
        spec.outline(
            cls.start,  # prepare init population and parameters
            while_(cls.not_finished)(
                cls.launch_evaluation,    # calc fitness of current generation
                cls.get_results,
                cls.crossover,
                cls.mutate,
                cls.local_search,
                cls.combine_pop,
            ),
            # finalize run
            cls.launch_evaluation,
            cls.get_results,
            cls.combine_pop,
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
        
    def _init_population(self, num_pop, genes, seed):
        """return populations of one generation as a numpy array"""
        random.seed(f'init_pop_{seed}')
        
        # set an unassigned array
        pop = np.empty([num_pop, len(genes)], dtype=float)

        for i in range(num_pop):
            _inds = {}
            for k, v in genes.items():
                # the first for to collect all not related range setting for base.
                space = v['space']
                gene_type = v['type']
                
                refto = space.get("refto", None)
                if refto is None:
                    x = random.uniform(space['low'], space['high'])

                    if gene_type == 'int':
                        x = int(round(x))
                    else:
                        x = round(x, 4)

                    _inds[k] = x                
                
            for k, v in genes.items():
                # the second for to set the relavent range from base
                space = v['space']
                gene_type = v['type']
                refto = space.get("refto", None)
                if refto is not None:
                    base = _inds[refto]
                    x = base + random.uniform(space['low'], space['high'])
                    
                    if gene_type == 'int':
                        x = int(round(x))
                    else:
                        x = round(x, 4)

                    _inds[k] = x  
                
            # Set pop
            for j, key in enumerate(genes.keys()):
                pop[i][j] = _inds[key]
                
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
        genes = self.ctx.genes = self.inputs.vars_info.get_dict()
        # built-in dict class gained the ability to remember 
        # insertion order (this new behavior became guaranteed in Python 3.7).
        self.ctx.population = self._init_population(num_pop, genes, seed=seed)
        
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
        
    def _merge_nested_inputs(self, input_mapping, input_values, fixture_inputs):
        """list of mapping key and value"""
        target_inputs = dict(fixture_inputs)
        
        nested_key_inputs = {}
        for key, value in zip(input_mapping, input_values):
            # nest key separated by `.`
            nested_key_inputs[key] = value
            
        inputs = _merge_nested_keys(nested_key_inputs, target_inputs)
            
        return inputs
            
    def _validate_ind(self, ind, genes):
        """validate and convert the ind to the correct type"""
        vind = []
        for i, (k, v) in enumerate(genes.items()):
            x = ind[i]
            if v["space"].get("refto", None) is None:
                if x < v["space"]["low"] or x > v["space"]["high"]:
                    self.report(f"!!!WARNING: gene {k} = {x} is out of range {v['space']['low']} < x < {v['space']['high']}.")
            # else:
            # TODO: add contruct _inds dict function and check the range of it.
                
                
            if v["type"] == "int":
                x = int(x)
                
            vind.append(x)
            
        return vind
                
        
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
        input_mapping = [i["key_name"] for i in self.inputs.vars_info.get_dict().values()]
        for idx, ind in enumerate(self.ctx.population):
            vind = self._validate_ind(ind, self.ctx.genes)
            inputs = self._merge_nested_inputs(
                input_mapping=input_mapping, 
                input_values=list(vind), 
                fixture_inputs=self.inputs.get('fixture_inputs', {})
            )
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
                    outputs[idx] = math.inf
                else:
                    return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            else:
                outputs[idx] = eval_proc.outputs['result'].value
            
        self.ctx.fitness = outputs
        self.ctx.best_solution = self._get_best_solution(outputs)
        
        output_report = []
        for idx, ind in enumerate(self.ctx.population):
            key = self.eval_key(idx)
            proc = self.ctx[key]
            fitness = outputs[idx]
            output_report.append(f'idx={idx}  pk={proc.pk}: {ind} -> fitness={fitness}')
            
        output_report_str = '\n'.join(output_report)
        
        self.report(f'population and process pk:')
        self.report(f'\n{output_report_str}')
        self.report(self.ctx.best_solution)
        
    def _get_best_solution(self, outputs):
        import operator
        
        # get the max value and idx of fitness
        idx, best_fitness =  min(outputs.items(), key=operator.itemgetter(1))
        key = self.eval_key(idx)
        eval_proc = self.ctx[key]
        process_uuid = eval_proc.id
        best_ind = self.ctx.population[idx]
        
        # TODO store more than one best solutions
        return {
            'best_fitness': best_fitness,
            'process_uuid': process_uuid,
            'best_ind': best_ind,
        }
        
    def crossover(self):
        """crossover"""
        self.ctx.current_generation += 1    # IMPORTANT, otherwise infinity loop
        self.ctx.seed += 1 # IMPORTANT the seed should update for every generation otherwise mutate offspring is the same
        
        # keep and mating parents selection
        self.ctx.elitism, mating_parents = _rank_selection(
            self.ctx.population, 
            self.ctx.fitness, 
            self.ctx.const_parameters['num_elitism'],
            self.ctx.const_parameters['num_mating_parents'],
        )
        
        # EXPERIMENTAL!!
        # N_offspring = N_pop - 2 * N_elitism
        # sinec mutate using gaussing for the other N_elitism
        
        # crossover
        num_offsprings = self.ctx.const_parameters['num_pop_per_generation'] - 2 * self.ctx.const_parameters['num_elitism']
        self.ctx.offspring = _crossover(mating_parents, num_offsprings, seed=self.ctx.seed)
        
        
    def mutate(self):
        """breed new generation"""
        
        # mutation elitism
        self.ctx.mut_elitism = _mutate(
            self.ctx.elitism,
            individual_mutate_probability=self.ctx.const_parameters['individual_mutate_probability'],
            gene_mutate_probability=self.ctx.const_parameters['gene_mutate_elitism_probability'], 
            genes=self.ctx.genes,
            seed=self.ctx.seed,
            gaussian=True,
        )
        
        # mutation offspring
        self.ctx.mut_offspring = _mutate(
            self.ctx.offspring, 
            individual_mutate_probability=self.ctx.const_parameters['individual_mutate_probability'], 
            gene_mutate_probability=self.ctx.const_parameters['gene_mutate_mediocrity_probability'], 
            genes=self.ctx.genes,
            seed=self.ctx.seed
        )
        
        # population generation: update ctx population for next generation
        # self.ctx.population = np.vstack((self.ctx.elitism, mut_elitism, mut_offspring))
        # self.report(f'new population: {self.ctx.population}')
        
    def local_search(self):
        """ local_search of elitism
        """
        for ind in self.ctx.elitism:
            ls_parameters = self.ctx.const_parameters['local_search_base_parameters']
            ls_parameters['init_vars'] = list(ind)
            
            inputs = {
                'parameters': orm.Dict(dict=ls_parameters),
                'evaluate_process': self.inputs.evaluate_process,
                'vars_info': self.inputs.vars_info,
                'result_key': self.inputs.result_key,
                'fixture_inputs': self.inputs.fixture_inputs,
            }
            running = self.submit(LocalSearchWorkChain, **inputs)
            self.to_context(workchain_elitism=append_(running))
    
    def combine_pop(self):
        local_min_elitism_lst = []
        local_min_elitism_y_list = []
        for child in self.ctx.workchain_elitism:
            if not child.is_finished_ok:
                self.logger.warning(
                    f"Local search not finished ok",
                )
                return self.exit_codes.ERROR_LOCAL_SEARCH_NOT_FINISHED_OK
            
            local_min_elitism_lst.append(child.outputs.result['xs'])
            local_min_elitism_y_list.append(child.outputs.result['y'])
            
        self.ctx.local_min_elitism = np.array(local_min_elitism_lst)
        self.ctx.local_min_elitism_y = local_min_elitism_y_list
            
        # TODO check the local_search workchains are finished.
        # self.ctx.population = np.vstack((self.ctx.local_min_elitism, self.ctx.local_min_mut_elitism, self.ctx.local_min_mut_offspring))
        self.ctx.population = np.vstack((self.ctx.local_min_elitism, self.ctx.mut_elitism, self.ctx.mut_offspring))
    
    def finalize(self):
        # this step will come after `combine_pop`,
        # the populations will correspond with the solutions.
        # will store all population in a 2D list and fitness.
        # will give the best one (the first one with idx = 0) and its fitness.
        self.report('on stop')
        self.out('result', orm.Dict(dict={
                "populations": list(self.ctx.local_min_elitism), 
                "fitness": self.ctx.local_min_elitism_y,
                'current_generation': self.ctx.current_generation,
            }).store())
        
def _rank_selection(population, fitness, num_elitism, num_mating_parents):
    """
    Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
    It accepts 2 parameters:
        -fitness: The fitness values of the solutions in the current population.
        -num_parents: The number of parents to be selected.
    It returns two arrays of the selected keep parents and mating parents respectively.
    """

    fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.

    keep_parents = np.empty((num_elitism, population.shape[1]), dtype=object)
    for i in range(num_elitism):
        # set i-th best ind to i row
        keep_parents[i, :] = population[fitness_sorted[i], :].copy()
        
    mating_parents = np.empty((num_mating_parents, population.shape[1]), dtype=object)
    for i in range(num_mating_parents):
        # set i-th best ind to i row
        mating_parents[i, :] = population[fitness_sorted[i], :].copy()

    return keep_parents, mating_parents

def _crossover(parents, num_offsprings, seed):
    """In principle the max number of un-duplicated offsprings this procedure can produce
    is P(nparents, 2).
    """
    # TODO a warning for too much offspring required with too less parents.
    random.seed(f'crossover_{seed}')
    
    num_parents, num_genes = parents.shape
    offspring = np.empty((num_offsprings, num_genes), dtype=object)
    for i in range(num_offsprings):
        m_idx, f_idx = random.sample(range(num_parents), 2)
        mother = parents[m_idx] # mother from mother idx
        father = parents[f_idx] # father from father idx
        
        #mating
        # TODO: two points crossover if more gene
        # k = 0 for random one of parents. e.g for crossover probability=0.0 no crossover
        k = 0   # TODO -> range(1:num_genes)
        child = np.hstack((mother[:k], father[k:]))
        offspring[i, :] = child
        
    return offspring

def _mutate(inds, individual_mutate_probability, gene_mutate_probability, genes, seed, gaussian=False):
    """docstring"""
    random.seed(f'mutate_{seed}')
    
    # whether mutate this indvidual.
    # this is a suplement for keep parent,
    # usually this set to 1 so every individual not ranking to 
    # keep parents will mutate.
    if random.random() > individual_mutate_probability:
        return inds
    
    num_inds, num_genes = inds.shape    
    mut_inds = np.empty([num_inds, num_genes], dtype=float)
    for i in range(num_inds):
        _d_ind = {}
        for j, (k, v) in enumerate(genes.items()):
            _d_ind[k] = inds[i][j]
        
        for j, (k, v) in enumerate(genes.items()):
            space = v['space']
            gene_type = v['type']
            # based of probability, keep original value if not being hit
            # TODO: if not fully random but a pertubation may better??
            if random.random() < gene_mutate_probability:
                if gaussian:
                    old_value = inds[i, j]
                    x = random.gauss(old_value, sigma=old_value/10)
                    
                    if gene_type == 'int':
                        x = int(round(x))
                    else:
                        x = round(x, 4)

                    _d_ind[k] = x    
                else:
                    refto = space.get("refto", None)
                    if refto:
                        base = _d_ind[refto]
                        x = base + random.uniform(space['low'], space['high'])
                    else:
                        x = random.uniform(space['low'], space['high'])
                        
                    if gene_type == 'int':
                        x = int(round(x))
                    else:
                        x = round(x, 4)

                    _d_ind[k] = x    
            else:
                mut_inds[i, j] = inds[i, j]
                
        # Set pop
        for j, key in enumerate(genes.keys()):
            mut_inds[i, j] = _d_ind[key]                
            
    return mut_inds

def _merge_nested_keys(nested_key_inputs, target_inputs):
    """
    Maps nested_key_inputs onto target_inputs with support for nested keys:
        x.y:a.b -> x.y['a']['b']
    Note: keys will be python str; values will be AiiDA data types
    """
    def _get_nested_dict(in_dict, split_path):
        res_dict = in_dict
        for path_part in split_path:
            res_dict = res_dict.setdefault(path_part, {})
        return res_dict

    destination = _copy_nested_dict(target_inputs)

    for key, value in nested_key_inputs.items():
        full_port_path, *full_attr_path = key.split(':')
        *port_path, port_name = full_port_path.split('.')
        namespace = _get_nested_dict(in_dict=destination, split_path=port_path)

        if not full_attr_path:
            if not isinstance(value, orm.Node):
                value = to_aiida_type(value).store()
            res_value = value
        else:
            if len(full_attr_path) != 1:
                raise ValueError(f"Nested key syntax can contain at most one ':'. Got '{key}'")

            # Get or create the top-level dictionary.
            try:
                res_dict = namespace[port_name].get_dict()
            except KeyError:
                res_dict = {}

            *sub_dict_path, attr_name = full_attr_path[0].split('.')
            sub_dict = _get_nested_dict(in_dict=res_dict, split_path=sub_dict_path)
            sub_dict[attr_name] = _from_aiida_type(value)
            res_value = orm.Dict(dict=res_dict).store()

        namespace[port_name] = res_value
    return destination


def _copy_nested_dict(value):
    """
    Copy nested dictionaries. `AttributesFrozendict` is converted into
    a (mutable) plain Python `dict`.

    This is needed because `copy.deepcopy` would create new AiiDA nodes.
    """
    if isinstance(value, (dict, AttributesFrozendict)):
        return {k: _copy_nested_dict(v) for k, v in value.items()}
    return value


def _from_aiida_type(value):
    """
    Convert an AiiDA data object to the equivalent Python object
    """
    if not isinstance(value, orm.Node):
        return value
    if isinstance(value, orm.BaseType):
        return value.value
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.List):
        return value.get_list()
    raise TypeError(f'value of type {type(value)} is not supported')