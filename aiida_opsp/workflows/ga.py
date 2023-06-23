import numpy as np
import random

from aiida import orm
from aiida.engine import WorkChain, while_, append_
import math

from plumpy.utils import AttributesFrozendict

from aiida.orm.nodes.data.base import to_aiida_type

from aiida_opsp.workflows.ls import LocalSearchWorkChain
from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS
from aiida_opsp.workflows.individual import GenerateValidIndividual
from aiida_opsp.utils.merge_input import individual_to_inputs

class GeneticAlgorithmWorkChain(WorkChain):
    """WorkChain to run GA """
    
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
            cls.init_setup,
            cls.prepare_init_population_run,
            cls.prepare_init_population_inspect,
            while_(cls.should_continue)(
                cls.evaluation_run,    # calc fitness of current generation
                cls.evaluation_inspect,
                # cls.crossover,
                cls.mutate,
                cls.local_search,
                cls.combine_pop,
            ),
            # finalize run
            cls.launch_final_evaluation,
            cls.get_final_results,
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
        

    def init_setup(self):
        """prepare initial ctx"""
        
        # to store const parameters in ctx over GA procedure
        parameters = self.ctx.const_parameters = self.inputs.parameters.get_dict()
        
        # init current optimize session aka generation in GA
        self.ctx.current_generation = 0
        
        # population
        self.ctx.seed = parameters['seed']
        self.ctx.num_population = parameters['num_pop_per_generation']
        self.ctx.genes = self.inputs.vars_info.get_dict()
        self.ctx.num_elitism = parameters['num_elitism']
        self.ctx.num_mating_parents = parameters['num_mating_parents']

        self.ctx.population = np.array([], dtype=np.float64).reshape(0, len(self.ctx.genes))
        
        # initialize the ctx variable to update during GA
        self.ctx.fitness = None
        
        # set base evaluate process
        self.ctx.evaluate_process = load_object(self.inputs.evaluate_process.value)

        # tmp_retrive_key_storage
        # This is for store the key so the parser step know which process to fetch from ctx
        # It needs to be kept empty in between a run-inspect session.
        self.ctx.tmp_retrive_key_storage = []
        
        # solution
        self.ctx.best_solution = None
    
    def prepare_init_population_run(self):
        inputs = {
            'evaluate_process': self.ctx.evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
        }

        evaluates = dict()
        for idx in range(self.ctx.num_population):
            new_seed = self.ctx.seed + idx  # increment the seed, since we don't want every individual is the same ;)
            inputs['seed'] = orm.Int(new_seed)
            node = self.submit(GenerateValidIndividual, **inputs)

            retrive_key = f'_VALID_IND_{idx}'
            evaluates[retrive_key] = node

            self.ctx.tmp_retrive_key_storage.append(retrive_key)


        return self.to_context(**evaluates)

    def prepare_init_population_inspect(self):
        population = []
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f"Retriving output for evaluation {key}")

            proc: orm.WorkChainNode = self.ctx[key]
            if not proc.is_finished_okay:
                return self.exit_codes.ERROR_PREPARE_INIT_POPULATION_FAILED
            else:
                population.append(proc.outputs.final_individual.get_dict())          

        self.ctx.population = population

        if not len(self.ctx.population) == self.ctx.num_population:
            return self.exit_codes.ERROR_PREPARE_INIT_POPULATION_FAILED            
            
        
    def should_continue(self):
        """return a bool, whether create new generation"""
        if self.ctx.current_generation > self.ctx.const_parameters['num_generation']:
            return False
        else:
            return True
        
    def evaluation_run(self):
        """
        Calculating the fitness values of all solutions in the current population. 
        It returns:
            -fitness: An array of the calculated fitness values.
            
        evaluate_process is the name of process
        """
        self.report(f'On fitness at generation: {self.ctx.current_generation}')

        # submit evaluation for the individuals of a population
        evaluates = dict()
        for idx, individual in enumerate(self.ctx.population):
            inputs = individual_to_inputs(individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
            node = self.submit(self.ctx.evaluate_process, **inputs)

            retrive_key = f'_EVAL_IND_{idx}'
            evaluates[retrive_key] = node
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)
    
    def launch_final_evaluation(self):
        self.report("I am final")
        self.launch_evaluation()

    def evaluation_inspect(self):
        """-> fitness parser, calc best solution"""
        self.report('Checking finished evaluations.')

        # dict of retrive key and score of the individual
        fitness = {}
    
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            key = self.ctx.tmp_retrive_key_storage.pop(0)

            self.report(f'Retrieving output for evaluation {key}')

            proc: orm.WorkChainNode = self.ctx[key]
            if not proc.is_finished_ok:
                # When evaluate process failed it can be
                # - the parameters are not proper, this should result the bad score for the GA input
                # - the evalute process failed for resoure reason, should raise and reported.
                # - TODO: Test configuration 0 is get but no other configuration results -> check what output look like
                if proc.exit_status == 201: # ERROR_PSPOT_HAS_NODE
                    fitness[key] = math.inf
                else:
                    return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            else:
                fitness[key] = proc.outputs['result'].value
            
        # fitness store the results of individual score of a population of this generation
        self.ctx.fitness = fitness
        if len(self.ctx.fitness) != self.ctx.num_generation:
            return self.exit_codes.ERROR_FITNESS_HAS_WRONG_NUM_OF_RESULTS

        # self.ctx.best_solution = self._get_solutions(fitness, [0])[0]
        
        # output_report = []
        # for idx, ind in enumerate(self.ctx.population):
        #     key = self.eval_key(idx)
        #     proc = self.ctx[key]
        #     fitness = outputs[idx]
        #     output_report.append(f'idx={idx}  pk={proc.pk}: {ind} -> fitness={fitness}')
        #     
        # output_report_str = '\n'.join(output_report)
        
        # self.report(f'population and process pk:')
        # self.report(f'\n{output_report_str}')
        # self.report(self.ctx.best_solution)
        
    def get_final_results(self):
        self.report("I am final")
        self.get_results()
        
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
        self.ctx.current_optimize_session += 1    # IMPORTANT, otherwise infinity loop
        self.ctx.seed += 1 # IMPORTANT the seed should update for every generation otherwise mutate offspring is the same
        
        # keep and mating parents selection
        self.ctx.elitism, mating_parents = _rank_selection(
            self.ctx.population, 
            self.ctx.fitness, 
            self.ctx.num_elitism,
            self.ctx.num_mating_parents,
        )
        
        # EXPERIMENTAL!!
        # N_offspring = N_pop - 2 * N_elitism
        # since mutate using gaussing for the other N_elitism
        
        # crossover
        num_offsprings = self.ctx.num_population - 2 * self.ctx.num_elitism
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
        # TODO: very tricky, find a better way to do this
        # Now for the first generation, run the local search for unmutated elitism
        # for the rest of generation, run the local search for mutated elitism
        self.report(f'current_optimize_session: {self.ctx.current_optimize_session}')
        if self.ctx.current_optimize_session == 1:
            to_mutated_elitism = self.ctx.elitism
        else:
            to_mutated_elitism = self.ctx.mut_elitism
            
        for ind in to_mutated_elitism:
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
        for _ in range(self.ctx.num_elitism):
            child = self.ctx.workchain_elitism.pop()
            if not child.is_finished_ok:
                self.logger.warning(
                    f"Local search not finished ok",
                )
                return self.exit_codes.ERROR_LOCAL_SEARCH_NOT_FINISHED_OK
            
            local_min_elitism_lst.append(child.outputs.result['xs'])
            local_min_elitism_y_list.append(child.outputs.result['y'])
            
        # make sure workchain_eltism ctx is empty
        assert len(self.ctx.workchain_elitism) == 0
            
        self.ctx.local_min_elitism = np.array(local_min_elitism_lst)
        self.ctx.local_min_elitism_y = local_min_elitism_y_list
            
        # TODO check the local_search workchains are finished.
        # self.ctx.population = np.vstack((self.ctx.local_min_elitism, self.ctx.local_min_mut_elitism, self.ctx.local_min_mut_offspring))
        
        print("@@@number to comb", len(self.ctx.local_min_elitism), len(self.ctx.mut_elitism), len(self.ctx.mut_offspring))
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
                'current_generation': self.ctx.current_optimize_session,
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

    print("HERERRR:", num_elitism, population.shape[1])
    keep_parents = np.empty((num_elitism, population.shape[1]), dtype=np.float64)
    for i in range(num_elitism):
        # set i-th best ind to i row
        keep_parents[i, :] = population[fitness_sorted[i], :].copy()
        
    mating_parents = np.empty((num_mating_parents, population.shape[1]), dtype=np.float64)
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
    offspring = np.empty((num_offsprings, num_genes), dtype=np.float64)
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
    mut_inds = np.empty([num_inds, num_genes], dtype=np.float64)
    for i in range(num_inds):
        _d_ind = {}

        # get all genes items key is the name of gene parameter e.g. rc(5)
        for j, (k, v) in enumerate(genes.items()):
            _d_ind[k] = inds[i][j]
            
        for j, (k, v) in enumerate(genes.items()):
            old_value = inds[i, j]
            space = v['space']
            gene_type = v['type']
            # based on the probability, keep original value if not being hit
            refto_param = space.get("refto", None) # change from 0.0 as base
            if refto_param:
                base = _d_ind[refto_param]
            else:
                base = 0.0
            if random.random() < gene_mutate_probability:

                if gaussian and gene_type == 'int':
                    # when gaussian set, which means run mutation for elitism only mutate continous gene.
                    _d_ind[k] = old_value
                    continue

                if gaussian:
                    x = random.gauss(old_value, sigma=old_value/10)
                else:
                    x = random.uniform(space['low'], space['high'])
                    
                # Get the new mutated value
                new_value = x + base
                
                if gene_type == 'int':
                    new_value = int(round(new_value))
                else:
                    new_value = round(new_value, 4)
                    
                # when gaussian applied to int type gene, it may not change because of rounding
                # we force to change it by +1 or -1
                if new_value == old_value:
                    # if it is lower bound, add 1
                    if new_value == space['low']:
                        new_value += 1
                    # if it is upper bound, minus 1
                    elif new_value == space['high']:
                        new_value -= 1
                    # otherwise, random +1 or -1
                    else:
                        new_value += random.choice([-1, 1])

                # Set the mutated gene value
                _d_ind[k] = new_value
                     
            else:
                # Set the gene value to original value
                _d_ind[k] = old_value

            
        # Set the individual's chromosome
        mut_inds[i, :] = list(_d_ind.values())
                
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