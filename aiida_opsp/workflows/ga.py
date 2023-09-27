import random

from aiida import orm
from aiida.engine import WorkChain, while_, if_
from aiida.plugins.entry_point import load_entry_point_from_string
import math

from aiida_opsp.workflows.individual import GenerateRandomValidIndividual, GenerateMutateValidIndividual, GenerateCrossoverValidIndividual
from aiida_opsp.utils.merge_input import individual_to_inputs

def validate_ga_parameters(ga_parameters, _):
    parameters = ga_parameters.get_dict()
    num_individuals = parameters.get('num_individuals')
    if num_individuals is None:
        return 'num_individuals is not defined'
    
    num_mating_individuals = parameters.get('num_mating_individuals')
    if num_mating_individuals is None:
        return 'num_mating_individuals is not defined'
    
    if num_mating_individuals < 2:
        return 'num_mating_individuals should not less than 2'
    
    num_offspring_individuals = parameters.get('num_offspring_individuals')
    if num_offspring_individuals is None:
        return 'num_offspring_individuals is not defined'

    num_elite_individuals = parameters.get('num_elite_individuals')
    if num_elite_individuals is None:
        return 'num_elite_individuals is not defined'

    num_new_individuals = parameters.get('num_new_individuals')
    if num_new_individuals is None:
        return 'num_new_individuals is not defined'    

    num_mediocre_individuals = num_individuals - 2 * num_elite_individuals - num_new_individuals - num_offspring_individuals
    if num_mediocre_individuals < 0:
        return 'num_mediocre_individuals should not less than 0'

    return None
    

class GeneticAlgorithmWorkChain(WorkChain):
    """WorkChain to run GA """

    # the maximum number of the best individuals not updated
    _MAX_THEBEST_COUNT = 10
    
    @classmethod
    def define(cls, spec):
        """Specify imputs and outputs"""
        super().define(spec)
        spec.input('ga_parameters', valid_type=orm.Dict, validator=validate_ga_parameters)
        spec.input('generate_evaluate_process', valid_type=orm.Str, help='Process which produces the pseudopotential, no need to get the score.')
        spec.input('score_evaluate_process', valid_type=orm.Str, help='Process which produces the result to be optimized, produce the score of the sample.')
        spec.input_namespace('score_evaluate_parameters', required=False, dynamic=True)
        spec.input('variable_info', valid_type=orm.Dict)
        spec.input('result_key', valid_type=orm.Str)   
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)

        spec.input('local_optimization_process', valid_type=orm.Str, help='Process which produces the result to be optimized.', required=False)
        spec.input('local_optimization_parameters', valid_type=orm.Dict, required=False)
        
        spec.outline(
            cls.init_setup,
            cls.prepare_init_population_run,
            cls.prepare_init_population_inspect,
            while_(cls.should_continue)(
                cls.evaluation_run,    # calc fitness of current generation
                cls.evaluation_inspect,
                cls.group_individuals_by_scores,
                cls.state_update,
                # ----------------- selection -----------------
                # the individuals are update in the ctx
                # to stop the iteration I set `should_continue` flag in previous state_update step, 
                # Which means the following steps also cross-by therefore the following steps need to be skipped
                # in the final iteration after the should_continue flag is set to False.
                cls.crossover_run,
                cls.crossover_inspect,
                cls.mutate_run,
                cls.mutate_inspect,
                cls.new_individuals_run,
                cls.new_individuals_inspect,
                cls.combine_population,
                if_(cls.should_run_local_optimization)(
                    cls.local_optimization_run,
                    cls.local_optimization_inspect,
                ),
                # ----------------- selection -----------------
            ),
            cls.finalize,   # stop iteration and get results
        )
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('final_individual', valid_type=orm.Dict)
        
        spec.exit_code(201, 'ERROR_PREPARE_INIT_POPULATION_FAILED', message='Failed to prepare init population')
        spec.exit_code(202, 'ERROR_SCORE_EVALUATE_PROCESS_FAILED', message='Failed to evaluate')
        spec.exit_code(203, 'ERROR_FITNESS_HAS_WRONG_NUM_OF_RESULTS', message='Fitness has wrong number of results')
        spec.exit_code(204, 'ERROR_MUTATE_NOT_FINISHED_OK', message='Mutate not finished okay')
        spec.exit_code(205, 'ERROR_NEW_INDIVIDUALS_NOT_FINISHED_OK', message='New individuals not finished okay')
        spec.exit_code(206, 'ERROR_LOCAL_OPTIMIZATION_NOT_FINISHED_OK', message='Local optimization not finished okay')

         
    def init_setup(self):
        """prepare initial ctx"""
        
        # to store const parameters in ctx over GA procedure
        ga_parameters = self.inputs.ga_parameters.get_dict()
        
        # init current optimize session aka generation in GA
        self.ctx.current_generation = 1

        self.ctx.should_continue = True
        
        self.ctx.seed = ga_parameters['seed']

        self.ctx.num_generations = ga_parameters['num_generations']
        self.ctx.num_individuals = ga_parameters['num_individuals']
        self.ctx.num_elite_individuals = ga_parameters['num_elite_individuals']
        self.ctx.num_new_individuals = ga_parameters['num_new_individuals']
        self.ctx.num_offspring_individuals = ga_parameters['num_offspring_individuals']
        self.ctx.num_mediocre_individuals = self.ctx.num_individuals - 2 * self.ctx.num_elite_individuals - self.ctx.num_new_individuals - self.ctx.num_offspring_individuals
        
        self.ctx.num_mating_individuals = ga_parameters['num_mating_individuals']
        self.ctx.elite_individual_mutate_probability = ga_parameters['elite_individual_mutate_probability']
        self.ctx.mediocre_individual_mutate_probability = ga_parameters['mediocre_individual_mutate_probability']

        individual_generate_max_iteration = ga_parameters.get('individual_generate_max_iteration', 20)
        self.ctx.individual_generate_max_iteration = orm.Int(individual_generate_max_iteration)

        # set base evaluate process
        self.ctx.generate_evaluate_process = load_entry_point_from_string(self.inputs.generate_evaluate_process.value)
        self.ctx.score_evaluate_process = load_entry_point_from_string(self.inputs.score_evaluate_process.value)

        # evaluate process for local optimization
        if 'local_optimization_process' in self.inputs:
            self.ctx._should_run_local_optimization = True
            
            self.ctx.local_optimization_process = load_entry_point_from_string(self.inputs.local_optimization_process.value) # nelder_mead
            self.ctx.local_optimize_evaluate_process = load_entry_point_from_string(self.inputs.local_optimization_parameters['evaluate_process']) # inner evaluate process
            # local optimization interval is for how many generations to run local optimization once
            self.ctx.local_optimization_interval = self.inputs.local_optimization_parameters.get_dict().get('interval', 1)
        else:
            self.ctx._should_run_local_optimization = False

        # counting the best individual appear times by generations
        self.ctx.max_thebest_count = ga_parameters.get('max_thebest_count', self._MAX_THEBEST_COUNT)

        # tmp_retrive_key_storage
        # This is for store the key so the parser step know which process to fetch from ctx
        # It needs to be kept empty in between a run-inspect session.
        self.ctx.tmp_retrive_key_storage = []
        
    def prepare_init_population_run(self):
        inputs = {
            'evaluate_process': self.ctx.generate_evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
            'max_iteration': self.ctx.individual_generate_max_iteration,
        }

        evaluates = dict()
        for idx in range(self.ctx.num_individuals):
            new_seed = self.ctx.seed + idx  # increment the seed, since we don't want every individual is the same ;)
            inputs['seed'] = orm.Int(new_seed)
            node = self.submit(GenerateRandomValidIndividual, **inputs)

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
            if not proc.is_finished_ok:
                return self.exit_codes.ERROR_PREPARE_INIT_POPULATION_FAILED
            else:
                population.append(proc.outputs.final_individual.get_dict())          

        self.ctx.population = population

        if not len(self.ctx.population) == self.ctx.num_individuals:
            return self.exit_codes.ERROR_PREPARE_INIT_POPULATION_FAILED            
            
        
    def should_continue(self):
        """return a bool, whether create new generation"""
        return self.ctx.should_continue
        
    def evaluation_run(self):
        """run the pseudo evaluate process to calc the loss function of each individual
        """
        self.report(f'On fitness at generation: {self.ctx.current_generation}')

        # submit evaluation for the individuals of a population
        evaluates = dict()
        for idx, individual in enumerate(self.ctx.population):
            oncvpsp_inputs = individual_to_inputs(individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
            if 'score_evaluate_parameters' not in self.inputs:
                inputs = oncvpsp_inputs
            else:
                inputs = dict(self.inputs.score_evaluate_parameters)
                inputs['oncvpsp'] = oncvpsp_inputs

            node = self.submit(self.ctx.score_evaluate_process, **inputs)

            retrive_key = f'_EVAL_IND_{idx}'
            
            # store the individual object in the extra for after use
            node.base.extras.set('individual', individual)

            optimize_info = {
                'generation': self.ctx.current_generation,
                'retrive_key': retrive_key,
            }
            node.base.extras.set('optimize_mode', 'genetic-algorithm')
            node.base.extras.set('optimize_info', optimize_info)

            evaluates[retrive_key] = node
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)
    
    def evaluation_inspect(self):
        """-> fitness parser, calc best solution"""
        self.report('Checking finished evaluations.')

        # dict of retrive key and score of the individual
        scores = {}
        individuals = {}
    
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            key = self.ctx.tmp_retrive_key_storage.pop(0)

            self.report(f'Retrieving output for evaluation {key}')

            proc: orm.WorkChainNode = self.ctx[key]
            individuals[key] = proc.base.extras.get('individual')
            if not proc.is_finished_ok:
                # When evaluate process failed it can be
                # - the parameters are not proper, this should result the bad score for the GA input
                # - the evalute process failed for resoure reason, should raise and reported.
                # - TODO: Test configuration 0 is get but no other configuration results -> check what output look like
                if proc.exit_status > 200: # All the exit code > 200 are considered as evaluation error and set the score to inf
                    scores[key] = math.inf
                else:
                    return self.exit_codes.ERROR_SCORE_EVALUATE_PROCESS_FAILED
            else:
                result_key = self.inputs.result_key.value
                scores[key] = proc.outputs[result_key].value
            
        # scores store the results of individual score of a population of this generation
        if len(scores) != self.ctx.num_individuals:
            self.report(f'Number of scores ({len(scores)}) does not match number of individuals ({self.ctx.num_individuals})')
            return self.exit_codes.ERROR_FITNESS_HAS_WRONG_NUM_OF_RESULTS

        # order the fitness dict by score
        self.ctx.sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        self.ctx.sorted_individuals = {key: individuals[key] for key in self.ctx.sorted_scores}

        self.logger.warning(f"scores: {list(self.ctx.sorted_scores.values())}")

    def group_individuals_by_scores(self):
        self.report('Grouping individuals to categories.')

        # create ctx for each group
        self.ctx.elite_individuals = []
        self.ctx.mutate_elite_individuals = []
        self.ctx.mediocre_individuals = []
         
        individuals_in_order = list(self.ctx.sorted_individuals.values())
        for _ in range(self.ctx.num_elite_individuals):
            self.ctx.elite_individuals.append(individuals_in_order.pop(0))
            
        for _ in range(self.ctx.num_elite_individuals):
            self.ctx.mutate_elite_individuals.append(individuals_in_order.pop(0))
            
        # XXX EXPERIMENTAL, not sure if it is good
        # The number of mediocre individuals total - 2 * num_elite_individuals - num_new_individuals
        self.ctx.num_mediorce_individuals = self.ctx.num_individuals - 2 * self.ctx.num_elite_individuals - self.ctx.num_new_individuals
        for _ in range(self.ctx.num_mediorce_individuals):
            self.ctx.mediocre_individuals.append(individuals_in_order.pop(0))

    def state_update(self):
        # GA iteration state update and check
        self.ctx.current_generation += 1
        if self.ctx.current_generation > self.ctx.num_generations:
            self.report(f'Final generation {self.ctx.num_generations} reached, stopping.')
            self.ctx.should_continue = False

        # the best fitness is not improved for a maximum times, stop the optimization
        epsilon = 1e-3
        thebest = list(self.ctx.sorted_scores.values())[0]
        if 'thebest' not in self.ctx:
            # the first generation
            self.ctx.thebest = thebest
            self.ctx.thebest_count = 0

        if thebest < self.ctx.thebest - epsilon:
            self.report(f'The best score is improved from {self.ctx.thebest} to {thebest}, from generation {self.ctx.current_generation-1} to {self.ctx.current_generation}')
            self.ctx.thebest = thebest
            self.ctx.thebest_count = 0
        elif abs(self.ctx.thebest - thebest) < epsilon:
            # the best fitness is not improved
            self.ctx.thebest_count += 1
        else:
            self.ctx.thebest_count = 0
            self.logger.warning(f'The best score is increase, happened because local minimization make it worse in SSSP verification. Chaning {self.ctx.thebest} -> {thebest}, check if it is a problem.')

        if self.ctx.thebest_count > self.ctx.max_thebest_count:
            self.report(f'The best score is not improved for {self.ctx.thebest_count} times, stopping.')
            self.ctx.should_continue = False

    def crossover_run(self):
        """Use previous half of the population with better fitness to crossover to create new individuals
        """
        if not self.ctx.should_continue:
            return None

        inputs = {
            'evaluate_process': self.ctx.generate_evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
            'max_iteration': self.ctx.individual_generate_max_iteration,
        }
        
        evaluates = dict()

        # use half of the population to crossover
        mating_pool = self.ctx.population[:self.ctx.num_mating_individuals]
        self.ctx.offspring_individuals = []
        
        for idx in range(self.ctx.num_new_individuals):
            # set seed for mating parents selection
            new_seed = self.ctx.seed + idx
            random.seed(new_seed)

            # select two individuals from the mating pool
            parent1, parent2 = random.sample(mating_pool, 2)

            inputs['seed'] = orm.Int(new_seed)
            inputs['parent1'] = orm.Dict(dict=parent1)
            inputs['parent2'] = orm.Dict(dict=parent2)
            
            # crossover the two individuals
            node = self.submit(GenerateCrossoverValidIndividual, **inputs)
            
            retrive_key = f'_CROSSOVER_IND_{idx}'
            evaluates[retrive_key] = node
            
            self.ctx.tmp_retrive_key_storage.append(retrive_key)

        return self.to_context(**evaluates)
        
    def crossover_inspect(self):
        if not self.ctx.should_continue:
            return None
        
        offspring_individuals = list()
        
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            retrive_key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f'Retrieving output for crossover {retrive_key}')
            node = self.ctx[retrive_key]
            if not node.is_finished_ok:
                return self.exit_codes.ERROR_CROSSOVER_NOT_FINISHED_OK
            else:
                offspring_individuals.append(node.outputs['final_individual'].get_dict())
                
        self.ctx.offspring_individuals = offspring_individuals

    def mutate_run(self):
        if not self.ctx.should_continue:
            return None

        inputs = {
            'evaluate_process': self.ctx.generate_evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
            'max_iteration': self.ctx.individual_generate_max_iteration,
        }
        
        evaluates = dict()

        # mutation elitism
        for idx, individual in enumerate(self.ctx.elite_individuals):
            new_seed = self.ctx.seed + idx  # increment the seed, since we don't want every individual is the same ;)
            inputs['seed'] = orm.Int(new_seed)
            inputs['init_individual'] = orm.Dict(dict=individual)
            inputs['probability'] = orm.Float(self.ctx.elite_individual_mutate_probability)

            node = self.submit(GenerateMutateValidIndividual, **inputs)

            retrive_key = f'_VALID_MUTATE_ELITE_{idx}'
            evaluates[retrive_key] = node

            self.ctx.tmp_retrive_key_storage.append(retrive_key)

        # mutation mediocre
        for idx, individual in enumerate(self.ctx.mediocre_individuals):
            new_seed = self.ctx.seed + idx  # increment the seed, since we don't want every individual is the same ;)
            inputs['seed'] = orm.Int(new_seed)
            inputs['init_individual'] = orm.Dict(dict=individual)
            inputs['probability'] = orm.Float(self.ctx.mediocre_individual_mutate_probability)
            
            node = self.submit(GenerateMutateValidIndividual, **inputs)
            
            retrive_key = f'_VALID_MUTATE_MEDIOCRE_{idx}'
            evaluates[retrive_key] = node
            
            self.ctx.tmp_retrive_key_storage.append(retrive_key)

        return self.to_context(**evaluates)

    def mutate_inspect(self):
        if not self.ctx.should_continue:
            return None
        
        new_mutate_elite_individuals = list()
        new_mutate_mediocre_individuals = list()

        while len(self.ctx.tmp_retrive_key_storage) > 0:
            retrive_key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f"Retriving output for evaluation {retrive_key}")
            node = self.ctx[retrive_key]
            if not node.is_finished_ok:
                self.report(f'node {node.pk} is not finished ok')
                return self.exit_codes.ERROR_MUTATE_NOT_FINISHED_OK
            else:
                if "_VALID_MUTATE_ELITE_" in retrive_key:
                    new_mutate_elite_individuals.append(node.outputs['final_individual'].get_dict())
                if "_VALID_MUTATE_MEDIOCRE_" in retrive_key:
                    new_mutate_mediocre_individuals.append(node.outputs['final_individual'].get_dict())
                    
        self.ctx.mutate_elite_individuals = new_mutate_elite_individuals
        self.ctx.mutate_mediocre_individuals = new_mutate_mediocre_individuals

    def new_individuals_run(self):
        """Create new individuals to increase the diviersity of the population"""
        if not self.ctx.should_continue:
            return None
        
        inputs = {
            'evaluate_process': self.ctx.generate_evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
            'max_iteration': self.ctx.individual_generate_max_iteration,
        }
        
        evaluates = dict()
        for idx in range(self.ctx.num_new_individuals):
            new_seed = self.ctx.seed + idx + self.ctx.current_generation + 42  # increment the seed in a way depend also on the generation, since we don't want every individual is the same ;)
            inputs['seed'] = orm.Int(new_seed)
            node = self.submit(GenerateRandomValidIndividual, **inputs)
            
            retrive_key = f'_VALID_NEW_IND_{idx}'
            evaluates[retrive_key] = node
            
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)

    def new_individuals_inspect(self):
        if not self.ctx.should_continue:
            return None
        
        new_individuals = list()
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            retrive_key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f"Retriving output for evaluation {retrive_key}")
            node = self.ctx[retrive_key]
            if not node.is_finished_ok:
                self.report(f'node {node.pk} is not finished ok')
                return self.exit_codes.ERROR_NEW_INDIVIDUAL_NOT_FINISHED_OK
            else:
                new_individuals.append(node.outputs['final_individual'].get_dict())
                    
        self.ctx.new_individuals = new_individuals

    def combine_population(self):
        """combine population"""
        if not self.ctx.should_continue:
            return None

        self.ctx.population = self.ctx.elite_individuals + self.ctx.mutate_elite_individuals + self.ctx.mutate_mediocre_individuals + self.ctx.new_individuals

    def should_run_local_optimization(self):
        """The condition to run local optimization"""
        # run local optimization every 5 generations
        if not self.ctx._should_run_local_optimization:
            return False
        else:
            return self.ctx.current_generation % self.ctx.local_optimization_interval == 0
        
    def local_optimization_run(self):
        """local optimization for the population"""
        if not self.ctx.should_continue:
            return None
        
        inputs = {
            'evaluate_process': self.ctx.local_optimize_evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
        }
        
        evaluates = dict()
        for idx, individual in enumerate(self.ctx.population):
            new_seed = self.ctx.seed + idx
            inputs['seed'] = orm.Int(new_seed)
            inputs['init_individual'] = orm.Dict(dict=individual)
            inputs['parameters'] = self.inputs.local_optimization_parameters
            inputs['result_key'] = self.inputs.result_key

            node = self.submit(self.ctx.local_optimization_process, **inputs)
            
            retrive_key = f'_LOCAL_OPT_{idx}'
            evaluates[retrive_key] = node
            
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)
    
    def local_optimization_inspect(self):
        if not self.ctx.should_continue:
            return None
        
        population = list()
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            retrive_key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f"Retriving output for evaluation {retrive_key}")
            node = self.ctx[retrive_key]
            if not node.is_finished_ok:
                self.logger.warning(f'node {node.pk} is not finished ok')

                # There is chance that for some very unlucky individuals, the local optimization process is not finished ok.
                # We don't want to stop the whole process because of this, so we just ignore this individual and use the original one.
                # Those are usually the individuals that are not very good, the low quality ones means it will be replaced by the new individuals in the next generation.
                population.append(node.inputs['init_individual'].get_dict())
            else:
                population.append(node.outputs['final_individual'].get_dict())
                    
        self.ctx.population = population
            
    def finalize(self):
        """Write to output the best individual and its fitness, and the its uuid of evaluation process"""
        self.report('finalize')

        best_key, final_individual = list(self.ctx.sorted_individuals.items())[0]
        score = self.ctx.sorted_scores[best_key]

        best_process = self.ctx[best_key]

        output_parameters = orm.Dict(dict={
            'score': score,
            'uuid': best_process.uuid,
        })

        self.out('output_parameters', output_parameters.store())
        self.out('final_individual', orm.Dict(dict=final_individual).store())
        