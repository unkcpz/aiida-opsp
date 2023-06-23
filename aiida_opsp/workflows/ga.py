import numpy as np
import random

from aiida import orm
from aiida.engine import WorkChain, while_, append_
import math

from aiida_opsp.workflows.ls import LocalSearchWorkChain
from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS
from aiida_opsp.workflows.individual import GenerateRandomValidIndividual, GenerateMutateValidIndividual
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
        spec.input('variable_info', valid_type=orm.Dict)
        spec.input('result_key', valid_type=orm.Str)   
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)
        
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
                # when the iteration stop, the step following needs to be skipped
                # cls.crossover,
                cls.mutate_run,
                cls.mutate_inspect,
                cls.combine_population,
                # cls.local_optimization_run,
                # cls.local_optimization_inspect,
                # ----------------- selection -----------------
            ),
            cls.finalize,   # stop iteration and get results
        )
        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('final_individual', valid_type=orm.Dict)
        
        spec.exit_code(201, 'ERROR_PREPARE_INIT_POPULATION_FAILED', message='Failed to prepare init population')
        spec.exit_code(202, 'ERROR_EVALUATE_PROCESS_FAILED', message='Failed to evaluate')
        spec.exit_code(203, 'ERROR_FITNESS_HAS_WRONG_NUM_OF_RESULTS', message='Fitness has wrong number of results')
        spec.exit_code(204, 'ERROR_MUTATE_NOT_FINISHED_OK', message='Mutate not finished okay')

         
    def init_setup(self):
        """prepare initial ctx"""
        
        # to store const parameters in ctx over GA procedure
        parameters = self.inputs.parameters.get_dict()
        
        # init current optimize session aka generation in GA
        self.ctx.current_generation = 0

        self.ctx.should_continue = True
        
        self.ctx.seed = parameters['seed']
        self.ctx.num_generations = parameters['num_generations']
        self.ctx.num_individuals = parameters['num_individuals']
        self.ctx.num_elite_individuals = parameters['num_elite_individuals']
        self.ctx.num_mating_parents = parameters['num_mating_individuals']
        self.ctx.elite_individual_mutate_probability = parameters['elite_individual_mutate_probability']
        self.ctx.mediocre_individual_mutate_probability = parameters['mediocre_individual_mutate_probability']

        # set base evaluate process
        self.ctx.evaluate_process = load_object(self.inputs.evaluate_process.value)

        # tmp_retrive_key_storage
        # This is for store the key so the parser step know which process to fetch from ctx
        # It needs to be kept empty in between a run-inspect session.
        self.ctx.tmp_retrive_key_storage = []
        
    def prepare_init_population_run(self):
        inputs = {
            'evaluate_process': self.ctx.evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
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
            inputs = individual_to_inputs(individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
            node = self.submit(self.ctx.evaluate_process, **inputs)
            
            # store the individual object in the extra for after use
            node.base.extras.set('individual', individual)

            retrive_key = f'_EVAL_IND_{idx}'
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
            if not proc.is_finished_ok:
                # When evaluate process failed it can be
                # - the parameters are not proper, this should result the bad score for the GA input
                # - the evalute process failed for resoure reason, should raise and reported.
                # - TODO: Test configuration 0 is get but no other configuration results -> check what output look like
                if proc.exit_status == 201: # ERROR_PSPOT_HAS_NODE
                    scores[key] = math.inf
                else:
                    return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            else:
                result_key = self.inputs.result_key.value
                scores[key] = proc.outputs[result_key].value
                individuals[key] = proc.base.extras.get('individual')
            
        # scores store the results of individual score of a population of this generation
        if len(scores) != self.ctx.num_individuals:
            self.report(f'Number of scores ({len(scores)}) does not match number of individuals ({self.ctx.num_individuals})')
            return self.exit_codes.ERROR_FITNESS_HAS_WRONG_NUM_OF_RESULTS

        # order the fitness dict by score
        self.ctx.sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1]))
        self.ctx.sorted_individuals = {key: individuals[key] for key in self.ctx.sorted_scores}

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
        # The number of mediocre individuals total - 2 * num_elite_individuals
        number_mediocre_individual = self.ctx.num_individuals - 2 * self.ctx.num_elite_individuals
        for _ in range(number_mediocre_individual):
            self.ctx.mediocre_individuals.append(individuals_in_order.pop(0))

    def state_update(self):
        # GA iteration state update and check
        self.ctx.current_generation += 1
        if self.ctx.current_generation > self.ctx.num_generations:
            self.report(f'Final generation {self.ctx.num_generations} reached, stopping.')
            self.ctx.should_continue = False

        # TODO if the best fitness is not improved for a maximum times, stop the optimization

    # def crossover(self):
    # TODO: crossover will use the group tag in variable_info to decide which group of variables to crossover
    #     """crossover"""
    #     self.ctx.current_optimize_session += 1    # IMPORTANT, otherwise infinity loop
    #     self.ctx.seed += 1 # IMPORTANT the seed should update for every generation otherwise mutate offspring is the same
    #     
    #     # keep and mating parents selection
    #     self.ctx.elitism, mating_parents = _rank_selection(
    #         self.ctx.population, 
    #         self.ctx.fitness, 
    #         self.ctx.num_elitism,
    #         self.ctx.num_mating_parents,
    #     )
    #     
    #     # EXPERIMENTAL!!
    #     # N_offspring = N_pop - 2 * N_elitism
    #     # since mutate using gaussing for the other N_elitism
    #     
    #     # crossover
    #     num_offsprings = self.ctx.num_individuals - 2 * self.ctx.num_elitism
    #     self.ctx.offspring = _crossover(mating_parents, num_offsprings, seed=self.ctx.seed)
        

    def mutate_run(self):
        if not self.ctx.should_continue:
            return None

        inputs = {
            'evaluate_process': self.ctx.evaluate_process,
            'variable_info': self.inputs.variable_info,
            'fixture_inputs': self.inputs.fixture_inputs,
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

    def combine_population(self):
        """combine population"""
        if not self.ctx.should_continue:
            return None

        self.ctx.population = self.ctx.elite_individuals + self.ctx.mutate_elite_individuals + self.ctx.mutate_mediocre_individuals

        
    # def local_search(self):
    #     """ local_search of elitism
    #     """
    #     # TODO: very tricky, find a better way to do this
    #     # Now for the first generation, run the local search for unmutated elitism
    #     # for the rest of generation, run the local search for mutated elitism
    #     self.report(f'current_optimize_session: {self.ctx.current_optimize_session}')
    #     if self.ctx.current_optimize_session == 1:
    #         to_mutated_elitism = self.ctx.elitism
    #     else:
    #         to_mutated_elitism = self.ctx.mut_elitism
    #         
    #     for ind in to_mutated_elitism:
    #         ls_parameters = self.ctx.const_parameters['local_search_base_parameters']
    #         ls_parameters['init_vars'] = list(ind)
    #         
    #         inputs = {
    #             'parameters': orm.Dict(dict=ls_parameters),
    #             'evaluate_process': self.inputs.evaluate_process,
    #             'vars_info': self.inputs.vars_info,
    #             'result_key': self.inputs.result_key,
    #             'fixture_inputs': self.inputs.fixture_inputs,
    #         }
    #         running = self.submit(LocalSearchWorkChain, **inputs)
    #         self.to_context(workchain_elitism=append_(running))
    
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
        