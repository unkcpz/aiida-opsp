import random
from aiida.engine import WorkChain, while_, ToContext
from aiida import orm

from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS
from aiida_opsp.utils.merge_input import individual_to_inputs

class _MixinGenerateValidIndividual(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('evaluate_process', **PROCESS_INPUT_KWARGS)
        spec.input('variable_info', valid_type=orm.Dict)
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)
        
        spec.input('seed', valid_type=orm.Int, default=lambda: orm.Int(2022))
        spec.input('max_iteration', valid_type=orm.Int, default=lambda: orm.Int(20))
        
        spec.outline(
            cls.setup,
            while_(cls.should_continue)(
                cls.generate,
                cls.evaluate,
                cls.inspect,
            ),
            cls.finalize,
        )
        spec.output('final_individual', valid_type=orm.Dict)

        spec.exit_code(201, 'ERROR_INVALID_INDIVIDUAL', message='The individual is invalid.')
        spec.exit_code(202, 'ERROR_CANNOT_GENERATE_VALID_INDIVIDUAL', message='Cannot generate valid individual')

    def setup(self):
        """Setup inputs"""
        self.ctx.count = 0
        self.ctx.should_continue = True

        # the seed need to be update upon the iter number, otherwise will always give the same result
        self.ctx.seed = self.inputs.seed.value

    def should_continue(self):
        if not self.ctx.count < self.inputs.max_iteration.value:
            self.report("reach the maximum iteration")
            return False

        return self.ctx.should_continue

    def generate(self):
        """The generate step need to be implemented in the subclass"""
        raise NotImplementedError("Please implement the generate method in the subclass")

    def evaluate(self):
        if not self.ctx.should_continue:
            return None

        evaluate_process = load_object(self.inputs.evaluate_process.value)

        # submit evaluation process for the individual
        inputs = individual_to_inputs(self.ctx.individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
        process = self.submit(evaluate_process, **inputs)
        # at most did MAX_ITERATION
        self.ctx.count += 1

        # update the seed to generate new input
        self.ctx.seed += self.ctx.count
        
        return ToContext(evaluate=process)
    
    def inspect(self):
        # check if the continue needed
        process = self.ctx.evaluate
        
        if process.is_finished_ok:
            self.ctx.final_individual = self.ctx.individual
            self.ctx.should_continue = False

    def finalize(self):
        if "final_individual" in self.ctx:
            self.out("final_individual", orm.Dict(dict=self.ctx.final_individual).store())
        else:
            return self.exit_codes.ERROR_CANNOT_GENERATE_VALID_INDIVIDUAL 

class GenerateRandomValidIndividual(_MixinGenerateValidIndividual):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

    def generate(self):
        # before generate and continue, check if the previous run is okay
        # don't do it the first time where the self.ctx.individual is not set
        if "individual" in self.ctx:
            process = self.ctx.evaluate
            
            if process.is_finished_ok:
                self.ctx.final_individual = self.ctx.individual
                self.ctx.should_continue = False        
        
        self.ctx.individual = generate_random_individual(self.inputs.variable_info.get_dict(), self.ctx.seed)

class GenerateMutateValidIndividual(_MixinGenerateValidIndividual):

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('init_individual', valid_type=orm.Dict)
        spec.input('probability', valid_type=orm.Float, default=lambda: orm.Float(0.5))
        
    def setup(self):
        """Setup inputs"""
        super().setup()

        # set the probability
        self.ctx.probability = self.inputs.probability.value

    def generate(self):
        # before generate and continue, check if the previous run is okay
        # don't do it the first time where the self.ctx.individual is not set
        if "individual" in self.ctx:
            process = self.ctx.evaluate
            
            if process.is_finished_ok:
                self.ctx.final_individual = self.ctx.individual
                self.ctx.should_continue = False        
        
        self.ctx.individual = generate_mutate_individual(self.inputs.init_individual.get_dict(), self.ctx.probability, self.inputs.variable_info.get_dict(), self.ctx.seed)

class GenerateCrossoverValidIndividual(_MixinGenerateValidIndividual):

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('parent1', valid_type=orm.Dict)
        spec.input('parent2', valid_type=orm.Dict) 
        
    def generate(self):
        # before generate and continue, check if the previous run is okay
        # don't do it the first time where the self.ctx.individual is not set
        if "individual" in self.ctx:
            process = self.ctx.evaluate
            
            if process.is_finished_ok:
                self.ctx.final_individual = self.ctx.individual
                self.ctx.should_continue = False        
        
        self.ctx.individual = generate_crossover_individual(self.inputs.parent1.get_dict(), self.inputs.parent2.get_dict(), self.inputs.variable_info.get_dict(), self.ctx.seed)

def hash_dict(d: dict):
    import hashlib
    import json

    # dict to JSON string representation
    json_str = json.dumps(d, sort_keys=True)
    
    # Hash the JSON using SHA-256
    hash_object = hashlib.sha256(json_str.encode())
    
    return hash_object.hexdigest()

def generate_mutate_individual(init_individual: dict, probability: float, variable_info, seed=None):
    """Generate a mutate individual slightly different from given one.
    
    The mutate happened for every gene when the probability hit.
    The seed is the combination of base seed and a int generate from init_individual.
    
    For the continues parameters (gene), gaussion will be applied, 
    For the interge parameterrs (gene), jump +/- 1 in the boundary.
    """
    random.seed(f'{hash_dict(init_individual)}_{seed}')
    individual = init_individual.copy()
    
    for key, info in variable_info.items():
        if not random.random() < probability:
            # not mutate
            continue

        var_range = info["space"]["range"]
        var_type = info["var_type"]

        old_value = init_individual[key]
        
        ref_to_key = info["space"].get("ref_to", None)
        if ref_to_key is not None:
            # update the range based on the ref_to_key
            var_range = [i + init_individual[ref_to_key] for i in var_range]

        if var_type == "int":
            # randint is inclusive
            new_value = old_value + random.choice([-1, 1])

            # make sure the new value is in the range
            individual[key] = min(max(new_value, var_range[0]), var_range[1])
        elif var_type == "float":
            # uniform is inclusive
            new_value = round(random.gauss(old_value, sigma=old_value * 0.1), 4)

            # make sure the new value is in the range
            individual[key] = min(max(new_value, var_range[0]), var_range[1])
        else:
            raise ValueError("Unknown variable type")

    return individual


def generate_random_individual(variable_info, seed=None):
    """Generate a random individual"""
    random.seed(seed)
    individual = dict()

    # iterate over the variable_info to set variables that are not dependent on others
    for key, info in variable_info.items():
        var_range = info["space"]["range"]
        var_type = info["var_type"]
        
        ref_to_key = info["space"].get("ref_to", None)
        if ref_to_key is not None:
            continue
        else:
            if var_type == "int":
                # randint is inclusive
                individual[key] = random.randint(var_range[0], var_range[1])
            elif var_type == "float":
                # uniform is inclusive
                var = random.uniform(var_range[0], var_range[1])
                individual[key] = round(var, 4)
            else:
                raise ValueError("Unknown variable type")

    # iterate over the variable_info to set variables that are dependent on others
    for key, info in variable_info.items():
        var_range = info["space"]["range"]
        var_type = info["var_type"]
        
        ref_to_key = info["space"].get("ref_to", None)

        if ref_to_key is None:
            continue
        else:
            # get the new range based on the ref_to_key
            var_range = [i + individual[ref_to_key] for i in var_range]

            if var_type == "int":
                # randint is inclusive
                individual[key] = random.randint(var_range[0], var_range[1])
            elif var_type == "float":
                # uniform is inclusive
                var = random.uniform(var_range[0], var_range[1])
                individual[key] = round(var, 4)
            else:
                raise ValueError("Unknown variable type")

    return individual

def generate_crossover_individual(parent1: dict, parent2: dict, variable_info, seed=None):
    """Generate a offspring individual from two parents"""
    random.seed(f'{hash_dict(parent1)}_{hash_dict(parent2)}_{seed}')
    
    group_mapping = dict()
    child = dict()
    for key, info in variable_info.items():
        g = info.get("group", None)
        if g is not None:
            parent_selected = group_mapping.get(g, None)
            
            if parent_selected is None:
                parent_selected = random.choice([-1, 1])
                group_mapping[g] = parent_selected
            
            # select the parent
            if parent_selected < -1:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        else:
            parent_selected = random.choice([-1, 1])
            
            if parent_selected < -1:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]

    return child
        
        

def validate_individual(individual, variable_info):
    """Validate the individual"""
    for key, info in variable_info.items():
        var_range = info["space"]["range"]
        ref_to = info["space"].get("ref_to", None)
        var_type = info["var_type"]
        
        value = individual[key]
        
        if ref_to is not None:
            ref_to_val = individual[ref_to]
            var_range = [var_range[0] + ref_to_val, var_range[1] + ref_to_val]
        
        if var_type == "int":
            if not isinstance(value, int):
                raise ValueError(f"The variable type is {type(value)} not int")
            if not var_range[0] <= value <= var_range[1]:
                raise ValueError(f"The variable value {value} is not in the range")
        elif var_type == "float":
            if not isinstance(value, float):
                raise ValueError(f"The variable type is {type(value)} not float")
            if not var_range[0] <= value <= var_range[1]:
                raise ValueError(f"The variable value {value} is not in the range")
        else:
            raise ValueError("Unknown variable type")
    
    return True