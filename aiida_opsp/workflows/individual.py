import random
from aiida.engine import WorkChain, while_, ToContext
from aiida import orm

from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS
from aiida_opsp.utils.merge_input import individual_to_inputs


class GenerateValidIndividual(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('evaluate_process', **PROCESS_INPUT_KWARGS)
        spec.input('variable_info', valid_type=orm.Dict)
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)
        
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

    def setup(self):
        """Setup inputs"""
        self.ctx.count = 0
        self.ctx.should_continue = True

    def should_continue(self):
        if not self.ctx.count < 10:
            self.report("reach the maximum iteration")
            return False

        return self.ctx.should_continue

    def generate(self):
        # before generate and continue, check if the previous run is okay
        # don't do it the first time where the self.ctx.individual is not set
        if "individual" in self.ctx:
            process = self.ctx.evaluate
            
            if process.is_finished_ok:
                self.ctx.final_individual = self.ctx.individual
                self.ctx.should_continue = False        
        
        self.ctx.individual = generate_random_individual(self.inputs.variable_info.get_dict())

    def evaluate(self):
        if not self.ctx.should_continue:
            return None

        evaluate_process = load_object(self.inputs.evaluate_process.value)

        # submit evaluation process for the individual
        inputs = individual_to_inputs(self.ctx.individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
        process = self.submit(evaluate_process, **inputs)
        # at most did MAX_ITERATION
        self.ctx.count += 1
        
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
            # get the base value that the current variable depends on
            ref_to_val = individual[ref_to_key]

            # only support refer to the same type
            assert isinstance(ref_to_val, eval(var_type))
            
            if var_type == "int":
                # randint is inclusive
                individual[key] = random.randint(var_range[0], var_range[1])
                individual[key] += ref_to_val
            elif var_type == "float":
                # uniform is inclusive
                var = random.uniform(var_range[0], var_range[1])
                individual[key] = round(var, 4) + ref_to_val
            else:
                raise ValueError("Unknown variable type")

    return individual

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