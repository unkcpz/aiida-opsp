from aiida.engine import WorkChain, while_, ToContext
from aiida import orm

from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS


class GenerateValidIndividual(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('parameters', valid_type=orm.Dict)
        spec.input('evaluate_process', **PROCESS_INPUT_KWARGS)
        spec.input('variable_info', valid_type=orm.Dict)
        spec.input('result_key', valid_type=orm.Str)
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
        
        self.ctx.individual = _random_individual()

    def evaluate(self):
        if not self.ctx.should_continue:
            return None

        evaluate_process = load_object(self.inputs.evaluate_process.value)

        # submit evaluation process for the individual
        inputs = self._individual_to_inputs(self.ctx.individual)
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
            self.out("final_individual", self.ctx.final_individual)
        else:
            return self.exit_codes.ERROR_CANNOT_GENERATE_VALID_INDIVIDUAL
