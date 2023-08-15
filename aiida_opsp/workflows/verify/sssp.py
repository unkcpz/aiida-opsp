from aiida.engine import WorkChain, ToContext
from aiida import orm

from aiida_opsp.calcjob import OncvPseudoCalculation

from aiida_sssp_workflow.workflows.measure.precision import PrecisionMeasureWorkChain

# generate pseudo and run SSSP verfication on it
class SSSPVerificationWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.expose_inputs(OncvPseudoCalculation, namespace='oncvpsp')
        spec.expose_inputs(PrecisionMeasureWorkChain, namespace='sssp', exclude=('pseudo'))
        spec.expose_outputs(OncvPseudoCalculation, namespace='oncvpsp')
        spec.expose_outputs(PrecisionMeasureWorkChain, namespace='sssp')
        spec.output('result', valid_type=orm.Float)

        spec.outline(
            cls.generate,
            cls.generate_inspect,
            cls.evaluate,
            cls.evaluate_inspect,
            cls.finalize,
        )
        spec.exit_code(201, 'ERROR_GENERATE_PSEUDO_FAILED', message='The `generate` step failed.')
        spec.exit_code(202, 'ERROR_EVALUATE_PSEUDO_FAILED', message='The `evaluate` step, e.g. SSSP measure workflow failed.')
        
    def generate(self):
        inputs = self.exposed_inputs(OncvPseudoCalculation, namespace='oncvpsp')
        # always dump psp for SSSP verification 
        if 'dump_psp' in inputs and inputs['dump_psp'].value is False:
            self.logger.warning('dump_psp is set to True for SSSP verification.')
            inputs['dump_psp'] = orm.Bool(True)
        else:
            inputs['dump_psp'] = orm.Bool(True)

        self.report("generating pseudo...")
        running = self.submit(OncvPseudoCalculation, **inputs)
        
        return ToContext(oncvwf=running)

    def generate_inspect(self):
        workchain = self.ctx.oncvwf
        
        # Same from OncvPseudoBaseWorkChain.finalize need to refactor and reuse
        # only parse and set result when finish ok
        if not workchain.is_finished_ok:
            if 500 < workchain.exit_status < 600:
                self.report(
                    f"WF for oncv failed with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_GENERATE_PSEUDO_FAILED
            else:
                self.report(
                    f"WF for oncv finished with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_GENERATE_PSEUDO_FAILED

        self.out_many(
            self.exposed_outputs(workchain, OncvPseudoCalculation)
        )

        # Get the pseudopotential
        self.ctx.pseudo = workchain.outputs.output_pseudo

    def evaluate(self):
        # SSSP verification 
        inputs = self.exposed_inputs(PrecisionMeasureWorkChain, namespace='sssp')

        if "options" in inputs:
            options = inputs.options.get_dict()
        else:
            from aiida_sssp_workflow.utils import get_default_options

            options = get_default_options(
                with_mpi=True,
            )

        if "parallelization" in inputs:
            parallelization = inputs.parallelization.get_dict()
        else:
            parallelization = {}

        inputs['pseudo'] = self.ctx.pseudo
        inputs['options'] = orm.Dict(dict=options)
        inputs['parallelization'] = orm.Dict(dict=parallelization)

        self.report("running SSSP measure workflow...")
        running = self.submit(PrecisionMeasureWorkChain, **inputs)

        return ToContext(measurewf=running)

    def evaluate_inspect(self):
        workchain = self.ctx.measurewf
        
        if not workchain.is_finished_ok:
            return self.exit_codes.ERROR_EVALUATE_PSEUDO_FAILED
        
        self.out_many(
            self.exposed_outputs(
                workchain, 
                PrecisionMeasureWorkChain,
                namespace='sssp'
            )
        )

        self.ctx.conf_nu_mapping = dict()
        for key, value in workchain.outputs.output_parameters.get_dict().items():
            nu = value['nu']
            self.ctx.conf_nu_mapping[key] = nu
            
    def finalize(self):
        # calculate the result from the mapping we use the maximum nu of all configurations
        self.out('result', orm.Float(max(self.ctx.conf_nu_mapping.values())))