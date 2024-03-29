from aiida.engine import WorkChain, ToContext, calcfunction
from aiida import orm

from aiida_opsp.calcjob import OncvPseudoCalculation

from aiida_sssp_workflow.workflows.measure.precision import PrecisionMeasureWorkChain

# generate pseudo and run SSSP verfication on it
class SSSPVerificationWorkChain(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('upper_bound_ecutwfc', valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.expose_inputs(OncvPseudoCalculation, namespace='oncvpsp')
        spec.expose_inputs(PrecisionMeasureWorkChain, namespace='sssp', exclude=('pseudo', 'wavefunction_cutoff', 'charge_density_cutoff'))
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
            self.logger.warning('dump_psp is set to True for SSSP verification, not needed to set it manually.')

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
            self.exposed_outputs(workchain, OncvPseudoCalculation, namespace='oncvpsp')
        )

        # Get the pseudopotential
        self.ctx.pseudo = workchain.outputs.output_pseudo

        # read the max cutoff in Ha, need to convert to Ry
        max_ecutwfc = workchain.outputs.output_parameters['max_ecut'] * 2

        # If the max_ecutwfc is larger than the upper bound, use the upper bound
        # I assume when using not enough cutoff will lead to a larger nu.
        if max_ecutwfc > self.inputs.upper_bound_ecutwfc.value:
            self.logger.warning(f"max_ecutwfc {max_ecutwfc} > upper_bound_ecutwfc {self.inputs.upper_bound_ecutwfc.value}, will use upper_bound_ecutwfc")
            self.ctx.max_ecutwfc = self.inputs.upper_bound_ecutwfc.value
        else:
            self.ctx.max_ecutwfc = max_ecutwfc

        self.ctx.max_ecutrho = self.ctx.max_ecutwfc * 4 # since we only test for NC, the plugin is integrated with oncvpsp

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

        inputs["wavefunction_cutoff"] = orm.Float(self.ctx.max_ecutwfc)
        inputs["charge_density_cutoff"] = orm.Float(self.ctx.max_ecutrho)
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

        self.ctx.sssp_output_parameters = workchain.outputs.output_parameters
            
    def finalize(self):
        # calculate the result from the mapping we use the maximum nu of all configurations
        result = get_max_nu(self.ctx.sssp_output_parameters)
        self.out('result', result)

@calcfunction
def get_max_nu(output_parameters):
    nu_lst = list()
    for value in output_parameters.get_dict().values():
        nu_lst.append(value['nu'])
        
    return orm.Float(max(nu_lst))