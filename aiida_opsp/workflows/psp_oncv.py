from aiida.engine import WorkChain, submit
from aiida import orm
from aiida.engine import ToContext
from aiida_opsp.calcjob import OncvPseudoCalculation

def penalty(crop_ldd=None, max_ecut=None, state_err_avg=None):
    if crop_ldd is None:
        return abs(9999)

    # set range for crop intg (-inf, -5) (-5, -2) (-2, 0) (0, 2) (2, 5) (5, inf)
    crop_weight_dict = {
        "ninf_n5": 0, 
        "n5_n2": 3, 
        "n2_0": 2.5, 
        "0_2": 2.5, 
        "2_5": 3, 
        "5_inf": 0.2,
    }
    
    state_type_weight_dict = {
        "bound": 1,
        "unbound": 0.1,
    }
    
    res_cost = 0.
    for ldd in crop_ldd:
        integ = ldd["integ"]
        st_w = state_type_weight_dict.get(ldd["state_type"])
        cw_w = crop_weight_dict.get(ldd["crop_range"])
        res_cost += integ * st_w * cw_w
    
    # an example of penalty function TBD

    # TODO: state_err_avg
    
    res_cost += max_ecut * 0.2
    
    # Search function need use min for best results error, the smaller the better so close to 0 is best
    return res_cost
    

class OncvPseudoBaseWorkChain(WorkChain):
    """Wrap of OncvPseudoCalculation calcjob
    calculate and output `results` as result_key for GA and add error handler"""
    
    @classmethod
    def define(cls, spec):
        super(OncvPseudoBaseWorkChain, cls).define(spec)

        spec.expose_inputs(OncvPseudoCalculation, exclude=['metadata'])
        spec.expose_outputs(OncvPseudoCalculation)
        spec.output('result', valid_type=orm.Float)

        spec.outline(
            cls.evaluate,
            cls.finalize,
        )
        spec.exit_code(201, 'ERROR_PARAMETERS_NOT_PROPER',
                    message='The oncv gereration failed because of bad parameters.')        
        spec.exit_code(301, 'ERROR_SUB_PROCESS_FAILED_ONCV',
                    message='The `ONCV` sub process failed.')

    def evaluate(self):
        # This is a bit improper: The new value should be created in a calculation.
        inputs = self.exposed_inputs(OncvPseudoCalculation)
        # import ipdb; ipdb.set_trace()
        running = self.submit(OncvPseudoCalculation, **inputs)
        
        return ToContext(oncvwf=running)
        
    def finalize(self):
        workchain = self.ctx.oncvwf
        
        # only parse and set result when finish ok
        # it can be test configuration parse error that no test is done but psp is generated
        if not workchain.is_finished_ok:
            if 500 < workchain.exit_status < 600:
                self.report(
                    f"WF for oncv failed parameters not proper with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_PARAMETERS_NOT_PROPER
            else:
                self.report(
                    f"WF for oncv failed with exit status {workchain.exit_status}"
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_ONCV
        
        self.out_many(
            self.exposed_outputs(workchain, OncvPseudoCalculation)
        )
        
        # a very experiment way to define evaluate value for accuracy of psp.
        d = dict(workchain.outputs.output_parameters)
        
        inputs = {
            "crop_ldd": d.get("crop_ldd", None),
            "max_ecut": d.get("max_ecut", 99),
            "state_err_avg": d.get("state_err_avg", 99),
        }
        
        result = penalty(**inputs)
                
        self.out('result', orm.Float(result).store())
        