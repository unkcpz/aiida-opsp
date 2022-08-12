import numpy as np
import random
from plumpy import if_
from pygments import highlight
import yaml

import aiida
from aiida import orm
from aiida.engine import WorkChain, while_
from aiida.engine.persistence import ObjectLoader
from functools import singledispatch
import math
import scipy.linalg as la

from plumpy.utils import AttributesFrozendict

from aiida.orm.nodes.data.base import to_aiida_type


# Generic process evaluation the code is ref from aiida-optimize
_YAML_IDENTIFIER = '!!YAML!!'

# simplex parameters from scipy.minimize
RHO = 1
CHI = 2
PSI = 0.5
SIGMA = 0.5

class Rosenbrock(WorkChain):
    
    @classmethod
    def define(cls, spec):
        super(Rosenbrock, cls).define(spec)
        
        spec.input('x')
        spec.input('y')
        spec.output('result')
        
        spec.outline(
            cls.run,
        )
        
    def run(self):
        x, y = self.inputs.x.value, self.inputs.y.value
        self.out('result', orm.Float((1 - x) ** 2 + 100 * (y - x**2) ** 2).store())
        
def create_init_simplex(xs, tol=0.2):
    ret = [xs]
    for i in range(len(xs)):
        #
        np.random.seed(1992+i)
        xss = list(np.array(xs) + np.random.uniform(low=-tol, high=tol, size=len(xs)))
        
        ret.append(xss)
        
    return ret

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


class LocalSearchWorkChain(WorkChain):
    """WorkChain to run GA """
    
    _EVAL_PREFIX = 'eval_'
    
    @classmethod
    def define(cls, spec):
        """Specify imputs and outputs"""
        super().define(spec)
        spec.input('parameters', valid_type=orm.Dict)
        spec.input('evaluate_process', help='Process which produces the result to be optimized.',
            **PROCESS_INPUT_KWARGS)
        spec.input('input_nested_keys', valid_type=orm.List)    # map gene to input of evaluate process in oder
        spec.input('result_key', valid_type=orm.Str)    # name of key to be the result for fitness
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)  # The fixed input parameters that will combined with change parameters
        
        spec.outline(
            cls.submit_init,  # prepare init population and parameters
            cls.update_init,
            while_(cls.do_iter)(
                cls.init_iter,
                cls.submit_reflection,
                cls.update_reflection,
                if_(cls.do_expansion)(
                    cls.submit_expansion,
                    cls.update_expansion,
                ),
                if_(cls.do_contraction)(
                    cls.submit_contraction,
                    cls.update_contraction,
                ),
                if_(cls.do_inside_contraction)(
                    cls.submit_inside_contraction,
                    cls.update_inside_contraction,
                ),
                if_(cls.do_shrink)(
                    cls.submit_shrink,
                    cls.update_shrink,
                ),
            ),
            cls.finalize,   # stop iteration and get results
        )
        spec.output('result', valid_type=orm.Dict)
        
        spec.exit_code(
            201,
            'ERROR_EVALUATE_PROCESS_FAILED',
            message='GA optimization failed because one of the evaluate processes did not finish ok.'
        )

    def _submit(self, name="unknow operation"):
        """
        Calculating the res of current simplex. 
            
        evaluate_process is the name of process
        """
        self.report(f'On evaluate at simplex: {self.ctx.num_iter} by operation {name}')

        evaluate_process = load_object(self.inputs.evaluate_process.value)
        
        # submit evaluation for the pop ind
        evals = {}
        for idx, ind in enumerate(self.ctx._to_evaluate_simplex):
            inputs = self._merge_nested_inputs(
                self.inputs.input_nested_keys.get_list(), 
                list(ind), 
                self.inputs.get('fixture_inputs', {})
            )
            node = self.submit(evaluate_process, **inputs)
            evals[self.eval_key(idx)] = node
            self.indices_to_retrieve.append(idx)

        return self.to_context(**evals)
        
    def submit_init(self):
        # to store const parameters in ctx over GA procedure
        parameters = self.inputs.parameters.get_dict()
        self.xtol = parameters['xtol']
        self.ftol = parameters['ftol']
        self.max_iter = parameters['max_iter']
        
        # new_simplex keep track the entities to evaluated can be three for initialize and then 
        # one for expansion, contraction and shrink
        # simplex only keep the final simplex
        self.ctx._to_evaluate_simplex = self.ctx.simplex = np.array(parameters['init_simplex'])
        
        assert len(self.ctx.simplex) == self.ctx.simplex.shape[1] + 1
        self.len_simplex = len(self.ctx.simplex)
        self.ctx.fun_simplex = [np.nan for _ in range(self.len_simplex)]
        self.ctx.num_iter = 0
        self.ctx.finished = False
        
        # record the next operation, can be expansion, contraction or shrink
        self.ctx.next_operation = "CONTINUE"
        self._submit(name='init')
        
    def update_init(self):
        _, outputs = self._parse_outputs(single=False)
        self.ctx.fun_simplex = outputs
    
    def _parse_outputs(self, single=True):
        """From evaluate result update res and simplex kept
        The result the less (a positive number close to 0) the better.
        """
        self.report('Checking finished evaluations and update simplex.')
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
                # !!! IMPORTANT, the small panalty function the better.
                outputs[idx] = eval_proc.outputs['result'].value
            
        fun_simplex = np.array([outputs[i] for i in range(len(self.ctx._to_evaluate_simplex))])
        
        self.report(f"Evaluate results: simplex inputs: {self.ctx._to_evaluate_simplex}, fun_simplex output {fun_simplex}.")
        
        if single:
            # take the only one of the evaluation
            return self.ctx._to_evaluate_simplex[0], fun_simplex[0]
        else:
            return self.ctx._to_evaluate_simplex, fun_simplex
        
        # output_report = []
        # for idx, ind in enumerate(self.ctx.population):
        #     key = self.eval_key(idx)
        #     proc = self.ctx[key]
        #     fitness = outputs[idx]
        #     output_report.append(f'idx={idx}  pk={proc.pk}: {ind} -> fitness={fitness}')
            
        # output_report_str = '\n'.join(output_report)
        
        # self.report(f'population and process pk:')
        # self.report(f'\n{output_report_str}')
        # self.report(self.ctx.best_solution)

    
    def do_iter(self):
        """return a bool, whether create new generation"""
        return not self.ctx.finished
        
    def init_iter(self):
        # do sort
        idx = np.argsort(self.ctx.fun_simplex)
        self.ctx.fun_simplex = np.take(self.ctx.fun_simplex, idx, axis=0)
        self.ctx.simplex = np.take(self.ctx.simplex, idx, axis=0)
        
        self.report(f"simplex is {self.ctx.simplex}")
        self.report(f"fun_simplex is {self.ctx.fun_simplex}")
        
        # check finished
        x_dist_max = np.max(la.norm(self.ctx.simplex[1:] - self.ctx.simplex[0], axis=-1))
        self.report(f"Maximum distance value for the simplex: {x_dist_max}")
        f_diff_max = np.max(np.abs(self.ctx.fun_simplex[1:] - self.ctx.fun_simplex[0]))
        self.report(f"Maximum function difference: {f_diff_max}")
        self.ctx.finished = (x_dist_max < self.xtol) and (f_diff_max < self.ftol)
        self.report(
            f"End of Nelder-Mead iteration {self.ctx.num_iter}, max number of iterations: {self.max_iter}."
        )
        if not self.ctx.finished:
            if self.ctx.num_iter >= self.max_iter:
                self.report("Number of iterations exceeded the maximum. Stop.")
                self.ctx.next_operation = "CONTINUE"    # no furture operation, continue as start from iter
                self.ctx.finished = True
                
        self.ctx.num_iter += 1
        
    @property
    def xbar(self):
        return np.average(self.ctx.simplex[:-1], axis=0)
        
    def submit_reflection(self):
        xr = (1 + RHO) * self.xbar - RHO * self.ctx.simplex[-1]
        self.ctx._to_evaluate_simplex = [xr]
        self._submit(name='reflection')
        
    def update_reflection(self):
        xr, fxr = self._parse_outputs()
        self.ctx.extra_points = {"xr": (xr, fxr)}
        
        if fxr < self.ctx.fun_simplex[0]:
            self.ctx.next_operation = "EXPANSION"
        else:
            if fxr < self.ctx.fun_simplex[-2]:
                self._update_last(xr, fxr)
                self.ctx.next_operation = "CONTINUE"
            else:
                if fxr < self.ctx.fun_simplex[-1]:
                    self.ctx.next_operation = "CONTRACTION"
                else:
                    self.ctx.next_operation = "INSIDE_CONTRACTION"
                    
    def _update_last(self, x, f):
        self.ctx.simplex[-1] = x
        self.ctx.fun_simplex[-1] = f
                
    def do_expansion(self):
        return self.ctx.next_operation == "EXPANSION"
    
    def submit_expansion(self):
        self.report("Submitting expansion step.")
        xe = (1 + RHO * CHI) * self.xbar - RHO * CHI * self.ctx.simplex[-1]
        self.ctx._to_evaluate_simplex = [xe]
        self._submit(name='expansion')
        
    def update_expansion(self):
        """
        Retrieve the results of an expansion step.
        """
        xe, fxe = self._parse_outputs()
        xr, fxr = self.ctx.extra_points["xr"]
        if fxe < fxr:
            self._update_last(xe, fxe)
        else:
            self._update_last(xr, fxr)
            
        self.ctx.next_operation = "CONTINUE"
        
    def do_contraction(self):
        return self.ctx.next_operation == "CONTRACTION"
    
    def submit_contraction(self):
        self.report("Submitting contraction step.")
        xc = (1 + PSI * RHO) * self.xbar - PSI * RHO * self.ctx.simplex[-1]
        self.ctx._to_evaluate_simplex = [xc]
        self._submit(name='contraction')
        
    def update_contraction(self):
        """
        Retrieve the results of an contraction step.
        """
        xc, fxc = self._parse_outputs()
        _, fxr = self.ctx.extra_points["xr"]
        if fxc < fxr:
            self._update_last(xc, fxc)
            self.ctx.next_operation = "CONTINUE"
        else:
            # shrink
            self.ctx.next_operation = "SHRINK"
            
    def do_inside_contraction(self):
        return self.ctx.next_operation == "INSIDE_CONTRACTION"
    
    def submit_inside_contraction(self):
        self.report("Submitting inside contraction step.")
        xcc = (1 - PSI) * self.xbar + PSI * self.ctx.simplex[-1]
        self.ctx._to_evaluate_simplex = [xcc]
        self._submit(name='inside_contraction')
        
    def update_inside_contraction(self):
        """
        Retrieve the results of an inside contraction step.
        """
        xcc, fxcc = self._parse_outputs()
        if fxcc < self.ctx.fun_simplex[-1]:
            self._update_last(xcc, fxcc)
            self.ctx.next_operation = "CONTINUE"
        else:
            # shrink
            self.ctx.next_operation = "SHRINK"
            
    def do_shrink(self):
        return self.ctx.next_operation == "SHRINK"
    
    def submit_shrink(self):
        self.report("Submitting shrink step.")
        self.ctx._to_evaluate_simplex = self.ctx.simplex[1:] = self.ctx.simplex[0] + SIGMA * (self.ctx.simplex[1:] - self.ctx.simplex[0])
        self.ctx.fun_simplex[1:] = np.nan
        self._submit(name='shrink')
        
    def update_shrink(self):
        """
        Retrieve the results of an shrink step.
        """
        self.ctx.simplex[1:], self.ctx.fun_simplex[1:] = self._parse_outputs(single=False)

        self.ctx.next_operation = "CONTINUE"
        
    @property
    def indices_to_retrieve(self):
        return self.ctx.setdefault('indices_to_retrieve', [])

    @indices_to_retrieve.setter
    def indices_to_retrieve(self, value):
        self.ctx.indices_to_retrieve = value
    
    def eval_key(self, index):
        """
        Returns the evaluation key corresponding to a given index.
        """
        return self._EVAL_PREFIX + str(index)
        
    def _merge_nested_inputs(self, input_mapping, input_values, fixture_inputs):
        """list of mapping key and value"""
        target_inputs = dict(fixture_inputs)
        
        nested_key_inputs = {}
        for key, value in zip(input_mapping, input_values):
            # nest key separated by `.`
            nested_key_inputs[key] = value
            
        inputs = _merge_nested_keys(nested_key_inputs, target_inputs)
            
        return inputs

    def finalize(self):
        self.report(f'on stop: simplex is {self.ctx.simplex}, fun_simplex is {self.ctx.fun_simplex}')
        self.out('result', orm.Dict(dict={
                'num_iter': self.ctx.num_iter,
            }).store())

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