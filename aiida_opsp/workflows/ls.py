import numpy as np
import random
import copy
from enum import Enum

from aiida import orm
from aiida.engine import WorkChain, while_, if_, calcfunction
import math
import scipy.linalg as la

from aiida_opsp.workflows import load_object, PROCESS_INPUT_KWARGS
from aiida_opsp.utils import hash_dict
from aiida_opsp.utils.merge_input import individual_to_inputs
from aiida_opsp.workflows.individual import GenerateValidSimplexIndividual

# simplex parameters from scipy.minimize
RHO = 1
CHI = 2
PSI = 0.5
SIGMA = 0.5

class Operation(Enum):
    CONTINUE = 'continue'
    INIT = 'init'
    REFLECTION = 'reflection'
    EXPANSION = 'expansion'
    CONTRACTION = 'contraction'
    INSIDE_CONTRACTION = 'inside_contraction'
    SHRINK = 'shrink'

def extract_search_variables(individual, variable_info):
    """Extract and make the simplex from variables that tagged as
    local_optimize in variable_info, and the rest of variables are fixed.
    """
    point = dict()
    fixture_variables = dict()
    for key, value in individual.items():
        if variable_info[key].get("local_optimize", False):
            point[key] = value
        else:
            fixture_variables[key] = value
            
    return point, fixture_variables


@calcfunction
def sort_simplex(simplex, scores, uuids):
    """Sort the simplex by the scores of the points in the simplex"""
    argsort = np.argsort(scores)
    simplex = [simplex[i] for i in argsort]
    scores = [scores[i] for i in argsort]
    uuids = [uuids[i] for i in argsort]

    return orm.Dict(dict={
        "simplex": simplex, 
        "scores": scores,
        "best_point": simplex[0],
        "best_score": scores[0],
        "best_uuid": uuids[0],
    })

class NelderMeadWorkChain(WorkChain):
    """WorkChain to run GA """
    
    @classmethod
    def define(cls, spec):
        """Specify imputs and outputs"""
        super().define(spec)
        spec.input('evaluate_process', help='Process which produces the result to be optimized.',
            **PROCESS_INPUT_KWARGS)
        spec.input('variable_info', valid_type=orm.Dict)    # map gene to input of evaluate process in oder
        spec.input('result_key', valid_type=orm.Str)    # name of key to be the result for fitness
        spec.input_namespace('fixture_inputs', required=False, dynamic=True)  # The fixed input parameters that will combined with change parameters

        spec.input('parameters', valid_type=orm.Dict)
        spec.input('init_individual', valid_type=orm.Dict)
        spec.input('seed', valid_type=orm.Int, default=lambda: orm.Int(2022))
        
        spec.outline(
            cls.setup,
            cls.prepare_init_simplex,
            cls.prepare_init_simplex_inspect,
            cls.submit_preparation,
            cls.update_preparation,
            while_(cls.should_continue)(
                cls.initialize_iteration,
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
        spec.output('final_individual', valid_type=orm.Dict)
        spec.output('final_score', valid_type=orm.Float)
        
        spec.exit_code(
            201,
            'ERROR_EVALUATE_PROCESS_FAILED',
            message='GA optimization failed because one of the evaluate processes did not finish ok.'
        )
        spec.exit_code(
            202,
            'ERROR_PREPARE_INIT_SIMPLEX_FAILED',
            message='local Nelder-Mead optimization failed because prepare init simplex failed.',
        )
        
    def _restore_individual(self, point, fixture_variables):
        """restore inputs for submission from variable_info"""
        individual = dict()
        for key, value in point.items():
            individual[key] = value
        for key, value in fixture_variables.items():
            individual[key] = value
            
        return individual

    def _submit(self, op):
        """
        Calculating the res of current simplex (the score of its every points). 
            
        evaluate_process is the name of process
        """
        self.report(f'On evaluate at simplex point: {self.ctx.num_iteration} by operation {op.value}')

        evaluate_process = load_object(self.inputs.evaluate_process.value)
        
        # submit evaluation for the pop ind
        evaluates = dict()
        for idx, point in enumerate(self.ctx._points_to_evaluate):
            individual = self._restore_individual(point, self.ctx.fixture_variables)
            inputs = individual_to_inputs(individual, self.inputs.variable_info.get_dict(), self.inputs.fixture_inputs)
            node = self.submit(evaluate_process, **inputs)

            retrive_key = f'_EVAL_NM_{op.value.upper()}_{idx}'

            optimize_info = {
                'retrive_key': retrive_key,
                'iteration': self.ctx.num_iteration,
            }
            node.base.extras.set('point', point)
            node.base.extras.set('individual', individual)
            node.base.extras.set('optimize_mode', 'nelder-mead')
            node.base.extras.set('optimize_info', optimize_info)
            
            evaluates[retrive_key] = node
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)

    def _inspect(self):
        """Inspect the results of current simplex evaluation.
        
        Update the simplex and scores of the simplex points.
        """
        self.report('Checking finished evaluations.')
        scores = dict()
        points = dict() 
        uuids = dict()

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
                    scores[key] = 999.0
                    points[key] = proc.base.extras.get('point')
                    uuids[key] = proc.uuid
                else:
                    return self.exit_codes.ERROR_EVALUATE_PROCESS_FAILED
            else:
                # !!! IMPORTANT, the smaller the value of penalty function the better the points is.
                result_key = self.inputs.result_key.value
                scores[key] = proc.outputs[result_key].value                
                points[key] = proc.base.extras.get('point')
                uuids[key] = proc.uuid
            
        self.report(f"Evaluate results: points are {points}, the corresponding output {scores}.")
        
        points_lst = list(points.values())
        scores_lst = list(scores.values())
        uuids_lst = list(uuids.values())

        return points_lst, scores_lst, uuids_lst
             
    def setup(self):
        # to store const parameters in ctx over GA procedure
        parameters = self.inputs.parameters.get_dict()
        self.ctx.xtol = parameters['xtol']
        self.ctx.ftol = parameters['ftol']

        self.ctx.max_iter = parameters['max_iter']
        self.ctx.num_iteration = 0
        self.ctx.should_continue = True
        
        self.ctx.variable_info = self.inputs.variable_info.get_dict()
        
        # tmp_retrive_key_storage
        # This is for store the key so the parser step know which process to fetch from ctx
        # It needs to be kept empty in between a run-inspect session.
        self.ctx.tmp_retrive_key_storage = []
        
    def prepare_init_simplex(self):
        self.ctx.init_point, self.ctx.fixture_variables = extract_search_variables(self.inputs.init_individual, self.ctx.variable_info)
        self.report(f'On preparing initial simplex')
        self.report(f"Split the individual into: point: {self.ctx.init_point}, fixture_variables: {self.ctx.fixture_variables}.")

        # submit evaluation for the pop ind
        evaluates = dict()
        for idx, key in enumerate(self.ctx.init_point.keys()):
            inputs = {
                'evaluate_process': self.inputs.evaluate_process,
                'variable_info': self.inputs.variable_info,
                'fixture_inputs': self.inputs.fixture_inputs,
                'init_point': orm.Dict(dict=self.ctx.init_point),
                'fixture_variables': orm.Dict(dict=self.ctx.fixture_variables),
                'mutate_key': orm.Str(key),
            }
            node = self.submit(GenerateValidSimplexIndividual, **inputs)

            retrive_key = f'_EVAL_NM_INIT_SIMPLEX_{idx}'

            optimize_info = {
                'retrive_key': retrive_key,
            }
            node.base.extras.set('optimize_mode', 'nelder-mead')
            node.base.extras.set('optimize_info', optimize_info)
            
            evaluates[retrive_key] = node
            self.ctx.tmp_retrive_key_storage.append(retrive_key)
            
        return self.to_context(**evaluates)

    def prepare_init_simplex_inspect(self):
        simplex = [self.ctx.init_point]
        while len(self.ctx.tmp_retrive_key_storage) > 0:
            key = self.ctx.tmp_retrive_key_storage.pop(0)
            self.report(f'Retrieving output for evaluation {key}')
            
            proc: orm.WorkChainNode = self.ctx[key]
            if not proc.is_finished_ok:
                return self.exit_codes.ERROR_PREPARE_INIT_POPULATION_FAILED
            else:
                valid_individual = proc.outputs.final_individual.get_dict()
                point, _ = extract_search_variables(valid_individual, self.ctx.variable_info)
                simplex.append(point)

        self.ctx.simplex = simplex

        # assert the number of points of a simplex is one more than the number of variables
        if not len(self.ctx.simplex) == len(self.ctx.init_point) + 1:
            return self.exit_codes.ERROR_PREPARE_INIT_SIMPLEX_FAILED
        
    def submit_preparation(self):
        # XXX it is important to select a good initial point, 
        # however it is not the focus of this workchain, we just use a 
        # random points in the range of the initial individual
        # there are posibility that the simplex is invalid to produce the pseudopotentail,
        # in this case, in principle, we should re-sample the simplex, but we just use as small sigma
        # to make sure the simplex is valid.
        #self.ctx.simplex = create_random_simplex(point, self.ctx.fixture_variables, self.ctx.variable_info, seed=self.inputs.seed.value)
        self.ctx._points_to_evaluate = self.ctx.simplex

        # set the next operation to "CONTINUE" so only the reflection is run, 
        self.ctx.next_operation = Operation.CONTINUE
        self._submit(op=Operation.INIT)
        
    def update_preparation(self):
        self.ctx.simplex, self.ctx.scores, self.ctx.uuids = self._inspect()
    
    def should_continue(self):
        """return a bool, whether create new generation"""
        return self.ctx.should_continue

    def compute_x_dist_max(self, simplex):
        """Compute the maximum distance between points in simplex"""
        simplex = np.array([list(p.values()) for p in simplex])
        return np.max(la.norm(simplex[1:] - simplex[0], axis=-1)) 

    def compute_f_diff_max(self, scores):
        """compute the maximum difference between function values in simplex"""
        f_diff = np.max(np.abs(np.array(scores[1:]) - scores[0]))
        return f_diff
        
    def initialize_iteration(self):
        # sort the simplex upon the values of points
        res = sort_simplex(self.ctx.simplex, self.ctx.scores, self.ctx.uuids)
        self.ctx.simplex, self.ctx.scores = res['simplex'], res['scores']
        self.report(f"Sorted scores: {self.ctx.scores}")
        self.logger.info(f"Sorted simplex: {self.ctx.simplex}")
        
        # check if should continue the iteration
        x_dist_max = self.compute_x_dist_max(self.ctx.simplex)
        self.report(f"Maximum distance value for the simplex: {x_dist_max}")

        
        # the scores may be inf, so we need to filter them out
        valid_scores = [s for s in self.ctx.scores if s != math.inf]
        f_diff_max = self.compute_f_diff_max(valid_scores)
        f_diff_rel = f_diff_max / np.abs(valid_scores[0])
        self.report(f"Maximum function difference: {f_diff_max}")
        self.report(f"Relative function difference: {f_diff_rel}")
        
        # if the points are too close or the function values are too close, stop the iteration
        # use the relative tolerance to compare the function values
        self.ctx.should_continue = (x_dist_max > self.ctx.xtol) and (f_diff_rel > self.ctx.ftol)
        self.report(
            f"End of Nelder-Mead iteration {self.ctx.num_iteration}, max number of iterations: {self.ctx.max_iter}."
        )
        
        if not self.ctx.num_iteration < self.ctx.max_iter:
            self.report(f"Number of iterations exceeded the maximum {self.ctx.max_iter}. Stop.")
            self.ctx.next_operation = Operation.CONTINUE    # no furture operations
            self.ctx.should_continue = False
        else:     
            self.ctx.num_iteration += 1
        
    def centroid(self, arr_simplex):
        return np.average(arr_simplex[:-1], axis=0)
        
    def submit_reflection(self):
        self.report(f"Submit reflection operation.")
        key_list = list(self.ctx.simplex[0].keys())
        arr_simplex = np.array([list(p.values()) for p in self.ctx.simplex])
        xr = (1 + RHO) * self.centroid(arr_simplex) - RHO * arr_simplex[-1]
        self.ctx._points_to_evaluate = [dict(zip(key_list, list(xr)))]
        self._submit(op=Operation.REFLECTION)
        
    def update_reflection(self):
        points, scores, uuids = self._inspect()
        xr, fxr = points[0], scores[0]
        self.ctx.point_reflected = (xr, fxr, uuids[0])
        
        if fxr < self.ctx.scores[0]:
            "if the reflected point is better than the best point, try expansion"
            self.ctx.next_operation = Operation.EXPANSION
        else:
            if fxr < self.ctx.scores[-2]:
                self._update_last(xr, fxr, uuids[0])
                self.ctx.next_operation = Operation.CONTINUE
            else:
                if fxr < self.ctx.scores[-1]:
                    self.ctx.next_operation = Operation.CONTRACTION
                else:
                    self.ctx.next_operation = Operation.INSIDE_CONTRACTION
                    
    def _update_last(self, x, f, uuid):
        self.ctx.simplex[-1] = x
        self.ctx.scores[-1] = f
        self.ctx.uuids[-1] = uuid
                
    def do_expansion(self):
        return self.ctx.next_operation == Operation.EXPANSION
    
    def submit_expansion(self):
        self.report("Submitting expansion simplex reshaping.")
        key_list = list(self.ctx.simplex[0].keys())
        arr_simplex = np.array([list(p.values()) for p in self.ctx.simplex])
        xe = (1 + RHO * CHI) * self.centroid(arr_simplex) - RHO * CHI * arr_simplex[-1]
        self.ctx._points_to_evaluate = [dict(zip(key_list, list(xe)))]
        self._submit(op=Operation.EXPANSION)

        
    def update_expansion(self):
        """
        Retrieve the results of an expansion step.
        """
        points, scores, uuids = self._inspect()
        xe, fxe = points[0], scores[0]
        _, fxr, _ = self.ctx.point_reflected
        if fxe < fxr:
            self._update_last(xe, fxe, uuids[0])
        else:
            self._update_last(*self.ctx.point_reflected)
            
        self.ctx.next_operation = Operation.CONTINUE
        
    def do_contraction(self):
        return self.ctx.next_operation == Operation.CONTRACTION
    
    def submit_contraction(self):
        self.report("Submitting contraction simplex reshaping.")
        key_list = list(self.ctx.simplex[0].keys())
        arr_simplex = np.array([list(p.values()) for p in self.ctx.simplex])
        xc = (1 + PSI * RHO) * self.centroid(arr_simplex) - PSI * RHO * arr_simplex[-1]
        self.ctx._points_to_evaluate = [dict(zip(key_list, list(xc)))]
        self._submit(op=Operation.EXPANSION)
        
    def update_contraction(self):
        """
        Retrieve the results of an contraction step.
        """
        points, scores, uuids = self._inspect()
        xc, fxc = points[0], scores[0]
        _, fxr, _ = self.ctx.point_reflected
        if fxc < fxr:
            self._update_last(xc, fxc, uuids[0])
            self.ctx.next_operation = Operation.CONTINUE
        else:
            # shrink
            self.ctx.next_operation = Operation.SHRINK
            
    def do_inside_contraction(self):
        return self.ctx.next_operation == Operation.INSIDE_CONTRACTION
    
    def submit_inside_contraction(self):
        self.report("Submitting inside contraction simplex reshaping.")
        key_list = list(self.ctx.simplex[0].keys())
        arr_simplex = np.array([list(p.values()) for p in self.ctx.simplex])
        xcc = (1 - PSI) * self.centroid(arr_simplex) + PSI * arr_simplex[-1]
        self.ctx._points_to_evaluate = [dict(zip(key_list, list(xcc)))]
        self._submit(op=Operation.INSIDE_CONTRACTION)
        
    def update_inside_contraction(self):
        """
        Retrieve the results of an inside contraction step.
        """
        points, scores, uuids = self._inspect()
        xcc, fxcc = points[0], scores[0]
        if fxcc < self.ctx.scores[-1]:
            self._update_last(xcc, fxcc, uuids[0])
            self.ctx.next_operation = Operation.CONTINUE
        else:
            # shrink
            self.ctx.next_operation = Operation.SHRINK
            
    def do_shrink(self):
        return self.ctx.next_operation == Operation.SHRINK
    
    def submit_shrink(self):
        self.report("Submitting shrink simplex reshaping.")
        key_list = list(self.ctx.simplex[0].keys())
        arr_point0 = np.array(list(self.ctx.simplex[0].values()))
        points = []
        for point in self.ctx.simplex[1:]:
            arr_point = np.array(list(point.values()))
            arr_point = arr_point0 + SIGMA * (arr_point - arr_point0) 
            points.append(dict(zip(key_list, list(arr_point))))

        self.ctx._points_to_evaluate = points
        self._submit(op=Operation.SHRINK)


    def update_shrink(self):
        """
        Retrieve the results of an shrink step.
        """
        self.ctx.simplex[1:], self.ctx.scores[1:], self.ctx.uuids[1:] = self._inspect()

        self.ctx.next_operation = Operation.CONTINUE
        
    def finalize(self):
        bp = self.ctx.simplex[0]
        score = self.ctx.scores[0]
        self.report(f"best point is {bp}")
        self.report(f"best score is {score}")
        
        self.out('final_individual', orm.Dict(dict=self._restore_individual(point=bp, fixture_variables=self.ctx.fixture_variables)).store())
        self.out('final_score', orm.Float(score).store())
