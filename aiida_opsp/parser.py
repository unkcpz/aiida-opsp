from aiida.parsers import Parser
from aiida import orm
import io
import tempfile
import numpy as np
from scipy.integrate import cumtrapz
import os
from abipy.ppcodes.oncv_parser import OncvParser
from collections import namedtuple
import numpy as np

from aiida import plugins

UpfData = plugins.DataFactory('pseudo.upf')
Fd_Params = namedtuple("Fermi_Dirac", ['energy', 'sigma', 'mirror']) # mirror for upside down upon 0.5

def fermi_dirac(e, fd_params: Fd_Params):
    """array -> array"""
    res = 1.0 / (np.exp((e - fd_params.energy) / fd_params.sigma) + 1.0)
    
    assert isinstance(fd_params.mirror, bool)
    
    if fd_params.mirror:
        return 1.0 - res
    else:
        return res
    
def create_weights(xs, fd1: Fd_Params, fd2: Fd_Params):
    assert np.all(xs[:-1] <= xs[1:])    # make sure that the energies is in acsending order
    
    boundary = (fd1.energy + fd2.energy) / 2.0
    
    condition = (xs < boundary)
    _energies = np.extract(condition, xs)
    weights1 = fermi_dirac(_energies, fd_params=fd1)
    
    condition = (xs >= boundary)
    _energies = np.extract(condition, xs)
    weights2 = fermi_dirac(_energies, fd_params=fd2)

    weights = np.concatenate((weights1, weights2))
    
    return weights

def compute_lderr(atan_logders, lmax, weight_unbound=0.1):
    """
    We having this function to process the atan logder in advance since we 
    don't want to store lots of data in the file repository.
    
    Using four values to construct the fermi-dirac functions for weight the function.
    Four values are hard coded.
    """
    fd1 = Fd_Params._make([0.0, 0.25, True])
    fd2 = Fd_Params._make([6.0, 0.25, False])

    ldderr = 0.0
    for l in atan_logders.ae:
        # diff with counting the weight on fermi dirac distribution
        f1, f2 = atan_logders.ae[l], atan_logders.ps[l]
                
        sortind = np.argsort(f1.energies) # must do the sort since we use concatenate to combine split range
        energies = f1.energies[sortind]
        
        abs_diff = np.abs(f1.values - f2.values)    # compare the absolute diff
        abs_diff = abs_diff[sortind]
        
        weights = create_weights(energies, fd1, fd2)  # !do not sort since it require enegies sorted in acsend order
        
        integ = cumtrapz(abs_diff * weights, x=energies) / (energies[-1] - energies[0]) # normalized cumulated integ
        integ_final = integ[-1]
        
        if not l < lmax+1:
            # unbound states
            integ_final *= weight_unbound
            
        ldderr += integ_final
    
    return ldderr

class OncvPseudoParser(Parser):
    """Parser for `OncvPseudoCalculation` parse output to pseudo and verifi results"""
    
    def parse(self, **kwargs):
        """Parse the contets of output of oncv to veri results and pseudo files"""
        
        output_folder = self.retrieved
        
        with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
            stdout = handle.read()
        
        with tempfile.NamedTemporaryFile() as fp:
            fp.write(stdout.encode('utf-8'))
            fpath = fp.name
            abi_parser = OncvParser(fpath)
            
            try:
                abi_parser.scan()
                
                lderr = compute_lderr(abi_parser.atan_logders, abi_parser.lmax, self.node.inputs.weight_unbound.value)
            except:
                # not finish okay therefore not parsed
                # TODO re-check the following exit states, will be override by this one
                output_parameters = {}
                self.out('output_parameters', orm.Dict(dict=output_parameters))
                return self.exit_codes.get('ERROR_ABIPY_NOT_PARSED')
            else:
                if abi_parser._errors:
                    output_parameters = {
                        "error": abi_parser._errors,
                    }
                    self.out('output_parameters', orm.Dict(dict=output_parameters))
                    return self.exit_codes.get('ERROR_GHOST_OR_FCFACT')
            
            
            results = abi_parser.get_results()
        
        output_parameters = {}
        output_parameters['ldderr'] = lderr
        output_parameters['max_atan_logder_l1err'] = float(results['max_atan_logder_l1err'])
        output_parameters['max_ecut'] = float(results['max_ecut'])
        output_parameters['weight_unbound'] = self.node.inputs.weight_unbound.value
        
        # Separate the input string into separate lines
        data_lines = stdout.split('\n') 
        logs = {'error': []}
        for count, line in enumerate(data_lines):
            
            # ERROR_PSPOT_HAS_NODE
            if 'pseudo wave function has node' in line:
                logs['error'].append('ERROR_PSPOT_HAS_NODE')
                
            if 'lschvkbb ERROR' in line:
                logs['error'].append('ERROR_LSCHVKBB')
            
            # line idx for PSP UPF part
            if 'Begin PSP_UPF' in line:
                start_idx = count
            
            if 'END_PSP' in line:
                end_idx = count
                
            # For configuration test results 
            # idx 0 always exist for the setting configuration
            if 'Test configuration'  in line:
                test_idx = line.strip()[-1]
                i = count
                
                while True:
                    if ('PSP excitation error' in data_lines[i]
                    or 'WARNING no output for configuration' in data_lines[i]):
                        end_count = i
                        break
                    
                    i += 1
                    
                test_ctx = data_lines[count+2:end_count+1]
                
                output_parameters[f'tc_{test_idx}'] = parse_configuration_test(test_idx, test_ctx)
                
        self.out('output_parameters', orm.Dict(dict=output_parameters))
                    
        if self.node.inputs.dump_psp.value:
            upf_lines = data_lines[start_idx+1:end_idx-1]
            upf_txt = '\n'.join(upf_lines)
            
            
            pseudo = UpfData.get_or_create(io.BytesIO(upf_txt.encode('utf-8')))
            self.out('output_pseudo', pseudo)
            
        for error_label in [
            'ERROR_PSPOT_HAS_NODE',
            'ERROR_LSCHVKBB',
        ]:
            if error_label in logs['error']:
                return self.exit_codes.get(error_label)
                
def parse_configuration_test(test_idx, test_ctx):
    out = {}
    
    out['idx'] = test_idx
    
    # error of every state
    state_error = []
    for count, line in enumerate(test_ctx[:]):
        if not line:
            # When encounter blank line where separate error of every angular momentum
            # and excitatior error summary, go to last line and parse excitation error
            excitation_line = test_ctx[count+4]
            out['excitation_error'] = float(excitation_line.split()[-1].replace('D', 'E'))
            
            break
        
        if 'n   l     f' in line:
            continue
        
        if 'WARNING lschvkbb convergence error' in line:
            # the test failed because of lschvkbk convergence error
            # regard it as very bad pseudopotential (the less value the better pseudo)
            out['excitation_err'] = 99.
            out['state_err_avg'] = 99.
            
            return out
        
        # parse line of angular momentum
        try:
            n, l, f, eae, eps, diff = line.split()
        except ValueError:
            # conduction states are not parsed and raise not enough values error
            continue
        
        state_error.append(float(diff.replace('D', 'E')))
        
    out['state_err'] = state_error
    out['state_err_avg'] = sum(state_error) / len(state_error)
    
    return out