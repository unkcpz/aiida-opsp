import math
from aiida.parsers import Parser
from aiida import orm
import io
import tempfile
import numpy as np
from scipy.integrate import cumtrapz
import os
from abipy.ppcodes.oncv_parser import OncvParser

from aiida import plugins

UpfData = plugins.DataFactory('pseudo.upf')

def compute_crop_l1err(atan_logders, lmax):
    """
    return a dict key is the range of crop, value is the integral and is the 
    state type (bound/unbound)
    energy integ ranges are:
        -inf -> -5
        -5. -> -2.
        -2 -> 0
        0 -> 2
        2 -> 6
        6 -> 8
        8 -> inf
    """
    r_dict = {
        "ninf_n5": (-math.inf, -5.), 
        "n5_n2": (-5, -2), 
        "n2_0": (-2, 0), 
        "0_2": (0, 2), 
        "2_6": (2, 6), 
        "6_8": (6, 8),
        "8_inf": (8, math.inf),
    }
    crop_ldd = [] 
    for l in atan_logders.ae:
        for k, r in r_dict.items():
            f1, f2 = atan_logders.ae[l], atan_logders.ps[l]
            abs_diff = np.abs(f1.values - f2.values)
            
            # crop
            condition = (r[0] < f1.energies) * (f1.energies < r[1])
            energies = np.extract(condition, f1.energies)
            abs_diff = np.extract(condition, abs_diff)
            
            integ = cumtrapz(abs_diff, x=energies) / (energies[-1] - energies[0])
            
            if l < lmax+1:
                # bound states
                state_type = "bound"
            else:
                # unbound states
                state_type = "unbound"
                
            crop_ldd.append({
                "crop_range": k,
                "l": l,
                "state_type": state_type,
                "integ": integ[-1],
                # "int_range": r,
            })
    
    return crop_ldd

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
                
                # crop_ldd is a dictionary for the ldd of every crop section
                crop_ldd = compute_crop_l1err(abi_parser.atan_logders, abi_parser.lmax)
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
        output_parameters['crop_ldd'] = crop_ldd
        output_parameters['max_atan_logder_l1err'] = float(results['max_atan_logder_l1err'])
        output_parameters['max_ecut'] = float(results['max_ecut'])
        
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