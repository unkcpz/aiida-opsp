from unittest import result
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

def compute_crop_l1err(atan_logders, emin, emax):
    max_l1err = 0.0
    for l in atan_logders.ae:
        f1, f2 = atan_logders.ae[l], atan_logders.ps[l]
        abs_diff = np.abs(f1.values - f2.values)
        
        # crop
        condition = (emin < f1.energies) * (f1.energies < emax)
        energies = np.extract(condition, f1.energies)
        abs_diff = np.extract(condition, abs_diff)
        
        integ = cumtrapz(abs_diff, x=energies) / (energies[-1] - energies[0])
        max_l1err = max(max_l1err, integ[-1])
        
    return float(max_l1err)

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
                crop_0_5_atan_logder_l1err = compute_crop_l1err(abi_parser.atan_logders, 0., 5)
            except:
                # not finish okay therefore not parsed
                # TODO re-check the following exit states, will be override by this one
                output_parameters = {}
                self.out('output_parameters', orm.Dict(dict=output_parameters))
                return self.exit_codes.get('ERROR_ABIPY_NOT_PARSED')
            
            results = abi_parser.get_results()
        
        output_parameters = {}
        
        output_parameters['crop_0_5_atan_logder_l1err'] = crop_0_5_atan_logder_l1err
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
            out['excitation_error'] = 99.
            out['state_error_avg'] = 99.
            
            return out
        
        # parse line of angular momentum
        try:
            n, l, f, eae, eps, diff = line.split()
        except ValueError:
            # conduction states are not parsed and raise not enough values error
            continue
        
        state_error.append(float(diff.replace('D', 'E')))
        
    out['state_error'] = state_error
    out['state_error_avg'] = sum(state_error) / len(state_error)
    
    return out