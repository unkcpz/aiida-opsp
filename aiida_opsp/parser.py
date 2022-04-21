from aiida.parsers import Parser
from aiida import orm
import io

from aiida import plugins

UpfData = plugins.DataFactory('pseudo.upf')

class OncvPseudoParser(Parser):
    """Parser for `OncvPseudoCalculation` parse output to pseudo and verifi results"""
    
    def parse(self, **kwargs):
        """Parse the contets of output of oncv to veri results and pseudo files"""
        
        output_folder = self.retrieved
        
        with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
            stdout = handle.read()
        
        
        output_parameters = {}
        
        # Separate the input string into separate lines
        data_lines = stdout.split('\n') 
        
        for count, line in enumerate(data_lines):
            
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
                    if 'PSP excitation error' in data_lines[i]:
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
                
def parse_configuration_test(test_idx, test_ctx):
    out = {}
    
    out['idx'] = test_idx
    
    # error of every state
    state_error = []
    for count, line in enumerate(test_ctx[1:]):
        if not line:
            # When encounter blank line where separate error of every angular momentum
            # and excitatior error summary, go to last line and parse excitation error
            excitation_line = test_ctx[count+4]
            out['excitation_error'] = float(excitation_line.split()[-1].replace('D', 'E'))
            
            break
        
        # parse line of angular momentum
        n, l, f, eae, eps, diff = line.split()
        
        state_error.append(float(diff.replace('D', 'E')))
        
    out['state_error'] = state_error
    out['state_error_avg'] = sum(state_error) / len(state_error)
    
    return out