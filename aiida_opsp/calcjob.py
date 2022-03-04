from aiida.engine import CalcJob
from aiida.engine.processes.process_spec import CalcJobProcessSpec
from aiida import orm
from aiida.common.folders import Folder
from aiida.common.datastructures import CalcInfo, CodeInfo


class OncvPseudoCalculation(CalcJob):
    """Generate pseudopotential with oncvpsp"""
    
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the specification"""
        
        super().define(spec)
        spec.input('input', valid_type=orm.Dict)
        spec.output('output', valid_type=orm.SinglefileData)
        
        spec.inputs['metadata']['options']['input_filename'].default = 'aiida.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'aiida.out'
        spec.inputs['metadata']['options']['parser_name'].default = 'opsp.pseudo.oncv'
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        
    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission"""
        inp_str = self.input_generator(self.inputs.input.get_dict())
        
        with folder.open(self.options.input_filename, 'w', encoding='utf8') as handle:
            handle.write(inp_str)
            
        codeinfo = CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdin_name = self.options.input_filename
        codeinfo.stdout_name = self.options.output_filename
        
        calcinfo = CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.retrieve_list = [self.options.output_filename]
        
        return calcinfo
    
    @staticmethod
    def input_generator(inp_params: dict) -> str:
        inp = []
        # ATOM CONF
        inp.append('# ATOM AND REFERENCE CONFIGURATION')
        inp.append('# atsym, z, nc, nv, iexc   psfile')
        inp.append(' '.join(str(e) for e in inp_params['atom_info']))
        inp.append('#')
        
        inp.append('# n, l, f  (nc+nv lines)')
        for lst in inp_params['atom_conf']:
            inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        # PSEUDO AND OPT
        inp.append('# PSEUDOPOTENTIAL AND OPTIMIZATION')
        inp.append('# lmax')
        inp.append(str(inp_params['opt_lmax']))
        inp.append('#')
        
        inp.append("# l, rc, ep, ncon, nbas, qcut  (lmax+1 lines, l's must be in order)")
        for lst in inp_params['opt_rrkj']:
            inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# LOCAL POTENTIAL')
        inp.append(' '.join(str(e) for e in inp_params['local_potential']))
        inp.append('#')
        
        inp.append('# VANDERBILT-KLEINMAN-BYLANDER PROJECTORs')
        for lst in inp_params['projectors']:
            inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# MODEL CORE CHARGE')
        inp.append(' '.join(str(e) for e in inp_params['core_corr']))
        inp.append('#')
        
        return '\n'.join(inp)