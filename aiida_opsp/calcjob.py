from aiida.engine import CalcJob
from aiida.engine.processes.process_spec import CalcJobProcessSpec
from aiida import orm
from aiida.common.folders import Folder
from aiida.common.datastructures import CalcInfo, CodeInfo

from ase.data import atomic_numbers

class OncvPseudoCalculation(CalcJob):
    """Generate pseudopotential with oncvpsp"""
    
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the specification"""
        
        super().define(spec)
        # atom_info part
        spec.input('element', valid_type=orm.Str)
        spec.input('number_core_states', valid_type=orm.Float)
        spec.input('number_valence_states', valid_type=orm.Float)
        spec.input('exc_functional', valid_type=orm.Str)
        spec.input('output_type', valid_type=orm.Str)   # upf, psp8, both
        
        # atom configuration part
        # example
        # confs = {
        #     '1s': 2.0,
        #     '2s': 2.0,
        #     '2p': 6.0,
        #     '3s': 2.0,
        #     '3p': 6.0,
        #     '3d': 10.0,
        #     '4s': 2.0,
        #     '4p': 2.0,
        # }
        # -> 
        # n, l, f  (nc+nv lines)
        # 1    0    2.0
        # 2    0    2.0
        # 2    1    6.0
        # 3    0    2.0
        # 3    1    6.0
        # 3    2   10.0
        # 4    0    2.0
        # 4    1    2.0
        spec.input('configuration', valid_type=orm.Dict)
        # lmax maximum angular momentum for which psp is calculated (<=3)
        spec.input('lmax', valid_type=orm.Int)
        
        # d = {
        #     's': {
        #         'rc': 2.60,
        #         'ncon': 4,
        #         'nbas': 8,
        #         'qcut': 5.0,
        #         'nproj': 2,
        #         'debl': 1.5,
        #     },
        #     'p': {
        #         'rc': 2.60,
        #         'ncon': 4,
        #         'nbas': 8,
        #         'qcut': 5.2,
        #         'nproj': 2,
        #         'debl': 1.5,
        #     },
        #     'd': {
        #         'rc': 2.60,
        #         # ep if not set -> -0.0
        #         # positive energy must be specified 
        #         # for barrier-confined "scattering" state for unoccupied l <=lmax
        #         # A small positive energy is usually  good (0.1-0.25 Ha).
        #         'ep': 0.1, 
        #         'ncon': 4,
        #         'nbas': 8,
        #         'qcut': 8.4,
        #         'nproj': 2,
        #         'debl': 1.5,
        #     }
        # }
        spec.input('angular_momentum_settings', valid_type=orm.Dict)
        
        # local potential
        # d = {
        #     'llcol': 4, # fix
        #     'lpopt': 5, # 1-5, algorithm enum set
        #     'rc(5)': 2.0,
        #     'dvloc0': 0.0,
        # }
        spec.input('local_potential_settings', valid_type=orm.Dict)
        # nlcc
        # d = {
        #     'icmod': 3,
        #     'fcfact': 5.0,
        #     'rcfact': 1.4,
        # }
        spec.input('nlcc_settings', valid_type=orm.Dict) # fix from start

        spec.output('output', valid_type=orm.SinglefileData)
        
        spec.inputs['metadata']['options']['input_filename'].default = 'aiida.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'aiida.out'
        spec.inputs['metadata']['options']['parser_name'].default = 'opsp.pseudo.oncv'
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        
    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission"""
        element = self.inputs.element.value
        number_core_states = self.inputs.nubmer_core_states.value
        number_valence_states = self.inputs.number_valence_states.value
        exc_functional = self.inputs.exc_functional.value
        output_type = self.inputs.output_type.value
        configuration = self.inputs.configuration.get_dict()
        lmax = self.inputs.lmax.value
        angular_momentum_settings = self.inputs.angular_momentum_settings.get_dict()
        local_potential_settings = self.inputs.local_potential_settings.get_dict()
        nlcc_settings = self.inputs.nlcc_settings.get_dict()
        
        inp_str = self.input_generator(
            element,
            number_core_states,
            number_valence_states,
            exc_functional,
            output_type,
            configuration,
            lmax,
            angular_momentum_settings,
            local_potential_settings,
            nlcc_settings,
        )
        
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
    def input_generator(            
        element,
        number_core_states,
        number_valence_states,
        exc_functional,
        output_type,
        configuration,
        lmax,
        angular_momentum_settings,
        local_potential_settings,
        nlcc_settings,
    ) -> str:
        inp = []
        # ATOM CONF
        inp.append('# ATOM AND REFERENCE CONFIGURATION')
        inp.append('# atsym, z, nc, nv, iexc   psfile')
        z = atomic_numbers[element]
        lst_atom_info = [
            element, 
            z, 
            number_core_states, 
            number_valence_states,
            exc_functional,
            output_type
        ]
        inp.append(' '.join(str(e) for e in lst_atom_info))
        inp.append('#')
        
        # configuration setting
        def o2l(orbital_symbol):
            if orbital_symbol == 's':
                return '0'
            if orbital_symbol == 'p':
                return '1'
            if orbital_symbol == 'd':
                return '2'
             
            return '3'  # for f but also for wrongly set orbital symbol, check!
        
        inp.append('# n, l, f  (nc+nv lines)')
        for k, v in configuration:
            n = k[0]    # principal quantum number
            l = o2l(k[1])
            f = str(v)
        inp.append(' '.join([n, l, f]))
        inp.append('#')
        
        # PSEUDO AND OPT
        inp.append('# PSEUDOPOTENTIAL AND OPTIMIZATION')
        inp.append('# lmax')
        inp.append(str(lmax))
        inp.append('#')
        
        inp.append("# l, rc, ep, ncon, nbas, qcut  (lmax+1 lines, l's must be in order)")
        for k, v in angular_momentum_settings:
            l = o2l(k) 
            
            rc = v.get('rc')
            ncon = v.get('ncon')
            nbas = v.get('nbas')
            qcut = v.get('qcut')
            ep = v.get('ep', -0.0)
            lst = [n, rc, ep, ncon, nbas, qcut]
            inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# LOCAL POTENTIAL')
        llcol = local_potential_settings.get('llcol')
        lpopt = local_potential_settings.get('lpopt')
        rc_5 = local_potential_settings.get('rc(5)')
        dvloc0 = local_potential_settings.get('dvloc0')
        lst = [llcol, lpopt, rc_5, dvloc0]
        inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# VANDERBILT-KLEINMAN-BYLANDER PROJECTORs')
        for k, v in angular_momentum_settings:
            l = o2l(k) 
            
            nproj = v.get('nproj')
            debl = v.get('debl')
            lst = [n, nproj, debl]
            inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# MODEL CORE CHARGE')
        icmod = local_potential_settings.get('icmod')
        fcfact = local_potential_settings.get('fcfact')
        rcfact = local_potential_settings.get('rcfact', None)
        lst = [icmod, fcfact, rcfact]
        inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        return '\n'.join(inp)