from aiida.engine import CalcJob
from aiida.engine.processes.process_spec import CalcJobProcessSpec
from aiida import orm
from aiida.common.folders import Folder
from aiida.common.datastructures import CalcInfo, CodeInfo

from ase.data import atomic_numbers

import importlib

from aiida import plugins

UpfData = plugins.DataFactory('pseudo.upf')

class OncvPseudoCalculation(CalcJob):
    """Generate pseudopotential with oncvpsp"""
    
    @classmethod
    def define(cls, spec: CalcJobProcessSpec):
        """Define the specification"""
        
        super().define(spec)
        # atom_info part read from configuration cards
        spec.input('conf_name', valid_type=orm.Str)
        spec.input('lmax', valid_type=orm.Int)
        spec.input('angular_momentum_settings', valid_type=orm.Dict)
        spec.input('local_potential_settings', valid_type=orm.Dict)
        spec.input('nlcc_settings', valid_type=orm.Dict) 
        spec.input('dump_psp', valid_type=orm.Bool, required=False)
        spec.input('weight_unbound', valid_type=orm.Float, default=lambda: orm.Float(0.1))

        spec.output('output_parameters', valid_type=orm.Dict)
        spec.output('output_pseudo', valid_type=UpfData, required=False)
        
        spec.exit_code(501, 'ERROR_PSPOT_HAS_NODE',
            message='The pseudo wave function has node.')   # TODO can record which l is wrong and used to tune inputs of GA
        spec.exit_code(502, 'ERROR_LSCHVKBB',
            message='The lschvkbb error.')  
        spec.exit_code(503, 'ERROR_ABIPY_NOT_PARSED',
            message='Exception while parsing using abipy.')
        spec.exit_code(504, 'ERROR_GHOST_OR_FCFACT',
            message='Has Ghost or for example fcfact > 0.0. for icmod=1.')
        
        
        spec.inputs['metadata']['options']['input_filename'].default = 'aiida.in'
        spec.inputs['metadata']['options']['output_filename'].default = 'aiida.out'
        spec.inputs['metadata']['options']['parser_name'].default = 'opsp.pseudo.oncvpsp'
        spec.inputs['metadata']['options']['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        
    def prepare_for_submission(self, folder: Folder) -> CalcInfo:
        """Prepare the calculation for submission"""
        conf_name = self.inputs.conf_name.value
        lmax = self.inputs.lmax.value
        angular_momentum_settings = self.inputs.angular_momentum_settings.get_dict()
        local_potential_settings = self.inputs.local_potential_settings.get_dict()
        nlcc_settings = self.inputs.nlcc_settings.get_dict()
        
        inp_str = self.input_generator(
            conf_name,
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
        conf_name,
        lmax,
        angular_momentum_settings,
        local_potential_settings,
        nlcc_settings,
    ) -> str:
        inp = []
        # configuration setting
        def o2l(orbital_symbol):
            if orbital_symbol == 's':
                return '0'
            if orbital_symbol == 'p':
                return '1'
            if orbital_symbol == 'd':
                return '2'
             
            return '3'  # for f but also for wrongly set orbital symbol, check!
        
        import_path = importlib.resources.path(
            "aiida_opsp.statics.configurations", f"{conf_name}.dat"
        )

        with import_path as path, open(path, "r") as handle:
            content = handle.read()
            inp.append(content)
        
        # PSEUDO AND OPT
        inp.append('# PSEUDOPOTENTIAL AND OPTIMIZATION')
        inp.append('# lmax')
        inp.append(str(lmax))
        inp.append('#')
        
        inp.append("# l, rc, ep, ncon, nbas, qcut  (lmax+1 lines, l's must be in order)")
        rcmax = 0.0
        rcmin = 99.0
        for k in ['s', 'p', 'd']:
            l = o2l(k) 
            v = angular_momentum_settings.get(k, None)
            
            if v:
                rc = v.get('rc')
                rcmax = max(rcmax, rc)
                rcmin = min(rcmin, rc)
                ncon = v.get('ncon')
                nbas = v.get('nbas')
                qcut = v.get('qcut')
                ep = v.get('ep', -0.0)
                lst = [l, rc, ep, ncon, nbas, qcut]
                inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# LOCAL POTENTIAL')
        llcol = local_potential_settings.get('llcol')
        lpopt = local_potential_settings.get('lpopt')
        
        # if rc(5) not set use default min rc
        rc_5 = local_potential_settings.get('rc(5)', rcmin)
        
        dvloc0 = local_potential_settings.get('dvloc0')
        lst = [llcol, lpopt, rc_5, dvloc0]
        inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# VANDERBILT-KLEINMAN-BYLANDER PROJECTORs')
        for k in ['s', 'p', 'd']:
            l = o2l(k) 
            v = angular_momentum_settings.get(k, None)
            
            if v:
                nproj = v.get('nproj')
                debl = v.get('debl')
                lst = [l, nproj, debl]
                inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# MODEL CORE CHARGE')
        icmod = nlcc_settings.get('icmod')
        fcfact = nlcc_settings.get('fcfact', None)
        if not fcfact:
            raise "fcfact can't be empty."
        rcfact = nlcc_settings.get('rcfact', '')
        lst = [icmod, fcfact, rcfact]
        inp.append(' '.join(str(e) for e in lst))
        inp.append('#')
        
        inp.append('# epsh1 epsh2 depsh')
        inp.append('-12.0 12.0 0.02')
        
        inp.append('# rlmax drl')
        #rlmax = rcmax * 2
        rlmax = 6.0
        inp.append(' '.join([str(rlmax), str(0.01)]))
        
        try:
            # read and append verify card content to input by element
            # inp.append(verify_card(element))
            import_path = importlib.resources.path(
                "aiida_opsp.statics.test_configurations", f"{conf_name}.dat"
            )

            with import_path as path, open(path, "r") as handle:
                content = handle.read()
                inp.append(content)
        except FileNotFoundError:
            inp.append('0')
            
        
        return '\n'.join(inp)