from aiida.parsers import Parser

class OncvPseudoParser(Parser):
    """Parser for `OncvPseudoCalculation` parse output to pseudo and verifi results"""
    
    def parse(self, **kwargs):
        """Parse the contets of output of oncv to veri results and pseudo files"""
        from aiida.orm import SinglefileData
        import io
        
        output_folder = self.retrieved
        
        with output_folder.open(self.node.get_option('output_filename'), 'r') as handle:
            result = handle.read()
            
        self.out('output', SinglefileData(file=io.BytesIO(bytes(result, encoding='utf8'))))