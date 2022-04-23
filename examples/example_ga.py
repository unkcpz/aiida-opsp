from aiida.engine import run_get_pk
from aiida import orm

from aiida_opsp.workflows.ga import GeneticAlgorithmWorkChain
from aiida_opsp.workflows.psp_oncv import OncvPseudoWrapWorkChain 

def run():
    inputs = {
        'parameters': orm.Dict(dict={
            'num_generation': 2,
            'num_pop_per_generation': 20,
            'num_genes': 2,
            'num_mating_parents': 15,
            'num_keep_parents': 3,
            'num_mutation_genes': 2,
            'mutate_probability': 0.8,
            'gene_space': [{'low': 1.1, 'high': 2.0}, {'low': 8.0, 'high': 16.0}],
            'gene_type': ['float', 'float'],
            'seed': 989,
        }),
        'evaluate_process': OncvPseudoWrapWorkChain,
        'input_mapping': orm.List(list=['rc_s', 'qcut_s']), # for genes defined by gene_type and gene_space.
        'output_mapping': orm.Str('result'),
    }
    res, pk = run_get_pk(GeneticAlgorithmWorkChain, **inputs)

    return res, pk

if __name__ == '__main__':
    res, pk = run()
    # print(res['result'].get_dict(), pk)