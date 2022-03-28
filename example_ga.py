from aiida.engine import run_get_pk

from aiida_opsp.workflow import GeneticAlgorithmWorkChain

def run():
    inputs = {}
    res, pk = run_get_pk(GeneticAlgorithmWorkChain, **inputs)

    return res, pk

if __name__ == '__main__':
    res, pk = run()
    # print(res['result'].get_dict(), pk)