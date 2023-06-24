import pytest

from aiida_opsp.workflows.individual import validate_individual, generate_random_individual, generate_mutate_individual, generate_crossover_individual

@pytest.fixture
def variable_info():
    d = {
        'a': {
            'var_type': 'int',
            'space': {
                'range': [0, 10],
                'ref_to': None,
            }
        },
        'b': {
            'var_type': 'float',
            'space': {
                'range': [0, 5],
                'ref_to': None,
            },
            'group': 'beta',
        },
        'b0': {
            'var_type': 'float',
            'space': {
                'range': [1, 2],
                'ref_to': 'b',
            },
            'group': 'beta',
        },
    }

    return d

def test_validate_individual(variable_info):
    # test valid individual
    individual = {
        'a': 5,
        'b': 2.5,
        'b0': 4.5,
    }
    
    assert validate_individual(individual, variable_info)

def test_generate_random_individual(variable_info, data_regression):
    individual = generate_random_individual(variable_info, seed=1)
    
    data_regression.check(individual)
    assert validate_individual(individual, variable_info)

def test_mutate_individual(variable_info, data_regression):
    probability = 0.5
    individual = generate_random_individual(variable_info, seed=1)
    mutated_individual = generate_mutate_individual(individual, probability, variable_info, seed=1)
    
    data_regression.check(mutated_individual)
    assert validate_individual(mutated_individual, variable_info)

    # TODO: might need to test more cases, such as checking if the change also depends on the original dict
    # I test manually and it seems to be working fine

def test_crossover_individual(variable_info):
    parent1 = {
        'a': 5,
        'b': 2.5,
        'b0': 4.5,
    }

    parent2 = {
        'a': 15,
        'b': 12.5,
        'b0': 14.5,
    }

    for idx in range(10):        
        crossover_individual = generate_crossover_individual(parent1, parent2, variable_info, seed=idx)
        
        a = crossover_individual['a']
        b = crossover_individual['b']
        b0 = crossover_individual['b0']
        
        if b == 2.5:
            assert b0 == 4.5
        if b == 12.5:
            assert b0 == 14.5

    
    