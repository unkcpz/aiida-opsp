import pytest

from aiida_opsp.workflows.individual import validate_individual, generate_random_individual, generate_mutate_individual

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
        },
        'b0': {
            'var_type': 'float',
            'space': {
                'range': [1, 2],
                'ref_to': 'b',
            },
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

def test_mutata_inddividual(variable_info, data_regression):
    probability = 0.5
    individual = generate_random_individual(variable_info, seed=1)
    mutated_individual = generate_mutate_individual(individual, probability, variable_info, seed=1)
    
    data_regression.check(mutated_individual)
    assert validate_individual(mutated_individual, variable_info)

    # TODO: might need to test more cases, such as checking if the change also depends on the original dict
    # I test manually and it seems to be working fine
