from aiida_opsp.workflows.individual import validate_individual, generate_random_individual

def test_validate_individual():
    variable_info = {
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

    # test valid individual
    individual = {
        'a': 5,
        'b': 2.5,
        'b0': 4.5,
    }
    
    assert validate_individual(individual, variable_info)

def test_generate_random_individual(data_regression):
    variable_info = {
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

    individual = generate_random_individual(variable_info, seed=1)
    
    data_regression.check(individual)
    assert validate_individual(individual, variable_info)
    