import pytest

from aiida_opsp.workflows.ls import extract_search_variables, create_random_simplex, sort_simplex

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
            'local_optimize': True,
        },
    }

    return d

def test_extract_search_variables(variable_info):
    individual = {
        'a': 5,
        'b': 2.5,
        'b0': 4.5,
    }
    
    point, _ = extract_search_variables(individual, variable_info)
    assert point['b0'] == 4.5

@pytest.mark.parametrize('seed', range(5))
def test_create_random_simplex(variable_info, seed, data_regression):
    individual = {
        'a': 5,
        'b': 2.5,
        'b0': 4.5,
    }
    
    point, fixture_variables = extract_search_variables(individual, variable_info)

    simplex = create_random_simplex(point, fixture_variables, variable_info, seed=seed)

    assert len(simplex) == 2
    assert 3.5 < simplex[1]['b0'] < 4.5

    # data regression test to make sure the seed is working
    data_regression.check(simplex)
    
def test_sort_simplex():
    simplex = [{'b0': 3.0}, {'b0': 4.0}]
    scores = [2.0, 1.1] 

    sorted_simplex, sorted_scores = sort_simplex(simplex, scores)

    assert sorted_simplex[0]['b0'] == 4.0
    assert sorted_scores[0] == 1.1