from aiida import orm
from aiida_opsp.utils.merge_input import individual_to_inputs

def test_individual_to_inputs(data_regression):
    individual = {
        'a': 5,
        'b': 2.5,
        'nested_dict_a_c': 4.5,
        'inner_a_extra': 1.5,
    }
    
    variable_info = {
        'a': {
            'key_name': 'a',
            'var_type': 'int',
        },
        'b': {
            'key_name': 'b',
            'var_type': 'float',
        },
        'nested_dict_a_c': {
            'key_name': 'nested_dict:a.c',
            'var_type': 'float',
        },
        'inner_a_extra': {
            'key_name': 'nested_dict:inner_a.extra_param',
            'var_type': 'float',
        },
    }
    
    fixed_inputs = {
        'nested_dict': orm.Dict(dict={
            'inner_a': {
                'inner_b': 1.0,
            }
        }),
        'other': orm.Str('other'),
    }

    inputs = individual_to_inputs(individual, variable_info, fixed_inputs)
    
    assert 'a' in inputs
    assert 'b' in inputs
    assert inputs['a'].value == 5
    assert inputs['b'].value == 2.5
    
    data_regression.check(inputs["nested_dict"].get_dict())