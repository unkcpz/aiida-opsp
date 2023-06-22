from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type

from plumpy.utils import AttributesFrozendict

def individual_to_inputs(individual, variable_info, fixture_inputs):
    """
    Convert an individual into a dictionary of inputs for a calculation.
    
    :param individual: individual to convert
    :param fixture_inputs: inputs of the calculation
    :return: inputs for the calculation
    """
    input_mapping = {}
    for key, value in individual.items():
        key_name = variable_info[key]["key_name"]
        input_mapping[key_name] = value

    inputs = _merge_nested_inputs(input_mapping, fixture_inputs)

    return inputs

def _merge_nested_inputs(input_mapping: dict, fixture_inputs):
    """iterate through input_mapping and merge the nested keys into fixture_inputs

    Args:
        input_mapping (dict): key, value pairs of nested keys
        fixture_inputs (dict): the fixture inputs of the calculation

    Returns:
        dict: the merged inputs
    """
    target_inputs = dict(fixture_inputs)

    nested_key_inputs = {}
    for key, value in input_mapping.items():
        # nest key separated by '.'
        nested_key_inputs[key] = value
        

    inputs = _merge_nested_keys(nested_key_inputs, target_inputs)
    
    return inputs

def _copy_nested_dict(value):
    """
    Copy nested dictionaries. `AttributesFrozendict` is converted into
    a (mutable) plain Python `dict`.

    This is needed because `copy.deepcopy` would create new AiiDA nodes.
    """
    if isinstance(value, (dict, AttributesFrozendict)):
        return {k: _copy_nested_dict(v) for k, v in value.items()}

    return value

def _merge_nested_keys(nested_key_inputs, target_inputs):
    """
    Maps nested_key_inputs onto target_inputs with support for nested keys:
        x.y:a.b -> x.y['a']['b']
    Note: keys will be python str; values will be AiiDA data types
    """
    def _get_nested_dict(in_dict, split_path):
        res_dict = in_dict
        for path_part in split_path:
            res_dict = res_dict.setdefault(path_part, {})
        return res_dict

    destination = _copy_nested_dict(target_inputs)

    for key, value in nested_key_inputs.items():
        full_port_path, *full_attr_path = key.split(':')
        *port_path, port_name = full_port_path.split('.')
        namespace = _get_nested_dict(in_dict=destination, split_path=port_path)

        if not full_attr_path:
            if not isinstance(value, orm.Node):
                value = to_aiida_type(value).store()
            res_value = value
        else:
            if len(full_attr_path) != 1:
                raise ValueError(f"Nested key syntax can contain at most one ':'. Got '{key}'")

            # Get or create the top-level dictionary.
            try:
                res_dict = namespace[port_name].get_dict()
            except KeyError:
                res_dict = {}

            *sub_dict_path, attr_name = full_attr_path[0].split('.')
            sub_dict = _get_nested_dict(in_dict=res_dict, split_path=sub_dict_path)
            sub_dict[attr_name] = _from_aiida_type(value)
            res_value = orm.Dict(dict=res_dict).store()

        namespace[port_name] = res_value

    return destination

def _from_aiida_type(value):
    """
    Convert an AiiDA data object to the equivalent Python object
    """
    if not isinstance(value, orm.Node):
        return value
    if isinstance(value, orm.BaseType):
        return value.value
    if isinstance(value, orm.Dict):
        return value.get_dict()
    if isinstance(value, orm.List):
        return value.get_list()
    raise TypeError(f'value of type {type(value)} is not supported')