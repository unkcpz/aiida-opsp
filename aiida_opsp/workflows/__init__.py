import yaml

from aiida.engine.persistence import ObjectLoader
from aiida import orm

from functools import singledispatch

# Generic process evaluation the code is ref from aiida-optimize
_YAML_IDENTIFIER = '!!YAML!!'

@singledispatch
def get_fullname(cls_obj):
    """
    Serializes an AiiDA process class / function to an AiiDA String.
    :param cls_obj: Object to be serialized
    :type cls_obj: Process
    """
    try:
        return orm.Str(ObjectLoader().identify_object(cls_obj))
    except ValueError:
        return orm.Str(_YAML_IDENTIFIER + yaml.dump(cls_obj))

#: Keyword arguments to be passed to ``spec.input`` for serializing an input which is a class / process into a string.
PROCESS_INPUT_KWARGS = {
    'valid_type': orm.Str,
    'serializer': get_fullname,
}

def load_object(cls_name):
    """
    Loads the process from the serialized string.
    """
    if isinstance(cls_name, orm.Str):
        cls_name_str = cls_name.value
    else:
        cls_name_str = str(cls_name)
    try:
        return ObjectLoader().load_object(cls_name_str)
    except ValueError as err:
        if cls_name_str.startswith(_YAML_IDENTIFIER):
            return yaml.load(cls_name_str[len(_YAML_IDENTIFIER):])
        raise ValueError(f"Could not load class name '{cls_name_str}'.") from err
