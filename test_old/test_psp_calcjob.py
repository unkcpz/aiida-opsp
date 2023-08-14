from aiida.common import datastructures

def test_oncv_default(fixture_sandbox, generate_calc_job, generate_inputs_oncv, file_regression):
    """Test a default `opsp.pseudo.oncvpsp`."""
    entry_point_name = 'opsp.pseudo.oncvpsp'

    inputs = generate_inputs_oncv()
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    file_regression.check(input_written, encoding='utf-8', extension='.in')
    
def test_oncv_with_test_configurations(fixture_sandbox, generate_calc_job, generate_inputs_oncv, file_regression):
    """Test a inputs with test configurations `opsp.pseudo.oncvpsp`."""
    entry_point_name = 'opsp.pseudo.oncvpsp'

    inputs = generate_inputs_oncv(True)
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    file_regression.check(input_written, encoding='utf-8', extension='.in')
