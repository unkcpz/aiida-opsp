

def test_oncv_default(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_oncv, data_regression):
    """Test oncv parser
    """
    name = 'default'
    entry_point_calc_job = 'opsp.pseudo.oncv'
    entry_point_parser = 'opsp.pseudo.oncv'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, True))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_parameters' in results
    assert 'output_pseudo' in results

    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict(),
    })
    
def test_oncv_no_pseudo_dump(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_oncv, data_regression):
    """Test oncv parser
    """
    name = 'no_pseudo_dump'
    entry_point_calc_job = 'opsp.pseudo.oncv'
    entry_point_parser = 'opsp.pseudo.oncv'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, False))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_pseudo' not in results

def test_oncv_error_pspot_has_node(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_oncv, data_regression):
    """Test oncv parser
    """
    name = 'pspot_has_node'
    entry_point_calc_job = 'opsp.pseudo.oncv'
    entry_point_parser = 'opsp.pseudo.oncv'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, False))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    expected_exit_status = node.process_class.exit_codes.ERROR_PSPOT_HAS_NODE.status

    assert calcfunction.is_failed
    assert calcfunction.exit_status == expected_exit_status
    assert 'output_parameters' in results

    # TODO: it is now empty, can add some general metadata instead of output parsed only
    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict(),
    })
    
def test_oncv_error_lschvkbb(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_oncv, data_regression):
    """Test oncv parser
    """
    name = 'error_lschvkbb'
    entry_point_calc_job = 'opsp.pseudo.oncv'
    entry_point_parser = 'opsp.pseudo.oncv'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, False))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)
    expected_exit_status = node.process_class.exit_codes.ERROR_LSCHVKBB.status

    assert calcfunction.is_failed
    assert calcfunction.exit_status == expected_exit_status
    assert 'output_parameters' in results

    # TODO: it is now empty, can add some general metadata instead of output parsed only
    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict(),
    })
    
def test_oncv_warning_test_lschvkbb(fixture_localhost, generate_calc_job_node, generate_parser, generate_inputs_oncv, data_regression):
    """Test oncv parser
    """
    name = 'warning_lschvkbb_not_converge'
    entry_point_calc_job = 'opsp.pseudo.oncv'
    entry_point_parser = 'opsp.pseudo.oncv'

    node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, name, generate_inputs_oncv(True, False))
    parser = generate_parser(entry_point_parser)
    results, calcfunction = parser.parse_from_node(node, store_provenance=False)

    assert calcfunction.is_finished, calcfunction.exception
    assert calcfunction.is_finished_ok, calcfunction.exit_message
    assert 'output_pseudo' not in results

    # TODO: it is now empty, can add some general metadata instead of output parsed only
    data_regression.check({
        'output_parameters': results['output_parameters'].get_dict(),
    })
    
    