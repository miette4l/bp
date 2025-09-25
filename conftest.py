import pytest
import importlib

def pytest_addoption(parser):
    parser.addoption(
        "--bp_module",
        action="store",
        default="bp_dicts",
        help="BP module to test (e.g., bp_dicts or bp_vector)"
    )

@pytest.fixture(scope="module")
def bp_module(request):
    module_name = request.config.getoption("--bp_module")
    return importlib.import_module(module_name)