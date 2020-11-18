import pytest


###############################################################################
# Test arguments
###############################################################################


def pytest_addoption(parser):
    """Parse command-line arguments"""
    parser.addoption(
        '--htk_directory',
        action='store',
        default=path())


@pytest.fixture(scope='session')
def htk_directory(pytest_config):
    """Handle htk_directory command-line argument"""
    return pytest_config.getoption('htk_directory')
