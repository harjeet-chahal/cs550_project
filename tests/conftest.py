"""
Pytest config: put src/ on sys.path so tests can import project modules
without an editable install, and cache the Cora dataset for the session.
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))


@pytest.fixture(scope='session')
def cora():
    """Load Cora once per test session (parsing the .content / .cites
    files takes ~1s — sharing it keeps the suite snappy)."""
    from data_preprocessing import load_cora
    return load_cora()
