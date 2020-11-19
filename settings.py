"""
This file attempts loading some environment configuration variables, and
provides a sane default otherwise.
"""

try:
    from config import NCPUS
except ImportError:
    NCPUS = 2

try:
    from config import HEADLESS
except ImportError:
    HEADLESS = False

try:
    from config import RESULTS_PATH
except ImportError:
    RESULTS_PATH = "./results/"
