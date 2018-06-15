import os
import sys  # NOQA
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))
from python.utils import logsetup


logsetup.init_log_handler("pse")