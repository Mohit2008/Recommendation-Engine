import os
from os import path
import logging

here = path.abspath(path.dirname(__file__))
LOG_LEVEL = logging.INFO
LOG_DIR = os.path.join(here, os.pardir, "log")


class RecommendationEngineConfig:
    pass