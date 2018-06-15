from logging.handlers import WatchedFileHandler  # NOQA
import logging  # NOQA
import os  # NOQA
import sys  # NOQA

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from python import seconfig
except ImportError:
    if os.geteuid() == 0:
        raise Exception("don't run script as root, please")
    else:
        raise Exception("PYTHONPATH should point to seconfig.py")

FORMAT = ("%(asctime)s.%(msecs)03d;%(thread)x;%(filename)s;"
          "%(funcName)s;%(levelname)s;%(message)s")
DATEFMT = '%Y%m%d%I%M%S'

logger = logging.getLogger(__name__)


def get_handler(basename):
    logfile = "{}.log".format(basename)
    logfile_full_path = os.path.join(seconfig.LOG_DIR, logfile)
    log_handler = WatchedFileHandler(logfile_full_path)
    log_handler.setFormatter(logging.Formatter(FORMAT, datefmt=DATEFMT))
    return log_handler


def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """log uncaught exceptions before exiting the program
    """
    if not issubclass(exc_type, KeyboardInterrupt):
        logger.fatal("1558;Exit with uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def init_root_logger():

    # shorten the more common level names i.e. DEBUG -> D
    logging.addLevelName(logging.DEBUG, 'D')
    logging.addLevelName(logging.INFO, 'I')
    logging.addLevelName(logging.ERROR, 'E')
    logging.addLevelName(logging.FATAL, 'F')

    sys.excepthook = handle_uncaught_exception


def init_log_handler(basename, replace_existing=False):
    main_logger = logging.getLogger()
    if replace_existing:
        main_logger.handlers = []
    if not main_logger.handlers:
        # the root logger has no handler, so create one
        handler = get_handler(basename)
        main_logger.addHandler(handler)
        main_logger.setLevel(seconfig.LOG_LEVEL)
        main_logger.info("1559;started %s logger", basename)
