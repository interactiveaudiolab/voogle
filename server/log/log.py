import logging
import logging.config
import os

# Setup logging
logging.config.fileConfig(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf'))

loggers = {}

def get_logger(name):
    global loggers
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        if name != 'root':
            logger.propagate = False
        loggers[name] = logger
        return logger
