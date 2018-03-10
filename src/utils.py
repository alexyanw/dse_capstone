import logging
logger = logging.getLogger('dp')
# stream handler
sh = logging.StreamHandler()
#sh.setLevel(level)
sformatter = logging.Formatter('[%(name)s] %(message)s')
sh.setFormatter(sformatter)
logger.addHandler(sh)

__all__ = ['set_logger']

def set_logger(level):
    logger.setLevel(level)

def fatal(msg):
    print("Error:", msg)
    exit(1)
