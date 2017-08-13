import logging
import sys


class Log(object):
    level_set = False

    @staticmethod
    def set_loglevel(loglevel):
        global logger
        Log.get_logger().setLevel(loglevel)
        Log.get_logger().info("Set loglevel to %d" % loglevel)
        logger = Log.get_logger()
        Log.level_set = True

    @staticmethod
    def get_logger():
        return logging.getLogger("kernel_exp_family")


if not Log.level_set:
    level = logging.INFO
    logging.basicConfig(format='KERNEL_EXP_FAMILY: %(levelname)s: %(asctime)s: %(module)s.%(funcName)s(): %(message)s',
                        level=level)
    Log.get_logger().info("Global logger initialised with loglevel %d" % level)
    Log.level_set = True


class SimpleLogger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # needed for python 3 compatibility
        pass
