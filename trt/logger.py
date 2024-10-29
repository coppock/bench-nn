import tensorrt as trt

import logging

# This is all because trt.Logger _actually_ writes to stdout!
class Logger(trt.ILogger):
    _levels = [
        logging.CRITICAL,
        logging.ERROR,
        logging.WARNING,
        logging.INFO,
        logging.DEBUG,
    ]

    def __init__(self, severity=trt.ILogger.WARNING):
        trt.ILogger.__init__(self)
        logging.basicConfig(level=self._levels[severity.value])

    def log(self, severity, msg):
        logging.log(self._levels[severity.value], msg)
