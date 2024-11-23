import argparse
import os
import signal
import sys
import time

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__()
        self.add_argument('-m', '--iteration-count', type=int)
        self.add_argument('-n', '--batch-size', default=1, type=int)

class Iterator:
    def _handler(self, *_):
        self.done = True

    def __init__(self, n):
        self.stdout = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

        self.done = False
        for signum in signal.SIGTERM, signal.SIGINT:
            signal.signal(signum, self._handler)

        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        t_f = time.time()
        # On first iteration, we don't want to print anything.
        try: print(t_f, t_f - self.t_i, file=self.stdout, flush=True)
        except AttributeError: pass
        if self.done or self.n and self.i >= self.n: raise StopIteration
        self.i += 1
        self.t_i = time.time()
