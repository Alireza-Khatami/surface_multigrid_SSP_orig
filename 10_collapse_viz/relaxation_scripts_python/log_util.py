"""One log sink for every module (the C++ prints everything to stderr): stderr,
plus an optional file set with set_log_file."""
import sys

_file = None


def set_log_file(f):
    global _file
    _file = f


def log(s):
    print(s, file=sys.stderr, flush=True)
    if _file is not None:
        _file.write(s + '\n')
        _file.flush()
