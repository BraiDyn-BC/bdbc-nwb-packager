# MIT License
#
# Copyright (c) 2024-2025 Keisuke Sehara and Ryo Aoki
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from typing import Optional
from pathlib import Path
from datetime import datetime
import logging as _logging
import os as _os


PathLike = str | Path

FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOGDIR_NAME = "bdbc-logs"
DEFAULT_TIMESTAMP_FMT = "%Y%m%d-%H%M%S"

APP_LOGGER: Optional[_logging.Logger] = None
LOGGING_FORMATTER: Optional[_logging.Formatter] = None
FILE_OUTPUT: Optional[_logging.FileHandler] = None
FILE_OUTPUT_PATH: Optional[Path] = None

DEBUG = _logging.DEBUG
INFO  = _logging.INFO
WARNING = _logging.WARNING
ERROR = _logging.ERROR
CRITICAL = _logging.CRITICAL


def get_logger(
    file_output: bool = False,
    file: Optional[PathLike] = None,
    logdir: Optional[PathLike] = None,
    prefix: str = 'log',
    timestamp_format: str = DEFAULT_TIMESTAMP_FMT,
    postfix: str = '',
    suffix: str = '.txt',
    file_mode: str = 'a',
    base_level: int = DEBUG,
    console_level: int = INFO,
) -> _logging.Logger:
    global APP_LOGGER
    if APP_LOGGER is None:
        APP_LOGGER = _init(
            base_level=base_level,
            console_level=console_level,
        )
    if file_output:
        set_file_output(
            root=APP_LOGGER,
            file=file,
            logdir=logdir,
            prefix=prefix,
            timestamp_format=timestamp_format,
            postfix=postfix,
            suffix=suffix,
            mode=file_mode,
        )
    return APP_LOGGER


def default_log_dir() -> Path:
    home = Path(_os.path.expanduser("~"))
    return home / DEFAULT_LOGDIR_NAME


def has_file_output() -> bool:
    return (FILE_OUTPUT is not None)


def get_file_path() -> Optional[Path]:
    return FILE_OUTPUT_PATH


def set_file_output(
    root: Optional[_logging.Logger] = None,
    file: Optional[PathLike] = None,
    logdir: Optional[PathLike] = None,
    prefix: str = 'log',
    timestamp_format: str = DEFAULT_TIMESTAMP_FMT,
    postfix: str = '',
    suffix: str = '.txt',
    mode: str = 'a',
):
    global FILE_OUTPUT
    global FILE_OUTPUT_PATH
    if FILE_OUTPUT is not None:
        # TODO: closing/re-opening the log output may cause
        # issues that could be hard to locate or debug.
        # so I leave this error as it is for the time being.
        raise RuntimeError('log file output has been already set')
    file = _setup_filepath(
        file=file,
        logdir=logdir,
        prefix=prefix,
        timestamp_format=timestamp_format,
        postfix=postfix,
        suffix=suffix,
    )
    if not file.parent.exists():
        file.parent.mkdir(parents=True)
    FILE_OUTPUT = _logging.FileHandler(
        file,
        mode=mode
    )
    FILE_OUTPUT_PATH = file
    if root is None:
        root = get_logger()
    FILE_OUTPUT.setLevel(root.level)
    FILE_OUTPUT.setFormatter(LOGGING_FORMATTER)
    root.addHandler(FILE_OUTPUT)


def _setup_filepath(
    file: Optional[PathLike] = None,
    logdir: Optional[PathLike] = None,
    prefix: str = 'log',
    timestamp_format: str = DEFAULT_TIMESTAMP_FMT,
    postfix: str = '',
    suffix: str = '.txt'
) -> Path:
    if file is not None:
        return Path(file)
    if logdir is None:
        logdir = default_log_dir()
    else:
        logdir = Path(logdir)
    filename = f"{prefix}{datetime.now().strftime(timestamp_format)}{postfix}{suffix}"
    return logdir / filename


def _init(
    base_level: int = DEBUG,
    console_level: int = INFO,
) -> _logging.Logger:
    global LOGGING_FORMATTER
    bdbc = _logging.getLogger('bdbc')
    existing = list(bdbc.handlers)
    for h in existing:
        bdbc.removeHandler(h)
    bdbc.propagate = False
    bdbc.setLevel(base_level)
    LOGGING_FORMATTER = _logging.Formatter(FORMAT, datefmt=DATEFMT)
    console = _logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(LOGGING_FORMATTER)
    bdbc.addHandler(console)
    return bdbc


def critical(msg: str, *args, **kwargs):
    get_logger().critical(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def debug(msg: str, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def exception(exc: BaseException, *args, **kwargs):
    get_logger().exception(exc, *args, **kwargs)


def test():
    info('testing info msg')
    debug('testing debug msg')
