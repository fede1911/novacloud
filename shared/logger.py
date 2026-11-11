"""Application logging of the blueprint.

Everything goes to var/logs/blueprint.log; the console gets only the errors,
so the log does not mix with the chat printed by the CLI. Follow the file with
`tail -f var/logs/blueprint.log` to see warnings and the flow of every turn.
"""

import logging
import pathlib

ROOT_LOGGER_NAME = "blueprint"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s - %(message)s"
LOG_FILE_NAME = "blueprint.log"


def get_logger(name: str) -> logging.Logger:
    """Returns the logger of a module, child of the blueprint logger.

    Args:
        name: Name of the module, usually __name__.

    Returns:
        The module logger.
    """
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def configure_logging(*, logs_dir: pathlib.Path, level: str) -> pathlib.Path:
    """Sends the blueprint logs to a file, and only the errors to stderr.

    Args:
        logs_dir: Folder of the log file, created if missing.
        level: Level of the file log (e.g. "INFO").

    Returns:
        Path of the log file.
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / LOG_FILE_NAME

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    stderr_handler = logging.StreamHandler()
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root_logger = logging.getLogger(ROOT_LOGGER_NAME)
    root_logger.setLevel(level)
    root_logger.handlers = [file_handler, stderr_handler]
    root_logger.propagate = False

    # The Strands logs are useful in the file, not on the console
    strands_logger = logging.getLogger("strands")
    strands_logger.setLevel(logging.WARNING)
    strands_logger.handlers = [file_handler]
    strands_logger.propagate = False
    return log_path
