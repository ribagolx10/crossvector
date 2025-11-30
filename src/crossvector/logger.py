import logging
from typing import Optional

from crossvector.settings import settings as api_settings

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

_configured = False


def setup_global_logging(level: str = "INFO") -> None:
    """Configure the root logger once in a standardized format.

    Args:
        level: Log level name (e.g., "DEBUG", "INFO")
    """
    global _configured
    if _configured:
        return
    lvl = _LEVELS.get(level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    _configured = True


def get_logger(name: Optional[str] = None) -> "Logger":
    """Return a module/class logger. Ensures global logging is configured.

    Args:
        name: Logger name, usually __name__
    """
    return Logger(name or __name__)


class Logger:
    """Thin wrapper over standard logging with a convenience message method.

    - Honors global configuration via `setup_global_logging` invoked from settings.
    - Provides `.message(text)` which logs at `INFO` when LOG_LEVEL is INFO or higher,
      otherwise logs at `DEBUG`.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        if not _configured:
            setup_global_logging(api_settings.LOG_LEVEL)
        self._logger = logging.getLogger(name or __name__)

    def debug(self, msg: str, *args, **kwargs) -> None:
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        self._logger.critical(msg, *args, **kwargs)

    def message(self, msg: str, *args, **kwargs) -> None:
        # Decide level based on configured LOG_LEVEL
        level = (api_settings.LOG_LEVEL or "").upper()
        if level == "DEBUG":
            self.debug(msg, *args, **kwargs)
        elif level == "INFO" or level == "":  # UNSET treated as INFO
            self.info(msg, *args, **kwargs)
        else:
            # Respect other configured levels without overriding
            lvl = _LEVELS.get(level, logging.INFO)
            self._logger.log(lvl, msg, *args, **kwargs)
