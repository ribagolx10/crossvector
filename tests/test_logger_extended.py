"""Extended tests for logger module to increase coverage."""

import logging
from unittest.mock import patch

import pytest

from crossvector.logger import (
    _LEVELS,
    Logger,
    get_logger,
    setup_global_logging,
)
from crossvector.settings import settings


class TestSetupGlobalLogging:
    """Tests for setup_global_logging function."""

    def test_setup_global_logging_with_default_level(self):
        """Test setting up global logging with default INFO level."""
        # Reset global state
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging()
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.INFO

    def test_setup_global_logging_with_debug_level(self):
        """Test setting up global logging with DEBUG level."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="DEBUG")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.DEBUG

    def test_setup_global_logging_with_warning_level(self):
        """Test setting up global logging with WARNING level."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="WARNING")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.WARNING

    def test_setup_global_logging_with_error_level(self):
        """Test setting up global logging with ERROR level."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="ERROR")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.ERROR

    def test_setup_global_logging_with_critical_level(self):
        """Test setting up global logging with CRITICAL level."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="CRITICAL")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.CRITICAL

    def test_setup_global_logging_with_invalid_level(self):
        """Test setting up global logging with invalid level defaults to INFO."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="INVALID")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.INFO

    def test_setup_global_logging_with_lowercase_level(self):
        """Test setting up global logging with lowercase level name."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging(level="info")
            mock_basicconfig.assert_called_once()
            args, kwargs = mock_basicconfig.call_args
            assert kwargs["level"] == logging.INFO

    def test_setup_global_logging_idempotent(self):
        """Test that setup_global_logging only configures once."""
        import crossvector.logger as logger_module

        logger_module._configured = True

        with patch("logging.basicConfig") as mock_basicconfig:
            setup_global_logging()
            # Should not be called if already configured
            mock_basicconfig.assert_not_called()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_with_name(self):
        """Test getting logger with explicit name."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig"):
            logger = get_logger("test_module")
            assert isinstance(logger, Logger)
            assert logger._logger.name == "test_module"

    def test_get_logger_without_name(self):
        """Test getting logger without name defaults to __name__."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig"):
            logger = get_logger()
            assert isinstance(logger, Logger)
            # Name will be the module name where get_logger is called
            assert logger._logger.name is not None

    def test_get_logger_with_none(self):
        """Test getting logger with explicit None."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig"):
            logger = get_logger(None)
            assert isinstance(logger, Logger)
            assert logger._logger.name is not None


class TestLoggerClass:
    """Tests for Logger class methods."""

    @pytest.fixture
    def logger(self):
        """Create a logger instance for testing."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig"):
            return Logger("test_logger")

    def test_logger_debug(self, logger):
        """Test debug method."""
        with patch.object(logger._logger, "debug") as mock_debug:
            logger.debug("Debug message")
            mock_debug.assert_called_once_with("Debug message")

    def test_logger_debug_with_args(self, logger):
        """Test debug method with format args."""
        with patch.object(logger._logger, "debug") as mock_debug:
            logger.debug("Debug %s", "message")
            mock_debug.assert_called_once_with("Debug %s", "message")

    def test_logger_info(self, logger):
        """Test info method."""
        with patch.object(logger._logger, "info") as mock_info:
            logger.info("Info message")
            mock_info.assert_called_once_with("Info message")

    def test_logger_info_with_kwargs(self, logger):
        """Test info method with keyword arguments."""
        with patch.object(logger._logger, "info") as mock_info:
            logger.info("Info %s", "message", extra={"key": "value"})
            mock_info.assert_called_once_with("Info %s", "message", extra={"key": "value"})

    def test_logger_warning(self, logger):
        """Test warning method."""
        with patch.object(logger._logger, "warning") as mock_warning:
            logger.warning("Warning message")
            mock_warning.assert_called_once_with("Warning message")

    def test_logger_error(self, logger):
        """Test error method."""
        with patch.object(logger._logger, "error") as mock_error:
            logger.error("Error message")
            mock_error.assert_called_once_with("Error message")

    def test_logger_critical(self, logger):
        """Test critical method."""
        with patch.object(logger._logger, "critical") as mock_critical:
            logger.critical("Critical message")
            mock_critical.assert_called_once_with("Critical message")

    def test_logger_message_with_debug_level(self, logger):
        """Test message method when LOG_LEVEL is DEBUG."""
        with patch.object(settings, "LOG_LEVEL", "DEBUG"):
            with patch.object(logger, "debug") as mock_debug:
                logger.message("Test message")
                mock_debug.assert_called_once()

    def test_logger_message_with_info_level(self, logger):
        """Test message method when LOG_LEVEL is INFO."""
        with patch.object(settings, "LOG_LEVEL", "INFO"):
            with patch.object(logger, "info") as mock_info:
                logger.message("Test message")
                mock_info.assert_called_once()

    def test_logger_message_with_empty_level(self, logger):
        """Test message method when LOG_LEVEL is empty (defaults to INFO)."""
        with patch.object(settings, "LOG_LEVEL", ""):
            with patch.object(logger, "info") as mock_info:
                logger.message("Test message")
                mock_info.assert_called_once()

    def test_logger_message_with_warning_level(self, logger):
        """Test message method when LOG_LEVEL is WARNING."""
        with patch.object(settings, "LOG_LEVEL", "WARNING"):
            with patch.object(logger._logger, "log") as mock_log:
                logger.message("Test message")
                mock_log.assert_called_once()
                args, kwargs = mock_log.call_args
                assert args[0] == logging.WARNING

    def test_logger_message_with_error_level(self, logger):
        """Test message method when LOG_LEVEL is ERROR."""
        with patch.object(settings, "LOG_LEVEL", "ERROR"):
            with patch.object(logger._logger, "log") as mock_log:
                logger.message("Test message")
                mock_log.assert_called_once()
                args, kwargs = mock_log.call_args
                assert args[0] == logging.ERROR

    def test_logger_message_with_critical_level(self, logger):
        """Test message method when LOG_LEVEL is CRITICAL."""
        with patch.object(settings, "LOG_LEVEL", "CRITICAL"):
            with patch.object(logger._logger, "log") as mock_log:
                logger.message("Test message")
                mock_log.assert_called_once()
                args, kwargs = mock_log.call_args
                assert args[0] == logging.CRITICAL

    def test_logger_message_with_lowercase_level(self, logger):
        """Test message method with lowercase log level."""
        with patch.object(settings, "LOG_LEVEL", "debug"):
            with patch.object(logger, "debug") as mock_debug:
                logger.message("Test message")
                mock_debug.assert_called_once()

    def test_logger_initialization_calls_setup(self):
        """Test that Logger initialization calls setup_global_logging if needed."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("crossvector.logger.setup_global_logging") as mock_setup:
            with patch("logging.basicConfig"):
                Logger("test")
                mock_setup.assert_called_once()

    def test_logger_initialization_with_none_name(self):
        """Test Logger initialization with None name."""
        import crossvector.logger as logger_module

        logger_module._configured = False

        with patch("logging.basicConfig"):
            logger = Logger(None)
            assert logger._logger.name is not None


class TestLevelsMapping:
    """Tests for _LEVELS mapping."""

    def test_levels_mapping_contains_all_standard_levels(self):
        """Test that _LEVELS contains all standard logging levels."""
        assert "CRITICAL" in _LEVELS
        assert "ERROR" in _LEVELS
        assert "WARNING" in _LEVELS
        assert "INFO" in _LEVELS
        assert "DEBUG" in _LEVELS

    def test_levels_mapping_values_are_correct(self):
        """Test that _LEVELS maps to correct logging level integers."""
        assert _LEVELS["CRITICAL"] == logging.CRITICAL
        assert _LEVELS["ERROR"] == logging.ERROR
        assert _LEVELS["WARNING"] == logging.WARNING
        assert _LEVELS["INFO"] == logging.INFO
        assert _LEVELS["DEBUG"] == logging.DEBUG
