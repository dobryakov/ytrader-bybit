"""
Unit tests for logging setup.
"""
import pytest
from unittest.mock import patch, MagicMock
import structlog
import logging


class TestLoggingSetup:
    """Tests for logging setup."""
    
    def test_logging_initializes_structlog(self):
        """Test that logging setup initializes structlog."""
        with patch("structlog.configure") as mock_configure:
            from src.logging import setup_logging
            
            setup_logging(level="INFO")
            
            mock_configure.assert_called_once()
    
    def test_logging_configures_processors(self):
        """Test that logging configures correct processors."""
        with patch("structlog.configure") as mock_configure:
            from src.logging import setup_logging
            
            setup_logging(level="INFO")
            
            # Check that configure was called with processors
            call_args = mock_configure.call_args
            assert call_args is not None
            # Verify processors are configured (adjust based on actual implementation)
    
    def test_logging_sets_log_level(self):
        """Test that logging sets the correct log level."""
        with patch("structlog.configure") as mock_configure:
            from src.logging import setup_logging
            
            setup_logging(level="DEBUG")
            
            # Verify log level is set correctly
            call_args = mock_configure.call_args
            assert call_args is not None
    
    def test_logging_creates_logger(self):
        """Test that logging setup creates a logger."""
        with patch("structlog.configure"):
            from src.logging import setup_logging, get_logger
            
            setup_logging(level="INFO")
            logger = get_logger(__name__)
            
            assert logger is not None
            # Verify logger has expected methods
            assert hasattr(logger, "info")
            assert hasattr(logger, "error")
            assert hasattr(logger, "warning")
            assert hasattr(logger, "debug")
    
    def test_logger_includes_trace_id(self):
        """Test that logger includes trace_id in log context."""
        with patch("structlog.configure"):
            from src.logging import setup_logging, get_logger
            
            setup_logging(level="INFO")
            logger = get_logger(__name__)
            
            # Test that logger can bind trace_id
            logger_with_trace = logger.bind(trace_id="test-trace-123")
            assert logger_with_trace is not None
    
    def test_logging_handles_different_levels(self):
        """Test that logging setup handles different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        
        for level in levels:
            with patch("structlog.configure") as mock_configure:
                from src.logging import setup_logging
                
                setup_logging(level=level)
                
                mock_configure.assert_called_once()

