"""
Custom exception classes for the model service.

Provides structured error handling with appropriate exception types
for different error scenarios.
"""


class ModelServiceError(Exception):
    """Base exception for all model service errors."""

    pass


class ConfigurationError(ModelServiceError):
    """Raised when configuration is invalid or missing."""

    pass


class DatabaseError(ModelServiceError):
    """Raised when database operations fail."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    pass


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""

    pass


class MessageQueueError(ModelServiceError):
    """Raised when message queue operations fail."""

    pass


class MessageQueueConnectionError(MessageQueueError):
    """Raised when message queue connection fails."""

    pass


class MessageQueuePublishError(MessageQueueError):
    """Raised when message publishing fails."""

    pass


class MessageQueueConsumeError(MessageQueueError):
    """Raised when message consumption fails."""

    pass


class ModelStorageError(ModelServiceError):
    """Raised when model storage operations fail."""

    pass


class ModelNotFoundError(ModelStorageError):
    """Raised when a model file is not found."""

    pass


class ModelLoadError(ModelStorageError):
    """Raised when model loading fails."""

    pass


class ModelSaveError(ModelStorageError):
    """Raised when model saving fails."""

    pass


class ModelTrainingError(ModelServiceError):
    """Raised when model training fails."""

    pass


class DatasetError(ModelServiceError):
    """Raised when dataset operations fail."""

    pass


class DatasetInsufficientError(DatasetError):
    """Raised when dataset is too small for training."""

    pass


class SignalGenerationError(ModelServiceError):
    """Raised when signal generation fails."""

    pass


class SignalValidationError(SignalGenerationError):
    """Raised when signal validation fails."""

    pass


class RateLimitExceededError(SignalGenerationError):
    """Raised when rate limit is exceeded."""

    pass

