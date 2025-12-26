"""
Configuration management using pydantic-settings.

Loads configuration from environment variables with validation and type conversion.
"""

from typing import Optional, List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Service Configuration
    model_service_port: int = Field(default=4500, alias="MODEL_SERVICE_PORT")
    model_service_api_key: str = Field(..., alias="MODEL_SERVICE_API_KEY")
    model_service_log_level: str = Field(default="INFO", alias="MODEL_SERVICE_LOG_LEVEL")
    model_service_service_name: str = Field(default="model-service", alias="MODEL_SERVICE_SERVICE_NAME")

    # Database Configuration
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(..., alias="POSTGRES_DB")
    postgres_user: str = Field(..., alias="POSTGRES_USER")
    postgres_password: str = Field(..., alias="POSTGRES_PASSWORD")

    # RabbitMQ Configuration
    rabbitmq_host: str = Field(default="rabbitmq", alias="RABBITMQ_HOST")
    rabbitmq_port: int = Field(default=5672, alias="RABBITMQ_PORT")
    rabbitmq_user: str = Field(default="guest", alias="RABBITMQ_USER")
    rabbitmq_password: str = Field(default="guest", alias="RABBITMQ_PASSWORD")

    # WebSocket Gateway Configuration
    ws_gateway_host: str = Field(default="ws-gateway", alias="WS_GATEWAY_HOST")
    ws_gateway_port: int = Field(default=4400, alias="WS_GATEWAY_PORT")
    ws_gateway_api_key: str = Field(..., alias="WS_GATEWAY_API_KEY")

    # Model Storage Configuration
    model_storage_path: str = Field(default="/models", alias="MODEL_STORAGE_PATH")

    # Model Training Configuration
    model_training_min_dataset_size: int = Field(default=1000, alias="MODEL_TRAINING_MIN_DATASET_SIZE")
    model_training_max_duration_seconds: int = Field(default=1800, alias="MODEL_TRAINING_MAX_DURATION_SECONDS")
    # Top-K Analysis Configuration
    model_signal_top_k_percentage: int = Field(
        default=10,
        alias="MODEL_SIGNAL_TOP_K_PERCENTAGE",
        description="Top-K percentage to use for dynamic confidence threshold from model quality metrics. Default: 10 (top-10%). If top-k threshold is available, it will be used instead of static min_confidence_threshold. Valid values: 10, 20, 30, 50."
    )
    model_optimal_top_k_selection_strategy: str = Field(
        default="lift_then_accuracy",
        alias="MODEL_OPTIMAL_TOP_K_SELECTION_STRATEGY",
        description="Strategy for selecting optimal top-k percentage for each model. Options: 'lift_then_accuracy' (default - requires lift > 1.0, then maximizes accuracy), 'accuracy_only' (maximizes accuracy), 'lift_only' (maximizes lift), 'balanced' (maximizes accuracy * lift)."
    )
    
    model_activation_threshold: float = Field(
        default=0.75, 
        alias="MODEL_ACTIVATION_THRESHOLD",
        description="Minimum threshold for model activation (used with metric from MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC). Default: 0.75 (75%)"
    )
    model_quality_threshold_r2: float = Field(
        default=0.0,
        alias="MODEL_QUALITY_THRESHOLD_R2",
        description="Minimum R² score threshold for regression models to be auto-activated. R² > 0 means model is better than predicting mean. Default: 0.0 (model should be at least as good as mean prediction)"
    )
    model_quality_threshold_rmse: Optional[float] = Field(
        default=None,
        alias="MODEL_QUALITY_THRESHOLD_RMSE",
        description="Maximum RMSE threshold for regression models to be auto-activated (optional). If set, model will be activated only if RMSE <= this value. Default: None (not used)"
    )
    # Class Balancing and Hyperparameter Tuning Configuration
    model_training_use_smote: bool = Field(default=False, alias="MODEL_TRAINING_USE_SMOTE")
    model_training_class_weight_method: str = Field(default="inverse_frequency", alias="MODEL_TRAINING_CLASS_WEIGHT_METHOD")
    model_training_hyperparameter_tuning: bool = Field(default=False, alias="MODEL_TRAINING_HYPERPARAMETER_TUNING")
    model_training_tuning_method: str = Field(default="grid_search", alias="MODEL_TRAINING_TUNING_METHOD")
    model_training_tuning_max_iterations: int = Field(default=50, alias="MODEL_TRAINING_TUNING_MAX_ITERATIONS")
    model_training_quality_checks_enabled: bool = Field(default=True, alias="MODEL_TRAINING_QUALITY_CHECKS_ENABLED")
    model_training_threshold_optimization_metric: str = Field(
        default="f1",
        alias="MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC",
        description="Metric to optimize for threshold calibration: 'f1', 'pr_auc', 'balanced_accuracy', 'recall'. Default: 'f1'"
    )
    
    # Time-Based Retraining Configuration (for market-data-only training)
    model_retraining_interval_days: int = Field(
        default=7, 
        alias="MODEL_RETRAINING_INTERVAL_DAYS",
        description="Interval in days between automatic retraining. Default: 7 days."
    )
    model_retraining_check_interval_hours: int = Field(
        default=6,
        alias="MODEL_RETRAINING_CHECK_INTERVAL_HOURS",
        description="Interval in hours for checking if retraining is needed. Default: 6 hours."
    )
    model_retraining_train_period_days: int = Field(default=30, alias="MODEL_RETRAINING_TRAIN_PERIOD_DAYS")
    model_retraining_validation_period_days: int = Field(default=7, alias="MODEL_RETRAINING_VALIDATION_PERIOD_DAYS")
    model_retraining_test_period_days: int = Field(default=1, alias="MODEL_RETRAINING_TEST_PERIOD_DAYS")
    
    # Prediction Horizon Configuration
    model_prediction_horizon_seconds: int = Field(default=180, alias="MODEL_PREDICTION_HORIZON_SECONDS", description="Prediction horizon in seconds. Default: 180 (3 minutes) for better market movement capture.")
    
    # Target Registry Configuration
    target_registry_version: str = Field(default="latest", alias="TARGET_REGISTRY_VERSION", description="Target Registry version to use for dataset building. Default: 'latest'")
    
    # Model Hyperparameters Configuration
    model_hyperparams_config_path: str = Field(
        default="config/model_hyperparams.yaml",
        alias="MODEL_HYPERPARAMS_CONFIG_PATH",
        description="Path to YAML file with default model hyperparameters by model/task type."
    )
    
    # Timestamp Continuity Validation Configuration
    model_training_min_dataset_coverage_ratio: float = Field(
        default=0.8,
        alias="MODEL_TRAINING_MIN_DATASET_COVERAGE_RATIO",
        description="Minimum acceptable timestamp coverage ratio for training datasets. Datasets with coverage below this ratio will trigger warnings. Default: 0.8 (80% coverage). Lower coverage may indicate missing data periods that could affect model training quality."
    )
    model_training_warn_on_large_gaps: bool = Field(
        default=True,
        alias="MODEL_TRAINING_WARN_ON_LARGE_GAPS",
        description="Enable warnings for large timestamp gaps in training datasets. Default: true. Large gaps may indicate missing trading sessions or data collection issues."
    )
    model_training_critical_gap_threshold_seconds: int = Field(
        default=3600,
        alias="MODEL_TRAINING_CRITICAL_GAP_THRESHOLD_SECONDS",
        description="Critical gap threshold in seconds. Gaps exceeding this duration will trigger warnings. Default: 3600 (1 hour). Gaps larger than this may significantly affect temporal dependencies in model training."
    )
    
    # Version Mismatch Auto-Retraining Configuration
    version_mismatch_retraining_interval_hours: int = Field(
        default=24,
        alias="VERSION_MISMATCH_RETRAINING_INTERVAL_HOURS",
        description="Minimum hours between automatic retraining triggers due to version mismatches. Default: 24 (once per day max). Prevents spam retraining when versions change frequently."
    )
    
    # Classification Threshold Configuration
    model_classification_threshold: float = Field(default=0.005, alias="MODEL_CLASSIFICATION_THRESHOLD")
    
    # Regression Threshold Configuration (for converting predicted return to BUY/SELL/HOLD signal)
    model_regression_threshold: float = Field(
        default=0.001,
        alias="MODEL_REGRESSION_THRESHOLD",
        description="Threshold for regression models to convert predicted return to signal. If predicted_return > threshold: BUY, if < -threshold: SELL, else: HOLD. Default: 0.001 (0.1%)"
    )
    
    # Classification Probability Difference Threshold (hysteresis for BUY/SELL signals)
    model_min_probability_diff: float = Field(
        default=0.05,
        alias="MODEL_MIN_PROBABILITY_DIFF",
        description="Minimum difference between buy_probability and sell_probability required to generate a signal. If |buy_probability - sell_probability| < min_probability_diff, signal is HOLD (None). This provides hysteresis to prevent signals when model is uncertain. Default: 0.05 (5%). Only applies to classification models without calibrated thresholds."
    )
    model_regression_max_expected_return: float = Field(
        default=0.01,
        alias="MODEL_REGRESSION_MAX_EXPECTED_RETURN",
        description="Maximum expected return for confidence calculation in regression models. Used to normalize confidence (0-1). Default: 0.01 (1%)"
    )
    
    # Prediction Threshold Configuration (for improving recall of minority classes)
    # These thresholds are applied to class probabilities instead of using argmax
    # If probability for a class exceeds its threshold, that class is predicted
    # If multiple classes exceed thresholds, the one with highest probability is chosen
    # If no class exceeds threshold, falls back to argmax
    model_prediction_threshold_class_0: Optional[float] = Field(
        default=None, 
        alias="MODEL_PREDICTION_THRESHOLD_CLASS_0",
        description="Prediction threshold for class 0 (flat). If None, uses argmax. Range: 0.0-1.0"
    )
    model_prediction_threshold_class_1: Optional[float] = Field(
        default=None,
        alias="MODEL_PREDICTION_THRESHOLD_CLASS_1", 
        description="Prediction threshold for class 1 (up). If None, uses argmax. Lower threshold improves recall. Range: 0.0-1.0"
    )
    model_prediction_threshold_class_neg1: Optional[float] = Field(
        default=None,
        alias="MODEL_PREDICTION_THRESHOLD_CLASS_NEG1",
        description="Prediction threshold for class -1 (down). If None, uses argmax. Lower threshold improves recall. Range: 0.0-1.0"
    )
    model_prediction_use_threshold_calibration: bool = Field(
        default=False,
        alias="MODEL_PREDICTION_USE_THRESHOLD_CALIBRATION",
        description="If True, use threshold-based prediction instead of argmax for classification"
    )

    # Training Buffer Persistence Configuration
    buffer_persistence_enabled: bool = Field(default=True, alias="BUFFER_PERSISTENCE_ENABLED")
    buffer_recovery_on_startup: bool = Field(default=True, alias="BUFFER_RECOVERY_ON_STARTUP")
    buffer_max_recovery_events: int = Field(default=10000, alias="BUFFER_MAX_RECOVERY_EVENTS")

    # Training Queue Configuration
    training_queue_enabled: bool = Field(default=True, alias="TRAINING_QUEUE_ENABLED")
    training_queue_max_size: int = Field(default=10, alias="TRAINING_QUEUE_MAX_SIZE")
    training_force_cancel_on_critical: bool = Field(default=False, alias="TRAINING_FORCE_CANCEL_ON_CRITICAL")

    # Training Orchestrator Advanced Configuration
    max_parallel_training: int = Field(default=1, alias="MAX_PARALLEL_TRAINING")
    batch_buffer_update_interval_seconds: int = Field(default=10, alias="BATCH_BUFFER_UPDATE_INTERVAL_SECONDS")

    # Signal Generation Configuration
    signal_generation_rate_limit: int = Field(default=60, alias="SIGNAL_GENERATION_RATE_LIMIT")
    signal_generation_burst_allowance: int = Field(default=10, alias="SIGNAL_GENERATION_BURST_ALLOWANCE")
    signal_generation_skip_if_open_order: bool = Field(default=True, alias="SIGNAL_GENERATION_SKIP_IF_OPEN_ORDER")
    signal_generation_check_opposite_orders_only: bool = Field(default=False, alias="SIGNAL_GENERATION_CHECK_OPPOSITE_ORDERS_ONLY")

    # Warm-up Mode Configuration
    warmup_mode_enabled: bool = Field(default=True, alias="WARMUP_MODE_ENABLED")
    warmup_signal_frequency: int = Field(default=60, alias="WARMUP_SIGNAL_FREQUENCY")
    warmup_min_amount: float = Field(default=100.0, alias="WARMUP_MIN_AMOUNT")
    warmup_max_amount: float = Field(default=1000.0, alias="WARMUP_MAX_AMOUNT")
    warmup_randomness_level: float = Field(default=0.5, alias="WARMUP_RANDOMNESS_LEVEL")

    # Intelligent Mode Configuration
    # Frequency of intelligent (model-based) signals, signals per minute.
    # Works independently of warm-up frequency. If not set explicitly in .env,
    # falls back to a reasonable default (1 signal per second).
    intelligent_signal_frequency: float = Field(default=60.0, alias="INTELLIGENT_SIGNAL_FREQUENCY")

    # Trading Strategy Configuration
    trading_strategies: Optional[str] = Field(default=None, alias="TRADING_STRATEGIES")

    # Position Manager Configuration (for risk management)
    position_manager_host: str = Field(default="position-manager", alias="POSITION_MANAGER_HOST")
    position_manager_port: int = Field(default=4600, alias="POSITION_MANAGER_PORT")
    position_manager_api_key: str = Field(..., alias="POSITION_MANAGER_API_KEY")

    # Risk Management Configuration
    model_service_take_profit_pct: float = Field(default=3.0, alias="MODEL_SERVICE_TAKE_PROFIT_PCT")
    model_service_max_position_size_ratio: float = Field(default=0.8, alias="MODEL_SERVICE_MAX_POSITION_SIZE_RATIO")
    order_manager_max_position_size: float = Field(default=2.0, alias="ORDERMANAGER_MAX_POSITION_SIZE")

    # Balance Adaptation & Sync Configuration
    balance_adaptation_safety_margin: float = Field(
        default=0.95,
        alias="BALANCE_ADAPTATION_SAFETY_MARGIN",
    )
    balance_data_max_age_seconds: int = Field(default=60, alias="BALANCE_DATA_MAX_AGE_SECONDS")
    balance_sync_enabled: bool = Field(
        default=True,
        alias="BALANCE_SYNC_ENABLED",
        description="Enable on-demand balance sync via ws-gateway when DB snapshot is stale or missing",
    )
    balance_sync_min_interval_seconds: int = Field(
        default=30,
        alias="BALANCE_SYNC_MIN_INTERVAL_SECONDS",
        description="Minimum interval (seconds) between successive balance sync requests to ws-gateway",
    )
    balance_sync_timeout_seconds: float = Field(
        default=5.0,
        alias="BALANCE_SYNC_TIMEOUT_SECONDS",
        description="Timeout (seconds) for HTTP requests to ws-gateway balance sync endpoint",
    )

    # Market Data Cache Freshness Configuration
    market_data_max_age_seconds: int = Field(default=60, alias="MARKET_DATA_MAX_AGE_SECONDS")
    market_data_stale_warning_threshold_seconds: int = Field(
        default=30,
        alias="MARKET_DATA_STALE_WARNING_THRESHOLD_SECONDS",
    )

    # Signal Processing Delay Monitoring
    signal_processing_delay_alert_threshold_seconds: int = Field(
        default=300,
        alias="SIGNAL_PROCESSING_DELAY_ALERT_THRESHOLD_SECONDS",
    )

    # Target Evaluation Configuration
    target_evaluation_base_interval_seconds: int = Field(
        default=10,
        alias="TARGET_EVALUATION_BASE_INTERVAL_SECONDS",
        description="Base interval for target evaluation task (seconds). Default: 10",
    )
    target_evaluation_min_interval_seconds: int = Field(
        default=5,
        alias="TARGET_EVALUATION_MIN_INTERVAL_SECONDS",
        description="Minimum interval when many pending targets (seconds). Default: 5",
    )
    target_evaluation_max_interval_seconds: int = Field(
        default=60,
        alias="TARGET_EVALUATION_MAX_INTERVAL_SECONDS",
        description="Maximum interval when no pending targets (seconds). Default: 60",
    )
    target_evaluation_obsolete_age_days: int = Field(
        default=3,
        alias="TARGET_EVALUATION_OBSOLETE_AGE_DAYS",
        description="Age in days to mark targets as obsolete (stop processing). Default: 3",
    )

    # Position Cache Configuration (for optimization)
    position_cache_enabled: bool = Field(default=True, alias="POSITION_CACHE_ENABLED")
    position_cache_ttl_seconds: int = Field(default=30, alias="POSITION_CACHE_TTL_SECONDS")
    position_cache_max_size: int = Field(default=1000, alias="POSITION_CACHE_MAX_SIZE")

    # Feature Service Configuration
    feature_service_host: str = Field(default="feature-service", alias="FEATURE_SERVICE_HOST")
    feature_service_port: int = Field(default=4900, alias="FEATURE_SERVICE_PORT")
    feature_service_api_key: str = Field(..., alias="FEATURE_SERVICE_API_KEY")
    feature_service_use_queue: bool = Field(default=True, alias="FEATURE_SERVICE_USE_QUEUE")
    feature_service_feature_cache_ttl_seconds: int = Field(default=30, alias="FEATURE_SERVICE_FEATURE_CACHE_TTL_SECONDS")
    feature_service_feature_timeout_seconds: float = Field(
        default=60.0,
        alias="FEATURE_SERVICE_FEATURE_TIMEOUT_SECONDS",
        description="Timeout in seconds for feature requests to Feature Service API. Default: 60.0 seconds. Increase if feature-service is slow to respond."
    )
    
    # Feature Service Dataset Building Configuration
    feature_service_dataset_build_timeout_seconds: int = Field(default=3600, alias="FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS")
    feature_service_dataset_metadata_timeout_seconds: float = Field(default=60.0, alias="FEATURE_SERVICE_DATASET_METADATA_TIMEOUT_SECONDS", description="Timeout for getting dataset metadata from Feature Service API")
    feature_service_dataset_download_timeout_seconds: float = Field(default=600.0, alias="FEATURE_SERVICE_DATASET_DOWNLOAD_TIMEOUT_SECONDS", description="Timeout for downloading dataset files from Feature Service API")
    feature_service_target_computation_max_lookback_seconds: int = Field(default=300, alias="FEATURE_SERVICE_TARGET_COMPUTATION_MAX_LOOKBACK_SECONDS", description="Maximum lookback window for target computation data availability fallback in seconds (default: 300)")
    feature_service_dataset_poll_interval_seconds: int = Field(default=60, alias="FEATURE_SERVICE_DATASET_POLL_INTERVAL_SECONDS")
    feature_service_dataset_storage_path: str = Field(default="/datasets", alias="FEATURE_SERVICE_DATASET_STORAGE_PATH")
    
    # Legacy Feature Compatibility (for models trained before Feature Service integration)
    # Set to true only if you have old models that require legacy feature names (e.g., spread_percent)
    # New models should be trained on Feature Service features directly
    feature_service_legacy_feature_compatibility: bool = Field(default=False, alias="FEATURE_SERVICE_LEGACY_FEATURE_COMPATIBILITY")

    # Exit Strategy Configuration
    exit_strategy_enabled: bool = Field(default=True, alias="EXIT_STRATEGY_ENABLED")
    exit_strategy_rate_limit: int = Field(default=10, alias="EXIT_STRATEGY_RATE_LIMIT")

    # Take Profit Configuration
    # Note: threshold_pct now uses MODEL_SERVICE_TAKE_PROFIT_PCT (unified with intelligent_signal_generator)
    take_profit_enabled: bool = Field(default=True, alias="TAKE_PROFIT_ENABLED")
    take_profit_partial_exit: bool = Field(default=False, alias="TAKE_PROFIT_PARTIAL_EXIT")
    take_profit_partial_amount_pct: float = Field(default=50.0, alias="TAKE_PROFIT_PARTIAL_AMOUNT_PCT")

    # Stop Loss Configuration
    stop_loss_enabled: bool = Field(default=True, alias="STOP_LOSS_ENABLED")
    stop_loss_threshold_pct: float = Field(default=-2.0, alias="STOP_LOSS_THRESHOLD_PCT")

    # Time-Based Exit Configuration
    time_based_exit_enabled: bool = Field(default=False, alias="TIME_BASED_EXIT_ENABLED")
    time_based_exit_max_hours: int = Field(default=24, alias="TIME_BASED_EXIT_MAX_HOURS")
    time_based_exit_profit_target_pct: float = Field(default=1.0, alias="TIME_BASED_EXIT_PROFIT_TARGET_PCT")

    @field_validator("model_service_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper

    @field_validator("model_activation_threshold")
    @classmethod
    def validate_activation_threshold(cls, v: float) -> float:
        """Validate activation threshold is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Accuracy threshold must be between 0.0 and 1.0")
        return v
    
    @field_validator("model_prediction_threshold_class_0", "model_prediction_threshold_class_1", "model_prediction_threshold_class_neg1")
    @classmethod
    def validate_prediction_threshold(cls, v: Optional[float]) -> Optional[float]:
        """Validate prediction threshold is between 0 and 1 if provided."""
        if v is not None and not 0.0 <= v <= 1.0:
            raise ValueError("Prediction threshold must be between 0.0 and 1.0")
        return v

    @field_validator("model_training_min_dataset_size")
    @classmethod
    def validate_min_dataset_size(cls, v: int) -> int:
        """Validate minimum dataset size is positive."""
        if v <= 0:
            raise ValueError("Minimum dataset size must be positive")
        return v

    @field_validator("model_retraining_interval_days")
    @classmethod
    def validate_retraining_interval_days(cls, v: int) -> int:
        """Validate retraining interval is positive."""
        if v <= 0:
            raise ValueError("MODEL_RETRAINING_INTERVAL_DAYS must be positive")
        return v

    @field_validator("model_retraining_train_period_days")
    @classmethod
    def validate_retraining_train_period_days(cls, v: int) -> int:
        """Validate training period length is positive."""
        if v <= 0:
            raise ValueError("MODEL_RETRAINING_TRAIN_PERIOD_DAYS must be positive")
        return v

    @field_validator("model_retraining_validation_period_days")
    @classmethod
    def validate_retraining_validation_period_days(cls, v: int) -> int:
        """Validate validation period length is positive."""
        if v <= 0:
            raise ValueError("MODEL_RETRAINING_VALIDATION_PERIOD_DAYS must be positive")
        return v

    @field_validator("model_retraining_test_period_days")
    @classmethod
    def validate_retraining_test_period_days(cls, v: int) -> int:
        """Validate test period length is positive."""
        if v <= 0:
            raise ValueError("MODEL_RETRAINING_TEST_PERIOD_DAYS must be positive")
        return v

    @field_validator("model_training_min_dataset_coverage_ratio")
    @classmethod
    def validate_min_dataset_coverage_ratio(cls, v: float) -> float:
        """Validate minimum dataset coverage ratio is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("MODEL_TRAINING_MIN_DATASET_COVERAGE_RATIO must be between 0.0 and 1.0")
        return v

    @field_validator("model_training_critical_gap_threshold_seconds")
    @classmethod
    def validate_critical_gap_threshold_seconds(cls, v: int) -> int:
        """Validate critical gap threshold is positive."""
        if v <= 0:
            raise ValueError("MODEL_TRAINING_CRITICAL_GAP_THRESHOLD_SECONDS must be positive")
        return v

    @field_validator("model_training_class_weight_method")
    @classmethod
    def validate_class_weight_method(cls, v: str) -> str:
        """Validate class weight method is one of the supported options."""
        valid_methods = {"inverse_frequency", "balanced", "custom"}
        v_lower = v.lower()
        if v_lower not in valid_methods:
            raise ValueError(f"MODEL_TRAINING_CLASS_WEIGHT_METHOD must be one of {valid_methods}, got {v}")
        return v_lower

    @field_validator("model_training_threshold_optimization_metric")
    @classmethod
    def validate_threshold_optimization_metric(cls, v: str) -> str:
        """Validate threshold optimization metric is one of the supported options."""
        valid_metrics = {"f1", "pr_auc", "balanced_accuracy", "recall"}
        v_lower = v.lower()
        if v_lower not in valid_metrics:
            raise ValueError(f"MODEL_TRAINING_THRESHOLD_OPTIMIZATION_METRIC must be one of {valid_metrics}, got {v}")
        return v_lower

    @field_validator("model_training_tuning_method")
    @classmethod
    def validate_tuning_method(cls, v: str) -> str:
        """Validate tuning method is one of the supported options."""
        valid_methods = {"grid_search", "bayesian"}
        v_lower = v.lower()
        if v_lower not in valid_methods:
            raise ValueError(f"MODEL_TRAINING_TUNING_METHOD must be one of {valid_methods}, got {v}")
        return v_lower

    @field_validator("model_training_tuning_max_iterations")
    @classmethod
    def validate_tuning_max_iterations(cls, v: int) -> int:
        """Validate tuning max iterations is positive."""
        if v <= 0:
            raise ValueError("MODEL_TRAINING_TUNING_MAX_ITERATIONS must be positive")
        return v

    @field_validator("buffer_max_recovery_events")
    @classmethod
    def validate_buffer_max_recovery_events(cls, v: int) -> int:
        """Validate maximum recovery events is positive."""
        if v <= 0:
            raise ValueError("BUFFER_MAX_RECOVERY_EVENTS must be positive")
        return v

    @field_validator("training_queue_max_size")
    @classmethod
    def validate_training_queue_max_size(cls, v: int) -> int:
        """Validate training queue max size is positive."""
        if v <= 0:
            raise ValueError("TRAINING_QUEUE_MAX_SIZE must be positive")
        return v

    @field_validator("max_parallel_training")
    @classmethod
    def validate_max_parallel_training(cls, v: int) -> int:
        """Validate maximum parallel training tasks is positive."""
        if v <= 0:
            raise ValueError("MAX_PARALLEL_TRAINING must be positive")
        return v

    @field_validator("signal_generation_rate_limit")
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        """Validate rate limit is positive."""
        if v <= 0:
            raise ValueError("Rate limit must be positive")
        return v

    @field_validator("warmup_signal_frequency")
    @classmethod
    def validate_warmup_frequency(cls, v: int) -> int:
        """Validate warm-up signal frequency is positive."""
        if v <= 0:
            raise ValueError("Warm-up signal frequency must be positive")
        return v

    @field_validator("intelligent_signal_frequency")
    @classmethod
    def validate_intelligent_frequency(cls, v: float) -> float:
        """Validate intelligent signal frequency is positive."""
        if v <= 0:
            raise ValueError("Intelligent signal frequency must be positive")
        return v

    @field_validator("warmup_randomness_level")
    @classmethod
    def validate_randomness_level(cls, v: float) -> float:
        """Validate randomness level is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Randomness level must be between 0.0 and 1.0")
        return v

    @property
    def database_url(self) -> str:
        """Get PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def rabbitmq_url(self) -> str:
        """Get RabbitMQ connection URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )

    @property
    def ws_gateway_url(self) -> str:
        """Get WebSocket Gateway base URL."""
        return f"http://{self.ws_gateway_host}:{self.ws_gateway_port}"

    @property
    def position_manager_url(self) -> str:
        """Get Position Manager base URL."""
        return f"http://{self.position_manager_host}:{self.position_manager_port}"

    @property
    def feature_service_url(self) -> str:
        """Get Feature Service base URL."""
        return f"http://{self.feature_service_host}:{self.feature_service_port}"

    @property
    def trading_strategy_list(self) -> List[str]:
        """Parse trading strategies from comma-separated string."""
        if not self.trading_strategies:
            return []
        return [s.strip() for s in self.trading_strategies.split(",") if s.strip()]

    def validate_on_startup(self) -> None:
        """
        Validate configuration on startup.

        Checks file paths, database connectivity, RabbitMQ connectivity, and other critical settings.

        Raises:
            ValueError: If any validation fails
            ConfigurationError: If configuration is invalid
        """
        import os
        from pathlib import Path
        from .exceptions import ConfigurationError
        from .logging import get_logger

        logger = get_logger(__name__)
        errors = []

        # Validate model storage path exists and is writable
        try:
            storage_path = Path(self.model_storage_path)
            if not storage_path.exists():
                try:
                    storage_path.mkdir(parents=True, exist_ok=True)
                    logger.info("Created model storage directory", path=str(storage_path))
                except OSError as e:
                    errors.append(f"Cannot create model storage directory: {e}")

            if storage_path.exists() and not os.access(storage_path, os.W_OK):
                errors.append(f"Model storage path is not writable: {self.model_storage_path}")

            if storage_path.exists() and not os.access(storage_path, os.R_OK):
                errors.append(f"Model storage path is not readable: {self.model_storage_path}")
        except Exception as e:
            errors.append(f"Error validating model storage path: {e}")

        # Validate port ranges
        if not 1 <= self.model_service_port <= 65535:
            errors.append(f"Model service port must be between 1 and 65535, got {self.model_service_port}")

        if not 1 <= self.postgres_port <= 65535:
            errors.append(f"PostgreSQL port must be between 1 and 65535, got {self.postgres_port}")

        if not 1 <= self.rabbitmq_port <= 65535:
            errors.append(f"RabbitMQ port must be between 1 and 65535, got {self.rabbitmq_port}")

        if not 1 <= self.ws_gateway_port <= 65535:
            errors.append(f"WebSocket Gateway port must be between 1 and 65535, got {self.ws_gateway_port}")

        if not 1 <= self.position_manager_port <= 65535:
            errors.append(f"Position Manager port must be between 1 and 65535, got {self.position_manager_port}")

        # Validate risk management configuration
        if self.model_service_take_profit_pct < 0:
            errors.append(f"MODEL_SERVICE_TAKE_PROFIT_PCT must be non-negative, got {self.model_service_take_profit_pct}")

        if not 0.0 <= self.model_service_max_position_size_ratio <= 1.0:
            errors.append(f"MODEL_SERVICE_MAX_POSITION_SIZE_RATIO must be between 0.0 and 1.0, got {self.model_service_max_position_size_ratio}")

        if self.order_manager_max_position_size <= 0:
            errors.append(f"ORDERMANAGER_MAX_POSITION_SIZE must be positive, got {self.order_manager_max_position_size}")

        # Validate balance adaptation configuration
        if not 0.0 < self.balance_adaptation_safety_margin <= 1.0:
            errors.append(
                f"BALANCE_ADAPTATION_SAFETY_MARGIN must be between 0.0 (exclusive) and 1.0 (inclusive), "
                f"got {self.balance_adaptation_safety_margin}"
            )
        if self.balance_data_max_age_seconds <= 0:
            errors.append(
                f"BALANCE_DATA_MAX_AGE_SECONDS must be positive, got {self.balance_data_max_age_seconds}"
            )

        if self.balance_sync_min_interval_seconds <= 0:
            errors.append(
                f"BALANCE_SYNC_MIN_INTERVAL_SECONDS must be positive, got {self.balance_sync_min_interval_seconds}"
            )
        if self.balance_sync_timeout_seconds <= 0:
            errors.append(
                f"BALANCE_SYNC_TIMEOUT_SECONDS must be positive, got {self.balance_sync_timeout_seconds}"
            )

        # Validate market data cache freshness configuration
        if self.market_data_max_age_seconds <= 0:
            errors.append(
                f"MARKET_DATA_MAX_AGE_SECONDS must be positive, got {self.market_data_max_age_seconds}"
            )
        if self.market_data_stale_warning_threshold_seconds <= 0:
            errors.append(
                "MARKET_DATA_STALE_WARNING_THRESHOLD_SECONDS must be positive, "
                f"got {self.market_data_stale_warning_threshold_seconds}"
            )
        if self.market_data_stale_warning_threshold_seconds >= self.market_data_max_age_seconds:
            errors.append(
                "MARKET_DATA_STALE_WARNING_THRESHOLD_SECONDS must be less than "
                "MARKET_DATA_MAX_AGE_SECONDS "
                f"(got {self.market_data_stale_warning_threshold_seconds} "
                f"and {self.market_data_max_age_seconds})"
            )

        # Validate signal processing delay alert threshold
        if self.signal_processing_delay_alert_threshold_seconds <= 0:
            errors.append(
                "SIGNAL_PROCESSING_DELAY_ALERT_THRESHOLD_SECONDS must be positive, "
                f"got {self.signal_processing_delay_alert_threshold_seconds}"
            )

        # Validate position cache configuration
        if self.position_cache_enabled:
            if self.position_cache_ttl_seconds <= 0:
                errors.append(f"POSITION_CACHE_TTL_SECONDS must be positive, got {self.position_cache_ttl_seconds}")
            if self.position_cache_max_size <= 0:
                errors.append(f"POSITION_CACHE_MAX_SIZE must be positive, got {self.position_cache_max_size}")

        # Validate Feature Service configuration
        if not 1 <= self.feature_service_port <= 65535:
            errors.append(f"Feature Service port must be between 1 and 65535, got {self.feature_service_port}")
        if self.feature_service_feature_cache_ttl_seconds <= 0:
            errors.append(f"FEATURE_SERVICE_FEATURE_CACHE_TTL_SECONDS must be positive, got {self.feature_service_feature_cache_ttl_seconds}")
        if self.feature_service_dataset_build_timeout_seconds <= 0:
            errors.append(f"FEATURE_SERVICE_DATASET_BUILD_TIMEOUT_SECONDS must be positive, got {self.feature_service_dataset_build_timeout_seconds}")
        if self.feature_service_dataset_metadata_timeout_seconds <= 0:
            errors.append(f"FEATURE_SERVICE_DATASET_METADATA_TIMEOUT_SECONDS must be positive, got {self.feature_service_dataset_metadata_timeout_seconds}")
        if self.feature_service_dataset_download_timeout_seconds <= 0:
            errors.append(f"FEATURE_SERVICE_DATASET_DOWNLOAD_TIMEOUT_SECONDS must be positive, got {self.feature_service_dataset_download_timeout_seconds}")
        if self.feature_service_dataset_poll_interval_seconds <= 0:
            errors.append(f"FEATURE_SERVICE_DATASET_POLL_INTERVAL_SECONDS must be positive, got {self.feature_service_dataset_poll_interval_seconds}")

        # Validate time-based retraining configuration
        if self.model_retraining_interval_days <= 0:
            errors.append(f"MODEL_RETRAINING_INTERVAL_DAYS must be positive, got {self.model_retraining_interval_days}")
        if self.model_retraining_train_period_days <= 0:
            errors.append(f"MODEL_RETRAINING_TRAIN_PERIOD_DAYS must be positive, got {self.model_retraining_train_period_days}")
        if self.model_retraining_validation_period_days <= 0:
            errors.append(f"MODEL_RETRAINING_VALIDATION_PERIOD_DAYS must be positive, got {self.model_retraining_validation_period_days}")
        if self.model_retraining_test_period_days <= 0:
            errors.append(f"MODEL_RETRAINING_TEST_PERIOD_DAYS must be positive, got {self.model_retraining_test_period_days}")
        
        # Validate period lengths are logical (train >= validation >= test)
        if self.model_retraining_train_period_days < self.model_retraining_validation_period_days:
            errors.append(
                f"MODEL_RETRAINING_TRAIN_PERIOD_DAYS ({self.model_retraining_train_period_days}) "
                f"should be >= MODEL_RETRAINING_VALIDATION_PERIOD_DAYS ({self.model_retraining_validation_period_days})"
            )
        if self.model_retraining_validation_period_days < self.model_retraining_test_period_days:
            errors.append(
                f"MODEL_RETRAINING_VALIDATION_PERIOD_DAYS ({self.model_retraining_validation_period_days}) "
                f"should be >= MODEL_RETRAINING_TEST_PERIOD_DAYS ({self.model_retraining_test_period_days})"
            )

        # Validate API key is not empty
        if not self.model_service_api_key or len(self.model_service_api_key.strip()) == 0:
            errors.append("MODEL_SERVICE_API_KEY is required and cannot be empty")

        if not self.ws_gateway_api_key or len(self.ws_gateway_api_key.strip()) == 0:
            errors.append("WS_GATEWAY_API_KEY is required and cannot be empty")

        if not self.position_manager_api_key or len(self.position_manager_api_key.strip()) == 0:
            errors.append("POSITION_MANAGER_API_KEY is required and cannot be empty")

        if not self.feature_service_api_key or len(self.feature_service_api_key.strip()) == 0:
            errors.append("FEATURE_SERVICE_API_KEY is required and cannot be empty")

        # Validate database credentials
        if not self.postgres_db or len(self.postgres_db.strip()) == 0:
            errors.append("POSTGRES_DB is required and cannot be empty")

        if not self.postgres_user or len(self.postgres_user.strip()) == 0:
            errors.append("POSTGRES_USER is required and cannot be empty")

        if not self.postgres_password or len(self.postgres_password.strip()) == 0:
            errors.append("POSTGRES_PASSWORD is required and cannot be empty")

        # Validate training configuration
        if self.model_training_min_dataset_size <= 0:
            errors.append(f"MODEL_TRAINING_MIN_DATASET_SIZE must be positive, got {self.model_training_min_dataset_size}")

        if self.model_training_max_duration_seconds <= 0:
            errors.append(f"MODEL_TRAINING_MAX_DURATION_SECONDS must be positive, got {self.model_training_max_duration_seconds}")

        # Validate timestamp continuity validation configuration
        if not 0.0 <= self.model_training_min_dataset_coverage_ratio <= 1.0:
            errors.append(
                f"MODEL_TRAINING_MIN_DATASET_COVERAGE_RATIO must be between 0.0 and 1.0, "
                f"got {self.model_training_min_dataset_coverage_ratio}"
            )
        if self.model_training_critical_gap_threshold_seconds <= 0:
            errors.append(
                f"MODEL_TRAINING_CRITICAL_GAP_THRESHOLD_SECONDS must be positive, "
                f"got {self.model_training_critical_gap_threshold_seconds}"
            )

        # Validate signal generation configuration
        if self.signal_generation_rate_limit <= 0:
            errors.append(f"SIGNAL_GENERATION_RATE_LIMIT must be positive, got {self.signal_generation_rate_limit}")

        if self.signal_generation_burst_allowance < 0:
            errors.append(f"SIGNAL_GENERATION_BURST_ALLOWANCE must be non-negative, got {self.signal_generation_burst_allowance}")

        # Validate warm-up configuration
        if self.warmup_mode_enabled:
            if self.warmup_signal_frequency <= 0:
                errors.append(f"WARMUP_SIGNAL_FREQUENCY must be positive, got {self.warmup_signal_frequency}")

            if self.warmup_min_amount < 0:
                errors.append(f"WARMUP_MIN_AMOUNT must be non-negative, got {self.warmup_min_amount}")

            if self.warmup_max_amount < self.warmup_min_amount:
                errors.append(f"WARMUP_MAX_AMOUNT ({self.warmup_max_amount}) must be >= WARMUP_MIN_AMOUNT ({self.warmup_min_amount})")

        # Validate URL formats (basic check)
        try:
            # Test database URL format
            db_url = self.database_url
            if not db_url.startswith("postgresql://"):
                errors.append(f"Invalid database URL format: {db_url}")

            # Test RabbitMQ URL format
            rmq_url = self.rabbitmq_url
            if not rmq_url.startswith("amqp://"):
                errors.append(f"Invalid RabbitMQ URL format: {rmq_url}")

            # Test WebSocket Gateway URL format
            ws_url = self.ws_gateway_url
            if not ws_url.startswith("http://") and not ws_url.startswith("https://"):
                errors.append(f"Invalid WebSocket Gateway URL format: {ws_url}")

            # Test Position Manager URL format
            pm_url = self.position_manager_url
            if not pm_url.startswith("http://") and not pm_url.startswith("https://"):
                errors.append(f"Invalid Position Manager URL format: {pm_url}")

            # Test Feature Service URL format
            fs_url = self.feature_service_url
            if not fs_url.startswith("http://") and not fs_url.startswith("https://"):
                errors.append(f"Invalid Feature Service URL format: {fs_url}")
        except Exception as e:
            errors.append(f"Error validating URL formats: {e}")

        # Check for conflicting configurations
        if not self.warmup_mode_enabled and not self.trading_strategy_list:
            logger.warning("Warm-up mode is disabled and no trading strategies configured - signal generation may not work")

        # Raise error if any validation failed
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            logger.error("Configuration validation failed", errors=errors)
            raise ConfigurationError(error_message)

        logger.info("Configuration validation passed")


# Global settings instance
settings = Settings()

