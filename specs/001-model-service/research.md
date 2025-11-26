# Research: Model Service Technology Decisions

**Feature**: Model Service - Trading Decision and ML Training Microservice  
**Date**: 2025-01-27  
**Purpose**: Resolve technical clarification items from implementation plan

## Research Questions

### 1. ML Framework Selection for Trading Signal Models

**Question**: Should we use scikit-learn, XGBoost, PyTorch, or TensorFlow for trading signal prediction models?

**Decision**: **scikit-learn** with **XGBoost** as primary model library

**Rationale**:
- **scikit-learn**: Provides comprehensive ML toolkit (preprocessing, feature engineering, model selection, evaluation metrics) essential for trading signal models
- **XGBoost**: Excellent performance for tabular data (order execution events, market features), handles non-linear relationships well, fast training and inference
- Both libraries are well-suited for structured/tabular data from order execution events and market data
- Fast training times (critical for <30 min requirement for 1M records)
- Low inference latency (critical for <5s signal generation requirement)
- Excellent Python ecosystem integration (pandas, numpy)
- Mature, stable libraries with extensive documentation
- Support for both batch training and incremental learning patterns
- Model interpretability features (feature importance, SHAP values) useful for trading strategy analysis
- Lower memory footprint compared to deep learning frameworks
- Easier to version, serialize, and deploy compared to deep learning models

**Alternatives Considered**:
- **PyTorch/TensorFlow**: Deep learning frameworks - overkill for initial implementation, longer training times, higher memory requirements, more complex deployment. Could be added later for more sophisticated models (LSTM for time series, reinforcement learning)
- **LightGBM**: Similar to XGBoost but XGBoost has broader adoption and more examples in trading contexts
- **Pure scikit-learn**: Good baseline but XGBoost typically provides better performance for structured data with non-linear patterns

**Implementation Notes**:
- Use `scikit-learn>=1.3.0` for preprocessing, feature engineering, and model evaluation
- Use `xgboost>=2.0.0` for gradient boosting models (primary model type)
- Support multiple model types (XGBoost, Random Forest, Logistic Regression) for strategy comparison
- Implement feature engineering pipeline using scikit-learn transformers
- Use scikit-learn's `partial_fit` for online learning support (incremental updates)
- For batch retraining, use full training pipeline with XGBoost

---

### 2. Model Serialization Format

**Question**: Should we use pickle, joblib, or format-specific serialization for ML models?

**Decision**: **joblib** for scikit-learn models, **XGBoost native format** for XGBoost models

**Rationale**:
- **joblib**: Optimized for numpy arrays and scikit-learn models, faster and more efficient than pickle for numerical data
- **XGBoost native format** (`.json` or `.ubj`): More portable, version-independent, human-readable (JSON), better cross-platform compatibility
- Both formats support model metadata and version information
- joblib is the recommended serialization method for scikit-learn (official documentation)
- XGBoost native format allows inspection of model structure and parameters
- Better error handling and validation compared to raw pickle
- Smaller file sizes for tree-based models (XGBoost native format)

**Alternatives Considered**:
- **pickle**: Standard Python serialization, but less efficient for numpy arrays, potential security concerns, version-dependent
- **ONNX**: Cross-platform format, but adds complexity and conversion overhead, not needed for single-platform deployment
- **PMML**: XML-based format, verbose and less efficient than native formats

**Implementation Notes**:
- Store scikit-learn models using `joblib.dump()` with compression (`.pkl` extension)
- Store XGBoost models using `model.save_model()` with `.json` format (`.json` extension)
- Store model files in `/models/v{version}/` directory structure
- Include model metadata (version, training timestamp, quality metrics) in filename or separate metadata file
- Implement model loading with error handling and validation
- Use file system with database metadata (file path stored in PostgreSQL)

---

### 3. Online Learning vs Batch Retraining Strategy

**Question**: How should we implement both online learning (incremental updates) and periodic batch retraining?

**Decision**: **Hybrid approach** - batch retraining as primary, online learning as optional enhancement

**Rationale**:
- **Batch retraining**: More reliable, better model quality, easier to version and rollback, supports full feature engineering pipeline
- **Online learning**: Faster adaptation to new patterns, but requires careful implementation to avoid catastrophic forgetting
- Batch retraining aligns better with requirements (scheduled periodic, data accumulation thresholds, quality degradation triggers)
- Online learning can be implemented later as an optimization for rapid adaptation
- Batch retraining provides better observability and quality tracking

**Alternatives Considered**:
- **Pure online learning**: Faster adaptation but risk of model degradation, harder to version and rollback
- **Pure batch retraining**: More reliable but slower adaptation to market changes

**Implementation Notes**:
- Primary: Implement batch retraining with configurable triggers (scheduled, data threshold, quality degradation)
- Secondary: Support online learning using scikit-learn's `partial_fit` for compatible models (Logistic Regression, SGD-based models)
- For XGBoost: Use batch retraining only (XGBoost doesn't support true online learning, but can retrain frequently)
- Implement training cancellation and restart mechanism for retraining conflicts
- Track training dataset size and quality metrics for both approaches

---

### 4. Model File Storage and Versioning Strategy

**Question**: How should we organize model files on the file system with versioning?

**Decision**: **Directory-based versioning** with database metadata

**Rationale**:
- Directory structure: `/models/v{version}/model.{ext}` provides clear organization
- Database stores metadata (version ID, timestamps, quality metrics, file path) for querying and management
- File system storage is efficient for large model files (avoiding database bloat)
- Easy to implement rollback (switch active version pointer in database)
- Supports multiple model files per version (if needed for ensemble models)

**Alternatives Considered**:
- **Flat file structure with version in filename**: Simpler but harder to organize and manage
- **Database BLOB storage**: Would bloat database, less efficient for large files
- **Object storage (S3)**: Overkill for single-server deployment, adds complexity

**Implementation Notes**:
- Directory structure: `/models/v{version}/model.json` (XGBoost) or `/models/v{version}/model.pkl` (scikit-learn)
- Database table: `model_versions` with columns: `version_id`, `file_path`, `trained_at`, `quality_metrics`, `training_config`, `is_active`
- Implement cleanup policy for old model versions (keep last N versions, archive others)
- Use Docker volume mount for model storage directory
- Implement file system health checks (disk space, write permissions)

---

## Summary of Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | Async/await support, ML ecosystem maturity |
| ML Framework | scikit-learn | 1.3+ | Comprehensive ML toolkit, preprocessing, evaluation |
| ML Model Library | XGBoost | 2.0+ | Fast, accurate models for tabular data |
| Data Processing | pandas | 2.0+ | Data manipulation and feature engineering |
| Model Serialization | joblib | 1.3+ | Efficient serialization for scikit-learn models |
| Message Queue Client | aio-pika | 9.0+ | Async RabbitMQ client (consistent with existing services) |
| Database Driver | asyncpg | 0.29+ | Fastest async PostgreSQL driver (consistent with existing services) |
| REST API | FastAPI | 0.104+ | Async support, OpenAPI generation (for health checks, monitoring) |
| Configuration | pydantic-settings | 2.0+ | Type-safe configuration management |
| Logging | structlog | 23.2+ | Structured logging with trace IDs |
| Testing | pytest + pytest-asyncio | Latest | Async test support |

## Dependencies Summary

**Core Dependencies**:
- `scikit-learn>=1.3.0` - ML framework (preprocessing, evaluation, baseline models)
- `xgboost>=2.0.0` - Gradient boosting models (primary model type)
- `pandas>=2.0.0` - Data manipulation and feature engineering
- `numpy>=1.24.0` - Numerical computing (dependency of scikit-learn, XGBoost)
- `joblib>=1.3.0` - Model serialization for scikit-learn models
- `aio-pika>=9.0.0` - RabbitMQ async client
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `fastapi>=0.104.0` - REST API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic-settings>=2.0.0` - Configuration management
- `structlog>=23.2.0` - Structured logging

**Optional Dependencies** (for future enhancements):
- `shap>=0.42.0` - Model interpretability (feature importance analysis)
- `matplotlib>=3.7.0` - Visualization for model quality metrics
- `scipy>=1.10.0` - Statistical functions for quality metrics calculation

