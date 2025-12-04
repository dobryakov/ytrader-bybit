# Research: Feature Service Technology Decisions

**Feature**: Feature Service for Real-Time Feature Computation and Dataset Building  
**Date**: 2025-01-27  
**Purpose**: Resolve technical clarification items from implementation plan

## Research Questions

### 1. Time Series Processing Library Selection

**Question**: Should we use `pandas` or `polars` for time series processing and rolling window computations?

**Decision**: `pandas` library

**Rationale**:
- `pandas` is the de facto standard for time series data processing in Python
- Excellent support for rolling windows, time-based indexing, and resampling operations
- Mature ecosystem with extensive documentation and community support
- Well-integrated with NumPy for numerical computations
- Sufficient performance for our use case (real-time features computed at 1s, 3s, 15s, 1m intervals)
- Better compatibility with existing Python data science stack (numpy, scipy)
- Easier to maintain and debug due to widespread familiarity

**Alternatives Considered**:
- `polars`: Faster for large datasets, but:
  - Less mature ecosystem and fewer examples for time series operations
  - Steeper learning curve for team members
  - Overkill for our real-time computation needs (small rolling windows)
  - Less intuitive API for time-based operations compared to pandas
  - Our use case doesn't require the performance benefits of polars (we process small windows in real-time, not large batch operations)

**Implementation Notes**:
- Use `pandas>=2.0.0` for Python 3.11+ compatibility
- Leverage `pandas.DataFrame.rolling()` for rolling window computations (1s, 3s, 15s, 1m)
- Use `pandas.Timestamp` for time-based indexing and alignment
- Consider `pandas.DataFrame.resample()` for time-based aggregations if needed

---

### 2. Parquet Storage Library Selection

**Question**: Should we use `pyarrow` or `fastparquet` for reading/writing Parquet files?

**Decision**: `pyarrow` library

**Rationale**:
- `pyarrow` is the official Apache Arrow implementation and the most widely used Parquet library
- Better performance for both read and write operations, especially for large files
- Native integration with pandas (can convert directly to/from pandas DataFrames)
- Better support for complex data types and nested structures
- Active development and maintenance by Apache Arrow project
- Better compression options and performance optimizations
- Industry standard for Parquet file operations

**Alternatives Considered**:
- `fastparquet`: 
  - Slower performance compared to pyarrow
  - Less active maintenance
  - Limited support for newer Parquet format features
  - Not as well-integrated with pandas ecosystem

**Implementation Notes**:
- Use `pyarrow>=14.0.0` for Python 3.11+ compatibility
- Use `pyarrow.parquet.ParquetFile` for reading historical data
- Use `pyarrow.parquet.write_table()` for writing raw market data
- Organize Parquet files by data type and date (e.g., `orderbook/2025-01-27.parquet`, `trades/2025-01-27.parquet`)
- Use partitioning by symbol if needed for better query performance

---

### 3. Feature Computation Libraries

**Question**: What libraries should we use for feature computation (rolling aggregations, VWAP, volatility calculations)?

**Decision**: `pandas` + `numpy` + `scipy` (for statistical functions)

**Rationale**:
- `pandas`: Rolling windows, time-based aggregations, resampling (as established in Research Question 1)
- `numpy`: Fast numerical operations, array computations, mathematical functions
- `scipy`: Statistical functions (e.g., for volatility calculations, z-score computations)
- This combination provides all necessary functionality for our feature types:
  - Price features: pandas rolling for returns, VWAP calculations
  - Volatility: scipy or numpy for standard deviation calculations
  - Orderflow features: pandas groupby and aggregations
  - Orderbook features: numpy for depth calculations, pandas for rolling aggregations
  - Temporal features: numpy/pandas for time_of_day, scipy for z-score

**Implementation Notes**:
- Use `numpy>=1.24.0` for numerical computations
- Use `scipy>=1.10.0` for statistical functions (if needed for advanced volatility calculations)
- Implement feature computation functions that accept pandas DataFrames and return feature values
- Ensure all feature computation functions are pure (no side effects) to guarantee online/offline identity

---

### 4. Orderbook Reconstruction Data Structures and Algorithms

**Question**: What data structures and algorithms should we use for orderbook state management (snapshot + delta reconstruction)?

**Decision**: In-memory orderbook state using Python dictionaries/collections + ordered data structures

**Rationale**:
- For real-time processing: Maintain in-memory orderbook state as nested dictionaries:
  - `orderbook[symbol] = {'bids': SortedDict, 'asks': SortedDict, 'sequence': int, 'timestamp': datetime}`
  - Use `sortedcontainers.SortedDict` or `bisect` module for maintaining sorted price levels
  - Fast O(log n) insertions and deletions for delta updates
  - O(1) access to best bid/ask prices
- For offline reconstruction: Read snapshot + all deltas sequentially, apply deltas in order
- Sequence number tracking: Store sequence number with each orderbook state to detect gaps
- Snapshot recovery: When desynchronization detected, request snapshot from ws-gateway or read from Parquet files

**Alternatives Considered**:
- Full database storage of orderbook state: Too slow for real-time updates, unnecessary overhead
- Redis for orderbook state: Adds external dependency, not needed for single-service state
- Complex tree structures: Overkill for typical orderbook depth (top 5-10 levels sufficient)

**Implementation Notes**:
- Use `sortedcontainers>=2.4.0` for efficient sorted data structures (or `bisect` module from stdlib)
- Maintain orderbook state per symbol in memory
- Store all deltas in Parquet files for offline reconstruction
- Implement snapshot + delta reconstruction algorithm:
  1. Load snapshot at time T
  2. Apply all deltas sequentially from T to target time
  3. Validate sequence numbers to detect gaps
  4. Request snapshot if gap detected
- For offline mode: Read snapshot and deltas from Parquet files, reconstruct state for each timestamp

---

### 5. Feature Identity Testing Strategy

**Question**: How should we test feature identity between online and offline computation modes?

**Decision**: Derivation tests comparing online vs offline feature values on identical input data

**Rationale**:
- Create test fixtures with known market data sequences
- Run same data through both online engine (real-time processing) and offline engine (batch processing)
- Compare feature values with exact match (or within floating-point tolerance)
- Use pytest parameterization to test multiple scenarios (different symbols, time periods, feature types)
- Store test data in Parquet format to ensure consistency

**Implementation Notes**:
- Create test fixtures with synthetic or recorded market data
- Implement test function: `test_feature_identity_online_vs_offline(symbol, time_period, feature_list)`
- Use `pytest.approx()` for floating-point comparisons (tolerance: 1e-9 for exact match requirement)
- Test all feature types: price, orderflow, orderbook, perpetual, temporal, position features
- Test edge cases: missing data, orderbook desynchronization recovery, position data fallback
- Ensure Feature Registry configuration is identical for both modes

**Test Structure**:
```python
def test_feature_identity_online_vs_offline():
    # 1. Load test data (market data sequence)
    # 2. Process through online engine (simulate real-time)
    # 3. Process through offline engine (batch mode)
    # 4. Compare feature vectors at each timestamp
    # 5. Assert exact match (within tolerance)
```

---

## Summary of Technology Stack

| Component | Technology | Version | Rationale |
|-----------|-----------|---------|-----------|
| Language | Python | 3.11+ | Async/await support, ecosystem maturity |
| REST API | FastAPI | 0.104+ | Async support, OpenAPI generation |
| Message Queue Client | aio-pika | 9.0+ | Async RabbitMQ client |
| Database Driver | asyncpg | 0.29+ | Fastest async PostgreSQL driver |
| HTTP Client | httpx | 0.25+ | Async HTTP client for Position Manager API |
| Time Series Processing | pandas | 2.0+ | Standard library for time series, rolling windows |
| Numerical Computing | numpy | 1.24+ | Fast array operations, mathematical functions |
| Statistical Functions | scipy | 1.10+ | Statistical calculations (volatility, z-score) |
| Parquet Storage | pyarrow | 14.0+ | Official Apache Arrow implementation, best performance |
| Data Structures | sortedcontainers | 2.4+ | Efficient sorted data structures for orderbook |
| Configuration | pydantic-settings | 2.0+ | Type-safe configuration management |
| Logging | structlog | 23.2+ | Structured logging with trace IDs |
| Testing | pytest + pytest-asyncio | Latest | Async test support, feature identity tests |

## Dependencies Summary

**Core Dependencies**:
- `fastapi>=0.104.0` - REST API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `aio-pika>=9.0.0` - RabbitMQ async client
- `asyncpg>=0.29.0` - PostgreSQL async driver
- `httpx>=0.25.0` - Async HTTP client
- `pandas>=2.0.0` - Time series processing, rolling windows
- `numpy>=1.24.0` - Numerical computations
- `scipy>=1.10.0` - Statistical functions
- `pyarrow>=14.0.0` - Parquet file read/write
- `sortedcontainers>=2.4.0` - Efficient sorted data structures for orderbook
- `pydantic-settings>=2.0.0` - Configuration management
- `structlog>=23.2.0` - Structured logging

**Testing Dependencies**:
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.21.0` - Async test support
- `pytest-mock>=3.11.0` - Mocking utilities

