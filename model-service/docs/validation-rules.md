# Order Execution Event Validation Rules

**Purpose**: Specify validation rules for order execution events consumed from RabbitMQ queue `order-manager.order_events` per FR-025.

**Date**: 2025-01-27  
**Feature**: Model Service - Trading Decision and ML Training Microservice

## Overview

Order execution events are consumed from the `order-manager.order_events` RabbitMQ queue and must be validated before being processed by the training pipeline. Invalid or corrupted events are logged and skipped to ensure system stability.

## Required Fields

### Top-Level Fields

| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `event_id` | string (UUID) | Unique event identifier | Optional (auto-generated if missing) |
| `order_id` | string | Order identifier from order manager | **Required**, non-empty string |
| `signal_id` | string (UUID) | Original trading signal identifier | **Required**, non-empty string |
| `strategy_id` | string | Trading strategy identifier | **Required**, non-empty string |
| `asset` | string | Trading pair (e.g., 'BTCUSDT') | **Required**, non-empty string, will be uppercased |
| `side` | string | Order side | **Required**, must be 'buy' or 'sell' (case-insensitive) |
| `execution_price` | number | Actual execution price | **Required**, must be positive number (> 0) |
| `execution_quantity` | number | Executed quantity | **Required**, must be positive number (> 0) |
| `execution_fees` | number | Total fees paid | **Required**, must be non-negative number (>= 0) |
| `executed_at` | string (ISO 8601) or datetime | Execution timestamp | **Required**, must be valid ISO 8601 format, cannot be in future |
| `signal_price` | number | Original signal price | **Required**, must be positive number (> 0) |
| `signal_timestamp` | string (ISO 8601) or datetime | Original signal timestamp | **Required**, must be valid ISO 8601 format, cannot be in future |
| `market_conditions` | object | Market data at execution time | **Required**, see Market Conditions below |
| `performance` | object | Trade performance metrics | **Required**, see Performance Metrics below |
| `trace_id` | string | Trace ID for request flow tracking | Optional |

### Market Conditions Object

| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `spread` | number | Bid-ask spread | **Required**, must be non-negative number (>= 0) |
| `volume_24h` | number | 24-hour trading volume | **Required**, must be non-negative number (>= 0) |
| `volatility` | number | Current volatility measure | **Required**, must be non-negative number (>= 0) |

### Performance Metrics Object

| Field | Type | Description | Validation Rules |
|-------|------|-------------|------------------|
| `slippage` | number | Price difference (execution - signal) | **Required**, any number (can be negative) |
| `slippage_percent` | number | Slippage as percentage | **Required**, any number (can be negative) |
| `realized_pnl` | number | Realized profit/loss (if closed) | Optional, any number |
| `return_percent` | number | Return percentage | Optional, any number |

## Format Constraints

### Datetime Format

- **Format**: ISO 8601 with timezone (e.g., `2025-01-27T10:00:00Z` or `2025-01-27T10:00:00+00:00`)
- **Validation**: Must parse correctly, cannot be in the future
- **Timezone**: UTC preferred (Z suffix or +00:00)

### Asset Format

- **Format**: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
- **Validation**: Non-empty string, minimum 3 characters
- **Normalization**: Automatically uppercased

### Side Format

- **Valid Values**: 'buy' or 'sell' (case-insensitive)
- **Normalization**: Automatically lowercased

## Corruption Detection Criteria

### JSON Parsing Errors

- **Symptom**: `json.JSONDecodeError` when parsing message body
- **Action**: Log error with body preview (first 200 characters), skip message
- **Recovery**: Continue processing next message

### Missing Required Fields

- **Symptom**: Required field not present in event data
- **Action**: Log warning with missing field names, skip message
- **Recovery**: Continue processing next message

### Invalid Field Types

- **Symptom**: Field has incorrect type (e.g., string instead of number)
- **Action**: Log warning with field name and value, skip message
- **Recovery**: Continue processing next message

### Invalid Field Values

- **Symptom**: Field value violates constraints (e.g., negative price, invalid side)
- **Action**: Log warning with field name and value, skip message
- **Recovery**: Continue processing next message

### Invalid Datetime Format

- **Symptom**: Datetime string cannot be parsed or is in the future
- **Action**: Log warning with field name and value, skip message
- **Recovery**: Continue processing next message

### Nested Object Validation Failures

- **Symptom**: `market_conditions` or `performance` object missing required fields
- **Action**: Log warning with missing field names, skip message
- **Recovery**: Continue processing next message

## Error Handling Procedures

### Per FR-025 Requirements

1. **Graceful Continuation**: Invalid events are logged and skipped, processing continues with next message
2. **Comprehensive Logging**: All validation failures are logged with:
   - Event data keys (for debugging)
   - Missing field names
   - Invalid field values
   - Error messages
3. **No System Disruption**: Single invalid event does not stop the consumer
4. **Traceability**: All errors include trace_id if available

### Error Logging Levels

- **Warning**: Missing fields, invalid values, format issues (non-critical)
- **Error**: JSON parsing failures, unexpected exceptions (critical but recoverable)

### Example Error Logs

```json
{
  "level": "warning",
  "message": "Execution event missing required fields",
  "missing_fields": ["execution_price", "execution_quantity"],
  "event_data_keys": ["order_id", "signal_id", "asset", "side"]
}
```

```json
{
  "level": "warning",
  "message": "Invalid execution_price",
  "execution_price": -100.0
}
```

```json
{
  "level": "error",
  "message": "Failed to parse execution event JSON",
  "error": "Expecting value: line 1 column 1 (char 0)",
  "body_preview": "invalid json content..."
}
```

## Validation Flow

1. **JSON Parsing**: Parse message body as JSON
2. **Required Fields Check**: Verify all required top-level fields are present
3. **Type Validation**: Verify field types match expected types
4. **Value Range Validation**: Verify numeric fields are within valid ranges
5. **Nested Object Validation**: Validate `market_conditions` and `performance` objects
6. **Datetime Parsing**: Parse and validate datetime strings
7. **Pydantic Validation**: Use `OrderExecutionEvent.from_dict()` for final validation
8. **Success**: Return parsed `OrderExecutionEvent` object
9. **Failure**: Log error/warning and return `None`

## Implementation Location

- **Consumer**: `model-service/src/consumers/execution_event_consumer.py`
- **Model**: `model-service/src/models/execution_event.py`
- **Validation Method**: `ExecutionEventConsumer._validate_and_parse_event()`

## Testing

Validation rules should be tested with:
- Valid events (all required fields, correct types)
- Missing required fields
- Invalid field types
- Invalid field values (negative prices, invalid sides, etc.)
- Invalid datetime formats
- Corrupted JSON
- Missing nested objects (`market_conditions`, `performance`)
- Missing nested object fields

