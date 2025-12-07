"""
Metadata storage using asyncpg connection pool.
"""
import asyncpg
from typing import Optional, AsyncContextManager, Any
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from src.config import config
from src.logging import get_logger

logger = get_logger(__name__)


class MetadataStorage:
    """PostgreSQL metadata storage using asyncpg connection pool."""
    
    def __init__(self, pool: Optional[asyncpg.Pool] = None):
        """
        Initialize metadata storage.
        
        Args:
            pool: Optional asyncpg pool (for testing). If None, creates new pool.
        """
        self._pool = pool
        self._own_pool = pool is None
    
    async def initialize(self) -> None:
        """Initialize database connection pool."""
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                host=config.postgres_host,
                port=config.postgres_port,
                database=config.postgres_db,
                user=config.postgres_user,
                password=config.postgres_password,
                min_size=2,
                max_size=10,
            )
            logger.info("Database connection pool initialized")
    
    async def close(self) -> None:
        """Close database connection pool."""
        if self._pool and self._own_pool:
            await self._pool.close()
            logger.info("Database connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[asyncpg.Connection]:
        """
        Get a database connection from the pool.
        
        Yields:
            asyncpg.Connection: Database connection
        """
        if self._pool is None:
            await self.initialize()
        
        async with self._pool.acquire() as connection:
            yield connection
    
    @asynccontextmanager
    async def transaction(self) -> AsyncContextManager[asyncpg.Connection]:
        """
        Get a database connection with transaction.
        
        Yields:
            asyncpg.Connection: Database connection in transaction
        """
        async with self.get_connection() as connection:
            async with connection.transaction():
                yield connection
    
    @property
    def pool(self) -> Optional[asyncpg.Pool]:
        """Get the connection pool (for testing)."""
        return self._pool
    
    @staticmethod
    def _normalize_datetime(dt: Any) -> Optional[datetime]:
        """
        Normalize datetime to timezone-aware UTC datetime.
        
        asyncpg requires datetime objects (not strings) for timestamp parameters.
        However, asyncpg compares datetime objects internally during encoding,
        which causes errors when mixing timezone-aware and timezone-naive datetimes.
        
        This function ensures ALL datetime objects are timezone-aware UTC,
        preventing asyncpg from encountering mixed types during encoding.
        
        CRITICAL: Always creates a NEW datetime object to avoid any reference issues
        that might cause asyncpg to see the original naive datetime.
        
        Args:
            dt: datetime object (timezone-aware or timezone-naive)
            
        Returns:
            timezone-aware datetime in UTC, or None if input is None
        """
        if dt is None:
            return None
        
        if not isinstance(dt, datetime):
            return dt
        
        # CRITICAL: Always create a NEW datetime object to avoid reference issues
        # asyncpg may compare datetime objects by reference, so we must create new objects
        if dt.tzinfo is None:
            # If timezone-naive, assume UTC and create NEW timezone-aware datetime
            # Use datetime constructor to ensure it's a completely new object
            normalized = datetime(
                dt.year, dt.month, dt.day,
                dt.hour, dt.minute, dt.second, dt.microsecond,
                tzinfo=timezone.utc
            )
        else:
            # If timezone-aware, convert to UTC and create NEW datetime object
            utc_dt = dt.astimezone(timezone.utc)
            normalized = datetime(
                utc_dt.year, utc_dt.month, utc_dt.day,
                utc_dt.hour, utc_dt.minute, utc_dt.second, utc_dt.microsecond,
                tzinfo=timezone.utc
            )
        
        # Verify it's actually timezone-aware UTC
        assert normalized.tzinfo is not None, f"Failed to normalize datetime: {dt}"
        assert normalized.tzinfo == timezone.utc, f"Datetime not in UTC: {normalized}"
        
        return normalized
    
    @staticmethod
    def _normalize_datetime_in_dict(data: Any) -> Any:
        """
        Recursively normalize all datetime objects in a dictionary or nested structure.
        
        This ensures that ALL datetime objects (including nested ones in walk_forward_config
        or target_config) are timezone-aware UTC, preventing asyncpg from encountering
        mixed timezone-aware and timezone-naive datetimes during encoding.
        
        Args:
            data: Dictionary, list, or other data structure that may contain datetime objects
            
        Returns:
            Data structure with all datetime objects normalized to timezone-aware UTC
        """
        if data is None:
            return None
        
        if isinstance(data, datetime):
            return MetadataStorage._normalize_datetime(data)
        
        if isinstance(data, dict):
            return {
                key: MetadataStorage._normalize_datetime_in_dict(value)
                for key, value in data.items()
            }
        
        if isinstance(data, (list, tuple)):
            normalized_list = [
                MetadataStorage._normalize_datetime_in_dict(item)
                for item in data
            ]
            return type(data)(normalized_list) if isinstance(data, tuple) else normalized_list
        
        return data
    
    @staticmethod
    def _convert_datetime_to_iso_string(data: Any) -> Any:
        """
        Recursively convert all datetime objects to ISO format strings.
        
        This is needed for JSON serialization, as datetime objects cannot be
        directly serialized to JSON. All datetime objects are converted to ISO
        format strings with timezone information.
        
        Args:
            data: Dictionary, list, or other data structure that may contain datetime objects
            
        Returns:
            Data structure with all datetime objects converted to ISO format strings
        """
        if data is None:
            return None
        
        if isinstance(data, datetime):
            # Convert to ISO format string with timezone
            return data.isoformat()
        
        if isinstance(data, dict):
            return {
                key: MetadataStorage._convert_datetime_to_iso_string(value)
                for key, value in data.items()
            }
        
        if isinstance(data, (list, tuple)):
            converted_list = [
                MetadataStorage._convert_datetime_to_iso_string(item)
                for item in data
            ]
            return type(data)(converted_list) if isinstance(data, tuple) else converted_list
        
        return data
    
    async def create_dataset(self, dataset_data: dict) -> str:
        """
        Create a new dataset record.
        
        Args:
            dataset_data: Dataset data dictionary
            
        Returns:
            Dataset ID (UUID as string)
        """
        # CRITICAL: Create a copy of dataset_data to avoid any reference issues
        # asyncpg may inspect the original dict, so we must ensure it doesn't contain
        # naive datetime objects that could be compared with aware ones
        dataset_data = dict(dataset_data)  # Shallow copy
        # Normalize all datetime objects to timezone-aware UTC before passing to query
        # This ensures ALL datetime objects (including nested ones) are timezone-aware,
        # preventing asyncpg from encountering mixed timezone-aware and timezone-naive
        # datetimes during encoding. asyncpg compares datetime objects internally, and
        # mixing types causes "can't subtract offset-naive and offset-aware datetimes" error.
        # 
        # CRITICAL: We MUST normalize BEFORE creating any connection or transaction,
        # because asyncpg may inspect the parameters during connection setup.
        train_period_start = self._normalize_datetime(dataset_data.get("train_period_start"))
        train_period_end = self._normalize_datetime(dataset_data.get("train_period_end"))
        validation_period_start = self._normalize_datetime(dataset_data.get("validation_period_start"))
        validation_period_end = self._normalize_datetime(dataset_data.get("validation_period_end"))
        test_period_start = self._normalize_datetime(dataset_data.get("test_period_start"))
        test_period_end = self._normalize_datetime(dataset_data.get("test_period_end"))
        
        # CRITICAL: Verify normalization immediately after calling it
        # This helps catch any issues before they reach asyncpg
        for name, dt in [
            ("train_period_start", train_period_start),
            ("train_period_end", train_period_end),
            ("validation_period_start", validation_period_start),
            ("validation_period_end", validation_period_end),
            ("test_period_start", test_period_start),
            ("test_period_end", test_period_end),
        ]:
            if dt is not None and isinstance(dt, datetime):
                if dt.tzinfo is None:
                    logger.error(f"CRITICAL: {name} is still naive after normalization: {dt}")
                    # Force re-normalization
                    if name == "train_period_start":
                        train_period_start = self._normalize_datetime(dt)
                    elif name == "train_period_end":
                        train_period_end = self._normalize_datetime(dt)
                    elif name == "validation_period_start":
                        validation_period_start = self._normalize_datetime(dt)
                    elif name == "validation_period_end":
                        validation_period_end = self._normalize_datetime(dt)
                    elif name == "test_period_start":
                        test_period_start = self._normalize_datetime(dt)
                    elif name == "test_period_end":
                        test_period_end = self._normalize_datetime(dt)
                elif dt.tzinfo != timezone.utc:
                    logger.warning(f"{name} is not UTC, converting: {dt}")
                    # Force re-normalization to UTC
                    if name == "train_period_start":
                        train_period_start = self._normalize_datetime(dt)
                    elif name == "train_period_end":
                        train_period_end = self._normalize_datetime(dt)
                    elif name == "validation_period_start":
                        validation_period_start = self._normalize_datetime(dt)
                    elif name == "validation_period_end":
                        validation_period_end = self._normalize_datetime(dt)
                    elif name == "test_period_start":
                        test_period_start = self._normalize_datetime(dt)
                    elif name == "test_period_end":
                        test_period_end = self._normalize_datetime(dt)
        
        # Final normalization pass - ensure ALL datetime objects are new timezone-aware UTC objects
        # This is critical because asyncpg may compare datetime objects by reference or value,
        # and we must ensure no naive datetime objects exist anywhere
        datetime_params = [
            self._normalize_datetime(train_period_start),
            self._normalize_datetime(train_period_end),
            self._normalize_datetime(validation_period_start),
            self._normalize_datetime(validation_period_end),
            self._normalize_datetime(test_period_start),
            self._normalize_datetime(test_period_end),
        ]
        
        # Final verification - all must be timezone-aware UTC or None
        for i, dt in enumerate(datetime_params):
            if dt is not None:
                if not isinstance(dt, datetime):
                    logger.error(f"CRITICAL: Parameter {i} is not a datetime: {type(dt)}")
                elif dt.tzinfo is None:
                    logger.error(f"CRITICAL: Parameter {i} is still naive: {dt}")
                    # Emergency fix - create new UTC datetime
                    datetime_params[i] = datetime(
                        dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond,
                        tzinfo=timezone.utc
                    )
                elif dt.tzinfo != timezone.utc:
                    logger.warning(f"Parameter {i} is not UTC, converting: {dt}")
                    utc_dt = dt.astimezone(timezone.utc)
                    datetime_params[i] = datetime(
                        utc_dt.year, utc_dt.month, utc_dt.day,
                        utc_dt.hour, utc_dt.minute, utc_dt.second, utc_dt.microsecond,
                        tzinfo=timezone.utc
                    )
        
        # Normalize datetime objects in nested structures (walk_forward_config, target_config)
        # These may contain datetime objects that need to be normalized
        walk_forward_config = self._normalize_datetime_in_dict(dataset_data.get("walk_forward_config"))
        target_config = self._normalize_datetime_in_dict(dataset_data.get("target_config"))
        
        # Final verification: ensure ALL datetime objects are timezone-aware UTC
        # This is critical - asyncpg compares ALL datetime objects in the query,
        # and mixing naive/aware causes the error
        all_datetime_values = [
            datetime_params[0], datetime_params[1], datetime_params[2],
            datetime_params[3], datetime_params[4], datetime_params[5],
        ]
        # Also check nested structures for datetime objects
        def check_all_datetimes(obj, path=""):
            """Recursively check all datetime objects in nested structures."""
            if isinstance(obj, datetime):
                if obj.tzinfo is None:
                    logger.error(f"CRITICAL: Found naive datetime at {path}: {obj}")
                    return False
                elif obj.tzinfo != timezone.utc:
                    logger.warning(f"Found non-UTC datetime at {path}: {obj}, converting to UTC")
                    return False
            elif isinstance(obj, dict):
                for key, value in obj.items():
                    if not check_all_datetimes(value, f"{path}.{key}"):
                        return False
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    if not check_all_datetimes(item, f"{path}[{i}]"):
                        return False
            return True
        
        if not all(check_all_datetimes(dt, f"datetime_params[{i}]") for i, dt in enumerate(all_datetime_values)):
            logger.error("CRITICAL: Some datetime objects are not properly normalized!")
        if walk_forward_config and not check_all_datetimes(walk_forward_config, "walk_forward_config"):
            logger.error("CRITICAL: walk_forward_config contains improperly normalized datetime objects!")
        if target_config and not check_all_datetimes(target_config, "target_config"):
            logger.error("CRITICAL: target_config contains improperly normalized datetime objects!")
        
        # Log datetime parameters at debug level (only if needed for troubleshooting)
        # Note: structlog doesn't support isEnabledFor, so we just log at debug level
        logger.debug("Datetime parameters before query:")
        for i, dt in enumerate(datetime_params):
            if dt is not None:
                logger.debug(f"  Param {i}: {dt} (type: {type(dt).__name__}, tzinfo: {dt.tzinfo}, naive: {dt.tzinfo is None})")
            else:
                logger.debug(f"  Param {i}: None")
        
        async with self.transaction() as conn:
            # Prepare query parameters - ensure all datetime are new objects
            query_params = [
                dataset_data["symbol"],
                dataset_data["status"],
                dataset_data["split_strategy"],
                datetime_params[0],  # train_period_start
                datetime_params[1],  # train_period_end
                datetime_params[2],  # validation_period_start
                datetime_params[3],  # validation_period_end
                datetime_params[4],  # test_period_start
                datetime_params[5],  # test_period_end
                walk_forward_config,
                target_config,
                dataset_data["feature_registry_version"],
                dataset_data.get("output_format", "parquet"),
            ]
            
            # Final check of all datetime in query_params
            for i, param in enumerate(query_params):
                if isinstance(param, datetime):
                    if param.tzinfo is None:
                        logger.error(f"CRITICAL: Query param {i} is naive datetime: {param}")
                        # Emergency fix
                        query_params[i] = datetime(
                            param.year, param.month, param.day,
                            param.hour, param.minute, param.second, param.microsecond,
                            tzinfo=timezone.utc
                        )
                    elif param.tzinfo != timezone.utc:
                        logger.warning(f"Query param {i} is not UTC: {param}")
                        utc_dt = param.astimezone(timezone.utc)
                        query_params[i] = datetime(
                            utc_dt.year, utc_dt.month, utc_dt.day,
                            utc_dt.hour, utc_dt.minute, utc_dt.second, utc_dt.microsecond,
                            tzinfo=timezone.utc
                        )
            
            # CRITICAL FIX: asyncpg compares ALL datetime objects in the query internally.
            # The error "can't subtract offset-naive and offset-aware datetimes" occurs
            # when asyncpg tries to encode datetime objects and encounters mixed types.
            # 
            # Even though all our datetime are timezone-aware UTC, asyncpg may still
            # compare them with other datetime objects (e.g., from connection pool,
            # cached prepared statements, or internal asyncpg datetime).
            #
            # Solution: Create completely new datetime objects right before the query,
            # ensuring they are all timezone-aware UTC and have no references to original objects.
            # Also, we'll use a fresh connection to avoid any cached prepared statements.
            
            # Create brand new datetime objects right before query execution
            # This ensures no references to original objects and all are timezone-aware UTC
            final_datetime_params = []
            for dt in datetime_params:
                if dt is None:
                    final_datetime_params.append(None)
                else:
                    # Create completely new datetime object with explicit UTC timezone
                    final_datetime_params.append(datetime(
                        dt.year, dt.month, dt.day,
                        dt.hour, dt.minute, dt.second, dt.microsecond,
                        tzinfo=timezone.utc
                    ))
            
            # Log final datetime parameters at debug level (only if needed for troubleshooting)
            # Note: structlog doesn't support isEnabledFor, so we just log at debug level
            logger.debug("Final datetime parameters (new objects created):")
            for i, dt in enumerate(final_datetime_params):
                if dt is not None:
                    logger.debug(f"  Param {i}: {dt} (type: {type(dt).__name__}, tzinfo: {dt.tzinfo}, naive: {dt.tzinfo is None})")
                else:
                    logger.debug(f"  Param {i}: None")
            
            # Build final query parameters with new datetime objects
            # For JSONB fields, asyncpg should accept dict, but some versions may require JSON strings
            # Let's convert dict to JSON string to be safe
            import json
            
            # Convert JSONB fields to JSON strings if they are dicts
            # First, convert any datetime objects in the dicts to ISO strings for JSON serialization
            walk_forward_config_final = query_params[9]
            if isinstance(walk_forward_config_final, dict):
                # Convert datetime objects to ISO strings before JSON serialization
                walk_forward_config_final = self._convert_datetime_to_iso_string(walk_forward_config_final)
                walk_forward_config_final = json.dumps(walk_forward_config_final)
            elif walk_forward_config_final is not None and not isinstance(walk_forward_config_final, str):
                # Convert datetime objects to ISO strings before JSON serialization
                walk_forward_config_final = self._convert_datetime_to_iso_string(walk_forward_config_final)
                walk_forward_config_final = json.dumps(walk_forward_config_final)
            
            target_config_final = query_params[10]
            if isinstance(target_config_final, dict):
                # Convert datetime objects to ISO strings before JSON serialization
                target_config_final = self._convert_datetime_to_iso_string(target_config_final)
                target_config_final = json.dumps(target_config_final)
            elif target_config_final is not None and not isinstance(target_config_final, str):
                # Convert datetime objects to ISO strings before JSON serialization
                target_config_final = self._convert_datetime_to_iso_string(target_config_final)
                target_config_final = json.dumps(target_config_final)
            
            final_params = [
                query_params[0],  # symbol
                query_params[1],  # status
                query_params[2],  # split_strategy
                final_datetime_params[0],  # train_period_start
                final_datetime_params[1],  # train_period_end
                final_datetime_params[2],  # validation_period_start
                final_datetime_params[3],  # validation_period_end
                final_datetime_params[4],  # test_period_start
                final_datetime_params[5],  # test_period_end
                walk_forward_config_final,  # walk_forward_config (JSON string or None)
                target_config_final,  # target_config (JSON string)
                query_params[11],  # feature_registry_version
                query_params[12],  # output_format
            ]
            
            # Execute query with fresh datetime objects
            # CRITICAL: asyncpg may compare datetime objects with internal values during encoding.
            # To avoid this, we'll use execute() with a fresh prepared statement,
            # or we can try to ensure all datetime are created in the same way.
            #
            # Another approach: use format() to insert values directly (less safe but avoids asyncpg's datetime comparison)
            # However, this is not recommended due to SQL injection risks.
            #
            # Best approach: Ensure all datetime are timezone-aware UTC and created fresh,
            # and use execute() which may handle prepared statements differently.
            
            # CRITICAL FIX: asyncpg has a bug where it compares datetime objects during encoding
            # and throws "can't subtract offset-naive and offset-aware datetimes" even when
            # all datetime objects are timezone-aware. This happens because asyncpg may
            # compare our datetime objects with internal datetime values (e.g., from connection
            # pool, cached prepared statements, or internal asyncpg datetime objects).
            #
            # Solution: Use execute() with explicit type casting in SQL to force PostgreSQL
            # to handle the conversion, bypassing asyncpg's datetime comparison logic.
            #
            # This workaround ensures that PostgreSQL receives the datetime values and handles
            # the type conversion, avoiding asyncpg's internal datetime comparison.
            
            try:
                # First, try the normal approach
                result = await conn.fetchrow(
                    """
                    INSERT INTO datasets (
                        symbol, status, split_strategy,
                        train_period_start, train_period_end,
                        validation_period_start, validation_period_end,
                        test_period_start, test_period_end,
                        walk_forward_config, target_config,
                        feature_registry_version, output_format
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                    """,
                    *final_params
                )
                dataset_id = result['id']
            except Exception as e:
                error_msg = str(e)
                if "can't subtract offset-naive and offset-aware" in error_msg:
                    logger.warning(f"asyncpg datetime comparison error detected, using workaround: {e}")
                    logger.debug("Using explicit type casting to bypass asyncpg's datetime comparison")
                    
                    # Workaround: Use execute() with explicit type casting in SQL
                    # This forces PostgreSQL to handle conversion, bypassing asyncpg's datetime comparison
                    # We use execute() and then fetch the ID separately to avoid the comparison issue
                    await conn.execute(
                        """
                        INSERT INTO datasets (
                            symbol, status, split_strategy,
                            train_period_start, train_period_end,
                            validation_period_start, validation_period_end,
                            test_period_start, test_period_end,
                            walk_forward_config, target_config,
                            feature_registry_version, output_format
                        ) VALUES (
                            $1, $2, $3,
                            $4::timestamptz, $5::timestamptz,
                            $6::timestamptz, $7::timestamptz,
                            $8::timestamptz, $9::timestamptz,
                            $10, $11, $12, $13
                        )
                        """,
                        *final_params
                    )
                    
                    # Get the ID separately using a simple query
                    result = await conn.fetchrow(
                        """
                        SELECT id FROM datasets 
                        WHERE symbol = $1 
                        AND status = $2 
                        AND train_period_start = $3
                        ORDER BY created_at DESC 
                        LIMIT 1
                        """,
                        final_params[0],  # symbol
                        final_params[1],  # status
                        final_datetime_params[0]  # train_period_start for uniqueness
                    )
                    if result is None:
                        raise ValueError("Failed to retrieve dataset ID after insert")
                    dataset_id = result['id']
                    logger.debug(f"Successfully inserted dataset using datetime workaround, ID: {dataset_id}")
                elif "expected str, got dict" in error_msg or "jsonb" in error_msg.lower():
                    # JSONB field error - convert dict to JSON string
                    logger.warning(f"JSONB field error detected, converting dict to JSON string: {e}")
                    import json
                    
                    # Convert JSONB fields to JSON strings
                    final_params_jsonb = final_params.copy()
                    if isinstance(final_params_jsonb[9], dict):
                        final_params_jsonb[9] = json.dumps(final_params_jsonb[9])
                    elif final_params_jsonb[9] is None:
                        final_params_jsonb[9] = None
                    else:
                        final_params_jsonb[9] = json.dumps(final_params_jsonb[9]) if final_params_jsonb[9] else None
                    
                    if isinstance(final_params_jsonb[10], dict):
                        final_params_jsonb[10] = json.dumps(final_params_jsonb[10])
                    elif final_params_jsonb[10] is None:
                        final_params_jsonb[10] = None
                    else:
                        final_params_jsonb[10] = json.dumps(final_params_jsonb[10]) if final_params_jsonb[10] else None
                    
                    # Retry with JSON strings
                    result = await conn.fetchrow(
                        """
                        INSERT INTO datasets (
                            symbol, status, split_strategy,
                            train_period_start, train_period_end,
                            validation_period_start, validation_period_end,
                            test_period_start, test_period_end,
                            walk_forward_config, target_config,
                            feature_registry_version, output_format
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                        RETURNING id
                        """,
                        *final_params_jsonb
                    )
                    dataset_id = result['id']
                    logger.debug(f"Successfully inserted dataset using JSON string workaround, ID: {dataset_id}")
                else:
                    # Re-raise if it's a different error
                    raise
            return str(dataset_id)
    
    async def get_dataset(self, dataset_id: str) -> Optional[dict]:
        """
        Get dataset by ID.
        
        Args:
            dataset_id: Dataset ID (UUID string)
            
        Returns:
            Dataset data dictionary or None if not found
        """
        async with self.get_connection() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM datasets WHERE id = $1
                """,
                dataset_id,
            )
            if row is None:
                return None
            # Normalize all datetime objects in the result to timezone-aware UTC
            # This prevents mixing timezone-aware and timezone-naive datetimes
            # when using the dataset data in other parts of the code
            dataset_dict = dict(row)
            return self._normalize_datetime_in_dict(dataset_dict)
    
    async def update_dataset(
        self,
        dataset_id: str,
        updates: dict,
    ) -> None:
        """
        Update dataset record.
        
        Args:
            dataset_id: Dataset ID (UUID string)
            updates: Dictionary of fields to update
        """
        if not updates:
            return
        
        # Build dynamic UPDATE query
        set_clauses = []
        values = []
        param_num = 1
        import json
        
        for key, value in updates.items():
            # Normalize datetime values recursively
            # This ensures ALL datetime objects (including nested ones) are timezone-aware UTC,
            # preventing asyncpg from encountering mixed timezone-aware and timezone-naive datetimes
            if isinstance(value, datetime):
                # Normalize datetime and create new object to avoid reference issues
                normalized_dt = self._normalize_datetime(value)
                # Create completely new datetime object
                final_dt = datetime(
                    normalized_dt.year, normalized_dt.month, normalized_dt.day,
                    normalized_dt.hour, normalized_dt.minute, normalized_dt.second, normalized_dt.microsecond,
                    tzinfo=timezone.utc
                )
                set_clauses.append(f"{key} = ${param_num}::timestamptz")
                values.append(final_dt)
            elif key.endswith("_at") or key.endswith("_start") or key.endswith("_end"):
                # Potential datetime field
                if isinstance(value, datetime):
                    normalized_dt = self._normalize_datetime(value)
                    final_dt = datetime(
                        normalized_dt.year, normalized_dt.month, normalized_dt.day,
                        normalized_dt.hour, normalized_dt.minute, normalized_dt.second, normalized_dt.microsecond,
                        tzinfo=timezone.utc
                    )
                    set_clauses.append(f"{key} = ${param_num}::timestamptz")
                    values.append(final_dt)
                else:
                    set_clauses.append(f"{key} = ${param_num}")
                    values.append(value)
            elif isinstance(value, dict):
                # JSONB field - normalize datetime objects and convert to JSON string
                normalized = self._normalize_datetime_in_dict(value)
                # Convert datetime objects to ISO strings for JSON serialization
                json_ready = self._convert_datetime_to_iso_string(normalized)
                set_clauses.append(f"{key} = ${param_num}")
                values.append(json.dumps(json_ready))
            elif isinstance(value, list):
                # List that may contain datetime objects
                normalized = self._normalize_datetime_in_dict(value)
                # Convert datetime objects to ISO strings if needed for JSON serialization
                json_ready = self._convert_datetime_to_iso_string(normalized)
                set_clauses.append(f"{key} = ${param_num}")
                values.append(json.dumps(json_ready))
            else:
                set_clauses.append(f"{key} = ${param_num}")
                values.append(value)
            param_num += 1
        
        values.append(dataset_id)  # WHERE clause parameter
        
        async with self.transaction() as conn:
            try:
                await conn.execute(
                    f"""
                    UPDATE datasets
                    SET {', '.join(set_clauses)}
                    WHERE id = ${param_num}
                    """,
                    *values,
                )
            except Exception as e:
                error_msg = str(e)
                if "can't subtract offset-naive and offset-aware" in error_msg:
                    logger.warning(f"asyncpg datetime comparison error in update_dataset, retrying with explicit casting: {e}")
                    # Retry with explicit type casting for all datetime fields
                    set_clauses_retry = []
                    values_retry = []
                    param_num_retry = 1
                    
                    for key, value in updates.items():
                        if isinstance(value, datetime) or (key.endswith("_at") or key.endswith("_start") or key.endswith("_end")):
                            if isinstance(value, datetime):
                                normalized_dt = self._normalize_datetime(value)
                                final_dt = datetime(
                                    normalized_dt.year, normalized_dt.month, normalized_dt.day,
                                    normalized_dt.hour, normalized_dt.minute, normalized_dt.second, normalized_dt.microsecond,
                                    tzinfo=timezone.utc
                                )
                                set_clauses_retry.append(f"{key} = ${param_num_retry}::timestamptz")
                                values_retry.append(final_dt)
                            else:
                                set_clauses_retry.append(f"{key} = ${param_num_retry}")
                                values_retry.append(value)
                        elif isinstance(value, dict):
                            normalized = self._normalize_datetime_in_dict(value)
                            json_ready = self._convert_datetime_to_iso_string(normalized)
                            set_clauses_retry.append(f"{key} = ${param_num_retry}")
                            values_retry.append(json.dumps(json_ready))
                        elif isinstance(value, list):
                            normalized = self._normalize_datetime_in_dict(value)
                            json_ready = self._convert_datetime_to_iso_string(normalized)
                            set_clauses_retry.append(f"{key} = ${param_num_retry}")
                            values_retry.append(json.dumps(json_ready))
                        else:
                            set_clauses_retry.append(f"{key} = ${param_num_retry}")
                            values_retry.append(value)
                        param_num_retry += 1
                    
                    values_retry.append(dataset_id)
                    await conn.execute(
                        f"""
                        UPDATE datasets
                        SET {', '.join(set_clauses_retry)}
                        WHERE id = ${param_num_retry}
                        """,
                        *values_retry,
                    )
                else:
                    raise
    
    async def list_datasets(
        self,
        symbol: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> list:
        """
        List datasets with optional filters.
        
        Args:
            symbol: Filter by symbol (optional)
            status: Filter by status (optional)
            limit: Maximum number of results
            
        Returns:
            List of dataset dictionaries
        """
        async with self.get_connection() as conn:
            query = "SELECT * FROM datasets WHERE 1=1"
            params = []
            param_num = 1
            
            if symbol:
                query += f" AND symbol = ${param_num}"
                params.append(symbol)
                param_num += 1
            
            if status:
                query += f" AND status = ${param_num}"
                params.append(status)
                param_num += 1
            
            query += f" ORDER BY created_at DESC LIMIT ${param_num}"
            params.append(limit)
            
            rows = await conn.fetch(query, *params)
            # Normalize all datetime objects in results to timezone-aware UTC
            # This prevents mixing timezone-aware and timezone-naive datetimes
            return [self._normalize_datetime_in_dict(dict(row)) for row in rows]
    
    async def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete dataset record from database.
        
        Args:
            dataset_id: Dataset ID (UUID string)
            
        Returns:
            True if dataset was deleted, False if not found
        """
        async with self.transaction() as conn:
            result = await conn.execute(
                """
                DELETE FROM datasets WHERE id = $1
                """,
                dataset_id,
            )
            # result is the number of rows affected
            deleted = result == "DELETE 1"
            if deleted:
                logger.info("dataset_deleted", dataset_id=dataset_id)
            else:
                logger.warning("dataset_not_found_for_deletion", dataset_id=dataset_id)
            return deleted

