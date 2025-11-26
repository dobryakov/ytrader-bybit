"""
Database repository base class.

Provides common database operations and transaction management for repositories.
"""

from typing import Optional, List, Dict, Any, TypeVar, Generic
import asyncpg
from abc import ABC, abstractmethod

from .connection import db_pool
from ..config.exceptions import DatabaseQueryError, DatabaseError
from ..config.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Base class for database repositories."""

    def __init__(self):
        """Initialize the repository."""
        self._table_name: Optional[str] = None

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Return the table name for this repository."""
        pass

    async def _execute(self, query: str, *args) -> str:
        """
        Execute a query and return the result.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Query result

        Raises:
            DatabaseQueryError: If execution fails
        """
        try:
            return await db_pool.execute(query, *args)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error("Repository query execution failed", query=query, error=str(e), exc_info=True)
            raise DatabaseQueryError(f"Query execution failed: {e}") from e

    async def _fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """
        Execute a query and fetch all results.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of records

        Raises:
            DatabaseQueryError: If execution fails
        """
        try:
            return await db_pool.fetch(query, *args)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error("Repository query fetch failed", query=query, error=str(e), exc_info=True)
            raise DatabaseQueryError(f"Query fetch failed: {e}") from e

    async def _fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """
        Execute a query and fetch a single row.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single record or None

        Raises:
            DatabaseQueryError: If execution fails
        """
        try:
            return await db_pool.fetchrow(query, *args)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error("Repository query fetchrow failed", query=query, error=str(e), exc_info=True)
            raise DatabaseQueryError(f"Query fetchrow failed: {e}") from e

    async def _fetchval(self, query: str, *args) -> Optional[Any]:
        """
        Execute a query and fetch a single value.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single value or None

        Raises:
            DatabaseQueryError: If execution fails
        """
        try:
            return await db_pool.fetchval(query, *args)
        except DatabaseError:
            raise
        except Exception as e:
            logger.error("Repository query fetchval failed", query=query, error=str(e), exc_info=True)
            raise DatabaseQueryError(f"Query fetchval failed: {e}") from e

    async def _transaction(self):
        """
        Get a transaction context manager.

        Returns:
            Transaction context manager

        Example:
            async with await self._transaction() as conn:
                await conn.execute("INSERT INTO ...")
        """
        pool = await db_pool.get_pool()
        return pool.transaction()

    def _record_to_dict(self, record: asyncpg.Record) -> Dict[str, Any]:
        """
        Convert an asyncpg record to a dictionary.

        Args:
            record: Database record

        Returns:
            Dictionary representation
        """
        return dict(record)

    def _records_to_dicts(self, records: List[asyncpg.Record]) -> List[Dict[str, Any]]:
        """
        Convert a list of asyncpg records to dictionaries.

        Args:
            records: List of database records

        Returns:
            List of dictionary representations
        """
        return [self._record_to_dict(record) for record in records]

