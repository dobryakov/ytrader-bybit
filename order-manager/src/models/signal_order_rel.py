"""Signal-Order Relationship model for tracking signal-to-order mappings."""

from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class SignalOrderRelationship(BaseModel):
    """Signal-Order Relationship entity tracking mappings between signals and orders.

    Tracks which orders were created from which signals, enabling relationship
    analysis and position building across multiple signals.
    """

    id: UUID = Field(default_factory=uuid4, description="Unique relationship identifier")
    signal_id: UUID = Field(..., description="Trading signal identifier")
    order_id: UUID = Field(..., description="Order identifier (references orders.id)")
    relationship_type: str = Field(
        ...,
        description="Relationship type: 'one_to_one', 'one_to_many', 'many_to_one'",
        max_length=20,
    )
    execution_sequence: Optional[int] = Field(None, description="Sequence number for ordering (for 1:N relationships)", ge=1)
    allocation_amount: Optional[Decimal] = Field(None, description="Amount allocated from signal to this order (for partial allocations)", gt=0)
    allocation_quantity: Optional[Decimal] = Field(None, description="Quantity allocated from signal to this order", gt=0)
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When relationship was created")

    @field_validator("relationship_type")
    @classmethod
    def validate_relationship_type(cls, v: str) -> str:
        """Validate relationship type."""
        valid_types = {"one_to_one", "one_to_many", "many_to_one"}
        v_lower = v.lower()
        if v_lower not in valid_types:
            raise ValueError(f"Relationship type must be one of {valid_types}")
        return v_lower

    @field_validator("allocation_amount", "allocation_quantity")
    @classmethod
    def validate_allocation(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Validate allocation values are positive if set."""
        if v is not None and v <= 0:
            raise ValueError("Allocation values must be positive if set")
        return v

    def to_dict(self) -> dict:
        """Convert relationship to dictionary for database operations."""
        return {
            "id": str(self.id),
            "signal_id": str(self.signal_id),
            "order_id": str(self.order_id),
            "relationship_type": self.relationship_type,
            "execution_sequence": self.execution_sequence,
            "allocation_amount": str(self.allocation_amount) if self.allocation_amount is not None else None,
            "allocation_quantity": str(self.allocation_quantity) if self.allocation_quantity is not None else None,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SignalOrderRelationship":
        """Create relationship from dictionary (e.g., from database)."""
        # Convert string UUIDs to UUID objects
        for field in ["id", "signal_id", "order_id"]:
            if field in data and isinstance(data[field], str):
                data[field] = UUID(data[field])

        # Convert string decimals to Decimal
        for field in ["allocation_amount", "allocation_quantity"]:
            if field in data and data[field] is not None:
                if isinstance(data[field], str):
                    data[field] = Decimal(data[field])

        return cls(**data)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            UUID: str,
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }

