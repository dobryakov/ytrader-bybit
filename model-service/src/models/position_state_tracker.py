"""
Position state tracking data model.

Tracks position lifecycle data including entry price, entry time, peak price,
highest unrealized PnL, and last exit signal timestamp for exit strategy evaluation.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class PositionState(BaseModel):
    """
    Position state data model for tracking position lifecycle.

    Used for exit strategy evaluation, particularly for trailing stop
    and time-based exits. Tracks key metrics throughout position lifecycle.
    """

    asset: str = Field(..., description="Trading pair symbol (e.g., 'BTCUSDT')")
    entry_price: float = Field(..., description="Price at which position was entered", gt=0)
    entry_time: datetime = Field(..., description="Timestamp when position was entered")
    peak_price: float = Field(..., description="Highest price reached since entry (for trailing stop)", gt=0)
    highest_unrealized_pnl: float = Field(
        ..., description="Highest unrealized PnL achieved since entry (for trailing stop)"
    )
    last_exit_signal_time: Optional[datetime] = Field(
        default=None, description="Timestamp of last exit signal generated for this position"
    )

    def update_peak_price(self, current_price: float) -> bool:
        """
        Update peak price if current price is higher.

        Args:
            current_price: Current market price

        Returns:
            True if peak price was updated, False otherwise
        """
        if current_price > self.peak_price:
            self.peak_price = current_price
            return True
        return False

    def update_highest_pnl(self, current_pnl: float) -> bool:
        """
        Update highest unrealized PnL if current PnL is higher.

        Args:
            current_pnl: Current unrealized PnL

        Returns:
            True if highest PnL was updated, False otherwise
        """
        if current_pnl > self.highest_unrealized_pnl:
            self.highest_unrealized_pnl = current_pnl
            return True
        return False

    def get_time_held_minutes(self) -> Optional[float]:
        """
        Get time position has been held in minutes.

        Returns:
            Time held in minutes or None if entry_time is not set
        """
        if not self.entry_time:
            return None
        delta = datetime.utcnow() - self.entry_time
        return delta.total_seconds() / 60.0

    def to_dict(self) -> dict:
        """Convert position state to dictionary."""
        return {
            "asset": self.asset,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat() + "Z" if self.entry_time else None,
            "peak_price": self.peak_price,
            "highest_unrealized_pnl": self.highest_unrealized_pnl,
            "last_exit_signal_time": (
                self.last_exit_signal_time.isoformat() + "Z" if self.last_exit_signal_time else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PositionState":
        """
        Create PositionState from dictionary.

        Args:
            data: Dictionary with position state data

        Returns:
            PositionState instance
        """
        # Parse datetime strings if needed
        entry_time = data.get("entry_time")
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))

        last_exit_signal_time = data.get("last_exit_signal_time")
        if isinstance(last_exit_signal_time, str):
            last_exit_signal_time = datetime.fromisoformat(last_exit_signal_time.replace("Z", "+00:00"))
        elif last_exit_signal_time is None:
            last_exit_signal_time = None

        return cls(
            asset=data["asset"],
            entry_price=float(data["entry_price"]),
            entry_time=entry_time,
            peak_price=float(data.get("peak_price", data["entry_price"])),
            highest_unrealized_pnl=float(data.get("highest_unrealized_pnl", 0.0)),
            last_exit_signal_time=last_exit_signal_time,
        )

