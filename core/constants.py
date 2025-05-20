"""
Constants used throughout the QuantumSpectre trading system.

This module defines system-wide constants to ensure consistency
across all components.
"""

from enum import Enum, auto
from typing import Dict

# Exchange types
class ExchangeType(str, Enum):
    """Supported exchange types."""
    SPOT = "spot"
    FUTURES = "futures" 
    MARGIN = "margin"
    OPTIONS = "options"
    DERIV = "deriv"

# Order types
class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    POST_ONLY = "post_only"
    FOK = "fill_or_kill"
    IOC = "immediate_or_cancel"

# Order sides
class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"

# Order status
class OrderStatus(str, Enum):
    """Order status values."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING = "pending"

# Timeframes
class Timeframe(str, Enum):
    """Standard timeframes for candlestick data."""
    M1 = "1m"     # 1 minute
    M3 = "3m"     # 3 minutes
    M5 = "5m"     # 5 minutes
    M15 = "15m"   # 15 minutes
    M30 = "30m"   # 30 minutes
    H1 = "1h"     # 1 hour
    H2 = "2h"     # 2 hours
    H4 = "4h"     # 4 hours
    H6 = "6h"     # 6 hours
    H8 = "8h"     # 8 hours
    H12 = "12h"   # 12 hours
    D1 = "1d"     # 1 day
    D3 = "3d"     # 3 days
    W1 = "1w"     # 1 week
    M1 = "1M"     # 1 month

# Signal types
class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    EXIT = "exit"
    REDUCE = "reduce"
    INCREASE = "increase"

# Signal strength
class SignalStrength(str, Enum):
    """Signal strength indicators."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

# Trading modes
class TradingMode(str, Enum):
    """System trading modes."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"
    OPTIMIZATION = "optimization"
    RESEARCH = "research"

# System status
class SystemStatus(str, Enum):
    """System status indicators."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"

# Risk levels
class RiskLevel(str, Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Asset types
class AssetType(str, Enum):
    """Asset type classifications."""
    CRYPTO = "cryptocurrency"
    FOREX = "forex"
    STOCK = "stock"
    COMMODITY = "commodity"
    INDEX = "index"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"
    CFD = "cfd"

# Exchange mapping - maps exchange names to their normalized form
EXCHANGE_MAPPING: Dict[str, str] = {
    "binance": "binance",
    "binanceus": "binance_us",
    "deriv": "deriv",
    "deribit": "deribit",
    "ftx": "ftx",
    "kucoin": "kucoin",
    "kraken": "kraken",
    "coinbasepro": "coinbase_pro",
    "huobi": "huobi",
    "bitfinex": "bitfinex",
    "bitstamp": "bitstamp",
    "bybit": "bybit",
    "okex": "okex",
    "bitmex": "bitmex",
}

# Event types for the system's event bus
class EventType(str, Enum):
    """System event types."""
    # Market data events
    TICK = "tick"
    TRADE = "trade"
    ORDERBOOK_UPDATE = "orderbook_update"
    CANDLE_CLOSED = "candle_closed"
    MARKET_DEPTH_UPDATE = "market_depth_update"
    FUNDING_RATE_UPDATE = "funding_rate_update"
    
    # Trading events
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"
    
    # Signal events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_EXECUTED = "signal_executed"
    STRATEGY_DECISION = "strategy_decision"
    
    # System events
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    EXCHANGE_CONNECTION_CHANGED = "exchange_connection_changed"
    CONFIG_UPDATED = "config_updated"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    
    # User interface events
    UI_INTERACTION = "ui_interaction"
    UI_STATE_CHANGED = "ui_state_changed"
    NOTIFICATION = "notification"
    ALERT = "alert"
    
    # Intelligence events
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_DETECTED = "pattern_detected"
    PREDICTION_UPDATED = "prediction_updated"
    SENTIMENT_CHANGED = "sentiment_changed"
    
    # Performance events
    PERFORMANCE_METRIC_UPDATED = "performance_metric_updated"
    RESOURCE_USAGE_WARNING = "resource_usage_warning"

# Export Enums as dictionaries for easier access
EXCHANGE_TYPES = {e.name: e.value for e in ExchangeType}
ORDER_TYPES = {e.name: e.value for e in OrderType}
ORDER_SIDES = {e.name: e.value for e in OrderSide}
ORDER_STATUSES = {e.name: e.value for e in OrderStatus}
TIMEFRAMES = {e.name: e.value for e in Timeframe}
SIGNAL_TYPES = {e.name: e.value for e in SignalType}
SIGNAL_STRENGTHS = {e.name: e.value for e in SignalStrength}
TRADING_MODES = {e.name: e.value for e in TradingMode}
SYSTEM_STATUSES = {e.name: e.value for e in SystemStatus}
RISK_LEVELS = {e.name: e.value for e in RiskLevel}
ASSET_TYPES = {e.name: e.value for e in AssetType}
EVENT_TYPES = {e.name: e.value for e in EventType}

# System-wide configuration defaults
DEFAULT_CONFIG = {
    "system": {
        "name": "QuantumSpectre",
        "version": "1.0.0",
        "mode": TRADING_MODES["PAPER"],
        "log_level": "INFO",
        "timezone": "UTC",
    },
    "security": {
        "api_key_encryption": True,
        "jwt_expiration_minutes": 60,
        "password_hash_rounds": 10,
        "require_2fa": False,
    },
    "trading": {
        "default_exchange": "binance",
        "default_asset_type": ASSET_TYPES["CRYPTO"],
        "default_timeframe": TIMEFRAMES["H1"],
        "default_risk_level": RISK_LEVELS["MEDIUM"],
        "max_open_positions": 10,
        "max_open_orders": 50,
        "enable_stop_loss": True,
        "default_stop_loss_pct": 2.0,
        "enable_take_profit": True,
        "default_take_profit_pct": 4.0,
    },
    "execution": {
        "default_order_type": ORDER_TYPES["LIMIT"],
        "slippage_tolerance_pct": 0.1,
        "max_retry_attempts": 3,
        "retry_delay_ms": 500,
        "order_timeout_ms": 5000,
        "use_dynamic_sizing": True,
    },
    "brain_council": {
        "min_confidence_threshold": 0.65,
        "min_agreement_pct": 60.0,
        "default_weights": {
            "momentum": 1.0,
            "mean_reversion": 1.0, 
            "ml_prediction": 1.5,
            "sentiment": 0.8,
            "pattern": 0.9,
            "orderflow": 1.2,
            "volatility": 0.7,
            "onchain": 0.6,
        },
        "adaptive_weighting": True,
        "recalibration_interval_minutes": 60,
    },
    "data": {
        "max_historical_days": 365,
        "default_resample_rule": "1H",
        "cache_expiry_seconds": 3600,
        "missing_data_tolerance_pct": 1.0,
        "use_adjusted_prices": True,
    },
    "performance": {
        "monitor_system_resources": True,
        "resource_warning_threshold_pct": 80,
        "max_memory_usage_gb": 4,
        "record_execution_metrics": True,
        "slow_operation_threshold_ms": 100,
    },
    "ui": {
        "theme": "dark",
        "default_chart_type": "candle",
        "auto_refresh_seconds": 5,
        "max_notifications": 100,
        "enable_sound_alerts": True,
        "show_advanced_features": False,
    },
    "notifications": {
        "email": {
            "enabled": False,
            "critical_only": True,
        },
        "telegram": {
            "enabled": False,
            "level": "INFO",
        },
        "desktop": {
            "enabled": True,
            "level": "WARNING",
        },
    },
}
