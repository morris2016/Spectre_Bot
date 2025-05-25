#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
System Constants and Enumerations

This module provides system-wide constants, enumerations, and configuration defaults
used throughout the QuantumSpectre Elite Trading System.
"""

import os
import enum
from pathlib import Path


# ======================================
# System Core Constants
# ======================================

VERSION = "1.0.0"
CONFIG_SCHEMA_VERSION = 1
SYSTEM_NAME = "QuantumSpectre Elite Trading System"
AUTHOR = "QuantumSpectre Team"
LICENSE = "MIT"

# Environment settings
ENV_PRODUCTION = "production"
ENV_DEVELOPMENT = "development"
ENV_TESTING = "testing"

# Default configuration paths
DEFAULT_CONFIG_PATH = os.environ.get(
    "QUANTUM_SPECTRE_CONFIG", str(Path.home() / ".quantumspectre" / "config.yml")
)
DEFAULT_DATA_DIR = os.environ.get(
    "QUANTUM_SPECTRE_DATA", str(Path.home() / ".quantumspectre" / "data")
)
STORAGE_ROOT_PATH = os.environ.get(
    "QUANTUM_SPECTRE_STORAGE", str(Path.home() / ".quantumspectre" / "storage")
)
DEFAULT_LOG_DIR = os.environ.get(
    "QUANTUM_SPECTRE_LOGS", str(Path.home() / ".quantumspectre" / "logs")
)
DEFAULT_MODEL_DIR = os.environ.get(
    "QUANTUM_SPECTRE_MODELS", str(Path.home() / ".quantumspectre" / "models")
)


# ======================================
# System Architecture Configuration
# ======================================

SERVICE_NAMES = {
    "data_ingest": "Data Ingestion Service",
    "data_feeds": "Data Feeds Service",
    "feature_service": "Feature Service",
    "intelligence": "Intelligence Service",
    "ml_models": "ML Models Service",
    "strategy_brains": "Strategy Brains Service",
    "brain_council": "Brain Council Service",
    "execution_engine": "Execution Engine Service",
    "risk_manager": "Risk Manager Service",
    "backtester": "Backtester Service",
    "monitoring": "Monitoring Service",
    "api_gateway": "API Gateway Service",
    "ui": "UI Service",
}

SERVICE_DEPENDENCIES = {
    "data_ingest": [],
    "data_feeds": ["data_ingest"],
    "feature_service": ["data_feeds"],
    "intelligence": ["feature_service"],
    "ml_models": ["feature_service"],
    "strategy_brains": ["intelligence", "ml_models"],
    "brain_council": ["strategy_brains"],
    "execution_engine": ["brain_council", "risk_manager"],
    "risk_manager": ["data_feeds"],
    "backtester": ["feature_service", "strategy_brains", "risk_manager"],
    "monitoring": [],
    "api_gateway": ["brain_council", "execution_engine", "monitoring"],
    "ui": ["api_gateway"],
}

SERVICE_STARTUP_ORDER = [
    "data_ingest", "data_feeds", "feature_service", "intelligence", "ml_models",
    "strategy_brains", "risk_manager", "brain_council", "execution_engine",
    "backtester", "monitoring", "api_gateway", "ui"
]

DATA_INGEST_METRICS_PREFIX = "data_ingest"


# ======================================
# Resource Management
# ======================================

DEFAULT_THREAD_POOL_SIZE = 10
MAX_THREAD_POOL_SIZE = 100
DEFAULT_PROCESS_POOL_SIZE = 4
MAX_PROCESS_POOL_SIZE = 16
MARKET_DATA_MAX_WORKERS = 16

MEMORY_WARNING_THRESHOLD = 0.85
MEMORY_CRITICAL_THRESHOLD = 0.95

LOG_LEVELS = {
    "CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0,
}
DEFAULT_LOG_LEVEL = "INFO"


# ======================================
# Network Configuration
# ======================================

# API rate limits (requests per minute)
API_RATE_LIMIT_DEFAULT = 100
API_RATE_LIMIT_TRADING = 20
API_RATE_LIMIT_AUTH = 10

# WebSocket configuration
WEBSOCKET_MAX_CONNECTIONS = 10000
WEBSOCKET_PING_INTERVAL = 30
WEBSOCKET_PING_TIMEOUT = 10
WEBSOCKET_CLOSE_TIMEOUT = 5

# HTTP configuration
HTTP_TIMEOUT_DEFAULT = 10
HTTP_TIMEOUT_FEED = 30
HTTP_TIMEOUT_LONG = 120
HTTP_MAX_RETRIES = 3
HTTP_RETRY_BACKOFF = 2.0


# ======================================
# Security Configuration
# ======================================

TOKEN_EXPIRY_ACCESS = 3600  # 1 hour
TOKEN_EXPIRY_REFRESH = 2592000  # 30 days
PASSWORD_MIN_LENGTH = 10
PASSWORD_HASH_ALGORITHM = "pbkdf2_sha256"
PASSWORD_SALT_LENGTH = 32
PASSWORD_HASH_ITERATIONS = 200000


# ======================================
# Database Configuration
# ======================================

DATABASE_POOL_MIN_SIZE = 5
DATABASE_POOL_MAX_SIZE = 20
DATABASE_MAX_QUERIES = 50000
DATABASE_CONNECTION_TIMEOUT = 60
DATABASE_COMMAND_TIMEOUT = 60


# ======================================
# Cache Configuration
# ======================================

CACHE_DEFAULT_TTL = 300
CACHE_LONG_TTL = 3600
CACHE_VERY_LONG_TTL = 86400


# ======================================
# Exchange and Trading Enums
# ======================================

class Exchange(enum.Enum):
    """Supported trading exchanges."""
    BINANCE = "binance"
    DERIV = "deriv"
    BACKTEST = "backtest"


class AssetClass(enum.Enum):
    """Supported asset classes."""
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCK = "stock"
    INDEX = "index"
    COMMODITY = "commodity"
    SYNTHETIC = "synthetic"


class Timeframe(enum.Enum):
    """Supported trading timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"
    MN1 = "1M"


class OrderType(enum.Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT_MARKET = "take_profit_market"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(enum.Enum):
    """Order sides for trading."""
    BUY = "buy"
    SELL = "sell"


class PositionSide(enum.Enum):
    """Position sides for trading."""
    LONG = "long"
    SHORT = "short"


class OrderStatus(enum.Enum):
    """Order status states."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    PENDING_CANCEL = "pending_cancel"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(enum.Enum):
    """Position status states."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED = "closed"
    FAILED = "failed"


class TimeInForce(enum.Enum):
    """Time in force options."""
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date


class SignalDirection(enum.Enum):
    """Trade signal directions."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SignalStrength(enum.Enum):
    """Signal strength levels."""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5


class MarketRegime(enum.Enum):
    """Market condition types."""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


class RiskLevel(enum.Enum):
    """Risk assessment levels."""
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5


# ======================================
# Trading Platform Configuration
# ======================================

SUPPORTED_PLATFORMS = ["deriv", "binance"]

SUPPORTED_ASSETS = {
    "binance": [
        "BTC", "ETH", "USDT", "BNB", "ADA", "XRP", "DOGE", "SOL", "DOT", "LTC",
        "BCH", "LINK", "MATIC", "ATOM", "AVAX", "TRX", "XLM", "NEAR", "FIL",
        "EOS", "AAVE", "UNI", "SAND", "MANA", "SHIB", "ALGO", "FTM", "ETC",
        "ZIL", "VET", "THETA", "XTZ", "GRT", "CHZ", "ENJ", "BAT", "ZRX",
        "1INCH", "COMP", "SNX", "YFI", "CRV", "KSM", "DASH", "OMG", "QTUM",
        "ICX", "ONT", "WAVES", "LRC", "BTT", "HOT", "NANO", "SC", "ZEN",
        "STMX", "ANKR", "CELR", "CVC", "DENT", "IOST", "KAVA", "NKN", "OCEAN",
        "RLC", "STORJ", "TOMO", "WRX", "XEM", "ZEC"
    ],
    "deriv": ["BTC", "ETH", "LTC", "USDC", "USDT", "XRP"]
}

ASSET_TYPES = [
    "crypto", "forex", "stocks", "indices", "commodities", "futures", "options",
]

STRATEGY_TYPES = [
    "trend_following", "mean_reversion", "breakout", "momentum", "statistical_arbitrage",
    "market_making", "sentiment_based", "machine_learning", "pattern_recognition",
    "volatility_based", "order_flow", "market_structure", "multi_timeframe",
    "adaptive", "ensemble", "reinforcement_learning", "regime_based",
]

EXECUTION_MODES = [
    "live", "paper", "backtest", "simulation", "optimization", "stress_test",
]

SLIPPAGE_MODELS = [
    "fixed", "percentage", "volume_based", "volatility_based", "orderbook_based", "impact_based",
]


# ======================================
# Risk Management Configuration
# ======================================

class RiskControlMethod(enum.Enum):
    """Risk control methods."""
    FIXED_STOP_LOSS = "fixed_stop_loss"
    TRAILING_STOP = "trailing_stop"
    ATR_STOP = "atr_stop"
    VOLATILITY_STOP = "volatility_stop"
    SUPPORT_RESISTANCE_STOP = "support_resistance_stop"
    TIME_STOP = "time_stop"
    EQUITY_STOP = "equity_stop"
    DRAWDOWN_STOP = "drawdown_stop"


class PositionSizingMethod(enum.Enum):
    """Position sizing methods."""
    FIXED_SIZE = "fixed_size"
    FIXED_VALUE = "fixed_value"
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_CRITERION = "kelly_criterion"
    OPTIMAL_F = "optimal_f"
    RISK_PARITY = "risk_parity"


class ExecutionAlgorithm(enum.Enum):
    """Trade execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    PEG = "peg"
    SNIPER = "sniper"
    ADAPTIVE = "adaptive"


# Default risk parameters
DEFAULT_RISK_PERCENT_PER_TRADE = 1.0
DEFAULT_MAX_OPEN_TRADES = 5
DEFAULT_MAX_CORRELATED_TRADES = 2
DEFAULT_MAX_DRAWDOWN_PERCENT = 20.0
DEFAULT_PROFIT_FACTOR_THRESHOLD = 1.5
DEFAULT_WIN_RATE_THRESHOLD = 65.0
DEFAULT_TRAILING_STOP_ACTIVATION = 1.0
DEFAULT_KELLY_FRACTION = 0.5
DEFAULT_GROWTH_FACTOR = 1.05
DEFAULT_FIXED_STOP_PERCENTAGE = 2.0
DEFAULT_MIN_STOP_DISTANCE = 0.005
DEFAULT_STOP_LOSS_MULTIPLIER = 1.5
DEFAULT_TAKE_PROFIT_MULTIPLIER = 2.0

# Position management
PARTIAL_CLOSE_LEVELS = [0.25, 0.5, 0.75]
POSITION_SIZE_PRECISION = 4
MAX_LEVERAGE_BINANCE = 125
MAX_LEVERAGE_DERIV = 100


# ======================================
# Technical Analysis Configuration
# ======================================

FIBONACCI_RATIOS = {
    "0": 0.0, "23.6": 0.236, "38.2": 0.382, "50": 0.5, "61.8": 0.618,
    "76.4": 0.764, "78.6": 0.786, "100": 1.0, "127.2": 1.272, "138.2": 1.382,
    "150": 1.5, "161.8": 1.618, "200": 2.0, "223.6": 2.236, "261.8": 2.618,
    "361.8": 3.618, "423.6": 4.236
}

SMA_PERIODS = [10, 20, 50, 100, 200]
EMA_PERIODS = [9, 12, 26, 50, 200]
RSI_PERIODS = [7, 14, 21]
MACD_PARAMS = {"FAST": 12, "SLOW": 26, "SIGNAL": 9}
BOLLINGER_BANDS_PARAMS = {"PERIOD": 20, "STD_DEV": 2}
STOCHASTIC_PARAMS = {"K_PERIOD": 14, "K_SLOWING": 3, "D_PERIOD": 3}


# ======================================
# Machine Learning Configuration
# ======================================

class ModelType(enum.Enum):
    """Machine learning model types."""
    LINEAR_REGRESSION = "linear_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    SVM = "svm"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    CNN = "cnn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


class ScalingMethod(enum.Enum):
    """Feature scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    MAXABS = "maxabs"
    NONE = "none"


# ======================================
# Notification Configuration
# ======================================

class NotificationType(enum.Enum):
    """Notification types."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TAKE_PROFIT_HIT = "take_profit_hit"
    NEW_SIGNAL = "new_signal"
    PATTERN_DETECTED = "pattern_detected"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"


class NotificationPriority(enum.Enum):
    """Notification priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class NotificationChannel(enum.Enum):
    """Notification channels."""
    INTERNAL = "internal"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    WEBHOOK = "webhook"
    TELEGRAM = "telegram"
    SLACK = "slack"
    DISCORD = "discord"


# ======================================
# Deriv-specific Configuration
# ======================================

MAX_RECONNECT_ATTEMPTS = 5
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 60.0
DEFAULT_SUBSCRIPTION_TIMEOUT = 30.0
DEFAULT_PING_INTERVAL = 30.0
MARKET_ORDER_BOOK_DEPTH = 10
DERIV_PRICE_REFRESH_RATE = 1.0

DERIV_ENDPOINTS = {
    "websocket": "wss://ws.binaryws.com/websockets/v3",
    "oauth": "https://oauth.deriv.com",
    "api": "https://api.deriv.com",
}

DERIV_ASSET_CLASSES = {
    "forex": "forex",
    "indices": "indices",
    "commodities": "commodities",
    "synthetic": "synthetic_index",
}


# ======================================
# Feature Engineering
# ======================================

FEATURE_PRIORITY_LEVELS = ["high", "normal", "low"]
DEFAULT_FEATURE_PARAMS = {}


# ======================================
# Helper Lists for Runtime Use
# ======================================

EXCHANGE_TYPES = [ex.value for ex in Exchange]
TIME_FRAMES = [tf.value for tf in Timeframe]
ORDER_TYPES = [ot.value for ot in OrderType]
ORDER_SIDES = [side.value for side in OrderSide]
POSITION_SIDES = [ps.value for ps in PositionSide]
ORDER_STATUSES = [ps.value for ps in OrderStatus]
POSITION_STATUSES = [ps.value for ps in PositionStatus]
SIGNAL_STRENGTHS = [ss.value for ss in SignalStrength]


# ======================================
# Export Interface
# ======================================

__all__ = [
    # System constants
    "VERSION", "CONFIG_SCHEMA_VERSION", "SYSTEM_NAME", "AUTHOR", "LICENSE",
    "ENV_PRODUCTION", "ENV_DEVELOPMENT", "ENV_TESTING",
    "DEFAULT_CONFIG_PATH", "DEFAULT_DATA_DIR", "STORAGE_ROOT_PATH",
    "DEFAULT_LOG_DIR", "DEFAULT_MODEL_DIR",
    
    # Service configuration
    "SERVICE_NAMES", "SERVICE_DEPENDENCIES", "SERVICE_STARTUP_ORDER",
    "DATA_INGEST_METRICS_PREFIX",
    
    # Resource management
    "DEFAULT_THREAD_POOL_SIZE", "MAX_THREAD_POOL_SIZE", "DEFAULT_PROCESS_POOL_SIZE",
    "MAX_PROCESS_POOL_SIZE", "MARKET_DATA_MAX_WORKERS",
    "MEMORY_WARNING_THRESHOLD", "MEMORY_CRITICAL_THRESHOLD",
    "LOG_LEVELS", "DEFAULT_LOG_LEVEL",
    
    # Network configuration
    "API_RATE_LIMIT_DEFAULT", "API_RATE_LIMIT_TRADING", "API_RATE_LIMIT_AUTH",
    "WEBSOCKET_MAX_CONNECTIONS", "WEBSOCKET_PING_INTERVAL", "WEBSOCKET_PING_TIMEOUT",
    "WEBSOCKET_CLOSE_TIMEOUT", "HTTP_TIMEOUT_DEFAULT", "HTTP_TIMEOUT_FEED",
    "HTTP_TIMEOUT_LONG", "HTTP_MAX_RETRIES", "HTTP_RETRY_BACKOFF",
    
    # Security configuration
    "TOKEN_EXPIRY_ACCESS", "TOKEN_EXPIRY_REFRESH", "PASSWORD_MIN_LENGTH",
    "PASSWORD_HASH_ALGORITHM", "PASSWORD_SALT_LENGTH", "PASSWORD_HASH_ITERATIONS",
    
    # Database configuration
    "DATABASE_POOL_MIN_SIZE", "DATABASE_POOL_MAX_SIZE", "DATABASE_MAX_QUERIES",
    "DATABASE_CONNECTION_TIMEOUT", "DATABASE_COMMAND_TIMEOUT",
    
    # Cache configuration
    "CACHE_DEFAULT_TTL", "CACHE_LONG_TTL", "CACHE_VERY_LONG_TTL",
    
    # Trading enums
    "Exchange", "AssetClass", "Timeframe", "OrderType", "OrderSide", "PositionSide",
    "OrderStatus", "PositionStatus", "TimeInForce", "SignalDirection", "SignalStrength",
    "MarketRegime", "RiskLevel",
    
    # Trading platform configuration
    "SUPPORTED_PLATFORMS", "SUPPORTED_ASSETS", "ASSET_TYPES", "STRATEGY_TYPES",
    "EXECUTION_MODES", "SLIPPAGE_MODELS",
    
    # Risk management
    "RiskControlMethod", "PositionSizingMethod", "ExecutionAlgorithm",
    "DEFAULT_RISK_PERCENT_PER_TRADE", "DEFAULT_MAX_OPEN_TRADES", "DEFAULT_MAX_CORRELATED_TRADES",
    "DEFAULT_MAX_DRAWDOWN_PERCENT", "DEFAULT_PROFIT_FACTOR_THRESHOLD", "DEFAULT_WIN_RATE_THRESHOLD",
    "DEFAULT_TRAILING_STOP_ACTIVATION", "DEFAULT_KELLY_FRACTION", "DEFAULT_GROWTH_FACTOR",
    "DEFAULT_FIXED_STOP_PERCENTAGE", "DEFAULT_MIN_STOP_DISTANCE",
    "DEFAULT_STOP_LOSS_MULTIPLIER", "DEFAULT_TAKE_PROFIT_MULTIPLIER",
    "PARTIAL_CLOSE_LEVELS", "POSITION_SIZE_PRECISION", "MAX_LEVERAGE_BINANCE", "MAX_LEVERAGE_DERIV",
    
    # Technical analysis
    "FIBONACCI_RATIOS", "SMA_PERIODS", "EMA_PERIODS", "RSI_PERIODS",
    "MACD_PARAMS", "BOLLINGER_BANDS_PARAMS", "STOCHASTIC_PARAMS",
    
    # Machine learning
    "ModelType", "ScalingMethod",
    
    # Notifications
    "NotificationType", "NotificationPriority", "NotificationChannel",
    
    # Deriv configuration
    "MAX_RECONNECT_ATTEMPTS", "INITIAL_RECONNECT_DELAY", "MAX_RECONNECT_DELAY",
    "DEFAULT_SUBSCRIPTION_TIMEOUT", "DEFAULT_PING_INTERVAL", "MARKET_ORDER_BOOK_DEPTH",
    "DERIV_PRICE_REFRESH_RATE", "DERIV_ENDPOINTS", "DERIV_ASSET_CLASSES",
    
    # Feature engineering
    "FEATURE_PRIORITY_LEVELS", "DEFAULT_FEATURE_PARAMS",
    
    # Helper lists
    "EXCHANGE_TYPES", "TIME_FRAMES", "ORDER_TYPES", "ORDER_SIDES", "POSITION_SIDES",
    "ORDER_STATUSES", "POSITION_STATUSES", "SIGNAL_STRENGTHS",
]
