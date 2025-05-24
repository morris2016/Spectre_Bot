#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Exception Hierarchy

This module provides a comprehensive exception hierarchy for the QuantumSpectre Elite Trading System,
with specialized exceptions for different error scenarios.
"""

class QuantumSpectreError(Exception):
    """Base exception for all QuantumSpectre system errors."""
    pass


class ConfigurationError(QuantumSpectreError):
    """Raised when there is an error in the system configuration."""
    pass


class ServiceError(QuantumSpectreError):
    """Base class for service-related errors."""
    pass


class ServiceStartupError(ServiceError):
    """Raised when a service fails to start."""
    pass


class ServiceShutdownError(ServiceError):
    """Raised when a service fails to shut down properly."""
    pass


class SystemCriticalError(QuantumSpectreError):
    """Raised for critical system errors that require immediate shutdown."""
    pass


class DataError(QuantumSpectreError):
    """Base class for data-related errors."""
    pass


class DataIngestionError(DataError):
    """Raised when there is an error during data ingestion."""
    pass


class DataProcessorError(DataError):
    """Raised when there is an error in a data processor."""
    pass


class ProcessorNotFoundError(DataProcessorError):
    """Raised when a requested data processor is not found."""
    pass


class SourceNotFoundError(DataError):
    """Raised when a requested data source is not found."""
    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass


class FeedError(QuantumSpectreError):
    """Base class for feed-related errors."""
    pass


class FeedConnectionError(FeedError):
    """Raised when a feed connection fails."""
    pass

class BlockchainConnectionError(FeedConnectionError): # New Exception
    """Raised specifically for blockchain node connection errors."""
    pass

class FeedDisconnectedError(FeedError):
    """Raised when a feed unexpectedly disconnects."""
    pass


class FeedTimeoutError(FeedError):
    """Raised when a feed operation times out."""
    pass


class FeedRateLimitError(FeedError):
    """Raised when a feed rate limit is exceeded."""
    pass

class FeedPriorityError(FeedError):
    """Raised when there is an error with feed priority handling."""
    pass


class DatabaseError(QuantumSpectreError):
    """Base class for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when a database connection fails."""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when a database query fails."""
    pass


class RedisError(QuantumSpectreError):
    """Base class for Redis-related errors."""
    pass


class RedisConnectionError(RedisError):
    """Raised when a Redis connection fails."""
    pass


class SecurityError(QuantumSpectreError):
    """Base class for security-related errors."""
    pass


class APIKeyError(SecurityError):
    """Raised when there is an issue with API key validation."""
    pass


class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SecurityError):
    """Raised when authorization fails."""
    pass


class ExecutionError(QuantumSpectreError):
    """Base class for execution engine errors."""
    pass


class OrderError(ExecutionError):
    """Base class for order-related errors."""
    pass


class OrderRejectedError(OrderError):
    """Raised when an order is rejected by an exchange."""
    pass


class OrderTimeoutError(OrderError):
    """Raised when an order times out."""
    pass


class OrderExecutionError(OrderError):
    """Raised when there is an error executing an order."""
    pass


class InsufficientFundsError(OrderError):
    """Raised when there are insufficient funds for an order."""
    pass


class PositionError(ExecutionError):
    """Base class for position-related errors."""
    pass


class PositionExecutionError(PositionError):
    """Raised when there is an error executing a position."""
    pass


class InvalidPositionStateError(PositionError):
    """Raised when a position is in an invalid state."""
    pass


class InsufficientBalanceError(PositionError):
    """Raised when there is insufficient balance for a position."""
    pass


class MarginCallError(PositionError):
    """Raised when a margin call occurs on a position."""
    pass


class PositionLiquidationError(PositionError):
    """Raised when a position is forcibly liquidated."""
    pass


class RiskError(QuantumSpectreError):
    """Base class for risk management errors."""
    pass


class RiskLimitExceededError(RiskError):
    """Raised when a risk limit is exceeded."""
    pass


class BacktestError(QuantumSpectreError):
    """Base class for backtesting errors."""
    pass


class IntelligenceError(QuantumSpectreError):
    """Base class for intelligence system errors."""
    pass

class ModelError(QuantumSpectreError):
    """Base class for ML model errors."""
    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""
    pass


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""
    pass


class StrategyError(QuantumSpectreError):
    """Base class for strategy errors."""
    pass


class SignalGenerationError(StrategyError):
    """Raised when signal generation fails."""
    pass


class MonitoringError(QuantumSpectreError):
    """Base class for monitoring errors."""
    pass


class AlertError(MonitoringError):
    """Raised when alert generation or delivery fails."""
    pass


class ResourceError(QuantumSpectreError):
    """Base class for resource-related errors."""
    pass
class ResourceExhaustionError(ResourceError):
    """Raised when a system resource (e.g., memory, disk, GPU) is exhausted."""
    pass


class RedundancyFailureError(ResourceError):
    """Raised when a redundancy mechanism fails."""
    pass


class TimeoutError(QuantumSpectreError):
    """Raised when an operation times out."""
    pass

class ExchangeError(Exception):
    """Raised when there is a problem related to exchange operations."""
    pass

class RateLimitError(QuantumSpectreError):
    """Raised when a system or API rate limit is exceeded."""
    def __init__(self, message="Rate limit exceeded", retry_after=None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

    def __str__(self):
        if self.retry_after:
            return f"{self.message}. Retry after {self.retry_after} seconds."
        return self.message

class FeedNotFoundError(Exception):
    """Raised when there is a problem related to exchange operations."""
    pass

class FeedInitializationError(Exception):
    """Raised when there is a problem related to exchange operations."""
    pass

# Add these new exception classes to your exceptions.py file

class FeedAuthenticationError(FeedError):
    """Raised when authentication with a feed service fails."""
    pass

class DataSourceError(DataError):
    """Raised when there is an error with a data source."""
    pass
class FeedSubscriptionError(FeedError):
    """Raised when there is an error with feed subscription."""
    pass

class FeedDataError(FeedError):
    """Raised when there is an error with feed data."""
    pass
class ParsingError(DataError):
    """Raised when there is an error parsing data."""
    pass
class DataFeedConnectionError(FeedError):
    """Raised when there is a connection error with a data feed."""
    pass
# Removed duplicate ParsingError definition
# class ParsingError(DataError):
#     """Raised when there is an error parsing data."""
#     pass

# Removed duplicate DataFeedConnectionError definition
# class DataFeedConnectionError(FeedError):
#     """Raised when there is a connection error with a data feed."""
#     pass

class ModelLoadError(QuantumSpectreError):
    """Raised when there is an error loading a machine learning model."""
    pass

class DataParsingError(DataError):
    """Raised when there is an error parsing data."""
    pass

class CredentialError(SecurityError):
    """Raised when there is an error with credentials."""
    pass

class SecurityViolationError(SecurityError):
    """Raised when a security violation is detected."""
    pass

class RegimeDetectionError(QuantumSpectreError):
    """Raised when there is an error in regime detection."""
    pass

class LoopholeDetectionError(IntelligenceError):
    """Raised when there is an error in loophole detection."""
    pass
class AdaptiveLearningError(IntelligenceError):
    """Raised when there is an error in adaptive learning."""
    pass

class IntelligenceServiceError(IntelligenceError):
    """Raised when there is an error in the intelligence service."""
    pass


class NewsFeedError(QuantumSpectreError):
    """Raised when there is an error in the news feed module."""
    pass

class NewsParsingError(DataError):
    """Raised when there is an error parsing news data."""
    pass

class NewsSourceUnavailableError(FeedError):
    """Raised when a news source is unavailable."""
    pass

class FeatureNotFoundError(QuantumSpectreError):
    """Raised when a requested feature is not found."""
    pass

class FeatureCalculationError(QuantumSpectreError):
    """Raised when there is an error calculating a feature."""
    pass

class FeatureServiceError(QuantumSpectreError):
    """Base class for feature service errors."""
    pass

class InvalidTimeframeError(QuantumSpectreError):
    """Raised when an invalid timeframe is provided."""
    pass

class InvalidParameterError(QuantumSpectreError):
    """Raised when an invalid parameter is provided to a function or method."""
    pass
# --- Auto-generated missing exceptions ---

class AlertDeliveryError(AlertError):
    """Raised when alert delivery fails."""
    pass

class AlertConfigurationError(AlertError):
    """Raised when there is an error in alert configuration."""
    pass

class RiskManagerError(RiskError):
    """Raised for errors specific to the Risk Manager service."""
    pass

class PositionSizingError(RiskError):
    """Raised for errors in position sizing calculations."""
    pass

class StopLossError(OrderError):
    """Raised for errors related to stop-loss order management."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a requested ML model is not found."""
    pass

class DashboardError(QuantumSpectreError):
    """Raised for errors related to dashboard operations."""
    pass

class InsufficientLiquidityError(ExecutionError):
    """Raised when there is insufficient liquidity to execute a trade."""
    pass

class ArbitrageOpportunityExpiredError(StrategyError):
    """Raised when an arbitrage opportunity is no longer valid."""
    pass

class DrawdownLimitExceededException(RiskLimitExceededError):
    """Raised when a drawdown limit is exceeded."""
    pass


class MaxDrawdownExceededError(RiskLimitExceededError):
    """Raised when the maximum allowed drawdown is exceeded."""
    pass

class RiskManagementException(RiskError):
    """General exception for risk management issues."""
    pass

class ModelVersionError(ModelError):
    """Raised for issues related to model versioning."""
    pass

class LogAnalysisError(MonitoringError):
    """Raised for errors during log analysis."""
    pass

class InsufficientDataError(DataError):
    """Raised when there is insufficient data for an operation."""
    pass

class EncodingError(DataError):
    """Raised for errors during data encoding or decoding."""
    pass

class MetricCollectionError(MonitoringError):
    """Raised for errors during metric collection."""
    pass

class ServiceConnectionError(ServiceError):
    """Raised for errors connecting to an internal or external service."""
    pass

class DataStoreError(DatabaseError):
    """Raised for errors interacting with a generic data store."""
    pass

class StorageError(DatabaseError):
    """Raised for errors related to data storage operations."""
    pass

class DataIntegrityError(DataError):
    """Raised when data integrity is compromised."""
    pass

class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity is compromised."""
    pass

class InvalidPopulationError(AdaptiveLearningError):
    """Raised when a genetic algorithm population is invalid."""
    pass

class ConvergenceError(AdaptiveLearningError):
    """Raised when an algorithm fails to converge."""
    pass

class GeneticOperationError(AdaptiveLearningError):
    """Raised when a genetic algorithm operation fails."""
    pass

class DatabaseTimeoutError(DatabaseError):
    """Raised when a database operation times out."""
    pass

class InvalidDataError(DataValidationError):
    """Raised when provided data is invalid for the operation."""
    pass

class TrainingError(ModelTrainingError):
    """General error during a training process."""
    pass

class ArbitrageValidationError(StrategyError):
    """Raised for validation errors in arbitrage strategies."""
    pass

class PredictionError(ModelPredictionError):
    """General error during a prediction process."""
    pass

class AnalysisError(QuantumSpectreError):
    """Raised for general errors during analysis tasks."""
    pass

class RecoveryStrategyError(StrategyError):
    """Raised for errors in recovery strategies."""
    pass

class OptimizationError(QuantumSpectreError):
    """Raised for errors during optimization processes."""
    pass

class CorrelationCalculationError(FeatureCalculationError):
    """Raised for errors during correlation calculations."""
    pass

class SamplingError(DataError):
    """Raised for errors during data sampling."""
    pass

class DataQualityError(DataValidationError):
    """Raised for issues related to data quality."""
    pass

class HardwareError(ResourceError):
    """Raised for errors related to hardware interaction or failure."""
    pass

class EnsembleConfigError(ModelError):
    """Raised for configuration errors in ensemble models."""
    pass

class ServiceUnavailableError(ServiceError):
    """Raised when a required service is unavailable."""
    pass

class ModelNotSupportedError(ModelError):
    """Raised when a model type or version is not supported."""
    pass

class InvalidFeatureFormatError(FeatureCalculationError):
    """Raised when feature data has an invalid format."""
    pass

class StrategyExecutionError(StrategyError):
    """Raised for errors during strategy execution."""
    pass

class AdaptationError(StrategyError):
    """Raised for errors during adaptive learning or strategy adaptation."""
    pass

class EvolutionError(AdaptiveLearningError):
    """Raised for errors during genetic algorithm evolution."""
    pass

class InferenceError(ModelPredictionError):
    """Raised for errors during model inference."""
    pass

class CircuitBreakerTrippedException(SystemCriticalError):
    """Raised when a circuit breaker is tripped."""
    pass

class PatternRecognitionError(FeatureCalculationError):
    """Raised for errors during pattern recognition."""
    pass

class PatternDetectionError(PatternRecognitionError):
    """Raised when there is an error detecting patterns in data."""
    pass

class PatternNotFoundError(FeatureNotFoundError):
    """Raised when a specific pattern is not found."""
    pass

class DataAlignmentError(DataError):
    """Raised for errors aligning data from different sources or timeframes."""
    pass

class RESTClientError(FeedError):
    """Raised for errors in a REST API client."""
    pass

class RequestError(FeedError):
    """Raised for general errors making external requests."""
    pass

class DataTransformationError(DataProcessorError):
    """Raised for errors during data transformation."""
    pass

class CapitalManagementError(RiskError):
    """Raised for errors in capital management."""
    pass

class MicrostructureAnalysisError(FeatureCalculationError):
    """Raised for errors in market microstructure analysis."""
    pass

class MigrationError(DatabaseError):
    """Raised for errors during database migrations."""
    pass

class ModelValidationError(ModelError):
    """Raised for validation errors related to models."""
    pass

class WebSocketError(FeedConnectionError):
    """Raised for errors related to WebSocket connections."""
    pass

class SubscriptionError(FeedError):
    """Raised for errors subscribing to data feeds or topics."""
    pass

class DataFetchError(FeedError):
    """Raised for errors fetching data from external sources."""
    pass

class BacktestConfigError(BacktestError):
    """Raised for configuration errors in the backtester."""
    pass

class BacktestDataError(BacktestError):
    """Raised for data-related errors in the backtester."""
    pass

class BacktestStrategyError(BacktestError):
    """Raised for strategy-related errors in the backtester."""
    pass

class AssetCouncilError(QuantumSpectreError):
    """Raised for errors in the Asset Council."""
    pass

class BrainNotFoundError(StrategyError):
    """Raised when a required Strategy Brain is not found."""
    pass

class CouncilError(QuantumSpectreError):
    """Raised for general errors in a Council."""
    pass

class DecisionError(StrategyError):
    """Raised for errors in decision-making processes."""
    pass

class PerformanceTrackerError(MonitoringError):
    """Raised for errors in the performance tracker."""
    pass

class InvalidStrategyError(StrategyError):
    """Raised when an invalid strategy is encountered or configured."""
    pass

class SimulationError(BacktestError):
    """Raised for errors during backtest simulations."""
    pass

class SentimentAnalysisError(FeatureCalculationError):
    """Raised for errors during sentiment analysis."""
    pass

class RegimeCouncilError(QuantumSpectreError):
    """Raised for errors in the Regime Council."""
    pass

class ReportGenerationError(QuantumSpectreError):
    """Raised for errors during report generation."""
    pass

class OperationNotPermittedError(SecurityError):
    """Raised when an operation is not permitted for the current user/context."""
    pass

class BacktestScenarioError(BacktestError):
    """Raised for errors related to backtesting scenarios."""
    pass

class DataNotFoundError(DataError):
    """Raised when expected data is not found."""
    pass

class VoiceAdvisorError(QuantumSpectreError):
    """Raised for errors in the Voice Advisor system."""
    pass

class TTSEngineError(ServiceError):
    """Raised for errors in the Text-to-Speech engine."""
    pass

class VotingError(QuantumSpectreError):
    """Raised for errors in voting systems."""
    pass

class PermissionDeniedError(AuthorizationError):
    """Raised when an action is denied due to insufficient permissions."""
    pass

class WeightingSystemError(QuantumSpectreError):
    """Raised for errors in weighting systems."""
    pass

class FeedCoordinationError(FeedError):
    """Raised for errors in coordinating multiple data feeds."""
    pass

class DataInsufficientError(InsufficientDataError):
    """Raised when data is present but insufficient in quantity for an operation."""
    pass

class InvalidAssetError(DataValidationError):
    """Raised when an invalid asset is specified."""
    pass

class InvalidTimeRangeError(DataValidationError):
    """Raised when an invalid time range is specified."""
    pass

class InvalidFeatureDefinitionError(FeatureCalculationError):
    """Raised when a feature definition is invalid or malformed."""
    pass

class FeatureTimeoutError(FeatureServiceError):
    """Raised when a feature calculation operation times out."""
    pass
# --- End of auto-generated missing exceptions ---

class InvalidOrderError(OrderError):
    """Raised when an order is invalid or has invalid parameters."""
    pass

class OrderCancellationError(OrderError):
    """Raised when there is an error cancelling an order."""
    pass

class SlippageExceededError(OrderError):
    """Raised when slippage exceeds the allowed threshold."""
    pass

class NetworkError(QuantumSpectreError):
    """Raised when there is a network-related error."""
    pass


class TimeSeriesConnectionError(Exception):
    """
    Exception raised when a connection to the time series database fails
    """
    def __init__(self, message="Failed to connect to time series database", details=None):
        self.message = message
        self.details = details
        super().__init__(self.message)
        
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message

class TimeSeriesQueryError(Exception):
    """
    Exception raised when a query to the time series database fails
    """
    def __init__(self, message="Failed to execute time series query", query=None, details=None):
        self.message = message
        self.query = query
        self.details = details
        super().__init__(self.message)
        
    def __str__(self):
        result = self.message
        if self.query:
            result += f"\nQuery: {self.query}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

class TimeSeriesDataError(Exception):
    """
    Exception raised when there's an issue with time series data
    (missing data, corrupted data, etc.)
    """
    def __init__(self, message="Time series data error", data_info=None, details=None):
        self.message = message
        self.data_info = data_info
        self.details = details
        super().__init__(self.message)
        
    def __str__(self):
        result = self.message
        if self.data_info:
            result += f"\nData info: {self.data_info}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

class TimeSeriesConfigError(Exception):
    """
    Exception raised when there's a configuration error with the time series database
    (invalid settings, connection parameters, etc.)
    """
    def __init__(self, message="Time series configuration error", config_key=None, details=None):
        self.message = message
        self.config_key = config_key
        self.details = details
        super().__init__(self.message)
        
    def __str__(self):
        result = self.message
        if self.config_key:
            result += f"\nConfig key: {self.config_key}"
        if self.details:
            result += f"\nDetails: {self.details}"
        return result

__all__ = [
    'QuantumSpectreError', 'ConfigurationError', 'ServiceError', 'ServiceStartupError',
    'ServiceShutdownError', 'SystemCriticalError', 'DataError', 'DataIngestionError',
    'DataProcessorError', 'ProcessorNotFoundError', 'SourceNotFoundError', 'DataValidationError',
    'FeedError', 'FeedConnectionError', 'BlockchainConnectionError', 'FeedDisconnectedError',
    'FeedTimeoutError', 'FeedRateLimitError', 'DatabaseError', 'DatabaseConnectionError',
    'DatabaseQueryError', 'RedisError', 'RedisConnectionError', 'SecurityError',
    'APIKeyError', 'AuthenticationError', 'AuthorizationError', 'ExecutionError',
    'OrderError', 'OrderRejectedError', 'OrderTimeoutError', 'InsufficientFundsError',
    'InvalidOrderError', 'OrderCancellationError', 'SlippageExceededError', 'NetworkError',
    'RiskError', 'RiskLimitExceededError', 'BacktestError', 'ModelError',
    'ModelTrainingError', 'ModelPredictionError', 'StrategyError', 'SignalGenerationError',
    'MonitoringError', 'AlertError', 'ResourceError', 'ResourceExhaustionError',
    'TimeoutError', 'ExchangeError', 'RateLimitError', 'FeedNotFoundError',
    'FeedInitializationError', 'FeedAuthenticationError', 'DataSourceError',
    'FeedSubscriptionError', 'FeedDataError', 'ParsingError', 'DataFeedConnectionError',
    'ModelLoadError', 'DataParsingError', 'CredentialError', 'SecurityViolationError',
    'RegimeDetectionError', 'NewsFeedError', 'NewsParsingError', 'NewsSourceUnavailableError',
    'FeatureNotFoundError', 'FeatureCalculationError', 'FeatureServiceError',
    'InvalidTimeframeError', 'InvalidParameterError', 'AlertDeliveryError',
    'AlertConfigurationError', 'RiskManagerError', 'PositionSizingError', 'StopLossError',
    'PositionError', 'PositionExecutionError', 'InvalidPositionStateError', 'InsufficientBalanceError',
    'MarginCallError', 'PositionLiquidationError',
    'ModelNotFoundError', 'DashboardError', 'InsufficientLiquidityError',
    'ArbitrageOpportunityExpiredError', 'DrawdownLimitExceededException',
    'MaxDrawdownExceededError',
    'RiskManagementException', 'ModelVersionError', 'LogAnalysisError',
    'InsufficientDataError', 'EncodingError', 'MetricCollectionError',
    'ServiceConnectionError', 'DataStoreError', 'InvalidDataError', 'TrainingError',
    'ArbitrageValidationError', 'PredictionError', 'AnalysisError', 'RecoveryStrategyError',
    'OptimizationError', 'CorrelationCalculationError', 'SamplingError', 'DataQualityError',
    'HardwareError', 'EnsembleConfigError', 'ServiceUnavailableError', 'ModelNotSupportedError',
    'InvalidFeatureFormatError', 'StrategyExecutionError', 'AdaptationError', 'InferenceError',
    'CircuitBreakerTrippedException', 'PatternRecognitionError', 'PatternNotFoundError',
    'DataAlignmentError', 'RESTClientError', 'RequestError', 'DataTransformationError',
    'CapitalManagementError', 'MicrostructureAnalysisError', 'MigrationError',
    'ModelValidationError', 'WebSocketError', 'SubscriptionError', 'DataFetchError',
    'BacktestConfigError', 'BacktestDataError', 'BacktestStrategyError', 'AssetCouncilError',
    'BrainNotFoundError', 'CouncilError', 'DecisionError', 'PerformanceTrackerError',
    'InvalidStrategyError', 'SimulationError', 'SentimentAnalysisError', 'RegimeCouncilError',
    'ReportGenerationError', 'OperationNotPermittedError', 'BacktestScenarioError',
    'DataNotFoundError', 'VoiceAdvisorError', 'TTSEngineError', 'VotingError',
    'PermissionDeniedError', 'WeightingSystemError', 'FeedCoordinationError',
    'DataInsufficientError', 'InvalidAssetError', 'InvalidTimeRangeError',
    'InvalidFeatureDefinitionError', 'FeatureTimeoutError',
    'TimeSeriesConnectionError', 'TimeSeriesQueryError', 'TimeSeriesDataError',
    'TimeSeriesConfigError'
]
