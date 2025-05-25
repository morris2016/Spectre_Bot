#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Feature Service - Main Application

This module implements the main application for the feature service,
which calculates technical indicators, pattern recognition features,
and other trading signals.
"""

import os
import sys
import time
import json
import asyncio
import logging
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import signal
import numpy as np
import pandas as pd

# Internal imports
from config import Config
from common.logger import get_logger
from common.utils import (
    setup_event_loop, cancel_all_tasks, 
    time_execution, generate_uuid
)
from common.metrics import MetricsCollector
from common.exceptions import (
    FeatureCalculationError, FeatureServiceError,
    InvalidTimeframeError, InvalidParameterError
)
from common.redis_client import RedisClient
from common.db_client import get_db_client, DatabaseClient
from common.async_utils import TaskGroup, PeriodicTask, Throttler

# Feature service imports
from feature_service import (
    init_feature_service, shutdown_feature_service,
    get_feature_calculator, get_feature_transformer,
    get_processor, get_all_feature_calculators
)
from feature_service.processor import FeatureProcessor
from feature_service.feature_extraction import FeatureExtractor
from feature_service.multi_timeframe import MultiTimeframeAnalyzer

# Service name and logger initialization
SERVICE_NAME = "feature_service"
logger = get_logger(SERVICE_NAME)

class FeatureService:
    """
    Main service for feature calculation and pattern recognition.
    
    This service is responsible for:
    1. Calculating technical indicators and other trading features
    2. Recognizing patterns across multiple timeframes
    3. Providing feature data to the intelligence systems
    4. Managing feature calculation pipelines
    """
    
    def __init__(self, config: Config):
        """
        Initialize the feature service.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.processor = None
        self.extractor = None
        self.mtf_analyzer = None
        self.redis_client = None
        self.db_client = None
        self.metrics = MetricsCollector(SERVICE_NAME)
        
        # Cached data
        self._cache = {}
        
        # Service state
        self.running = False
        self.initialized = False
        
        # Task management
        self.tasks = set()
        self.executor = None
        self._throttler = Throttler(
            rate_limit=config.get("feature_service.rate_limit", 100),
            time_period=1.0
        )
        
        # Initialize event loop
        self.loop = None
        
        logger.info("Feature service instance created")
    
    async def initialize(self, db_connector: Optional[DatabaseClient] = None) -> None:
        """Initialize the service and its components."""
        if self.initialized:
            logger.warning("Feature service already initialized")
            return
        
        logger.info("Initializing feature service")
        
        try:
            # Initialize common resources
            self.redis_client = RedisClient(
                host=self.config.get("redis.host", "localhost"),
                port=self.config.get("redis.port", 6379),
                db=self.config.get("redis.feature_service_db", 1),
                password=self.config.get("redis.password", None),
            )

            if db_connector is not None:
                self.db_client = db_connector
            if self.db_client is None:
                self.db_client = await get_db_client(
                    connection_string=self.config.get("database.connection_string"),
                    pool_size=self.config.get("database.pool_size", 10),
                    pool_recycle=self.config.get("database.pool_recycle", 3600),
                )
            if getattr(self.db_client, "pool", None) is None:
                await self.db_client.initialize()
                await self.db_client.create_tables()

            
            # Initialize component resources
            max_workers = self.config.get(
                "feature_service.max_workers", 
                min(32, (os.cpu_count() or 1) + 4)
            )
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            
            # Initialize feature service components
            self.processor = FeatureProcessor(
                self.config,
                redis_client=self.redis_client,
                db_client=self.db_client
            )
            
            self.extractor = FeatureExtractor(
                self.config,
                redis_client=self.redis_client,
                db_client=self.db_client
            )
            
            self.mtf_analyzer = MultiTimeframeAnalyzer(
                self.config,
                redis_client=self.redis_client,
                db_client=self.db_client
            )
            
            # Initialize feature service
            init_feature_service()
            
            # Create periodic tasks
            await self._create_tasks()
            
            self.initialized = True
            logger.info("Feature service initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing feature service: {e}")
            logger.error(traceback.format_exc())
            raise FeatureServiceError(f"Initialization failed: {e}")
    
    async def _create_tasks(self):
        """
        Create periodic tasks for the service.
        """
        # Task to refresh feature cache
        refresh_cache_task = PeriodicTask(
            self._refresh_feature_cache,
            interval=self.config.get("feature_service.cache_refresh_interval", 60),
            name="refresh_feature_cache"
        )
        self.tasks.add(refresh_cache_task)
        
        # Task to clean expired cache entries
        clean_cache_task = PeriodicTask(
            self._clean_expired_cache,
            interval=self.config.get("feature_service.cache_cleanup_interval", 300),
            name="clean_expired_cache"
        )
        self.tasks.add(clean_cache_task)
        
        # Task to update feature statistics
        update_stats_task = PeriodicTask(
            self._update_feature_statistics,
            interval=self.config.get("feature_service.stats_update_interval", 600),
            name="update_feature_statistics"
        )
        self.tasks.add(update_stats_task)
        
        # Task to health check feature calculators
        health_check_task = PeriodicTask(
            self._health_check,
            interval=self.config.get("feature_service.health_check_interval", 300),
            name="health_check"
        )
        self.tasks.add(health_check_task)
        
        logger.info(f"Created {len(self.tasks)} periodic tasks")
    
    async def start(self):
        """
        Start the feature service.
        """
        if self.running:
            logger.warning("Feature service already running")
            return
        
        logger.info("Starting feature service")
        
        if not self.initialized:
            await self.initialize()
        
        # Start periodic tasks
        for task in self.tasks:
            await task.start()
        
        self.running = True
        logger.info("Feature service started successfully")
    
    async def stop(self):
        """
        Stop the feature service.
        """
        if not self.running:
            logger.warning("Feature service not running")
            return
        
        logger.info("Stopping feature service")
        
        # Stop periodic tasks
        for task in self.tasks:
            await task.stop()
        
        # Shutdown components
        if self.executor:
            self.executor.shutdown(wait=True)
        
        shutdown_feature_service()
        
        self.running = False
        logger.info("Feature service stopped successfully")
    
    async def cleanup(self):
        """
        Clean up resources used by the service.
        """
        logger.info("Cleaning up feature service resources")
        
        if self.running:
            await self.stop()
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_client:
            await self.db_client.close()
        
        self.initialized = False
        logger.info("Feature service cleanup complete")
    
    async def _refresh_feature_cache(self):
        """
        Refresh the feature cache with latest data.
        """
        try:
            assets = await self._get_active_assets()
            timeframes = self.config.get("feature_service.timeframes", ["1m", "5m", "15m", "1h", "4h", "1d"])
            
            logger.debug(f"Refreshing feature cache for {len(assets)} assets across {len(timeframes)} timeframes")
            
            for asset in assets:
                for timeframe in timeframes:
                    # Use throttler to prevent overwhelming the system
                    async with self._throttler:
                        task_name = f"refresh_cache_{asset}_{timeframe}"
                        await self._calculate_and_cache_features(asset, timeframe)
            
            self.metrics.gauge("feature_service.cache_size", len(self._cache))
            self.metrics.counter("feature_service.cache_refresh", 1)
            
        except Exception as e:
            logger.error(f"Error refreshing feature cache: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.cache_refresh_errors", 1)
    
    async def _clean_expired_cache(self):
        """
        Clean expired entries from the feature cache.
        """
        try:
            now = time.time()
            expiry = self.config.get("feature_service.cache_expiry_seconds", 300)
            expired_keys = []
            
            for key, (timestamp, _) in self._cache.items():
                if now - timestamp > expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
            
            logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
            self.metrics.gauge("feature_service.expired_entries_cleaned", len(expired_keys))
            
        except Exception as e:
            logger.error(f"Error cleaning expired cache: {e}")
            self.metrics.counter("feature_service.cache_cleanup_errors", 1)
    
    async def _update_feature_statistics(self):
        """
        Update statistics about feature calculations.
        """
        try:
            # Collect and store various statistics about features
            calculators = get_all_feature_calculators()
            
            stats = {
                "total_calculators": len(calculators),
                "calculation_counts": {},
                "performance_metrics": {},
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Store statistics in Redis for monitoring
            await self.redis_client.set(
                "feature_service:statistics",
                json.dumps(stats),
                expire=3600
            )
            
            logger.debug("Updated feature statistics")
            self.metrics.counter("feature_service.stats_updates", 1)
            
        except Exception as e:
            logger.error(f"Error updating feature statistics: {e}")
            self.metrics.counter("feature_service.stats_update_errors", 1)
    
    async def _health_check(self):
        """
        Perform health check on feature calculators.
        """
        try:
            calculators = get_all_feature_calculators()
            all_healthy = True
            issues = []
            
            for name, calculator in calculators.items():
                # Basic check - ensure calculator is callable
                if not callable(getattr(calculator, "calculate", None)):
                    all_healthy = False
                    issues.append(f"Calculator '{name}' missing calculate method")
            
            # Report health status
            health_status = {
                "status": "healthy" if all_healthy else "unhealthy",
                "issues": issues,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self.redis_client.set(
                "feature_service:health",
                json.dumps(health_status),
                expire=600
            )
            
            logger.debug(f"Health check completed: {health_status['status']}")
            self.metrics.gauge("feature_service.health_status", 1 if all_healthy else 0)
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
            self.metrics.counter("feature_service.health_check_errors", 1)
    
    async def _get_active_assets(self) -> List[str]:
        """
        Get the list of currently active assets.
        
        Returns:
            List of active asset symbols
        """
        try:
            # First try to get from Redis cache
            assets_json = await self.redis_client.get("active_assets")
            if assets_json:
                return json.loads(assets_json)
            
            # Fallback to config if Redis doesn't have it
            return self.config.get("trading.active_assets", [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT",
                "SOL/USDT", "ADA/USDT", "DOGE/USDT", "MATIC/USDT"
            ])
            
        except Exception as e:
            logger.error(f"Error getting active assets: {e}")
            # Return default assets in case of error
            return ["BTC/USDT", "ETH/USDT"]
    
    @time_execution("feature_calculation")
    async def _calculate_and_cache_features(self, asset: str, timeframe: str):
        """
        Calculate features for an asset and timeframe and store in cache.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string (e.g., "1m", "1h")
        """
        cache_key = f"{asset}_{timeframe}"
        
        try:
            # Get market data
            market_data = await self._get_market_data(asset, timeframe)
            if market_data is None or market_data.empty:
                logger.warning(f"No market data for {asset} {timeframe}")
                return
            
            # Calculate basic technical features
            tech_features = await self.executor.submit(
                self.extractor.calculate_technical_features,
                market_data
            )
            
            # Calculate volatility features
            volatility_features = await self.executor.submit(
                self.extractor.calculate_volatility_features,
                market_data
            )
            
            # Calculate volume features
            volume_features = await self.executor.submit(
                self.extractor.calculate_volume_features,
                market_data
            )
            
            # Calculate pattern features
            pattern_features = await self.executor.submit(
                self.extractor.calculate_pattern_features,
                market_data
            )
            
            # Merge all features
            all_features = {
                **tech_features,
                **volatility_features,
                **volume_features,
                **pattern_features
            }
            
            # Store in cache with timestamp
            self._cache[cache_key] = (time.time(), all_features)
            
            # Also store the latest in Redis for other services
            await self.redis_client.set(
                f"features:{cache_key}",
                json.dumps(all_features),
                expire=self.config.get("feature_service.redis_cache_expiry", 300)
            )
            
            logger.debug(f"Calculated and cached features for {asset} {timeframe}")
            self.metrics.counter("feature_service.calculations", 1)
            
        except Exception as e:
            logger.error(f"Error calculating features for {asset} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.calculation_errors", 1)
            raise FeatureCalculationError(f"Feature calculation failed for {asset} {timeframe}: {e}")
    
    async def _get_market_data(self, asset: str, timeframe: str) -> pd.DataFrame:
        """
        Get market data for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            DataFrame containing OHLCV data
        """
        try:
            # Try to get from Redis first for faster access
            data_key = f"market_data:{asset}:{timeframe}"
            cached_data = await self.redis_client.get(data_key)
            
            if cached_data:
                # Deserialize and convert to DataFrame
                data_dict = json.loads(cached_data)
                return pd.DataFrame(data_dict)
            
            # If not in Redis, fetch from database
            query = """
                SELECT 
                    timestamp, open, high, low, close, volume
                FROM 
                    market_data.ohlcv
                WHERE 
                    asset = %s AND timeframe = %s
                ORDER BY 
                    timestamp DESC
                LIMIT 
                    %s
            """
            
            lookback = self.config.get("feature_service.lookback_periods", 500)
            
            result = await self.db_client.fetch_all(
                query, (asset, timeframe, lookback)
            )
            
            if not result:
                logger.warning(f"No data found for {asset} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(result, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Set timestamp as index
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Sort by time
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data for {asset} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.data_fetch_errors", 1)
            return pd.DataFrame()  # Return empty dataframe in case of error
    
    async def get_features(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Get features for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary of features
        """
        cache_key = f"{asset}_{timeframe}"
        
        # Check if in memory cache first
        if cache_key in self._cache:
            timestamp, features = self._cache[cache_key]
            # Check if cache is fresh enough
            if time.time() - timestamp < self.config.get("feature_service.cache_expiry_seconds", 300):
                return features
        
        # Not in cache or expired, recalculate
        await self._calculate_and_cache_features(asset, timeframe)
        
        # Return from cache now
        if cache_key in self._cache:
            return self._cache[cache_key][1]
        
        # If still not in cache, something went wrong
        raise FeatureCalculationError(f"Failed to calculate features for {asset} {timeframe}")
    
    async def get_multi_timeframe_features(
        self, 
        asset: str, 
        timeframes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get features for multiple timeframes for the same asset.
        
        Args:
            asset: Asset symbol
            timeframes: List of timeframe strings
            
        Returns:
            Dictionary of features keyed by timeframe
        """
        result = {}
        
        for timeframe in timeframes:
            features = await self.get_features(asset, timeframe)
            result[timeframe] = features
        
        # Add multi-timeframe correlation features
        mtf_features = await self.executor.submit(
            self.mtf_analyzer.analyze_timeframes,
            asset, result
        )
        
        # Add to result
        result["mtf"] = mtf_features
        
        return result
    
    async def detect_patterns(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Detect patterns for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary of detected patterns with confidence scores
        """
        try:
            # Get features first
            features = await self.get_features(asset, timeframe)
            
            # Get market data
            market_data = await self._get_market_data(asset, timeframe)
            
            # Detect patterns using the pattern recognition module
            patterns = await self.executor.submit(
                self.processor.detect_patterns,
                market_data, features
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {asset} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.pattern_detection_errors", 1)
            return {}  # Return empty dict in case of error
    
    async def analyze_market_structure(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze market structure for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary of market structure analysis
        """
        try:
            # Get features first
            features = await self.get_features(asset, timeframe)
            
            # Get market data
            market_data = await self._get_market_data(asset, timeframe)
            
            # Analyze market structure
            structure = await self.executor.submit(
                self.processor.analyze_market_structure,
                market_data, features
            )
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing market structure for {asset} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.market_structure_errors", 1)
            return {}  # Return empty dict in case of error
    
    async def identify_support_resistance(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Identify support and resistance levels for a specific asset and timeframe.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary of support and resistance levels
        """
        try:
            # Get features first
            features = await self.get_features(asset, timeframe)
            
            # Get market data
            market_data = await self._get_market_data(asset, timeframe)
            
            # Identify support and resistance levels
            levels = await self.executor.submit(
                self.processor.identify_support_resistance,
                market_data, features
            )
            
            return levels
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance for {asset} {timeframe}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.support_resistance_errors", 1)
            return {}  # Return empty dict in case of error
    
    async def analyze_order_flow(self, asset: str, timeframe: str) -> Dict[str, Any]:
        """
        Analyze order flow for a specific asset.
        
        Args:
            asset: Asset symbol
            timeframe: Timeframe string
            
        Returns:
            Dictionary of order flow analysis
        """
        try:
            # For order flow, we need L2 book data
            # First check if we have cached order flow analysis
            cache_key = f"order_flow:{asset}:{timeframe}"
            cached_flow = await self.redis_client.get(cache_key)
            
            if cached_flow:
                return json.loads(cached_flow)
            
            # If not cached, we need to perform the analysis
            # Get order book data
            book_data = await self._get_order_book_data(asset)
            
            # Get market data
            market_data = await self._get_market_data(asset, timeframe)
            
            # Analyze order flow
            order_flow = await self.executor.submit(
                self.processor.analyze_order_flow,
                book_data, market_data
            )
            
            # Cache the result
            await self.redis_client.set(
                cache_key,
                json.dumps(order_flow),
                expire=self.config.get("feature_service.order_flow_cache_expiry", 60)
            )
            
            return order_flow
            
        except Exception as e:
            logger.error(f"Error analyzing order flow for {asset}: {e}")
            logger.error(traceback.format_exc())
            self.metrics.counter("feature_service.order_flow_errors", 1)
            return {}  # Return empty dict in case of error
    
    async def _get_order_book_data(self, asset: str) -> Dict[str, Any]:
        """
        Get order book data for a specific asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            Dictionary of order book data
        """
        try:
            # First check Redis cache
            cache_key = f"order_book:{asset}"
            cached_book = await self.redis_client.get(cache_key)
            
            if cached_book:
                return json.loads(cached_book)
            
            # If not found in cache, return empty dict
            # Actual implementation would fetch from market data service
            logger.warning(f"Order book data not available for {asset}")
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching order book data for {asset}: {e}")
            return {}

# Main service instance creation function
def create_feature_service(config: Config) -> FeatureService:
    """
    Create a new feature service instance.
    
    Args:
        config: Application configuration
        
    Returns:
        Initialized feature service
    """
    return FeatureService(config)

# Command-line entry point
if __name__ == "__main__":
    import argparse
    from config import load_config
    
    parser = argparse.ArgumentParser(description="QuantumSpectre Feature Service")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    args = parser.parse_args()
    
    # Setup logging
    from common.logger import setup_logging
    setup_logging(level=args.log_level.upper())
    
    # Load configuration
    config = load_config(args.config if args.config else None)
    
    # Create and run service
    async def main():
        service = create_feature_service(config)
        
        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(service)))
        
        try:
            await service.initialize()
            await service.start()
            
            # Keep the service running
            while service.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in feature service: {e}")
            logger.error(traceback.format_exc())
        finally:
            await service.cleanup()
    
    async def shutdown(service):
        logger.info("Shutdown signal received")
        await service.stop()
        await service.cleanup()
        
        # Stop the event loop
        loop = asyncio.get_running_loop()
        loop.stop()
    
    # Run the service
    asyncio.run(main())
