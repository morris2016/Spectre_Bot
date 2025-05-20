#!/usr/bin/env python3
"""
QuantumSpectre Elite Trading System
Asset Council Module

This module implements the Asset Council, which coordinates strategy brains
specialized for specific assets. It personalizes trading strategies based on
asset-specific characteristics to achieve optimal performance.
"""

import os
import time
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
import logging

from common.constants import TIMEFRAMES
from common.logger import get_logger
from common.utils import generate_id
from common.async_utils import gather_with_concurrency
from common.exceptions import AssetCouncilError, BrainNotFoundError
from data_storage.models.market_data import AssetCharacteristics
from data_storage.models.strategy_data import StrategyPerformance

from brain_council.base_council import BaseCouncil
from brain_council.signal_generator import TradeSignal, SignalStrength, SignalConfidence


class AssetCouncil(BaseCouncil):
    """
    Asset Council coordinates strategies specialized for specific assets.
    
    This council manages a set of strategy brains that are specifically tuned
    for a particular trading asset, ensuring that strategies are personalized
    to each asset's unique characteristics and behavior patterns.
    """
    
    def __init__(self, 
                 asset_id: str, 
                 platform: str,
                 config: Dict[str, Any] = None):
        """
        Initialize the Asset Council for a specific asset.
        
        Args:
            asset_id: Unique identifier for the asset
            platform: Trading platform (Binance or Deriv)
            config: Configuration parameters for this council
        """
        super().__init__(f"asset_council_{platform}_{asset_id}", config)
        self.asset_id = asset_id
        self.platform = platform
        self.logger = get_logger(f"asset_council.{platform}.{asset_id}")
        
        # Asset-specific characteristics
        self.characteristics = None
        
        # Dictionary of brain strategies assigned to this asset
        self.strategy_brains = {}
        
        # Historical performance of strategies for this asset
        self.strategy_performance = {}
        
        # Strategy weights based on performance
        self.strategy_weights = {}
        
        # Asset-specific parameters
        self.volatility_profile = None
        self.correlation_matrix = None
        self.liquidity_profile = None
        self.pattern_effectiveness = {}
        
        # Dictionary of timeframe-specific models
        self.timeframe_models = {}
        
        # Last model update time
        self.last_model_update = {}
        
        # Asset-specific market regime
        self.current_regime = None
        
        # Council operation flags
        self.initialized = False
        self.running = False
        
        # Asset-specific advisories
        self.advisories = []
        
        # Recent signals
        self.recent_signals = []
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Initialize the council
        self._initialize()
    
    async def _initialize(self) -> None:
        """Initialize the Asset Council with asset-specific data and strategies."""
        try:
            self.logger.info(f"Initializing Asset Council for {self.platform}:{self.asset_id}")
            
            # Load asset characteristics
            await self._load_asset_characteristics()
            
            # Load appropriate strategy brains based on asset characteristics
            await self._load_strategy_brains()
            
            # Initialize performance tracking
            await self._initialize_performance_tracking()
            
            # Analyze historical patterns for this asset
            await self._analyze_historical_patterns()
            
            # Compute initial strategy weights based on historical performance
            await self._compute_strategy_weights()
            
            # Initialize market regime detection
            await self._initialize_regime_detection()
            
            self.initialized = True
            self.logger.info(f"Asset Council initialized for {self.platform}:{self.asset_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Asset Council: {str(e)}")
            raise AssetCouncilError(f"Initialization error: {str(e)}")
    
    async def _load_asset_characteristics(self) -> None:
        """Load and analyze asset-specific characteristics."""
        try:
            # Retrieve asset characteristics from database
            self.characteristics = await AssetCharacteristics.get_by_asset_id(
                self.asset_id, self.platform)
            
            if not self.characteristics:
                # If no characteristics exist, initialize them
                self.logger.info(f"No characteristics found for {self.asset_id}, creating initial profile")
                self.characteristics = await self._create_initial_characteristics()
            
            # Analyze volatility profile
            self.volatility_profile = await self._analyze_volatility_profile()
            
            # Analyze liquidity profile
            self.liquidity_profile = await self._analyze_liquidity_profile()
            
            # Load pattern effectiveness for this asset
            self.pattern_effectiveness = await self._load_pattern_effectiveness()
            
            self.logger.info(f"Asset characteristics loaded for {self.platform}:{self.asset_id}")
        
        except Exception as e:
            self.logger.error(f"Failed to load asset characteristics: {str(e)}")
            raise AssetCouncilError(f"Asset characteristics loading error: {str(e)}")
    
    async def _create_initial_characteristics(self) -> AssetCharacteristics:
        """Create initial asset characteristics profile."""
        # Create a new characteristics profile
        characteristics = AssetCharacteristics(
            asset_id=self.asset_id,
            platform=self.platform,
            volatility_profile={
                'daily_avg': None,
                'weekly_avg': None,
                'trend_strength': None
            },
            liquidity_profile={
                'avg_volume': None,
                'spread': None,
                'depth': None
            },
            correlation_data={},
            seasonal_patterns={},
            behavioral_traits={},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save to database
        await characteristics.save()
        
        # Schedule a comprehensive analysis
        asyncio.create_task(self._comprehensive_asset_analysis())
        
        return characteristics
    
    async def _comprehensive_asset_analysis(self) -> None:
        """Perform comprehensive analysis of asset characteristics."""
        self.logger.info(f"Starting comprehensive analysis for {self.platform}:{self.asset_id}")
        
        # This would be a long-running task analyzing:
        # - Historical volatility patterns
        # - Liquidity characteristics
        # - Seasonal tendencies
        # - Correlation with other assets/indices
        # - Response to news events
        # - Support/resistance level adherence
        # - Gap behavior
        # - Response to various technical patterns
        
        # For demonstration purposes, we'll simulate this process
        await asyncio.sleep(2)  # Simulate analysis time
        
        # Update characteristics with deeper analysis
        await self._update_asset_characteristics()
        
        self.logger.info(f"Completed comprehensive analysis for {self.platform}:{self.asset_id}")
    
    async def _update_asset_characteristics(self) -> None:
        """Update asset characteristics with latest analysis results."""
        # Update the characteristics with new data
        self.characteristics.volatility_profile = self.volatility_profile
        self.characteristics.liquidity_profile = self.liquidity_profile
        self.characteristics.updated_at = datetime.now()
        
        # Save updated characteristics
        await self.characteristics.save()
        
        self.logger.info(f"Updated asset characteristics for {self.platform}:{self.asset_id}")
    
    async def _analyze_volatility_profile(self) -> Dict[str, float]:
        """Analyze volatility profile for this asset."""
        # In a real implementation, this would:
        # - Retrieve historical price data
        # - Calculate various volatility metrics (ATR, standard deviation, etc.)
        # - Analyze volatility regimes and patterns
        # - Build a comprehensive volatility profile
        
        # For now, we'll return a simulated profile
        return {
            'daily_avg': 1.2,  # Average daily volatility %
            'weekly_avg': 2.8,  # Average weekly volatility %
            'monthly_avg': 5.7,  # Average monthly volatility %
            'intraday_pattern': 'u-shaped',  # Volatility pattern within the day
            'volatility_regime': 'moderate',  # Current volatility regime
            'atr_daily': 0.45,  # Average True Range (daily)
            'trend_volatility_ratio': 0.68,  # Ratio of trend to noise
            'garch_params': {  # Generalized AutoRegressive Conditional Heteroskedasticity parameters
                'alpha': 0.05,
                'beta': 0.85
            }
        }
    
    async def _analyze_liquidity_profile(self) -> Dict[str, Any]:
        """Analyze liquidity profile for this asset."""
        # In a real implementation, this would:
        # - Analyze order book depth
        # - Calculate typical spreads
        # - Measure market impact for different order sizes
        # - Determine optimal execution strategies
        
        # For now, we'll return a simulated profile
        return {
            'avg_spread': 0.03,  # Average spread in percentage
            'avg_depth_100k': 0.25,  # Average price impact for $100k order
            'daily_volume_avg': 25000000,  # Average daily volume in base currency
            'liquidity_score': 78,  # 0-100 score of overall liquidity
            'optimal_order_size': 50000,  # Optimal order size for minimal impact
            'time_to_liquidate': {  # Estimated time to liquidate positions of different sizes
                '10k': '1m',
                '100k': '5m',
                '1M': '45m'
            },
            'hourly_liquidity_pattern': [
                65, 60, 52, 45, 40, 35,  # 00:00-05:59
                38, 55, 72, 85, 92, 95,  # 06:00-11:59
                90, 88, 85, 82, 80, 85,  # 12:00-17:59
                80, 75, 70, 68, 66, 65   # 18:00-23:59
            ]
        }
    
    async def _load_pattern_effectiveness(self) -> Dict[str, float]:
        """Load pattern effectiveness data for this asset."""
        # This would retrieve historical data on how effective different
        # chart patterns and indicators have been for this specific asset
        
        # For now, we'll return simulated effectiveness scores (0-1)
        return {
            'engulfing': 0.78,
            'doji': 0.65,
            'hammer': 0.71,
            'shooting_star': 0.63,
            'marubozu': 0.58,
            'three_white_soldiers': 0.83,
            'three_black_crows': 0.81,
            'morning_star': 0.76,
            'evening_star': 0.75,
            'bullish_harami': 0.62,
            'bearish_harami': 0.64,
            'rsi_oversold': 0.68,
            'rsi_overbought': 0.65,
            'macd_crossover': 0.73,
            'double_top': 0.77,
            'double_bottom': 0.79,
            'head_shoulders': 0.81,
            'inverse_head_shoulders': 0.82,
            'triangle_ascending': 0.74,
            'triangle_descending': 0.73,
            'triangle_symmetrical': 0.69,
            'rectangle': 0.65,
            'wedge_rising': 0.67,
            'wedge_falling': 0.68,
            'cup_with_handle': 0.76,
            'fibonacci_retracement': 0.82,
            'support_bounce': 0.84,
            'resistance_rejection': 0.81
        }
    
    async def _load_strategy_brains(self) -> None:
        """Load appropriate strategy brains for this asset."""
        try:
            # In a production system, this would:
            # 1. Analyze which strategies historically work best for this asset
            # 2. Retrieve strategy performance metrics
            # 3. Initialize the appropriate strategy brain instances
            # 4. Configure them with asset-specific parameters
            
            from strategy_brains.momentum_brain import MomentumBrain
            from strategy_brains.mean_reversion_brain import MeanReversionBrain
            from strategy_brains.breakout_brain import BreakoutBrain
            from strategy_brains.pattern_brain import PatternBrain
            from strategy_brains.volatility_brain import VolatilityBrain
            from strategy_brains.trend_brain import TrendBrain
            from strategy_brains.order_flow_brain import OrderFlowBrain
            from strategy_brains.sentiment_brain import SentimentBrain
            from strategy_brains.market_structure_brain import MarketStructureBrain
            from strategy_brains.ml_brain import MLBrain
            
            # Create instances of strategy brains with asset-specific configurations
            self.strategy_brains = {
                "momentum": MomentumBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("momentum")
                ),
                "mean_reversion": MeanReversionBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("mean_reversion")
                ),
                "breakout": BreakoutBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("breakout")
                ),
                "pattern": PatternBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("pattern", {
                        "pattern_effectiveness": self.pattern_effectiveness
                    })
                ),
                "volatility": VolatilityBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("volatility", {
                        "volatility_profile": self.volatility_profile
                    })
                ),
                "trend": TrendBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("trend")
                ),
                "order_flow": OrderFlowBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("order_flow", {
                        "liquidity_profile": self.liquidity_profile
                    })
                ),
                "sentiment": SentimentBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("sentiment")
                ),
                "market_structure": MarketStructureBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("market_structure")
                ),
                "ml": MLBrain(
                    asset_id=self.asset_id,
                    platform=self.platform,
                    config=self._get_brain_config("ml")
                )
            }
            
            # Initialize each brain
            init_tasks = []
            for name, brain in self.strategy_brains.items():
                init_tasks.append(brain.initialize())
            
            await gather_with_concurrency(5, *init_tasks)
            
            self.logger.info(f"Loaded {len(self.strategy_brains)} strategy brains for {self.platform}:{self.asset_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load strategy brains: {str(e)}")
            raise AssetCouncilError(f"Strategy brain loading error: {str(e)}")
    
    def _get_brain_config(self, brain_type: str, extra_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get configuration for a specific brain type with asset-specific customizations."""
        # Base configuration for this brain type
        if brain_type not in self.config.get('brains', {}):
            base_config = {}
        else:
            base_config = self.config['brains'][brain_type].copy()
        
        # Add asset-specific configurations
        base_config.update({
            "asset_id": self.asset_id,
            "platform": self.platform,
            "volatility_profile": self.volatility_profile,
            "liquidity_profile": self.liquidity_profile,
            "pattern_effectiveness": self.pattern_effectiveness
        })
        
        # Add any extra configuration
        if extra_config:
            base_config.update(extra_config)
        
        return base_config
    
    async def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for this asset's strategies."""
        try:
            # Load historical performance data for each strategy
            strategies = list(self.strategy_brains.keys())
            
            performance_data = await StrategyPerformance.get_by_asset(
                self.asset_id, self.platform, strategies)
            
            # Initialize performance metrics
            for strategy_name in self.strategy_brains:
                # Get historical performance for this strategy
                strategy_perf = next((p for p in performance_data 
                                    if p.strategy_name == strategy_name), None)
                
                if strategy_perf:
                    self.strategy_performance[strategy_name] = strategy_perf
                else:
                    # Create new performance record if none exists
                    self.strategy_performance[strategy_name] = StrategyPerformance(
                        strategy_name=strategy_name,
                        asset_id=self.asset_id,
                        platform=self.platform,
                        win_rate=0.0,
                        profit_factor=0.0,
                        avg_profit=0.0,
                        avg_loss=0.0,
                        sharpe_ratio=0.0,
                        trades_count=0,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    await self.strategy_performance[strategy_name].save()
            
            self.logger.info(f"Initialized performance tracking for {len(strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance tracking: {str(e)}")
            raise AssetCouncilError(f"Performance tracking initialization error: {str(e)}")
    
    async def _analyze_historical_patterns(self) -> None:
        """Analyze historical price patterns for this asset."""
        try:
            # In a production system, this would:
            # 1. Retrieve historical price data
            # 2. Identify recurring patterns and their effectiveness
            # 3. Analyze correlation with external factors
            # 4. Build a pattern library specific to this asset
            
            # For now, we'll simulate this process
            self.logger.info(f"Analyzing historical patterns for {self.platform}:{self.asset_id}")
            await asyncio.sleep(0.5)  # Simulate analysis time
            
            # Example pattern analysis results - would come from actual analysis
            self.historical_patterns = {
                "open_gap_fill_probability": 0.76,
                "support_level_respect": 0.82,
                "resistance_level_respect": 0.79,
                "trend_continuation_after_pullback": 0.68,
                "reversal_after_exhaustion": 0.73,
                "news_impact_duration": {
                    "minor": "2h",
                    "moderate": "1d",
                    "major": "3d"
                },
                "effective_indicators": [
                    "rsi", "macd", "volume_profile", "support_resistance"
                ],
                "volatility_cycle": {
                    "duration": "14d",
                    "current_phase": "expanding",
                    "next_phase_probability": 0.72
                }
            }
            
            self.logger.info(f"Completed historical pattern analysis")
            
        except Exception as e:
            self.logger.error(f"Failed to analyze historical patterns: {str(e)}")
            raise AssetCouncilError(f"Historical pattern analysis error: {str(e)}")
    
    async def _compute_strategy_weights(self) -> None:
        """Compute weights for each strategy based on historical performance."""
        try:
            weights = {}
            total_score = 0
            
            # Calculate a score for each strategy
            for name, performance in self.strategy_performance.items():
                # Base score combines win rate and profit factor
                base_score = (performance.win_rate * 0.6) + (performance.profit_factor * 0.4)
                
                # Adjust for statistical significance
                trades_confidence = min(performance.trades_count / 100, 1.0)
                adjusted_score = base_score * trades_confidence
                
                # Adjustment based on current market regime
                if self.current_regime and hasattr(self, 'regime_compatibility'):
                    regime_factor = self.regime_compatibility.get(name, {}).get(self.current_regime, 0.5)
                    adjusted_score *= regime_factor
                
                # Store score
                weights[name] = max(adjusted_score, 0.01)  # Ensure minimum weight
                total_score += weights[name]
            
            # Normalize weights
            if total_score > 0:
                for name in weights:
                    weights[name] /= total_score
            else:
                # Equal weights if no performance data
                equal_weight = 1.0 / len(weights)
                weights = {name: equal_weight for name in weights}
            
            self.strategy_weights = weights
            
            self.logger.info(f"Computed strategy weights: {', '.join([f'{k}={v:.2f}' for k, v in weights.items()])}")
            
        except Exception as e:
            self.logger.error(f"Failed to compute strategy weights: {str(e)}")
            raise AssetCouncilError(f"Strategy weight computation error: {str(e)}")
    
    async def _initialize_regime_detection(self) -> None:
        """Initialize market regime detection for this asset."""
        try:
            # In a production system, this would:
            # 1. Load historical regime classification models
            # 2. Analyze current market conditions
            # 3. Determine current market regime
            
            # Simplified regime detection for now
            # Possible regimes: trending_up, trending_down, ranging, volatile, breakout
            self.regimes = ["trending_up", "trending_down", "ranging", "volatile", "breakout"]
            
            # Define strategy compatibility with each regime (0-1 score)
            self.regime_compatibility = {
                "momentum": {
                    "trending_up": 0.95,
                    "trending_down": 0.90,
                    "ranging": 0.30,
                    "volatile": 0.45,
                    "breakout": 0.75
                },
                "mean_reversion": {
                    "trending_up": 0.40,
                    "trending_down": 0.40,
                    "ranging": 0.95,
                    "volatile": 0.65,
                    "breakout": 0.25
                },
                "breakout": {
                    "trending_up": 0.35,
                    "trending_down": 0.35,
                    "ranging": 0.55,
                    "volatile": 0.70,
                    "breakout": 0.95
                },
                "pattern": {
                    "trending_up": 0.75,
                    "trending_down": 0.75,
                    "ranging": 0.80,
                    "volatile": 0.50,
                    "breakout": 0.85
                },
                "volatility": {
                    "trending_up": 0.60,
                    "trending_down": 0.60,
                    "ranging": 0.40,
                    "volatile": 0.90,
                    "breakout": 0.75
                },
                "trend": {
                    "trending_up": 0.95,
                    "trending_down": 0.95,
                    "ranging": 0.20,
                    "volatile": 0.50,
                    "breakout": 0.70
                },
                "order_flow": {
                    "trending_up": 0.80,
                    "trending_down": 0.80,
                    "ranging": 0.75,
                    "volatile": 0.85,
                    "breakout": 0.90
                },
                "sentiment": {
                    "trending_up": 0.70,
                    "trending_down": 0.70,
                    "ranging": 0.50,
                    "volatile": 0.65,
                    "breakout": 0.80
                },
                "market_structure": {
                    "trending_up": 0.85,
                    "trending_down": 0.85,
                    "ranging": 0.75,
                    "volatile": 0.60,
                    "breakout": 0.90
                },
                "ml": {
                    "trending_up": 0.80,
                    "trending_down": 0.80,
                    "ranging": 0.80,
                    "volatile": 0.80,
                    "breakout": 0.80
                }
            }
            
            # Set initial regime (would be determined through analysis)
            await self._detect_current_regime()
            
            self.logger.info(f"Initialized regime detection, current regime: {self.current_regime}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize regime detection: {str(e)}")
            raise AssetCouncilError(f"Regime detection initialization error: {str(e)}")
    
    async def _detect_current_regime(self) -> None:
        """Detect the current market regime for this asset."""
        # In a production system, this would use sophisticated algorithms
        # to classify the current market state
        
        # For now, we'll use a simplified approach
        try:
            # Placeholder for actual regime detection logic
            # This would analyze:
            # - Recent price action
            # - Volatility levels
            # - Trending/ranging behavior
            # - Support/resistance interactions
            
            # Simulated regime detection
            import random
            self.current_regime = random.choice(self.regimes)
            
            # When regime changes, update strategy weights
            if hasattr(self, 'last_regime') and self.last_regime != self.current_regime:
                await self._compute_strategy_weights()
            
            self.last_regime = self.current_regime
            
        except Exception as e:
            self.logger.error(f"Failed to detect current regime: {str(e)}")
            self.current_regime = "unknown"
    
    async def process_data(self, data: Dict[str, Any]) -> None:
        """
        Process new market data through the asset council.
        
        Args:
            data: New market data for this asset
        """
        if not self.initialized:
            self.logger.warning("Asset Council not initialized, initializing now")
            await self._initialize()
        
        try:
            # Update asset characteristics if needed
            if self._should_update_characteristics():
                await self._update_asset_characteristics()
            
            # Detect current market regime
            await self._detect_current_regime()
            
            # Process data with each strategy brain
            brain_signals = {}
            for name, brain in self.strategy_brains.items():
                try:
                    signals = await brain.process_data(data)
                    brain_signals[name] = signals
                except Exception as e:
                    self.logger.error(f"Error in brain {name}: {str(e)}")
                    continue
            
            # Generate aggregate signal
            aggregate_signal = await self._generate_aggregate_signal(brain_signals)
            
            # Update performance metrics
            await self._update_performance_metrics(brain_signals)
            
            # Store advisory messages
            self._update_advisories(brain_signals, aggregate_signal)
            
            # Store recent signals
            self._update_recent_signals(aggregate_signal)
            
            return aggregate_signal
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise AssetCouncilError(f"Data processing error: {str(e)}")
    
    def _should_update_characteristics(self) -> bool:
        """Determine if asset characteristics should be updated."""
        if not hasattr(self, 'last_characteristics_update'):
            self.last_characteristics_update = datetime.now() - timedelta(days=1)
            return True
        
        # Update characteristics every 12 hours
        time_since_update = datetime.now() - self.last_characteristics_update
        if time_since_update > timedelta(hours=12):
            self.last_characteristics_update = datetime.now()
            return True
        
        return False
    
    async def _generate_aggregate_signal(self, 
                                        brain_signals: Dict[str, List[TradeSignal]]
                                        ) -> List[TradeSignal]:
        """
        Generate aggregate signals from multiple strategy brains.
        
        Args:
            brain_signals: Dictionary of signals from each strategy brain
            
        Returns:
            List of aggregated trade signals
        """
        try:
            # Combine and weight signals
            aggregated_signals = []
            
            # Group signals by timeframe
            timeframe_signals = {}
            for brain_name, signals in brain_signals.items():
                for signal in signals:
                    if signal.timeframe not in timeframe_signals:
                        timeframe_signals[signal.timeframe] = []
                    
                    # Apply brain-specific weight
                    weighted_signal = signal.copy()
                    weighted_signal.confidence *= self.strategy_weights.get(brain_name, 0.1)
                    
                    # Apply regime compatibility factor
                    if self.current_regime in self.regime_compatibility.get(brain_name, {}):
                        regime_factor = self.regime_compatibility[brain_name][self.current_regime]
                        weighted_signal.confidence *= regime_factor
                    
                    timeframe_signals[signal.timeframe].append((brain_name, weighted_signal))
            
            # Process each timeframe separately
            for timeframe, signals in timeframe_signals.items():
                # Group by signal type (BUY/SELL)
                buy_signals = [s for _, s in signals if s.signal_type == "BUY"]
                sell_signals = [s for _, s in signals if s.signal_type == "SELL"]
                
                # If we have both buy and sell signals, evaluate the stronger case
                if buy_signals and sell_signals:
                    buy_confidence = sum(s.confidence for s in buy_signals)
                    sell_confidence = sum(s.confidence for s in sell_signals)
                    
                    # Only proceed if there's a clear winner
                    confidence_diff = abs(buy_confidence - sell_confidence)
                    if confidence_diff > 0.3:  # Minimum threshold for clear signal
                        if buy_confidence > sell_confidence:
                            # Generate buy signal
                            agg_signal = self._create_aggregate_buy_signal(timeframe, buy_signals)
                            aggregated_signals.append(agg_signal)
                        else:
                            # Generate sell signal
                            agg_signal = self._create_aggregate_sell_signal(timeframe, sell_signals)
                            aggregated_signals.append(agg_signal)
                
                # If we only have buy signals
                elif buy_signals:
                    buy_confidence = sum(s.confidence for s in buy_signals)
                    if buy_confidence > 0.5:  # Minimum threshold
                        agg_signal = self._create_aggregate_buy_signal(timeframe, buy_signals)
                        aggregated_signals.append(agg_signal)
                
                # If we only have sell signals
                elif sell_signals:
                    sell_confidence = sum(s.confidence for s in sell_signals)
                    if sell_confidence > 0.5:  # Minimum threshold
                        agg_signal = self._create_aggregate_sell_signal(timeframe, sell_signals)
                        aggregated_signals.append(agg_signal)
            
            return aggregated_signals
            
        except Exception as e:
            self.logger.error(f"Error generating aggregate signal: {str(e)}")
            return []
    
    def _create_aggregate_buy_signal(self, 
                                    timeframe: str, 
                                    buy_signals: List[TradeSignal]
                                    ) -> TradeSignal:
        """Create an aggregate buy signal from multiple buy signals."""
        # Calculate weighted average confidence
        total_confidence = sum(s.confidence for s in buy_signals)
        weighted_confidence = total_confidence / len(buy_signals)
        
        # Calculate strength based on confidence
        if weighted_confidence > 0.8:
            strength = SignalStrength.STRONG
        elif weighted_confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Combine reasons
        reasons = []
        for signal in sorted(buy_signals, key=lambda s: s.confidence, reverse=True)[:3]:
            reasons.extend(signal.reasons)
        
        # Create aggregate signal
        return TradeSignal(
            asset_id=self.asset_id,
            platform=self.platform,
            timeframe=timeframe,
            signal_type="BUY",
            confidence=min(weighted_confidence, 0.95),  # Cap at 0.95
            strength=strength,
            entry_price=self._calculate_aggregate_price(buy_signals, 'entry_price'),
            stop_loss=self._calculate_aggregate_price(buy_signals, 'stop_loss'),
            take_profit=self._calculate_aggregate_price(buy_signals, 'take_profit'),
            timestamp=datetime.now(),
            reasons=reasons[:5],  # Limit to top 5 reasons
            source="asset_council",
            expiration=datetime.now() + timedelta(hours=2)
        )
    
    def _create_aggregate_sell_signal(self, 
                                     timeframe: str, 
                                     sell_signals: List[TradeSignal]
                                     ) -> TradeSignal:
        """Create an aggregate sell signal from multiple sell signals."""
        # Calculate weighted average confidence
        total_confidence = sum(s.confidence for s in sell_signals)
        weighted_confidence = total_confidence / len(sell_signals)
        
        # Calculate strength based on confidence
        if weighted_confidence > 0.8:
            strength = SignalStrength.STRONG
        elif weighted_confidence > 0.6:
            strength = SignalStrength.MODERATE
        else:
            strength = SignalStrength.WEAK
        
        # Combine reasons
        reasons = []
        for signal in sorted(sell_signals, key=lambda s: s.confidence, reverse=True)[:3]:
            reasons.extend(signal.reasons)
        
        # Create aggregate signal
        return TradeSignal(
            asset_id=self.asset_id,
            platform=self.platform,
            timeframe=timeframe,
            signal_type="SELL",
            confidence=min(weighted_confidence, 0.95),  # Cap at 0.95
            strength=strength,
            entry_price=self._calculate_aggregate_price(sell_signals, 'entry_price'),
            stop_loss=self._calculate_aggregate_price(sell_signals, 'stop_loss'),
            take_profit=self._calculate_aggregate_price(sell_signals, 'take_profit'),
            timestamp=datetime.now(),
            reasons=reasons[:5],  # Limit to top 5 reasons
            source="asset_council",
            expiration=datetime.now() + timedelta(hours=2)
        )
    
    def _calculate_aggregate_price(self, 
                                  signals: List[TradeSignal], 
                                  price_attr: str
                                  ) -> Optional[float]:
        """Calculate aggregate price point from multiple signals."""
        valid_prices = [getattr(s, price_attr) for s in signals 
                      if getattr(s, price_attr) is not None]
        
        if not valid_prices:
            return None
        
        # For entry price, use confidence-weighted average
        if price_attr == 'entry_price':
            weighted_sum = sum(getattr(s, price_attr) * s.confidence for s in signals 
                             if getattr(s, price_attr) is not None)
            total_weight = sum(s.confidence for s in signals 
                             if getattr(s, price_attr) is not None)
            
            if total_weight > 0:
                return weighted_sum / total_weight
            return None
        
        # For stop loss, use most conservative (for risk management)
        elif price_attr == 'stop_loss':
            buy_signals = [s for s in signals if s.signal_type == "BUY"]
            sell_signals = [s for s in signals if s.signal_type == "SELL"]
            
            if buy_signals:
                # For buy signals, stop loss is lower
                valid_stops = [getattr(s, price_attr) for s in buy_signals 
                             if getattr(s, price_attr) is not None]
                return max(valid_stops) if valid_stops else None
            
            elif sell_signals:
                # For sell signals, stop loss is higher
                valid_stops = [getattr(s, price_attr) for s in sell_signals 
                             if getattr(s, price_attr) is not None]
                return min(valid_stops) if valid_stops else None
        
        # For take profit, use middle ground approach
        elif price_attr == 'take_profit':
            buy_signals = [s for s in signals if s.signal_type == "BUY"]
            sell_signals = [s for s in signals if s.signal_type == "SELL"]
            
            if buy_signals:
                # For buy signals, take profit is higher
                valid_targets = [getattr(s, price_attr) for s in buy_signals 
                               if getattr(s, price_attr) is not None]
                # Use median for reasonable target
                return np.median(valid_targets) if valid_targets else None
            
            elif sell_signals:
                # For sell signals, take profit is lower
                valid_targets = [getattr(s, price_attr) for s in sell_signals 
                               if getattr(s, price_attr) is not None]
                # Use median for reasonable target
                return np.median(valid_targets) if valid_targets else None
        
        # Fallback to average
        return sum(valid_prices) / len(valid_prices)
    
    async def _update_performance_metrics(self, 
                                         brain_signals: Dict[str, List[TradeSignal]]
                                         ) -> None:
        """Update performance metrics for each strategy brain."""
        # In a production system, this would update based on trade outcomes
        # We'll simulate this for demonstration purposes
        for brain_name, signals in brain_signals.items():
            if brain_name in self.strategy_performance:
                # Update metrics periodically based on actual trade outcomes
                if random.random() < 0.05:  # 5% chance to update
                    self.strategy_performance[brain_name].trades_count += 1
                    # Other metrics would be updated based on actual outcomes
    
    def _update_advisories(self, 
                          brain_signals: Dict[str, List[TradeSignal]], 
                          aggregate_signal: List[TradeSignal]
                          ) -> None:
        """Update asset-specific advisories based on signals."""
        # Clear old advisories
        self.advisories = []
        
        # Add general advisories based on market regime
        self.advisories.append({
            "type": "regime",
            "message": f"Current market regime: {self.current_regime}",
            "importance": "medium"
        })
        
        # Add signal-specific advisories
        for signal in aggregate_signal:
            if signal.strength == SignalStrength.STRONG:
                importance = "high"
            elif signal.strength == SignalStrength.MODERATE:
                importance = "medium"
            else:
                importance = "low"
                
            self.advisories.append({
                "type": "signal",
                "message": f"{signal.signal_type} signal on {self.asset_id} ({signal.timeframe}) - {signal.confidence:.2f} confidence",
                "importance": importance,
                "reasons": signal.reasons,
                "timeframe": signal.timeframe,
                "expiration": signal.expiration
            })
        
        # Add brain-specific advisories
        for brain_name, signals in brain_signals.items():
            for signal in signals:
                if signal.confidence > 0.75:
                    self.advisories.append({
                        "type": "strategy",
                        "message": f"{brain_name} strategy detected {signal.signal_type} opportunity",
                        "importance": "medium" if signal.confidence > 0.85 else "low",
                        "strategy": brain_name,
                        "reasons": signal.reasons[:2]
                    })
    
    def _update_recent_signals(self, signals: List[TradeSignal]) -> None:
        """Update the list of recent signals."""
        # Add new signals
        self.recent_signals.extend(signals)
        
        # Keep only signals from the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_signals = [s for s in self.recent_signals 
                             if s.timestamp > cutoff_time]
        
        # Cap the list at 50 signals
        if len(self.recent_signals) > 50:
            self.recent_signals = sorted(
                self.recent_signals, 
                key=lambda s: s.timestamp, 
                reverse=True
            )[:50]
    
    async def get_advisories(self) -> List[Dict[str, Any]]:
        """Get current advisories for this asset."""
        return self.advisories
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for this asset's strategies."""
        metrics = {}
        for name, performance in self.strategy_performance.items():
            metrics[name] = {
                "win_rate": performance.win_rate,
                "profit_factor": performance.profit_factor,
                "avg_profit": performance.avg_profit,
                "avg_loss": performance.avg_loss,
                "sharpe_ratio": performance.sharpe_ratio,
                "trades_count": performance.trades_count,
                "current_weight": self.strategy_weights.get(name, 0)
            }
        return metrics
    
    async def get_signals(self, 
                        timeframe: str = None,
                        signal_type: str = None,
                        min_confidence: float = 0.0
                        ) -> List[TradeSignal]:
        """
        Get filtered signals for this asset.
        
        Args:
            timeframe: Optional timeframe filter
            signal_type: Optional signal type filter (BUY/SELL)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of trade signals matching the criteria
        """
        filtered_signals = self.recent_signals.copy()
        
        # Apply filters
        if timeframe:
            filtered_signals = [s for s in filtered_signals 
                              if s.timeframe == timeframe]
        
        if signal_type:
            filtered_signals = [s for s in filtered_signals 
                              if s.signal_type == signal_type]
        
        if min_confidence > 0:
            filtered_signals = [s for s in filtered_signals 
                              if s.confidence >= min_confidence]
        
        return sorted(filtered_signals, key=lambda s: s.timestamp, reverse=True)
    
    async def reset_brain(self, brain_name: str) -> bool:
        """
        Reset a specific strategy brain.
        
        Args:
            brain_name: Name of the brain to reset
            
        Returns:
            True if reset successful, False otherwise
        """
        if brain_name not in self.strategy_brains:
            raise BrainNotFoundError(f"Brain {brain_name} not found")
        
        try:
            await self.strategy_brains[brain_name].reset()
            self.logger.info(f"Reset brain {brain_name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset brain {brain_name}: {str(e)}")
            return False
    
    async def optimize_brain(self, brain_name: str) -> bool:
        """
        Optimize a specific strategy brain.
        
        Args:
            brain_name: Name of the brain to optimize
            
        Returns:
            True if optimization successful, False otherwise
        """
        if brain_name not in self.strategy_brains:
            raise BrainNotFoundError(f"Brain {brain_name} not found")
        
        try:
            await self.strategy_brains[brain_name].optimize()
            self.logger.info(f"Optimized brain {brain_name}")
            
            # Update weights after optimization
            await self._compute_strategy_weights()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize brain {brain_name}: {str(e)}")
            return False
    
    async def stop(self) -> None:
        """Stop the asset council and all strategy brains."""
        try:
            stop_tasks = []
            for name, brain in self.strategy_brains.items():
                stop_tasks.append(brain.stop())
            
            await gather_with_concurrency(10, *stop_tasks)
            self.running = False
            self.logger.info(f"Stopped Asset Council for {self.platform}:{self.asset_id}")
        
        except Exception as e:
            self.logger.error(f"Error stopping Asset Council: {str(e)}")
            raise AssetCouncilError(f"Stop error: {str(e)}")