"""
Strategies
"""

from trademl.strategies.meta_labeling_primary import (
    vix_change_strategy, crossover_strategy, bbands_Strategy)
from trademl.strategies.test_strategies import (
    TestStrategy, TestStrategyIndicators, SmaOptimizationStrategy)
from trademl.strategies.benchmark_strategies import (
    BuyAndHold_Buy, BuyAndHold_Target, BuyAndHold_More, BuyAndHold_More_Fund, PandasData)
import trademl.strategies.benchmark_strategies as benchmark_strategies

