from typing import Any, Dict, List, Optional
import pandas as pd
import quantstats as qs

class PerformanceAnalyzer:
    """Performance metrics calculator leveraging quantstats when available."""

    def compute(
        self,
        trades: pd.DataFrame,
        equity_curve: Optional[pd.Series] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        metrics = metrics or ["total_return", "win_rate", "sharpe"]

        result: Dict[str, Any] = {m: 0 for m in metrics}

        if not trades.empty and "profit_pct" in trades.columns:
            if "total_return" in metrics:
                result["total_return"] = trades["profit_pct"].sum()
            if "win_rate" in metrics:
                result["win_rate"] = float((trades["profit_pct"] > 0).mean() * 100)

        if equity_curve is not None and not equity_curve.empty:
            returns = equity_curve.pct_change().dropna()
            if "sharpe" in metrics:
                result["sharpe"] = float(qs.stats.sharpe(returns))
            if "max_drawdown" in metrics:
                result["max_drawdown"] = float(qs.stats.max_drawdown(returns))
            if "cagr" in metrics:
                result["cagr"] = float(qs.stats.cagr(returns))
            if "sortino" in metrics:
                result["sortino"] = float(qs.stats.sortino(returns))

        return result

    def generate_report(self, equity_curve: pd.Series, output_file: str) -> str:
        """Generate an HTML performance report using quantstats."""
        returns = equity_curve.pct_change().dropna()
        qs.reports.html(returns, output=output_file, title="Backtest Report")
        return output_file

class Performance:
    """Wrapper providing async API for performance analysis."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.analyzer = PerformanceAnalyzer()

    async def analyze(self, backtest_result: Dict[str, Any], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        trades = pd.DataFrame(backtest_result.get("trades", []))
        equity_df = pd.DataFrame(backtest_result.get("equity_curve", []))
        equity_curve = equity_df["equity"] if not equity_df.empty and "equity" in equity_df.columns else None
        return self.analyzer.compute(trades, equity_curve, metrics)
