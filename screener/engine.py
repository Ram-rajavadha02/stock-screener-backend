"""
Screening engine.

Evaluates a list of Condition rules against every stock in the mock database
and returns the ones that satisfy the criteria (match='all' → AND, 'any' → OR).
"""
import pandas as pd
from indicators.calculator import get_series_tf
from models.schemas import Condition, ScreenRequest, ScreenResponse, StockMatch


# ── Operator implementations ──────────────────────────────────────────────────

def _op_gt(a: pd.Series, b: pd.Series)  -> pd.Series: return a > b
def _op_lt(a: pd.Series, b: pd.Series)  -> pd.Series: return a < b
def _op_gte(a: pd.Series, b: pd.Series) -> pd.Series: return a >= b
def _op_lte(a: pd.Series, b: pd.Series) -> pd.Series: return a <= b
def _op_eq(a: pd.Series, b: pd.Series)  -> pd.Series: return (a - b).abs() < 1e-6

def _op_crosses_above(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crossed above b (was below on prior bar)."""
    return (a.shift(1) < b.shift(1)) & (a >= b)

def _op_crosses_below(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on the bar where a crossed below b (was above on prior bar)."""
    return (a.shift(1) > b.shift(1)) & (a <= b)


OPERATORS: dict[str, callable] = {
    ">":             _op_gt,
    "<":             _op_lt,
    ">=":            _op_gte,
    "<=":            _op_lte,
    "=":             _op_eq,
    "crosses_above": _op_crosses_above,
    "crosses_below": _op_crosses_below,
}


# ── Engine ────────────────────────────────────────────────────────────────────

class ScreenerEngine:
    def run(self, request: ScreenRequest, all_data: dict[str, pd.DataFrame]) -> ScreenResponse:
        matched: list[StockMatch] = []

        for symbol, df in all_data.items():
            try:
                passed = self._evaluate(df, request)
            except Exception:
                passed = False

            if passed:
                last  = df.iloc[-1]
                prev  = df.iloc[-2] if len(df) > 1 else last
                chg   = ((last["close"] - prev["close"]) / prev["close"]) * 100
                matched.append(StockMatch(
                    symbol=symbol,
                    last_close=round(float(last["close"]), 2),
                    last_volume=int(last["volume"]),
                    change_pct=round(float(chg), 2),
                ))

        return ScreenResponse(
            matched_stocks=matched,
            total_matched=len(matched),
            total_screened=len(all_data),
            conditions_used=request.conditions,
            match_mode=request.match,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _evaluate(self, df: pd.DataFrame, request: ScreenRequest) -> bool:
        results: list[bool] = []

        for cond in request.conditions:
            result = self._check_condition(df, cond)
            results.append(result)

        if not results:
            return False

        return all(results) if request.match == "all" else any(results)

    def _check_condition(self, df: pd.DataFrame, cond: Condition) -> bool:
        op_func = OPERATORS.get(cond.operator)
        if op_func is None:
            raise ValueError(f"Unknown operator: '{cond.operator}'")

        lhs_tf = cond.lhs_timeframe
        rhs_tf = cond.rhs_timeframe

        lhs = get_series_tf(df, cond.indicator, lhs_tf)
        rhs = get_series_tf(df, cond.value,     rhs_tf)

        result_series: pd.Series = op_func(lhs, rhs)

        # Check only the most recent valid (non-NaN) value
        valid = result_series.dropna()
        if valid.empty:
            return False
        return bool(valid.iloc[-1])
