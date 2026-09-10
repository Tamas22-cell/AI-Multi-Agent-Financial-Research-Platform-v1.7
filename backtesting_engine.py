import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import yfinance as yf


class BacktestingEngine:
    """
    AI Multi-Agent Financial Research Platform
    v1.7 Backtesting Engine

    Capabilities:
    - Real historical signal evaluation
    - Time-aware 1D / 7D / 30D horizons
    - BUY / HOLD / AVOID performance analysis
    - SPY / BTC-USD benchmark comparison
    - Agent-by-agent accuracy analysis
    - Benchmark snapshot quality validation
    - Research equity curve
    - Cumulative strategy performance
    - Peak equity
    - Drawdown
    - Maximum drawdown
    - Win rate
    - Average win / loss
    - Profit factor
    - Exposure rate
    - Strategy volatility
    - Sharpe-like performance metric
    - JSON and TXT backtest reporting

    Important:
    Only real signals stored in signal_history.json
    are evaluated.

    No synthetic historical signals are created.

    Cumulative performance uses completed 1D observations only.

    Strategy interpretation:
    BUY   -> SPY exposure
    HOLD  -> cash
    AVOID -> cash

    AVOID is NOT treated as a short position.
    """

    HORIZONS = {
        "1D": 1,
        "7D": 7,
        "30D": 30,
    }

    BENCHMARK_THRESHOLDS = {
        "SPY": {
            "1D": 0.5,
            "7D": 1.5,
            "30D": 3.0,
        },
        "BTC-USD": {
            "1D": 1.0,
            "7D": 3.0,
            "30D": 7.0,
        },
    }

    AGENT_BENCHMARKS = {
        "macro": "SPY",
        "stock": "SPY",
        "crypto": "BTC-USD",
        "onchain": "BTC-USD",
        "derivatives": "BTC-USD",
        "technical": "SPY",
        "news": "SPY",
        "geopolitical": "SPY",
        "risk": "SPY",
    }

    DECISIONS = [
        "BUY",
        "HOLD",
        "AVOID",
    ]

    MAX_ENTRY_SNAPSHOT_AGE_DAYS = 7

    STARTING_EQUITY = 100.0

    TRADING_PERIODS_PER_YEAR = 252

    def __init__(
        self,
        history_file: str = "reports/output/signal_history.json",
        output_json: str = "reports/output/backtest_results.json",
        output_txt: str = "reports/output/backtest_report.txt",
    ):
        self.history_file = Path(history_file)
        self.output_json = Path(output_json)
        self.output_txt = Path(output_txt)

        self.output_json.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.output_txt.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._price_cache: Dict[
            str,
            Dict[str, float],
        ] = {}

    # ============================================================
    # HISTORY
    # ============================================================

    def load_history(
        self,
    ) -> List[Dict[str, Any]]:

        if not self.history_file.exists():
            print(
                f"[BACKTEST] Signal history not found: "
                f"{self.history_file}"
            )
            return []

        try:
            with self.history_file.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(
                data,
                list,
            ):
                print(
                    "[BACKTEST] signal_history.json must "
                    "contain a JSON list."
                )
                return []

            return data

        except Exception as exc:
            print(
                f"[BACKTEST] Could not read signal "
                f"history: {exc}"
            )
            return []

    # ============================================================
    # DATE / TIME
    # ============================================================

    @staticmethod
    def _parse_date(
        value: str,
    ) -> Optional[datetime]:

        try:
            return datetime.strptime(
                value,
                "%Y-%m-%d",
            )

        except Exception:
            return None

    @staticmethod
    def _parse_signal_timestamp(
        timestamp_value: Any,
        fallback_date: Optional[str] = None,
    ) -> Optional[datetime]:

        if timestamp_value:
            try:
                parsed = datetime.fromisoformat(
                    str(
                        timestamp_value
                    ).strip()
                )

                if parsed.tzinfo is not None:
                    parsed = parsed.replace(
                        tzinfo=None
                    )

                return parsed

            except Exception:
                pass

        if fallback_date:
            try:
                return datetime.strptime(
                    fallback_date,
                    "%Y-%m-%d",
                )

            except Exception:
                return None

        return None

    @staticmethod
    def _parse_market_timestamp(
        timestamp_value: Any,
    ) -> Optional[datetime]:

        if not timestamp_value:
            return None

        try:
            parsed = datetime.fromisoformat(
                str(
                    timestamp_value
                ).strip()
            )

            if parsed.tzinfo is not None:
                parsed = parsed.replace(
                    tzinfo=None
                )

            return parsed

        except Exception:
            return None

    @staticmethod
    def _now() -> datetime:
        return datetime.now()

    # ============================================================
    # HORIZON MATURITY
    # ============================================================

    def _horizon_status(
        self,
        signal_timestamp: datetime,
        horizon_days: int,
    ) -> Dict[str, Any]:

        target_timestamp = (
            signal_timestamp
            + timedelta(
                days=horizon_days
            )
        )

        now = self._now()

        elapsed_seconds = (
            now
            - signal_timestamp
        ).total_seconds()

        required_seconds = (
            timedelta(
                days=horizon_days
            ).total_seconds()
        )

        remaining_seconds = max(
            required_seconds
            - elapsed_seconds,
            0.0,
        )

        return {
            "mature": (
                now
                >= target_timestamp
            ),
            "signal_timestamp": (
                signal_timestamp.isoformat()
            ),
            "target_timestamp": (
                target_timestamp.isoformat()
            ),
            "remaining_hours": round(
                remaining_seconds
                / 3600.0,
                2,
            ),
        }

    # ============================================================
    # ENTRY SNAPSHOT VALIDATION
    # ============================================================

    def _validate_entry_snapshot(
        self,
        signal_timestamp: datetime,
        market_timestamp_value: Any,
    ) -> Dict[str, Any]:

        market_timestamp = (
            self._parse_market_timestamp(
                market_timestamp_value
            )
        )

        if market_timestamp is None:
            return {
                "status": "UNKNOWN",
                "valid": True,
                "age_days": None,
                "message": (
                    "Benchmark timestamp unavailable "
                    "or could not be parsed."
                ),
            }

        age_days = (
            signal_timestamp.date()
            - market_timestamp.date()
        ).days

        if age_days < 0:
            return {
                "status": "INVALID_FUTURE",
                "valid": False,
                "age_days": age_days,
                "message": (
                    "Benchmark snapshot is dated "
                    "after the signal."
                ),
            }

        if (
            age_days
            > self.MAX_ENTRY_SNAPSHOT_AGE_DAYS
        ):
            return {
                "status": "STALE",
                "valid": False,
                "age_days": age_days,
                "message": (
                    "Benchmark snapshot is older "
                    "than the accepted age window."
                ),
            }

        if age_days == 0:
            return {
                "status": "CURRENT",
                "valid": True,
                "age_days": age_days,
                "message": (
                    "Benchmark snapshot is from "
                    "the signal date."
                ),
            }

        return {
            "status": (
                "VALID_PREVIOUS_MARKET_DATA"
            ),
            "valid": True,
            "age_days": age_days,
            "message": (
                "Benchmark snapshot predates the "
                "signal but remains within the "
                "accepted market-data window."
            ),
        }

    # ============================================================
    # MARKET DATA
    # ============================================================

    def _download_price_window(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, float]:

        cache_key = (
            f"{symbol}:"
            f"{start_date.strftime('%Y-%m-%d')}:"
            f"{end_date.strftime('%Y-%m-%d')}"
        )

        if cache_key in self._price_cache:
            return self._price_cache[
                cache_key
            ]

        result: Dict[
            str,
            float,
        ] = {}

        try:
            ticker = yf.Ticker(
                symbol
            )

            data = ticker.history(
                start=start_date.strftime(
                    "%Y-%m-%d"
                ),
                end=end_date.strftime(
                    "%Y-%m-%d"
                ),
                interval="1d",
                auto_adjust=False,
            )

            if (
                data is None
                or data.empty
            ):
                self._price_cache[
                    cache_key
                ] = result

                return result

            for index, row in data.iterrows():
                try:
                    close_price = float(
                        row["Close"]
                    )

                    if close_price <= 0:
                        continue

                    result[
                        index.strftime(
                            "%Y-%m-%d"
                        )
                    ] = close_price

                except Exception:
                    continue

        except Exception as exc:
            print(
                f"[BACKTEST] Price download "
                f"failed for {symbol}: {exc}"
            )

        self._price_cache[
            cache_key
        ] = result

        return result

    def _get_forward_price(
        self,
        symbol: str,
        target_timestamp: datetime,
    ) -> Optional[Dict[str, Any]]:

        if (
            target_timestamp
            > self._now()
        ):
            return None

        target_date = datetime(
            target_timestamp.year,
            target_timestamp.month,
            target_timestamp.day,
        )

        search_end = min(
            target_date
            + timedelta(days=8),
            self._now()
            + timedelta(days=1),
        )

        prices = (
            self._download_price_window(
                symbol=symbol,
                start_date=target_date,
                end_date=(
                    search_end
                    + timedelta(days=1)
                ),
            )
        )

        if not prices:
            return None

        for date_key in sorted(
            prices.keys()
        ):
            price_date = (
                self._parse_date(
                    date_key
                )
            )

            if price_date is None:
                continue

            if (
                price_date.date()
                >= target_date.date()
            ):
                return {
                    "date": date_key,
                    "price": round(
                        prices[
                            date_key
                        ],
                        4,
                    ),
                }

        return None

    # ============================================================
    # RETURN
    # ============================================================

    @staticmethod
    def _calculate_return_pct(
        entry_price: float,
        exit_price: float,
    ) -> Optional[float]:

        try:
            if entry_price <= 0:
                return None

            result = (
                (
                    exit_price
                    - entry_price
                )
                / entry_price
            ) * 100.0

            return round(
                result,
                4,
            )

        except Exception:
            return None

    # ============================================================
    # SIGNAL CLASSIFICATION
    # ============================================================

    def _get_threshold(
        self,
        benchmark: str,
        horizon_name: str,
    ) -> float:

        return float(
            self.BENCHMARK_THRESHOLDS
            .get(
                benchmark,
                {},
            )
            .get(
                horizon_name,
                0.0,
            )
        )

    def _evaluate_direction(
        self,
        signal: str,
        return_pct: float,
        threshold: float,
    ) -> Dict[str, Any]:

        if (
            return_pct
            > threshold
        ):
            actual_direction = (
                "bullish"
            )

        elif (
            return_pct
            < -threshold
        ):
            actual_direction = (
                "bearish"
            )

        else:
            actual_direction = (
                "neutral"
            )

        signal_map = {
            "buy": "bullish",
            "bullish": "bullish",
            "hold": "neutral",
            "neutral": "neutral",
            "avoid": "bearish",
            "sell": "bearish",
            "bearish": "bearish",
        }

        predicted_direction = (
            signal_map.get(
                str(
                    signal
                ).strip().lower(),
                "unknown",
            )
        )

        if (
            predicted_direction
            == "unknown"
        ):
            correct = None

        else:
            correct = (
                predicted_direction
                == actual_direction
            )

        return {
            "predicted_direction": (
                predicted_direction
            ),
            "actual_direction": (
                actual_direction
            ),
            "correct": correct,
        }

    # ============================================================
    # BENCHMARK HORIZON
    # ============================================================

    def _evaluate_benchmark_horizon(
        self,
        benchmark: str,
        entry_price: float,
        signal_timestamp: datetime,
        horizon_name: str,
        horizon_days: int,
        entry_quality: Dict[str, Any],
    ) -> Dict[str, Any]:

        maturity = (
            self._horizon_status(
                signal_timestamp,
                horizon_days,
            )
        )

        target_timestamp = (
            signal_timestamp
            + timedelta(
                days=horizon_days
            )
        )

        base_result = {
            "target_timestamp": (
                target_timestamp.isoformat()
            ),
            "entry_price": round(
                entry_price,
                4,
            ),
            "exit_date": None,
            "exit_price": None,
            "return_pct": None,
            "remaining_hours": (
                maturity[
                    "remaining_hours"
                ]
            ),
            "entry_quality": (
                entry_quality[
                    "status"
                ]
            ),
        }

        if not entry_quality.get(
            "valid",
            True,
        ):
            base_result[
                "status"
            ] = "INVALID_ENTRY"

            return base_result

        if not maturity["mature"]:
            base_result[
                "status"
            ] = "PENDING"

            return base_result

        exit_data = (
            self._get_forward_price(
                benchmark,
                target_timestamp,
            )
        )

        if exit_data is None:
            base_result[
                "status"
            ] = "PENDING_MARKET_DATA"

            base_result[
                "remaining_hours"
            ] = 0.0

            return base_result

        exit_price = float(
            exit_data["price"]
        )

        return_pct = (
            self._calculate_return_pct(
                entry_price,
                exit_price,
            )
        )

        base_result.update(
            {
                "status": "COMPLETED",
                "exit_date": (
                    exit_data[
                        "date"
                    ]
                ),
                "exit_price": round(
                    exit_price,
                    4,
                ),
                "return_pct": (
                    return_pct
                ),
                "remaining_hours": 0.0,
            }
        )

        return base_result

    # ============================================================
    # RECORD EVALUATION
    # ============================================================

    def _evaluate_record(
        self,
        record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        date_value = record.get(
            "date"
        )

        if not date_value:
            return None

        signal_timestamp = (
            self._parse_signal_timestamp(
                record.get(
                    "timestamp"
                ),
                fallback_date=str(
                    date_value
                ),
            )
        )

        if signal_timestamp is None:
            return None

        decision = str(
            record.get(
                "decision",
                "UNKNOWN",
            )
        ).upper()

        benchmarks = record.get(
            "benchmarks",
            {},
        )

        agents = record.get(
            "agents",
            {},
        )

        result: Dict[
            str,
            Any,
        ] = {
            "timestamp": record.get(
                "timestamp"
            ),
            "date": date_value,
            "decision": decision,
            "market_regime": (
                record.get(
                    "market_regime"
                )
            ),
            "market_score": (
                record.get(
                    "market_score"
                )
            ),
            "average_confidence": (
                record.get(
                    "average_confidence"
                )
            ),
            "benchmarks": {},
            "decision_evaluation": {},
            "agent_evaluation": {},
            "data_quality": {},
        }

        for benchmark in [
            "SPY",
            "BTC-USD",
        ]:
            benchmark_info = (
                benchmarks.get(
                    benchmark,
                    {},
                )
            )

            entry_price = (
                benchmark_info.get(
                    "price"
                )
            )

            if entry_price is None:
                continue

            try:
                entry_price = float(
                    entry_price
                )

            except Exception:
                continue

            entry_timestamp = (
                benchmark_info.get(
                    "price_timestamp"
                )
            )

            entry_quality = (
                self._validate_entry_snapshot(
                    signal_timestamp,
                    entry_timestamp,
                )
            )

            result[
                "data_quality"
            ][benchmark] = (
                entry_quality
            )

            benchmark_result = {
                "entry_price": round(
                    entry_price,
                    4,
                ),
                "entry_timestamp": (
                    entry_timestamp
                ),
                "source": (
                    benchmark_info.get(
                        "source"
                    )
                ),
                "entry_quality": (
                    entry_quality
                ),
                "horizons": {},
            }

            for (
                horizon_name,
                horizon_days,
            ) in self.HORIZONS.items():

                benchmark_result[
                    "horizons"
                ][horizon_name] = (
                    self
                    ._evaluate_benchmark_horizon(
                        benchmark=benchmark,
                        entry_price=entry_price,
                        signal_timestamp=(
                            signal_timestamp
                        ),
                        horizon_name=(
                            horizon_name
                        ),
                        horizon_days=(
                            horizon_days
                        ),
                        entry_quality=(
                            entry_quality
                        ),
                    )
                )

            result[
                "benchmarks"
            ][benchmark] = (
                benchmark_result
            )

        spy_data = (
            result[
                "benchmarks"
            ].get(
                "SPY"
            )
        )

        for horizon_name in self.HORIZONS:

            if not spy_data:
                result[
                    "decision_evaluation"
                ][horizon_name] = {
                    "status": "PENDING",
                    "correct": None,
                }
                continue

            horizon_data = (
                spy_data[
                    "horizons"
                ].get(
                    horizon_name,
                    {},
                )
            )

            if (
                horizon_data.get(
                    "status"
                )
                != "COMPLETED"
            ):
                result[
                    "decision_evaluation"
                ][horizon_name] = {
                    "status": (
                        horizon_data.get(
                            "status",
                            "PENDING",
                        )
                    ),
                    "correct": None,
                    "remaining_hours": (
                        horizon_data.get(
                            "remaining_hours"
                        )
                    ),
                }
                continue

            return_pct = (
                horizon_data.get(
                    "return_pct"
                )
            )

            threshold = (
                self._get_threshold(
                    "SPY",
                    horizon_name,
                )
            )

            directional = (
                self._evaluate_direction(
                    signal=decision,
                    return_pct=return_pct,
                    threshold=threshold,
                )
            )

            result[
                "decision_evaluation"
            ][horizon_name] = {
                "status": "COMPLETED",
                "benchmark": "SPY",
                "threshold_pct": threshold,
                "return_pct": return_pct,
                **directional,
            }

        for (
            agent_name,
            agent_data,
        ) in agents.items():

            benchmark = (
                self.AGENT_BENCHMARKS.get(
                    agent_name,
                    "SPY",
                )
            )

            signal = agent_data.get(
                "signal",
                "neutral",
            )

            agent_result = {
                "signal": signal,
                "score": agent_data.get(
                    "score"
                ),
                "confidence": (
                    agent_data.get(
                        "confidence"
                    )
                ),
                "benchmark": benchmark,
                "horizons": {},
            }

            benchmark_data = (
                result[
                    "benchmarks"
                ].get(
                    benchmark
                )
            )

            for horizon_name in self.HORIZONS:

                if benchmark_data is None:
                    agent_result[
                        "horizons"
                    ][horizon_name] = {
                        "status": "PENDING",
                        "correct": None,
                    }
                    continue

                horizon_data = (
                    benchmark_data[
                        "horizons"
                    ].get(
                        horizon_name,
                        {},
                    )
                )

                if (
                    horizon_data.get(
                        "status"
                    )
                    != "COMPLETED"
                ):
                    agent_result[
                        "horizons"
                    ][horizon_name] = {
                        "status": (
                            horizon_data.get(
                                "status",
                                "PENDING",
                            )
                        ),
                        "correct": None,
                        "remaining_hours": (
                            horizon_data.get(
                                "remaining_hours"
                            )
                        ),
                    }
                    continue

                return_pct = (
                    horizon_data.get(
                        "return_pct"
                    )
                )

                threshold = (
                    self._get_threshold(
                        benchmark,
                        horizon_name,
                    )
                )

                directional = (
                    self._evaluate_direction(
                        signal=signal,
                        return_pct=return_pct,
                        threshold=threshold,
                    )
                )

                agent_result[
                    "horizons"
                ][horizon_name] = {
                    "status": "COMPLETED",
                    "threshold_pct": (
                        threshold
                    ),
                    "return_pct": (
                        return_pct
                    ),
                    **directional,
                }

            result[
                "agent_evaluation"
            ][agent_name] = (
                agent_result
            )

        return result

    # ============================================================
    # GENERIC PERFORMANCE
    # ============================================================

    @staticmethod
    def _performance_block(
        evaluations: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        completed = 0
        correct = 0
        pending = 0
        invalid = 0

        returns: List[
            float
        ] = []

        for evaluation in evaluations:

            status = evaluation.get(
                "status"
            )

            if status == "COMPLETED":
                completed += 1

                if (
                    evaluation.get(
                        "correct"
                    )
                    is True
                ):
                    correct += 1

                return_pct = (
                    evaluation.get(
                        "return_pct"
                    )
                )

                if return_pct is not None:
                    try:
                        returns.append(
                            float(
                                return_pct
                            )
                        )
                    except Exception:
                        pass

            elif status == "INVALID_ENTRY":
                invalid += 1

            else:
                pending += 1

        incorrect = (
            completed
            - correct
        )

        accuracy = (
            round(
                (
                    correct
                    / completed
                )
                * 100.0,
                2,
            )
            if completed > 0
            else None
        )

        if returns:
            average_return = round(
                mean(
                    returns
                ),
                4,
            )

            best_return = round(
                max(
                    returns
                ),
                4,
            )

            worst_return = round(
                min(
                    returns
                ),
                4,
            )

        else:
            average_return = None
            best_return = None
            worst_return = None

        return {
            "completed": completed,
            "correct": correct,
            "incorrect": incorrect,
            "pending": pending,
            "invalid": invalid,
            "accuracy_pct": accuracy,
            "average_return_pct": (
                average_return
            ),
            "best_return_pct": (
                best_return
            ),
            "worst_return_pct": (
                worst_return
            ),
        }

    # ============================================================
    # BUY / HOLD / AVOID PERFORMANCE
    # ============================================================

    def _build_decision_stats(
        self,
        evaluated_records: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        decision_stats = {}

        for decision_name in self.DECISIONS:

            decision_records = [
                record
                for record
                in evaluated_records
                if str(
                    record.get(
                        "decision",
                        ""
                    )
                ).upper()
                == decision_name
            ]

            decision_stats[
                decision_name
            ] = {
                "total_signals": len(
                    decision_records
                ),
                "horizons": {},
            }

            for horizon_name in self.HORIZONS:

                evaluations = []

                for record in decision_records:

                    evaluations.append(
                        record.get(
                            "decision_evaluation",
                            {},
                        ).get(
                            horizon_name,
                            {},
                        )
                    )

                decision_stats[
                    decision_name
                ][
                    "horizons"
                ][horizon_name] = (
                    self._performance_block(
                        evaluations
                    )
                )

        return decision_stats

    # ============================================================
    # EQUITY HELPERS
    # ============================================================

    @staticmethod
    def _apply_return(
        equity: float,
        return_pct: float,
    ) -> float:

        return (
            equity
            * (
                1.0
                + (
                    return_pct
                    / 100.0
                )
            )
        )

    @staticmethod
    def _calculate_drawdown_pct(
        equity: float,
        peak_equity: float,
    ) -> float:

        if peak_equity <= 0:
            return 0.0

        return round(
            (
                (
                    equity
                    - peak_equity
                )
                / peak_equity
            )
            * 100.0,
            4,
        )

    @staticmethod
    def _strategy_return(
        decision: str,
        market_return_pct: float,
    ) -> float:

        decision = str(
            decision
        ).upper()

        if decision == "BUY":
            return market_return_pct

        return 0.0

    # ============================================================
    # PROFESSIONAL STRATEGY STATISTICS
    # ============================================================

    def _build_strategy_statistics(
        self,
        strategy_returns: List[float],
        completed_periods: int,
        buy_periods: int,
    ) -> Dict[str, Any]:

        if completed_periods == 0:
            return {
                "status": "PENDING",
                "samples": 0,
                "winning_periods": 0,
                "losing_periods": 0,
                "flat_periods": 0,
                "win_rate_pct": None,
                "average_return_pct": None,
                "average_win_pct": None,
                "average_loss_pct": None,
                "best_period_pct": None,
                "worst_period_pct": None,
                "gross_profit_pct": None,
                "gross_loss_pct": None,
                "profit_factor": None,
                "exposure_rate_pct": None,
                "volatility_pct": None,
                "annualized_volatility_pct": None,
                "sharpe_like": None,
            }

        winning_returns = [
            value
            for value in strategy_returns
            if value > 0
        ]

        losing_returns = [
            value
            for value in strategy_returns
            if value < 0
        ]

        flat_returns = [
            value
            for value in strategy_returns
            if value == 0
        ]

        winning_periods = len(
            winning_returns
        )

        losing_periods = len(
            losing_returns
        )

        flat_periods = len(
            flat_returns
        )

        active_outcomes = (
            winning_periods
            + losing_periods
        )

        if active_outcomes > 0:
            win_rate = (
                winning_periods
                / active_outcomes
            ) * 100.0

        else:
            win_rate = None

        average_return = mean(
            strategy_returns
        )

        average_win = (
            mean(
                winning_returns
            )
            if winning_returns
            else None
        )

        average_loss = (
            mean(
                losing_returns
            )
            if losing_returns
            else None
        )

        best_period = max(
            strategy_returns
        )

        worst_period = min(
            strategy_returns
        )

        gross_profit = sum(
            winning_returns
        )

        gross_loss = abs(
            sum(
                losing_returns
            )
        )

        if gross_loss > 0:
            profit_factor = (
                gross_profit
                / gross_loss
            )

        elif gross_profit > 0:
            profit_factor = float(
                "inf"
            )

        else:
            profit_factor = None

        exposure_rate = (
            buy_periods
            / completed_periods
        ) * 100.0

        volatility = None
        annualized_volatility = None
        sharpe_like = None

        if len(
            strategy_returns
        ) >= 2:

            try:
                volatility = stdev(
                    strategy_returns
                )

                annualized_volatility = (
                    volatility
                    * math.sqrt(
                        self.TRADING_PERIODS_PER_YEAR
                    )
                )

                if volatility > 0:
                    daily_mean = mean(
                        strategy_returns
                    )

                    sharpe_like = (
                        daily_mean
                        / volatility
                    ) * math.sqrt(
                        self.TRADING_PERIODS_PER_YEAR
                    )

            except Exception:
                pass

        if (
            profit_factor is not None
            and math.isinf(
                profit_factor
            )
        ):
            profit_factor_output = (
                "INF"
            )

        elif profit_factor is not None:
            profit_factor_output = round(
                profit_factor,
                4,
            )

        else:
            profit_factor_output = None

        return {
            "status": "AVAILABLE",
            "samples": (
                completed_periods
            ),
            "winning_periods": (
                winning_periods
            ),
            "losing_periods": (
                losing_periods
            ),
            "flat_periods": (
                flat_periods
            ),
            "win_rate_pct": (
                round(
                    win_rate,
                    2,
                )
                if win_rate is not None
                else None
            ),
            "average_return_pct": round(
                average_return,
                4,
            ),
            "average_win_pct": (
                round(
                    average_win,
                    4,
                )
                if average_win is not None
                else None
            ),
            "average_loss_pct": (
                round(
                    average_loss,
                    4,
                )
                if average_loss is not None
                else None
            ),
            "best_period_pct": round(
                best_period,
                4,
            ),
            "worst_period_pct": round(
                worst_period,
                4,
            ),
            "gross_profit_pct": round(
                gross_profit,
                4,
            ),
            "gross_loss_pct": round(
                gross_loss,
                4,
            ),
            "profit_factor": (
                profit_factor_output
            ),
            "exposure_rate_pct": round(
                exposure_rate,
                2,
            ),
            "volatility_pct": (
                round(
                    volatility,
                    4,
                )
                if volatility is not None
                else None
            ),
            "annualized_volatility_pct": (
                round(
                    annualized_volatility,
                    4,
                )
                if annualized_volatility is not None
                else None
            ),
            "sharpe_like": (
                round(
                    sharpe_like,
                    4,
                )
                if sharpe_like is not None
                else None
            ),
        }

    # ============================================================
    # STRATEGY EQUITY CURVE
    # ============================================================

    def _build_strategy_equity_curve(
        self,
        evaluated_records: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        strategy_equity = (
            self.STARTING_EQUITY
        )

        benchmark_equity = (
            self.STARTING_EQUITY
        )

        strategy_peak = (
            self.STARTING_EQUITY
        )

        benchmark_peak = (
            self.STARTING_EQUITY
        )

        strategy_max_drawdown = 0.0
        benchmark_max_drawdown = 0.0

        curve = []

        strategy_returns = []

        completed_periods = 0
        buy_periods = 0
        cash_periods = 0

        sorted_records = sorted(
            evaluated_records,
            key=lambda record: str(
                record.get(
                    "timestamp",
                    ""
                )
            ),
        )

        for record in sorted_records:

            benchmark_data = (
                record.get(
                    "benchmarks",
                    {}
                )
                .get(
                    "SPY",
                    {}
                )
                .get(
                    "horizons",
                    {}
                )
                .get(
                    "1D",
                    {}
                )
            )

            if (
                benchmark_data.get(
                    "status"
                )
                != "COMPLETED"
            ):
                continue

            market_return = (
                benchmark_data.get(
                    "return_pct"
                )
            )

            if market_return is None:
                continue

            try:
                market_return = float(
                    market_return
                )

            except Exception:
                continue

            decision = str(
                record.get(
                    "decision",
                    "HOLD",
                )
            ).upper()

            strategy_return = (
                self._strategy_return(
                    decision,
                    market_return,
                )
            )

            strategy_returns.append(
                strategy_return
            )

            strategy_equity = (
                self._apply_return(
                    strategy_equity,
                    strategy_return,
                )
            )

            benchmark_equity = (
                self._apply_return(
                    benchmark_equity,
                    market_return,
                )
            )

            strategy_peak = max(
                strategy_peak,
                strategy_equity,
            )

            benchmark_peak = max(
                benchmark_peak,
                benchmark_equity,
            )

            strategy_drawdown = (
                self._calculate_drawdown_pct(
                    strategy_equity,
                    strategy_peak,
                )
            )

            benchmark_drawdown = (
                self._calculate_drawdown_pct(
                    benchmark_equity,
                    benchmark_peak,
                )
            )

            strategy_max_drawdown = min(
                strategy_max_drawdown,
                strategy_drawdown,
            )

            benchmark_max_drawdown = min(
                benchmark_max_drawdown,
                benchmark_drawdown,
            )

            completed_periods += 1

            if decision == "BUY":
                buy_periods += 1

            else:
                cash_periods += 1

            curve.append(
                {
                    "date": record.get(
                        "date"
                    ),
                    "timestamp": record.get(
                        "timestamp"
                    ),
                    "decision": decision,
                    "market_return_pct": round(
                        market_return,
                        4,
                    ),
                    "strategy_return_pct": round(
                        strategy_return,
                        4,
                    ),
                    "strategy_equity": round(
                        strategy_equity,
                        4,
                    ),
                    "strategy_peak_equity": round(
                        strategy_peak,
                        4,
                    ),
                    "strategy_drawdown_pct": (
                        strategy_drawdown
                    ),
                    "benchmark_equity": round(
                        benchmark_equity,
                        4,
                    ),
                    "benchmark_peak_equity": round(
                        benchmark_peak,
                        4,
                    ),
                    "benchmark_drawdown_pct": (
                        benchmark_drawdown
                    ),
                }
            )

        strategy_total_return = (
            (
                strategy_equity
                / self.STARTING_EQUITY
            )
            - 1.0
        ) * 100.0

        benchmark_total_return = (
            (
                benchmark_equity
                / self.STARTING_EQUITY
            )
            - 1.0
        ) * 100.0

        excess_return = (
            strategy_total_return
            - benchmark_total_return
        )

        professional_stats = (
            self._build_strategy_statistics(
                strategy_returns=(
                    strategy_returns
                ),
                completed_periods=(
                    completed_periods
                ),
                buy_periods=(
                    buy_periods
                ),
            )
        )

        return {
            "status": (
                "AVAILABLE"
                if completed_periods > 0
                else "PENDING"
            ),
            "method": (
                "Completed 1D signal observations"
            ),
            "strategy_policy": {
                "BUY": "SPY exposure",
                "HOLD": "cash",
                "AVOID": "cash",
            },
            "starting_equity": (
                self.STARTING_EQUITY
            ),
            "completed_periods": (
                completed_periods
            ),
            "buy_periods": (
                buy_periods
            ),
            "cash_periods": (
                cash_periods
            ),
            "ending_strategy_equity": round(
                strategy_equity,
                4,
            ),
            "strategy_total_return_pct": round(
                strategy_total_return,
                4,
            ),
            "strategy_peak_equity": round(
                strategy_peak,
                4,
            ),
            "strategy_max_drawdown_pct": round(
                strategy_max_drawdown,
                4,
            ),
            "ending_spy_equity": round(
                benchmark_equity,
                4,
            ),
            "spy_total_return_pct": round(
                benchmark_total_return,
                4,
            ),
            "spy_peak_equity": round(
                benchmark_peak,
                4,
            ),
            "spy_max_drawdown_pct": round(
                benchmark_max_drawdown,
                4,
            ),
            "strategy_excess_return_pct": round(
                excess_return,
                4,
            ),
            "strategy_statistics": (
                professional_stats
            ),
            "curve": curve,
        }

    # ============================================================
    # BENCHMARK EQUITY
    # ============================================================

    def _build_benchmark_equity_curve(
        self,
        evaluated_records: List[
            Dict[str, Any]
        ],
        benchmark: str,
    ) -> Dict[str, Any]:

        equity = (
            self.STARTING_EQUITY
        )

        peak_equity = (
            self.STARTING_EQUITY
        )

        max_drawdown = 0.0

        curve = []

        benchmark_returns = []

        sorted_records = sorted(
            evaluated_records,
            key=lambda record: str(
                record.get(
                    "timestamp",
                    ""
                )
            ),
        )

        for record in sorted_records:

            horizon_data = (
                record.get(
                    "benchmarks",
                    {}
                )
                .get(
                    benchmark,
                    {}
                )
                .get(
                    "horizons",
                    {}
                )
                .get(
                    "1D",
                    {}
                )
            )

            if (
                horizon_data.get(
                    "status"
                )
                != "COMPLETED"
            ):
                continue

            return_pct = (
                horizon_data.get(
                    "return_pct"
                )
            )

            if return_pct is None:
                continue

            try:
                return_pct = float(
                    return_pct
                )

            except Exception:
                continue

            benchmark_returns.append(
                return_pct
            )

            equity = (
                self._apply_return(
                    equity,
                    return_pct,
                )
            )

            peak_equity = max(
                peak_equity,
                equity,
            )

            drawdown = (
                self._calculate_drawdown_pct(
                    equity,
                    peak_equity,
                )
            )

            max_drawdown = min(
                max_drawdown,
                drawdown,
            )

            curve.append(
                {
                    "date": record.get(
                        "date"
                    ),
                    "return_pct": round(
                        return_pct,
                        4,
                    ),
                    "equity": round(
                        equity,
                        4,
                    ),
                    "peak_equity": round(
                        peak_equity,
                        4,
                    ),
                    "drawdown_pct": (
                        drawdown
                    ),
                }
            )

        total_return = (
            (
                equity
                / self.STARTING_EQUITY
            )
            - 1.0
        ) * 100.0

        volatility = None
        annualized_volatility = None

        if len(
            benchmark_returns
        ) >= 2:

            try:
                volatility = stdev(
                    benchmark_returns
                )

                annualized_volatility = (
                    volatility
                    * math.sqrt(
                        self.TRADING_PERIODS_PER_YEAR
                    )
                )

            except Exception:
                pass

        return {
            "benchmark": benchmark,
            "status": (
                "AVAILABLE"
                if curve
                else "PENDING"
            ),
            "starting_equity": (
                self.STARTING_EQUITY
            ),
            "completed_periods": len(
                curve
            ),
            "ending_equity": round(
                equity,
                4,
            ),
            "cumulative_return_pct": round(
                total_return,
                4,
            ),
            "peak_equity": round(
                peak_equity,
                4,
            ),
            "max_drawdown_pct": round(
                max_drawdown,
                4,
            ),
            "volatility_pct": (
                round(
                    volatility,
                    4,
                )
                if volatility is not None
                else None
            ),
            "annualized_volatility_pct": (
                round(
                    annualized_volatility,
                    4,
                )
                if annualized_volatility is not None
                else None
            ),
            "curve": curve,
        }

    # ============================================================
    # SUMMARY
    # ============================================================

    def _build_summary(
        self,
        evaluated_records: List[
            Dict[str, Any]
        ],
    ) -> Dict[str, Any]:

        summary = {
            "records": len(
                evaluated_records
            ),
            "decision_performance": {},
            "decision_stats": {},
            "agent_performance": {},
            "data_quality": {
                "valid_snapshots": 0,
                "invalid_snapshots": 0,
                "unknown_snapshots": 0,
            },
            "cumulative_performance": {},
            "benchmark_equity": {},
        }

        for record in evaluated_records:

            for quality in (
                record.get(
                    "data_quality",
                    {},
                ).values()
            ):
                status = quality.get(
                    "status"
                )

                if status in {
                    "CURRENT",
                    "VALID_PREVIOUS_MARKET_DATA",
                }:
                    summary[
                        "data_quality"
                    ][
                        "valid_snapshots"
                    ] += 1

                elif status in {
                    "STALE",
                    "INVALID_FUTURE",
                }:
                    summary[
                        "data_quality"
                    ][
                        "invalid_snapshots"
                    ] += 1

                else:
                    summary[
                        "data_quality"
                    ][
                        "unknown_snapshots"
                    ] += 1

        for horizon_name in self.HORIZONS:

            evaluations = []

            for record in evaluated_records:

                evaluations.append(
                    record.get(
                        "decision_evaluation",
                        {},
                    ).get(
                        horizon_name,
                        {},
                    )
                )

            summary[
                "decision_performance"
            ][horizon_name] = (
                self._performance_block(
                    evaluations
                )
            )

        summary[
            "decision_stats"
        ] = self._build_decision_stats(
            evaluated_records
        )

        agent_names = set()

        for record in evaluated_records:
            agent_names.update(
                record.get(
                    "agent_evaluation",
                    {},
                ).keys()
            )

        for agent_name in sorted(
            agent_names
        ):

            agent_summary = {
                "benchmark": (
                    self.AGENT_BENCHMARKS.get(
                        agent_name,
                        "SPY",
                    )
                ),
                "horizons": {},
            }

            for horizon_name in self.HORIZONS:

                evaluations = []

                for record in evaluated_records:

                    evaluation = (
                        record.get(
                            "agent_evaluation",
                            {},
                        )
                        .get(
                            agent_name,
                            {},
                        )
                        .get(
                            "horizons",
                            {},
                        )
                        .get(
                            horizon_name,
                            {},
                        )
                    )

                    evaluations.append(
                        evaluation
                    )

                agent_summary[
                    "horizons"
                ][horizon_name] = (
                    self._performance_block(
                        evaluations
                    )
                )

            summary[
                "agent_performance"
            ][agent_name] = (
                agent_summary
            )

        summary[
            "cumulative_performance"
        ] = (
            self._build_strategy_equity_curve(
                evaluated_records
            )
        )

        for benchmark in [
            "SPY",
            "BTC-USD",
        ]:

            summary[
                "benchmark_equity"
            ][benchmark] = (
                self._build_benchmark_equity_curve(
                    evaluated_records,
                    benchmark,
                )
            )

        return summary

    # ============================================================
    # JSON REPORT
    # ============================================================

    def _save_json_report(
        self,
        report: Dict[str, Any],
    ) -> None:

        with self.output_json.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False,
            )

    # ============================================================
    # TEXT REPORT
    # ============================================================

    def _save_text_report(
        self,
        report: Dict[str, Any],
    ) -> None:

        summary = report.get(
            "summary",
            {},
        )

        lines = []

        lines.append(
            "=" * 78
        )

        lines.append(
            "AI MULTI-AGENT FINANCIAL RESEARCH PLATFORM"
        )

        lines.append(
            "v1.7 BACKTESTING ENGINE REPORT"
        )

        lines.append(
            "=" * 78
        )

        lines.append("")

        lines.append(
            f"Generated: "
            f"{report.get('generated_at')}"
        )

        lines.append(
            f"Historical records: "
            f"{summary.get('records', 0)}"
        )

        lines.append("")

        lines.append(
            "DATA QUALITY"
        )

        lines.append(
            "-" * 78
        )

        quality = summary.get(
            "data_quality",
            {},
        )

        lines.append(
            f"Valid benchmark snapshots: "
            f"{quality.get('valid_snapshots', 0)}"
        )

        lines.append(
            f"Invalid benchmark snapshots: "
            f"{quality.get('invalid_snapshots', 0)}"
        )

        lines.append(
            f"Unknown benchmark snapshots: "
            f"{quality.get('unknown_snapshots', 0)}"
        )

        lines.append("")

        lines.append(
            "FINAL DECISION PERFORMANCE"
        )

        lines.append(
            "-" * 78
        )

        decision_performance = (
            summary.get(
                "decision_performance",
                {},
            )
        )

        for horizon_name in self.HORIZONS:

            data = decision_performance.get(
                horizon_name,
                {},
            )

            accuracy = data.get(
                "accuracy_pct"
            )

            accuracy_text = (
                f"{accuracy:.2f}%"
                if accuracy is not None
                else "PENDING"
            )

            lines.append(
                f"{horizon_name}: "
                f"completed={data.get('completed', 0)} | "
                f"correct={data.get('correct', 0)} | "
                f"incorrect={data.get('incorrect', 0)} | "
                f"pending={data.get('pending', 0)} | "
                f"invalid={data.get('invalid', 0)} | "
                f"accuracy={accuracy_text}"
            )

        lines.append("")

        lines.append(
            "BUY / HOLD / AVOID PERFORMANCE"
        )

        lines.append(
            "-" * 78
        )

        decision_stats = (
            summary.get(
                "decision_stats",
                {},
            )
        )

        for decision_name in self.DECISIONS:

            decision_data = (
                decision_stats.get(
                    decision_name,
                    {},
                )
            )

            lines.append(
                f"{decision_name} "
                f"[Total signals: "
                f"{decision_data.get('total_signals', 0)}]"
            )

            horizons = (
                decision_data.get(
                    "horizons",
                    {},
                )
            )

            for horizon_name in self.HORIZONS:

                data = horizons.get(
                    horizon_name,
                    {},
                )

                accuracy = data.get(
                    "accuracy_pct"
                )

                accuracy_text = (
                    f"{accuracy:.2f}%"
                    if accuracy is not None
                    else "PENDING"
                )

                lines.append(
                    f"  {horizon_name}: "
                    f"completed={data.get('completed', 0)} | "
                    f"correct={data.get('correct', 0)} | "
                    f"incorrect={data.get('incorrect', 0)} | "
                    f"pending={data.get('pending', 0)} | "
                    f"accuracy={accuracy_text}"
                )

            lines.append("")

        cumulative = (
            summary.get(
                "cumulative_performance",
                {},
            )
        )

        lines.append(
            "CUMULATIVE STRATEGY PERFORMANCE"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Completed 1D periods: "
            f"{cumulative.get('completed_periods', 0)}"
        )

        lines.append(
            f"Ending strategy equity: "
            f"{cumulative.get('ending_strategy_equity', 100.0):.4f}"
        )

        lines.append(
            f"Strategy cumulative return: "
            f"{cumulative.get('strategy_total_return_pct', 0.0):.4f}%"
        )

        lines.append(
            f"Strategy max drawdown: "
            f"{cumulative.get('strategy_max_drawdown_pct', 0.0):.4f}%"
        )

        lines.append(
            f"SPY cumulative return: "
            f"{cumulative.get('spy_total_return_pct', 0.0):.4f}%"
        )

        lines.append(
            f"SPY max drawdown: "
            f"{cumulative.get('spy_max_drawdown_pct', 0.0):.4f}%"
        )

        lines.append(
            f"Excess return vs SPY: "
            f"{cumulative.get('strategy_excess_return_pct', 0.0):.4f}%"
        )

        lines.append("")

        stats = cumulative.get(
            "strategy_statistics",
            {},
        )

        lines.append(
            "PROFESSIONAL STRATEGY STATISTICS"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Samples: "
            f"{stats.get('samples', 0)}"
        )

        lines.append(
            f"Winning periods: "
            f"{stats.get('winning_periods', 0)}"
        )

        lines.append(
            f"Losing periods: "
            f"{stats.get('losing_periods', 0)}"
        )

        lines.append(
            f"Flat periods: "
            f"{stats.get('flat_periods', 0)}"
        )

        win_rate = stats.get(
            "win_rate_pct"
        )

        lines.append(
            f"Win rate: "
            f"{f'{win_rate:.2f}%' if win_rate is not None else 'PENDING'}"
        )

        average_return = stats.get(
            "average_return_pct"
        )

        lines.append(
            f"Average strategy return: "
            f"{f'{average_return:.4f}%' if average_return is not None else 'PENDING'}"
        )

        average_win = stats.get(
            "average_win_pct"
        )

        lines.append(
            f"Average win: "
            f"{f'{average_win:.4f}%' if average_win is not None else 'PENDING'}"
        )

        average_loss = stats.get(
            "average_loss_pct"
        )

        lines.append(
            f"Average loss: "
            f"{f'{average_loss:.4f}%' if average_loss is not None else 'PENDING'}"
        )

        best_period = stats.get(
            "best_period_pct"
        )

        lines.append(
            f"Best period: "
            f"{f'{best_period:.4f}%' if best_period is not None else 'PENDING'}"
        )

        worst_period = stats.get(
            "worst_period_pct"
        )

        lines.append(
            f"Worst period: "
            f"{f'{worst_period:.4f}%' if worst_period is not None else 'PENDING'}"
        )

        lines.append(
            f"Profit factor: "
            f"{stats.get('profit_factor', 'PENDING')}"
        )

        exposure = stats.get(
            "exposure_rate_pct"
        )

        lines.append(
            f"Exposure rate: "
            f"{f'{exposure:.2f}%' if exposure is not None else 'PENDING'}"
        )

        volatility = stats.get(
            "volatility_pct"
        )

        lines.append(
            f"Period volatility: "
            f"{f'{volatility:.4f}%' if volatility is not None else 'PENDING'}"
        )

        annualized_vol = stats.get(
            "annualized_volatility_pct"
        )

        lines.append(
            f"Annualized volatility: "
            f"{f'{annualized_vol:.4f}%' if annualized_vol is not None else 'PENDING'}"
        )

        sharpe = stats.get(
            "sharpe_like"
        )

        lines.append(
            f"Sharpe-like metric: "
            f"{f'{sharpe:.4f}' if sharpe is not None else 'PENDING'}"
        )

        lines.append("")

        lines.append(
            "BENCHMARK CUMULATIVE PERFORMANCE"
        )

        lines.append(
            "-" * 78
        )

        benchmark_equity = (
            summary.get(
                "benchmark_equity",
                {},
            )
        )

        for benchmark in [
            "SPY",
            "BTC-USD",
        ]:

            data = benchmark_equity.get(
                benchmark,
                {},
            )

            lines.append(
                f"{benchmark}:"
            )

            lines.append(
                f"  Completed periods: "
                f"{data.get('completed_periods', 0)}"
            )

            lines.append(
                f"  Ending equity: "
                f"{data.get('ending_equity', 100.0):.4f}"
            )

            lines.append(
                f"  Cumulative return: "
                f"{data.get('cumulative_return_pct', 0.0):.4f}%"
            )

            lines.append(
                f"  Max drawdown: "
                f"{data.get('max_drawdown_pct', 0.0):.4f}%"
            )

        lines.append("")

        lines.append(
            "AGENT PERFORMANCE"
        )

        lines.append(
            "-" * 78
        )

        agent_performance = (
            summary.get(
                "agent_performance",
                {},
            )
        )

        for (
            agent_name,
            agent_data,
        ) in agent_performance.items():

            benchmark = (
                agent_data.get(
                    "benchmark",
                    "UNKNOWN",
                )
            )

            lines.append(
                f"{agent_name.upper()} "
                f"[Benchmark: {benchmark}]"
            )

            for horizon_name in self.HORIZONS:

                data = (
                    agent_data.get(
                        "horizons",
                        {},
                    ).get(
                        horizon_name,
                        {},
                    )
                )

                accuracy = (
                    data.get(
                        "accuracy_pct"
                    )
                )

                accuracy_text = (
                    f"{accuracy:.2f}%"
                    if accuracy is not None
                    else "PENDING"
                )

                lines.append(
                    f"  {horizon_name}: "
                    f"completed={data.get('completed', 0)} | "
                    f"correct={data.get('correct', 0)} | "
                    f"incorrect={data.get('incorrect', 0)} | "
                    f"pending={data.get('pending', 0)} | "
                    f"accuracy={accuracy_text}"
                )

            lines.append("")

        lines.append(
            "-" * 78
        )

        lines.append(
            "METHODOLOGY"
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            "1D requires a full 24 hours from the original signal timestamp."
        )

        lines.append(
            "7D requires a full 7 x 24 hours."
        )

        lines.append(
            "30D requires a full 30 x 24 hours."
        )

        lines.append(
            "SPY is the primary benchmark for the final decision."
        )

        lines.append(
            "Crypto, OnChain and Derivatives agents use BTC-USD."
        )

        lines.append(
            "Other specialist agents use SPY."
        )

        lines.append(
            "Cumulative performance uses completed 1D observations only."
        )

        lines.append(
            "BUY is treated as SPY exposure."
        )

        lines.append(
            "HOLD and AVOID are treated as cash."
        )

        lines.append(
            "AVOID is not treated as a short position."
        )

        lines.append(
            "Sharpe-like metric is calculated only when at least two "
            "return observations exist and return volatility is non-zero."
        )

        lines.append(
            "Profit factor is based on gross winning returns divided "
            "by absolute gross losing returns."
        )

        lines.append(
            "Only real stored historical signals are evaluated."
        )

        lines.append(
            "No synthetic historical signals or fake historical backfill is used."
        )

        lines.append(
            "=" * 78
        )

        with self.output_txt.open(
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                "\n".join(
                    lines
                )
            )

    # ============================================================
    # RUN
    # ============================================================

    def run(
        self,
    ) -> Dict[str, Any]:

        print("")

        print(
            "=" * 78
        )

        print(
            "v1.7 BACKTESTING ENGINE"
        )

        print(
            "=" * 78
        )

        history = self.load_history()

        if not history:

            report = {
                "version": "1.7",
                "status": "NO_DATA",
                "generated_at": (
                    datetime.now().isoformat()
                ),
                "summary": {
                    "records": 0,
                },
                "records": [],
            }

            self._save_json_report(
                report
            )

            print(
                "[BACKTEST] No historical signals found."
            )

            return report

        print(
            f"[BACKTEST] Historical signals: "
            f"{len(history)}"
        )

        evaluated_records = []

        for (
            index,
            record,
        ) in enumerate(
            history,
            start=1,
        ):

            print(
                f"[BACKTEST] Evaluating "
                f"{index}/{len(history)} "
                f"- {record.get('date', 'UNKNOWN')}"
            )

            evaluated = (
                self._evaluate_record(
                    record
                )
            )

            if evaluated is not None:
                evaluated_records.append(
                    evaluated
                )

        summary = (
            self._build_summary(
                evaluated_records
            )
        )

        report = {
            "version": "1.7",
            "engine": (
                "Historical Signal Backtesting Engine"
            ),
            "status": "SUCCESS",
            "generated_at": (
                datetime.now().isoformat()
            ),
            "architecture": (
                "Time-Aware Historical Signal Evaluation + "
                "BUY/HOLD/AVOID Analysis + "
                "SPY/BTC Benchmarking + "
                "Agent Performance + "
                "Data Quality Validation + "
                "Cumulative Performance + "
                "Drawdown Analysis + "
                "Professional Strategy Statistics"
            ),
            "horizons": list(
                self.HORIZONS.keys()
            ),
            "benchmark_thresholds": (
                self.BENCHMARK_THRESHOLDS
            ),
            "summary": summary,
            "records": (
                evaluated_records
            ),
        }

        self._save_json_report(
            report
        )

        self._save_text_report(
            report
        )

        print("")

        print(
            "[BACKTEST] SUCCESS"
        )

        print(
            f"[BACKTEST] JSON report: "
            f"{self.output_json}"
        )

        print(
            f"[BACKTEST] TXT report: "
            f"{self.output_txt}"
        )

        print("")

        quality = summary.get(
            "data_quality",
            {},
        )

        print(
            "DATA QUALITY"
        )

        print(
            f"  Valid snapshots: "
            f"{quality.get('valid_snapshots', 0)}"
        )

        print(
            f"  Invalid snapshots: "
            f"{quality.get('invalid_snapshots', 0)}"
        )

        print(
            f"  Unknown snapshots: "
            f"{quality.get('unknown_snapshots', 0)}"
        )

        print("")

        print(
            "FINAL DECISION PERFORMANCE"
        )

        for horizon_name in self.HORIZONS:

            performance = (
                summary[
                    "decision_performance"
                ][horizon_name]
            )

            accuracy = (
                performance.get(
                    "accuracy_pct"
                )
            )

            accuracy_text = (
                f"{accuracy:.2f}%"
                if accuracy is not None
                else "PENDING"
            )

            print(
                f"  {horizon_name}: "
                f"{accuracy_text} "
                f"(completed="
                f"{performance.get('completed', 0)}, "
                f"pending="
                f"{performance.get('pending', 0)}, "
                f"invalid="
                f"{performance.get('invalid', 0)})"
            )

        print("")

        print(
            "BUY / HOLD / AVOID PERFORMANCE"
        )

        for decision_name in self.DECISIONS:

            decision_data = (
                summary[
                    "decision_stats"
                ].get(
                    decision_name,
                    {},
                )
            )

            print(
                f"  {decision_name}: "
                f"{decision_data.get('total_signals', 0)} "
                f"historical signals"
            )

            horizons = (
                decision_data.get(
                    "horizons",
                    {},
                )
            )

            for horizon_name in self.HORIZONS:

                data = horizons.get(
                    horizon_name,
                    {},
                )

                accuracy = (
                    data.get(
                        "accuracy_pct"
                    )
                )

                accuracy_text = (
                    f"{accuracy:.2f}%"
                    if accuracy is not None
                    else "PENDING"
                )

                print(
                    f"    {horizon_name}: "
                    f"{accuracy_text} "
                    f"(completed="
                    f"{data.get('completed', 0)}, "
                    f"pending="
                    f"{data.get('pending', 0)})"
                )

        print("")

        cumulative = (
            summary.get(
                "cumulative_performance",
                {},
            )
        )

        print(
            "CUMULATIVE PERFORMANCE"
        )

        print(
            f"  Completed 1D periods: "
            f"{cumulative.get('completed_periods', 0)}"
        )

        print(
            f"  Strategy equity: "
            f"{cumulative.get('ending_strategy_equity', 100.0):.4f}"
        )

        print(
            f"  Strategy return: "
            f"{cumulative.get('strategy_total_return_pct', 0.0):.4f}%"
        )

        print(
            f"  Strategy max drawdown: "
            f"{cumulative.get('strategy_max_drawdown_pct', 0.0):.4f}%"
        )

        print(
            f"  SPY return: "
            f"{cumulative.get('spy_total_return_pct', 0.0):.4f}%"
        )

        print(
            f"  Excess return vs SPY: "
            f"{cumulative.get('strategy_excess_return_pct', 0.0):.4f}%"
        )

        print("")

        stats = cumulative.get(
            "strategy_statistics",
            {},
        )

        print(
            "PROFESSIONAL STRATEGY STATISTICS"
        )

        print(
            f"  Samples: "
            f"{stats.get('samples', 0)}"
        )

        print(
            f"  Winning periods: "
            f"{stats.get('winning_periods', 0)}"
        )

        print(
            f"  Losing periods: "
            f"{stats.get('losing_periods', 0)}"
        )

        print(
            f"  Flat periods: "
            f"{stats.get('flat_periods', 0)}"
        )

        win_rate = (
            stats.get(
                "win_rate_pct"
            )
        )

        print(
            f"  Win rate: "
            f"{f'{win_rate:.2f}%' if win_rate is not None else 'PENDING'}"
        )

        exposure = (
            stats.get(
                "exposure_rate_pct"
            )
        )

        print(
            f"  Exposure rate: "
            f"{f'{exposure:.2f}%' if exposure is not None else 'PENDING'}"
        )

        print(
            f"  Profit factor: "
            f"{stats.get('profit_factor') if stats.get('profit_factor') is not None else 'PENDING'}"
        )

        sharpe = (
            stats.get(
                "sharpe_like"
            )
        )

        print(
            f"  Sharpe-like: "
            f"{f'{sharpe:.4f}' if sharpe is not None else 'PENDING'}"
        )

        print("")

        print(
            "BENCHMARK EQUITY"
        )

        benchmark_equity = (
            summary.get(
                "benchmark_equity",
                {},
            )
        )

        for benchmark in [
            "SPY",
            "BTC-USD",
        ]:

            data = (
                benchmark_equity.get(
                    benchmark,
                    {},
                )
            )

            print(
                f"  {benchmark}: "
                f"equity="
                f"{data.get('ending_equity', 100.0):.4f} | "
                f"return="
                f"{data.get('cumulative_return_pct', 0.0):.4f}% | "
                f"max DD="
                f"{data.get('max_drawdown_pct', 0.0):.4f}%"
            )

        print(
            "=" * 78
        )

        print("")

        return report


# ================================================================
# DIRECT EXECUTION
# ================================================================

if __name__ == "__main__":

    engine = BacktestingEngine()
    engine.run()