from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class DataIntegrityError(RuntimeError):
    """Raised when critical data invariants are violated (e.g., lot size changes mid-trade)."""


@dataclass(frozen=True)
class BaseStrategyConfig:
    input_parquet_path: str
    require_expiry_rank_1: bool = True
    settle_fallback_on_zero_or_nan: bool = True
    warn_max_examples: int = 25

    def resolved_input_path(self) -> Path:
        # supports "data>curated>file.parquet" style paths
        normalized = self.input_parquet_path.replace(">", "/").replace("\\", "/")
        return Path(normalized)


@dataclass(frozen=True)
class TradeBlotterSchema:
    trade_id: str = "trade_id"
    symbol: str = "symbol"
    instrument: str = "instrument"
    expiry_dt: str = "expiry_dt"
    strike_pr: str = "strike_pr"
    option_typ: str = "option_typ"
    entry_date: str = "entry_date"
    position_sign: str = "position_sign"  # +1 long, -1 short
    entry_price: str = "entry_price"      # optional override; if NaN uses settle on entry row


class BaseStrategy(ABC):
    """
    Abstract parent for strategies:
      - Loads and filters curated derivatives dataset.
      - Identifies monthly entry days.
      - Computes vectorized Rupee MTM P&L for strategy-defined trades.
    """

    REQUIRED_MARKET_COLS = [
        "date",
        "symbol",
        "instrument",
        "expiry_dt",
        "strike_pr",
        "option_typ",
        "open",                # Contract Open for Overnight P&L
        "index_open_price",    # Index Open for Gap Risk Trigger
        "close",
        "settle_pr",
        "lot_size",
        "expiry_rank",
        "is_trading_day",
        "is_opt_monthly_expiry",
    ]

    def __init__(self, config: BaseStrategyConfig, logger: Optional[logging.Logger] = None) -> None:
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def load_market_data(self) -> pd.DataFrame:
        path = self.config.resolved_input_path()
        self.logger.info("Loading market data from: %s", path)

        df = pd.read_parquet(path)

        missing = [c for c in self.REQUIRED_MARKET_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Market data missing required columns: {missing}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["expiry_dt"] = pd.to_datetime(df["expiry_dt"]).dt.normalize()

        if df["date"].isna().any() or df["expiry_dt"].isna().any():
            raise ValueError("Found nulls in date/expiry_dt after parsing.")

        if (df["lot_size"] <= 0).any():
            raise ValueError("Found non-positive lot_size values in market data.")

        sort_cols = ["symbol", "instrument", "expiry_dt", "strike_pr", "option_typ", "date"]
        df = df.sort_values(sort_cols, kind="mergesort")
        self.logger.info("Loaded rows: %d", len(df))

        if self.config.require_expiry_rank_1:
            before = len(df)
            df = df.loc[df["expiry_rank"] == 1].copy()
            self.logger.info("Filtered expiry_rank==1: %d -> %d rows", before, len(df))

        return df

    @staticmethod
    def identify_entry_days(market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Entry Day:
        First `date` where `is_trading_day == True` strictly after a day where
        `is_opt_monthly_expiry == True`, computed per symbol.

        Returns DataFrame: [symbol, entry_date]
        """
        df = market_df[["symbol", "date", "is_trading_day", "is_opt_monthly_expiry"]].drop_duplicates()
        df = df.sort_values(["symbol", "date"], kind="mergesort")

        # cycle_id increments on monthly expiry days; expiry day belongs to the new cycle.
        df["cycle_id"] = df.groupby("symbol")["is_opt_monthly_expiry"].cumsum()

        expiry_dates = (
            df.loc[df["is_opt_monthly_expiry"], ["symbol", "cycle_id", "date"]]
            .rename(columns={"date": "monthly_expiry_date"})
        )
        df = df.merge(expiry_dates, on=["symbol", "cycle_id"], how="left")

        is_candidate = (df["cycle_id"] > 0) & (df["is_trading_day"]) & (df["date"] > df["monthly_expiry_date"])
        candidates = df.loc[is_candidate, ["symbol", "cycle_id", "date"]]

        entry_days = (
            candidates.groupby(["symbol", "cycle_id"], as_index=False)["date"].min()
            .rename(columns={"date": "entry_date"})
        )

        return entry_days[["symbol", "entry_date"]]

    @abstractmethod
    def build_trades(self, market_df: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
        """
        Strategy must return a blotter with at least:
          trade_id, symbol, instrument, expiry_dt, strike_pr, option_typ, entry_date, position_sign
        and optionally entry_price.
        """
        raise NotImplementedError

    def compute_mtm_pnl_rupee(
        self,
        market_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        schema: TradeBlotterSchema = TradeBlotterSchema(),
    ) -> pd.DataFrame:
        """
        Vectorized MTM Rupee P&L:
          Day 0 (entry): sign * (settle_t - entry_price) * lot_size
          Day t (hold):  sign * (settle_t - settle_{t-1}) * lot_size

        Guards:
          - settle_pr null/0 -> fallback to close (warning with examples)
          - lot_size changes within a trade -> DataIntegrityError
          - require a market row on expiry_dt (force-close on expiry)
        """

        required_trade_cols = [
            schema.trade_id,
            schema.symbol,
            schema.instrument,
            schema.expiry_dt,
            schema.strike_pr,
            schema.option_typ,
            schema.entry_date,
            schema.position_sign,
        ]
        missing_trade_cols = [c for c in required_trade_cols if c not in trades_df.columns]
        if missing_trade_cols:
            raise ValueError(f"Trades missing required columns: {missing_trade_cols}")

        trades = trades_df.copy()
        trades[schema.entry_date] = pd.to_datetime(trades[schema.entry_date]).dt.normalize()
        trades[schema.expiry_dt] = pd.to_datetime(trades[schema.expiry_dt]).dt.normalize()

        if schema.entry_price not in trades.columns:
            trades[schema.entry_price] = np.nan

        key_cols = [schema.symbol, schema.instrument, schema.expiry_dt, schema.strike_pr, schema.option_typ]

        merged = market_df.merge(trades, on=key_cols, how="inner")

        before = len(merged)
        merged = merged.loc[
            (merged["date"] >= merged[schema.entry_date]) & (merged["date"] <= merged[schema.expiry_dt])
        ].copy()
        self.logger.info(
            "Expanded trades onto market rows: %d -> %d rows after date-window filter",
            before,
            len(merged),
        )

        if merged.empty:
            raise ValueError("No market rows matched trades after merge/date-window filtering.")

        merged = merged.sort_values([schema.trade_id, "date"], kind="mergesort")

        # settle fallback
        settle = merged["settle_pr"]
        is_missing_settle = settle.isna() | (settle == 0)

        if self.config.settle_fallback_on_zero_or_nan and is_missing_settle.any():
            examples = merged.loc[
                is_missing_settle, ["date", "symbol", "instrument", "expiry_dt", "strike_pr", "option_typ"]
            ].head(self.config.warn_max_examples)
            self.logger.warning(
                "Missing/zero settle_pr encountered; falling back to close. Examples (up to %d): %s",
                self.config.warn_max_examples,
                examples.to_dict(orient="records"),
            )
            merged["settle_used"] = np.where(is_missing_settle, merged["close"], merged["settle_pr"])
        else:
            merged["settle_used"] = merged["settle_pr"]

        # lot size integrity within trade
        lot_nunique = merged.groupby(schema.trade_id)["lot_size"].nunique(dropna=False)
        bad_trades = lot_nunique[lot_nunique > 1]
        if not bad_trades.empty:
            raise DataIntegrityError(
                "Lot size changed mid-trade for trade_id(s): "
                f"{bad_trades.index.tolist()}. This likely indicates a bad data join."
            )

        # force-close on expiry: require last row date == expiry_dt per trade
        last_dates = merged.groupby(schema.trade_id)["date"].max()
        expiry_dates = merged.groupby(schema.trade_id)[schema.expiry_dt].max()
        missing_expiry_row = last_dates[last_dates != expiry_dates]
        if not missing_expiry_row.empty:
            raise ValueError(
                "Trade(s) missing expiry_dt row; cannot force-close without final exchange settlement. "
                f"trade_id(s): {missing_expiry_row.index.tolist()}"
            )

        # resolve entry price:
        # - use blotter entry_price if provided
        # - else use settle_used from the entry_date row
        is_entry_row = merged["date"] == merged[schema.entry_date]
        entry_settle = (
            merged["settle_used"].where(is_entry_row).groupby(merged[schema.trade_id]).transform("max")
        )

        merged["entry_price_resolved"] = np.where(
            merged[schema.entry_price].notna(),
            merged[schema.entry_price],
            entry_settle,
        )

        if merged["entry_price_resolved"].isna().any():
            bad = merged.loc[merged["entry_price_resolved"].isna(), schema.trade_id].unique().tolist()
            raise ValueError(f"Could not resolve entry_price for trade_id(s): {bad}")

        # vectorized MTM
        settle_prev = merged.groupby(schema.trade_id)["settle_used"].shift(1)
        position_sign = merged[schema.position_sign].astype(int)

        holding_pnl = position_sign * (merged["settle_used"] - settle_prev) * merged["lot_size"]
        entry_pnl = position_sign * (merged["settle_used"] - merged["entry_price_resolved"]) * merged["lot_size"]

        merged["daily_pnl_rupee"] = np.where(is_entry_row, entry_pnl, holding_pnl)
        merged["daily_pnl_rupee"] = merged["daily_pnl_rupee"].fillna(0.0)
        merged["cum_pnl_rupee"] = merged.groupby(schema.trade_id)["daily_pnl_rupee"].cumsum()

        # PASS-THROUGH FOR ANALYTICS: Add open and index_open_price to the output
        out = merged[
            [
                "date",
                "symbol",
                "strike_pr",
                "option_typ",
                "open",                # Contract Open snap for Overnight P&L
                "index_open_price",    # Index Open snap for Gap analysis
                "settle_used",
                "entry_price_resolved",
                "daily_pnl_rupee",
                "cum_pnl_rupee",
                schema.trade_id,
            ]
        ].rename(columns={"settle_used": "settle_pr", "entry_price_resolved": "entry_price"})

        return out
