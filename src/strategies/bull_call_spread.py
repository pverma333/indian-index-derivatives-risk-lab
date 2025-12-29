from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.strategies.engine_pnl import BaseStrategy, BaseStrategyConfig, DataIntegrityError, TradeBlotterSchema
except Exception:  # pragma: no cover
    from engine_pnl import BaseStrategy, BaseStrategyConfig, DataIntegrityError, TradeBlotterSchema


@dataclass(frozen=True)
class BullCallSpreadStrategyConfig(BaseStrategyConfig):
    symbol: str = "NIFTY"

    # Monthly option logic typically requires access to non-near expiries too.
    require_expiry_rank_1: bool = False

    strike_interval: int = 50
    otm_points: int = 200

    # Liquidity abort guard at entry
    max_abort_ratio: float = 0.25

    # Optional: curated sets sometimes include multiple option instruments
    option_instrument: Optional[str] = "OPTIDX"


class BullCallSpreadStrategy(BaseStrategy):
    REQUIRED_LIQ_COLS = ("open_int", "volume")

    LONG_SUFFIX = "_LONG_CE"
    SHORT_SUFFIX = "_SHORT_CE"

    def __init__(self, config: BullCallSpreadStrategyConfig, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(config=config, logger=logger)
        self.config: BullCallSpreadStrategyConfig = config

    @staticmethod
    def _parent_trade_id_from_leg_trade_id(trade_id: str) -> str:
        if trade_id.endswith(BullCallSpreadStrategy.LONG_SUFFIX):
            return trade_id[: -len(BullCallSpreadStrategy.LONG_SUFFIX)]
        if trade_id.endswith(BullCallSpreadStrategy.SHORT_SUFFIX):
            return trade_id[: -len(BullCallSpreadStrategy.SHORT_SUFFIX)]
        return trade_id

    def _validate_market_has_liquidity_columns(self, market_df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_LIQ_COLS if c not in market_df.columns]
        if missing:
            raise ValueError(
                f"Market data missing required liquidity columns for bull call spread: {missing}. "
                "Ticket requires open_int and volume checks."
            )

    def _get_monthly_expiry_dates(self, market_df: pd.DataFrame, symbol: str) -> pd.Series:
        dates = (
            market_df.loc[(market_df["symbol"] == symbol) & (market_df["is_opt_monthly_expiry"] == True), "date"]
            .drop_duplicates()
            .sort_values(kind="mergesort")
        )
        return dates

    def _next_monthly_expiry(self, monthly_expiries: pd.Series, entry_date: pd.Timestamp) -> Optional[pd.Timestamp]:
        nxt = monthly_expiries.loc[monthly_expiries > entry_date]
        if nxt.empty:
            return None
        return pd.Timestamp(nxt.iloc[0]).normalize()

    def _select_atm_call_strike(self, rows_on_entry: pd.DataFrame, spot_close: float) -> int:
        step = int(self.config.strike_interval)
        if step <= 0:
            raise ValueError("strike_interval must be positive.")

        df = rows_on_entry.copy()
        df = df.loc[df["option_typ"] == "CE"]
        df = df.loc[df["strike_pr"].notna()]
        df["strike_int"] = df["strike_pr"].astype(int)
        df = df.loc[(df["strike_int"] % step) == 0]

        if df.empty:
            raise ValueError("No CE candidate strikes after strike_interval filtering.")

        strikes = df["strike_int"].drop_duplicates().sort_values(kind="mergesort")
        diffs = (strikes.astype(float) - float(spot_close)).abs()
        min_diff = diffs.min()

        # Deterministic tie-break: choose the lower strike among equally-close
        best = strikes.loc[diffs == min_diff].min()
        return int(best)

    def _pick_single_row(self, df: pd.DataFrame) -> pd.Series:
        # Deterministic row selection in case of duplicates
        sort_cols = [c for c in ["instrument", "lot_size", "close", "settle_pr"] if c in df.columns]
        if not sort_cols:
            return df.iloc[0]
        return df.sort_values(sort_cols, kind="mergesort").iloc[0]

    def build_trades(self, market_df: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
        self._validate_market_has_liquidity_columns(market_df)

        symbol = self.config.symbol
        entry_days = entry_days.loc[entry_days["symbol"] == symbol].copy()
        entry_days["entry_date"] = pd.to_datetime(entry_days["entry_date"]).dt.normalize()
        entry_days = entry_days.sort_values(["entry_date"], kind="mergesort")

        monthly_expiries = self._get_monthly_expiry_dates(market_df, symbol=symbol)
        if monthly_expiries.empty:
            raise ValueError(f"No monthly expiry dates found for symbol={symbol} (is_opt_monthly_expiry==True).")

        option_df = market_df.loc[(market_df["symbol"] == symbol) & (market_df["option_typ"] == "CE")].copy()
        if self.config.option_instrument is not None:
            option_df = option_df.loc[option_df["instrument"] == self.config.option_instrument].copy()

        trades_out: list[dict] = []
        aborted = 0

        for entry_date in entry_days["entry_date"].tolist():
            expiry_dt = self._next_monthly_expiry(monthly_expiries, entry_date=entry_date)
            if expiry_dt is None:
                self.logger.warning("No next monthly expiry after entry_date=%s; skipping.", entry_date.date())
                continue

            rows_on_entry = option_df.loc[(option_df["date"] == entry_date) & (option_df["expiry_dt"] == expiry_dt)]
            if rows_on_entry.empty:
                self.logger.critical(
                    "ABORT BCS: no CE option rows on entry_date=%s for monthly expiry_dt=%s (symbol=%s).",
                    entry_date.date(),
                    expiry_dt.date(),
                    symbol,
                )
                aborted += 1
                continue

            # spot_close is present in curated dataset (options rows have it)
            spot_close = float(rows_on_entry["spot_close"].iloc[0])
            atm_strike = self._select_atm_call_strike(rows_on_entry, spot_close=spot_close)
            otm_strike = int(atm_strike + int(self.config.otm_points))

            atm_rows = rows_on_entry.loc[rows_on_entry["strike_pr"].astype(int) == atm_strike]
            otm_rows = rows_on_entry.loc[rows_on_entry["strike_pr"].astype(int) == otm_strike]

            if atm_rows.empty or otm_rows.empty:
                self.logger.critical(
                    "ABORT BCS: missing leg(s) on entry_date=%s symbol=%s expiry_dt=%s "
                    "(ATM=%s rows=%d | OTM=%s rows=%d).",
                    entry_date.date(),
                    symbol,
                    expiry_dt.date(),
                    atm_strike,
                    len(atm_rows),
                    otm_strike,
                    len(otm_rows),
                )
                aborted += 1
                continue

            atm_row = self._pick_single_row(atm_rows)
            otm_row = self._pick_single_row(otm_rows)

            # Validation: same expiry_dt by construction, but ensure metadata consistency
            if pd.Timestamp(atm_row["expiry_dt"]).normalize() != pd.Timestamp(otm_row["expiry_dt"]).normalize():
                self.logger.critical(
                    "ABORT BCS: expiry mismatch ATM=%s OTM=%s on entry_date=%s.",
                    pd.Timestamp(atm_row["expiry_dt"]).date(),
                    pd.Timestamp(otm_row["expiry_dt"]).date(),
                    entry_date.date(),
                )
                aborted += 1
                continue
            if str(atm_row["instrument"]) != str(otm_row["instrument"]):
                self.logger.critical(
                    "ABORT BCS: instrument mismatch ATM=%s OTM=%s on entry_date=%s expiry=%s.",
                    atm_row["instrument"],
                    otm_row["instrument"],
                    entry_date.date(),
                    expiry_dt.date(),
                )
                aborted += 1
                continue
            if int(atm_row["lot_size"]) != int(otm_row["lot_size"]):
                self.logger.critical(
                    "ABORT BCS: lot_size mismatch ATM=%s OTM=%s on entry_date=%s expiry=%s.",
                    atm_row["lot_size"],
                    otm_row["lot_size"],
                    entry_date.date(),
                    expiry_dt.date(),
                )
                aborted += 1
                continue

            # Liquidity guard
            atm_liq_ok = (float(atm_row["open_int"]) > 0.0) and (float(atm_row["volume"]) > 0.0)
            otm_liq_ok = (float(otm_row["open_int"]) > 0.0) and (float(otm_row["volume"]) > 0.0)
            if not (atm_liq_ok and otm_liq_ok):
                self.logger.critical(
                    "ABORT BCS: illiquid leg(s) on entry_date=%s symbol=%s expiry_dt=%s "
                    "(ATM oi=%.2f vol=%.2f | OTM oi=%.2f vol=%.2f).",
                    entry_date.date(),
                    symbol,
                    expiry_dt.date(),
                    float(atm_row["open_int"]),
                    float(atm_row["volume"]),
                    float(otm_row["open_int"]),
                    float(otm_row["volume"]),
                )
                aborted += 1
                continue

            parent_trade_id = f"{symbol}_BCS_{entry_date:%Y%m%d}"

            # IMPORTANT: compute_mtm_pnl_rupee groups by trade_id only,
            # so each leg must have its own unique trade_id to avoid settle_prev interleaving.
            trades_out.append(
                {
                    "trade_id": f"{parent_trade_id}{self.LONG_SUFFIX}",
                    "symbol": symbol,
                    "instrument": str(atm_row["instrument"]),
                    "expiry_dt": expiry_dt,
                    "strike_pr": int(atm_strike),
                    "option_typ": "CE",
                    "entry_date": entry_date,
                    "position_sign": +1,
                }
            )
            trades_out.append(
                {
                    "trade_id": f"{parent_trade_id}{self.SHORT_SUFFIX}",
                    "symbol": symbol,
                    "instrument": str(otm_row["instrument"]),
                    "expiry_dt": expiry_dt,
                    "strike_pr": int(otm_strike),
                    "option_typ": "CE",
                    "entry_date": entry_date,
                    "position_sign": -1,
                }
            )

        trades_df = pd.DataFrame(trades_out)
        if trades_df.empty:
            self.logger.warning("No trades produced by BullCallSpreadStrategy.build_trades().")
            return trades_df

        # Abort-ratio guard (relative to attempted entry days)
        attempted = max(1, len(entry_days))
        ratio = aborted / attempted
        if ratio > float(self.config.max_abort_ratio):
            raise DataIntegrityError(
                f"Too many BCS entries aborted: {aborted}/{attempted} ({ratio:.2%}). "
                "This likely indicates missing monthly chains or bad filtering."
            )

        return trades_df.sort_values(["trade_id"], kind="mergesort").reset_index(drop=True)

    def _drop_unsynced_parent_trades(self, legs_mtm: pd.DataFrame) -> pd.DataFrame:
        legs = legs_mtm.copy()
        legs["parent_trade_id"] = legs["trade_id"].map(self._parent_trade_id_from_leg_trade_id)

        counts = legs.groupby(["parent_trade_id", "date"])["trade_id"].nunique(dropna=False)
        bad_parents = counts[counts != 2].reset_index()["parent_trade_id"].unique().tolist()

        if not bad_parents:
            return legs.drop(columns=["parent_trade_id"])

        self.logger.critical("ABORT BCS parent trade(s): leg date-calendar mismatch: %s", bad_parents)

        before = legs["parent_trade_id"].nunique()
        out = legs.loc[~legs["parent_trade_id"].isin(bad_parents)].copy()
        after = out["parent_trade_id"].nunique()

        aborted = before - after
        ratio = aborted / max(1, before)
        self.logger.warning("Dropped unsynced BCS parent trades: %d/%d (%.2f%%).", aborted, before, 100.0 * ratio)

        if ratio > float(self.config.max_abort_ratio):
            raise DataIntegrityError(
                f"Too many parent trades aborted due to leg synchronization failures: "
                f"{aborted}/{before} ({ratio:.2%})."
            )

        return out.drop(columns=["parent_trade_id"])

    def _aggregate_legs_to_parent(self, legs_mtm: pd.DataFrame, trades_df: pd.DataFrame) -> pd.DataFrame:
        legs = legs_mtm.copy()
        legs["parent_trade_id"] = legs["trade_id"].map(self._parent_trade_id_from_leg_trade_id)

        meta = trades_df[["trade_id", "expiry_dt", "strike_pr", "position_sign", "symbol"]].copy()
        meta["parent_trade_id"] = meta["trade_id"].map(self._parent_trade_id_from_leg_trade_id)

        parent_meta = (
            meta.groupby(["parent_trade_id", "symbol", "expiry_dt"], as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "atm_strike": int(g.loc[g["position_sign"] == 1, "strike_pr"].iloc[0]),
                        "otm_strike": int(g.loc[g["position_sign"] == -1, "strike_pr"].iloc[0]),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )

        daily = (
            legs.groupby(["date", "parent_trade_id", "symbol"], as_index=False)["daily_pnl_rupee"]
            .sum()
            .rename(columns={"daily_pnl_rupee": "strategy_pnl_rupee", "parent_trade_id": "trade_id"})
        )

        daily = daily.merge(
            parent_meta.rename(columns={"parent_trade_id": "trade_id"}),
            on=["trade_id", "symbol"],
            how="left",
        )

        if daily[["atm_strike", "otm_strike"]].isna().any().any():
            bad = daily.loc[daily["atm_strike"].isna() | daily["otm_strike"].isna(), "trade_id"].unique().tolist()
            raise ValueError(f"Could not resolve strikes for parent trade_id(s): {bad}")

        daily = daily.sort_values(["trade_id", "date"], kind="mergesort").reset_index(drop=True)
        daily["cum_pnl_rupee"] = daily.groupby("trade_id")["strategy_pnl_rupee"].cumsum()

        return daily[["date", "symbol", "atm_strike", "otm_strike", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"]]

    def run(self) -> pd.DataFrame:
        market_df = self.load_market_data()
        self._validate_market_has_liquidity_columns(market_df)

        entry_days = self.identify_entry_days(market_df)
        trades_df = self.build_trades(market_df, entry_days)

        if trades_df.empty:
            return pd.DataFrame(
                columns=["date", "symbol", "atm_strike", "otm_strike", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"]
            )

        legs_mtm = self.compute_mtm_pnl_rupee(market_df, trades_df, schema=TradeBlotterSchema())
        legs_mtm = self._drop_unsynced_parent_trades(legs_mtm)

        if legs_mtm.empty:
            return pd.DataFrame(
                columns=["date", "symbol", "atm_strike", "otm_strike", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"]
            )

        return self._aggregate_legs_to_parent(legs_mtm, trades_df=trades_df)
