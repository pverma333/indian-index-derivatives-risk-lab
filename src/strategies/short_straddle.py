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
class ShortStraddleStrategyConfig(BaseStrategyConfig):
    symbol: str = "NIFTY"
    strike_interval: int = 50

    # Spot resolution inside market_df
    spot_instrument_candidates: tuple[str, ...] = ("SPOT", "IDX", "INDEX", "NIFTY_SPOT")
    spot_settlement_use_close: bool = True

    intrinsic_tolerance: float = 1e-6

    # If too many parent straddles get aborted due to sync mismatch, hard fail
    max_abort_ratio: float = 0.25

    # Optional filter if curated set contains multiple option instruments
    option_instrument: Optional[str] = None


class ShortStraddleStrategy(BaseStrategy):
    OPTION_TYPES = ("CE", "PE")
    REQUIRED_LIQ_COLS = ("open_int", "volume")
    LEG_SUFFIX_SEP = "__"  # used to build per-leg trade_id

    def __init__(self, config: ShortStraddleStrategyConfig, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(config=config, logger=logger)
        self.config: ShortStraddleStrategyConfig = config

    @staticmethod
    def _parent_trade_id_from_leg_trade_id(trade_id: str) -> str:
        # "NIFTY_..._12000__CE" -> "NIFTY_..._12000"
        return trade_id.split(ShortStraddleStrategy.LEG_SUFFIX_SEP, 1)[0]

    def _validate_market_has_liquidity_columns(self, market_df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_LIQ_COLS if c not in market_df.columns]
        if missing:
            raise ValueError(
                f"Market data missing required liquidity columns for straddle: {missing}. "
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

    def _get_spot_price(self, market_df: pd.DataFrame, symbol: str, on_date: pd.Timestamp) -> float:
        day = market_df.loc[(market_df["symbol"] == symbol) & (market_df["date"] == on_date)].copy()
        if day.empty:
            raise ValueError(f"No rows found for spot resolution: symbol={symbol}, date={on_date.date()}")

        # Prefer explicit spot instruments
        for inst in self.config.spot_instrument_candidates:
            cand = day.loc[day["instrument"] == inst]
            if not cand.empty:
                cand = cand.sort_values(["instrument", "expiry_dt", "strike_pr", "option_typ"], kind="mergesort")
                return float(cand["close"].iloc[0])

        # Fallback: non-option rows
        cand = day.loc[day["option_typ"].isna() | (~day["option_typ"].isin(self.OPTION_TYPES))]
        if cand.empty:
            raise ValueError(
                f"Could not resolve spot close: symbol={symbol}, date={on_date.date()}. "
                "No instrument match and no non-option rows available."
            )
        cand = cand.sort_values(["instrument", "expiry_dt", "strike_pr"], kind="mergesort")
        return float(cand["close"].iloc[0])

    def _select_atm_strike(self, option_rows_on_entry: pd.DataFrame, spot_close: float) -> int:
        step = int(self.config.strike_interval)
        if step <= 0:
            raise ValueError("strike_interval must be positive.")

        df = option_rows_on_entry.copy()
        df = df.loc[df["strike_pr"].notna()]
        df = df.loc[(df["strike_pr"].astype(int) % step) == 0]

        if df.empty:
            raise ValueError("No candidate strikes after strike_interval filtering.")

        strikes = df["strike_pr"].astype(int).drop_duplicates().sort_values(kind="mergesort")
        diffs = (strikes - spot_close).abs()
        min_diff = diffs.min()
        best = strikes.loc[diffs == min_diff].min()  # deterministic tie-break
        return int(best)

    def build_trades(self, market_df: pd.DataFrame, entry_days: pd.DataFrame) -> pd.DataFrame:
        self._validate_market_has_liquidity_columns(market_df)

        symbol = self.config.symbol
        entry_days = entry_days.loc[entry_days["symbol"] == symbol].copy()
        entry_days["entry_date"] = pd.to_datetime(entry_days["entry_date"]).dt.normalize()
        entry_days = entry_days.sort_values(["entry_date"], kind="mergesort")

        monthly_expiries = self._get_monthly_expiry_dates(market_df, symbol=symbol)
        if monthly_expiries.empty:
            raise ValueError(f"No monthly expiry dates found for symbol={symbol} (is_opt_monthly_expiry==True).")

        option_df = market_df.loc[
            (market_df["symbol"] == symbol) & (market_df["option_typ"].isin(self.OPTION_TYPES))
        ].copy()

        if self.config.option_instrument is not None:
            option_df = option_df.loc[option_df["instrument"] == self.config.option_instrument].copy()

        trades_out: list[dict] = []

        for entry_date in entry_days["entry_date"].tolist():
            expiry_dt = self._next_monthly_expiry(monthly_expiries, entry_date=entry_date)
            if expiry_dt is None:
                self.logger.warning("No next monthly expiry after entry_date=%s; skipping.", entry_date.date())
                continue

            rows_on_entry = option_df.loc[(option_df["date"] == entry_date) & (option_df["expiry_dt"] == expiry_dt)]
            if rows_on_entry.empty:
                self.logger.critical(
                    "ABORT straddle: no option rows on entry_date=%s for expiry_dt=%s (symbol=%s).",
                    entry_date.date(),
                    expiry_dt.date(),
                    symbol,
                )
                continue

            spot_close = self._get_spot_price(market_df, symbol=symbol, on_date=entry_date)
            atm_strike = self._select_atm_strike(rows_on_entry, spot_close=spot_close)

            ce = rows_on_entry.loc[
                (rows_on_entry["strike_pr"].astype(int) == atm_strike) & (rows_on_entry["option_typ"] == "CE")
            ]
            pe = rows_on_entry.loc[
                (rows_on_entry["strike_pr"].astype(int) == atm_strike) & (rows_on_entry["option_typ"] == "PE")
            ]

            if ce.empty or pe.empty:
                self.logger.critical(
                    "ABORT straddle: missing leg(s) on entry_date=%s symbol=%s expiry_dt=%s strike=%s "
                    "(CE_rows=%d, PE_rows=%d).",
                    entry_date.date(),
                    symbol,
                    expiry_dt.date(),
                    atm_strike,
                    len(ce),
                    len(pe),
                )
                continue

            ce_row = ce.sort_values(["instrument", "lot_size"], kind="mergesort").iloc[0]
            pe_row = pe.sort_values(["instrument", "lot_size"], kind="mergesort").iloc[0]

            # Symbol / lot / instrument consistency
            if ce_row["symbol"] != pe_row["symbol"]:
                self.logger.critical(
                    "ABORT straddle: symbol mismatch CE=%s PE=%s on entry_date=%s strike=%s expiry=%s.",
                    ce_row["symbol"],
                    pe_row["symbol"],
                    entry_date.date(),
                    atm_strike,
                    expiry_dt.date(),
                )
                continue
            if int(ce_row["lot_size"]) != int(pe_row["lot_size"]):
                self.logger.critical(
                    "ABORT straddle: lot_size mismatch CE=%s PE=%s on entry_date=%s strike=%s expiry=%s.",
                    ce_row["lot_size"],
                    pe_row["lot_size"],
                    entry_date.date(),
                    atm_strike,
                    expiry_dt.date(),
                )
                continue
            if ce_row["instrument"] != pe_row["instrument"]:
                self.logger.critical(
                    "ABORT straddle: instrument mismatch CE=%s PE=%s on entry_date=%s strike=%s expiry=%s.",
                    ce_row["instrument"],
                    pe_row["instrument"],
                    entry_date.date(),
                    atm_strike,
                    expiry_dt.date(),
                )
                continue

            # Liquidity guard
            ce_liq_ok = (float(ce_row["open_int"]) > 0.0) and (float(ce_row["volume"]) > 0.0)
            pe_liq_ok = (float(pe_row["open_int"]) > 0.0) and (float(pe_row["volume"]) > 0.0)
            if not (ce_liq_ok and pe_liq_ok):
                self.logger.critical(
                    "ABORT straddle: illiquid leg(s) on entry_date=%s symbol=%s expiry_dt=%s strike=%s "
                    "(CE_oi=%.2f CE_vol=%.2f | PE_oi=%.2f PE_vol=%.2f).",
                    entry_date.date(),
                    symbol,
                    expiry_dt.date(),
                    atm_strike,
                    float(ce_row["open_int"]),
                    float(ce_row["volume"]),
                    float(pe_row["open_int"]),
                    float(pe_row["volume"]),
                )
                continue

            parent_trade_id = f"{symbol}_STRADDLE_{entry_date:%Y%m%d}_{expiry_dt:%Y%m%d}_{atm_strike}"

            # IMPORTANT: compute_mtm_pnl_rupee groups by trade_id only,
            # so each leg must have its own unique trade_id.
            for opt_typ in ("CE", "PE"):
                leg_trade_id = f"{parent_trade_id}{self.LEG_SUFFIX_SEP}{opt_typ}"
                trades_out.append(
                    {
                        "trade_id": leg_trade_id,
                        "symbol": symbol,
                        "instrument": str(ce_row["instrument"]),
                        "expiry_dt": expiry_dt,
                        "strike_pr": int(atm_strike),
                        "option_typ": opt_typ,
                        "entry_date": entry_date,
                        "position_sign": -1,  # short per ticket; matches schema comment too:contentReference[oaicite:1]{index=1}
                    }
                )

        trades_df = pd.DataFrame(trades_out)
        if trades_df.empty:
            self.logger.warning("No trades produced by ShortStraddleStrategy.build_trades().")
            return trades_df

        # Sanity: each parent should have exactly 2 legs
        trades_df["parent_trade_id"] = trades_df["trade_id"].map(self._parent_trade_id_from_leg_trade_id)
        leg_counts = trades_df.groupby("parent_trade_id")["option_typ"].nunique(dropna=False)
        bad = leg_counts[leg_counts != 2]
        if not bad.empty:
            raise ValueError(f"Internal error: some parent straddles do not have 2 legs: {bad.to_dict()}")

        return trades_df.drop(columns=["parent_trade_id"]).sort_values(["trade_id"], kind="mergesort").reset_index(drop=True)

    def _validate_expiry_intrinsic(self, market_df: pd.DataFrame, legs_mtm: pd.DataFrame, trades_df: pd.DataFrame) -> None:
        meta = trades_df[["trade_id", "expiry_dt", "strike_pr", "option_typ", "symbol"]].copy()
        meta["expiry_dt"] = pd.to_datetime(meta["expiry_dt"]).dt.normalize()

        df = legs_mtm.merge(meta, on=["trade_id", "option_typ", "symbol", "strike_pr"], how="left")
        if df["expiry_dt"].isna().any():
            bad = df.loc[df["expiry_dt"].isna(), "trade_id"].unique().tolist()
            raise ValueError(f"Could not map expiry_dt for trade_id(s): {bad}")

        expiry_rows = df.loc[df["date"] == df["expiry_dt"]].copy()
        if expiry_rows.empty:
            raise ValueError("No expiry rows found in legs MTM output; cannot validate intrinsic settlement.")

        unique_trades = expiry_rows[["trade_id", "symbol", "expiry_dt"]].drop_duplicates()

        spot_settles = []
        for r in unique_trades.itertuples(index=False):
            spot_px = self._get_spot_price(market_df, symbol=r.symbol, on_date=r.expiry_dt)
            spot_settles.append({"trade_id": r.trade_id, "spot_settle": float(spot_px)})

        spot_df = pd.DataFrame(spot_settles)
        expiry_rows = expiry_rows.merge(spot_df, on="trade_id", how="left")

        strike = expiry_rows["strike_pr"].astype(float)
        spot = expiry_rows["spot_settle"].astype(float)

        intrinsic = np.where(
            expiry_rows["option_typ"] == "CE",
            np.maximum(0.0, spot - strike),
            np.maximum(0.0, strike - spot),
        )

        diff = (expiry_rows["settle_pr"].astype(float) - intrinsic).abs()
        bad = expiry_rows.loc[
            diff > float(self.config.intrinsic_tolerance),
            ["trade_id", "option_typ", "date", "strike_pr", "settle_pr", "spot_settle"],
        ]
        if not bad.empty:
            examples = bad.head(10).to_dict(orient="records")
            raise DataIntegrityError(
                "Expiry intrinsic validation failed for option settle_pr. "
                f"Examples (up to 10): {examples}"
            )

    def _drop_unsynced_parent_trades(self, legs_mtm: pd.DataFrame) -> pd.DataFrame:
        legs = legs_mtm.copy()
        legs["parent_trade_id"] = legs["trade_id"].map(self._parent_trade_id_from_leg_trade_id)

        # For each parent trade and date, require both legs
        counts = legs.groupby(["parent_trade_id", "date"])["option_typ"].nunique(dropna=False)
        bad_parents = counts[counts != 2].reset_index()["parent_trade_id"].unique().tolist()

        if not bad_parents:
            return legs_mtm

        self.logger.critical("ABORT straddle parent trade(s): leg date-calendar mismatch: %s", bad_parents)

        before = legs["parent_trade_id"].nunique()
        out = legs.loc[~legs["parent_trade_id"].isin(bad_parents)].copy()
        after = out["parent_trade_id"].nunique()

        aborted = before - after
        ratio = aborted / max(1, before)
        self.logger.warning("Dropped unsynced parent trades: %d/%d (%.2f%%).", aborted, before, 100.0 * ratio)

        if ratio > float(self.config.max_abort_ratio):
            raise DataIntegrityError(
                f"Too many parent trades aborted due to leg synchronization failures: {aborted}/{before} ({ratio:.2%})."
            )

        return out.drop(columns=["parent_trade_id"])

    def run(self) -> pd.DataFrame:
        market_df = self.load_market_data()
        self._validate_market_has_liquidity_columns(market_df)

        entry_days = self.identify_entry_days(market_df)
        trades_df = self.build_trades(market_df, entry_days)

        if trades_df.empty:
            return pd.DataFrame(columns=["date", "symbol", "strike_pr", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"])

        # compute MTM per-leg (trade_id must be per-leg)
        legs_mtm = self.compute_mtm_pnl_rupee(market_df, trades_df, schema=TradeBlotterSchema())

        # expiry intrinsic validation (hard fail)
        self._validate_expiry_intrinsic(market_df, legs_mtm=legs_mtm, trades_df=trades_df)

        # enforce leg synchronization at parent straddle level
        legs_mtm = self._drop_unsynced_parent_trades(legs_mtm)
        if legs_mtm.empty:
            return pd.DataFrame(columns=["date", "symbol", "strike_pr", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"])

        # Aggregate CE+PE into strategy pnl by parent_trade_id
        legs_mtm = legs_mtm.copy()
        legs_mtm["parent_trade_id"] = legs_mtm["trade_id"].map(self._parent_trade_id_from_leg_trade_id)

        daily = (
            legs_mtm.groupby(["date", "parent_trade_id", "symbol", "strike_pr"], as_index=False)["daily_pnl_rupee"]
            .sum()
            .rename(columns={"daily_pnl_rupee": "strategy_pnl_rupee", "parent_trade_id": "trade_id"})
        )
        daily = daily.sort_values(["trade_id", "date"], kind="mergesort")
        daily["cum_pnl_rupee"] = daily.groupby("trade_id")["strategy_pnl_rupee"].cumsum()

        return daily[["date", "symbol", "strike_pr", "strategy_pnl_rupee", "cum_pnl_rupee", "trade_id"]].reset_index(drop=True)
