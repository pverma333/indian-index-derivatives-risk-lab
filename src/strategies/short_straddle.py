from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

if __package__ in (None, ""):
    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    from src.strategies import contract_selectors as cs  # type: ignore
    from src.strategies import expiry_selectors as es  # type: ignore
else:
    from . import contract_selectors as cs  # type: ignore
    from . import expiry_selectors as es  # type: ignore


_REQUIRED_CFG_KEYS = [
    "symbol",
    "tenor",
    "qty_lots",
    "strike_band_n",
    "max_atm_search_steps",
    "liquidity_mode",
    "min_contracts",
    "min_open_int",
    "liquidity_percentile",
    "exit_rule",
    "exit_k_days",
    "fees_bps",
    "fixed_fee_per_lot",
    "width_points",
    "otm_distance_points",
]


@dataclass(frozen=True)
class _SkipContext:
    strategy_name: str
    symbol: str
    tenor: str
    expiry_dt: pd.Timestamp
    entry_date: pd.Timestamp
    reason: str

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "tenor": self.tenor,
            "expiry_dt": str(self.expiry_dt.date()),
            "entry_date": str(self.entry_date.date()),
            "reason": self.reason,
        }


class ShortStraddleStrategy:
    strategy_name = "short_straddle"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or {}

    def build_trades(self, market_df: pd.DataFrame, cfg: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        effective_cfg = dict(self.cfg)
        if cfg:
            effective_cfg.update(cfg)

        self._validate_cfg(effective_cfg)

        symbol: str = effective_cfg["symbol"]
        tenor: str = effective_cfg["tenor"]

        cycles_df = es.build_expiry_cycles(market_df=market_df, symbol=symbol, tenor=tenor)
        if cycles_df.empty:
            logger.info(
                "No cycles produced for short straddle.",
                extra={"strategy_name": self.strategy_name, "symbol": symbol, "tenor": tenor},
            )
            return self._empty_trades_df()

        cycles_df = cycles_df.sort_values(["expiry_dt", "entry_date"]).reset_index(drop=True)

        rows: List[Dict[str, Any]] = []
        leg_seq = 0

        for _, cyc in cycles_df.iterrows():
            expiry_dt = pd.Timestamp(cyc["expiry_dt"]).normalize()
            entry_date = pd.Timestamp(cyc["entry_date"]).normalize()
            exit_date = pd.Timestamp(cyc["exit_date"]).normalize()

            chain_df = cs.get_chain(market_df=market_df, symbol=symbol, expiry_dt=expiry_dt, entry_date=entry_date)
            if chain_df.empty:
                self._log_skip(_SkipContext(self.strategy_name, symbol, tenor, expiry_dt, entry_date, "EMPTY_CHAIN"))
                continue

            strike_interval_used = cs.infer_strike_interval(chain_df)

            if "spot_close" not in chain_df.columns or chain_df["spot_close"].isna().all():
                self._log_skip(
                    _SkipContext(self.strategy_name, symbol, tenor, expiry_dt, entry_date, "MISSING_SPOT_CLOSE")
                )
                continue
            spot_close = float(chain_df["spot_close"].iloc[0])

            band_df = cs.apply_strike_band(
                chain_df=chain_df,
                spot_close=spot_close,
                strike_band_n=int(effective_cfg["strike_band_n"]),
            )

            filt_df = cs.apply_liquidity_filters(
                band_df=band_df,
                liquidity_mode=str(effective_cfg["liquidity_mode"]),
                min_contracts=int(effective_cfg["min_contracts"]),
                min_open_int=int(effective_cfg["min_open_int"]),
                liquidity_percentile=int(effective_cfg["liquidity_percentile"]),
            )

            atm_strike = cs.select_atm_strike(
                filt_df=filt_df,
                spot_close=spot_close,
                max_atm_search_steps=int(effective_cfg["max_atm_search_steps"]),
            )

            if atm_strike is None:
                self._log_skip(_SkipContext(self.strategy_name, symbol, tenor, expiry_dt, entry_date, "NO_VALID_ATM_STRIKE"))
                continue

            has_ce = ((filt_df["strike_pr"] == atm_strike) & (filt_df["option_typ"] == "CE")).any()
            has_pe = ((filt_df["strike_pr"] == atm_strike) & (filt_df["option_typ"] == "PE")).any()
            if not (has_ce and has_pe):
                self._log_skip(_SkipContext(self.strategy_name, symbol, tenor, expiry_dt, entry_date, "ATM_MISSING_CE_OR_PE"))
                continue

            trade_id = self._make_trade_id(symbol=symbol, tenor=tenor, expiry_dt=expiry_dt, entry_date=entry_date)

            for opt_type in ["CE", "PE"]:
                leg_seq += 1
                leg_id = f"{trade_id}::L{leg_seq:06d}"

                rows.append(
                    {
                        "strategy_name": self.strategy_name,
                        "tenor": tenor,
                        "trade_id": trade_id,
                        "leg_id": leg_id,
                        "symbol": symbol,
                        "instrument": "OPTIDX",
                        "expiry_dt": expiry_dt,
                        "strike_pr": float(atm_strike),
                        "option_typ": opt_type,
                        "side": -1,
                        "qty_lots": int(effective_cfg["qty_lots"]),
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "strike_band_n": int(effective_cfg["strike_band_n"]),
                        "width_points": effective_cfg["width_points"],
                        "otm_distance_points": effective_cfg["otm_distance_points"],
                        "max_atm_search_steps": int(effective_cfg["max_atm_search_steps"]),
                        "liquidity_mode": str(effective_cfg["liquidity_mode"]),
                        "min_contracts": int(effective_cfg["min_contracts"]),
                        "min_open_int": int(effective_cfg["min_open_int"]),
                        "liquidity_percentile": int(effective_cfg["liquidity_percentile"]),
                        "exit_rule": str(effective_cfg["exit_rule"]),
                        "exit_k_days": effective_cfg["exit_k_days"],
                        "fees_bps": float(effective_cfg["fees_bps"]),
                        "fixed_fee_per_lot": float(effective_cfg["fixed_fee_per_lot"]),
                        "strike_interval_used": strike_interval_used,
                    }
                )

        if not rows:
            return self._empty_trades_df()

        trades_df = pd.DataFrame(rows)
        trades_df["qty_lots"] = trades_df["qty_lots"].astype(int)
        trades_df["side"] = trades_df["side"].astype(int)
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"]).dt.normalize()
        trades_df["exit_date"] = pd.to_datetime(trades_df["exit_date"]).dt.normalize()
        trades_df["expiry_dt"] = pd.to_datetime(trades_df["expiry_dt"]).dt.normalize()
        return trades_df

    @staticmethod
    def _make_trade_id(symbol: str, tenor: str, expiry_dt: pd.Timestamp, entry_date: pd.Timestamp) -> str:
        return f"short_straddle|{symbol}|{tenor}|{expiry_dt:%Y%m%d}|{entry_date:%Y%m%d}"

    @staticmethod
    def _log_skip(ctx: _SkipContext) -> None:
        logger.info("Skipping short straddle cycle.", extra=ctx.to_log_dict())

    @staticmethod
    def _empty_trades_df() -> pd.DataFrame:
        cols = [
            "strategy_name",
            "tenor",
            "trade_id",
            "leg_id",
            "symbol",
            "instrument",
            "expiry_dt",
            "strike_pr",
            "option_typ",
            "side",
            "qty_lots",
            "entry_date",
            "exit_date",
            "strike_band_n",
            "width_points",
            "otm_distance_points",
            "max_atm_search_steps",
            "liquidity_mode",
            "min_contracts",
            "min_open_int",
            "liquidity_percentile",
            "exit_rule",
            "exit_k_days",
            "fees_bps",
            "fixed_fee_per_lot",
            "strike_interval_used",
        ]
        return pd.DataFrame(columns=cols)

    @staticmethod
    def _validate_cfg(cfg: Dict[str, Any]) -> None:
        missing = [k for k in _REQUIRED_CFG_KEYS if k not in cfg]
        if missing:
            raise ValueError(f"ShortStraddleStrategy cfg missing required keys: {missing}")

        tenor = str(cfg["tenor"])
        if tenor not in {"WEEKLY", "MONTHLY"}:
            raise ValueError(f"ShortStraddleStrategy expects single tenor per call, got tenor={tenor!r}")

        qty_lots = cfg["qty_lots"]
        if not isinstance(qty_lots, int) or qty_lots < 1:
            raise ValueError(f"qty_lots must be int >= 1, got {qty_lots!r}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    print("short_straddle.py imported OK.")
    print("Note: This file is meant to be imported by the engine/tests.")
