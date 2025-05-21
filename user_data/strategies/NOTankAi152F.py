import logging
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import talib.abstract as ta
import pandas_ta as pta  # pandas_ta is imported but not explicitly used in the provided code.
# If it's for future use or part of an older version, that's okay.
# Otherwise, it can be removed if not needed.
from scipy.signal import argrelextrema

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]


class NOTankAi152F(IStrategy):
    """
    Enhanced strategy on the 15-minute timeframe.

    Key improvements:
      - Dynamic stoploss based on ATR.
      - Dynamic leverage calculation.
      - Murrey Math level calculation (rolling window for performance).
      - Enhanced DCA (Average Price) logic.
      - Translated to English and code structured for clarity.
      - Parameterization of internal constants for optimization.
    """

    # General strategy parameters
    timeframe = "15m"
    startup_candle_count: int = 200
    stoploss = -0.99
    trailing_stop = False
    position_adjustment_enable = True
    can_short = False
    use_exit_signal = True
    ignore_roi_if_entry_signal = True

    max_entry_position_adjustment = 7
    process_only_new_candles = True

    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 6, default=2, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)
    indicator_mml_window = IntParameter(32, 128, default=64, space="buy", optimize=True, load=True)

    # Dynamic Stoploss parameters
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="sell", optimize=True,
                                               load=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.10, default=-0.20, decimals=2, space="sell", optimize=True,
                                               load=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=10.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 10, default=5, space="buy", optimize=True, load=True)
    indicator_rolling_window_threshold = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    indicator_rolling_check_window = IntParameter(2, 10, default=4, space="buy", optimize=True, load=True)

    # ROI table (minutes to decimal)
    minimal_roi = {
        "0": 0.5,
        "60": 0.45,
        "120": 0.4,
        "240": 0.3,
        "360": 0.25,
        "720": 0.2,
        "1440": 0.15,
        "2880": 0.1,
        "3600": 0.05,
        "7200": 0.02,
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {},
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "line"},
                "minima": {"color": "#aac7fc", "type": "line"},
                "maxima_check": {"color": "#a29db9", "type": "line"},
                "minima_check": {"color": "#aac7fc", "type": "line"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},
                "[8/8]P": {"color": "red", "type": "line"},
                "[0/8]P": {"color": "red", "type": "line"},
            }
        },
    }
    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        rolling_high = df["high"].rolling(window=window_size, min_periods=window_size).max()
        rolling_low = df["low"].rolling(window=window_size, min_periods=window_size).min()
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        for i in range(len(df)):
            if i < window_size - 1:
                continue
            mn_period = rolling_low.iloc[i]
            mx_period = rolling_high.iloc[i]
            current_close = df["close"].iloc[i]
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue
            finalH_period = mx_period
            finalL_period = mn_period
            if finalH_period == finalL_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][i] = current_close
                continue
            levels = NOTankAi152F._calculate_mml_core(mn_period, finalH_period, mx_period, finalL_period, mml_c1, mml_c2)
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][i] = levels.get(key, current_close)
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        calculated_max_dca_multiplier = 1.0
        if self.position_adjustment_enable:
            num_safety_orders = int(self.max_safety_orders.value)
            volume_scale = self.safety_order_volume_scale.value
            if num_safety_orders > 0 and volume_scale > 0:
                current_order_relative_size = 1.0
                for _ in range(num_safety_orders):
                    current_order_relative_size *= volume_scale
                    calculated_max_dca_multiplier += current_order_relative_size
            else:
                logger.warning(f"{pair}: Could not calculate max_dca_multiplier due to "
                               f"invalid max_safety_orders ({num_safety_orders}) or "
                               f"safety_order_volume_scale ({volume_scale}). Defaulting to 1.0.")
        else:
            logger.debug(f"{pair}: Position adjustment not enabled. max_dca_multiplier is 1.0.")

        if calculated_max_dca_multiplier > 0:
            stake_amount = proposed_stake / calculated_max_dca_multiplier
            logger.info(f"{pair} Initial stake calculated: {stake_amount:.8f} (Proposed: {proposed_stake:.8f}, "
                        f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f})")
            if min_stake is not None and stake_amount < min_stake:
                logger.info(f"{pair} Initial stake {stake_amount:.8f} was below min_stake {min_stake:.8f}. "
                            f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
                stake_amount = min_stake
            return stake_amount
        else:
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty:
            logger.warning(f"{pair} Empty DataFrame in custom_entry_price. Returning proposed_rate.")
            return proposed_rate
        last_candle = dataframe.iloc[-1]
        entry_price = (last_candle["close"] + last_candle["open"] + proposed_rate) / 3.0
        if side == "long":
            if proposed_rate < entry_price:
                entry_price = proposed_rate
        elif side == "short":
            if proposed_rate > entry_price:
                entry_price = proposed_rate
        logger.info(
            f"{pair} Calculated Entry Price: {entry_price:.8f} | Last Close: {last_candle['close']:.8f}, "
            f"Last Open: {last_candle['open']:.8f}, Proposed Rate: {proposed_rate:.8f}")
        if self.last_entry_price is not None and abs(entry_price - self.last_entry_price) < 0.000005:
            increment_factor = self.increment_for_unique_price.value if side == "long" else (
                1.0 / self.increment_for_unique_price.value)
            entry_price *= increment_factor
            logger.info(
                f"{pair} Entry price incremented to {entry_price:.8f} (previous: {self.last_entry_price:.8f}) due to proximity.")
        self.last_entry_price = entry_price
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe.empty or 'atr' not in dataframe.columns or dataframe['atr'].isnull().all():
            logger.warning(
                f"{pair} ATR not available or all NaN for dynamic stoploss. Using default stoploss: {self.stoploss}")
            return self.stoploss
        last_atr = dataframe["atr"].iat[-1]
        if pd.isna(last_atr) or last_atr == 0:
            valid_atr = dataframe["atr"].dropna()
            if not valid_atr.empty:
                last_atr = valid_atr.iat[-1]
            else:
                logger.warning(
                    f"{pair} All ATR values are NaN or no valid ATR found. Using fallback stoploss -0.10.")
                return -0.10
        if last_atr == 0:
            logger.warning(f"{pair} ATR is 0. Using fallback stoploss (-0.10 for 10%).")
            return -0.10
        atr_multiplier = self.stoploss_atr_multiplier.value
        if current_rate == 0:
            logger.warning(
                f"{pair} Current rate is 0. Cannot compute dynamic stoploss. Using default: {self.stoploss}")
            return self.stoploss
        dynamic_sl_ratio = atr_multiplier * last_atr / current_rate
        calculated_stoploss = -abs(dynamic_sl_ratio)
        max_reasonable_sl = self.stoploss_max_reasonable.value
        final_stoploss = max(calculated_stoploss, max_reasonable_sl)
        logger.info(
            f"{pair} Dynamic Stoploss: {final_stoploss:.4f} (ATR: {last_atr:.8f}, Rate: {current_rate:.8f}, SL Ratio: {dynamic_sl_ratio:.4f})")
        return final_stoploss
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float,
                           time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        current_profit_ratio = trade.calc_profit_ratio(rate)
        logger.info(f"{pair} Confirming exit due to reason '{exit_reason}'. Current profit: {current_profit_ratio:.4f}")
        if exit_reason == "trailing_stop_loss":
            logger.info(f"{pair} Trailing stop loss triggered. Exit allowed.")
            return True
        if exit_reason == "partial_exit" and current_profit_ratio < 0:
            logger.info(
                f"{pair} Exit signal '{exit_reason}' for partial exit with a loss was rejected. Profit: {current_profit_ratio:.4f} < 0.")
            return False
        return True

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> Optional[float]:
        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits

        # --- Profit-taking logic ---
        if current_profit > 0.25 and count_of_exits == 0:
            logger.info(
                f"{trade.pair} Taking partial profit (25% of current position value) at {current_profit:.2%}")
            if trade.amount == 0:
                logger.warning(f"{trade.pair} Attempted partial exit, but trade.amount is 0. No action taken.")
                return None
            amount_to_sell_in_stake = (trade.amount * current_rate) * 0.25
            return -amount_to_sell_in_stake

        if current_profit > 0.40 and count_of_exits == 1:
            logger.info(
                f"{trade.pair} Taking additional profit (approx. 33.3% of current position value) at {current_profit:.2%}")
            if trade.amount == 0:
                logger.warning(f"{trade.pair} Attempted partial exit, but trade.amount is 0. No action taken.")
                return None
            amount_to_sell_in_stake = (trade.amount * current_rate) * (1 / 3)
            return -amount_to_sell_in_stake

        # --- DCA logic ---
        if not self.position_adjustment_enable:
            return None

        if (current_profit > self.initial_safety_order_trigger.value / 2.0 and count_of_entries == 1) or \
           (current_profit > self.initial_safety_order_trigger.value and count_of_entries == 2) or \
           (current_profit > self.initial_safety_order_trigger.value * 1.5 and count_of_entries == 3):
            logger.info(
                f"{trade.pair} DCA condition met by Freqtrade, but current profit {current_profit:.2%} "
                f"is above strategy threshold for {count_of_entries} entries. Skipping strategy-side DCA.")
            return None

        if count_of_entries >= self.max_safety_orders.value + 1:
            logger.info(
                f"{trade.pair} Max safety orders ({self.max_safety_orders.value}) reached. No more DCA.")
            return None

        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                logger.error(
                    f"{trade.pair} No filled entry orders found for DCA calculation, although entry count is {count_of_entries}.")
                return None
            last_order_cost = filled_entry_orders[-1].cost
            dca_stake_amount = abs(last_order_cost * self.safety_order_volume_scale.value)
            if min_stake is not None and dca_stake_amount < min_stake:
                logger.warning(
                    f"{trade.pair} DCA stake {dca_stake_amount:.8f} below min_stake {min_stake:.8f}. Adjusting to min_stake.")
                dca_stake_amount = min_stake
            if max_stake is not None and (trade.stake_amount + dca_stake_amount) > max_stake:
                available_for_dca = max_stake - trade.stake_amount
                if dca_stake_amount > available_for_dca and available_for_dca > (min_stake or 0):
                    logger.warning(
                        f"{trade.pair} DCA stake {dca_stake_amount:.8f} reduced to {available_for_dca:.8f} due to max_stake limit.")
                    dca_stake_amount = available_for_dca
                elif available_for_dca <= (min_stake or 0):
                    logger.warning(
                        f"{trade.pair} Cannot DCA. Adding {available_for_dca:.8f} would exceed max_stake or is below min_stake.")
                    return None
            logger.info(f"{trade.pair} Adjusting position with DCA. Adding {dca_stake_amount:.8f}. "
                        f"Entry count: {count_of_entries}, Max safety: {self.max_safety_orders.value}")
            return dca_stake_amount
        except IndexError:
            logger.error(
                f"Error calculating DCA stake for {trade.pair}: IndexError accessing last_order. Filled orders: {filled_entry_orders}")
            return None
        except Exception as e:
            logger.error(f"Error calculating DCA stake for {trade.pair}: {e}")
            return None

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        extrema_order = self.indicator_extrema_order.value
        maxima_indices = argrelextrema(dataframe["close"].values, np.greater, order=extrema_order)[0]
        minima_indices = argrelextrema(dataframe["close"].values, np.less, order=extrema_order)[0]

        dataframe["maxima"] = 0
        dataframe.loc[dataframe.index[maxima_indices], "maxima"] = 1
        dataframe["minima"] = 0
        dataframe.loc[dataframe.index[minima_indices], "minima"] = 1
        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe.index[minima_indices], "s_extrema"] = -1
        dataframe.loc[dataframe.index[maxima_indices], "s_extrema"] = 1

        # Murrey Math levels
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels(dataframe, window_size=mml_window)
        for level_name, level_series in murrey_levels.items():
            dataframe[level_name] = level_series

        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator.replace(0, np.nan))
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).max()

        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0).astype(int)

        pair = metadata.get("pair", "UNKNOWN_PAIR")
        if len(dataframe) > rolling_check_window and len(dataframe) > extrema_order:
            if not dataframe[['maxima', 'maxima_check']].empty:
                if dataframe["maxima"].iloc[-3] == 1 and dataframe["maxima_check"].iloc[-1] == 0:
                    pass
                if dataframe["minima"].iloc[-3] == 1 and dataframe["minima_check"].iloc[-1] == 0:
                    pass
        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[
            (df["DI_catch"] == 1) &
            (df["maxima_check"] == 1) &
            (df["s_extrema"] < 0) &
            (df["minima"].shift(1) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 35),
            ["enter_long", "enter_tag"]
        ] = (1, "Confirmed_Min_Entry")

        df.loc[
            (df["minima_check"] == 0) &
            (df["volume"] > 0) &
            (df["rsi"] < 30),
            ["enter_long", "enter_tag"]
        ] = (1, "Aggressive_Min_Entry")

        df.loc[
            (df["DI_catch"] == 1) &
            (df["minima_check"] == 0) &
            (df["minima_check"].shift(5) == 1) &
            (df["volume"] > 0) &
            (df["rsi"] < 32),
            ["enter_long", "enter_tag"]
        ] = (1, "Transitional_Min_Entry")

        if self.can_short:
            df.loc[
                (df["DI_catch"] == 1) &
                (df["minima_check"] == 1) &
                (df["s_extrema"] > 0) &
                (df["maxima"].shift(1) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 65),
                ["enter_short", "enter_tag"]
            ] = (1, "Confirmed_Max_Entry")

            df.loc[
                (df["maxima_check"] == 0) &
                (df["volume"] > 0) &
                (df["rsi"] > 70),
                ["enter_short", "enter_tag"]
            ] = (1, "Aggressive_Max_Entry")

            df.loc[
                (df["DI_catch"] == 1) &
                (df["maxima_check"] == 0) &
                (df["maxima_check"].shift(5) == 1) &
                (df["volume"] > 0) &
                (df["rsi"] > 68),
                ["enter_short", "enter_tag"]
            ] = (1, "Transitional_Max_Entry")
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df.loc[
            (df["maxima_check"] == 0) &
            (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "Exit_Max_Check")

        df.loc[
            (df["DI_catch"] == 1) &
            (df["s_extrema"] > 0) &
            (df["maxima"].shift(1) == 1) &
            (df["volume"] > 0),
            ["exit_long", "exit_tag"]
        ] = (1, "Exit_Max_Confirmed")

        if self.can_short:
            df.loc[
                (df["minima_check"] == 0) &
                (df["volume"] > 0),
                ["exit_short", "exit_tag"]
            ] = (1, "Exit_Min_Check")

            df.loc[
                (df["DI_catch"] == 1) &
                (df["s_extrema"] < 0) &
                (df["minima"].shift(1) == 1) &
                (df["volume"] > 0),
                ["exit_short", "exit_tag"]
            ] = (1, "Exit_Min_Confirmed")
        return df
