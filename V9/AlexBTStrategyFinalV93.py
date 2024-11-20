import numpy as np
import pandas_ta as pta
from typing import Optional
import math
import freqtrade.vendor.qtpylib.indicators as qtpylib
import logging
from functools import reduce
from typing import Dict
from datetime import timedelta, datetime, timezone
from freqtrade.persistence import Trade
from pandas import DataFrame, Series
import talib.abstract as ta
from pandas import DataFrame

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy,BooleanParameter
from technical.pivots_points import pivots_points

from scipy.signal import argrelextrema

import json
import logging
from functools import reduce

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
from freqtrade.strategy import IStrategy, RealParameter
from technical.pivots_points import pivots_points

logger = logging.getLogger(__name__)

class AlexBTStrategyFinalV93(IStrategy):
    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.20, -0.01, decimals=3, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(6, 30, name='max_open_trades'),
            ]

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v9.3.15"

    plot_config = {
        "subplots": {
            "T": {
                'T': {'color': 'red'},
            },
            "S": {
                'predicted_S': {'color': 'blue'},
            },
            "R": {
                'predicted_R': {'color': 'yellow'},
            },
            "R2": {
                'predicted_R2': {'color': 'pink'},
            },
            "V": {
                'predicted_V': {'color': 'purple'},
            },
            "V2": {
                'predicted_V2': {'color': 'green'},
            },
        },
    }
    
    
    
    

    # Hyperspace parameters:
    buy_params = {
        # "threshold_buy": 0.59453,
        #'sell_multiplier': 0.6932253866156644,
        "target_pct_change": 9.96036948605142,
        "w0": 0.6304233141068925,
        "w1": 0.7092233514651597,
        "w2": 0.14124677288812248,
        "w3": 0.4763296358666367,
        "w4": 0.4199181354573633,
        "w5": 0.9930040857717386,
        "w6": 0.2891076817113472,
        "w7": 0.90757453371422,
        "w8": 0.36027388382603626,
        "predicted_R_factor": 1.478026518015907,
        "predicted_V_factor": 1.0830431739672923,
        "predicted_S_factor": 1.4946617812677905,
        "predicted_R2_factor": 0.7605720333307935,
        "predicted_V2_factor": 1.464012237821537
    }
    sell_params = {
        # "threshold_sell": 0.80573,
        # "sell_multiplier": sell_multiplier
    }

    # ROI table:
    minimal_roi = {}

    # Stoploss:
    stoploss = -0.08
    max_open_trades = 30

    target_pct_change = DecimalParameter(0.1, 10, default=9.96036, decimals=5, space="buy")
    mem_target_pct_change = target_pct_change.value
    
    # Hyperoptable parameters for predicted components
    predicted_R_factor = DecimalParameter(0.1, 2.0, default=1.47802, decimals=5, space="buy")
    predicted_V_factor = DecimalParameter(0.1, 2.0, default=1.08304, decimals=5, space="buy")
    predicted_S_factor = DecimalParameter(0.1, 2.0, default=1.49466, decimals=5, space="buy")
    predicted_R2_factor = DecimalParameter(0.1, 2.0, default=0.76057, decimals=5, space="buy")
    predicted_V2_factor = DecimalParameter(0.1, 2.0, default=1.46401, decimals=5, space="buy")
    
    mem_predicted_R_factor = predicted_R_factor.value
    mem_predicted_V_factor = predicted_V_factor.value
    mem_predicted_S_factor = predicted_S_factor.value
    mem_predicted_R2_factor = predicted_R2_factor.value
    mem_predicted_V2_factor = predicted_V2_factor.value
    
    #threshold_buy = DecimalParameter(-1, 1, default=0.63042, decimals=5, space="buy")
    #threshold_sell = DecimalParameter(-1, 1,default=0.63042, decimals=5, space='sell')

    # Weights for calculating the aggregate score - normalized to sum to 1
    w0 = DecimalParameter(0, 1, default=0.63042, decimals=5, space="buy")  # moving average (normalized_ma)
    w1 = DecimalParameter(0, 1, default=0.70922, decimals=5, space="buy")  # MACD (normalized_macd)
    w2 = DecimalParameter(0, 1, default=0.14124, decimals=5, space="buy")  # Rate of Change (ROC)
    w3 = DecimalParameter(0, 1, default=0.47632, decimals=5, space="buy")  # RSI (normalized_rsi)
    w4 = DecimalParameter(0, 1, default=0.41991, decimals=5, space="buy")  # Bollinger Band width
    w5 = DecimalParameter(0, 1, default=0.99300, decimals=5, space="buy")  # CCI (normalized_cci)
    w6 = DecimalParameter(0, 1, default=0.28910, decimals=5, space="buy")  # OBV (normalized_obv)
    w7 = DecimalParameter(0, 1, default=0.90757, decimals=5, space="buy")  # ATR (normalized_atr)
    w8 = DecimalParameter(0, 1, default=0.36027, decimals=5, space="buy")  # Stochastic Oscillator (normalized_stoch)

    mem_w0 = w0.value
    mem_w1 = w1.value
    mem_w2 = w2.value
    mem_w3 = w3.value
    mem_w4 = w4.value
    mem_w5 = w5.value
    mem_w6 = w6.value
    mem_w7 = w7.value
    mem_w8 = w8.value

    # Trailing stop:
    use_custom_stoploss = False
    trailing_stop = False
    trailing_only_offset_is_reached = False

    # Define ATR-based dynamic trailing stop and offset
    min_trailing_stop = DecimalParameter(0.05, 0.25, default=0.10, decimals=2, space="buy", optimize = use_custom_stoploss, load = use_custom_stoploss)
    max_trailing_stop = DecimalParameter(0.10, 0.50, default=0.30, decimals=2, space="buy", optimize = use_custom_stoploss, load = use_custom_stoploss)
    min_trailing_offset = DecimalParameter(0.05, 0.30, default=0.10, decimals=2, space="buy", optimize = use_custom_stoploss, load = use_custom_stoploss)
    max_trailing_offset = DecimalParameter(0.10, 0.50, default=0.30, decimals=2, space="buy", optimize = use_custom_stoploss, load = use_custom_stoploss)

    mem_min_trailing_stop = min_trailing_stop.value
    mem_max_trailing_stop = max_trailing_stop.value
    mem_min_trailing_offset = min_trailing_offset.value
    mem_max_trailing_offset = max_trailing_offset.value
    
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    startup_candle_count = 20
    leverage_value = 10.0

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
# #----------------------------------
#         dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
#         dataframe["%-adx-period"] = ta.ADX(dataframe, window=period)
#         dataframe["%-er-period"] = pta.er(dataframe['close'], length=period)
#         dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
#         dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
#         dataframe["%-tcp-period"] = top_percent_change(dataframe, period)
#         dataframe["%-cti-period"] = pta.cti(dataframe['close'], length=period)
#         dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
#         dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe['close'], timeperiod=period)
#         dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
#         dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / dataframe['close'] * 1000
# #----------------------------------

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"]
            - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(dataframe['close'], slowperiod=12,
                                                                                    fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)

        # Step 1: Normalize Indicators:
        # Why? Normalizing the indicators will make them comparable and allow us to assign weights to them.
        # How? We will calculate the z-score of each indicator by subtracting the rolling mean and dividing by the
        # rolling standard deviation. This will give us a normalized value that is centered around 0 with a standard
        # deviation of 1.
        dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe['stoch'].rolling(window=14).std()
        dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe['atr'].rolling(window=14).std()
        dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe['obv'].rolling(window=14).std()
        dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe['close'].rolling(window=10).std()
        dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe['macd'].rolling(window=26).std()
        dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe['roc'].rolling(window=2).std()
        dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                           dataframe['momentum'].rolling(window=4).std()
        dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe['rsi'].rolling(window=10).std()
        dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
            window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
        dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe['cci'].rolling(window=20).std()

        # Dynamic Weights Adjustment
        # Calculate trend strength as the absolute difference between MA and close price
        trend_strength = abs(dataframe['ma'] - dataframe['close'])
        # Calculate rolling mean and stddev once to avoid redundancy
        rolling_mean = trend_strength.rolling(window=14).mean()
        rolling_stddev = trend_strength.rolling(window=14).std()
        # Calculate a more dynamic strong trend threshold
        strong_trend_threshold = rolling_mean + 1.5 * rolling_stddev
        # Determine strong trend condition
        is_strong_trend = trend_strength > strong_trend_threshold
        # Apply dynamic weight adjustment, could also consider a more gradual adjustment
        dataframe['w_momentum'] = self.mem_w3 * (1 + 0.5 * (trend_strength / strong_trend_threshold))
        # Optional: Clip the w_momentum values to prevent extreme cases
        dataframe['w_momentum'] = dataframe['w_momentum'].clip(lower=self.mem_w3, upper=self.mem_w3 * 2)

        # Step 2: Calculate aggregate score S
        # Calculate aggregate score S
        #w = [self.mem_w0, self.mem_w1, self.mem_w2, self.mem_w3, self.mem_w4, self.mem_w5, self.mem_w6, self.mem_w7, self.mem_w8]
      
        dataframe['S'] = self.mem_w0 * dataframe['normalized_ma'] + \
                         self.mem_w1 * dataframe['normalized_macd'] + \
                         self.mem_w2 * dataframe['normalized_roc'] + \
                         self.mem_w3 * dataframe['normalized_rsi'] + \
                         self.mem_w4 * dataframe['normalized_bb_width'] + \
                         self.mem_w5 * dataframe['normalized_cci'] + \
                         dataframe['w_momentum'] * dataframe['normalized_momentum'] + \
                         self.mem_w8 * dataframe['normalized_stoch'] + \
                         self.mem_w7 * dataframe['normalized_atr'] + \
                         self.mem_w6 * dataframe['normalized_obv']

        # Step 3: Market Regime Filter R

        dataframe['R'] = 0
        dataframe.loc[dataframe['close'] > dataframe['bb_upperband'], 'R'] = 1
        dataframe.loc[dataframe['close'] < dataframe['bb_lowerband'], 'R'] = -1
        buffer_pct = 0.01  # 1% buffer
        
        dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'] * (1 + buffer_pct), 1, 
                                    (np.where(dataframe['close'] < dataframe['ma_100'] * (1 - buffer_pct), -1, 0)))

        # Step 4: Volatility Adjustment V
        # EXPLANATION: Calculate the Bollinger Band width and assign it to V. The Bollinger Band width is the
        # difference between the upper and lower Bollinger Bands divided by the middle Bollinger Band. The idea is
        # that when the Bollinger Bands are wide, the market is volatile, and when the Bollinger Bands are narrow,
        # the market is less volatile. So we are using the Bollinger Band width as a measure of volatility. You can
        # use other indicators to measure volatility as well. For example, you can use the ATR (Average True Range)

        bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['V_mean'] = 1 / (bb_width + 1e-8)  # Avoid division by zero

        # ATR-based Volatility Measure
        dataframe['V2_mean'] = 1 / (dataframe['atr'] + 1e-8)  # Avoid division by zero

        # Rolling window size for adaptive normalization
        rolling_window = 50

        # Normalize V_mean using a rolling window
        mean_v = dataframe['V_mean'].rolling(window=rolling_window).mean()
        std_v = dataframe['V_mean'].rolling(window=rolling_window).std()
        dataframe['V_norm'] = (dataframe['V_mean'] - mean_v) / std_v
        dataframe['V_norm'] = dataframe['V_norm'].fillna(0) 

        # Normalize V2_mean using a rolling window
        mean_v2 = dataframe['V2_mean'].rolling(window=rolling_window).mean()
        std_v2 = dataframe['V2_mean'].rolling(window=rolling_window).std()
        dataframe['V2_norm'] = (dataframe['V2_mean'] - mean_v2) / std_v2
        dataframe['V2_norm'] = dataframe['V2_norm'].fillna(0)

        # Signal assignment using hysteresis
        upper_threshold = 1.0
        lower_threshold = -1.0

        dataframe['V'] = np.where(dataframe['V_norm'] > upper_threshold, 1,
                          np.where(dataframe['V_norm'] < lower_threshold, -1, 0))
        dataframe['V2'] = np.where(dataframe['V2_norm'] > upper_threshold, 1,
                           np.where(dataframe['V2_norm'] < lower_threshold, -1, 0))

        # Forward-fill to maintain the last state of the signal
        dataframe['V'] = dataframe['V'].ffill()  # Correct ffill usage
        dataframe['V2'] = dataframe['V2'].ffill()  # Correct ffill usage

        # Example: Predict 'R'
        #dataframe['predicted_R'] = self.freqai.predict(dataframe['R'], metadata) * self.mem_predicted_R_factor
        #dataframe['predicted_V'] = self.freqai.predict(dataframe['V'], metadata) * self.mem_predicted_V_factor
        #dataframe['predicted_S'] = self.freqai.predict(dataframe['S'], metadata) * self.mem_predicted_S_factor
        # Optional: Predict R2 and V2 and adjust them with hyperoptable parameters
        #dataframe['predicted_R2'] = self.freqai.predict(dataframe['R2'], metadata) * self.mem_predicted_R2_factor.value
        #dataframe['predicted_V2'] = self.freqai.predict(dataframe['V2'], metadata) * self.mem_predicted_V2_factor

        dataframe['predicted_R'] = dataframe['R'] * self.mem_predicted_R_factor
        dataframe['predicted_V'] = dataframe['V'] * self.mem_predicted_V_factor
        dataframe['predicted_S'] = dataframe['S'] * self.mem_predicted_S_factor
        dataframe['predicted_R2'] = dataframe['R2'] * self.mem_predicted_R2_factor
        dataframe['predicted_V2'] = dataframe['V2'] * self.mem_predicted_V2_factor
        
        # Recalculate final target using predicted components
        dataframe['T'] = dataframe['predicted_S'] * dataframe['predicted_R'] * dataframe['predicted_R2'] * dataframe['predicted_V'] * dataframe['predicted_V2']


        # Get Final Target Score to incorporate new calculations
        #dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['R2'] * dataframe['V'] * dataframe['V2']

        # Assign the target score T to the AI target column
        target_horizon = 1  # Define your prediction horizon here
        dataframe['&-target'] = dataframe['T'].shift(-target_horizon)
        

        return dataframe
        

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Calculate percentage change in &-target
        
        df['&-target'] = df['&-target'].fillna(0)
           
        df['&-target_ptc_change'] = df['&-target'].pct_change(fill_method=None) * 100

        # Long Entry Condition
        enter_long_conditions = [
			
            df['&-target_ptc_change'] > self.mem_target_pct_change,  # Significant change in &-target
            df['&-target'].shift(2).notna(),  # Ensure no NaN in shifted values
            df['&-target'].shift(2) > df['&-target'].shift(1),  # Previous two values indicate a drop (potential valley)
            df['&-target'].shift(1) < df['&-target'],  # Now &-target is increasing
            df['&-target'] > 0,  # Ensure positive prediction (bullish)
            df['volume'] > 0  # Only trade if there is volume
        ]

    # Short Entry Condition
        
        enter_short_conditions = [
			
            df['&-target_ptc_change'] < -(self.mem_target_pct_change), # * self.sell_multiplier),  # Significant negative change in &-target
            df['&-target'].shift(2).notna(),  # Ensure no NaN in shifted values
            df['&-target'].shift(2) < df['&-target'].shift(1),  # Previous two values indicate a rise (potential peak)
            df['&-target'].shift(1) > df['&-target'],  # Now &-target is decreasing
            df['&-target'] < 0,  # Ensure negative prediction (bearish)
            df['volume'] > 0  # Only trade if there is volume
        ]

        # Apply Long Entry
        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long")

        # Apply Short Entry
        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Calculate percentage change in &-target
        
        df['&-target'] = df['&-target'].fillna(0)
        
        df['&-target_ptc_change'] = df['&-target'].pct_change(fill_method=None) * 100
    
        # Long Exit Condition (peak detected)
        exit_long_conditions = [
            df['&-target_ptc_change'] < -self.mem_target_pct_change,  # Significant negative change
            df['&-target'].shift(2).notna(),  # Ensure no NaN in shifted values
            df['&-target'].shift(2) < df['&-target'].shift(1),  # Indicates a previous rise
            df['&-target'].shift(1) > df['&-target'],  # Now it's decreasing (peak)
            df['&-target'] < 0,  # Bearish prediction
        ]

        # Short Exit Condition (valley detected)
        exit_short_conditions = [
            df['&-target_ptc_change'] > self.mem_target_pct_change, # * self.sell_multiplier,  # Significant positive change
            df['&-target'].shift(2).notna(),  # Ensure no NaN in shifted values
            df['&-target'].shift(2) > df['&-target'].shift(1),  # Indicates a previous drop
            df['&-target'].shift(1) < df['&-target'],  # Now it's increasing (valley)
            df['&-target'] > 0,  # Bullish prediction
        ]

        # Apply Long Exit
        df.loc[
            reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
        ] = (1, "exit_long")

        # Apply Short Exit
        df.loc[
            reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
        ] = (1, "exit_short")

        return df
    
    #def leverage(self, pair: str, current_time: 'datetime', current_rate: float, proposed_leverage: float, **kwargs) -> float:
    #    return self.leverage_value
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Dynamically adjust leverage based on volatility (ATR).
        """
        # Get the analyzed dataframe for the pair and timeframe
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            # Fallback to default leverage if dataframe is unavailable or empty
            if dataframe is None or (isinstance(dataframe, pd.DataFrame) and dataframe.empty):
                return self.leverage_value

            # Get the latest ATR value from the last candle
            atr = dataframe['atr'].iat[-1]  # Most recent ATR

            # Fallback if ATR is NaN
            if pd.isna(atr):
                return self.leverage_value

            # Calculate the rolling mean and standard deviation of ATR
            rolling_mean = dataframe['atr'].rolling(window=14).mean().iat[-1]
            rolling_stddev = dataframe['atr'].rolling(window=14).std().iat[-1]

            # Debugging prints (can be replaced with logging)
            if (not self.dp.runmode.value in ("backtest", "plot", "hyperopt")):
                print(f"ATR: {atr}, Rolling Mean: {rolling_mean}, Stddev: {rolling_stddev}")

            # Dynamically adjust leverage based on ATR volatility
            if atr > rolling_mean + 1.5 * rolling_stddev:
                reduced_leverage = self.leverage_value * 0.5  # Halve leverage when ATR is high
                if (not self.dp.runmode.value in ("backtest", "plot", "hyperopt")):
                    print(f"Volatility high. Reducing leverage to: {reduced_leverage}")
                return reduced_leverage  # This should be 5.0 if default is 10.0

            # Return the default leverage if volatility is low
            if (not self.dp.runmode.value in ("backtest", "plot", "hyperopt")):
                print(f"Volatility normal. Keeping leverage: {self.leverage_value}")
        except:
            pass
        return self.leverage_value

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Calculate custom stoploss based on dynamic trailing stop logic.
        """
        # Unpack the tuple returned by get_analyzed_dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        row = -1  # Use the last row in the dataframe

        # If no valid dataframe is available (initial trades)
        if dataframe is None or dataframe.empty:
            return self.stoploss

        # Get trailing stop and offset based on ATR
        trailing_stop, trailing_offset = self.trailing_stop_logic(dataframe, row)

        # If the price has increased above the trailing offset, adjust stoploss
        if current_profit > trailing_offset:
            trade.exit_tag = "trailing_stop"
            return -trailing_stop

        trade.exit_tag = "stoploss"
        
        return self.stoploss

    def trailing_stop_logic(self, dataframe: DataFrame, row: int) -> (float, float):
        """
        Calculate the trailing stop and offset dynamically based on ATR.
        """
        # Ensure the ATR column exists before accessing it
        if "atr" not in dataframe.columns:
            raise KeyError("ATR column is missing in the dataframe")

        # Access the ATR value using the row index
        atr = dataframe["atr"].iloc[row]  # Ensure row is a valid integer (-1 for the last row)

        # Normalize ATR between min and max values for trailing stop and offset
        atr_norm = (atr - dataframe["atr"].min()) / (dataframe["atr"].max() - dataframe["atr"].min())

        # Dynamically calculate trailing stop and offset based on ATR
        trailing_stop = self.mem_min_trailing_stop + atr_norm * (
                    self.mem_max_trailing_stop - self.mem_min_trailing_stop)
        trailing_offset = self.mem_min_trailing_offset + atr_norm * (
                    self.mem_max_trailing_offset - self.mem_min_trailing_offset)

        return trailing_stop, trailing_offset


def chaikin_mf(df, periods=20):
    close = df["close"]
    low = df["low"]
    high = df["high"]
    volume = df["volume"]
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name="cmf")

def top_percent_change(dataframe: DataFrame, length: int) -> float:
    if length == 0:
        return (dataframe["open"] - dataframe["close"]) / dataframe["close"]
    else:
        return (dataframe["open"].rolling(length).max() - dataframe["close"]) / dataframe["close"]
