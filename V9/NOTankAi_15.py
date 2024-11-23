import logging
import warnings
import gc

from functools import reduce
import datetime
import talib.abstract as ta
import pandas_ta as pta
import numpy as np
import pandas as pd
#from murrey_math import calculate_murrey_math_levels
import freqtrade.vendor.qtpylib.indicators as qtpylib
from technical import qtpylib
from datetime import timedelta, datetime, timezone
from pandas import DataFrame, Series
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from technical.pivots_points import pivots_points
from freqtrade.exchange import timeframe_to_prev_date, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute, informative)
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

from scipy.signal import argrelextrema
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore
from scipy.stats import linregress
import math

logger = logging.getLogger(__name__)

class NOTankAi_15(IStrategy):
    '''
    _________   __  .__    ________                                       
    \______  \_/  |_|  |__ \______ \____________     ____   ____   ____   
        /    /\   __\  |  \ |    |  \_  __ \__  \   / ___\ /  _ \ /    \  
       /    /  |  | |   Y  \|    `   \  | \// __ \_/ /_/  >  <_> )   |  \ 
      /____/   |__| |___|  /_______  /__|  (____  /\___  / \____/|___|  / 
                         \/        \/           \//_____/             \/  
                         
    * no hyper needed
    * dropped RL version < v1.2.2
    * freqai config for regressor:
        // PyTorchLSTMRegressorMultiTarget - needs cuda compiled torch package
            "gamma": 0.9,
            "verbose": 1,
            "learning_rate": 0.003,
            "device": "cuda", // or cpu only
            "output_dim": 3,  // 1 is single target, you need to set your strat number of targets
            "trainer_kwargs": {
                "n_epochs": 100,
                "n_steps": null,
                "batch_size": 64,
                "n_jobs": 2,
          },
            "model_kwargs": {
                "num_lstm_layers": 3,
                "hidden_dim": 128,
                "dropout_percent": 0.4,
                "window_size": 5,
                }
    * WIP
    '''

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v2.0.4.6"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.30, -0.01, decimals=2, name='stoploss')]
            
    exit_profit_only = False ### No selling at a loss
    use_custom_stoploss = True
    trailing_stop = False
    position_adjustment_enable = True
    ignore_roi_if_entry_signal = True
    max_entry_position_adjustment = 4
    max_open_trades = 20 # split balance in 20
    #max_dca_multiplier = 2.5
    process_only_new_candles = True
    can_short = True
    use_exit_signal = True
    startup_candle_count: int = 200
    stoploss = -0.1
    timeframe = '5m'
    trail_enable = True
    stoploss_protection = True         # for more than 5min since trade open
    trail_delta = 0.05
    trail_delta_natr = True
    trail_start = 0.03 # lower than stop
    trail_stop = abs(stoploss) # higher than start, better to be low or equan than abs(stoploss), 999 if unlimited

    minimal_roi = {
    #    '0'   : 1.00,
    #    '300' : 0.50,
    #    '600' : 0.45,
    #    '1200': 0.40,
    #    '2400': 0.30,
    #    '3600': 0.20,
    #    '4800': 0.10,
    #    '7200': 0.05,
    #    '9600': 0.01,
    }
    
    plot_config = {
        "main_plot": {},
        "subplots": {
        "extrema": {
          "&s-minima_sort_threshold": {
            "color": "#4ae747",
            "type": "line"
          },
          "&s-maxima_sort_threshold": {
            "color": "#5b5e4b",
            "type": "line"
          },
          "extrema_norm": {
            "color": "#1b3dd1"
          }
        },
        "min_max": {
          "maxima": {
            "color": "#a29db9",
            "type": "line"
          },
          "minima": {
            "color": "#ac7fc",
            "type": "line"
          },
          "maxima_check": {
            "color": "#a29db9",
            "type": "line"
          },
          "minima_check": {
            "color": "#ac7fc",
            "type": "line"
          }
        },
        "predict": {
          "do_predict": {
            "color": "#fbff00",
            "type": "line"
          },
          "DI_catch": {
            "color": "#4924bb"
          }
        },
        "extrema_predicted": {
          "&s-extrema": {
            "color": "#75aca3"
          },
          "extrema": {
            "color": "#89b04d",
            "type": "line"
          }
        }
        }
        }

    '''
    # protections
    cooldown_lookback = IntParameter(24, 48, default=12, space="protection", optimize=False, load=True)
    stop_duration = IntParameter(12, 200, default=5, space="protection", optimize=False, load=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=False, load=True)

    ### protections ###
    @property
    def protections(self):
        prot = []

        prot.append({
            "method": "CooldownPeriod",
            "stop_duration_candles": self.cooldown_lookback.value
        })
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 24 * 3,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False
            })

        return prot
    '''

    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> Optional[float]:

        n = self.freqai_info["feature_parameters"]["label_period_candles"]
        
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)        
        current_candle = df.iloc[-1].squeeze()
        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        signal_time = entry_time - timedelta(minutes=int(timeframe_minutes))
        signal_candle = df.loc[df['date'] == signal_time]
        signal_candle = signal_candle.iloc[-1].squeeze()
        #filled_entries = trade.select_filled_orders(trade.entry_side)
        #count_of_entries = trade.nr_of_successful_entries
        #trade_duration = (current_time - trade.open_date_utc).seconds / 60
        

        # Extremas
        if trade.is_short:
            need_partial_exit = (
                #(df["do_predict"] == 1)
                #(df["'%%-DI_catch"].rolling(window=1).max().shift(-1) >= 0)
                #  (df["&-s_minima"].shift(2) == 0)  # partial minimum                
                (df["extrema"].rolling(window=2).min().shift(-1) < 0)      # full minimum 1
                #& (df["natr"].shift(2) < df["natr"].shift(1))
                & (df['volume'] > 0)  # Make sure Volume is not 0
            )
        else:
            need_partial_exit = (
                #(df["do_predict"] == 1)
                #(df["'%%-DI_catch"].rolling(window=1).min().shift(-1) <= 0)
                # (df["&-s_maxima"].shift(2) == 0)  # partial minimum                
                (df["extrema"].rolling(window=2).max().shift(-1) > 0)      # full minimum 1
                #& (df["natr"].shift(2) > df["natr"].shift(1))
                & (df['volume'] > 0)  # Make sure Volume is not 0
            )
        need_partial_exit_last = need_partial_exit.iloc[-1]
        def valid_exits(a: int, x: int, c: int) -> list:
            # a number of exits 3
            # x number of repeated exits 2 times
            # c nr_of_successful_exits from 0 a-1
            return [c + a * b for b in range(x + 1)]
        
        if not signal_candle.empty and 'natr' in df.columns:
            last_natr = signal_candle['natr']
        else:
            last_natr = 1.0

        # Normalize NATR to a range from atr_min value to n (for example)
        n = 4 # n>1
        atr_min = abs(self.stoploss)/n
        atr_max = abs(self.stoploss)*n*n*last_natr  # Set atr_max adjustement based on RR>3/1

        normalized_atr = (last_natr - atr_min) / (atr_max - atr_min) * n
        normalized_atr = max(0, min(normalized_atr, n))  # Ensure it's within range [0, n]

        # Adjust position size based on current_profit and normalized ATR
        position_adjustment = normalized_atr

        if current_profit > abs(position_adjustment*0) and (trade.nr_of_successful_exits in valid_exits(n,2,0)) and need_partial_exit_last:
        #if current_profit > 0 and (trade.nr_of_successful_exits in valid_exits(3,2,0)) and need_partial_exit_last:
            # Take half of the profit at +25%
            return -(trade.stake_amount / (n+1))
        if current_profit > abs(position_adjustment*1) and (trade.nr_of_successful_exits in valid_exits(n,2,1)) and need_partial_exit_last:
        #if current_profit > 0 and (trade.nr_of_successful_exits in valid_exits(3,2,1)) and need_partial_exit_last:
            # Take half of the profit at +25%
            return -(trade.stake_amount / (n+0))
        if current_profit > abs(position_adjustment*2) and (trade.nr_of_successful_exits in valid_exits(n,2,2)) and need_partial_exit_last:
        #if current_profit > 0 and (trade.nr_of_successful_exits in valid_exits(3,2,2)) and need_partial_exit_last:
            # Take half of the profit at +25%
            return -(trade.stake_amount / (n-1))
        if current_profit > abs(position_adjustment*3) and (trade.nr_of_successful_exits in valid_exits(n,2,3)) and need_partial_exit_last:
        #if current_profit > 0 and (trade.nr_of_successful_exits in valid_exits(3,2,2)) and need_partial_exit_last:
            # Take half of the profit at +25%
            return -(trade.stake_amount / (n-2))
        # if more conditions increase n, adjust n+/-x values accordingly
            
        return None


    # profit assured on market change
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, length: int = 14, multiplier: float = 1.5, **kwargs) -> float:

        if after_fill:
            return stoploss_from_open(self.stoploss, current_profit, is_short=trade.is_short, leverage=trade.leverage)
                        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()
        
        entry_time = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        timeframe_minutes = timeframe_to_minutes(self.timeframe)
        signal_time = entry_time - timedelta(minutes=int(timeframe_minutes))
        signal_candle = dataframe.loc[dataframe['date'] == signal_time]
        #signal_candle = dataframe.iloc[(dataframe['date'] - signal_time).abs().argsort()[:1]]
        if not signal_candle.empty:
            signal_candle = signal_candle.iloc[-1].squeeze()  # Only squeeze if it's not empty
        else:
            signal_candle = None  # Handle the case where there's no matching candle
        trade_duration = (current_time - trade.open_date_utc).seconds / 60


        if self.trail_delta_natr:
            trail_delta = current_candle['natr']/trade.leverage
        else:    
            trail_delta = self.trail_delta
            
        #self.trail_delta = current_profit-*current_candle['natr']
        if self.trail_enable:
            logger.info(f"Trail profit {pair} is {current_profit:.3f}, [from:{self.trail_start:.3f}, to:{(self.trail_stop + trail_delta):.3f}, delta:-{trail_delta:.3f}]")
            if current_profit > self.trail_start and current_profit < (self.trail_stop+self.trail_delta):
                if trade.is_short:
                    logger.info(f"Short trail {pair} @ {stoploss_from_absolute(current_rate+current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(current_rate+current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)
                else:
                    logger.info(f"Long trail {pair} @ {stoploss_from_absolute(current_rate-current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(current_rate-current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)
        if (self.stoploss_protection and trade_duration > 5) and (not self.trail_delta_natr) and (current_profit > (self.trail_stop+trail_delta)):
                if trade.is_short:
                    logger.info(f"Short protection1 {pair} | {trail_delta} @ {stoploss_from_absolute(trade.open_rate-trade.open_rate*abs(self.trail_stop)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(trade.open_rate-trade.open_rate*abs(self.trail_stop)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)
                else:
                    logger.info(f"Long protection1 {pair} | {trail_delta} @ {stoploss_from_absolute(trade.open_rate+trade.open_rate*abs(self.trail_stop)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(trade.open_rate+trade.open_rate*abs(self.trail_stop)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)
        if (self.stoploss_protection and trade_duration > 5) and self.trail_delta_natr:
                if trade.is_short:
                    logger.info(f"Short protection2 {pair} | natr:{trail_delta} | sl:{stoploss_from_absolute(current_rate+current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(current_rate+current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)
                else:
                    logger.info(f"Long protection2 {pair} | natr:{trail_delta} | sl:{stoploss_from_absolute(current_rate-current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)}")
                    return stoploss_from_absolute(current_rate-current_rate*abs(trail_delta)/trade.leverage, current_rate, is_short=trade.is_short, leverage=trade.leverage)

        
        return None

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-rocr-period"] = ta.ROCR(dataframe, timeperiod=period)
        dataframe["%-cmf-period"] = chaikin_mf(dataframe, periods=period)
        dataframe["%-chop-period"] = qtpylib.chopiness(dataframe, period)
        dataframe["%-linear-period"] = ta.LINEARREG_ANGLE(dataframe["close"], timeperiod=period)
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-atr-periodp"] = dataframe["%-atr-period"] / dataframe["close"] * 1000
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
        dataframe["%-raw_volume"] = dataframe["volume"]
        
        dataframe["%-obv"] = ta.OBV(dataframe)
        dataframe["%-dpo"] = pta.dpo(dataframe['close'], length=40, centered=False)

        # Williams R%
        dataframe['%-willr14'] = pta.willr(dataframe['high'], dataframe['low'], dataframe['close'])

        dataframe['ema_2'] = ta.EMA(dataframe, timeperiod=2)

        # VWAP
        vwap_low, vwap, vwap_high = VWAPB(dataframe, 20, 1)
        dataframe['%-vwap_upperband'] = vwap_high
        dataframe['%-vwap_middleband'] = vwap
        dataframe['%-vwap_lowerband'] = vwap_low
        dataframe['%-vwap_width'] = ((dataframe['%-vwap_upperband'] - dataframe['%-vwap_lowerband']) / dataframe['%-vwap_middleband']) * 100

        dataframe['%-dist_to_vwap_upperband'] = get_distance(dataframe['close'], dataframe['%-vwap_upperband'])
        dataframe['%-dist_to_vwap_middleband'] = get_distance(dataframe['close'], dataframe['%-vwap_middleband'])
        dataframe['%-dist_to_vwap_lowerband'] = get_distance(dataframe['close'], dataframe['%-vwap_lowerband'])
        dataframe['%-tail'] = (dataframe['close'] - dataframe['low']).abs()
        dataframe['%-wick'] = (dataframe['high'] - dataframe['close']).abs()

        #### heikinashi

        heikinashi = qtpylib.heikinashi(dataframe)

        dataframe['%-ha_open'] = heikinashi['open']
        dataframe['%-ha_close'] = heikinashi['close']
        dataframe['%-ha_high'] = heikinashi['high']
        dataframe['%-ha_low'] = heikinashi['low']
        dataframe['%-ha_closedelta'] = (heikinashi['close'] - heikinashi['close'].shift())
        dataframe['%-ha_tail'] = (heikinashi['close'] - heikinashi['low'])
        dataframe['%-ha_wick'] = (heikinashi['high'] - heikinashi['close'])
        dataframe['%-ha_HLC3'] = (heikinashi['high'] + heikinashi['low'] + heikinashi['close'])/3
        
        #####

        ''' 
        dataframe["%-+3/8"] = dataframe["[+3/8]P"]
        dataframe["%-+2/8"] = dataframe["[+2/8]P"]
        dataframe["%-+1/8"] = dataframe["[+1/8]P"]
        dataframe["%-8/8"] = dataframe["[8/8]P"]
        dataframe["%-7/8"] = dataframe["[7/8]P"]
        dataframe["%-6/8"] = dataframe["[6/8]P"]
        dataframe["%-5/8"] = dataframe["[5/8]P"]
        dataframe["%-4/8"] = dataframe["[4/8]P"]
        dataframe["%-3/8"] = dataframe["[3/8]P"]
        dataframe["%-2/8"] = dataframe["[2/8]P"]
        dataframe["%-1/8"] = dataframe["[1/8]P"]
        dataframe["%-0/8"] = dataframe["[0/8]P"]
        dataframe["%--1/8"] = dataframe["[-1/8]P"]
        dataframe["%--2/8"] = dataframe["[-2/8]P"]
        dataframe["%--3/8"] = dataframe["[-3/8]P"]

        dataframe['ema_2'] = ta.EMA(dataframe['close'], timeperiod=2)
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+3/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+2/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[+1/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[8/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[4/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[0/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-1/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-2/8]P'])
        dataframe['%-distema2'] = get_distance(dataframe['ema_2'], dataframe['[-3/8]P'])

        dataframe['%-entrythreshold4'] = (dataframe['%-tail'] - dataframe['[0/8]P'])
        dataframe["%-entrythreshold5"] = (dataframe['%-tail'] - dataframe['[-1/8]P'])
        dataframe["%-entrythreshold6"] = (dataframe['%-tail'] - dataframe['[-2/8]P'])
        dataframe["%-entrythreshold7"] = (dataframe['%-tail'] - dataframe['[-3/8]P'])

        dataframe["%-exitthreshold4"] = (dataframe['%-wick'] - dataframe['[8/8]P'])
        dataframe["%-exitthreshold5"] = (dataframe['%-wick'] - dataframe['[+1/8]P'])
        dataframe["%-exitthreshold6"] = (dataframe['%-wick'] - dataframe['[+2/8]P'])
        dataframe["%-exitthreshold7"] = (dataframe['%-wick'] - dataframe['[+3/8]P'])
        '''
        murrey_math_levels = calculate_murrey_math_levels(dataframe)
        for level, value in murrey_math_levels.items():
            dataframe[level] = value

        #dataframe['mmlextreme_oscillator'] = 100 * ((dataframe['close'] - dataframe["[-3/8]P"]) / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]))
        #dataframe['%-mmlextreme_oscillator'] = dataframe['mmlextreme_oscillator']
        dataframe['%-mmlextreme_oscillator'] = 100 * ((dataframe['close'] - dataframe["[-3/8]P"]) / (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]))
    

        # Calculate the percentage change between the high and open prices for each 5-minute candle
        #dataframe['%-perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        # Create a custom indicator that checks if any of the past 100 5-minute candles' high price is 3% or more above the open price
        #dataframe['%-candle_1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 1, 1, 0).sum()).shift()
        #dataframe['%-candle_2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 2, 1, 0).sum()).shift()
        #dataframe['%-candle_3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 3, 1, 0).sum()).shift()
        #dataframe['%-candle_5perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x >= 5, 1, 0).sum()).shift()

        #dataframe['%-candle_-1perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -1, -1, 0).sum()).shift()
        #dataframe['%-candle_-2perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -2, -1, 0).sum()).shift()
        #dataframe['%-candle_-3perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -3, -1, 0).sum()).shift()
        #dataframe['%-candle_-5perc_50'] = dataframe['%-perc_change'].rolling(50).apply(lambda x: np.where(x <= -5, -1, 0).sum()).shift()

        # Calculate the percentage of the current candle's range where the close price is
        #dataframe['%-close_percentage'] = (dataframe['close'] - dataframe['low']) / (dataframe['high'] - dataframe['low'])

        #dataframe['%-body_size'] = abs(dataframe['open'] - dataframe['close'])
        #dataframe['%-range_size'] = dataframe['high'] - dataframe['low']
        #dataframe['%-body_range_ratio'] = dataframe['%-body_size'] / dataframe['%-range_size']

        #dataframe['%-upper_wick_size'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        #dataframe['%-upper_wick_range_ratio'] = dataframe['%-upper_wick_size'] / dataframe['%-range_size']
        
        #lookback_period = 10
        dataframe['%-max_high'] = dataframe['high'].rolling(50).max()
        dataframe['%-min_low'] = dataframe['low'].rolling(50).min()
        dataframe['%-close_position'] = (dataframe['close'] - dataframe['%-min_low']) / (dataframe['%-max_high'] - dataframe['%-min_low'])
        dataframe['%-current_candle_perc_change'] = (dataframe['high'] / dataframe['open'] - 1) * 100

        dataframe['%-hi'] = ta.SMA(dataframe['high'], timeperiod = 28)

        dataframe['%-lo'] = ta.SMA(dataframe['low'], timeperiod = 28)

        dataframe['%-ema1'] = ta.EMA(dataframe['%-ha_HLC3'], timeperiod = 28)
        dataframe['%-ema2'] = ta.EMA(dataframe['%-ema1'], timeperiod = 28)

        # 200 SMA and distance
        dataframe['%-200sma'] = ta.SMA(dataframe, timeperiod = 200)
        dataframe['%-200sma_dist'] = get_distance(dataframe["%-ha_close"], dataframe['%-200sma'])
        '''
        ########### Now, the wave_columns DataFrame can be used directly
        new_columns = new_columns.loc[:, ~new_columns.columns.duplicated()]
        new_columns = new_columns[~new_columns.index.duplicated(keep='first')]
        new_columns = new_columns.reindex(new_columns.index)

        dataframe = pd.concat([new_columns,dataframe], axis=1)
        ###########
        '''
        dataframe['d'] = dataframe['%-ema1'] - dataframe['%-ema2']
        dataframe['mi'] = dataframe['%-ema1'] + dataframe['d']
        dataframe['%-md'] = np.where(dataframe['mi'] > dataframe['%-hi'], 
            dataframe['mi'] - dataframe['%-hi'], 
            np.where(dataframe['mi'] < dataframe['%-lo'], 
            dataframe['mi'] - dataframe['%-lo'], 0))
        
        dataframe['%-sb'] = ta.SMA(dataframe['%-md'], timeperiod = 8)
        dataframe['%-sh'] = dataframe['%-md'] - dataframe['%-sb']
        '''
        newframe = dataframe.copy()
        # WaveTrend using OHLC4 or HA close - 3/21
        newframe['ap'] = (0.333 * (newframe['%-ha_high'] + newframe['%-ha_low'] + newframe["%-ha_close"]))
        
        newframe['esa'] = ta.EMA(newframe['ap'], timeperiod = 9)
        newframe['d'] = ta.EMA(abs(newframe['ap'] - newframe['esa']), timeperiod = 9)
        newframe['%-wave_ci'] = (newframe['ap']-newframe['esa']) / (0.015 * newframe['d'])
        newframe['%-wave_t1'] = ta.EMA(newframe['%-wave_ci'], timeperiod = 12)
        newframe['%-wave_t2'] = ta.SMA(newframe['%-wave_t1'], timeperiod = 4)
        
        dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]
        newframe = newframe.loc[:, ~newframe.columns.duplicated()]
        dataframe = pd.concat([newframe,dataframe], axis=1)
        del newframe
        gc.collect()
        '''


        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]
    
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        return dataframe


    def set_freqai_targets(self, dataframe, **kwargs):
        #
        n = self.freqai_info["feature_parameters"]["label_period_candles"]

        # Check if 'extrema_norm' exists in the dataframe
        if 'extrema_norm' not in dataframe.columns:
            # Initialize 'extrema_norm' with zeros if it does not exist
            dataframe['extrema_norm'] = 0

        #dataframe['extrema_norm'] = dataframe['extrema_norm'].replace(0, pd.NA).ffill().fillna(0)
        # Use the filled 'extrema_norm' column for further calculations
        dataframe['&s-extrema'] = dataframe['extrema_norm'].shift(-n)
        
        # predict the expected range
        #dataframe['&-s_max'] = dataframe["close"].shift(-n).rolling(n).max()/dataframe["close"] - 1
        #dataframe['&-s_min'] = dataframe["close"].shift(-n).rolling(n).min()/dataframe["close"] - 1
        
        return dataframe


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        if '&s-extrema' in dataframe.columns:
            dataframe['&s-extrema'] = dataframe['&s-extrema'].replace(0, pd.NA).ffill().fillna(0)

        n = self.freqai_info["feature_parameters"]["label_period_candles"]
        lookback = 1
        lookback_window = n

        dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], length=14)
    
    
        dataframe["extrema"] = 0
        min_peaks = argrelextrema(
            dataframe["low"].values, np.less,
            order=n
        )
        max_peaks = argrelextrema(
            dataframe["high"].values, np.greater,
            order=n
        )
        #for mp in min_peaks[0]:
        #    dataframe.at[mp, "extrema"] = -1
        #for mp in max_peaks[0]:
        #    dataframe.at[mp, "extrema"] = 1
            
        dataframe.loc[min_peaks[0], "extrema"] = -1
        dataframe.loc[max_peaks[0], "extrema"] = 1
        
        dataframe["minima"] = np.where(dataframe["extrema"] == -1, 1, 0)
        dataframe["maxima"] = np.where(dataframe["extrema"] == 1, 1, 0)
        
        #dataframe.loc[min_peaks[0], "extrema"] = -1
        #dataframe.loc[max_peaks[0], "extrema"] = 1
        #dataframe['extrema_norm'] = dataframe['extrema'].rolling(window=lookback_window, win_type='gaussian', center=True, min_periods=1).mean(std=0.5)
        dataframe['extrema_norm'] = dataframe['extrema'].rolling(window=lookback_window, win_type='gaussian', center=True).mean(std=0.5)
        #print(dataframe['extrema_norm'])


        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1,
        )
        
        # load default values 
        dataframe["minima_sort_threshold"] = dataframe["s-minima_sort_threshold"]
        dataframe["maxima_sort_threshold"] = dataframe["s-maxima_sort_threshold"]

        dataframe['min_threshold_mean'] = dataframe["minima_sort_threshold"].expanding().mean()
        dataframe['max_threshold_mean'] = dataframe["maxima_sort_threshold"].expanding().mean()

        dataframe['maxima_check'] = dataframe['maxima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)
        dataframe['minima_check'] = dataframe['minima'].rolling(12).apply(lambda x: int((x != 1).all()), raw=True).fillna(0)

        pair = metadata['pair']
        if dataframe['maxima'].iloc[-3] == 1 and dataframe['maxima_check'].iloc[-1] == 0:
            self.dp.send_msg(f'*** {pair} *** Maxima Detected - Potential Short!!!'
                )

        if dataframe['minima'].iloc[-3] == 1 and dataframe['minima_check'].iloc[-1] == 0:
            self.dp.send_msg(f'*** {pair} *** Minima Detected - Potential Long!!!'
                )
        gc.collect()
        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        n = self.freqai_info["feature_parameters"]["label_period_candles"] # this must be read ~20 for 5m   
        lookback = 2
        lookback_window = n//4
        '''
        if df['&s-extrema'].notnull().tail(lookback_window).count() == lookback_window:
            x = np.arange(lookback_window)  # Array from 0 to lookback_window-1
            y = df['&s-extrema'].tail(lookback_window).to_numpy(dtype=float)  # Convert to a float array
            
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(x, y)
        '''
            
        df.loc[
            (
                (df["do_predict"] >= 0) &  # Guard: tema is raising
                #(df["DI_catch"] == 1) &
                #(df["maxima_check"] == 1) &
                (df["&s-extrema"] > df["&s-extrema"].shift(1)) &
                #(df["&s-extrema"].shift(1) > df["&s-extrema"].shift(2)) &
                (df["&s-extrema"].iloc[-n:] != 0).all() &
                #(slope > 0) &
                #(df["natr"] > 0.5) &
                #(df["minima"].iloc[-1:] == 1).any() &
                (df['minima'].rolling(window=lookback).max() == 1) &
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['enter_long', 'enter_tag']] = (1, 'Minima')

        df.loc[
            (
                (df["do_predict"] >= 0) & # Guard: tema is raising
                #(df["DI_catch"] == 1) &
                #(df["minima_check"] == 1) &
                (df["&s-extrema"] < df["&s-extrema"].shift(1)) &
                #(df["&s-extrema"].shift(1) < df["&s-extrema"].shift(2)) &
                (df["&s-extrema"].iloc[-n:] != 0).all() &
                #(slope < 0) &
                #(df["natr"] > 0.5) &
                #(df["maxima"].iloc[-1:] == 1).any() &
                (df['maxima'].rolling(window=lookback).max() == 1) &
                (df['volume'] > 0)  # Make sure Volume is not 0

            ),
            ['enter_short', 'enter_tag']] = (1, 'Maxima')

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:

        n = self.freqai_info["feature_parameters"]["label_period_candles"] # this must be read ~20 for 5m       
        lookback = 1
        lookback_window = n//4
        '''
        if df['&s-extrema'].notnull().tail(lookback_window).count() == lookback_window:
            x = np.arange(lookback_window)  # Array from 0 to lookback_window-1
            y = df['&s-extrema'].tail(lookback_window).to_numpy(dtype=float)  # Convert to a float array
            
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(x, y)
        '''
        df.loc[
            (
                (df["do_predict"] >= 0) &  # Guard: tema is raising
                #(df["DI_catch"] == 1) &
                (df["&s-extrema"] < df["&s-extrema"].shift(1)) &
                (df["&s-extrema"].iloc[-n:] != 0).all() &
                #(df["&s-extrema"].shift(1) != 0) &
                #(slope < 0) &
                (df['maxima'].rolling(window=lookback_window).max() == 1) &
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_long', 'exit_tag']] = (1, 'Maxima')
        df.loc[
            (
                (df["do_predict"] >= 0) & # Guard: tema is raising
                #(df["DI_catch"] == 1) &
                (df["&s-extrema"] > df["&s-extrema"].shift(1)) &
                (df["&s-extrema"].iloc[-n:] != 0).all() &
                #(df["&s-extrema"].shift(1) != 0) &
                #(slope > 0) &
                (df['minima'].rolling(window=lookback_window).max() == 1) &
                (df['volume'] > 0)   # Make sure Volume is not 0

            ),
            ['exit_short', 'exit_tag']] = (1, 'Minima')

        return df

   # Leverage
    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:
        """
        Customize leverage for each new trade.
    
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on this pair
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        :return: A leverage amount, which is between 1.0 and max_leverage.
        """
        window_size = 50
        # Obtain historical candle data for the given pair and timeframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        # Extract required historical data for indicators
        historical_close_prices = dataframe['close'].tail(window_size)
        historical_high_prices = dataframe['high'].tail(window_size)
        historical_low_prices = dataframe['low'].tail(window_size)

        # Set base leverage
        base_leverage = 10

        # Calculate RSI and ATR based on historical data using TA-Lib
        rsi_values = ta.RSI(historical_close_prices, timeperiod=14)  # Adjust the time period as needed
        atr_values = ta.ATR(historical_high_prices, historical_low_prices, historical_close_prices,
                            timeperiod=14)  # Adjust the time period as needed

        # Calculate MACD and SMA based on historical data
        macd_line, signal_line, _ = ta.MACD(historical_close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        sma_values = ta.SMA(historical_close_prices, timeperiod=20)

        # Get the current RSI and ATR values from the last data point in the historical window
        current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50.0  # Default value if no data available
        current_atr = atr_values[-1] if len(atr_values) > 0 else 0.0  # Default value if no data available

        # Get the current MACD and SMA values
        current_macd = macd_line[-1] - signal_line[-1] if len(macd_line) > 0 and len(signal_line) > 0 else 0.0
        current_sma = sma_values[-1] if len(sma_values) > 0 else 0.0

        # Define dynamic thresholds for RSI and ATR for leverage adjustments
        # Set default values or use non-NaN values if available
        dynamic_rsi_low = np.nanmin(rsi_values) if len(rsi_values) > 0 and not np.isnan(np.nanmin(rsi_values)) else 30.0
        dynamic_rsi_high = np.nanmax(rsi_values) if len(rsi_values) > 0 and not np.isnan(
            np.nanmax(rsi_values)) else 70.0
        dynamic_atr_low = np.nanmin(atr_values) if len(atr_values) > 0 and not np.isnan(
            np.nanmin(atr_values)) else 0.002
        dynamic_atr_high = np.nanmax(atr_values) if len(atr_values) > 0 and not np.isnan(
            np.nanmax(atr_values)) else 0.005

        # Print variables for debugging
        print("Historical Close Prices:", historical_close_prices)
        print("RSI Values:", rsi_values)
        print("ATR Values:", atr_values)
        print("Current RSI:", current_rsi)
        print("Current ATR:", current_atr)
        print("Current MACD:", current_macd)
        print("Current SMA:", current_sma)
        print("Dynamic RSI Low:", dynamic_rsi_low)
        print("Dynamic RSI High:", dynamic_rsi_high)
        print("Dynamic ATR Low:", dynamic_atr_low)
        print("Dynamic ATR High:", dynamic_atr_high)

        # Leverage adjustment factors
        long_increase_factor = 1.5
        long_decrease_factor = 0.5
        short_increase_factor = 1.5
        short_decrease_factor = 0.5
        volatility_decrease_factor = 0.8
        # Adjust leverage for long trades
        if side == 'long':
            # Adjust leverage for short trades based on dynamic thresholds and current RSI
            if current_rsi < dynamic_rsi_low:
                base_leverage *= long_increase_factor
            elif current_rsi > dynamic_rsi_high:
                base_leverage *= long_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            # Adjust leverage based on MACD and SMA
            if current_macd > 0:
                base_leverage *= long_increase_factor
            if current_rate < current_sma:
                base_leverage *= long_decrease_factor

        # Adjust leverage for short trades
        elif side == 'short':
            # Adjust leverage for short trades based on dynamic thresholds and current RSI
            if current_rsi > dynamic_rsi_high:
                base_leverage *= short_increase_factor
            elif current_rsi < dynamic_rsi_low:
                base_leverage *= short_decrease_factor

            if current_atr > (current_rate * 0.03):
                base_leverage *= volatility_decrease_factor

            # Adjust leverage based on MACD and SMA
            if current_macd < 0:
                base_leverage *= short_increase_factor  # Increase leverage for potential downward movement
            if current_rate > current_sma:
                base_leverage *= short_decrease_factor  # Decrease leverage if price is above the moving average

        else:
            return proposed_leverage  # Return the proposed leverage if side is neither 'long' nor 'short'

        # Apply maximum and minimum limits to the adjusted leverage
        adjusted_leverage = max(min(base_leverage, max_leverage), 1.0)  # Apply max and min limits

        # Print variables for debugging
        print("Proposed Leverage:", proposed_leverage)
        print("Adjusted Leverage:", adjusted_leverage)

        return adjusted_leverage  # Return the adjusted leverage


def chaikin_mf(df, periods=20):
    close = df['close']
    low = df['low']
    high = df['high']
    volume = df['volume']
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0)
    mfv *= volume
    cmf = mfv.rolling(periods).sum() / volume.rolling(periods).sum()
    return Series(cmf, name='cmf')

def VWAPB(dataframe, window_size=20, num_of_std=1):
    df = dataframe.copy()
    df['vwap'] = qtpylib.rolling_vwap(df, window=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def get_distance(p1, p2):
    return abs((p1) - (p2))

def calculate_murrey_math_levels(df, window_size=64):

    #df = df.iloc[-window_size:]
    
    # Calculate rolling 64-bar maximum and minimum values
    rolling_max_H = df['high'].rolling(window=window_size).max()
    rolling_min_L = df['low'].rolling(window=window_size).min()

    max_H = rolling_max_H
    min_L = rolling_min_L
    range_HL = max_H - min_L

    def calculate_fractal(v2):
        fractal = 0
        if 25000 < v2 <= 250000:
            fractal = 100000
        elif 2500 < v2 <= 25000:
            fractal = 10000
        elif 250 < v2 <= 2500:
            fractal = 1000
        elif 25 < v2 <= 250:
            fractal = 100
        elif 12.5 < v2 <= 25:
            fractal = 12.5
        elif 6.25 < v2 <= 12.5:
            fractal = 12.5
        elif 3.125 < v2 <= 6.25:
            fractal = 6.25
        elif 1.5625 < v2 <= 3.125:
            fractal = 3.125
        elif 0.390625 < v2 <= 1.5625:
            fractal = 1.5625
        elif 0 < v2 <= 0.390625:
            fractal = 0.1953125
        return fractal

    def calculate_octave(v1, v2, mn, mx):
        range_ = v2 - v1
        sum_ = np.floor(np.log(calculate_fractal(v1) / range_) / np.log(2))
        octave = calculate_fractal(v1) * (0.5 ** sum_)
        mn = np.floor(v1 / octave) * octave
        if mn + octave > v2:
            mx = mn + octave
        else:
            mx = mn + (2 * octave)
        return mx

    def calculate_x_values(v1, v2, mn, mx):
        dmml = (v2 - v1) / 8
        x_values = []

        # Calculate the midpoints of each segment
        midpoints = [mn + i * dmml for i in range(8)]

        for i in range(7):
            x_i = (midpoints[i] + midpoints[i + 1]) / 2
            x_values.append(x_i)

        finalH = max(x_values)  # Maximum of the x_values is the finalH

        return x_values, finalH

    def calculate_y_values(x_values, mn):
        y_values = []

        for x in x_values:
            if x > 0:
                y = mn
            else:
                y = 0
            y_values.append(y)

        return y_values

    def calculate_mml(mn, finalH, mx):
        dmml = ((finalH - finalL) / 8) * 1.0699
        mml = (float([mx][0]) * 0.99875) + (dmml * 3) 
        # mml = (float([mx]) * 0.99875) + (dmml * 3) 

        ml = []
        for i in range(0, 16):
            calc = mml - (dmml * (i))
            ml.append(calc)

        murrey_math_levels = {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0]
        }


        return mml, murrey_math_levels

    for i in range(len(df)):
        mn = np.min(min_L.iloc[:i + 1])
        mx = np.max(max_H.iloc[:i + 1])
        x_values, finalH = calculate_x_values(mn, mx, mn, mx)
        y_values = calculate_y_values(x_values, mn)
        finalL = np.min(y_values)
        mml, murrey_math_levels = calculate_mml(finalL, finalH, mx)

        # Add Murrey Math levels to the DataFrame at each time step
        for level, value in murrey_math_levels.items():
            df.at[df.index[i], level] = value

    return df


def PC(dataframe, in1, in2):
    df = dataframe.copy()
    pc = ((in2-in1)/in1) * 100
    return pc
