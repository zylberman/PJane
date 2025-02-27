# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union
#------------------------------------
import random
#------------------------------------
from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib
import utils.app as news_analyzer


# This class is a sample. Feel free to customize it.
class PJStrategy(IStrategy):
    """
    This is a sample strategy to inspire you.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.10

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = "5m"

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "tema": {},
            "sar": {"color": "white"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []
    # ------------------------------------

    def get_trend_for_block(self, first_date, first_value, metadata) -> str:
        """
        Calcula la tendencia de un bloque de 16 horas usando el primer valor del bloque.

        :param first_date: Fecha del primer valor del bloque
        :param first_value: Primer valor del bloque (puede ser precio, RSI, volumen, etc.)
        :return: 'positive', 'negative' o 'neutral' (valor de tendencia)
        """


        # Convertir first_date a string
        first_date_str = first_date.strftime('%Y-%m-%d %H:%M:%S')

        news_analyzer.decir_hola()

        # Obtener la divisa automáticamente desde `metadata`
        currency_pair = metadata.get("pair", "BTC/USDT")  # Si no está definido, usa BTC/USDT por defecto

        trend = news_analyzer.obtener_sentimiento(currency_pair, first_date_str)

        print(f"📌 Fecha del primer dato del bloque: {first_date}, Valor: {first_value}")
        print(f"La tendecia determinada por el analizador es : {trend}")

        trend_options = ['positive', 'negative', 'neutral']
        tendencia = random.choice(trend_options)  # Aquí puedes cambiar la lógica si necesitas algo más específico

        return tendencia

    # ------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame.

        :param dataframe: Dataframe with data from the exchange.
        :param metadata: Additional information, like the currently traded pair.
        :return: A DataFrame with all mandatory indicators for the strategies.
        """

        print("🔍 DataFrame antes de agregar indicadores:")
        print(dataframe.head(3))  # Muestra las primeras 5 filas para ver el formato de la fecha

        # Asegurar que la columna 'date' esté en formato datetime
        if 'date' in dataframe.columns:
            dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')

        # **Hacer que el tamaño del bloque sea configurable**
        trend_block_hours = 16  # ⬅️ Tamaño del bloque en horas
        timeframe_minutes = timeframe_to_minutes(self.timeframe)  
        rows_per_block = (trend_block_hours * 60) // timeframe_minutes  # Conversión a filas

        # Crear la columna 'trend' con valores vacíos (NaN)
        dataframe['trend'] = None

        # Nombre del archivo de caché basado en el par de trading
        cache_filename = f"trend_cache_{metadata['pair'].replace('/', '_')}.csv"

        # Si el archivo de caché existe, cargar 'trend' desde allí
        if os.path.exists(cache_filename):
            print(f"🔄 Cargando datos de tendencia desde {cache_filename}...")
            trend_cache = pd.read_csv(cache_filename, parse_dates=["date"])  # Cargar cache
            dataframe = dataframe.merge(trend_cache, on="date", how="left")  # Combinar con el DataFrame principal

        else:
            print("⚡ Calculando tendencias desde cero...")

            # Asegurar que la columna 'date' está en formato datetime
            if "date" in dataframe.columns:
                dataframe["date"] = pd.to_datetime(dataframe["date"], errors="coerce")

            # Crear la columna 'trend' vacía
            dataframe["trend"] = None

            # Asignar tendencias en bloques configurables
            total_filas = len(dataframe)
            for start in range(0, total_filas, rows_per_block):
                end = min(start + rows_per_block, total_filas)

                first_date = dataframe.iloc[start]["date"]
                first_value = dataframe.iloc[start]["close"]

                tendencia = self.get_trend_for_block(first_date, first_value, metadata)
                dataframe.loc[start:end - 1, "trend"] = tendencia

            # Guardar los datos calculados en un archivo CSV
            dataframe[["date", "trend"]].to_csv(cache_filename, index=False)
            print(f"✅ Datos de tendencia guardados en {cache_filename} para futuras ejecuciones.")
    
        '''' 
        for start in range(0, total_filas, rows_per_16h):
            end = min(start + rows_per_16h, total_filas)  # Evita pasarnos del límite

            # ✅ Obtener la fecha y el primer valor del bloque
            first_date = dataframe.iloc[start]['date']
            first_value = dataframe.iloc[start]['close']  # O 'rsi' si prefieres

            # ✅ Calcular la tendencia basada en el primer valor del bloque y su fecha
            tendencia = self.get_trend_for_block(first_date, first_value, metadata)

            # ✅ Asignar la tendencia al bloque
            dataframe.loc[start:end-1, 'trend'] = tendencia

        # ✅ Mostrar los primeros valores de cada bloque
        for start in range(0, total_filas, rows_per_16h):
            end = min(start + rows_per_16h, total_filas)
            print(dataframe.iloc[start:start+2][['date', 'trend']])  # Mostrar las primeras 3 filas de cada bloque
        '''
        
        # ------------------------------------
        # Momentum Indicators
        # ------------------------------------

        # ADX
        dataframe["adx"] = ta.ADX(dataframe)

        # # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)

        # # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # # Aroon, Aroon Oscillator
        # aroon = ta.AROON(dataframe)
        # dataframe['aroonup'] = aroon['aroonup']
        # dataframe['aroondown'] = aroon['aroondown']
        # dataframe['aroonosc'] = ta.AROONOSC(dataframe)

        # # Awesome Oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)

        # # Keltner Channel
        # keltner = qtpylib.keltner_channel(dataframe)
        # dataframe["kc_upperband"] = keltner["upper"]
        # dataframe["kc_lowerband"] = keltner["lower"]
        # dataframe["kc_middleband"] = keltner["mid"]
        # dataframe["kc_percent"] = (
        #     (dataframe["close"] - dataframe["kc_lowerband"]) /
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"])
        # )
        # dataframe["kc_width"] = (
        #     (dataframe["kc_upperband"] - dataframe["kc_lowerband"]) / dataframe["kc_middleband"]
        # )

        # # Ultimate Oscillator
        # dataframe['uo'] = ta.ULTOSC(dataframe)

        # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
        # dataframe['cci'] = ta.CCI(dataframe)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

        # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

        # # Stochastic Slow
        # stoch = ta.STOCH(dataframe)
        # dataframe['slowd'] = stoch['slowd']
        # dataframe['slowk'] = stoch['slowk']

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # # Stochastic RSI
        # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
        # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
        # stoch_rsi = ta.STOCHRSI(dataframe)
        # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # MFI
        dataframe["mfi"] = ta.MFI(dataframe)

        # # ROC
        # dataframe['roc'] = ta.ROC(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (dataframe["bb_upperband"] - dataframe["bb_lowerband"]) / dataframe[
            "bb_middleband"
        ]

        # Bollinger Bands - Weighted (EMA based instead of SMA)
        # weighted_bollinger = qtpylib.weighted_bollinger_bands(
        #     qtpylib.typical_price(dataframe), window=20, stds=2
        # )
        # dataframe["wbb_upperband"] = weighted_bollinger["upper"]
        # dataframe["wbb_lowerband"] = weighted_bollinger["lower"]
        # dataframe["wbb_middleband"] = weighted_bollinger["mid"]
        # dataframe["wbb_percent"] = (
        #     (dataframe["close"] - dataframe["wbb_lowerband"]) /
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"])
        # )
        # dataframe["wbb_width"] = (
        #     (dataframe["wbb_upperband"] - dataframe["wbb_lowerband"]) /
        #     dataframe["wbb_middleband"]
        # )

        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SMA - Simple Moving Average
        # dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
        # dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
        # dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
        # dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
        # dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        # dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

        # Parabolic SAR
        dataframe["sar"] = ta.SAR(dataframe)

        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # Cycle Indicator
        # ------------------------------------
        # Hilbert Transform Indicator - SineWave
        hilbert = ta.HT_SINE(dataframe)
        dataframe["htsine"] = hilbert["sine"]
        dataframe["htleadsine"] = hilbert["leadsine"]

        # Pattern Recognition - Bullish candlestick patterns
        # ------------------------------------
        # # Hammer: values [0, 100]
        # dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        # # Inverted Hammer: values [0, 100]
        # dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)
        # # Dragonfly Doji: values [0, 100]
        # dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)
        # # Piercing Line: values [0, 100]
        # dataframe['CDLPIERCING'] = ta.CDLPIERCING(dataframe) # values [0, 100]
        # # Morningstar: values [0, 100]
        # dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe) # values [0, 100]
        # # Three White Soldiers: values [0, 100]
        # dataframe['CDL3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(dataframe) # values [0, 100]

        # Pattern Recognition - Bearish candlestick patterns
        # ------------------------------------
        # # Hanging Man: values [0, 100]
        # dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)
        # # Shooting Star: values [0, 100]
        # dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        # # Gravestone Doji: values [0, 100]
        # dataframe['CDLGRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(dataframe)
        # # Dark Cloud Cover: values [0, 100]
        # dataframe['CDLDARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(dataframe)
        # # Evening Doji Star: values [0, 100]
        # dataframe['CDLEVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(dataframe)
        # # Evening Star: values [0, 100]
        # dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)

        # Pattern Recognition - Bullish/Bearish candlestick patterns
        # ------------------------------------
        # # Three Line Strike: values [0, -100, 100]
        # dataframe['CDL3LINESTRIKE'] = ta.CDL3LINESTRIKE(dataframe)
        # # Spinning Top: values [0, -100, 100]
        # dataframe['CDLSPINNINGTOP'] = ta.CDLSPINNINGTOP(dataframe) # values [0, -100, 100]
        # # Engulfing: values [0, -100, 100]
        # dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe) # values [0, -100, 100]
        # # Harami: values [0, -100, 100]
        # dataframe['CDLHARAMI'] = ta.CDLHARAMI(dataframe) # values [0, -100, 100]
        # # Three Outside Up/Down: values [0, -100, 100]
        # dataframe['CDL3OUTSIDE'] = ta.CDL3OUTSIDE(dataframe) # values [0, -100, 100]
        # # Three Inside Up/Down: values [0, -100, 100]
        # dataframe['CDL3INSIDE'] = ta.CDL3INSIDE(dataframe) # values [0, -100, 100]

        # # Chart type
        # # ------------------------------------
        # # Heikin Ashi Strategy
        # heikinashi = qtpylib.heikinashi(dataframe)
        # dataframe['ha_open'] = heikinashi['open']
        # dataframe['ha_close'] = heikinashi['close']
        # dataframe['ha_high'] = heikinashi['high']
        # dataframe['ha_low'] = heikinashi['low']

        # Retrieve best bid and best ask from the orderbook
        # ------------------------------------
        """
        # first check if dataprovider is available
        if self.dp:
            if self.dp.runmode.value in ('live', 'dry_run'):
                ob = self.dp.orderbook(metadata['pair'], 1)
                dataframe['best_bid'] = ob['bids'][0][0]
                dataframe['best_ask'] = ob['asks'][0][0]
        """
        print("🔍 DataFrame después de agregar indicadores:\n", dataframe.iloc[190:200][['date', 'sar', 'tema', 'trend']])
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe["rsi"], self.buy_rsi.value))
                & (dataframe["tema"] <= dataframe["bb_middleband"])  # Guard: tema below BB middle
                & (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
                & (dataframe["trend"] == 'positive')  # Guard: tendencia positiva
            ),
            "enter_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], self.short_rsi.value))
                & (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
                & (dataframe["trend"] == 'negative')  # Guard: tendencia negativa
            ),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[
            (
                # Signal: RSI crosses above 70
                (qtpylib.crossed_above(dataframe["rsi"], self.sell_rsi.value))
                & (dataframe["tema"] > dataframe["bb_middleband"])  # Guard: tema above BB middle
                & (dataframe["tema"] < dataframe["tema"].shift(1))  # Guard: tema is falling
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        dataframe.loc[
            (
                # Signal: RSI crosses above 30
                (qtpylib.crossed_above(dataframe["rsi"], self.exit_short_rsi.value))
                &
                # Guard: tema below BB middle
                (dataframe["tema"] <= dataframe["bb_middleband"])
                & (dataframe["tema"] > dataframe["tema"].shift(1))  # Guard: tema is raising
                & (dataframe["volume"] > 0)  # Make sure Volume is not 0
            ),
            "exit_short",
        ] = 1

        # **NUEVA CONDICIÓN**: Cierre de operaciones si cambia la tendencia y hay profit
        if "trend" in dataframe.columns:
            dataframe["trend_shift"] = dataframe["trend"] != dataframe["trend"].shift(1)  # Detectar cambios en tendencia

            # Verificar si la estrategia tiene trades activos
            if self.dp and self.dp.runmode.value in ("live", "dry_run"):  
                active_trades = self.dp.get_trades(metadata["pair"])  # Obtener operaciones activas para el par

                for trade in active_trades:
                    open_price = trade.open_rate  # Precio al que se abrió la operación
                    current_price = dataframe["close"]  # Precio actual de la vela
                    profit = (current_price - open_price) / open_price  # Cálculo del profit en porcentaje

                    # Cierre en largo si la tendencia cambia a negativa y hay profit
                    dataframe.loc[
                        (
                            (dataframe["trend_shift"])  # Cambio en tendencia
                            & (dataframe["trend"] == "negative")  # Nueva tendencia negativa
                            & (profit > 0)  # Solo cerrar si hay ganancia
                        ),
                        "exit_long",
                    ] = 1

                    # Cierre en corto si la tendencia cambia a positiva y hay profit
                    dataframe.loc[
                        (
                            (dataframe["trend_shift"])  # Cambio en tendencia
                            & (dataframe["trend"] == "positive")  # Nueva tendencia positiva
                            & (profit < 0)  # Solo cerrar si hay ganancia en short (precio cae)
                        ),
                        "exit_short",
                    ] = 1

        return dataframe
