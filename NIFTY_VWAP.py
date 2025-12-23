# BhavCopy.py

import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import pandas_ta as ta
from datetime import date
from nsepy import get_history
import datetime
import numpy as np
import yfinance as yf
import mplfinance as mpf

import mysql.connector
from nsedt import equity as eq
from tradingview_ta import TA_Handler, Interval, Exchange
import tradingview_ta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import streamlit as st
import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import datetime
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
#import Fyers
import warnings
warnings.filterwarnings('ignore')
import time
import os
import datetime
import pandas as pd
import json
import requests
import time
import pyotp
import os
import requests
from urllib.parse import parse_qs,urlparse
import sys
import warnings
warnings.filterwarnings('ignore')
import pandas_ta as ta
from fyers_api import fyersModel, accessToken
from datetime import datetime, timedelta
import time
import os
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  # Import Matplotlib
#import Angel

from nsepy import get_history
from datetime import date
import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from nselib import derivatives
import numpy as np
npNaN = np.nan

import mysql.connector
import pandas as pd
import requests
import numpy as np
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import plotly.graph_objects as go


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

import mysql.connector
import pandas as pd
import requests
import numpy as np
import numpy as np
import pandas as pd
import yfinance as yf
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import plotly.graph_objects as go


def app():
    st.subheader("NIFTY50 Price Prediction By Data Analysis :")
    selected = option_menu(
        menu_title=None,  # required
        options=["VWAP-EMA-Strategy", "EMA-SMA Strategy"],  # required
        icons=["house", "book", "envelope"],  # optional
        menu_icon="cast",  # optional
        default_index=0,  # optional
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#818589"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "blue"},
        },
    )


    if selected == "VWAP-EMA-Strategy":
        from datetime import time
        import pyotp
        import pandas as pd
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        from angel_login import angel_login
        from indicators import add_indicators
        import Angel
        import Angel_DF
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import time
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # --- Auto-refresh using st_autorefresh ---
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh

        # ---- Index Selection ----
        Commodity_Selection = st.sidebar.selectbox(
            "Select Commodity for Trade: ",
            ["CRUDEOIL", "NATURALGAS", "GOLD", "SILVER"],
            index=0,
            key="Commodity_select"
        )

        Candle_Length = 300

        # ---- Index Selection ----
        Position_Selection = st.sidebar.selectbox(
            "Select ATM/OTM Position: ",
            ["ATM", "OTM-50", "OTM-100", "OTM-150", "OTM-200"],
            index=0,
            key="position_select"
        )

        # ---- Timeframe Selection ----
        timeframe = st.sidebar.selectbox(
            "Timeframe",
            ["1m", "3m", "5m", "15m", "30m"],
            index=2,
            key="tf_select"
        )

        # ---- Timeframe Selection ----
        Leg_Loss = st.sidebar.selectbox(
            "Total Leg Max Loss:",
            ["1500", "1250", "1000", "750", "500", "250"],
            index=2,
            key="loss_select"
        )

        # ---- Timeframe Selection ----
        Max_Loss = st.sidebar.selectbox(
            "Individual Leg Max Loss:",
            ["2000", "", "2500", "2800", "3000", "3200"],
            index=2,
            key="Max_select"
        )
        # --- Auto-refresh every 10 seconds ---
        count = st_autorefresh(interval=30_000, key="datarefresh")

        # ---- NIFTY 50 Token (Angel One Fixed Token) ---- 

        expiry = Angel_DF.instruments[(Angel_DF.instruments['name']==Commodity_Selection) &  (Angel_DF.instruments['instrumenttype']=='FUTCOM') &  (Angel_DF.instruments['exch_seg']=='MCX')]["expiry"].unique()
        expiry

        today = datetime.today()

        dates = pd.to_datetime(expiry, format="%d%b%Y")
        next_expiry = dates[dates > today].min()
        st.write(f"{Commodity_Selection} " " Next Expiry:", next_expiry.strftime("%d%b%y").upper())


        Symbol = (f"{Commodity_Selection}{next_expiry.strftime("%d%b%y").upper()}FUT")

        def get_option_token(symbol):
            row = Angel_DF.instruments[Angel_DF.instruments["symbol"] == symbol]
            if row.empty:
                raise Exception(f"Token not found for {symbol}")
            return row.iloc[0]["token"]

        Commodity_token = get_option_token(Symbol)
        #st.write(Commodity_token) 


        NIFTY_TOKEN = Commodity_token     # NIFTY 50 Index
        EXCHANGE = "MCX"

        # ---- Fetch LTP ----
        ltp_data = Angel.smartApi.ltpData(
            exchange=EXCHANGE,
            tradingsymbol="NIFTY",
            symboltoken=NIFTY_TOKEN
        )

        #st.write(ltp_data)

        LTP = float(ltp_data['data']['ltp'])
        prev_close = ltp_data['data']['close']

        ATM = round(LTP / 50) * 50

        # ---- Calculate Change ----
        change = LTP - prev_close
        pct_change = (change / prev_close) * 100

        # ---- Determine arrow ----
        arrow = "🔺" if change > 0 else "🔻" if change < 0 else "⏺️"  # Up / Down / Neutral

        # ---- Display in Streamlit ----
        st.write(f"📈 {Commodity_Selection} LTP: {LTP:.2f} {arrow}")
        st.write(f"📍 {Commodity_Selection} Spot Price (ATM): {ATM}")
        st.write(f"{Commodity_Selection} Change: {change:.2f} {arrow}")
        st.write(f"{Commodity_Selection} % Change: {pct_change:.2f}% {arrow}")





        import pandas_ta as ta
        def add_indicators(df):
            df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
            df["EMA14"] = ta.ema(df["Close"], length=20)
            df["RSI"] = ta.rsi(df["Close"], length=14)

            macd = ta.macd(df["Close"])
            df["MACD"] = macd["MACD_12_26_9"]
            df["MACD_SIGNAL"] = macd["MACDs_12_26_9"]
            return df
        

        # ---------- FETCH 5-MIN CANDLES ---------- #
        to_date = datetime.now()
        from_date = to_date - timedelta(days=5)

        params = {
            "exchange": "MCX",
            "symboltoken": Commodity_token,
            "interval": "THREE_MINUTE",
            "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
            "todate": to_date.strftime("%Y-%m-%d %H:%M")
        }

        data = Angel.smartApi.getCandleData(params)
        NIFTY = pd.DataFrame(
            data["data"],
            columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
        )
        NIFTY["Datetime"] = pd.to_datetime(NIFTY["Datetime"])
        NIFTY['Datetime'] = pd.to_datetime(NIFTY['Datetime']).dt.tz_localize(None)
        NIFTY['NIFTY_Change'] = NIFTY['Close'].diff()
        NIFTY = add_indicators(NIFTY)        
        #st.write(NIFTY)


        NIFTY1 = NIFTY.tail(Candle_Length).copy()
        # Ensure datetime
        NIFTY1["Datetime"] = pd.to_datetime(NIFTY1["Datetime"])
        # Pre-format datetime for hover (candlestick limitation)
        dt_text = NIFTY1["Datetime"].dt.strftime("%Y-%m-%d %H:%M")

        # ---------------- PLOT ----------------
        st.subheader(f"📈 {Commodity_Selection} Candle + EMA + RSI", divider="rainbow")

        fig1 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price + EMA", "RSI")
        )

        # ---------- CANDLESTICK ----------
        fig1.add_trace(
            go.Candlestick(
                x=NIFTY1.index,  # ✅ index-based x-axis

                open=NIFTY1["Open"],
                high=NIFTY1["High"],
                low=NIFTY1["Low"],
                close=NIFTY1["Close"],

                name="Price",

                hoverinfo="text",
                hovertext=[
                    f"<b>Date:</b> {d}<br>"
                    f"<b>Open:</b> {o}<br>"
                    f"<b>High:</b> {h}<br>"
                    f"<b>Low:</b> {l}<br>"
                    f"<b>Close:</b> {c}"
                    for d, o, h, l, c in zip(
                        dt_text,
                        NIFTY1["Open"],
                        NIFTY1["High"],
                        NIFTY1["Low"],
                        NIFTY1["Close"]
                    )
                ]
            ),
            row=1, col=1
        )

        # ---------- EMA14 ----------
        fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["EMA14"],
                name="EMA14",
                customdata=dt_text,
                hovertemplate=
                    "<b>Date:</b> %{customdata}<br>"
                    "<b>EMA14:</b> %{y:.2f}<br>"
                    "<extra></extra>",
                line=dict(color="cyan", width=2)
            ),
            row=1, col=1
        )

        # ---------- RSI ----------
        fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["RSI"],
                name="RSI",
                customdata=dt_text,
                hovertemplate=
                    "<b>Date:</b> %{customdata}<br>"
                    "<b>RSI:</b> %{y:.2f}<br>"
                    "<extra></extra>",
                line=dict(color="orange", width=2)
            ),
            row=2, col=1
        )

        # ---------- RSI LEVELS ----------
        fig1.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
        fig1.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")

        # ---------- LAYOUT ----------
        fig1.update_layout(
            height=750,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",   # 🔥 sync crosshair
            legend=dict(orientation="h", y=1.08)
        )

        st.plotly_chart(fig1, use_container_width=True)
        #st.write(NIFTY)





        import numpy as np
        import pandas as pd

        # df must have: ['Open','High','Low','Close']
        # Optional but recommended: Volume, RSI

        # -----------------------------
        # 1️⃣ LOG RETURN VOLATILITY
        # -----------------------------
        df = NIFTY
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility_20'] = df['log_return'].rolling(20).std() * np.sqrt(252)

        # -----------------------------
        # 2️⃣ TRUE RANGE & ATR
        # -----------------------------
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = (df['High'] - df['Close'].shift()).abs()
        df['tr3'] = (df['Low'] - df['Close'].shift()).abs()

        df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['ATR_14'] = df['TR'].rolling(14).mean()

        # -----------------------------
        # 3️⃣ PARKINSON VOLATILITY (HL)
        # -----------------------------
        df['parkinson_vol'] = (
            (np.log(df['High'] / df['Low']) ** 2)
            .rolling(20)
            .mean()
            / (4 * np.log(2))
        )

        df['parkinson_vol'] = np.sqrt(df['parkinson_vol']) * np.sqrt(252)

        # -----------------------------
        # 4️⃣ GARMAN-KLASS VOLATILITY (OHLC)
        # -----------------------------
        df['gk_vol'] = (
            0.5 * (np.log(df['High'] / df['Low']) ** 2)
            - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
        )

        df['gk_vol'] = np.sqrt(df['gk_vol'].rolling(20).mean()) * np.sqrt(252)

        # -----------------------------
        # 5️⃣ BOLLINGER BAND WIDTH
        # -----------------------------
        df['SMA20'] = df['Close'].rolling(20).mean()
        df['BB_std'] = df['Close'].rolling(20).std()
        df['BB_width'] = (2 * df['BB_std']) / df['SMA20']

        # -----------------------------
        # 6️⃣ RSI VOLATILITY (OPTIONAL)
        # -----------------------------
        if 'RSI' in df.columns:
            df['RSI_vol'] = df['RSI'].rolling(14).std()
        else:
            df['RSI_vol'] = np.nan

        # -----------------------------
        # 7️⃣ VOLATILITY REGIME
        # -----------------------------
        df['vol_regime'] = np.where(
            df['ATR_14'] > df['ATR_14'].rolling(50).mean(),
            'HIGH_VOL',
            'LOW_VOL'
        )

        # -----------------------------
        # 8️⃣ FINAL FEATURE SET
        # -----------------------------
        volatility_features = [
            'volatility_20',
            'ATR_14',
            'parkinson_vol',
            'gk_vol',
            'BB_width',
            'RSI_vol',
            'vol_regime'
        ]

        #st.write(df[volatility_features].tail())

        import numpy as np
        # -----------------------------
        # NORMALIZATION FUNCTION
        # -----------------------------
        def normalize(series, window=100):
            rolling_min = series.rolling(window).min()
            rolling_max = series.rolling(window).max()
            return (series - rolling_min) / (rolling_max - rolling_min)

        # -----------------------------
        # NORMALIZE COMPONENTS
        # -----------------------------
        df['ATR_norm'] = normalize(df['ATR_14'])
        df['Parkinson_norm'] = normalize(df['parkinson_vol'])
        df['BB_norm'] = normalize(df['BB_width'])
        df['ReturnVol_norm'] = normalize(df['volatility_20'])

        # -----------------------------
        # WEIGHTS (Tuned for Index Trading)
        # -----------------------------
        W_ATR = 0.35
        W_PARK = 0.30
        W_BB = 0.20
        W_RET = 0.15

        # -----------------------------
        # VOLATILITY SCORE (0–100)
        # -----------------------------
        df['Volatility_Score'] = (
            W_ATR * df['ATR_norm'] +
            W_PARK * df['Parkinson_norm'] +
            W_BB * df['BB_norm'] +
            W_RET * df['ReturnVol_norm']
        ) * 100

        # -----------------------------
        # CLEANUP
        # -----------------------------
        df['Volatility_Score'] = df['Volatility_Score'].clip(0, 100)

        st.subheader(f"{Commodity_Selection} Market Volatility Index:",divider='rainbow')
        st.write(df[['ATR_14','parkinson_vol','BB_width','volatility_20','Volatility_Score']].tail())





        st.subheader(f"",divider='rainbow')
        st.subheader(f"{Commodity_Selection} " " CE/PE Option Trading:",divider='rainbow')
        expiry = Angel_DF.instruments[(Angel_DF.instruments['name']==Commodity_Selection) &  (Angel_DF.instruments['instrumenttype']=='OPTFUT') &  (Angel_DF.instruments['exch_seg']=='MCX')]["expiry"].unique()
        expiry
        today = datetime.today()
        dates = pd.to_datetime(expiry, format="%d%b%Y")
        next_expiry = dates[dates > today].min()
        st.write(f"{Commodity_Selection} " " FNO Next Expiry:", next_expiry.strftime("%d%b%y").upper())



        Commodity_Symbol_CE = f"{Commodity_Selection}{next_expiry.strftime("%d%b%y").upper()}{ATM+50}CE"
        def get_option_token(symbol):
            row = Angel_DF.instruments[Angel_DF.instruments["symbol"] == symbol]
            if row.empty:
                raise Exception(f"Token not found for {symbol}")
            return row.iloc[0]["token"]
        Commodity_token_CE = get_option_token(Commodity_Symbol_CE)
        st.write(f"{Symbol} " " Token Is:", Commodity_token_CE)


        Commodity_Symbol_PE = f"{Commodity_Selection}{next_expiry.strftime("%d%b%y").upper()}{ATM-50}PE"
        def get_option_token(symbol):
            row = Angel_DF.instruments[Angel_DF.instruments["symbol"] == symbol]
            if row.empty:
                raise Exception(f"Token not found for {symbol}")
            return row.iloc[0]["token"]
        Commodity_token_PE = get_option_token(Commodity_Symbol_PE)
        st.write(f"{Symbol} " " Token Is:", Commodity_token_PE)


        
        def fetch_5min_candles(token):
            to_date = datetime.now()
            from_date = to_date - timedelta(days=3)

            response = Angel.smartApi.getCandleData({
                "exchange": "MCX",
                "symboltoken": token,
                "interval": "THREE_MINUTE",
                "fromdate": from_date.strftime("%Y-%m-%d %H:%M"),
                "todate": to_date.strftime("%Y-%m-%d %H:%M")
            })

            if not response or "data" not in response or response["data"] is None:
                raise Exception("No candle data")

            df = pd.DataFrame(
                response["data"],
                columns=["Datetime", "Open", "High", "Low", "Close", "Volume"]
            )

            # 🔑 CRITICAL FIX
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
            df.sort_index(inplace=True)

            return df



        def apply_indicators(df):
            df = df.copy()

            df["EMA14"] = ta.ema(df["Close"], length=5)
            df["EMA20"] = ta.ema(df["Close"], length=10)
            df["RSI"] = ta.rsi(df["Close"], length=14)

            df["VWAP"] = ta.vwap(
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                volume=df["Volume"]
            )

            df.dropna(inplace=True)
            return df

        ce_df = fetch_5min_candles(Commodity_token_CE)
        pe_df = fetch_5min_candles(Commodity_token_PE)
        ce_df = apply_indicators(ce_df)
        pe_df = apply_indicators(pe_df)





        
        
        
        # ---- Two-column layout ----
        col1, col2 = st.columns(2)

        # ---- Display in columns ----
        with col1:
            NIFTY1 = ce_df.tail(Candle_Length)


            import numpy as np

            # ---------------- INITIALIZE COLUMNS ----------------
            NIFTY1 = NIFTY1.copy()
            NIFTY1["signal"] = 0
            NIFTY1["position"] = 0
            NIFTY1["entry_price"] = np.nan
            NIFTY1["exit_price"] = np.nan
            NIFTY1["exit_reason"] = ""
            NIFTY1["pnl"] = 0.0

            # ---------------- SELL SIGNAL (EMA < VWAP CROSSDOWN) ----------------
            sell_condition = (
                (NIFTY1["EMA14"] < NIFTY1["VWAP"]) & (NIFTY1["RSI"] < 55) &
                (NIFTY1["EMA14"].shift(1) >= NIFTY1["VWAP"].shift(1))
            )

            NIFTY1.loc[sell_condition, "signal"] = -1

            # ---------------- TRADE MANAGEMENT ----------------
            from datetime import time as dt_time
            from datetime import datetime, time
            import time
            import time as time_module


            ENTRY_START = dt_time(9, 45)
            ENTRY_END   = dt_time(13, 0)
            EXIT_TIME   = dt_time(23, 0)

            in_trade = False
            sell_price = None

            for i in range(len(NIFTY1)):

                idx   = NIFTY1.index[i]
                tm    = idx.time()
                price = NIFTY1["EMA14"].iloc[i]
                rsi   = NIFTY1["RSI"].iloc[i]

                # -------- ENTRY (SELL) --------
                if (
                    NIFTY1["signal"].iloc[i] == -1 and
                    not in_trade and
                    ENTRY_START <= tm < ENTRY_END
                ):
                    in_trade = True
                    sell_price = price

                    NIFTY1.at[idx, "position"] = -1
                    NIFTY1.at[idx, "entry_price"] = sell_price

                # -------- EXIT --------
                elif in_trade:

                    # ⏰ 3:00 PM SQUARE-OFF
                    if tm >=EXIT_TIME:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "3:00 PM Auto Square-off"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

                    # ✅ PROFIT BOOKING
                    elif rsi < 25:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "RSI < 25 (Profit)"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

                    # ❌ STOP LOSS
                    elif price > 1.15 * sell_price:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "SL Hit (15%)"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

            NIFTY1.to_csv('CE_Trades.csv')


            NIFTY1["pnl"] = 0.0

            for i in range(len(NIFTY1)):
                if (
                    pd.notna(NIFTY1["exit_price"].iloc[i]) and
                    pd.notna(NIFTY1["entry_price"].iloc[i])
                ):
                    entry = NIFTY1["entry_price"].iloc[i]
                    exitp = NIFTY1["exit_price"].iloc[i]

                    # OPTION SELL → profit when price falls
                    pnl = entry - exitp

                    NIFTY1.at[NIFTY1.index[i], "pnl"] = pnl

            #st.write(NIFTY1)
            # ---------------- PLOTLY CHART ----------------            
            st.subheader(f"📈 {Commodity_Symbol_CE} "  " Candle + RSI", divider="rainbow")

            fig1 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            #subplot_titles=("Price + VWAP + EMA + Signals", "RSI")
            )

            # -------- Candlestick --------
            fig1.add_trace(
            go.Candlestick(
                x=NIFTY1.index,
                open=NIFTY1["Open"],
                high=NIFTY1["High"],
                low=NIFTY1["Low"],
                close=NIFTY1["Close"],
                name="Price"
            ),
            row=1, col=1
            )

            # -------- EMA14 --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["EMA14"],
                mode="lines",
                line=dict(color="cyan", width=2),
                name="EMA14"
            ),
            row=1, col=1
            )

            # -------- VWAP --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["VWAP"],
                mode="lines",
                line=dict(color="orange", dash="dot", width=2),
                name="VWAP"
            ),
            row=1, col=1
            )

            # -------- BUY Signals --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1[NIFTY1["signal"] == 1].index,
                y=NIFTY1[NIFTY1["signal"] == 1]["Close"],
                mode="markers",
                marker=dict(color="green", size=16, symbol="triangle-up"),
                name="BUY"
            ),
            row=1, col=1
            )

            # -------- SELL Signals --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1[NIFTY1["signal"] == -1].index,
                y=NIFTY1[NIFTY1["signal"] == -1]["Close"],
                mode="markers",
                marker=dict(color="red", size=16, symbol="triangle-down"),
                name="SELL"
            ),
            row=1, col=1
            )

            # -------- RSI --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["RSI"],
                mode="lines",
                line=dict(color="orange", width=2),
                name="RSI"
            ),
            row=2, col=1
            )

            # Profit exits
            fig1.add_trace(
                go.Scatter(
                    x=NIFTY1[NIFTY1["exit_reason"].str.contains("Profit", na=False)].index,
                    y=NIFTY1[NIFTY1["exit_reason"].str.contains("Profit", na=False)]["Close"],
                    mode="markers",
                    marker=dict(color="lime", size=14, symbol="circle"),
                    name="BOOK PROFIT"
                ),
                row=1, col=1
            )

            # SL exits
            fig1.add_trace(
                go.Scatter(
                    x=NIFTY1[NIFTY1["exit_reason"].str.contains("SL", na=False)].index,
                    y=NIFTY1[NIFTY1["exit_reason"].str.contains("SL", na=False)]["Close"],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="x"),
                    name="STOP LOSS"
                ),
                row=1, col=1
            )

            fig1.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
            fig1.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")

            # -------- Layout --------
            fig1.update_layout(
            height=750,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08)
            )

            st.plotly_chart(fig1, use_container_width=True)


            st.subheader("📖 Trade Book Details :")

            NIFTY1 = pd.read_csv('CE_Trades.csv')
            NIFTY2 = NIFTY1[(NIFTY1['entry_price'] > 0) | (NIFTY1['exit_price'] > 0)]

            if len(NIFTY2) < 1 :
                st.write('No Trade Initiated Yet...')

            else:
                NIFTY2 = NIFTY2[['Datetime','Close','EMA14','RSI','VWAP','entry_price','exit_price','exit_reason','pnl']]
                st.write(NIFTY2)

                Lots = 1*75
                Trade = NIFTY1[(NIFTY1['pnl'] != 0)]
                Total_Trades = len(Trade)
                Total_Profit_Trades= len(Trade[Trade['pnl'] > 0])
                Win_Rate = (Total_Profit_Trades/Total_Trades)*100
                total_Income = (NIFTY1['pnl'].sum())*Lots

                st.write(f"Total No of Trades : {Total_Trades}")
                st.write(f"Total No of Profit Trade : {Total_Profit_Trades}")
                st.write(f"Win Rate : {round(Win_Rate,2)}%")
                st.write(f"Total Income(Rs/-) : {round(total_Income,2)}")
                st.subheader("📖 Trade Book Trades")

                Trade = Trade[['Datetime','Close','EMA14','RSI','VWAP','entry_price','exit_price','exit_reason','pnl']]
                st.write(Trade)



            # ---- Display in columns ----
        with col2:
            NIFTY1 = pe_df.tail(Candle_Length)


            import numpy as np

            # ---------------- INITIALIZE COLUMNS ----------------
            NIFTY1 = NIFTY1.copy()
            NIFTY1["signal"] = 0
            NIFTY1["position"] = 0
            NIFTY1["entry_price"] = np.nan
            NIFTY1["exit_price"] = np.nan
            NIFTY1["exit_reason"] = ""
            NIFTY1["pnl"] = 0.0

            # ---------------- SELL SIGNAL (EMA < VWAP CROSSDOWN) ----------------
            sell_condition = (
                (NIFTY1["EMA14"] < NIFTY1["VWAP"]) & (NIFTY1["RSI"] < 55) &
                (NIFTY1["EMA14"].shift(1) >= NIFTY1["VWAP"].shift(1))
            )

            NIFTY1.loc[sell_condition, "signal"] = -1

            # ---------------- TRADE MANAGEMENT ----------------
            from datetime import time as dt_time
            from datetime import datetime, time
            import time
            import time as time_module


            ENTRY_START = dt_time(9, 45)
            ENTRY_END   = dt_time(13, 0)
            EXIT_TIME   = dt_time(23, 0)

            in_trade = False
            sell_price = None

            for i in range(len(NIFTY1)):

                idx   = NIFTY1.index[i]
                tm    = idx.time()
                price = NIFTY1["EMA14"].iloc[i]
                rsi   = NIFTY1["RSI"].iloc[i]

                # -------- ENTRY (SELL) --------
                if (
                    NIFTY1["signal"].iloc[i] == -1 and
                    not in_trade and
                    ENTRY_START <= tm < ENTRY_END
                ):
                    in_trade = True
                    sell_price = price

                    NIFTY1.at[idx, "position"] = -1
                    NIFTY1.at[idx, "entry_price"] = sell_price

                # -------- EXIT --------
                elif in_trade:

                    # ⏰ 3:00 PM SQUARE-OFF
                    if tm >=EXIT_TIME:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "3:00 PM Auto Square-off"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

                    # ✅ PROFIT BOOKING
                    elif rsi < 25:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "RSI < 25 (Profit)"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

                    # ❌ STOP LOSS
                    elif price > 1.15 * sell_price:
                        in_trade = False
                        NIFTY1.at[idx, "exit_price"] = price
                        NIFTY1.at[idx, "exit_reason"] = "SL Hit (15%)"
                        NIFTY1.at[idx, "pnl"] = sell_price - price

            NIFTY1.to_csv('PE_Trades.csv')


            NIFTY1["pnl"] = 0.0

            for i in range(len(NIFTY1)):
                if (
                    pd.notna(NIFTY1["exit_price"].iloc[i]) and
                    pd.notna(NIFTY1["entry_price"].iloc[i])
                ):
                    entry = NIFTY1["entry_price"].iloc[i]
                    exitp = NIFTY1["exit_price"].iloc[i]

                    # OPTION SELL → profit when price falls
                    pnl = entry - exitp

                    NIFTY1.at[NIFTY1.index[i], "pnl"] = pnl

            #st.write(NIFTY1)
            # ---------------- PLOTLY CHART ----------------            
            st.subheader(f"📈 {Commodity_Symbol_PE} "  " Candle + RSI", divider="rainbow")

            fig1 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
            #subplot_titles=("Price + VWAP + EMA + Signals", "RSI")
            )

            # -------- Candlestick --------
            fig1.add_trace(
            go.Candlestick(
                x=NIFTY1.index,
                open=NIFTY1["Open"],
                high=NIFTY1["High"],
                low=NIFTY1["Low"],
                close=NIFTY1["Close"],
                name="Price"
            ),
            row=1, col=1
            )

            # -------- EMA14 --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["EMA14"],
                mode="lines",
                line=dict(color="cyan", width=2),
                name="EMA14"
            ),
            row=1, col=1
            )

            # -------- VWAP --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["VWAP"],
                mode="lines",
                line=dict(color="orange", dash="dot", width=2),
                name="VWAP"
            ),
            row=1, col=1
            )

            # -------- BUY Signals --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1[NIFTY1["signal"] == 1].index,
                y=NIFTY1[NIFTY1["signal"] == 1]["Close"],
                mode="markers",
                marker=dict(color="green", size=16, symbol="triangle-up"),
                name="BUY"
            ),
            row=1, col=1
            )

            # -------- SELL Signals --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1[NIFTY1["signal"] == -1].index,
                y=NIFTY1[NIFTY1["signal"] == -1]["Close"],
                mode="markers",
                marker=dict(color="red", size=16, symbol="triangle-down"),
                name="SELL"
            ),
            row=1, col=1
            )

            # -------- RSI --------
            fig1.add_trace(
            go.Scatter(
                x=NIFTY1.index,
                y=NIFTY1["RSI"],
                mode="lines",
                line=dict(color="orange", width=2),
                name="RSI"
            ),
            row=2, col=1
            )

            # Profit exits
            fig1.add_trace(
                go.Scatter(
                    x=NIFTY1[NIFTY1["exit_reason"].str.contains("Profit", na=False)].index,
                    y=NIFTY1[NIFTY1["exit_reason"].str.contains("Profit", na=False)]["Close"],
                    mode="markers",
                    marker=dict(color="lime", size=14, symbol="circle"),
                    name="BOOK PROFIT"
                ),
                row=1, col=1
            )

            # SL exits
            fig1.add_trace(
                go.Scatter(
                    x=NIFTY1[NIFTY1["exit_reason"].str.contains("SL", na=False)].index,
                    y=NIFTY1[NIFTY1["exit_reason"].str.contains("SL", na=False)]["Close"],
                    mode="markers",
                    marker=dict(color="red", size=14, symbol="x"),
                    name="STOP LOSS"
                ),
                row=1, col=1
            )

            fig1.add_hline(y=70, row=2, col=1, line_dash="dash", line_color="red")
            fig1.add_hline(y=30, row=2, col=1, line_dash="dash", line_color="green")

            # -------- Layout --------
            fig1.update_layout(
            height=750,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.08)
            )

            st.plotly_chart(fig1, use_container_width=True)


            st.subheader("📖 Trade Book Details :")
            NIFTY1 = pd.read_csv('PE_Trades.csv')
            NIFTY2 = NIFTY1[(NIFTY1['entry_price'] > 0) | (NIFTY1['exit_price'] > 0)]
            
            if len(NIFTY2) < 1 :
                st.write('No Trade Initiated Yet...')

            else:
                NIFTY2 = NIFTY2[['Datetime','Close','EMA14','RSI','VWAP','entry_price','exit_price','exit_reason','pnl']]
                st.write(NIFTY2)

                Lots = 1*75
                Trade = NIFTY1[(NIFTY1['pnl'] != 0)]
                Total_Trades = len(Trade)
                Total_Profit_Trades= len(Trade[Trade['pnl'] > 0])
                Win_Rate = (Total_Profit_Trades/Total_Trades)*100
                total_Income = (NIFTY1['pnl'].sum())*Lots

                st.write(f"Total No of Trades : {Total_Trades}")
                st.write(f"Total No of Profit Trade : {Total_Profit_Trades}")
                st.write(f"Win Rate : {round(Win_Rate,2)}%")
                st.write(f"Total Income(Rs/-) : {round(total_Income,2)}")
                st.subheader("📖 Trade Book Trades")

                Trade = Trade[['Datetime','Close','EMA14','RSI','VWAP','entry_price','exit_price','exit_reason','pnl']]
                st.write(Trade)






        
        