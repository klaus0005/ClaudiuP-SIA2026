"""
Sistem de Suport Decizional (DSS) pentru achiziția de uree
Disertație — model BiLSTM pentru predicția prețului ureei
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime

plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25,
    'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold'
})

# ==========================================================================
# CONFIGURARE PAGINA
# ==========================================================================
st.set_page_config(page_title="DSS Preț Uree", page_icon="🌾", layout="wide",
                   initial_sidebar_state="collapsed")

# CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1100px;}
    h1 {color: #1a3a5c; font-weight: 700;}
    h2 {color: #2c5f8a; border-bottom: 2px solid #e8eef4; padding-bottom: 0.3rem; margin-top: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    .stAlert {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# ==========================================================================
# DEFINIREA MODELULUI (trebuie identica cu cea din notebook)
# ==========================================================================
class UreaBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==========================================================================
# INCARCARE MODEL + DATE (cache ca sa nu reincarce la fiecare interactiune)
# ==========================================================================
@st.cache_resource
def load_model_and_config():
    config = joblib.load('models/config.pkl')
    scaler = joblib.load('models/scaler_final.pkl')
    model = UreaBiLSTM(config['n_features'], config['hidden_size'])
    model.load_state_dict(torch.load('models/bilstm6_urea_final.pt'))
    model.eval()
    return model, scaler, config

@st.cache_data(ttl=3600)  # reîncarcă maxim o dată pe oră
def load_data(_config):
    features = _config['features']
    pink_cols = [c for c in features if c != 'EXUSEU']

    # --- Pink Sheet (local, se publică lunar) ---
    # Fisierul World Bank foloseste nume descriptive in antet (randul 4), nu coduri scurte
    rename_map = {
        'Crude oil, average': 'CRUDE_PETRO',
        'Natural gas, US': 'NGAS_US',
        'Natural gas, Europe': 'NGAS_EUR',
        'Liquefied natural gas, Japan': 'NGAS_JP',
        'Maize': 'MAIZE',
        'Wheat, US SRW': 'WHEAT_US_SRW',
        'Urea': 'UREA_EE_BULK',
        'DAP': 'DAP',
        'Potassium chloride **': 'POTASH',
    }
    df_pink = pd.read_excel('data/CMO-Historical-Data-Monthly.xlsx',
                            sheet_name='Monthly Prices', skiprows=4)
    df_pink = df_pink.rename(columns={df_pink.columns[0]: 'Date'})
    df_pink.columns = [str(c).strip() for c in df_pink.columns]
    df_pink = df_pink.rename(columns=rename_map)
    # Aruncam randul cu unitatile si pastram doar lunile valide 'YYYYMmm'
    df_pink = df_pink[df_pink['Date'].astype(str).str.match(r'^\d{4}M\d{2}$')].copy()
    df_pink['Date'] = pd.to_datetime(df_pink['Date'], format='%YM%m')
    for col in pink_cols:
        df_pink[col] = pd.to_numeric(df_pink[col], errors='coerce')
    df_pink = df_pink[['Date'] + pink_cols]

    # --- EXUSEU: încercăm FRED live, cu fallback la fișierul local ---
    sursa_curs = "live (FRED)"
    try:
        start = datetime.datetime(1999, 1, 1)
        end = datetime.datetime.now()
        df_fx = web.DataReader('EXUSEU', 'fred', start, end).reset_index()
        df_fx.columns = ['Date', 'EXUSEU']
        df_fx = df_fx.dropna()
    except Exception:
        # Dacă nu e internet / FRED pică -> fișierul local
        sursa_curs = "local (fallback)"
        df_fx = pd.read_csv('data/EXUSEU.csv')
        df_fx['observation_date'] = pd.to_datetime(df_fx['observation_date'])
        df_fx = df_fx.rename(columns={'observation_date': 'Date'})

    df = pd.merge(df_pink, df_fx, on='Date', how='inner')
    df = df.sort_values('Date').set_index('Date').ffill()
    df = df[features]
    return df, sursa_curs

model, scaler, config = load_model_and_config()
df_prices, sursa_curs = load_data(config)
st.caption(f"📡 Sursă curs valutar: {sursa_curs} | Ultima lună: {df_prices.index[-1].strftime('%Y-%m')}")
st.divider()
LOOK_BACK = config['look_back']
features = config['features']

# ==========================================================================
# FUNCTIE DE PREDICTIE (predictie T+1 pe baza ultimelor luni)
# ==========================================================================
def predict_next(df_prices, model, scaler, look_back):
    # Calculam log-returns pe tot setul
    df_logret = np.log(df_prices / df_prices.shift(1)).dropna()

    # Luam ultimele `look_back` luni ca input
    last_window = df_logret.iloc[-look_back:].values
    last_window_scaled = scaler.transform(last_window)

    X = torch.tensor(last_window_scaled[np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(X).numpy().flatten()[0]

    # De-scalam (doar coloana ureei, index 0)
    dummy = np.zeros((1, scaler.n_features_in_))
    dummy[0, 0] = pred_scaled
    pred_logret = scaler.inverse_transform(dummy)[0, 0]

    # Reconstruim pretul: pret_urmator = pret_curent * exp(log_return)
    current_price = df_prices['UREA_EE_BULK'].iloc[-1]
    predicted_price = current_price * np.exp(pred_logret)
    pct_change = (np.exp(pred_logret) - 1) * 100

    return current_price, predicted_price, pct_change

# ==========================================================================
# INTERFATA
# ==========================================================================
st.title("🌾 Sistem de Suport Decizional — Achiziție Uree")
st.markdown("Predicția prețului ureei pe luna următoare și recomandare de achiziție, "
            "bazată pe un model BiLSTM cu variabile exogene (energie, curs valutar).")

current_price, predicted_price, pct_change = predict_next(df_prices, model, scaler, LOOK_BACK)
last_date = df_prices.index[-1]

# --- SECTIUNEA 1: PREDICTIE T+1 ---
st.header("📊 Predicție lună următoare")
col1, col2, col3 = st.columns(3)
col1.metric("Preț curent", f"{current_price:.2f} $/t",
            help=f"Ultima lună disponibilă: {last_date.strftime('%Y-%m')}")
col2.metric("Preț prezis (T+1)", f"{predicted_price:.2f} $/t",
            delta=f"{pct_change:+.2f}%")
col3.metric("Variație prezisă", f"{pct_change:+.2f}%")

# --- SECTIUNEA 2: RECOMANDARE ---
st.header("🎯 Recomandare")
PRAG = 5.0  # pragul de +/- 5%

if pct_change > PRAG:
    st.error(f"### 🔴 CUMPĂRĂ ACUM\n"
             f"Modelul prezice o creștere de **{pct_change:+.2f}%**. "
             f"Achiziția acum evită scumpirea anticipată.")
    recomandare = "CUMPĂRĂ"
elif pct_change < -PRAG:
    st.success(f"### 🟢 AȘTEAPTĂ\n"
               f"Modelul prezice o scădere de **{pct_change:+.2f}%**. "
               f"Amânarea achiziției poate reduce costul.")
    recomandare = "AȘTEAPTĂ"
else:
    st.warning(f"### 🟡 MONITORIZEAZĂ\n"
               f"Variația prezisă (**{pct_change:+.2f}%**) este sub pragul de ±{PRAG}%. "
               f"Nu există un semnal clar; se recomandă monitorizarea pieței.")
    recomandare = "MONITORIZEAZĂ"

# --- SECTIUNEA 3: GRAFIC ISTORIC ---
st.header("📈 Evoluție istorică")
n_months = st.slider("Câte luni de istoric să afișez?", 12, len(df_prices), 60)

fig, ax = plt.subplots(figsize=(11, 4))
hist = df_prices['UREA_EE_BULK'].iloc[-n_months:]
ax.plot(hist.index, hist.values, color='#2c5f8a', linewidth=1.5, label='Preț istoric')
# Punctul de predictie
next_date = last_date + pd.DateOffset(months=1)
ax.scatter([next_date], [predicted_price], color='#d1495b', s=80, zorder=5,
           label=f'Predicție ({predicted_price:.0f} $/t)')
ax.plot([last_date, next_date], [current_price, predicted_price],
        color='#d1495b', linestyle='--', linewidth=1.5)
ax.set_xlabel('Data')
ax.set_ylabel('Preț ($/tonă)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# --- SECTIUNEA 4: SIMULARE IMPACT FINANCIAR ---
st.header("💰 Simulare impact financiar")
st.markdown("Estimează impactul deciziei pentru o cantitate dată de uree.")

cantitate = st.number_input("Cantitate de achiziționat (tone)", min_value=1, value=100, step=10)

cost_acum = cantitate * current_price
cost_prezis = cantitate * predicted_price
diferenta = cost_prezis - cost_acum

col1, col2, col3 = st.columns(3)
col1.metric("Cost dacă cumperi ACUM", f"{cost_acum:,.0f} $")
col2.metric("Cost estimat luna viitoare", f"{cost_prezis:,.0f} $")
col3.metric("Diferență", f"{diferenta:+,.0f} $",
            delta=f"{pct_change:+.2f}%", delta_color="inverse")

if diferenta > 0:
    st.info(f"Cumpărând acum, economisești estimativ **{abs(diferenta):,.0f} $** "
            f"față de achiziția de luna viitoare.")
elif diferenta < 0:
    st.info(f"Așteptând, poți economisi estimativ **{abs(diferenta):,.0f} $** "
            f"față de achiziția de acum.")

# --- SECTIUNEA 5: BACKTESTING ---
st.header("🔬 Backtesting — performanța istorică a strategiei")
st.markdown("Simulează: dacă ai fi urmat recomandările modelului în ultimele luni, "
            "cum ar fi evoluat deciziile vs. o achiziție lunară constantă?")

@st.cache_data
def run_backtest(_df_prices, _config, n_test_months=24):
    df_logret = np.log(_df_prices / _df_prices.shift(1)).dropna()
    prices = _df_prices['UREA_EE_BULK'].values
    look_back = _config['look_back']

    correct_calls = 0
    total_calls = 0
    results = []

    start = len(df_logret) - n_test_months
    for i in range(start, len(df_logret)):
        window = df_logret.iloc[i-look_back:i].values
        window_scaled = scaler.transform(window)
        X = torch.tensor(window_scaled[np.newaxis, :, :], dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = model(X).numpy().flatten()[0]
        dummy = np.zeros((1, scaler.n_features_in_))
        dummy[0, 0] = pred_scaled
        pred_logret = scaler.inverse_transform(dummy)[0, 0]

        actual_logret = df_logret['UREA_EE_BULK'].iloc[i]

        # Directie prezisa vs reala
        pred_dir = np.sign(pred_logret)
        actual_dir = np.sign(actual_logret)
        if pred_dir == actual_dir:
            correct_calls += 1
        total_calls += 1

        results.append({
            'data': df_logret.index[i],
            'prezis_%': (np.exp(pred_logret) - 1) * 100,
            'real_%': (np.exp(actual_logret) - 1) * 100,
            'directie_corecta': pred_dir == actual_dir
        })

    accuracy = correct_calls / total_calls * 100
    return pd.DataFrame(results), accuracy

bt_df, bt_accuracy = run_backtest(df_prices, config)

col1, col2 = st.columns(2)
col1.metric("Acuratețe direcțională (backtest)", f"{bt_accuracy:.1f}%",
            help="Procentul de luni în care modelul a prezis corect direcția (sus/jos)")
col2.metric("Luni testate", f"{len(bt_df)}")

st.markdown("**Detaliu predicții vs. realitate:**")
bt_display = bt_df.copy()
bt_display['data'] = bt_display['data'].dt.strftime('%Y-%m')
bt_display['prezis_%'] = bt_display['prezis_%'].round(2)
bt_display['real_%'] = bt_display['real_%'].round(2)
bt_display['directie_corecta'] = bt_display['directie_corecta'].map({True: '✅', False: '❌'})
bt_display.columns = ['Lună', 'Prezis (%)', 'Real (%)', 'Direcție corectă']
st.dataframe(bt_display, width='stretch', hide_index=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Model: BiLSTM (fereastră 6 luni, 10 variabile exogene). "
           "Notă: predicțiile au caracter orientativ și nu constituie consultanță financiară.")
