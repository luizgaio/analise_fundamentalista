# -------------------------------------------------------------
# Step 1 — Seleção de Ativos (.SA) para o Dashboard de Valuation
# Autor: Luiz E. Gaio + ChatGPT
# Descrição: Componente inicial do app Streamlit para escolher
#            tickers da B3, validar, normalizar e salvar no estado.
# -------------------------------------------------------------

import re
import pandas as pd
import streamlit as st
from typing import List

st.set_page_config(page_title="Step 1 – Seleção de Ativos", layout="wide")
st.title("📌 Step 1 — Seleção de Ativos (.SA)")
st.caption("Informe os tickers da B3. Aceita: PETR4, VALE3, ITUB4, BOVA11, já com ou sem o sufixo .SA.")

# -----------------------------
# 🔧 Utilitários
# -----------------------------
TICKER_PATTERN = re.compile(r"^[A-Z]{3,5}\d{1,2}(?:\.SA)?$")

def normalize_ticker(t: str) -> str:
    """Normaliza o ticker: maiúsculas, adiciona .SA se faltar e aparentar ser B3.
    Mantém .SA quando já presente.
    """
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.endswith(".SA"):
        return t
    # heurística: se parece com ticker B3 (letras+digitos), adiciona .SA
    if re.match(r"^[A-Z]{3,5}\d{1,2}$", t):
        return f"{t}.SA"
    return t  # deixa como está p/ casos internacionais

def parse_tickers(texto: str) -> List[str]:
    """Divide por vírgula, espaço ou quebra de linha."""
    if not texto:
        return []
    raw = re.split(r"[\s,;]+", texto.strip())
    return [x for x in raw if x]

def dedup_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------
# 📋 Lista de referência (você pode editar)
# -----------------------------
populares = [
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
    "ABEV3.SA", "BOVA11.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA",
    "PRIO3.SA", "GGBR4.SA", "LREN3.SA", "RAIL3.SA", "HAPV3.SA",
]

colA, colB = st.columns([1.2, 1])
with colA:
    st.subheader("1) Escolha a partir de uma lista sugerida")
    selecao = st.multiselect(
        "Selecione ativos (você pode digitar para buscar):",
        options=populares,
        default=["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    )

with colB:
    st.subheader("2) Informe manualmente (opcional)")
    texto_livre = st.text_area(
        "Cole/Digite tickers separados por vírgula, espaço ou quebra de linha",
        value="PETR4, VALE3, ITUB4, BOVA11",
        height=120,
    )

st.subheader("3) Upload por arquivo (opcional)")
st.write("Aceita `.csv` com uma coluna chamada `ticker`.")
up = st.file_uploader("Envie um CSV de tickers", type=["csv"])

# -----------------------------
# 🧮 Consolidação & Validação
# -----------------------------
lista_texto = [normalize_ticker(t) for t in parse_tickers(texto_livre)]
lista_upload = []
if up is not None:
    try:
        df_up = pd.read_csv(up)
        col = None
        for c in df_up.columns:
            if c.strip().lower() in {"ticker", "tickers", "papel"}:
                col = c
                break
        if col is None:
            st.error("CSV deve conter uma coluna `ticker` (ou `papel`).")
        else:
            lista_upload = [normalize_ticker(str(x)) for x in df_up[col].astype(str).tolist()]
    except Exception as e:
        st.error(f"Falha ao ler CSV: {e}")

todos_raw = list(selecao) + lista_texto + lista_upload
# remove vazios, normaliza e deduplica
norm = [normalize_ticker(t) for t in todos_raw if str(t).strip()] 
lista_final = dedup_order(norm)

# validação
validos = []
invalidos = []
for t in lista_final:
    if TICKER_PATTERN.match(t):
        validos.append(t)
    else:
        invalidos.append(t)

st.markdown("---")
st.subheader("✅ Pré-visualização da seleção")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Válidos ({len(validos)}):**")
    if validos:
        st.code(", ".join(validos))
    else:
        st.info("Nenhum ticker válido até o momento.")
with col2:
    st.write(f"**Suspeitos/Inválidos ({len(invalidos)}):**")
    if invalidos:
        st.warning(", ".join(invalidos))
    else:
        st.success("Nenhum inválido detectado.")

st.markdown("---")
confirm = st.button("✅ Confirmar seleção e salvar no estado", type="primary")
if confirm:
    if not validos:
        st.error("Seleção vazia. Adicione ao menos 1 ticker válido.")
    else:
        st.session_state["tickers_selecionados"] = validos
        st.success(f"Selecionados {len(validos)} tickers.")
        st.toast("Tickers salvos! Siga para a Etapa 2 (coleta de dados).", icon="✅")

# Exibe no rodapé o estado atual (útil ao integrar com as próximas etapas)
if "tickers_selecionados" in st.session_state:
    st.caption("**Estado atual**: tickers selecionados → " + ", ".join(st.session_state["tickers_selecionados"]))
else:
    st.caption("**Estado atual**: ainda não há tickers salvos. Clique em Confirmar quando estiver pronto.")

# ============================================================
# ETAPA 2 — Coleta e preparação de dados (yfinance)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

st.markdown("## 📥 Etapa 2 — Coleta e preparação de dados")

# 1) Recupera tickers salvos na Etapa 1
tickers_sel = st.session_state.get("tickers_selecionados", [])
if not tickers_sel:
    st.info("Nenhum ticker confirmado na Etapa 1. Volte e clique em **Confirmar seleção**.")
    st.stop()

# 2) Parâmetros de coleta
colp1, colp2 = st.columns([1,1])
with colp1:
    period_prices = st.selectbox("Período de preços (para momentum)", ["1y","2y","5y"], index=1)
with colp2:
    st.caption("Dica: se algum ativo não tiver campo em `info`, ele aparece como NaN.")

# 3) Funções utilitárias e cache
def _pct_change_over(prices: pd.Series, days: int) -> float:
    if len(prices) < days + 1:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[-(days+1)] - 1.0)

@st.cache_data(show_spinner=True)
def fetch_info_and_prices(tickers, period_prices="2y"):
    rows = []
    price_hist = {}
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            info = t.info  # dicionário com fundamentos (pode ter ausências)
            hist = t.history(period=period_prices, interval="1d")
            if "Close" in hist:
                px = hist["Close"].dropna()
            else:
                px = pd.Series(dtype=float)
            price_hist[tk] = px

            rows.append({
                "Ticker": tk,
                "Empresa": info.get("longName"),
                "Setor": info.get("sector"),

                # --- Valuation ---
                "P/L": info.get("trailingPE"),
                "P/VP": info.get("priceToBook"),
                "EV/EBITDA": info.get("enterpriseToEbitda"),
                "P/Sales": info.get("priceToSalesTrailing12Months"),
                "Dividend Yield (%)": (info.get("dividendYield") or np.nan) * 100 if info.get("dividendYield") else np.nan,

                # --- Rentabilidade ---
                "ROE (%)": (info.get("returnOnEquity") or np.nan) * 100 if info.get("returnOnEquity") else np.nan,
                "ROA (%)": (info.get("returnOnAssets") or np.nan) * 100 if info.get("returnOnAssets") else np.nan,
                "Margem Bruta (%)": (info.get("grossMargins") or np.nan) * 100 if info.get("grossMargins") else np.nan,
                "Margem Operacional (%)": (info.get("operatingMargins") or np.nan) * 100 if info.get("operatingMargins") else np.nan,
                "Margem Líquida (%)": (info.get("profitMargins") or np.nan) * 100 if info.get("profitMargins") else np.nan,
                "Margem EBITDA (%)": (info.get("ebitdaMargins") or np.nan) * 100 if info.get("ebitdaMargins") else np.nan,

                # --- Endividamento / Liquidez (quando disponível) ---
                "Debt/Equity": info.get("debtToEquity"),
                "Current Ratio": info.get("currentRatio"),
                "Quick Ratio": info.get("quickRatio"),

                # --- Mercado ---
                "Market Cap (R$ bi)": (info.get("marketCap") or 0) / 1e9,
            })
        except Exception as e:
            st.warning(f"Falha ao coletar {tk}: {e}")

    df_info = pd.DataFrame(rows)

    # 4) Momentum (retornos aproximando janelas de pregões)
    for tk, px in price_hist.items():
        if px.empty:
            continue
        df_info.loc[df_info["Ticker"] == tk, "Ret 1M (%)"]  = _pct_change_over(px, 21)  * 100
        df_info.loc[df_info["Ticker"] == tk, "Ret 3M (%)"]  = _pct_change_over(px, 63)  * 100
        df_info.loc[df_info["Ticker"] == tk, "Ret 6M (%)"]  = _pct_change_over(px, 126) * 100
        df_info.loc[df_info["Ticker"] == tk, "Ret 12M (%)"] = _pct_change_over(px, 252) * 100

    return df_info, price_hist

with st.spinner("Coletando dados do Yahoo Finance..."):
    df_info, price_hist = fetch_info_and_prices(tickers_sel, period_prices=period_prices)

# 5) Exibição e persistência para próximas etapas
if df_info.empty:
    st.error("Não foi possível coletar dados. Verifique os tickers e tente novamente.")
    st.stop()

st.success(f"Dados coletados para {len(df_info)} ativos.")
st.dataframe(df_info.sort_values("Ticker").reset_index(drop=True), use_container_width=True)

# Salva no estado para Etapa 3 (rankings/scores)
st.session_state["df_info"] = df_info
st.session_state["price_hist"] = price_hist

# Opcional: download CSV
csv_bytes = df_info.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Baixar CSV (fundamentos + momentum)", data=csv_bytes, file_name="dados_yfinance.csv", mime="text/csv")

