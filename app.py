# -------------------------------------------------------------
# Step 1 — Seleção de Ativos (.SA) por Setor ou por Ticker
# Autor: Luiz E. Gaio + ChatGPT
# Descrição: Componente inicial do app Streamlit para escolher
#            tickers da B3 a partir de uma classificação setorial
#            (arquivo ClassifSetorial.xlsx no repositório) OU
#            diretamente por multiselect de tickers.
# -------------------------------------------------------------

import re
import pandas as pd
import streamlit as st
from typing import List

st.set_page_config(page_title="Step 1 – Seleção de Ativos", layout="wide")
st.title("📌 Step 1 — Seleção de Ativos (.SA)")
st.caption("Selecione as empresas por **setor** (a partir do arquivo `ClassifSetorial.xlsx`) ou diretamente por **ticker**. O sufixo .SA é adicionado automaticamente.")

# -----------------------------
# 🔧 Utilitários
# -----------------------------
TICKER_PATTERN = re.compile(r"^[A-Z]{3,5}\d{1,2}(?:\.SA)?$")

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.endswith(".SA"):
        return t
    if re.match(r"^[A-Z]{3,5}\d{1,2}$", t):
        return f"{t}.SA"
    return t

def dedup_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -----------------------------
# 📂 Carregamento da classificação setorial
# -----------------------------
@st.cache_data(show_spinner=False)
def load_classificacao(path: str = "ClassifSetorial.xlsx") -> pd.DataFrame:
    """Lê o Excel de classificação e normaliza nomes de colunas.
    Espera colunas pelo menos: Ticker, Setor (opcionalmente Empresa/SubSetor/Segmento).
    """
    df = pd.read_excel(path)
    # normaliza nomes
    df.columns = [c.strip().lower() for c in df.columns]
    # mapeia aliases comuns
    colmap = {}
    for alvo, aliases in {
        "ticker": ["ticker", "papel", "ativo"],
        "empresa": ["empresa", "nome", "companhia"],
        "setor": ["setor", "sector"],
        "subsetor": ["subsetor", "sub-setor", "subsector"],
        "segmento": ["segmento", "segment"]
    }.items():
        for a in aliases:
            if a in df.columns:
                colmap[a] = alvo
                break
    df = df.rename(columns=colmap)
    if "ticker" not in df.columns or "setor" not in df.columns:
        raise ValueError("O arquivo ClassifSetorial.xlsx deve conter ao menos as colunas 'Ticker' e 'Setor'.")
    # normaliza tickers
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip().apply(normalize_ticker)
    # remove linhas vazias
    df = df.dropna(subset=["ticker", "setor"]).reset_index(drop=True)
    return df

# tenta carregar
try:
    df_class = load_classificacao()
    setores = sorted(df_class["setor"].dropna().unique().tolist())
except Exception as e:
    df_class = pd.DataFrame()
    setores = []
    st.warning(f"Não foi possível carregar `ClassifSetorial.xlsx`: {e}")

# -----------------------------
# 🎛️ Modo de seleção
# -----------------------------
modo = st.radio(
    "Como deseja selecionar?",
    ["Por setor", "Por ticker"],
    horizontal=True,
)

validos = []
invalidos = []

if modo == "Por setor":
    st.subheader("Selecionar por Setor")
    if df_class.empty:
        st.error("O arquivo `ClassifSetorial.xlsx` não foi encontrado ou está inválido. Envie o arquivo para o repositório e atualize a página.")
    else:
        cols = st.columns([1.2, 1.2, 1])
        with cols[0]:
            setores_sel = st.multiselect("Setores", options=setores)
        # filtra por setor
        if setores_sel:
            df_filtrado = df_class[df_class["setor"].isin(setores_sel)].copy()
            # subsetor e segmento, se existirem
            if "subsetor" in df_filtrado.columns:
                subsets = sorted(df_filtrado["subsetor"].dropna().unique().tolist())
            else:
                subsets = []
            if "segmento" in df_filtrado.columns:
                segs = sorted(df_filtrado["segmento"].dropna().unique().tolist())
            else:
                segs = []
            with cols[1]:
                if subsets:
                    subset_sel = st.multiselect("Subsetores (opcional)", options=subsets)
                    if subset_sel:
                        df_filtrado = df_filtrado[df_filtrado["subsetor"].isin(subset_sel)]
                if segs:
                    seg_sel = st.multiselect("Segmentos (opcional)", options=segs)
                    if seg_sel:
                        df_filtrado = df_filtrado[df_filtrado["segmento"].isin(seg_sel)]
            with cols[2]:
                st.metric("Empresas filtradas", len(df_filtrado))

            # lista de tickers filtrados e escolha final
            # se existir coluna empresa, mostra no label
            if "empresa" in df_filtrado.columns:
                opcoes = df_filtrado[["ticker", "empresa"]].drop_duplicates()
                opcoes["label"] = opcoes["ticker"] + " — " + opcoes["empresa"].astype(str)
                labels = opcoes["label"].tolist()
                label_to_ticker = dict(zip(opcoes["label"], opcoes["ticker"]))
                escolha = st.multiselect("Escolha as empresas do filtro:", options=labels)
                validos = [label_to_ticker[l] for l in escolha]
            else:
                opcoes = sorted(df_filtrado["ticker"].drop_duplicates().tolist())
                validos = st.multiselect("Escolha as empresas do filtro:", options=opcoes)
        else:
            st.info("Selecione ao menos um setor para ver as empresas.")

else:  # Por ticker
    st.subheader("Selecionar por Ticker")
    # Sugestões (opcionais)
    sugestoes = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
        "ABEV3.SA", "BOVA11.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA",
        "PRIO3.SA", "GGBR4.SA", "LREN3.SA", "RAIL3.SA", "HAPV3.SA",
    ]
    # Se tivermos df_class, usamos tickers de lá como options
    options = sorted(df_class["ticker"].unique().tolist()) if not df_class.empty else sugestoes
    selecionados = st.multiselect(
        "Digite ou escolha tickers:",
        options=options,
        default=["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
    )
    validos = [normalize_ticker(t) for t in selecionados if t]

# validação simples
for t in validos:
    if not TICKER_PATTERN.match(t):
        invalidos.append(t)

st.markdown("---")
st.subheader("✅ Pré-visualização da seleção")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Válidos ({len(validos)}):**")
    if validos:
        st.code(", ".join(dedup_order(validos)))
    else:
        st.info("Nenhum ticker válido até o momento.")
with col2:
    st.write(f"**Suspeitos/Inválidos ({len(invalidos)}):**")
    if invalidos:
        st.warning(", ".join(dedup_order(invalidos)))
    else:
        st.success("Nenhum inválido detectado.")

st.markdown("---")
confirm = st.button("✅ Confirmar seleção e salvar no estado", type="primary")
if confirm:
    if not validos:
        st.error("Seleção vazia. Adicione ao menos 1 ticker.")
    else:
        final = dedup_order([normalize_ticker(t) for t in validos])
        st.session_state["tickers_selecionados"] = final
        st.success(f"Selecionados {len(final)} tickers.")
        st.toast("Tickers salvos! Siga para a Etapa 2 (coleta de dados).", icon="✅")

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

