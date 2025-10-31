# -------------------------------------------------------------
# Step 1 — Seleção de Ativos (.SA) por Setor ou por Ticker
# Autor: Luiz E. Gaio + ChatGPT
# Descrição: Componente inicial do app Streamlit para escolher
#            tickers da B3 a partir de uma classificação setorial
#            (arquivo ClassifSetorial.xlsx no repositório) OU
#            diretamente por multiselect de tickers.
# -------------------------------------------------------------

import re
import unicodedata
from typing import List, Tuple, Dict

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Step 1 – Seleção de Ativos", layout="wide")
st.title("📌 Step 1 — Seleção de Ativos (.SA)")
st.caption(
    "Selecione empresas por **setor** (a partir do arquivo `ClassifSetorial.xlsx`) ou diretamente por **ticker**. "
    "Se a planilha não tiver a coluna Ticker, o app monta automaticamente a partir de **CÓDIGO + série padrão + `.SA`**."
)

# --------- Utilitários ---------
TICKER_PATTERN = re.compile(r"^[A-Z]{3,6}\d{0,2}(?:\.SA)?$")

def normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.endswith(".SA"):
        return t
    # Se vier já com letras+numero, acrescenta .SA
    if re.match(r"^[A-Z]{3,6}\d{1,2}$", t):
        return f"{t}.SA"
    # Se vier só letras (ex.: PETR), deixa para montagem com série
    return t

def dedup_order(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def strip_accents_lower(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.lower().strip()

@st.cache_data(show_spinner=False)
def load_classificacao(path: str = "ClassifSetorial.xlsx") -> Tuple[pd.DataFrame, str | None]:
    """
    Lê o Excel e padroniza colunas.
    Aceita headers como: SETOR, SUBSETOR, SEGMENTO, NOME DE PREGÃO, CÓDIGO, TICKER (opcional).
    Retorna (df_padronizado, msg_erro).
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except FileNotFoundError:
        return pd.DataFrame(), "Arquivo 'ClassifSetorial.xlsx' não encontrado na raiz do repositório."
    except Exception as e:
        return pd.DataFrame(), f"Falha ao abrir o Excel: {e}"

    original_cols = list(df.columns)

    # Normaliza nomes para matching flexível
    norm_map: Dict[str, str] = {}
    for c in original_cols:
        k = strip_accents_lower(c)
        k = k.replace(" ", "").replace("_", "").replace("-", "")
        norm_map[k] = c

    # Possíveis aliases (case-insensitive e sem acentos)
    want = {
        "Setor":    ["setor", "setoreconomico", "setorfinanceiro"],
        "Subsetor": ["subsetor", "subsector", "sub-setor"],
        "Segmento": ["segmento", "segmentoeconomico", "segment"],
        "Empresa":  ["nomedepregao", "empresa", "companhia", "nomepregao"],
        "Codigo":   ["codigo", "codigodenegociacao", "codneg", "tickerb3", "papel", "ativo"],
        "Ticker":   ["ticker"]  # se já existir na planilha, usamos preferencialmente
    }

    # Monta renomeação
    rename = {}
    for target, candidates in want.items():
        for key in candidates:
            if key in norm_map:
                rename[norm_map[key]] = target
                break

    df = df.rename(columns=rename)

    # Checks mínimos
    if "Setor" not in df.columns:
        return pd.DataFrame(), "Planilha sem coluna equivalente a **SETOR**."
    if "Ticker" not in df.columns and "Codigo" not in df.columns:
        return pd.DataFrame(), "Planilha precisa ter **Ticker** ou **CÓDIGO**."

    # Normaliza colunas que existirem
    keep_cols = ["Setor", "Subsetor", "Segmento", "Empresa", "Ticker", "Codigo"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = pd.NA

    # Limpeza básica
    for c in ["Setor", "Subsetor", "Segmento", "Empresa", "Codigo", "Ticker"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Linhas válidas de setor
    df = df[df["Setor"].notna() & (df["Setor"] != "")].copy().reset_index(drop=True)
    return df, None

# --------- Carrega classificação ---------
df_class, msg = load_classificacao()
if msg:
    st.warning(msg)

# Se não houver planilha ou faltar colunas mínimas, ofereça apenas seleção por ticker de fallback
tem_base = not df_class.empty and ("Setor" in df_class.columns)

# Série padrão (usada apenas quando não existe coluna Ticker na planilha)
serie_default = st.sidebar.selectbox("Série padrão (quando faltar Ticker na planilha)", ["3", "4", "11"], index=0)

# Monta coluna TickerFinal (preferência: Ticker; senão, Codigo + série)
def build_ticker_final(row) -> str:
    if pd.notna(row.get("Ticker")) and str(row.get("Ticker")).strip():
        return normalize_ticker(str(row["Ticker"]))
    cod = str(row.get("Codigo") or "").strip().upper()
    if not cod:
        return ""
    base = cod if cod.endswith(".SA") else f"{cod}{serie_default}.SA"
    return normalize_ticker(base)

if tem_base:
    df_class["TickerFinal"] = df_class.apply(build_ticker_final, axis=1)
    # remove vazios
    df_class = df_class[df_class["TickerFinal"].astype(bool)].reset_index(drop=True)

# --------- Modo de seleção ---------
modo = st.radio("Como deseja selecionar?", ["Por setor", "Por ticker"], horizontal=True)

validos, invalidos = [], []

if modo == "Por setor":
    st.subheader("Selecionar por Setor")

    if not tem_base:
        st.warning("Classificação não disponível. Use a seleção por ticker.")
    else:
        # Contagem por setor
        set_counts = (
            df_class.groupby("Setor")["TickerFinal"]
            .nunique()
            .sort_index()
            .to_dict()
        )
        setores_opcoes = [f"{s} ({set_counts.get(s,0)})" for s in sorted(set_counts.keys())]
        map_display_to_setor = {disp: disp.rsplit(" (", 1)[0] for disp in setores_opcoes}

        colA, colB, colC = st.columns([1.4, 1.2, 1])
        with colA:
            sel_disp = st.multiselect("Setores", options=setores_opcoes)
            setores_sel = [map_display_to_setor[x] for x in sel_disp]

        df_f = df_class.copy()
        if setores_sel:
            df_f = df_f[df_f["Setor"].isin(setores_sel)]

        # Subfiltros
        subsetores = sorted(df_f["Subsetor"].dropna().unique().tolist()) if "Subsetor" in df_f.columns else []
        segmentos  = sorted(df_f["Segmento"].dropna().unique().tolist()) if "Segmento" in df_f.columns else []

        with colB:
            if subsetores:
                sub_sel = st.multiselect("Subsetores (opcional)", options=subsetores)
                if sub_sel:
                    df_f = df_f[df_f["Subsetor"].isin(sub_sel)]
            if segmentos:
                seg_sel = st.multiselect("Segmentos (opcional)", options=segmentos)
                if seg_sel:
                    df_f = df_f[df_f["Segmento"].isin(seg_sel)]

        with colC:
            st.metric("Empresas filtradas", int(df_f["TickerFinal"].nunique()))

        # Opções finais (Ticker — Empresa)
        if "Empresa" in df_f.columns and df_f["Empresa"].notna().any():
            op = df_f[["TickerFinal", "Empresa"]].drop_duplicates()
            op["label"] = op["TickerFinal"] + " — " + op["Empresa"].astype(str)
            labels = op["label"].tolist()
            label_to_ticker = dict(zip(op["label"], op["TickerFinal"]))
            escolhidas = st.multiselect("Escolha as empresas:", options=labels)
            validos = [label_to_ticker[l] for l in escolhidas]
        else:
            op = sorted(df_f["TickerFinal"].drop_duplicates().tolist())
            validos = st.multiselect("Escolha as empresas:", options=op)

else:  # Por ticker
    st.subheader("Selecionar por Ticker")
    if tem_base:
        options = sorted(df_class["TickerFinal"].drop_duplicates().tolist())
    else:
        # fallback mínimo
        options = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "ABEV3.SA", "WEGE3.SA", "PRIO3.SA"]
    selecionados = st.multiselect("Digite ou escolha tickers:", options=options, default=options[:3])
    validos = [normalize_ticker(t) for t in selecionados if t]

# --------- Validação ---------
invalidos = [t for t in validos if not TICKER_PATTERN.match(t)]

st.markdown("---")
st.subheader("✅ Pré-visualização da seleção")
c1, c2 = st.columns(2)
with c1:
    st.write(f"**Válidos ({len(validos)}):**")
    st.code(", ".join(dedup_order(validos)) if validos else "—")
with c2:
    st.write(f"**Suspeitos/Inválidos ({len(invalidos)}):**")
    if invalidos:
        st.warning(", ".join(dedup_order(invalidos)))
    else:
        st.success("Nenhum inválido detectado.")

st.markdown("---")
if st.button("✅ Confirmar seleção e salvar no estado", type="primary"):
    final = dedup_order([normalize_ticker(t) for t in validos if t])
    if not final:
        st.error("Seleção vazia. Adicione ao menos 1 ticker.")
    else:
        st.session_state["tickers_selecionados"] = final
        st.success(f"Selecionados {len(final)} tickers.")
        st.toast("Tickers salvos! Siga para a Etapa 2 (coleta de dados).", icon="✅")

st.caption("**Estado atual**: " + (", ".join(st.session_state.get("tickers_selecionados", [])) or "nenhum ticker salvo."))


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

