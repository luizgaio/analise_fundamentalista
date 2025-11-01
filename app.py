# =============================================================
# Dashboard B3 — Layout inicial com dois modos de análise
# Modo 1: Análise Individual (empresa)
# Modo 2: Screener / Ranking
# -------------------------------------------------------------
# Este arquivo é um esqueleto organizado para evoluir o app.
# Nas próximas etapas, plugaremos a coleta de dados, scores etc.
# =============================================================

from __future__ import annotations
import streamlit as st
from urllib.parse import urlencode
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
def apply_dark_fig(fig):
    return fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a", font_color="#e2e8f0"
    )
import re
import pandas as pd
import streamlit as st
from typing import Tuple, Dict

# ============================================================
# ETAPA 2 — Coleta e preparação de dados (yfinance)
# ============================================================


TRADING_DAYS = {"1M": 21, "3M": 63, "6M": 126, "12M": 252}

def _safe_pct(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _as_pct(x):
    return float(x) * 100 if x is not None else np.nan

def _momentum_from_series(px: pd.Series, days: int) -> float:
    if px is None or px.empty or len(px) <= days:
        return np.nan
    try:
        return float(px.iloc[-1] / px.iloc[-(days+1)] - 1.0)
    except Exception:
        return np.nan

@st.cache_data(show_spinner=True)
def fetch_yf_info_and_prices(ticker: str, period_prices: str = "2y"):
    """Baixa info do yfinance e preços históricos do ativo e do IBOV (^BVSP)."""
    t = yf.Ticker(ticker)
    info = t.info  # dicionário; pode faltar campos
    hist = t.history(period=period_prices, interval="1d")
    px = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)

    # Benchmark (Ibovespa) para referência
    try:
        ibov = yf.Ticker("^BVSP").history(period=period_prices, interval="1d")
        ibov_px = ibov["Close"].dropna() if "Close" in ibov else pd.Series(dtype=float)
    except Exception:
        ibov_px = pd.Series(dtype=float)

    return info, px, ibov_px

def _num(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _pct(v):
    return _num(v) * 100.0 if v is not None else np.nan

def _build_overview_from_info(info: dict) -> pd.DataFrame:
    ev          = _num(info.get("enterpriseValue"))
    ebit        = _num(info.get("ebit")) if info.get("ebit") is not None else _num(info.get("operatingIncome"))
    total_debt  = _num(info.get("totalDebt"))
    total_cash  = _num(info.get("totalCash"))
    net_debt    = (total_debt - total_cash) if (not np.isnan(total_debt) and not np.isnan(total_cash)) else np.nan
    equity      = _num(info.get("totalStockholderEquity"))
    total_assets= _num(info.get("totalAssets"))

    trailing_pe = _num(info.get("trailingPE"))
    forward_pe  = _num(info.get("forwardPE"))
    pb          = _num(info.get("priceToBook"))
    peg         = _num(info.get("pegRatio"))
    ps          = _num(info.get("priceToSalesTrailing12Months"))

    roe = _pct(info.get("returnOnEquity"))
    roa = _pct(info.get("returnOnAssets"))
    dy  = _pct(info.get("dividendYield"))
    beta= _num(info.get("beta"))

    gross_m     = _pct(info.get("grossMargins"))
    op_m        = _pct(info.get("operatingMargins"))    # "Margem Operacional (%)"
    ebitda_m    = _pct(info.get("ebitdaMargins"))
    net_m       = _pct(info.get("profitMargins"))

    rows = [{
        # Identidade
        "Empresa": info.get("longName"),
        "Setor":   info.get("sector"),

        # Valor da Empresa
        "Market Cap":        _num(info.get("marketCap")),
        "Enterprise Value":  ev,

        # Análise de Mercado
        "P/L (Trailing)": trailing_pe,
        "P/L (Forward)":  forward_pe,
        "P/VP":           pb,
        "PEG":            peg,
        "P/Sales":        ps,

        # EV múltiplos
        "EV/EBITDA":   _num(info.get("enterpriseToEbitda")),
        "EV/Revenue":  _num(info.get("enterpriseToRevenue")),
        "EV/EBIT (calc)": (ev/ebit) if (not np.isnan(ev) and not np.isnan(ebit) and ebit != 0) else np.nan,

        # Rentabilidade e Risco
        "ROE (%)": roe,
        "ROA (%)": roa,
        "Dividend Yield (%)": dy,
        "Beta": beta,

        # Eficiência / Margens
        "Margem Bruta (%)":    gross_m,
        "Margem EBIT (%)":     op_m,        # mantemos como EBIT no seu layout…
        "Margem EBITDA (%)":   ebitda_m,
        "Margem Líquida (%)":  net_m,
        # …mas também expomos o nome que a Etapa 3 já usa:
        "Margem Operacional (%)": op_m,

        # Endividamento
        "Net Debt/Equity": (net_debt/equity) if (not np.isnan(net_debt) and not np.isnan(equity) and equity != 0) else np.nan,
        "Net Debt/Assets": (net_debt/total_assets) if (not np.isnan(net_debt) and not np.isnan(total_assets) and total_assets != 0) else np.nan,
        "Debt/Equity":    _num(info.get("debtToEquity")),  # alias p/ Etapa 3

        # Liquidez / Giro
        "Liquidez Corrente": _num(info.get("currentRatio")),
        "Liquidez Seca":     _num(info.get("quickRatio")),
        "Giro do Ativo":     _num(info.get("assetTurnover")),
        "Giro de Estoque":   _num(info.get("inventoryTurnover")),
    }]

    df = pd.DataFrame(rows)

    # 🔁 Aliases para compatibilidade com a Etapa 3
    df["P/L"] = df["P/L (Trailing)"]      # Etapa 3 espera "P/L"
    # "P/VP" já existe com esse nome
    return df

# ------------------------------
# Configuração básica da página
# ------------------------------
st.set_page_config(
    page_title="Análise Fundamentalista de Ações",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== THEME / STYLES (dark consistente em toda a página) ======
def inject_base_styles():
    st.markdown("""
    <style>
      :root {
        --bg: #0f172a;
        --bg2: #0b1220;
        --tx: #e2e8f0;
        --tx2: #cbd5e1;
        --bd: rgba(255,255,255,.08);
      }

      /* ====== Layout geral ====== */
      html, body, .stApp, div[data-testid="stAppViewContainer"] {
        background: var(--bg)!important;
        color: var(--tx)!important;
      }
      section[data-testid="stSidebar"] {
        background: var(--bg2)!important;
        border-right: 1px solid var(--bd);
      }
      .block-container { background:transparent!important; padding-top: 0.5rem !important; }

      /* ====== DataFrames e Tabelas ====== */
      .stDataFrame, .stTable {
        background: var(--bg2)!important;
        border: 1px solid var(--bd)!important;
        border-radius: 10px;
      }
      .stDataFrame thead, .stTable thead {
        background: var(--bg)!important;
        color: var(--tx)!important;
      }
      .stDataFrame tbody, .stTable tbody { color: var(--tx)!important; }

      /* ====== Select / Dropdown ====== */
      div[data-baseweb="select"] {
        background: var(--bg2)!important;
        color: var(--tx)!important;
        border: 1px solid var(--bd)!important;
      }
      div[role="listbox"] {
        background: var(--bg2)!important;
        color: var(--tx)!important;
        border: 1px solid var(--bd)!important;
      }
      div[role="option"] { color: var(--tx)!important; }
      input::placeholder { color: var(--tx2)!important; }

      /* ====== Expanders ====== */
      .st-expander {
        background: var(--bg2)!important;
        border: 1px solid var(--bd)!important;
        border-radius: 12px;
      }

      /* ====== ABAS (Tabs) ====== */
      /* Container das abas */
      .stTabs [data-baseweb="tab-list"] {
        display: flex;
        gap: 12px;                           /* espaçamento entre as abas */
        background: transparent;
        border: none;
        margin-bottom: 1rem;
      }

      /* Cada aba como um “bloquinho” */
      .stTabs [data-baseweb="tab"] {
        border: 1px solid var(--bd);
        border-radius: 10px;
        background: var(--bg2);
        color: var(--tx2)!important;
        padding: 8px 14px;
        font-weight: 500;
        transition: all 0.2s ease-in-out;
      }

      /* Aba ativa */
      .stTabs [aria-selected="true"] {
        color: var(--tx)!important;
        border-color: #60a5fa!important;
        box-shadow: 0 0 0 1px #60a5fa inset;
      }

      /* ====== Métricas ====== */
      div[data-testid="stMetric"] {
        background: rgba(255,255,255,.06)!important;
        border: 1px solid var(--bd);
        border-radius: 12px;
        text-align: center;
      }
      div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        color: var(--tx)!important;
        text-align: center;
        width: 100%;
        display: block;
      }

      /* ====== Botões ====== */
      .stButton > button {
        background: var(--bg2)!important;
        color: var(--tx)!important;
        border: 1px solid var(--bd)!important;
      }
    </style>
    """, unsafe_allow_html=True)


inject_base_styles()

# ------------------------------
# Utils de navegação
# ------------------------------
MODES = {"home": "Início", "single": "Análise Individual", "screener": "Screener"}

def set_mode(mode: str):
    st.session_state["mode"] = mode
    st.experimental_set_query_params(**{"mode": mode})

# Query param → carrega modo ao abrir o app
params = st.experimental_get_query_params()
mode_param = params.get("mode", ["home"]) [0]
if "mode" not in st.session_state:
    st.session_state["mode"] = mode_param if mode_param in MODES else "home"

# ------------------------------
# Estilos
# ------------------------------
CARD_CSS = """
<style>
:root {
  --radius: 18px;
}
.card {
  border: 1px solid rgba(0,0,0,0.07);
  border-radius: var(--radius);
  padding: 18px 18px 14px 18px;
  transition: all .18s ease;
  background: #0b2545;              /* azul escuro */
}
[data-theme="dark"] .card {
  background: #0b2545;              /* azul escuro no tema escuro também */
  border-color: rgba(255,255,255,0.08);
}
.btn-primary { background:#2F6BFF; color:white; }
.btn-ghost   { background:transparent; border-color:rgba(0,0,0,0.15); }
[data-theme="dark"] .btn-ghost { border-color:rgba(255,255,255,0.2); color:#fff; }
.badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; opacity:.9; }
.badge-green { background:rgba(16,185,129,.15); color:#0F9D58; }
.badge-blue  { background:rgba(47,107,255,.12); color:#2F6BFF; }
hr.soft { border:none; border-top:1px solid rgba(0,0,0,0.08); margin:18px 0; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)

# Botão pequeno customizado (para "Começar")
st.markdown("""
<style>
.btn-small {
  display:inline-block;
  background-color:#1d4ed8; /* Azul forte */
  color:#fff !important;
  padding:4px 10px;
  border-radius:8px;
  font-size:0.85rem;
  text-decoration:none;
  transition:background 0.2s ease;
}
.btn-small:hover {
  background-color:#2563eb;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* badge maior para o SETOR */
.badge-sector{
  display:inline-block; padding:6px 14px; border-radius:999px;
  font-size:14px; font-weight:700; letter-spacing:.4px;
  background:rgba(47,107,255,.18); color:#e2e8f0; text-transform:uppercase;
}
/* deixa subsetor/segmento em MAIÚSCULAS mantendo seu estilo */
.badge-blue{ text-transform:uppercase; }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Cabeçalho
# ------------------------------
col_logo, col_title = st.columns([0.08, 0.92])

with col_title:
    # ===============================
    # Cabeçalho da página
    # ===============================
    modo = st.session_state.get("mode", "home")

    if modo == "home":
        st.markdown("""
        <div style='text-align: center; margin-bottom: 1rem;'>
            <h1 style='font-size:3rem; font-weight:700;'>Análise Fundamentalista de Ações</h1>
            <p style='color:#cbd5e1; font-size:1.05rem;'>
                Escolha o modo de análise: <b>Individual</b> (uma empresa) ou <b>Screener</b> (ranking de várias).
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif modo == "single":
        st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)


# ------------------------------
# Sidebar de navegação
# ------------------------------
with st.sidebar:
    st.markdown("### Navegação")
    st.markdown("<div style='margin-top: -10px'></div>", unsafe_allow_html=True)
    # Mapeia os modos com rótulos
    nav_items = [
        ("home",    "Início"),
        ("single",  "Análise Individual"),
        ("screener","Screener"),
    ]

    # Renderiza 3 botões verticais; o modo atual fica como "primary"
    for mode_key, label in nav_items:
        is_current = st.session_state.get("mode") == mode_key
        if st.button(label, type=("primary" if is_current else "secondary"), use_container_width=True):
            set_mode(mode_key)

# ------------------------------
# Páginas
# ------------------------------

def render_home():
    st.markdown("---")
    c1, c2 = st.columns(2)

    # ----------- CARD 1: Análise Individual -----------
    with c1:
        st.markdown(
            """
            <div class='card'>
              <span class='badge badge-blue'>Modo 1</span>
              <h3>🔎 Análise Individual</h3>
              <p>Estude profundamente uma empresa: múltiplos, rentabilidade, endividamento, histórico de preços e comparativos de setor.</p>
              <a href='?mode=single' class='btn-small'>Começar</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------- CARD 2: Screener / Ranking -----------
    with c2:
        st.markdown(
            """
            <div class='card'>
              <span class='badge badge-green'>Modo 2</span>
              <h3>📈 Screener / Ranking</h3>
              <p>Monte um ranking de empresas por múltiplos e qualidade. Filtros por setor, pesos customizados e exportação para CSV.</p>
              <a href='?mode=screener' class='btn-small'>Começar</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# MODO: ANÁLISE INDIVIDUAL
# Visão  (por lista OU por setor→subsetor→segmento)
# ============================================================

# ---------- Utils de leitura e normalização ----------
def _normalize_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t:
        return ""
    if t.endswith(".SA"):
        return t
    # padrão B3: letras + classe numérica (3, 4, 11)
    if re.match(r"^[A-Z]{3,6}\d{1,2}$", t):
        return f"{t}.SA"
    return t

@st.cache_data(show_spinner=False)
def load_classif_setorial(path: str = "ClassifSetorial.xlsx") -> Tuple[pd.DataFrame, str | None]:
    """
    Espera colunas (qualquer caixa/acento): SETOR, SUBSETOR, SEGMENTO, NOME DE PREGÃO, CÓDIGO e/ou TICKER.
    Se TICKER não existir, monta a partir de CÓDIGO + série escolhida na UI (definida fora desta função).
    Aqui apenas padronizamos nomes; a montagem final do ticker fica em etapa1_selecao_empresa().
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except FileNotFoundError:
        return pd.DataFrame(), "Arquivo 'ClassifSetorial.xlsx' não encontrado na raiz do repositório."
    except Exception as e:
        return pd.DataFrame(), f"Falha ao abrir o Excel: {e}"

    # padroniza nomes de colunas para comparação
    up = {c: c.strip().upper() for c in df.columns}
    df.rename(columns=up, inplace=True)

    # mapeia para nomes-alvo
    colmap: Dict[str, str] = {}
    for srcs, dst in [
        (["SETOR"], "Setor"),
        (["SUBSETOR"], "Subsetor"),
        (["SEGMENTO"], "Segmento"),
        (["NOME DE PREGÃO", "NOME PREGÃO", "NOME PREGAO", "NOME DE PREGAO"], "Empresa"),
        (["TICKER"], "Ticker"),
        (["CÓDIGO", "CODIGO", "CÓDIGO DE NEGOCIAÇÃO", "CODIGO DE NEGOCIACAO", "CODNEG"], "Codigo"),
    ]:
        for s in srcs:
            if s in df.columns:
                colmap[s] = dst
                break
    df.rename(columns=colmap, inplace=True)

    # checagem mínima
    if "Setor" not in df.columns:
        return pd.DataFrame(), "A planilha precisa ter a coluna 'SETOR'."
    if ("Ticker" not in df.columns) and ("Codigo" not in df.columns):
        return pd.DataFrame(), "A planilha precisa ter 'TICKER' ou 'CÓDIGO'."

    # normalizações leves
    for c in ["Setor", "Subsetor", "Segmento", "Empresa", "Ticker", "Codigo"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # remove linhas sem setor
    df = df[df["Setor"].notna() & (df["Setor"] != "")].copy().reset_index(drop=True)
    return df, None

def render_sidebar_selector():
    """Seleciona a empresa na SIDEBAR e preenche:
       empresa_escolhida, empresa_nome, empresa_setor, empresa_subsetor, empresa_segmento
    """
    with st.sidebar:
        st.markdown("### 🔎 Seleção da Empresa")
        df_class, msg = load_classif_setorial()
        if msg:
            st.warning(msg)
            return None

        serie = st.selectbox("Tipo de ação", ["3","4","5","6","11"], index=1,
                             help="3 = ON; 4/5/6 = PN; 11 = Unit.")

        def _build_ticker(row):
            tk = str(row.get("Ticker", "")).strip()
            if tk:
                return _normalize_ticker(tk)
            cod = str(row.get("Codigo","")).strip().upper()
            return _normalize_ticker(f"{cod}{serie}.SA") if cod else ""

        df_class["TickerFinal"] = df_class.apply(_build_ticker, axis=1)
        df_valid = df_class[df_class["TickerFinal"]!=""].copy()

        modo = st.radio("Como deseja selecionar?",
                        ["Por lista de tickers", "Por Setor"],
                        index=0)

        chosen = {"ticker": None, "empresa": None, "setor": None, "subsetor": None, "segmento": None}

        if modo == "Por lista de tickers":
            if "Empresa" in df_valid.columns and df_valid["Empresa"].notna().any():
                df_valid["label"] = df_valid["TickerFinal"] + " — " + df_valid["Empresa"].astype(str)
                label = st.selectbox("Empresa", df_valid["label"].sort_values(), index=0)
                row  = df_valid.loc[df_valid["label"] == label].iloc[0]
            else:
                tk  = st.selectbox("Ticker", sorted(df_valid["TickerFinal"].unique()))
                row = df_valid.loc[df_valid["TickerFinal"] == tk].iloc[0]
        else:
            df_f = df_valid.copy()
            setor = st.selectbox("Setor", sorted(df_f["Setor"].dropna().unique()))
            df_f = df_f[df_f["Setor"]==setor]

            subsetores = sorted(df_f["Subsetor"].dropna().unique()) if "Subsetor" in df_f.columns else []
            subsetor   = st.selectbox("Subsetor", subsetores) if subsetores else None
            if subsetor: df_f = df_f[df_f["Subsetor"]==subsetor]

            segmentos  = sorted(df_f["Segmento"].dropna().unique()) if "Segmento" in df_f.columns else []
            segmento   = st.selectbox("Segmento", segmentos) if segmentos else None
            if segmento: df_f = df_f[df_f["Segmento"]==segmento]

            if "Empresa" in df_f.columns and df_f["Empresa"].notna().any():
                df_f["label"] = df_f["TickerFinal"] + " — " + df_f["Empresa"].astype(str)
                label = st.selectbox("Empresa", df_f["label"].sort_values())
                row   = df_f.loc[df_f["label"] == label].iloc[0]
            else:
                tk  = st.selectbox("Ticker", sorted(df_f["TickerFinal"].unique()))
                row = df_f.loc[df_f["TickerFinal"] == tk].iloc[0]

        chosen["ticker"]   = row["TickerFinal"]
        chosen["empresa"]  = row.get("Empresa") or row["TickerFinal"]
        chosen["setor"]    = row.get("Setor")
        chosen["subsetor"] = row.get("Subsetor")
        chosen["segmento"] = row.get("Segmento")

        st.session_state["empresa_escolhida"] = chosen["ticker"]
        st.session_state["empresa_nome"]     = chosen["empresa"]
        st.session_state["empresa_setor"]    = chosen["setor"]
        st.session_state["empresa_subsetor"] = chosen["subsetor"]
        st.session_state["empresa_segmento"] = chosen["segmento"]

        st.success(f"Selecionado: **{chosen['ticker']}**")
        return chosen

def etapa1_selecao_empresa():
    st.markdown("### Seleção da Empresa")

    # 1) Carrega base setorial
    df_class, msg = load_classif_setorial()
    if msg:
        st.warning(msg)
        st.stop()

    # 2) Série padrão (somente usada se Ticker não existir na planilha)
    col_ser = st.columns([1, 3])[0]
    serie_padrao = col_ser.selectbox("Escolha o tipo de Ação", ["3", "4", "5", "6", "11"], index=0,
                                     help="3 para ON; 4, 5 e 6 para PN e 11 para Unit. Será usado para montar PETR + 4 → PETR4.SA.")

    # 3) Monta coluna TickerFinal (preferência por Ticker; senão Codigo + série)
    def _build_ticker(row: pd.Series) -> str:
        if "Ticker" in row and str(row["Ticker"]).strip():
            return _normalize_ticker(str(row["Ticker"]))
        cod = str(row.get("Codigo", "")).strip().upper()
        if not cod:
            return ""
        return _normalize_ticker(f"{cod}{serie_padrao}.SA")

    df_class["TickerFinal"] = df_class.apply(_build_ticker, axis=1)
    df_class = df_class[df_class["TickerFinal"] != ""].copy().reset_index(drop=True)

    # 4) UI: modo de seleção
    modo = st.radio("Como deseja selecionar a empresa?",
                    ["Por lista de tickers", "Por Setor → Subsetor → Segmento → Empresa"],
                    horizontal=True)

    empresa_nome = None
    ticker_escolhido = None

    if modo == "Por lista de tickers":
        # lista ordenada por Empresa (quando existir) ou por ticker
        op = df_class.copy()
        if "Empresa" in op.columns and op["Empresa"].notna().any():
            op["label"] = op["TickerFinal"] + " — " + op["Empresa"].astype(str)
            op = op.sort_values(["Empresa", "TickerFinal"])
            labels = op["label"].tolist()
            map_label_ticker = dict(zip(op["label"], op["TickerFinal"]))
            sel = st.selectbox("Escolha a empresa (lista completa)", labels, index=0)
            ticker_escolhido = map_label_ticker.get(sel)
            empresa_nome = op.loc[op["label"] == sel, "Empresa"].iloc[0]
        else:
            tickers = sorted(op["TickerFinal"].unique().tolist())
            ticker_escolhido = st.selectbox("Escolha o ticker", tickers, index=0)
            # Após escolher o ticker_escolhido:
            # tenta buscar o nome completo na planilha (coluna "Empresa")
            try:
                nome_df = df_class.loc[df_class["TickerFinal"] == ticker_escolhido, "Empresa"].dropna()
                empresa_nome = nome_df.iloc[0] if not nome_df.empty else ticker_escolhido
            except Exception:
                empresa_nome = ticker_escolhido


    else:
        # seleção encadeada: setor -> subsetor -> segmento -> empresa
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.4])

        # Setor
        setores = sorted(df_class["Setor"].dropna().unique().tolist())
        setor_sel = c1.selectbox("Setor", ["—"] + setores, index=0)
        df_f = df_class.copy()
        if setor_sel != "—":
            df_f = df_f[df_f["Setor"] == setor_sel]

        # Subsetor
        subsetores = sorted(df_f["Subsetor"].dropna().unique().tolist()) if "Subsetor" in df_f.columns else []
        subsetor_sel = c2.selectbox("Subsetor", ["—"] + subsetores, index=0)
        if subsetor_sel != "—":
            df_f = df_f[df_f["Subsetor"] == subsetor_sel]

        # Segmento
        segmentos = sorted(df_f["Segmento"].dropna().unique().tolist()) if "Segmento" in df_f.columns else []
        segmento_sel = c3.selectbox("Segmento", ["—"] + segmentos, index=0)
        if segmento_sel != "—":
            df_f = df_f[df_f["Segmento"] == segmento_sel]

        # Empresa (label = TickerFinal — Empresa)
        if "Empresa" in df_f.columns and df_f["Empresa"].notna().any():
            op = df_f[["TickerFinal", "Empresa"]].drop_duplicates().copy()
            op["label"] = op["TickerFinal"] + " — " + op["Empresa"].astype(str)
            op = op.sort_values("Empresa")
            labels = ["—"] + op["label"].tolist()
            escolha = c4.selectbox("Empresa", labels, index=0)
            if escolha != "—":
                ticker_escolhido = op.loc[op["label"] == escolha, "TickerFinal"].iloc[0]
                empresa_nome = op.loc[op["label"] == escolha, "Empresa"].iloc[0]
        else:
            # fallback: só ticker
            op = sorted(df_f["TickerFinal"].drop_duplicates().tolist())
            escolha = c4.selectbox("Empresa (por ticker)", ["—"] + op, index=0)
            if escolha != "—":
                ticker_escolhido = escolha
                empresa_nome = escolha

    st.markdown("---")
    # 5) Confirmar e salvar no estado
    if ticker_escolhido:
        st.success(f"Selecionado: **{ticker_escolhido}**" + (f" ({empresa_nome})" if empresa_nome else ""))
        if st.button("✅ Confirmar empresa e avançar para a Etapa 2", type="primary"):
            st.session_state["empresa_escolhida"] = ticker_escolhido
            st.session_state["empresa_nome"] = empresa_nome or ticker_escolhido
            st.session_state["empresa_nome_completo"] = st.session_state["empresa_nome"]  # <- adicione esta linha
            st.toast("Empresa selecionada com sucesso!", icon="✅")
    else:
        st.info("Selecione uma empresa para continuar.")

    # Rodapé da etapa
    if "empresa_escolhida" in st.session_state:
        st.caption(f"**Estado atual**: {st.session_state['empresa_escolhida']} — {st.session_state.get('empresa_nome','')}")
    else:
        st.caption("**Estado atual**: nenhuma empresa confirmada.")

def etapa2_coleta_dados():
    st.markdown("### Visão geral")
    ticker = st.session_state.get("empresa_escolhida")
    if not ticker:
        st.info("Selecione uma empresa na Etapa 1 para continuar.")
        return

    # -------------------------------
    # Parâmetros e coleta
    # -------------------------------
    cols = st.columns([1, 1, 1.2])
    with cols[0]:
        period_prices = st.selectbox("Período de preços", ["1y", "2y", "5y"], index=1)
    with cols[1]:
        show_benchmark = st.toggle("Comparar com Ibovespa", value=True, help="Usa ^BVSP como benchmark.")
    with cols[2]:
        st.caption("Indicadores podem vir incompletos do Yahoo. Campos ausentes aparecem como NaN.")

    with st.spinner(f"Baixando dados de {ticker}…"):
        info, px, ibov_px = fetch_yf_info_and_prices(ticker, period_prices=period_prices)

    if (px is None) or px.empty:
        st.error("Não foi possível obter preços do ativo selecionado.")
        return

    # -------------------------------
    # Overview (info -> dataframe)
    # -------------------------------
    df_info = _build_overview_from_info(info)
    nome = df_info.at[0, "Empresa"] if not df_info.empty else ticker
    st.session_state["empresa_nome_completo"] = nome
    r = df_info.iloc[0] if not df_info.empty else pd.Series(dtype=float)

    # -------------------------------
    # Helpers visuais
    # -------------------------------
    def fmt_money_brl_bi(v):
        return "—" if (v is None or np.isnan(v)) else f"R$ {v/1e9:,.1f} bi"

    def fmt_num(v, d=2):
        return "—" if (v is None or np.isnan(v)) else f"{v:.{d}f}"

    def fmt_pct(v, d=1):
        return "—" if (v is None or np.isnan(v)) else f"{v:.{d}f}%"

    def box(label: str, value: str | float):
        # Um "card" compacto: título pequeno e valor grande
        with st.container(border=True):
            st.caption(label)
            st.markdown(f"<div style='font-size:1.8rem; font-weight:700; line-height:1'>{value}</div>", unsafe_allow_html=True)

    # ==========================================================
    # LINHA 1 — Valor da Empresa (2 boxes)
    # ==========================================================
    st.subheader("Valor da Empresa")
    c1, c2 = st.columns(2)
    with c1: box("Valor de Mercado", fmt_money_brl_bi(r.get("Market Cap")))
    with c2: box("Enterprise Value", fmt_money_brl_bi(r.get("Enterprise Value")))

    # ==========================================================
    # LINHA 2 — Análise de Mercado (4 boxes)
    # ==========================================================
    st.subheader("Análise de Mercado")
    c1, c2, c3, c4 = st.columns(4)
    with c1: box("P/L (Trailing)", fmt_num(r.get("P/L (Trailing)")))
    with c2: box("P/L (Forward)",  fmt_num(r.get("P/L (Forward)")))
    with c3: box("P/VP",           fmt_num(r.get("P/VP")))
    with c4: box("PEG",            fmt_num(r.get("PEG")))

    # ==========================================================
    # LINHA 3 — Análise de Mercado (cont.) (4 boxes)
    # ==========================================================
    st.subheader("Análise de Mercado (cont.)")
    c1, c2, c3, c4 = st.columns(4)
    with c1: box("EV/EBITDA",  fmt_num(r.get("EV/EBITDA")))
    with c2: box("EV/EBIT",    fmt_num(r.get("EV/EBIT (calc)")))
    with c3: box("EV/Revenue", fmt_num(r.get("EV/Revenue")))
    with c4: box("P/Sales",    fmt_num(r.get("P/Sales")))

    # ==========================================================
    # LINHA 4 — Rentabilidade e Risco (4 boxes)
    # ==========================================================
    st.subheader("Análise de Rentabilidade e Risco")
    c1, c2, c3, c4 = st.columns(4)
    with c1: box("ROE",            fmt_pct(r.get("ROE (%)")))
    with c2: box("ROA",            fmt_pct(r.get("ROA (%)")))
    with c3: box("Dividend Yield", fmt_pct(r.get("Dividend Yield (%)")))
    with c4: box("Beta",           fmt_num(r.get("Beta")))

    # ==========================================================
    # LINHA 5 — Eficiência (4 boxes)
    # ==========================================================
    st.subheader("Análise de Eficiência")
    c1, c2, c3, c4 = st.columns(4)
    with c1: box("Margem Bruta",    fmt_pct(r.get("Margem Bruta (%)")))
    with c2: box("Margem EBIT",     fmt_pct(r.get("Margem EBIT (%)")))
    with c3: box("Margem EBITDA",   fmt_pct(r.get("Margem EBITDA (%)")))
    with c4: box("Margem Líquida",  fmt_pct(r.get("Margem Líquida (%)")))

    # ==========================================================
    # LINHA 6 — Endividamento (2 boxes)
    # ==========================================================
    st.subheader("Endividamento")
    c1, c2 = st.columns(2)
    with c1: box("Dívida Líquida/PL",   fmt_num(r.get("Net Debt/Equity")))
    with c2: box("Dívida Líquida/Ativo", fmt_num(r.get("Net Debt/Assets")))

    # ==========================================================
    # LINHA 7 — Índices de Liquidez (4 boxes)
    # ==========================================================
    st.subheader("Índices de Liquidez")
    c1, c2, c3, c4 = st.columns(4)
    with c1: box("Liquidez Corrente", fmt_num(r.get("Liquidez Corrente")))
    with c2: box("Liquidez Seca",     fmt_num(r.get("Liquidez Seca")))
    with c3: box("Giro do Ativo",     fmt_num(r.get("Giro do Ativo")))
    with c4: box("Giro Estoque",      fmt_num(r.get("Giro de Estoque")))

    # ==========================================================
    # Métricas rápidas + Momentum (mantidos)
    # ==========================================================
    st.markdown("---")
    m1, m2 = st.columns(2)
    with m1: st.metric("P/L (Trailing)", fmt_num(r.get("P/L (Trailing)")))
    with m2: st.metric("ROE (%)",        fmt_pct(r.get("ROE (%)")))

    ret_1m  = _momentum_from_series(px, TRADING_DAYS["1M"])
    ret_3m  = _momentum_from_series(px, TRADING_DAYS["3M"])
    ret_6m  = _momentum_from_series(px, TRADING_DAYS["6M"])
    ret_12m = _momentum_from_series(px, TRADING_DAYS["12M"])

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Retorno 1M",  f"{ret_1m*100:,.1f}%"  if not np.isnan(ret_1m)  else "—")
    with m2: st.metric("Retorno 3M",  f"{ret_3m*100:,.1f}%" if not np.isnan(ret_3m)  else "—")
    with m3: st.metric("Retorno 6M",  f"{ret_6m*100:,.1f}%" if not np.isnan(ret_6m)  else "—")
    with m4: st.metric("Retorno 12M", f"{ret_12m*100:,.1f}%" if not np.isnan(ret_12m) else "—")

    st.markdown("---")

    # Gráfico de preço (com benchmark opcional, normalizado = 100)
    def _normalize_100(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        base = s.iloc[0]
        return s / base * 100.0

    base_df = pd.DataFrame({"Data": px.index, ticker: _normalize_100(px).values})
    if show_benchmark and ibov_px is not None and not ibov_px.empty:
        # Alinha datas
        _tmp = pd.DataFrame({"Data": ibov_px.index, "IBOV": _normalize_100(ibov_px).values})
        base_df = base_df.merge(_tmp, on="Data", how="left")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=base_df["Data"], y=base_df[ticker], mode="lines", name=ticker))
    if "IBOV" in base_df.columns:
        fig.add_trace(go.Scatter(x=base_df["Data"], y=base_df["IBOV"], mode="lines", name="IBOV (100=base)"))
    fig.update_layout(
        title=f"Evolução normalizada (100 = início) — {ticker}",
        xaxis_title="Data", yaxis_title="Índice (base 100)", height=420, legend_title="Séries"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabela de fundamentos (compacta)
    st.markdown("#### Indicadores do Yahoo Finance (info)")
    st.dataframe(
        df_info.T.rename(columns={0: "Valor"}),
        use_container_width=True,
        height=420
    )

    # Download CSV
    csv_bytes = df_info.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV (overview da empresa)", data=csv_bytes,
                       file_name=f"{ticker}_overview.csv", mime="text/csv")
    
    # ✅ Salva no estado para a Etapa 3
    st.session_state["empresa_info_df"] = df_info
    st.session_state["empresa_px"] = px

# ============================================================
# ETAPA 3 — Análise avançada (cards + comparativo setorial)
# ============================================================

# ——— Utilitários de normalização/score ———
def _minmax(x):
    # Aceita Series OU DataFrame
    if isinstance(x, pd.DataFrame):
        s = x.copy().astype(float)
        col_min = s.min(numeric_only=True)
        col_max = s.max(numeric_only=True)
        rng = col_max - col_min
        # evita divisão por zero: onde rng==0, devolve 0.5
        rng = rng.replace(0, np.nan)
        out = (s - col_min) / rng
        return out.fillna(0.5)
    else:
        s = pd.to_numeric(pd.Series(x), errors="coerce")
        if s.empty or s.max() == s.min():
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - s.min()) / (s.max() - s.min())

def _safe(v):
    try:
        return float(v)
    except Exception:
        return np.nan

def _score_value(df: pd.DataFrame) -> pd.Series:
    cols = ["P/L", "P/VP", "EV/EBITDA", "P/Sales"]
    tmp = df[cols].applymap(_safe)
    # normaliza coluna a coluna e inverte (menor = melhor)
    inv = 1 - _minmax(tmp)
    return inv.mean(axis=1)

def _score_profit(df: pd.DataFrame) -> pd.Series:
    cols = ["ROE (%)", "ROA (%)", "Margem Líquida (%)", "Margem Operacional (%)", "Margem EBITDA (%)"]
    tmp = df[cols].applymap(_safe)
    return _minmax(tmp).mean(axis=1)

def _score_strength(df: pd.DataFrame) -> pd.Series:
    cols_low  = ["Debt/Equity"]                  # menor melhor
    cols_high = ["Current Ratio", "Quick Ratio"] # maior melhor

    parts = []
    if all(c in df.columns for c in cols_low):
        tmp_low = df[cols_low].applymap(_safe)
        parts.append(1 - _minmax(tmp_low))  # inverte para "menor=melhor"
    if all(c in df.columns for c in cols_high):
        tmp_high = df[cols_high].applymap(_safe)
        parts.append(_minmax(tmp_high))

    if not parts:
        return pd.Series([np.nan] * len(df), index=df.index)

    return pd.concat(parts, axis=1).mean(axis=1)

def _score_momentum(px: pd.Series) -> dict:
    out = {}
    out["1M"]  = _momentum_from_series(px, TRADING_DAYS["1M"])
    out["3M"]  = _momentum_from_series(px, TRADING_DAYS["3M"])
    out["6M"]  = _momentum_from_series(px, TRADING_DAYS["6M"])
    out["12M"] = _momentum_from_series(px, TRADING_DAYS["12M"])
    return out

@st.cache_data(show_spinner=True)
def _fetch_peers_overview(tickers: list, period_prices: str = "2y"):
    """Coleta info de múltiplos para uma lista de tickers (sem preços)."""
    rows = []
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            info = t.info
            rows.append({
                "Ticker": tk,
                "P/L": _safe(info.get("trailingPE")),
                "P/VP": _safe(info.get("priceToBook")),
                "EV/EBITDA": _safe(info.get("enterpriseToEbitda")),
                "P/Sales": _safe(info.get("priceToSalesTrailing12Months")),
                "Dividend Yield (%)": _as_pct(info.get("dividendYield")),
                "ROE (%)": _as_pct(info.get("returnOnEquity")),
                "ROA (%)": _as_pct(info.get("returnOnAssets")),
                "Margem Líquida (%)": _as_pct(info.get("profitMargins")),
                "Margem Operacional (%)": _as_pct(info.get("operatingMargins")),
                "Margem EBITDA (%)": _as_pct(info.get("ebitdaMargins")),
                "Debt/Equity": _safe(info.get("debtToEquity")),
                "Current Ratio": _safe(info.get("currentRatio")),
                "Quick Ratio": _safe(info.get("quickRatio")),
                "Setor": info.get("sector"),
                "Empresa": info.get("longName"),
                "Market Cap (R$ bi)": (_safe(info.get("marketCap"))/1e9 if info.get("marketCap") else np.nan),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)
  
def etapa3_analise_avancada():
    st.markdown("### Análise Financeira")
    ticker = st.session_state.get("empresa_escolhida")
    if not ticker:
        st.info("Selecione e confirme uma empresa nas etapas anteriores.")
        return

    # — Parâmetros de análise —
    c1, c2 = st.columns([1,1])
    with c1:
        period_prices = st.selectbox("Período para momentum", ["1y", "2y", "5y"], index=1, key="e3_period")
    with c2:
        topN = st.slider("Máx. de pares (mesmo setor)", min_value=4, max_value=20, value=10, step=1,
                         help="Para não pesar a coleta, limitamos a quantidade de pares.")

    # — Dados da empresa (reutiliza cache da etapa 2 quando possível) —
    if "empresa_info_df" in st.session_state and "empresa_px" in st.session_state:
        df_info_self = st.session_state["empresa_info_df"].copy()
        px_self = st.session_state["empresa_px"].copy()
    else:
        info, px_self, _ = fetch_yf_info_and_prices(ticker, period_prices=period_prices)
        df_info_self = _build_overview_from_info(info)

    # Scores e momentum da empresa
    df_self = df_info_self.copy()
    mom = _score_momentum(px_self)
    df_self["Momentum 1M (%)"]  = mom["1M"]*100 if mom["1M"]==mom["1M"] else np.nan
    df_self["Momentum 3M (%)"]  = mom["3M"]*100 if mom["3M"]==mom["3M"] else np.nan
    df_self["Momentum 6M (%)"]  = mom["6M"]*100 if mom["6M"]==mom["6M"] else np.nan
    df_self["Momentum 12M (%)"] = mom["12M"]*100 if mom["12M"]==mom["12M"] else np.nan

    # — Encontrar pares (mesmo setor) a partir do Excel —
    df_class, msg = load_classif_setorial()
    if msg:
        st.warning(msg + " — Mostrando apenas a empresa, sem comparativos de setor.")
        peers_list = []  # segue sem pares
    else:
        mserie = re.search(r"(\d{1,2})\.SA$", ticker)
        serie_sel = mserie.group(1) if mserie else "3"

    # Monta TickerFinal para o Excel (usa TICKER se existir; senão CÓDIGO + a MESMA série da escolhida)
    def _build_tickerfinal(row):
        tk = str(row.get("Ticker", "")).strip()
        if tk:
            return _normalize_ticker(tk)
        cod = str(row.get("Codigo", "")).strip().upper()
        return _normalize_ticker(f"{cod}{serie_sel}.SA") if cod else ""

    df_class["TickerFinal"] = df_class.apply(_build_tickerfinal, axis=1)

    # Descobre o setor pelo Excel (garante o mesmo idioma/estrutura da sua planilha)
    linha = df_class[df_class["TickerFinal"].str.upper() == ticker.upper()]
    setor_self = linha["Setor"].iloc[0] if not linha.empty else None

    # Lista de pares do mesmo setor (do Excel)
    if setor_self:
        df_sector = df_class[df_class["Setor"] == setor_self].copy()
        peers_list = [t for t in df_sector["TickerFinal"].dropna().unique().tolist() if isinstance(t, str)]
        peers_list = [t for t in peers_list if t != ticker][:topN]  # remove o próprio e limita
    else:
        peers_list = []

    # Diagnóstico visual
    with st.expander("🔍 Diagnóstico dos pares", expanded=False):
        st.write("Setor (Excel):", setor_self)
        st.write("Qtd. pares:", len(peers_list))
        st.write(peers_list[:20])

    # — Coleta dos pares —
    df_peers = _fetch_peers_overview(peers_list, period_prices=period_prices) if peers_list else pd.DataFrame()

    # — Consolidação (empresa + pares) —
    df_all = pd.concat([
        df_self.assign(Ticker=ticker)[[
            "Ticker","Empresa","Setor","P/L","P/VP","EV/EBITDA","P/Sales",
            "Dividend Yield (%)","ROE (%)","ROA (%)","Margem Líquida (%)",
            "Margem Operacional (%)","Margem EBITDA (%)","Debt/Equity",
            "Current Ratio","Quick Ratio","Market Cap (R$ bi)",
            "Momentum 1M (%)","Momentum 3M (%)","Momentum 6M (%)","Momentum 12M (%)",
        ]],
        df_peers.reindex(columns=[
            "Ticker","Empresa","Setor","P/L","P/VP","EV/EBITDA","P/Sales",
            "Dividend Yield (%)","ROE (%)","ROA (%)","Margem Líquida (%)",
            "Margem Operacional (%)","Margem EBITDA (%)","Debt/Equity",
            "Current Ratio","Quick Ratio","Market Cap (R$ bi)"
        ])
    ], ignore_index=True)

    # — Cálculo de scores —
    df_scores = df_all.copy()
    df_scores["Score Value"]   = _score_value(df_scores)
    df_scores["Score Profit"]  = _score_profit(df_scores)
    df_scores["Score Strength"]= _score_strength(df_scores)
    # Momentum composto (média dos disponíveis)
    mom_cols = ["Momentum 1M (%)","Momentum 3M (%)","Momentum 6M (%)","Momentum 12M (%)"]
    df_scores["Score Momentum"] = _minmax(df_scores[mom_cols].applymap(_safe)).mean(axis=1)

    # Score geral (pesos ajustáveis – por enquanto fixos; depois podemos expor sliders)
    W_VALUE, W_PROFIT, W_STRENGTH, W_MOM = 0.30, 0.30, 0.20, 0.20
    df_scores["Score Total"] = (
        W_VALUE*df_scores["Score Value"] +
        W_PROFIT*df_scores["Score Profit"] +
        W_STRENGTH*df_scores["Score Strength"] +
        W_MOM*df_scores["Score Momentum"]
    )

    # — Cards da empresa (destaques) —
    st.markdown("#### 🧾 Destaques (empresa selecionada)")
    c1, c2, c3, c4 = st.columns(4)
    sel = df_scores[df_scores["Ticker"] == ticker]
    if sel.empty:
        st.warning("Não encontrei a linha da empresa nos scores. Mostrando apenas a tabela.")
        st.dataframe(df_scores.sort_values("Score Total", ascending=False).reset_index(drop=True),
                 use_container_width=True, height=420)
        return
    row_self = sel.iloc[0]
    with c1: st.metric("Score Value",     f"{row_self['Score Value']*100:,.0f} / 100")
    with c2: st.metric("Score Profit",    f"{row_self['Score Profit']*100:,.0f} / 100")
    with c3: st.metric("Score Strength",  f"{row_self['Score Strength']*100:,.0f} / 100")
    with c4: st.metric("Score Momentum",  f"{row_self['Score Momentum']*100:,.0f} / 100")

    # --- RADAR: Scores (empresa x mediana do setor) ---
    st.markdown("#### 🕸️ Radar de Scores (0–100) — empresa vs. setor")

    score_cols = ["Score Value", "Score Profit", "Score Strength", "Score Momentum"]

    # seleciona linha da empresa
    sel = df_scores[df_scores["Ticker"] == ticker]
    if sel.empty:
        st.info("Não foi possível localizar os scores da empresa selecionada.")
    else:
        row_self = sel.iloc[0]

        # mediana do setor (usa todas as linhas exceto NaN)
        med_setor = df_scores[score_cols].median(numeric_only=True)

        # prepara vetores (0–100)
        labels = ["Value", "Profit", "Strength", "Momentum"]
        emp_vals = [(row_self[c] * 100) if pd.notna(row_self[c]) else np.nan for c in score_cols]
        setor_vals = [(med_setor[c] * 100) if pd.notna(med_setor[c]) else np.nan for c in score_cols]

        # remove categorias completamente ausentes (ambos NaN)
        mask = [not (np.isnan(e) and np.isnan(s)) for e, s in zip(emp_vals, setor_vals)]
        labels = [l for l, m in zip(labels, mask) if m]
        emp_vals = [v for v, m in zip(emp_vals, mask) if m]
        setor_vals = [v for v, m in zip(setor_vals, mask) if m]

        if not labels:
            st.info("Sem dados suficientes para montar o radar de scores.")
        else:
            import plotly.graph_objects as go
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=emp_vals, theta=labels, fill="toself", name=ticker
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=setor_vals, theta=labels, fill="toself", name="Setor (mediana)"
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Radar de Scores (0–100)"
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # — Gráficos —
    st.markdown("#### 📊 Comparativos do setor")
    g1, g2 = st.columns(2)

    # (a) Dispersão P/L × ROE com tamanho por Market Cap
    with g1:
        if not df_scores.empty:
            # Saneamento para o Plotly
            df_plot = df_scores.copy()
            for col in ["P/L", "ROE (%)", "Market Cap (R$ bi)"]:
                df_plot[col] = pd.to_numeric(df_plot[col], errors="coerce")

            # remove linhas sem X ou Y válidos
            df_plot = df_plot.dropna(subset=["P/L", "ROE (%)"]).copy()

            if df_plot.empty:
                st.info("Sem dados numéricos suficientes para P/L e ROE dos pares.")
            else:
                # tamanho (não pode ter NaN/negativo/inf)
                sz = df_plot["Market Cap (R$ bi)"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                sz = sz.clip(lower=0.0) + 0.1  # evita zero puro

                fig_sc = px.scatter(
                    df_plot,
                    x="P/L",
                    y="ROE (%)",
                    color="Ticker",
                    size=sz,                       # ← usa série saneada
                    hover_name="Empresa",
                    title="P/L × ROE (bolha = Market Cap)"
                )

                # destaca a empresa selecionada (se houver ponto válido)
                try:
                    row_self = df_plot[df_plot["Ticker"] == ticker].iloc[0]
                    x_self = float(row_self["P/L"])
                    y_self = float(row_self["ROE (%)"])
                    if np.isfinite(x_self) and np.isfinite(y_self):
                        fig_sc.add_scatter(
                            x=[x_self], y=[y_self],
                            mode="markers+text", text=[ticker], textposition="top center",
                            marker=dict(size=14, symbol="star")
                        )
                except Exception:
                    pass

                st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Sem pares (setor não encontrado ou sem tickers válidos).")

    # (b) Barras margens (empresa vs. mediana do setor)
    with g2:
        if not df_scores.empty:
            cols_marg = ["Margem Líquida (%)","Margem Operacional (%)","Margem EBITDA (%)"]
            med_setor = df_scores[cols_marg].median(numeric_only=True)
            bar_df = pd.DataFrame({
                "Indicador": cols_marg,
                "Empresa": [row_self[c] for c in cols_marg],
                "Setor (mediana)": [med_setor[c] for c in cols_marg]
            })
            fig_bar = go.Figure()
            fig_bar.add_bar(x=bar_df["Indicador"], y=bar_df["Empresa"], name=ticker)
            fig_bar.add_bar(x=bar_df["Indicador"], y=bar_df["Setor (mediana)"], name="Setor (mediana)")
            fig_bar.update_layout(barmode="group", title="Margens: empresa vs. setor (mediana)")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Sem dados suficientes para margens do setor.")

    h1, h2 = st.columns(2)

    # (c) Barras: ROE e ROA — empresa vs mediana do setor
    with h1:
        if not df_scores.empty:
            # garante numérico
            df_tmp = df_scores.copy()
            for col in ["ROE (%)", "ROA (%)"]:
                df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce")

            med_setor = df_tmp[["ROE (%)", "ROA (%)"]].median(numeric_only=True)
            row_plot = df_tmp[df_tmp["Ticker"] == ticker].iloc[0]

            bar_df = pd.DataFrame({
                "Indicador": ["ROE (%)", "ROA (%)"],
                "Empresa": [row_plot["ROE (%)"], row_plot["ROA (%)"]],
                "Setor (mediana)": [med_setor["ROE (%)"], med_setor["ROA (%)"]],
            })

            fig_bar_rent = go.Figure()
            fig_bar_rent.add_bar(x=bar_df["Indicador"], y=bar_df["Empresa"], name=ticker)
            fig_bar_rent.add_bar(x=bar_df["Indicador"], y=bar_df["Setor (mediana)"], name="Setor (mediana)")
            fig_bar_rent.update_layout(barmode="group", title="Rentabilidades: empresa vs. setor (mediana)",
                                       yaxis_title="%")
            st.plotly_chart(fig_bar_rent, use_container_width=True)
        else:
            st.info("Sem dados suficientes para ROE/ROA.")

    # (d) Barras: Liquidez — Current Ratio e Quick Ratio (empresa vs mediana do setor)
    with h2:
        if not df_scores.empty:
            df_tmp = df_scores.copy()
            for col in ["Current Ratio", "Quick Ratio"]:
                df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce")

            med_setor = df_tmp[["Current Ratio", "Quick Ratio"]].median(numeric_only=True)
            row_plot = df_tmp[df_tmp["Ticker"] == ticker].iloc[0]

            bar_df = pd.DataFrame({
                "Indicador": ["Current Ratio", "Quick Ratio"],
                "Empresa": [row_plot["Current Ratio"], row_plot["Quick Ratio"]],
                "Setor (mediana)": [med_setor["Current Ratio"], med_setor["Quick Ratio"]],
            })

            fig_bar_liq = go.Figure()
            fig_bar_liq.add_bar(x=bar_df["Indicador"], y=bar_df["Empresa"], name=ticker)
            fig_bar_liq.add_bar(x=bar_df["Indicador"], y=bar_df["Setor (mediana)"], name="Setor (mediana)")
            fig_bar_liq.update_layout(barmode="group", title="Liquidez: empresa vs. setor (mediana)")
            st.plotly_chart(fig_bar_liq, use_container_width=True)
        else:
            st.info("Sem dados suficientes para liquidez.")    

    @st.cache_data(show_spinner=True)
    
    def fetch_prices_multi(tickers: list, period: str = "2y"):
    
        """Baixa preços ajustados de vários tickers e retorna um DF de Close."""
        if not tickers:
            return pd.DataFrame()
        try:
            df = yf.download(
                tickers=tickers,
                period=period, interval="1d",
                auto_adjust=True, group_by="ticker",
                threads=False, progress=False,
            )
            # extrai a coluna Close em qualquer formato que vier
            if isinstance(df.columns, pd.MultiIndex):
                close = df.xs("Close", axis=1, level=1, drop_level=False).copy()
                # rearranja para colunas simples com os tickers
                close = close.droplevel(1, axis=1)
            else:
                # único ticker
                close = pd.DataFrame({tickers[0]: df["Close"]})
            close = close.dropna(how="all")
            return close
        except Exception:
            return pd.DataFrame()

    def _normalize_base100(col: pd.Series) -> pd.Series:
        """Normaliza cada série individualmente para 100 no primeiro ponto válido."""
        if col is None or col.dropna().empty:
            return col
        base = col.dropna().iloc[0]
        return (col / base) * 100.0

        # ====== Evolução de Preços (base 100) — empresa + pares ======
    st.markdown("#### 📈 Preço normalizado (base 100) — empresa e pares")

    # recupera a lista de pares que você montou acima na mesma função
    # (se preferir, salve peers_list no session_state quando montar)
    peers_for_chart = []
    try:
        peers_for_chart = peers_list.copy()
    except NameError:
        peers_for_chart = []

    # monta universo: empresa + pares (sem duplicados)
    univ = [ticker] + [t for t in peers_for_chart if t != ticker]
    univ = list(dict.fromkeys([t for t in univ if isinstance(t, str) and t]))

    cols = st.columns([1,1,1])
    with cols[0]:
        per = st.selectbox("Período do gráfico", ["6mo", "1y", "2y", "5y"], index=2, key="prices_base100_period")
    with cols[1]:
        show_legend = st.toggle("Mostrar legenda completa", value=False)
    with cols[2]:
        st.caption("Séries ajustadas e normalizadas para 100 no 1º ponto válido.")

    # coleta e plota
    prices = fetch_prices_multi(univ, period=per)
    if prices.empty:
        st.info("Sem dados de preço para o universo selecionado.")
    else:
        # normaliza cada coluna individualmente
        base100 = prices.apply(_normalize_base100)
        base100 = base100.dropna(how="all")

        fig_norm = go.Figure()
        for col in base100.columns:
            # destaque para o ticker selecionado
            is_sel = (col == ticker)
            fig_norm.add_trace(go.Scatter(
                x=base100.index, y=base100[col],
                mode="lines", name=col,
                line=dict(width=3 if is_sel else 1.5),
                opacity=1.0 if is_sel else 0.8
            ))
        fig_norm.update_layout(
            height=420,
            title="Evolução do preço (100 = início de cada série)",
            xaxis_title="Data", yaxis_title="Índice (base 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0) if show_legend else dict()
        )
        st.plotly_chart(fig_norm, use_container_width=True)


    
    st.markdown("#### 📋 Tabela (empresa + pares)")
    st.dataframe(
        df_scores.sort_values("Score Total", ascending=False).reset_index(drop=True),
        use_container_width=True, height=420
    )
    
    # Export
    st.download_button(
        "⬇️ Baixar CSV (empresa + pares + scores)",
        data=df_scores.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_pares_scores.csv",
        mime="text/csv"
    )

    # Guarda em sessão para eventuais próximas etapas
    st.session_state["etapa3_df_scores"] = df_scores

# ============================================================
# ETAPA 4 — Valuation (Target Price por múltiplos e Ben Graham)
# ============================================================

def etapa4_valuation():
    st.markdown("### Valuation (Target Price)")

    ticker = st.session_state.get("empresa_escolhida")
    if not ticker:
        st.info("Selecione e confirme uma empresa nas etapas anteriores.")
        return

    # -------------------------
    # Parâmetros da avaliação
    # -------------------------
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        lookback = st.selectbox("Janela p/ histórico de preço", ["2y", "5y"], index=1,
                                help="Usada p/ calcular P/L e P/VP históricos (média ao longo do período).")
    with c2:
        g_pct = st.number_input("g (crescimento anual, %)", value=5.0, step=0.5, min_value=-10.0, max_value=50.0,
                                help="Usado na fórmula de Ben Graham: EPS × (8,5 + 2g) × (4,4/Y)")
    with c3:
        y_pct = st.number_input("Y (taxa livre de risco, %)", value=10.0, step=0.5, min_value=1.0, max_value=30.0,
                                help="Usado na fórmula de Ben Graham; 4,4 é o yield base da fórmula.")

    # -------------------------
    # Coleta básica
    # -------------------------
    # Reuso da Etapa 2 quando possível
    info_df = st.session_state.get("empresa_info_df")
    px_2 = st.session_state.get("empresa_px")

    # Sempre buscamos uma série longa p/ histórico de múltiplos
    try:
        t = yf.Ticker(ticker)
        info = t.info if info_df is None else None  # se já temos df_info, usamos abaixo
        hist_long = t.history(period=lookback, interval="1d")
        px_long = hist_long["Close"].dropna() if "Close" in hist_long else pd.Series(dtype=float)
    except Exception:
        info = {}
        px_long = pd.Series(dtype=float)

    # Último preço (referência)
    try:
        p_now = float(px_long.iloc[-1] if not px_long.empty else (px_2.iloc[-1] if px_2 is not None and not px_2.empty else np.nan))
    except Exception:
        p_now = np.nan

    # EPS (trailing) e BVPS (book per share) — tentativas com fallback
    def _num(v): 
        try: return float(v)
        except: return np.nan

    if info_df is not None and not info_df.empty and "P/L" in info_df.columns and "P/VP" in info_df.columns:
        # Podemos inferir EPS/BVPS por: EPS ≈ Price / (P/L), BVPS ≈ Price / (P/VP)
        pl_now = _num(info_df.at[0, "P/L"])
        pb_now = _num(info_df.at[0, "P/VP"])
    else:
        pl_now = _num((info or {}).get("trailingPE"))
        pb_now = _num((info or {}).get("priceToBook"))

    # Se priceToBook e trailingPE vieram vazios, tentamos ler 'bookValue' e 'trailingEps'
    eps_ttm = _num((info or {}).get("trailingEps"))
    bvps = _num((info or {}).get("bookValue"))

    # Fallbacks a partir de múltiplos + preço atual
    if (np.isnan(eps_ttm) or eps_ttm == 0) and (not np.isnan(pl_now)) and (pl_now > 0) and (not np.isnan(p_now)):
        eps_ttm = p_now / pl_now
    if (np.isnan(bvps) or bvps == 0) and (not np.isnan(pb_now)) and (pb_now > 0) and (not np.isnan(p_now)):
        bvps = p_now / pb_now

    # Segurança: evita divisões por zero/negativas
    eps_ttm = eps_ttm if (not np.isnan(eps_ttm) and eps_ttm > 0) else np.nan
    bvps    = bvps    if (not np.isnan(bvps) and bvps > 0) else np.nan

    # -------------------------
    # P/L e P/VP HISTÓRICOS (médias)
    # -------------------------
    # Observação: usamos EPS e BVPS atuais como aproximadores para derivar uma "média histórica" de múltiplos:
    # PE_hist_avg ≈ média(Preço(t)/EPS_atual) ao longo da janela; idem para PB utilizando BVPS_atual.
    if px_long is not None and not px_long.empty:
        pe_hist_series = px_long / eps_ttm if (eps_ttm and not np.isnan(eps_ttm) and eps_ttm > 0) else pd.Series(dtype=float)
        pb_hist_series = px_long / bvps    if (bvps    and not np.isnan(bvps)    and bvps    > 0) else pd.Series(dtype=float)

        pe_hist_avg = float(np.nanmean(pe_hist_series)) if not pe_hist_series.empty else np.nan
        pb_hist_avg = float(np.nanmean(pb_hist_series)) if not pb_hist_series.empty else np.nan
    else:
        pe_hist_avg, pb_hist_avg = np.nan, np.nan

    # -------------------------
    # P/L e P/VP — MEDIANA DO SETOR
    # -------------------------
    df_scores = st.session_state.get("etapa3_df_scores")
    if df_scores is None or df_scores.empty:
        pl_med_setor, pb_med_setor = np.nan, np.nan
    else:
        dfp = df_scores.copy()
        dfp["P/L"]  = pd.to_numeric(dfp["P/L"], errors="coerce")
        dfp["P/VP"] = pd.to_numeric(dfp["P/VP"], errors="coerce")
        pl_med_setor = float(dfp["P/L"].median(skipna=True))
        pb_med_setor = float(dfp["P/VP"].median(skipna=True))

    # -------------------------
    # Targets por múltiplos
    # -------------------------
    targets = []

    def add_target(label, price):
        if np.isnan(price) or price <= 0 or np.isnan(p_now):
            up = np.nan
            verdict = "—"
        else:
            up = (price / p_now) - 1.0
            verdict = "Desconto (↑)" if up > 0 else "Prêmio (↓)"
        targets.append({"Método": label, "Target": price, "Upside (%)": (up * 100.0) if not np.isnan(up) else np.nan, "Veredito": verdict})

    # P/L histórico
    if not np.isnan(eps_ttm) and not np.isnan(pe_hist_avg) and pe_hist_avg > 0:
        add_target("P/L histórico", eps_ttm * pe_hist_avg)
    else:
        add_target("P/L histórico", np.nan)

    # P/L mediana do setor
    if not np.isnan(eps_ttm) and not np.isnan(pl_med_setor) and pl_med_setor > 0:
        add_target("P/L mediana do setor", eps_ttm * pl_med_setor)
    else:
        add_target("P/L mediana do setor", np.nan)

    # P/VP histórico
    if not np.isnan(bvps) and not np.isnan(pb_hist_avg) and pb_hist_avg > 0:
        add_target("P/VP histórico", bvps * pb_hist_avg)
    else:
        add_target("P/VP histórico", np.nan)

    # P/VP mediana do setor
    if not np.isnan(bvps) and not np.isnan(pb_med_setor) and pb_med_setor > 0:
        add_target("P/VP mediana do setor", bvps * pb_med_setor)
    else:
        add_target("P/VP mediana do setor", np.nan)

    # -------------------------
    # Ben Graham
    # P_graham = EPS × (8,5 + 2g) × (4,4 / Y)
    # onde g e Y são percentuais (ex.: g=5% → 5; Y=10% → 10)
    # -------------------------
    if not np.isnan(eps_ttm):
        g = float(g_pct)
        Y = float(y_pct)
        if Y <= 0: Y = 10.0
        p_graham = eps_ttm * (8.5 + 2 * g) * (4.4 / Y)
        add_target("Ben Graham", p_graham)
    else:
        add_target("Ben Graham", np.nan)

    df_targets = pd.DataFrame(targets)

    # -------------------------
    # Ben Graham (clássica simplificada)
    # VI = 22.5 × LPA × VPA
    # -------------------------
    if not np.isnan(eps_ttm) and not np.isnan(bvps):
        vi_graham = 22.5 * eps_ttm * bvps
        add_target("Ben Graham (22,5×LPA×VPA)", vi_graham)
    else:
        add_target("Ben Graham (22,5×LPA×VPA)", np.nan)

    if not np.isnan(eps_ttm) and not np.isnan(bvps):
        vi_graham_simplificada = 22.5 * eps_ttm * bvps
        add_target("Ben Graham (22,5×LPA×VPA)", vi_graham_simplificada)

    if not np.isnan(eps_ttm):
        g = float(g_pct)
        Y = float(y_pct)
        if Y <= 0: Y = 10.0
        vi_graham_ajustada = eps_ttm * (8.5 + 2 * g) * (4.4 / Y)
        add_target("Ben Graham ajustada (8,5+2g)", vi_graham_ajustada)

    # -------------------------
    # Exibição
    # -------------------------
    k1, k2, k3 = st.columns(3)
    with k1: st.metric("Preço atual", f"{p_now:,.2f}" if not np.isnan(p_now) else "—")
    with k2: st.metric("Lucro por Ação (LPA)",   f"{eps_ttm:,.2f}" if not np.isnan(eps_ttm) else "—")
    with k3: st.metric("Valor Patrimonial por Ação (VPA)",        f"{bvps:,.2f}" if not np.isnan(bvps) else "—")

    st.markdown("#### 🎯 Preços-alvo")
    st.dataframe(df_targets, use_container_width=True, height=260)

    # Gráfico: barras horizontais Target vs Preço Atual
    if not df_targets.empty and not np.isnan(p_now):
        plot_df = df_targets.copy()
        plot_df["Target"] = pd.to_numeric(plot_df["Target"], errors="coerce")
        plot_df = plot_df.dropna(subset=["Target"])
        if not plot_df.empty:
            fig_tp = go.Figure()
            fig_tp.add_bar(y=plot_df["Método"], x=plot_df["Target"], orientation="h", name="Target")
            fig_tp.add_vline(x=p_now, line_dash="dash", annotation_text=f"Preço atual: {p_now:,.2f}",
                             annotation_position="top right")
            fig_tp.update_layout(height=420, title="Targets por metodologia (linha tracejada = preço atual)",
                                 xaxis_title="Preço", yaxis_title="")
            st.plotly_chart(fig_tp, use_container_width=True)

    # Export
    st.download_button(
        "⬇️ Baixar CSV (targets de valuation)",
        data=df_targets.to_csv(index=False).encode("utf-8"),
        file_name=f"{ticker}_valuation_targets.csv",
        mime="text/csv"
    )

    st.caption("Notas: P/L e P/VP históricos são aproximados usando EPS/BVPS atuais como denominadores sobre a série de preços "
               f"({lookback}). A fórmula de Ben Graham usa g={g_pct:.1f}% e Y={y_pct:.1f}%. Ajuste conforme seu cenário.")

def render_company_header():
    titulo = st.session_state.get("empresa_nome", st.session_state.get("empresa_escolhida", "—"))
    nome_completo = st.session_state.get("empresa_nome_completo") or titulo

    setor    = (st.session_state.get("empresa_setor") or "").upper()
    subsetor = (st.session_state.get("empresa_subsetor") or "").upper()
    segmento = (st.session_state.get("empresa_segmento") or "").upper()

    st.markdown(f"""
    <div style="margin-top:-2rem; margin-bottom:1.2rem;">
      <h1 style="margin:0; font-size:3rem; font-weight:800; letter-spacing:.5px;">
        {titulo.upper()}
      </h1>
      <h3 style="margin:.3rem 0 .8rem 0; font-size:1.3rem; font-weight:400; color:#cbd5e1;">
        {nome_completo}
      </h3>
      <div style="display:flex; flex-wrap:wrap; gap:.5rem; margin-top:.25rem;">
        {f'<span class="badge badge-blue">{setor}</span>' if setor else ''}
        {f'<span class="badge badge-blue">{subsetor}</span>' if subsetor else ''}
        {f'<span class="badge badge-blue">{segmento}</span>' if segmento else ''}
      </div>
    </div>
    """, unsafe_allow_html=True)

def render_single_with_tabs():
    render_company_header()

    tab1, tab2, tab3 = st.tabs([
        "📊 Análise Financeira",
        "📈 Comparativo do Setor",
        "💰 Valuation (Target Price)"
    ])

    with tab1:
        etapa2_coleta_dados()

    with tab2:
        etapa3_analise_avancada()

    with tab3:
        etapa4_valuation()


def render_single_layout():
    sel = render_sidebar_selector()   # seleção na sidebar
    if sel and sel.get("ticker"):
        render_single_with_tabs()
    else:
        st.info("Escolha uma empresa na barra lateral para iniciar a análise.")

def render_screener_layout():
    st.subheader("📈 Screener / Ranking — layout")
    st.caption("Esqueleto visual para filtros, pesos e ranking. Próxima etapa: dados.")

    f1, f2, f3 = st.columns([1.2,1,1])
    with f1:
        st.multiselect("Setores", ["Energia","Financeiro","Materiais Básicos","Consumo"], help="Carregados do Excel quando integrarmos")
    with f2:
        st.slider("Peso: Value", 0.0, 1.0, 0.25, 0.05)
    with f3:
        st.slider("Peso: Quality", 0.0, 1.0, 0.25, 0.05)

    f4, f5 = st.columns(2)
    with f4:
        st.slider("Peso: Momentum", 0.0, 1.0, 0.25, 0.05)
    with f5:
        st.slider("Peso: Crescimento", 0.0, 1.0, 0.25, 0.05)

    st.container(border=True).markdown("**Tabela placeholder** — Ranking com colunas essenciais (Ticker, Setor, P/L, P/VP, EV/EBITDA, ROE, Momentum, Score)")
    st.container(height=6)
    st.container(border=True).markdown("**Gráfico placeholder** — Dispersão P/L × ROE (bolhas por Market Cap)")

# ------------------------------
# Roteamento simples por modo
# ------------------------------
if st.session_state["mode"] == "home":
    render_home()
elif st.session_state["mode"] == "single":
    render_single_layout()
elif st.session_state["mode"] == "screener":
    render_screener_layout()

# Rodapé
st.markdown("""
<hr class='soft'/>
<small style='opacity:.7'>Elaborado pelo Prof. Luiz Eduardo Gaio (UNICAMP) para fins educacionais.</small>
""", unsafe_allow_html=True)


