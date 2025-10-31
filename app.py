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

# ------------------------------
# Configuração básica da página
# ------------------------------
st.set_page_config(
    page_title="Dashboard B3 — Valuation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
  background: rgba(255,255,255,0.65);
}
[data-theme="dark"] .card { background: rgba(0,0,0,0.25); border-color: rgba(255,255,255,0.08); }
.card:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(0,0,0,0.08); }
.card h3 { margin: 0 0 8px 0; }
.btn {
  display:inline-block; padding:10px 14px; border-radius:12px; 
  text-decoration:none; font-weight:600; border:1px solid transparent;
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

# ------------------------------
# Cabeçalho
# ------------------------------
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("<div style='font-size:44px'>📊</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0'>Dashboard B3 — Valuation</h1>
    <p style='margin-top:6px;opacity:.8'>Escolha o modo de análise: <b>Individual</b> (uma empresa) ou <b>Screener</b> (ranking de várias).
    </p>
    """, unsafe_allow_html=True)

# ------------------------------
# Sidebar de navegação
# ------------------------------
with st.sidebar:
    st.markdown("### Navegação")
    sel = st.radio("", [MODES[m] for m in ("home","single","screener")], index=(0 if st.session_state["mode"]=="home" else 1 if st.session_state["mode"]=="single" else 2))
    # converte label → chave
    rev = {v:k for k,v in MODES.items()}
    set_mode(rev[sel])

    st.markdown("---")
    st.markdown("**Atalhos**")
    cols = st.columns(2)
    if cols[0].button("🏠 Início", use_container_width=True):
        set_mode("home")
    if cols[1].button("🔄 Limpar sessão", use_container_width=True):
        st.session_state.clear()
        set_mode("home")

# ------------------------------
# Páginas
# ------------------------------

def render_home():
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class='card'>
              <span class='badge badge-blue'>Modo 1</span>
              <h3>🔎 Análise Individual</h3>
              <p>Estude profundamente uma empresa: múltiplos, rentabilidade, endividamento, histórico de preços e comparativos de setor.</p>
              <a class='btn btn-primary' href='?""" + urlencode({"mode":"single"}) + """'>Começar</a>
              <a class='btn btn-ghost' href='?""" + urlencode({"mode":"single"}) + """'>Ver layout</a>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class='card'>
              <span class='badge badge-green'>Modo 2</span>
              <h3>📈 Screener / Ranking</h3>
              <p>Monte um ranking de empresas por múltiplos e qualidade. Filtros por setor, pesos customizados e exportação para CSV.</p>
              <a class='btn btn-primary' href='?""" + urlencode({"mode":"screener"}) + """'>Começar</a>
              <a class='btn btn-ghost' href='?""" + urlencode({"mode":"screener"}) + """'>Ver layout</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("""
    <hr class='soft'/>
    <small style='opacity:.7'>Versão layout • Próximas etapas: conexão com dados (yfinance), seleção por setor (ClassifSetorial.xlsx), 
    cálculo de scores e gráficos interativos.</small>
    """, unsafe_allow_html=True)


def render_single_layout():
    st.subheader("🔎 Análise Individual — Etapa 1: Seleção da Empresa")
    etapa1_selecao_empresa()  # <-- chama a etapa 1 aqui

# ============================================================
# MODO: ANÁLISE INDIVIDUAL
# ETAPA 1 — Seleção da Empresa (por lista OU por setor→subsetor→segmento)
# ============================================================

import re
import pandas as pd
import streamlit as st
from typing import Tuple, Dict

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


def etapa1_selecao_empresa():
    st.markdown("### Etapa 1 — Seleção da Empresa")

    # 1) Carrega base setorial
    df_class, msg = load_classif_setorial()
    if msg:
        st.warning(msg)
        st.stop()

    # 2) Série padrão (somente usada se Ticker não existir na planilha)
    col_ser = st.columns([1, 3])[0]
    serie_padrao = col_ser.selectbox("Série padrão quando faltar Ticker", ["3", "4", "11"], index=0,
                                     help="Usada para montar PETR + 4 → PETR4.SA quando a planilha só tiver 'CÓDIGO'.")

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
            st.toast("Empresa selecionada com sucesso!", icon="✅")
    else:
        st.info("Selecione uma empresa para continuar.")

    # Rodapé da etapa
    if "empresa_escolhida" in st.session_state:
        st.caption(f"**Estado atual**: {st.session_state['empresa_escolhida']} — {st.session_state.get('empresa_nome','')}")
    else:
        st.caption("**Estado atual**: nenhuma empresa confirmada.")


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



