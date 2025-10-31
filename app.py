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
    st.subheader("🔎 Análise Individual — layout")
    st.caption("Nesta etapa temos só o esqueleto visual. Em seguida plugaremos os dados.")

    tcol = st.columns([1,1,1])
    with tcol[0]:
        st.selectbox("Empresa (ticker)", ["Selecione...", "PETR4.SA", "VALE3.SA", "ITUB4.SA"], index=0)
    with tcol[1]:
        st.selectbox("Período de preços", ["1y","2y","5y"], index=1)
    with tcol[2]:
        st.multiselect("Comparar com (opcional)", ["VALE3.SA","ITUB4.SA","WEGE3.SA"])

    g1, g2, g3 = st.columns(3)
    with g1: st.container(border=True).markdown("**Card** — Múltiplos (P/L, P/VP, EV/EBITDA, DY)")
    with g2: st.container(border=True).markdown("**Card** — Rentabilidade (ROE, ROA, Margens)")
    with g3: st.container(border=True).markdown("**Card** — Endividamento e Liquidez")

    st.container(height=10)
    st.container(border=True).markdown("**Gráfico placeholder** — Preço vs. Benchmark, retorno acumulado")
    st.container(height=6)
    st.container(border=True).markdown("**Tabela placeholder** — Histórico resumido de indicadores")


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


