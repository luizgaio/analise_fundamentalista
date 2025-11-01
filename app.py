# =============================================================
# Dashboard B3 — Layout moderno com design inspirado na imagem
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
import re
import pandas as pd
import streamlit as st
from typing import Tuple, Dict
from datetime import datetime, timedelta

# ====== CONFIGURAÇÃO INICIAL ======
st.set_page_config(
    page_title="Stock Peer Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====== NOVO STYLE - DESIGN MODERNO ======
def inject_modern_styles():
    st.markdown("""
    <style>
    /* ---------- TEMA PRINCIPAL ---------- */
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
        --bg-card: #334155;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-purple: #8b5cf6;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border: #475569;
    }
    
    /* ---------- FUNDO GERAL ---------- */
    .stApp, .main, .block-container {
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    /* ---------- HEADER MODERNO ---------- */
    .main-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%);
        padding: 2rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 2rem;
    }
    
    .header-title {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    
    .header-subtitle {
        color: var(--text-secondary) !important;
        font-size: 1.1rem !important;
        margin-bottom: 0 !important;
    }
    
    /* ---------- CARDS MODERNOS ---------- */
    .modern-card {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.3s ease !important;
        height: 100% !important;
    }
    
    .modern-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 12px 28px rgba(0,0,0,0.3) !important;
        border-color: var(--accent-blue) !important;
    }
    
    .card-title {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    /* ---------- BADGES ---------- */
    .badge {
        display: inline-block !important;
        padding: 0.25rem 0.75rem !important;
        border-radius: 20px !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .badge-blue {
        background: rgba(59, 130, 246, 0.2) !important;
        color: var(--accent-blue) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    .badge-green {
        background: rgba(16, 185, 129, 0.2) !important;
        color: var(--accent-green) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    
    /* ---------- BOTÕES MODERNOS ---------- */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple)) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(59, 130, 246, 0.3) !important;
    }
    
    /* ---------- SIDEBAR MODERNA ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }
    
    .sidebar-title {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 2rem !important;
        text-align: center !important;
    }
    
    /* ---------- METRIC CARDS ---------- */
    .metric-card {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        text-align: center !important;
    }
    
    .metric-value {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.25rem !important;
    }
    
    .metric-label {
        font-size: 0.875rem !important;
        color: var(--text-secondary) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    /* ---------- TABS MODERNAS ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: transparent !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--bg-secondary) !important;
        border-radius: 12px 12px 0 0 !important;
        padding: 0.75rem 1.5rem !important;
        border: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-blue) !important;
        color: white !important;
        border-color: var(--accent-blue) !important;
    }
    
    /* ---------- GRÁFICOS ---------- */
    .plotly-chart-container {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
    }
    
    /* ---------- RESPONSIVIDADE ---------- */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

inject_modern_styles()

# ====== FUNÇÕES EXISTENTES (mantidas do código original) ======
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
    info = t.info
    hist = t.history(period=period_prices, interval="1d")
    px = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)

    try:
        ibov = yf.Ticker("^BVSP").history(period=period_prices, interval="1d")
        ibov_px = ibov["Close"].dropna() if "Close" in ibov else pd.Series(dtype=float)
    except Exception:
        ibov_px = pd.Series(dtype=float)

    return info, px, ibov_px

def _build_overview_from_info(info: dict) -> pd.DataFrame:
    """Monta um DataFrame enxuto com os principais indicadores do yfinance.info."""
    rows = [{
        "Empresa": info.get("longName"),
        "Setor": info.get("sector"),
        "P/L": _safe_pct(info.get("trailingPE")),
        "P/VP": _safe_pct(info.get("priceToBook")),
        "EV/EBITDA": _safe_pct(info.get("enterpriseToEbitda")),
        "P/Sales": _safe_pct(info.get("priceToSalesTrailing12Months")),
        "Dividend Yield (%)": _as_pct(info.get("dividendYield")),
        "ROE (%)": _as_pct(info.get("returnOnEquity")),
        "ROA (%)": _as_pct(info.get("returnOnAssets")),
        "Margem Líquida (%)": _as_pct(info.get("profitMargins")),
        "Margem Operacional (%)": _as_pct(info.get("operatingMargins")),
        "Margem EBITDA (%)": _as_pct(info.get("ebitdaMargins")),
        "Debt/Equity": _safe_pct(info.get("debtToEquity")),
        "Current Ratio": _safe_pct(info.get("currentRatio")),
        "Quick Ratio": _safe_pct(info.get("quickRatio")),
        "Market Cap (R$ bi)": (_safe_pct(info.get("marketCap")) / 1e9) if info.get("marketCap") else np.nan,
    }]
    df = pd.DataFrame(rows)
    return df

# ====== COMPONENTES DE UI MODERNOS ======
def render_modern_header():
    """Header moderno inspirado na imagem"""
    st.markdown("""
    <div class="main-header">
        <div class="block-container">
            <h1 class="header-title">Stock Peer Analysis</h1>
            <p class="header-subtitle">Easily compare stocks against others in their peer group</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_metric_card(value: str, label: str, delta: str = None):
    """Componente de card de métrica moderno"""
    delta_html = f'<div style="color: #10b981; font-size: 0.875rem; font-weight: 600;">{delta}</div>' if delta else ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        {delta_html}
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_feature_card(icon: str, title: str, description: str, badge: str = None, badge_color: str = "blue"):
    """Card de feature moderno"""
    badge_html = f'<span class="badge badge-{badge_color}">{badge}</span>' if badge else ""
    
    st.markdown(f"""
    <div class="modern-card">
        {badge_html}
        <div class="card-title">
            <span>{icon}</span>
            {title}
        </div>
        <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

# ====== LAYOUT DA PÁGINA INICIAL ======
def render_modern_home():
    """Página inicial moderna"""
    render_modern_header()
    
    # Cards de features
    col1, col2 = st.columns(2)
    
    with col1:
        render_feature_card(
            "🔎", 
            "Individual Analysis", 
            "Deep dive into a single company: multiples, profitability, debt, price history and sector comparisons.",
            "Mode 1", "blue"
        )
    
    with col2:
        render_feature_card(
            "📈", 
            "Screener / Ranking", 
            "Build company rankings by multiples and quality. Sector filters, custom weights and CSV export.",
            "Mode 2", "green"
        )
    
    st.markdown("---")
    
    # Seção de análise rápida (inspirada na imagem)
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="color: var(--text-primary); margin-bottom: 1rem;">Quick Stock Analysis</h2>
        <p style="color: var(--text-secondary);">Get instant insights for any stock symbol</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input rápido para análise
    quick_col1, quick_col2, quick_col3 = st.columns([2, 1, 1])
    
    with quick_col1:
        quick_ticker = st.text_input("Enter stock symbol", placeholder="e.g., PETR4.SA, VALE3.SA...")
    
    with quick_col2:
        time_horizon = st.selectbox("Time horizon", ["1M", "3M", "6M", "1Y", "5Y"])
    
    with quick_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("Analyze Stock", type="primary")
    
    if analyze_btn and quick_ticker:
        st.session_state["mode"] = "single"
        st.session_state["quick_ticker"] = quick_ticker
        st.rerun()

# ====== LAYOUT DE ANÁLISE INDIVIDUAL MODERNO ======
def render_modern_single_analysis():
    """Layout moderno para análise individual"""
    
    # Header da análise
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: var(--text-primary); margin-bottom: 0.5rem;">Stock Analysis</h1>
            <p style="color: var(--text-secondary);">Comprehensive analysis and peer comparison</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True):
            st.session_state["mode"] = "home"
            st.rerun()
    
    # Seção de seleção de ações
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📊 Stock Selection</div>
    """, unsafe_allow_html=True)
    
    col_select1, col_select2, col_select3 = st.columns(3)
    
    with col_select1:
        main_stock = st.text_input("Main Stock", value=st.session_state.get("quick_ticker", "PETR4.SA"), 
                                 placeholder="e.g., PETR4.SA")
    
    with col_select2:
        peer_stocks = st.multiselect(
            "Peer Stocks", 
            ["VALE3.SA", "ITUB4.SA", "BBDC4.SA", "WEGE3.SA", "MGLU3.SA"],
            default=["VALE3.SA", "ITUB4.SA"]
        )
    
    with col_select3:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years", "10 Years", "20 Years"],
            index=3
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Métricas principais
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="color: var(--text-primary);">Performance Overview</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Grid de métricas
    metric_cols = st.columns(6)
    
    with metric_cols[0]:
        render_metric_card("+23.5%", "1M Return", "↑ 2.3%")
    
    with metric_cols[1]:
        render_metric_card("+45.2%", "3M Return", "↑ 5.1%")
    
    with metric_cols[2]:
        render_metric_card("+67.8%", "6M Return", "↑ 8.2%")
    
    with metric_cols[3]:
        render_metric_card("+125.3%", "1Y Return", "↑ 15.6%")
    
    with metric_cols[4]:
        render_metric_card("12.4", "P/L Ratio", "Sector: 15.2")
    
    with metric_cols[5]:
        render_metric_card("2.1x", "P/VP", "Sector: 1.8x")
    
    # Gráficos e análise
    st.markdown("<br>", unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns([2, 1])
    
    with chart_col1:
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📈 Normalized Price Comparison</div>
        """, unsafe_allow_html=True)
        
        # Gráfico placeholder (será substituído pelo gráfico real)
        fig = go.Figure()
        
        # Dados de exemplo para o gráfico
        dates = pd.date_range(start='2024-01-01', end='2024-10-01', freq='M')
        stocks = {
            'PETR4.SA': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145],
            'VALE3.SA': [100, 102, 108, 112, 118, 122, 128, 132, 138, 142],
            'ITUB4.SA': [100, 98, 101, 104, 107, 110, 113, 116, 119, 122]
        }
        
        for stock, prices in stocks.items():
            fig.add_trace(go.Scatter(
                x=dates, 
                y=prices, 
                mode='lines',
                name=stock,
                line=dict(width=3)
            ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#f1f5f9'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with chart_col2:
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">🏆 Best Performing Stock</div>
            <div style="text-align: center; padding: 2rem 1rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🥇</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--accent-green); margin-bottom: 0.5rem;">PETR4.SA</div>
                <div style="font-size: 1.25rem; color: var(--text-primary);">+145%</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;">1 Year Return</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="modern-card">
            <div class="card-title">📊 Peer Comparison</div>
            <div style="margin-top: 1rem;">
        """, unsafe_allow_html=True)
        
        # Tabela de comparação simples
        comparison_data = {
            'Stock': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
            'Return': ['+145%', '+142%', '+122%'],
            'P/L': ['12.4', '14.2', '10.8'],
            'Rating': ['A', 'A-', 'B+']
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# ====== LAYOUT DO SCREENER MODERNO ======
def render_modern_screener():
    """Layout moderno para screener"""
    
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.markdown("""
        <div style="margin-bottom: 2rem;">
            <h1 style="color: var(--text-primary); margin-bottom: 0.5rem;">Stock Screener</h1>
            <p style="color: var(--text-secondary);">Filter and rank stocks by multiple criteria</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True):
            st.session_state["mode"] = "home"
            st.rerun()
    
    # Filtros
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">⚙️ Screening Criteria</div>
    """, unsafe_allow_html=True)
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        st.multiselect("Sectors", ["Energy", "Financial", "Materials", "Consumption", "Technology"])
        st.slider("Max P/L Ratio", 0.0, 50.0, 25.0, 1.0)
    
    with filter_col2:
        st.slider("Min ROE %", 0.0, 50.0, 10.0, 1.0)
        st.slider("Max Debt/Equity", 0.0, 5.0, 2.0, 0.1)
    
    with filter_col3:
        st.slider("Weight: Value", 0.0, 1.0, 0.3, 0.05)
        st.slider("Weight: Quality", 0.0, 1.0, 0.3, 0.05)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Resultados
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="modern-card">
        <div class="card-title">📋 Screening Results</div>
    """, unsafe_allow_html=True)
    
    # Tabela de resultados (placeholder)
    results_data = {
        'Rank': [1, 2, 3, 4, 5],
        'Ticker': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'WEGE3.SA', 'BBDC4.SA'],
        'Sector': ['Energy', 'Materials', 'Financial', 'Industrial', 'Financial'],
        'P/L': [12.4, 14.2, 10.8, 25.3, 8.7],
        'ROE %': [18.5, 22.1, 15.3, 28.7, 12.4],
        'Score': [88, 85, 82, 79, 76],
        'Rating': ['A', 'A-', 'B+', 'B+', 'B']
    }
    
    df_results = pd.DataFrame(results_data)
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# ====== NAVEGAÇÃO E ROTEAMENTO ======
MODES = {"home": "Home", "single": "Individual Analysis", "screener": "Screener"}

def set_mode(mode: str):
    st.session_state["mode"] = mode

# Sidebar moderna
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">Stock Analysis</div>
    """, unsafe_allow_html=True)
    
    # Navegação principal
    selected_mode = st.radio(
        "Navigation",
        ["Home", "Individual Analysis", "Screener"],
        index=0 if st.session_state.get("mode", "home") == "home" else 1 if st.session_state.get("mode") == "single" else 2
    )
    
    # Converter seleção para mode key
    mode_map = {"Home": "home", "Individual Analysis": "single", "Screener": "screener"}
    set_mode(mode_map[selected_mode])
    
    st.markdown("---")
    
    # Atalhos rápidos
    st.markdown("**Quick Actions**")
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        if st.button("🏠 Home", use_container_width=True):
            set_mode("home")
    
    with quick_col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.clear()
            set_mode("home")
    
    st.markdown("---")
    
    # Informações
    st.markdown("""
    <div style="color: var(--text-secondary); font-size: 0.875rem;">
        <p><strong>Version:</strong> 2.0 Modern</p>
        <p><strong>Data Source:</strong> Yahoo Finance</p>
        <p><strong>Last Update:</strong> Today</p>
    </div>
    """, unsafe_allow_html=True)

# ====== ROTEAMENTO PRINCIPAL ======
if st.session_state.get("mode", "home") == "home":
    render_modern_home()
elif st.session_state.get("mode") == "single":
    render_modern_single_analysis()
elif st.session_state.get("mode") == "screener":
    render_modern_screener()

# ====== FOOTER ======
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem 0; border-top: 1px solid var(--border); text-align: center;">
    <p style="color: var(--text-secondary); font-size: 0.875rem; margin: 0;">
        Developed for educational purposes • Modern Stock Analysis Dashboard
    </p>
</div>
""", unsafe_allow_html=True)


