"""
Cylinder Violation Index (CVI) Explorer
========================================
NBA 2026 Data Science Competition Entry

Streamlit application for evaluating foul legitimacy using skeletal
tracking data. Three-tab layout: Spatial Initiation (offense),
Cylinder Intrusion (defense), Event Explorer (game-level analysis).

Usage:  streamlit run cvi_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("data")
OFFENSE_CSV = "cvi_results_clean.csv"
BASELINES_CSV = "player_baselines_v5-2.csv"
PLAYERS_CSV = "nba_players_2024_25.csv"
DEFENSE_CSV = "cvi_defense_results.csv"

RISK_HIGH = 0.40
RISK_MODERATE = 0.70

# Semantic colors used in Python logic
CLR = {
    "green":   "#27AE60",
    "red":     "#C9082A",
    "amber":   "#E67E22",
    "blue":    "#17408B",
    "blue_lt": "#5B9BF5",
}

# =============================================================================
# THEME CSS — Light mode ("Broadcast" theme)
# =============================================================================

theme_css = """
<style>
    /* --- Core --- */
    :root { color-scheme: light !important; }
    .stApp { background-color: #F4F5F7 !important; color: #1A1A2E !important; color-scheme: light !important; }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Helvetica, Arial, sans-serif;
        color: #1A1A2E;
        color-scheme: light !important;
    }
    header[data-testid="stHeader"] { background: #F4F5F7 !important; border-bottom: none !important; }
    .block-container { padding-top: 3.5rem !important; max-width: 1200px; }

    /* --- Sidebar --- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #17408B 0%, #102D62 100%) !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] * { color: rgba(255,255,255,.8) !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.15) !important; }

    /* Sidebar expander */
    section[data-testid="stSidebar"] details {
        background: rgba(255,255,255,.08) !important;
        border: 1px solid rgba(255,255,255,.15) !important;
        border-radius: 8px !important;
    }
    section[data-testid="stSidebar"] details summary,
    section[data-testid="stSidebar"] details summary span { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] details div { color: rgba(255,255,255,.75) !important; }
    section[data-testid="stSidebar"] details p,
    section[data-testid="stSidebar"] details li,
    section[data-testid="stSidebar"] details strong { color: rgba(255,255,255,.85) !important; }
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background-color: rgba(255,255,255,.1) !important;
        border: 1px solid rgba(255,255,255,.15) !important;
        color: #FFFFFF !important;
    }

    /* --- NBA Header Bar --- */
    .nba-header {
        display: flex; align-items: center; gap: 1rem;
        padding: .75rem 0 1rem 0; margin-bottom: .75rem;
        border-bottom: 2px solid #C9082A;
    }
    .nba-header .nba-logo { height: 44px; flex-shrink: 0; object-fit: contain; }
    .nba-header .header-text h1 {
        font-size: 1.4rem !important; font-weight: 800;
        color: #1A1A2E !important; margin: 0 !important;
    }
    .nba-header .header-text p { font-size: .8rem; color: #6C757D; margin: .15rem 0 0 0; }

    /* --- Sidebar brand --- */
    .sidebar-brand { display: flex; align-items: center; gap: .75rem; margin-bottom: .25rem; }
    .sidebar-brand img { height: 36px; flex-shrink: 0; object-fit: contain; }
    .sidebar-brand .brand-text .title {
        font-size: 1rem; font-weight: 800; color: #FFFFFF !important; line-height: 1.2;
    }
    .sidebar-brand .brand-text .sub {
        font-size: .65rem; color: rgba(255,255,255,.6) !important;
        text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
    }

    /* --- KPI Cards --- */
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E0E4EA;
        border-radius: 8px; padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    div[data-testid="stMetric"] label {
        color: #6C757D !important; font-size: .72rem !important;
        text-transform: uppercase; letter-spacing: .8px; font-weight: 600;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1A1A2E !important; font-size: 1.75rem !important; font-weight: 700 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { color: #6C757D !important; }

    /* --- Verdict Card --- */
    .verdict-card {
        background: #FFFFFF; border: 1px solid #E0E4EA;
        border-radius: 10px; padding: 2rem 2.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .verdict-card h1 { font-size: 1.85rem; font-weight: 800; margin: 0 0 .2rem 0; color: #1A1A2E; }
    .verdict-card .subtitle { font-size: .95rem; color: #6C757D; margin-bottom: 1.5rem; }
    .verdict-card .big-number { font-size: 3.5rem; font-weight: 900; line-height: 1; color: #17408B; }
    .verdict-card .rate-bar { width: 80px; height: 4px; border-radius: 2px; margin-top: .5rem; }
    .verdict-card .metric-label {
        font-size: .82rem; color: #6C757D; margin-top: .35rem;
        text-transform: uppercase; letter-spacing: .5px; font-weight: 600;
    }
    .verdict-card .detail { font-size: .9rem; color: #8E99A4; margin-top: .4rem; }

    /* --- Badges --- */
    .badge {
        display: inline-block; padding: 6px 18px; border-radius: 6px;
        font-weight: 700; font-size: .78rem; letter-spacing: .6px; margin-top: .75rem;
    }
    .badge-low { background: rgba(39,174,96,.12); color: #1E8449; border: 1px solid rgba(39,174,96,.35); }
    .badge-mod { background: rgba(241,196,15,.15); color: #B7950B; border: 1px solid rgba(241,196,15,.5); }
    .badge-high { background: rgba(201,8,42,.12); color: #C9082A; border: 1px solid rgba(201,8,42,.35); }

    /* --- Defender Card --- */
    .defender-card {
        background: #FFFFFF; border: 1px solid #E0E4EA;
        border-radius: 10px; padding: 1.75rem 2rem; margin-bottom: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
    }
    .defender-card h2 { margin: 0 0 .2rem 0; font-size: 1.5rem; font-weight: 800; color: #1A1A2E; }
    .defender-card .sub { color: #6C757D; margin-bottom: 1.25rem; font-size: .9rem; }
    .defender-card .stat-row { display: flex; gap: 2.5rem; flex-wrap: wrap; margin-bottom: .75rem; }
    .defender-card .stat-item .num { font-size: 2rem; font-weight: 800; color: #1A1A2E; }
    .defender-card .stat-item .lbl {
        font-size: .78rem; color: #6C757D; text-transform: uppercase; letter-spacing: .5px;
    }
    .defender-card .insight {
        background: #F4F5F7; border-radius: 6px; padding: .85rem 1.1rem;
        color: #6C757D; margin-top: 1rem; border: 1px solid #E0E4EA;
        font-size: .9rem; line-height: 1.5;
    }
    .defender-card .insight strong { color: #1A1A2E; }

    /* --- Stats Panel --- */
    .stats-panel {
        background: #FFFFFF; border: 1px solid #E0E4EA;
        border-radius: 10px; padding: 1.5rem;
    }
    .stats-panel h3 { color: #1A1A2E !important; font-size: 1.1rem; }
    .stats-panel .row {
        display: flex; justify-content: space-between;
        padding: 8px 0; border-bottom: 1px solid #F0F1F3;
    }
    .stats-panel .row:last-child { border-bottom: none; }
    .stats-panel .row .k { color: #6C757D; font-size: .85rem; }
    .stats-panel .row .v { font-weight: 700; font-size: .9rem; color: #1A1A2E; }

    /* --- Placeholder --- */
    .placeholder-state {
        text-align: center; padding: 4rem 2rem;
        background: #FFFFFF; border: 1px solid #E0E4EA;
        border-radius: 10px; margin: 2rem 0;
    }
    .placeholder-state h2 { color: #17408B; margin-bottom: .5rem; }
    .placeholder-state p { color: #6C757D; max-width: 520px; margin: 0 auto; }

    /* --- Section label --- */
    .section-label {
        font-size: .78rem; text-transform: uppercase; letter-spacing: 1px;
        color: #8E99A4; font-weight: 600;
        margin: 1.5rem 0 .75rem 0; padding-bottom: .5rem;
        border-bottom: 1px solid #E0E4EA;
    }

    /* --- Tabs --- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0; border-bottom: 2px solid #E0E4EA; background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px; font-weight: 600; font-size: .88rem;
        color: #8E99A4; background: transparent;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 3px solid #C9082A !important; color: #1A1A2E !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: #1A1A2E !important; }

    /* --- Inputs --- */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #E0E4EA !important; color: #1A1A2E !important;
    }
    div[data-baseweb="select"] span { color: #1A1A2E !important; }

    /* Dropdown menu / popover — force light mode */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    ul[role="listbox"],
    ul[role="listbox"] > li,
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] > div,
    div[data-baseweb="menu"] ul,
    div[data-baseweb="menu"] li {
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
    }
    ul[role="listbox"] > li:hover,
    ul[role="listbox"] > li[aria-selected="true"],
    li[data-highlighted="true"],
    div[data-baseweb="menu"] li:hover {
        background-color: #F0F2F5 !important;
        color: #1A1A2E !important;
    }
    /* Select input text and placeholder */
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [data-baseweb="tag"],
    div[data-baseweb="select"] svg {
        color: #1A1A2E !important;
        fill: #6C757D !important;
    }
    .stSelectbox label, .stSelectbox label p, .stSelectbox label span { color: #1A1A2E !important; }

    /* --- Tables / DataFrames — force light mode --- */
    .stDataFrame { border-radius: 8px; overflow: hidden; border: 1px solid #E0E4EA !important; }
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] iframe {
        background-color: #FFFFFF !important;
        color-scheme: light !important;
    }
    [data-testid="stDataFrame"] * {
        color: #1A1A2E !important;
    }
    /* Force glide-data-grid (Streamlit's canvas table renderer) to light */
    [data-testid="stDataFrame"] .dvn-scroller,
    [data-testid="stDataFrame"] .dvn-underlay,
    [data-testid="stDataFrame"] .dvn-stack,
    [data-testid="stDataFrame"] canvas + div,
    [data-testid="stDataFrame"] [class*="glideDataEditor"],
    [data-testid="stDataFrame"] [data-testid="glideDataEditor"] {
        background-color: #FFFFFF !important;
        color-scheme: light !important;
    }
    /* Column headers */
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stDataFrame"] th {
        background-color: #F8F9FA !important;
        color: #1A1A2E !important;
    }
    /* Cell backgrounds */
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stDataFrame"] td {
        background-color: #FFFFFF !important;
        color: #1A1A2E !important;
    }

    /* --- Expander (main) --- */
    details { background: #FFFFFF !important; border: 1px solid #E0E4EA !important; border-radius: 8px !important; }
    details summary { color: #1A1A2E !important; }
    details div { color: #6C757D !important; }

    /* --- Alerts --- */
    div[data-testid="stAlert"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E0E4EA !important; color: #1A1A2E !important;
    }

    /* --- Overview Hero (glassmorphism) --- */
    .overview-hero {
        background: linear-gradient(135deg, #0F2341 0%, #17408B 50%, #1A4FA0 100%);
        border-radius: 16px;
        padding: 0;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .overview-hero::before {
        content: '';
        position: absolute;
        top: -80px; right: -40px;
        width: 300px; height: 300px;
        background: radial-gradient(circle, rgba(201,8,42,.18) 0%, transparent 65%);
        pointer-events: none;
    }
    .overview-hero::after {
        content: '';
        position: absolute;
        bottom: -100px; left: -60px;
        width: 280px; height: 280px;
        background: radial-gradient(circle, rgba(91,155,245,.12) 0%, transparent 65%);
        pointer-events: none;
    }
    .hero-topbar {
        display: flex; align-items: center; gap: 1rem;
        padding: 1.5rem 2.5rem 1.25rem 2.5rem;
        border-bottom: 1px solid rgba(255,255,255,.08);
        position: relative; z-index: 1;
    }
    .hero-topbar img { height: 40px; object-fit: contain; }
    .hero-topbar .hero-brand {
        font-size: .7rem; color: rgba(255,255,255,.80);
        text-transform: uppercase; letter-spacing: 2px; font-weight: 600;
    }
    .hero-body {
        padding: 2rem 2.5rem 2.5rem 2.5rem;
        position: relative; z-index: 1;
    }
    .hero-body .hero-title {
        font-size: 2rem; font-weight: 900; color: #FFFFFF;
        line-height: 1.25; margin: 0 0 .3rem 0; letter-spacing: -.5px;
    }
    .hero-body .hero-subtitle {
        font-size: 1.1rem; font-weight: 400;
        color: rgba(255,255,255,.5); margin: 0 0 1.5rem 0;
        font-style: italic;
    }
    .hero-body .hero-desc {
        font-size: .92rem; color: rgba(255,255,255,.7);
        line-height: 1.65; max-width: 680px; margin: 0 0 2rem 0;
    }
    .hero-stats { display: flex; gap: .75rem; flex-wrap: wrap; }
    .hero-stat-card {
        background: rgba(255,255,255,.07);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,.12);
        border-radius: 12px;
        padding: 1rem 1.5rem; min-width: 130px;
        transition: background .2s, border-color .2s;
    }
    .hero-stat-card:hover {
        background: rgba(255,255,255,.12);
        border-color: rgba(255,255,255,.25);
    }
    .hero-stat-card .stat-num {
        font-size: 1.6rem; font-weight: 800; color: #FFFFFF; line-height: 1;
    }
    .hero-stat-card .stat-label {
        font-size: .68rem; color: rgba(255,255,255,.5);
        text-transform: uppercase; letter-spacing: .8px;
        font-weight: 600; margin-top: .35rem;
    }

    .pillar-card {
        background: #FFFFFF;
        border: 1px solid #E0E4EA;
        border-radius: 10px;
        padding: 1.75rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
        height: 100%;
        transition: border-color .2s;
    }
    .pillar-card:hover {
        border-color: #17408B;
    }
    .pillar-card .pillar-icon {
        width: 48px; height: 48px;
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .pillar-card .pillar-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #1A1A2E;
        margin-bottom: .4rem;
    }
    .pillar-card .pillar-metric {
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .5px;
        font-weight: 600;
        margin-bottom: .6rem;
    }
    .pillar-card .pillar-desc {
        font-size: .88rem;
        color: #6C757D;
        line-height: 1.55;
    }

    .guide-card {
        background: #FFFFFF;
        border: 1px solid #E0E4EA;
        border-radius: 10px;
        padding: 1.5rem 2rem;
        box-shadow: 0 1px 4px rgba(0,0,0,.06);
        margin-bottom: 1rem;
    }
    .guide-card h3 {
        font-size: 1rem !important;
        font-weight: 700;
        margin: 0 0 .75rem 0;
    }
    .guide-card .guide-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: .6rem 0;
        border-bottom: 1px solid #F0F1F3;
    }
    .guide-card .guide-row:last-child { border-bottom: none; }
    .guide-card .guide-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 5px;
        font-weight: 700;
        font-size: .75rem;
        letter-spacing: .4px;
        white-space: nowrap;
        min-width: 140px;
        text-align: center;
    }
    .guide-card .guide-text {
        font-size: .88rem;
        color: #6C757D;
        line-height: 1.4;
    }

    .workflow-step {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem 0;
    }
    .workflow-step .step-num {
        width: 32px; height: 32px;
        background: #17408B;
        color: #FFFFFF;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 800;
        font-size: .85rem;
        flex-shrink: 0;
    }
    .workflow-step .step-content .step-title {
        font-weight: 700;
        font-size: .95rem;
        color: #1A1A2E;
        margin-bottom: .2rem;
    }
    .workflow-step .step-content .step-desc {
        font-size: .85rem;
        color: #6C757D;
        line-height: 1.5;
    }

    /* --- Global --- */
    h1, h2, h3, h4, h5, h6 { color: #1A1A2E !important; }
    p, li, span { color: #1A1A2E; }
    .stCaption, .stCaption p { color: #8E99A4 !important; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
"""

# =============================================================================
# PAGE CONFIG
# =============================================================================

# Ensure .streamlit/config.toml exists to force light theme
# (This is the ONLY reliable way to fix dark-mode tables/dropdowns in Streamlit)
_config_dir = Path(".streamlit")
_config_file = _config_dir / "config.toml"
if not _config_file.exists():
    _config_dir.mkdir(exist_ok=True)
    _config_file.write_text(
        '[theme]\n'
        'base = "light"\n'
        'primaryColor = "#C9082A"\n'
        'backgroundColor = "#F4F5F7"\n'
        'secondaryBackgroundColor = "#FFFFFF"\n'
        'textColor = "#1A1A2E"\n'
        'font = "sans serif"\n'
    )

st.set_page_config(
    page_title="CVI Explorer — NBA Foul Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# NBA LOGO — reusable component
# =============================================================================

NBA_LOGO = "https://upload.wikimedia.org/wikipedia/en/0/03/National_Basketball_Association_logo.svg"

def nba_logo_img(h=44):
    return f'<img src="{NBA_LOGO}" height="{h}" style="border-radius: 4px; object-fit: contain;">'

def nba_header(title, subtitle):
    return f"""
    <div class="nba-header">
        {nba_logo_img(44)}
        <div class="header-text">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
    </div>"""


# =============================================================================
# STYLED TABLE — renders as HTML to guarantee light mode
# =============================================================================

def render_table(df, max_height=400):
    """Render a DataFrame as a styled HTML table that respects light mode.
    This bypasses Streamlit's glide-data-grid canvas which ignores theme
    settings on macOS dark mode."""
    html = df.to_html(index=False, escape=True, classes="cvi-table")
    st.markdown(f"""
    <div style="max-height:{max_height}px; overflow-y:auto; border:1px solid #E0E4EA;
                border-radius:8px; margin-bottom:1rem;">
    <style>
        .cvi-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: .85rem;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            background: #FFFFFF;
        }}
        .cvi-table thead th {{
            background: #F8F9FA;
            color: #1A1A2E;
            font-weight: 700;
            font-size: .75rem;
            text-transform: uppercase;
            letter-spacing: .5px;
            padding: 10px 12px;
            border-bottom: 2px solid #E0E4EA;
            position: sticky;
            top: 0;
            z-index: 1;
        }}
        .cvi-table tbody td {{
            padding: 8px 12px;
            color: #1A1A2E;
            border-bottom: 1px solid #F0F1F3;
            white-space: nowrap;
        }}
        .cvi-table tbody tr:hover {{
            background: #F4F5F7;
        }}
        .cvi-table tbody tr:nth-child(even) {{
            background: #FAFBFC;
        }}
        .cvi-table tbody tr:nth-child(even):hover {{
            background: #F4F5F7;
        }}
    </style>
    {html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SAMPLE DATA
# =============================================================================

def _sample_offense():
    np.random.seed(42)
    players = [
        (201566,"Harrison Barnes","SAC"),(1628369,"Jayson Tatum","BOS"),
        (203507,"Giannis Antetokounmpo","MIL"),(201935,"James Harden","LAC"),
        (203081,"Damian Lillard","MIL"),(1629029,"Luka Doncic","DAL"),
        (203954,"Joel Embiid","PHI"),(1630162,"Anthony Edwards","MIN"),
        (203110,"Draymond Green","GSW"),(201142,"Kevin Durant","PHX")]
    defenders = [
        (1631102,"Jake LaRavia","MEM"),(1628398,"Pascal Siakam","IND"),
        (203507,"Giannis Antetokounmpo","MIL"),(203935,"Marcus Smart","MEM"),
        (1627759,"Jaylen Brown","BOS"),(203952,"Andrew Wiggins","GSW"),
        (1628983,"Shai Gilgeous-Alexander","OKC"),(203897,"Bam Adebayo","MIA"),
        (1629627,"Cam Reddish","LAL"),(201933,"Blake Griffin","BOS")]
    rows = []
    games = [f"002240{i:04d}" for i in range(1,11)]
    for ev in range(1,352):
        p=players[np.random.randint(len(players))]
        d=defenders[np.random.randint(len(defenders))]
        lat=np.random.exponential(8)+4; thr=np.random.uniform(10,16)
        z=(lat-np.random.uniform(5,8))/np.random.uniform(2,4)
        lk=bool(z>2.5 and np.random.random()>.6)
        bc=bool(z>2.0 and np.random.random()>.7 and not lk)
        ji=bool(np.random.random()>.9)
        rows.append(dict(
            game_id=games[np.random.randint(10)],
            period=np.random.randint(1,5),gc_seconds=np.random.randint(0,720),
            event_num=ev,
            foul_category=np.random.choice(["Shooting","Personal","Offensive","Loose Ball"]),
            offensive_player_id=p[0],offensive_player_name=p[1],
            defensive_player_id=d[0],defensive_player_name=d[1],
            team_abbr=p[2],
            pbp_description=f"Foul: {np.random.choice(['Shooting','Personal'])} — {d[1]} on {p[1]}",
            osi_leg_kick=lk,osi_brakecheck=bc,osi_jump_into=ji,
            osi_any=lk or bc or ji,
            max_lateral_dev_in=round(lat,1),personal_threshold_in=round(thr,1),
            deviation_z_score=round(z,2),
            max_deceleration_fps2=round(np.random.uniform(5,30),1),
            decel_threshold_fps2=round(np.random.uniform(15,25),1),
            jump_into_displacement_ft=round(np.random.uniform(0,3),1) if ji else 0.0,
            jump_into_defender_dist_ft=round(np.random.uniform(1,10),1) if ji else 0.0,
            nearest_defender_dist_ft=round(np.random.uniform(1,12),1),
            defender_in_cylinder=bool(np.random.random()>.7),
            n_frames=np.random.randint(10,50),
            peak_frame_id=np.random.randint(100,10000)))
    return pd.DataFrame(rows)

def _sample_baselines():
    np.random.seed(42)
    ids=[201566,1628369,203507,201935,203081,1629029,203954,1630162,203110,
         201142,1631102,1628398,203935,1627759,203952,1628983,203897,1629627,201933]
    rows=[]
    for pid in ids:
        m,s=np.random.uniform(5,10),np.random.uniform(1.5,4)
        rows.append(dict(personId=pid,n_shots=np.random.randint(30,300),
            mean_lateral_dev=round(m,2),std_lateral_dev=round(s,2),
            p50_lateral_dev=round(m-.5,2),p75_lateral_dev=round(m+s*.5,2),
            p90_lateral_dev=round(m+s*1.3,2),p95_lateral_dev=round(m+s*1.65,2),
            max_lateral_dev=round(m+s*3,2)))
    return pd.DataFrame(rows)

def _sample_players():
    data=[
        (201566,"Harrison Barnes","Harrison","Barnes",True,"SAC"),
        (1628369,"Jayson Tatum","Jayson","Tatum",True,"BOS"),
        (203507,"Giannis Antetokounmpo","Giannis","Antetokounmpo",True,"MIL"),
        (201935,"James Harden","James","Harden",True,"LAC"),
        (203081,"Damian Lillard","Damian","Lillard",True,"MIL"),
        (1629029,"Luka Doncic","Luka","Doncic",True,"DAL"),
        (203954,"Joel Embiid","Joel","Embiid",True,"PHI"),
        (1630162,"Anthony Edwards","Anthony","Edwards",True,"MIN"),
        (203110,"Draymond Green","Draymond","Green",True,"GSW"),
        (201142,"Kevin Durant","Kevin","Durant",True,"PHX"),
        (1631102,"Jake LaRavia","Jake","LaRavia",True,"MEM"),
        (1628398,"Pascal Siakam","Pascal","Siakam",True,"IND"),
        (203935,"Marcus Smart","Marcus","Smart",True,"MEM"),
        (1627759,"Jaylen Brown","Jaylen","Brown",True,"BOS"),
        (203952,"Andrew Wiggins","Andrew","Wiggins",True,"GSW"),
        (1628983,"Shai Gilgeous-Alexander","Shai","Gilgeous-Alexander",True,"OKC"),
        (203897,"Bam Adebayo","Bam","Adebayo",True,"MIA"),
        (1629627,"Cam Reddish","Cam","Reddish",True,"LAL"),
        (201933,"Blake Griffin","Blake","Griffin",False,"BOS")]
    return pd.DataFrame(data,columns=[
        "personId","player_name","first_name","last_name","is_active","current_team_abbr"])


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_offense():
    p = DATA_DIR / OFFENSE_CSV
    df = pd.read_csv(p) if p.exists() else _sample_offense()
    df["offensive_player_id"] = df["offensive_player_id"].astype(int)
    df["defensive_player_id"] = df["defensive_player_id"].astype(int)
    for c in ["osi_leg_kick","osi_brakecheck","osi_jump_into"]:
        if c not in df.columns: df[c] = False
    df["osi_any"] = df[["osi_leg_kick","osi_brakecheck","osi_jump_into"]].any(axis=1)
    return df

@st.cache_data(show_spinner=False)
def load_baselines():
    p=DATA_DIR/BASELINES_CSV
    if p.exists():
        df = pd.read_csv(p)
        df["personId"] = df["personId"].astype(int)
        return df
    return _sample_baselines()

@st.cache_data(show_spinner=False)
def load_players():
    p = DATA_DIR / PLAYERS_CSV
    if p.exists():
        df = pd.read_csv(p)
        df["personId"] = df["personId"].astype(int)
        return df
    return _sample_players()

@st.cache_data(show_spinner=False)
def load_defense():
    p = DATA_DIR / DEFENSE_CSV
    if p.exists():
        df = pd.read_csv(p)
        if "defender_id" in df.columns:
            df["defender_id"] = df["defender_id"].astype(int)
        return df
    return None

def _name(pid,pdf):
    m=pdf[pdf["personId"]==pid]
    return m.iloc[0]["player_name"] if not m.empty else str(pid)

def _team(pid,pdf):
    m=pdf[pdf["personId"]==pid]
    return m.iloc[0].get("current_team_abbr","—") if not m.empty else "—"


# =============================================================================
# PLOTLY HELPERS (light mode)
# =============================================================================

PC = dict(text="#1A1A2E", text2="#6C757D", grid="#E0E4EA", bg="rgba(0,0,0,0)")

def _layout(**kw):
    base=dict(paper_bgcolor=PC["bg"],plot_bgcolor=PC["bg"],
        font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif",
                  color=PC["text"],size=14))
    if "margin" not in kw: base["margin"]=dict(l=20,r=20,t=40,b=20)
    base.update(kw)
    return base

def chart_rate_bar(rate):
    fig=go.Figure()
    fig.add_trace(go.Bar(y=[""],x=[rate],orientation="h",marker_color=CLR["blue"],
        text=f"{rate:.0%}",textposition="inside",textfont=dict(size=15,color="#FFF"),
        hovertemplate="Natural Trajectory: %{x:.1%}<extra></extra>",name="Natural"))
    fig.add_trace(go.Bar(y=[""],x=[1-rate],orientation="h",marker_color=CLR["red"],
        text=f"{1-rate:.0%}",textposition="inside",textfont=dict(size=15,color="#FFF"),
        hovertemplate="Cylinder Violation: %{x:.1%}<extra></extra>",name="Violation"))
    fig.update_layout(**_layout(barmode="stack",height=56,showlegend=False,
        xaxis=dict(visible=False,range=[0,1]),yaxis=dict(visible=False),
        margin=dict(l=0,r=0,t=0,b=0)))
    return fig

def chart_flags(lk,bc,ji):
    cats=["Leg Kicks","Brake-Checks","Jump-Intos"]; vals=[lk,bc,ji]
    colors=[CLR["red"],CLR["amber"],CLR["blue"]]
    fig=go.Figure()
    for c,v,co in zip(cats,vals,colors):
        fig.add_trace(go.Bar(y=[c],x=[v],orientation="h",marker_color=co,
            text=str(v),textposition="outside",textfont=dict(size=13,color=PC["text"]),
            hovertemplate=f"{c}: {v}<extra></extra>"))
    fig.update_layout(**_layout(height=170,showlegend=False,
        xaxis=dict(title="Count",dtick=1,gridcolor=PC["grid"],tickfont=dict(color=PC["text2"])),
        yaxis=dict(autorange="reversed",tickfont=dict(color=PC["text"])),
        margin=dict(l=120,r=40,t=10,b=40)))
    return fig

def chart_targets(counts):
    top=counts.head(8)
    fig=go.Figure(go.Bar(y=top.index[::-1],x=top.values[::-1],orientation="h",
        marker_color=CLR["red"],text=top.values[::-1],textposition="outside",
        textfont=dict(size=13,color=PC["text"]),
        hovertemplate="%{y}: %{x} violations<extra></extra>"))
    fig.update_layout(**_layout(height=max(200,len(top)*42+60),showlegend=False,
        xaxis=dict(title="Cylinder Violations Drawn",dtick=1,gridcolor=PC["grid"],
                   tickfont=dict(color=PC["text2"])),
        yaxis=dict(tickfont=dict(color=PC["text"])),
        margin=dict(l=170,r=40,t=10,b=40)))
    return fig

def chart_deviation_scatter(game_df):
    """Scatter plot: lateral deviation vs personal threshold for each foul event.
    Flagged events shown in red, clean events in blue. Dot size = |z-score|.
    Diagonal line = threshold boundary (above = flagged territory)."""
    flagged = game_df[game_df["osi_any"]]
    clean = game_df[~game_df["osi_any"]]

    fig = go.Figure()

    # Threshold boundary line
    max_val = max(game_df["max_lateral_dev_in"].max(),
                  game_df["personal_threshold_in"].max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color="#E0E4EA", width=2, dash="dash"),
        name="Threshold Boundary", showlegend=True,
        hoverinfo="skip"))

    # Clean events
    if not clean.empty:
        fig.add_trace(go.Scatter(
            x=clean["personal_threshold_in"],
            y=clean["max_lateral_dev_in"],
            mode="markers",
            marker=dict(
                size=clean["deviation_z_score"].abs().clip(2, 12) * 3,
                color=CLR["blue"], opacity=0.4,
                line=dict(width=1, color="#FFF")),
            name="Natural Trajectory",
            text=clean["offensive_player_name"],
            customdata=np.stack([
                clean["deviation_z_score"],
                clean["foul_category"]], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Lateral Dev: %{y:.1f}\"<br>"
                "Threshold: %{x:.1f}\"<br>"
                "Z-Score: %{customdata[0]:.2f}σ<br>"
                "Type: %{customdata[1]}"
                "<extra>Natural</extra>")))

    # Flagged events
    if not flagged.empty:
        # Build flag type labels
        ftypes = []
        for _, r in flagged.iterrows():
            t = []
            if r["osi_leg_kick"]: t.append("Leg Kick")
            if r["osi_brakecheck"]: t.append("Brake-Check")
            if r["osi_jump_into"]: t.append("Jump-Into")
            ftypes.append(", ".join(t) if t else "Flagged")

        fig.add_trace(go.Scatter(
            x=flagged["personal_threshold_in"],
            y=flagged["max_lateral_dev_in"],
            mode="markers",
            marker=dict(
                size=flagged["deviation_z_score"].abs().clip(2, 12) * 3,
                color=CLR["red"], opacity=0.8,
                line=dict(width=1.5, color="#FFF"),
                symbol="diamond"),
            name="Cylinder Violation",
            text=flagged["offensive_player_name"],
            customdata=np.stack([
                flagged["deviation_z_score"],
                ftypes], axis=-1),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Lateral Dev: %{y:.1f}\"<br>"
                "Threshold: %{x:.1f}\"<br>"
                "Z-Score: %{customdata[0]:.2f}σ<br>"
                "Flag: %{customdata[1]}"
                "<extra>Violation</extra>")))

    fig.update_layout(**_layout(
        height=420, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=.5),
        xaxis=dict(title="Personal Threshold (in)", gridcolor=PC["grid"],
                   tickfont=dict(color=PC["text2"])),
        yaxis=dict(title="Lateral Deviation (in)", gridcolor=PC["grid"],
                   tickfont=dict(color=PC["text2"])),
        margin=dict(l=60, r=20, t=50, b=50)))

    # Add annotation for the regions
    fig.add_annotation(
        x=max_val * 0.3, y=max_val * 0.85,
        text="⬆ ABOVE THRESHOLD<br><i>Higher violation probability</i>",
        showarrow=False, font=dict(size=11, color=CLR["red"]),
        opacity=0.6)
    fig.add_annotation(
        x=max_val * 0.75, y=max_val * 0.2,
        text="⬇ BELOW THRESHOLD<br><i>Natural trajectory zone</i>",
        showarrow=False, font=dict(size=11, color=CLR["blue"]),
        opacity=0.5)
    return fig


def chart_game_violation_types(game_df):
    """Donut chart showing violation type distribution for a game."""
    lk = int(game_df["osi_leg_kick"].sum())
    bc = int(game_df["osi_brakecheck"].sum())
    ji = int(game_df["osi_jump_into"].sum())
    clean = int((~game_df["osi_any"]).sum())

    labels = ["Natural Trajectory", "Leg Kicks", "Brake-Checks", "Jump-Intos"]
    values = [clean, lk, bc, ji]
    colors = [CLR["blue"], CLR["red"], CLR["amber"], CLR["blue_lt"]]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55, marker=dict(colors=colors, line=dict(color="#FFF", width=2)),
        textinfo="label+value", textposition="outside",
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value} events (%{percent})<extra></extra>"))

    fig.update_layout(**_layout(
        height=320, showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)))

    fig.add_annotation(
        text=f"<b>{int(game_df['osi_any'].sum())}</b><br><span style='font-size:11px;color:#6C757D'>Flagged</span>",
        x=0.5, y=0.5, font=dict(size=24, color="#1A1A2E"),
        showarrow=False)
    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(df):
    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            {nba_logo_img(36)}
            <div class="brand-text">
                <div class="title">Cylinder Violation<br/>Index</div>
                <div class="sub">CVI Explorer</div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.caption(
            "Evaluate whether a player's drawn fouls result from natural "
            "shooting trajectories or unnatural cylinder violations.")
        st.markdown("---")
        # st.caption("NBA 2026 Data Science Competition")


# =============================================================================
# TAB 0 — OVERVIEW (Executive Summary / Onboarding)
# =============================================================================

def render_overview(df):
    n_games = df["game_id"].nunique()
    n_fouls = len(df)
    n_flagged = int(df["osi_any"].sum())
    n_players = df["offensive_player_id"].nunique()
    flag_rate = n_flagged / n_fouls if n_fouls else 0

    # --- Hero Section ---
    st.markdown(f"""
    <div class="overview-hero">
        <div class="hero-topbar">
            <span class="hero-brand">NBA Future Analytics Stars 2026 · Rome Delgado-Gonzalez & Naisha Sinha</span>
        </div>
        <div class="hero-body">
            <div style="display:flex; align-items:center; gap:1rem; margin-bottom:.3rem;">
                {nba_logo_img(45)}
                <div class="hero-title">Cylinder Violation Index (CVI) Explorer</div>
            </div>
            <div class="hero-subtitle">Upgrading the eye test: Quantifying foul baiting with high-fidelity skeletal tracking.</div>
            <div class="hero-desc">
                Built on NBA Hawk-Eye skeletal tracking data, CVI compares every
                player's body kinematics during foul events against their own
                biomechanical baseline instead of a league average to flag unnatural
                deviations with statistical precision.
            </div>
            <div class="hero-stats">
                <div class="hero-stat-card">
                    <div class="stat-num">{n_games}</div>
                    <div class="stat-label">Games Analyzed</div>
                </div>
                <div class="hero-stat-card">
                    <div class="stat-num">{n_fouls}</div>
                    <div class="stat-label">Foul Events</div>
                </div>
                <div class="hero-stat-card">
                    <div class="stat-num">{n_flagged}</div>
                    <div class="stat-label">Violations Detected</div>
                </div>
                <div class="hero-stat-card">
                    <div class="stat-num">{flag_rate:.0%}</div>
                    <div class="stat-label">Overall Flag Rate</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- The Three Pillars ---
    st.markdown('<div class="section-label">What CVI Measures: Three Violation Types</div>',
                unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)

    with p1:
        lk_count = int(df["osi_leg_kick"].sum())
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-icon" style="background:rgba(201,8,42,.1);">🦵</div>
            <div class="pillar-title">Leg Kick</div>
            <div class="pillar-metric" style="color:#C9082A;">Lateral Foot Deviation</div>
            <div class="pillar-desc">
                During a shooting foul, the offensive player extends their leg
                unnaturally outward to initiate contact with the defender. CVI
                compares the shooter's <strong>lateral foot position</strong>
                against their own baseline from non-foul shooting possessions.
                If the deviation exceeds their personal 95th percentile, it's flagged.
            </div>
            <div style="margin-top:1rem; padding-top:.75rem; border-top:1px solid #F0F1F3;">
                <span style="font-size:1.3rem; font-weight:800; color:#C9082A;">{lk_count}</span>
                <span style="font-size:.8rem; color:#6C757D;"> detected in dataset</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with p2:
        bc_count = int(df["osi_brakecheck"].sum())
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-icon" style="background:rgba(230,126,34,.1);">🛑</div>
            <div class="pillar-title">Brake-Check</div>
            <div class="pillar-metric" style="color:#E67E22;">Sudden Deceleration</div>
            <div class="pillar-desc">
                The ball handler abruptly stops or decelerates to force a trailing
                defender into a collision. CVI measures the player's
                <strong>deceleration profile</strong> and compares it to their
                normal stopping patterns. If the stop is significantly more
                sudden than their baseline, contact was manufactured.
            </div>
            <div style="margin-top:1rem; padding-top:.75rem; border-top:1px solid #F0F1F3;">
                <span style="font-size:1.3rem; font-weight:800; color:#E67E22;">{bc_count}</span>
                <span style="font-size:.8rem; color:#6C757D;"> detected in dataset</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with p3:
        ji_count = int(df["osi_jump_into"].sum())
        st.markdown(f"""
        <div class="pillar-card">
            <div class="pillar-icon" style="background:rgba(23,64,139,.1);">⬆️</div>
            <div class="pillar-title">Jump-Into</div>
            <div class="pillar-metric" style="color:#17408B;">Unnatural Forward Displacement</div>
            <div class="pillar-desc">
                During a shooting motion, the offensive player displaces their body
                forward toward a nearby defender rather than shooting with a natural
                vertical trajectory. CVI tracks <strong>forward displacement</strong>
                relative to the defender's position and flags cases where the
                shooter closed distance unnaturally.
            </div>
            <div style="margin-top:1rem; padding-top:.75rem; border-top:1px solid #F0F1F3;">
                <span style="font-size:1.3rem; font-weight:800; color:#17408B;">{ji_count}</span>
                <span style="font-size:.8rem; color:#6C757D;"> detected in dataset</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- How to Read the Data ---
    st.markdown('<div class="section-label">How to Read CVI Results</div>',
                unsafe_allow_html=True)

    g1, g2 = st.columns([1, 1])

    with g1:
        st.markdown(f"""
        <div class="guide-card">
            <h3>Understanding Deviation Badges</h3>
            <p style="font-size:.85rem; color:#6C757D; margin-bottom:1rem;">
                Each player receives a <strong>Natural Trajectory Rate</strong>:
                the percentage of their drawn fouls that occurred through normal
                basketball movement. The badge indicates deviation level:
            </p>
            <div class="guide-row">
                <span class="guide-badge" style="background:rgba(39,174,96,.12); color:#1E8449; border:1px solid rgba(39,174,96,.35);">LOW DEVIATION</span>
                <span class="guide-text">70%+ natural trajectory. This player earns fouls through legitimate play. Low risk in free agency evaluation.</span>
            </div>
            <div class="guide-row">
                <span class="guide-badge" style="background:rgba(241,196,15,.15); color:#B7950B; border:1px solid rgba(241,196,15,.5);">MODERATE DEVIATION</span>
                <span class="guide-text">40–70% natural trajectory. Some manufactured contact detected. Warrants deeper review of specific events.</span>
            </div>
            <div class="guide-row">
                <span class="guide-badge" style="background:rgba(201,8,42,.12); color:#C9082A; border:1px solid rgba(201,8,42,.35);">HIGH DEVIATION</span>
                <span class="guide-text">Below 40% natural trajectory. Significant foul manufacturing pattern. FT production may decline in playoff officiating.</span>
            </div>
        </div>""", unsafe_allow_html=True)

    with g2:
        st.markdown(f"""
        <div class="guide-card">
            <h3>Key Metrics Explained</h3>
            <div class="guide-row">
                <span class="guide-badge" style="background:#17408B; color:#FFF;">NATURAL TRAJ. RATE</span>
                <span class="guide-text">Percentage of fouls drawn through normal basketball movement. Higher = cleaner player.</span>
            </div>
            <div class="guide-row">
                <span class="guide-badge" style="background:#F4F5F7; color:#1A1A2E; border:1px solid #E0E4EA;">Z-SCORE</span>
                <span class="guide-text">How many standard deviations above the player's own baseline. Higher z-score = more abnormal the movement.</span>
            </div>
            <div class="guide-row">
                <span class="guide-badge" style="background:#F4F5F7; color:#1A1A2E; border:1px solid #E0E4EA;">LATERAL DEV (in)</span>
                <span class="guide-text">Maximum sideways body displacement during the foul event, measured in inches from skeletal tracking data.</span>
            </div>
            <div class="guide-row">
                <span class="guide-badge" style="background:#F4F5F7; color:#1A1A2E; border:1px solid #E0E4EA;">P95 THRESHOLD</span>
                <span class="guide-text">The player's personal 95th percentile deviation. Movements beyond this are statistically abnormal for that specific player.</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # --- Getting Started ---
    st.markdown('<div class="section-label">Getting Started</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="guide-card">
        <div class="workflow-step">
            <div class="step-num">1</div>
            <div class="step-content">
                <div class="step-title">Spatial Initiation → Evaluate an offensive player</div>
                <div class="step-desc">
                    Type a player name to see their Natural Trajectory Rate, violation
                    breakdown, and every flagged event. Start here when scouting a
                    free agent's foul-drawing legitimacy.
                </div>
            </div>
        </div>
        <div class="workflow-step">
            <div class="step-num">2</div>
            <div class="step-content">
                <div class="step-title">Cylinder Intrusion → Evaluate a defender</div>
                <div class="step-desc">
                    Search a defender to see how many of their fouls were caused by
                    offensive manufacturing — not their own mistakes. Use this when
                    evaluating whether a defender is actually undisciplined or just targeted.
                </div>
            </div>
        </div>
        <div class="workflow-step">
            <div class="step-num">3</div>
            <div class="step-content">
                <div class="step-title">Event Explorer → Drill into a specific game</div>
                <div class="step-desc">
                    Select a game to see a deviation scatter plot comparing every foul
                    event against player thresholds. Expand individual flagged events
                    for full biomechanical detail.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # --- Methodology (moved from sidebar) ---
    with st.expander("Full Methodology: How CVI Works"):
        st.markdown("""
**Data Source**

CVI is built on NBA Hawk-Eye optical tracking data, which captures
29 skeletal joints at 25 frames per second for every player on the court.
This gives us sub-inch precision on body positioning during every play.

**Personal Baselines**

For each player, we compute a biomechanical baseline from their non-foul
shooting possessions — how their body naturally moves when they shoot
without drawing contact. This creates a player-specific distribution of
lateral deviation, deceleration, and forward displacement. The 95th
percentile of this distribution becomes that player's personal threshold.

**Detection Logic**

During each foul event, we compare the offensive player's kinematics
against their own baseline:

1. **Leg Kick**: lateral foot deviation exceeds the player's personal
   95th percentile during a shooting foul.
2. **Brake-Check**: deceleration exceeds the player's normal stopping
   profile, forcing trailing contact.
3. **Jump-Into**: forward displacement toward a defender during a
   shooting motion reduces the gap unnaturally.

If any of these triggers fire, the foul is flagged as a cylinder violation.
The **Natural Trajectory Rate** is simply the percentage of a player's
drawn fouls where none of these flags triggered.

**Why Personal Baselines Matter**

A player like James Harden naturally has wider body movements than a
catch-and-shoot specialist. Using a league-wide threshold would
over-flag physical players and under-flag subtle manipulators. By
comparing each player to themselves, CVI produces fair, individualized
assessments.

**Defensive Analysis (Cylinder Intrusion)**

The same framework applies in reverse: we track whether defenders
step into shooters' landing zones or aggressively crowd the offensive
cylinder. This data is generated separately and can be loaded into
the Cylinder Intrusion tab when available.
""")


# =============================================================================
# TAB 1 — SPATIAL INITIATION (Offense)
# =============================================================================

def render_spatial_initiation(df, bl, pl):
    st.markdown(nba_header(
        "Spatial Initiation Analysis",
        "Evaluate offensive players for unnatural cylinder deviations: "
        "leg kicks, brake-checks, and jump-into movements during foul events."),
        unsafe_allow_html=True)

    # KPIs
    # k1,k2,k3,k4 = st.columns(4)
    # k1.metric("Games Analyzed", df["game_id"].nunique())
    # k2.metric("Total Foul Events", len(df))
    # k3.metric("Flagged Events", int(df["osi_any"].sum()))
    # k4.metric("Players in Dataset", df["offensive_player_id"].nunique())
    # st.markdown("")

    # Player search
    nmap = {pid: _name(pid,pl) for pid in df["offensive_player_id"].unique()}
    sel = st.selectbox("Search for a player", [""]+sorted(nmap.values()),
                       index=0, placeholder="Type a player name...", key="t1p")
    if not sel:
        st.info("Select a player to view their spatial initiation profile.", icon="👆")
        return

    pid=[k for k,v in nmap.items() if v==sel][0]
    pf=df[df["offensive_player_id"]==pid]
    tot=len(pf); flg=int(pf["osi_any"].sum()); leg=tot-flg
    rate=leg/tot if tot else 1.0
    lk,bc,ji=int(pf["osi_leg_kick"].sum()),int(pf["osi_brakecheck"].sum()),int(pf["osi_jump_into"].sum())

    if rate>=RISK_MODERATE: lbl,cls,ic="LOW DEVIATION","badge-low",CLR["green"]
    elif rate>=RISK_HIGH: lbl,cls,ic="MODERATE DEVIATION","badge-mod","#F1C40F"
    else: lbl,cls,ic="HIGH DEVIATION","badge-high",CLR["red"]

    st.markdown(f"""
    <div class="verdict-card">
        <h1>{sel}</h1>
        <div class="subtitle">{tot} fouls drawn in dataset</div>
        <div style="display:flex;gap:3rem;align-items:flex-end;flex-wrap:wrap;">
            <div>
                <div class="big-number">{rate:.0%}</div>
                <div class="rate-bar" style="background:{ic}"></div>
                <div class="metric-label">Natural Trajectory Rate</div>
            </div>
            <div>
                <div class="detail">{leg} of {tot} fouls drawn via natural trajectory</div>
                <div class="detail">{flg} flagged as cylinder violations</div>
                <div><span class="badge {cls}">{lbl}</span></div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.plotly_chart(chart_rate_bar(rate), use_container_width=True,
                    config={"displayModeBar":False})

    # Breakdown
    st.markdown('<div class="section-label">Violation Breakdown</div>',
                unsafe_allow_html=True)

    # Full-width chart
    st.plotly_chart(chart_flags(lk, bc, ji), use_container_width=True,
                    config={"displayModeBar": False})
    # Per-type context
    if flg > 0:
        type_details = []
        if lk > 0: type_details.append(f"**Leg Kicks:** {lk} ({lk/tot:.0%} of all fouls)")
        if bc > 0: type_details.append(f"**Brake-Checks:** {bc} ({bc/tot:.0%} of all fouls)")
        if ji > 0: type_details.append(f"**Jump-Intos:** {ji} ({ji/tot:.0%} of all fouls)")
        st.caption(" · ".join(type_details))

    # Player Baseline + League Context — side by side BELOW the chart
    st.markdown('<div class="section-label">Player Baseline & League Context</div>',
                unsafe_allow_html=True)
    b1, b2, b3, b4, b5 = st.columns(5)

    row = bl[bl["personId"] == pid]
    if not row.empty:
        r = row.iloc[0]
        b1.metric("Median Lateral Dev", f'{r["p50_lateral_dev"]:.1f}"')
        b2.metric("P95 Threshold", f'{r["p95_lateral_dev"]:.1f}"')
        b3.metric("Shots in Baseline", int(r["n_shots"]))
    else:
        b1.caption("No baseline data")

    rates = [1 - df[df["offensive_player_id"] == u]["osi_any"].mean()
             for u in df["offensive_player_id"].unique()
             if len(df[df["offensive_player_id"] == u]) >= 3]
    la = np.mean(rates) if rates else .5
    rank_num = sum(1 for r2 in rates if r2 <= rate)
    b4.metric("League Avg Traj. Rate", f"{la:.0%}",
              delta=f"{rate - la:+.0%} vs league", delta_color="normal")
    b5.metric("Player Rank", f"{rank_num}/{len(rates)}",
              delta="by natural traj. rate", delta_color="off")

    # Flagged events table
    st.markdown('<div class="section-label">Flagged Foul Events</div>',
                unsafe_allow_html=True)
    fl=pf[pf["osi_any"]].copy()
    if fl.empty:
        st.success("No flagged events — all fouls exhibit natural trajectory.", icon="✅")
    else:
        ft=[]
        for _,r in fl.iterrows():
            t=[]
            if r["osi_leg_kick"]: t.append("Leg Kick")
            if r["osi_brakecheck"]: t.append("Brake-Check")
            if r["osi_jump_into"]: t.append("Jump-Into")
            ft.append(", ".join(t) if t else "—")
        fl["Flag Type"]=ft
        cols={"game_id":"Game","period":"Qtr","gc_seconds":"Time (s)",
              "pbp_description":"Description","Flag Type":"Violation Type",
              "deviation_z_score":"Z-Score","max_lateral_dev_in":"Lat. Dev (in)",
              "nearest_defender_dist_ft":"Def. Dist (ft)","defensive_player_name":"Defender"}
        av={k:v for k,v in cols.items() if k in fl.columns}
        render_table(fl[list(av.keys())].rename(columns=av),
                     max_height=min(400,len(fl)*40+40))


# =============================================================================
# TAB 2 — CYLINDER INTRUSION (Defense)
# =============================================================================

def render_cylinder_intrusion(df, pl, defense_df):
    st.markdown(nba_header(
        "Cylinder Intrusion Analysis",
        "Evaluate defenders: adjusted foul profiles from offensive baiting, "
        "plus landing space and crowding violations when available."),
        unsafe_allow_html=True)

    # ----- Section A: Defender Adjusted Profile (always available) -----
    st.markdown('<div class="section-label">Defender Adjusted Foul Profile</div>',
                unsafe_allow_html=True)
    st.caption("Identify defenders whose foul counts are inflated by "
               "offensive cylinder violations committed against them.")

    dmap={d:_name(d,pl) for d in df["defensive_player_id"].unique()}
    sel=st.selectbox("Search for a defender",[""]+sorted(dmap.values()),
                     index=0,placeholder="Type a defender name...",key="t2d")
    if sel:
        did=[k for k,v in dmap.items() if v==sel][0]
        dd=df[df["defensive_player_id"]==did]; tot=len(dd)
        bait=int(dd["osi_any"].sum()); adj=tot-bait
        pct=bait/tot if tot else 0

        st.markdown(f"""
        <div class="defender-card">
            <h2>{sel}</h2>
            <div class="sub">{tot} fouls committed in dataset</div>
            <div class="stat-row">
                <div class="stat-item"><div class="num">{tot}</div><div class="lbl">Raw Fouls</div></div>
                <div class="stat-item"><div class="num" style="color:{CLR['red']}">{bait}</div><div class="lbl">Offensive Violations</div></div>
                <div class="stat-item"><div class="num" style="color:{CLR['blue_lt']}">{adj}</div><div class="lbl">Adjusted Count</div></div>
            </div>
            <div class="insight"><strong>{pct:.1%}</strong> of this player's fouls were the result of offensive cylinder violations — not defensive error.</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">Who Targeted This Defender?</div>',
                    unsafe_allow_html=True)
        bd=dd[dd["osi_any"]]
        if bd.empty:
            st.success("No cylinder violations drawn against this defender.", icon="✅")
        else:
            st.plotly_chart(chart_targets(bd["offensive_player_name"].value_counts()),
                            use_container_width=True,config={"displayModeBar":False})

        st.markdown('<div class="section-label">All Foul Events</div>',
                    unsafe_allow_html=True)
        disp=dd.copy(); disp["Violation?"]=disp["osi_any"].map({True:"⚑ Yes",False:"—"})
        cols={"game_id":"Game","period":"Qtr","gc_seconds":"Time (s)",
              "offensive_player_name":"Offensive Player","pbp_description":"Description",
              "Violation?":"Cylinder Violation?","deviation_z_score":"Z-Score"}
        av={k:v for k,v in cols.items() if k in disp.columns}
        render_table(disp[list(av.keys())].rename(columns=av),
                     max_height=min(400,len(disp)*40+40))
    else:
        st.info("Select a defender to view their adjusted foul profile.", icon="👆")

    # ----- Section B: Defensive CVI (landing space / crowding) -----
    st.markdown("---")
    st.markdown('<div class="section-label">Landing Space & Crowding Violations</div>',
                unsafe_allow_html=True)

    if defense_df is None:
        st.markdown(f"""
        <div class="placeholder-state">
            <h2>Defensive CVI Data Pending</h2>
            <p style="font-size:1rem;margin-bottom:.75rem;">
                Landing space and crowding analysis is being computed.</p>
            <p>When ready, drop <code style="color:{CLR['blue']}">cvi_defense_results.csv</code>
                into <code style="color:{CLR['blue']}">data/</code> and this section
                activates automatically.</p>
        </div>""", unsafe_allow_html=True)
    else:
        # Defense data is available — show defender CVI search
        if "defender_name" in defense_df.columns:
            dn=sorted(defense_df["defender_name"].dropna().unique())
        elif "defender_id" in defense_df.columns:
            dn=sorted(defense_df["defender_id"].astype(str).unique())
        else:
            st.warning("Unexpected column structure in defense CSV.")
            render_table(defense_df.head(20)); return

        sel2=st.selectbox("Search defender (landing space)",[""]+list(dn),
                          index=0,placeholder="Type a defender name...",key="t2d_ls")
        if sel2:
            if "defender_name" in defense_df.columns:
                dd2=defense_df[defense_df["defender_name"]==sel2]
            else:
                dd2=defense_df[defense_df["defender_id"].astype(str)==sel2]
            tot2=len(dd2)
            v=int(dd2["cvi_landing_space_flag"].sum()) if "cvi_landing_space_flag" in dd2.columns else 0
            st.markdown(f"""
            <div class="defender-card">
                <h2>{sel2}</h2>
                <div class="stat-row">
                    <div class="stat-item"><div class="num">{tot2}</div><div class="lbl">Defensive Events</div></div>
                    <div class="stat-item"><div class="num" style="color:{CLR['red']}">{v}</div><div class="lbl">Landing Violations</div></div>
                    <div class="stat-item"><div class="num" style="color:{CLR['blue_lt']}">{tot2-v}</div><div class="lbl">Clean Contests</div></div>
                </div>
            </div>""", unsafe_allow_html=True)
            render_table(dd2, max_height=min(400,len(dd2)*40+40))
        else:
            st.info("Select a defender to view landing space violations.", icon="👆")


# =============================================================================
# TAB 3 — EVENT EXPLORER (game-filtered analysis with visualizations)
# =============================================================================

def render_event_explorer(df, pl):
    st.markdown(nba_header(
        "Event Explorer",
        "Game-level foul analysis with deviation visualizations. "
        "Select a game to explore individual foul events and threshold comparisons."),
        unsafe_allow_html=True)

    # Game filter
    games = sorted(df["game_id"].unique())
    game_labels = {g: f"Game {str(g)[-4:]}  ({len(df[df['game_id']==g])} fouls)" for g in games}
    sel_game = st.selectbox(
        "Select a game",
        options=[""]+games,
        format_func=lambda x: game_labels.get(x, "Select a game...") if x else "Select a game...",
        key="t3_game")

    if not sel_game:
        st.info("Select a game above to explore foul events.", icon="👆")

        # Show aggregate stats while no game is selected
        st.markdown('<div class="section-label">Dataset Overview</div>',
                    unsafe_allow_html=True)
        k1,k2,k3 = st.columns(3)
        flag_rate = df["osi_any"].mean()
        k1.metric("Overall Flag Rate", f"{flag_rate:.1%}")
        k2.metric("Avg Z-Score (flagged)",
                  f'{df[df["osi_any"]]["deviation_z_score"].mean():.2f}σ')
        k3.metric("Games Available", len(games))
        return

    gdf = df[df["game_id"]==sel_game].copy()
    n_fouls = len(gdf)
    n_flagged = int(gdf["osi_any"].sum())
    flag_rate = n_flagged / n_fouls if n_fouls else 0

    # Game-level KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Foul Events", n_fouls)
    k2.metric("Flagged", n_flagged)
    k3.metric("Flag Rate", f"{flag_rate:.0%}")
    k4.metric("Players Involved", gdf["offensive_player_id"].nunique())

    # Visualizations side by side
    st.markdown('<div class="section-label">Deviation Analysis</div>',
                unsafe_allow_html=True)

    viz1, viz2 = st.columns([3, 2])
    with viz1:
        st.plotly_chart(chart_deviation_scatter(gdf),
                        use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Each dot is a foul event. Dots above the dashed line exceeded "
                   "the player's personal deviation threshold. Size reflects z-score magnitude.")

    with viz2:
        st.plotly_chart(chart_game_violation_types(gdf),
                        use_container_width=True,
                        config={"displayModeBar": False})

    # Event detail table with expandable rows
    st.markdown('<div class="section-label">All Foul Events</div>',
                unsafe_allow_html=True)

    # Add flag type column
    ft_col = []
    for _, r in gdf.iterrows():
        if not r["osi_any"]:
            ft_col.append("—")
        else:
            t=[]
            if r["osi_leg_kick"]: t.append("Leg Kick")
            if r["osi_brakecheck"]: t.append("Brake-Check")
            if r["osi_jump_into"]: t.append("Jump-Into")
            ft_col.append(", ".join(t) if t else "Flagged")
    gdf["Violation Type"] = ft_col
    gdf["Status"] = gdf["osi_any"].map({True: "⚑ Flagged", False: "✓ Natural"})

    display_cols = {
        "period": "Qtr", "gc_seconds": "Time (s)",
        "offensive_player_name": "Offensive Player",
        "defensive_player_name": "Defender",
        "foul_category": "Foul Type",
        "Status": "Status",
        "Violation Type": "Violation",
        "deviation_z_score": "Z-Score",
        "max_lateral_dev_in": "Lat. Dev (in)",
        "personal_threshold_in": "Threshold (in)",
        "nearest_defender_dist_ft": "Def. Dist (ft)",
    }
    av = {k:v for k,v in display_cols.items() if k in gdf.columns}
    render_table(
        gdf[list(av.keys())].rename(columns=av).sort_values(["Qtr","Time (s)"]),
        max_height=min(500, len(gdf)*38+40))

    # Expandable event detail
    if n_flagged > 0:
        st.markdown('<div class="section-label">Flagged Event Details</div>',
                    unsafe_allow_html=True)
        flagged_events = gdf[gdf["osi_any"]]
        for idx, (_, ev) in enumerate(flagged_events.iterrows()):
            ft_list = []
            if ev["osi_leg_kick"]: ft_list.append("Leg Kick")
            if ev["osi_brakecheck"]: ft_list.append("Brake-Check")
            if ev["osi_jump_into"]: ft_list.append("Jump-Into")
            ftype = " / ".join(ft_list)

            with st.expander(
                f"Q{ev['period']} {ev['gc_seconds']}s — "
                f"{ev['offensive_player_name']} ({ftype}) "
                f"— Z: {ev['deviation_z_score']:.2f}σ"):
                c1,c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class="stats-panel">
                        <div class="row"><span class="k">Violation Type</span><span class="v" style="color:{CLR['red']}">{ftype}</span></div>
                        <div class="row"><span class="k">Offensive Player</span><span class="v">{ev['offensive_player_name']}</span></div>
                        <div class="row"><span class="k">Defender</span><span class="v">{ev.get('defensive_player_name','—')}</span></div>
                        <div class="row"><span class="k">Foul Category</span><span class="v">{ev.get('foul_category','—')}</span></div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class="stats-panel">
                        <div class="row"><span class="k">Lateral Deviation</span><span class="v">{ev.get('max_lateral_dev_in','—')}"</span></div>
                        <div class="row"><span class="k">Personal Threshold</span><span class="v">{ev.get('personal_threshold_in','—')}"</span></div>
                        <div class="row"><span class="k">Z-Score</span><span class="v">{ev.get('deviation_z_score','—')}σ</span></div>
                        <div class="row"><span class="k">Defender Distance</span><span class="v">{ev.get('nearest_defender_dist_ft','—')} ft</span></div>
                    </div>""", unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = load_offense()
    bl = load_baselines()
    pl = load_players()
    ddf = load_defense()

    st.markdown(theme_css, unsafe_allow_html=True)
    render_sidebar(df)

    t0, t1, t2, t3 = st.tabs([
        "Overview",
        "Spatial Initiation",
        "Cylinder Intrusion",
        "Event Explorer"])

    with t0: render_overview(df)
    with t1: render_spatial_initiation(df, bl, pl)
    with t2: render_cylinder_intrusion(df, pl, ddf)
    with t3: render_event_explorer(df, pl)


if __name__ == "__main__":
    main()