"""Dashboard CSS and theming — modern, clean, professional."""

CUSTOM_CSS = """
<style>
    /* ---- Global ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Header ---- */
    .dash-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0c4a6e 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .dash-header h1 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .dash-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ---- Metric Cards ---- */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 8px rgba(0,0,0,0.06);
        border: 1px solid #e2e8f0;
        flex: 1;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .kpi-icon {
        font-size: 1.6rem;
        margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    .kpi-delta {
        font-size: 0.8rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    .kpi-delta.positive { color: #16a34a; }
    .kpi-delta.negative { color: #dc2626; }
    .kpi-delta.neutral { color: #64748b; }

    /* Card accent borders */
    .kpi-card.accent-blue { border-top: 3px solid #3b82f6; }
    .kpi-card.accent-red { border-top: 3px solid #ef4444; }
    .kpi-card.accent-amber { border-top: 3px solid #f59e0b; }
    .kpi-card.accent-green { border-top: 3px solid #22c55e; }
    .kpi-card.accent-purple { border-top: 3px solid #8b5cf6; }
    .kpi-card.accent-cyan { border-top: 3px solid #06b6d4; }

    /* ---- Section Headers ---- */
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
        transition: all 0.15s;
    }

    .sidebar-section {
        background: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .sidebar-section h3 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.8rem;
    }

    /* ---- Risk Badges ---- */
    .risk-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .risk-badge.high { background: #fef2f2; color: #dc2626; }
    .risk-badge.medium { background: #fffbeb; color: #d97706; }
    .risk-badge.low { background: #f0fdf4; color: #16a34a; }

    /* ---- Data Tables ---- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }

    /* ---- Status Banner ---- */
    .status-banner {
        background: linear-gradient(90deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .status-banner.warning {
        background: linear-gradient(90deg, #fffbeb, #fef3c7);
        border-color: #fde68a;
    }
    .status-banner.success {
        background: linear-gradient(90deg, #f0fdf4, #dcfce7);
        border-color: #bbf7d0;
    }

    /* ---- Charts ---- */
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid #f1f5f9;
    }

    /* ---- Hide Streamlit branding ---- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
