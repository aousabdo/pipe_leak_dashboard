"""Dashboard CSS and theming."""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1a365d;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.6rem;
        color: #2a4a7f;
        padding-top: 1rem;
        font-weight: 600;
    }
    .subsection-header {
        font-size: 1.2rem;
        color: #3a6ea5;
        padding-top: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid #3182ce;
    }
    .risk-high { border-left-color: #e53e3e; }
    .risk-medium { border-left-color: #dd6b20; }
    .risk-low { border-left-color: #38a169; }
</style>
"""
