# ui/theme.py

PRIMARY_COLOR = "#1F4E79"   # azul corporativo
PRIMARY_DARK = "#0B1F3B"
TEXT_COLOR = "#1A1F36"
MUTED_TEXT = "#475569"
BG_LIGHT = "#F6F8FB"

def global_css() -> str:
    return f"""
    <style>
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {TEXT_COLOR};
    }}

    h1, h2, h3 {{
        color: {PRIMARY_DARK};
        font-weight: 600;
    }}

    section[data-testid="stSidebar"] {{
        background-color: #F8FAFC;
    }}

    section[data-testid="stSidebar"] nav a {{
        border-radius: 8px;
        margin: 2px 0;
    }}

    section[data-testid="stSidebar"] nav a[aria-current="page"] {{
        background-color: rgba(31,78,121,0.15);
        font-weight: 600;
    }}
    </style>
    """
