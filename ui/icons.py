# ui/icons.py

ICON_COLOR = "#1F4E79"
STROKE = 2

ICONS = {
    "home": f"""
    <svg viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="{STROKE}">
      <path d="M3 9l9-7 9 7"/>
      <path d="M9 22V12h6v10"/>
    </svg>
    """,

    "history": f"""
    <svg viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="{STROKE}">
      <path d="M3 3v18h18"/>
      <path d="M18 9l-5 5-4-4-3 3"/>
    </svg>
    """,

    "simulator": f"""
    <svg viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="{STROKE}">
      <rect x="3" y="4" width="18" height="14"/>
      <path d="M8 20h8"/>
    </svg>
    """,

    "planner": f"""
    <svg viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="{STROKE}">
      <circle cx="12" cy="7" r="4"/>
      <path d="M5.5 21a6.5 6.5 0 0113 0"/>
    </svg>
    """,

    "pricing": f"""
    <svg viewBox="0 0 24 24" fill="none" stroke="{ICON_COLOR}" stroke-width="{STROKE}">
      <path d="M12 1v22"/>
      <path d="M5 6h14"/>
      <path d="M5 18h14"/>
    </svg>
    """,
}

def icon(name: str, size: int = 24) -> str:
    svg = ICONS.get(name, "")
    return f"""
    <div style="width:{size}px;height:{size}px;display:inline-flex;">
        {svg}
    </div>
    """
