def kpi_card(title: str, value: str, subtitle: str = "") -> str:
    return f"""
    <div style="
        background: linear-gradient(180deg,#FFFFFF,#F6F8FB);
        border-radius:14px;
        padding:16px;
        border:1px solid rgba(11,31,59,0.08);
        box-shadow:0 8px 20px rgba(11,31,59,0.08);
    ">
        <div style="font-size:13px;color:#64748B;margin-bottom:4px;">{title}</div>
        <div style="font-size:22px;font-weight:600;color:#0B1F3B;">{value}</div>
        <div style="font-size:12px;color:#64748B;margin-top:2px;">{subtitle}</div>
    </div>
    """
