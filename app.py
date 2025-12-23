# app.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st

# módulos do projeto (devem existir)
from tesouro_data import TesouroDiretoRepository, VnaRepository
from renda_mais_domain import RendaMaisSimulator, RendaMaisOperation, AposentadoriaPlanner, compute_retirement_metrics
from pricing_domain import IPCAIndexedPricer, TesouroPriceMatcher, BondSpec

# desing UI themes/icons
from ui.theme import global_css
from ui.icons import icon
from ui.kpi import kpi_card
import streamlit.components.v1 as components

st.markdown(global_css(), unsafe_allow_html=True)

# -----------------------
# Config geral / tema
# -----------------------
st.set_page_config(
    page_title="Ferramentas Tesouro Direto – RendA+, IPCA+ & Precificação",
    layout="wide",
)

pio.templates.default = "plotly_white"

CUSTOM_COLORS = [
    "#0B1F3B",
    "#1F4E79",
    "#2F6FA3",
    "#5A8BB5",
    "#8FA4C3",
    "#B0BECF",
    "#D0D7E3",
]

st.markdown(
    """
    <style>
    .main { background-color: #F6F8FB !important; }
    html, body, [class*="css"]  { font-family: "Segoe UI", Roboto, sans-serif; color: #1A1F36; }
    h1, h2, h3 { color: #0B1F3B !important; font-weight: 600 !important; }
    .big-button {
        display: inline-block;
        margin: 8px 8px;
        padding: 18px 26px;
        background: linear-gradient(90deg,#1F4E79,#2F6FA3);
        color: white !important;
        font-weight: 600;
        border-radius: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# CACHE: repositórios e dados
# -----------------------
@st.cache_resource
def get_tesouro_repo() -> TesouroDiretoRepository:
    return TesouroDiretoRepository()


@st.cache_resource
def get_vna_repo() -> VnaRepository:
    return VnaRepository()


@st.cache_data(ttl=3600, show_spinner=False)
def load_and_prepare_history() -> pd.DataFrame:
    """
    Baixa e prepara o histórico do Tesouro Direto.
    O processamento é vetorizado para performance.
    """
    repo = get_tesouro_repo()
    df = repo.load_history_standardized()

    # garantir tipos e ordenar
    df["Data_Base"] = pd.to_datetime(df["Data_Base"], errors="coerce")
    df["Data_Vencimento"] = pd.to_datetime(df["Data_Vencimento"], errors="coerce")
    df = df.dropna(subset=["Data_Base", "PU", "TituloRaw"]).sort_values("Data_Base").reset_index(drop=True)

    # Ano de vencimento e ajuste para renda+/educa+
    df["Ano_Vencimento"] = df["Data_Vencimento"].dt.year

    titulo_lower = df["TituloRaw"].astype(str).str.lower()
    mask_renda = titulo_lower.str.contains("renda", na=False)
    mask_educa = titulo_lower.str.contains("educa", na=False)

    df["Ano_Ajustado"] = df["Ano_Vencimento"]
    df.loc[mask_renda, "Ano_Ajustado"] = df.loc[mask_renda, "Ano_Vencimento"] - 19
    df.loc[mask_educa, "Ano_Ajustado"] = df.loc[mask_educa, "Ano_Vencimento"] - 4

    df["Ano_Ajustado_int"] = df["Ano_Ajustado"].astype("Int64")

    df["Titulo"] = df["TituloRaw"].astype(str)
    has_ano = df["Ano_Ajustado_int"].notna()
    df.loc[has_ano, "Titulo"] = df.loc[has_ano, "TituloRaw"].astype(str) + " " + df.loc[has_ano, "Ano_Ajustado_int"].astype(str)

    # Taxa média vetorizada
    if "Taxa_Compra" in df.columns and "Taxa_Venda" in df.columns:
        df["Taxa_media"] = (df["Taxa_Compra"] + df["Taxa_Venda"]) / 2.0
    elif "Taxa_Compra" in df.columns:
        df["Taxa_media"] = df["Taxa_Compra"]
    else:
        df["Taxa_media"] = pd.NA

    # opcional: reduzir colunas usadas pela UI (economia de memória)
    keep_cols = [
        "Data_Base",
        "TituloRaw",
        "Titulo",
        "Data_Vencimento",
        "Ano_Vencimento",
        "Ano_Ajustado",
        "PU",
        "Taxa_media",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols + [c for c in df.columns if c not in keep_cols]]

    return df


def clear_all_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


# ==============================
# Páginas (OOP)
# ==============================
@dataclass
class BasePage:
    title: str

    def render(self) -> None:
        raise NotImplementedError


import streamlit.components.v1 as components

class HomePage(BasePage):
    def __init__(self, pages_meta: list[tuple[str, str]]):
        super().__init__(title="HOME")
        self.pages_meta = pages_meta


    def render(self) -> None:
        components.html(
            f"""
            <style>
            .container {{
                max-width: 1100px;
                padding: 20px 10px;
            }}
            .cards {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
                gap: 24px;
            }}
            .card {{
                background: linear-gradient(180deg, #FFFFFF, #F6F8FB);
                border-radius: 18px;
                padding: 24px;
                border: 1px solid rgba(11,31,59,0.08);
                box-shadow: 0 10px 30px rgba(11,31,59,0.08);
                transition: transform .18s ease, box-shadow .18s ease;
            }}
            .card:hover {{
                transform: translateY(-6px);
                box-shadow: 0 18px 45px rgba(11,31,59,0.14);
            }}
            .hint {{
                margin-top: 28px;
                font-size: 14px;
                color: #64748B;
            }}
            </style>

            <div class="container">
              <h1>Planejamento Financeiro com Tesouro Direto</h1>
              <p style="font-size:18px;color:#475569;margin-bottom:32px;">
                Ferramentas para análise de preços, fluxo de caixa e planejamento de renda real.
              </p>

              <div class="cards">
                <div class="card">{icon("history",42)}<h3>Histórico de Preços</h3><p>Séries históricas de PU e taxas reais.</p></div>
                <div class="card">{icon("simulator",42)}<h3>Simulador de Fluxo</h3><p>Fluxos mensais de renda do RendA+.</p></div>
                <div class="card">{icon("planner",42)}<h3>Planejador</h3><p>Alocação ótima para renda real.</p></div>
                <div class="card">{icon("pricing",42)}<h3>Precificador</h3><p>Preço teórico vs Tesouro Direto.</p></div>
              </div>

              <div class="hint">👉 Use o menu lateral para navegar</div>
            </div>
            """,
            height=620,
        )



class HistoricoPage(BasePage):
    def __init__(self):
        super().__init__(title="Histórico de Preços")

    def render(self) -> None:
        components.html(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                {icon("history", 30)}
                <h2 style="margin:0;">{self.title}</h2>
            </div>
            """,
            height=60,
        )
        st.markdown("Escolha os filtros abaixo e clique em **Aplicar filtros**.")

        with st.spinner("Carregando histórico do Tesouro Direto (cache)..."):
            df = load_and_prepare_history()
        st.success("Dados prontos (cache).")

        titulos_unicos = sorted(df["Titulo"].drop_duplicates().tolist())

        with st.form("form_hist_main"):
            c1, c2 = st.columns(2)
            with c1:
                modo_dado = st.radio(
                    "Série para visualizar",
                    options=["PU", "Taxa (média compra/venda)"],
                    index=0,
                )
            with c2:
                periodo = st.radio("Período", options=["ALL", "5Y", "1Y"], index=0)

            termo_busca = st.text_input("Buscar por nome do título (string)")

            if termo_busca.strip():
                titulos_filtrados = [t for t in titulos_unicos if termo_busca.lower() in t.lower()]
            else:
                titulos_filtrados = titulos_unicos

            titulos_selecionados = st.multiselect(
                "Selecione os títulos para plotar",
                options=titulos_filtrados,
                default=(titulos_filtrados[:3] if len(titulos_filtrados) >= 3 else titulos_filtrados),
            )

            aplicar = st.form_submit_button("✅ Aplicar filtros")

        if not aplicar:
            st.info("Preencha os filtros e clique em **Aplicar filtros** para ver o gráfico.")
            return

        if not titulos_selecionados:
            st.warning("Selecione ao menos um título.")
            return

        df_sel = df[df["Titulo"].isin(titulos_selecionados)].copy()
        if df_sel.empty:
            st.warning("Nenhum dado para os títulos selecionados.")
            return

        data_max = df_sel["Data_Base"].max().date()
        if periodo == "ALL":
            data_min = df_sel["Data_Base"].min().date()
        elif periodo == "5Y":
            try:
                data_min = data_max.replace(year=data_max.year - 5)
            except ValueError:
                data_min = data_max - timedelta(days=5 * 365)
        else:
            try:
                data_min = data_max.replace(year=data_max.year - 1)
            except ValueError:
                data_min = data_max - timedelta(days=365)

        mask = (df_sel["Data_Base"].dt.date >= data_min) & (df_sel["Data_Base"].dt.date <= data_max)
        df_sel = df_sel[mask]

        if df_sel.empty:
            st.warning("Não há dados no período selecionado para esses títulos.")
            return

        if modo_dado == "PU":
            serie_col = "PU"
            y_label = "PU (Preço unitário)"
            titulo_graf = "Histórico de Preços (PU)"
            df_plot = df_sel[["Data_Base", "Titulo", "PU"]].copy()
        else:
            if df_sel["Taxa_media"].isna().all():
                st.warning("Não há informações de taxa média nesse arquivo para os títulos selecionados.")
                return
            serie_col = "Taxa_media"
            y_label = "Taxa média (% a.a.)"
            titulo_graf = "Histórico de Taxas – média compra/venda"
            df_plot = df_sel[["Data_Base", "Titulo", "Taxa_media"]].copy().dropna()

        df_plot = df_plot.sort_values("Data_Base")

        # limitar número de séries para performance
        max_series = 12
        unique_titles = df_plot["Titulo"].unique()
        if len(unique_titles) > max_series:
            st.info(f"{len(unique_titles)} séries selecionadas — mostrando as primeiras {max_series} por performance.")
            titles_to_keep = unique_titles[:max_series]
            df_plot = df_plot[df_plot["Titulo"].isin(titles_to_keep)]

        st.subheader(f"{titulo_graf} – Período: {periodo}")
        fig = px.line(
            df_plot,
            x="Data_Base",
            y=serie_col,
            color="Titulo",
            labels={"Data_Base": "Data", serie_col: y_label, "Titulo": "Título"},
            template="plotly_white",
            color_discrete_sequence=CUSTOM_COLORS,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Estatísticas resumidas
        st.subheader("Estatísticas por título")
        stats = []
        for t in titulos_selecionados:
            df_t = df_sel[df_sel["Titulo"] == t]
            if df_t.empty:
                continue
            serie = df_t[serie_col]
            stats.append({
                "Titulo": t,
                "Primeira data": df_t["Data_Base"].min().date(),
                "Última data": df_t["Data_Base"].max().date(),
                "Valor mínimo": serie.min(),
                "Valor máximo": serie.max(),
                "Valor mais recente": serie.iloc[-1],
            })

        if stats:
            df_stats = pd.DataFrame(stats)
            if modo_dado == "PU":
                fmt = "R$ {:,.2f}".format
            else:
                fmt = "{:,.2f}%".format
            st.dataframe(df_stats.style.format({"Valor mínimo": fmt, "Valor máximo": fmt, "Valor mais recente": fmt}))

        csv = df_sel.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Baixar dados filtrados (CSV)", data=csv, file_name="historico_tesouro_filtrado.csv", mime="text/csv")


class SimuladorPage(BasePage):
    def __init__(self, simulator: RendaMaisSimulator):
        super().__init__(title="Simulador de Fluxo (RendA+)")
        self.simulator = simulator

    def render(self) -> None:
        components.html(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                {icon("simulator", 30)}
                <h2 style="margin:0;">{self.title}</h2>
            </div>
            """,
            height=60,
        )

        vencimentos_validos = [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065]
        hoje = date.today()
        data_padrao = date(hoje.year, hoje.month, min(hoje.day, 28))

        with st.form("form_simulador_fluxo"):
            ipca_padrao = st.number_input(
                "IPCA médio esperado (a.a., %) – aplicado a todas as operações",
                min_value=-5.0,
                max_value=20.0,
                value=0.0,
                step=0.50,
            )

            n_operacoes = int(st.number_input("Número de operações (compras diferentes)", min_value=1, max_value=20, value=2, step=1))

            st.subheader("Operações – parâmetros em tabela")
            dados_ops = []
            for i in range(n_operacoes):
                dados_ops.append({
                    "Operação": i + 1,
                    "Data compra": pd.to_datetime(data_padrao),
                    "Ano conversão": vencimentos_validos[min(i, len(vencimentos_validos) - 1)],
                    "Taxa real (%)": 7.0,
                    "Valor investido (R$)": 10000.0 if i == 0 else 0.0,
                })
            df_ops = pd.DataFrame(dados_ops)

            edited_ops = st.data_editor(
                df_ops,
                column_config={
                    "Operação": st.column_config.NumberColumn("Operação", disabled=True),
                    "Data compra": st.column_config.DateColumn("Data compra"),
                    "Ano conversão": st.column_config.SelectboxColumn("Ano conversão", options=vencimentos_validos),
                    "Taxa real (%)": st.column_config.NumberColumn("Taxa real (%)", step=0.25, format="%.2f"),
                    "Valor investido (R$)": st.column_config.NumberColumn("Valor investido (R$)", min_value=0.0, step=1000.0, format="%.2f"),
                },
                hide_index=True,
            )

            mostrar_liquido = st.checkbox("Mostrar fluxo líquido de IR no gráfico", value=True)
            run_sim = st.form_submit_button("🚀 Rodar simulação de fluxo")

        if not run_sim:
            return

        dfs = []
        resumos = []
        for _, row in edited_ops.iterrows():
            valor_inv = float(row["Valor investido (R$)"])
            if valor_inv <= 0:
                continue

            data_compra_val = row["Data compra"]
            if hasattr(data_compra_val, "date"):
                data_compra_val = data_compra_val.date()

            ano_conv = int(row["Ano conversão"])
            taxa_real_pct = float(row["Taxa real (%)"])
            nome_titulo = f"RendA+ {ano_conv} – Op {int(row['Operação'])}"

            op = RendaMaisOperation(
                data_compra=data_compra_val,
                ano_conversao=ano_conv,
                taxa_real_aa=taxa_real_pct / 100.0,
                ipca_aa=ipca_padrao / 100.0,
                valor_investido=valor_inv,
                nome_titulo=nome_titulo,
            )

            df_op, resumo_op = self.simulator.simulate_operation(op)
            dfs.append(df_op.assign(titulo=nome_titulo))
            resumos.append(resumo_op)

        if not dfs:
            st.warning("Nenhuma operação válida (todos os valores investidos são zero).")
            return

        df_tudo = pd.concat(dfs, ignore_index=True)

        st.subheader("Resumo por operação")
        df_resumo = pd.DataFrame(resumos)
        cols_res = ["titulo", "data_compra", "ano_conversao", "valor_investido", "meses_ate_conversao", "valor_real_conversao", "valor_nominal_conversao", "total_bruto", "total_ir", "total_liquido"]
        st.dataframe(df_resumo[cols_res].style.format({
            "valor_investido": "R$ {:,.2f}".format,
            "valor_real_conversao": "R$ {:,.2f}".format,
            "valor_nominal_conversao": "R$ {:,.2f}".format,
            "total_bruto": "R$ {:,.2f}".format,
            "total_ir": "R$ {:,.2f}".format,
            "total_liquido": "R$ {:,.2f}".format,
        }))

        st.subheader("Fluxo consolidado por data de pagamento")
        df_fluxo = df_tudo.groupby("data_pagamento", as_index=False).agg({"parcela_nominal_bruta": "sum","ir": "sum","parcela_nominal_liquida": "sum"})
        csv = df_fluxo.to_csv(index=False).encode("utf-8")
        st.download_button(label="📥 Baixar fluxo consolidado (CSV)", data=csv, file_name="fluxo_renda_mais_consolidado.csv", mime="text/csv")

        y_col = "parcela_nominal_liquida" if mostrar_liquido else "parcela_nominal_bruta"
        titulo_graf = ("Fluxo mensal líquido de IR" if mostrar_liquido else "Fluxo mensal bruto")

        fig = px.line(df_fluxo, x="data_pagamento", y=y_col, title=titulo_graf, labels={"data_pagamento": "Data de pagamento", y_col: "Valor (R$)"}, template="plotly_white", color_discrete_sequence=CUSTOM_COLORS)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Detalhe por título / operação")
        titulos = df_tudo["titulo"].unique().tolist()
        titulo_sel = st.selectbox("Escolha um título/operacão para detalhar:", options=titulos)
        df_det = df_tudo[df_tudo["titulo"] == titulo_sel].copy()
        st.dataframe(df_det.head(30).style.format({
            "parcela_nominal_bruta": "R$ {:,.2f}".format,
            "juros_nominal": "R$ {:,.2f}".format,
            "amortizacao_nominal": "R$ {:,.2f}".format,
            "ir": "R$ {:,.2f}".format,
            "parcela_nominal_liquida": "R$ {:,.2f}".format,
        }))


class PlanejadorPage(BasePage):
    def __init__(self, planner: AposentadoriaPlanner):
        super().__init__(title="Planejador de Aposentadoria")
        self.planner = planner

    def render(self) -> None:
        components.html(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                {icon("planner", 30)}
                <h2 style="margin:0;">{self.title}</h2>
            </div>
            """,
            height=60,
        )
        hoje = date.today()

        with st.form("form_planejador"):
            c1, c2 = st.columns(2)
            with c1:
                idade_atual = st.number_input("Idade atual", min_value=18, max_value=90, value=30, step=1)
                idade_aposentadoria = st.number_input(
                    "Idade em que deseja se aposentar",
                    min_value=idade_atual + 1,
                    max_value=100,
                    value=45,
                    step=1,
                )
                idade_fim_recebimento = st.number_input(
                    "Até que idade espera receber renda?",
                    min_value=idade_aposentadoria + 1,
                    max_value=120,
                    value=90,
                    step=1,
                    help="Horizonte para garantir a renda média solicitada (anos)."
                )            
            with c2:
                renda_desejada = st.number_input(
                    "Renda mensal REAL líquida desejada (R$)",
                    min_value=0.0,
                    value=10000.0,
                    step=1000.0,
                    format="%.2f",
                )
                st.number_input(
                    "IPCA esperado (a.a., %, apenas informativo)",
                    min_value=-5.0,
                    max_value=20.0,
                    value=0.0,
                    step=0.50,
                )

            st.subheader("Parâmetros dos títulos RendA+")
            vencimentos_validos = [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065]
            dados_venc = []
            for ano in vencimentos_validos:
                dados_venc.append(
                    {
                        "Ano conversão": ano,
                        "Usar": ano >= hoje.year + 5,
                        "Taxa real (%)": 7.0,
                    }
                )
            df_venc = pd.DataFrame(dados_venc)

            edited_venc = st.data_editor(
                df_venc,
                column_config={
                    "Ano conversão": st.column_config.NumberColumn("Ano conversão", disabled=True),
                    "Usar": st.column_config.CheckboxColumn("Usar"),
                    "Taxa real (%)": st.column_config.NumberColumn("Taxa real (%)", step=0.25, format="%.2f"),
                },
                hide_index=True,
            )

            run_plan = st.form_submit_button("📐 Calcular alocação sugerida")

        if not run_plan:
            return

        df_usar = edited_venc[edited_venc["Usar"] == True].copy()
        if df_usar.empty:
            st.warning("Selecione ao menos um vencimento (coluna 'Usar').")
            return

        venc_sel = df_usar["Ano conversão"].astype(int).tolist()
        taxas_sel = (df_usar["Taxa real (%)"] / 100.0).astype(float).tolist()



        try:
            df_aloc, df_renda_otimizacao, df_fluxo_real = self.planner.montar_alocacao(
                idade_atual=int(idade_atual),
                idade_aposentadoria=int(idade_aposentadoria),
                idade_fim_recebimento=int(idade_fim_recebimento),
                renda_real_desejada=float(renda_desejada),
                hoje=hoje,
                vencimentos=venc_sel,
                taxas_reais_aa=taxas_sel,
            )

            metrics = compute_retirement_metrics(
                df_renda=df_fluxo_real.rename(
                    columns={
                        "data_pagamento": "data",
                        "parcela_nominal_liquida": "renda_real_liquida",
                    }
                ),
                renda_objetivo=renda_desejada,
                idade_atual=int(idade_atual),
                idade_aposentadoria=int(idade_aposentadoria),
                idade_fim_recebimento=int(idade_fim_recebimento),
                hoje=hoje,
            )


            # -----------------------------
            # Table: Alocação sugerida
            # -----------------------------
            st.header("Resultado da otimização")
            st.subheader("Sugestão de alocação para fluxo mensal (reais de hoje)")
            st.dataframe(df_aloc.style.format({"taxa_real_anual": "{:.2%}".format, "investimento_sugerido": "R$ {:,.2f}".format}))


            # -----------------------------
            # Card: Linha do tempo 
            # -----------------------------
            
            eventos_timeline = [
                {
                    "label": "Hoje",
                    "data": hoje,
                    "idade": idade_atual,
                    "ano": hoje.year,
                },
                {
                    "label": "Início da renda",
                    "data": metrics["data_inicio"],
                    "idade": metrics["idade_inicio_real"],
                    "ano": metrics["ano_inicio_pagamento"],
                },
                {
                    "label": "Aposentadoria",
                    "data": metrics["data_aposentadoria"],
                    "idade": idade_aposentadoria,
                    "ano": metrics["data_aposentadoria"].year,
                },
                {
                    "label": "Fim da renda",
                    "data": metrics["data_fim"],
                    "idade": metrics["idade_fim_real"],
                    "ano": metrics["ano_fim_pagamento"],
                },
            ]

            eventos_timeline = sorted(eventos_timeline, key=lambda e: e["data"])

            timeline_html = """
            <div class="timeline">
            """
            for e in eventos_timeline:
                timeline_html += f"""
                <div class="milestone">
                    <div class="dot"></div>
                    <div class="value">{e['idade']} anos</div>
                    <div class="label">{e['label']} ({e['ano']})</div>
                </div>
                """
            timeline_html += "</div>"

            components.html(
                f"""
                <style>
                    .timeline-wrapper {{
                        margin-top: 12px;
                    }}

                    .timeline-title {{
                        font-weight: 600;
                        margin-bottom: 12px;
                    }}

                    .timeline {{
                        display: flex;
                        align-items: center;
                        justify-content: space-between;
                        position: relative;
                        margin-top: 10px;
                    }}

                    .timeline::before {{
                        content: "";
                        position: absolute;
                        top: 22px;
                        left: 0;
                        right: 0;
                        height: 2px;
                        background: #CBD5E1;
                        z-index: 0;
                    }}

                    .milestone {{
                        position: relative;
                        text-align: center;
                        z-index: 1;
                        background: #F8FAFC;
                        padding: 10px 14px;
                        border-radius: 10px;
                        min-width: 120px;
                        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
                    }}

                    .dot {{
                        width: 12px;
                        height: 12px;
                        background: #1F4E79;
                        border-radius: 50%;
                        margin: 0 auto 6px auto;
                    }}

                    .value {{
                        font-weight: 700;
                        font-size: 14px;
                    }}

                    .label {{
                        font-size: 12px;
                        color: #475569;
                    }}
                </style>

                <div class="timeline-wrapper">
                    <h3 class="timeline-title">Timeline</h3>
                    {timeline_html}
                </div>
                """,
                height=150,
            )

            # -----------------------------
            # Card: Resumo da Aposentadoria
            # -----------------------------
            investimento_total = df_aloc["investimento_sugerido"].sum()

            components.html(
                f"""
                <h3>Resumo da Aposentadoria</h3>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;">
                    {kpi_card("Início da renda", f"{metrics['idade_inicio_real']} anos", f"Ano {metrics['ano_inicio_pagamento']}")}
                    {kpi_card("Início da aposentadoria", f"{idade_aposentadoria} anos", f"Ano {metrics["data_aposentadoria"].year}")}
                    {kpi_card("Fim da renda", f"{metrics['idade_fim_real']} anos", f"Ano {metrics['ano_fim_pagamento']}")}
                    {kpi_card("Duração total", f"{metrics['duracao_anos']} anos", f"{metrics['duracao_meses']} meses")}
                </div>
                """,
                height=180,
            )

            # -----------------------------
            # Card: Metricas Renda Mensal
            # -----------------------------
            components.html(
                f"""
                <h3>Renda Real Mensal (considerando apenas o período após aposentadoria)</h3>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-bottom:20px;">
                    {kpi_card("Investimento total", f"R$ {investimento_total:,.2f}")}
                    {kpi_card("Renda média", f"R$ {metrics['renda_media']:,.0f}")}
                    {kpi_card("Renda mínima", f"R$ {metrics['renda_min']:,.0f}")}
                    {kpi_card("Renda máxima", f"R$ {metrics['renda_max']:,.0f}")}
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;;margin-bottom:20px;">
                    {kpi_card("Meses abaixo do alvo", f"{metrics['meses_abaixo_objetivo']}", f"{(metrics['meses_abaixo_objetivo']/metrics['duracao_meses'])*100:.1f}% do total")}
                    {kpi_card("Pior mês", f"R$ {metrics['pior_gap_abs']:,.0f}", f"{metrics['pior_gap_pct']*100:.1f}% vs alvo")}
                </div>

                """,
                height=270,
            )


            # -----------------------------
            # Table: Grafico do fluxo
            # -----------------------------
            st.subheader("Renda mensal REAL líquida simulada")

            fig = px.line(
                df_fluxo_real.rename(
                columns={
                    "data_pagamento": "data",
                    "parcela_nominal_liquida": "renda_real_liquida",
                }),
                x="data",
                y="renda_real_liquida",title="Renda mensal real líquida durante a aposentadoria", labels={"data": "Data", "renda_real_liquida": "Renda (R$)"}, template="plotly_white", color_discrete_sequence=CUSTOM_COLORS)
            fig.add_hline(
                y=renda_desejada,
                line_dash="dash",
                line_color="#EF4444",  # vermelho corporativo
                line_width=2,
                annotation_text="Renda desejada",
                annotation_position="top left",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**Renda média:** R$ {metrics['renda_media']:,.2f} – **mín:** R$ {metrics['renda_min']:,.2f} – **máx:** R$ {metrics['renda_max']:,.2f}")
        

        except ValueError as e:
            st.error(str(e))
            return



class PrecificadorPage(BasePage):
    def __init__(self, pricer: IPCAIndexedPricer, matcher: TesouroPriceMatcher):
        super().__init__(title="Precificador (IPCA+ / RendA+)")
        self.pricer = pricer
        self.matcher = matcher

    def render(self) -> None:
        components.html(
            f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
                {icon("pricer", 30)}
                <h2 style="margin:0;">{self.title}</h2>
            </div>
            """,
            height=60,
        )
        hoje = date.today()

        with st.form("form_precificador"):
            tipo_label = st.selectbox("Tipo de título", ["Tesouro IPCA+ (NTN-B Principal)", "Tesouro RendA+"])
            tipo_interno = "NTNB_PRINCIPAL" if "IPCA+" in tipo_label else "RENDA_MAIS"

            c1, c2 = st.columns(2)
            with c1:
                data_compra = st.date_input("Data de compra", value=hoje)
                data_calculo = st.date_input("Data de cálculo", value=hoje)
            with c2:
                taxa_compra_pct = st.number_input("Taxa real acordada na compra (a.a., %)", min_value=-5.0, max_value=25.0, value=6.0, step=0.25)
                taxa_atual_pct = st.number_input("Taxa real de mercado atual (a.a., %)", min_value=-5.0, max_value=25.0, value=6.5, step=0.25)

            data_vencimento = None
            ano_conversao = None
            if tipo_interno == "NTNB_PRINCIPAL":
                data_vencimento = st.date_input("Data de vencimento (Tesouro IPCA+ Principal)", value=date(2035, 5, 15))
            else:
                ano_conversao = st.selectbox("Ano de conversão (RendA+)", [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065], index=3)

            run = st.form_submit_button("📊 Calcular preço e sensibilidade")

        if not run:
            return

        try:
            df_vna = get_vna_repo().load_vna_ntnb()
            spec = BondSpec(tipo=tipo_interno, data_vencimento=data_vencimento, ano_conversao=ano_conversao)
            result = self.pricer.price(
                spec=spec,
                data_compra=data_compra,
                data_calculo=data_calculo,
                taxa_compra_real_aa=taxa_compra_pct / 100.0,
                taxa_atual_real_aa=taxa_atual_pct / 100.0,
            )
        except Exception as e:
            st.error(f"Erro ao precificar: {e}")
            return

        st.subheader("Preço teórico (modelo)")
        if tipo_interno == "NTNB_PRINCIPAL":
            preco_modelo = result.get("preco_atual")
            vna_compra = result.get("vna_compra")
            vna_calc = result.get("vna_atual")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Preço teórico hoje (por 1 unidade nominal)", f"R$ {preco_modelo:,.2f}")
                st.metric("VNA na data de compra", f"R$ {vna_compra:,.4f}")
            with c2:
                st.metric("VNA na data de cálculo", f"R$ {vna_calc:,.4f}")
        else:
            preco_modelo = result.get("preco_nominal_atual_por_1_real_investido")
            pv_real = result.get("pv_real_atual")
            pmt_real = result.get("pmt_real")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Preço nominal hoje (por R$1,00 investido)", f"R$ {preco_modelo:,.4f}")
            with c2:
                st.metric("PV real dos fluxos (hoje)", f"{pv_real:,.4f}")
            with c3:
                st.metric("PMT real estimada (fase de renda)", f"{pmt_real:,.4f}/mês")

        st.subheader("Comparação: modelo × preço oficial Tesouro Direto")
        preco_oficial = self.matcher.find_official_price(spec=spec, data_calculo=data_calculo)
        if preco_oficial is None:
            st.warning("Não encontrei preço oficial correspondente no CSV do Tesouro Direto para esse tipo/vencimento.")
        else:
            df_bar = pd.DataFrame({"Fonte": ["Modelo", "Tesouro Direto"], "Preço": [preco_modelo, preco_oficial]})
            fig_bar = px.bar(df_bar, x="Fonte", y="Preço", text="Preço", title="Preço teórico vs. preço oficial", labels={"Preço": "Preço (R$)"}, color="Fonte", color_discrete_sequence=CUSTOM_COLORS[:2])
            fig_bar.update_traces(texttemplate="R$ %{y:,.2f}", textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Sensibilidade do preço à taxa real (DV01 caseiro)")
        df_curve = self.pricer.price_curve_vs_rate(spec=spec, data_compra=data_compra, data_calculo=data_calculo, taxa_compra_real_aa=taxa_compra_pct / 100.0, taxa_central_real_aa=taxa_atual_pct / 100.0, span_pct=0.02, n_pontos=21)
        fig_curve = px.line(df_curve, x="taxa_real_pct", y="preco_teorico", title="Preço teórico em função da taxa real (±2 p.p.)", labels={"taxa_real_pct": "Taxa real (a.a., %)", "preco_teorico": "Preço (R$)"}, template="plotly_white", color_discrete_sequence=CUSTOM_COLORS[:1])
        st.plotly_chart(fig_curve, use_container_width=True)


# -----------------------
# App principal (navegação lateral)
# -----------------------
class TesouroApp:
    def __init__(self) -> None:
        # Instanciar repositórios e domínios (uma vez)
        self.tesouro_repo = get_tesouro_repo()
        self.vna_repo = get_vna_repo()
        self.simulator = RendaMaisSimulator()
        self.planner = AposentadoriaPlanner(self.simulator)
        self.pricer = IPCAIndexedPricer(self.vna_repo)
        self.matcher = TesouroPriceMatcher(self.tesouro_repo)

        # páginas: HOME primeiro
        # slugs must be unique
        self.page_slugs = ["home", "historico", "simulador", "planejador", "precificador"]
        # map slugs -> title for home buttons
        pages_meta = [(s, t) for s, t in zip(self.page_slugs[1:], ["Histórico de Preços", "Simulador de Fluxo", "Planejador de Aposentadoria", "Precificador"])]

        self.pages = [
            HomePage(pages_meta),
            HistoricoPage(),
            SimuladorPage(self.simulator),
            PlanejadorPage(self.planner),
            PrecificadorPage(self.pricer, self.matcher),
        ]

    def run(self) -> None:
        pages_def = []
        for page, slug in zip(self.pages, self.page_slugs):
            pages_def.append(st.Page(page.render, title=page.title, url_path=slug))
        # navigation with default position = sidebar (lateral)
        nav = st.navigation(pages_def)
        nav.run()


def main():
    app = TesouroApp()
    app.run()


if __name__ == "__main__":
    main()
