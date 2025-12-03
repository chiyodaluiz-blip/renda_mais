from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------
# Utilidades de data
# -----------------------------------------

def adicionar_meses(d: date, meses: int) -> date:
    """Soma 'meses' a uma data, cuidando de ano/mês e limitando o dia a 28."""
    ano = d.year + (d.month - 1 + meses) // 12
    mes = (d.month - 1 + meses) % 12 + 1
    dia = min(d.day, 28)
    return date(ano, mes, dia)


def meses_entre(inicio: date, fim: date) -> int:
    """Número aproximado de meses inteiros entre duas datas."""
    return (fim.year - inicio.year) * 12 + (fim.month - inicio.month)


# -----------------------------------------
# Taxas e IR
# -----------------------------------------

def taxa_anual_para_mensal(taxa_anual: float) -> float:
    """
    Converte taxa efetiva anual em efetiva mensal.
    Exemplo: 0.06 -> (1+0.06)**(1/12)-1
    """
    return (1 + taxa_anual) ** (1 / 12) - 1


def aliquota_ir_renda_fixa(dias: int) -> float:
    """
    Tabela regressiva padrão de renda fixa (Imposto de Renda):

      - até 180 dias:    22,5%
      - 181 a 360 dias:  20,0%
      - 361 a 720 dias:  17,5%
      - acima de 720:    15,0%
    """
    if dias <= 180:
        return 0.225
    elif dias <= 360:
        return 0.20
    elif dias <= 720:
        return 0.175
    else:
        return 0.15


# -----------------------------------------
# Simulação de UMA operação do RendA+
# (simulador de fluxo)
# -----------------------------------------

def simular_operacao_renda_mais(
    data_compra: date,
    ano_conversao: int,
    taxa_real_aa: float,
    ipca_aa: float,
    valor_investido: float,
    nome_titulo: str = None,
):
    """
    Simula uma compra única de Tesouro RendA+ com:
      - taxa real (IPCA + X% a.a.)
      - IPCA médio esperado
      - valor investido único

    Retorna:
      - df: DataFrame com 240 fluxos mensais
      - resumo: dicionário com estatísticas da operação
    """
    if nome_titulo is None:
        nome_titulo = f"Tesouro RendA+ {ano_conversao}"

    data_conversao = date(ano_conversao, 1, 15)

    if data_conversao <= data_compra:
        raise ValueError("A data de conversão deve ser posterior à data de compra.")

    meses_ate_conversao = meses_entre(data_compra, data_conversao)
    if meses_ate_conversao <= 0:
        raise ValueError("Meses até a conversão deve ser positivo.")

    # Taxas mensais
    taxa_real_m = taxa_anual_para_mensal(taxa_real_aa)
    ipca_m = taxa_anual_para_mensal(ipca_aa)

    # 1) Valor REAL na data de conversão
    fator_real = (1 + taxa_real_m) ** meses_ate_conversao
    valor_real_conversao = valor_investido * fator_real

    # 2) Valor NOMINAL aproximado na conversão (considerando IPCA)
    indice_ipca_conv = (1 + ipca_m) ** meses_ate_conversao
    valor_nominal_conversao = valor_real_conversao * indice_ipca_conv

    # 3) Parcela REAL (anuidade em 240 meses) -> Calculo da PMT
    n_parcelas = 240
    parcela_real = (
        valor_real_conversao
        * taxa_real_m
        / (1 - (1 + taxa_real_m) ** -n_parcelas)
    )

    # 4) Geração do fluxo nominal com IR
    saldo_real = valor_real_conversao
    registros = []

    for k in range(1, n_parcelas + 1):
        data_pagamento = adicionar_meses(data_conversao, k - 1)
        dias_da_compra = (data_pagamento - data_compra).days
        aliq_ir = aliquota_ir_renda_fixa(dias_da_compra)

        juros_real = saldo_real * taxa_real_m
        amort_real = parcela_real - juros_real
        saldo_real = max(saldo_real - amort_real, 0.0)

        meses_totais = meses_ate_conversao + (k - 1)
        indice_ipca_k = (1 + ipca_m) ** meses_totais

        parcela_nominal = parcela_real * indice_ipca_k
        juros_nominal = juros_real * indice_ipca_k
        amort_nominal = amort_real * indice_ipca_k

        ir = aliq_ir * juros_nominal
        parcela_liquida = parcela_nominal - ir

        registros.append(
            {
                "titulo": nome_titulo,
                "ano_conversao": ano_conversao,
                "data_compra": data_compra,
                "data_pagamento": data_pagamento,
                "n_parcela": k,
                "dias_da_compra": dias_da_compra,
                "aliquota_ir": aliq_ir,
                "parcela_nominal_bruta": parcela_nominal,
                "juros_nominal": juros_nominal,
                "amortizacao_nominal": amort_nominal,
                "ir": ir,
                "parcela_nominal_liquida": parcela_liquida,
            }
        )

    df = pd.DataFrame(registros)

    resumo = {
        "titulo": nome_titulo,
        "ano_conversao": ano_conversao,
        "data_compra": data_compra,
        "valor_investido": valor_investido,
        "meses_ate_conversao": meses_ate_conversao,
        "valor_real_conversao": valor_real_conversao,
        "valor_nominal_conversao": valor_nominal_conversao,
        "parcela_real": parcela_real,
        "total_bruto": df["parcela_nominal_bruta"].sum(),
        "total_ir": df["ir"].sum(),
        "total_liquido": df["parcela_nominal_liquida"].sum(),
    }

    return df, resumo


# -----------------------------------------
# Planejador de aposentadoria – helpers
# -----------------------------------------

def fluxo_real_liquido_por_unidade(
    hoje: date,
    ano_conversao: int,
    taxa_real_aa: float,
    aliquota_ir_constante: float = 0.15,
) -> np.ndarray:
    """
    Para CADA R$ 1 investido hoje em um RendA+ com:
      - ano de conversão
      - taxa real anual

    Calcula o vetor de 240 pagamentos MENSAIS em termos REAIS,
    já LÍQUIDOS de IR (assumindo alíquota constante sobre os juros).

    Simplificações:
      - IPCA = 0 (tudo em reais de hoje)
      - IR fixo (15%) porque todos os prazos são muito longos.
    """
    data_conversao = date(ano_conversao, 1, 15)
    meses_ate_conversao = meses_entre(hoje, data_conversao)
    if meses_ate_conversao <= 0:
        return np.zeros(240)

    taxa_real_m = taxa_anual_para_mensal(taxa_real_aa)

    # PV = 1 real hoje
    valor_real_conversao = (1 + taxa_real_m) ** meses_ate_conversao

    n_parcelas = 240
    parcela_real_bruta = (
        valor_real_conversao
        * taxa_real_m
        / (1 - (1 + taxa_real_m) ** -n_parcelas)
    )

    saldo_real = valor_real_conversao
    pagamentos_liquidos = []

    for _ in range(n_parcelas):
        juros_real = saldo_real * taxa_real_m
        amort_real = parcela_real_bruta - juros_real
        saldo_real = max(saldo_real - amort_real, 0.0)

        ir = aliquota_ir_constante * juros_real
        parcela_liquida = parcela_real_bruta - ir
        pagamentos_liquidos.append(parcela_liquida)

    return np.array(pagamentos_liquidos)


def montar_alocacao_aposentadoria(
    idade_atual: int,
    idade_aposentadoria: int,
    renda_real_desejada: float,
    hoje: date,
    vencimentos: list,
    taxas_reais_aa: list,
) -> tuple:
    """
    Calcula sugestão de quanto investir em cada vencimento do RendA+ hoje
    para tentar estabilizar a renda REAL LÍQUIDA de IR ao redor de
    'renda_real_desejada' durante a aposentadoria.

    Retorna:
      - df_aloc: DataFrame com investimento sugerido por vencimento
      - df_renda: DataFrame com renda mensal líquida simulada (reais de hoje)
    """
    if idade_aposentadoria <= idade_atual:
        raise ValueError("Idade de aposentadoria deve ser maior que a idade atual.")

    ano_aposentadoria = hoje.year + (idade_aposentadoria - idade_atual)
    inicio_aposentadoria = date(ano_aposentadoria, 1, 15)

    if not vencimentos:
        raise ValueError("Nenhum vencimento selecionado.")

    ultimo_ano = max(vencimentos) + 20
    data_final = date(ultimo_ano, 1, 15)
    n_meses = meses_entre(inicio_aposentadoria, data_final) + 1

    # Matriz A: para cada mês da aposentadoria, quanto 1 real em cada título entrega
    A = np.zeros((n_meses, len(vencimentos)), dtype=float)
    # Vetor alvo: renda real desejada em cada mês
    y = np.full(n_meses, renda_real_desejada, dtype=float)

    for j, (ano_conv, taxa_real_aa) in enumerate(zip(vencimentos, taxas_reais_aa)):
        data_conv = date(ano_conv, 1, 15)
        pagamentos_liq = fluxo_real_liquido_por_unidade(hoje, ano_conv, taxa_real_aa)

        # índice do primeiro pagamento líquido na linha do tempo da aposentadoria
        idx_primeiro = meses_entre(inicio_aposentadoria, data_conv)

        for k in range(240):
            idx = idx_primeiro + k
            if 0 <= idx < n_meses:
                A[idx, j] += pagamentos_liq[k]

    # Resolver mínimos quadrados: A x ≈ y
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    x = np.where(x < 0, 0, x)  # sem posições vendidas

    # Alocação sugerida
    registros_aloc = []
    for ano_conv, taxa_real_aa, montante in zip(vencimentos, taxas_reais_aa, x):
        registros_aloc.append(
            {
                "ano_conversao": ano_conv,
                "taxa_real_anual": taxa_real_aa,
                "investimento_sugerido": montante,
            }
        )
    df_aloc = pd.DataFrame(registros_aloc)

    # Renda efetiva
    renda = A @ x
    datas = [adicionar_meses(inicio_aposentadoria, i) for i in range(n_meses)]
    df_renda = pd.DataFrame({"data": datas, "renda_real_liquida": renda})

    return df_aloc, df_renda


# -----------------------------------------
# Aba 1 – Simulador de fluxo
# -----------------------------------------

def aba_simulador_fluxo():
    st.header("📈 Simulador de Fluxo – Tesouro RendA+")

    st.markdown(
        """
        Aqui você simula o **fluxo de caixa mensal** de uma ou mais compras
        de Tesouro RendA+, considerando:

        - taxa real (IPCA + X% ao ano)
        - IPCA médio esperado
        - valor investido único por operação
        - IR regressivo sobre os juros de cada parcela
        """
    )

    vencimentos_validos = [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065]

    hoje = date.today()
    data_padrao = date(hoje.year, hoje.month, min(hoje.day, 28))

    ipca_padrao = st.number_input(
        "IPCA médio esperado (a.a., %) – padrão",
        min_value=-5.0,
        max_value=20.0,
        value=4.0,
        step=0.25,
    )

    n_operacoes = st.number_input(
        "Número de operações (compras diferentes)",
        min_value=1,
        max_value=20,
        value=2,
        step=1,
    )

    mostrar_liquido = st.checkbox(
        "Mostrar fluxo líquido de IR no gráfico",
        value=True,
    )

    st.subheader("Definição das operações")

    operacoes = []
    for i in range(int(n_operacoes)):
        st.markdown(f"**Operação {i+1}**")
        c1, c2, c3 = st.columns(3)

        with c1:
            data_compra = st.date_input(
                f"Data da compra (Op {i+1})",
                value=data_padrao,
                key=f"data_compra_{i}",
            )
            ano_conv = st.selectbox(
                f"Ano de conversão (Op {i+1})",
                options=vencimentos_validos,
                index=min(i, len(vencimentos_validos) - 1),
                key=f"ano_conv_{i}",
            )

        with c2:
            taxa_real_pct = st.number_input(
                f"Taxa real IPCA+ (a.a., %) – Op {i+1}",
                min_value=-5.0,
                max_value=15.0,
                value=6.0,
                step=0.25,
                key=f"taxa_real_{i}",
            )
            ipca_pct = st.number_input(
                f"IPCA esperado (a.a., %) – Op {i+1}",
                min_value=-5.0,
                max_value=20.0,
                value=ipca_padrao,
                step=0.25,
                key=f"ipca_{i}",
            )

        with c3:
            valor_inv = st.number_input(
                f"Valor investido (R$) – Op {i+1}",
                min_value=0.0,
                value=10000.0 if i == 0 else 0.0,
                step=1000.0,
                key=f"valor_inv_{i}",
                format="%.2f",
            )

        nome_titulo = f"RendA+ {ano_conv} – Op {i+1}"

        operacoes.append(
            {
                "nome": nome_titulo,
                "data_compra": data_compra,
                "ano_conversao": ano_conv,
                "taxa_real_aa": taxa_real_pct / 100.0,
                "ipca_aa": ipca_pct / 100.0,
                "valor_investido": valor_inv,
            }
        )

        st.markdown("---")

    if st.button("🚀 Rodar simulação de fluxo"):
        dfs = []
        resumos = []

        for op in operacoes:
            if op["valor_investido"] <= 0:
                continue

            df_op, resumo_op = simular_operacao_renda_mais(
                data_compra=op["data_compra"],
                ano_conversao=op["ano_conversao"],
                taxa_real_aa=op["taxa_real_aa"],
                ipca_aa=op["ipca_aa"],
                valor_investido=op["valor_investido"],
                nome_titulo=op["nome"],
            )
            dfs.append(df_op)
            resumos.append(resumo_op)

        if not dfs:
            st.warning("Nenhuma operação válida (todos os valores investidos são zero).")
            return

        df_tudo = pd.concat(dfs, ignore_index=True)

        st.subheader("Resumo por operação")
        df_resumo = pd.DataFrame(resumos)
        cols_res = [
            "titulo",
            "data_compra",
            "ano_conversao",
            "valor_investido",
            "meses_ate_conversao",
            "valor_real_conversao",
            "valor_nominal_conversao",
            "total_bruto",
            "total_ir",
            "total_liquido",
        ]
        st.dataframe(
            df_resumo[cols_res].style.format(
                {
                    "valor_investido": "R$ {:,.2f}".format,
                    "valor_real_conversao": "R$ {:,.2f}".format,
                    "valor_nominal_conversao": "R$ {:,.2f}".format,
                    "total_bruto": "R$ {:,.2f}".format,
                    "total_ir": "R$ {:,.2f}".format,
                    "total_liquido": "R$ {:,.2f}".format,
                }
            )
        )

        st.subheader("Fluxo consolidado por data de pagamento")
        df_fluxo = df_tudo.groupby("data_pagamento", as_index=False).agg(
            {
                "parcela_nominal_bruta": "sum",
                "ir": "sum",
                "parcela_nominal_liquida": "sum",
            }
        )
        st.dataframe(
            df_fluxo.head(20).style.format(
                {
                    "parcela_nominal_bruta": "R$ {:,.2f}".format,
                    "ir": "R$ {:,.2f}".format,
                    "parcela_nominal_liquida": "R$ {:,.2f}".format,
                }
            )
        )

        csv = df_fluxo.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Baixar fluxo consolidado (CSV)",
            data=csv,
            file_name="fluxo_renda_mais_consolidado.csv",
            mime="text/csv",
        )

        df_plot = df_fluxo.set_index("data_pagamento")
        if mostrar_liquido:
            st.line_chart(df_plot[["parcela_nominal_liquida"]])
            st.caption("Fluxo **líquido de IR** ao longo do tempo.")
        else:
            st.line_chart(df_plot[["parcela_nominal_bruta"]])
            st.caption("Fluxo **bruto** (sem IR).")

        st.subheader("Detalhe por título / operação")
        titulos = df_tudo["titulo"].unique().tolist()
        titulo_sel = st.selectbox(
            "Escolha um título/operacão para detalhar:",
            options=titulos,
        )
        df_det = df_tudo[df_tudo["titulo"] == titulo_sel].copy()
        st.dataframe(
            df_det.head(30).style.format(
                {
                    "parcela_nominal_bruta": "R$ {:,.2f}".format,
                    "juros_nominal": "R$ {:,.2f}".format,
                    "amortizacao_nominal": "R$ {:,.2f}".format,
                    "ir": "R$ {:,.2f}".format,
                    "parcela_nominal_liquida": "R$ {:,.2f}".format,
                }
            )
        )


# -----------------------------------------
# Aba 2 – Planejador de aposentadoria
# -----------------------------------------

def aba_planejador_aposentadoria():
    st.header("🧓 Planejador de Aposentadoria – RendA+")

    st.markdown(
        """
        Esta ferramenta sugere **quanto investir hoje em cada RendA+**
        para tentar garantir uma **renda mensal REAL líquida de IR**
        próxima do valor desejado, a partir da idade de aposentadoria.

        Hipóteses do modelo:
        - Você faz apenas **um investimento hoje** (sem novos aportes depois).
        - Todos os cálculos são em **reais de hoje** (IPCA = 0 no modelo).
        - IR considerado: **15% fixo** sobre os juros reais (prazo muito longo).
        """
    )

    hoje = date.today()
    c1, c2 = st.columns(2)
    with c1:
        idade_atual = st.number_input(
            "Idade atual",
            min_value=18,
            max_value=90,
            value=30,
            step=1,
        )
        idade_aposentadoria = st.number_input(
            "Idade em que deseja se aposentar",
            min_value=idade_atual + 1,
            max_value=100,
            value=65,
            step=1,
        )
    with c2:
        renda_desejada = st.number_input(
            "Renda mensal REAL líquida desejada na aposentadoria (R$)",
            min_value=0.0,
            value=5000.0,
            step=500.0,
            format="%.2f",
        )
        st.number_input(
            "IPCA esperado (a.a., %, apenas informativo)",
            min_value=-5.0,
            max_value=20.0,
            value=4.0,
            step=0.25,
        )

    st.subheader("Taxa real de cada vencimento RendA+")

    vencimentos_validos = [2030, 2035, 2040, 2045, 2050, 2055, 2060, 2065]
    venc_sel = []
    taxas_sel = []

    for ano in vencimentos_validos:
        cols = st.columns(3)
        with cols[0]:
            usar = st.checkbox(
                f"Usar RendA+ {ano}",
                value=(ano >= hoje.year + 5),
            )
        with cols[1]:
            taxa_real_pct = st.number_input(
                f"Taxa real IPCA+ (a.a., %) – {ano}",
                min_value=-5.0,
                max_value=15.0,
                value=6.0,
                step=0.25,
                key=f"taxa_real_planner_{ano}",
            )
        with cols[2]:
            st.write("")
        st.markdown("---")

        if usar:
            venc_sel.append(ano)
            taxas_sel.append(taxa_real_pct / 100.0)

    if st.button("📐 Calcular alocação sugerida"):
        if not venc_sel:
            st.warning("Selecione ao menos um vencimento para usar no plano.")
            return

        try:
            df_aloc, df_renda = montar_alocacao_aposentadoria(
                idade_atual=idade_atual,
                idade_aposentadoria=idade_aposentadoria,
                renda_real_desejada=renda_desejada,
                hoje=hoje,
                vencimentos=venc_sel,
                taxas_reais_aa=taxas_sel,
            )
        except ValueError as e:
            st.error(str(e))
            return

        st.subheader("Investimento sugerido hoje (em reais de hoje)")
        st.dataframe(
            df_aloc.style.format(
                {
                    "taxa_real_anual": "{:.2%}".format,
                    "investimento_sugerido": "R$ {:,.2f}".format,
                }
            )
        )

        investimento_total = df_aloc["investimento_sugerido"].sum()
        st.write(f"**Investimento total sugerido: R$ {investimento_total:,.2f}**")

        st.subheader("Renda mensal REAL líquida simulada na aposentadoria")
        st.dataframe(df_renda.head(30))

        st.line_chart(
            df_renda.set_index("data")[["renda_real_liquida"]],
        )

        renda_media = df_renda["renda_real_liquida"].mean()
        renda_min = df_renda["renda_real_liquida"].min()
        renda_max = df_renda["renda_real_liquida"].max()

        st.markdown(
            f"""
            **Estatísticas (reais de hoje, já líquidos de IR):**
            - Renda média: R$ {renda_media:,.2f}  
            - Renda mínima: R$ {renda_min:,.2f}  
            - Renda máxima: R$ {renda_max:,.2f}  
            """
        )


# -----------------------------------------
# App principal
# -----------------------------------------

def main():
    st.set_page_config(
        page_title="Ferramentas Tesouro RendA+",
        layout="wide",
    )

    aba1, aba2 = st.tabs(["Simulador de Fluxo", "Planejador de Aposentadoria"])

    with aba1:
        aba_simulador_fluxo()

    with aba2:
        aba_planejador_aposentadoria()


if __name__ == "__main__":
    main()
