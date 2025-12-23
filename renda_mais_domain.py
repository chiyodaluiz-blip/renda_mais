# renda_mais_domain.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Tuple, Dict
from math import floor

import numpy as np
import pandas as pd


# ------------------- Domínio básico -------------------

@dataclass
class RendaMaisOperation:
    data_compra: date
    ano_conversao: int
    taxa_real_aa: float      # em DECIMAL (ex: 0.07)
    ipca_aa: float           # em DECIMAL (ex: 0.04)
    valor_investido: float
    nome_titulo: str


class IncomeTaxRule:
    """
    Regra de IR simplificada para renda fixa:
    - até 180 dias    -> 22.5%
    - até 360 dias    -> 20.0%
    - até 720 dias    -> 17.5%
    - acima de 720    -> 15.0%
    """

    def aliquota(self, dias: int) -> float:
        if dias <= 180:
            return 0.225
        elif dias <= 360:
            return 0.20
        elif dias <= 720:
            return 0.175
        else:
            return 0.15


# ------------------- Simulador RendA+ -------------------

class RendaMaisSimulator:
    """
    Responsável por simular o fluxo de caixa de uma operação de RendA+.
    """

    def __init__(self, ir_rule: IncomeTaxRule | None = None) -> None:
        self.ir_rule = ir_rule or IncomeTaxRule()

    @staticmethod
    def _to_monthly_rate(rate_aa: float) -> float:
        return (1 + rate_aa) ** (1 / 12) - 1

    def simulate_operation(
        self, op: RendaMaisOperation
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Retorna:
        - df_fluxo: DataFrame com colunas
          [data_pagamento, parcela_nominal_bruta, juros_nominal,
           amortizacao_nominal, ir, parcela_nominal_liquida]
        - resumo: dict com informações agregadas da operação
        """

        # 1) Quantos meses até a conversão
        data_conversao = date(op.ano_conversao, 1, 1)
        meses_ate_conversao = (data_conversao.year - op.data_compra.year) * 12 + (
            data_conversao.month - op.data_compra.month
        )
        meses_ate_conversao = max(meses_ate_conversao, 0)

        real_m = self._to_monthly_rate(op.taxa_real_aa)
        ipca_m = self._to_monthly_rate(op.ipca_aa)

        # 2) Valor real acumulado até a conversão
        valor_real_conversao = op.valor_investido * (1 + real_m) ** meses_ate_conversao
        # valor nominal nessa data
        valor_nominal_conversao = valor_real_conversao * (1 + ipca_m) ** meses_ate_conversao

        # 3) Fase de renda: 240 meses de anuidade real
        n_pagamentos = 240
        if real_m == 0:
            pmt_real = valor_real_conversao / n_pagamentos
        else:
            pmt_real = valor_real_conversao * real_m / (1 - (1 + real_m) ** -n_pagamentos)

        # 4) Construir fluxo mês a mês (nominal) após a conversão
        datas_pag = []
        juros_nom = []
        amort_nom = []
        parcela_nom = []
        ir_list = []
        parcela_liq = []

        saldo_real = valor_real_conversao

        # dias entre compra e cada recebimento (aprox: 30 dias por mês)
        dias_desde_compra_base = meses_ate_conversao * 30

        for k in range(1, n_pagamentos + 1):
            # data de pagamento nominal (aprox)
            mes = data_conversao.month + (k - 1)
            ano = data_conversao.year + (mes - 1) // 12
            mes_corrig = ((mes - 1) % 12) + 1
            data_pag = date(ano, mes_corrig, 15)  # dia 15 arbitrário
            datas_pag.append(data_pag)

            # juros + amortização em termos REAIS
            juros_real = saldo_real * real_m
            amort_real = pmt_real - juros_real
            saldo_real -= amort_real

            # fator de inflação até esse pagamento (contínua após conversão)
            meses_total = meses_ate_conversao + k
            fator_ipca = (1 + ipca_m) ** meses_total

            # converter pmt_real, juros_real, amort_real para NOMINAL
            pmt_nom_k = pmt_real * fator_ipca
            juros_nom_k = juros_real * fator_ipca
            amort_nom_k = amort_real * fator_ipca

            parcela_nom.append(pmt_nom_k)
            juros_nom.append(juros_nom_k)
            amort_nom.append(amort_nom_k)

            # IR sobre juros (simplificado: base = juros_nom_k)
            dias_total = dias_desde_compra_base + k * 30
            aliq = self.ir_rule.aliquota(dias_total)
            ir_k = juros_nom_k * aliq
            ir_list.append(ir_k)
            parcela_liq.append(pmt_nom_k - ir_k)

        df_fluxo = pd.DataFrame(
            {
                "data_pagamento": datas_pag,
                "parcela_nominal_bruta": parcela_nom,
                "juros_nominal": juros_nom,
                "amortizacao_nominal": amort_nom,
                "ir": ir_list,
                "parcela_nominal_liquida": parcela_liq,
            }
        )

        resumo = {
            "titulo": op.nome_titulo,
            "data_compra": op.data_compra,
            "ano_conversao": op.ano_conversao,
            "valor_investido": op.valor_investido,
            "meses_ate_conversao": meses_ate_conversao,
            "valor_real_conversao": valor_real_conversao,
            "valor_nominal_conversao": valor_nominal_conversao,
            "total_bruto": float(df_fluxo["parcela_nominal_bruta"].sum()),
            "total_ir": float(df_fluxo["ir"].sum()),
            "total_liquido": float(df_fluxo["parcela_nominal_liquida"].sum()),
        }

        return df_fluxo, resumo


# ------------------- Planejador de aposentadoria -------------------

class AposentadoriaPlanner:
    """
    Usa o simulador RendA+ para sugerir alocação que gera uma certa renda
    mensal real líquida desejada.
    """

    def __init__(self, simulator: RendaMaisSimulator) -> None:
        self.simulator = simulator

    def montar_alocacao(
        self,
        idade_atual: int,
        idade_aposentadoria: int,
        idade_fim_recebimento: int,
        renda_real_desejada: float,
        hoje: date,
        vencimentos: list[int],
        taxas_reais_aa: list[float],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        NNLS-like allocation by least squares, using horizon from idade_aposentadoria
        until idade_fim_recebimento.

        The key fix here: the monthly index now starts at a year computed from `hoje`
        and the age difference, so we avoid passing raw ages as calendar years.
        """
        import numpy as np
        import pandas as pd

        # Validations
        if len(vencimentos) != len(taxas_reais_aa):
            raise ValueError("Listas 'vencimentos' e 'taxas_reais_aa' devem ter mesmo tamanho.")
        if idade_aposentadoria <= idade_atual:
            raise ValueError("Idade de aposentadoria deve ser maior que idade atual.")
        if idade_fim_recebimento <= idade_aposentadoria:
            raise ValueError("Idade fim do recebimento deve ser maior que a idade de aposentadoria.")

        # Horizon (months)
        anos_horizonte = idade_fim_recebimento - idade_aposentadoria
        meses_aposentadoria = anos_horizonte * 12
        if meses_aposentadoria <= 0:
            raise ValueError("Horizonte calculado <= 0 meses — verifique as idades.")

        # --- FIX: construir índice mensal usando o ano atual + diferença de idades ---
        # Se hoje.year = 2025, idade_atual = 35, idade_aposentadoria = 65:
        # ano_inicio = 2025 + (65 - 35) = 2055 -> começamos em Jan/2055 (matches expected timeline)
        ano_inicio = hoje.year + (idade_aposentadoria - idade_atual)
        month_index = pd.date_range(
            start=pd.Timestamp(year=ano_inicio, month=1, day=1),
            periods=meses_aposentadoria,
            freq="MS",
        )

        # For each vencimento simulate 1.0 invested today (real terms: ipca=0)
        series = []
        for ano_conv, taxa_real in zip(vencimentos, taxas_reais_aa):
            op = RendaMaisOperation(
                data_compra=hoje,
                ano_conversao=ano_conv,
                taxa_real_aa=taxa_real,
                ipca_aa=0.0,
                valor_investido=1.0,
                nome_titulo=f"RendA+ {ano_conv}",
            )
            df_fluxo, _ = self.simulator.simulate_operation(op)

            # aggregate by month and align with month_index
            # extrair série com índice temporal e garantir DatetimeIndex
            s = df_fluxo.set_index("data_pagamento")["parcela_nominal_liquida"].copy()

            # garantir que o índice é datetime (pode vir como date ou string)
            s.index = pd.to_datetime(s.index)

            # ordenar por índice (requerido para resample)
            s = s.sort_index()

            # agora resample mensal (MS = Month Start) e somar pagamentos no mês
            s_monthly = s.resample("MS").sum()

            # criar janela alvo e alinhar valores existentes
            s_window = pd.Series(0.0, index=month_index)

            # usar intersection direto e atribuição vetorial (mais rápido que loop)
            common_idx = s_monthly.index.intersection(s_window.index)
            if not common_idx.empty:
                s_window.loc[common_idx] = s_monthly.loc[common_idx].values

            series.append(s_window.values)

        # Build matrix A (months x n_venc)
        A = np.column_stack(series)
        n_venc = A.shape[1]

        # Target vector b: constant desired monthly REAL income
        b = np.full((meses_aposentadoria,), renda_real_desejada)

        # Initial least squares solution
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        x = x_ls.copy()

        # If negatives appear, apply simple iterative NNLS-like fix:
        neg_mask = x < 0
        if neg_mask.any():
            active_mask = ~neg_mask
            for _ in range(n_venc):
                if active_mask.sum() == 0:
                    x = np.zeros(n_venc)
                    break
                A_active = A[:, active_mask]
                x_active, *_ = np.linalg.lstsq(A_active, b, rcond=None)
                if (x_active >= 0).all():
                    x = np.zeros(n_venc)
                    x[active_mask] = x_active
                    break
                neg_active = x_active < 0
                if not neg_active.any():
                    x = np.zeros(n_venc)
                    x[active_mask] = np.maximum(x_active, 0.0)
                    break
                active_idx = np.where(active_mask)[0]
                neg_global_idx = active_idx[neg_active]
                active_mask[neg_global_idx] = False
            x = np.maximum(x, 0.0)

        # Scale to reach exactly the average desired income across the window
        renda_resultante = A @ x
        renda_media_result = renda_resultante.mean()
        if renda_media_result <= 0:
            raise ValueError("Solução encontrou renda média zero — verifique as taxas/vencimentos.")
        scale = renda_real_desejada / renda_media_result
        x = x * scale
        renda_resultante = renda_resultante * scale


        # --------------------------------------------------
        # NOVO: construir fluxo REAL completo (desde o 1º pagamento)
        # --------------------------------------------------
        fluxos_reais = []

        for ano_conv, taxa_real, investimento in zip(vencimentos, taxas_reais_aa, x):
            if investimento <= 0:
                continue

            op = RendaMaisOperation(
                data_compra=hoje,
                ano_conversao=ano_conv,
                taxa_real_aa=taxa_real,
                ipca_aa=0.0,
                valor_investido=investimento,
                nome_titulo=f"RendA+ {ano_conv}",
            )

            df_fluxo_op, _ = self.simulator.simulate_operation(op)
            fluxos_reais.append(df_fluxo_op)

        if fluxos_reais:
            df_fluxo_real = (
                pd.concat(fluxos_reais)
                .groupby("data_pagamento", as_index=False)
                .sum(numeric_only=True)
                .sort_values("data_pagamento")
            )
        else:
            df_fluxo_real = pd.DataFrame(
                columns=["data_pagamento", "parcela_nominal_liquida"]
            )

        # Build outputs
        df_aloc = pd.DataFrame({
            "ano_conversao": vencimentos,
            "taxa_real_anual": taxas_reais_aa,
            "investimento_sugerido": x,
        })

        df_renda_otimizacao = pd.DataFrame({
            "data": month_index,
            "renda_real_liquida": renda_resultante
        })


        return df_aloc, df_renda_otimizacao, df_fluxo_real


# ------------------- Metrics -------------------

def compute_retirement_metrics(
    df_renda: pd.DataFrame,
    renda_objetivo: float,
    idade_atual: int,
    idade_aposentadoria: int,
    idade_fim_recebimento: int,
    hoje: date,
) -> dict:
    """
    df_renda: DataFrame com colunas ['data', 'renda_real_liquida']
    """

    data_aposentadoria = date(
        hoje.year + (idade_aposentadoria - idade_atual),
        hoje.month,
        hoje.day,
    )

    df_pos_apos = df_renda[
        df_renda["data"] >= data_aposentadoria
    ].copy()


    renda = df_pos_apos["renda_real_liquida"].values

    renda_media = float(np.mean(renda))
    renda_min = float(np.min(renda))
    renda_max = float(np.max(renda))
    volatilidade = float(np.std(renda))
    volatilidade_pct = volatilidade / renda_media if renda_media > 0 else 0.0

    meses_abaixo = int(np.sum(renda < renda_objetivo))
    pior_gap_abs = float(np.min(renda - renda_objetivo))
    pior_gap_pct = pior_gap_abs / renda_objetivo if renda_objetivo > 0 else 0.0

    margem_seguranca = (renda_media - renda_objetivo) / renda_objetivo

    duracao_anos = idade_fim_recebimento - idade_aposentadoria
    duracao_meses = duracao_anos * 12

    expectativa_media_ibge = 80  # proxy simples e conservador
    anos_folga = idade_fim_recebimento - expectativa_media_ibge

    data_inicio = pd.to_datetime(df_renda["data"].min()).date()
    data_fim = pd.to_datetime(df_renda["data"].max()).date()



    # idade real no primeiro e último pagamento
    anos_ate_inicio = (data_inicio - hoje).days / 365.25
    anos_ate_fim = (data_fim - hoje).days / 365.25

    idade_inicio_real = idade_atual + anos_ate_inicio
    idade_fim_real = idade_atual + anos_ate_fim

    ano_inicio_pagamento = data_inicio.year
    ano_fim_pagamento = data_fim.year

    return {
        # Tempo
        "idade_inicio_real": int(floor(idade_inicio_real)),
        "idade_fim_real": int(floor(idade_fim_real)),
        "ano_inicio_pagamento": ano_inicio_pagamento,
        "ano_fim_pagamento": ano_fim_pagamento,
        "duracao_anos": duracao_anos,
        "duracao_meses": duracao_meses,

        # Renda
        "renda_media": renda_media,
        "renda_min": renda_min,
        "renda_max": renda_max,
        "volatilidade_abs": volatilidade,
        "volatilidade_pct": volatilidade_pct,

        # Segurança
        "meses_abaixo_objetivo": meses_abaixo,
        "pior_gap_abs": pior_gap_abs,
        "pior_gap_pct": pior_gap_pct,
        "margem_seguranca": margem_seguranca,

        # Longevidade
        "expectativa_ibge": expectativa_media_ibge,
        "anos_folga": anos_folga,

        # Datas
        "data_inicio": data_inicio,
        "data_fim": data_fim,
        "data_aposentadoria": data_aposentadoria,
    }
