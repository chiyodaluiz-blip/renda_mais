# pricing_domain.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict

import numpy as np
import pandas as pd

from tesouro_data import TesouroDiretoRepository, VnaRepository


@dataclass
class BondSpec:
    tipo: str                      # "NTNB_PRINCIPAL" ou "RENDA_MAIS"
    data_vencimento: Optional[date] = None  # para IPCA+
    ano_conversao: Optional[int] = None     # para RendA+


class IPCAIndexedPricer:
    """
    Precifica títulos IPCA+ / RendA+ usando VNA + taxa real.
    """

    def __init__(self, vna_repo: VnaRepository) -> None:
        self.vna_repo = vna_repo

    @staticmethod
    def _to_monthly_rate(rate_aa: float) -> float:
        return (1 + rate_aa) ** (1 / 12) - 1

    def _vna_at(self, df_vna: pd.DataFrame, d: date) -> float:
        # supõe df_vna com colunas ["data", "vna"]
        # pega o último VNA <= d
        df = df_vna[df_vna["data"] <= pd.Timestamp(d)]
        if df.empty:
            raise ValueError(f"Não há VNA disponível para data <= {d}.")
        return float(df.iloc[-1]["vna"])

    def price(
        self,
        spec: BondSpec,
        data_compra: date,
        data_calculo: date,
        taxa_compra_real_aa: float,
        taxa_atual_real_aa: float,
    ) -> Dict:
        """
        Para simplificar:
        - IPCA+: preço aproximado = VNA_atual * fator de deságio (taxa_atual)
        - RendA+: aproximação usando PV real dos 240 PMTs (como no simulador).
        """

        df_vna = self.vna_repo.load_vna_ntnb()

        if spec.tipo == "NTNB_PRINCIPAL":
            vna_compra = self._vna_at(df_vna, data_compra)
            vna_atual = self._vna_at(df_vna, data_calculo)

            # PV com taxa "compra" e "atual" são conceitos diferentes;
            # aqui focamos no preço pela taxa atual:
            # preço teórico ~ VNA_atual / (1 + taxa_atual)^(du)
            # onde du = duration em anos até vencimento.
            if spec.data_vencimento is None:
                raise ValueError("data_vencimento é obrigatória para NTNB_PRINCIPAL.")

            anos_ate_venc = (spec.data_vencimento - data_calculo).days / 365.0
            fator_desconto = (1 + taxa_atual_real_aa) ** (-anos_ate_venc)
            preco_atual = vna_atual * fator_desconto

            return {
                "tipo": spec.tipo,
                "vna_compra": vna_compra,
                "vna_atual": vna_atual,
                "preco_atual": preco_atual,
            }

        elif spec.tipo == "RENDA_MAIS":
            if spec.ano_conversao is None:
                raise ValueError("ano_conversao é obrigatório para RENDA_MAIS.")

            # Para preço: PV real dos fluxos futuros com taxa_atual
            real_m = self._to_monthly_rate(taxa_atual_real_aa)
            n_pag = 240

            # supõe 1 real de valor real na data de conversão
            if real_m == 0:
                pmt_real = 1.0 / n_pag
            else:
                pmt_real = 1.0 * real_m / (1 - (1 + real_m) ** -n_pag)

            # PV real hoje: descontar número de meses até conversão
            data_conversao = date(spec.ano_conversao, 1, 1)
            meses_ate_conv = max(
                0,
                (data_conversao.year - data_calculo.year) * 12
                + (data_conversao.month - data_calculo.month),
            )
            pv_real_na_conversao = 1.0  # suposto
            pv_real_hoje = pv_real_na_conversao / ((1 + real_m) ** meses_ate_conv)

            # PV nominal hoje "por 1 real investido" ~ pv_real_hoje (real),
            # se ignorarmos IPCA (ou trabalharmos em termos reais).
            # Aqui retornamos esses valores para o app usar.
            return {
                "tipo": spec.tipo,
                "pv_real_atual": pv_real_hoje,
                "pmt_real": pmt_real,
                "preco_nominal_atual_por_1_real_investido": pv_real_hoje,
            }

        else:
            raise ValueError(f"Tipo de título não suportado: {spec.tipo}")

    def price_curve_vs_rate(
        self,
        spec: BondSpec,
        data_compra: date,
        data_calculo: date,
        taxa_compra_real_aa: float,
        taxa_central_real_aa: float,
        span_pct: float = 0.02,
        n_pontos: int = 21,
    ) -> pd.DataFrame:
        taxas = np.linspace(
            taxa_central_real_aa - span_pct, taxa_central_real_aa + span_pct, n_pontos
        )
        rows = []
        for t in taxas:
            res = self.price(
                spec=spec,
                data_compra=data_compra,
                data_calculo=data_calculo,
                taxa_compra_real_aa=taxa_compra_real_aa,
                taxa_atual_real_aa=t,
            )
            if spec.tipo == "NTNB_PRINCIPAL":
                preco = res["preco_atual"]
            else:
                preco = res["preco_nominal_atual_por_1_real_investido"]
            rows.append({"taxa_real_pct": t * 100.0, "preco_teorico": preco})

        return pd.DataFrame(rows)


class TesouroPriceMatcher:
    """
    Usa o histórico do Tesouro para encontrar o preço oficial na data.
    """

    def __init__(self, repo: TesouroDiretoRepository) -> None:
        self.repo = repo

    def find_official_price(
        self,
        spec: BondSpec,
        data_calculo: date,
    ) -> Optional[float]:
        df = self.repo.load_history_standardized()

        # Você pode sofisticar este "match" pelo padrão exato de nome.
        # Aqui fazemos algo genérico:
        df_dia = df[df["Data_Base"].dt.date == data_calculo]
        if df_dia.empty:
            return None

        # Heurística para o nome do título:
        if spec.tipo == "NTNB_PRINCIPAL":
            # tenta achar por "IPCA+" e ano de vencimento
            ano = spec.data_vencimento.year if spec.data_vencimento else None
            candidatos = df_dia["TituloRaw"].astype(str)
            mask = candidatos.str.contains("IPCA", case=False)
            if ano:
                mask &= candidatos.str.contains(str(ano))
            df_sel = df_dia[mask]
        else:  # RENDA_MAIS
            ano = spec.ano_conversao
            candidatos = df_dia["TituloRaw"].astype(str)
            mask = candidatos.str.contains("Renda", case=False)
            if ano:
                mask &= candidatos.str.contains(str(ano))
            df_sel = df_dia[mask]

        if df_sel.empty:
            return None

        # Se tiver mais de um, pega o primeiro
        return float(df_sel.iloc[0]["PU"])
