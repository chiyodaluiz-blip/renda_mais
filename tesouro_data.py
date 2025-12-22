# tesouro_data.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from io import BytesIO

import pandas as pd
import requests
from bs4 import BeautifulSoup


# =============================================================================
# Repositório de histórico de preços/taxas do Tesouro Direto
# =============================================================================


@dataclass
class TesouroDiretoRepository:
    """
    Responsável por carregar e padronizar o histórico de preços/taxas.
    Usa como fonte o CSV oficial do Tesouro Transparente:
    "Taxas dos Títulos Ofertados pelo Tesouro Direto"
    """

    session: Optional[requests.Session] = None

    # URL atual (CKAN) do arquivo precotaxatesourodireto.csv
    HIST_CSV_URL: str = (
        "https://www.tesourotransparente.gov.br/ckan/dataset/"
        "df56aa42-484a-4a59-8184-7676580c81e3/resource/"
        "796d2059-14e9-44e3-80c9-2d9e30b405c1/download/precotaxatesourodireto.csv"
    )

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    # ------------------- DOWNLOAD / CARREGAMENTO -------------------

    def load_raw_history(self) -> pd.DataFrame:
        """
        Baixa o CSV de preços/taxas do Tesouro Transparente e devolve um DataFrame cru.

        Observações:
        - O arquivo vem com separador ';' e decimais com vírgula.
        - Usamos encoding 'latin-1' porque é padrão em planilhas públicas BR.
        """
        resp = self.session.get(self.HIST_CSV_URL, timeout=60)
        resp.raise_for_status()

        content = resp.content

        # A leitura padrão é ; com decimal ','. Se no futuro mudarem, é fácil ajustar.
        df = pd.read_csv(
            BytesIO(content),
            sep=";",
            encoding="latin-1",
            decimal=",",
            low_memory=False,
        )

        return df

    # ------------------- PADRONIZAÇÃO / DETECÇÃO -------------------

    @staticmethod
    def _detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        cols_lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            for col_lc, col_real in cols_lower.items():
                if cand in col_lc:
                    return col_real
        return None

    def detect_col_titulo(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["titulo", "título", "nome_titulo", "nome do papel", "descricao"],
        )

    def detect_col_data_base(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["data_base", "data base", "data do pregão", "data_referencia", "data"],
        )

    def detect_col_preco(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["pu", "preco", "preço", "preco_base", "preco unit"],
        )

    def detect_col_vencimento(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["venc", "vencimento", "data venc", "data_vencimento"],
        )

    def detect_col_taxa_compra(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["taxa_compra", "taxa compra", "taxa de compra"]
        )

    def detect_col_taxa_venda(self, df: pd.DataFrame) -> Optional[str]:
        return self._detect_column(
            df,
            ["taxa_venda", "taxa venda", "taxa de venda"]
        )

    # ------------------- INTERFACE PRINCIPAL -------------------

    def load_history_standardized(self) -> pd.DataFrame:
        """
        Carrega o histórico bruto e devolve um DataFrame padronizado com colunas:
        - TituloRaw
        - Data_Base (datetime)
        - PU
        - Data_Vencimento (datetime)
        - Taxa_Compra (se houver)
        - Taxa_Venda (se houver)
        """

        df_raw = self.load_raw_history()

        col_titulo = self.detect_col_titulo(df_raw)
        col_data_base = self.detect_col_data_base(df_raw)
        col_preco = self.detect_col_preco(df_raw)
        col_venc = self.detect_col_vencimento(df_raw)
        col_tc = self.detect_col_taxa_compra(df_raw)
        col_tv = self.detect_col_taxa_venda(df_raw)

        if not all([col_titulo, col_data_base, col_preco, col_venc]):
            raise ValueError(
                "Não foi possível detectar as colunas principais "
                "(título, data base, preço, vencimento). "
                f"Colunas do arquivo: {list(df_raw.columns)}"
            )

        rename_map = {
            col_titulo: "TituloRaw",
            col_data_base: "Data_Base",
            col_preco: "PU",
            col_venc: "Data_Vencimento",
        }
        if col_tc:
            rename_map[col_tc] = "Taxa_Compra"
        if col_tv:
            rename_map[col_tv] = "Taxa_Venda"

        df = df_raw.rename(columns=rename_map).copy()

        df["Data_Base"] = pd.to_datetime(
            df["Data_Base"], dayfirst=True, errors="coerce"
        )
        df["Data_Vencimento"] = pd.to_datetime(
            df["Data_Vencimento"], dayfirst=True, errors="coerce"
        )

        df = df.dropna(subset=["Data_Base", "TituloRaw", "PU"])
        df = df.sort_values("Data_Base")

        return df


# =============================================================================
# Repositório de VNA de NTN-B
# =============================================================================


@dataclass
class VnaRepository:
    """
    Carrega a série de VNA das NTN-B a partir do Excel oficial
    "Valor Nominal de NTN-B" no Tesouro Transparente.
    """

    session: Optional[requests.Session] = None

    # Página que contém o link do Excel (histórico completo de NTN-B)
    VNA_PAGE_URL: str = (
        "https://www.tesourotransparente.gov.br/publicacoes/"
        "valor-nominal-de-ntn-b"
    )

    def __post_init__(self) -> None:
        if self.session is None:
            self.session = requests.Session()

    def _find_vna_xlsx_url(self) -> str:
        """
        Acessa a página do Tesouro Transparente e procura o primeiro link
        para um arquivo .xlsx (planilha de valores nominais de NTN-B).
        """
        resp = self.session.get(self.VNA_PAGE_URL, timeout=60)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        link = None

        # Procura qualquer <a> com href terminando em .xlsx
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".xlsx"):
                link = href
                break

        if not link:
            raise RuntimeError(
                "Não foi possível localizar o link do arquivo XLSX de VNA NTN-B "
                "na página do Tesouro Transparente."
            )

        # Se o href for relativo, completa com o domínio
        if link.startswith("http"):
            return link
        else:
            return "https://www.tesourotransparente.gov.br" + link

    def load_vna_ntnb(self) -> pd.DataFrame:
        """
        Retorna DataFrame com colunas:
        - data (datetime)
        - vna  (float)

        O layout típico do Excel é:
        - linhas iniciais com cabeçalho institucional
        - uma linha com "DATA" na primeira coluna
        - a partir da linha seguinte, duas colunas: data e valor nominal
        """

        xlsx_url = self._find_vna_xlsx_url()
        resp = self.session.get(xlsx_url, timeout=60)
        resp.raise_for_status()

        # lê sem header para localizar manualmente a linha do cabeçalho
        df_raw = pd.read_excel(BytesIO(resp.content), header=None)

        # encontra a linha onde a primeira coluna contém "DATA"
        mask_header = df_raw.iloc[:, 0].astype(str).str.upper().str.contains("DATA")
        if not mask_header.any():
            raise RuntimeError(
                "Não encontrei linha com cabeçalho 'DATA' na planilha de VNA NTN-B."
            )

        header_idx = mask_header[mask_header].index[0]

        # dados começam logo após a linha de cabeçalho
        df_data = df_raw.iloc[header_idx + 1 :, 0:2].copy()
        df_data.columns = ["data", "vna"]

        # remove linhas vazias
        df_data = df_data.dropna(subset=["data", "vna"])

        # garante tipos corretos
        df_data["data"] = pd.to_datetime(df_data["data"], dayfirst=True, errors="coerce")
        df_data["vna"] = pd.to_numeric(df_data["vna"], errors="coerce")

        df_data = df_data.dropna(subset=["data", "vna"])
        df_data = df_data.sort_values("data").reset_index(drop=True)

        return df_data
