"""Testes unitários da engenharia de atributos e do pré-processamento.

Os testes usam uma base sintética pequena. Portanto, podem ser executados no
GitHub Actions ou localmente sem acessar o Databricks e sem carregar a base
Gold do projeto.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse


# Permite executar ``pytest`` a partir da raiz do repositório.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.preprocessing import (  # noqa: E402
    FEATURES_CATEGORICAS,
    FEATURES_INDICADORES,
    FEATURES_NUMERICAS,
    IndicadoresAusencia,
    converter_para_float32,
    criar_pre_processamento,
)


@pytest.fixture
def dados_exemplo() -> pd.DataFrame:
    """Cria registros representativos, incluindo ausências por fonte."""

    return pd.DataFrame(
        {
            "co_uf": [35, 33, 29, 43],
            "tp_dependencia": [2, 3, 2, 4],
            "atlas_idhm": [0.80, np.nan, 0.68, 0.75],
            "atlas_idhm_e": [0.77, np.nan, 0.61, 0.70],
            "atlas_renda_pc": [1_400.0, np.nan, 650.0, 950.0],
            "atlas_indice_gini": [0.48, np.nan, 0.55, 0.51],
            "atlas_prop_pobreza_criancas": [0.10, np.nan, 0.31, 0.19],
            "atlas_taxa_criancas_dom_sem_fund": [0.02, np.nan, 0.09, 0.04],
            "censo_prop_mat_2ano_internet_aprendizagem": [0.90, 0.72, np.nan, 0.81],
            "censo_prop_mat_2ano_alimentacao": [0.95, 0.91, np.nan, 0.93],
            "censo_prop_mat_2ano_biblioteca_sala_leitura": [0.63, 0.51, np.nan, 0.58],
            "fundeb_receita_contribuicao": [10_000.0, 8_000.0, 6_000.0, np.nan],
            "fundeb_complementacao_uniao": [500.0, 1_000.0, 2_000.0, np.nan],
            "fundeb_receita_total": [10_500.0, 9_000.0, 8_000.0, np.nan],
            "indicador_meta_final_2025": [0.80, np.nan, 0.72, 0.78],
            "indicador_pc_aluno_alfabetizado_2024": [0.75, np.nan, 0.63, 0.70],
        }
    )


def test_listas_de_features_nao_possuem_sobreposicao() -> None:
    """Cada coluna deve possuir somente um tratamento na pipeline."""

    grupos = [
        set(FEATURES_NUMERICAS),
        set(FEATURES_CATEGORICAS),
        set(FEATURES_INDICADORES),
    ]

    assert not (grupos[0] & grupos[1])
    assert not (grupos[0] & grupos[2])
    assert not (grupos[1] & grupos[2])
    assert len(FEATURES_NUMERICAS) == 14
    assert len(FEATURES_CATEGORICAS) == 2
    assert len(FEATURES_INDICADORES) == 4


def test_indicadores_ausencia_cria_flags_corretas(
    dados_exemplo: pd.DataFrame,
) -> None:
    """Uma ausência na coluna de referência deve gerar o flag da fonte."""

    original = dados_exemplo.copy(deep=True)
    transformador = IndicadoresAusencia()
    resultado = transformador.fit_transform(dados_exemplo)

    assert resultado[FEATURES_INDICADORES].to_dict("list") == {
        "atlas_dados_ausentes": [0, 1, 0, 0],
        "censo_dados_ausentes": [0, 0, 1, 0],
        "fundeb_dados_ausentes": [0, 0, 0, 1],
        "indicadores_dados_ausentes": [0, 1, 0, 0],
    }
    pd.testing.assert_frame_equal(dados_exemplo, original)


def test_transformador_fit_nao_aprende_estado(
    dados_exemplo: pd.DataFrame,
) -> None:
    """O transformer de flags deve seguir a interface do scikit-learn."""

    transformador = IndicadoresAusencia()
    assert transformador.fit(dados_exemplo) is transformador


def test_pipeline_padrao_imputa_e_preserva_quantidade_de_linhas(
    dados_exemplo: pd.DataFrame,
) -> None:
    """A saída padrão não pode conter nulos após a imputação."""

    pipeline = criar_pre_processamento()
    resultado = pipeline.fit_transform(dados_exemplo)

    assert resultado.shape[0] == len(dados_exemplo)
    assert resultado.shape[1] >= (
        len(FEATURES_NUMERICAS) + len(FEATURES_CATEGORICAS)
        + len(FEATURES_INDICADORES)
    )
    valores = resultado.data if sparse.issparse(resultado) else resultado
    assert np.isfinite(valores).all()


def test_pipeline_aceita_categoria_desconhecida(
    dados_exemplo: pd.DataFrame,
) -> None:
    """Categorias novas não devem interromper a inferência."""

    pipeline = criar_pre_processamento()
    pipeline.fit(dados_exemplo)

    novo = dados_exemplo.iloc[[0]].copy()
    novo.loc[:, "co_uf"] = 99
    novo.loc[:, "tp_dependencia"] = 99

    resultado = pipeline.transform(novo)
    assert resultado.shape[0] == 1


def test_pipeline_densa_retorna_float32(
    dados_exemplo: pd.DataFrame,
) -> None:
    """HistGradientBoosting deve receber matriz densa e econômica."""

    pipeline = criar_pre_processamento(
        padronizar=True,
        saida_densa=True,
    )
    resultado = pipeline.fit_transform(dados_exemplo)

    assert isinstance(resultado, np.ndarray)
    assert resultado.dtype == np.float32
    assert np.isfinite(resultado).all()


def test_converter_para_float32_evita_copia_desnecessaria() -> None:
    """A conversão deve manter arrays que já estejam no dtype esperado."""

    original = np.array([[1.0, 2.0]], dtype=np.float32)
    resultado = converter_para_float32(original)

    assert resultado.dtype == np.float32
    assert np.shares_memory(original, resultado)

