"""Contratos mínimos dos dados utilizados na modelagem.

Este módulo documenta e testa as regras que devem continuar verdadeiras mesmo
quando uma nova safra de dados for processada. Os testes são independentes do
Databricks e utilizam estruturas sintéticas pequenas.
"""

from pathlib import Path
import sys

import pandas as pd
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.preprocessing import (  # noqa: E402
    FEATURES_CATEGORICAS,
    FEATURES_NUMERICAS,
)


TARGET = "in_alfabetizado"

FEATURES_ESPERADAS = FEATURES_CATEGORICAS + FEATURES_NUMERICAS

COLUNAS_AUXILIARES = [
    "id_aluno",
    "id_escola",
    "co_municipio",
]

COLUNAS_PROIBIDAS_POR_LEAKAGE = {
    "vl_proficiencia_lp",
    "in_presenca_lp",
    "in_preenchimento_lp",
    "co_caderno_lp",
    "co_bloco_1",
    "tx_resposta_bloco_1",
    "tx_gabarito_bloco_1",
    "co_bloco_2",
    "tx_resposta_bloco_2",
    "tx_gabarito_bloco_2",
    "co_bloco_3",
    "tx_resposta_bloco_3",
    "tx_gabarito_bloco_3",
    "co_bloco_4",
    "tx_resposta_bloco_4",
    "tx_gabarito_bloco_4",
}


def validar_colunas_inferencia(dados: pd.DataFrame) -> pd.DataFrame:
    """Valida presença, excesso e ordem das features de inferência."""

    recebidas = set(dados.columns)
    esperadas = set(FEATURES_ESPERADAS)
    ausentes = sorted(esperadas - recebidas)
    extras = sorted(recebidas - esperadas)

    if ausentes or extras:
        raise ValueError(
            "Contrato de entrada inválido. "
            f"Ausentes: {ausentes}. Extras: {extras}."
        )

    return dados.loc[:, FEATURES_ESPERADAS].copy()


def validar_target(target: pd.Series) -> None:
    """Exige target binário, completo e limitado aos valores zero e um."""

    if target.isna().any():
        raise ValueError("O target contém valores ausentes.")

    dominio = set(target.unique())
    if not dominio.issubset({0, 1}):
        raise ValueError(f"Domínio inválido do target: {sorted(dominio)}.")


def validar_separacao_municipal(
    municipios_treino: pd.Series,
    municipios_validacao: pd.Series,
    municipios_teste: pd.Series,
) -> None:
    """Garante independência territorial entre as três partições."""

    grupos_treino = set(municipios_treino.dropna())
    grupos_validacao = set(municipios_validacao.dropna())
    grupos_teste = set(municipios_teste.dropna())

    if grupos_treino & grupos_validacao:
        raise ValueError("Há municípios compartilhados entre treino e validação.")
    if grupos_treino & grupos_teste:
        raise ValueError("Há municípios compartilhados entre treino e teste.")
    if grupos_validacao & grupos_teste:
        raise ValueError("Há municípios compartilhados entre validação e teste.")


def base_minima() -> pd.DataFrame:
    """Produz uma entrada mínima com as features na ordem invertida."""

    dados = {coluna: [1.0] for coluna in FEATURES_ESPERADAS}
    dados["co_uf"] = [35]
    dados["tp_dependencia"] = [2]
    return pd.DataFrame(dados).loc[:, list(reversed(FEATURES_ESPERADAS))]


def test_contrato_possui_as_16_features_aprovadas() -> None:
    assert len(FEATURES_ESPERADAS) == 16
    assert len(FEATURES_ESPERADAS) == len(set(FEATURES_ESPERADAS))


def test_features_nao_contem_target_identificadores_ou_leakage() -> None:
    features = set(FEATURES_ESPERADAS)

    assert TARGET not in features
    assert features.isdisjoint(COLUNAS_AUXILIARES)
    assert features.isdisjoint(COLUNAS_PROIBIDAS_POR_LEAKAGE)


def test_validador_reordena_colunas_corretamente() -> None:
    validada = validar_colunas_inferencia(base_minima())
    assert validada.columns.tolist() == FEATURES_ESPERADAS


@pytest.mark.parametrize("coluna", FEATURES_ESPERADAS[:3])
def test_validador_rejeita_feature_ausente(coluna: str) -> None:
    dados = base_minima().drop(columns=coluna)
    with pytest.raises(ValueError, match="Ausentes"):
        validar_colunas_inferencia(dados)


def test_validador_rejeita_feature_extra() -> None:
    dados = base_minima().assign(coluna_indevida=1)
    with pytest.raises(ValueError, match="Extras"):
        validar_colunas_inferencia(dados)


def test_target_binario_e_completo_e_aceito() -> None:
    validar_target(pd.Series([0, 1, 0, 1], name=TARGET))


@pytest.mark.parametrize(
    "valores, mensagem",
    [([0, None, 1], "ausentes"), ([0, 1, 2], "Domínio inválido")],
)
def test_target_invalido_e_rejeitado(valores, mensagem: str) -> None:
    with pytest.raises(ValueError, match=mensagem):
        validar_target(pd.Series(valores, name=TARGET))


def test_municipios_distintos_entre_particoes_sao_aceitos() -> None:
    validar_separacao_municipal(
        pd.Series([1001, 1002]),
        pd.Series([2001]),
        pd.Series([3001, 3002]),
    )


def test_municipio_compartilhado_e_rejeitado() -> None:
    with pytest.raises(ValueError, match="treino e teste"):
        validar_separacao_municipal(
            pd.Series([1001, 1002]),
            pd.Series([2001]),
            pd.Series([1002, 3001]),
        )

