from inspect import signature

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler
)


FEATURES_CATEGORICAS = [
    "co_uf",
    "tp_dependencia"
]


FEATURES_NUMERICAS = [
    "atlas_idhm",
    "atlas_idhm_e",
    "atlas_renda_pc",
    "atlas_indice_gini",
    "atlas_prop_pobreza_criancas",
    "atlas_taxa_criancas_dom_sem_fund",
    "censo_prop_mat_2ano_internet_aprendizagem",
    "censo_prop_mat_2ano_alimentacao",
    "censo_prop_mat_2ano_biblioteca_sala_leitura",
    "fundeb_receita_contribuicao",
    "fundeb_complementacao_uniao",
    "fundeb_receita_total",
    "indicador_meta_final_2025",
    "indicador_pc_aluno_alfabetizado_2024"
]


FEATURES_INDICADORES = [
    "atlas_dados_ausentes",
    "censo_dados_ausentes",
    "fundeb_dados_ausentes",
    "indicadores_dados_ausentes"
]


class IndicadoresAusencia(BaseEstimator, TransformerMixin):
    """Cria indicadores binários de ausência por fonte de dados."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformado = X.copy()

        X_transformado["atlas_dados_ausentes"] = (
            X_transformado["atlas_idhm"]
            .isna()
            .astype(int)
        )

        X_transformado["censo_dados_ausentes"] = (
            X_transformado[
                "censo_prop_mat_2ano_internet_aprendizagem"
            ]
            .isna()
            .astype(int)
        )

        X_transformado["fundeb_dados_ausentes"] = (
            X_transformado[
                "fundeb_receita_contribuicao"
            ]
            .isna()
            .astype(int)
        )

        X_transformado["indicadores_dados_ausentes"] = (
            X_transformado[
                "indicador_meta_final_2025"
            ]
            .isna()
            .astype(int)
        )

        return X_transformado


def converter_para_float32(X):
    """Reduz o consumo de memória da saída densa."""

    return X.astype(
        np.float32,
        copy=False
    )


def criar_pre_processamento(
    padronizar=False,
    saida_densa=False
):
    """Constrói a arquitetura de pré-processamento.

    Parameters
    ----------
    padronizar : bool, default=False
        Ativa a padronização das variáveis numéricas.
    saida_densa : bool, default=False
        Quando True, configura o OneHotEncoder para
        produzir uma matriz densa. Essa configuração
        é necessária para estimadores que não aceitam
        matrizes esparsas, como o
        HistGradientBoostingClassifier.

    O valor padrão False preserva o comportamento
    utilizado pelos modelos já existentes.
    """

   
    etapas_numericas: list[
        tuple[str, BaseEstimator]
    ] = [
        (
            "imputacao",
            SimpleImputer(strategy="median")
        )
    ]

    if padronizar:
        etapas_numericas.append(
            (
                "padronizacao",
                StandardScaler()
            )
        )
        
    pipeline_numerica = Pipeline(
        steps=etapas_numericas
    )

    parametros_encoder = {
        "handle_unknown": "ignore"
    }

    if "sparse_output" in signature(
        OneHotEncoder
    ).parameters:
        parametros_encoder["sparse_output"] = (
            not saida_densa
        )
    else:
        parametros_encoder["sparse"] = (
            not saida_densa
        )

    pipeline_categorica = Pipeline(
        steps=[
            (
                "encoding",
                OneHotEncoder(**parametros_encoder)
            )
        ]
    )

    pre_processamento = ColumnTransformer(
        transformers=[
            (
                "numericas",
                pipeline_numerica,
                FEATURES_NUMERICAS
            ),
            (
                "categoricas",
                pipeline_categorica,
                FEATURES_CATEGORICAS
            ),
            (
                "indicadores_ausencia",
                "passthrough",
                FEATURES_INDICADORES
            )
        ]
    )

    etapas_pre_processamento = [
            (
                "engenharia_atributos",
                IndicadoresAusencia()
            ),
            (
                "pre_processamento",
                pre_processamento
            )
    ]

    if saida_densa:
        etapas_pre_processamento.append(
            (
                "conversao_float32",
                FunctionTransformer(
                    converter_para_float32
                )
            )
        )

    return Pipeline(
        steps=etapas_pre_processamento
    )
