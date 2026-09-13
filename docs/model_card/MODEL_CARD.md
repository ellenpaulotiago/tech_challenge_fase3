# Model Card — Tech Challenge Fase 3

## Identificação

- Modelo: Hist_Gradient_Boosting_balanceado
- Target: in_alfabetizado
- Classe de interesse: 0 — não alfabetizado
- Limiar: 0.5000
- Data UTC: 2026-09-09T20:44:49.769154+00:00

## Finalidade

Estimar risco de não alfabetização para apoiar análises e priorização de políticas educacionais.

## Dados

- Treino: 1,303,743 registros
- Validação: 363,471 registros
- Teste: 298,881 registros
- Separação territorial por município
- 16 features de entrada

## Desempenho no teste

- PR-AUC: 0.470955
- ROC-AUC: 0.661130
- Recall da classe 0: 0.626182
- Precisão da classe 0: 0.446927
- F1 da classe 0: 0.521583
- Balanced accuracy: 0.617898
- Brier Score: 0.228198

## Generalização e estabilidade

- Recall na validação: 0.712683
- Recall no teste: 0.626182
- Gap teste - validação: -0.086501
- Recall mínimo de desenvolvimento mantido no teste: False
- Região de menor recall: Centro-Oeste (0.096249)
- Região de maior recall: Norte (0.898176)
- UFs com recall igual a zero: 12, 23, 32, 41, 52
- O teste não foi utilizado para reajustar modelo, features ou limiar.

## Comparação pós-hoc

- Comparador: Random_Forest_balanceada otimizado
- Diferença média de PR-AUC: 0.001116
- IC 95% da diferença: [-0.002861, 0.005844]
- Superioridade em PR-AUC confirmada: False
- A comparação não altera a seleção realizada na validação.

## Uso permitido

- Apoio à análise de risco educacional.
- Priorização de investigações e ações complementares.
- Análises agregadas com controle de cobertura e incerteza.

## Uso proibido

- Rotular definitivamente um aluno.
- Substituir avaliação pedagógica ou decisão humana qualificada.
- Aplicar sanções, restringir direitos ou excluir beneficiários.
- Interpretar importância preditiva como causalidade.

## Limitações

- Features contextuais municipais se repetem entre alunos.
- O modelo depende da qualidade e temporalidade das fontes públicas.
- Desempenho pode variar por UF, região e dependência administrativa.
- Probabilidades exigem monitoramento de calibração em novos períodos.
- Mudanças de distribuição exigem nova validação antes do uso.

## Monitoramento recomendado

- Schema, ausências e categorias desconhecidas.
- Drift das features e das probabilidades.
- Recall, precisão, PR-AUC e calibração quando o target estiver disponível.
- Desempenho por subgrupos territoriais e educacionais.
