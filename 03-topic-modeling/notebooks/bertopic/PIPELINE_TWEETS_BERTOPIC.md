# Tweets BR 2022 — Documentação do Corpus e Estratégia de Amostragem

> **Corpus:** Tweets eleições presidenciais Brasil 2022 — PT-BR, textos curtos  
> **Notebooks:** `lda/02_lda_tweets_bre2022.ipynb` · `bertopic/01_bertopic_tweets_bre2022.ipynb`  
> **Configuração:** `01-preprocessing/configs/params.yaml` → corpus `tweets_bre2022`  
> **Última revisão:** 2026-06-29 — corpus `tweets_bre2022_20260629_215159` (8.811 tweets, 4 meses)

---

## Índice

1. [Origem do dataset](#1-origem-do-dataset)
2. [Distribuição bruta por mês](#2-distribuição-bruta-por-mês)
3. [Por que setembro está ausente](#3-por-que-setembro-está-ausente)
4. [Estratégia de amostragem e decisões](#4-estratégia-de-amostragem-e-decisões)
5. [Pipeline de pré-processamento e atrito](#5-pipeline-de-pré-processamento-e-atrito)
6. [Corpus final](#6-corpus-final)
7. [Características para topic modeling](#7-características-para-topic-modeling)
8. [Configuração params.yaml](#8-configuração-paramsyaml)

---

## 1. Origem do dataset

**Fonte:** Zenodo 14834749 — *Brazilian Elections 2022 Twitter Dataset*  
**Dataset original:** ~7 milhões de tweets coletados durante o período eleitoral brasileiro de 2022  
**Amostra disponível localmente:** 200.000 tweets (`data/raw/tweets_bre2022/tweets_bre2022.csv`, 163 MB)

A amostra de 200k foi extraída via amostragem uniforme aleatória (seed=42) do dataset completo do Zenodo. O arquivo bruto contém as colunas:

| Coluna original | Renomeada para | Descrição |
|---|---|---|
| `tweet_content` | `message` | Texto do tweet |
| `created_at` | `data` | Data/hora UTC |
| `tweet_id` | `tweet_id` | ID original do Twitter |
| `post_id` | `post_id` | ID interno do dataset Zenodo |
| `user`, `user_info` | — | Metadados do autor |
| `has_mention`, `mentions`, etc. | — | Metadados de interação |

---

## 2. Distribuição bruta por mês

A coluna `category` (YYYY-MM) é derivada de `data` via `derive_category_from_date` no pré-processamento.

Distribuição nos 200k tweets brutos:

| Mês | Tweets brutos | % do total válido |
|---|---|---|
| 2022-08 | 29.489 | 15,5% |
| **2022-09** | **0** | **0% — AUSENTE** |
| 2022-10 | 21.690 | 11,4% |
| 2022-11 | 64.940 | 34,1% |
| 2022-12 | 74.372 | 39,1% |
| Datas inválidas (NaT) | 9.509 | — |
| **Total válido** | **190.491** | |

> A concentração em novembro e dezembro reflete o calendário eleitoral: o 2º turno foi em 30/10/2022, e a polarização continuou intensa até a posse em 01/01/2023.

---

## 3. Por que setembro está ausente

Setembro de 2022 tem **zero tweets** na amostra de 200k, apesar de ser um mês de alta atividade política (campanha pré-1º turno, 02/10/2022).

A causa provável é a amostragem uniforme sobre o dataset do Zenodo: se o dataset original não cobriu setembro de forma sistemática (por exemplo, por falha de coleta no período), nenhuma instância de setembro aparece na amostra de 200k.

**Decisão:** setembro foi **removido das categorias-alvo** no `params.yaml`. O corpus final cobre os 4 meses disponíveis: agosto, outubro, novembro e dezembro de 2022.

---

## 4. Estratégia de amostragem e decisões

### Objetivo

Produzir um corpus balanceado de ~10k tweets para topic modeling, com representação dos principais momentos da campanha:

| Mês | Evento eleitoral |
|---|---|
| 2022-08 | Campanha presidencial — início oficial (16/08) |
| 2022-10 | 1º turno (02/10) e 2º turno (30/10) |
| 2022-11 | Pós-eleição, contestações e transição |
| 2022-12 | Acampamentos pró-golpe, posse prevista |

### Problema: atrito no pré-processamento

A primeira tentativa usou `n_per_category=2000`, resultando em apenas **4.240 tweets** finais. O motivo foi que o pré-processamento (deduplicação de retweets + detecção de idioma) descarta ~40–70% dos tweets por mês antes de atingir o corpus limpo:

| Mês | `n=2.000` amostrados | Sobreviveram | Taxa de retenção |
|---|---|---|---|
| 2022-08 | 2.000 | 1.210 | 60,5% |
| 2022-10 | 2.000 | 684 | 34,2% |
| 2022-11 | 2.000 | 1.052 | 52,6% |
| 2022-12 | 2.000 | 1.294 | 64,7% |

### Solução: sobre-amostragem no pré-processamento

Para compensar o atrito, `n_per_category` foi aumentado para **5.000** — valor que alimenta o pipeline com mais tweets brutos antes dos filtros, sem risco de esgotar nenhum mês (o menor pool, outubro, tem 21.690 disponíveis):

| Mês | `n=5.000` amostrados | Sobreviveram | Taxa de retenção |
|---|---|---|---|
| 2022-08 | 5.000 | 2.621 | 52,4% |
| 2022-10 | 5.000 | 1.381 | 27,6% |
| 2022-11 | 5.000 | 2.070 | 41,4% |
| 2022-12 | 5.000 | 2.739 | 54,8% |

> Outubro tem retenção consistentemente baixa (~28–34%) mesmo com mais amostras. Isso indica que os tweets de outubro têm alta proporção de retweets e/ou texto não-português detectado pelo `langdetect` — possivelmente reflexo do pico de engajamento (período de votação gera muito retweet de conteúdo institucional).

---

## 5. Pipeline de pré-processamento e atrito

```
200.000 tweets brutos (tweets_bre2022.csv)
        │
        ▼
[1.2] Rename: tweet_content→message, created_at→data
[1.3] add_constants: platform=twitter, election=br_2022
        │
        ▼
[1.5a] derive_category_from_date: data → category (YYYY-MM)
        Filtro implícito: datas inválidas (NaT) → sem categoria → excluídos na amostragem
        190.491 tweets com categoria válida
        │
        ▼
[1.5]  Amostragem estratificada (n_per_category=5.000, seed=42)
        Categorias: ["2022-08", "2022-10", "2022-11", "2022-12"]
        → 20.000 tweets selecionados (5.000 × 4 meses)
        │
        ▼
[2]    Deduplicação de texto exato
        Principal fonte de atrito: retweets têm o prefixo "RT @user: " removido
        pelo demojize, mas o texto citado é frequentemente idêntico
        │
        ▼
[3]    Limpeza: demojize PT-BR, collapse de caracteres repetidos
        URLs e @mentions preservados (relevantes para contexto eleitoral)
        │
        ▼
[4]    Detecção de idioma (langdetect)
        Tweets curtos têm classificação instável — fonte significativa de perdas
        em outubro (campanha com muito conteúdo compartilhado de outras línguas)
        │
        ▼
[5]    Filtros de comprimento: min_doc_length=20 chars, min_words=3
        │
        ▼
[6]    Atribuição de post_id sequencial
        │
        ▼
corpus_limpo.csv  →  8.811 tweets
```

---

## 6. Corpus final

**Versão de produção:** `tweets_bre2022_20260629_215159`  
**Arquivo:** `01-preprocessing/data/output/tweets_bre2022/tweets_bre2022_20260629_215159/corpus_limpo.csv`

### Distribuição por mês

| Mês | Tweets | % do corpus | Palavras médias | Mediana |
|---|---|---|---|---|
| 2022-08 | 2.621 | 29,7% | 19,3 | 16 |
| 2022-10 | 1.381 | 15,7% | 21,6 | 18 |
| 2022-11 | 2.070 | 23,5% | 23,5 | 20 |
| 2022-12 | 2.739 | 31,1% | 26,1 | 24 |
| **Total** | **8.811** | | **22,8 média** | **19 mediana** |

> Outubro está sub-representado (15,7% vs ~25% esperado em distribuição uniforme). Isso é estrutural — reflexo do alto atrito desse mês no pré-processamento, não de viés na amostragem.

### Estatísticas de comprimento (palavras por tweet)

```
count   8.811
mean    22,8
std     13,1
min      5
P25     12
P50     19
P75     33
max     88
```

Distribuição característica de textos curtos: mediana de 19 palavras vs ~300 palavras/doc na Folha. Isso tem impacto direto nas escolhas do BERTopic (`min_cluster_size` menor, `min_samples` menor para lidar com ruído) e no LDA (vocabulário mais enxuto por `filter_extremes`).

---

## 7. Características para topic modeling

| Característica | Valor | Impacto |
|---|---|---|
| Comprimento médio | 22,8 palavras | Embeddings qwen3 de textos curtos têm menor densidade semântica |
| Setembro ausente | 0 tweets | Cobertura temporal: ago–out–nov–dez 2022 |
| Outubro sub-representado | 1.381 (15,7%) | Análise temporal comparativa deve considerar peso diferente |
| @mentions e #hashtags preservados | sim | Enriquecem semântica eleitoral no BERTopic |
| Retweets | removidos por dedup | Reduz redundância, mas remove sinal de amplificação |
| Idioma | PT-BR (langdetect) | ~40–70% de atrito por mês inclui ruído multilíngue |
| Contexto temporal | 4 meses eleitorais | `category` disponível para validação temporal no 03 |

### Ajustes esperados nos modelos em relação à Folha

**BERTopic:**
- `min_cluster_size` menor (textos curtos formam clusters menores)
- `min_samples` menor (reduz outliers, que são especialmente altos em textos curtos)
- `reduce_outliers` pode precisar de sweep específico — parâmetros ótimos da Folha não generalizam diretamente

**LDA:**
- `filter_extremes` com `no_below` provavelmente menor que 5 (corpus menor, vocab mais restrito)
- K provavelmente menor que 20 (menos diversidade temática esperada num corpus focado numa eleição)
- Tópicos de ruído (URLs, @mentions repetidos) mais prováveis

---

## 8. Configuração params.yaml

```yaml
# 01-preprocessing/configs/params.yaml
tweets_bre2022:
  derive_category_from_date:
    date_column: data        # após rename (created_at → data)
    format: "%Y-%m"
    output_column: category

  sample:
    n_per_category: 5000     # sobre-amostragem para compensar atrito (~28-55% por mês)
    category_column: category
    categories:
      - "2022-08"
      # "2022-09" ausente do dataset bruto (0 tweets nos 200k)
      - "2022-10"
      - "2022-11"
      - "2022-12"
```

```yaml
# 03-topic-modeling/configs/params.yaml  (a preencher antes de rodar os modelos)
tweets_bre2022:
  text_column: message
  id_column: post_id
  subdir: tweets_bre2022
  language: pt
  date_column: data
  # ... (ver template e ajustar min_cluster_size, min_samples, K para textos curtos)
```
