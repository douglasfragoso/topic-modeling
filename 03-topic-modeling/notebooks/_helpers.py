"""Inlined helpers for the 04-topic-modeling notebooks.

This module consolidates what used to live under ``04-topic-modeling/src/``
(deleted in commit 002f06ac) into a single file co-located with the notebooks
so that ``from _helpers import ...`` works without any sys.path tweaks.

Originating modules (concatenated in order, imports merged + de-duplicated):

1. config.py        — params loading, multi-corpus resolution.
2. embeddings.py    — Ollama + SentenceTransformer embedding helpers with cache.
3. lemmatize.py     — language-aware spaCy lemmatization (PT/EN).
4. naming.py        — topic naming via Ollama LLM (retry + temperature escalation).
5. topic_utils.py   — coherence / exclusivity / FREX / diversity / NPMI etc.
6. lda_pipeline.py  — gensim LDA grid search, train, extract, doc distributions.
7. stm_pipeline.py  — prepare STM input + R subprocess orchestration.

``nmf_model.py`` was intentionally omitted because no notebook imports from it.

Original function bodies are preserved verbatim. Only the import blocks were
merged at the top of the file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from itertools import combinations
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import httpx
import numpy as np
import ollama
import pandas as pd
import yaml
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaMulticore
from sentence_transformers import SentenceTransformer
from sklearn.metrics import cohen_kappa_score


# ===========================================================================
# config.py
# ===========================================================================
"""Configuration loading from params.yaml.

Multi-corpus aware. Backward-compatible with notebooks that pass the full
``params`` dict to ``get_column_names`` — in that case the default corpus
(``params['default_corpus']`` or first key) is used.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_params(path: str = None) -> dict:
    """Load parameters from a YAML file."""
    if path is None:
        path = PROJECT_ROOT / "configs" / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        result = yaml.safe_load(f)
    if not isinstance(result, dict):
        raise ValueError(f"Expected dict from {path}, got {type(result)}")
    return result


def get_corpus_config(params: dict, corpus_id: str = None) -> tuple[str, dict]:
    """Resolve which corpus to use and return its configuration dict.

    Resolution order: explicit arg → params['default_corpus'] → first key.
    Returns (corpus_id, corpus_cfg). Falls back to ('default', params) for
    legacy flat layouts (no 'corpora' key).
    """
    corpora = params.get("corpora")
    if corpora is None:
        return ("default", params)

    if corpus_id is None:
        corpus_id = params.get("default_corpus")
    if corpus_id is None:
        corpus_id = next(iter(corpora))

    if corpus_id not in corpora:
        available = ", ".join(corpora.keys())
        raise KeyError(
            f"Corpus '{corpus_id}' not found in params['corpora']. "
            f"Available: {available}"
        )
    return (corpus_id, corpora[corpus_id])


def apply_bertopic_overrides(bert_cfg: dict, corpus_cfg: dict) -> dict:
    """Overlay ``corpus_cfg['bertopic_overrides']`` onto the shared ``bertopic:``
    block, one level deep (dict-valued keys like ``umap``/``hdbscan``/
    ``reduce_outliers`` are merged key-by-key; scalar keys are replaced).

    The ``bertopic:`` section in params.yaml is shared across corpora — e.g.
    umap/hdbscan/reduce_outliers were calibrated for folha via its own sweep.
    A per-corpus ``bertopic_overrides`` lets another corpus (e.g. tweets, via
    its own ``sweep_bertopic_grid`` run) recalibrate just the knobs its sweep
    touched, without mutating the shared dict or affecting other corpora.
    Returns a NEW dict; ``bert_cfg`` and ``corpus_cfg`` are left untouched.
    """
    import copy
    merged = copy.deepcopy(bert_cfg)
    overrides = corpus_cfg.get("bertopic_overrides") or {}
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **val}
        else:
            merged[key] = val
    return merged


def get_column_names(params_or_cfg: dict, corpus_id: str = None) -> dict:
    """Extract column names. Accepts full ``params`` dict or a ``corpus_cfg``.

    - If a full params dict is passed (has ``corpora`` key), the active corpus
      is resolved via ``get_corpus_config(params, corpus_id)``.
    - If a corpus_cfg dict is passed (has ``text_column`` at top), uses it.
    - Falls back to the legacy nested layout (``data.columns.text``) for
      compatibility with old configs.

    Returns dict with keys: 'text', 'date', 'post_id', 'covariates'.
    """
    # New layout — full params with corpora dict
    if "corpora" in params_or_cfg:
        _, cfg = get_corpus_config(params_or_cfg, corpus_id)
    else:
        cfg = params_or_cfg

    # Per-corpus flat layout
    if "text_column" in cfg:
        return {
            "text": cfg.get("text_column", "message"),
            "date": cfg.get("date_column", "data"),
            "post_id": cfg.get("post_id_column", "post_id"),
            "covariates": cfg.get("covariates", []) or [],
        }

    # Legacy nested layout (data.columns.*)
    columns = cfg.get("data", {}).get("columns", {})
    return {
        "text": columns.get("text", "message"),
        "date": columns.get("date", "data"),
        "post_id": columns.get("post_id", "post_id"),
        "covariates": columns.get("covariates", []) or [],
    }


def get_seed(params: dict) -> int:
    """Extract the global random seed."""
    return params.get("seed", 42)


def make_run_output_dir(base_dir, corpus_id, *, create: bool = True):
    """Diretorio de saida por execucao: ``base_dir/<corpus>_<YYYYmmdd_HHMMSS>``.

    Garante que rodar o mesmo notebook varias vezes (ex.: sweeps com configs
    diferentes, ou re-execucoes) NAO sobrescreva as saidas anteriores — cada
    run grava num subdiretorio carimbado com nome do dataset + data + hora.

    Parameters
    ----------
    base_dir : str | Path
        Diretorio-base do corpus (ex.: ``../data/output/<corpus>``). Caches que
        devem persistir entre runs (ex.: ``lda_metrics.csv`` do grid search)
        continuam vivendo aqui, no PAI do diretorio de run.
    corpus_id : str
        Nome do corpus, usado como prefixo legivel do subdiretorio.
    create : bool
        Cria o diretorio (``mkdir -p``) quando True (default).

    Returns
    -------
    Path
        ``base_dir/<corpus_id>_<timestamp>``.
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{corpus_id}_{stamp}"
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_latest_dir(base_output_dir, *, contains, fallback_dir=None, verbose: bool = True):
    """Resolve o diretório com a versão MAIS RECENTE de ``contains`` (latest-wins).

    Implementa o fluxo "input de um módulo = output do anterior, sem cópia": cada
    módulo grava num subdiretório carimbado ``<nome>_<AAAAMMDD_HHMMSS>`` no seu
    ``data/output/<corpus>/`` e o downstream lê a versão mais nova direto de lá.

    Ordem de resolução:
      1. subdiretórios carimbados de ``base_output_dir`` que contenham ``contains``
         → retorna o de nome (= timestamp) MAIOR;
      2. layout plano legado (``contains`` direto em ``base_output_dir``);
      3. ``fallback_dir`` legado (ex.: ``data/input/<corpus>``).

    Parameters
    ----------
    base_output_dir : str | Path
        ``data/output/<corpus>`` do módulo produtor (ex.: ``../../01-preprocessing/data/output/folha``).
    contains : str
        Nome do arquivo que identifica uma versão válida (ex.: ``corpus_limpo.csv``).
    fallback_dir : str | Path | None
        Diretório legado a usar se nada versionado/plano for encontrado.
    verbose : bool
        Imprime a versão resolvida.

    Returns
    -------
    Path
        Diretório que contém ``contains``. Use ``Path.name`` como string de versão
        para rastreabilidade (provenance) nos metrics.
    """
    base = Path(base_output_dir)
    stamped = (
        [d for d in base.iterdir() if d.is_dir() and (d / contains).exists()]
        if base.exists()
        else []
    )
    if stamped:
        chosen = max(stamped, key=lambda d: d.name)
        if verbose:
            print(f"  versão resolvida: {chosen.name}/ (latest)")
        return chosen
    if base.exists() and (base / contains).exists():
        if verbose:
            print(f"  layout plano legado: {base}")
        return base
    if fallback_dir is not None and (Path(fallback_dir) / contains).exists():
        if verbose:
            print(f"  fallback legado: {fallback_dir}")
        return Path(fallback_dir)
    raise FileNotFoundError(
        f"'{contains}' não encontrado em {base} (subdirs carimbados ou plano) "
        f"nem no fallback {fallback_dir}"
    )


def load_corpus(input_dir, encoding: str = "utf-8", verbose: bool = True) -> pd.DataFrame:
    """Carrega o corpus do diretório de input com auto-detecção de sentimento.

    Prefere ``corpus_com_sentimento.csv`` (gerado pelo módulo 03-sentiment —
    inclui colunas ``sentiment``, ``confidence``, ``sim_positive``,
    ``sim_negative`` além das colunas originais). Cai para ``corpus_limpo.csv``
    (gerado pelo 01-preprocessing) se a versão enriquecida não estiver presente.

    Isso permite que os outputs do 04-topic-modeling (``bertopic_results.csv``,
    ``lda_results.csv``, ``stm_results.csv``) herdem automaticamente a coluna
    ``sentiment`` quando o 03-sentiment tiver rodado antes — sem precisar de
    merge manual posterior.

    Parameters
    ----------
    input_dir : Path or str
        Diretório que contém o(s) CSV(s) (ex.: ``data/input/<corpus>/``).
    encoding : str
        Encoding do CSV (default: utf-8).
    verbose : bool
        Imprime qual arquivo foi carregado e se a coluna ``sentiment`` existe.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        Se nem ``corpus_com_sentimento.csv`` nem ``corpus_limpo.csv``
        estiverem em ``input_dir``.
    """
    input_dir = Path(input_dir)
    enriched = input_dir / "corpus_com_sentimento.csv"
    plain = input_dir / "corpus_limpo.csv"
    if enriched.exists():
        df = pd.read_csv(enriched, encoding=encoding)
        if verbose:
            print(f"Carregado: {enriched.name} ({len(df)} docs, com coluna 'sentiment')")
        return df
    if plain.exists():
        df = pd.read_csv(plain, encoding=encoding)
        if verbose:
            print(
                f"Carregado: {plain.name} ({len(df)} docs, SEM coluna 'sentiment' — "
                "rode 03-sentiment antes para ter cruzamento topic x sentiment automatico)"
            )
        return df
    raise FileNotFoundError(
        f"Nenhum corpus encontrado em '{input_dir}'. "
        "Esperado: 'corpus_com_sentimento.csv' (do 03-sentiment) ou "
        "'corpus_limpo.csv' (do 01-preprocessing)."
    )


# ===========================================================================
# embeddings.py
# ===========================================================================
"""Sentence-transformer embeddings with disk cache."""

logger = logging.getLogger(__name__)


async def _get_ollama_embedding(client, text, model, dimension, timeout=120.0, _retries=5):
    """Get embedding for a single text via Ollama API.

    If ``dimension`` is provided and smaller than the model's native size, the
    returned vector is **truncated to the first ``dimension`` components and
    re-normalized to unit length** — this is the Matryoshka-style truncation
    used by Qwen3-Embedding (and any model trained with MRL). When the native
    size is already smaller, returns as-is and logs a warning (the request
    cannot be honoured without retraining).

    Retries up to ``_retries`` times on 5xx errors **and on transient transport
    failures** (ConnectError/timeout) with exponential backoff — needed when
    Ollama runs in CPU/hybrid mode and the server momentarily drops the
    connection or restarts (queues saturated, model reload under memory
    pressure). Only 4xx (client) errors propagate immediately.
    """
    import asyncio as _asyncio
    last_exc = None
    for attempt in range(_retries):
        try:
            resp = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            # Servidor caiu/reiniciou ou timeout — retenta com backoff.
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Ollama conexão falhou (%s) (attempt %d/%d), retrying in %ds…",
                type(exc).__name__, attempt + 1, _retries, wait,
            )
            await _asyncio.sleep(wait)
            continue
        if resp.status_code < 500:
            resp.raise_for_status()
            break
        last_exc = resp
        wait = 2 ** attempt
        logger.warning("Ollama 5xx (attempt %d/%d), retrying in %ds…", attempt + 1, _retries, wait)
        await _asyncio.sleep(wait)
    else:
        # Retries esgotados: relança 5xx (Response) ou a última exceção de transporte.
        if isinstance(last_exc, httpx.Response):
            last_exc.raise_for_status()
        elif last_exc is not None:
            raise last_exc
    emb = resp.json()["embedding"]

    if dimension and len(emb) != dimension:
        if len(emb) > dimension:
            # Matryoshka truncation: take first D components, renormalize.
            arr = np.asarray(emb[:dimension], dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm > 0:
                arr = arr / norm
            return arr.tolist()
        else:
            logger.warning(
                "Ollama returned %dd embedding, requested %dd > native; "
                "returning native size.", len(emb), dimension,
            )
    return emb


async def _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent=1, timeout=120.0):
    """Get embeddings for multiple texts with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    async with httpx.AsyncClient() as client:
        async def bounded(text):
            async with semaphore:
                return await _get_ollama_embedding(client, text, model, dimension, timeout=timeout)
        tasks = [bounded(t) for t in texts]
        return await asyncio.gather(*tasks)


def get_ollama_embeddings(
    texts: list[str],
    model: str,
    dimension: int = None,
    max_concurrent: int = 5,
    timeout: float = 120.0,
) -> np.ndarray:
    """Get embeddings from Ollama API. Returns numpy array.

    Handles Jupyter event loop via nest_asyncio if available.
    """
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(
                    asyncio.run,
                    _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
                ).result()
        else:
            results = loop.run_until_complete(
                _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
            )
    except RuntimeError:
        results = asyncio.run(
            _get_ollama_embeddings_batch(texts, model, dimension, max_concurrent, timeout=timeout)
        )
    return np.array(results, dtype=np.float32)


def get_or_compute_embeddings(
    texts: list[str],
    model_name: str,
    cache_path: str,
    backend: str = "sentence_transformers",
    dimension: int = None,
    batch_size: int = 32,
    show_progress: bool = True,
    timeout: float = 120.0,
) -> np.ndarray:
    """Load embeddings from cache or compute and save them.

    Supports two backends:
    - "sentence_transformers": uses SentenceTransformer (default)
    - "ollama": uses Ollama API via async HTTP

    If cache exists and shape matches, returns cached embeddings.
    """
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if cached.shape[0] == len(texts):
            return cached

    if backend == "ollama":
        embeddings = get_ollama_embeddings(texts, model_name, dimension=dimension, timeout=timeout)
    else:
        model = SentenceTransformer(model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings


# ===========================================================================
# lemmatize.py
# ===========================================================================
"""Lemmatization with language-aware spaCy model loading.

Supports PT-BR (`pt_core_news_lg`) and EN (`en_core_web_sm`).
Models are lazy-loaded and cached at module level.
"""

_NLP_CACHE: dict = {}

_LANG_TO_MODEL = {
    "pt": "pt_core_news_lg",
    "en": "en_core_web_sm",
}


def _get_nlp(lang: str):
    if lang not in _LANG_TO_MODEL:
        raise ValueError(
            f"Unsupported lang '{lang}'. Supported: {list(_LANG_TO_MODEL)}"
        )
    if lang not in _NLP_CACHE:
        import spacy
        try:
            _NLP_CACHE[lang] = spacy.load(_LANG_TO_MODEL[lang])
        except OSError as e:
            raise OSError(
                f"spaCy model '{_LANG_TO_MODEL[lang]}' not installed. "
                f"Run: python -m spacy download {_LANG_TO_MODEL[lang]}"
            ) from e
    return _NLP_CACHE[lang]


def lemmatize_corpus(
    docs: list[str],
    lang: str,
    params: dict,
    batch_size: int = 256,
    n_process: int = 1,
) -> tuple[list[list[str]], Dictionary]:
    """Lemmatize a list of documents and build a filtered gensim Dictionary.

    Args:
        docs: raw document strings.
        lang: 'pt' or 'en'.
        params: dict with `lda.no_below` and `lda.no_above` for filter_extremes.
        batch_size: spaCy nlp.pipe batch size.
        n_process: spaCy nlp.pipe parallel workers (1 on Windows recommended).

    Returns:
        (tokenized_docs, dictionary) where dictionary already had
        filter_extremes applied per params['lda'].
    """
    nlp = _get_nlp(lang)

    tokenized: list[list[str]] = []
    # Process via nlp.pipe for speed on large corpora
    for spacy_doc in nlp.pipe(docs, batch_size=batch_size, n_process=n_process):
        tokens = [
            tok.lemma_.lower()
            for tok in spacy_doc
            if (not tok.is_stop) and tok.is_alpha and len(tok.lemma_) > 2
        ]
        tokenized.append(tokens)

    dictionary = Dictionary(tokenized)
    lda_cfg = params.get("lda", {})
    dictionary.filter_extremes(
        no_below=lda_cfg.get("no_below", 5),
        no_above=lda_cfg.get("no_above", 0.5),
    )
    return tokenized, dictionary


# ===========================================================================
# naming.py
# ===========================================================================
"""Topic naming via Ollama LLM (PT-BR, with retry/temperature escalation)."""


def warmup_ollama(
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    timeout: float = 120.0,
) -> bool:
    """Warm-up call para carregar o modelo na memoria antes do batch real.

    Cold start de modelos grandes em CPU pode demorar ~30-60s. Sem warm-up,
    a primeira chamada de naming sofre esse custo + risco maior de timeout.

    Returns
    -------
    bool
        True se o modelo respondeu, False caso contrario.
    """
    try:
        client = ollama.Client(host=base_url)
        t0 = time.time()
        response = client.chat(
            model=model,
            messages=[{"role": "user", "content": "Responda apenas: OK"}],
            options={"temperature": 0.0, "num_predict": 8, "think": False},
        )
        dt = time.time() - t0
        content = response.get("message", {}).get("content", "")
        print(f"  [warmup] Modelo {model} respondeu em {dt:.1f}s: {content[:50]!r}")
        return True
    except Exception as e:
        print(f"  [warmup] Falhou: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Prompt + response cleaning
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks and surrounding quotes.

    Qwen3 may emit <think>...</think> followed by the answer. If the model
    returns *only* a thinking block (sem resposta após), extract a fallback
    candidate from inside the thinking block (last short line that looks like
    a label) — this prevents 100% fallback rate observed em prod.
    """
    original = text or ""
    cleaned = re.sub(r"<think>.*?</think>", "", original, flags=re.DOTALL).strip()
    cleaned = cleaned.strip('"').strip("'").strip()

    if cleaned:
        return cleaned

    # Modelo emitiu apenas <think>...</think> sem resposta. Tentar extrair
    # uma frase candidata de dentro do thinking (ultima linha curta).
    inner = re.findall(r"<think>(.*?)</think>", original, flags=re.DOTALL)
    if not inner:
        return ""
    last_block = inner[-1]
    # Pegar linhas curtas (3-50 chars) que pareçam título/rótulo
    candidates = [
        ln.strip(' "\'.,:;-')
        for ln in last_block.split("\n")
        if 3 <= len(ln.strip()) <= 50
    ]
    # Ultima candidata costuma ser a conclusao do raciocinio
    return candidates[-1] if candidates else ""


def _clean_label(raw: str) -> str:
    """Extract first useful line and strip prefixes/punctuation."""
    if not raw:
        return ""
    line = next((l.strip() for l in _strip_thinking(raw).split("\n") if l.strip()), "")
    line = line.strip('"\'`').rstrip(".!?;:")
    for pref in ("Rótulo:", "Rotulo:", "Label:", "Topic:", "Tópico:", "Nome:"):
        if line.lower().startswith(pref.lower()):
            line = line[len(pref):].strip()
    # Cap at 7 words — enough for descriptive labels without truncating mid-phrase.
    parts = line.split()
    if len(parts) > 7:
        line = " ".join(parts[:7])
    return line


_HTML_ENTITY_TOKENS = frozenset({
    "atilde", "ccedil", "iacute", "eacute", "aacute", "otilde", "uacute",
    "ocirc", "acirc", "ecirc", "iuml", "agrave", "egrave", "ograve",
    "amp", "nbsp", "quot", "apos", "lt", "gt",
    "ccedil atilde", "novamente representando",
})


def _filter_keywords(keywords: list[str]) -> list[str]:
    """Remove HTML entity tokens and other known-spurious terms from keyword list."""
    return [k for k in keywords if k.lower() not in _HTML_ENTITY_TOKENS]


def _build_prompt_pt(
    keywords: list[str],
    example_docs: list[str] | None,
    top_n: int,
    doc_max_chars: int,
    max_docs: int,
) -> str:
    """PT-BR prompt body (rótulos de tópico, jornalístico/notícias)."""
    clean_kws = _filter_keywords(keywords)
    keywords_str = ", ".join(clean_kws[:top_n])
    docs_block = ""
    if example_docs:
        snippets = []
        for i, d in enumerate(example_docs[:max_docs]):
            text = d if isinstance(d, str) else str(d)
            snippets.append(f"Documento {i + 1}: {text[:doc_max_chars].strip()}...")
        if snippets:
            docs_block = "\n\n" + "\n\n".join(snippets)

    return (
        "Voce e um especialista em analise de topicos de noticias jornalisticas brasileiras.\n\n"
        f"Palavras-chave: {keywords_str}"
        f"{docs_block}\n\n"
        "Tarefa: criar um rotulo descritivo em portugues brasileiro (4-7 palavras).\n"
        "O rotulo deve nomear o TEMA central, nao descrever uma acao.\n"
        "Responda APENAS com o rotulo, sem explicacoes, sem aspas, sem prefixos.\n"
        "Use substantivos concretos. Capitalize apenas a primeira palavra.\n\n"
        "/no_think\n\n"
        "Rotulo:"
    )


def _build_prompt_en(
    keywords: list[str],
    example_docs: list[str] | None,
    top_n: int,
    doc_max_chars: int,
    max_docs: int,
) -> str:
    """English prompt body (topic labels, generic news/benchmark domain)."""
    keywords_str = ", ".join(keywords[:top_n])
    docs_block = ""
    if example_docs:
        snippets = []
        for i, d in enumerate(example_docs[:max_docs]):
            text = d if isinstance(d, str) else str(d)
            snippets.append(f"Document {i + 1}: {text[:doc_max_chars].strip()}...")
        if snippets:
            docs_block = "\n\n" + "\n\n".join(snippets)

    return (
        "You are an expert in topic analysis for English-language news corpora.\n\n"
        f"Keywords: {keywords_str}"
        f"{docs_block}\n\n"
        "Task: create a short label in English (3-5 words).\n"
        "Respond ONLY with the label, no explanations, no quotes, no prefixes.\n"
        "Use concrete nouns. Capitalize only the first word.\n\n"
        "/no_think\n\n"
        "Label:"
    )


def build_prompt(
    keywords: list[str],
    example_docs: list[str] | None = None,
    top_n: int = 15,
    doc_max_chars: int = 200,
    max_docs: int = 1,
    lang: str = "pt",
) -> str:
    """Build a topic-naming prompt, language-aware via ``lang``.

    F2 — calibração para Qwen3:4b em CPU:
    - top_n=15 (vs 25 anterior) — reduz ruído da cauda longa
    - doc_max_chars=200 (vs 400) — encurta prompt
    - max_docs=1 (vs 2-3) — apenas o doc mais representativo
    - sufixo /no_think — instrui Qwen3 a desabilitar thinking mode

    Parameters
    ----------
    lang : str
        ``"pt"`` (default, preserva comportamento histórico — rotulos em
        portugues brasileiro) ou ``"en"`` (rotulos em ingles, para corpora
        EN). Qualquer outro valor cai em ``"pt"``.
    """
    builder = _build_prompt_en if lang == "en" else _build_prompt_pt
    return builder(keywords, example_docs, top_n, doc_max_chars, max_docs)


# ---------------------------------------------------------------------------
# Backend call (single attempt + retry wrapper)
# ---------------------------------------------------------------------------


_DEFAULT_SYSTEM_PROMPT_PT = (
    "Voce e um especialista em topicos. Responda APENAS com "
    "um rotulo curto (3-5 palavras), sem explicacoes."
)

_DEFAULT_SYSTEM_PROMPT_EN = (
    "You are a topic-labeling expert. Respond ONLY with "
    "a short label (3-5 words), no explanations."
)


def _call_ollama_once(
    keywords: list[str],
    example_docs: list[str] | None,
    model: str,
    base_url: str,
    temperature: float,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> str:
    """One Ollama call. Raises on transport/HTTP error.

    Parameters
    ----------
    prompt_builder : callable, optional
        Funcao customizada `(keywords, example_docs) -> str` para gerar o user
        prompt. Quando None, usa ``build_prompt`` padrao (language-aware via
        ``lang``). Permite que o notebook edite o prompt direto na cell sem
        mexer no naming.py.
    system_prompt : str, optional
        System prompt customizado. Default: enxuto otimizado para topic
        naming, selecionado por ``lang`` ("pt" ou "en").
    lang : str
        Idioma do rotulo quando ``prompt_builder``/``system_prompt`` nao sao
        passados explicitamente. Default ``"pt"`` preserva comportamento
        historico.
    """
    if prompt_builder is None:
        prompt_builder = lambda kws, docs: build_prompt(kws, docs, lang=lang)  # noqa: E731
    if system_prompt is None:
        system_prompt = (
            _DEFAULT_SYSTEM_PROMPT_EN if lang == "en" else _DEFAULT_SYSTEM_PROMPT_PT
        )

    client = ollama.Client(host=base_url)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_builder(keywords, example_docs)},
        ],
        options={
            "temperature": temperature,
            "top_p": 0.95,
            "num_predict": 256,        # Maior para acomodar thinking + resposta
            "repeat_penalty": 1.1,
            "think": False,            # Ollama recente: desabilita thinking no Qwen3
        },
    )
    return _clean_label(response["message"]["content"])


def _fallback_label(keywords: list[str], n: int = 3, lang: str = "pt") -> str:
    """Emergency label: top-N keywords joined.

    ``lang`` apenas afeta a mensagem usada quando ``keywords`` esta vazio
    (default ``"pt"`` preserva comportamento historico).
    """
    kws = [k for k in keywords[:n] if k]
    if not kws:
        return "Topic without label" if lang == "en" else "Tópico sem rótulo"
    label = ", ".join(kws)
    return label[0].upper() + label[1:]


def _smart_fallback(keywords: list[str], lang: str = "pt") -> str:
    """Fallback inteligente: gera frase nominal em vez de lista de keywords.

    Em vez de "saude, hospital, leitos" (lista, parece MMR keywords), gera
    "Saude, hospital e leitos" — pelo menos é uma frase nominal natural.

    ``lang="en"`` gera o conector em ingles ("and") em vez de "e", para
    corpora EN. Default ``"pt"`` preserva comportamento historico.
    """
    kws = [k for k in keywords[:3] if k and len(k) >= 2]
    no_label_msg = "Topic without label" if lang == "en" else "Topico sem rotulo"
    connector = "and" if lang == "en" else "e"
    if not kws:
        return no_label_msg
    if len(kws) == 1:
        return kws[0].capitalize()
    if len(kws) == 2:
        return f"{kws[0].capitalize()} {connector} {kws[1]}"
    return f"{kws[0].capitalize()}, {kws[1]} {connector} {kws[2]}"


def _looks_like_keyword_list(label: str, keywords: list[str], threshold: float = 0.5) -> bool:
    """Detecta se o LLM devolveu lista de keywords disfarçada de rótulo.

    Um rótulo legítimo é uma frase nominal coerente. Se >threshold das partes
    separadas por vírgula são keywords do input, é lista (ruim).
    """
    if not label or "," not in label:
        return False
    parts = [p.strip().lower().rstrip(".") for p in label.split(",")]
    if len(parts) < 2:
        return False
    kws_lower = {k.lower() for k in keywords[:15] if k}
    matches = sum(1 for p in parts if p in kws_lower)
    return matches / len(parts) > threshold


def name_topic(
    keywords: list[str],
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    example_docs: list[str] | None = None,
    max_attempts: int = 3,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> str:
    """Name a single topic with retry + temperature escalation.

    Strategy
    --------
    Up to ``max_attempts`` calls; temperature escalates 0.2 → 0.5 → 0.8 to
    encourage variety on retries when a previous attempt returned an empty or
    too-short label. After all attempts fail, returns the top-3 keyword
    fallback so the pipeline never crashes on a single bad topic.

    Parameters
    ----------
    keywords : list[str]
        Representative keywords (top-10 recommended).
    model : str
        Ollama model identifier.
    base_url : str
        Ollama server base URL.
    example_docs : list[str] | None
        Optional representative documents to ground the prompt.
    max_attempts : int
        Max LLM calls before falling back.
    prompt_builder : callable, optional
        Override do prompt builder. Quando None, usa ``build_prompt`` com
        ``lang`` aplicado.
    system_prompt : str, optional
        Override do system prompt. Quando None, usa o default de ``lang``.
    lang : str
        Idioma do rotulo ("pt" ou "en") quando ``prompt_builder`` /
        ``system_prompt`` nao sao passados explicitamente. Tambem usado para
        selecionar o idioma do fallback (``_smart_fallback``). Default
        ``"pt"`` preserva comportamento historico.
    """
    fallback = _smart_fallback(keywords, lang=lang)
    temperatures = [0.2, 0.5, 0.8]
    last_error: str | None = None

    for attempt in range(max_attempts):
        temp = temperatures[min(attempt, len(temperatures) - 1)]
        try:
            label = _call_ollama_once(
                keywords, example_docs, model, base_url, temp,
                prompt_builder=prompt_builder, system_prompt=system_prompt,
                lang=lang,
            )
            if not label or len(label) < 3:
                last_error = (
                    f"resposta vazia/curta (tentativa {attempt + 1}, "
                    f"temp={temp})"
                )
            elif _looks_like_keyword_list(label, keywords):
                # LLM devolveu lista de keywords disfarcada de rotulo. Rejeita.
                last_error = (
                    f"resposta parece lista de keywords (tentativa "
                    f"{attempt + 1}, temp={temp}): {label!r}"
                )
            else:
                # Rotulo valido
                return label
        except Exception as e:
            last_error = (
                f"erro na chamada (tentativa {attempt + 1}): "
                f"{type(e).__name__}: {e}"
            )

        time.sleep(1)

    print(f"  [naming] fallback acionado: {last_error} -> '{fallback}'")
    return fallback


def name_all_topics(
    topics_keywords: dict[int, list[str]],
    model: str = "qwen3:4b",
    base_url: str = "http://localhost:11434",
    example_docs_map: dict[int, list[str]] | None = None,
    max_attempts: int = 3,
    prompt_builder=None,
    system_prompt: str | None = None,
    lang: str = "pt",
) -> dict[int, str]:
    """Name every topic. ``example_docs_map`` is optional per-topic context.

    Pass ``prompt_builder`` and/or ``system_prompt`` to override the defaults
    direto do notebook (sem editar src/naming.py).

    Parameters
    ----------
    lang : str
        Idioma do rotulo ("pt" ou "en"), usado para selecionar o prompt
        builder/system prompt/fallback padrao quando ``prompt_builder`` /
        ``system_prompt`` nao sao passados. Tipicamente vem de
        ``corpus_cfg["language"]`` (ver ``params.yaml``). Default ``"pt"``
        preserva comportamento historico para notebooks que nao passam este
        argumento.
    """
    return {
        tid: name_topic(
            kws,
            model=model,
            base_url=base_url,
            example_docs=(example_docs_map or {}).get(tid),
            max_attempts=max_attempts,
            prompt_builder=prompt_builder,
            system_prompt=system_prompt,
            lang=lang,
        )
        for tid, kws in topics_keywords.items()
    }


# ===========================================================================
# topic_utils.py
# ===========================================================================
"""Topic evaluation utilities: coherence, stability, Likert, Kappa, export."""


def compute_coherence_cv(
    topics_keywords: dict[int, list[str]],
    texts: list[list[str]],
    dictionary,
) -> float:
    """Compute c_v coherence score using gensim."""
    from gensim.models import CoherenceModel

    topics_list = [
        [w for w in kws if isinstance(w, str) and w]
        for kws in topics_keywords.values()
    ]
    topics_list = [kws for kws in topics_list if kws]
    if not topics_list:
        return 0.0
    cm = CoherenceModel(
        topics=topics_list,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1,  # avoid multiprocessing issues on Windows
    )
    return cm.get_coherence()


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


def compute_stability(
    results_per_seed: dict[int, dict[int, list[str]]],
) -> tuple[float, float]:
    """Compute topic stability across seeds via greedy Jaccard matching."""
    seeds = list(results_per_seed.keys())
    if len(seeds) < 2:
        return float("nan"), float("nan")

    pair_scores = []
    for seed_a, seed_b in combinations(seeds, 2):
        topics_a = results_per_seed[seed_a]
        topics_b = results_per_seed[seed_b]

        matched_scores = []
        used_b = set()
        for tid_a, kws_a in topics_a.items():
            best_score = 0.0
            best_tid = None
            for tid_b, kws_b in topics_b.items():
                if tid_b in used_b:
                    continue
                score = compute_jaccard(set(kws_a), set(kws_b))
                if score > best_score:
                    best_score = score
                    best_tid = tid_b
            if best_tid is not None:
                used_b.add(best_tid)
            matched_scores.append(best_score)

        if matched_scores:
            pair_scores.append(np.mean(matched_scores))

    if not pair_scores:
        return 0.0, 0.0
    return float(np.mean(pair_scores)), float(np.std(pair_scores))


def generate_likert_sheet(
    all_results: list[dict],
    output_path: str,
    mapping_path: str,
    seed: int = 42,
) -> None:
    """Generate blinded Likert evaluation CSV and secret mapping."""
    df = pd.DataFrame(all_results)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "eval_id", range(1, len(df) + 1))

    mapping = df[["eval_id", "model"]].copy()
    mapping.to_csv(mapping_path, index=False)

    likert = df.drop(columns=["model", "topic_id"])
    likert["representatividade"] = ""
    likert["coerencia"] = ""
    likert["utilidade"] = ""
    likert.to_csv(output_path, index=False)


def compute_kappa(ratings_a: list[int], ratings_b: list[int]) -> float:
    """Cohen's Kappa between two raters."""
    return float(cohen_kappa_score(ratings_a, ratings_b))


def export_results(
    topics: list[int],
    probs: list[list[float]],
    names: dict[int, str],
    texts: list[str],
    output_path: str,
    topic_type: str = "probabilistic",
    granularity: str = "unit",
    post_ids: list[str] = None,  # NEW: original post_id values for cross-axis joins
) -> None:
    """Export model results to standardized CSV.

    Parameters
    ----------
    topics: dominant topic index per document
    probs: full probability distribution per document
    names: mapping from topic_id to human-readable name
    texts: raw text per document
    output_path: destination CSV path
    topic_type: 'semantic' or 'probabilistic'
    granularity: 'unit' or 'monthly'
    post_ids: optional list of original post_id strings.  When provided a
        ``post_id`` column is inserted as the first column so outputs can be
        joined doc-a-doc com `03-sentiment/data/output/<corpus>/sentiment_results.csv`.
        ``doc_id`` (sequential integer) is kept for backward compatibility.
    """
    def _to_native(p):
        """Convert a probability row to native Python floats for JSON serialization."""
        return [float(v) for v in p]

    df = pd.DataFrame({
        "doc_id": range(len(topics)),
        "text": texts,
        "topic_id": topics,
        "topic_name": [names.get(t, f"Topic_{t}") for t in topics],
        "topic_prob_distribution": [json.dumps(_to_native(p)) for p in probs],
    })
    df["topic_type"] = topic_type
    df["granularity"] = granularity

    if post_ids is not None:
        df.insert(0, "post_id", post_ids)

    df.to_csv(output_path, index=False, encoding="utf-8")


def export_topics_for_eval(
    topics_keywords: dict[int, list[str]],
    topics_names: dict[int, str],
    model_name: str,
    output_path: str,
) -> None:
    """Export topics in standard format for comparative evaluation."""
    rows = []
    for tid, kws in topics_keywords.items():
        rows.append({
            "topic_id": tid,
            "topic_name": topics_names.get(tid, f"Topic_{tid}"),
            "keywords": ", ".join(kws),
            "model": model_name,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def compute_exclusivity(topics_keywords: dict[int, list[str]]) -> float:
    """Proportion of keywords that appear in only one topic.

    Coarse binary version (presence/absence). Use ``compute_exclusivity_ctfidf``
    for the continuous version that follows the BERTopic literature.
    """
    all_words = []
    for kws in topics_keywords.values():
        all_words.extend(kws)
    if not all_words:
        return 0.0
    unique_count = sum(1 for w in set(all_words) if all_words.count(w) == 1)
    return unique_count / len(set(all_words))


def compute_exclusivity_ctfidf(
    topics_keywords: dict[int, list[str]],
    topic_word_scores: dict[int, dict[str, float]],
    top_n: int = 10,
) -> tuple[float, dict[int, float]]:
    """Continuous exclusivity via c-TF-IDF weight distribution.

    For each top-N keyword w of topic t, exclusivity is the share of total
    c-TF-IDF mass of w concentrated in t:

        excl(w, t) = score(w, t) / sum_{t'} score(w, t')

    Topic exclusivity is the mean over its top-N keywords. The function returns
    both the corpus-level mean and the per-topic dict.

    Parameters
    ----------
    topics_keywords : dict[int, list[str]]
        topic_id -> ordered keyword list (top-N already truncated or fuller).
    topic_word_scores : dict[int, dict[str, float]]
        topic_id -> {word: c-TF-IDF score}. Required.
    top_n : int
        Number of keywords per topic to score.
    """
    per_topic: dict[int, float] = {}
    for tid, kws in topics_keywords.items():
        if not kws:
            per_topic[tid] = 0.0
            continue
        scores = []
        for w in kws[:top_n]:
            num = topic_word_scores.get(tid, {}).get(w, 0.0)
            denom = sum(
                topic_word_scores.get(other_tid, {}).get(w, 0.0)
                for other_tid in topics_keywords
            )
            if denom > 0:
                scores.append(num / denom)
        per_topic[tid] = float(np.mean(scores)) if scores else 0.0

    mean_excl = float(np.mean(list(per_topic.values()))) if per_topic else 0.0
    return mean_excl, per_topic


def compute_topic_diversity(
    topics_keywords: dict[int, list[str]],
    top_k: int = 10,
) -> float:
    """Topic Diversity (Dieng et al., 2020).

    TD = |unique(top-k keywords across all topics)| / (k * |valid_topics|)

    Range: [0, 1]. Higher = topics share less vocabulary.

    Skips topics with fewer than 2 valid keywords (typical of -1/outlier or
    degenerate clusters when MMR returns empty strings).
    """
    valid_word_lists: list[list[str]] = []
    for tid, kws in topics_keywords.items():
        if tid == -1:
            continue
        clean = [w for w in kws[:top_k] if w]
        if len(clean) < 2:
            continue
        valid_word_lists.append(clean)

    if not valid_word_lists:
        return 0.0

    all_words = [w for kws in valid_word_lists for w in kws]
    unique_count = len(set(all_words))
    return unique_count / (top_k * len(valid_word_lists))


def compute_semantic_coherence(
    topics_keywords: dict[int, list[str]],
    embedding_func,
) -> float:
    """Mean cosine similarity between embeddings of top-N terms per topic.

    Parameters
    ----------
    topics_keywords: dict mapping topic_id to list of keywords
    embedding_func: callable that takes list[str] and returns np.ndarray of shape (n, dim)
    """
    from sklearn.metrics.pairwise import cosine_similarity
    scores = []
    for tid, kws in topics_keywords.items():
        if len(kws) < 2:
            continue
        try:
            embs = embedding_func(kws)
        except Exception as exc:  # backend de embeddings indisponível (ex.: Ollama fora)
            import warnings
            warnings.warn(
                "compute_semantic_coherence: backend de embeddings indisponível "
                f"({type(exc).__name__}: {exc}). Retornando NaN. "
                "Verifique se o Ollama está no ar (ollama serve) na porta 11434.",
                RuntimeWarning,
                stacklevel=2,
            )
            return float("nan")
        sim = cosine_similarity(embs)
        n = len(kws)
        total = sum(float(sim[i][j]) for i in range(n) for j in range(i + 1, n))
        pairs = n * (n - 1) / 2
        scores.append(total / pairs if pairs > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_frex(
    topics_keywords: dict[int, list[str]],
    topic_word_scores: dict[int, dict[str, float]],
    weight: float = 0.5,
) -> dict[int, list[str]]:
    """FREX-ranked keywords using a *linear* combination (legacy).

    Kept for backwards compatibility with earlier evaluation scripts. New code
    should prefer ``compute_frex_score`` which implements the harmonic-mean
    formulation of Airoldi & Bischof (2016) used in STM and reported in the
    BERTopic literature.

    topic_word_scores: REQUIRED. Dict[topic_id, Dict[word, score]].
    weight: 0=pure exclusivity, 1=pure frequency.
    """
    word_topic_count = {}
    for kws in topics_keywords.values():
        for w in kws:
            word_topic_count[w] = word_topic_count.get(w, 0) + 1

    frex_results = {}
    for tid, kws in topics_keywords.items():
        scores = topic_word_scores.get(tid, {})
        scored_raw = []
        for w in kws:
            freq_score = scores.get(w, 0.0)
            excl_score = 1.0 / word_topic_count[w] if word_topic_count[w] > 0 else 0.0
            scored_raw.append((w, freq_score, excl_score))

        if scored_raw:
            all_freq = [fs for _, fs, _ in scored_raw]
            max_freq = max(all_freq)
            min_freq = min(all_freq)
            range_freq = max_freq - min_freq if max_freq != min_freq else 1.0
            scored = [
                (w, weight * ((fs - min_freq) / range_freq) + (1 - weight) * es)
                for w, fs, es in scored_raw
            ]
        else:
            scored = []

        scored.sort(key=lambda x: x[1], reverse=True)
        frex_results[tid] = [w for w, _ in scored]

    return frex_results


def compute_frex_score(
    topics_keywords: dict[int, list[str]],
    topic_word_matrix: np.ndarray,
    vocab_index: dict[str, int],
    topic_index: dict[int, int],
    top_n: int = 10,
    w_freq: float = 0.5,
) -> tuple[float, dict[int, float]]:
    """FREX score (Airoldi & Bischof, 2016) — harmonic mean of percentile ranks.

    For each topic t and each top-N keyword w:

        F(w, t) = percentile rank of c-TF-IDF(w, t) in topic t's row
        E(w, t) = percentile rank of exclusivity(w, t) in topic t's row
        FREX(w, t) = 1 / (w_freq / F + (1 - w_freq) / E)      (harmonic mean)

    Topic FREX is the mean over its top-N keywords; the corpus-level FREX is
    the mean across topics. Returns (mean_frex, per_topic_dict).

    Parameters
    ----------
    topics_keywords : dict[int, list[str]]
        topic_id -> ordered keyword list.
    topic_word_matrix : np.ndarray
        Matrix of shape (n_topics, vocab_size) with c-TF-IDF (or equivalent
        topic-word weights). Rows correspond to topic_index keys.
    vocab_index : dict[str, int]
        Maps vocabulary token -> column index in ``topic_word_matrix``.
    topic_index : dict[int, int]
        Maps topic_id -> row index in ``topic_word_matrix``. Must include
        every topic appearing in ``topics_keywords``; -1 (outliers) is ignored
        if absent.
    top_n : int
        Keywords per topic to score.
    w_freq : float
        Weight on frequency in the harmonic mean (0.5 is the canonical value).
    """
    from scipy.stats import rankdata

    if topic_word_matrix.size == 0:
        return 0.0, {}

    # Total word mass across topics for exclusivity computation.
    valid_rows = [topic_index[t] for t in topics_keywords if t in topic_index and t != -1]
    if not valid_rows:
        return 0.0, {}
    word_total = topic_word_matrix[valid_rows].sum(axis=0) + 1e-12

    per_topic: dict[int, float] = {}
    for tid, kws in topics_keywords.items():
        if tid == -1 or tid not in topic_index:
            continue
        row = topic_word_matrix[topic_index[tid]]
        excl = row / word_total

        # Percentile ranks within this topic
        f_rank = rankdata(row) / len(row)
        e_rank = rankdata(excl) / len(excl)

        idxs = [vocab_index[w] for w in kws[:top_n] if w in vocab_index]
        if not idxs:
            per_topic[tid] = 0.0
            continue

        scores = []
        for i in idxs:
            f = max(f_rank[i], 1e-12)
            e = max(e_rank[i], 1e-12)
            frex = 1.0 / (w_freq / f + (1 - w_freq) / e)
            scores.append(frex)
        per_topic[tid] = float(np.mean(scores))

    mean_frex = float(np.mean(list(per_topic.values()))) if per_topic else 0.0
    return mean_frex, per_topic


def _build_cooccurrence_matrix(texts: list[list[str]], vocab: set[str]) -> tuple[dict, dict]:
    """Pre-compute co-occurrence counts for all word pairs in vocab."""
    from collections import defaultdict
    cooccur = defaultdict(int)
    doc_freq = defaultdict(int)

    for doc in texts:
        words_in_doc = set(doc) & vocab
        for w in words_in_doc:
            doc_freq[w] += 1
        words_list = sorted(words_in_doc)
        for i, wa in enumerate(words_list):
            for wb in words_list[i + 1:]:
                cooccur[(wa, wb)] += 1

    return dict(cooccur), dict(doc_freq)


def compute_npmi(
    topics_keywords_a: dict[int, list[str]],
    topics_keywords_b: dict[int, list[str]],
    texts: list[list[str]],
) -> float:
    """Normalized Pointwise Mutual Information between two sets of topic keywords.

    Measures concordance between topics discovered by different models.
    Pre-computes co-occurrence matrix for efficiency.
    """
    from math import log

    vocab = set()
    for kws in list(topics_keywords_a.values()) + list(topics_keywords_b.values()):
        vocab.update(kws)

    n_docs = len(texts)
    if n_docs == 0:
        return 0.0

    cooccur, doc_freq = _build_cooccurrence_matrix(texts, vocab)

    def pairwise_npmi(words_a, words_b):
        scores = []
        for wa in words_a:
            for wb in words_b:
                if wa == wb:
                    continue
                p_a = doc_freq.get(wa, 0) / n_docs
                p_b = doc_freq.get(wb, 0) / n_docs
                key = tuple(sorted([wa, wb]))
                co = cooccur.get(key, 0) / n_docs
                if co == 0 or p_a == 0 or p_b == 0:
                    continue
                pmi = log(co / (p_a * p_b))
                log_co = -log(co)
                if log_co == 0:
                    # co == 1.0: words always co-occur → perfect NPMI = 1.0
                    scores.append(1.0)
                    continue
                npmi_val = pmi / log_co
                scores.append(npmi_val)
        return float(np.mean(scores)) if scores else 0.0

    all_scores = []
    for tid_a, kws_a in topics_keywords_a.items():
        best = max(
            (pairwise_npmi(kws_a, kws_b) for kws_b in topics_keywords_b.values()),
            default=0.0,
        )
        all_scores.append(best)

    return float(np.mean(all_scores)) if all_scores else 0.0


def compute_diversity(topic_distributions: list[list[float]]) -> float:
    """Compute topic diversity via entropy of document-topic distribution.

    Higher entropy = topics distribute more evenly across documents.
    """
    if not topic_distributions:
        return 0.0
    arr = np.array(topic_distributions)
    avg_dist = arr.mean(axis=0)
    avg_dist = avg_dist[avg_dist > 0]
    if len(avg_dist) == 0:
        return 0.0
    entropy = -np.sum(avg_dist * np.log2(avg_dist))
    max_entropy = np.log2(len(avg_dist)) if len(avg_dist) > 1 else 1.0
    return float(entropy / max_entropy)


# ===========================================================================
# bertopic_sweep.py
# ===========================================================================
"""BERTopic outlier-strategy/threshold sweep — multi-seed scoring of
reduce_outliers variants via coherence, diversity, exclusivity, FREX and
Jaccard stability.

Corpus-agnostic, same calling convention as ``grid_search_k`` below: pure
functions, no hardcoded corpus/paths/seeds, no file I/O — the notebook
supplies ``build_model`` + the corpus's docs/embeddings/tokenized/dictionary
and owns any caching of the returned DataFrames. Fills the gap the BERTopic
docs leave open: there's no built-in way to compare
off/c-tf-idf/embeddings/probabilities/distributions (or a threshold sweep
within one strategy) empirically across seeds.
"""


def _bertopic_postprocess(
    model,
    docs: list[str],
    embeddings: np.ndarray,
    strategy: str,
    threshold: float = 0.0,
    reduce_nr: int | None = None,
) -> tuple[float, float, int]:
    """Apply reduce_outliers(strategy, threshold) + conditional reduce_topics,
    in the canonical B18 order (reduce_outliers -> update_topics ->
    reduce_topics -> update_topics). ``strategy="off"`` skips step 1 entirely.

    Returns (outlier_pre, outlier_post, n_raw_topics).
    """
    topics0 = model.topics_
    outlier_pre = sum(1 for t in topics0 if t == -1) / len(topics0)
    n_raw = len([t for t in set(topics0) if t != -1])

    if strategy != "off":
        kw = dict(documents=docs, topics=topics0, strategy=strategy, threshold=threshold)
        if strategy == "embeddings":
            kw["embeddings"] = embeddings
        elif strategy == "probabilities":
            kw["probabilities"] = model.probabilities_
        new_topics = model.reduce_outliers(**kw)
        model.update_topics(
            docs, topics=new_topics, vectorizer_model=model.vectorizer_model,
            ctfidf_model=model.ctfidf_model, representation_model=model.representation_model,
        )

    n_now = len([t for t in model.get_topic_info()["Topic"] if t != -1])
    if reduce_nr and n_now > reduce_nr:
        model.reduce_topics(docs, nr_topics=reduce_nr)
        model.update_topics(
            docs, topics=model.topics_, vectorizer_model=model.vectorizer_model,
            ctfidf_model=model.ctfidf_model, representation_model=model.representation_model,
        )

    outlier_post = sum(1 for t in model.topics_ if t == -1) / len(model.topics_)
    return outlier_pre, outlier_post, n_raw


def _bertopic_sweep_metrics(
    model,
    tokenized: list[list[str]],
    dictionary,
    top_n_metrics: int = 20,
) -> tuple[dict, dict[int, list[str]]]:
    """Score a post-processed BERTopic model: K, c_v, diversity, exclusivity, FREX.

    Returns (metrics_dict, {topic_id: keywords}); the keyword map is what
    callers accumulate across seeds to feed ``compute_stability``.
    """
    topics = model.topics_
    valid_ids = sorted(t for t in set(topics) if t != -1)
    topics_keywords = {
        tid: [w for w, _ in (model.get_topic(tid) or []) if isinstance(w, str)]
        for tid in valid_ids
    }
    topic_index = {tid: i for i, tid in enumerate(sorted(model.get_topics().keys()))}
    ctfidf_matrix = model.c_tf_idf_.toarray() if model.c_tf_idf_ is not None else np.zeros((1, 1))
    vocab = model.vectorizer_model.get_feature_names_out()
    vocab_index = {w: i for i, w in enumerate(vocab)}

    # Exclusividade (F1): topic_word_scores DEVE vir da MATRIZ COMPLETA c-TF-IDF,
    # nao do top-N de model.get_topic() (cache). Se usar so o cache, o denominador
    # da exclusividade ignora o peso real de uma palavra nos OUTROS topicos (tratado
    # como 0 quando ela nao esta no top-N deles), inflando a exclusividade. Espelha
    # o fix F1 ja presente no notebook da folha — mantem a coluna Exclus do
    # sweep comparavel com a metrica do pipeline principal.
    topic_word_scores: dict[int, dict[str, float]] = {}
    if model.c_tf_idf_ is not None:
        for tid in valid_ids:
            if tid in topic_index:
                row = ctfidf_matrix[topic_index[tid]]
                topic_word_scores[tid] = {
                    vocab[j]: float(row[j]) for j in range(len(vocab)) if row[j] > 0
                }
            else:
                topic_word_scores[tid] = {}
    else:
        topic_word_scores = {tid: {w: float(s) for w, s in model.get_topic(tid)} for tid in valid_ids}

    tk_metrics = {tid: kws[:top_n_metrics] for tid, kws in topics_keywords.items()}
    try:
        cv = compute_coherence_cv(tk_metrics, tokenized, dictionary)
    except ValueError:
        # gensim CoherenceModel exige que cada topico tenha >=1 token no
        # dictionary (unigramas); com ngram_range=(1,2) um topico pode cair
        # 100% em bigramas fora do vocabulario unigrama — mesmo modo de falha
        # ja tratado no pipeline principal (notebook, celula "Metricas
        # quantitativas completas"), aqui so precisa nao derrubar o sweep.
        cv = float("nan")
    div = compute_topic_diversity(topics_keywords, top_k=top_n_metrics)
    excl, _ = compute_exclusivity_ctfidf(topics_keywords, topic_word_scores, top_n=top_n_metrics)
    frex, _ = compute_frex_score(
        topics_keywords, topic_word_matrix=ctfidf_matrix,
        vocab_index=vocab_index, topic_index=topic_index, top_n=top_n_metrics, w_freq=0.5,
    )
    seed_kws = {tid: [w for w in kws if w] for tid, kws in topics_keywords.items()}
    seed_kws = {tid: kws for tid, kws in seed_kws.items() if len(kws) >= 2}
    return dict(K=len(valid_ids), cv=cv, div=div, excl=excl, frex=frex), seed_kws


def sweep_outlier_strategies(
    build_model,
    docs: list[str],
    embeddings: np.ndarray,
    tokenized: list[list[str]],
    dictionary,
    seeds: Iterable[int],
    strategies: Iterable[str] = ("off", "c-tf-idf", "embeddings", "probabilities", "distributions"),
    reduce_nr: int | None = None,
    top_n_metrics: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-seed sweep over ``reduce_outliers`` strategies.

    For each seed, fits ONE base model via ``build_model(seed)`` then
    deep-copies it once per strategy so every variant starts from the same
    fit — only post-processing differs. Scores each variant with
    c_v/diversity/exclusivity/FREX plus, per strategy, multi-seed Jaccard
    stability (``compute_stability``).

    Parameters
    ----------
    build_model : Callable[[int], BERTopic]
        Factory returning a fresh, *unfit* BERTopic instance configured for
        the given seed (UMAP/HDBSCAN ``random_state``). The caller owns the
        embedding model, vectorizer, ctfidf and representation model choices
        — this function only fits/copies/post-processes/scores.
    docs, embeddings : corpus texts and pre-computed embeddings (aligned).
    tokenized, dictionary : gensim inputs for c_v coherence.
    seeds : seeds to fit the base model with (stability needs >= 2).
    strategies : reduce_outliers strategies to compare; ``"off"`` skips
        reduce_outliers entirely (B18 minus step 1).
    reduce_nr : if set and the post-RO topic count exceeds it, applies
        ``reduce_topics(nr_topics=reduce_nr)`` — B18 step 3.
    top_n_metrics : keywords per topic fed to the metric functions.

    Returns
    -------
    (raw, agg) : ``raw`` has one row per (seed, strategy); ``agg`` has one
        row per strategy with means + Jaccard stability across seeds.
    """
    import copy

    rows = []
    kws_by_strategy: dict[str, dict[int, dict[int, list[str]]]] = {s: {} for s in strategies}

    for seed in seeds:
        base = build_model(seed)
        base.fit_transform(docs, embeddings=embeddings)
        for strat in strategies:
            m = copy.deepcopy(base)
            o_pre, o_post, n_raw = _bertopic_postprocess(
                m, docs, embeddings, strat, reduce_nr=reduce_nr,
            )
            met, seed_kws = _bertopic_sweep_metrics(m, tokenized, dictionary, top_n_metrics)
            kws_by_strategy[strat][seed] = seed_kws
            rows.append(dict(strategy=strat, seed=seed, n_raw=n_raw,
                              outlier_pre=o_pre, outlier_post=o_post, **met))

    raw = pd.DataFrame(rows)

    agg_rows = []
    for strat in strategies:
        sub = raw[raw.strategy == strat]
        if sub.empty:
            continue
        stab_m, stab_s = (
            compute_stability(kws_by_strategy[strat])
            if len(kws_by_strategy[strat]) >= 2 else (float("nan"), float("nan"))
        )
        agg_rows.append(dict(
            strategy=strat, n_seeds=len(sub),
            K=round(sub["K"].mean(), 1),
            outlier_pre=round(sub["outlier_pre"].mean(), 4),
            outlier_post=round(sub["outlier_post"].mean(), 4),
            C_v=round(sub["cv"].mean(), 4), Diversity=round(sub["div"].mean(), 4),
            Exclus=round(sub["excl"].mean(), 4), FREX=round(sub["frex"].mean(), 4),
            Stability=round(stab_m, 4), Stab_std=round(stab_s, 4),
        ))

    agg = pd.DataFrame(agg_rows)
    return raw, agg


def sweep_outlier_threshold(
    build_model,
    docs: list[str],
    embeddings: np.ndarray,
    tokenized: list[list[str]],
    dictionary,
    seeds: Iterable[int],
    grids: dict[str, list[float]],
    reduce_nr: int | None = None,
    top_n_metrics: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-seed sweep over per-strategy ``reduce_outliers`` thresholds.

    Same fit-once-per-seed/copy-per-variant structure as
    ``sweep_outlier_strategies``, but holds the strategy fixed per grid entry
    and varies ``threshold`` instead — answers "does partial reallocation
    beat threshold=0 (reallocate everything) or strategy='off' (reallocate
    nothing)?". ``grids`` maps strategy -> list of thresholds to try, e.g.
    ``{"c-tf-idf": [0.08, 0.12, 0.15]}`` (calibrate per strategy: c-tf-idf and
    probabilities scores live in [0, 1]; embeddings cosine similarity is
    typically high — calibrate against a seed=42 probe first).
    """
    import copy

    rows = []
    kws_by: dict[tuple[str, float], dict[int, dict[int, list[str]]]] = {}

    for seed in seeds:
        base = build_model(seed)
        base.fit_transform(docs, embeddings=embeddings)
        for strat, grid in grids.items():
            for thr in grid:
                m = copy.deepcopy(base)
                o_pre, o_post, n_raw = _bertopic_postprocess(
                    m, docs, embeddings, strat, threshold=thr, reduce_nr=reduce_nr,
                )
                met, seed_kws = _bertopic_sweep_metrics(m, tokenized, dictionary, top_n_metrics)
                kws_by.setdefault((strat, thr), {})[seed] = seed_kws
                rows.append(dict(strategy=strat, threshold=thr, seed=seed, n_raw=n_raw,
                                  outlier_pre=o_pre, outlier_post=o_post, **met))

    raw = pd.DataFrame(rows)

    agg_rows = []
    for (strat, thr), kmap in kws_by.items():
        sub = raw[(raw.strategy == strat) & (raw.threshold == thr)]
        stab_m, stab_s = compute_stability(kmap) if len(kmap) >= 2 else (float("nan"), float("nan"))
        agg_rows.append(dict(
            strategy=strat, threshold=thr, n_seeds=len(sub),
            K=round(sub["K"].mean(), 1),
            outlier_post=round(sub["outlier_post"].mean(), 4),
            C_v=round(sub["cv"].mean(), 4), Diversity=round(sub["div"].mean(), 4),
            Exclus=round(sub["excl"].mean(), 4), FREX=round(sub["frex"].mean(), 4),
            Stability=round(stab_m, 4), Stab_std=round(stab_s, 4),
        ))

    agg = (
        pd.DataFrame(agg_rows).sort_values(["strategy", "threshold"]).reset_index(drop=True)
        if agg_rows else pd.DataFrame(agg_rows)
    )
    return raw, agg


def sweep_bertopic_grid(
    build_model,
    docs: list[str],
    embeddings: np.ndarray,
    tokenized: list[list[str]],
    dictionary,
    seeds: Iterable[int],
    n_neighbors_grid: Iterable[int],
    min_cluster_size_grid: Iterable[int],
    min_samples_grid: Iterable[int | None] = (None,),
    reduce_nr_grid: Iterable[int | None] = (None,),
    cluster_selection_methods_grid: Iterable[str] = ("leaf",),
    outlier_strategy: str = "off",
    outlier_threshold: float = 0.0,
    top_n_metrics: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Multi-seed grid search over UMAP ``n_neighbors`` x HDBSCAN
    ``min_cluster_size`` x HDBSCAN ``min_samples`` x ``reduce_topics(nr_topics)``
    target — the structural-hyperparameter analogue of ``grid_search_k`` for LDA.

    UMAP ``n_neighbors`` and the two HDBSCAN knobs (``min_cluster_size`` +
    ``min_samples``) each force a fresh UMAP+HDBSCAN fit, so this refits once
    per ``(seed, n_neighbors, min_cluster_size, min_samples)`` and reuses that
    fit across ``reduce_nr_grid`` (post-processing only, same fit-once/copy-many
    pattern as the other sweeps here). ``min_samples`` controls how conservative
    HDBSCAN is (higher -> more points pushed to -1/outlier): the knob that most
    affects outlier rate on short/noisy text (e.g. tweets), which is exactly why
    it belongs in the grid alongside ``min_cluster_size``.

    A single-seed version of this grid search is cheap but unreliable: ARJ's
    own validation (see ``relatorio_arj.md`` sec 7.1) found a single-seed
    n_neighbors/min_cluster_size grid picked the wrong n_neighbors, only
    caught by multi-seed stability scoring. Always pass >= 2 seeds and trust
    ``Stability``/``Stab_std`` over a single run's coherence number.

    Parameters
    ----------
    build_model : Callable[[int, int, int, int | None, str], BERTopic]
        Factory ``(seed, n_neighbors, min_cluster_size, min_samples,
        cluster_selection_method) -> unfit BERTopic``. The caller owns
        embedding/vectorizer/ctfidf/representation choices.
    min_samples_grid : Iterable[int | None]
        HDBSCAN ``min_samples`` values to try. Default ``(None,)`` runs a single
        point using the factory's own default.
    cluster_selection_methods_grid : Iterable[str]
        HDBSCAN ``cluster_selection_method`` values to try. Default ``("leaf",)``.
        Pass ``("leaf", "eom")`` to compare both methods. ``"eom"`` (Excess of
        Mass) tends to produce fewer, larger clusters; ``"leaf"`` tends to produce
        more, smaller clusters.
    outlier_strategy, outlier_threshold : applied identically to every grid
        point via ``_bertopic_postprocess`` (B18 order). Should always be
        ``"off"`` here to isolate the structural-hyperparameter effect — compare
        outlier strategies separately with ``sweep_outlier_strategies``.

    Returns
    -------
    (raw, agg) : ``raw`` has one row per (seed, n_neighbors, min_cluster_size,
        min_samples, cluster_selection_method, reduce_nr); ``agg`` has one row
        per grid point with means + Jaccard stability across seeds.
    """
    import copy

    rows = []
    kws_by_combo: dict[tuple, dict[int, dict[int, list[str]]]] = {}

    for seed in seeds:
        for nn in n_neighbors_grid:
            for mcs in min_cluster_size_grid:
                for ms in min_samples_grid:
                    for csm in cluster_selection_methods_grid:
                        base = build_model(seed, nn, mcs, ms, csm)
                        base.fit_transform(docs, embeddings=embeddings)
                        for nr in reduce_nr_grid:
                            m = copy.deepcopy(base)
                            o_pre, o_post, n_raw = _bertopic_postprocess(
                                m, docs, embeddings, outlier_strategy,
                                threshold=outlier_threshold, reduce_nr=nr,
                            )
                            met, seed_kws = _bertopic_sweep_metrics(m, tokenized, dictionary, top_n_metrics)
                            key = (nn, mcs, ms, csm, nr)
                            kws_by_combo.setdefault(key, {})[seed] = seed_kws
                            rows.append(dict(
                                n_neighbors=nn, min_cluster_size=mcs, min_samples=ms,
                                cluster_selection_method=csm, reduce_nr=nr, seed=seed,
                                n_raw=n_raw, outlier_pre=o_pre, outlier_post=o_post, **met,
                            ))

    raw = pd.DataFrame(rows)

    def _nan_safe_mask(col: pd.Series, value) -> pd.Series:
        # A None grid value (min_samples=None / reduce_nr=None) becomes NaN once
        # mixed with ints in the DataFrame; `col == None` is always False (not
        # NaN-aware), so it must be special-cased rather than compared directly.
        return col.isna() if value is None else col == value

    agg_rows = []
    for (nn, mcs, ms, csm, nr), kmap in kws_by_combo.items():
        sub = raw[
            (raw.n_neighbors == nn) & (raw.min_cluster_size == mcs)
            & _nan_safe_mask(raw.min_samples, ms)
            & (raw.cluster_selection_method == csm)
            & _nan_safe_mask(raw.reduce_nr, nr)
        ]
        stab_m, stab_s = compute_stability(kmap) if len(kmap) >= 2 else (float("nan"), float("nan"))
        agg_rows.append(dict(
            n_neighbors=nn, min_cluster_size=mcs, min_samples=ms,
            cluster_selection_method=csm, reduce_nr=nr,
            n_seeds=len(sub),
            K=round(sub["K"].mean(), 1),
            outlier_pre=round(sub["outlier_pre"].mean(), 4),
            outlier_post=round(sub["outlier_post"].mean(), 4),
            C_v=round(sub["cv"].mean(), 4), Diversity=round(sub["div"].mean(), 4),
            Exclus=round(sub["excl"].mean(), 4), FREX=round(sub["frex"].mean(), 4),
            Stability=round(stab_m, 4), Stab_std=round(stab_s, 4),
        ))

    agg = (
        pd.DataFrame(agg_rows)
        .sort_values(["n_neighbors", "min_cluster_size", "min_samples", "cluster_selection_method", "reduce_nr"])
        .reset_index(drop=True)
        if agg_rows else pd.DataFrame(agg_rows)
    )
    return raw, agg


# ===========================================================================
# lda_pipeline.py
# ===========================================================================
"""LDA pipeline: grid search K, train, extract topics, qualitative report.

Corpus-agnostic. Importado pelos notebooks de 04-topic-modeling.
"""


def grid_search_k(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k_range: Iterable[int],
    seed: int = 42,
    passes: int = 10,
    workers: int | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Train LDA for each K and return (cv_scores, perplexity_scores).

    Fits each model once and computes both C_v coherence and perplexity
    (exp(-log_perplexity_per_word)) to avoid refitting.
    """
    cv_scores: dict[int, float] = {}
    perplexity_scores: dict[int, float] = {}
    for k in k_range:
        model = LdaMulticore(
            corpus=corpus_bow,
            id2word=dictionary,
            num_topics=k,
            random_state=seed,
            passes=passes,
            workers=workers,
        )
        cm = CoherenceModel(
            model=model,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        cv_scores[k] = float(cm.get_coherence())
        perplexity_scores[k] = float(np.exp(-model.log_perplexity(corpus_bow)))
    return cv_scores, perplexity_scores


def grid_search_alpha_eta(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    tokenized: list[list[str]],
    k: int,
    alpha_grid: list | None = None,
    eta_grid: list | None = None,
    seed: int = 42,
    passes: int = 10,
    workers: int | None = None,
) -> tuple[pd.DataFrame, object, object]:
    """Varre combinações alpha × eta com K fixo. Retorna (df_results, best_alpha, best_eta).

    alpha — prior Dirichlet doc→tópico:
      'symmetric': uniforme (default gensim); 'asymmetric': 1/k (favorece temas dominantes);
      float baixo (0.01): docs concentrados em poucos tópicos; alto (0.5): difusos.

    eta — prior Dirichlet tópico→palavra:
      None: gensim default (1/K); float baixo (0.01): tópicos com vocab concentrado
      (mais distintos); alto (0.1+): maior sobreposição de vocabulário entre tópicos.

    df_results ordenado por cv decrescente — primeira linha é o vencedor.
    """
    if alpha_grid is None:
        alpha_grid = ["symmetric", "asymmetric", 0.1, 0.5]
    if eta_grid is None:
        eta_grid = [None, 0.01, 0.1]
    rows: list[dict] = []
    for alpha in alpha_grid:
        for eta in eta_grid:
            model = LdaMulticore(
                corpus=corpus_bow,
                id2word=dictionary,
                num_topics=k,
                random_state=seed,
                passes=passes,
                workers=workers,
                alpha=alpha,
                eta=eta,
            )
            cm = CoherenceModel(
                model=model, texts=tokenized, dictionary=dictionary, coherence="c_v"
            )
            rows.append({
                "alpha": alpha,
                "eta": eta,
                "cv": float(cm.get_coherence()),
                "perplexity": float(np.exp(-model.log_perplexity(corpus_bow))),
            })
    df = (
        pd.DataFrame(rows)
        .sort_values("cv", ascending=False)
        .reset_index(drop=True)
    )
    return df, df.iloc[0]["alpha"], df.iloc[0]["eta"]


def train_lda(
    corpus_bow: list[list[tuple[int, int]]],
    dictionary: Dictionary,
    k: int,
    seed: int = 42,
    passes: int = 20,
    workers: int | None = None,
    alpha: str | float | list = "symmetric",
    eta: str | float | list | None = None,
) -> LdaMulticore:
    """Train final LDA model with chosen K, alpha and eta."""
    return LdaMulticore(
        corpus=corpus_bow,
        id2word=dictionary,
        num_topics=k,
        random_state=seed,
        passes=passes,
        workers=workers,
        alpha=alpha,
        eta=eta,
    )


def extract_topics_keywords(
    model: LdaMulticore,
    k: int,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """Return {topic_id: [top_n keywords]} ordered by topic-word probability."""
    out: dict[int, list[str]] = {}
    for tid in range(k):
        terms = model.show_topic(tid, topn=top_n)
        out[tid] = [w for w, _ in terms]
    return out


def compute_doc_distributions(
    model: LdaMulticore,
    corpus_bow: list[list[tuple[int, int]]],
    k: int,
) -> tuple[list[int], list[list[float]]]:
    """Return (dominant_topic_per_doc, full_distribution_per_doc)."""
    dominant: list[int] = []
    full: list[list[float]] = []
    for bow in corpus_bow:
        dist = dict(model.get_document_topics(bow, minimum_probability=0.0))
        row = [float(dist.get(t, 0.0)) for t in range(k)]
        full.append(row)
        dominant.append(int(np.argmax(row)))
    return dominant, full


def qualitative_report(
    topics_keywords: dict[int, list[str]],
    names: dict[int, str] | None = None,
    coherent_threshold: float = 0.0,
) -> str:
    """Markdown structured report. Heuristic categorization placeholders.

    The bulk of qualitative analysis is annotated manually after this report
    in the notebook (or in docs/baselines/). This function produces the
    template + auto-grouped sections by simple heuristics.
    """
    names = names or {tid: ", ".join(kws[:3]) for tid, kws in topics_keywords.items()}
    lines: list[str] = []
    lines.append("# Qualitative report\n")
    lines.append(f"**Total topics:** {len(topics_keywords)}\n")
    lines.append("## Tópicos coesos\n_(preencher manualmente: tópicos com keywords semanticamente próximas e tema único)_\n")
    lines.append("## Tópicos genéricos / discurso\n_(preencher manualmente: keywords genéricas, sem tema concreto)_\n")
    lines.append("## Tópicos lixo / boilerplate\n_(preencher manualmente: keywords são fórmulas, hashtags, headers, etc.)_\n")
    lines.append("## Redundâncias\n_(preencher manualmente: tópicos diferentes que cobrem o mesmo tema)_\n")
    lines.append("## Candidatos a stop word\n_(preencher manualmente: termos repetidos em 3+ tópicos sem agregar diferenciação)_\n")
    lines.append("## Tabela de tópicos\n")
    lines.append("| ID | Nome | Top keywords |\n|---|---|---|\n")
    for tid in sorted(topics_keywords):
        kws = topics_keywords[tid]
        kws_s = ", ".join(kws[:10])
        nm = names.get(tid, "—")
        lines.append(f"| T{tid} | {nm} | {kws_s} |\n")
    return "".join(lines)


# ===========================================================================
# stm_pipeline.py
# ===========================================================================
"""STM pipeline: prepare input, run R subprocess, parse JSON output.

Corpus-agnostic. The R script (`scripts/run_stm.R`) does searchK + train +
estimateEffect natively. Python only orchestrates.
"""


def prepare_stm_input(
    df: pd.DataFrame,
    text_col: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Validate and reshape df → STM-friendly input.

    Required columns: ``text_col`` plus all listed ``covariates``.
    Output columns: ['text', *covariates] in this order. R reads the first
    column as the text column when no explicit `text` exists.
    """
    if text_col not in df.columns:
        raise ValueError(
            f"text_col '{text_col}' ausente. Cols: {df.columns.tolist()}"
        )
    missing = [c for c in covariates if c not in df.columns]
    if missing:
        raise ValueError(
            f"Covariável(s) ausente(s): {missing}. "
            f"Cols disponíveis: {df.columns.tolist()}"
        )
    out = df[[text_col, *covariates]].copy()
    out = out.rename(columns={text_col: "text"})
    out = out.dropna(subset=["text", *covariates]).reset_index(drop=True)
    return out


def run_stm_subprocess(
    input_csv: str,
    k_min: int,
    k_max: int,
    prevalence_formula: str,
    rscript_path: str,
    seed: int,
    output_dir: str,
    output_json: str,
    rscript_file: str,
    top_n: int = 10,
    language: str = "portuguese",
    timeout_sec: int = 3600,
) -> tuple[dict[int, list[str]], np.ndarray, dict, dict]:
    """Run scripts/run_stm.R and parse its JSON output.

    Returns:
        keywords: {topic_id: [frex words]}
        theta: ndarray (n_docs, k) doc-topic distribution
        effects: dict of prevalence effects per covariate
        meta: misc metadata (k, convergence_iters, formula, etc.)
    """
    cmd = [
        rscript_path,
        rscript_file,
        "--input", input_csv,
        "--k_min", str(k_min),
        "--k_max", str(k_max),
        "--prevalence", prevalence_formula,
        "--seed", str(seed),
        "--output_dir", output_dir,
        "--top_n", str(top_n),
        "--language", language,
        "--output_json", output_json,
    ]
    print(f"[stm] cmd: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(
            f"run_stm.R exit {r.returncode}\n=== stdout ===\n{r.stdout}\n=== stderr ===\n{r.stderr}"
        )
    print(r.stdout[-2000:])  # tail of R log

    if not Path(output_json).exists():
        raise RuntimeError(f"R did not write {output_json}. stdout tail:\n{r.stdout[-500:]}")

    with open(output_json, "r", encoding="utf-8") as f:
        d = json.load(f)

    k = int(d["k"])
    keywords = {int(tid): list(kws) for tid, kws in d["topics"].items()}
    theta = np.array(d["theta"])
    effects = d.get("prevalence_effects", {}) or {}
    meta = {
        "k": k,
        "convergence_iters": d.get("convergence_iters"),
        **d.get("metadata", {}),
    }
    return keywords, theta, effects, meta
