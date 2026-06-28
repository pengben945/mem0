"""
BM25 lemmatization for consistent keyword matching.

Uses spaCy's lemmatizer for better handling of:
- Verb forms: attending/attends/attended -> attend
- Comparatives/superlatives: older/oldest -> old
- Plurals: memories -> memory
- Avoids over-stemming: organization != organize

Also includes original -ing forms alongside lemmas to handle cases
where spaCy's context-dependent lemmatization produces inconsistent
results (e.g., "meeting" as noun vs verb -> different lemmas).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def lemmatize_for_bm25(text: str) -> str:
    """
    为 BM25 关键词检索做词形还原预处理。

    会把输入文本转为小写后送入 spaCy，仅保留“非标点、非停用词”的词元，
    并返回以空格拼接的 lemma 字符串（用于后续全文检索/倒排匹配）。

    额外规则：若原词以 `-ing` 结尾且与 lemma 不同，会把原词也一并保留，
    以缓解名词/动词语境导致的词形差异（如 meeting/meet）。

    当 spaCy 不可用时，回退为原始文本，保证主流程不被阻断。
    """
    from mem0.utils.spacy_models import get_nlp_lemma

    nlp = get_nlp_lemma()
    if nlp is None:
        return text

    doc = nlp(text.lower())
    tokens = []

    for token in doc:
        if token.is_punct or token.is_stop:
            continue

        lemma = token.lemma_
        if lemma.isalnum():
            tokens.append(lemma)

        # Also add original if it ends in -ing and differs from lemma.
        # This handles noun/verb ambiguity (meeting/meet, attending/attend).
        if token.text.endswith("ing") and token.text != lemma and token.text.isalnum():
            tokens.append(token.text)

    return " ".join(tokens)
