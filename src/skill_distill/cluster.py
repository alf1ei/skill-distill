"""零依赖的 TF-IDF + 余弦相似度聚类（中文按字符 bigram，英文按单词）。"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .models import Conversation

STOP = set("不要 希望 喜欢 帮我 示例 例子 example phone email token id_number ".split()) | set("的了是我你他她它在和就都也要有这那吗呢吧啊么一个不会可以什么怎么如果因为所以但是还是然后然后the a an and or to of in is are for on with it this that be i you".split())
_CJK = re.compile(r"[\u4e00-\u9fff]+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.-]{1,}")


def tokenize(text: str) -> list[str]:
    toks: list[str] = []
    for w in _WORD.findall(text):
        w = w.lower().rstrip(".")
        if w not in STOP:
            toks.append(w)
    for seg in _CJK.findall(text):
        for i in range(len(seg) - 1):
            bg = seg[i:i + 2]
            if bg in STOP or (bg[0] in STOP and bg[1] in STOP):
                continue
            toks.append(bg)
    return toks


def tfidf(docs: list[list[str]], max_df: float = 0.5) -> tuple[list[dict[str, float]], dict[str, float]]:
    """max_df：在超过该比例文档中出现的词视为"全局噪声"丢弃（文档数 >= 4 时生效）。"""
    n = len(docs)
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    if n >= 4:
        noisy = {t for t, c in df.items() if c / n > max_df}
        docs = [[t for t in d if t not in noisy] for d in docs]
        df = Counter({t: c for t, c in df.items() if t not in noisy})
    idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {t: (c / max(len(d), 1)) * idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    return vecs, idf


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


@dataclass
class Cluster:
    id: int
    members: list[int] = field(default_factory=list)
    centroid: dict[str, float] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)


def _centroid(vecs: list[dict[str, float]]) -> dict[str, float]:
    acc: Counter[str] = Counter()
    for v in vecs:
        acc.update(v)
    norm = math.sqrt(sum(x * x for x in acc.values())) or 1.0
    return {t: x / norm for t, x in acc.items()}


def cluster_conversations(convs: list[Conversation], threshold: float = 0.12, top_k: int = 8) -> list[Cluster]:
    """单遍 leader 聚类 + 一轮重分配，结果对输入顺序稳定（确定性）。"""
    docs = [tokenize((c.title if c.source == "chatgpt" else "") + "\n" + c.user_text()) for c in convs]
    vecs, _ = tfidf(docs)
    clusters: list[Cluster] = []
    for i, v in enumerate(vecs):
        best, best_sim = None, threshold
        for cl in clusters:
            s = cosine(v, cl.centroid)
            if s >= best_sim:
                best, best_sim = cl, s
        if best is None:
            best = Cluster(len(clusters))
            clusters.append(best)
        best.members.append(i)
        best.centroid = _centroid([vecs[j] for j in best.members])
    # 一轮重分配
    for cl in clusters:
        cl.members = []
    for i, v in enumerate(vecs):
        best = max(clusters, key=lambda cl: (cosine(v, cl.centroid), -cl.id))
        best.members.append(i)
    clusters = [c for c in clusters if c.members]
    for new_id, cl in enumerate(clusters):
        cl.id = new_id
        cl.centroid = _centroid([vecs[j] for j in cl.members])
        cl.keywords = [t for t, _ in sorted(cl.centroid.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]
    return clusters
