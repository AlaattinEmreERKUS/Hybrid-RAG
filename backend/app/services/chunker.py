import hashlib
import re
import uuid

import tiktoken

from app.models.schemas import Chunk

_enc = tiktoken.get_encoding("cl100k_base")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_MD_HEADER = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (breadcrumb, body) pairs. Plain text -> one section."""
    matches = list(_MD_HEADER.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))
    stack: list[tuple[int, str]] = []
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        stack = [s for s in stack if s[0] < level] + [(level, title)]
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
    return sections


def chunk_document(text: str, doc_id: str | None = None,
                   max_tokens: int = 512, overlap: int = 64) -> tuple[str, list[Chunk]]:
    doc_id = doc_id or str(uuid.uuid4())
    chunks: list[Chunk] = []
    idx = 0
    for breadcrumb, body in _sections(text):
        sents = _SENT_SPLIT.split(body)
        window: list[str] = []
        window_toks = 0
        for sent in sents:
            if not sent.strip():
                continue
            stoks = len(_enc.encode(sent))
            if window and window_toks + stoks > max_tokens:
                chunks.append(_emit(doc_id, idx, breadcrumb, window))
                idx += 1
                # keep tail sentences as overlap
                tail: list[str] = []
                t = 0
                for s in reversed(window):
                    t += len(_enc.encode(s))
                    if t > overlap:
                        break
                    tail.insert(0, s)
                window, window_toks = tail, t
            window.append(sent)
            window_toks += stoks
        if window:
            chunks.append(_emit(doc_id, idx, breadcrumb, window))
            idx += 1
    return doc_id, chunks


def _emit(doc_id: str, idx: int, section: str, sents: list[str]) -> Chunk:
    text = " ".join(sents).strip()
    prefixed = f"[{section}] {text}" if section else text
    return Chunk(
        chunk_id=f"{doc_id}:{idx}",
        doc_id=doc_id,
        index=idx,
        text=prefixed,
        section=section,
        token_count=len(_enc.encode(prefixed)),
    )
