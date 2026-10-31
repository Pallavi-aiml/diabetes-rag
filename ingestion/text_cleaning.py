"""PDF text normalisation for guideline documents.

Fixes extraction artefacts that hurt BM25, embeddings and NLI:
  * ligatures (ﬂ, ﬁ, ﬀ …)          -> fl, fi, ff …   (Unicode NFKC)
  * hyphenation across line breaks   -> "glyce- mic" / "erythro-\\ncyte" -> "glycemic"
  * control characters from symbol fonts (bullets, <, ≈, ± …) -> space
  * runs of whitespace

LIMITATION: ADA's symbol font has no Unicode mapping, so a sign such as "<" in
"A1C <7%" is lost at extraction time and cannot be recovered reliably. We replace it
with a space (the number survives) instead of guessing the symbol.
"""
import re
import unicodedata

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SOFT_HYPHEN = re.compile(r"(?<=[a-z])-\s*\n\s*(?=[a-z])")
_SPACE_HYPHEN = re.compile(r"(?<=[a-z]{3})- (?=[a-z]{3})")   # "glyce- mic"


def normalize_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _SPACE_HYPHEN.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
