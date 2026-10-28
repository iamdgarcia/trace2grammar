"""trace2grammar: turn-level trajectory grammars for voice agents.

Public API:
    extract(obj, fmt)               one-call normalize + infer (all dialects)
    to_grammar(raw, fmt, meta)      conversation -> latent representation (embedding)
    fit_basis(results) / project_features(feats, basis)   the 2D behavior plane
    register_adapter(name, fn)      plug in a new transport
    ExtractionConfig                the domain-adaptable keyword patterns
"""

from .adapters import (
    ADAPTERS,
    fit_basis,
    project_features,
    register_adapter,
    to_grammar,
)
from .coverage import detect_fmt
from .extract import (
    KINDS,
    PROGRESS_STATES,
    TERMINALS,
    Event,
    ExtractionConfig,
    extract,
    extract_transcript,
    infer_states,
    normalize_trace,
)

__all__ = [
    "ADAPTERS",
    "Event",
    "ExtractionConfig",
    "KINDS",
    "PROGRESS_STATES",
    "TERMINALS",
    "detect_fmt",
    "extract",
    "extract_transcript",
    "fit_basis",
    "infer_states",
    "normalize_trace",
    "project_features",
    "register_adapter",
    "to_grammar",
]
