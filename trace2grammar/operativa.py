"""Operativa (business-operation) labels for conversations.

Same philosophy as the state classifier: deterministic, config-exposed pattern
tables, honest baseline. The operativa identity already travels in the trace
(domain tool names + first user utterance); tau was carrying it as attrs
metadata and dropping it from the label. This module makes it a first-class
label so coverage can be conditioned on it.

Label precedence:
  1. domain tool names (exclude auth/handoff tools — they exist in every call)
  2. keywords in the first user utterance
  3. 'general_info'
The outcome taxonomy (contained/abandoned/handoff) is the SHAPE taxonomy; the
operativa taxonomy is the BUSINESS taxonomy. Both are labels on the same walk.
"""
from __future__ import annotations

import re

# tool-name → operativa (domain tools only; auth + handoff tools excluded upstream)
TOOL_OPERATIVA: dict[str, str] = {
    "reset_password": "account_access",
    "send_otp": "account_access",
    "verify_otp": "account_access",
    "unblock_login": "account_access",
    "block_card": "card_status",
    "unblock_card": "card_status",
    "lookup_debits": "direct_debit",
    "reschedule_debit": "direct_debit",
    "add_payee": "payee_transfer",
    "lookup_payment": "payments",
    "generate_statement": "statements",
    "book_appointment": "appointment",
    "lookup_appointment": "appointment",
    "reschedule_appointment": "appointment",
    "cancel_appointment": "appointment",
    "check_availability": "appointment",
    "check_coverage": "insurance",
    "check_balance": "balance",
}

# auth / handoff tool patterns (mirror extract.py) — never label an operativa
AUTH_TOOL_PAT = re.compile(r"verify|identity|auth|otp|login", re.I)
HANDOFF_TOOL_PAT = re.compile(r"escalate|transfer|handoff|human|agent_queue", re.I)

# first-utterance keyword fallback → operativa (ordered: first hit wins)
UTTERANCE_OPERATIVA: tuple[tuple[str, str], ...] = (
    (r"charge|payment|pay(ment)? .*(without|no )|unrecognized|no reconoc|cargo|domicili", "dispute_charge"),
    (r"appointment|cita|cleaning|limpieza|dental|visit|visita|clinic|cl[ií]nica", "appointment"),
    (r"medication|refill|prescription|\blab\b|test results|nurse|doctor|enfermera|medicaci|receta|an[aá]lisis|aid", "clinical_service"),
    (r"cancel my account|want out|dar de baja|cancelar la cuenta", "account_cancellation"),
    (r"\bcard\b|tarjeta|block|bloque|locked", "card_status"),
    (r"password|log ?in|log ?on|entrar|acceder|access|contrase", "account_access"),
    (r"payee|beneficiario|transfer|sent money", "payee_transfer"),
    (r"certificate|statement|certificado|taxes|impuest", "statements"),
    (r"debit|recibo", "direct_debit"),
    (r"coverage|seguro|p[oó]liza", "insurance"),
    (r"balance|saldo|how much.*(left|have)|money i have", "balance"),
)


def label_operativa(transcript: str, tool_names: list[str] | None = None) -> str:
    """Operativa label for one conversation. Deterministic; see module docstring."""
    # 1) domain tool names, in call order, first hit wins
    for name in tool_names or []:
        if AUTH_TOOL_PAT.search(name) or HANDOFF_TOOL_PAT.search(name):
            continue
        key = name.lower()
        if key in TOOL_OPERATIVA:
            return TOOL_OPERATIVA[key]
    # 2) first user utterance keywords
    for line in transcript.split("\n"):
        if line.lower().startswith("user:"):
            text = line[5:].lower()
            for pat, op in UTTERANCE_OPERATIVA:
                if re.search(pat, text, re.I):
                    return op
            break
    # 3) give up honestly
    return "general_info"
