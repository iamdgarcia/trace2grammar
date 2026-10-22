# Generate 500 production-shaped sample conversations (fixed seed -> deterministic,
# regenerable) in the transcript dialect, English, two verticals: customer support
# (Acme Bank) and appointment booking (Northside Clinic).
#
# HONEST PROVENANCE: these are TEMPLATE+VOCAB compositions, not human transcripts and
# not LLM generations. Variety comes from combinatorics (intents x phrasings x tool
# fates x human noise); the grammar walker decides every outcome; the summary table
# printed at the end is the check. This file IS the provenance: rerun it, same 580.
import json, os, random, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
os.makedirs(SAMPLES, exist_ok=True)

rng = random.Random(20261022)

# ---------------- vocab ----------------
GREET_SUPPORT = [
    "Welcome to Acme Bank support. I can help with accounts, cards and payments. What do you need?",
    "Hi, this is the Acme Bank assistant. How can I help you today?",
    "Acme Bank support, how can I help?",
    "Hello and welcome to Acme Bank. What can I do for you today?",
]
GREET_CLINIC = [
    "Good morning, this is Northside Clinic booking. How can I help?",
    "Northside Clinic, how may I help you?",
    "Hello, Northside Clinic support. What do you need?",
    "Hi, this is the Northside Clinic assistant. How can I help you today?",
]
CLOSERS = [
    "Thanks for calling Acme Bank, have a great day!",
    "Thanks for calling, have a good day!",
    "Is there anything else I can help you with? ... Have a great day!",
    "Thanks for your call, have a nice day!",
]
CLOSERS_CLINIC = [
    "Thank you for choosing Northside Clinic, have a great day!",
    "Thanks for calling, have a nice day!",
    "Anything else we can do for you? ... Have a good day!",
]

# tool fates for generic tools
def fate():
    r = rng.random()
    if r < 0.12: return "error"
    return "ok"

INTENTS_SUPPORT = ["balance", "card_replacement", "dispute", "cancel_subscription",
                   "password_reset", "transfer_status", "add_payee"]
INTENTS_CLINIC = ["book", "reschedule", "cancel_appointment", "refill", "ask_hours"]

USER_PHRASES = {
    "balance": ["I want to check my account balance", "What's my balance?",
                "Can you tell me how much money I have?", "Check my balance please"],
    "card_replacement": ["My card is damaged, I need a replacement", "I lost my card",
                         "My card doesn't work, I think it's broken",
                         "The ATM ate my card, I need a new one"],
    "dispute": ["I want to dispute a charge", "There is a charge I don't recognize",
                "You charged me twice, I want my money back",
                "I was overcharged last week"],
    "cancel_subscription": ["I want to cancel my subscription",
                            "Please cancel my account", "I want out of my plan"],
    "password_reset": ["I can't log in, I need to reset my password",
                       "I forgot my password", "My password doesn't work anymore"],
    "transfer_status": ["Did my transfer arrive?", "I sent money on Friday, where is it?",
                        "I want to know the status of a transfer"],
    "add_payee": ["I want to send money to a new payee", "Add a new payee to my account"],
    "book": ["I'd like to book an appointment", "Can I get a dental cleaning?",
             "I need to see the doctor this week", "Do you have anything available tomorrow?"],
    "reschedule": ["I need to move my appointment", "Can I change my Monday appointment?",
                   "I can't make it to my visit, I need another time"],
    "cancel_appointment": ["I need to cancel my appointment", "Cancel my visit on Thursday please"],
    "refill": ["I need a refill of my prescription", "My medication ran out, I need a refill",
               "Can you renew my prescription?"],
    "ask_hours": ["What are your opening hours?", "Are you open on Saturdays?",
                  "What time does the clinic close?"],
}
CLARIFY_USER = ["Sorry, what did you say?", "Come again?", "I don't understand",
                "Can you repeat that?", "What do you mean?"]
CONFIRM_USER = ["Yes", "Yes, that's correct", "Correct", "Sure", "Yes please"]
TOOLNAME = {
    "balance": "check_balance", "card_replacement": "order_card",
    "dispute": "file_dispute", "cancel_subscription": "cancel_subscription",
    "password_reset": "reset_password", "transfer_status": "lookup_payment",
    "add_payee": "add_payee", "book": "book_appointment", "reschedule": "reschedule_appointment",
    "cancel_appointment": "cancel_appointment", "refill": "request_refill",
    "ask_hours": None,
}

# ---------------- story builders ----------------
def story_contained(intent, clinic):
    L = []
    greet = rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT)
    L.append("assistant: " + greet)
    L.append("user: " + rng.choice(USER_PHRASES[intent]))
    tool = TOOLNAME[intent]
    if tool: L.append("tool(verify_identity): ok" if intent != "ask_hours" else "tool(lookup_hours): ok")
    # occasional clarify loop
    if rng.random() < 0.30:
        L.append("assistant: " + rng.choice([
            "Just to confirm, could you repeat the details?", "Which account or date would that be?"]))
        L.append("user: " + rng.choice(CLARIFY_USER))
        L.append("assistant: Of course: " + rng.choice([
            "which day would you prefer?", "could you confirm the last four digits?"]))
        L.append("user: " + rng.choice(CONFIRM_USER))
    # occasional tool error with recovery
    if tool and rng.random() < 0.25:
        L.append(f"tool({tool}): error timeout")
        L.append(f"tool({tool}): ok")
    elif tool:
        L.append(f"tool({tool}): ok")
    L.append("assistant: " + rng.choice([
        "Done. Anything else I can do for you?",
        "All set. Is there anything else?",
        "That's completed. Anything else today?"]))
    if rng.random() < 0.5:
        L.append("user: No, thank you.")
    else:
        L.append("user: No, that's everything.")
    L.append("assistant: " + rng.choice(CLOSERS_CLINIC if clinic else CLOSERS))
    return L

def story_abandoned(intent, clinic):
    L = []
    L.append("assistant: " + rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT))
    L.append("user: " + rng.choice(USER_PHRASES[intent]))
    tool = TOOLNAME[intent]
    kind = rng.random()
    if kind < 0.3:
        # dies during auth
        L += ["tool(verify_identity): error no_answer", "tool(verify_identity): error no_answer",
              "assistant: I'm having trouble verifying your identity. Could you confirm your date of birth?",
              "user:"]
    elif kind < 0.6 and tool:
        L.append(f"tool(verify_identity): ok")
        L.append(f"tool({tool}): ok")
        L.append("assistant: " + rng.choice([
            "We have two options: morning or afternoon. Which do you prefer?",
            "I can process that now. Shall I go ahead?"]))
        L.append("user:")  # silence
    elif kind < 0.85:
        # gives up after clarify
        L.append("tool(verify_identity): ok")
        L.append("assistant: Could you give me a few more details?")
        L.append("user: " + rng.choice(CLARIFY_USER))
        L.append("assistant: Sure: could you confirm the details on your screen?")
        L.append("user: I can't find it now. Forget it.")
    else:
        # dies right after a tool error
        L.append("tool(verify_identity): ok")
        L.append(f"tool({tool}): error rejected")
        L.append("assistant: I'm sorry, that didn't go through. Do you want me to try again?")
        L.append("user:")
    return L

def story_handoff(intent, clinic):
    L = []
    L.append("assistant: " + rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT))
    L.append("user: " + rng.choice(USER_PHRASES[intent]))
    kind = rng.random()
    if kind < 0.25:
        # three auth failures force escalation
        L += ["tool(verify_identity): error mismatch", "tool(verify_identity): error mismatch",
              "tool(verify_identity): error mismatch",
              "assistant: I can't verify your identity automatically, so for security I need to pass you to a human agent.",
              "tool(escalate_to_human): ok",
              "assistant: Connecting you with our verification team, please hold."]
    elif kind < 0.5:
        # out of scope vertical
        L.append("tool(verify_identity): ok")
        L.append("assistant: " + rng.choice([
            "That request is handled by our specialist team, I can't do it myself. Shall I transfer you?",
            "That's above what I can arrange here. Let me pass you to a human agent."]))
        L.append("user: " + rng.choice(["Yes, please transfer me.", "Yes, put me through."]))
        L.append("tool(transfer_to_agent): ok")
        L.append("assistant: Transferring you now, one moment please.")
    elif kind < 0.75:
        # angry caller demands a human
        L.append("tool(verify_identity): ok")
        L.append("assistant: I'm really sorry about that. Let me look into it right away.")
        L.append("user: I don't want a robot, I want a person. Now.")
        L.append("tool(escalate_to_human): ok")
        L.append("assistant: Understood, connecting you with a human agent. Please hold the line.")
    else:
        # authority limit
        L.append("tool(verify_identity): ok")
        L.append("assistant: The amount is above the limit I can approve. I'll connect you with a human agent who can decide it.")
        L.append("tool(escalate_to_human): ok")
        L.append("assistant: Connecting you now, please stay on the line.")
    return L

def story_sim(intent, clinic):
    """Over-cooperative simulated set: never errors, never leaves, never escalates."""
    L = ["assistant: Hello! How can I help you today?",
         "user: " + rng.choice(USER_PHRASES[intent])]
    tool = TOOLNAME[intent]
    if tool:
        L.append("tool(verify_identity): ok" if intent != "ask_hours" else "tool(lookup_hours): ok")
        L.append(f"tool({tool}): ok")
    L += ["assistant: All done. Anything else?",
          "user: No.",
          "assistant: Thanks for calling, have a great day!"]
    return L

# ---------------- compose ----------------
N_REAL, N_SIM = 500, 80
written = []
counts = {"cont": 0, "aban": 0, "hand": 0}

# clean old samples except taubench-era? (samples dir only has generated ones)
for f in os.listdir(SAMPLES):
    os.remove(os.path.join(SAMPLES, f))

i = 0
while len(written) < N_REAL:
    i += 1
    r = rng.random()
    clinic = rng.random() < 0.45
    intents = INTENTS_CLINIC if clinic else INTENTS_SUPPORT
    intent = rng.choice(intents)
    if r < 0.42:
        L, fam = story_contained(intent, clinic), "cont"
    elif r < 0.72:
        L, fam = story_abandoned(intent, clinic), "aban"
    else:
        L, fam = story_handoff(intent, clinic), "hand"
    name = f"{fam}{len(written)+1:03d}_{i:04d}"
    with open(os.path.join(SAMPLES, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    written.append(name)
    counts[fam] += 1

for j in range(N_SIM):
    clinic = rng.random() < 0.45
    intents = INTENTS_CLINIC if clinic else INTENTS_SUPPORT
    intent = rng.choice([x for x in intents if TOOLNAME[x]])
    L = story_sim(intent, clinic)
    name = f"sim{j+1:03d}"
    with open(os.path.join(SAMPLES, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    written.append(name)

print("wrote", len(written), "transcripts:", counts, "+", N_SIM, "sim")

# ---------------- walker decides; summary is the check ----------------
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract

from collections import Counter
oc = Counter()
for name in written:
    with open(os.path.join(SAMPLES, name + ".txt"), encoding="utf-8") as fh:
        r = extract(fh.read(), "transcript")
    fam = name[:4] if name.startswith("sim") else name[:4]
    oc[(name[:4], r["outcome"])] += 1
for k in sorted(oc):
    print(k, oc[k])
json.dump({k[0] + ":" + k[1]: v for k, v in oc.items()},
          open(os.path.join(SAMPLES, "_summary.json"), "w", encoding="utf-8"), indent=1)
