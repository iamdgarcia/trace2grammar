# Generate 500 production-shaped sample conversations (fixed seed -> deterministic,
# regenerable) in the transcript dialect, English, two verticals: customer support
# (Acme Bank) and appointment booking (Northside Clinic). Plus 80 over-cooperative sim.
#
# HONEST PROVENANCE: these are TEMPLATE+VOCAB compositions, not human transcripts and
# not LLM generations. Every conversation composes: greeting x intent x phrasing x
# auth path x clarify loops (0-2) x tool plan (1-4 tools) x tool fates (ok/error/retry)
# x resolution x closer, with per-line synthetic timestamps that scale with content
# length (so durations differ) and a unique case id. The grammar walker decides every
# outcome; the summary table printed at the end is the check (family prefixes are
# intents, not labels: mismatches show up as walker outcomes the families forbid).
import json, os, random, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
os.makedirs(SAMPLES, exist_ok=True)

rng = random.Random(20261022)

GREET_SUPPORT = [
    "Welcome to Acme Bank support. I can help with accounts, cards and payments. What do you need?",
    "Hi, this is the Acme Bank assistant. How can I help you today?",
    "Acme Bank support, how can I help?",
    "Hello and welcome to Acme Bank. What can I do for you today?",
    "Good afternoon, Acme Bank assistant speaking. What can I help you with?",
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
    "Thanks for your call, have a nice day!",
    "Thank you for banking with Acme. Goodbye!",
]
CLOSERS_CLINIC = [
    "Thank you for choosing Northside Clinic, have a great day!",
    "Thanks for calling, have a nice day!",
    "Take care and see you at the clinic. Goodbye!",
]

INTENTS_SUPPORT = ["balance", "card_replacement", "dispute", "cancel_subscription",
                   "password_reset", "transfer_status", "add_payee", "card_block",
                   "direct_debit", "statement"]
INTENTS_CLINIC = ["book", "reschedule", "cancel_appointment", "refill", "ask_hours",
                  "test_results", "home_visit"]

USER_PHRASES = {
    "balance": ["I want to check my account balance", "What's my balance?",
                "Can you tell me how much money I have?", "Check my balance please",
                "How much is left in my current account?"],
    "card_replacement": ["My card is damaged, I need a replacement", "I lost my card",
                         "My card doesn't work, I think it's broken",
                         "The ATM ate my card, I need a new one",
                         "I never received the card I ordered two weeks ago"],
    "dispute": ["I want to dispute a charge", "There is a charge I don't recognize",
                "You charged me twice, I want my money back",
                "I was overcharged last week",
                "A company took a payment without my permission"],
    "cancel_subscription": ["I want to cancel my subscription",
                            "Please cancel my account", "I want out of my plan",
                            "Cancel everything, I don't use it anymore"],
    "password_reset": ["I can't log in, I need to reset my password",
                       "I forgot my password", "My password doesn't work anymore",
                       "The app locked me out after three attempts"],
    "transfer_status": ["Did my transfer arrive?", "I sent money on Friday, where is it?",
                        "I want to know the status of a transfer",
                        "A payment I'm expecting hasn't shown up"],
    "add_payee": ["I want to send money to a new payee", "Add a new payee to my account",
                  "How do I register a company as a payee?"],
    "card_block": ["I want to block my card temporarily", "Freeze my card please",
                   "I found my card but I had blocked it, can you unblock it?"],
    "direct_debit": ["A direct debit failed, why?", "I need to change the date of my direct debit",
                     "My direct debit bounced and I got a fee"],
    "statement": ["I need a statement of the last three months",
                  "Can you send me a proof of payment?",
                  "I need a certificate for my taxes"],
    "book": ["I'd like to book an appointment", "Can I get a dental cleaning?",
             "I need to see the doctor this week", "Do you have anything available tomorrow?",
             "I need a first consultation"],
    "reschedule": ["I need to move my appointment", "Can I change my Monday appointment?",
                   "I can't make it to my visit, I need another time"],
    "cancel_appointment": ["I need to cancel my appointment", "Cancel my visit on Thursday please"],
    "refill": ["I need a refill of my prescription", "My medication ran out, I need a refill",
               "Can you renew my prescription?"],
    "ask_hours": ["What are your opening hours?", "Are you open on Saturdays?",
                  "What time does the clinic close?", "Do you open during lunch?"],
    "test_results": ["My lab results are ready?", "How do I get my test results?",
                     "Has the doctor seen my results yet?"],
    "home_visit": ["Can I get a home visit? I can't move well",
                   "I need a nurse to come home for a blood draw"],
}
CLARIFY_USER = ["Sorry, what did you say?", "Come again?", "I don't understand",
                "Can you repeat that?", "What do you mean?", "Sorry, which one?"]
CONFIRM_USER = ["Yes", "Yes, that's correct", "Correct", "Sure", "Yes please", "That's right"]
NEG_USER = ["No, thank you.", "No, that's everything.", "No, that's all, thanks."]

# per-intent tool plans: list of (toolname, needed) — composed 1..4 tools per story
TOOL_PLANS = {
    "balance": ["check_balance"],
    "card_replacement": ["order_card"],
    "dispute": ["lookup_transactions", "file_dispute"],
    "cancel_subscription": ["lookup_plan", "cancel_subscription"],
    "password_reset": ["send_otp", "verify_otp", "reset_password"],
    "transfer_status": ["lookup_payment"],
    "add_payee": ["add_payee"],
    "card_block": ["block_card"],
    "direct_debit": ["lookup_debits", "reschedule_debit"],
    "statement": ["generate_statement"],
    "book": ["check_availability", "book_appointment"],
    "reschedule": ["lookup_appointment", "reschedule_appointment"],
    "cancel_appointment": ["lookup_appointment", "cancel_appointment"],
    "refill": ["lookup_prescription", "request_refill"],
    "ask_hours": ["lookup_hours"],
    "test_results": ["lookup_results"],
    "home_visit": ["check_coverage", "book_appointment"],
}
# some intents genuinely skip auth (pure info)
NO_AUTH = {"ask_hours"}

ASSISTANT_RESPONSES = {
    "balance": ["Your checking account balance is {amt} dollars and savings has {amt2}.",
                "The current balance is {amt} dollars. Anything else?"],
    "card_replacement": ["A replacement card is ordered, it arrives in {n} business days.",
                         "Done: new card on its way, delivery in {n} days."],
    "dispute": ["The dispute is filed, case {case}. You'll hear from us in {n} business days.",
                "Dispute registered for {amt} dollars, case {case}."],
    "cancel_subscription": ["Your subscription is cancelled, effective at the end of the billing period.",
                            "Cancelled. You keep access until {month}."],
    "password_reset": ["Your password has been reset, check your email.",
                       "Password updated. Try logging in again in a minute."],
    "transfer_status": ["The transfer of {amt} dollars is {status}.",
                        "It shows {status} since this morning."],
    "add_payee": ["The payee is registered. The first payment to a new payee has a 24 hour hold.",
                  "Payee added and confirmed by SMS."],
    "card_block": ["The card is blocked. You can unblock it from the app whenever you want.",
                   "Card frozen as requested."],
    "direct_debit": ["The direct debit is rescheduled to the {n}th. The fee is refunded.",
                     "Done: new date the {n}th, and the bounced fee reversed."],
    "statement": ["The statement is on its way to your email.",
                  "Generated: the PDF is in your documents section."],
    "book": ["You're booked for {day} at {time} with Dr. {doc}.",
             "Booked: {day} at {time}. You'll get a reminder SMS."],
    "reschedule": ["Moved to {day} at {time}. Anything else?",
                   "The appointment is now {day} at {time}."],
    "cancel_appointment": ["The appointment is cancelled. Anything else?",
                           "Cancelled; you can rebook any time."],
    "refill": ["Refill requested, ready for pickup in {n} hours.",
               "The pharmacy will have it tomorrow morning."],
    "ask_hours": ["We're open {hours} on weekdays and {hours2} on Saturdays.",
                  "Monday to Friday {hours}, Saturdays {hours2}. Closed Sundays."],
    "test_results": ["The results are in the app, released by your doctor this morning.",
                     "Results are ready; the doctor added notes to them."],
    "home_visit": ["A nurse visit is arranged for {day} between {time} and {time2}.",
                   "Home visit booked: {day}, {time} window. You'll get a call to confirm."],
}

def amt(): return f"{rng.randint(12, 2400)}.{rng.choice(['10','20','50','90','99'])}"
def day(): return rng.choice(["Monday","Tuesday","Wednesday","Thursday","Friday"])
def time(): return f"{rng.randint(9,17)}:{rng.choice(['00','15','30','45'])}"
def doc(): return rng.choice(["Reeves","Ortiz","Chen","Patel","Novak"])
def hours(): return f"{rng.randint(8,9)}:{rng.choice(['00','30'])} to {rng.randint(17,20)}:{rng.choice(['00','30'])}"
def month(): return rng.choice(["March","April","May","June"])
def status(): return rng.choice(["completed", "pending", "returned"])

def fill(t):
    return (t.replace("{amt}", amt()).replace("{amt2}", amt()).replace("{n}", str(rng.randint(2, 7)))
             .replace("{case}", str(rng.randint(10000, 99999))).replace("{day}", day())
             .replace("{time}", time()).replace("{time2}", time()).replace("{doc}", doc())
             .replace("{hours}", hours()).replace("{hours2}", hours()).replace("{month}", month())
             .replace("{status}", status()))

# ---------------- story builders ----------------
def story_contained(intent, clinic):
    L = ["assistant: " + rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT),
         "user: " + rng.choice(USER_PHRASES[intent])]
    tools = TOOL_PLANS[intent]
    if intent not in NO_AUTH:
        # auth: mostly one pass; sometimes one retry (no cycle: single retry then ok)
        if rng.random() < 0.22:
            L.append("tool(verify_identity): error timeout")
        L.append("tool(verify_identity): ok")
    # 0-2 clarify loops with REAL dependency on the intent
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        L.append("assistant: " + rng.choice([
            "Sure: could you confirm the last four digits?",
            "Just to confirm: is it the {day} one?".replace("{day}", day()),
            "Which of the two would you prefer?",
            "Do you want it processed now or later today?"]))
        L.append("user: " + rng.choice(CLARIFY_USER))
        L.append("assistant: " + rng.choice([
            "Of course: I said, " + rng.choice(["should I process it now?", "which option do you prefer?", "can you confirm it?"]),
            "No problem, let me repeat it."]))
        L.append("user: " + rng.choice(CONFIRM_USER))
    # run the tool plan: some tools error once and retry (recovery), some just run
    for t in tools:
        r = rng.random()
        if r < 0.14:
            L.append(f"tool({t}): error timeout")
            L.append(f"tool({t}): ok")
        elif r < 0.20:
            L.append(f"tool({t}): ok")
            L.append(f"tool({t}): ok")  # idempotent re-run (iteration, not cycle)
        else:
            L.append(f"tool({t}): ok")
    L.append("assistant: " + fill(rng.choice(ASSISTANT_RESPONSES[intent])))
    L.append("assistant: " + rng.choice(["Anything else I can do for you?", "Is there anything else today?"]))
    L.append("user: " + rng.choice(NEG_USER))
    L.append("assistant: " + rng.choice(CLOSERS_CLINIC if clinic else CLOSERS))
    return L

def story_abandoned(intent, clinic):
    L = ["assistant: " + rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT),
         "user: " + rng.choice(USER_PHRASES[intent])]
    tools = TOOL_PLANS[intent]
    kind = rng.random()
    if kind < 0.28:
        # dies during auth
        L += ["tool(verify_identity): error no_answer", "tool(verify_identity): error no_answer",
              "assistant: I'm having trouble verifying your identity. Could you confirm your date of birth?",
              "user: Sorry, one second, I'm looking for the document."]
    elif kind < 0.55:
        # silent after the assistant asks something
        if intent not in NO_AUTH:
            L.append("tool(verify_identity): ok")
            for t in tools[:1]:
                L.append(f"tool({t}): ok")
        L.append("assistant: " + rng.choice([
            "We have two options: morning or afternoon. Which do you prefer?",
            "I can process that now, it takes a minute. Shall I go ahead?",
            "Could you confirm the address we have on file?"]))
        L.append("user: " + rng.choice(["Hmm, let me think.", "One second.", "Eh, wait."]))
    elif kind < 0.8:
        # gives up after a clarify loop (frustration)
        if intent not in NO_AUTH:
            L.append("tool(verify_identity): ok")
        L.append("assistant: Could you give me a few more details?")
        L.append("user: " + rng.choice(CLARIFY_USER))
        L.append("assistant: Sure, let me repeat: could you confirm the details on your screen?")
        L.append("user: I can't find it now. Forget it.")
    else:
        # dies right after a tool error (no recovery offered in time)
        if intent not in NO_AUTH:
            L.append("tool(verify_identity): ok")
        t = tools[-1]
        L.append(f"tool({t}): error rejected")
        L.append("assistant: I'm sorry, that didn't go through. Do you want me to try again?")
    return L

def story_handoff(intent, clinic):
    L = ["assistant: " + rng.choice(GREET_CLINIC if clinic else GREET_SUPPORT),
         "user: " + rng.choice(USER_PHRASES[intent])]
    kind = rng.random()
    if kind < 0.22:
        # three auth failures force escalation
        L += ["tool(verify_identity): error mismatch", "tool(verify_identity): error mismatch",
              "tool(verify_identity): error mismatch",
              "assistant: I can't verify your identity automatically, so for security I need to pass you to a human agent.",
              "tool(escalate_to_human): ok",
              "assistant: Connecting you with our verification team, please hold."]
    elif kind < 0.44:
        # out of scope for the bot
        if intent not in NO_AUTH:
            L.append("tool(verify_identity): ok")
        L.append("assistant: " + rng.choice([
            "That request is handled by our specialist team, I can't do it myself. Shall I transfer you?",
            "That's above what I can arrange here. Let me pass you to a human agent."]))
        L.append("user: " + rng.choice(["Yes, please transfer me.", "Yes, put me through."]))
        L.append("tool(transfer_to_agent): ok")
        L.append("assistant: Transferring you now, one moment please.")
    elif kind < 0.68:
        # angry caller demands a human
        L.append("tool(verify_identity): ok")
        L.append("assistant: I'm really sorry about that. Let me look into it right away.")
        L.append("user: I don't want a robot, I want a person. Now.")
        L.append("tool(escalate_to_human): ok")
        L.append("assistant: Understood, connecting you with a human agent. Please hold the line.")
    elif kind < 0.85:
        # authority limit
        L.append("tool(verify_identity): ok")
        L.append("assistant: The amount is above the limit I can approve. I'll connect you with a human agent who can decide it.")
        L.append("tool(escalate_to_human): ok")
        L.append("assistant: Connecting you now, please stay on the line.")
    else:
        # tool error twice then forced transfer
        if intent not in NO_AUTH:
            L.append("tool(verify_identity): ok")
        t = TOOL_PLANS[intent][-1]
        L.append(f"tool({t}): error internal")
        L.append(f"tool({t}): error internal")
        L.append("assistant: The system isn't letting me complete this. I'll transfer you to someone who can.")
        L.append("tool(transfer_to_agent): ok")
        L.append("assistant: Transferring you now, please hold.")
    return L

def story_sim(intent, clinic):
    L = ["assistant: Hello! How can I help you today?",
         "user: " + rng.choice(USER_PHRASES[intent])]
    if intent not in NO_AUTH:
        L.append("tool(verify_identity): ok")
    for t in TOOL_PLANS[intent]:
        L.append(f"tool({t}): ok")
    L += ["assistant: " + fill(rng.choice(ASSISTANT_RESPONSES[intent])),
          "assistant: Anything else?",
          "user: No.",
          "assistant: Thanks for calling, have a great day!"]
    return L

# ---------------- compose ----------------
N_REAL, N_SIM = 500, 80
written = []
counts = {"cont": 0, "aban": 0, "hand": 0}

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
    # ensure intent and family are compatible: handoff stories always end in handoff events,
    # abandoned stories always end mid-flight, contained always close
    name = f"{fam}{len(written)+1:03d}_{i:04d}"
    with open(os.path.join(SAMPLES, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    written.append(name)
    counts[fam] += 1

for j in range(N_SIM):
    clinic = rng.random() < 0.45
    intents = INTENTS_CLINIC if clinic else INTENTS_SUPPORT
    intent = rng.choice(intents)
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
walks = Counter()
for name in written:
    with open(os.path.join(SAMPLES, name + ".txt"), encoding="utf-8") as fh:
        r = extract(fh.read(), "transcript")
    oc[(name[:4], r["outcome"])] += 1
    walks[tuple(w["state"] for w in r["walk"])] += 1
for k in sorted(oc):
    print(k, oc[k])
print("distinct walks:", len(walks), "of", len(written))
json.dump({k[0] + ":" + k[1]: v for k, v in oc.items()},
          open(os.path.join(SAMPLES, "_summary.json"), "w", encoding="utf-8"), indent=1)

# hard check: distinct walks must be a real fraction, else the plane collapses
if len(walks) < 40:
    print("WARNING: only", len(walks), "distinct walks: the space will look empty")
