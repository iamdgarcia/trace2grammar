# Generate the sample-conversation set for trace2grammar (22-oct).
# Hand-written synthetic transcripts, English, two verticals (customer support,
# appointment booking). Each becomes samples/<name>.txt in the transcript dialect:
#   assistant: / user: lines, tool(name): ok|error <msg>, blank lines ignored.
# Outcomes are DERIVED by the walker, not labeled by filename: the summary table
# printed at the end is the check (mismatch = bug in the transcript or the walker).
import json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
os.makedirs(SAMPLES, exist_ok=True)

# (filename, transcript) — variety: cycle depths 0-2, tool errors with and without
# recovery, mid-auth hangs, escalations, one over-cooperative sim batch at the end.
CONVOS = [
    # ---------------- contained ----------------
    ("cont01_balance_straight", """assistant: Welcome to Acme Bank support. I can help with accounts, cards and payments. What do you need?
user: Hi, I just want to check my account balance.
tool(verify_identity): ok
tool(check_balance): ok
assistant: Your checking account balance is 1,240.50 dollars. Anything else I can do for you?
user: No, that's everything. Thanks!
assistant: Thanks for calling Acme Bank, have a great day!"""),
    ("cont02_card_replace_clarify", """assistant: Hi, this is the Acme Bank assistant. How can I help you today?
user: My card is damaged, I need a replacement.
tool(verify_identity): ok
assistant: Sure, I can order a replacement. Should we send it to your address on Main Street 42?
user: Sorry, what address did you say?
assistant: The address we have on file: Main Street 42, apartment 3.
user: Yes, that's correct.
tool(order_card): ok
assistant: Your new card will arrive in 5 business days. Anything else?
user: No, thanks a lot.
assistant: Thanks for your call, have a good day!"""),
    ("cont03_booking_simple", """assistant: Good morning, this is Northside Clinic booking. How can I help?
user: I'd like to book a dental cleaning.
tool(check_availability): ok
assistant: We have Thursday at 10:00 or Friday at 15:30. Which works better?
user: Thursday at 10:00, please.
tool(book_appointment): ok
assistant: You're booked for Thursday at 10:00 with Dr. Reeves. Anything else?
user: No, that's all.
assistant: Thank you for choosing Northside Clinic, have a great day!"""),
    ("cont04_reschedule_clarify", """assistant: Northside Clinic, how may I help you?
user: I need to move my appointment on Monday.
tool(verify_identity): ok
tool(lookup_appointment): ok
assistant: You have a check-up on Monday at 9:00. What day would you like instead?
user: What was the appointment again?
assistant: Your check-up with Dr. Reeves, Monday at 9:00.
user: Ah yes, move it to Wednesday afternoon.
tool(reschedule_appointment): ok
assistant: Done, moved to Wednesday at 16:00. Anything else?
user: No, thanks.
assistant: Thanks for calling, have a nice day!"""),
    ("cont05_otp_retry", """assistant: Acme Bank support, how can I help?
user: I can't log in, I need to reset my password.
tool(send_otp): error timeout
tool(send_otp): ok
assistant: I've sent a code to your phone. Can you read it to me?
user: Yes, it's 448210.
tool(verify_otp): ok
tool(reset_password): ok
assistant: Your password has been reset, check your email. Anything else?
user: No, perfect.
assistant: Thanks for calling Acme Bank, have a great day!"""),
    ("cont06_dispute_two_clarifies", """assistant: Acme Bank support, how can I help you today?
user: I want to dispute a charge.
tool(verify_identity): ok
assistant: Of course. Which charge: the one from the 3rd or the one from the 12th?
user: The one from the 3rd, sorry I mixed up the dates.
assistant: Got it, the 3rd, 89.90 at TechMart. What is the problem with it?
user: The goods never arrived, I already contacted the shop.
tool(file_dispute): ok
assistant: The dispute is filed, case number 58291. You'll hear from us in 3 business days. Anything else?
user: No, thank you.
assistant: Thanks for calling Acme Bank, have a great day!"""),
    ("cont07_refill_declines_human", """assistant: Northside Clinic support, how can I help?
user: I need a refill of my prescription.
tool(verify_identity): ok
tool(lookup_prescription): ok
assistant: I can request the refill for the blood pressure medication. It takes 24 hours. Should I go ahead?
user: Yes please.
tool(request_refill): ok
assistant: Refill requested. Would you like me to transfer you to a nurse for any questions?
user: No, no need, I'm fine.
assistant: Alright. Thanks for calling Northside Clinic, have a good day!"""),
    ("cont08_cancel_subscription", """assistant: Acme support, how can I help?
user: I want to cancel my subscription.
tool(verify_identity): ok
assistant: Before I do it: I can offer you a 20 percent discount for 3 months if you stay. Interested?
user: No, thank you, I really want to cancel it.
tool(cancel_subscription): ok
assistant: Your subscription is cancelled, effective at the end of the billing period. Anything else?
user: No, that's all.
assistant: Thanks for calling Acme, have a great day!"""),
    # ---------------- abandoned ----------------
    ("aban01_never_replies", """assistant: Welcome to Acme Bank support. I can help with accounts, cards and payments. What do you need?
user: Hi, I have a question about a transfer.
tool(verify_identity): ok
assistant: Sure, could you tell me the amount of the transfer?
user: It was 300 euros, on Friday.
assistant: I see a transfer of 300 euros from Friday, status pending. Would you like more details?
user: Yes please, when will it arrive?
assistant: It should arrive within 1 business day. Anything else about it?
user:"""),
    ("aban02_hangs_mid_auth", """assistant: Acme Bank support, how can I help?
user: I need to check a payment.
tool(verify_identity): error no_answer
tool(verify_identity): error no_answer
tool(verify_identity): error no_answer
assistant: I'm having trouble verifying your identity. Could you confirm your date of birth?
user:"""),
    ("aban03_silent_after_offer", """assistant: Northside Clinic booking, how can I help?
user: I'd like to see the doctor this week.
tool(check_availability): ok
assistant: We have Wednesday at 12:00 or Thursday at 9:30. Which one do you prefer?
user: Hmm, let me think about the morning one.
assistant: Sure, take your time: Thursday at 9:30 with Dr. Reeves. Shall I book it?
user:"""),
    ("aban04_after_two_clarify", """assistant: Acme Bank support, how can I help?
user: There is a problem with my account.
tool(verify_identity): ok
assistant: I'm sorry to hear that. What kind of problem are you seeing?
user: I don't understand the app, it says something strange.
assistant: Could you tell me the exact message the app shows?
user: Come again?
assistant: Of course: what exact message does the app show on the screen?
user: I can't find it now. Forget it.
user:"""),
    ("aban05_silent_on_date", """assistant: Northside Clinic, how may I help you?
user: I need a follow-up with Dr. Reeves.
tool(verify_identity): ok
tool(lookup_appointment): ok
assistant: Dr. Reeves has openings on Tuesday at 11:00 or Friday at 13:00. Which would you like?
user:"""),
    ("aban06_after_tool_error", """assistant: Acme Bank support, how can I help you today?
user: I want to send money to a new payee.
tool(verify_identity): ok
tool(add_payee): error rejected_by_compliance
assistant: I'm sorry, the system rejected adding that payee. It may need a manual review. Do you want me to try again?
user:"""),
    # ---------------- handoff ----------------
    ("hand01_angry_demands_human", """assistant: Acme Bank support, how can I help?
user: This is ridiculous, I've been charged twice and nobody fixes it!
assistant: I'm really sorry about that. Let me look into the duplicate charge right away.
tool(verify_identity): ok
tool(lookup_charges): ok
user: I don't want a robot, I want a person. Now.
tool(escalate_to_human): ok
assistant: Understood, connecting you with a human agent. Please hold the line."""),
    ("hand02_mortgage_out_of_scope", """assistant: Acme Bank support, how can I help you today?
user: I have questions about switching my mortgage to a fixed rate.
tool(verify_identity): ok
assistant: Mortgage restructuring is handled by our specialist team, I can't quote rates myself. Shall I transfer you?
user: Yes, please transfer me.
tool(transfer_to_agent): ok
assistant: Transferring you to the mortgage team, one moment please."""),
    ("hand03_auth_fail_forced", """assistant: Acme Bank support, how can I help?
user: I can't access my account.
tool(verify_identity): error mismatch
tool(verify_identity): error mismatch
tool(verify_identity): error mismatch
assistant: I can't verify your identity automatically, so for security I need to pass you to a human agent.
tool(escalate_to_human): ok
assistant: Connecting you with our verification team, please hold."""),
    ("hand04_special_request", """assistant: Northside Clinic booking, how can I help?
user: I need an appointment but I need sign language interpretation during the visit.
tool(check_availability): ok
assistant: Booking interpretation is beyond what I can arrange here. Let me pass you to our patient support team.
tool(transfer_to_agent): ok
assistant: Transferring you now, they will arrange the interpreter. One moment."""),
    ("hand05_refund_over_authority", """assistant: Acme support, how can I help?
user: The washing machine I bought broke in a week, I want all my money back.
tool(verify_identity): ok
tool(lookup_order): ok
assistant: The refund for that order is above the limit I can approve. I'll connect you with a human agent who can decide it.
tool(escalate_to_human): ok
assistant: Connecting you now, please stay on the line."""),
    # ---------------- simulated set: over-cooperative v1 (never errors, never leaves,
    # never transfers) — the collapse configuration of §5 ----------------
    ("sim01_balance", """assistant: Hello! Welcome to Acme Bank. How can I help you?
user: Check my balance please.
tool(verify_identity): ok
tool(check_balance): ok
assistant: Your balance is 980 dollars. Anything else?
user: No.
assistant: Thanks for calling, have a great day!"""),
    ("sim02_book", """assistant: Hello! Northside Clinic here. What do you need?
user: A cleaning appointment.
tool(check_availability): ok
tool(book_appointment): ok
assistant: Booked for Tuesday at 9:00. Anything else?
user: No.
assistant: Thanks for calling, have a great day!"""),
    ("sim03_card", """assistant: Hello! How can I help you today?
user: I lost my card.
tool(verify_identity): ok
tool(block_card): ok
tool(order_card): ok
assistant: Card blocked and a replacement ordered. Anything else?
user: No, that's it.
assistant: Thanks for calling, have a great day!"""),
    ("sim04_refill", """assistant: Hello! How can I help?
user: Refill my prescription.
tool(verify_identity): ok
tool(request_refill): ok
assistant: Refill requested, ready in 24 hours. Anything else?
user: No.
assistant: Thanks for calling, have a great day!"""),
    ("sim05_transfer_question", """assistant: Hello! How can I help you today?
user: Did my transfer arrive?
tool(verify_identity): ok
tool(lookup_payment): ok
assistant: Yes, the transfer arrived this morning. Anything else?
user: No, thanks.
assistant: Thanks for calling, have a great day!"""),
    ("sim06_cancel", """assistant: Hello! How can I help?
user: Cancel my subscription.
tool(verify_identity): ok
tool(cancel_subscription): ok
assistant: Subscription cancelled. Anything else?
user: No.
assistant: Thanks for calling, have a great day!"""),
]

written = []
for name, txt in CONVOS:
    p = os.path.join(SAMPLES, name + ".txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(txt.strip() + "\n")
    written.append(name)
print("wrote", len(written), "transcripts to samples/")

# ---- summary table: the walker decides every outcome; mismatches with the
# intended family in the filename are bugs, checked right here.
sys.path.insert(0, ROOT)
from trace2grammar.extract import extract

rows = []
for name in written:
    with open(os.path.join(SAMPLES, name + ".txt"), encoding="utf-8") as fh:
        text = fh.read()
    r = extract(text, "transcript")
    states = " > ".join(w["state"] for w in r["walk"])
    rows.append((name, r["outcome"], r["cycle_depth"], r["error_count"], states))
    print(f"{name:32s} {r['outcome']:11s} D={r['cycle_depth']} err={r['error_count']}  {states[:86]}")

json.dump([{"name": n, "outcome": o, "cycle_depth": d, "error_count": e}
           for n, o, d, e, _ in rows],
          open(os.path.join(SAMPLES, "_summary.json"), "w", encoding="utf-8"),
          indent=1)
print("samples/_summary.json written")
