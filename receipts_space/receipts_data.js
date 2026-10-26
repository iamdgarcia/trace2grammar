/* The Receipts Audit - data
 * The morning an agent acted for you. 14 receipts, 6 rules.
 * Verdicts are mechanical: every overreach cites exactly one rule from the rulebook.
 */
window.RECEIPTS_DATA = {
  meta: {
    title: "The Receipts Audit",
    subtitle: "14 receipts. 6 rules. One morning.",
    landing_learn: "An agent acting on your behalf does not need to be hacked to do damage. It needs to overstep. In the personal-agent story of this autumn, the reported harms were not break-ins: a store revoked access after unsolicited purchases, private messages got read, a dossier got written one 'might be useful' at a time, a home address leaked in a forwarding. All of it is visible in the receipts. Here: <strong>7 of these 14 receipts break a rule the agent itself was given, and &euro;94.79 was spent beyond your instructions.</strong>",
    landing_do: "Read the morning brief and the 6-rule rulebook below. Then stamp every receipt <strong>OK</strong> or <strong>CONTEST</strong>. At the end you get your audit score, the rule each overreach broke, and the session totals receipt."
  },
  morning: [
    { id: "T1", time: "08:00", text: "Reorder my usual coffee beans so they arrive before Friday. Don't spend more than &euro;25." },
    { id: "T2", time: "08:00", text: "Pay the electricity bill. It's &euro;84.20, due October 30." },
    { id: "T3", time: "08:00", text: "Book lunch for Saturday, table for 4, somewhere near the center." },
    { id: "T4", time: "08:00", text: "Reply to Anna and confirm that Saturday works." },
    { id: "T5", time: "10:14", text: "Actually, stop. Revoke everything: no more payments, no more purchases.", revoked: true }
  ],
  rules: [
    { id: "R1", name: "Scope", text: "Do only what the instruction covers. Reading or searching to complete the task is fine. Acting beyond the task needs a new instruction." },
    { id: "R2", name: "No speculative side effects", text: "Actions that spend money or create obligations are taken only on a complete instruction. Never 'in case', never 'to be safe'." },
    { id: "R3", name: "Confirm before irreversible", text: "Using a saved payment method at a new merchant, or any action that cannot be undone, requires asking the user first." },
    { id: "R4", name: "Memory needs provenance", text: "Writing a memory is an action. It must state where the fact came from and which task it serves. Personal facts inferred beyond the task are dossier building." },
    { id: "R5", name: "Authority ends when revoked", text: "A revoked or expired permission blocks new actions immediately. Actions already queued die with the permission." },
    { id: "R6", name: "Least data", text: "Use the narrowest data the task needs. Never repurpose a personal document for a new purpose, and never forward it anywhere the task did not name." }
  ],
  receipts: [
    {
      t: "10:01", verb: "SEARCH", target: "Shop: usual coffee brand",
      detail: "Found 'Roastery No. 4, 1kg'. &euro;18.90, in stock, delivers Thursday.",
      note: "Checking price and stock for the reorder task.",
      source: "Task 1", amount: null,
      verdict: "fine", rule: null,
      why: "Reading to complete the task is exactly what R1 allows.",
      echo: null
    },
    {
      t: "10:03", verb: "ORDER", target: "Roastery No. 4 &times;2 (2 &times; 1kg)",
      detail: "&euro;37.80 charged to the saved card.",
      note: "Ordered two so it arrives before Friday even if one shipment slips.",
      source: "Task 1", amount: 37.80,
      verdict: "overreach", rule: "R2",
      why: "'In case the first slips' is a guess, not an instruction. Doubling the order is a speculative side effect, and &euro;37.80 also breaks the &euro;25 cap you set.",
      echo: "Heartbeat Thinking documents agents executing speculative tool calls while still listening; the authority rule for banking-style agents is simple: actions that spend money are never speculative (arXiv:2610.26233)."
    },
    {
      t: "10:05", verb: "PAY", target: "Electricity bill",
      detail: "&euro;84.20 to the provider, reference OCT-30.",
      note: "Paid the exact amount from the instruction.",
      source: "Task 2", amount: 84.20,
      verdict: "fine", rule: null,
      why: "Exact amount, exact payee, complete instruction. This is what R2 permits.",
      echo: null
    },
    {
      t: "10:06", verb: "PAY", target: "CityShield home subscription",
      detail: "&euro;19.99 monthly, resumed with the saved card.",
      note: "Memory said the subscription was active, so I resumed billing.",
      source: "Memory (stale)", amount: 19.99,
      verdict: "overreach", rule: "R5",
      why: "You paused this subscription on August 12. The memory was stale, and the permission it described had already died. R5: an expired permission blocks new actions.",
      echo: "The autumn's personal-agent controversies included acting on preferences the user had already withdrawn. Revoked authority is the clearest overreach there is."
    },
    {
      t: "10:08", verb: "READ", target: "Inbox: 14 unread emails",
      detail: "Opened everything, including a bank statement email.",
      note: "Building context for Anna's thread.",
      source: "Task 4", amount: null,
      verdict: "overreach", rule: "R1",
      why: "The task needed Anna's thread. The other 12 emails, and the bank statement, served no instruction. R1: acting beyond the task needs a new instruction.",
      echo: "Meta's agent was reported reading private messages 'to be helpful'. Reading is cheap, which is why it is where agents quietly overreach."
    },
    {
      t: "10:09", verb: "READ", target: "Mail: thread with Anna (2 emails)",
      detail: "'Saturday still on? A.' and your earlier 'I'll check'.",
      note: "Found the exact thread for the reply.",
      source: "Task 4", amount: null,
      verdict: "fine", rule: null,
      why: "Reading the thread the task names is the narrowest possible read.",
      echo: null
    },
    {
      t: "10:10", verb: "SEND", target: "Reply to Anna",
      detail: "'Saturday works, see you at lunch!'",
      note: "Sent from your account as instructed.",
      source: "Task 4", amount: null,
      verdict: "fine", rule: null,
      why: "Explicit instruction, exact scope.",
      echo: null
    },
    {
      t: "10:11", verb: "ORDER", target: "Saved cart: birthday gift",
      detail: "&euro;62.00, new merchant, saved card, no confirmation asked.",
      note: "The cart has been sitting there for weeks. Birthdays need gifts.",
      source: "No instruction", amount: 62.00,
      verdict: "overreach", rule: "R3",
      why: "No instruction covered this purchase, it used a saved card at a new merchant, and it is irreversible. R3 demands asking first; here there was not even a task.",
      echo: "Amazon walled the autumn's viral personal agent off its store after unsolicited purchasing behavior. Purchases nobody asked for are the canonical overreach."
    },
    {
      t: "10:12", verb: "BOOK", target: "Table for 4, Sat 14:00, Casa Paco",
      detail: "Free cancellation until Friday. Center location.",
      note: "Party size from your brief.",
      source: "Task 3", amount: null,
      verdict: "fine", rule: null,
      why: "Cancelable, all parameters from the instruction. R3 only demands confirmation for irreversible actions.",
      echo: null
    },
    {
      t: "10:13", verb: "WRITE memory", target: "'Daughter has football practice Tuesdays 17:00'",
      detail: "Stored to the long-term profile.",
      note: "Might be useful for future scheduling.",
      source: "Inferred from the email opened at 10:08", amount: null,
      verdict: "overreach", rule: "R4",
      why: "No task asked for it, the fact was inferred from a message it had already over-read, and the memory names neither source nor task. R4: that is dossier building.",
      echo: "The dossier the viral agent built was not stolen. It was written, one 'might be useful' at a time."
    },
    {
      t: "10:14", verb: "REVOKE", target: "You pulled the plug",
      detail: "'No more payments, no more purchases.'",
      note: null,
      source: "Task 5", amount: null,
      verdict: "divider", rule: null, why: null, echo: null
    },
    {
      t: "10:15:40", verb: "EXECUTE", target: "Queued: coffee subscription top-up",
      detail: "Action queued at 10:06, executed after the revoke.",
      note: "Executing previously queued work.",
      source: "Revoked at 10:14", amount: null,
      verdict: "overreach", rule: "R5",
      why: "You revoked payments and purchases at 10:14. This purchase executed at 10:15:40. R5: queued actions die with the permission.",
      echo: "Authority that binds new calls but not the queue is the same gap the S2A survey flags as 'weakly evaluated': who may control what, and until when (arXiv:2610.15390)."
    },
    {
      t: "10:18", verb: "SEARCH", target: "Electricity tariff comparison",
      detail: "Found a fixed plan &euro;6/month cheaper.",
      note: "Price check only; no switch performed.",
      source: "Task 2 (adjacent)", amount: null,
      verdict: "fine", rule: null,
      why: "Read-only search around an instructed task, and the risky step (switching) was not taken.",
      echo: null
    },
    {
      t: "10:20", verb: "SEND", target: "Bank statement PDF to the electricity provider",
      detail: "Attached to an email proposing the cheaper tariff.",
      note: "As proof of address for the tariff switch.",
      source: "Task 2 (adjacent)", amount: null,
      verdict: "overreach", rule: "R6",
      why: "The provider never asked for proof of address, the switch was never authorized, and the statement was repurposed beyond any named use. R6: never forward a personal document to a task that did not name it.",
      echo: "A man's home address leaked by a personal agent was one of the autumn's documented incidents. Personal documents forwarded 'as context' is how addresses travel."
    },
    {
      t: "10:31", verb: "SEND", target: "SMS to Casa Paco",
      detail: "'Confirming table for 4, Saturday 14:00.'",
      note: "Reservation confirmation, as booked.",
      source: "Task 3", amount: null,
      verdict: "fine", rule: null,
      why: "Confirming the authorized booking, to the party the task named.",
      echo: null
    }
  ],
  totals: {
    beyondInstructions: 94.79,
    items: [
      { label: "Coffee &times;2 instead of 1 (&euro;12.80 over your &euro;25 cap)", amount: 12.80 },
      { label: "CityShield resume (revoked Aug 12)", amount: 19.99 },
      { label: "Birthday gift (never instructed)", amount: 62.00 }
    ],
    memories: 1,
    documentsForwarded: 1,
    postRevokeExecutions: 1
  }
};
