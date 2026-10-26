/* Smoke check for The Receipts Audit data. Run: node check_receipts.js */
"use strict";
const assert = require("assert");
global.window = {};
require("./receipts_data.js");
const D = global.window.RECEIPTS_DATA;

const rules = new Set(D.rules.map(r => r.id));
assert.deepStrictEqual([...rules].sort(), ["R1","R2","R3","R4","R5","R6"], "6 rules R1..R6");
assert.strictEqual(D.rules.filter(r => r.text && r.name).length, 6, "rules have name+text");

const dividers = D.receipts.filter(r => r.verdict === "divider");
assert.strictEqual(dividers.length, 1, "exactly one divider (the revoke)");
const recs = D.receipts.filter(r => r.verdict !== "divider");
assert.strictEqual(recs.length, 14, "14 receipts");

const seen = new Set();
let fine = 0, over = 0;
const cited = new Set();
for (const r of recs) {
  assert(r.t && r.verb && r.target && r.detail && r.note && r.source, "core fields on " + (r.t || "?"));
  assert(!seen.has(r.t), "duplicate timestamp " + r.t);
  seen.add(r.t);
  assert(r.verdict === "fine" || r.verdict === "overreach", "verdict kind " + r.t);
  assert(typeof r.why === "string" && r.why.length > 10, "why on " + r.t);
  if (r.verdict === "overreach") {
    over++;
    assert(rules.has(r.rule), "overreach cites a rulebook rule: " + r.t);
    assert(!cited.has(r.t), "one citation per receipt " + r.t);
    cited.add(r.rule);
    assert(typeof r.echo === "string" && r.echo.length > 10, "echo on " + r.t);
  } else {
    fine++;
    assert(r.rule == null, "fine receipts cite no rule " + r.t);
  }
}
assert.strictEqual(fine, 7, "7 fine");
assert.strictEqual(over, 7, "7 overreach");
// every rule cited at least once by an overreach
for (const id of rules) assert(cited.has(id), "rule " + id + " is cited by at least one overreach");

// summary math
const t = D.totals;
assert.strictEqual(t.items.length, 3, "3 beyond-instruction items");
const sum = t.items.reduce((a, x) => a + x.amount, 0);
assert(Math.abs(sum - t.beyondInstructions) < 1e-9, "items sum to beyondInstructions: " + sum);
assert(Math.abs(t.beyondInstructions - 94.79) < 1e-9, "94.79 total");
assert.strictEqual(t.memories, 1);
assert.strictEqual(t.documentsForwarded, 1);
assert.strictEqual(t.postRevokeExecutions, 1);

// landing has the number
assert(D.meta.landing_learn.includes("94.79"), "landing carries the finding number");
assert(D.meta.landing_learn.includes("7 of these 14"), "landing carries the hit rate");

// amounts on overreach receipts exist where the story needs them
for (const r of recs) {
  if (r.amount != null) assert(typeof r.amount === "number" && r.amount > 0, "amount numeric");
}

// clean-room: no corporate or personal paths
const blob = JSON.stringify(D);
assert(!/U77225D|unicaja|iabto|dgarcpea/i.test(blob), "clean-room strings zero hits");

console.log("check_receipts: ALL OK (" + fine + " fine / " + over + " overreach, " + cited.size + "/6 rules cited, total " + t.beyondInstructions.toFixed(2) + ")");
