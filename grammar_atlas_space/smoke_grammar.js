/* Smoke test for the Trajectory Grammar Atlas: grammar induction over the embedded
 * real tau-bench walks. Run: node smoke_grammar.js   (from this directory)
 * Writes grammar_derived.json (digest) for the python parity check.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const GRAMMAR = require('./grammar.js');

const src = fs.readFileSync(path.join(__dirname, 'atlas_data.js'), 'utf8');
const blob = JSON.parse(src.slice(src.indexOf('=') + 1).replace(/;\s*$/, ''));
const REAL = blob.points.filter(p => p.kind === 'real');

let fails = 0;
function check(name, cond, detail) {
  if (cond) console.log('  PASS ' + name);
  else { fails++; console.log('  FAIL ' + name + (detail ? ' — ' + detail : '')); }
}

const g = GRAMMAR.induce(REAL.map(p => ({ walk: p.walk, outcome: p.outcome })));

console.log('smoke_grammar.js');
check('1,980 real runs', g.nRuns === 1980, String(g.nRuns));
const sumStates = REAL.reduce((s, p) => s + p.walk.length, 0);
check('transitions = total walk states', g.nTransitions === sumStates, g.nTransitions + ' vs ' + sumStates);
check('24 distinct edges', g.nDistinctEdges === 24, String(g.nDistinctEdges));
check('top edge q_tool>q_confirm 9310', g.edges[0].a === 'q_tool' && g.edges[0].b === 'q_confirm' && g.edges[0].count === 9310,
  JSON.stringify(g.edges[0]));
check('unique templates 636', g.paths.length === 636, String(g.paths.length));
check('top template 87 runs', g.paths[0].count === 87, String(g.paths[0].count));
check('contained reach q_close 378/378', g.reachClose.theta_cont[0] === 378 && g.reachClose.theta_cont[1] === 378,
  JSON.stringify(g.reachClose.theta_cont));
check('abandoned reach q_close 1021/1602', g.reachClose.theta_aban[0] === 1021 && g.reachClose.theta_aban[1] === 1602,
  JSON.stringify(g.reachClose.theta_aban));

// per-source shares must sum to 1
const bySource = {};
g.edges.forEach(e => { (bySource[e.a] = bySource[e.a] || []).push(e.share); });
let shareOK = true;
Object.keys(bySource).forEach(s => {
  const sum = bySource[s].reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-9) { shareOK = false; console.log('    source ' + s + ' shares sum ' + sum); }
});
check('per-source shares sum to 1', shareOK);

// edge typing sanity
check('type q_tool>q_confirm progress', GRAMMAR.edgeType('q_tool', 'q_confirm') === 'progress');
check('type q_confirm>q_tool backward', GRAMMAR.edgeType('q_confirm', 'q_tool') === 'backward');
check('type q_close>theta_cont terminal', GRAMMAR.edgeType('q_close', 'theta_cont') === 'terminal');
check('type q_tool>q_tool self', GRAMMAR.edgeType('q_tool', 'q_tool') === 'self');

// sub-selection grammar: contained only -> every terminal edge is theta_cont
const gCont = GRAMMAR.induce(REAL.filter(p => p.outcome === 'theta_cont').map(p => ({ walk: p.walk, outcome: p.outcome })));
check('contained-only grammar has only theta_cont terminals',
  gCont.edges.every(e => e.b !== 'theta_aban' && e.b !== 'theta_hand' && e.b !== 'theta_esc'));
check('contained-only terminal count 378', gCont.terminalCounts.theta_cont === 378, String(gCont.terminalCounts.theta_cont));

// digest for the python parity check
const digest = {
  nRuns: g.nRuns, nTransitions: g.nTransitions, nDistinctEdges: g.nDistinctEdges,
  top6Share: g.top6Share,
  edges: g.edges.map(e => ({ a: e.a, b: e.b, count: e.count, type: e.type })),
  topTemplates: g.paths.slice(0, 10).map(p => ({ seq: p.seq.join('>'), outcome: p.outcome, count: p.count })),
  nTemplates: g.paths.length,
  reachClose: g.reachClose,
  visits: g.visits
};
fs.writeFileSync(path.join(__dirname, 'grammar_derived.json'), JSON.stringify(digest, null, 1));
console.log(fails === 0 ? 'SMOKE OK (' + digest.edges.length + ' edges, ' + digest.nTemplates + ' templates)' : 'SMOKE FAILED: ' + fails);
process.exit(fails === 0 ? 0 : 1);
