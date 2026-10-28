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

const g = GRAMMAR.induce(REAL.map(p => ({ walk: p.walk })));

console.log('smoke_grammar.js');
check('1,980 real runs', g.nRuns === 1980, String(g.nRuns));
const sumTrans = REAL.reduce((s, p) => s + p.walk.length - 1, 0);
check('transitions = total bigrams', g.nTransitions === sumTrans, g.nTransitions + ' vs ' + sumTrans);
check('28,576 transitions', g.nTransitions === 28576, String(g.nTransitions));
check('19 distinct edges', g.nDistinctEdges === 19, String(g.nDistinctEdges));
check('top edge q_tool>q_confirm 9310', g.edges[0].a === 'q_tool' && g.edges[0].b === 'q_confirm' && g.edges[0].count === 9310,
  JSON.stringify(g.edges[0]));
check('top6 share 81.1%', Math.abs(g.top6Share - 0.8113) < 0.001, String(g.top6Share));
check('unique templates 636', g.paths.length === 636, String(g.paths.length));
check('top template 87 runs', g.paths[0].count === 87, String(g.paths[0].count));
check('top template shape', g.paths[0].seq.join('>') === 'q_greet>q_intent>q_auth>q_tool>q_intent>q_tool>q_intent>q_tool>q_intent',
  g.paths[0].seq.join('>'));

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
check('type q_tool>q_tool self', GRAMMAR.edgeType('q_tool', 'q_tool') === 'self');
check('no terminal edges in the induced grammar', g.edges.every(e => !e.b.startsWith('theta_') && !e.a.startsWith('theta_')));

// sub-selection: a region grammar is still well-formed (shares sum to 1)
const gSub = GRAMMAR.induce(REAL.filter(p => p.D > 0).map(p => ({ walk: p.walk })));
check('D>0 subset induces 628 runs', gSub.nRuns === 628, String(gSub.nRuns));
const subShares = {};
gSub.edges.forEach(e => { (subShares[e.a] = subShares[e.a] || []).push(e.share); });
let subOK = true;
Object.keys(subShares).forEach(s => {
  const sum = subShares[s].reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 1e-9) subOK = false;
});
check('subset shares sum to 1', subOK);

// digest for the python parity check
const digest = {
  nRuns: g.nRuns, nTransitions: g.nTransitions, nDistinctEdges: g.nDistinctEdges,
  top6Share: g.top6Share,
  edges: g.edges.map(e => ({ a: e.a, b: e.b, count: e.count, type: e.type })),
  topTemplates: g.paths.slice(0, 10).map(p => ({ seq: p.seq.join('>'), count: p.count })),
  nTemplates: g.paths.length,
  visits: g.visits
};
fs.writeFileSync(path.join(__dirname, 'grammar_derived.json'), JSON.stringify(digest, null, 1));
console.log(fails === 0 ? 'SMOKE OK (' + digest.edges.length + ' edges, ' + digest.nTemplates + ' templates)' : 'SMOKE FAILED: ' + fails);
process.exit(fails === 0 ? 0 : 1);
