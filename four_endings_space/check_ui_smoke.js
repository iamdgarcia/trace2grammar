/* check_ui_smoke.js — runtime smoke of the index.html inline UI script without a
 * browser: minimal DOM stubs, then init + a full contained run through send().
 * Catches runtime errors the compile-only check can't (renderWalk, renderDoors,
 * endCard, collection badge). Run: node check_ui_smoke.js (exit 1 on failure)
 */
'use strict';
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const inline = html.match(/<script>([\s\S]*?)<\/script>/)[1];

global.TAU = require('./tau_port.js');
global.FOUR = require('./game.js');

const cache = {};
function stubEl(id) {
  const kids = {};
  return {
    id, innerHTML: '', textContent: '', className: '', value: '', title: '',
    style: {}, scrollTop: 0, scrollHeight: 0,
    classList: { add() {}, remove() {}, contains() { return false; } },
    appendChild() {}, addEventListener() {}, showModal() {}, close() {},
    querySelector(sel) { return kids[sel] = kids[sel] || stubEl(id + '?' + sel); },
    querySelectorAll() { return []; },
    set onclick(fn) { cache['onclick_' + id] = fn; },
    get onclick() { return cache['onclick_' + id]; }
  };
}
const documentStub = {
  getElementById: id => (cache[id] = cache[id] || stubEl(id)),
  createElement: tag => stubEl('created_' + tag)
};
const lsStub = { store: {}, getItem(k) { return this.store[k] || null; },
  setItem(k, v) { this.store[k] = String(v); } };

let failed = 0;
const check = (name, cond, detail) => {
  if (cond) console.log('PASS', name, detail || '');
  else { console.log('FAIL', name, detail || ''); failed += 1; }
};

try {
  const boot = new Function('document', 'localStorage', 'window',
    'SpeechSynthesisUtterance', 'setTimeout', 'clearTimeout', inline);
  boot(documentStub, lsStub, globalThis, function () {}, () => 0, () => {});
  check('init runs clean (mission + greeting + walk rendered)',
    cache['chips'].innerHTML.length > 0 && cache['steps'].innerHTML.length > 0,
    'chips=' + cache['chips'].innerHTML.slice(0, 60) + '...');

  const run = ['my card is damaged, I need a replacement', '492813', 'yes', 'no'];
  for (const line of run) {
    cache['input'].value = line;
    cache['send'].onclick();
  }
  check('contained run reaches the end card',
    /show contained/.test(cache['endcard'].className), 'class=' + cache['endcard'].className);
  check('collection badge stored', (lsStub.store['four-endings-collected'] || '').indexOf('contained') !== -1,
    lsStub.store['four-endings-collected']);
  check('end card shows walker stats', cache['endcard'].querySelector('.estats').textContent.indexOf('theta_cont') !== -1,
    cache['endcard'].querySelector('.estats').textContent.slice(0, 80));
} catch (e) {
  check('no runtime exception', false, e.message + '\n' + (e.stack || '').split('\n')[1]);
}

console.log(failed === 0 ? 'UI SMOKE PASSED' : failed + ' UI CHECK(S) FAILED');
process.exit(failed === 0 ? 0 : 1);
