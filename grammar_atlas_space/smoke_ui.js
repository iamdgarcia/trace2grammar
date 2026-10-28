/* UI smoke with a DOM stub: runs the page's app code headlessly and asserts the
 * grammar SVG and rules lists are built. Run: node smoke_ui.js
 */
'use strict';
const fs = require('fs');
const path = require('path');

/* ---- minimal DOM stub ---- */
function makeEl(tag) {
  const el = {
    tag, children: [], dataset: {}, style: {}, _class: new Set(),
    classList: {
      add: c => el._class.add(c), remove: c => el._class.delete(c),
      toggle: (c, v) => { (v === undefined ? !el._class.has(c) : v) ? el._class.add(c) : el._class.delete(c); return el._class.has(c); },
      contains: c => el._class.has(c)
    },
    setAttribute(k, v) { el.attrs[k] = v; },
    getAttribute(k) { return el.attrs[k]; },
    appendChild(c) { el.children.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    querySelectorAll() { return []; },
    closest() { return null; },
    getContext() { return new Proxy({}, { get: (t, k) => (k === 'canvas' ? canvas : () => { }) }); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 760, height: 560 }; },
    innerHTML: '', textContent: '', value: 'all',
    attrs: {}
  };
  return el;
}
const byId = {};
function get(id) { return byId[id] || (byId[id] = makeEl('div#' + id)); }
const ths = ['n_states', 'D'].map(k => { const e = makeEl('th'); e.dataset.sort = k; return e; });

global.window = { addEventListener() {} };
global.document = {
  getElementById: get,
  createElementNS: (ns, tag) => makeEl(tag),
  createElement: tag => makeEl(tag),
  querySelectorAll: sel => (sel.indexOf('sortable') >= 0 ? ths : [])
};
const canvas = makeEl('canvas');
get('map').getContext = canvas.getContext;
global.fetch = () => Promise.resolve({ ok: false, status: 404, text: () => Promise.resolve('') });

/* load scripts in order, like the browser would */
const dir = __dirname;
const vm = require('vm');
const ctx = vm.createContext({ window: global.window, document: global.document, fetch: global.fetch, console, setTimeout, clearTimeout });
vm.runInContext(fs.readFileSync(path.join(dir, 'atlas_data.js'), 'utf8'), ctx);
vm.runInContext(fs.readFileSync(path.join(dir, 'grammar.js'), 'utf8'), ctx);
vm.runInContext('window.GRAMMAR = GRAMMAR', ctx); // vm alias; in a browser `this` === window already
const html = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const blocks = html.match(/<script>[\s\S]*?<\/script>/g) || [];
vm.runInContext(blocks[blocks.length - 1].replace(/^<script>/, '').replace(/<\/script>$/, ''), ctx);

let fails = 0;
function check(name, cond, detail) {
  if (cond) console.log('  PASS ' + name);
  else { fails++; console.log('  FAIL ' + name + (detail ? ' — ' + detail : '')); }
}

console.log('smoke_ui.js (DOM stub)');
const svg = get('grammar-svg');
function walkAll(n, acc) {
  (n.children || []).forEach(c => { acc.push(c); walkAll(c, acc); });
  return acc;
}
const all = walkAll(svg, []);
const rects = all.filter(c => c.tag === 'rect');
const texts = all.filter(c => c.tag === 'text');
const paths = all.filter(c => c.tag === 'path');
check('svg has exactly the 7 progress state rects', rects.length === 7, String(rects.length));
check('svg has node labels', texts.length >= 7, String(texts.length));
check('svg has edge paths', paths.length >= 15, String(paths.length));
check('no terminal nodes drawn', all.every(c => !(c.textContent || '').includes('θ_')));
const of = get('grammar-of').textContent;
check('grammar-of mentions 1,980 runs', of.indexOf('1,980') >= 0, of);
check('grammar-of mentions 19 edges', of.indexOf('19 edges') >= 0, of);
const flow = get('rules-flow-list').innerHTML;
check('flow rules list has q_tool edges', flow.indexOf('q_tool') >= 0, flow.slice(0, 80));
const loops = get('rules-loops-list').innerHTML;
check('loops list has backward edges', loops.indexOf('q_confirm') >= 0, loops.slice(0, 80));
const pathsList = get('rules-paths-list').innerHTML;
check('templates list populated', pathsList.indexOf('runs') >= 0, pathsList.slice(0, 80));
check('no outcome words anywhere in the rendered UI', [flow, loops, pathsList, of].every(s => s.indexOf('contained') < 0 && s.indexOf('abandoned') < 0));
check('landing nRuns filled', get('ln-nruns').textContent === '1,980', get('ln-nruns').textContent);
check('landing transitions filled', get('ln-trans').textContent === '28,576', get('ln-trans').textContent);
check('footer has tau-bench provenance', get('foot').innerHTML.indexOf('tau-bench') >= 0);
console.log(fails === 0 ? 'UI SMOKE OK' : 'UI SMOKE FAILED: ' + fails);
process.exit(fails === 0 ? 0 : 1);
