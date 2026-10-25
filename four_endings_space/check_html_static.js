/* check_html_static.js — static checks on index.html without a browser:
 *  1) every id referenced by getElementById/$('...') exists in the markup
 *  2) the inline <script> compiles (new Function, no execution)
 *  3) README frontmatter: required keys + short_description <= 60 chars
 * Run: node check_html_static.js  (exit 1 on any failure)
 */
'use strict';
const fs = require('fs');
const path = require('path');
let fails = 0;
const check = (name, cond, detail) => {
  if (cond) console.log('PASS', name, detail || '');
  else { console.log('FAIL', name, detail || ''); fails += 1; }
};

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');

/* 1) ids referenced vs defined */
const defined = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
const referenced = new Set([...html.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]));
const missing = [...referenced].filter(id => !defined.has(id));
check('all referenced ids exist in markup', missing.length === 0,
  missing.length ? 'missing: ' + missing.join(', ') : referenced.size + ' ids checked');

/* dynamic ids built in JS: door-<k> from ['contained','handoff','abandoned','escape'] */
const dynIds = ['door-contained','door-handoff','door-abandoned','door-escape'];
const dynMissing = dynIds.filter(id => !defined.has(id));
check('dynamic door ids exist', dynMissing.length === 0,
  dynMissing.length ? 'missing: ' + dynMissing.join(', ') : dynIds.length + ' doors');

/* 2) inline script compiles */
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
check('exactly one inline script block', scripts.length === 1, scripts.length + ' blocks');
let compileOk = false, compileErr = '';
try { new Function(scripts[0]); compileOk = true; } catch (e) { compileErr = e.message; }
check('inline script compiles', compileOk, compileErr);

/* 3) README frontmatter */
const readme = fs.readFileSync(path.join(__dirname, 'README.md'), 'utf8');
const fm = readme.match(/^---\n([\s\S]*?)\n---/);
check('README frontmatter present', !!fm);
if (fm) {
  const keys = {};
  fm[1].split('\n').forEach(l => { const m = l.match(/^(\w+):\s*(.*)$/); if (m) keys[m[1]] = m[2]; });
  ['title','emoji','colorFrom','colorTo','sdk','app_file','short_description'].forEach(k =>
    check('frontmatter key: ' + k, !!keys[k], keys[k] || 'MISSING'));
  check('sdk is static', keys.sdk === 'static', keys.sdk);
  check('app_file is index.html', keys.app_file === 'index.html', keys.app_file);
  check('short_description <= 60 chars', (keys.short_description || '').length <= 60,
    (keys.short_description || '').length + ' chars');
}

/* 4) files shipped */
['index.html','tau_port.js','game.js','llm.js','README.md'].forEach(f =>
  check('file present: ' + f, fs.existsSync(path.join(__dirname, f))));

/* 5) tau_port byte-identical to the parity-verified original */
const crypto = require('crypto');
const a = crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname, 'tau_port.js'))).digest('hex');
const b = crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname, '..', 'trace2grammar', 'autopsy_space', 'tau_port.js'))).digest('hex');
check('tau_port.js sha256 identical to parity-verified original', a === b, a.slice(0, 16) + '...');

console.log(fails === 0 ? 'ALL STATIC CHECKS PASSED' : fails + ' CHECK(S) FAILED');
process.exit(fails === 0 ? 0 : 1);
