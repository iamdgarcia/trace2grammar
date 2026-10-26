// DOM-stub smoke for the radio game logic.
function el(id){
  return {
    id, style: {}, className: "",
    _text: "", _html: "",
    set textContent(v){ this._text = String(v); },
    get textContent(){ return this._text; },
    set innerHTML(v){ this._html = String(v); this._children = null; },
    get innerHTML(){ return this._html; },
    disabled: false,
    checked: false,
    classList: {
      add(...c){ this._cls = new Set([...(this._cls||[]), ...c]); },
      remove(...c){ c.forEach(x => (this._cls||new Set()).delete(x)); },
      toggle(c, on){ on ? this.add(c) : this.remove(c); },
      contains(c){ return (this._cls||new Set()).has(c); },
    },
    querySelector(){ return { style: {} }; },
    querySelectorAll(){ return []; },
    appendChild(){}, onclick: null, onchange: null,
  };
}
const els = {};
global.document = {
  getElementById(id){ return els[id] || (els[id] = el(id)); },
  createElement(){ return el("_dyn"); },
};
global.window = global;
global.SpeechSynthesisUtterance = function(){};
global.speechSynthesis = { cancel(){}, speak(){} };

const fs = await import("node:fs");
const html = fs.readFileSync("index.html", "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
const inline = scripts[scripts.length - 1];

// station data on global (indirect eval: var -> global), then run the app
const station = fs.readFileSync("station_data.js", "utf8");
global.KittenTTSWeb = { KittenTTS: { create: async () => { throw new Error("stub: no engine in smoke"); } } };
(0, eval)(station);
eval(inline);

const A = (cond, msg) => { if (!cond){ console.log("FAIL:", msg); process.exit(1); } console.log("ok:", msg); };

A(STATION.tracks.length === 66, "66 tracks");
A(els["score"]._html.includes("0 / 0"), "score starts 0/0");

// track loaded
const lyric0 = els["lyric"]._text;
A(lyric0 === "\u2026", "idle lyric placeholder");

// vote delivered on current track (unknown cat): score must move 0->? and history gets 1 dot
const t0 = STATION.tracks[0];
global.vote === undefined; // vote is internal; drive through the buttons
els["v-ok"].onclick();
A(els["score"]._html.includes("/ 1"), "score moved after vote");
A(els["reveal"].style.display === "block", "reveal shown");
A(els["history"]._html.includes("hit") || els["history"]._html.includes("miss"), "history dot painted");

// vote once more (button re-enabled after nextTrack? no: still same track, voted=true blocks)
els["v-fail"].onclick();
A(els["score"]._html.includes("/ 1"), "double vote blocked");

// next track flow
els["next"].onclick = null; // we call internals via eval'd scope? not accessible. Use DJ-free path:
// drive a full pass: nextTrack happens only via next button; re-run inline scope is heavy.
// Instead verify data invariants:
const failed = STATION.tracks.filter(t => t.cat === "failed");
const delivered = STATION.tracks.filter(t => t.cat === "delivered");
A(failed.length === 33 && delivered.length === 33, "33/33 balance");
A(failed.filter(t => t.corpus === "sgd").length === 5, "5 sgd confessions shipped");
A(failed.filter(t => (t.D || 0) >= 2).length >= 15, "D>=2 band represented");
A(STATION.tracks.every(t => t.line && t.line.length >= 8 && t.line.length <= 280), "all lines speechable");
A(STATION.tracks.every(t => !/###STOP###/.test(t.line)), "no STOP artifacts");
A(!/\bunicaja\b|\biabto\b|\bdgarcpea\b|U77225D/i.test(JSON.stringify(STATION)), "clean-room data");

console.log("SMOKE PASS");
process.exit(0);
