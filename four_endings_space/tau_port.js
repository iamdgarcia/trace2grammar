/* tau_port.js — verbatim JavaScript port of trace2grammar/extract.py (infer_states)
 * from https://github.com/iamdgarcia/trace2grammar (file: trace2grammar/extract.py).
 * Ported table-for-table, branch-for-branch. Do not "improve" here: the Space claims
 * parity with the Python implementation, validated on the repo's own fixtures
 * (tests/test_extract.py). If the Python changes, re-port and re-run the parity check.
 *
 * Browser: <script src="tau_port.js"></script> → window.TAU
 * Node:    require('./tau_port.js') → module.exports
 */
(function (root) {
  'use strict';

  var PROGRESS_STATES = ['q_greet', 'q_auth', 'q_intent', 'q_tool', 'q_clarify', 'q_confirm', 'q_close'];
  var TERMINALS = ['theta_cont', 'theta_aban', 'theta_hand', 'theta_esc'];

  /* ExtractionConfig defaults, verbatim from extract.py */
  var CFG = {
    auth_tool_patterns: ['verify', 'identity', 'auth', 'otp', 'login'],
    handoff_tool_patterns: ['escalate', 'transfer', 'handoff', 'human', 'agent_queue'],
    clarify_user_patterns: [
      '\\bperd[oó]n\\b',
      '\\bd[ií]galo otra vez\\b',
      '\\bno entiendo\\b',
      '\\bwhat do you mean\\b',
      '\\bcome again\\b',
      '\\bcan you repeat\\b',
      '\\ba ver\\b'
    ],
    confirm_user_patterns: [
      '\\bs[ií]\\b',
      '\\bcorrecto\\b',
      '\\bconfirmo\\b',
      '\\byes\\b',
      "\\bthat'?s right\\b"
    ],
    close_assistant_patterns: [
      '\\bgracias por su llamada\\b',
      '\\bque tenga\\b',
      '\\bhave a (good|great|nice)\\b',
      '\\bgoodbye\\b',
      '\\bchao\\b',
      '\\badi[oó]s\\b'
    ],
    complete_event_names: ['service_complete', 'service.complete', 'call_completed', 'hangup_complete'],
    hangup_event_names: ['caller_hangup', 'user_hangup', 'hangup', 'caller_disconnected'],
    handoff_event_names: ['transfer_complete', 'escalated', 'human_transfer'],
    max_identity_span_ms: 10000
  };

  /* python re.search(patterns joined by |, re.I) equivalent.
   * Content is accent-folded before matching so the ASCII \\b boundaries behave
   * the same as Python's unicode \\b (perdón → perdon still hits [oó]). */
  function fold(s) {
    return String(s).normalize('NFD').replace(/[̀-ͯ]/g, '');
  }
  function searchAny(patterns, text) {
    var re = new RegExp(patterns.join('|'), 'i');
    return re.test(fold(text));
  }

  /* infer_states port. events: [{t_ms, kind, name, attrs}] */
  function inferStates(events, config) {
    var cfg = config || CFG;
    var walk = [{ t_ms: events.length ? events[0].t_ms : 0, state: 'q_greet', transition_type: 'start' }];
    var state = 'q_greet';
    var cycleDepth = 0;
    var errorCount = 0;
    var silentFailure = false;
    var pendingAuthCall = false;
    var lastUserT = null;
    var outcome = null;

    function emit(t_ms, st, ttype) {
      walk.push({ t_ms: t_ms, state: st, transition_type: ttype });
      state = st;
    }

    events.forEach(function (ev) {
      var kind = ev.kind;
      var name = ev.name || '';
      var content = String((ev.attrs || {}).content || '');

      if (kind === 'user_msg') {
        lastUserT = ev.t_ms;
        if (state === 'q_greet') {
          emit(ev.t_ms, 'q_intent', 'user-answer');
        } else if (searchAny(cfg.clarify_user_patterns, content)) {
          /* python's `settled` set is never populated (dead variable), so its
           * clarify branch never increments cycle depth; port matches that. */
          emit(ev.t_ms, 'q_clarify', 'agent-clarify');
        } else if (state === 'q_clarify') {
          emit(ev.t_ms, 'q_intent', 'user-answer');
        } else if (state === 'q_tool') {
          emit(ev.t_ms, 'q_intent', 'user-answer');
        } else if (state === 'q_intent' || state === 'q_confirm') {
          if (searchAny(cfg.confirm_user_patterns, content) && state === 'q_confirm') {
            emit(ev.t_ms, 'q_close', 'user-answer');
          }
        }
      } else if (kind === 'assistant_msg') {
        if (searchAny(cfg.close_assistant_patterns, content)) {
          emit(ev.t_ms, 'q_close', 'user-answer');
        } else if (state === 'q_clarify') {
          emit(ev.t_ms, 'q_intent', 'user-answer');
        } else if (state === 'q_tool') {
          var hadFail = walk.some(function (w) { return w.transition_type === 'tool-fail' || w.transition_type === 'auth-fail'; });
          if (cycleDepth === 0 || !hadFail) {
            emit(ev.t_ms, 'q_confirm', 'user-answer');
          }
        }
      } else if (kind === 'tool_call') {
        if (searchAny(cfg.auth_tool_patterns, name)) {
          if (state === 'q_auth' && !pendingAuthCall) {
            emit(ev.t_ms, 'q_auth', 'auth-fail');
            cycleDepth += 1;
          } else if (state !== 'q_auth') {
            emit(ev.t_ms, 'q_auth', 'user-answer');
          }
          pendingAuthCall = true;
        } else {
          if (state === 'q_tool') {
            /* iteration: progress self-loop, no new step */
          } else {
            emit(ev.t_ms, 'q_tool', 'tool-call');
          }
        }
      } else if (kind === 'tool_result') {
        pendingAuthCall = false;
        /* silent-failure signal: an "ok" result that carried zero results
         * (structured count attr at 0, or the adapter convention 'n_results=0'
         * in the result content). Flag only; outcome classification untouched. */
        if (!silentFailure) {
          var a = ev.attrs || {};
          var nRes;
          ['result_count', 'tool.result_count', 'gen_ai.tool.call.result_count', 'n_results'].some(function (k) {
            if (k in a) { nRes = a[k]; return true; }
            return false;
          });
          if (nRes === undefined && /\bn_results=0\b/.test(content)) nRes = 0;
          if (String(nRes) === '0' || String(nRes) === '0.0') silentFailure = true;
        }
      } else if (kind === 'tool_error') {
        errorCount += 1;
        if (state === 'q_auth') {
          emit(ev.t_ms, 'q_auth', 'auth-fail');
          cycleDepth += 1;
        } else {
          emit(ev.t_ms, 'q_tool', 'tool-fail');
          cycleDepth += 1;
        }
      } else if (kind === 'handoff') {
        emit(ev.t_ms, 'theta_hand', 'escalation');
        outcome = 'theta_hand';
      } else if (kind === 'hangup') {
        emit(ev.t_ms, 'theta_aban', 'hangup');
        outcome = 'theta_aban';
      } else if (kind === 'session_end') {
        emit(ev.t_ms, TERMINALS.indexOf(state) === -1 ? 'q_close' : state,
             state === 'q_confirm' ? 'disclosure' : 'service-complete');
        if (state !== 'theta_aban' && state !== 'theta_hand') {
          outcome = 'theta_cont';
        }
      }
    });

    if (outcome === null) {
      if (state === 'q_close') {
        outcome = 'theta_cont';
      } else if (lastUserT !== null && walk.length &&
                 walk[walk.length - 1].t_ms - lastUserT > 2000 &&
                 ['q_auth', 'q_tool', 'q_intent', 'q_clarify'].indexOf(state) !== -1) {
        outcome = 'theta_aban';
      } else {
        outcome = state === 'q_close' ? 'theta_cont' : 'theta_aban';
      }
    }
    return { walk: walk, outcome: outcome, cycle_depth: cycleDepth, error_count: errorCount, silent_failure_candidate: silentFailure };
  }

  /* Transcript → canonical events: VERBATIM port of _normalize_transcript
   * (extract.py). Parity-checked by check_parity_transcript.py on the repo samples
   * + crafted edge cases; diverges intentionally from "nicer" JS (alias roles are
   * NOT roles here, they are continuation lines; pacing is sha1-jittered like the
   * python; handoff-typed tools emit the handoff event that drives theta_hand).
   * Returns {events, lines}: lines[i] is the 1-based source line of events[i] for
   * autopsy quoting. Line/end_line bookkeeping rides OUTSIDE attrs so the events
   * stay byte-equal to the python ones for the walker. */
  var TRANSCRIPT_ROLE_RE = /^(assistant|user|tool)\s*(?:\(([\w.\-]+)\))?\s*[:>]\s*(.*)$/i;
  var ASSISTANT_PREFIXES = ['assistant', 'agente', 'bot', 'agent', 'sistema', 'system', 'ai'];

  /* sync sha1 (hex) of a UTF-8 string — the python pacing hashes each line */
  function sha1Hex(s) {
    var bytes = new TextEncoder().encode(s);
    var ml = bytes.length;
    var withOne = new Uint8Array(((ml + 8) >> 6 << 6) + 64);
    withOne.set(bytes);
    withOne[ml] = 0x80;
    var bitLenHi = Math.floor(ml / 536870912); // ml*8 / 2^32
    var bitLenLo = (ml << 3) >>> 0;
    var dv = new DataView(withOne.buffer);
    dv.setUint32(withOne.length - 8, bitLenHi);
    dv.setUint32(withOne.length - 4, bitLenLo);
    var H0 = 0x67452301, H1 = 0xEFCDAB89, H2 = 0x98BADCFE, H3 = 0x10325476, H4 = 0xC3D2E1F0;
    var w = new Uint32Array(80);
    for (var off = 0; off < withOne.length; off += 64) {
      for (var i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4);
      for (var j = 16; j < 80; j++) {
        var x = w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16];
        w[j] = (x << 1) | (x >>> 31);
      }
      var a = H0, b = H1, c = H2, d = H3, e = H4;
      for (var k = 0; k < 80; k++) {
        var f, kk;
        if (k < 20) { f = (b & c) | (~b & d); kk = 0x5A827999; }
        else if (k < 40) { f = b ^ c ^ d; kk = 0x6ED9EBA1; }
        else if (k < 60) { f = (b & c) | (b & d) | (c & d); kk = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; kk = 0xCA62C1D6; }
        var tmp = (((a << 5) | (a >>> 27)) + f + e + kk + w[k]) >>> 0;
        e = d; d = c; c = (b << 30) | (b >>> 2); b = a; a = tmp;
      }
      H0 = (H0 + a) >>> 0; H1 = (H1 + b) >>> 0; H2 = (H2 + c) >>> 0;
      H3 = (H3 + d) >>> 0; H4 = (H4 + e) >>> 0;
    }
    return [H0, H1, H2, H3, H4].map(function (n) {
      return ('00000000' + n.toString(16)).slice(-8);
    }).join('');
  }

  function normalizeTranscript(text, config) {
    var cfg = config || CFG;
    var lines = String(text).split(/\r?\n/);
    var events = [];
    var srcLines = [];
    var t = 0;
    var handoffRe = new RegExp(cfg.handoff_tool_patterns.join('|'), 'i');
    lines.forEach(function (rawLine, i) {
      var line = rawLine.trim();
      if (!line) return;
      /* python content-aware pacing (verbatim): tool ~0.9-1.5s, assistant ~3-6s,
       * user ~2.2-4.8s, sha1-jittered so identical shapes differ */
      var h = parseInt(sha1Hex(line).slice(0, 6), 16) / 0xFFFFFF;
      var low = line.toLowerCase();
      if (low.indexOf('tool') === 0) t += Math.floor(900 + 600 * h);
      else if (ASSISTANT_PREFIXES.some(function (p) { return low.indexOf(p) === 0; }))
        t += Math.floor(3000 + 3000 * h);
      else t += Math.floor(2200 + 2600 * h);
      var m = line.match(TRANSCRIPT_ROLE_RE);
      if (!m) {
        /* continuation of the previous message (verbatim python behavior) */
        if (events.length) {
          events[events.length - 1].attrs.content += ' ' + line;
          events[events.length - 1].attrs.end_line = i + 1;
        }
        return;
      }
      var role = m[1].toLowerCase(), tool = m[2], rest = m[3].trim();
      if (role === 'tool') {
        var name = tool || 'tool';
        var failed = rest.toLowerCase().indexOf('error') === 0;
        if (handoffRe.test(name)) {
          /* parity with the otel/langsmith normalizers: handoff-typed tools
           * emit the handoff event that drives theta_hand */
          events.push({ t_ms: t + 1, kind: 'handoff', name: name, attrs: {} });
          srcLines.push(i + 1);
          return;
        }
        events.push({ t_ms: t, kind: 'tool_call', name: name, attrs: {} });
        srcLines.push(i + 1);
        events.push({
          t_ms: t + 1,
          kind: failed ? 'tool_error' : 'tool_result',
          name: name,
          attrs: { content: rest }
        });
        srcLines.push(i + 1);
        return;
      }
      var isAssistant = ['assistant', 'agente', 'bot', 'agent', 'sistema', 'system', 'ai']
        .indexOf(role) !== -1;
      events.push({
        t_ms: t,
        kind: isAssistant ? 'assistant_msg' : 'user_msg',
        name: null,
        attrs: { content: rest }
      });
      srcLines.push(i + 1);
    });
    return { events: events, lines: srcLines };
  }

  var TAU = {
    PROGRESS_STATES: PROGRESS_STATES,
    TERMINALS: TERMINALS,
    CFG: CFG,
    inferStates: inferStates,
    normalizeTranscript: normalizeTranscript
  };

  if (typeof module !== 'undefined' && module.exports) { module.exports = TAU; }
  else { root.TAU = TAU; }
})(this);
