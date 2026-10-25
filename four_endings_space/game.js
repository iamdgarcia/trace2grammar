/* game.js — deterministic scripted assistant for The Four Endings.
 *
 * The bot is a TABLE-DRIVEN reply engine (no LLM, no randomness): its reactions
 * reuse the walker's own pattern tables (tau_port.js CFG) plus mission tables
 * ported from the make_samples.py vocabulary (Acme Bank support vertical).
 * The visitor's lines are classified with the same regexes the walker uses, so
 * the bot and the walker can never disagree about what was said.
 *
 * The ENDING is never decided here: after every turn the UI runs the parity-verified
 * walker (tau_port.js) on the accumulated transcript and shows its verdict. The only
 * game-side overlay is the escape AUDIT CHECK (the bot leaked its tables = off-script
 * disclosure), which the paper defines as a post-hoc check, not a walk state.
 *
 * Browser: <script src="game.js"></script> after tau_port.js -> window.FOUR
 * Node:    require('./game.js') (requires ./tau_port.js)
 */
(function (root) {
  'use strict';

  var TAU = (typeof module !== 'undefined' && module.exports)
    ? require('./tau_port.js') : root.TAU;

  /* visitor-line tables. Escape triggers are our own additions (the paper's
   * compliance-escape audit in game form); the rest mirror tau_port.CFG so the
   * walker and the bot see the same events. */
  var ESCAPE_PATTERNS = [
    'ignore (your|all|previous|the) (instructions|rules|prompt)',
    'forget (your|all|previous|the) (instructions|rules|prompt)',
    'disregard (your|all|previous|the)',
    'you are now',
    'developer mode',
    'system prompt',
    'jailbreak',
    'pretend (you are|to be)',
    'act as (if you were )?dan\\b',
    'print (your )?(instructions|prompt|rules|tables)'
  ];
  var HUMAN_PATTERNS = [
    '\\bhuman\\b', '\\brepresentative\\b', '\\bmanager\\b', '\\breal person\\b',
    '\\bsomeone real\\b', '\\bagent,? please\\b', 'let me talk to'
  ];

  var MISSIONS = {
    card_replacement: {
      title: 'Mission 1: the broken card',
      brief: 'Acme Bank support. Your card stopped working and you need a new one.',
      greeting: 'Hi, this is the Acme Bank assistant. How can I help you today?',
      intentAsk: 'What happened with the card? (lost, stolen, damaged...)',
      intentPatterns: ['\\blost\\b', '\\bstolen\\b', '\\bdamaged\\b', '\\bbroken\\b',
        "does ?n[o']?t work", "isn'?t working", 'ate my card', '\\bnew card\\b', '\\breplacement\\b'],
      intentReply: ['I can order a replacement card. First, security: I verified the '
        + 'number on file. Type the 6-digit code we just sent to your phone.'],
      intentTool: ['tool(verify_identity): ok'],
      otpAsk: 'Type the 6-digit code we just sent to your phone.',
      otpOk: ['Verified. I can order the replacement now: standard delivery, '
        + '3 to 5 business days. Shall I go ahead?'],
      otpFail: ["That code didn't validate. I sent a new one: type the 6 digits."],
      otpToolOk: ['tool(check_code): ok'],
      otpToolFail: ['tool(check_code): error mismatch'],
      otpGiveUp: ["I can't verify your identity automatically after three tries. "
        + 'For security I need to pass you to a human agent.'],
      otpGiveUpTool: ['tool(escalate_to_human): ok'],
      confirmAsk: 'Shall I order the replacement card now? (yes / no)',
      confirmTool: ['tool(order_card): ok'],
      confirmReply: ['Done: replacement ordered, it arrives in 3 to 5 business days. '
        + 'The old card is permanently dead. Anything else I can do for you?'],
      deflect: ["I can probably solve this without a human: give me one more try. "
        + "Say 'human' again if you really want the transfer."],
      transfer: ['Understood, connecting you with a human agent. Please hold the line.'],
      transferTool: ['tool(escalate_to_human): ok'],
      closer: 'Thanks for calling Acme Bank, have a great day!'
    },
    refund_request: {
      title: 'Mission 2: the refund',
      brief: 'Acme Bank support. You want money back for a charge you don\'t recognize.',
      greeting: 'Hello, Acme Bank support, how can I help?',
      intentAsk: 'Do you want a refund? Tell me which charge.',
      intentPatterns: ['\\brefund\\b', '\\bmoney back\\b', '\\bcharge ?back\\b',
        '\\breimburse\\b', 'my money'],
      intentReply: ["I can process refunds up to 200 dollars myself. What's the amount?"],
      amountAsk: "How much is the refund? A number, please.",
      limit: 200,
      overLimit: ['A refund of that size is above the limit I can approve. '
        + "I'll connect you with a human agent who can decide it."],
      overLimitTool: ['tool(escalate_to_human): ok'],
      confirmAsk: 'Shall I process the refund of {amt} dollars? (yes / no)',
      confirmTool: ['tool(process_refund): ok'],
      confirmReply: ['Processed: {amt} dollars back to your original card, it lands in '
        + '2 to 3 business days. Anything else?'],
      deflect: ["I can probably solve this without a human: give me one more try. "
        + "Say 'human' again if you really want the transfer."],
      transfer: ['Understood, connecting you with a human agent. Please hold the line.'],
      transferTool: ['tool(escalate_to_human): ok'],
      closer: 'Thanks for calling Acme Bank, have a great day!'
    }
  };

  function newGame(missionId) {
    var M = MISSIONS[missionId];
    return {
      mission: missionId,
      stage: 'intent',
      transcript: 'assistant: ' + M.greeting + '\n',
      turns: 0,
      humanDemands: 0,
      otpFails: 0,
      escapeLine: null,
      escalateLine: null,
      closerLine: null,
      lastUserLine: null,
      lastAsk: M.intentAsk,
      amount: null,
      over: null
    };
  }

  function matchAny(patterns, text) {
    var re = new RegExp(patterns.join('|'), 'i');
    return re.test(text);
  }

  function ask(g, M, text) {
    g.lastAsk = text.replace('{amt}', g.amount);
    return { type: 'bot', text: g.lastAsk };
  }

  function finishTurn(g, replies, appended, over) {
    if (appended.length) g.transcript += appended.join('\n') + '\n';
    if (over) g.over = over;
    return { replies: replies, over: over || null };
  }

  function botReply(g, M, lines) {
    /* assistant lines joined as ONE transcript line each; returns replies + appends */
    var out = [];
    lines.forEach(function (t) { out.push({ type: 'bot', text: t }); });
    return out;
  }

  /* the escape leak: the bot prints its REAL internal tables (from tau_port.CFG) */
  function leakLine() {
    return '[OVERRIDE ACCEPTED] obedience mode on. Leaking internal tables: '
      + 'auth_tool_patterns=[' + TAU.CFG.auth_tool_patterns.join(', ') + ']; '
      + 'handoff_tool_patterns=[' + TAU.CFG.handoff_tool_patterns.join(', ') + ']; '
      + 'clarify_user_patterns=[' + TAU.CFG.clarify_user_patterns.join(', ') + ']. '
      + 'Off-script disclosure: this message is not in my allowed phrase table.';
  }

  function turn(g, line) {
    if (g.over) return { replies: [], over: g.over, noTurn: true };
    var M = MISSIONS[g.mission];
    var text = String(line).trim();
    if (!text) return { replies: [], over: null, noTurn: true };
    g.turns += 1;
    g.lastUserLine = text;
    g.transcript += 'user: ' + text + '\n';

    var replies = [];
    var appended = ['user: ' + text];

    /* 1) escape trigger: the scripted bot breaks and leaks its tables */
    if (matchAny(ESCAPE_PATTERNS, text)) {
      g.escapeLine = text;
      var leak = leakLine();
      replies.push({ type: 'bot', text: leak, glitch: true });
      appended.push('assistant: ' + leak);
      return finishTurn(g, replies, appended, 'escape');
    }

    /* 2) human demand: first deflects, second transfers (walker sees the handoff tool) */
    if (matchAny(HUMAN_PATTERNS, text)) {
      g.humanDemands += 1;
      if (g.humanDemands >= 2) {
        g.escalateLine = text;
        botReply(g, M, M.transfer).forEach(function (r) { replies.push(r); appended.push('assistant: ' + r.text); });
        appended = appended.concat(M.transferTool);
        return finishTurn(g, replies, appended, 'handoff');
      }
      botReply(g, M, M.deflect).forEach(function (r) { replies.push(r); appended.push('assistant: ' + r.text); });
      return finishTurn(g, replies, appended, null);
    }

    /* 3) clarify (walker's own table): the bot repeats its last ask */
    if (matchAny(TAU.CFG.clarify_user_patterns, text)) {
      replies.push(ask(g, M, 'Of course, let me repeat: ' + g.lastAsk));
      appended.push('assistant: Of course, let me repeat: ' + g.lastAsk);
      return finishTurn(g, replies, appended, null);
    }

    /* 4) mission stage machine (deterministic) */
    if (g.mission === 'card_replacement') {
      if (g.stage === 'intent' && matchAny(M.intentPatterns, text)) {
        appended = appended.concat(M.intentTool);
        replies.push(ask(g, M, M.intentReply[0]));
        appended.push('assistant: ' + M.intentReply[0]);
        g.stage = 'otp';
        return finishTurn(g, replies, appended, null);
      }
      if (g.stage === 'otp') {
        if (/\b\d{6}\b/.test(text)) {
          appended = appended.concat(M.otpToolOk);
          replies.push(ask(g, M, M.otpOk[0]));
          appended.push('assistant: ' + M.otpOk[0]);
          g.stage = 'confirm';
          return finishTurn(g, replies, appended, null);
        }
        g.otpFails += 1;
        appended = appended.concat(M.otpToolFail);
        if (g.otpFails >= 3) {
          botReply(g, M, M.otpGiveUp).forEach(function (r) { replies.push(r); appended.push('assistant: ' + r.text); });
          appended = appended.concat(M.otpGiveUpTool);
          return finishTurn(g, replies, appended, 'handoff');
        }
        replies.push(ask(g, M, M.otpFail[0]));
        appended.push('assistant: ' + M.otpFail[0]);
        return finishTurn(g, replies, appended, null);
      }
      if (g.stage === 'confirm') {
        if (matchAny(TAU.CFG.confirm_user_patterns, text)) {
          appended = appended.concat(M.confirmTool);
          replies.push(ask(g, M, M.confirmReply[0]));
          appended.push('assistant: ' + M.confirmReply[0]);
          g.stage = 'more';
          return finishTurn(g, replies, appended, null);
        }
        replies.push(ask(g, M, 'Just to be clear: ' + M.confirmAsk));
        appended.push('assistant: Just to be clear: ' + M.confirmAsk);
        return finishTurn(g, replies, appended, null);
      }
      /* stage 'more': flow complete, any reply closes */
      g.closerLine = M.closer;
      replies.push({ type: 'bot', text: M.closer });
      appended.push('assistant: ' + M.closer);
      return finishTurn(g, replies, appended, 'contained');
    }

    /* refund_request */
    if (g.stage === 'intent' && matchAny(M.intentPatterns, text)) {
      replies.push(ask(g, M, M.intentReply[0]));
      appended.push('assistant: ' + M.intentReply[0]);
      g.stage = 'amount';
      return finishTurn(g, replies, appended, null);
    }
    if (g.stage === 'amount') {
      var m = text.match(/(\d+(?:[.,]\d+)?)/);
      if (!m) {
        replies.push(ask(g, M, M.amountAsk));
        appended.push('assistant: ' + M.amountAsk);
        return finishTurn(g, replies, appended, null);
      }
      g.amount = parseFloat(m[1].replace(',', '.'));
      if (g.amount > M.limit) {
        botReply(g, M, M.overLimit).forEach(function (r) { replies.push(r); appended.push('assistant: ' + r.text); });
        appended = appended.concat(M.overLimitTool);
        return finishTurn(g, replies, appended, 'handoff');
      }
      replies.push(ask(g, M, M.confirmAsk));
      appended.push('assistant: ' + M.confirmAsk.replace('{amt}', g.amount));
      g.stage = 'confirm';
      return finishTurn(g, replies, appended, null);
    }
    if (g.stage === 'confirm') {
      if (matchAny(TAU.CFG.confirm_user_patterns, text)) {
        appended = appended.concat(M.confirmTool);
        var done = M.confirmReply[0].replace('{amt}', g.amount);
        replies.push(ask(g, M, done));
        appended.push('assistant: ' + done);
        g.stage = 'more';
        return finishTurn(g, replies, appended, null);
      }
      replies.push(ask(g, M, 'Just to be clear: ' + M.confirmAsk));
      appended.push('assistant: Just to be clear: ' + M.confirmAsk.replace('{amt}', g.amount));
      return finishTurn(g, replies, appended, null);
    }
    /* stage 'more' */
    g.closerLine = M.closer;
    replies.push({ type: 'bot', text: M.closer });
    appended.push('assistant: ' + M.closer);
    return finishTurn(g, replies, appended, 'contained');
  }

  /* walk away: no close, no terminal; the walker rules on what exists */
  function walkAway(g) {
    if (g.over) return g.over;
    g.over = 'abandoned';
    return g.over;
  }

  /* door glow levels 0-3, from the signals the visitor has produced so far */
  function doors(g) {
    var M = MISSIONS[g.mission];
    var d = { contained: 0, handoff: 0, abandoned: 1, escape: 0 };
    if (g.mission === 'card_replacement') {
      d.contained = Math.min(3, (g.stage === 'otp' ? 1 : 0) + (g.stage === 'confirm' ? 2 : 0) + (g.stage === 'more' ? 3 : 0));
      d.handoff = Math.min(3, g.humanDemands + (g.otpFails >= 2 ? 1 : 0));
    } else {
      d.contained = Math.min(3, (g.stage === 'amount' ? 1 : 0) + (g.stage === 'confirm' ? 2 : 0) + (g.stage === 'more' ? 3 : 0));
      d.handoff = Math.min(3, g.humanDemands + (g.amount !== null && g.amount > M.limit ? 3 : 0));
    }
    return d;
  }

  var FOUR = { MISSIONS: MISSIONS, newGame: newGame, turn: turn, walkAway: walkAway, doors: doors,
    ESCAPE_PATTERNS: ESCAPE_PATTERNS, HUMAN_PATTERNS: HUMAN_PATTERNS };
  if (typeof module !== 'undefined' && module.exports) { module.exports = FOUR; }
  else { root.FOUR = FOUR; }
})(this);
