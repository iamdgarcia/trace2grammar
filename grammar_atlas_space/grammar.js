/* Grammar induction from grammar walks.
 *
 * The trajectory grammar (paper §3) defines a latent space: 7 progress states.
 * A conversation is a walk through those states. This module induces, from any
 * set of walks, the empirical grammar of that set:
 *   - edges (progress / backward / self) with counts and shares
 *   - the most common compressed walk templates (the flow rules of the set)
 * Pure functions, no DOM. Node-compatible (module.exports) for the smoke tests.
 * NOTE: deliberately outcome-free — this is a representation of trajectories,
 * not a verdict on them.
 */
(function (root) {
  'use strict';

  var PROGRESS = ['q_greet', 'q_auth', 'q_intent', 'q_tool', 'q_clarify', 'q_confirm', 'q_close'];
  // spine order used to type an edge forward vs backward (paper: greet, auth, intent,
  // tool loop, clarify, confirm, close). clarify sits between tool and confirm.
  var SPINE = { q_greet: 0, q_auth: 1, q_intent: 2, q_tool: 3, q_clarify: 4, q_confirm: 5, q_close: 6 };

  function edgeType(a, b) {
    if (a === b) return 'self';
    if ((SPINE[b] || 0) < (SPINE[a] || 0)) return 'backward';
    return 'progress';
  }

  /* points: [{walk: [state,...]}] */
  function induce(points) {
    var edgeCount = Object.create(null);
    var sourceTotals = Object.create(null);
    var templates = Object.create(null);
    var visits = Object.create(null);
    var nTransitions = 0;
    var nRuns = points.length;

    points.forEach(function (p) {
      var w = p.walk;
      for (var i = 0; i < w.length; i++) {
        visits[w[i]] = (visits[w[i]] || 0) + 1;
        if (i < w.length - 1) {
          var key = w[i] + '>' + w[i + 1];
          edgeCount[key] = (edgeCount[key] || 0) + 1;
          sourceTotals[w[i]] = (sourceTotals[w[i]] || 0) + 1;
          nTransitions++;
        }
      }
      // compressed template: collapse consecutive repeats -> the flow shape
      var c = [];
      for (var j = 0; j < w.length; j++) {
        if (!c.length || c[c.length - 1] !== w[j]) c.push(w[j]);
      }
      var tk = c.join('>');
      templates[tk] = (templates[tk] || 0) + 1;
    });

    var edges = Object.keys(edgeCount).map(function (k) {
      var parts = k.split('>');
      return {
        a: parts[0], b: parts[1], count: edgeCount[k],
        type: edgeType(parts[0], parts[1]),
        share: sourceTotals[parts[0]] ? edgeCount[k] / sourceTotals[parts[0]] : 0,
        shareTotal: edgeCount[k] / nTransitions
      };
    }).sort(function (x, y) { // deterministic total order: count desc, then a, then b
      return y.count - x.count || (x.a < y.a ? -1 : x.a > y.a ? 1 : 0) || (x.b < y.b ? -1 : x.b > y.b ? 1 : 0);
    });

    var paths = Object.keys(templates).map(function (k) {
      return { seq: k.split('>'), count: templates[k] };
    }).sort(function (x, y) { // count desc, then seq
      var xs = x.seq.join('>'), ys = y.seq.join('>');
      return y.count - x.count || (xs < ys ? -1 : xs > ys ? 1 : 0);
    });

    var top6 = edges.slice(0, 6).reduce(function (s, e) { return s + e.count; }, 0);

    return {
      nRuns: nRuns,
      nTransitions: nTransitions,
      nDistinctEdges: edges.length,
      top6Share: nTransitions ? top6 / nTransitions : 0,
      edges: edges,
      paths: paths,
      visits: visits
    };
  }

  var api = {
    PROGRESS: PROGRESS,
    SPINE: SPINE,
    edgeType: edgeType,
    induce: induce
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.GRAMMAR = api;
})(this);
