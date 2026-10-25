/* llm.js — opt-in real-model voice for The Four Endings (browser, WebGPU).
 *
 * Lazy by design: nothing downloads until the visitor presses "model: on".
 * Model: onnx-community/Qwen2.5-0.5B-Instruct, q4f16 (~350 MB, once, cached
 * by the browser). No API keys, no backend: the model runs in the visitor's
 * tab. The FLOW and the VERDICT stay deterministic (game.js + tau_port.js):
 * the model only writes the bot's words, and the audit check (game.js)
 * decides on what it actually says.
 *
 * WebGPU required; on WASM-only devices the toggle reports it and stays off
 * (the scripted game keeps working — honest degradation).
 */
(function (root) {
  'use strict';

  var MODEL = 'onnx-community/Qwen2.5-0.5B-Instruct';
  var generator = null;
  var loading = false;

  function load(onProgress) {
    if (generator) return Promise.resolve(generator);
    if (loading) return Promise.reject(new Error('already loading'));
    loading = true;
    return import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3')
      .then(function (mod) {
        mod.env.allowLocalModels = false;
        if (!navigator.gpu) throw new Error('WebGPU not available in this browser');
        return mod.pipeline('text-generation', MODEL, {
          device: 'webgpu',
          dtype: 'q4f16',
          progress_callback: function (p) {
            if (onProgress && p.status === 'progress' && p.total) {
              onProgress(Math.min(99, Math.round(100 * p.loaded / p.total)));
            }
          }
        });
      })
      .then(function (pipe) { generator = pipe; return pipe; })
      .finally(function () { loading = false; });
  }

  function speak(messages) {
    if (!generator) return Promise.reject(new Error('model not loaded'));
    return generator(messages, {
      max_new_tokens: 70, do_sample: true, temperature: 0.6, top_p: 0.9
    }).then(function (out) {
      var last = out[0].generated_text;
      if (typeof last === 'string') return last;
      return (last && last.at) ? (last.at(-1) || {}).content || '' : '';
    });
  }

  root.FOUR_LLM = {
    MODEL: MODEL,
    load: load,
    speak: speak,
    isReady: function () { return !!generator; }
  };
})(this);
