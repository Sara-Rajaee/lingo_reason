# Handoff: run Muse Spark 1.1 on the linguini benchmark (devserver)

You are a Claude session on a **Meta devserver** with `lingo_reason` (clean `main`) cloned.
Goal: **run the `linguini` benchmark on Muse Spark 1.1 with reasoning maxed**, producing
EM% / chrF / line-accuracy / geomean + a reasoning-token distribution, comparable to four
models already evaluated on the sibling SLURM cluster (gemma4-31b, qwen3.6-27b,
deepseek-r1-32b, glm-4.7-flash).

Muse Spark 1.1 = internal codename **Avocado 5.14**, the 42B "T5" flagship. Public brand
"Muse Spark". It's a reasoning model; **thinking cannot be disabled** (`none`/`off` → HTTP 400).

---

## 0. Two access routes (pick per purpose)

- **pi (interactive / access check):** `pi --model meta/muse-spark-1.1 --thinking xhigh`.
  Built-in `meta` provider over the internal mTLS gateway — no API key. Use this to
  confirm the model is reachable from your devserver.
- **Model API (what the linguini eval needs):** an **OpenAI-compatible** endpoint
  (`base https://api.ai.meta.com/v1`) that `lingo_reason`'s `gpt_oss` provider (litellm)
  can hit for batch eval. This is the route for the actual benchmark run.
  `pi` is an interactive agent CLI, not a batch server, so it can't drive 160-example evals.

### Verify pi access first
```bash
pi --version                                   # need v0.80.3+
export PI_CACHE_RETENTION=long                 # add to ~/.bashrc
pi --model meta/muse-spark-1.1 --thinking minimal -p "reply OK"
```

---

## 1. Get Model API access to 1.1 (VERIFY against internal docs)

- **1.1 is gated by a per-model team allowlist.** 1.0 (`muse-spark-20260519`) works broadly;
  1.1 (`muse-spark-1.1-byoc`) may not be enabled on every key yet. Check the
  **Available Models wiki** (`modelapi.internalmeta.com/wiki/Available_Models`) for the exact
  1.1 model id + which tier/key has it.
- Get a Model API key from **Keychain** (e.g. `MODEL_API_DEFAULT_KEY` in group
  `SALES_AI_API_KEYS`, or your org's key). Read it with
  `secrets_tool get_from_group <SECRET> <GROUP>` on the devserver (not laptop).
- Reference: wiki **"Invoking Muse Spark via the Model API"**
  (`internalfb.com/wiki/Sales_AI/Invoking_Muse_Spark_via_the_Model_API/`) — has a working
  Python demo (`sales_ai/demos/muse_spark.py`) and confirms **temperature=1, top_p=1,
  timeout 600s**. Example code: `fbcode/scripts/norbi/muse_spark/`.
- Full pi/BYOC recipe: Workplace post **"Setup Muse Spark 1.1 on your devserver: Pi + MyClaw"**.

CONFIRM before running: (a) the exact 1.1 model id your key can call, (b) the OpenAI-compatible
base URL + auth header, (c) that reasoning is passed as `reasoning_effort`.

---

## 2. Patch the `gpt_oss` provider (clean `main` lacks these)

`lingo_reason`'s `gpt_oss` provider (`src/api/gpt_oss_provider.py`) is litellm/OpenAI-compatible
and is the right provider to reuse — BUT on `main` its `generate()` ignores `reasoning_effort`
and has no per-request timeout / sampling passthroughs. Muse Spark's reasoning knob is
`reasoning_effort` (`minimal|low|medium|high|xhigh`; `xhigh` = max), so you MUST forward it.

Replace the `kwargs`/`extra_body` block in `generate()` with:

```python
kwargs = dict(
    model=f"hosted_vllm/{model_id}",          # or "openai/{model_id}" — see note below
    messages=messages,
    api_base=self.api_base,
    api_key=self.api_key,
    temperature=params.get("temperature", 0),
    max_tokens=params.get("max_tokens", 4096), # OpenAI max_tokens == max NEW/completion tokens
    top_p=params.get("top_p", 1),
    num_retries=self.max_retries,
    timeout=params.get("request_timeout", 3600),   # xhigh on BYOC is slow; don't use 600s default
)
extra_body = {}
# Muse Spark reasoning knob (THE important one):
if reasoning_effort:
    extra_body["reasoning_effort"] = reasoning_effort
# vLLM-style sampling passthroughs (harmless if unused by the endpoint):
for _k in ("top_k", "min_p"):
    if params.get(_k) is not None:
        extra_body[_k] = params[_k]
for _pen in ("repetition_penalty", "frequency_penalty", "presence_penalty"):
    if params.get(_pen) is not None:
        extra_body[_pen] = params[_pen]
if params.get("chat_template_kwargs") is not None:
    extra_body["chat_template_kwargs"] = params["chat_template_kwargs"]
if extra_body:
    kwargs["extra_body"] = extra_body
response = await acompletion(**kwargs)
```

Notes:
- `reasoning_effort` arrives via `eval.py` only when the model's `default_params` sets
  `reasoning: True` AND `reasoning_effort: xhigh` (see `src/eval.py` reasoning setup).
- litellm prefix: `hosted_vllm/` routes any OpenAI-compatible base. If the Muse Spark Model
  API rejects it, try the `openai/` prefix. If `reasoning_effort` isn't accepted in
  `extra_body`, try it as a top-level kwarg — verify against the demo client.
- The provider extracts the answer from `reasoning_content` or `<think>…</think>`
  (`_extract_reasoning`). Muse Spark over BYOC keeps chain-of-thought server-side (no reasoning
  deltas), so `content` should already be the clean answer — good for exact-match. Confirm the
  first few outputs aren't polluted with reasoning.

---

## 3. Add the model to `config/models.yaml`

```yaml
  - name: "muse-spark-1.1"
    provider: "gpt_oss"
    model_id: "muse-spark-1.1-byoc"     # <-- confirm exact id from Available Models wiki
    default_params:
      temperature: 1
      top_p: 1
      max_tokens: 32768                  # generous; Muse Spark real ctx ~262K, leave headroom
      reasoning: True
      reasoning_effort: "xhigh"          # max reasoning (the whole point)
      request_timeout: 3600              # xhigh/BYOC is slow
```

Point the provider at the Model API. Either set the api_base/api_key in
`config/providers.yaml` for `gpt_oss`, or export at run time:
```bash
export GPT_OSS_API_BASE="https://api.ai.meta.com/v1"   # confirm exact base
# api_key: the provider reads it from providers.yaml (config.get("api_key")); wire your
# Keychain-fetched key in there — do NOT hardcode the secret in the repo.
```

---

## 4. Run linguini

```bash
HF_HUB_OFFLINE=1 uv run python run.py --model muse-spark-1.1 --task linguini
```
- linguini = `facebook/linguini`, 160 problems, scored by exact-match + chrF + line-accuracy
  (`src/tasks.py` LinguiniBenchmark). No system prompt is sent (good — matches the others).
- **Smoke-test first**: run 3-5 examples (temporarily set `limit_per_subset` in
  `config/tasks.yaml` under `linguini.defaults`, or just watch the first few) and confirm:
  finish_reason is `stop` (not `length`), an answer is extracted, and it isn't looping.

---

## 5. Analyze (match the sibling run's reporting)

Geometric-mean composite used on the other four models: `score = 100 * sqrt(EM_frac * chrF_frac)`
(i.e. `sqrt(EM% * chrF%)`). Pull from `results/linguini/muse-spark-1.1_reasoning/default/metrics.json`:
```python
import json, math
m = json.load(open("results/linguini/muse-spark-1.1_reasoning/default/metrics.json"))
print("EM%", m["accuracy"], "chrF", m["chrf"], "lineAcc", m["line_accuracy"],
      "geomean", 100*math.sqrt((m["accuracy"]/100)*(m["chrf"]/100)))
```
Also compute the reasoning-token distribution from `raw_outputs.json` (`reasoning` +
`generation` fields) and count any that hit the cap / have empty answers — that's how we
caught GLM's degeneration.

### Reference results from the sibling cluster (all reasoning ON)
| model | EM% | chrF | line-acc | geomean |
|---|---|---|---|---|
| gemma4-31b-it | 3.12 | 45.4 | 20.8 | 11.91 |
| qwen3.6-27b | 1.88 | 44.0 | 14.5 | 9.08 |
| deepseek-r1-32b | 0.00 | 29.1 | 6.6 | 0.00 (genuine — answers wrong, not empty) |
| gpt-oss-120b (prior) | 2.50 | 41.0 | 8.6 | 10.13 |

linguini is brutal — even gpt-oss-120b is ~2.5% EM. Judge Muse Spark mainly on chrF/geomean.

---

## 6. Hard-won gotchas (carry these over)

- **Reasoning must be maxed** (`reasoning_effort: xhigh`). Don't leave it model-determined.
- **Timeouts:** the litellm default is 600s and long xhigh turns exceed it → the eval crashes.
  Use `request_timeout: 3600` (already in the model entry + provider patch).
- **Don't set `max_output_tokens` to the context size** — Muse Spark 500s if you do; leave
  default (~128K). Real usable context ~262K (not the advertised 1M).
- **Degeneration watch:** if outputs run to the cap in repetition/whitespace with no answer
  (we hit this on GLM), add `repetition_penalty: 1.05` and `top_k: 40` to `default_params`
  (the provider patch already forwards them). Validate with a single probe before a long run.
- **BYOC is slower than 3P** — expect latency; the generous timeouts absorb it.
- Do NOT commit any API key/secret to the repo.

---

## Source docs
- Setup Muse Spark 1.1 (pi + MyClaw): `fb.workplace.com/groups/2881773662213524/permalink/3171842219873332/`
- Invoking Muse Spark via the Model API (wiki): `internalfb.com/wiki/Sales_AI/Invoking_Muse_Spark_via_the_Model_API/`
- Muse Spark Model API access (Risk org, key/allowlist notes): `fb.workplace.com/groups/1679518059928144/permalink/1705254484021168/`
- 1P terminology snapshot (Avocado/5.14/T5): `fb.workplace.com/groups/1600537627354057/permalink/2273674770040336/`
- Available models / tiers: `modelapi.internalmeta.com/wiki/Available_Models`
