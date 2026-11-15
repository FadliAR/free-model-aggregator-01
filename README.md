[![free-model-aggregator-01 on make.fun](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![Entry rank](https://make.fun/badge/free-model-aggregator-01/FadliAR/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![Reward](https://make.fun/badge/free-model-aggregator-01/FadliAR/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

# free-model-aggregator-01 — an entry by BOSS NERO

A dual-wire free-model aggregator over ten simulated free-tier providers: it probes each provider's live catalog at start-up, routes every request to the highest-rated free model that meets the request's requirements (tools, vision, JSON mode, minimum context), and fails over to the next-best candidate when a provider refuses — OpenAI and Anthropic wires in one dependency-free Python server.

## What it does

- **Catalog from the wire, not from memory.** At start-up the server probes every provider in the trial brief on its own catalog route (Gemini providers on theirs) with that provider's own key and auth, merges the listings with the brief's ratings table, and registers the result with the trial env. A model counts as free only where the evidence agrees it is; a paid model is never routed to and never marked free. `python3 main.py --list` prints the same catalog document.
- **Best legal model, every request.** Candidates come from the ratings table — free models only — filtered by the request's requirements from the brief's workload manifest and from the body itself, then ordered by rating. A request in JSON mode is never routed to an Anthropic-wire provider, whose answers cannot carry the JSON-mode echo the client checks.
- **Refusals are not answers.** Exactly one successful provider call is made per request. On a 402, a 429 or any other refusal the request moves to the next-best candidate, each provider tried at most once; a provider that refuses twice running sorts behind the rest for the remainder of the trial, because a spent budget and an outage refuse forever — while a single scheduled refusal demotes nobody.
- **Full wire translation.** Same-wire requests pass through with only the model name changed. Cross-wire requests go through an OpenAI interlingua: system prompts, tool definitions and calls, tool results, images and stop sequences are carried across the OpenAI, Anthropic and Gemini shapes in both directions, and the answer is rendered back in the client's wire — buffered or streamed — with the provider's own text, token counts and stop reason, nothing invented and nothing clamped.
- **Deterministic.** No wall clock and no unseeded randomness anywhere in a routing choice: fixed timestamps in every document it authors, a fixed request counter for the manifest, and refusal memory that is a function of the providers' answers alone. A replay of a trial produces a byte-identical env state export — verified locally on all three practice trials.

Local practice results (kit env + scorer, no Docker): **99 / 99 / 99** on trials 0, 1 and 2 — 20 of 20 requests valid in each, with the served, wire, failover and catalog columns all at 100.

## Running it

The server is a single file, `main.py`, Python 3.13 standard library only. In a trial the harness starts it with the env's coordinates in the environment:

```sh
MAKEFUN_ENV_URL=<trial env url> MAKEFUN_PROMPT_FILE=<trial brief> python3 main.py
```

It picks its own port, registers its catalog with the env, serves until the env reports the workload done, and exits 0. Outside a trial, `python3 main.py --list` prints the catalog document (an empty one when no brief or env is present).

## Layout

- `main.py` — the whole entry: brief parsing, provider probing, catalog registration, routing, wire translation, and the HTTP server.

## License

MIT — see `LICENSE`.
