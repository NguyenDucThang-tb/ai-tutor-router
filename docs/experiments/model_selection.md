# Phase 3A candidate model pairs

Prices below are public list prices retrieved **2026-11-21**. They are USD per 1 million uncached text tokens unless a source says otherwise. The official pages used here did not provide a specific effective date for every displayed rate; the effective date is therefore **not published/verified** in this snapshot. They are research references, not account quotes or actual billing. None of the models was invoked. Region/account availability, rate limits, quota, exact provider-side deployment revision, and research-account terms must be checked by the owner before any approval.

| Candidate pair | Small model | Strong model | Input / output USD per 1M | Output/context limits | Rate limits / availability | LiteLLM |
|---|---|---|---|---|---|---|
| OpenAI GPT-4.1 | `openai/gpt-4.1-mini-2025-04-14` | `openai/gpt-4.1-2025-04-14` | $0.40 / $1.60; $1.50 / $6.00 | Not established from retrieved pricing source; verify current model docs/account | Account and region unknown; quota is account-specific | Provider listed; exact request path untested |
| Anthropic Claude 5.5 | `anthropic/claude-haiku-5-5` | `anthropic/claude-sonnet-5-5` | $0.10 / $0.50; $2.00 / $10.00 | Not captured here; Haiku rates change above 100k prompt tokens | Account/region/rate limits unknown | Provider listed; exact request path untested |
| Gemini 3.5 | `gemini/gemini-3.5-flash-lite` | `gemini/gemini-3.5-flash` | $0.30 / $2.50; $1.50 / $9.00 | Not captured here; thinking tokens count as output for Flash | AI Studio paid-tier availability unknown; free tier excluded | Provider listed; exact request path untested |
| Open-weight alternative | Qwen/Qwen3-4B | Qwen/Qwen3-32B | Hosted price unknown; self-hosted cost depends on hardware, utilization, quantization, and serving configuration | Verify model card and selected deployment | Weights published; local hardware/endpoint availability unknown | vLLM provider documented; endpoint not deployed/tested |

Official sources:

- OpenAI [API pricing](https://platform.openai.com/pricing/) and [GPT-4.1 mini model docs](https://platform.openai.com/docs/models/gpt-4.1-mini), [GPT-4.1 model docs](https://platform.openai.com/docs/models/gpt-4.1). The exact dated snapshot's current account availability was not queried.
- Anthropic [model pricing](https://platform.claude.com/docs/en/about-claude/pricing) and [model lifecycle page](https://platform.claude.com/docs/en/about-claude/model-deprecations).
- Google [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [model docs](https://ai.google.dev/gemini-api/docs/models). The paid standard tier is distinct from free-tier data-use terms.
- LiteLLM [provider list](https://docs.litellm.ai/docs/providers) and [vLLM provider docs](https://docs.litellm.ai/docs/providers/vllm).
- Qwen [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B), [Qwen3-32B model card](https://huggingface.co/Qwen/Qwen3-32B), and [official Qwen3 release page](https://github.com/QwenLM/Qwen3).

## Selection recommendation

For a compatibility pilot, consider the OpenAI GPT-4.1 pair only after verifying account access, endpoint IDs, terms, current price, and quota. This is not a quality recommendation and no performance result is claimed. Haiku/Sonnet has a larger listed-rate differential. Gemini rates are transparent, but the cited free tier has different data-use terms and must not be used for research collection. Qwen3 is a candidate when reproducible self-hosted hardware is available; without a concrete deployment there is no defensible API price.

No recommendation is currently executable: no provider call, capability check, regional availability check, output-limit verification, or paid-tier access check occurred. Provider-managed `latest` aliases are excluded because they are mutable.
