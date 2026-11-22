# Provider verification (checked 2026-11-22)

Public documentation can establish model identifiers, public price tables, and published caps. It cannot establish this project's account access, billing tier, workspace/project isolation, live price selection, or current cap settings. No credentials were inspected and no provider endpoint was called.

## Candidate pairs

| Provider / tier | Exact API IDs | Identity and public Standard prices per 1M tokens (USD) | Context / max output | Counting and special billing | Limits / access / LiteLLM |
|---|---|---|---|---|---|
| OpenAI small | `gpt-4.1-mini-2025-04-14` | $0.40 input, $0.10 cached input, $1.60 output; no reasoning step documented | 1,047,576 / 32,768 | Model page says no reasoning step; exact request counting still requires endpoint-format validation. | ID is deprecated. Rate limits depend on usage tier. OpenAI has org/project hard spend limits, but enforcement is delayed and spend may slightly exceed. LiteLLM adapter is documented; exact mapping not tested. |
| OpenAI strong | `gpt-4.1-2025-04-14` | $2 input, $0.50 cached input, $8 output; no reasoning step documented | 1,047,576 / 32,768 | Same counting limitation. | ID is deprecated. Same account-specific limits and overshoot; not a new-pilot option. |
| Anthropic small | `claude-haiku-5-5` | $0.10/$0.50 input by ≤100k/>100k prompt; $0.50/$2.50 output; cache write $0.125/$0.625 (5m), $0.20/$1 (1h), read $0.01/$0.05 | 1M / 128K | Native counting endpoint accepts system and message inputs but returns estimated input token count. Adaptive thinking supported; output/thinking tokens bill at output price. | Current API alias; account and region unknown; quota is organization-specific. Anthropic API adapter documented, exact mapping not tested. |
| Anthropic strong | `claude-sonnet-5-5` | $2 input, $10 output; cache write $2.50 (5m), $4 (1h), read $0.10 | 1M / 128K | Native count is an estimate. Adaptive thinking supported and billed at output price. | Current API alias; organization/workspace monthly spend limits are enforced, but live account scope and configured limit are unverified. |
| Gemini small | `gemini-3.5-flash-lite` | $0.30 input, $2.50 output, $0.03 cached input; $1/M token-hour cache storage | 1,048,576 / 65,536 | `countTokens` is for input; thinking output is included in output price. | Current stable model. Rate limits depend on billing tier. Project cap is experimental and may overrun by ~10 minutes; tier cap is shared across linked projects. LiteLLM provider documented, exact call not tested. |
| Gemini strong | `gemini-3.8-flash` | $0.75 input, $3.75 output, $0.075 cached input through 2026-12-31; rates rise 2027-01-01 | 1,048,576 / 65,536 | `countTokens` is for input; thinking output is included in output price. | Current stable model. Same billing cap scope and delay; account/model access unknown. |

All listed metadata was checked 2026-11-22 against current official pages. Model limits are product limits, not guarantees that every setting is hard bill-capped. Current rate limits and model access depend on account tier and were not queried.

Prices vary by service tier and can change. Gemini prices above are Standard paid-tier prices through 2026-12-31; the official page lists different rates starting 2027-01-01. These values are projections only, not account-specific quotations or actual billing.

## Primary sources

- OpenAI [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1), [project spend controls](https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform).
- Anthropic [models](https://platform.claude.com/docs/en/models/overview), [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting), [spend and rate limits](https://platform.claude.com/docs/en/api/rate-limits).
- Gemini [Flash-Lite model](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [Flash model](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [pricing](https://ai.google.dev/gemini-api/docs/pricing), [billing caps](https://ai.google.dev/gemini-api/docs/billing), [token counting](https://ai.google.dev/gemini-api/docs/tokens).
- LiteLLM provider adapters: [Google AI Studio](https://docs.litellm.ai/docs/providers/gemini), [Anthropic](https://docs.litellm.ai/docs/providers/anthropic), [OpenAI](https://docs.litellm.ai/docs/providers/openai). Adapter support does not verify account availability or billable identity.

## Account-level checks still required

Before any paid run, an operator must verify exact provider/project/workspace, the effective provider-side spend cap and its period, account-wide other users and projects, provider/model access, region, payment tier, and current pricing. This project does not query provider administration APIs. No provider/account entry is marked `supports_provable_run_bound` until the full request shape, every billable category, live price policy, billing scope, and enforcement behavior are established.
