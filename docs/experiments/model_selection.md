# Phase 3A-next candidate model pairs

Official sources checked 2026-11-22. Public documentation does not verify account access, region, effective billing tier, account settings, or current provider-side spend cap.

| Pair | Exact small / strong IDs | Public standard input/output price per 1M tokens | Limits and current status | Assessment |
|---|---|---|---|---|
| OpenAI GPT-4.1 | `openai/gpt-4.1-mini-2025-04-14` / `openai/gpt-4.1-2025-04-14` | $0.40/$1.60; $2/$8 (GPT-4.1 Mini page shows $0.40/$1.60; GPT-4.1 current page has its current listed rate) | Both dated model pages identify deprecation. Public docs list context/output limits, but deprecated identifiers are unsuitable for a new frozen pilot. | Reject for the pilot. |
| Anthropic Claude 5.5 | `anthropic/claude-haiku-5-5` / `anthropic/claude-sonnet-5-5` | Haiku $0.10/$0.50 up to 100k input and higher above; Sonnet $2/$10 | Current aliases. Haiku/Sonnet publish large context/output ceilings; Anthropic token-count API estimates input. Sonnet thinking tokens count as output billing. Account access/caps unverified. | Viable secondary candidate after exact account and LiteLLM verification. |
| Gemini | `gemini/gemini-3.5-flash-lite` / `gemini/gemini-3.8-flash` | $0.30/$2.50; $0.75/$3.75 through 2026-12-31; new rates from 2027-01-01 are higher. | Current stable IDs; 1,048,576 input and 65,536 output token limits. Thinking tokens billed with output. `countTokens` counts input, but isn't a full billing upper bound. Project caps are experimental with around 10 minutes of overage risk; tier caps are shared at billing-account scope. | Recommended documentation candidate pair: current IDs, shared provider and clear public token/price documentation. **Not executable.** |

The former catalog candidate `gemini/gemini-3.5-flash` is no longer used as the proposed strong model. The OpenAI dated candidates are retained only as historical catalog entries and are marked deprecated. The project’s versioned capability registry authorizes none of these models for paid execution. No arbitrary “latest” alias is used.

## Provider and adapter references

- OpenAI: [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini), [GPT-4.1](https://developers.openai.com/api/docs/models/gpt-4.1), [spend limits](https://help.openai.com/en/articles/9186755-managing-projects-in-the-api-platform).
- Anthropic: [model overview](https://platform.claude.com/docs/en/models/overview), [pricing](https://platform.claude.com/docs/en/about-claude/pricing), [token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting), [spend limits](https://platform.claude.com/docs/en/api/rate-limits).
- Gemini: [Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite), [3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash), [pricing](https://ai.google.dev/gemini-api/docs/pricing), [billing](https://ai.google.dev/gemini-api/docs/billing), [token counting](https://ai.google.dev/gemini-api/docs/tokens).
- LiteLLM provider pages: [Gemini](https://docs.litellm.ai/docs/providers/gemini), [Anthropic](https://docs.litellm.ai/docs/providers/anthropic), [OpenAI](https://docs.litellm.ai/docs/providers/openai). Documentation indicates adapter availability, not a tested live mapping or account access.

## Recommendation and limitation

The Gemini pair is recommended for a future protocol pilot based on documented stable identifiers, same-provider pairing, published token limits and transparent pricing, not because it is cheapest or proven best at math. Its exact LiteLLM request serialization/accounting has not been verified. Public project caps explicitly allow delayed-processing overages; a tier cap is shared across all linked projects. Since no exact provider billing maximum and suitably scoped instantaneous external limit have been established, local budget arithmetic cannot make the pilot safe. Read [provider verification](provider_verification.md) and [spend safety design](spend_safety_design.md).
