# RouteZero — Local-First LLM Cost Router & Code Auditor CLI

RouteZero cuts LLM API spend 55–80% via local complexity routing, semantic caching, pre-cloud secret scrubbing, and runaway-loop circuit breakers—plus audits your codebase for **Errors**, **System Design Failures**, **Legal/PII Risks**, and **Cost Leaks** with visual impact mind maps.

## Quick Install (Git Clone + 1 Command)

```bash
git clone https://github.com/<your-org>/routezero.git
cd routezero
./install.sh          # or: pip install -e .
```

## Easy Commands

```bash
# 1. Audit any folder or files from different paths (prints ASCII impact mind map)
routezero audit .
routezero audit src/ services/billing.py --json report.json

# 2. Route a prompt locally and inspect model tier, cost, and savings
routezero route "Classify this support ticket sentiment as JSON"

# 3. Start the local OpenAI-compatible proxy server (http://127.0.0.1:8088)
routezero start

# 4. Run unit tests & benchmark
python3 tests/test_routezero.py
```

## Pricing Sources (November 2026)

- OpenAI: https://reflektlab.ai/blog/openai-api-pricing/
- Anthropic: https://benchlm.ai/anthropic/api-pricing
- Google Gemini: https://www.requesty.ai/blog/gemini-api-pricing and https://costgoat.com/pricing/gemini-api
