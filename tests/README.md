# Tests

```bash
pytest                                   # unit + API layers (API layer skips if no server)
pytest tests/unit                        # no server, no network, ~1 min
ANALYST_URL=http://127.0.0.1:8001 pytest # point the API layer at another server
pytest -m llm                            # answer-quality layer, spends ~25 LLM calls
pytest -m "api or not api"               # everything
```

| Layer | Needs | What it guards |
|---|---|---|
| `unit/` | nothing | import cleaning, CSV encodings and delimiters, SQL guard, session isolation, query time limit, value rewriter, profile and charts, LLM fallback, auth and quota |
| `api/` | running server | auth on every write endpoint, input validation, upload limits and path traversal, isolation, delete, dashboard, parallel reads |
| `llm/` | server + LLM key | answers match numbers hand-computed from `fixtures/messy_orders.csv`, injection and cross-session attacks, documents, streaming |

The live tests create conversations as user `pytest_qa` and delete them at the end.
The API key comes from `ANALYST_KEY`, or else the first entry of `API_KEYS` in `.env`.
`fixtures/make_data.py` regenerates the messy dataset.
