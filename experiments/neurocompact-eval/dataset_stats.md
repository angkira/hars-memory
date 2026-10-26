# chains.jsonl dataset stats

chains: 10 | spans: 30 | QA items: 70 | items/chain: 7-7

## Spans per span_idx (chars / est tokens)

| span_idx | lag | n | chars mean | chars total | est tok mean | est tok total |
|---|---|---|---|---|---|---|
| 1 | 3 | 10 | 25187 | 251866 | 10494 | 104939 |
| 2 | 2 | 10 | 24242 | 242415 | 10100 | 101001 |
| 3 | 1 | 10 | 24126 | 241265 | 10052 | 100522 |

## QA items per lag x slice

| lag | slice | n |
|---|---|---|
| 0 | NONE | 10 |
| 1 | HEAD | 10 |
| 1 | TAIL | 10 |
| 2 | HEAD | 10 |
| 2 | TAIL | 10 |
| 3 | HEAD | 10 |
| 3 | TAIL | 10 |

## Split

- answerable: 60 (85.7%)
- unanswerable (must_abstain): 10 (14.3%)
- evidence words/item: min 10, mean 20.4, max 42

## Gold-anchor types

| type | n | share |
|---|---|---|
| identifier | 31 | 51.7% |
| path | 11 | 18.3% |
| number | 12 | 20.0% |
| version | 1 | 1.7% |
| other | 5 | 8.3% |

### Anchor examples per type

- `identifier`: importlib.metadata (chain_qq8Vqbow_c0)
- `identifier`: hars.execution.monitor.feedback (chain_qq8Vqbow_c0)
- `identifier`: create_cortex_agent (chain_qq8Vqbow_c0)
- `identifier`: DEFAULT_MODEL_SEQUENTIAL (chain_y062eGZR_c0)
- `path`: .plans/2026-08-11_argentic_full_runtime_poc.md (chain_qq8Vqbow_c0)
- `path`: openrouter/deepseek/deepseek-r1-distill-llama-70b (chain_y062eGZR_c0)
- `path`: tmp/connect_err.txt (chain_y062eGZR_c0)
- `path`: fix/56-grpc-transport-type (chain_gcKDB2Q7_c0)
- `number`: 1786274314 (chain_y062eGZR_c0)
- `number`: 120000 (chain_gcKDB2Q7_c0)
- `number`: 332 (chain_GFkz8FeB_c0)
- `number`: 164.9 (chain_GFkz8FeB_c0)
- `version`: 1.2.1 (chain_qq8Vqbow_c0)
- `other`: command_id=%s (chain_qq8Vqbow_c0)
- `other`: 14d (chain_gcKDB2Q7_c0)
- `other`: Chat Replay (chain_OAITa4mS_c0)
- `other`: mean_reward=-164.9 (chain_GFkz8FeB_c1)
