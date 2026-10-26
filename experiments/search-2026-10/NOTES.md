# Running notes (partial report; final in docs/search-research-2026-10-02.md)
Eval: 120 cases (gpt-6-luna generated, facts verified verbatim + inside a single indexed chunk), 49 docs, 42 ru / 78 en. Index: /tmp/eval-mig/after clone (4168 chunks).
Baseline (hybrid a=0.5, pool15, top5): file_hit@1 .633, hit@3 .850, hit@5 .875, fact@1 .625, fact@3 .808, fact@5 .858, MRR .729, chars@5 8674.
dense: fact@5 .692 / MRR .549; bm25: fact@5 .817 / MRR .702. hybrid beats dense strongly (CI excl 0), beats bm25 slightly (noise-level at @5).
By lang: ru fact@5 .762 vs en .910. By qtype: list weakest (.733 fact@5), identifier .871. Recall ceiling: fact absent from top-100 in 7/120 dense, 8/120 bm25.
## H2 fusion/tokenisation (offline)
- best alpha sweep: alpha .4 pool 50: fact_rr .759 vs .729 (d +.029 [-.001,+.056]), file_hit@1 +.050 [+.008,+.100]; 2-fold CV picks the same -> small, borderline. INCONCLUSIVE/weak.
- RRF variants: not better (see h2 results/ files).
- BM25 tokenisation: current identifier tokenizer ~= whole-run-only (.711) ~= stoplist (.715) vs current .702 (noise); plain \w+ words WORSE (.558*), char-4grams WORSE (.589*). k1=1.2,b=.75 hybrid fact_rr .738 (+.009 [0,.023]) noise. REFUTED (no gain from re-tokenising), current tokenizer is right.
- RRF (k20/k60, pool15-50): fact_rr -0.06* vs weighted fusion; weighted RRF no better. alpha .7 clearly worse (-0.068*). REFUTED for RRF; alpha .3-.4 +pool50 = +0.03 (noise-level).
## H5 query transformation (gpt-6-luna; NB first run had empty HyDE/rewrite outputs because reasoning tokens ate max_tokens=80/120; fixed to 1500 and re-ran)
- n=120, hybrid baseline fact@5 .858 / MRR .729. en_rewrite fact@5 .883 (+.025), MRR +.006; multi3 RRF fact@5 .883 (+.025) but file_hit@1 -.042, MRR -.009; HyDE variants MRR -0.02..0.00, hyde+raw fact@5 .900 (+.042, CI incl 0). All inside CI -> INCONCLUSIVE/REFUTED (no reliable gain).
- cost/latency per query: en_rewrite $0.00004 +1.6s median; multi $0.00007 +2.2s (+3 embeds); HyDE $0.00014 +3.2s median (p95 7.4s). Total H5 spend ~$0.05.
## H6 post-processing (offline, n=120; chunks avg ~1.7k chars, plain top5 = 8.7k chars)
- Duplicates in plain top5: exact text 0%, cos>.95 8.3%, same heading 13.3%. dedup_text/heading/cos95, MMR(.7/.5), max2/doc: fact@5 .858 identical, MRR delta <=.002, chars unchanged; MMR .5 raises distinct docs +0.43 [.32,.54] w/o fact loss -> harmless but no gain. REFUTED (nothing to collapse in this KB).
- Neighbour expansion (+-1 chunk): top3+nb1 fact .833 @10.2k chars vs plain5 .858 @8.7k; 0/120 facts split across adjacent chunks. REFUTED: worse per char than just more top-k.
- Plain depth curve: top3 .808 @5.2k chars, top5 .858 @8.7k, top8 .917 @13.9k (+.058* ), top10 .917 @17.4k. Diminishing: marginal 8->10 nil.
## H7 graph value (n=40 subset, real MCP call_tool on clone /tmp/search-exp/mcp; ll/hl keywords from gpt-6-luna; top_k=5)
(junie's own fact_any metric was wrong - substring test on JSON-escaped text; recomputed with parsed JSON.) Note: in this clone hybrid fused_chunks is IDENTICAL across all modes (fact@5 .95, fact@3 .85); modes differ only in `context`.
- naive: avg response 41.1k chars (context 19.4k + hybrid block 15.4k [of which chunk content 8.9k, snippet dup 1.5k, metadata ~1.3k] + rest). hybrid w/o keywords falls back to naive (identical).
- fact present in LightRAG `context`: naive .675, global .675, local .775, mix .775 (vs fused .95): the LightRAG context contains the fact LESS often than fused_chunks. Fact in either: .95 naive, .975 local/global/mix -> graph adds the fact in 1/40 cases that fused chunks lack, and loses nothing. 
- Response size: local 47.6k, mix 49.9k, global 39.3k (compact 36.7k) vs naive 40.4k. => graph modes = +0..+20% bloat for +1/40 (2.5pt, CI incl 0). Verdict: graph does NOT improve ranked-chunk retrieval; INCONCLUSIVE-to-REFUTED as retrieval win; its value would be multi-hop/synthesis (not measured by this fact-lookup set).
- reranker configured in MCP env fell back (model not cached offline) -> production here = no rerank.
## H8 bloat / answerability (judge gpt-6-luna, n=40, answerable / matches-gold)
- naive full response (29.5k ctx chars ~7.4k tok): .975 / .925; fused top-3 + metadata only (6.0k chars ~1.55k tok): .950 / .925; best graph full (global_kw 29.4k chars): .975 / .975 (+.05 [0,.125]). Differences within CI. => top-3 fused chunks give ~same answerability at ~1/5 the size. CONFIRMED (bloat is removable). Judge spend ~$0.09.
## H9 markdown chunking on the real KB (changed from plan: NO LLM build; re-chunked all 1057 clone docs in-process with the repo chunker, embedded with gemma locally, retrieval-only; graph not evaluated) n=120
- chunks: token 4168 (mean 1601 chars), md 7204 (849), md-merge(min200) 5964 (1019). Whole-fact split rate 0/120 in all three.
- hybrid fact@5: token .858, md .908 (+.050 [0,.100]), md-merge .892 (+.033 [-.017,.083]); fact_rr .729 / .754 / .737 (noise); file_hit@1 .633/.683/.650 (noise). chars@5 8674 / 4789 / 5603 (-45% / -35%, CI excl 0).
- dense only: fact@5 .692 -> md .817 (+.125 [.05,.20]*), md-merge .775 (+.083*). BM25 only: md .783 (-.033 ns), md-merge .817 (0).
- Verdict: CONFIRMED for dense/size (smaller chunks => dense better, half the chars), INCONCLUSIVE for hybrid accuracy (+3..5pt, CI touches 0). Caveat: equal top-5 means fewer chars; at equal chars token would need top-3 (.808). Heading localisation (text-based) .983 token / .826 md / .836 md-merge as measured by junie's text check (token chunks carry the [Document|Section] header text; metric not comparable) - ignore.
## H3 contextual prefix (gemma, "<source_path> > <heading_path>\n" prepended before embedding; chunks re-embedded offline) n=120
- dense fact@5 .692 -> .658 (-.033 [-.075,+.008]); prefix2 (stem+headings) .667; hybrid fact@5 .858 -> .867 (+.008 [0,.025]) / .875 (+.017); MRR 0.000/-.014. REFUTED (no gain; dense slightly worse). Sanity: gemma_plain reproduces baseline dense top-1 120/120.
## H4 Qwen3-Embedding-0.6B (4-bit MLX DWQ served by vLLM :8001; instruct query prefix) 
- dense fact@5 .692 -> .742 (+.050 [-.025,+.125]); hybrid fact@5 .858 -> .908 (+.050 [0,+.100]), file_hit@1 .633 -> .683 (+.050), MRR .729 -> .764 (+.035); ru hybrid fact@5 .857 vs en .936. With doc prefix: .892. alpha .3: .892; alpha .7 worse (.833).
- CI touches zero => INCONCLUSIVE-leaning-positive (+5pt consistent across file_hit@1, fact@5, MRR). Cost: index throughput 3.2 chunks/s via MLX server vs 8.2 chunks/s gemma on CPU; query embed 17ms both; needs a GPU/MLX service (not portable CPU artifact). Quantised variant used; fp16 might be better.
## H1 reranking (Qwen3-Reranker-0.6B Q4 via llama-server :8081 /v1/rerank; MiniLM cross-encoder NOT available offline -> skipped) n=120, baseline h1_none = hybrid pool20 top5
- top20->top5: file_hit@1 .625 -> .717 (+.092*), fact@3 .800 -> .883 (+.083*), fact@5 .867 -> .908 (+.042 ns), MRR .725 -> .796 (+.071*). top10->top5: file_hit@1 +.075, fact@3 +.083*, MRR +.062* (same within noise, half the docs).
- latency: ~4s/query uncontended (junie probe, 20 docs) but 14s median/30s p95 under load from other local jobs; llama-server single slot. Query-time cost, not index-time; no $.
- CONFIRMED: the biggest single gain on top-3/MRR. (instruct-prefixed query and top30 variants dropped: reranker server too slow under contention.)
## H10 combinations (n=120)
- mdmerge chunks + Qwen3 embeddings: dense fact@5 .708 (-.150* vs token+gemma dense? no: vs base_hybrid in table) - hybrid fact@5 .875 (+.017), MRR .745, chars@5 5428; alpha .4: .867. Not better than either lever alone -> levers do not stack (all within CI). Qwen3 dense on small md chunks was WORSE than on token chunks (.708 vs .742) - unexplained, maybe 4-bit model on short chunks.
- 3-way fusion gemma+qwen3+bm25: best (0,.5,.5, pool15) fact@5 .908 / MRR .746 = same as qwen3 hybrid alone; CV picks differ between folds -> no robust gain.
