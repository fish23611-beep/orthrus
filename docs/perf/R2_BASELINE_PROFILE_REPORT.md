# R2 bounded baseline profile report

Date: 2026-10-01 UTC.

## Result

No E3 or E5 timing result was produced in this container. All 2/10/20-window
cells are `NOT_RUN`; the dataset gates are `BLOCKED`, not `FAIL`.

This is intentional. Substituting the available older E3 hash, MAGIC output,
synthetic input, or CPU timing would not be a valid MSTC GPU baseline.

## E3 evidence

The effective `mstc_full` paths resolved against `/home/artifacts` are:

- graph: `graph_construction/THEIA_E3/build_graphs/3eb70b.../nx/` — present (522 files);
- Word2Vec: `edge_featurization/THEIA_E3/embed_nodes/642ecb.../word2vec_models/` — absent;
- edge embeddings: `edge_featurization/THEIA_E3/embed_edges/e8f6c7.../edge_embeds/` — absent.

An older E3 edge-embedding hash (`448b85...`) contains 431 files and about
9.7 GiB, but it is not the effective `mstc_full` task hash and was therefore not
used. An older Word2Vec hash (`635596...`) is likewise not interchangeable.

Status: E3 2/10/20 windows `NOT_RUN`; E3 profile gate `BLOCKED`.

## E5 evidence

The effective E5 graph, Word2Vec, and edge-embedding paths are absent. No E5
substitute was used.

Status: E5 2/10/20 windows `NOT_RUN`; E5 profile gate `BLOCKED`.

## Additional local execution blockers

- The isolated worktree's `Ground_Truth/darpa` submodule is not initialized.
  `/home` has a populated copy, but it was not linked or copied into the perf
  worktree because `/home` was explicitly protected.
- The container has an RTX 5070 Laptop GPU (compute capability `sm_120`), while
  PyTorch 1.13.1 reports support only through `sm_86`. CUDA availability alone
  is therefore insufficient for a valid GPU run. A one-element CUDA kernel
  smoke failed to complete after more than 180 seconds and was interrupted;
  no experiment process was left running.

Exact server-side commands and preflight checks are in
`docs/perf/R2_SERVER_COMMANDS.md`.

