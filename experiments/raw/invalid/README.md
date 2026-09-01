# Invalid runs

The two `device-memory-double-counted` pilot files completed without request
failures, but are excluded from summaries because their sampler summed vLLM's
shared allocation once per process and reported more memory than the L4 owns.
They are retained so the exclusion is auditable.

The four `pre-final-source` files are also excluded: the experiment worker was
not yet included in `repository_commit`, so those restarts cannot prove a
single harness identity. They completed without request failures and motivated
the final source-identity gate, but are not summarized.
