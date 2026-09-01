# Failed no-Triton pilot

The first no-Triton process failed during startup because the Torch reference
attention backend performs host-built indexed gathers that CUDA graph capture
cannot record. No timing cell started and no result entered `results.csv`.

The valid controlled comparison therefore uses two eager variants:
`no_cuda_graph` (Triton) versus `no_triton` (Torch reference). Their only
configuration difference is `use_triton_attention`.
