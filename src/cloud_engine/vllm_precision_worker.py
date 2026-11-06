"""Apply the shared policy inside each spawned vLLM worker before model loading."""
from cloud_engine.precision import configure_inference_precision, precision_status

configure_inference_precision()


class PrecisionWorker:
    def precision_status(self) -> dict:
        return precision_status()

    def numerical_policy_status(self) -> dict:
        from cloud_engine.numerical_policy import policy_status

        return {**policy_status(self.model_runner.model), "precision": precision_status()}

    def kv_capacity(self) -> dict:
        runner = self.model_runner
        return {"num_gpu_blocks": runner.kv_cache_config.num_blocks,
                "block_sizes": [table.block_size for table in runner.input_batch.block_table.block_tables],
                "tensor_parallel_size": runner.vllm_config.parallel_config.tensor_parallel_size,
                "source": "actual worker KV configuration and block tables"}

    def graph_status(self) -> dict:
        active = (_graph_class is not None and _graph_class.capture_end is _count_capture
                  and _graph_class.replay is _count_replay)
        return {"observer_active": active, **_graph_counts,
                "scope": "actual successful torch CUDA graph operations in this worker process"}


# Count actual successful CUDA graph operations; no GPU synchronization is added.
_graph_counts = {"capture_count": 0, "replay_count": 0}
_graph_class = None
_capture_end = None
_replay = None


def _count_capture(self, *args, **kwargs):
    result = _capture_end(self, *args, **kwargs)
    _graph_counts["capture_count"] += 1
    return result


def _count_replay(self, *args, **kwargs):
    result = _replay(self, *args, **kwargs)
    _graph_counts["replay_count"] += 1
    return result


def _install_graph_observer():
    global _graph_class, _capture_end, _replay
    import torch

    _graph_class = getattr(getattr(torch, "cuda", None), "CUDAGraph", None)
    if _graph_class is not None:
        _capture_end, _replay = _graph_class.capture_end, _graph_class.replay
        _graph_class.capture_end, _graph_class.replay = _count_capture, _count_replay


_install_graph_observer()


def _install_model_policy_loader():
    # Patch the actual lifecycle class, not the worker extension mixin: vLLM
    # rejects extension methods that conflict with existing worker methods.
    from vllm.v1.worker.gpu_worker import Worker
    if getattr(Worker.load_model, "_cie_numerical_policy_loader", False):
        return
    original = Worker.load_model
    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        from cloud_engine.numerical_policy import install_vllm_policy

        install_vllm_policy(self.model_runner.model, self.model_runner.model_config,
                            self.model_runner.vllm_config.parallel_config)
        return result
    load_model._cie_numerical_policy_loader = True
    Worker.load_model = load_model


_install_model_policy_loader()
