"""Reproducible benchmark runner (PRD §12).

Protocol per mode: fresh engine, identical weights, two unmeasured warmup
requests, reset peaks/metrics, three measured runs of the selected workload.
Every run is reported; the median headlines. Nothing is discarded or retried.
"""
