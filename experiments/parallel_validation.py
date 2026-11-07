"""Bounded paired-job dispatch and cooperative child cancellation; no retries."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

CANCEL_STORE = "cie-validation-cancellation"


def campaign_cancelled(store, campaign_id):
    value = store.get(campaign_id)
    if type(value) is not bool:
        raise RuntimeError("campaign cancellation state missing or malformed")
    return value


def run_bounded(jobs, run_job, retain_result, cancel_campaign, *, max_workers=4):
    if not 1 <= max_workers <= 4:
        raise ValueError("predeclared worker bound is 1–4")
    iterator = iter(jobs)
    pending = {}
    result = {"stopped": False, "submitted": 0, "completed": 0, "errors": []}

    def stop():
        if not result["stopped"]:
            result["stopped"] = True
            try:
                cancel_campaign()
            except Exception as exc:
                result["errors"].append({"kind": "cancel_signal_failed", "error": f"{type(exc).__name__}: {exc}"})

    def retain(future):
        job = pending.pop(future)
        try:
            payload = future.result()
        except Exception as exc:
            payload = {"orchestration_error": f"{type(exc).__name__}: {exc}"}
            stop()
        try:
            if retain_result(job, payload) is not True:
                stop()
        except Exception as exc:
            result["errors"].append({"kind": "retention_or_validation_error", "job": repr(job)[:1000],
                                      "error": f"{type(exc).__name__}: {exc}"})
            stop()
        result["completed"] += 1

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        def fill():
            while not result["stopped"] and len(pending) < max_workers:
                try:
                    job = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(run_job, job)] = job
                result["submitted"] += 1
        try:
            fill()
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                # Inspect every already completed job before submitting another.
                for future in completed:
                    retain(future)
                # Volume commits can take time: inspect failures that completed
                # during retention before filling any newly free worker slots.
                while ready := [future for future in pending if future.done()]:
                    for future in ready:
                        retain(future)
                fill()
        except BaseException:
            stop()
            for future in list(pending):
                retain(future)
            raise
    return result


def run_cancellable_process(command, *, env, stdout_path, stderr_path, cancelled,
                            timeout_seconds, poll_seconds=2):
    if timeout_seconds <= 0 or not 0 < poll_seconds <= 2:
        raise ValueError("invalid bounded process timing")
    result = {"returncode": None, "cancelled": False, "timeout": False,
              "cancel_check_error": None, "cancellation_reason": None}
    process = None
    started = time.monotonic()

    def cancellation():
        try:
            if cancelled():
                result.update(cancelled=True, cancellation_reason="campaign_cancelled")
                return True
        except Exception as exc:
            result.update(cancelled=True, cancellation_reason="cancel_check_failed",
                          cancel_check_error=f"{type(exc).__name__}: {exc}")
            return True
        return False

    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        if not cancellation():
            try:
                process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                while process.poll() is None:
                    if cancellation():
                        break
                    remaining = timeout_seconds - (time.monotonic()-started)
                    if remaining <= 0:
                        result.update(timeout=True, cancellation_reason="timeout")
                        break
                    try:
                        process.wait(timeout=min(poll_seconds, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except OSError as exc:
                result["launch_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                if process is not None:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    result["returncode"] = process.wait()
    result.update(stdout=stdout_path.read_text(errors="replace"), stderr=stderr_path.read_text(errors="replace"),
                  elapsed_seconds=time.monotonic()-started)
    return result


def open_durable_campaign(campaign_id, volume, root=None):
    """Single coordinator writer; commit named evidence before accepting a pair."""
    import hashlib
    import json
    import re
    from pathlib import Path
    if re.fullmatch(r"[0-9a-f-]{32,36}", campaign_id) is None:
        raise ValueError("invalid campaign ID")
    directory = (Path('/evidence') if root is None else Path(root)) / campaign_id
    directory.mkdir(parents=True, exist_ok=False)
    state = {'campaign_id': campaign_id, 'complete': False, 'error': None, 'artifacts': {}}
    def checkpoint(paths=(), **status):
        for path in paths:
            path = Path(path)
            name = path.relative_to(directory).as_posix()
            state['artifacts'][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        state.update(status)
        temporary = directory / 'index.tmp'
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(directory / 'index.json')
        volume.commit()
    checkpoint()
    return directory, state, checkpoint


def recover_durable(volume, destination, campaign_id):
    import hashlib
    import json
    from pathlib import Path
    destination = Path(destination)
    raw = b''.join(volume.read_file(f'{campaign_id}/index.json'))
    state = json.loads(raw)
    for name, expected in state['artifacts'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid durable artifact path')
        data = b''.join(volume.read_file(f'{campaign_id}/{name}'))
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('durable artifact hash mismatch')
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            previous = path.read_bytes()
            if path.suffix != '.jsonl' or not data.startswith(previous):
                raise ValueError('refusing to replace unequal retained evidence')
        path.write_bytes(data)
    (destination / 'remote-index.json').write_bytes(raw)
    return state


def save_dispatch(destination, campaign_id, call, app_id):
    import json
    from pathlib import Path
    record = {'campaign_id': campaign_id, 'call_id': call.object_id, 'app_id': app_id}
    with (Path(destination) / 'dispatch.json').open('x') as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return record


def wait_durable(call, volume, destination, dispatch):
    try:
        call.get()
    except BaseException:
        try:
            recover_durable(volume, destination, dispatch['campaign_id'])
        except Exception as exc:
            print(f'Evidence pending/unavailable: {type(exc).__name__}: {exc}; saved dispatch remains; do not resubmit.', flush=True)
        raise
    state = recover_durable(volume, destination, dispatch['campaign_id'])
    if state.get('complete') is not True or state.get('error') is not None:
        raise RuntimeError('campaign failed/incomplete; durable evidence retained; no resubmission')
    return state
