"""Prepare video inputs only from the completed real-GPU evidence bundle."""
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
raw_bytes = (root / 'raw.json').read_bytes()
raw = json.loads(raw_bytes)
data = json.loads((root / 'summary.json').read_text())
assert data['raw_sha256'] == hashlib.sha256(raw_bytes).hexdigest()
assert data['run_count'] == 10 and len(data['comparisons']) == 9, 'Complete study required'
assert data['environment']['gpu_identity_stable'], 'Comparisons require same physical GPU'
assert not data['crashed_runs'], 'Do not render incomplete GPU study as validated'
assert data['environment']['gpu']['name'], 'Actual GPU identity required'
assert all(c['status'] not in ('unavailable', 'incomplete') for c in data['comparisons'])
data['tokens'] = {engine: raw['runs'][f'{engine}_original']['result']['result']['output_tokens']
                  for engine in ('custom', 'vllm')}
assert all(len(tokens) == 3 for tokens in data['tokens'].values())
data['studyDate'] = data['environment']['started_utc'][:10]
Path(__file__).with_name('study-data.json').write_text(json.dumps(data, indent=2) + '\n')
