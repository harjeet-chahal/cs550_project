"""Demo / offline-pipeline consistency.

The demo's link-prediction decision threshold (loaded from
`demo_state.pkl`) MUST equal the threshold the offline pipeline
recorded in `link_prediction_results.json` under `gcn_lr.chosen_threshold`.
If they ever drift, the demo would classify edges differently from the
numbers reported in the README / JSON, which is exactly the kind of
silent inconsistency the audit flagged.
"""

import json
import math
import os
import pickle
import pytest

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'results',
)


def test_demo_threshold_matches_lp_json():
    state_path = os.path.join(RESULTS_DIR, 'demo_state.pkl')
    lp_path    = os.path.join(RESULTS_DIR, 'link_prediction_results.json')
    if not (os.path.exists(state_path) and os.path.exists(lp_path)):
        pytest.skip(
            "required artifacts not found "
            "(run `python src/main.py` to generate)"
        )

    with open(state_path, 'rb') as f:
        meta = pickle.load(f)
    with open(lp_path) as f:
        lp = json.load(f)

    assert 'link_threshold' in meta, "demo_state.pkl missing 'link_threshold'"
    assert 'gcn_lr' in lp, "link_prediction_results.json missing 'gcn_lr' block"

    demo_thr = float(meta['link_threshold'])
    json_thr = float(lp['gcn_lr']['chosen_threshold'])

    # JSON is rounded to 4 decimals; demo state stores the full float.
    # Allow up to 5e-4 absolute tolerance to absorb that rounding.
    assert math.isclose(demo_thr, json_thr, abs_tol=5e-4), (
        f"demo threshold {demo_thr} disagrees with "
        f"link_prediction_results.json gcn_lr threshold {json_thr}"
    )
