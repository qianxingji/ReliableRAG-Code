"""Frozen empirical-panel membership and feature contract; no model fitting."""
import hashlib

DATASETS = ("hotpotqa", "2wikimultihopqa", "musique")
RETRIEVERS = ("bm25", "dense", "hybrid")
FIELDS = {"ROA-FULL": list(range(11)), "ROA-NOGBV": list(range(10)),
          "HGB_GBV_R": [0, 10], "HGB_ONLY_R": [0], "GBV_ONLY_R": [10]}
FIT_PREFIX = "cas-q2-empirical-fit-v1|20260924"
FRESH_PREFIX = "cas-q2-empirical-fresh-v1|20260925"


def hash_order(prefix, dataset, sample_id):
    return hashlib.sha256(f"{prefix}|{dataset}|{sample_id}".encode("utf-8")).hexdigest(), sample_id


def split_development(keys):
    keys = set(keys)
    groups = {(k[0], k[2]) for k in keys}
    if len(keys) != 13500 or len(groups) != 4500:
        raise ValueError("Exact opened development cohort required")
    if {k[0] for k in keys} != set(DATASETS):
        raise ValueError("Exact dataset identities required")
    if any({(d,r,i) for r in RETRIEVERS} - keys for d,i in groups):
        raise ValueError("Every question must retain all three retrievers")
    calibration = set()
    for dataset in DATASETS:
        ids = [i for d,i in groups if d == dataset]
        if len(ids) != 1500:
            raise ValueError("Exactly 1500 development questions per dataset")
        ordered = sorted(ids, key=lambda i:hash_order(FIT_PREFIX, dataset, i))
        calibration.update((dataset,i) for i in ordered[:300])
    cal = {k for k in keys if (k[0],k[2]) in calibration}
    return {"fit": keys-cal, "cal": cal, "probe": keys}


def select_fresh_ids(sources, forbidden):
    if set(sources) != set(DATASETS):
        raise ValueError("All source datasets required")
    chosen = {}
    for dataset in DATASETS:
        if any(d != dataset for d,i in sources[dataset]):
            raise ValueError("Source dataset mismatch")
        available = sorted({i for d,i in sources[dataset] if (d,i) not in forbidden},
                           key=lambda i:hash_order(FRESH_PREFIX,dataset,i))
        if len(available) < 2000:
            raise ValueError("Insufficient fresh IDs: " + dataset)
        chosen[dataset] = available[:2000]
    return chosen
