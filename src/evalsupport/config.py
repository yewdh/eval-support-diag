from pathlib import Path
import yaml


DEFAULT = {
    "paths": {
        "base_dir": "outputs",
        "data_dir": "data/features",
        "rates_csv": "outputs/pr_rates_by_region.csv",
        "mech_csv": "outputs/pr_mechanism_check.csv",
    },
    "scales": ["1km", "2km", "5km"],
    "exp_name": "Model_E_Ultimate_V59",
    "classes": {"tobacco": 1, "maize": 2},
    "file_templates": {
        "pos": "YN_2024_TST_Ultra_v21_Positive_Samples{sc}.csv",
        "neg": "YN_2024_TST_Ultra_v21_Negative_Samples{sc}.csv",
    },
    "bootstrap": {
        "n_boot": 2000, "seed": 42, "min_total_gap": 0.05,
        "ci_low": 2.5, "ci_high": 97.5,
    },
    "simulation": {
        "n_mc": 2000, "seed": 42, "setting": "ood", "target_n": 52,
    },
}


def load_config(path=None):
    cfg = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULT.items()}
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
