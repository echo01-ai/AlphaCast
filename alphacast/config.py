import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml


@dataclass
class DatasetConfig:
    name: str
    training_csv: str
    test_csv: str
    look_back: int
    predicted_window: int
    sliding_window: int
    frequency: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    checkpoints: Dict[str, str] = field(default_factory=dict)
    context_prompt_file: Optional[str] = None

    def all_aliases(self) -> List[str]:
        base = {self.name.lower()}
        base.update(str(alias).lower() for alias in self.aliases)
        return sorted(base)


@dataclass
class ExperimentConfig:
    datasets: List[DatasetConfig]
    output_dir: str = "outputs"
    # New optional fields
    use_features: bool = True
    feature_selection_override: Optional[Dict] = None
    # Exogenous variable processing switch
    use_exogenous: bool = False
    sel_model: Optional[str] = None


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw_datasets = raw.get("datasets", [])
    if isinstance(raw_datasets, dict):
        dataset_entries = []
        for key, value in raw_datasets.items():
            if not isinstance(value, dict):
                continue
            entry = {"name": value.get("name", key), **value}
            dataset_entries.append(entry)
        raw_datasets = dataset_entries
    datasets = [DatasetConfig(**d) for d in raw_datasets]
    output_dir = raw.get("output_dir", "outputs")
    # New fields with defaults
    use_features = bool(raw.get("use_features", True))
    feature_selection_override = raw.get("feature_selection_override")
    use_exogenous = bool(raw.get("use_exogenous", False))
    sel_model_raw = raw.get("SEL_MODEL")
    sel_model = None
    if sel_model_raw is not None:
        sel_model_str = str(sel_model_raw).strip()
        sel_model = sel_model_str or None

    # Expand env vars and absolute paths
    for d in datasets:
        d.training_csv = os.path.expandvars(d.training_csv)
        d.test_csv = os.path.expandvars(d.test_csv)
        if d.context_prompt_file:
            d.context_prompt_file = os.path.expandvars(d.context_prompt_file)
        if d.checkpoints:
            d.checkpoints = {
                str(model): os.path.expandvars(path)
                for model, path in d.checkpoints.items()
            }
        d.aliases = DatasetConfig.all_aliases(d)
    return ExperimentConfig(
        datasets=datasets,
        output_dir=output_dir,
        use_features=use_features,
        feature_selection_override=feature_selection_override,
        use_exogenous=use_exogenous,
        sel_model=sel_model,
    )
