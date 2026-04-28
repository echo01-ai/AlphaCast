import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import yaml


@dataclass
class DatasetConfig:
    # config.yaml 中的单个数据集配置。和 run_experiment.py 对照阅读，
    # 可以看清一次实验启动前需要哪些字段。
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
    # 实验级开关：决定下游是否能使用目标序列特征、外生变量证据，
    # 以及是否强制指定某个模型。
    datasets: List[DatasetConfig]
    output_dir: str = "outputs"
    # New optional fields
    use_features: bool = True
    feature_selection_override: Optional[Dict] = None
    # Exogenous variable processing switch
    use_exogenous: bool = False
    sel_model: Optional[str] = None


def load_config(path: str) -> ExperimentConfig:
    # 将 YAML 规范化为 dataclass，并展开路径中的环境变量。
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw_datasets = raw.get("datasets", [])
    if isinstance(raw_datasets, dict):
        dataset_entries = []
        for key, value in raw_datasets.items():
            if not isinstance(value, dict):
                continue
            entry = {"name": value.get("name", key), **value}
            dataset_entries.append(entry)
        raw_datasets = dataset_entries
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError(f"No datasets configured in {path}. Fill the 'datasets' list before running.")

    required_str_fields = ("name", "training_csv", "test_csv")
    required_int_fields = ("look_back", "predicted_window", "sliding_window")

    def _is_empty(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    normalized_datasets = []
    for index, entry in enumerate(raw_datasets):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry #{index + 1} in {path} must be a mapping.")

        dataset_label = entry.get("name") or f"#{index + 1}"
        missing = [
            field_name
            for field_name in (*required_str_fields, *required_int_fields)
            if _is_empty(entry.get(field_name))
        ]
        if missing:
            raise ValueError(
                f"Dataset {dataset_label!r} in {path} is missing required field(s): "
                f"{', '.join(missing)}."
            )

        cleaned = dict(entry)
        cleaned["name"] = str(cleaned["name"])
        cleaned["training_csv"] = str(cleaned["training_csv"])
        cleaned["test_csv"] = str(cleaned["test_csv"])
        for field_name in required_int_fields:
            try:
                cleaned[field_name] = int(cleaned[field_name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Dataset {dataset_label!r} field '{field_name}' must be an integer."
                ) from exc
        if cleaned.get("aliases") is None:
            cleaned["aliases"] = []
        if cleaned.get("checkpoints") is None:
            cleaned["checkpoints"] = {}
        normalized_datasets.append(cleaned)

    datasets = [DatasetConfig(**d) for d in normalized_datasets]
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
                if path
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
