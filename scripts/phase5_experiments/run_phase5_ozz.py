"""Manual launcher for the Phase 5 real OZZ study."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osc_tools.ml.models.phasor_hierarchical_v3 import HierarchicalV3Config
from osc_tools.ml.phase5_hierarchical_ozz import OZZDataset, prepare_records
from osc_tools.ml.models.ozz_study import build_encoder
from scripts.phase5_experiments.run_phase5_hierarchical_ozz import run

VARIANT = "full"
EXTERNAL_FACTORY = None  # module:factory for timesnet or patchtst
CHECK_DATA_ONLY = True
EPOCHS = 100
IMPULSE_PROBABILITY = .02  # probability per training window
SEED = 42
WORKERS = 4
MANIFEST = ROOT / "data/phase5/real_ozz_prepared/manifest.json"
OUTPUT = ROOT / "experiments/phase5/ozz_study" / VARIANT


def main():
    architecture = HierarchicalV3Config()
    config = dict(
        architecture=asdict(architecture), study_variant=VARIANT, external_factory=EXTERNAL_FACTORY,
        prepared_real_manifest=str(MANIFEST), output=str(OUTPUT),
        resume=(OUTPUT / "latest_checkpoint.pt").is_file(), epochs=EPOCHS,
        seed=SEED, workers=WORKERS, use_real_no_ozz=True, presence_type=True,
        rtds_dir=str(ROOT / "data/Simulated_OZZ_v1"), samples_per_epoch=2000,
        validation_samples=200, max_seconds=30., target_tokens=640, save_every=500,
        console_interval=2., backbone_lr=2e-4, head_lr=5e-4,
        lr_warmup_epochs=min(10., EPOCHS*.1), balanced_events=True,
        current_dropout_all=.20, current_dropout_phase=.25, voltage_dropout_phase=.03,
        label_error_probability=.05, no_checkpoint_max_tokens=800,
        impulse_probability=IMPULSE_PROBABILITY,
    )
    if CHECK_DATA_ONLY:
        ssl = json.loads((ROOT / "data/phase5/research_strict_splits.json").read_text(encoding="utf-8"))
        records, manifest = prepare_records(ROOT, config, ssl)
        dataset = OZZDataset(records["train"], 8, SEED, 1., False,
                             feature_mode="hierarchical_v2", label_tail_periods=1, study_mixture=True)
        groups = [int(dataset[i]["source_group"]) for i in range(len(dataset))]
        model = build_encoder(VARIANT, architecture, external_factory=EXTERNAL_FACTORY)
        print(f"Data OK. Window groups: {groups}. Real test sections: {len(manifest['real_test'])}. "
              f"Encoder parameters: {sum(p.numel() for p in model.parameters()):,}")
        return
    run(config)


if __name__ == "__main__":
    main()
