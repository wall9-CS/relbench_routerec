# Rel-Trial Route-Collapsed CF

This CF route collapses the Rel-Trial recommendation path
`source -> *_studies -> studies -> sponsors_studies -> sponsor` into
seed-time-specific `source -> trial_cf -> sponsor` shortcut candidates.

Build snapshots:

```bash
python -m examples.build_trial_cf_snapshots \
  --task condition-sponsor-run \
  --all-history \
  --output-root /path/to/cf_snapshots
```

Train with the snapshots:

```bash
python -m examples.idgnn_recommendation_trial_cf \
  --task condition-sponsor-run \
  --num_layers 2 \
  --cf-snapshot-dir /path/to/cf_snapshots/rel-trial/condition-sponsor-run/source_sponsor_all_history_alpha_0.5_support_1_top64
```

Use `--task site-sponsor-run` for the facility-to-sponsor recommendation task.

