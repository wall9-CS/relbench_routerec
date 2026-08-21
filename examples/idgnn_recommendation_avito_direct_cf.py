from __future__ import annotations

from .direct_cf_recommendation import (
    DirectCFTrainingComponents,
    run_direct_cf_recommendation,
)
from .avito_direct_cf.coverage import compute_direct_cf_coverage_for_split
from .avito_direct_cf.graph import (
    DIRECT_AD_CF,
    attach_direct_cf_snapshot,
    build_direct_cf_num_neighbors,
    build_direct_cf_schema_template,
    direct_ad_cf_col_stats,
)
from .avito_direct_cf.io import load_snapshot, validate_manifest_for_training


def main() -> None:
    run_direct_cf_recommendation(
        DirectCFTrainingComponents(
            description="Train 2-layer ID-GNN with Rel-Avito direct-CF snapshots.",
            dataset="rel-avito",
            task="user-ad-visit",
            cf_node_type=DIRECT_AD_CF,
            validate_manifest_for_training=validate_manifest_for_training,
            load_snapshot=load_snapshot,
            attach_snapshot=attach_direct_cf_snapshot,
            build_schema_template=build_direct_cf_schema_template,
            build_num_neighbors=build_direct_cf_num_neighbors,
            cf_col_stats=direct_ad_cf_col_stats,
            compute_coverage_for_split=compute_direct_cf_coverage_for_split,
            source_count=lambda db: len(db.table_dict["UserInfo"]),
            dst_count=lambda task, db: task.num_dst_nodes,
            snapshot_size_kwargs=lambda num_source, num_dst: {
                "num_users": num_source,
                "num_ads": num_dst,
            },
            coverage_label="Avito",
        )
    )


if __name__ == "__main__":
    main()
