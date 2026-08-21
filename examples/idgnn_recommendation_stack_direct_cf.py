from __future__ import annotations

from .direct_cf_recommendation import (
    DirectCFTrainingComponents,
    run_direct_cf_recommendation,
)
from .stack_direct_cf.coverage import compute_direct_cf_coverage_for_split
from .stack_direct_cf.graph import (
    DIRECT_POST_CF,
    attach_direct_cf_snapshot,
    build_direct_cf_num_neighbors,
    build_direct_cf_schema_template,
    direct_post_cf_col_stats,
)
from .stack_direct_cf.io import load_snapshot, validate_manifest_for_training


def main() -> None:
    run_direct_cf_recommendation(
        DirectCFTrainingComponents(
            description="Train 2-layer ID-GNN with Rel-Stack direct-CF snapshots.",
            dataset="rel-stack",
            task="user-post-comment",
            cf_node_type=DIRECT_POST_CF,
            validate_manifest_for_training=validate_manifest_for_training,
            load_snapshot=load_snapshot,
            attach_snapshot=attach_direct_cf_snapshot,
            build_schema_template=build_direct_cf_schema_template,
            build_num_neighbors=build_direct_cf_num_neighbors,
            cf_col_stats=direct_post_cf_col_stats,
            compute_coverage_for_split=compute_direct_cf_coverage_for_split,
            source_count=lambda db: len(db.table_dict["users"]),
            dst_count=lambda task, db: task.num_dst_nodes,
            snapshot_size_kwargs=lambda num_source, num_dst: {
                "num_users": num_source,
                "num_posts": num_dst,
            },
            coverage_label="Stack",
        )
    )


if __name__ == "__main__":
    main()
