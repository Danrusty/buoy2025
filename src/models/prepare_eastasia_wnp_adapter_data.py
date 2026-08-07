"""准备 East Asia–western North Pacific adapter 行级数据。"""

from __future__ import annotations

import argparse

from eastasia_wnp_regional import prepare_eawnp_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="准备 15–45N、105–170E frozen-lineage adapter 数据"
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result = prepare_eawnp_dataset(code_commit=args.code_commit)
    print(f"Population: {result['population']}")
    print(f"Filtered data: {result['filtered_data_path']}")
    print(f"Split manifest: {result['split_manifest_path']}")


if __name__ == "__main__":
    main()
