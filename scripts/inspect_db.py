"""Inspect LanceDB database contents.

Run: python scripts/inspect_db.py [path]

Defaults to ./.spatial-memory if no path provided.
"""

import sys
from pathlib import Path
from datetime import datetime

import lancedb
import pandas as pd


def inspect_database(db_path: str = "./.spatial-memory") -> None:
    """Inspect a LanceDB database."""

    print("\n" + "=" * 70)
    print(f"DATABASE INSPECTION: {db_path}")
    print("=" * 70)

    # Check if exists
    if not Path(db_path).exists():
        print(f"\n  Database not found at: {db_path}")
        print("  Run some MCP operations first to create data.")
        return

    # Connect
    db = lancedb.connect(db_path)
    tables = db.table_names()

    print(f"\n  Tables: {tables}")

    for table_name in tables:
        print(f"\n{'-' * 70}")
        print(f"TABLE: {table_name}")
        print("-" * 70)

        table = db.open_table(table_name)

        # Schema
        print(f"\n  Schema:")
        for field in table.schema:
            print(f"    - {field.name}: {field.type}")

        # Stats
        row_count = table.count_rows()
        print(f"\n  Row count: {row_count}")

        if row_count == 0:
            continue

        # Get data
        df = table.to_pandas()

        # Columns to display (exclude vector)
        display_cols = [c for c in df.columns if c != 'vector']

        print(f"\n  Columns: {display_cols}")

        # Namespace distribution
        if 'namespace' in df.columns:
            print(f"\n  Namespaces:")
            for ns, count in df['namespace'].value_counts().items():
                print(f"    - {ns}: {count} memories")

        # Date range
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            oldest = df['created_at'].min()
            newest = df['created_at'].max()
            print(f"\n  Date range:")
            print(f"    Oldest: {oldest}")
            print(f"    Newest: {newest}")

        # Importance distribution
        if 'importance' in df.columns:
            print(f"\n  Importance stats:")
            print(f"    Min: {df['importance'].min():.2f}")
            print(f"    Max: {df['importance'].max():.2f}")
            print(f"    Mean: {df['importance'].mean():.2f}")

        # Content length stats
        if 'content' in df.columns:
            df['content_len'] = df['content'].str.len()
            print(f"\n  Content length stats:")
            print(f"    Min: {df['content_len'].min()} chars")
            print(f"    Max: {df['content_len'].max()} chars")
            print(f"    Mean: {df['content_len'].mean():.0f} chars")

        # Sample data
        print(f"\n  Sample memories (first 5):")
        print("  " + "-" * 66)
        for idx, row in df.head(5).iterrows():
            content = row.get('content', '')[:60]
            ns = row.get('namespace', 'default')
            importance = row.get('importance', 0.5)
            print(f"  [{ns}] (imp={importance:.2f}) {content}...")

        # Vector info
        if 'vector' in df.columns:
            sample_vector = df['vector'].iloc[0]
            if sample_vector is not None:
                vec_dim = len(sample_vector) if hasattr(sample_vector, '__len__') else 'unknown'
                print(f"\n  Vector dimensions: {vec_dim}")

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./.spatial-memory"
    inspect_database(db_path)


if __name__ == "__main__":
    main()
