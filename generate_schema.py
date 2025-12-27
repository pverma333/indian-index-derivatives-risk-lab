import pandas as pd
import pathlib

def document_schemas(directory="."):
    path = pathlib.Path(directory)
    with open("schema_documentation.md", "w") as f:
        f.write("# Data Schema Documentation\n\n")

        # Search for CSV and Parquet files
        for file in path.rglob("*"):
            if file.suffix in [".csv", ".parquet"]:
                f.write(f"### File: {file.name}\n")
                f.write(f"**Path:** `{file}`\n\n")

                # Read only the first few rows to get the schema
                df = pd.read_csv(file, nrows=0) if file.suffix == ".csv" else pd.read_parquet(file).head(0)

                f.write("| Column | Data Type |\n| --- | --- |\n")
                for col, dtype in df.dtypes.items():
                    f.write(f"| {col} | {dtype} |\n")
                f.write("\n---\n")

document_schemas()
