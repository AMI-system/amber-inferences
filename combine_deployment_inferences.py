import os
import pandas as pd
import requests
from glob import glob
import argparse
from amber_inferences.utils.config import load_credentials
from amber_inferences.utils.api_utils import get_deployments

# Recursively list all directories matching dep* pattern
def list_deployment_folders(base_dir):
    return [os.path.basename(f) for f in glob(os.path.join(base_dir, "**/dep*"), recursive=True) if os.path.isdir(f)]

# Read and combine all CSVs in a folder, preserving first file's column order
def read_and_combine_csvs(folder_path, dep_id):
    csv_files = [f for f in glob(os.path.join(folder_path, "*.csv")) if os.path.basename(f).startswith(dep_id)]
    dfs = []
    reference_cols = None

    for file in csv_files:
        df = pd.read_csv(file)
        if reference_cols is None:
            reference_cols = list(df.columns)
        else:
            if set(df.columns) != set(reference_cols):
                diff = set(df.columns).symmetric_difference(set(reference_cols))
                print(f"[WARN] Column mismatch in {file}: {diff}")
            df = df.reindex(columns=reference_cols)
        dfs.append(df)

    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df, reference_cols
    return pd.DataFrame(), []

# Process deployments
def process_all(base_dir, aws_credentials):
    deployments = get_deployments(aws_credentials['UKCEH_username'], aws_credentials['UKCEH_password'])
    deployments_dict = {d["deployment_id"]: d for d in deployments}
    folders = list_deployment_folders(base_dir)

    for dep_id in folders:
        folder_path = os.path.join(base_dir, dep_id)
        print(f"Processing: {dep_id}")

        df, col_order = read_and_combine_csvs(folder_path, dep_id)
        if df.empty:
            print(f"  [INFO] No CSVs found in {dep_id}")
            continue

        dep_meta = deployments_dict.get(dep_id, {})
        df["AMI_name"] = dep_meta.get("location_name")
        df["latitude"] = dep_meta.get("lat")
        df["longitude"] = dep_meta.get("lon")

        final_cols = col_order + ["AMI_name", "latitude", "longitude"]
        df = df[final_cols]

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{dep_id}_combined.csv")
        df.to_csv(output_path, index=False)
        print(f"  [DONE] Saved combined CSV to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine deployment CSVs and attach metadata.")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to the base directory containing deployment folders.")
    parser.add_argument("--creds", type=str, default="./credentials.json", help="Path to the credentials JSON file.")

    args = parser.parse_args()

    aws_credentials = load_credentials(args.creds)
    process_all(args.base_dir, aws_credentials)