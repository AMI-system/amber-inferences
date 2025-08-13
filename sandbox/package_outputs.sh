#!/bin/bash

# script to create zip file for each deployment folder containing all dep*.csv files

# Directory containing the deployment folders
DEPLOYMENTS_DIR="/gws/nopw/j04/ceh_generic/kgoldmann/costarica_inferences_tracking/"

# Move into the deployments directory
cd "$DEPLOYMENTS_DIR" || exit 1

# Loop over each subdirectory (deployment)
for dir in */ ; do
    # Remove trailing slash for zip name
    deployment_name="${dir%/}"
    echo "Processing deployment: $deployment_name"

    # Create zip archive
    zip_file="${DEPLOYMENTS_DIR}${deployment_name}.zip"

    # Check if there are any matching files first
    shopt -s nullglob
    csv_files=("$dir"/dep*.csv)
    if [ ${#csv_files[@]} -eq 0 ]; then
        echo "No dep*.csv files found in $dir, skipping..."
        continue
    fi

    zip "$zip_file" "${csv_files[@]}"
done

echo "To pull the zip files from remote to your local machine, run:"
echo "scp -i ~/.ssh/id_rsa_jasmin -r Sci1ViaLogin01:'${DEPLOYMENTS_DIR}*.zip' ~/"
echo
