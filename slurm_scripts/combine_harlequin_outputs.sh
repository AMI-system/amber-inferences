# for each directory in the input, run 05_combine_outputs.py

input_dirs='/home/users/katriona/amber-inferences/data/harlequin_flatbug/'

dirs=$(find ${input_dirs} -maxdepth 1 -type d)

for dir in $dirs; do
    dep=$(basename "$dir" .txt)

    # if dep doesnt start with 'dep', skip
    if [[ ! $dep == dep* ]]; then
        continue
    fi

    echo "Processing $dep"
    python 05_combine_outputs.py \
        --csv_file_pattern "./data/harlequin_flatbug/${dep}_*.csv" \
        --main_csv_file "./data/all_${dep}.csv"
done
