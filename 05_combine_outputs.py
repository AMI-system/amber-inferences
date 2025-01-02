import argparse
import glob
import os

import pandas as pd


def main(
    csv_file_pattern="results_*.csv",
    main_csv_file="results.csv",
    append=True,
    remove_chunk_files=True,
):
    """
    Main function to combine outputs from multiple chunks.

    Args:
        csv_file_pattern (str): Path pattern to save analysis results.
        main_csv_file (str): Path to the main results file.
        append (bool): Whether to append to existing results file or not.
        remove_chunk_files (bool): Whether to remove the chunk outputs once combined.
    """

    print(csv_file_pattern)

    # Combine and remove csv files
    glued_data = pd.DataFrame()
    for file_name in glob.glob(csv_file_pattern):
        print(f"Adding {file_name}")
        x = pd.read_csv(file_name, low_memory=False)
        glued_data = pd.concat([glued_data, x], axis=0)
        if remove_chunk_files:
            os.remove(file_name)

    if append:
        write_mode = "a"
        write_header = not os.path.isfile(main_csv_file)
    else:
        write_mode = "w"
        write_header = True

    print(f"Writting to {main_csv_file}")
    glued_data.to_csv(
        main_csv_file,
        mode=write_mode,
        header=write_header,
        index=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine outputs from multiple parallel sessions/chunks."
    )

    parser.add_argument(
        "--csv_file_pattern",
        default="results_*.csv",
        help="Path pattern for saved analysis results.",
    )
    parser.add_argument(
        "--main_csv_file", default="results.csv", help="Path to save all results."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Whether to append to existing results file or not.",
    )
    parser.add_argument(
        "--remove_chunk_files",
        action="store_true",
        help="Whether to remove the existing chunk outputs.",
    )

    args = parser.parse_args()

    main(
        csv_file_pattern=args.csv_file_pattern,
        main_csv_file=args.main_csv_file,
        append=args.append,
        remove_chunk_files=args.remove_chunk_files,
    )
