# Performing AMBER Inference on JASMIN

This directory is designed to download images from Jasmin object store and perform inference to:
- detect objects
- classify objects as moth or non-moth
- identify the order

This branch is designed to save crops of beetles.

## JASMIN Set-Up

To use this pipeline on JASMIN you must have access to the following services:
- **Login Services**: jasmin-login. This provides access to the JASMIN shared services, i.e. login, transfer, scientific analysis servers, Jupyter notebook and LOTUS.
- **Object Store**: ami-test-o. This is the data object store tenancy for the Automated Monitoring of Insects Trap.

The [JASMIN documentation](https://help.jasmin.ac.uk/docs/getting-started/get-started-with-jasmin/) provides useful infomration on how to get set-up with these services. Including:
1. [Generate an SSH key](https://help.jasmin.ac.uk/docs/getting-started/generate-ssh-key-pair/)
2. [Getting a JASMIN portal account](https://help.jasmin.ac.uk/docs/getting-started/get-jasmin-portal-account/)
3. [Request “jasmin-login” access](https://help.jasmin.ac.uk/docs/getting-started/get-login-account/) (access to the shared JASMIN servers and the LOTUS batch cluster)

## Models

You will need to add the models files to the ./models subdirectory. Following this you can pass in:
- binary_model_path: The path to the binary model weights
- order_model_path: The path to the binary model weights
- order_threshold_path: The path to the binary model weights
- localisation_model_path: The path to the binary model weights

AMBER team members can find these files on [OneDrive](https://thealanturininstitute.sharepoint.com/:f:/r/sites/Automatedbiodiversitymonitoring/Shared%20Documents/General/Data/models/jasmin?csf=1&web=1&e=HgjhgA). Others can contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the model files.

### Recommended Box thresholds

There are several object detection models which can be used in this analysis. These have varying recommended confidence thresholds to define object bounding boxes. The box threshold can be altered using the `--box_threshold` argument in `04_process_chunks.py`. The table below outlines the recommended thresholds for some models:

| Model file name                                   | Recommended box threshold |
|---------------------------------------------------|---------------------------|
| v1_localizmodel_2021-08-17-12-06.pt **(Default)** | 0.99 **(Default)**        |
| fasterrcnn_resnet50_fpn_tz53qv9v.pt               | 0.8                       |


## Conda Environment and Installation

Once you have access to JASMIN, you will need to [install miniforge](https://help.jasmin.ac.uk/docs/software-on-jasmin/creating-and-using-miniforge-environments/) to run condat. Then create a conda environment and install packages:

```bash
CONDA_ENV_PATH="~/conda_envs/moth_detector_env/"
source ~/miniforge3/bin/activate
conda create -p "${CONDA_ENV_PATH}" python=3.9
conda activate "${CONDA_ENV_PATH}"

conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install --yes --file requirements.txt
```

## Configs

To use the inference scripts you will need to set up a `credentials.json` file containing:

```json
{
  "AWS_ACCESS_KEY_ID": `SECRET`,
  "AWS_SECRET_ACCESS_KEY": `SECRET`,
  "AWS_REGION": `SECRET`,
  "AWS_URL_ENDPOINT": `SECRET`,
  "UKCEH_username": `SECRET`,
  "UKCEH_password": `SECRET`,
  "directory": './inferences/data'
}
```

Contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the AWS Access and UKCEH API configs.

## Usage

Load the conda env on Jasmin:

```bash
source ~/miniforge3/bin/activate
conda activate "~/conda_envs/moth_detector_env/"
```

_or on Baskerville_:

```bash
module load bask-apps/live
module load CUDA/11.7.0
module load Python/3.9.5-GCCcore-10.3.0
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"
mamba activate "/bask/projects/v/vjgo8416-amber/moth_detector_env"
```

The multi-core pipeline is run in several steps:

1. Listing All Available Deployments
2. Generate Key Files
3. Chop the keys into chunks
4. Analyse the chunks

### 01. Listing Available Deployments

To find information about the available deployments you can use the print_deployments function. For all deployments:

```bash
python 01_print_deployments.py --include_inactive
```

or for the UK only:

```bash
python 01_print_deployments.py \
  --subset_countries 'United Kingdom'
```

### 02. Generating the Keys

```bash
python 02_generate_keys.py --bucket 'gbr' --deployment_id 'dep000072' --output_file './keys/solar/dep000072_keys.txt'
```

### 03. Pre-chop the Keys into Chunks

```bash
python 03_pre_chop_files.py --input_file './keys/solar/dep000072_keys.txt' --file_extensions 'jpg' 'jpeg' --chunk_size 100 --output_file './keys/solar/dep000072_workload_chunks.json'
```

### 04. Process the Chunked Files

For a single chunk:

```bash
python 04_process_chunks.py \
  --chunk_id 1 \
  --json_file './keys/solar/dep000072_workload_chunks.json' \
  --output_dir './data/solar/dep000072' \
  --bucket_name 'gbr' \
  --credentials_file './credentials.json' \
  --csv_file 'dep000072.csv' \
  --localisation_model_path ./models/fasterrcnn_resnet50_fpn_tz53qv9v.pt \
  --species_model_path ./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt \
  --species_labels ./models/03_uk_data_category_map.json \
  --perform_inference \
  --remove_image \
  --save_crops
```

### 05. Combine Chunk Outputs (Optional)

If running using slurm, we typically write each chunk to an individual csv so ensure the do not overwrite one another. To combine into one file, run:

```bash
python 05_combine_outputs.py \
  --csv_file_pattern "./data/solar/gbr/dep000072_*.csv" \
  --main_csv_file "./data/solar/gbr/dep000072.csv" \
  --remove_chunk_files
```

## Running with slurm

To run with slurm you need to be logged in on the [scientific nodes](https://help.jasmin.ac.uk/docs/interactive-computing/sci-servers/).

It is recommended you set up a shell script to runfor your country and deployment of interest. For example, `solar_field_analysis.sh` peformes inferences for the UK's Solar 1 panels deployment. You can run this using:

```bash
sbatch solar_field_analysis.sh
```

Note to run slurm you will need to install miniforge on the scientific nodes.

To check the slurm queue:

```bash
squeue -u USERNAME
```
