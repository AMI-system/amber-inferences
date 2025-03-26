# Performing AMBER Inference on JASMIN

This directory is designed to download images from Jasmin object store and perform inference to:
- detect objects
- classify objects as moth or non-moth
- identify the order
- predict the species


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

| Model file name                      | Recommended box threshold |
|--------------------------------------|---------------------------|
| v1_localizmodel_2021-08-17-12-06.pt  | 0.99         |
| fasterrcnn_resnet50_fpn_tz53qv9v.pt  | 0.8          |
| flat_bug_M.pt                        | 0.0          |


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
conda activate "~/conda_envs/flatbug/"

cd amber-inferences
pip install -e .
```

<!-- _or on Baskerville_:

```bash
module load bask-apps/live
module load CUDA/11.7.0
module load Python/3.9.5-GCCcore-10.3.0
module load Miniforge3/24.1.2-0
eval "$(${EBROOTMINIFORGE3}/bin/conda shell.bash hook)"
source "${EBROOTMINIFORGE3}/etc/profile.d/mamba.sh"
mamba activate "/bask/projects/v/vjgo8416-amber/moth_detector_env"
``` -->

The multi-core pipeline is run in several steps:

1. Listing All Available Deployments
2. Generate Key Files
3. Chop the keys into chunks
4. Analyse the chunks


## Running with slurm

To run with slurm on JASMIN you need to be logged in on the [scientific nodes](https://help.jasmin.ac.uk/docs/interactive-computing/sci-servers/).

It is recommended you set up a shell script to runfor your country and deployment of interest. For example, `solar_field_analysis.sh` peformes inferences for the UK's Solar 1 panels deployment. You can run this using:

```bash
sbatch solar_field_analysis.sh
```

Note to run slurm you will need to install miniforge on the scientific nodes.

To check the slurm queue:

```bash
squeue -u $USER
```

## Running on Orchid

To run the flatbug model it is highly recommended that you use GPUs on orchid.

To get set up on orchid, first find the driver versions:

```bash
nvidia-smi
```

Then [find the correct torch command](https://pytorch.org/get-started/locally/) for that CUDA version. For Cuda 12.4:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Create an environment:

```bash
source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug/"

cd amber-inferences
pip install -e .
```

Install other requirements:

```bash
conda install --yes --file requirements.txt

# install flatbug
git clone git@github.com:darsa-group/flat-bug.git
cd flat-bug
git checkout develop
pip install -e .
```

To run:

```bash
source ~/miniforge3/bin/activate
conda activate "~/conda_envs/flatbug"

#TODO: provide updated example script
# sbatch ./slurm_scripts/flatbug_test.sh
```


# Package Version

## Set up and install

```
conda create -p "~/amber/" python=3.9
conda activate "~/amber/"
```

Deployment summary

```sh
pip install -e .
```

Install flat-bug

## Running

Once everything is installed you can run the inference pipeline. The following commands are available:


### Printing the Deployments Available

This will print the deployments, and files, available in the object store:

```sh
python -m amber_inferences.cli.deployments --subset_countries 'Panama'
# or
amber-deployments --subset_countries 'Panama'
```

### Generating keys for inference

Panama example:

```sh
python -m amber_inferences.cli.generate_keys --bucket 'pan' --deployment_id 'dep000022' --output_file './keys/dep000022_keys.json'

# or

amber-keys --bucket 'pan' --deployment_id 'dep000022' --output_file './keys/dep000022_keys.json' --file_extensions 'jpeg' 'jpg'
```

UK example:

```sh
python -m amber_inferences.generate_keys --input_file './keys/solar/dep000072_keys.txt' --file_extensions 'jpg' 'jpeg' --chunk_size 100 --output_file './keys/dep000072_workload_chunks.json'

# or

amber-keys --bucket 'gbr' --deployment_id 'dep000072' --output_file './keys/dep000072_keys.txt'
```

### Performing the inferences

```sh
python3 -m amber_inferences.cli.perform_inferences \
  --chunk_id 10 \
  --json_file '../examples/dep000022_keys.json' \
  --output_dir '../data/examples/dep000022' \
  --bucket_name 'pan' \
  --credentials_file '../credentials.json' \
  --csv_file '../data/examples/dep000022.csv' \
  --species_model_path ../models/turing-costarica_v03_resnet50_2024-06-04-16-17_state.pt \
  --species_labels ../models/03_costarica_data_category_map.json \
  --binary_model_path ../models/moth-nonmoth-effv2b3_20220506_061527_30.pth \
  --localisation_model_path ../models/v1_localizmodel_2021-08-17-12-06.pt \
  --order_model_path ../models/dhc_best_128.pth \
  --order_thresholds_path ../models/thresholdsTestTrain.csv \
  --perform_inference \
  --box_threshold 0.99 \
  --save_crops



  # or

amber-inferences --chunk_id 1 \
    --json_file './examples/dep000072_subset_keys.json' \
    --output_dir './data/examples/' \
    --bucket_name 'gbr' --credentials_file './credentials.json' \
    --csv_file 'dep000072.csv' \
    --species_model_path ./models/turing-uk_v03_resnet50_2024-05-13-10-03_state.pt \
    --species_labels ./models/03_uk_data_category_map.json \
    --perform_inference \
    --remove_image \
    --save_crops
```

# For Developers


```sh
python3 -m unittest discover -s tests
```
