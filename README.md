# Performing AMBER Inference

This Repository contains code to download images from the JASMIN object store and perform inference to: 
- Detect and isolate objects
- Track objects
- Classify objects as moth or non-moth
- Identify the order
- Predict the species

## JASMIN Set-Up

JASMIN is a data analysis facility that provides services as a High Performance Computing Cluster and Data Centre. To use this repository on JASMIN, you will require the following services:

1. [A JASMIN account](https://help.jasmin.ac.uk/docs/getting-started/get-jasmin-portal-account/)
2. [A JASMIN login account](https://help.jasmin.ac.uk/docs/getting-started/get-login-account/): This access to the JASMIN shared services, i.e. login, transfer, scientific analysis servers, Jupyter notebook and LOTUS.
3. [JASMIN ORCHID access (required for segmentation only)](https://accounts.jasmin.ac.uk/services/additional_services/orchid/): This will give you access to the GPU cluster, called ORCHID. The GPU cluster is required to effectively segment your objects using flatbug.
4. [Generate an ssh public and private key pair and register the public key in JASMIN](https://help.jasmin.ac.uk/docs/getting-started/generate-ssh-key-pair/): This will allow you to establish an ssh key connection to the JASMIN servers.Please note, it will take a couple of hours for the ssh key-pair to sync with JASMIN’s systems.
5. [Apply for the ceh_generic UKCEH workspace (UKCEH staff only)](https://accounts.jasmin.ac.uk/services/group_workspaces/ceh_generic/): This will allow you to access the CEH shared workspace, for a greater storage size allocation to and submit jobs under the `ceh_generic` account (needed for submissions to LOTUS only).
6. A JASMIN object store ssh key. You will need this to obtain files located on the data centre. The object store used to manage the automated monitoring data is called "ami-test-o". You will need a valid key generated for yourself, or for another member. It is recommended that you do the former if you need frequent access the object store, to prevent disruption if keys are invalidated. If you want to have a key generated for yourself, contact the object store manager, [Tom August](tomaug@ceh.ac.uk).

In JASMIN, you will need to install conda. Conda is required to manage a python installation and it's python module versions. The process for the installation of conda on JASMIN is detailed [here](https://help.jasmin.ac.uk/docs/software-on-jasmin/creating-and-using-miniforge-environments/):

After successfully installing conda, you must activate it.

```bash
source ~/miniforge3/bin/activate
```

## Models

The model files are managed in the ./models subdirectory. Following this you can pass in:
- species_model: The path to the regional species model
- species_labels: The path to the species labels
- binary_model_path: The path to the binary model weights
- order_model_path: The path to the binary model weights
- order_threshold_path: The path to the binary model weights
- localisation_model_path: The path to the binary model weights

AMBER team members can find these files on [OneDrive](https://thealanturininstitute.sharepoint.com/:f:/r/sites/Automatedbiodiversitymonitoring/Shared%20Documents/General/Data/models/jasmin?csf=1&web=1&e=HgjhgA). Others can contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the model files.

### Recommended Box thresholds

There are several object detection models which can be used in this analysis. These have varying recommended confidence thresholds to define object bounding boxes. The box threshold can be altered using the `--box_threshold` argument in `slurm_scripts/array_processor.sh`. The table below outlines the recommended thresholds for some models:

| Model file name                      | Recommended box threshold |
|--------------------------------------|---------------------------|
| flatbug_*.pt  **(Default)**          | 0.0          |
| v1_localizmodel_2021-08-17-12-06.pt  | 0.99         |
| fasterrcnn_resnet50_fpn_tz53qv9v.pt  | 0.8          |

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
}
```

Contact [Katriona Goldmann](kgoldmann@turing.ac.uk) for the AWS Access and UKCEH API configs.

## Setting up the Environment

The conda environment will be built inside the GPU interactive server. Whilst the conda environment can be accessed from the scientific servers (from which you will submit your inference jobs), building it inside the GPU server will ensure that torch is configured for the correct version of cuda.

```bash
ssh -A <jasmin_username>@login-01.jasmin.ac.uk
ssh <jasmin_username>@gpuhost001.jc.rl.ac.uk
```

First create a conda environment.

```bash
conda create -p ~/amber python=3.11
conda activate ~/amber
```

Next, you need to install the correct version of torch corresponding to your cuda version. To obtain the cuda version, check the drivers. You will see the cuda version in the top righthand corner of the output table.

```bash
nvidia-smi
```

Then [find the correct torch command](https://pytorch.org/get-started/locally/) for that CUDA version. For JASMIN users, the command you should run is:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Next, you need to install timm. The installation of timm needs to be completed using pip to avoid overwriting the GPU configuration of torch.

```bash
pip install timm
```

Now, if you have not already done so, clone this repository, and checkout to the `dev` brach

```bash
git clone https://github.com/AMI-system/amber-inferences.git
cd amber-inferences
git checkout dev
cd ..
```

Install the dependancies listed inside the requirements.txt file

```bash
conda install --yes --file ~/amber-inferences/requirements.txt
```

Now clone and install flatbug in editable mode

```bash
git clone git@github.com:darsa-group/flat-bug.git
cd flat-bug
git checkout develop
pip install -e .
cd ..
```

Likewise, install the amber-inferences repository in editable mode.

```bash
pip install -e ~/amber-inferences
```

Now confirm that cuda has been successfully configured

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If TRUE, you have successfully configured cuda!

## Running

Once everything is installed you can run the inference pipeline.
There is a tutorial set up in `./examples/tutorial.ipynb` which can be used to
run and explore the pipeline. This is recommended for first time users.
The commands below outline the process for running the pipeline from command line.


#### Printing the Deployments Available

```sh
python -m amber_inferences.cli.deployments --subset_countries 'Panama'
# or
amber-deployments --subset_countries 'Panama'
```

#### Generating keys for inference

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

#### Performing the inferences

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

### Using Slurm

If you are using Slurm, there are several examples scripts for running the inference pipeline. These are located in the `./slurm_scripts` directory. For examples to run all singapore inferences, using arrays and batches:

```sh
# for a full display of jobs running
squeue -u USERNAME -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

To check on the number of output files (replace . with the path to your output directory):

```sh
find . -maxdepth 1 -mindepth 1 -type d -exec sh -c 'echo "{} : $(find "{}" -type f | wc -l)" file\(s\)' \;
```
Or to check the logs:

```
head ./logs/cri/dep000031_batch_1.out
```

# Interpreting the Results

The results of the inference will be saved in the output directory specified by the `--output_dir` argument. The output will include:
- A CSV file containing the results of the inference, including species predictions, order predictions, and bounding box coordinates. The description of the columns in the CSV file are outlined in `output_description.md`.
- A directory containing the cropped images of the detected objects, if the `--save_crops` argument is specified.
- A directory containing the original images, if the `--remove_image` argument is not specified.


# For Developers


```sh
python3 -m unittest discover -s tests
```

For coverage:

```sh
pytest --cov=src/amber_inferences tests/ --cov-report=term-missing
```
