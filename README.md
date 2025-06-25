# Performing AMBER Inference

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

Create a conda environment:

```bash
conda create -p "~/amber/" python=3.11
conda activate "~/amber/"
```
If you are not using flatbut you can get away with python=3.9.


```sh
conda install --yes --file requirements.txt
pip install -e .
```

## If using GPU

You will need to use GPU to utilise Flatbug. To do this you will need to install the correct torch version for your CUDA version. Check the drivers:

```bash
nvidia-smi
```

Then [find the correct torch command](https://pytorch.org/get-started/locally/) for that CUDA version. For example, for Cuda 12.4:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

Create an environment:

```bash
source ~/miniforge3/bin/activate
conda activate "~/amber/"

cd amber-inferences
pip install -e .
```

## If installing Flatbug


```bash
cd ../
git clone git@github.com:darsa-group/flat-bug.git
cd flat-bug
git checkout develop
pip install -e .
cd ../amber_inferences
```



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
