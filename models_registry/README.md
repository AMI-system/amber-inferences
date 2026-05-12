rm README.md && cat > README.md <<'EOF'
# AMBER Species Model Registry

## Overview

This repository contains a curated registry of machine learning models used within the AMBER inference pipeline for automated insect monitoring.

This public repository currently includes:

- Species classification models
- Binary moth / non-moth classification model
- Associated category maps
- Model metadata and provenance documentation

The repository is intended to support reproducibility, transparency, and model management across AMBER deployments.

---

## AMBER Inference Pipeline Context

The full AMBER pipeline processes raw insect trap imagery through multiple stages:

1. Localisation (object detection)
2. Binary classification (moth / non-moth)
3. Order classification
4. Species classification
5. Tracking (post-processing)

This repository contains only the binary and species classification components.

The localisation and order classification components are maintained separately in their own repositories.

---

## Repository Structure

    models_registry/
    ├── binary_classifier/
    └── species_classifier/

Each model contains:

    weights/
    metadata/model.yaml

The metadata files document:

- model provenance
- associated category maps
- pipeline usage
- versioning
- archival status
- source information

---

## Included Models

### Binary Classification

#### moth_nonmoth

EfficientNetV2-B3 moth / non-moth classifier.

---

### Species Classification

Regional species classifiers currently include:

- Anguilla
- Costa Rica
- Japan
- Kenya / Uganda
- Madagascar
- Namibia
- Nigeria
- Singapore
- Thailand
- United Kingdom

The active UK model is:

- turing_uk_v03_resnet50

---

## Related Repositories

### Localisation / Object Detection

The active localisation model (flat_bug) is maintained separately:

https://github.com/darsa-group/flat-bug

---

### Order Classification

The order-level classification component is maintained separately and is associated with collaborative work involving Aarhus University and related insect monitoring projects:

https://github.com/kimbjerge/MCC24-trap

---

### AMBER Inference Framework

The broader AMBER inference framework is available here:

https://github.com/AMI-system/amber-inferences

---

## Notes

- The AMBER pipeline can be executed by specifying only a species model.
- Upstream models (localisation, binary, order) are automatically loaded by the inference framework.
- Models are stored independently and are not bundled together.

---

## Provenance

Where possible, model origins and usage have been reconstructed using:

- original file locations
- pipeline scripts
- OneDrive history
- output validation
- deployment usage

---

## Status Labels

- active → currently used in production pipeline
- legacy → retained for reference but no longer active
- experimental → development or testing model