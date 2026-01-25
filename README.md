# Road to SKA: Foundation Models, Embeddings, and Latent Spaces

An advanced tutorial for astrophysics students on machine learning concepts central to modern foundation models.

## Overview

This tutorial consists of four practical sessions designed to build intuition about latent spaces, embeddings, parameter-efficient fine-tuning, and real-world foundation model usage:

| Session | Topic | Type | Notebook |
|---------|-------|------|----------|
| **1A** | Autoencoders & Latent Spaces | Live | [Session1A_Autoencoders_LatentSpaces.ipynb](Session1A_Autoencoders_LatentSpaces.ipynb) |
| **1B** | Extension: MiraBest Radio Galaxies | Homework | [Session1B_Extension_MiraBest.ipynb](Session1B_Extension_MiraBest.ipynb) |
| **2A** | Embeddings-first Workflows | Live | [Session2A_Embeddings_Workflow.ipynb](Session2A_Embeddings_Workflow.ipynb) |
| **2B** | Extension: MiraBest Retrieval | Homework | [Session2B_Extension_MiraBest.ipynb](Session2B_Extension_MiraBest.ipynb) |
| **3A** | LoRA Fine-tuning | Live | [Session3A_LoRA_Finetuning.ipynb](Session3A_LoRA_Finetuning.ipynb) |
| **4A** | Generate Your Own Embeddings (Clay) | Live | [Session4A_Generate_Own_Embeddings.ipynb](Session4A_Generate_Own_Embeddings.ipynb) |
| **4B** | Advanced: TerraTorch | Homework | [Session4B_TerraTorch_Advanced.ipynb](Session4B_TerraTorch_Advanced.ipynb) |

All sessions are designed to be **CPU-friendly** and run on standard laptops. **Apple Silicon (M1/M2/M3) is supported** via MPS acceleration.

---

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate r2ska-tutorial
```

### 2. Launch Jupyter

```bash
jupyter lab
```

### 3. Open Session 1 and follow along

---

## Session Descriptions

### Session 1A: Autoencoders & Latent Spaces

**Goal:** Make latent spaces tangible.

You will:
- Train a small convolutional autoencoder on MNIST (or Fashion-MNIST)
- Extract latent vectors and visualise with PCA and UMAP
- Explore latent interpolation between samples
- Perform latent traversals (vary one dimension)
- Train a Variational Autoencoder (VAE) and compare

**Deliverable:** A figure showing a latent manifold + reconstructions + interpolations.


### Session 1B: Extension - MiraBest Radio Galaxies

**Goal:** Apply autoencoder concepts to real radio astronomy data.

You will:
- Load MiraBest radio galaxy images
- Train an autoencoder on radio galaxy morphologies
- Explore the latent space of FRI/FRII galaxy classifications

**Deliverable:** Latent space visualisation of radio galaxy morphologies.


### Session 2A: Embeddings-first Workflows with Clay

**Goal:** Treat a foundation model as an "embedding generator" and do downstream tasks cheaply.

You will:
- Load pre-computed embeddings from the Clay geospatial foundation model
- Train lightweight classifiers (Random Forest, Logistic Regression) on embeddings
- Build a similarity search index for retrieval

**Deliverable:** A classifier trained on embeddings + a retrieval demo (top-k similar patches).

**Key insight:** You can accomplish a lot *without* fine-tuning the foundation model.


### Session 2B: Extension - MiraBest Retrieval

**Goal:** Apply embedding-based retrieval to radio astronomy.

You will:
- Use pre-trained embeddings for MiraBest radio galaxies
- Build a similarity search system for radio galaxy morphologies
- Explore retrieval-based classification

**Deliverable:** A retrieval demo finding similar radio galaxies.


### Session 3A: Parameter-Efficient Fine-Tuning with LoRA

**Goal:** Understand what LoRA is doing and when to use it.

You will:
- Train a base CNN classifier
- Adapt it to a shifted domain (rotated + noisy images) using:
  - Full fine-tuning
  - Head-only fine-tuning
  - LoRA fine-tuning
- Compare accuracy, training time, and trainable parameters

**Deliverable:** A comparison table showing linear probe vs LoRA vs full fine-tune.


### Session 4A: Generate Your Own Embeddings with Clay

**Goal:** Learn to generate embeddings from scratch using a real foundation model.

You will:
- Set up the Clay foundation model environment
- Process your own imagery through the foundation model
- Generate and save embeddings for downstream use

**Deliverable:** Custom embeddings ready for classification or retrieval.


### Session 4B (Advanced): TerraTorch + Prithvi-EO-2.0

**Goal:** Experience a realistic geospatial fine-tuning workflow.

This optional session uses TerraTorch to fine-tune Prithvi-EO-2.0 for flood segmentation. It requires additional dependencies (GDAL, rasterio, terratorch) and is best suited for participants with:
- A working GDAL installation
- GPU access (recommended)
- Reliable internet for data downloads

---

## Prerequisites

### Required packages

The `environment.yml` includes everything needed for Sessions 1-3:

- **Core:** numpy, pandas, matplotlib, scikit-learn
- **Deep learning:** PyTorch, torchvision
- **Visualisation:** umap-learn, tqdm
- **Data:** pyarrow, shapely, requests, Pillow

### Optional packages (for advanced sections)

Uncomment in `environment.yml` if needed:

- **STAC workflows:** pystac-client, stackstac, planetary-computer
- **TerraTorch:** gdal, rasterio, terratorch

### Hardware requirements

| Session | CPU | GPU/MPS | RAM | Notes |
|---------|-----|---------|-----|-------|
| 1A, 1B | Required | Optional | 4GB+ | Faster with GPU or Apple Silicon (MPS) |
| 2A, 2B | Required | Optional | 4GB+ | Main cost is data download |
| 3A | Required | Optional | 4GB+ | Runs fine on CPU |
| 4A | Required | Recommended | 8GB+ | Clay model benefits from GPU |
| 4B | Required | Recommended | 8GB+ | GPU strongly recommended |

**Apple Silicon users:** The notebooks automatically detect and use MPS (Metal Performance Shaders) for GPU acceleration on M1/M2/M3 Macs.

---

## Troubleshooting

### Environment creation fails

Try creating a minimal environment first:

```bash
conda create -n r2ska-tutorial python=3.10 numpy pandas matplotlib scikit-learn -y
conda activate r2ska-tutorial
conda install pytorch torchvision -c pytorch -y
pip install jupyterlab umap-learn pyarrow shapely tqdm
```

### UMAP not working

UMAP is optional. PCA visualisations work without it. If you want UMAP:

```bash
pip install umap-learn
```

### Data download issues (Session 2)

Each notebook includes an **offline fallback** section that uses built-in sklearn datasets (e.g., `load_digits()`). You can learn the workflow pattern without external downloads.

### GDAL/rasterio issues (Session 4B)

GDAL can be tricky to install. Options:

1. Use conda: `conda install -c conda-forge gdal rasterio`
2. Skip Session 4B and use Session 3A instead
3. Use the pre-trained model for inference only (see fallback in notebook)

---

## Directory Structure

```
R2SKA_Advanced_Tutorial/
├── README.md
├── environment.yml
├── environment-clay.yml         # For Session 4A (Clay embeddings)
├── setup_clay_env.sh            # Helper script for Clay setup
├── Session1A_Autoencoders_LatentSpaces.ipynb
├── Session1B_Extension_MiraBest.ipynb
├── Session2A_Embeddings_Workflow.ipynb
├── Session2B_Extension_MiraBest.ipynb
├── Session3A_LoRA_Finetuning.ipynb
├── Session4A_Generate_Own_Embeddings.ipynb
├── Session4B_TerraTorch_Advanced.ipynb
└── _archive/                    # Original notebook versions
```

---

## Key References

### Foundational papers

- Hinton & Salakhutdinov (2006), [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/absps/science.pdf)
- Kingma & Welling (2013), [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Foundation models

- [Clay Foundation Model](https://clay-foundation.github.io/model/)
- [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- [TerraTorch](https://github.com/terrastackai/terratorch)

### Datasets

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- [MiraBest (radio galaxies)](https://doi.org/10.5281/zenodo.4288837)
- [Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11)

---

## License

Tutorial materials provided for educational use at the Road to SKA workshop.
