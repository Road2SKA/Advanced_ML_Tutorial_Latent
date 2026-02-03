# Road to SKA: Foundation Models, Embeddings, and Latent Spaces

This tutorial consists of four practical sessions designed to build intuition
about latent spaces, embeddings, parameter-efficient fine-tuning, and real-world
foundation model usage. Sessions 1-3 will be covered live, while Session 4 is an
extension to support Project H - *Flood Mapping from Orbit*.


| Session | Topic | Type | Notebook |
|---------|-------|------|----------|
| **1A** | Autoencoders & Latent Spaces | Live | [Session1A_Autoencoders_LatentSpaces.ipynb](Session1A_Autoencoders_LatentSpaces.ipynb) |
| **1B** | Extension: MiraBest Radio Galaxies | Live | [Session1B_Extension_MiraBest.ipynb](Session1B_Extension_MiraBest.ipynb) |
| **2A** | Embeddings-first Workflows | Live | [Session2A_Embeddings_Workflow.ipynb](Session2A_Embeddings_Workflow.ipynb) |
| **2B** | Extension: MiraBest Retrieval | Live | [Session2B_Extension_MiraBest.ipynb](Session2B_Extension_MiraBest.ipynb) |
| **3A** | LoRA Fine-tuning | Live | [Session3A_LoRA_Finetuning.ipynb](Session3A_LoRA_Finetuning.ipynb) |
| **4A** | Generate Your Own Embeddings (Clay) | Project H | [Session4A_Generate_Own_Embeddings.ipynb](Session4A_Generate_Own_Embeddings.ipynb) |

Sessions 1-3 are designed to be CPU-friendly and run on standard laptops, while
a GPU or Apple Silicon (M1/M2/M3/M4) is recommended for Session 4A.

---

## Run in Google Colab (Recommended for Beginners)

The quickest way to get started is using **Google Colab** — no installation required!

| Session | Topic | Colab Link |
|---------|-------|------------|
| **1A** | Autoencoders & Latent Spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Road2SKA/Advanced_ML_Tutorial_Latent/blob/colab/Session1A_Autoencoders_LatentSpaces.ipynb) |
| **1B** | Extension: MiraBest Radio Galaxies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Road2SKA/Advanced_ML_Tutorial_Latent/blob/colab/Session1B_Extension_MiraBest.ipynb) |
| **2A** | Embeddings-first Workflows | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Road2SKA/Advanced_ML_Tutorial_Latent/blob/colab/Session2A_Embeddings_Workflow.ipynb) |
| **2B** | Extension: MiraBest Retrieval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Road2SKA/Advanced_ML_Tutorial_Latent/blob/colab/Session2B_Extension_MiraBest.ipynb) |
| **3A** | LoRA Fine-tuning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Road2SKA/Advanced_ML_Tutorial_Latent/blob/colab/Session3A_LoRA_Finetuning.ipynb) |
| **4A** | Generate Your Own Embeddings | ⚠️ **Local only** — [see setup instructions](#session-4a-generate-your-own-embeddings-with-clay) |

### Colab Tips

- **Enable GPU** (optional but faster): `Runtime` → `Change runtime type` → `GPU`
- **Session timeout**: Colab disconnects after ~90 minutes of inactivity. Save your work!
- **Persistent storage**: Use Google Drive mounting if you want to save data between sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

---

## Quick Start (Local Installation)

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate r2ska-tutorial
```

### 2. Launch Jupyter

```bash
jupyter lab
```

### 3. Open Session 1A and follow along

---

## Session Descriptions

### Session 1A: Autoencoders & Latent Spaces

**Goal:** Make latent spaces tangible.

- Train a small convolutional autoencoder on MNIST (or Fashion-MNIST)
- Extract latent vectors and visualise with PCA and UMAP
- Explore latent interpolation between samples
- Perform latent traversals (vary one dimension)
- Train a Variational Autoencoder (VAE) and compare


### Session 1B: Extension - MiraBest Radio Galaxies

**Goal:** Apply autoencoder concepts to real radio astronomy data.

- Load MiraBest radio galaxy images
- Train an autoencoder on radio galaxy morphologies
- Explore the latent space of FRI/FRII galaxy classifications


### Session 2A: Embeddings-first Workflows with Clay

**Goal:** Treat a foundation model as an "embedding generator" and do downstream
tasks cheaply.

- Load pre-computed embeddings from the Clay geospatial foundation model
- Train lightweight classifiers (Random Forest, Logistic Regression) on embeddings
- Build a similarity search index for retrieval


### Session 2B: Extension - MiraBest Retrieval

**Goal:** Apply embedding-based retrieval to radio astronomy.

- Use pre-trained embeddings for MiraBest radio galaxies
- Build a similarity search system for radio galaxy morphologies
- Explore retrieval-based classification


### Session 3A: Parameter-Efficient Fine-Tuning with LoRA

**Goal:** Understand what LoRA is doing and when to use it.

- Train a base CNN classifier
- Adapt it to a shifted domain (rotated + noisy images) using:
  - Full fine-tuning
  - Head-only fine-tuning
  - LoRA fine-tuning
- Compare accuracy, training time, and trainable parameters


### Session 4A: Generate Your Own Embeddings with Clay

**Goal:** Learn to generate embeddings from scratch using a real foundation model.

- Set up the Clay foundation model environment
- Process your own imagery through the foundation model
- Generate and save embeddings for downstream use

> ⚠️ **Local execution required** — This notebook cannot run on Google Colab due to:
> - Clay requires Python 3.11+ (Colab has 3.10)
> - Large checkpoint download (1.2 GB)
> - GDAL/rasterio compilation issues
>
> **Setup:** Use the dedicated Clay environment:
> ```bash
> conda env create -f environment-clay.yml
> conda activate r2ska-clay
> ```


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

Each notebook includes an **offline fallback** section that uses built-in sklearn
datasets (e.g., `load_digits()`). You can learn the workflow pattern without
external downloads.

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
└── Session4A_Generate_Own_Embeddings.ipynb
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
