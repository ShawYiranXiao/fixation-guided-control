# Fixation-Driven Image Complexity Modeling and Control for Conditional Image Generation

This repository provides code for constructing fixation distribution maps (FDMs) with DeepGazeIII and using them as spatial conditions for Stable Diffusion with ControlNet. It includes fixation-map generation and fusion, dataset preparation, training and inference scripts, comparison visualizations, and a perceptual-complexity annotation interface.

The methods explore how fixation-based conditions can guide visual emphasis and perceived image complexity. Experiments include fixation-map fusion comparisons and preliminary conditional generation examples. A controlled perceptual evaluation remains future work.

## Workflow

1. Predict the next fixation with DeepGazeIII under sampled fixation histories.
2. Cache the log-density maps and compare fusion operators.
3. Pair the fused condition with an image and caption.
4. Train or run ControlNet with Stable Diffusion.
5. Inspect generated examples; use the annotation interface to support future perceptual evaluation.

The historical `mean` operator averages **log densities**, rather than normalized probabilities. Its definition is preserved for reproducibility. [Method notes](docs/method-notes.md) explain the interpretation and limitations.

## Code layout

| Directory | Contents |
| --- | --- |
| `code/hfdatasetmaker-vislab4/` | Main archived dataset and fixation-fusion pipeline, cached-feature tools and Flask service. |
| `code/hfdatasetmaker-local/` | Local pipeline variant, including `image_classification.py`. |
| `code/scanpathanalysis-*/` | Fixation analysis and visualization tools, including historical workspace variants. |
| `code/controlnet-training/` | Modified ControlNet training example, inference and Streamlit launchers, dataset utilities and LAION preparation scripts. |
| `code/comfyui-node/` | A ComfyUI node that calls a configurable DeepGaze service. |
| `code/mapgenerator-exploratory/` | Related GAN/VAE exploration, kept separate from the ControlNet workflow. |
| `scripts/` | Resource preparation and release checks. |
| `tests/` | CPU tests for fusion behavior and selected inference utilities. |

Multiple historical variants are retained as separate directories. Start with the vislab4 dataset pipeline and the consolidated training directory. The older playground files are experimental scripts, not interchangeable APIs.

## Setup and entry points

Use a separate Python 3.11 or 3.12 environment. Install PyTorch appropriate to your hardware before setting up the research dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# External model dependency; consult its upstream usage terms.
python -m pip install "git+https://github.com/matthias-k/DeepGaze.git@874f12e1ee519860f49860638cf7f6375956d45a"
python scripts/fetch_centerbias.py --output resources/centerbias_mit1003.npy
export DEEPGAZE_CENTERBIAS_PATH="$PWD/resources/centerbias_mit1003.npy"
```

DeepGaze and base diffusion weights are external resources and may download when a model is initialized. Training datasets, participant eye-tracking records, model weights, checkpoints, original photographs, and private presentation materials are not distributed here.

Run the local complexity annotation interface after setting the image directory:

```bash
export DEEPGAZE_IMAGE_DIR=/path/to/your/images
cd code/hfdatasetmaker-local
python image_classification.py
```

For dataset processing, inference, service configuration and the full list of environment variables, see [Reproducibility](docs/reproducibility.md). The code references a pinned diffusers commit and a historical DeepGaze commit. Dependency ranges do not fully reconstruct the original environment.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python scripts/check_release.py
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

The included GitHub Actions configuration runs these checks without downloading models or using a GPU. They cover English-only source text, syntax, unsafe defaults, selected inference regressions, and preserved fusion calculations. Full training, model inference and the historical interfaces still require validation with the external resources.

## License

The original code in this repository is licensed under the **MIT License**. The modified Hugging Face `train_controlnet.py` retains its **Apache-2.0** copyright and license; its license text is included under `LICENSES/`. External model code, weights and datasets retain their own terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff).

## Contributions

- **Weihao Qiu** ([@QIUWEIHAO](https://github.com/QIUWEIHAO)): Research direction, dataset-processing infrastructure, early interactive DeepGaze analysis tools, and integration of fixation maps and image-caption pairs with the ControlNet training environment.
- **Shaw Yiran Xiao** ([@ShawYiranXiao](https://github.com/ShawYiranXiao)): Fixation-map fusion methods and comparisons, standalone fixation-feature generation, NPZ caching, comparison visualizations, the perceptual-complexity annotation interface, and generation-pipeline integration including ControlNet input formatting and data flow.

See [AUTHORS.md](AUTHORS.md) for detailed contributions.

## Acknowledgements

The methods build on DeepGazeIII, Stable Diffusion, ControlNet, and Hugging Face diffusers. DeepGaze and diffusers are referenced as external dependencies. Source repositories include [HFDatasetMaker](https://github.com/QIUWEIHAO/HFDatasetMaker), [ScanpathAnalysis](https://github.com/QIUWEIHAO/ScanpathAnalysis), and [ScanpathAnalysis-shaw](https://github.com/ShawYiranXiao/ScanpathAnalysis-shaw). See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and licenses.
