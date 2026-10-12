# Reproducing the research workflow

The public release contains cleaned source variants from the 2024-2025 prototype. It uses external model code, weights and datasets. The original environments and all historical interfaces have not been reconstructed as a single validated GPU installation.

## Environment and resources

Follow the setup commands in the README. The core pipeline needs PyTorch, DeepGazeIII, the center-bias template and image data. Some analysis or upstream training utilities may additionally need `pysaliency` or `lmdb`; Tkinter is supplied by your Python installation or operating system.

The source files import `deepgaze_pytorch`; its pinned external checkout is installed separately. Set `DEEPGAZE_CENTERBIAS_PATH` to an absolute path to the downloaded template before running model code. The downloader verifies SHA-256 `051645b7da47884c138e81f28a4afbadd99b6c3262eaf086434a9626e3578be9` against the archived template.

## Dataset processing

The main script is `code/hfdatasetmaker-vislab4/process_dataset.py`. It loads a directory created by Hugging Face `datasets.save_to_disk`, expecting fields such as `key`, `image_downloaded` and `caption`. Its default JSON-looking path is historical; pass a real saved dataset directory explicitly.

```bash
cd code/hfdatasetmaker-vislab4
python process_dataset.py \
  --dataset_path /path/to/saved_dataset \
  --output_dir /path/to/new_output/parquet_chunks \
  --buffer_size 4
```

Use a new cache directory for each prediction configuration. The historical training dataset identifier is [qiuweihao/laion_4_to_3_deepgaze](https://huggingface.co/datasets/qiuweihao/laion_4_to_3_deepgaze). Its current contents and access are managed by its owner and are not a substitute for recording the exact dataset revision used in a new experiment.

Uploading is separate from local processing. Use the upload utility or the processing script's `--upload` flag, and set `HF_TOKEN` and `HF_USERNAME`. The older `main.py` and `main_playground.py` examples require `HF_UPLOAD=1` for their upload call and are retained as historical interfaces. Existing output is not silently deleted by `main.py`.

## Training and inference

Work from `code/controlnet-training/`. The training launch scripts expose `MODEL_DIR`, `OUTPUT_DIR` and `DATASET_NAME` as environment overrides. The historical launch parameters use a resolution argument of 512, learning rate `4e-6`, and a target of 50,000 steps. The modified training script also contains a fixed 384 by 512 image transform; check that behavior before changing dataset aspect ratios or interpreting the resolution argument.

The original validation photographs and condition maps are not distributed. Supply your own validation images at the paths named by the launch script, or edit the validation arrays in your working copy.

```bash
cd code/controlnet-training
python inference_controlnet.py \
  --base_model_path /path/to/available_sd15_model \
  --controlnet_checkpoint /path/to/controlnet_model \
  --cond_image_path /path/to/condition.png \
  --prompt "a group of people jumping on a beach" \
  --output_dir /path/to/new_outputs \
  --steps 20 --seed 0
```

The public inference utility preserves seed zero, uses float32 on CPU and applies model offload only for CUDA. The Streamlit inference interface accepts directories containing final ControlNet models or checkpoints with a `controlnet/` subdirectory and supports `.bin` and `.safetensors` weights.

## Interfaces and configuration

| Setting | Purpose |
| --- | --- |
| `DEEPGAZE_CENTERBIAS_PATH` | Center-bias template path. |
| `DEEPGAZE_IMAGE_DIR` | Image directory for the annotation interface. |
| `DEEPGAZE_DATA_ROOT` | Root for local data downloads. |
| `DEEPGAZE_NPZ_ROOT` | Prediction-cache root for comparison tools. |
| `DEEPGAZE_OUTPUT_ROOT` | Local comparison output root. |
| `DEEPGAZE_SERVICE_URL` | ComfyUI service endpoint; defaults to the local machine. |
| `DEEPGAZE_REQUEST_TIMEOUT` | ComfyUI request timeout in seconds. |
| `HF_USERNAME`, `HF_REPO_NAME`, `HF_TOKEN` | Explicit dataset-upload destination and credentials. |
| `HF_UPLOAD` | Set to `1` to enable upload in the older examples. |

From the dataset pipeline directory, `python deepgaze_service.py` starts the local Flask service on port 5234. The ComfyUI node can be installed into a separately managed ComfyUI environment. The analysis variants use Streamlit and expect the relevant external datasets and folder layouts.

## Publication changes and validation

Comments, docstrings and human-facing messages are English. Private paths and the lab endpoint have been replaced with local defaults or environment settings. The release fixes the seed-zero handling, CPU inference dtype, undefined Streamlit inference variables, unsafe cache filenames, missing keys in the NPZ generator and unconditional uploads in historical examples. It removes unreachable leftover validation code from the training script and adds formatting, lint, source and CPU regression checks.

These maintenance changes preserve the historical fusion operators. [source-map.json](source-map.json) records the archived file hashes and publication copies. The original lab code and private archive remain unchanged.
