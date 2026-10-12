# Third party software and resources

| Resource | Distribution and license scope |
| --- | --- |
| Hugging Face diffusers | An external Apache-2.0 dependency, pinned to `dcd77ce22273708294b7b9c2f7f0a4e45d7a9f33`. The project's modified `train_controlnet.py` preserves its original copyright and license, with a copy of [Apache-2.0](LICENSES/Apache-2.0.txt). |
| DeepGaze PyTorch | External dependency maintained by Matthias Kummerer and collaborators. Historical reference commit: `874f12e1ee519860f49860638cf7f6375956d45a`. No explicit license file was present in the archived source or upstream repository checked for this release. Its code, upstream notebooks, center-bias template and model weights are not redistributed or assigned this repository's MIT License. |
| Stable Diffusion and ControlNet weights | External models. Obtain the weights from their providers and follow their applicable terms. The historical base-model identifier is `runwayml/stable-diffusion-v1-5`; availability and access depend on the provider. |
| MIT1003, LAION and other research data | Dataset identifiers and loading code are retained for research reproducibility. Original photographs, bulk datasets, participant gaze records, annotation results and checkpoints are not included. |
| ComfyUI and TouchDesigner | External host applications. This repository includes the project-authored ComfyUI integration node. The private archive retains the original TouchDesigner projects and presentation materials; they are not part of this source release. |

Original repositories: [DeepGaze](https://github.com/matthias-k/DeepGaze), [diffusers](https://github.com/huggingface/diffusers), [HFDatasetMaker](https://github.com/QIUWEIHAO/HFDatasetMaker), and [ScanpathAnalysis](https://github.com/QIUWEIHAO/ScanpathAnalysis).

The resource downloader fetches the center-bias template directly from the original DeepGaze release and verifies the checksum recorded in the project archive. Running the downloader does not change the template's terms or include it in this repository's license.
