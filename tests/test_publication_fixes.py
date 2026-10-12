# SPDX-License-Identifier: MIT
"""Exercise maintenance changes without downloading models or importing UIs."""

import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_functions(relative, names, namespace):
    tree = ast.parse((ROOT / relative).read_text())
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if len(definitions) != len(names):
        raise AssertionError("Expected function definitions were not found")
    code = compile(ast.Module(body=definitions, type_ignores=[]), relative, "exec")
    exec(code, namespace)
    return namespace


class InferenceTests(unittest.TestCase):
    def fake_torch(self, cuda=False):
        class Generator:
            def __init__(self, device):
                self.device = device
                self.seed = None

            def manual_seed(self, seed):
                self.seed = seed
                return self

        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            float16="float16",
            float32="float32",
            Generator=Generator,
        )

    def test_seed_zero_is_forwarded(self):
        self.check_seed(0)

    def test_unspecified_seed_is_none(self):
        self.check_seed(None)

    def check_seed(self, seed):
        calls = []
        image = SimpleNamespace(save=lambda path: None)

        def pipeline(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(images=[image])

        namespace = {
            "torch": self.fake_torch(),
            "load_pipeline": lambda *args: pipeline,
            "load_conditioning_image": lambda path: "condition",
            "os": SimpleNamespace(
                makedirs=lambda *a, **k: None, path=SimpleNamespace(join=lambda *a: "/".join(a))
            ),
        }
        load_functions(
            "code/controlnet-training/inference_controlnet.py", {"run_inference"}, namespace
        )
        args = SimpleNamespace(
            base_model_path="base",
            controlnet_checkpoint="model",
            cond_image_path="condition.png",
            prompt="example",
            steps=1,
            seed=seed,
            output_dir="outputs",
        )
        namespace["run_inference"](args)
        generator = calls[0]["generator"]
        if seed is None:
            self.assertIsNone(generator)
        else:
            self.assertEqual(generator.seed, 0)

    def test_cpu_uses_float32_without_offload(self):
        self.check_device(False)

    def test_cuda_uses_float16_with_offload(self):
        self.check_device(True)

    def check_device(self, cuda):
        calls = {}
        pipe = SimpleNamespace(scheduler=SimpleNamespace(config={}))
        pipe.to = lambda device: calls.update(device=device)
        pipe.enable_model_cpu_offload = lambda: calls.update(offload=True)

        def build_pipeline(path, **kwargs):
            calls.update(kwargs)
            return pipe

        namespace = {
            "torch": self.fake_torch(cuda),
            "ControlNetModel": SimpleNamespace(from_pretrained=lambda *a, **k: "controlnet"),
            "StableDiffusionControlNetPipeline": SimpleNamespace(from_pretrained=build_pipeline),
            "UniPCMultistepScheduler": SimpleNamespace(from_config=lambda config: "scheduler"),
        }
        load_functions(
            "code/controlnet-training/inference_controlnet.py", {"load_pipeline"}, namespace
        )
        namespace["load_pipeline"]("base", "model")
        self.assertEqual(calls["torch_dtype"], "float16" if cuda else "float32")
        self.assertEqual(calls.get("offload", False), cuda)
        self.assertNotIn("safety_checker", calls)


class FusionAndCacheTests(unittest.TestCase):
    def test_constant_normalization_is_finite(self):
        namespace = load_functions(
            "code/hfdatasetmaker-vislab4/processors/deepgaze_feature.py",
            {"normalize_image_array_8bit"},
            {"np": np},
        )
        result = namespace["normalize_image_array_8bit"](np.ones((4, 4)))
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros((4, 4), dtype=np.uint8))

    def test_historical_mean_still_averages_log_values(self):
        namespace = load_functions(
            "code/hfdatasetmaker-vislab4/processors/deepgaze_feature.py",
            {"fusion_mean"},
            {"np": np},
        )
        values = np.array([[[-1.0, -3.0]], [[-2.0, -4.0]]])
        np.testing.assert_allclose(namespace["fusion_mean"](values), [[-1.5, -3.5]])

    def test_unsafe_cache_key_is_rejected_before_model_work(self):
        import re

        namespace = {
            "np": np,
            "re": re,
            "os": SimpleNamespace(path=SimpleNamespace(join=lambda *a: "/".join(a))),
        }
        load_functions(
            "code/hfdatasetmaker-vislab4/processors/deepgaze_feature.py",
            {"process_image_with_deepgaze_batch"},
            namespace,
        )
        with self.assertRaisesRegex(ValueError, "safe filename"):
            namespace["process_image_with_deepgaze_batch"](
                model=None,
                centerbias_template=None,
                image=np.zeros((4, 4, 3)),
                num_points=4,
                batch_size=1,
                total_iterations=1,
                dataset_key="../../outside",
                npz_output_dir="outputs",
            )

    def test_upload_requires_destination(self):
        namespace = load_functions(
            "code/hfdatasetmaker-vislab4/upload_dataset.py", {"upload_to_huggingface"}, {}
        )
        with self.assertRaisesRegex(ValueError, "HF_USERNAME"):
            namespace["upload_to_huggingface"](None, None, None, None)


class CheckpointTests(unittest.TestCase):
    def resolve(self, path):
        namespace = load_functions(
            "code/controlnet-training/checkpoints.py",
            {"resolve_controlnet_checkpoint"},
            {"Path": Path},
        )
        return namespace["resolve_controlnet_checkpoint"](path)

    def test_final_safetensors_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text("{}")
            (model / "diffusion_pytorch_model.safetensors").write_bytes(b"test")
            self.assertEqual(self.resolve(model), model)

    def test_nested_binary_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "controlnet"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "diffusion_pytorch_model.bin").write_bytes(b"test")
            self.assertEqual(self.resolve(root), model)

    def test_config_without_weights_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "weights"):
                self.resolve(root)


if __name__ == "__main__":
    unittest.main()
