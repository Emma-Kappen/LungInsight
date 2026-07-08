import importlib.util
import sys
import types
from pathlib import Path

import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def _install_stub(module_name, **attributes):
    module = types.ModuleType(module_name)
    for name, value in attributes.items():
        setattr(module, name, value)
    sys.modules[module_name] = module
    return module


def _import_module(module_name, relative_path):
    _install_stub('wandb')
    _install_stub('pylidc')
    _install_stub('pytorch_grad_cam', GradCAMPlusPlus=object)
    _install_stub('pytorch_grad_cam.utils')
    _install_stub(
        'pytorch_grad_cam.utils.model_targets',
        ClassifierOutputTarget=type('ClassifierOutputTarget', (), {'__init__': lambda self, *a, **k: None}),
    )
    _install_stub(
        'cir_multihead_pipeline',
        FEATURE_NAMES=[],
        PATCH_SIZE=64,
        _normalize_feature=lambda *args, **kwargs: None,
        _get_feature_value=lambda *args, **kwargs: None,
    )

    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_sparse_lung_mask_falls_back_to_full_volume():
    module = _import_module('detect_candidates_cpu', 'Imaging/detect_candidates_cpu.py')

    sparse_mask = np.zeros((8, 8, 8), dtype=bool)
    sparse_mask[3, 3, 3] = True

    resolved = module._resolve_detection_mask(sparse_mask, sparse_mask.shape)

    assert resolved.shape == sparse_mask.shape
    assert resolved.all()


def test_segment_lungs_keeps_internal_air_components():
    module = _import_module('detect_candidates_cpu', 'Imaging/detect_candidates_cpu.py')

    volume = np.zeros((40, 40, 40), dtype=np.float32)
    z, y, x = np.indices(volume.shape)
    lung_a = (z - 20) ** 2 + (y - 20) ** 2 + (x - 18) ** 2 <= 8 ** 2
    lung_b = (z - 20) ** 2 + (y - 20) ** 2 + (x - 22) ** 2 <= 8 ** 2
    volume[lung_a | lung_b] = -1000.0

    lung_mask = module.segment_lungs(volume)

    assert lung_mask.sum() > 0
    assert lung_mask.sum() > 0.5 * (lung_a | lung_b).sum()


def test_inference_cpu_saves_candidate_results(tmp_path):
    module = _import_module('inference_cpu', 'Imaging/inference_cpu.py')

    output_dir = tmp_path / 'inference_out'
    output_dir.mkdir()
    patch = np.zeros((64, 64, 64), dtype=np.float32)
    probs = {name: 0.25 for name in ['spiculation', 'lobulation', 'calcification', 'margin', 'texture', 'sphericity', 'subtlety', 'malignancy']}
    base = np.arange(64 * 64 * 64, dtype=np.float32).reshape(64, 64, 64)
    heatmaps = {name: base / (64 * 64 * 64) for name in probs}

    result_path = module.save_candidate_results(
        output_dir,
        'patient_cand000',
        patch,
        probs,
        heatmaps,
        center_zyx=(4, 5, 6),
    )

    assert result_path.endswith('_results.npz')
    assert (output_dir / 'candidate_results_manifest.csv').exists()
    manifest = pd.read_csv(output_dir / 'candidate_results_manifest.csv')
    assert manifest.iloc[0]['candidate_id'] == 'patient_cand000'
    assert manifest.iloc[0]['spiculation_score'] == 0.25

    payload = np.load(result_path)
    heatmap = payload['spiculation_heatmap']
    assert np.isfinite(heatmap).all()
    assert np.std(heatmap) > 0


def test_generate_characteristic_heatmaps_produces_distinct_heads():
    sys.path.insert(0, str(ROOT / 'Imaging'))
    module_path = ROOT / 'Imaging' / 'cir_multihead_pipeline.py'
    spec = importlib.util.spec_from_file_location('cir_multihead_pipeline_real', module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['cir_multihead_pipeline_real'] = module
    spec.loader.exec_module(module)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = torch.nn.Identity()

        def forward(self, x):
            act = torch.cat([x + 1.0, x * 2.0 + 3.0], dim=1)
            act = self.layer4(act)
            spatial = torch.arange(act.shape[2] * act.shape[3] * act.shape[4], dtype=act.dtype, device=act.device).reshape(1, 1, *act.shape[2:])
            head1 = (act[:, :1] * (spatial + 1.0)).sum()
            head2 = (act[:, 1:] * (spatial + 3.0)).sum()
            return {
                'spiculation': head1.unsqueeze(0),
                'malignancy': head2.unsqueeze(0),
            }

    model = DummyModel()
    x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    heatmaps = module.generate_characteristic_heatmaps(model, x, device='cpu', target_layer='layer4')

    assert heatmaps['spiculation'].shape == (8, 8, 8)
    assert heatmaps['malignancy'].shape == (8, 8, 8)
    assert np.isfinite(heatmaps['spiculation']).all()
    assert np.isfinite(heatmaps['malignancy']).all()
    assert not np.allclose(heatmaps['spiculation'], heatmaps['malignancy'])
