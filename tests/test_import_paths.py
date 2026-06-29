import importlib.util
import sys
import types
from pathlib import Path

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

    module_path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_lidc_preprocessing_imports_work():
    _import_module('preprocess_cpu', 'Imaging/01_lidc-preprocessing/preprocess_cpu.py')
    _import_module('train_colab_gpu', 'Imaging/01_lidc-preprocessing/train_colab_gpu.py')
    _import_module('inference_cpu', 'Imaging/01_lidc-preprocessing/inference_cpu.py')


def test_cir_main_imports_work():
    _import_module('cir_main', 'Imaging/03_CIR/main.py')
