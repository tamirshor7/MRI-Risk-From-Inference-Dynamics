import importlib
import importlib.util
import runpy
import sys
import os
from pathlib import Path

import tensorflow as tf
import tensorflow.python.distribute as distribute_package
from tensorflow.python.distribute import distribute_lib

vendor = Path(
    os.environ["MRI_RISK_XPDNET_VENDOR"]
).expanduser().resolve(strict=True)

sys.path.insert(0, str(vendor))

if not hasattr(distribute_lib, "has_strategy"):
    raise RuntimeError("TensorFlow distribution API is incompatible")

distribute_package.distribution_strategy_context = distribute_lib
sys.modules[
    "tensorflow.python.distribute.distribution_strategy_context"
] = distribute_lib

path = vendor / "fastmri_recon/models/training/compile.py"
source = path.read_text()

old_import = "import tensorflow_addons as tfa\n"
old_body = "    opt_kwargs = {}\n"

if source.count(old_import) != 1 or source.count(old_body) != 1:
    raise RuntimeError("Unexpected upstream compile module")

source = source.replace(old_import, "")
source = source.replace(
    old_body,
    "    import tensorflow_addons as tfa\n" + old_body,
)

name = "fastmri_recon.models.training.compile"
parent = importlib.import_module("fastmri_recon.models.training")

spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module

exec(compile(source, str(path), "exec"), module.__dict__)
parent.compile = module

from fastmri_recon.models.subclassed_models.xpdnet import XPDNet
from fastmri_recon.models.subclassed_models.denoisers.proposed_params import get_model_specs
from fastmri_recon.data.utils.masking.gen_mask_tf import gen_mask_equidistant_tf
from fastmri_recon.data.utils.multicoil.smap_extract import extract_smaps

print("TENSORFLOW=", tf.__version__, flush=True)
print("DISTRIBUTION_COMPAT=PASS", flush=True)
print("TRAINING_IMPORT_DEFERRED=PASS", flush=True)
print("XPDNET_IMPORT=PASS", flush=True)

if len(sys.argv) > 1:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
