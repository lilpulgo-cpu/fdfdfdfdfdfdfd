# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings, importlib, sys
from packaging.version import Version
import os, re, subprocess, inspect
import numpy as np

# Unsloth currently does not work on multi  setups - sadly we are a 2 brother team so
# enabling it will require much more work, so we have to prioritize. Please understand!
# We do have a beta version, which you can contact us about!
# Thank you for your understanding and we appreciate it immensely!

# Fixes https://github.com/unslothai/unsloth/issues/1266
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

if "_VISIBLE_DEVICES" in os.environ:
    os.environ["_DEVICE_ORDER"] = "PCI_BUS_ID"
    devices = os.environ["_VISIBLE_DEVICES"]
    # Check if there are multiple  devices set in env
    if not devices.isdigit():
        first_id = devices.split(",")[0]
        warnings.warn(
            f"Unsloth: '_VISIBLE_DEVICES' is currently {devices} \n"\
            "Unsloth currently does not support multi  setups - but we are working on it!\n"\
            "Multiple  devices detected but we require a single.\n"\
            f"We will override _VISIBLE_DEVICES to first: {first_id}."
        )
        os.environ["_VISIBLE_DEVICES"] = str(first_id)
else:
    # warnings.warn("Unsloth: '_VISIBLE_DEVICES' is not set. We shall set it ourselves.")
    os.environ["_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["_VISIBLE_DEVICES"] = "0"
pass

# Reduce VRAM usage by reducing fragmentation
# And optimize pinning of memory
os.environ["PYTORCH__ALLOC_CONF"] = \
    "expandable_segments:True,"\
    "roundup_power2_divisions:[32:256,64:128,256:64,>:32]"

# [TODO] Check why some s don't work
#    "pinned_use__host_register:True,"\
#    "pinned_num_register_threads:8"

# Hugging Face Hub faster downloads
if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
pass

# Log Unsloth is being used
os.environ["UNSLOTH_IS_PRESENT"] = "1"

try:
    import torch
except ModuleNotFoundError:
    raise ImportError(
        "Unsloth: Pytorch is not installed. Go to https://pytorch.org/.\n"\
        "We have some installation instructions on our Github page."
    )
except Exception as exception:
    raise exception
pass

# We support Pytorch 2
# Fixes https://github.com/unslothai/unsloth/issues/38
torch_version = torch.__version__.split(".")
major_torch, minor_torch = torch_version[0], torch_version[1]
major_torch, minor_torch = int(major_torch), int(minor_torch)
if (major_torch < 2):
    raise ImportError("Unsloth only supports Pytorch 2 for now. Please update your Pytorch to 2.1.\n"\
                      "We have some installation instructions on our Github page.")
elif (major_torch == 2) and (minor_torch < 2):
    # Disable expandable_segments
    del os.environ["PYTORCH__ALLOC_CONF"]
pass

# Fix Xformers performance issues since 0.0.25
import importlib.util
from pathlib import Path
from importlib.metadata import version as importlib_version
from packaging.version import Version
try:
    xformers_version = importlib_version("xformers")
    if Version(xformers_version) < Version("0.0.29"):
        xformers_location = importlib.util.find_spec("xformers").origin
        xformers_location = os.path.split(xformers_location)[0]
        cutlass = Path(xformers_location) / "ops" / "fmha" / "cutlass.py"

        if cutlass.exists():
            with open(cutlass, "r+") as f:
                text = f.read()
                # See https://github.com/facebookresearch/xformers/issues/1176#issuecomment-2545829591
                if "num_splits_key=-1," in text:
                    text = text.replace("num_splits_key=-1,", "num_splits_key=None,")
                    f.seek(0)
                    f.write(text)
                    f.truncate()
                    print("Unsloth: Patching Xformers to fix some performance issues.")
                pass
            pass
        pass
    pass
except:
    pass
pass

# Torch 2.4 has including_emulation
major_version, minor_version()
SUPPORTS_BFLOAT16 = (major_version >= 8)

old_is_bf16_supported = torch.is_bf16_supported
if "including_emulation" in str(inspect.signature(old_is_bf16_supported)):
    def is_bf16_supported(including_emulation = False):
        return old_is_bf16_supported(including_emulation)
    torch.is_bf16_supported = is_bf16_supported
else:
    def is_bf16_supported(): return SUPPORTS_BFLOAT16
    torch.is_bf16_supported = is_bf16_supported
pass

# For Gradio HF Spaces?
# if "SPACE_AUTHOR_NAME" not in os.environ and "SPACE_REPO_NAME" not in os.environ:
import triton
lib_dirs = lambda: None
if Version(triton.__version__) >= Version("3.0.0"):
    try: from triton.backends.driver import lib_dirs
    except: pass
else: from triton.common.build import lib_dirs

# Try loading bitsandbytes and triton
import bitsandbytes as bnb
try:
    cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
    lib_dirs()
except:
    warnings.warn(
        "Unsloth: Running `` to link ."\
    )

    if os.path.exists(""):
        os.system("")
    elif os.path.exists("/usr/local"):
        # Sometimes bitsandbytes cannot be linked properly in Runpod for example
        possible_s = subprocess.check_out(["ls", "-al", "/usr/local"]).decode("utf-8").split("\n")
        find_ = re.compile(r"[\s](\-[\d\.]{2,})$")
        possible_s = [find_.search(x) for x in possible_s]
        possible_s = [x.group(1) for x in possible_s if x is not None]

        # Try linking  folder, or everything in local
        if len(possible_s) == 0:
            os.system("ldconfig /usr/local/")
        else:
            find_number = re.compile(r"([\d\.]{2,})")
            latest_ = np.argsort([float(find_number.search(x).group(1)) for x in possible_s])[::-1][0]
            latest_ = possible_s[latest_]
            os.system(f"ldconfig /usr/local/{latest_}")
    pass

    importlib.reload(bnb)
    importlib.reload(triton)
    try:
        lib_dirs = lambda: None
        if Version(triton.__version__) >= Version("3.0.0"):
            try: from triton.backends.driver import lib_dirs
            except: pass
        else: from triton.common.build import lib_dirs
        cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
        lib_dirs()
    except:
        warnings.warn(
            "Unsloth:  is not linked properly.\n"\
            "Try running `python -m bitsandbytes` then `python -m xformers.info`\n"\
            "We tried running `` ourselves, but it didn't work.\n"\
            "You need to run in your terminal `sudo ` yourself, then import Unsloth.\n"\
            "Also try `sudo ldconfig /usr/local/-xx.x` - find the latest  version.\n"\
            "Unsloth will still run for now, but maybe it might crash - let's hope it works!"
        )
pass

# Check for unsloth_zoo
try:
    unsloth_zoo_version = importlib_version("unsloth_zoo")
    if Version(unsloth_zoo_version) < Version("2025.2.6"):
        try:
            os.system("pip install --upgrade --no-cache-dir --no-deps unsloth_zoo")
        except:
            try:
                os.system("pip install --upgrade --no-cache-dir --no-deps --user unsloth_zoo")
            except:
                raise ImportError("Unsloth: Please update unsloth_zoo via `pip install --upgrade --no-cache-dir --no-deps unsloth_zoo`")
    import unsloth_zoo
except:
    raise ImportError("Unsloth: Please install unsloth_zoo via `pip install unsloth_zoo`")
pass

from .models import *
from .save import *
from .chat_templates import *
from .tokenizer_utils import *
from .trainer import *

# Patch TRL trainers for backwards compatibility
_patch_trl_trainer()
