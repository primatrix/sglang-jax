"""Install the pinned SGLang text-module dependencies in an isolated GPU job.

Optional audio/video packages are excluded from these text-only module probes.
Torchvision is required by SGLang core imports and stays installed. In particular,
the source manifest pins torchaudio to a different torch release; don't allow that
optional package to select another torch or force an unsatisfiable installation.
"""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1])
manifest = tomllib.loads((root / "python/pyproject.toml").read_text())
skip = {"torchaudio", "torchcodec", "timm", "av", "decord2", "soundfile"}
requirements = [
    s for s in manifest["project"]["dependencies"] if re.split(r"[\[<>=!~; ]", s)[0] not in skip
]
subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchvision", "torchaudio"], check=True
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "uv", "setuptools", "wheel", "packaging", "ml_dtypes"],
    check=True,
)
path = Path("/tmp/sglang-text-requirements.txt")
path.write_text("\n".join(requirements) + "\n")
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--break-system-packages",
        "--prerelease=allow",
        "-r",
        str(path),
    ],
    check=True,
)
subprocess.run(
    [
        "uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--break-system-packages",
        "--no-deps",
        "--no-build-isolation",
        "-e",
        str(root / "python"),
    ],
    check=True,
)
