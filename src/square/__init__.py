import os
from pathlib import Path
from typing import Final

from gym.envs.registration import register

__version__ = "0.1.0"

PATH_MODULE_FOLDER: Final[Path] = Path(__file__).resolve().parent
PATH_PACKAGE_FOLDER: Final[Path] = PATH_MODULE_FOLDER.parent

PATH_GENERATED_FOLDER: Final[str] = os.path.join(PATH_PACKAGE_FOLDER, "generated")
PATH_RESOURCES_FOLDER: Final[str] = os.path.join(PATH_MODULE_FOLDER, "resources")
PATH_SRC_RESOURCES_FOLDER: Final[str] = os.path.join(
    PATH_PACKAGE_FOLDER, "src", "wordle", "resources"
)

register(id="Square-v0", entry_point="square.square:SquareEnv")
