"""Architecture plugins. Importing this package triggers each module's
`register(...)` call so the registry is populated by the time anyone calls
`build_compiler`. Add a new architecture = add one file here + one import
line below."""

from . import llama  # noqa: F401
from . import qwen  # noqa: F401
from . import deepseek  # noqa: F401
