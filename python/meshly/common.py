# Common type definitions
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Union


PathLike = Union[str, Path]

# Type for asset provider: either a dict or a callable that fetches by checksum
# Supports both sync and async fetch functions
# The callable can return None to indicate the asset should be read from cache
AssetFetcher = Callable[[str], Union[bytes, None, Awaitable[Optional[bytes]]]]
AssetProvider = Union[Dict[str, bytes], AssetFetcher]
