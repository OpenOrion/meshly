"""Constants for Packable zip file format."""


class ExportConstants:
    """Standard file paths and extensions for the Packable zip format."""

    EXPORT_TIME = (2020, 1, 1, 0, 0, 0)
    """Fixed date_time for deterministic zip output (2020-01-01 00:00:00)"""

    DATA_FILE = "data.json"
    """Instance data with $ref references to assets."""

    SCHEMA_FILE = "schema.json"
    """Pydantic JSON Schema with encoding info in json_schema_extra."""

    ASSETS_DIR = "assets"
    """Directory containing binary assets."""

    ASSET_EXT = ".bin"
    """Extension for binary assets (arrays, packables, resources)."""

    @staticmethod
    def asset_path(checksum: str) -> str:
        """Get the path for an asset in the zip archive.

        Args:
            checksum: Asset checksum

        Returns:
            Path like "assets/{checksum}.bin"
        """
        return f"{ExportConstants.ASSETS_DIR}/{checksum}{ExportConstants.ASSET_EXT}"

    @staticmethod
    def checksum_from_path(path: str) -> str:
        """Extract checksum from an asset path.

        Args:
            path: Asset path like "assets/{checksum}.bin"

        Returns:
            The checksum string
        """
        prefix_len = len(ExportConstants.ASSETS_DIR) + 1  # +1 for the /
        suffix_len = len(ExportConstants.ASSET_EXT)
        return path[prefix_len:-suffix_len]
