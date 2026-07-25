"""
uuid_utils.py
-------------
Handles two real-world messiness issues seen in Kaggle-hosted datasets:
- Duplicate-folder suffixes (e.g. 'uuid 2', 'uuid 3') -> fixed by truncating to 36 chars.
- Case differences between folder names and dict keys -> fixed by lowercasing both sides.
"""

from config import UUID_LEN


def clean_uuid(folder_name):
    """Strip any duplicate-folder suffix (e.g. 'uuid 2' -> 'uuid') and normalize case."""
    return folder_name[:UUID_LEN].strip().lower()


def build_normalized_lookup(id_to_raga):
    """Lowercased uuid -> raga_name lookup, so matching is case-insensitive."""
    return {k.strip().lower(): v for k, v in id_to_raga.items()}
