"""Expose source permissions and versioned evidence through one public package.

These models validate metadata; they do not fetch URLs, grant permissions,
or independently verify facts. Reviews remain a separate responsibility.
"""

from .evidence import Evidence, EvidenceStatus
from .source import PermissionStatus, Source

__all__ = ["Evidence", "EvidenceStatus", "PermissionStatus", "Source"]
