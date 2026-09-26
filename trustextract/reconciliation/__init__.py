"""
TrustExtract Cross-Document Reconciliation Package (Loop 5)
"""

from trustextract.reconciliation.entity_graph import (
    EntityClaim,
    EntityGraph,
    EntityNode,
    canonicalize_entity_key,
)
from trustextract.reconciliation.reconciler import (
    CrossDocumentReconciler,
    ReconciliationReport,
)

__all__ = [
    "EntityClaim",
    "EntityGraph",
    "EntityNode",
    "canonicalize_entity_key",
    "CrossDocumentReconciler",
    "ReconciliationReport",
]
