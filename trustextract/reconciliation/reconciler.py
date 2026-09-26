"""
Cross-Document Reconciliation Engine (Loop 5)
Reconciles claims about shared real-world entities across a bundle of related documents,
reusing Loop 3's field-type-aware comparison logic to surface conflicting assertions into
cross_document_conflicts[] and escalate document_status to NEEDS_REVIEW.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from trustextract.schema import (
    CrossDocumentConflict,
    DocumentStatus,
    ExtractionAgreement,
    FieldType,
    TrustExtractResult,
)
from trustextract.security.audit import audit_logger
from trustextract.extraction.pipeline import determine_field_agreement
from trustextract.reconciliation.entity_graph import EntityClaim, EntityGraph, EntityNode


@dataclass
class ReconciliationReport:
    """Result of bundle-wide cross-document reconciliation."""
    conflicts: List[CrossDocumentConflict] = field(default_factory=list)
    reconciled_entities_count: int = 0
    conflicting_entities_count: int = 0
    participating_document_ids: List[str] = field(default_factory=list)


class CrossDocumentReconciler:
    """
    Reconciles multi-document bundles by comparing claims on shared entity nodes.
    Reuses field-type-aware comparison logic:
      - IDENTIFIER: Strict character match (no near-match leniency)
      - NUMERIC: Parsed float comparison with delta tolerance
      - DATE: Canonical YYYY-MM-DD date comparison
      - FREE_TEXT: Normalized and near-match edit distance tolerance
    """

    def __init__(self) -> None:
        pass

    def reconcile_graph(self, graph: EntityGraph) -> List[CrossDocumentConflict]:
        """
        Inspects all shared entity nodes in the graph and detects cross-document discrepancies.
        """
        conflicts: List[CrossDocumentConflict] = []
        shared_nodes = graph.get_shared_nodes()

        for node in shared_nodes:
            # Compare claims across all unique pairs of documents for this entity
            claims = node.claims
            seen_doc_pairs = set()

            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    claim_a = claims[i]
                    claim_b = claims[j]

                    # Only compare claims from distinct documents
                    if claim_a.document_id == claim_b.document_id:
                        continue

                    doc_pair = tuple(sorted([claim_a.document_id, claim_b.document_id]))
                    if doc_pair in seen_doc_pairs:
                        continue

                    # Compare values using field-type-aware reconciliation
                    agreement, _ = determine_field_agreement(
                        val_a=claim_a.value,
                        val_b=claim_b.value,
                        field_name=node.entity_key
                    )

                    # Any disagreement (OCR_NEQ_VISION) constitutes a cross-document conflict
                    if agreement == ExtractionAgreement.OCR_NEQ_VISION:
                        seen_doc_pairs.add(doc_pair)
                        conflict = CrossDocumentConflict(
                            entity_key=node.entity_key,
                            field_name=claim_a.field_name,
                            document_ids=[claim_a.document_id, claim_b.document_id],
                            conflicting_values=[
                                {
                                    "document_id": claim_a.document_id,
                                    "field_name": claim_a.field_name,
                                    "value": str(claim_a.value),
                                    "confidence": claim_a.confidence,
                                    "crop_ref": claim_a.source.crop_ref if claim_a.source else None,
                                    "bbox": claim_a.source.bbox if claim_a.source else None,
                                    "page": claim_a.source.page if claim_a.source else 1
                                },
                                {
                                    "document_id": claim_b.document_id,
                                    "field_name": claim_b.field_name,
                                    "value": str(claim_b.value),
                                    "confidence": claim_b.confidence,
                                    "crop_ref": claim_b.source.crop_ref if claim_b.source else None,
                                    "bbox": claim_b.source.bbox if claim_b.source else None,
                                    "page": claim_b.source.page if claim_b.source else 1
                                }
                            ],
                            resolution_status="UNRESOLVED"
                        )
                        conflicts.append(conflict)

        return conflicts

    def reconcile_bundle(
        self,
        document_results: List[TrustExtractResult],
        actor_id: str = "reconciliation_worker"
    ) -> ReconciliationReport:
        """
        Orchestrates cross-document reconciliation across a bundle of TrustExtractResult objects.
        Mutates each TrustExtractResult in-place by attaching relevant conflicts, adding explicit
        escalation reasons, and escalating document_status to NEEDS_REVIEW.
        """
        graph = EntityGraph()
        doc_map: Dict[str, TrustExtractResult] = {}

        for res in document_results:
            doc_map[res.document_id] = res
            graph.add_document_result(res)

        conflicts = self.reconcile_graph(graph)

        # Distribute conflicts and escalation reasons to participating documents
        for conf in conflicts:
            for doc_id in conf.document_ids:
                if doc_id in doc_map:
                    doc_res = doc_map[doc_id]
                    # Append conflict object if not already present
                    if not any(c.entity_key == conf.entity_key and set(c.document_ids) == set(conf.document_ids) for c in doc_res.cross_document_conflicts):
                        doc_res.cross_document_conflicts.append(conf)

                    # Build human-readable escalation reason
                    other_vals = [
                        f"doc '{cv['document_id']}' ({cv['field_name']}='{cv['value']}')"
                        for cv in conf.conflicting_values
                    ]
                    reason = f"CROSS_DOCUMENT_CONFLICT: Discrepancy on entity '{conf.entity_key}' between " + " and ".join(other_vals)
                    if reason not in doc_res.escalation_reasons:
                        doc_res.escalation_reasons.append(reason)

                    # Escalate document status
                    doc_res.document_status = DocumentStatus.NEEDS_REVIEW

        shared_nodes = graph.get_shared_nodes()
        report = ReconciliationReport(
            conflicts=conflicts,
            reconciled_entities_count=len(shared_nodes),
            conflicting_entities_count=len(conflicts),
            participating_document_ids=list(doc_map.keys())
        )

        bundle_doc_id = "bundle:" + "+".join(sorted(doc_map.keys())) if doc_map else "bundle:empty"
        audit_logger.log_event(
            action="CROSS_DOC_RECONCILIATION_COMPLETE",
            actor_id=actor_id,
            document_id=bundle_doc_id,
            details={
                "bundle_size": len(document_results),
                "reconciled_entities": len(shared_nodes),
                "conflicts_count": len(conflicts),
                "conflicting_entity_keys": [c.entity_key for c in conflicts]
            }
        )

        return report
