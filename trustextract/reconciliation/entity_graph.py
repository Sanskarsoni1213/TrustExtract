"""
Entity Graph for Cross-Document Reconciliation (Loop 5)
Maintains shared real-world entities (vendors, customers, invoice/PO references, amounts, dates)
and maps claims from heterogeneous documents within a bundle to canonical graph nodes.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from trustextract.schema import FieldSource, FieldType, TrustExtractResult
from trustextract.extraction.pipeline import classify_field_type


# Canonical entity key aliases mapping heterogeneous document field names to shared entity nodes
ENTITY_CANONICAL_ALIASES: Dict[str, List[str]] = {
    "invoice_id": [
        "invoiceid", "invoicenumber", "invoicenum", "invoiceno",
        "invoice_no", "inv_num", "inv_no", "invoice_ref"
    ],
    "po_number": [
        "ponumber", "po_number", "ponum", "pocode",
        "purchaseordernumber", "purchase_order_number", "order_id", "orderid", "po_ref"
    ],
    "vendor_name": [
        "vendorname", "vendor", "suppliername", "supplier",
        "seller", "payee", "companyname", "merchant"
    ],
    "customer_name": [
        "customername", "customer", "buyername", "buyer",
        "clientname", "client", "billto", "soldto"
    ],
    "invoice_total": [
        "invoicetotal", "total", "totalamount", "amountdue",
        "balance", "grandtotal", "orderamount", "ordertotal", "po_amount", "po_total"
    ],
    "subtotal": [
        "subtotal", "sub_total", "netamount", "baseamount"
    ],
    "tax_amount": [
        "tax", "taxamount", "totaltax", "vat", "gst"
    ],
    "invoice_date": [
        "invoicedate", "date", "billdate", "orderdate", "podate", "issue_date", "documentdate"
    ],
    "due_date": [
        "duedate", "paymentdue", "expiry_date", "valid_until", "deliverydate", "requireddate"
    ],
    "vendor_address": [
        "vendoraddress", "remitto", "supplieraddress", "vendor_address"
    ],
    "customer_address": [
        "customeraddress", "shipto", "buyeraddress", "deliveryaddress", "destination"
    ],
    "employee_id": [
        "employeeid", "employee_id", "empid", "emp_number", "staff_id"
    ],
    "employee_name": [
        "employeename", "employee_name", "staff_name", "worker_name"
    ],
    "gross_pay": [
        "grosspay", "gross_pay", "gross_amount", "gross_earnings", "total_earnings"
    ],
    "net_pay": [
        "netpay", "net_pay", "net_amount", "take_home_pay"
    ],
    "deductions": [
        "totaldeductions", "deductions", "total_deductions"
    ],
    "pay_period_start": [
        "payperiodstart", "period_start", "pay_start"
    ],
    "pay_period_end": [
        "payperiodend", "period_end", "pay_end"
    ],
    "pay_date": [
        "paydate", "payment_date", "pay_day"
    ],
}


def canonicalize_entity_key(field_name: str) -> str:
    """
    Maps a raw document field name (e.g. 'PurchaseOrderNumber', 'InvoiceRef', 'SupplierName')
    to its canonical entity key in the cross-document graph.
    """
    clean_name = re.sub(r'[^a-zA-Z0-9]', '', str(field_name)).lower()
    for canonical_key, aliases in ENTITY_CANONICAL_ALIASES.items():
        cleaned_aliases = [re.sub(r'[^a-zA-Z0-9]', '', a).lower() for a in aliases]
        if clean_name in cleaned_aliases:
            return canonical_key
        # Substring heuristics if field name directly implies the entity
        for ca in cleaned_aliases:
            if len(ca) >= 6 and (ca in clean_name or clean_name in ca):
                return canonical_key
    return clean_name


@dataclass
class EntityClaim:
    """A specific factual claim made by a single document about an entity."""
    document_id: str
    field_name: str
    value: Any
    field_type: FieldType
    confidence: float
    source: Optional[FieldSource] = None


@dataclass
class EntityNode:
    """A canonical entity node in the cross-document graph accumulating claims across documents."""
    entity_key: str
    field_type: FieldType
    claims: List[EntityClaim] = field(default_factory=list)

    @property
    def participating_document_ids(self) -> Set[str]:
        return {c.document_id for c in self.claims}

    @property
    def is_multi_document(self) -> bool:
        """True if two or more distinct documents in the bundle contribute claims to this entity."""
        return len(self.participating_document_ids) >= 2


class EntityGraph:
    """
    Cross-document graph that aggregates factual claims across a bundle of documents.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, EntityNode] = {}

    def add_claim(
        self,
        document_id: str,
        field_name: str,
        value: Any,
        confidence: float = 1.0,
        source: Optional[FieldSource] = None,
        field_type: Optional[FieldType] = None
    ) -> None:
        """Adds a claim to the entity graph, creating or updating the canonical node."""
        if value is None or str(value).strip() == "":
            return

        canonical_key = canonicalize_entity_key(field_name)
        f_type = field_type or classify_field_type(field_name)

        if canonical_key not in self.nodes:
            self.nodes[canonical_key] = EntityNode(
                entity_key=canonical_key,
                field_type=f_type,
                claims=[]
            )

        claim = EntityClaim(
            document_id=document_id,
            field_name=field_name,
            value=value,
            field_type=f_type,
            confidence=confidence,
            source=source
        )
        self.nodes[canonical_key].claims.append(claim)

    def add_document_result(self, result: TrustExtractResult) -> None:
        """Populates claims from an existing TrustExtractResult extraction."""
        doc_id = result.document_id
        for fname, fv in result.fields.items():
            f_type = classify_field_type(fname)
            self.add_claim(
                document_id=doc_id,
                field_name=fname,
                value=fv.value,
                confidence=fv.confidence,
                source=fv.source,
                field_type=f_type
            )

    def get_shared_nodes(self) -> List[EntityNode]:
        """Returns all entity nodes that have claims across two or more documents."""
        return [node for node in self.nodes.values() if node.is_multi_document]
