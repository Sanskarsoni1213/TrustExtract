"""
Business-Logic Validators for TrustExtract (Loop 3)
Runs independently of Path A / Path B agreement signals.
A field that fails a business-logic check gets its confidence capped
regardless of what the OCR/vision agreement score says.

Supported validators:
  1. InvoiceArithmeticValidator  — line-item sum → Subtotal → InvoiceTotal
  2. DateLogicValidator          — InvoiceDate < DueDate
  3. IdentifierFormatValidator   — InvoiceId / DocumentNumber pattern checks
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

# Tolerance for floating-point arithmetic comparisons (currency is 2 d.p.)
_CURRENCY_TOLERANCE = 0.02


@dataclass
class FieldValidationResult:
    """Result of all validators applied to a single field."""
    field_name: str
    flags: List[str] = field(default_factory=list)          # e.g. ["ARITHMETIC_OK", "DATE_FAIL"]
    passed: bool = True                                      # False => confidence will be capped
    cap_confidence_at: Optional[float] = None               # when failed, cap here (default 0.50)


@dataclass
class DocumentValidationReport:
    """Aggregated validation results across all fields in a document."""
    per_field: Dict[str, FieldValidationResult] = field(default_factory=dict)
    summary_flags: List[str] = field(default_factory=list)  # document-level flags


# ---------------------------------------------------------------------------
# 1. Arithmetic validator
# ---------------------------------------------------------------------------

class InvoiceArithmeticValidator:
    """
    Verifies the three arithmetic relationships in an invoice:
      (a) sum(line_item totals) approx Subtotal
      (b) Subtotal + TotalTax approx InvoiceTotal
    Any discrepancy >= _CURRENCY_TOLERANCE flags the financial fields as FAIL.
    """

    FLAG_LINE_SUM_OK    = "LINE_SUM_OK"
    FLAG_LINE_SUM_FAIL  = "LINE_SUM_FAIL"
    FLAG_TAX_TOTAL_OK   = "TAX_TOTAL_OK"
    FLAG_TAX_TOTAL_FAIL = "TAX_TOTAL_FAIL"
    FLAG_ARITHMETIC_OK  = "ARITHMETIC_OK"

    def validate(
        self,
        fields: Dict[str, Any],
        table_rows: List[List[Any]],
        report: DocumentValidationReport,
    ) -> None:
        """Mutates report in-place with results for Subtotal, TotalTax, InvoiceTotal."""

        subtotal_val  = self._num(fields.get("Subtotal"))
        tax_val       = self._num(fields.get("TotalTax"))
        invoice_total = self._num(fields.get("InvoiceTotal"))

        # (a) Line-item sum -> Subtotal
        line_sum_flag: Optional[str] = None
        if table_rows and subtotal_val is not None:
            try:
                # Last cell of each row is treated as the line-item total
                line_totals = [float(row[-1]) for row in table_rows if isinstance(row[-1], (int, float))]
                computed_subtotal = round(sum(line_totals), 2)
                if abs(computed_subtotal - subtotal_val) <= _CURRENCY_TOLERANCE:
                    line_sum_flag = self.FLAG_LINE_SUM_OK
                else:
                    line_sum_flag = self.FLAG_LINE_SUM_FAIL
            except (TypeError, ValueError):
                line_sum_flag = self.FLAG_LINE_SUM_FAIL

        # (b) Subtotal + TotalTax == InvoiceTotal
        tax_total_flag: Optional[str] = None
        arithmetic_consistent = True
        if subtotal_val is not None and tax_val is not None and invoice_total is not None:
            expected_total = round(subtotal_val + tax_val, 2)
            if abs(expected_total - invoice_total) <= _CURRENCY_TOLERANCE:
                tax_total_flag = self.FLAG_TAX_TOTAL_OK
            else:
                tax_total_flag = self.FLAG_TAX_TOTAL_FAIL
                arithmetic_consistent = False

        # Write results per-field
        financial_fields = ["Subtotal", "TotalTax", "InvoiceTotal"]
        for fname in financial_fields:
            result = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
            if line_sum_flag:
                result.flags.append(line_sum_flag)
            if tax_total_flag:
                result.flags.append(tax_total_flag)
            if not arithmetic_consistent:
                result.passed = False
                result.cap_confidence_at = 0.50
            else:
                if tax_total_flag == self.FLAG_TAX_TOTAL_OK and self.FLAG_ARITHMETIC_OK not in result.flags:
                    result.flags.append(self.FLAG_ARITHMETIC_OK)

        if not arithmetic_consistent:
            report.summary_flags.append("ARITHMETIC_INCONSISTENCY")

    @staticmethod
    def _num(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# 2. Payslip arithmetic validator
# ---------------------------------------------------------------------------

class PayslipArithmeticValidator:
    """
    Verifies payslip arithmetic relationships:
      GrossPay - TotalDeductions approx NetPay
    Any discrepancy >= _CURRENCY_TOLERANCE flags financial fields as FAIL.
    """
    FLAG_PAYSLIP_ARITHMETIC_OK   = "PAYSLIP_ARITHMETIC_OK"
    FLAG_PAYSLIP_ARITHMETIC_FAIL = "PAYSLIP_ARITHMETIC_FAIL"

    def validate(
        self,
        fields: Dict[str, Any],
        report: DocumentValidationReport,
    ) -> None:
        gross_val = self._num(fields.get("GrossPay"))
        deductions_val = self._num(fields.get("TotalDeductions"))
        net_val = self._num(fields.get("NetPay"))

        if gross_val is not None and deductions_val is not None and net_val is not None:
            expected_net = round(gross_val - deductions_val, 2)
            passed = abs(expected_net - net_val) <= _CURRENCY_TOLERANCE
            flag = self.FLAG_PAYSLIP_ARITHMETIC_OK if passed else self.FLAG_PAYSLIP_ARITHMETIC_FAIL

            for fname in ("GrossPay", "TotalDeductions", "NetPay"):
                result = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
                result.flags.append(flag)
                if not passed:
                    result.passed = False
                    result.cap_confidence_at = 0.50

            if not passed:
                report.summary_flags.append("PAYSLIP_ARITHMETIC_INCONSISTENCY")

    @staticmethod
    def _num(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# 3. Date logic validator
# ---------------------------------------------------------------------------

class DateLogicValidator:
    """Checks InvoiceDate < DueDate and PayPeriodStart < PayPeriodEnd <= PayDate and that dates are parseable."""

    FLAG_DATE_OK          = "DATE_LOGIC_OK"
    FLAG_DATE_FAIL        = "DATE_LOGIC_FAIL"
    FLAG_DATE_UNPARSEABLE = "DATE_UNPARSEABLE"

    _DATE_FORMATS = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y",
        "%d.%m.%Y", "%m.%d.%Y",
    ]

    def validate(
        self,
        fields: Dict[str, Any],
        report: DocumentValidationReport,
    ) -> None:
        """Mutates report in-place with results for dates."""
        inv_date = self._parse(fields.get("InvoiceDate"))
        due_date = self._parse(fields.get("DueDate"))

        for fname, parsed in [("InvoiceDate", inv_date), ("DueDate", due_date)]:
            if fields.get(fname) is None:
                continue
            result = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
            if parsed is None:
                result.flags.append(self.FLAG_DATE_UNPARSEABLE)
                result.passed = False
                result.cap_confidence_at = 0.40

        if inv_date and due_date:
            if inv_date < due_date:
                for fname in ("InvoiceDate", "DueDate"):
                    r = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
                    r.flags.append(self.FLAG_DATE_OK)
            else:
                report.summary_flags.append("DATE_ORDER_VIOLATION")
                for fname in ("InvoiceDate", "DueDate"):
                    r = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
                    r.flags.append(self.FLAG_DATE_FAIL)
                    r.passed = False
                    r.cap_confidence_at = 0.45

        # Payslip dates: PayPeriodStart < PayPeriodEnd
        period_start = self._parse(fields.get("PayPeriodStart"))
        period_end   = self._parse(fields.get("PayPeriodEnd"))
        pay_date     = self._parse(fields.get("PayDate"))

        for fname, parsed in [("PayPeriodStart", period_start), ("PayPeriodEnd", period_end), ("PayDate", pay_date)]:
            if fields.get(fname) is None:
                continue
            result = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
            if parsed is None:
                result.flags.append(self.FLAG_DATE_UNPARSEABLE)
                result.passed = False
                result.cap_confidence_at = 0.40

        if period_start and period_end:
            if period_start < period_end:
                for fname in ("PayPeriodStart", "PayPeriodEnd"):
                    r = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
                    r.flags.append(self.FLAG_DATE_OK)
            else:
                report.summary_flags.append("PAY_PERIOD_ORDER_VIOLATION")
                for fname in ("PayPeriodStart", "PayPeriodEnd"):
                    r = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
                    r.flags.append(self.FLAG_DATE_FAIL)
                    r.passed = False
                    r.cap_confidence_at = 0.45

        if pay_date and period_end and pay_date >= period_end:
            r = report.per_field.setdefault("PayDate", FieldValidationResult(field_name="PayDate"))
            r.flags.append(self.FLAG_DATE_OK)

    def _parse(self, val: Any) -> Optional[datetime]:
        if val is None:
            return None
        s = str(val).strip()
        for fmt in self._DATE_FORMATS:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None


# ---------------------------------------------------------------------------
# 4. Identifier format validator
# ---------------------------------------------------------------------------

class IdentifierFormatValidator:
    """
    Checks InvoiceId / DocumentNumber / EmployeeId match expected structural patterns.
    """

    INVOICE_ID_PATTERN = re.compile(r"^[A-Z]{2,6}-\d{4}-[A-Za-z0-9]+$")
    EMPLOYEE_ID_PATTERN = re.compile(r"^[A-Z]{2,6}-\d{4}-[A-Za-z0-9]+$")
    FLAG_FORMAT_OK   = "ID_FORMAT_OK"
    FLAG_FORMAT_FAIL = "ID_FORMAT_FAIL"

    def validate(
        self,
        fields: Dict[str, Any],
        report: DocumentValidationReport,
    ) -> None:
        for fname in ("InvoiceId", "DocumentNumber", "EmployeeId"):
            if fname not in fields or fields[fname] is None:
                continue
            val = str(fields[fname]).strip()
            result = report.per_field.setdefault(fname, FieldValidationResult(field_name=fname))
            pattern = self.EMPLOYEE_ID_PATTERN if fname == "EmployeeId" else self.INVOICE_ID_PATTERN
            if pattern.match(val):
                result.flags.append(self.FLAG_FORMAT_OK)
            else:
                result.flags.append(self.FLAG_FORMAT_FAIL)
                result.passed = False
                result.cap_confidence_at = 0.55


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DocumentValidator:
    """Runs all validators and returns a consolidated DocumentValidationReport."""

    def __init__(self) -> None:
        self._arithmetic = InvoiceArithmeticValidator()
        self._payslip_arithmetic = PayslipArithmeticValidator()
        self._dates      = DateLogicValidator()
        self._ids        = IdentifierFormatValidator()

    def validate(
        self,
        fields: Dict[str, Any],
        table_rows: List[List[Any]],
    ) -> DocumentValidationReport:
        """
        Args:
            fields:     Dict mapping field name to raw extracted value
            table_rows: Rows from TableData (each row is a List[Any])
        Returns:
            DocumentValidationReport with per-field results and summary flags
        """
        report = DocumentValidationReport()
        self._arithmetic.validate(fields, table_rows, report)
        self._payslip_arithmetic.validate(fields, report)
        self._dates.validate(fields, report)
        self._ids.validate(fields, report)

        # Fields with no validator result get a default entry
        for fname in fields:
            if fname not in report.per_field:
                report.per_field[fname] = FieldValidationResult(
                    field_name=fname,
                    flags=["NO_VALIDATOR_APPLIED"],
                    passed=True
                )

        return report
