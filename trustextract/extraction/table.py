"""
Table Structure Extractor for TrustExtract
Detects tabular regions, extracts table grid rows/columns, and computes per-cell confidence scores.
"""

from typing import Any, Dict, List, Optional
from trustextract.schema import TableData


class TableStructureExtractor:
    """Extracts structured tables with cell-level confidence from OCR tokens and spatial layouts."""

    HEADER_KEYWORDS = ["description", "quantity", "qty", "unitprice", "unit price", "price", "linetotal", "line total", "item"]

    def extract_tables(self, tokens: List[Dict[str, Any]]) -> List[TableData]:
        """
        Identify table header and row structure from token coordinates and confidences.
        Returns TableData objects conforming to schema.
        """
        if not tokens:
            return []

        # Find potential table header tokens (excluding lines that contain colons or dollar signs)
        candidate_headers = []
        for t in tokens:
            text_clean = t["text"].lower().strip()
            if ":" in text_clean or "$" in text_clean:
                continue
            if any(h == text_clean or h in text_clean for h in self.HEADER_KEYWORDS):
                candidate_headers.append(t)

        if len(candidate_headers) < 2:
            return []

        # Group candidate headers into the same horizontal line (within 20px y-difference)
        header_lines: List[List[Dict[str, Any]]] = []
        for ch in candidate_headers:
            placed = False
            for hline in header_lines:
                if abs(hline[0]["bbox"][1] - ch["bbox"][1]) <= 20:
                    hline.append(ch)
                    placed = True
                    break
            if not placed:
                header_lines.append([ch])

        # Pick the header line with the most columns (at least 2)
        valid_header_lines = [hl for hl in header_lines if len(hl) >= 2]
        if not valid_header_lines:
            return []

        best_header = max(valid_header_lines, key=len)
        best_header.sort(key=lambda t: t["bbox"][0])

        col_xs = [t["bbox"][0] for t in best_header]
        num_cols = len(col_xs)
        header_y_max = max(t["bbox"][1] + t["bbox"][3] for t in best_header)

        # Find lower boundary of table (where Subtotal/Tax/Totals appear)
        totals_y = 999999.0
        for t in tokens:
            txt = t["text"].lower()
            if any(k in txt for k in ["subtotal", "invoice total", "tax (", "grand total"]):
                if t["bbox"][1] > header_y_max:
                    totals_y = min(totals_y, t["bbox"][1])

        # Extract body tokens between header and totals
        body_tokens = [
            t for t in tokens
            if (header_y_max + 10) <= t["bbox"][1] < totals_y
        ]
        if not body_tokens:
            return []

        # Group body tokens into table rows based on vertical proximity
        body_tokens.sort(key=lambda t: t["bbox"][1])
        row_clusters: List[List[Dict[str, Any]]] = []
        curr_cluster: List[Dict[str, Any]] = []
        curr_y: Optional[float] = None

        for bt in body_tokens:
            y = bt["bbox"][1]
            if curr_y is None or abs(y - curr_y) <= 25:
                curr_cluster.append(bt)
                curr_y = y if curr_y is None else (curr_y + y) / 2
            else:
                if curr_cluster:
                    row_clusters.append(sorted(curr_cluster, key=lambda item: item["bbox"][0]))
                curr_cluster = [bt]
                curr_y = y
        if curr_cluster:
            row_clusters.append(sorted(curr_cluster, key=lambda item: item["bbox"][0]))

        # Map each row's tokens into the column slots
        table_rows: List[List[Any]] = []
        conf_matrix: List[List[float]] = []

        for row_tokens in row_clusters:
            row_vals: List[Any] = [""] * num_cols
            row_confs: List[float] = [0.0] * num_cols

            for token in row_tokens:
                t_x = token["bbox"][0]
                # Find nearest column
                best_col_idx = 0
                min_dist = float("inf")
                for c_idx, c_x in enumerate(col_xs):
                    dist = abs(t_x - c_x)
                    if dist < min_dist:
                        min_dist = dist
                        best_col_idx = c_idx

                val = token["text"]
                if val.startswith("$"):
                    try:
                        val = float(val.replace("$", "").replace(",", ""))
                    except ValueError:
                        pass
                elif val.isdigit():
                    val = int(val)

                row_vals[best_col_idx] = val
                row_confs[best_col_idx] = round(token["confidence"], 3)

            # If row has content in at least one column
            if any(cell != "" for cell in row_vals):
                table_rows.append(row_vals)
                conf_matrix.append(row_confs)

        if not table_rows:
            return []

        return [TableData(rows=table_rows, confidence_per_cell=conf_matrix)]
