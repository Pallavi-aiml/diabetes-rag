r"""
scripts/csv_to_xlsx.py — turn claims_to_label.csv into a labeling sheet that opens correctly in Excel.

Why: Excel sometimes opens a CSV as ONE column (regional list-separator setting) and PDF line breaks inside
the evidence text then split rows apart. An .xlsx has none of these problems.

Usage:
    pip install openpyxl                      # once
    python scripts\csv_to_xlsx.py             # claims_to_label.csv -> claims_to_label.xlsx
    python scripts\csv_to_xlsx.py other.csv

Label in the `label` column (dropdown: 1 = evidence supports the claim, 0 = it does not; blank = unsure), save the
.xlsx, then score it directly:  python scripts\score_labeled_claims.py claims_to_label.xlsx
"""
import csv
import sys
from pathlib import Path


def main():
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.worksheet.datavalidation import DataValidation
    except ImportError:
        sys.exit("Please run:  pip install openpyxl")

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "claims_to_label.csv")
    if not src.exists():
        sys.exit(f"{src} not found (run this from the project folder).")
    dst = src.with_suffix(".xlsx")
    if dst.exists():
        sys.exit(f"{dst} already exists - rename or delete it first so your labels are not overwritten.")

    with open(src, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    col = {name: i for i, name in enumerate(header)}

    wb = Workbook()
    ws = wb.active
    ws.title = "claims"
    ws.append(header)
    for r in body:
        clean = [" ".join(str(c).split()) if i in (col["question"], col["claim"], col["evidence_shown"]) else c
                 for i, c in enumerate(r)]
        for name in ("nli_score",):
            try:
                clean[col[name]] = float(clean[col[name]])
            except (ValueError, KeyError):
                pass
        ws.append(clean)

    widths = {"id": 5, "question": 34, "claim": 60, "cited_sources": 9, "evidence_shown": 90, "nli_score": 9,
              "verifier_says_supported": 12, "answer_flagged": 10, "method": 8, "label": 8, "notes": 30}
    for name, i in col.items():
        letter = ws.cell(row=1, column=i + 1).column_letter
        ws.column_dimensions[letter].width = widths.get(name, 12)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="305496")
    lab = ws.cell(row=1, column=col["label"] + 1).column_letter
    for r in range(2, len(body) + 2):
        ws[f"{lab}{r}"].fill = PatternFill("solid", fgColor="FFF2CC")       # the cells you fill in
    dv = DataValidation(type="list", formula1='"1,0"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{lab}2:{lab}{len(body) + 1}")
    ws.freeze_panes = ws.cell(row=2, column=col["claim"] + 1)

    wb.save(dst)
    print(f"Wrote {dst} ({len(body)} claims). Open it, fill the yellow `label` column (1 or 0), save, then run:\n"
          f"    python scripts\\score_labeled_claims.py {dst.name}")


if __name__ == "__main__":
    main()
