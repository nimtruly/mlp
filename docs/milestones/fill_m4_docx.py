"""Fill the Milestone 4 .docx template from MILESTONE_4_ANSWERS.md.

    python docs/milestones/fill_m4_docx.py [--roll 21f1001234]

Markdown layout per question:

    ## Q1. Title
    ### Answer
    paragraphs; lines starting with "- " become bullets; a markdown table becomes a Word table
    ### Code
    ```python
    ...
    ```
    ### Figures
    - Q1.1 | Short title | notebooks/figures/xyz.png
"""
import argparse
import copy
import re
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEMPLATE = HERE / 'Milestone_4_Model_Building_and_Hyperparameter_Tuning.docx'
ANSWERS = HERE / 'MILESTONE_4_ANSWERS.md'
OUT = HERE / 'Milestone_4_filled.docx'


def parse_answers(md):
    qs = {}
    for block in re.split(r'^## (?=Q\d+\.)', md, flags=re.M)[1:]:
        num = int(re.match(r'Q(\d+)\.', block).group(1))
        parts = dict(re.findall(r'^### (Answer|Code|Figures)\n(.*?)(?=^### |\Z)', block, flags=re.M | re.S))
        code = re.search(r'```python\n(.*?)```', parts.get('Code', ''), flags=re.S)
        figs = [tuple(x.strip() for x in line[2:].split('|'))
                for line in parts.get('Figures', '').splitlines() if line.startswith('- ')]
        qs[num] = dict(answer=parts.get('Answer', '').strip(), code=code.group(1).rstrip() if code else '',
                       figures=figs)
    return qs


def clear_cell(cell):
    for p in cell.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    p = cell.paragraphs[0]
    for r in p.runs:
        r._r.getparent().remove(r._r)
    return p


def add_inline(par, text, size=10):
    """Add text with **bold** support."""
    for i, chunk in enumerate(re.split(r'\*\*', text)):
        if chunk:
            run = par.add_run(chunk.replace('`', ''))
            run.bold = (i % 2 == 1)
            run.font.size = Pt(size)


def fill_answer(cell, text):
    first = clear_cell(cell)
    lines = text.splitlines()
    i, used_first = 0, False

    def new_par():
        nonlocal used_first
        if not used_first:
            used_first = True
            return first
        return cell.add_paragraph()

    while i < len(lines):
        line = lines[i]
        if line.startswith('|'):                      # markdown table
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            if not used_first:                        # a cell must start with a paragraph
                used_first = True
            tbl = cell.add_table(rows=len(rows), cols=len(rows[0]))
            tbl.style = 'Table Grid'
            for r, row in enumerate(rows):
                for c, val in enumerate(row):
                    p = tbl.cell(r, c).paragraphs[0]
                    add_inline(p, f'**{val}**' if r == 0 else val, size=8.5)
            cell.add_paragraph()
            continue
        if not line.strip():
            i += 1
            continue
        p = new_par()
        p.paragraph_format.space_after = Pt(3)
        if line.startswith('- '):
            add_inline(p, '• ' + line[2:])
            p.paragraph_format.left_indent = Inches(0.15)
        else:
            add_inline(p, line)
        i += 1


def fill_code(cell, code):
    p = clear_cell(cell)
    for k, line in enumerate(code.splitlines()):
        if k:
            p = cell.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(line if line else ' ')
        run.font.name = 'Consolas'
        run._r.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), 'Consolas')
        run.font.size = Pt(7.5)


def fill_figures(doc, qs):
    paras = doc.paragraphs
    for num, q in qs.items():
        cap_idx = next((k for k, p in enumerate(paras) if p.text.startswith(f'Figure Q{num}.1:')), None)
        if cap_idx is None or not q['figures']:
            continue
        cap_tpl, img_tpl = paras[cap_idx], paras[cap_idx + 1]
        anchor = img_tpl._p
        for j, (fid, title, path) in enumerate(q['figures']):
            if j == 0:
                cap, img = cap_tpl, img_tpl
            else:
                cap_el, img_el = copy.deepcopy(cap_tpl._p), copy.deepcopy(img_tpl._p)
                anchor.addnext(cap_el)
                cap_el.addnext(img_el)
                cap = docx.text.paragraph.Paragraph(cap_el, cap_tpl._parent)
                img = docx.text.paragraph.Paragraph(img_el, img_tpl._parent)
            for p, text in [(cap, f'Figure {fid}: {title}'), (img, '')]:
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
                if p.runs:
                    p.runs[0].text = text
                else:
                    p.add_run(text)
            img_path = ROOT / path
            img.runs[0].add_picture(str(img_path), width=Inches(6.2))
            img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            anchor = img._p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--roll', default='')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    qs = parse_answers(ANSWERS.read_text())
    doc = docx.Document(str(TEMPLATE))
    if args.roll:
        doc.tables[0].cell(0, 1).paragraphs[0].add_run(args.roll)
    for num in range(1, 11):
        q = qs[num]
        fill_answer(doc.tables[2 * num - 1].cell(0, 0), q['answer'])
        fill_code(doc.tables[2 * num].cell(0, 0), q['code'])
    fill_figures(doc, qs)
    doc.save(args.out)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
