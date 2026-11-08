"""Convert notebook.py (percent-format cells) into the project .ipynb."""
import re
import sys
import nbformat

src = open(sys.argv[1]).read()
out = sys.argv[2]
cells = []
for chunk in re.split(r'^# %%', src, flags=re.M)[1:]:
    header, _, body = chunk.partition('\n')
    body = body.strip('\n')
    if header.strip().startswith('[markdown]'):
        text = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nb.metadata['language_info'] = {'name': 'python'}
nbformat.write(nb, out)
print(f'{len(cells)} cells -> {out}')
