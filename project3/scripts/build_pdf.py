"""Render docs/DOKUMENTACJA.md to docs/DOKUMENTACJA.pdf (Markdown -> HTML -> headless Chrome).

    uv run --with markdown python scripts/build_pdf.py [--chrome PATH]

Under WSL the Windows Chrome/Edge is used when no Linux Chrome is installed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

DOCS = Path(__file__).resolve().parents[1] / 'docs'
CANDIDATES = ['chromium', 'chromium-browser', 'google-chrome', 'google-chrome-stable',
              '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe',
              '/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe']

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm;
        @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
html { font-family: 'Segoe UI', 'DejaVu Sans', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
       color: #111; background: #fff; }
body { margin: 0; }
h1 { font-size: 20pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 15pt; margin: 20pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #ccc;
     break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 4pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #1a5fb4; text-decoration: none; }
img { max-width: 100%; max-height: 235mm; display: block; margin: 8pt auto; break-inside: avoid; }
img[src$="demo_renderer.png"] { max-height: 125mm; }
#spis-treści + ol { break-inside: avoid; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9pt; break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 3pt 6pt; vertical-align: top; }
th { background: #f0f0f0; }
code { font-family: Consolas, 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f4f4f4;
       padding: 0 2pt; border-radius: 2pt; }
pre { background: #f4f4f4; padding: 6pt 8pt; border-radius: 3pt; white-space: pre-wrap;
      break-inside: avoid; }
pre code { padding: 0; background: none; }
hr { border: 0; border-top: 1px solid #ccc; margin: 14pt 0; }
"""


def find_chrome(explicit):
    for c in [explicit] + CANDIDATES if explicit else CANDIDATES:
        path = shutil.which(c) or (c if Path(c).exists() else None)
        if path:
            return path
    raise SystemExit('Chrome/Chromium not found, pass --chrome PATH')


def to_browser_path(p: Path, chrome: str) -> str:
    if chrome.endswith('.exe'):
        return subprocess.check_output(['wslpath', '-w', str(p)], text=True).strip()
    return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chrome')
    args = ap.parse_args()
    md = (DOCS / 'DOKUMENTACJA.md').read_text(encoding='utf-8')
    body = markdown.markdown(md, extensions=['tables', 'fenced_code', 'toc', 'sane_lists'],
                             extension_configs={'toc': {'slugify': slugify_unicode}})
    html = DOCS / 'DOKUMENTACJA.html'
    html.write_text(f'<!doctype html><html lang="pl"><head><meta charset="utf-8">'
                    f'<title>SIGK - Projekt 3: Rendering neuralny</title><style>{CSS}</style></head>'
                    f'<body>{body}</body></html>', encoding='utf-8')
    chrome = find_chrome(args.chrome)
    pdf = DOCS / 'DOKUMENTACJA.pdf'
    src = to_browser_path(html, chrome)
    url = 'file:' + src.replace('\\', '/') if chrome.endswith('.exe') else html.as_uri()
    subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                    f'--print-to-pdf={to_browser_path(pdf, chrome)}', url], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    html.unlink()
    print('->', pdf)


if __name__ == '__main__':
    main()
