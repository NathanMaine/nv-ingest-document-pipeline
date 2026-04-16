# sample_pdfs/

Drop your PDF files here for extraction.

## Notes

- The Docker Compose mount maps `./data/sample_pdfs` → `/workspace/docs:ro` (read-only) inside the nv-ingest container, so any PDF you put here is immediately visible to nv-ingest.
- Files are loaded from this directory by `data/sample_pdfs/<your.pdf>` paths in the Python API.
- The repo's `.gitignore` excludes `*.pdf` by default to avoid accidental commits of large files. If you want to commit a small sample (recommended for tests), use `git add -f data/sample_pdfs/your-sample.pdf`.

## Recommended sample PDFs (public domain)

If you need a quick sanity check without your own data, try one of these:

- **NIST publications** — https://csrc.nist.gov/publications  (US government, public domain)
- **Internet RFCs** — https://www.rfc-editor.org/  (royalty-free, well-structured)
- **Project Gutenberg PDFs** — https://www.gutenberg.org/  (public domain books)
- **arXiv preprints** — https://arxiv.org/  (most are CC-BY or open)

For a benchmark, use a mix:
- One short native-text PDF (~10 pages)
- One long native-text PDF (~100 pages)
- One scanned PDF if you want OCR coverage
- One table-heavy PDF (technical spec, datasheet, regulation)
