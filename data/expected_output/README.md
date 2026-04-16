# expected_output/

Optional ground-truth JSONL files for validation testing.

## What goes here

If you want to verify your converter changes don't break expected output for a known PDF, generate a JSONL once with a known-good config, then commit it as `<pdf_name>.expected.jsonl`. Subsequent test runs can diff new output against the expected file.

## Example

```python
# Generate expected output once, after manual review of correctness
from src.extractor import DocumentExtractor
from src.converter import DocumentConverter

ex = DocumentExtractor()
results = ex.extract(["data/sample_pdfs/known-good.pdf"])
conv = DocumentConverter()
conv.convert_to_jsonl(results, "data/expected_output/known-good.expected.jsonl")
```

Then in a test:

```python
from pathlib import Path
import json

def test_converter_output_unchanged():
    expected = Path("data/expected_output/known-good.expected.jsonl").read_text()
    actual = Path("output/known-good.jsonl").read_text()
    assert json.loads(actual) == json.loads(expected)
```

This is opt-in. The unit test suite in `tests/` doesn't depend on this directory.
