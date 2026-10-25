# Test fixtures

There are no binary fixture files in this repository, and that is deliberate.

`tests/conftest.py` builds PDFs in memory with `build_pdf`, which takes a list of
pages, each a list of `(text, font_size)` blocks. Larger font sizes become
headings during parsing, which is how a fixture exercises the chunker's section
tracking.

Generating fixtures rather than committing them has three advantages: diffs stay
readable, the exact text a test asserts on is visible in the test, and there is no
risk of committing a real document by accident.

## Available fixtures

| fixture | shape | used for |
|---------|-------|----------|
| `sample_pdf` | two pages, headings, financial prose | ingestion, retrieval, citation |
| `injection_pdf` | one page whose body attempts a prompt injection | injection flagging and the verifier |

## Adding one

```python
from tests.conftest import build_pdf

pdf = build_pdf(
    [
        [("Heading", 20.0), ("Body text with the number 42 in it.", 10.0)],
        [("Second Page", 18.0), ("More body text.", 10.0)],
    ],
    title="My Fixture",
)
```

Keep the text short and keep the span a test will quote visible in the fixture
definition, so a reader can see why the assertion holds.

## Scanned pages

Generated fixtures have a text layer, so they always triage as `native`. To
exercise the OCR path you need a real scan: render a generated PDF to an image
and reassemble it without text, or add a small real scanned PDF under
`sample_pdfs/` and mark the test `integration`. The OCR path is otherwise
exercised only by its unit tests.
