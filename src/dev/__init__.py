"""Local development fixtures only - NOT part of the matching pipeline.

Nothing under `src/dev/` is imported by `blocking.py`, `features.py`, `model.py`, `decode.py` or
`pipeline.py`. It exists purely to generate a small fake dataset with the competition's exact file/column
schema, so the pipeline in `src/` can be run and sanity-checked end-to-end without the real competition
data (which was never provided to us). The name/city vocabulary hand-written below is only ever used to
*invent* fictitious test records here; it is never consulted by the actual matching pipeline.
"""
