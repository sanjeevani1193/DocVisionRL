# CORD v2: Milestone 1 data audit

Audit date: **2026-11-10**. Scope: exactly the first **three training rows** in
stored order, without shuffling. This is a reproducible inspection, not a
representative sample or an exhaustive label inventory. No extraction schema
has been selected.

## Source, provenance, revision, and license

- Dataset: [naver-clova-ix/cord-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2),
  published under NAVER CLOVA's information extraction namespace.
- Pinned Hub commit: `7f0115a4b758a71d6473b8d085751692da2fef98`.
  The Hub API reports last modification at `2022-07-19 23:43:33 UTC`.
- [Published metadata at this revision](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/7f0115a4b758a71d6473b8d085751692da2fef98/dataset_infos.json)
  declares train/validation/test sizes of **800/100/100**. These are metadata
  counts, not counts independently verified by iterating the splits.
- Three different version identifiers must be kept distinct: dataset name
  `cord-v2`, published builder metadata version `0.0.0`, and sampled annotation
  `meta.version` of `2.0.0`. The commit SHA pins the actual audited repository
  contents more precisely than the dataset name.
- The [dataset card](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/7f0115a4b758a71d6473b8d085751692da2fef98/README.md)
  contains only a `cc-by-4.0` license declaration. Published metadata spells out
  Creative Commons Attribution 4.0 International License.
- [Upstream CORD](https://github.com/clovaai/cord) describes Indonesian receipts
  and OCR/semantic parsing annotations. Its v2 release notes describe label
  corrections and `sub_group_id` for improved hierarchy. It also records removal
  of `store_info`, `payment_info`, and `etc` target classes from the public release.
  This historical class statement does not mean every nested key named `etc`
  is absent: `gt_parse.sub_total.etc` occurs in our first sample.
- The upstream repository also declares CC BY 4.0; preserve dataset attribution
  and its [license reference](https://github.com/clovaai/cord/blob/master/LICENSE-CC-BY)
  in future research artifacts. No independent rights/provenance investigation
  of individual receipt images was performed.

## Method and reproducibility

Run from the repository root:

```sh
.venv/bin/python scripts/inspect_cord.py
```

The verified environment is Python 3.12.16 on macOS 14.5 arm64, with
`datasets==5.2.0`, `Pillow==13.0.0`, and `pyarrow==26.0.1`. Exact Conda builds
and all installed Python dependency versions are recorded in
`conda-osx-arm64.lock` and `requirements.lock`. See the README for recreation.

The script reads Hub/card metadata and the small `dataset_infos.json`, then uses
`load_dataset(..., split="train", revision=REVISION, streaming=True)` and
`take(3)`. It returns an `IterableDataset`; no full split is prepared. No validation
or test examples are iterated. No non-streaming fallback is provided because that
could silently violate the download constraint. Sequential sampling needs no seed.

Stored images are exposed with `Image(decode=False)` and decoded individually
with Pillow. This lets image errors be reported separately from annotation errors.
The script parses `ground_truth` with `json.loads`, traverses every sampled
`gt_parse` key recursively, counts types and row presence, and limits annotation
display to 1,800 characters. Truncation affects printing only, not traversal.

Parquet streaming can read blocks/row groups containing uninspected examples.
We did not measure transferred network bytes. After execution, the local Hugging
Face cache occupied approximately 12 KiB and contained metadata/locks, with no
cached Parquet shards or prepared Arrow split. This verifies the absence of a
materialized local dataset; cache size does not measure streamed network traffic.
The full console transcript is local at `data/cord_inspection.txt` (Git-ignored).

## Observed image and annotation containers

Actual dataset features were `image: Image(decode=True)` and
`ground_truth: Value('string')`. Each row contained exactly those two columns.
The image storage was a dictionary; decoded images were Pillow `PngImageFile`
objects, RGB PNGs. Dimensions below are width × height, with no resizing.

| Training index / `meta.image_id` | Image dimensions | Menu entries | `valid_line` entries | `roi` keys | `repeating_symbol` entries |
| --- | --- | --- | --- | --- | --- |
| 0 | 864 × 1296 | 22 | 71 | 0 | 0 |
| 1 | 864 × 1296 | 6 | 24 | 0 | 0 |
| 2 | 720 × 1280 | 5 | 21 | 8 | 2 |

All three `ground_truth` strings parsed into dictionaries with these root keys:

| Key | Observed Python type | Observation |
| --- | --- | --- |
| `gt_parse` | `dict` | Three keys: `menu`, `sub_total`, `total` |
| `meta` | `dict` | `version` and `split` strings, integer `image_id`, `image_size` dictionary |
| `valid_line` | `list` | Nonempty annotation records; detailed contents not audited here |
| `roi` | `dict` | Empty in indices 0–1, nonempty in index 2 |
| `repeating_symbol` | `list` | Empty in indices 0–1, length 2 in index 2 |
| `dontcare` | `list` | Empty in all three |

All sampled metadata versions were `2.0.0`, splits were `train`, and declared
image dimensions matched decoded dimensions. No missing `ground_truth`, invalid
JSON, missing `gt_parse`, or image decoding failure occurred in the three rows.
An empty auxiliary container is observed emptiness, not evidence of missing data.

## Actual `gt_parse` paths and value types

`[]` denotes an element of a list, not a literal annotation key. Row presence
means a path appeared at least once in that receipt. All **116 sampled scalar
leaves** were strings; no numeric conversion was applied.

| Path below `gt_parse` | Type | Receipt presence | Value count |
| --- | --- | --- | --- |
| `menu` | `list` | 3/3 | 3 |
| `menu[]` | `dict` | 3/3 | 33 |
| `menu[].nm` | `str` | 3/3 | 33 |
| `menu[].cnt` | `str` | 3/3 | 33 |
| `menu[].price` | `str` | 3/3 | 33 |
| `menu[].sub` | `dict` | 1/3 | 2 |
| `menu[].sub.nm` | `str` | 1/3 | 2 |
| `sub_total` | `dict` | 3/3 | 3 |
| `sub_total.subtotal_price` | `str` | 3/3 | 3 |
| `sub_total.service_price` | `str` | 2/3 | 2 |
| `sub_total.tax_price` | `str` | 2/3 | 2 |
| `sub_total.etc` | `str` | 1/3 | 1 |
| `total` | `dict` | 3/3 | 3 |
| `total.total_price` | `str` | 3/3 | 3 |
| `total.cashprice` | `str` | 1/3 | 1 |
| `total.changeprice` | `str` | 1/3 | 1 |
| `total.menutype_cnt` | `str` | 1/3 | 1 |
| `total.menuqty_cnt` | `str` | 1/3 | 1 |

Example from training index 1 (excerpt, other fields/items omitted):

```json
{
  "menu": [
    {"nm": "SPGTHY BOLOGNASE", "cnt": "1", "price": "58,000"},
    {"nm": "PEPPER AUS", "cnt": "1", "price": "165,000", "sub": {"nm": "WELL DONE"}}
  ],
  "total": {"total_price": "580,965"}
}
```

This excerpt demonstrates hierarchy; its shortened menu is not an accounting
example. The script prints a larger, explicitly truncated view of each receipt.

## Variations, uncertainties, and schema implications

- Field presence varies. `service_price` and `tax_price` occur in indices 0–1
  but not 2. Cash/change/count fields occur only in index 2. Absence should not
  automatically become zero, an empty string, or a model error.
- Nested `sub` dictionaries occur for two items in index 1 and are absent from
  other sampled items. Flattening them into item names would lose hierarchy.
- Quantity formatting differs: index 0 uses strings such as `"1 x"`; index 1
  uses `"1"`. Prices include `"75,000"`, `"0"`, and the index-0 value `"40,000."`.
  String-preserving extraction and numeric normalization are separate decisions.
- Index 2 has `cashprice="350,000"`, `total_price="334,000"`, and
  `changeprice="-16,000"`. The sign needs comparison with the image and annotation
  conventions before interpreting it as an error. Arithmetic consistency has
  not been validated, so it cannot yet be assumed as a reward rule.
- All three sampled `menu` containers are lists. The public
  [Hub viewer](https://huggingface.co/datasets/naver-clova-ix/cord-v2) additionally
  displays a single-object `menu` at index 4. That is an external viewer
  observation, not a fourth locally loaded training example. Wider type variation
  and optional fields remain unaudited locally. Generic recursive inspection
  supports lists, dictionaries, scalars, and null without assuming menu shape.
- Three consecutive examples cannot establish field frequencies, label coverage,
  null/error rates, duplicate leakage, or representativeness. No split-level
  quality, cross-split duplication, or image/text alignment audit was performed.
- The minimal Hub card gives no detailed collection/consent methodology or
  transformation history. The exact v2 correction procedure, currency conventions,
  full auxiliary annotation types, and relation of every public label to upstream
  definitions remain unverified. The published citation field is empty; the
  research paper linked by upstream should be consulted before publication.
- Future schema work must decide hierarchy, optionality, list/object handling,
  source-text preservation, and normalization explicitly after broader evidence.
  Merchant/date/payment fields must not be invented from a generic receipt schema.
  Later comparison of base/SFT/GRPO models will require fixed serialization,
  evaluation rules, and split discipline; none is implemented in Milestone 1.

## Verification and encountered limitations

The live script exited successfully with **3/3 rows, 3 decoded images, and 3 valid
nonempty `gt_parse` objects**. Package compatibility was checked with
`python -m pip check`; no broken requirements were found. Training frameworks
were not installed.

Offline checks exercised malformed JSON, missing/null/empty `gt_parse`, mixed
dictionary/list/null nesting, preview truncation, and corrupt-image handling.
Git ignore checks confirmed that the environment, dataset files, local audit
transcript, caches, outputs, and checkpoints are excluded. A draft scalar-count
transcription error was corrected against the script's aggregate output: 116
string leaves, with no sampled non-string scalar leaves.

Initial Conda networking failed inside the restricted sandbox; retrying with
network access succeeded. No Homebrew source build was launched. The Hub emitted
an unauthenticated-request/rate-limit warning but allowed this public audit;
authentication was not needed. Interpreter and pip paths were checked explicitly
inside `.venv`. VS Code's default interpreter setting was written, but its active
GUI selection cannot be verified from this terminal session.
