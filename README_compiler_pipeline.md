# Pokémon TCG Compiler Pipeline v0.1

This replaces the manual-batch workflow with a repeatable local pipeline.

## What the pipeline does

1. Reads `data/all_cards.csv`
2. Analyzes card text patterns
3. Deduplicates cards into safe effect groups
4. Auto-compiles simple/common effects into simulator JSON
5. Sends unparsed or ambiguous cards to a review queue

## Files

Copy these scripts into your project:

```text
scripts/analyze_effect_corpus.py
scripts/compile_cards_auto.py
scripts/validate_compiled_cards.py
```

## Recommended first run

From PowerShell:

```powershell
cd C:\Users\maran\Documents\tcg-deck-analyzer

New-Item -ItemType Directory -Force -Path data\reports
New-Item -ItemType Directory -Force -Path data\compiled_cards\auto

python scripts\analyze_effect_corpus.py `
  --input data\all_cards.csv `
  --output-dir data\reports `
  --only-with-text

python scripts\compile_cards_auto.py `
  --input data\all_cards.csv `
  --output-dir data\compiled_cards\auto `
  --report-dir data\reports `
  --only-with-text

python scripts\validate_compiled_cards.py `
  --input data\compiled_cards\auto\compiled_cards_all.json
```

## Main outputs

```text
data/reports/effect_corpus_summary.json
data/reports/effect_template_frequency.csv
data/reports/effect_tag_counts.csv
data/reports/effect_text_lines.csv
data/reports/card_text_tags.csv
data/reports/compiler_coverage.json
data/reports/review_queue.csv

data/compiled_cards/auto/compiled_cards_all.json
data/compiled_cards/auto/complete/compiled_cards_complete.json
data/compiled_cards/auto/partial/compiled_cards_partial.json
data/compiled_cards/auto/needs_review/compiled_cards_needs_human_review.json
```

## Important idea

The compiler is intentionally conservative. It marks a card as `partial` when it cannot safely parse the text. That is good: false confidence is worse than a review queue.

Improve the compiler iteratively by adding patterns for the most frequent rows in:

```text
data/reports/review_queue.csv
data/reports/compiler_coverage.json
data/reports/effect_template_frequency.csv
```
