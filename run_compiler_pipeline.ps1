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
