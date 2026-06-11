# Pokémon TCG card JSON structure v1

Generated from analysis of the uploaded `all_cards.csv`.

Dataset observed:
- Rows: 7253
- Columns: 27
- Supertypes: {'Pokémon': np.int64(5787), 'Trainer': np.int64(1246), 'Energy': np.int64(220)}
- Non-empty rules text: 2753
- Non-empty abilities text: 1347
- Non-empty attacks text: 5798

The key design decision is to keep two layers:

1. `sources`: exact raw API/card text.
2. `compiled_effects`: parsed simulator actions.

Never throw away raw text. If the parser is unsure, preserve the text in `parser.unparsed_text` and set a low confidence score.

Main files:
- `pokemon_card_schema_v1.json`: JSON Schema for card definitions.
- `example_compiled_card_me4_122.json`: example compiled Mega Greninja ex.
- `pokemon_effect_ops_v1.md`: operation vocabulary for the simulator.
