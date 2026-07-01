from src.game_log.parser import parse_battle_log
from src.game_log.reducer import build_replay_frames


SAMPLE_LOG = """
Setup
FrunkUke chose tails for the opening coin flip.
BananaHammer33 won the coin toss.
BananaHammer33 decided to go second.
FrunkUke drew 7 cards for the opening hand.
- 7 drawn cards.
   • (me1_131) Ultra Ball, (mee_1) Basic Grass Energy, (sv5_123) Raging Bolt ex
BananaHammer33 drew 7 cards for the opening hand.
- 7 drawn cards.
FrunkUke played (sv5_123) Raging Bolt ex to the Active Spot.
BananaHammer33 played (sv8-5_4) Budew to the Active Spot.

FrunkUke's Turn
FrunkUke drew (me1_115) Energy Switch.
FrunkUke attached (mee_1) Basic Grass Energy to (sv5_123) Raging Bolt ex in the Active Spot.
FrunkUke's (sv5_123) Raging Bolt ex used Bellowing Thunder on BananaHammer33’s (sv8-5_4) Budew for 70 damage.
BananaHammer33's (sv8-5_4) Budew was Knocked Out!
FrunkUke took a Prize card.
(sv5_145) Ciphermaniac's Codebreaking was added to FrunkUke's hand.
Opponent conceded. FrunkUke wins.
"""


def test_parse_basic_battle_log_events():
    events = parse_battle_log(SAMPLE_LOG)

    assert any(e.event_type == "play_to_active" for e in events)
    assert any(e.event_type == "attach_energy" for e in events)
    assert any(e.event_type == "take_prize" for e in events)
    assert any(e.event_type == "add_to_hand_revealed" for e in events)

    card_ids = [card.exported_id for e in events for card in e.cards]
    assert "sv5_123" in card_ids
    assert "sv8-5_4" in card_ids


def test_reducer_tracks_revealed_prize():
    events = parse_battle_log(SAMPLE_LOG)
    frames = build_replay_frames(events)
    final = frames[-1].state

    frunk = final.players["FrunkUke"]
    assert len(frunk.prizes_taken) == 1
    assert frunk.prizes_taken[0].exported_id == "sv5_145"
