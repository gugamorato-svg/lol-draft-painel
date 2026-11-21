"""comfort_kda_diff: KDA do jogador no campeao acima do normal do campeao (23/09/2026).

Roda com:  python -m unittest tests.test_comfort_kda -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_pipeline.temporal_context import (  # noqa: E402
    COMFORT_KDA_PRIOR, COMFORT_KDA_SCALE, TemporalContext,
)

ROLES = ("top", "jng", "mid", "bot", "sup")


def jogo(blue_mid_player, red_mid_player="rmid", mid_kda=None, blue_won=1):
    """Jogo com Ahri no mid azul. mid_kda=(k, d, a) do mid azul; None = sem K/D/A (ao vivo)."""
    g = {"blue_won": blue_won, "blue_teamid": "T1", "red_teamid": "T2", "patch": 16.1,
         "league": "LCK", "game": 1}
    for r in ROLES:
        g[f"blue_{r}"] = "Ahri" if r == "mid" else f"B{r}"
        g[f"red_{r}"] = "Syndra" if r == "mid" else f"R{r}"
        g[f"blue_player_{r}"] = blue_mid_player if r == "mid" else f"b{r}"
        g[f"red_player_{r}"] = red_mid_player if r == "mid" else f"r{r}"
        for side in ("blue", "red"):
            for s, v in (("kills", 2), ("deaths", 2), ("assists", 4)):
                g[f"{side}_{s}_{r}"] = v
    if mid_kda is not None:
        g["blue_kills_mid"], g["blue_deaths_mid"], g["blue_assists_mid"] = mid_kda
    else:
        g = {k: v for k, v in g.items() if not any(f"_{s}_" in k for s in ("kills", "deaths", "assists"))}
    return g


class ComfortKdaTests(unittest.TestCase):
    def test_no_history_is_zero_and_no_leak(self):
        ctx = TemporalContext()
        g = jogo("faker", mid_kda=(10, 0, 10))
        self.assertEqual(ctx.build_features(g)["comfort_kda_diff"], 0.0)   # antes do update
        ctx.update(g)

    def test_kda_relative_to_champion_norm(self):
        ctx = TemporalContext()
        # 1) outro jogador de Ahri mid com KDA (2+4)/(2+1)=2 -> normal da Ahri mid = 2
        ctx.update(jogo("outro", mid_kda=(2, 2, 4)))
        # 2) faker de Ahri com KDA (5+11)/(1+1)=8 -> 6 acima do normal
        ctx.update(jogo("faker", mid_kda=(5, 1, 11)))
        f = ctx.build_features(jogo("faker"))["comfort_kda_diff"]
        esperado = 6.0 / (1 + COMFORT_KDA_PRIOR) / COMFORT_KDA_SCALE / 5.0
        self.assertAlmostEqual(f, esperado, places=9)
        self.assertGreater(f, 0)

    def test_red_side_is_negative(self):
        ctx = TemporalContext()
        ctx.update(jogo("outro", mid_kda=(2, 2, 4)))
        ctx.update(jogo("faker", mid_kda=(5, 1, 11)))
        g = jogo("zeka")
        g["red_mid"], g["blue_mid"] = "Ahri", "Syndra"          # faker de vermelho, de Ahri
        g["red_player_mid"], g["blue_player_mid"] = "faker", "zeka"
        self.assertLess(ctx.build_features(g)["comfort_kda_diff"], 0)

    def test_live_game_without_stats_does_not_break(self):
        ctx = TemporalContext()
        g = jogo("faker", mid_kda=None)          # feed ao vivo: sem K/D/A
        ctx.build_features(g)
        ctx.update(g)
        self.assertEqual(len(ctx.player_champion_kda), 0)


if __name__ == "__main__":
    unittest.main()
