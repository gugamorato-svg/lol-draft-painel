"""Widgets reutilizaveis: cards de stat, card de jogador, linha de matchup."""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class StatCard(QFrame):
    """Caixa com label superior pequena e um numero grande."""

    def __init__(self, title: str, value: str = "-", sub: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.setMinimumWidth(0)
        self.setMinimumHeight(54)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 5, 8, 5)
        v.setSpacing(1)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("cardHeader")
        self.title_lbl.setMinimumWidth(0)
        self.title_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("cardValue")
        self.value_lbl.setMinimumWidth(0)
        self.value_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.sub_lbl = QLabel(sub)
        self.sub_lbl.setObjectName("cardSub")
        self.sub_lbl.setMinimumWidth(0)
        self.sub_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        v.addWidget(self.title_lbl)
        v.addWidget(self.value_lbl)
        v.addWidget(self.sub_lbl)
        self.sub_lbl.setVisible(bool(sub))

    def set_value(self, value: str, sub: str = "") -> None:
        self.value_lbl.setPixmap(QPixmap())
        self.value_lbl.setText(value)
        self.sub_lbl.setText(sub)
        self.sub_lbl.setVisible(bool(sub))

    def set_icon_value(self, icon_path: str, fallback: str = "-", sub: str = "") -> None:
        pixmap = QPixmap(icon_path) if icon_path else QPixmap()
        if pixmap.isNull():
            self.set_value(fallback, sub=sub)
            return
        self.value_lbl.setText("")
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_lbl.setPixmap(
            pixmap.scaled(
                28,
                28,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self.sub_lbl.setText(sub)
        self.sub_lbl.setVisible(bool(sub))


class MiniStat(QFrame):
    """Stat pequeno para caber todos os jogadores sem scroll interno."""

    def __init__(self, title: str, value: str = "-", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("miniStat")
        self.setMinimumWidth(0)
        self.setFixedHeight(36)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        v = QVBoxLayout(self)
        v.setContentsMargins(5, 2, 5, 2)
        v.setSpacing(0)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("miniStatTitle")
        self.title_lbl.setMinimumWidth(0)
        self.title_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.value_lbl = QLabel(value)
        self.value_lbl.setObjectName("miniStatValue")
        self.value_lbl.setMinimumWidth(0)
        self.value_lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)

        v.addWidget(self.title_lbl)
        v.addWidget(self.value_lbl)

    def set_value(self, value: str) -> None:
        self.value_lbl.setText(value)

    _TONE_COLOR = {"good": "#5eff9a", "bad": "#ff6b6b"}

    def set_tone(self, tone: str = "") -> None:
        """Pinta o valor de verde/vermelho; tom vazio volta ao branco do tema."""
        color = self._TONE_COLOR.get(tone)
        self.value_lbl.setStyleSheet(f"color:{color};" if color else "")


class PlayerCard(QFrame):
    """Card de jogador: nome + role/liga + cards de K/D e WR do campeao."""

    def __init__(
        self,
        name: str = "-",
        role: str = "-",
        league: str = "-",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("playerCard")
        self.setMinimumWidth(0)
        self.setFixedHeight(112)  # +itens ao vivo
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 4, 8, 4)
        v.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(5)
        identity = QVBoxLayout()
        identity.setSpacing(0)
        self.name_lbl = QLabel(name)
        self.name_lbl.setStyleSheet("font-weight:700;color:#ffffff;font-size:12px;")
        self.name_lbl.setMinimumWidth(0)
        self.role_lbl = QLabel(f"{role} - {league}")
        self.role_lbl.setObjectName("cardSub")
        self.role_lbl.setMinimumWidth(0)
        identity.addWidget(self.name_lbl)
        identity.addWidget(self.role_lbl)
        self.champ_icon_lbl = QLabel("")
        self.champ_icon_lbl.setObjectName("champIcon")
        self.champ_icon_lbl.setFixedSize(28, 28)
        self.champ_icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_icon_lbl = QLabel("")
        self.empty_icon_lbl.setObjectName("emptyIcon")
        self.empty_icon_lbl.setFixedSize(28, 28)
        # Badge de nivel ao vivo (estilo Andy Danger). Vazio fora de jogo live.
        self.level_lbl = QLabel("")
        self.level_lbl.setFixedWidth(26)
        self.level_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.level_lbl.setStyleSheet(
            "color:#ffd479;font-size:12px;font-weight:800;")
        head.addLayout(identity, 1)
        head.addWidget(self.level_lbl)
        head.addWidget(self.champ_icon_lbl)
        head.addWidget(self.empty_icon_lbl)
        v.addLayout(head)

        stats = QHBoxLayout()
        stats.setSpacing(4)
        self.card_player = MiniStat("Player", "-/-")
        self.card_champ = MiniStat("Campeao", "-/-")
        self.card_champ_year_wr = MiniStat("WR Ano", "-/-")
        self.card_champ_career_wr = MiniStat("WR Carreira", "-/-")
        # WR do campeao no patch da partida (todos os jogadores pro, nao so este).
        self.card_champ_patch_wr = MiniStat("WR Patch", "-/-")
        for card in (
            self.card_player,
            self.card_champ,
            self.card_champ_year_wr,
            self.card_champ_career_wr,
            self.card_champ_patch_wr,
        ):
            # K/D/A ("3.25/2.10/5.40") precisa de mais largura que os WR ("12/20",
            # "63% (91)"): sem peso, 5 cards iguais cortam o K/D/A na janela de 1600px.
            stats.addWidget(card, 4 if card in (self.card_player, self.card_champ) else 3)
        v.addLayout(stats)

        # Linha de ITENS ao vivo (6 slots + trinket). Vazia ate haver jogo live.
        items = QHBoxLayout()
        items.setSpacing(2)
        self.item_lbls: list[QLabel] = []
        for _ in range(7):
            lbl = QLabel("")
            lbl.setObjectName("itemSlot")
            lbl.setFixedSize(20, 20)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.item_lbls.append(lbl)
            items.addWidget(lbl)
        items.addStretch(1)
        v.addLayout(items)

    def set_items(self, item_icon_paths: list) -> None:
        """Preenche os slots de item com os icones (lista de caminhos locais)."""
        paths = list(item_icon_paths or [])
        for i, lbl in enumerate(self.item_lbls):
            path = paths[i] if i < len(paths) else ""
            pix = QPixmap(path) if path else QPixmap()
            if pix.isNull():
                lbl.setPixmap(QPixmap())
                lbl.setText("")
            else:
                lbl.setPixmap(pix.scaled(
                    20, 20, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

    def clear_items(self) -> None:
        for lbl in self.item_lbls:
            lbl.setPixmap(QPixmap())
            lbl.setText("")
        self.level_lbl.setText("")

    def set_level(self, level) -> None:
        self.level_lbl.setText(
            f"Lv{int(level)}" if isinstance(level, (int, float)) and level else "")

    def set_patch_wr(self, value: str, title: str = "WR Patch",
                     tooltip: str = "", tone: str = "") -> None:
        card = self.card_champ_patch_wr
        card.title_lbl.setText(title)
        card.set_value(value)
        card.set_tone(tone)
        card.setToolTip(tooltip)

    def set_data(
        self,
        name: str,
        role: str,
        league: str,
        player_kd: str,
        champ_kd: str,
        champion_wr_year: str,
        champion_wr_career: str,
        champion_icon_path: str = "",
        champion_name: str = "",
        data_status: str = "",
        status_detail: str = "",
        resolve_reason: str = "",
    ) -> None:
        self.name_lbl.setText(name)
        self.role_lbl.setText(f"{role} - {league}")
        self._set_champion_icon(champion_icon_path, champion_name)
        self.card_player.set_value(player_kd)
        self.card_champ.set_value(champ_kd)
        self.card_champ_year_wr.set_value(champion_wr_year)
        self.card_champ_career_wr.set_value(champion_wr_career)
        tooltip = _player_tooltip(
            name=name,
            champion=champion_name,
            status=data_status,
            detail=status_detail,
            reason=resolve_reason,
        )
        self.setToolTip(tooltip)
        self.name_lbl.setToolTip(tooltip)
        self.role_lbl.setToolTip(tooltip)
        self.champ_icon_lbl.setToolTip(champion_name or "")

    def _set_champion_icon(self, icon_path: str, champion_name: str = "") -> None:
        self.champ_icon_lbl.setPixmap(QPixmap())
        self.champ_icon_lbl.setText("")
        pixmap = QPixmap(icon_path) if icon_path else QPixmap()
        if pixmap.isNull():
            return
        self.champ_icon_lbl.setText("")
        self.champ_icon_lbl.setPixmap(
            pixmap.scaled(
                28,
                28,
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


def _player_tooltip(
    *,
    name: str,
    champion: str,
    status: str,
    detail: str,
    reason: str,
) -> str:
    lines = [f"Player: {name}"]
    if champion:
        lines.append(f"Campeao: {champion}")
    if reason:
        lines.append(f"Resolver: {reason}")
    if status:
        lines.append(f"Dados: {status}")
    if detail:
        lines.append(f"Detalhe: {detail}")
    return "\n".join(lines)


class _AltProbBar(QWidget):
    """Barra fina de um modelo alternativo: titulo, % blue/red e delta vs o principal."""

    def __init__(self, title: str, tooltip: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet(
            "color:#9aa0a6; font-size:10px; font-weight:600; "
            "letter-spacing:0.5px; padding-top: 4px;"
        )
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if tooltip:
            self.setToolTip(tooltip)
        v.addWidget(self.title_lbl)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.blue_pct_lbl = QLabel("-")
        self.blue_pct_lbl.setStyleSheet("color:#7ab8ff; font-size:14px; font-weight:700;")
        self.blue_pct_lbl.setMinimumWidth(46)
        self.blue_pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.blue_pct_lbl)

        bar_outer = QFrame()
        bar_outer.setObjectName("probBarOuter")
        self._bar_layout = QHBoxLayout(bar_outer)
        self._bar_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_layout.setSpacing(2)
        for name in ("probBarBlue", "probBarRed"):
            bar = QFrame()
            bar.setObjectName(name)
            bar.setMinimumHeight(10)
            bar.setMaximumHeight(12)
            self._bar_layout.addWidget(bar, 50)
        row.addWidget(bar_outer, 1)

        self.red_pct_lbl = QLabel("-")
        self.red_pct_lbl.setStyleSheet("color:#ff8a8a; font-size:14px; font-weight:700;")
        self.red_pct_lbl.setMinimumWidth(46)
        self.red_pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self.red_pct_lbl)
        v.addLayout(row)

        # diff numerico (-2pp vs principal etc)
        self.delta_lbl = QLabel("")
        self.delta_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.delta_lbl.setStyleSheet("color:#9aa0a6; font-size:10px; padding: 0 0 4px 0;")
        v.addWidget(self.delta_lbl)
        self.setVisible(False)

    def set_prob(self, p_blue: Optional[float], main_blue_int: int,
                 title: Optional[str] = None) -> None:
        """None esconde a barra (modelo nao carregado ou sem previsao)."""
        if p_blue is None:
            self.blue_pct_lbl.setText("-")
            self.red_pct_lbl.setText("-")
            self.delta_lbl.setText("")
            self.setVisible(False)
            return
        if title:
            self.title_lbl.setText(title)
        blue = round(p_blue * 100)
        red = 100 - blue
        delta = blue - main_blue_int
        arrow = "^" if delta > 1 else ("v" if delta < -1 else "=")
        self.blue_pct_lbl.setText(f"{blue}%")
        self.red_pct_lbl.setText(f"{red}%")
        self._bar_layout.setStretch(0, max(blue, 1))
        self._bar_layout.setStretch(1, max(red, 1))
        self.delta_lbl.setText(f"{arrow} {delta:+d}pp vs principal")
        self.setVisible(True)


class WinProbWidget(QFrame):
    """Card de previsao de probabilidade de vitoria Blue vs Red.

    Exibe barra visual proporcional + percentuais + breakdown dos 3 fatores.
    """

    _OBJ = {"blue": "factorBlue", "red": "factorRed", "neutro": "factorNeutral"}
    _ARROW = {"blue": "^", "red": "v", "neutro": "~"}
    _CONF_LABEL = {"baixa": "Baixa", "media": "Media", "alta": "Alta"}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        head = QHBoxLayout()
        ttl = QLabel("PREVISAO DE VITORIA")
        ttl.setObjectName("winProbTitle")
        head.addWidget(ttl)
        head.addStretch(1)
        self.confidence_lbl = QLabel("")
        self.confidence_lbl.setObjectName("cardSub")
        head.addWidget(self.confidence_lbl)
        v.addLayout(head)

        bar_row = QHBoxLayout()
        bar_row.setSpacing(10)

        self.blue_pct_lbl = QLabel("-")
        self.blue_pct_lbl.setObjectName("winPctBlue")
        self.blue_pct_lbl.setMinimumWidth(46)
        self.blue_pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_row.addWidget(self.blue_pct_lbl)

        bar_outer = QFrame()
        bar_outer.setObjectName("probBarOuter")
        self._bar_layout = QHBoxLayout(bar_outer)
        self._bar_layout.setContentsMargins(0, 0, 0, 0)
        self._bar_layout.setSpacing(2)
        self._blue_bar = QFrame()
        self._blue_bar.setObjectName("probBarBlue")
        self._blue_bar.setMinimumHeight(16)
        self._red_bar = QFrame()
        self._red_bar.setObjectName("probBarRed")
        self._red_bar.setMinimumHeight(16)
        self._bar_layout.addWidget(self._blue_bar, 50)
        self._bar_layout.addWidget(self._red_bar, 50)
        bar_row.addWidget(bar_outer, 1)

        self.red_pct_lbl = QLabel("-")
        self.red_pct_lbl.setObjectName("winPctRed")
        self.red_pct_lbl.setMinimumWidth(46)
        self.red_pct_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_row.addWidget(self.red_pct_lbl)
        v.addLayout(bar_row)

        # ===== Barras alternativas (modelos sem a forca do time) =====
        # Util para cacar valor em underdog com draft superior ao favorito historico.
        self.no_pre_bar = _AltProbBar("So draft (sem pre-jogo)")
        v.addWidget(self.no_pre_bar)
        self.comfort_bar = _AltProbBar(
            "Draft + comfort do jogador (sem time)",
            tooltip=(
                "Draft + WR, volume e % do tempo de cada jogador no campeao.\n"
                "Sem a forca do time (Elo, forma, h2h).\n"
                "No backtest de 2026, foi o que melhor aguentou os playoffs."
            ),
        )
        v.addWidget(self.comfort_bar)

        factors_row = QHBoxLayout()
        factors_row.setSpacing(6)
        self._factor_lbls: List[QLabel] = []
        for _ in range(5):  # ate 5 fatores (4 do modelo treinado + champion MU opcional)
            fl = QLabel("-")
            fl.setObjectName("factorNeutral")
            fl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fl.setMinimumWidth(0)
            fl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            factors_row.addWidget(fl)
            self._factor_lbls.append(fl)
        v.addLayout(factors_row)

    def set_prediction(
        self,
        pred: Dict,
        blue_name: str = "Blue",
        red_name: str = "Red",
    ) -> None:
        p_blue = pred["win_prob_blue"]
        p_red = pred["win_prob_red"]
        blue_int = round(p_blue * 100)
        red_int = 100 - blue_int

        self.blue_pct_lbl.setText(f"{blue_int}%")
        self.red_pct_lbl.setText(f"{red_int}%")
        self._bar_layout.setStretch(0, max(blue_int, 1))
        self._bar_layout.setStretch(1, max(red_int, 1))

        conf = pred.get("confidence", "")
        self.confidence_lbl.setText(
            f"Confianca: {self._CONF_LABEL.get(conf, conf)}"
        )

        # Modelos alternativos (somem se o modelo nao foi carregado)
        self.no_pre_bar.set_prob(
            pred.get("win_prob_blue_no_pre"), blue_int,
            title=("Draft puro (sem time/player)"
                   if pred.get("no_pre_model_type") == "champs_only"
                   else "So draft (sem pre-jogo)"),
        )
        self.comfort_bar.set_prob(pred.get("win_prob_blue_comfort"), blue_int)

        side_name = {
            "blue": blue_name[:10],
            "red": red_name[:10],
            "neutro": "Neutro",
        }
        factors_list = pred.get("factors", [])
        for i, lbl in enumerate(self._factor_lbls):
            if i < len(factors_list):
                factor = factors_list[i]
                favors = factor.get("favors", "neutro")
                text = (
                    f"{self._ARROW[favors]} {side_name[favors]}\n"
                    f"{factor['name']}\n"
                    f"{factor.get('detail', '')}"
                )
                lbl.setText(text)
                lbl.setObjectName(self._OBJ[favors])
                lbl.setVisible(True)
            else:
                lbl.setVisible(False)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def reset(self) -> None:
        self.blue_pct_lbl.setText("-")
        self.red_pct_lbl.setText("-")
        self._bar_layout.setStretch(0, 50)
        self._bar_layout.setStretch(1, 50)
        self.confidence_lbl.setText("")
        self.no_pre_bar.set_prob(None, 50)
        self.comfort_bar.set_prob(None, 50)
        for fl in self._factor_lbls:
            fl.setText("-")
            fl.setObjectName("factorNeutral")
            fl.style().unpolish(fl)
            fl.style().polish(fl)


class MatchupRow(QFrame):
    """Linha do comparador player x player por lane."""

    def __init__(self, role: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)
        self.role_lbl = QLabel(role)
        self.role_lbl.setMinimumWidth(36)
        self.role_lbl.setObjectName("cardHeader")
        self.players_lbl = QLabel("- vs -")
        self.players_lbl.setStyleSheet("color:#ffffff;font-weight:600;")
        self.delta_lbl = QLabel("Delta kills 0.00")
        self.delta_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.delta_lbl.setMinimumWidth(110)

        h.addWidget(self.role_lbl)
        h.addWidget(self.players_lbl, 1)
        h.addWidget(self.delta_lbl)

    def set_data(self, role: str, blue: str, red: str, delta_kills: float | None) -> None:
        self.role_lbl.setText(role)
        self.players_lbl.setText(f"{blue}  vs  {red}")
        if delta_kills is None:
            self.delta_lbl.setText("Delta kills -")
            self.delta_lbl.setObjectName("cardSub")
        else:
            sign = "+" if delta_kills > 0 else ""
            self.delta_lbl.setText(f"Delta kills {sign}{delta_kills:.2f}")
            self.delta_lbl.setObjectName(
                "deltaPositive" if delta_kills > 0 else "deltaNegative"
            )
        # forca re-aplicacao do QSS:
        self.delta_lbl.style().unpolish(self.delta_lbl)
        self.delta_lbl.style().polish(self.delta_lbl)






_LIVE_OBJ = [
    ("dragons", "\U0001f409"),   # \ud83d\udc09
    ("barons", "\U0001f52e"),    # \ud83d\udd2e (roxo ~ baron)
    ("towers", "\U0001f3f0"),    # \ud83c\udff0
    ("gold", "\U0001fa99"),      # \ud83e\ude99
    ("kills", "\u2694"),          # \u2694
]
_BLUE = "#7ab8ff"
_RED = "#ff8a8a"


def _obj_stat(icon: str, color: str):
    """Mini-widget vertical: icone em cima, contagem embaixo. Retorna (widget, count_lbl)."""
    w = QWidget()
    vl = QVBoxLayout(w)
    vl.setContentsMargins(3, 0, 3, 0)
    vl.setSpacing(0)
    ic = QLabel(icon)
    ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
    ic.setStyleSheet(f"font-size:13px;color:{color};")
    ct = QLabel("0")
    ct.setAlignment(Qt.AlignmentFlag.AlignCenter)
    ct.setStyleSheet(f"color:{color};font-size:12px;font-weight:800;")
    vl.addWidget(ic)
    vl.addWidget(ct)
    return w, ct


class LiveStatePanel(QFrame):
    """Painel AO VIVO estilo Andy Danger: objetivos espelhados (icone+contagem),
    barra de proporcao de ouro e barra de WR ao vivo."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        v = QVBoxLayout(self)
        v.setContentsMargins(12, 6, 12, 8)
        v.setSpacing(5)

        head = QHBoxLayout()
        ttl = QLabel("\u26a1 AO VIVO")
        ttl.setObjectName("winProbTitle")
        head.addWidget(ttl)
        head.addStretch(1)
        self.min_lbl = QLabel("")
        self.min_lbl.setObjectName("cardSub")
        head.addWidget(self.min_lbl)
        v.addLayout(head)

        # Linha de objetivos espelhada: blue (esq) <-> red (dir)
        obj = QHBoxLayout()
        obj.setSpacing(1)
        self.blue_obj = {}
        self.red_obj = {}
        for name, icon in _LIVE_OBJ:
            w, ct = _obj_stat(icon, _BLUE)
            self.blue_obj[name] = ct
            obj.addWidget(w)
        obj.addStretch(1)
        for name, icon in reversed(_LIVE_OBJ):
            w, ct = _obj_stat(icon, _RED)
            self.red_obj[name] = ct
            obj.addWidget(w)
        v.addLayout(obj)

        # Barra de proporcao de OURO (assinatura do Andy Danger)
        gold_outer = QFrame()
        gold_outer.setObjectName("probBarOuter")
        self._gold = QHBoxLayout(gold_outer)
        self._gold.setContentsMargins(0, 0, 0, 0)
        self._gold.setSpacing(0)
        self._gb = QFrame(); self._gb.setObjectName("probBarBlue"); self._gb.setMinimumHeight(8)
        self._gr = QFrame(); self._gr.setObjectName("probBarRed"); self._gr.setMinimumHeight(8)
        self._gold.addWidget(self._gb, 50)
        self._gold.addWidget(self._gr, 50)
        v.addWidget(gold_outer)

        # Barra de WR ao vivo (nosso diferencial)
        bar_row = QHBoxLayout()
        bar_row.setSpacing(8)
        self.blue_pct = QLabel("-")
        self.blue_pct.setObjectName("winPctBlue")
        self.blue_pct.setMinimumWidth(44)
        self.blue_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_row.addWidget(self.blue_pct)
        bar_outer = QFrame()
        bar_outer.setObjectName("probBarOuter")
        self._bar = QHBoxLayout(bar_outer)
        self._bar.setContentsMargins(0, 0, 0, 0)
        self._bar.setSpacing(2)
        self._blue_bar = QFrame(); self._blue_bar.setObjectName("probBarBlue"); self._blue_bar.setMinimumHeight(15)
        self._red_bar = QFrame(); self._red_bar.setObjectName("probBarRed"); self._red_bar.setMinimumHeight(15)
        self._bar.addWidget(self._blue_bar, 50)
        self._bar.addWidget(self._red_bar, 50)
        bar_row.addWidget(bar_outer, 1)
        self.red_pct = QLabel("-")
        self.red_pct.setObjectName("winPctRed")
        self.red_pct.setMinimumWidth(44)
        self.red_pct.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bar_row.addWidget(self.red_pct)
        v.addLayout(bar_row)
        wr_lbl = QLabel("WR ao vivo")
        wr_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wr_lbl.setStyleSheet("color:#9aa0a6;font-size:9px;letter-spacing:0.5px;")
        v.addWidget(wr_lbl)
        self.hide()

    @staticmethod
    def _counts(t: dict) -> dict:
        return {
            "dragons": t.get("dragon_count", 0),
            "barons": t.get("barons", 0),
            "towers": t.get("towers", 0),
            "gold": t.get("gold", 0),
            "kills": t.get("kills", 0),
        }

    def _fill(self, labels: dict, t: dict, base_color: str) -> None:
        c = self._counts(t)
        for name, lbl in labels.items():
            val = c.get(name, 0)
            lbl.setText(f"{val:,}" if name == "gold" else str(val))
            if name == "dragons" and t.get("has_soul"):
                lbl.setText("SOUL")
                lbl.setStyleSheet("color:#ffd479;font-size:11px;font-weight:800;")
            else:
                lbl.setStyleSheet(f"color:{base_color};font-size:12px;font-weight:800;")

    def set_state(self, wr: dict, state: dict) -> None:
        b = state.get("blue", {}) or {}
        r = state.get("red", {}) or {}
        self._fill(self.blue_obj, b, _BLUE)
        self._fill(self.red_obj, r, _RED)

        bg = int(b.get("gold", 0) or 0)
        rg = int(r.get("gold", 0) or 0)
        tot = max(1, bg + rg)
        self._gold.setStretch(0, max(1, int(round(100 * bg / tot))))
        self._gold.setStretch(1, max(1, int(round(100 * rg / tot))))

        pb = wr["win_prob_blue"] * 100
        pr = wr["win_prob_red"] * 100
        self.blue_pct.setText(f"{pb:.0f}%")
        self.red_pct.setText(f"{pr:.0f}%")
        self._bar.setStretch(0, max(1, int(round(pb))))
        self._bar.setStretch(1, max(1, int(round(pr))))

        gl = wr.get("gold_lead_blue", 0)
        xl = wr.get("xp_lead_blue", 0)
        sg = "+" if gl >= 0 else ""
        sx = "+" if xl >= 0 else ""
        # gold EFETIVO (itens completos) \u2014 mostra o lead efetivo se disponivel
        beg, reg = b.get("eff_gold"), r.get("eff_gold")
        eff_txt = ""
        if beg is not None and reg is not None:
            eg = beg - reg
            eff_txt = f"  (efetivo {'+' if eg >= 0 else ''}{eg:,})"
        base = (f"min {wr.get('minute', 0):.0f}  \u00b7  ouro {sg}{gl:,}{eff_txt}"
                f"  xp {sx}{xl:,}  \u00b7  ~3min")
        if str(state.get("game_state", "")).lower() == "finished":
            # mapa acabou: avisa (senao parece jogo vivo congelado no min final)
            self.min_lbl.setText("\u26a0 MAPA FINALIZADO \u2014 re-clique p/ o mapa atual")
            self.min_lbl.setStyleSheet("color:#ffd479;font-weight:700;")
        else:
            self.min_lbl.setText(base)
            self.min_lbl.setStyleSheet("")
        self.show()

    def clear_state(self) -> None:
        self.hide()
