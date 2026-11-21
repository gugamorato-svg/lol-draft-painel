"""Janela principal do LoL Painel."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from api.cache import JsonCache
from api.esports import EsportsClient
from api.leaguepedia import LeaguepediaClient
from api.livefeed import LiveFeedClient
from api.oracle_stats import OracleStatsClient
from models.predictor import predict_win_prob
from ui.theme import QSS
from ui.widgets import MatchupRow, PlayerCard, StatCard, WinProbWidget, LiveStatePanel
from ui.worker import (LivePicksWorker, LiveWorker, MatchupWorker, LiveItemsWorker,
                       LiveStateWorker)
from api.item_icons import ItemIconCache


CACHE_PATH = Path.home() / ".lol_painel" / "cache.sqlite"

# 17/10 (dono): botao "Mapa N" p/ abrir qualquer mapa da serie (ex.: rever o Mapa 2 com o 3
# rolando). So mexe na tela: ui/worker.py entra na assinatura do codigo do teste prospectivo
# (models/model_meta.UPSTREAM_SOURCES) e nao pode mudar -- o worker ja aceita force_game_id.
_SERIE_ESTADO = {"completed": " ✓", "inProgress": " ● ao vivo"}
_SERIE_QSS = ("QPushButton { padding: 3px 12px; }"
              "QPushButton:checked { background-color: #2563eb; border-color: #2563eb; }")


class ReloadDataWorker(QThread):
    """24/10 (dono): botao "Recarregar dados". Le o Oracle de novo do disco e manda o modelo
    reler pesos + snapshot na proxima previsao (zera o cache em memoria do trained_predictor;
    nenhum arquivo protegido do teste prospectivo e alterado)."""
    done = pyqtSignal(object, str)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            oracle = OracleStatsClient()
            from models import trained_predictor as tp
            tp._state = None
            tp._load()
            ultimo = oracle.teams["date"].max() if "date" in oracle.teams else None
            treino = [v for v in (tp.model_version().get("treino_ate") or {}).values() if v]
            info = (f"jogos ate {ultimo:%d/%m %H:%M}" if ultimo is not None and ultimo == ultimo else "")
            if treino:
                info += f" | modelo treinado ate {max(treino)[8:10]}/{max(treino)[5:7]} {max(treino)[11:16]}"
            self.done.emit(oracle, info)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SeriesGamesWorker(QThread):
    """Mapas da serie (id, numero, estado) pelo getEventDetails da Riot, fora da thread da tela."""
    ready = pyqtSignal(str, list)

    def __init__(self, esports, event_id: str) -> None:
        super().__init__()
        self.esports, self.event_id = esports, event_id

    def run(self) -> None:
        try:
            det = self.esports.get_event_details(self.event_id, fresh=True)
            games = [{"id": str(g["id"]), "number": int(g.get("number") or 0), "state": g.get("state", "")}
                     for g in (det.get("match") or {}).get("games", []) if g.get("id")]
        except Exception:  # noqa: BLE001 - sem a lista, so nao mostra os botoes
            games = []
        self.ready.emit(self.event_id, sorted(games, key=lambda g: g["number"]))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LoL Painel")
        self.resize(1600, 920)
        self.setStyleSheet(QSS)

        cache = JsonCache(CACHE_PATH)
        self.lp = LeaguepediaClient(cache=cache)
        self.esports = EsportsClient(cache=cache)
        self.livefeed = LiveFeedClient()
        # OracleStatsClient: stats locais (substitui Leaguepedia para queries
        # que precisam de 2026 ou sao chamadas em rajada).
        print("[*] Carregando Oracle's Elixir CSVs em memoria...")
        self.oracle = OracleStatsClient()
        print(f"[OK] Oracle stats prontos ({len(self.oracle.players)} player-rows, "
              f"{len(self.oracle.teams)} team-rows)")
        self.current_worker = None

        # Itens ao vivo
        self.item_icons = ItemIconCache()
        self.items_worker = None
        self._sides_swapped = False
        self._last_data: Optional[Dict[str, Any]] = None
        self._last_items: Dict[str, Any] = {}
        self._roles = ("top", "jng", "mid", "bot", "sup")

        # Mapas da serie (botoes "Mapa N")
        self._series_event_id: Optional[str] = None
        self._shown_game_id: Optional[str] = None
        self.series_worker = None
        self.reload_worker = None

        # Estado + WR ao vivo
        self.state_worker = None
        self._last_state_wr = None   # (wr, state) do feed p/ re-aplicar no swap
        self._last_pred = None       # ultima previsao pre-draft (p/ P_comp)

        self._build_ui()
        self._set_status("Pronto. Informe dois times e clique Buscar.")

    # ---------- construcao ----------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # Top bar: inputs
        outer.addLayout(self._build_topbar())
        outer.addWidget(self._build_series_bar())

        # Conteudo principal: 3 colunas
        main = QHBoxLayout()
        main.setSpacing(10)
        main.addWidget(self._build_team_panel(blue=True), 3)
        main.addWidget(self._build_center_panel(), 4)
        main.addWidget(self._build_team_panel(blue=False), 3)
        outer.addLayout(main, 1)

        # Live matches list
        outer.addWidget(self._build_live_section())

        sb = QStatusBar()
        self.setStatusBar(sb)

    def _build_topbar(self) -> QHBoxLayout:
        h = QHBoxLayout()
        h.setSpacing(8)

        self.ed_blue = QLineEdit()
        self.ed_blue.setPlaceholderText("Time Blue (ex: T1, Gen.G, G2 Esports)")
        self.ed_red = QLineEdit()
        self.ed_red.setPlaceholderText("Time Red (ex: Hanwha Life Esports, KT Rolster)")

        self.cb_year = QComboBox()
        now_year = datetime.now().year
        self.cb_year.addItems([str(y) for y in range(now_year, now_year - 5, -1)])

        self.btn_search = QPushButton("Buscar")
        self.btn_search.setObjectName("primary")
        self.btn_search.clicked.connect(self._on_search)

        self.btn_live = QPushButton("Ver partidas ao vivo")
        self.btn_live.clicked.connect(self._on_load_live)

        # Corrige inversao de lado do feed (ex.: mapa anterior de serie BO,
        # Tencent so publica matchInfos apos o mapa acabar). Troca blue<->red
        # localmente, sem re-buscar (senao re-inverteria).
        self.btn_swap = QPushButton("⇄ Inverter lados")
        self.btn_swap.setToolTip("Troca Blue <-> Red (corrige lado invertido do feed)")
        self.btn_swap.clicked.connect(self._swap_sides)
        self.btn_swap.setEnabled(False)

        self.btn_reload = QPushButton("⟳ Recarregar dados")
        self.btn_reload.setToolTip("Lê de novo os jogos do Oracle e os pesos do modelo do disco "
                                   "(sem fechar o app). Leva ~1-2 min.")
        self.btn_reload.clicked.connect(self._on_reload_data)

        h.addWidget(QLabel("Blue:"))
        h.addWidget(self.ed_blue, 2)
        h.addSpacing(8)
        h.addWidget(QLabel("Red:"))
        h.addWidget(self.ed_red, 2)
        h.addSpacing(8)
        h.addWidget(QLabel("Ano:"))
        h.addWidget(self.cb_year)
        h.addStretch(1)
        h.addWidget(self.btn_reload)
        h.addWidget(self.btn_swap)
        h.addWidget(self.btn_live)
        h.addWidget(self.btn_search)
        return h

    def _build_series_bar(self) -> QWidget:
        """Faixa "Serie: Mapa 1 | Mapa 2 | Mapa 3": so aparece em partida ao vivo da Riot."""
        bar = QFrame()
        h = QHBoxLayout(bar)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(6)
        h.addWidget(QLabel("Série:"))
        self.series_btns = QHBoxLayout()
        self.series_btns.setSpacing(6)
        h.addLayout(self.series_btns)
        h.addStretch(1)
        bar.setStyleSheet(_SERIE_QSS)
        bar.setVisible(False)
        self.series_bar = bar
        return bar

    def _load_series(self) -> None:
        ev = self._series_event_id
        if not ev or str(ev).startswith("lplqq:"):   # Tencent nao tem a lista de mapas
            self.series_bar.setVisible(False)
            return
        w = SeriesGamesWorker(self.esports, ev)
        w.ready.connect(self._on_series_ready)
        w.start()
        self.series_worker = w

    def _on_series_ready(self, event_id: str, games: List[Dict[str, Any]]) -> None:
        if event_id != self._series_event_id:
            return                                    # resposta de uma serie antiga
        while self.series_btns.count():
            item = self.series_btns.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        games = [g for g in games if g["state"] != "unneeded"]
        if len(games) < 2:                            # MD1: nao ha o que escolher
            self.series_bar.setVisible(False)
            return
        for g in games:
            b = QPushButton(f"Mapa {g['number']}{_SERIE_ESTADO.get(g['state'], '')}")
            b.setCheckable(True)
            b.setChecked(g["id"] == self._shown_game_id)
            b.setEnabled(g["state"] in _SERIE_ESTADO)  # mapa que nao comecou nao tem draft
            b.setToolTip("Abrir este mapa da série" if b.isEnabled() else "Mapa ainda não começou")
            b.clicked.connect(lambda _=False, gid=g["id"]: self._open_series_game(gid))
            self.series_btns.addWidget(b)
        self.series_bar.setVisible(True)

    def _open_series_game(self, game_id: str) -> None:
        if self._series_event_id:
            self._on_search_live(self._series_event_id, force_game_id=game_id)

    def _on_live_search_failed(self, err: str) -> None:
        """Partida ao vivo que nao carregou (ex.: "Mapa 3 em transicao"): aviso na barra, sem
        janela de erro, e a faixa da serie aparece mesmo assim -> da p/ abrir o mapa anterior."""
        self.btn_search.setEnabled(True)
        self.progress_lbl.setText(err.splitlines()[0][:90] if err else "")
        self._set_status(" ".join(err.splitlines()))
        self._load_series()

    def _on_series_game_failed(self, err: str) -> None:
        """Mapa escolhido sem draft no feed: aviso na barra, sem janela de erro."""
        self.btn_search.setEnabled(True)
        self.progress_lbl.setText("")
        self._set_status(err)
        self._load_series()                           # volta o destaque p/ o mapa exibido

    def _on_reload_data(self) -> None:
        """Solta o Oracle velho ANTES de ler o novo (o app ocupa ~0,9 GB; os dois juntos
        podem faltar RAM). Buscar/partidas ao vivo ficam travados ate terminar."""
        import gc
        for b in (self.btn_reload, self.btn_search, self.btn_live):
            b.setEnabled(False)
        self.progress_lbl.setText("Recarregando dados...")
        self._set_status("Recarregando jogos do Oracle e pesos do modelo (~1-2 min)...")
        w = self.current_worker
        if w is not None and not w.isRunning():
            self.current_worker = None
        self.oracle = None
        gc.collect()
        rw = ReloadDataWorker()
        rw.done.connect(self._on_reload_done)
        rw.failed.connect(self._on_reload_failed)
        rw.start()
        self.reload_worker = rw

    def _on_reload_done(self, oracle, info: str) -> None:
        self.oracle = oracle
        for b in (self.btn_reload, self.btn_search, self.btn_live):
            b.setEnabled(True)
        self.progress_lbl.setText("")
        self._set_status(f"Dados recarregados: {info}" if info else "Dados recarregados.")
        # refaz a tela atual com o dado novo
        if self._series_event_id and self._shown_game_id:
            self._on_search_live(self._series_event_id, force_game_id=self._shown_game_id)
        elif self._last_data is not None and self.ed_blue.text().strip() and self.ed_red.text().strip():
            self._on_search()

    def _on_reload_failed(self, err: str) -> None:
        self.btn_reload.setEnabled(True)
        self.progress_lbl.setText("")
        self._set_status(f"Falhou ao recarregar ({err}). Feche e abra o app.")
        QMessageBox.warning(self, "LoL Painel", f"Não consegui recarregar os dados:\n{err}\n\n"
                            "Feche e abra o app.")

    def _build_team_panel(self, blue: bool) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        title = QLabel("Stats Players Blue" if blue else "Stats Players Red")
        title.setObjectName("sectionTitle" if blue else "sectionTitleRed")
        v.addWidget(title)

        # Header stats (2x4 grid)
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(5)
        cards = {}
        labels = [
            ("kills", "Total Kills (media)"),
            ("time", "Tempo (media, min)"),
            ("over_k", "Over kills"),
            ("under_k", "Under kills"),
            ("over_t", "Over tempo"),
            ("under_t", "Under tempo"),
        ]
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (key, lbl), pos in zip(labels, positions):
            c = StatCard(lbl, "-")
            grid.addWidget(c, *pos)
            cards[key] = c
        v.addLayout(grid)

        # Lista fixa de jogadores: cinco cards sempre visiveis, sem scroll interno.
        players_wrap = QWidget()
        pv = QVBoxLayout(players_wrap)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(5)
        player_cards: List[PlayerCard] = []
        for _ in range(5):
            pc = PlayerCard()
            pv.addWidget(pc)
            player_cards.append(pc)
        v.addWidget(players_wrap)
        v.addStretch(1)

        if blue:
            self.blue_cards = cards
            self.blue_players = player_cards
        else:
            self.red_cards = cards
            self.red_players = player_cards
        return wrap

    def _build_center_panel(self) -> QWidget:
        wrap = QWidget()
        v = QVBoxLayout(wrap)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        self.win_prob_widget = WinProbWidget()
        v.addWidget(self.win_prob_widget)

        # Painel de estado AO VIVO (gold/objetivos/WR ao vivo)
        self.live_panel = LiveStatePanel()
        v.addWidget(self.live_panel)

        # Comparador Time x Time
        ttl = QLabel("Comparador Time x Time")
        ttl.setObjectName("sectionTitleNeutral")
        v.addWidget(ttl)

        row = QHBoxLayout()
        self.h2h_winrate = StatCard("Winrate (Blue vs Red)", "-")
        self.h2h_kdiff = StatCard("Kills (diff)", "-")
        self.h2h_ddiff = StatCard("Mortes (diff)", "-")
        row.addWidget(self.h2h_winrate)
        row.addWidget(self.h2h_kdiff)
        row.addWidget(self.h2h_ddiff)
        v.addLayout(row)

        # Comparador Player x Player
        ttl2 = QLabel("Comparador Player x Player (lanes)")
        ttl2.setObjectName("sectionTitleNeutral")
        v.addWidget(ttl2)

        self.lane_rows = {
            r: MatchupRow(r) for r in ("TOP", "JNG", "MID", "BOT", "SUP")
        }
        for r in ("TOP", "JNG", "MID", "BOT", "SUP"):
            v.addWidget(self.lane_rows[r])
        v.addStretch(1)

        # progress label
        self.progress_lbl = QLabel("")
        self.progress_lbl.setObjectName("cardSub")
        v.addWidget(self.progress_lbl)
        return wrap

    def _build_live_section(self) -> QWidget:
        wrap = QFrame()
        wrap.setObjectName("card")
        v = QVBoxLayout(wrap)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(3)
        ttl = QLabel("Partidas (ao vivo / proximas)")
        ttl.setObjectName("sectionTitleNeutral")
        v.addWidget(ttl)
        self.live_list = QListWidget()
        self.live_list.setMinimumHeight(180)
        self.live_list.setMaximumHeight(240)
        self.live_list.itemDoubleClicked.connect(self._on_pick_live)
        v.addWidget(self.live_list)
        return wrap

    # ---------- handlers ----------

    def _set_status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    def _on_search(self) -> None:
        blue = self.ed_blue.text().strip()
        red = self.ed_red.text().strip()
        if not blue or not red:
            QMessageBox.warning(self, "LoL Painel", "Informe os dois times.")
            return
        year = int(self.cb_year.currentText())
        self.btn_search.setEnabled(False)
        self.progress_lbl.setText("Buscando...")
        self._series_event_id = None                 # busca manual: sem serie ao vivo
        self.series_bar.setVisible(False)

        w = MatchupWorker(self.oracle, blue, red, year)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_matchup_ready)
        w.failed.connect(self._on_matchup_failed)
        w.start()
        self.current_worker = w

    def _on_progress(self, msg: str) -> None:
        self.progress_lbl.setText(msg)
        self._set_status(msg)

    def _on_matchup_failed(self, err: str) -> None:
        self.btn_search.setEnabled(True)
        self.progress_lbl.setText("")
        QMessageBox.critical(self, "Erro", err)

    def _on_matchup_ready(self, data: Dict[str, Any]) -> None:
        self.btn_search.setEnabled(True)
        self.progress_lbl.setText("")
        self._last_data = data
        self._sides_swapped = False
        self.btn_swap.setEnabled(True)
        self._render(data)
        self._shown_game_id = str(data.get("live_game_id") or "") or None
        if self._shown_game_id:
            self._load_series()
        # Itens + estado/WR ao vivo: so para partidas live (tem live_game_id)
        self._start_items_polling(data.get("live_game_id"))
        self._start_state_polling(data.get("live_game_id"))

    def _start_items_polling(self, game_id: Optional[str]) -> None:
        self._stop_items_polling()
        # limpa itens antigos dos cards
        for pc in list(self.blue_players) + list(self.red_players):
            pc.clear_items()
        if not game_id:
            return
        w = LiveItemsWorker(self.livefeed, self.item_icons, str(game_id),
                            roles=self._roles, interval=20)
        w.items_ready.connect(self._on_items)
        w.start()
        self.items_worker = w

    def _stop_items_polling(self) -> None:
        w = self.items_worker
        if w is not None:
            try:
                w.stop()
                w.wait(1500)
            except Exception:
                pass
            self.items_worker = None

    def _on_items(self, items: Dict[str, Any]) -> None:
        self._last_items = items or {}
        # respeita o swap manual de lados: se o usuario inverteu, troca a fonte
        left_side, right_side = ("red", "blue") if self._sides_swapped else ("blue", "red")
        for cards, side in ((self.blue_players, left_side), (self.red_players, right_side)):
            for role, pc in zip(self._roles, cards):
                pc.set_items(items.get(f"{side}_{role}", []))
                pc.set_level(items.get(f"{side}_{role}_level"))

    # ---------- estado + WR ao vivo ----------

    def _start_state_polling(self, game_id: Optional[str]) -> None:
        self._stop_state_polling()
        self.live_panel.clear_state()
        self._last_state_wr = None
        if not game_id:
            return
        pred = self._last_pred or {}
        p_comp = float(pred.get("win_prob_blue", 0.5))
        w = LiveStateWorker(self.livefeed, str(game_id), p_comp, interval=20)
        w.state_ready.connect(self._on_state)
        w.start()
        self.state_worker = w

    def _stop_state_polling(self) -> None:
        w = self.state_worker
        if w is not None:
            try:
                w.stop()
                w.wait(1500)
            except Exception:
                pass
            self.state_worker = None

    def _on_state(self, wr: Dict[str, Any], state: Dict[str, Any]) -> None:
        self._last_state_wr = (wr, state)
        self._apply_state(wr, state)

    def _apply_state(self, wr: Dict[str, Any], state: Dict[str, Any]) -> None:
        if self._sides_swapped:
            wr, state = self._flip_wr(wr), self._flip_state(state)
        self.live_panel.set_state(wr, state)

    @staticmethod
    def _flip_wr(wr: Dict[str, Any]) -> Dict[str, Any]:
        w = dict(wr)
        w["win_prob_blue"], w["win_prob_red"] = wr.get("win_prob_red"), wr.get("win_prob_blue")
        w["gold_lead_blue"] = -wr.get("gold_lead_blue", 0)
        w["kills_lead_blue"] = -wr.get("kills_lead_blue", 0)
        return w

    @staticmethod
    def _flip_state(state: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(state)
        s["blue"], s["red"] = state.get("red"), state.get("blue")
        return s

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._stop_items_polling()
        self._stop_state_polling()
        super().closeEvent(event)

    @staticmethod
    def _swap_prefixed(d: Dict[str, Any]) -> Dict[str, Any]:
        """Novo dict com todas as chaves blue_*<->red_* trocadas."""
        out: Dict[str, Any] = {}
        for k, v in d.items():
            if k.startswith("blue_"):
                out["red_" + k[len("blue_"):]] = v
            elif k.startswith("red_"):
                out["blue_" + k[len("red_"):]] = v
            else:
                out[k] = v
        return out

    def _swap_sides(self) -> None:
        """Troca Blue<->Red no resultado JA carregado e re-renderiza (sem re-buscar)."""
        data = getattr(self, "_last_data", None)
        if not data:
            return
        s = dict(data)
        s["blue"], s["red"] = data.get("red"), data.get("blue")
        if isinstance(data.get("live_draft"), dict):
            s["live_draft"] = self._swap_prefixed(data["live_draft"])
        # h2h: winrate_a e as diffs sao do ponto de vista do blue -> inverte
        h = dict(data.get("h2h") or {})
        if h:
            wr = h.get("winrate_a")
            if isinstance(wr, (int, float)):
                h["winrate_a"] = round(100 - wr, 1)
            for kk in ("kills_diff", "deaths_diff"):
                if isinstance(h.get(kk), (int, float)):
                    h[kk] = -h[kk]
            s["h2h"] = h
        # lane matchups: troca nomes e nega o delta (blue - red)
        lanes = []
        for m in data.get("lane_matchups") or []:
            m2 = dict(m)
            m2["blue"], m2["red"] = m.get("red"), m.get("blue")
            if isinstance(m2.get("delta_kills"), (int, float)):
                m2["delta_kills"] = -m2["delta_kills"]
            lanes.append(m2)
        s["lane_matchups"] = lanes
        self._last_data = s
        self._render(s)
        # inverte tambem os itens ao vivo, na hora (nao espera o proximo poll)
        self._sides_swapped = not self._sides_swapped
        if self._last_items:
            self._on_items(self._last_items)
        if self._last_state_wr:
            self._apply_state(*self._last_state_wr)

    def _on_load_live(self) -> None:
        self.btn_live.setEnabled(False)
        self.live_list.clear()
        self.live_list.addItem("Carregando...")
        w = LiveWorker(self.esports)
        w.finished_ok.connect(self._on_live_ready)
        w.failed.connect(self._on_live_failed)
        w.start()
        self.current_worker = w

    def _on_live_ready(self, live: List[Dict[str, Any]], upcoming: List[Dict[str, Any]]) -> None:
        self.btn_live.setEnabled(True)
        self.live_list.clear()
        if not live and not upcoming:
            self.live_list.addItem("Nenhuma partida encontrada.")
            return
        for ev in live:
            self._add_event_item(ev, live=True)
        for ev in upcoming:
            self._add_event_item(ev, live=False)

    def _on_live_failed(self, err: str) -> None:
        self.btn_live.setEnabled(True)
        self.live_list.clear()
        self.live_list.addItem(f"Erro: {err}")

    def _add_event_item(self, ev: Dict[str, Any], live: bool) -> None:
        match = ev.get("match") or {}
        teams = match.get("teams") or []
        if len(teams) < 2:
            return
        t1 = teams[0].get("name") or teams[0].get("code") or "?"
        t2 = teams[1].get("name") or teams[1].get("code") or "?"
        when = ev.get("startTime") or ""
        league = (ev.get("league") or {}).get("name", "?")
        tag = "[AO VIVO]" if live else "[Em breve]"
        text = f"{tag} {league}: {t1} vs {t2}  ({when})"
        item = QListWidgetItem(text)
        # event_id = match.id (nao confundir com ev.id que pode ser None)
        item.setData(Qt.ItemDataRole.UserRole, {
            "t1": t1, "t2": t2,
            "event_id": match.get("id", ""),
            "is_live": live,
        })
        self.live_list.addItem(item)

    def _on_pick_live(self, item: QListWidgetItem) -> None:
        data = item.data(Qt.ItemDataRole.UserRole) or {}
        if not data:
            return
        self.ed_blue.setText(data["t1"])
        self.ed_red.setText(data["t2"])

        event_id = data.get("event_id", "")

        # Para partidas LPL/Tencent, tenta achar o mirror Riot ANTES de cair
        # no path Tencent. A Riot expoe o draft do mapa atual mesmo quando a
        # Tencent ainda esta no "gap entre mapas".
        if event_id and str(event_id).startswith("lplqq:"):
            mirror = self.esports.find_lpl_riot_mirror(data["t1"], data["t2"])
            if mirror:
                self._set_status(f"LPL via Riot mirror ({mirror[:12]}...)")
                event_id = mirror

        if event_id:
            # Caminho LIVE: usa LivePicksWorker para extrair picks
            self._on_search_live(event_id)
        else:
            # Sem event_id (raro): cai no flow normal de busca
            self._on_search()

    def _on_search_live(self, event_id: str, force_game_id: str = "") -> None:
        blue = self.ed_blue.text().strip()
        red = self.ed_red.text().strip()
        year = int(self.cb_year.currentText())
        self.btn_search.setEnabled(False)
        self.progress_lbl.setText("Buscando partida ao vivo...")
        self._series_event_id = event_id

        w = LivePicksWorker(self.esports, self.livefeed, self.oracle,
                            event_id=event_id, blue_name=blue, red_name=red,
                            year=year, force_game_id=force_game_id)
        w.progress.connect(self._on_progress)
        w.finished_ok.connect(self._on_matchup_ready)
        w.failed.connect(self._on_series_game_failed if force_game_id else self._on_live_search_failed)
        w.start()
        self.current_worker = w

    # ---------- render ----------

    def _render(self, data: Dict[str, Any]) -> None:
        self.ed_blue.setText(data["blue"]["name"])
        self.ed_red.setText(data["red"]["name"])
        self._render_team(data["blue"], blue=True)
        self._render_team(data["red"], blue=False)
        h2h = data["h2h"]
        self.h2h_winrate.set_value(f"{h2h['winrate_a']}%", sub=f"n={h2h['games']}")
        sign_k = "+" if h2h["kills_diff"] > 0 else ""
        sign_d = "+" if h2h["deaths_diff"] > 0 else ""
        self.h2h_kdiff.set_value(f"{sign_k}{h2h['kills_diff']}")
        self.h2h_ddiff.set_value(f"{sign_d}{h2h['deaths_diff']}")

        for r in ("TOP", "JNG", "MID", "BOT", "SUP"):
            self.lane_rows[r].set_data(r, "-", "-", 0.0)
        for m in data.get("lane_matchups", []):
            row = self.lane_rows.get(m["role"])
            if row:
                row.set_data(m["role"], m["blue"], m["red"], m["delta_kills"])

        pred = predict_win_prob(data)
        self._last_pred = pred
        self.win_prob_widget.set_prediction(
            pred, data["blue"]["name"], data["red"]["name"]
        )

    def _render_team(self, team_data: Dict[str, Any], blue: bool) -> None:
        cards = self.blue_cards if blue else self.red_cards
        players = self.blue_players if blue else self.red_players
        ts = team_data["team_stats"]
        cards["kills"].set_value(
            str(ts["total_kills_avg"]),
            sub=f"WR: {ts.get('winrate', 0.0):.0f}%  n={ts['games']}",
        )
        cards["time"].set_value(str(ts["duration_avg_min"]))
        cards["over_k"].set_value(f"{ts['over_kills_pct']}%", sub=f"n={ts['games']}")
        cards["under_k"].set_value(f"{ts['under_kills_pct']}%", sub="push=0.0%")
        cards["over_t"].set_value(f"{ts['over_time_pct']}%", sub=f"n={ts['games']}")
        cards["under_t"].set_value(f"{ts['under_time_pct']}%", sub="push=0.0%")

        for i, pc in enumerate(players):
            if i >= len(team_data["players"]):
                pc.set_data("-", "-", "-", "-/-", "-/-", "-/-", "-/-")
                pc.set_patch_wr("-/-")
                continue
            p = team_data["players"][i]
            role = {"Top": "TOP", "Jungle": "JNG", "Mid": "MID",
                    "Bot": "BOT", "Support": "SUP"}.get(p.get("role", ""), p.get("role", ""))
            # Player: K/D/A geral do jogador no ano.
            pkd = _format_kda(
                p.get("kills_avg"),
                p.get("deaths_avg"),
                p.get("assists_avg"),
                p.get("games_player", 0),
            )
            # Campeao escolhido na partida atual (vem do livefeed.extract_draft).
            # Mostra o nome + K/D historico no heroi (se houver amostra).
            champ_name = p.get("champion_name", "") or "-"
            kc = p.get("kills_avg_champ", 0.0)
            dc = p.get("deaths_avg_champ", 0.0)
            ac = p.get("assists_avg_champ", 0.0)
            n_champ = p.get("games_champ", 0)
            champ_label = _format_kda(kc, dc, ac, n_champ)
            champ_wr_year = _format_record(p.get("wins_champ_year", 0), p.get("games_champ_year", 0))
            champ_wr_career = _format_record(p.get("wins_champ", 0), n_champ)
            pc.set_data(
                p.get("player", "-"),
                role,
                team_data["name"],
                pkd,
                champ_label,
                champ_wr_year,
                champ_wr_career,
                p.get("champion_icon", ""),
                champ_name,
                p.get("data_status", ""),
                p.get("status_detail", ""),
                p.get("resolve_reason", ""),
            )
            pc.set_patch_wr(*_format_patch_wr(p.get("patch_wr"), champ_name))


# WR no patch so ganha cor com amostra minima e desvio real: com 30 jogos o erro
# padrao e ~9pp, entao 53% e ruido. Cor = >=55% verde / <=45% vermelho, n>=20.
_PATCH_TONE_MIN_GAMES = 20


def _format_patch_wr(ps: Optional[Dict[str, Any]], champion: str):
    """(valor, titulo, tooltip, tom) do card WR Patch."""
    if not ps or not ps.get("patch"):
        return "-/-", "WR Patch", "", ""
    n = int(ps.get("games") or 0)

    def rec(s):
        g = int(s.get("games") or 0)
        return f"{s['wins']}V/{g}J ({s['winrate']:.1f}%)" if g else "0 jogos"

    value = f"{ps['winrate']:.0f}% ({n})" if n else "0 jogos"
    title = f"WR {ps['patch']}{'*' if ps.get('fallback') else ''}"
    lines = [f"{champion} no patch {ps['patch']}: {rec(ps)}",
             "Todos os jogos pro da base (qualquer jogador e liga)."]
    if ps.get("fallback"):
        req = ps.get("requested")
        rs = ps.get("requested_stats")
        if not req:
            why = "patch da partida desconhecido"
        elif rs:
            why = (f"patch da partida {req} ainda tem so {rs['patch_games']} jogos "
                   f"na base ({champion}: {rec(rs)})")
        else:
            why = f"patch da partida {req} ainda sem jogos na base"
        lines.insert(0, f"* {why} -> mostrando {ps['patch']}")
    prev = ps.get("prev")
    if prev:
        line = f"Patch anterior {prev['patch']}: {rec(prev)}"
        if n >= _PATCH_TONE_MIN_GAMES and prev.get("games", 0) >= _PATCH_TONE_MIN_GAMES:
            line += f" | {ps['winrate'] - prev['winrate']:+.1f}pp"
        lines.append(line)
    tone = ""
    if n >= _PATCH_TONE_MIN_GAMES:
        shown = round(ps["winrate"])  # mesmo numero exibido no card
        tone = "good" if shown >= 55 else "bad" if shown <= 45 else ""
    return value, title, "\n".join(lines), tone


def _format_kda(kills: Any, deaths: Any, assists: Any, games: Any) -> str:
    try:
        games_int = int(games or 0)
    except (TypeError, ValueError):
        games_int = 0
    if games_int <= 0:
        return "sem dados"
    try:
        return f"{float(kills):.2f}/{float(deaths):.2f}/{float(assists):.2f}"
    except (TypeError, ValueError):
        return "sem dados"


def _format_record(wins: Any, games: Any) -> str:
    try:
        wins_int = int(wins or 0)
        games_int = int(games or 0)
    except (TypeError, ValueError):
        return "sem dados"
    if games_int <= 0:
        return "sem dados"
    return f"{wins_int}/{games_int}"


