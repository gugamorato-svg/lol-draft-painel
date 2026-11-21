"""Tema escuro do painel (QSS) - inspirado no LoL Painel da imagem."""

QSS = """
QWidget {
    background-color: #0e1014;
    color: #e6e6e6;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}
QMainWindow, QDialog { background-color: #0a0c10; }
QLabel { background-color: transparent; }

#card, #playerCard {
    background-color: #161a22;
    border: 1px solid #232836;
    border-radius: 8px;
}
#cardHeader { color: #9aa3b2; font-size: 10px; }
#cardValue  { color: #ffffff; font-size: 17px; font-weight: 700; }
#cardSub    { color: #6e7689; font-size: 9px; }
#miniStat {
    background-color: #10141c;
    border: 1px solid #253047;
    border-radius: 7px;
}
#miniStatTitle { color: #9fc7ff; font-size: 9px; }
#miniStatValue { color: #ffffff; font-size: 12px; font-weight: 800; }
#champIcon {
    background-color: #1b2638;
    border: 1px solid #33435f;
    border-radius: 6px;
    color: #9fc7ff;
    font-weight: 800;
}
#emptyIcon {
    background-color: #101722;
    border: 1px dashed #2f5788;
    border-radius: 10px;
}

#sectionTitle { color: #4aa9ff; font-size: 13px; font-weight: 700; }
#sectionTitleRed { color: #ff6b6b; font-size: 13px; font-weight: 700; }
#sectionTitleNeutral { color: #c9d1e0; font-size: 13px; font-weight: 700; }

#deltaPositive { color: #5eff9a; font-weight: 700; }
#deltaNegative { color: #ff6b6b; font-weight: 700; }

QLineEdit {
    background-color: #11141b;
    border: 1px solid #2a3040;
    border-radius: 6px;
    padding: 6px 10px;
    color: #ffffff;
}
QLineEdit:focus { border: 1px solid #4aa9ff; }

QPushButton {
    background-color: #1d2330;
    border: 1px solid #2a3040;
    border-radius: 6px;
    padding: 6px 14px;
    color: #ffffff;
}
QPushButton:hover { background-color: #263041; border-color: #4aa9ff; }
QPushButton:pressed { background-color: #1a2030; }
QPushButton:disabled { color: #555a66; }

#primary {
    background-color: #2563eb;
    border-color: #2563eb;
}
#primary:hover { background-color: #1d4ed8; }

QScrollArea { border: none; }
QListWidget {
    background-color: #0f131b;
    border: 1px solid #232836;
    border-radius: 6px;
    padding: 2px;
}
QListWidget::item { padding: 4px 6px; }
QListWidget::item:selected { background-color: #1f355c; color: #ffffff; }
QStatusBar { background-color: #0a0c10; color: #6e7689; }

/* Win Probability Widget */
#winProbTitle {
    color: #c9d1e0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
#winPctBlue {
    color: #4aa9ff;
    font-size: 26px;
    font-weight: 800;
}
#winPctRed {
    color: #ff6b6b;
    font-size: 26px;
    font-weight: 800;
}
#probBarOuter {
    background-color: #0e1014;
    border-radius: 4px;
    min-height: 16px;
}
#probBarBlue {
    background-color: #2563eb;
    border-radius: 4px 0px 0px 4px;
    min-height: 16px;
}
#probBarRed {
    background-color: #9b2335;
    border-radius: 0px 4px 4px 0px;
    min-height: 16px;
}
#factorBlue    { color: #4aa9ff; font-size: 10px; }
#factorRed     { color: #ff6b6b; font-size: 10px; }
#factorNeutral { color: #6e7689; font-size: 10px; }
"""
