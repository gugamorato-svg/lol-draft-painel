# LoL Draft Painel

[![tests](https://github.com/gugamorato-svg/lol-draft-painel/actions/workflows/tests.yml/badge.svg)](https://github.com/gugamorato-svg/lol-draft-painel/actions/workflows/tests.yml)

Painel desktop (Python + PyQt6) para acompanhar partidas **profissionais** de League of Legends ao vivo: puxa a partida que está rolando, mostra os jogadores e campeões reais do mapa atual e estima a chance de vitória de Blue e Red a partir do draft, dos times e dos jogadores.

> *English summary:* a desktop panel that follows pro League of Legends matches live, pulls the real players and champions of the current map from the official live stats feed, and estimates Blue vs Red win probability from the draft, teams and players. Python 3.11+, PyQt6, scikit-learn. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## O que ele faz

- Lista as partidas ao vivo (LoL Esports API) e, com dois cliques, abre o mapa atual.
- Corrige o lado Blue/Red real e puxa jogadores e campeões do **live stats feed** oficial.
- Mostra cards por jogador com histórico no campeão; o tooltip explica quando falta dado (nome não resolvido, sem amostra, erro de fallback).
- Calcula a probabilidade de vitória com um modelo de regressão logística treinado em dados públicos do [Oracle's Elixir](https://oracleselixir.com/).
- Mostra também uma barra de **draft puro**, que usa só os campeões, sem times nem jogadores.
- Cobre LPL pelo site da Tencent quando a partida não aparece na API oficial.

## Como rodar

Requer Python 3.11 ou mais novo.

```bash
pip install -r requirements.txt
```

**1. Baixe os dados de partidas** do Oracle's Elixir (um CSV por ano, publicado em pasta pública do Google Drive):

```bash
python scripts/update_oracle.py --years 2025 2026 --no-auth
```

Os arquivos vão para `data/oracle_elixir/` e não entram no git.

**2. Treine os modelos** com esses dados (gera os pesos em `data/`):

```bash
python scripts/refresh_and_retrain.py
```

**3. Abra o painel:**

```bash
python main.py
```

Clique em **Ver partidas ao vivo** e dê dois cliques numa partida.

> Este repositório traz o código e o método, não pesos treinados nem dados. Cada pessoa gera os seus a partir das fontes públicas acima.

## Testes

```bash
python -m pytest tests
```

Os testes rodam offline e não precisam dos CSVs.

## Estrutura

| Pasta | O que tem |
|---|---|
| `main.py` | Ponto de entrada e trava de instância única |
| `api/` | Clientes das fontes: LoL Esports, live stats feed, Leaguepedia, LPL/Tencent, ícones do Data Dragon |
| `data_pipeline/` | Leitura do Oracle's Elixir, contexto temporal (Elo, forma, conforto) e linha do tempo de treino sem vazamento |
| `models/` | Modelo treinado, barra de draft puro e wrapper de predição |
| `ui/` | Janela principal, cards e threads de trabalho |
| `scripts/` | Download de dados, treino e auditoria dos modelos |
| `tests/` | Testes offline |

## Modelo

As features são diferenças Blue − Red. A probabilidade é:

```text
prob_blue = sigmoid(intercepto + soma(feature * coeficiente))
prob_blue = clamp(prob_blue, 20%, 90%)
```

O treino respeita o tempo: cada partida só enxerga o que aconteceu antes dela (`data_pipeline/training_timeline.py`), para o modelo não "ver o futuro".

## Contribuindo

Contribuições são bem-vindas, de correção de texto a features novas. Veja o [CONTRIBUTING.md](CONTRIBUTING.md) e as issues marcadas como **good first issue**.

## Fontes de dados e créditos

- Dados de partidas: [Oracle's Elixir](https://oracleselixir.com/), de Tim Sevenhuysen.
- Partidas e live feed: LoL Esports (API pública usada pelo site oficial).
- Complemento de carreira: [Leaguepedia](https://lol.fandom.com/).
- Ícones: Data Dragon, da Riot Games.

LoL Draft Painel isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties. Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

## Licença

[MIT](LICENSE)
