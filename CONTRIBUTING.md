# Como contribuir

Obrigado pelo interesse! Qualquer ajuda conta: corrigir um texto, melhorar uma mensagem de erro, cobrir uma liga nova ou propor uma feature para o modelo.

*English:* issues and pull requests in English are welcome too.

## Antes de começar

1. Procure uma issue aberta. As marcadas como **good first issue** são pequenas e boas para começar.
2. Comente na issue que vai pegá-la, para ninguém fazer o mesmo trabalho ao mesmo tempo.
3. Tem uma ideia nova? Abra uma issue antes do PR para conversarmos sobre o caminho.

## Ambiente

```bash
git clone https://github.com/<seu-usuario>/lol-draft-painel.git
cd lol-draft-painel
pip install -r requirements.txt
python -m pytest tests
```

Os testes rodam offline. Para abrir o painel de verdade, siga os passos de dados e treino do [README](README.md#como-rodar).

## Pull requests

- Crie um branch a partir da `main` (ex.: `fix/mensagem-sem-dados`).
- Mantenha o PR focado em uma mudança. Refatorações à parte vão em outro PR.
- Toda correção de bug vem com um teste que falha sem a correção, quando der para testar sem rede.
- Rode `python -m pytest tests` antes de abrir o PR e diga na descrição o que você testou.
- Mudou algo no modelo (features, pesos, janela de treino)? Mostre a comparação antes/depois com `scripts/compare_models.py` ou `scripts/audit_model.py`. Mudança no modelo sem medida não entra.

## Regras do modelo

- **Nada de vazamento:** uma partida só pode usar informação de antes dela. A linha do tempo de treino (`data_pipeline/training_timeline.py`) existe para isso.
- **Dados não vão para o git:** CSVs do Oracle's Elixir, pesos treinados e caches ficam fora (ver `.gitignore`).

## Código

- Python 3.11+, com type hints onde ajudarem.
- Comentários explicam o porquê, não o quê.
- Mensagens de commit curtas e no imperativo, em português ou inglês.

## Dúvidas

Abra uma issue com a etiqueta **question**.
