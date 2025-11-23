# gramregex

OpenAI Responses API を使って CFG (context-free grammar) で出力を制約する CLI ツールです。`gramregex` コマンドで実行できます。grammar は tool call として渡し、Responses API の `text.verbosity` や `reasoning.effort` パラメータも指定できます。

## セットアップ

```bash
uv sync
```

必要な環境変数は `.env.example` を参考に設定してください。

- `OPENAI_API_KEY`: OpenAI もしくは OpenAI 互換エンドポイントの API キー (必須)
- `OPENAI_BASE_URL`: 互換エンドポイントを使う場合のベース URL (省略可)
- `OPENAI_MODEL`: 使用するモデル名 (デフォルト: `gpt-4.1-mini`)
- `PROVIDER`: LLM プロバイダ。現状 `openai` のみ対応。

## 使い方

CFG を直接指定する場合:

```bash
uv run gramregex "your prompt" --grammar "root ::= 'ok'"
```

CFG ファイルを指定する場合:

```bash
uv run gramregex --grammar-file path/to/grammar.cfg "your prompt"
```

主なオプション:

- `--grammar-syntax`: grammar ツールの `syntax` (`lark` もしくは `regex`。デフォルト: `lark`)
- `--verbosity`: Responses API の詳細度 (`low`/`medium`/`high` のいずれか)
- `--reasoning-effort`: 推論の強度 (`minimal`/`medium`/`high` のいずれか)
- `--model`: モデル名を一時的に上書き

## 開発

品質チェックは Nox で実行します。

```bash
uv run nox -s lint
uv run nox -s typing
uv run nox -s test
```

LLM 呼び出しを伴うテストはすべてモック化されているため、ネットワークなしで実行できます。
