# 汎用 SWETokenizer 実装計画

## 目標

`SWEEncoder_ja`（[encode_swe.py](encode_swe.py)）をベースに、下記を満たす汎用 Tokenizer を  
`tokenizer_swe.py` 単一ファイルで実装する。

| 要件 | 内容 |
|-----|------|
| 対象言語 | 日本語 + 英語 |
| HuggingFace 互換 | `transformers.PreTrainedTokenizer` を継承 |
| 語彙ファイル | 既存 `.txt` / `emoji.json` をそのまま使用 |
| 後方互換 | `encode(text)` / `decode(tokens)` API を維持 |
| 語彙学習（オプション） | 任意コーパスから `.txt` 語彙を学習する `SWETrainer` クラス |
| 実装形態 | 単一ファイル `tokenizer_swe.py` |

---

## アーキテクチャ

### クラス設計

`tokenizer_swe.py` に 2 クラスを実装する。

```python
# ① メインTokenizer（常に使用）
class SWETokenizer(PreTrainedTokenizer):
    vocab_files_names = {
        "vocab_file": "vocab.txt",   # 語彙ファイル（内部保存名）
        "emoji_file": "emoji.json",  # 絵文字マッピング
    }

    def __init__(
        self,
        vocab_file: str,
        emoji_file: str,
        bpe_dropout_rate: float = 0.0,
        clean: bool = False,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|nottoken|>",
        sep_token="<|separator|>",
        unk_token=None,   # バイトフォールバックで <unk> 不要
        **kwargs,
    ):
        ...

# ② 語彙学習クラス（オプション）
class SWETrainer:
    def __init__(
        self,
        vocab_size: int = 32000,
        emoji_file: str = "emoji.json",
        min_frequency: int = 2,
    ):
        ...

    def train(self, files: List[str], output_vocab: str) -> None:
        """テキストファイルリストから BPE 語彙を学習し .txt に出力する。"""
        ...

    def train_from_iterator(self, texts: Iterator[str], output_vocab: str) -> None:
        """テキストイテレータから語彙を学習する（インメモリ版）。"""
        ...
```

---

## V2 (`encode_swe.py`) からの主要変更点

### 1. 検索ウィンドウの汎用化（英語対応）

`encode_swe.py` の `encode()` では通常文字の探索が 3 文字に固定されており、  
英語の複数文字トークン（例: `"the"`, `"ing"` など）が語彙にあっても一致しない。

```python
# encode_swe.py（変更しない）
end = min(len(text), pos+self.maxlen+1) if text[pos]=='<' else pos+3
```

```python
# tokenizer_swe.py（全文字 maxlen まで探索）
end = min(len(text), pos + self.maxlen + 1)
```

この 1 行の変更で英語サブワードのマッチが有効になる。

### 2. `_tokenize` はトークン文字列を返す

HuggingFace の規約に従い、`_tokenize` は ID ではなく文字列リストを返す。  
ID 変換は `_convert_token_to_id` が担当する。

```
_tokenize("今日はHello") → ["今日", "は", "H", "e", "l", "l", "o"]
                              ↓ _convert_token_to_id
                           [669, 26639, ...]
```

### 3. `bpe_dropout_rate` / `clean` をインスタンス変数化

HuggingFace の `_tokenize(text: str)` はシグネチャが固定のため、  
これらは `__init__` 引数として受け取りインスタンス変数に保持する。

後方互換のため `encode(text, clean=..., bpe_dropout_rate=...)` メソッドも提供。

---

## HuggingFace 必須メソッド一覧

| メソッド | 実装内容 |
|---------|---------|
| `vocab_size` (property) | `len(self.bpe)` を返す |
| `get_vocab()` | `{token: id, ...}` の辞書を返す |
| `_tokenize(text)` | BPE マッチング → トークン文字列リスト |
| `_convert_token_to_id(token)` | `self.swe.get(token)` |
| `_convert_id_to_token(index)` | `self.bpe[index][0]` |
| `convert_tokens_to_string(tokens)` | バイト列復元を含むデコードロジック |
| `save_vocabulary(save_dir, prefix)` | `vocab.txt` / `emoji.json` をコピー保存 |

---

## 後方互換 API

```python
# SWEEncoder_ja と同じ呼び出しで動作する
tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json")

ids: List[int] = tokenizer.encode("今日はHello World!")
text: str      = tokenizer.decode(ids)
cleaned: str   = tokenizer.clean_text(content)
size: int      = len(tokenizer)
```

---

## 使用イメージ

### HuggingFace スタイル

```python
from tokenizer_swe import SWETokenizer

tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json")

# __call__ で input_ids / attention_mask を返す
out = tokenizer("今日はHello World!")
# {"input_ids": [...], "attention_mask": [...]}

# 保存・読み込み（from_pretrained 経由）
tokenizer.save_pretrained("./my-tokenizer")
tokenizer2 = SWETokenizer.from_pretrained("./my-tokenizer")
```

### BPE Dropout（学習時のデータ拡張）

```python
# インスタンス作成時に設定
tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json", bpe_dropout_rate=0.1)

# または後方互換 API で指定
ids = tokenizer.encode("テキスト", bpe_dropout_rate=0.1)
```

---

## 実装ステップ

### Phase 1: クラス骨格と初期化
- [ ] `PreTrainedTokenizer` 継承・`vocab_files_names` 定義
- [ ] `__init__`: 語彙・絵文字読み込み、`swe` 辞書・正規表現構築
- [ ] `vocab_size` property / `get_vocab()` 実装

### Phase 2: コアメソッド
- [ ] `_tokenize(text)` — BPE マッチング（ウィンドウ汎用化済み）、文字列返却
- [ ] `_convert_token_to_id` / `_convert_id_to_token`
- [ ] `convert_tokens_to_string(tokens)` — バイト列復元を含むデコード

### Phase 3: 保存・後方互換
- [ ] `save_vocabulary(save_directory, filename_prefix)` 実装
- [ ] `encode(text, clean=False, bpe_dropout_rate=0.0)` — 後方互換
- [ ] `decode(tokens, breakline='\n')` — 後方互換
- [ ] `clean_text(content)` — 移植

### Phase 4: 検証
- [ ] `SWEEncoder_ja` との出力比較（`bpe_dropout_rate=0.0` 時に一致するか）
- [ ] `transformers` 統合テスト（`__call__` / `save_pretrained` / `from_pretrained`）
- [ ] 英語テキストのトークン化確認

### Phase 5: 語彙学習（`SWETrainer`）— オプション
- [ ] BPE 学習アルゴリズム実装（文字ペア頻度集計 → 逐次マージ）
- [ ] 固定語彙ブロックの処理（特殊トークン・バイトトークンを先に確保）
- [ ] `train(files, output_vocab)` — ファイルリストから語彙を学習
- [ ] `train_from_iterator(texts, output_vocab)` — イテレータ版
- [ ] 出力フォーマット検証（既存 `.txt` と互換性があるか）

---

## `SWETrainer` の学習アルゴリズム

### 語彙構成（学習後の `.txt` フォーマット）

```
[学習済み BPE トークン]  ← コーパスから学習（vocab_size の大部分）
<BR>
<SP>
<TAB>
<URL> / <EMAIL> / <TEL> / <DATE> / <PRICE> / <BLOCK> / <KIGOU> / <U2000U2BFF>
<|emoji1|> 〜 <|emoji12|>
<|byte0|> 〜 <|byte255|>
<|startoftext|> / <|separator|> / <|nottoken|> / <|endoftext|>
```

固定トークン数: 約 280 個。`vocab_size - 280` 個を BPE で埋める。

### 学習ステップ

1. コーパスをスキャンし、初期単位（UTF-8 文字）の頻度を集計
2. 隣接する文字ペアの共起頻度を計算
3. 最頻出ペアを語彙に追加しコーパスを更新
4. 目標語彙数に達するまで 2〜3 を繰り返す
5. 固定トークンを末尾に追加して `.txt` に書き出す

### 使用イメージ

```python
from tokenizer_swe import SWETrainer, SWETokenizer

# 語彙学習
trainer = SWETrainer(vocab_size=32000, emoji_file="emoji.json")
trainer.train(["corpus_ja.txt", "corpus_en.txt"], output_vocab="my-vocab.txt")

# 学習済み語彙でトークナイザーを初期化
tokenizer = SWETokenizer("my-vocab.txt", "emoji.json")
```

---

## 未確認事項

- **英語トークン効率**: 語彙は日本語コーパスから学習済みのため、英語単語は  
  文字単位 or バイトフォールバックになる可能性が高い（許容範囲かを実装後に確認）
- **`tokenizer_config.json` の `auto_map`**: `AutoTokenizer` で読み込むには  
  `auto_map` の設定が必要。初期実装では不要とし、必要であれば追加する
