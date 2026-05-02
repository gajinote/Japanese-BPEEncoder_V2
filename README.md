# Japanese-BPEEncoder V2

日本語・英語テキスト（UTF-8）のSubWord Encoder / Tokenizerです。  
文字ペア単位で分かち書きし、整数列にエンコードします。

- **`encode_swe.py`** — オリジナルの日本語専用エンコーダー (`SWEEncoder_ja`)
- **`tokenizer_swe.py`** — HuggingFace `transformers` 互換トークナイザー (`SWETokenizer`) ＋ 語彙学習クラス (`SWETrainer`)



## ファイル構成

| ファイル | 内容 |
|---------|------|
| `encode_swe.py` | 日本語専用エンコーダー `SWEEncoder_ja` |
| `tokenizer_swe.py` | HuggingFace 互換 `SWETokenizer` / `SWETrainer` |
| `ja-swe32kfix.txt` | 推奨語彙（32K トークン・修正版） |
| `ja-swe32k.txt` | 語彙 32K |
| `ja-swe24k.txt` | 語彙 24K |
| `ja-swe16k.txt` | 語彙 16K |
| `ja-swe8k.txt` | 語彙 8K |
| `ja-sce.txt` | SingleCharacterEncode 用語彙（BPE なし・5.6K） |
| `emoji.json` | 絵文字マッピング |



## SWETokenizer（HuggingFace 互換）

`transformers.PreTrainedTokenizer` を継承した汎用トークナイザーです。  
日本語・英語に対応し、`<unk>` なしのバイトレベルフォールバックを備えます。

### 依存ライブラリ

```
pip install transformers numpy
```

### 基本的な使い方

```python
from tokenizer_swe import SWETokenizer

tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json")

# HuggingFace スタイル（input_ids / attention_mask を返す）
out = tokenizer("今日はHello World!")
print(out["input_ids"])

# バッチ処理・パディング
batch = tokenizer(["今日はいい天気", "Hello World"], padding=True)

# トークン文字列の確認
print(tokenizer.tokenize("今日はHello World"))
# ['今日', 'は', 'H', 'e', 'l', 'l', 'o', '<SP>', 'W', 'o', 'r', 'l', 'd']

# デコード
ids = tokenizer.encode("今日は日曜焼き肉定食をたべる")
print(tokenizer.decode(ids))
# 今日は日曜焼き肉定食をたべる
```

### 保存・読み込み

```python
# 保存
tokenizer.save_pretrained("./my-tokenizer")

# 読み込み
tokenizer = SWETokenizer.from_pretrained("./my-tokenizer")
```

### テキスト正規化（clean モード）

URL・メールアドレス・日付・価格などを特殊タグに変換します。

```python
# インスタンス作成時に有効化
tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json", clean=True)

# または encode 呼び出し時に指定
ids = tokenizer.encode("https://example.com 今日", clean=True)
print(tokenizer.decode(ids))
# <URL> 今日
```

### BPE Dropout（サブワード正則化）

学習時のデータ拡張として、分割方法をランダムに変えます。

```python
tokenizer = SWETokenizer("ja-swe32kfix.txt", "emoji.json", bpe_dropout_rate=0.1)

# または encode 呼び出し時に指定
ids = tokenizer.encode("今日は日曜焼き肉定食をたべる", bpe_dropout_rate=0.5)
print([tokenizer.decode([i]) for i in ids])
# ['今日', 'は', '日曜', '焼', 'き', '肉', '定食', 'を', 'た', 'べる']  ← 実行ごとに変わる
```

### 特殊トークン

| トークン | 用途 |
|---------|------|
| `<SP>` | スペース |
| `<BR>` | 改行 |
| `<TAB>` | タブ |
| `<URL>` / `<EMAIL>` / `<TEL>` / `<DATE>` / `<PRICE>` | clean モードの変換先 |
| `<BLOCK>` / `<KIGOU>` / `<U2000U2BFF>` | 記号・罫線 |
| `<\|emoji1\|>` 〜 `<\|emoji12\|>` | 絵文字（12 カテゴリ） |
| `<\|byte0\|>` 〜 `<\|byte255\|>` | バイトフォールバック（`<unk>` 不要） |
| `<\|startoftext\|>` | テキスト開始 |
| `<\|separator\|>` | セパレータ |
| `<\|nottoken\|>` | パディング |
| `<\|endoftext\|>` | テキスト終了 |



## SWETrainer（語彙学習）

任意のコーパスから BPE 語彙を学習し、`SWETokenizer` で読み込める `.txt` ファイルを生成します。

```python
from tokenizer_swe import SWETrainer, SWETokenizer

# 学習
trainer = SWETrainer(vocab_size=32000, min_frequency=2)
trainer.train(["corpus_ja.txt", "corpus_en.txt"], output_vocab="my-vocab.txt")

# 学習済み語彙でトークナイザーを初期化
tokenizer = SWETokenizer("my-vocab.txt", "emoji.json")
```

テキストイテレータからも学習できます。

```python
def text_iter():
    for line in open("corpus.txt"):
        yield line

trainer.train_from_iterator(text_iter(), output_vocab="my-vocab.txt")
```



## SWEEncoder_ja（オリジナル）

日本語専用のシンプルなエンコーダーです。依存ライブラリは `numpy` のみ。

```python
from encode_swe import SWEEncoder_ja
import json

with open('ja-swe32kfix.txt') as f:
    bpe = f.read().split('\n')

with open('emoji.json') as f:
    emoji = json.loads(f.read())

enc = SWEEncoder_ja(bpe, emoji)

p = enc.encode('今日は日曜焼き肉定食をたべる')
print(p)
# [669, 26639, 4282, 28351, 26620, 27448, 12781, 26683, 18161, 26651]

print(enc.decode(p))
# 今日は日曜焼き肉定食をたべる

print([enc.decode([i]) for i in p])
# ['今日', 'は', '日曜', '焼', 'き', '肉', '定食', 'を', 'たべ', 'る']
```

### 語彙の特徴

- 漢字・ひらがな・カタカナ毎に BPE を構築（文字種をまたぐ BPE なし）
- 異字体対応：「慶応」「𢙎応」「慶應」がすべて同じ ID にエンコード
- 旧字体（ゐ・ゑ）、異数字（①・⒉）、囲み文字（㊑・㋾）にも対応
- `<unk>` 不要：256 種のバイトタグで未知文字をすべてカバー

```
慶応𢙎応慶應　→　[17764, 17764, 17764]
```

### コマンドラインツール

テキストファイルをマルチプロセスでトークン化し、numpy 圧縮形式（.npz）で保存します。

```bash
python encode_swe.py \
    --src_dir ./texts \
    --dst_file output \
    --vocabulary ja-swe32kfix.txt \
    --num_process 8 \
    --clean_text
```

### サブワード正則化

```python
p = enc.encode('今日は日曜焼き肉定食をたべる', bpe_dropout_rate=0.5)
print([enc.decode([i]) for i in p])
# ['今日', 'は', '日曜', '焼', 'き', '肉', '定食', 'を', 'た', 'べる']
```



## ライセンス

MIT License
