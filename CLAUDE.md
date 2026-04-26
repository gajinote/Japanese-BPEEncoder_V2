---
# Japanese-BPEEncoder V2 — 汎用Tokenizer開発

## プロジェクト概要
`SWEEncoder_ja` をベースに、日本語・英語対応の HuggingFace 互換 Tokenizer を `tokenizer_swe.py` に実装する。

## 実装方針
- **単一ファイル**: `tokenizer_swe.py` に実装する（`encode_swe.py` は変更しない）
- **後方互換性**: `SWEEncoder_ja` の API（`encode` / `decode`）と同等の呼び出しを維持
- **HuggingFace 互換**: `transformers.PreTrainedTokenizer` を継承する
- **語彙ファイル**: 既存の `.txt` / `emoji.json` をそのまま使用する
- **対象言語**: 日本語 + 英語

## ファイル構成
| ファイル | 役割 |
|---------|------|
| `encode_swe.py` | 既存日本語専用エンコーダー（変更禁止） |
| `tokenizer_swe.py` | 新規：`SWETokenizer`（HuggingFace 互換）＋ `SWETrainer`（語彙学習）を実装 |
| `ja-swe32kfix.txt` | 推奨語彙ファイル（32K トークン） |
| `emoji.json` | 絵文字マッピング |
| `PLAN.md` | 実装計画の詳細 |

## 語彙ファイルの特殊トークン（ja-swe32kfix.txt）
| トークン | 用途 |
|---------|------|
| `<SP>` | スペース |
| `<BR>` | 改行 |
| `<TAB>` | タブ |
| `<URL>` / `<EMAIL>` / `<TEL>` / `<DATE>` / `<PRICE>` | clean_text 変換先 |
| `<BLOCK>` / `<KIGOU>` / `<U2000U2BFF>` | 記号・罫線 |
| `<\|emoji1\|>` 〜 `<\|emoji12\|>` | 絵文字（12カテゴリ） |
| `<\|byte0\|>` 〜 `<\|byte255\|>` | バイトフォールバック（`<unk>` 不要） |
| `<\|startoftext\|>` | テキスト開始 |
| `<\|separator\|>` | セパレータ |
| `<\|nottoken\|>` | パディング用 |
| `<\|endoftext\|>` | テキスト終了（末尾トークン） |

## コーディング規則
- インデント: 2スペース（グローバル CLAUDE.md に従う）
- コメント: 英語（既存コードに合わせる）
- 型ヒント: 使用する（HuggingFace 互換のため）
- `encode_swe.py` は絶対に変更しない
- 不要な機能追加・抽象化は行わない
