"""
tokenizer_swe.py — General-purpose SWE Tokenizer for Japanese + English.

Classes:
  SWETokenizer : transformers.PreTrainedTokenizer subclass (HuggingFace compatible).
  SWETrainer   : Optional BPE vocabulary trainer from a text corpus.
"""

import json
import os
import re
from collections import Counter
from typing import ClassVar, Dict, Iterator, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizer


class SWETokenizer(PreTrainedTokenizer):
  """
  HuggingFace-compatible tokenizer based on Japanese-BPEEncoder V2.

  Supports Japanese + English with byte-level fallback (no <unk>).
  Drop-in replacement for SWEEncoder_ja with additional HuggingFace interface.

  Args:
    vocab_file: Path to vocabulary .txt file (one token per line).
    emoji_file: Path to emoji.json mapping file.
    bpe_dropout_rate: BPE dropout rate for subword regularization (default 0.0).
    clean: Apply URL/email/date/price normalization in _tokenize (default False).
  """

  vocab_files_names = {
    "vocab_file": "vocab.txt",
    "emoji_file": "emoji.json",
  }

  def __init__(
    self,
    vocab_file: str,
    emoji_file: str,
    bpe_dropout_rate: float = 0.0,
    clean: bool = False,
    bos_token: str = "<|startoftext|>",
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<|nottoken|>",
    sep_token: str = "<|separator|>",
    unk_token: Optional[str] = None,
    **kwargs,
  ):
    with open(vocab_file, encoding="utf-8") as f:
      raw = f.read().split("\n")
    # Remove trailing empty lines
    while raw and raw[-1] == "":
      raw.pop()

    # bpe[i] = list of token strings mapping to ID i
    # (comma-separated variants encode same ID, e.g. "慶応,慶應")
    self.bpe: List[List[str]] = [
      [b] if (b == "," or "," not in b) else b.split(",")
      for b in raw
    ]

    # swe: token string → ID (primary lookup table)
    self.swe: Dict[str, int] = {}
    for idx, tokens in enumerate(self.bpe):
      for t in tokens:
        self.swe[t] = idx

    with open(emoji_file, encoding="utf-8") as f:
      self.emoji: Dict = json.loads(f.read())

    self.maxlen: int = max(len(w) for w in self.swe)
    self.bpe_dropout_rate = bpe_dropout_rate
    self.clean = clean

    # Regex patterns for clean_text
    self._pat_url = re.compile(
      r"(https?|ftp)(:\/\/[-_\.!~*\'()a-zA-Z0-9;\/?:\@&=\+$,%#]+)"
    )
    self._pat_email = re.compile(
      r"[A-Za-z0-9\._+]*@[\-_0-9A-Za-z]+(\.[A-Za-z]+)*"
    )
    self._pat_tel = re.compile(
      r"[\(]{0,1}[0-9]{2,4}[\)\-\(]{0,1}[0-9]{2,4}[\)\-]{0,1}[0-9]{3,4}"
    )
    self._pat_date1 = re.compile(
      r"([12]\d{3}[/\-年])*(0?[1-9]|1[0-2])[/\-月]"
      r"((0?[1-9]|[12][0-9]|3[01])日?)*"
      r"(\d{1,2}|:|\d{1,2}時|\d{1,2}分"
      r"|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)"
      r"|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
    )
    self._pat_date2 = re.compile(
      r"(明治|大正|昭和|平成|令和|㍾|㍽|㍼|㍻|㋿)"
      r"\d{1,2}年(0?[1-9]|1[0-2])月(0?[1-9]|[12][0-9]|3[01])日"
      r"(\d{1,2}|:|\d{1,2}時|\d{1,2}分"
      r"|\(日\)|\(月\)|\(火\)|\(水\)|\(木\)|\(金\)|\(土\)"
      r"|㈰|㈪|㈫|㈬|㈭|㈮|㈯)*"
    )
    self._pat_price = re.compile(
      r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*億)*"
      r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*万)*"
      r"((0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*千)*"
      r"(0|[1-9]\d*|[1-9]\d{0,2}(,\d{3})+)*"
      r"(千円|万円|千万円|円|千ドル|万ドル|千万ドル|ドル"
      r"|千ユーロ|万ユーロ|千万ユーロ|ユーロ)+"
      r"(\(税込\)|\(税抜\)|\+tax)*"
    )
    keisen = (
      "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫"
      "┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘"
      "╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
    )
    blocks = "▀▁▂▃▄▅▆▇█▉▊▋▌▍▎▏▐░▒▓▔▕▖▗▘▙▚▛▜▝▞▟"
    self._trans_block = str.maketrans({k: "<BLOCK>" for k in keisen + blocks})

    super().__init__(
      bos_token=bos_token,
      eos_token=eos_token,
      pad_token=pad_token,
      sep_token=sep_token,
      unk_token=unk_token,
      **kwargs,
    )

  # ------------------------------------------------------------------
  # HuggingFace PreTrainedTokenizer required interface
  # ------------------------------------------------------------------

  @property
  def vocab_size(self) -> int:
    return len(self.bpe)

  def get_vocab(self) -> Dict[str, int]:
    vocab = dict(self.swe)
    vocab.update(self.added_tokens_encoder)
    return vocab

  def _tokenize(self, text: str) -> List[str]:
    return self._tokenize_inner(text, self.clean, self.bpe_dropout_rate)

  def _convert_token_to_id(self, token: str) -> int:
    return self.swe.get(token, self.swe.get("<|byte0|>", 0))

  def _convert_id_to_token(self, index: int) -> str:
    return self.bpe[index][0]

  def convert_tokens_to_string(self, tokens: List[str]) -> str:
    words: List[str] = []
    byte_buf: List[int] = []
    for token in tokens:
      if token.startswith("<|byte") and token.endswith("|>"):
        byte_buf.append(int(token[6:-2]))
        continue
      if byte_buf:
        words.append(bytearray(byte_buf).decode("utf-8", errors="replace"))
        byte_buf = []
      if token.startswith("<|emoji") and token.endswith("|>"):
        words.append(self.emoji["emoji_inv"].get(token, ""))
      elif token == "<SP>":
        words.append(" ")
      elif token == "<BR>":
        words.append("\n")
      elif token == "<TAB>":
        words.append("\t")
      elif token == "<BLOCK>":
        words.append("▀")
      elif token == "<KIGOU>":
        words.append("ǀ")
      elif token == "<U2000U2BFF>":
        words.append("‖")
      else:
        words.append(token)
    if byte_buf:
      words.append(bytearray(byte_buf).decode("utf-8", errors="replace"))
    return "".join(words)

  def save_vocabulary(
    self,
    save_directory: str,
    filename_prefix: Optional[str] = None,
  ) -> Tuple[str, str]:
    if not os.path.isdir(save_directory):
      raise ValueError(f"Not a directory: {save_directory}")
    prefix = (filename_prefix + "-") if filename_prefix else ""
    vocab_path = os.path.join(save_directory, prefix + "vocab.txt")
    emoji_path = os.path.join(save_directory, prefix + "emoji.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
      for tokens in self.bpe:
        f.write(",".join(tokens) + "\n")
    with open(emoji_path, "w", encoding="utf-8") as f:
      json.dump(self.emoji, f, ensure_ascii=False)
    return vocab_path, emoji_path

  # ------------------------------------------------------------------
  # Backward-compatible API (SWEEncoder_ja compatible)
  # ------------------------------------------------------------------

  def __len__(self) -> int:
    return self.vocab_size

  def encode(  # type: ignore[override]
    self,
    text: str,
    text_pair: Optional[str] = None,
    clean: bool = False,
    bpe_dropout_rate: float = 0.0,
    **kwargs,
  ) -> List[int]:
    """
    Tokenize text and return token IDs.

    When clean or bpe_dropout_rate are specified, behaves like SWEEncoder_ja.encode().
    Otherwise delegates to PreTrainedTokenizer.encode() for full HuggingFace behaviour.
    """
    if clean or bpe_dropout_rate > 0.0:
      effective_clean = self.clean or clean
      effective_dropout = bpe_dropout_rate if bpe_dropout_rate > 0.0 else self.bpe_dropout_rate
      tokens = self._tokenize_inner(text, effective_clean, effective_dropout)
      return [self._convert_token_to_id(t) for t in tokens]
    return super().encode(text, text_pair=text_pair, **kwargs)  # type: ignore[return-value]

  def decode(  # type: ignore[override]
    self,
    token_ids,
    skip_special_tokens: bool = False,
    breakline: str = "\n",
    **kwargs,
  ) -> str:
    """
    Decode token IDs to text.

    breakline: replacement string for <BR> tokens (default: newline).
    """
    text = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    if breakline != "\n":
      text = text.replace("\n", breakline)
    return text

  def clean_text(self, content: str) -> str:
    """Apply URL / email / date / price / block normalization."""
    content = self._pat_url.sub("<URL>", content)
    content = self._pat_email.sub("<EMAIL>", content)
    content = self._pat_tel.sub("<TEL>", content)
    content = self._pat_date1.sub("<DATE>", content)
    content = self._pat_date2.sub("<DATE>", content)
    content = self._pat_price.sub("<PRICE>", content)
    content = content.translate(self._trans_block)
    while "<BLOCK><BLOCK>" in content:
      content = content.replace("<BLOCK><BLOCK>", "<BLOCK>")
    return content

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _tokenize_inner(
    self, text: str, clean: bool, bpe_dropout_rate: float
  ) -> List[str]:
    """Core BPE tokenization. Returns list of token strings."""
    text = text.replace(" ", "<SP>")
    text = text.replace("　", "<SP>")  # full-width space
    text = text.replace("\r\n", "<BR>")
    text = text.replace("\n", "<BR>")
    text = text.replace("\r", "<BR>")
    text = text.replace("\t", "<TAB>")
    text = text.replace("—", "ー")  # em dash
    text = text.replace("−", "ー")  # minus sign
    for k, v in self.emoji["emoji"].items():
      if k in text:
        text = text.replace(k, v)
    if clean:
      text = self.clean_text(text)

    result: List[str] = []
    pos = 0
    swe = self.swe
    maxlen = self.maxlen

    while pos < len(text):
      end = min(len(text), pos + maxlen + 1)
      kouho: List[Tuple[int, int]] = []  # (token_id, end_pos)
      for e in range(end, pos, -1):
        wd = text[pos:e]
        if wd in swe:
          if wd[0] == "<" and len(wd) > 2:
            # Special tag: treat as atomic, stop searching
            kouho = [(swe[wd], e)]
            break
          else:
            kouho.append((swe[wd], e))
      if kouho:
        s = sorted(kouho, key=lambda x: x[0])
        wp, e = s[0]
        if len(s) > 1 and bpe_dropout_rate > 0.0:
          if np.random.random() > bpe_dropout_rate:
            p = np.exp(np.arange(len(s) - 1))[::-1]
            p /= p.sum()
            wp, e = s[np.random.choice(np.arange(len(s) - 1) + 1, p=p)]
        result.append(self._convert_id_to_token(wp))
        pos = e
      else:
        wd = text[pos]
        if self._is_kigou(wd):
          result.append("<KIGOU>")
        elif self._is_u2e(wd):
          result.append("<U2000U2BFF>")
        else:
          for byte in wd.encode("utf-8"):
            result.append(f"<|byte{byte}|>")
        pos += 1
    return result

  @staticmethod
  def _is_kigou(x: str) -> bool:
    e = x.encode()
    if len(x) == 1 and len(e) == 2:
      c = (int(e[0]) << 8) + int(e[1])
      return (
        (0xC2A1 <= c <= 0xC2BF)
        or (0xC780 <= c <= 0xC783)
        or (0xCAB9 <= c <= 0xCBBF)
        or (0xCC80 <= c <= 0xCDA2)
      )
    return False

  @staticmethod
  def _is_u2e(x: str) -> bool:
    e = x.encode()
    if len(x) == 1 and len(e) == 3:
      c = (int(e[0]) << 16) + (int(e[1]) << 8) + int(e[2])
      return 0xE28080 <= c <= 0xE2B07F
    return False


# ----------------------------------------------------------------------


class SWETrainer:
  """
  Optional BPE vocabulary trainer for SWETokenizer.

  Learns merge rules from a text corpus and writes a vocab.txt compatible
  with SWETokenizer (same format as ja-swe32kfix.txt).

  Args:
    vocab_size: Target vocabulary size (default 32000).
    emoji_file: Path to emoji.json (used only to validate the file exists).
    min_frequency: Minimum pair frequency to merge (default 2).

  Example:
    trainer = SWETrainer(vocab_size=32000)
    trainer.train(["corpus_ja.txt", "corpus_en.txt"], "my-vocab.txt")
    tokenizer = SWETokenizer("my-vocab.txt", "emoji.json")
  """

  # Fixed special tokens appended after BPE tokens (283 total)
  _FIXED_TOKENS: ClassVar[List[str]] = [
    "<BR>", "<SP>", "<TAB>",
    "<URL>", "<EMAIL>", "<TEL>", "<DATE>", "<PRICE>",
    "<BLOCK>", "<KIGOU>", "<U2000U2BFF>",
    *[f"<|emoji{i}|>" for i in range(1, 13)],
    *[f"<|byte{i}|>" for i in range(256)],
    "<|startoftext|>", "<|separator|>", "<|nottoken|>", "<|endoftext|>",
  ]

  def __init__(
    self,
    vocab_size: int = 32000,
    emoji_file: str = "emoji.json",
    min_frequency: int = 2,
  ):
    if vocab_size <= len(self._FIXED_TOKENS):
      raise ValueError(
        f"vocab_size ({vocab_size}) must be larger than "
        f"the number of fixed special tokens ({len(self._FIXED_TOKENS)})."
      )
    self.vocab_size = vocab_size
    self.emoji_file = emoji_file
    self.min_frequency = min_frequency
    self._n_bpe = vocab_size - len(self._FIXED_TOKENS)

  def train(self, files: List[str], output_vocab: str) -> None:
    """Train BPE vocabulary from text files and save to output_vocab."""
    def _iter():
      for path in files:
        with open(path, encoding="utf-8") as f:
          yield f.read()
    self.train_from_iterator(_iter(), output_vocab)

  def train_from_iterator(self, texts: Iterator[str], output_vocab: str) -> None:
    """Train BPE vocabulary from a text iterator and save to output_vocab."""
    # Step 1: Build segment frequency table.
    # Each whitespace-separated segment becomes a char tuple.
    # Japanese text (no spaces) → one long tuple per line-segment, enabling
    # cross-character BPE merges (e.g. "今"+"日" → "今日").
    seg_freqs: Counter = Counter()
    for text in texts:
      for seg in re.findall(r"\S+", text):
        seg_freqs[tuple(seg)] += 1

    if self.min_frequency > 1:
      seg_freqs = Counter({
        w: f for w, f in seg_freqs.items() if f >= self.min_frequency
      })

    if not seg_freqs:
      raise ValueError("No segments found in corpus after frequency filtering.")

    # Step 2: Collect base character vocabulary (frequency-sorted).
    char_freqs: Counter = Counter()
    for word, freq in seg_freqs.items():
      for ch in word:
        char_freqs[ch] += freq
    sorted_chars: List[str] = [c for c, _ in char_freqs.most_common()]

    # Step 3: BPE training loop.
    # corpus maps each unique segment (tuple) to its current token list.
    corpus: Dict[Tuple, List[str]] = {w: list(w) for w in seg_freqs}
    n_target_merges = max(0, self._n_bpe - len(sorted_chars))
    merges: List[str] = []

    print(
      f"Training BPE: {len(seg_freqs)} unique segments, "
      f"{len(sorted_chars)} base chars, target {n_target_merges} merges …"
    )

    while len(merges) < n_target_merges:
      # Count adjacent pair frequencies across corpus
      pair_freqs: Counter = Counter()
      for word, tokens in corpus.items():
        freq = seg_freqs[word]
        for i in range(len(tokens) - 1):
          pair_freqs[(tokens[i], tokens[i + 1])] += freq

      if not pair_freqs:
        break
      best = max(pair_freqs, key=pair_freqs.get)
      if pair_freqs[best] < self.min_frequency:
        break

      merged = best[0] + best[1]
      merges.append(merged)

      # Apply merge to corpus
      new_corpus: Dict[Tuple, List[str]] = {}
      for word, tokens in corpus.items():
        new_toks: List[str] = []
        i = 0
        while i < len(tokens):
          if (
            i < len(tokens) - 1
            and tokens[i] == best[0]
            and tokens[i + 1] == best[1]
          ):
            new_toks.append(merged)
            i += 2
          else:
            new_toks.append(tokens[i])
            i += 1
        new_corpus[word] = new_toks
      corpus = new_corpus

      if len(merges) % 1000 == 0:
        print(f"  {len(merges)} merges done …")

    # Step 4: Build final token list and write vocab file.
    # Order: merged tokens (most frequent first) → base chars → fixed tokens
    n_chars = max(0, self._n_bpe - len(merges))
    bpe_tokens = merges + sorted_chars[:n_chars]
    final_tokens = bpe_tokens + self._FIXED_TOKENS

    with open(output_vocab, "w", encoding="utf-8") as f:
      for token in final_tokens:
        f.write(token + "\n")

    print(
      f"Saved {len(final_tokens)} tokens to {output_vocab} "
      f"({len(merges)} merges + {min(n_chars, len(sorted_chars))} chars "
      f"+ {len(self._FIXED_TOKENS)} fixed)."
    )
