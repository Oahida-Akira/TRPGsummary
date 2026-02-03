import os
import re
import subprocess
import datetime
import torch
import whisper
from dotenv import load_dotenv
from bert_score import score as bert_score
from openai import OpenAI

# ==== 設定 ====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY が .env に設定されていません。")
client = OpenAI(api_key=api_key)

# ==== 入力 ====
print("===あらすじ風要約文作成===")
title = input("シナリオタイトル：")
characters = input("主な登場人物（カンマ区切り）：")
keywords = input("キーワード／モチーフ：")
video_path = input("映像ファイル（.mp4パス）：").strip('"')

# ==== 出力フォルダ作成 ====
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{now}"
os.makedirs(output_dir, exist_ok=True)
cut_video_path = os.path.join(output_dir, "temp_video.mp4")
audio_path = os.path.join(output_dir, "audio.wav")

# ==== 映像を60分にカット ====
print("▶ 映像を60分にカットしています...")
subprocess.run([
    "ffmpeg", "-y", "-i", video_path,
    "-ss", "00:00:00", "-t", "01:00:00",
    "-c", "copy", cut_video_path
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==== 音声抽出 ====
print("▶ 音声を抽出しています...")
subprocess.run([
    "ffmpeg", "-y", "-i", cut_video_path,
    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==== Whisper文字起こし ====
print("📝 Whisperで文字起こし中...")
model = whisper.load_model("base")
transcription = model.transcribe(audio_path, fp16=False)
base_transcript = transcription["text"].strip()

with open(os.path.join(output_dir, "transcript.txt"), "w", encoding="utf-8") as f:
    f.write(base_transcript)

# ==== あらすじ案の作成 ====
print("🧠 GPTであらすじ案を生成中...")
prompt = f"""
あなたは、TRPGのセッションから映画予告風あらすじを作るプロのライターです。

以下はTRPGセッション「{title}」の全文文字起こしです。

主な登場人物：{characters}
キーワードやモチーフ：{keywords}

この文字起こしをもとに、TRPG未経験の人にも伝わるような、感情に訴える映画予告のようなあらすじを【3案】、それぞれ150〜300字程度で書いてください。
ただしあらすじタイトルなどは必要なく、本文だけを出力してください。

--- 登場人物 ---
{characters}

--- キーワード・モチーフ ---
{keywords}

--- セッション文字起こし ---
{base_transcript}
"""

res = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.9
)

raw_output = res.choices[0].message.content.strip()

summary_candidates = re.split(r"\n\s*\n+", raw_output)
summary_candidates = [s.strip() for s in summary_candidates if s.strip()]

# ==== BERTScoreによる評価 ====
print("📊 BERTScoreを計算中...")
references = [base_transcript] * len(summary_candidates)

P, R, F1 = bert_score(summary_candidates, references, lang="ja", verbose=False)
bert_scores = F1.tolist()

# ==== BERTScoreで最良案選定 ====
best_idx = bert_scores.index(max(bert_scores))
best_summary = summary_candidates[best_idx]

# ==== 出力 ====
with open(os.path.join(output_dir, "summary_candidates.txt"), "w", encoding="utf-8") as f:
    for i, s in enumerate(summary_candidates):
        f.write(f"【案{i+1}】\n{s}\n[BERTScore(F1)]: {bert_scores[i]:.4f}\n\n")

with open(os.path.join(output_dir, "final_summary.txt"), "w", encoding="utf-8") as f:
    f.write(best_summary)

# 不要ファイル削除（映像・音声）
os.remove(cut_video_path)
os.remove(audio_path)

# 完了通知
print("\n✅ 全プロセス完了")
print(f"📁 出力フォルダ: {output_dir}")
print(f"🎬 最良あらすじ案: {os.path.join(output_dir, 'final_summary.txt')}")
