from __future__ import annotations

from dotenv import load_dotenv; load_dotenv()
import os
import hmac
import base64
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta
import re
import sqlite3

import requests
import schedule
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ====== 設定 ======
DEFAULT_MINUTES_BEFORE = int(os.getenv("DEFAULT_MINUTES_BEFORE", "10"))

DOW_ATTR = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DOW_LABEL_TO_INDEX = {
    "mon": 0, "monday": 0, "tue": 1, "tuesday": 1, "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3, "fri": 4, "friday": 4, "sat": 5, "saturday": 5, "sun": 6, "sunday": 6,
    "月": 0, "月曜": 0, "月曜日": 0, "火": 1, "火曜": 1, "火曜日": 1, "水": 2, "水曜": 2, "水曜日": 2,
    "木": 3, "木曜": 3, "木曜日": 3, "金": 4, "金曜": 4, "金曜日": 4, "土": 5, "土曜": 5, "土曜日": 5,
    "日": 6, "日曜": 6, "日曜日": 6,
}

DEFAULT_PERIOD_MAP = {
    "1": {"start": "09:00", "end": "10:30"},
    "2": {"start": "10:40", "end": "12:10"},
    "3": {"start": "13:00", "end": "14:30"},
    "4": {"start": "14:40", "end": "16:10"},
    "5": {"start": "16:20", "end": "17:50"},
    "6": {"start": "18:00", "end": "19:30"},
}

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini")

VIEW_COMMANDS = {"一覧", "確認", "時間割", "schedule", "list"}

HELP_COMMANDS = {"help", "ヘルプ", "使い方"}

RESET_COMMANDS = {"リセット", "reset", "クリア", "clear"}
ADD_COMMANDS = {"追加", "add"}
DELETE_COMMANDS = {"削除", "delete"}
REPLACE_COMMANDS = {"置換", "修正", "変更", "replace"}
LOCATION_COMMANDS = {"教室", "場所", "location", "room"}
CONFIRM_POSITIVE = {"はい", "はい。", "yes", "y", "ok", "了解", "うん", "実行", "承認"}
CONFIRM_NEGATIVE = {"いいえ", "いいえ。", "no", "n", "やめる", "キャンセル", "不要"}
CONFIRM_POSITIVE_NORMALIZED = {s.lower() for s in CONFIRM_POSITIVE}
CONFIRM_NEGATIVE_NORMALIZED = {s.lower() for s in CONFIRM_NEGATIVE}



HELP_TEXT = (

    "利用できる主なコマンド：\n"

    "・通知 10\n　通知タイミングを分単位で設定\n"

    "・追加 / 削除 / 置換 / 教室\n　キーワードだけ送信すると詳細入力モードになります。続けて授業名や時間を自由な文章で送ると自動で解釈し、実行前に確認メッセージが届きます。\n"
    "　直接「追加 月曜1限 解析学」などと書いても同じ処理を行います。\n"

    "・リセット\n　登録済みの時間割をすべて削除\n"

    "・一覧\n　現在登録済みの時間割を表示\n"

    "・各コマンド実行前には確認メッセージ（はい／いいえ）が届きます\n"

    "\nそのほか時間割テキスト／画像を送信すると解析して登録します。"

)



DB_PATH = os.getenv("USER_STATE_DB", os.path.join(os.getcwd(), "user_state.db"))
DB_CONN: sqlite3.Connection | None = None
DB_LOCK = threading.Lock()

# ====== ユーティリティ ======
def normalize_day(day_str: str) -> int:
    key = day_str.strip().lower()
    key = {"thu.": "thu", "tue.": "tue"}.get(key, key)
    if key not in DOW_LABEL_TO_INDEX:
        key = key[:3]
    if key not in DOW_LABEL_TO_INDEX:
        raise ValueError(f"未知の曜日表現: {day_str}")
    return DOW_LABEL_TO_INDEX[key]

def hhm_to_time(hm: str) -> datetime:
    return datetime.strptime(hm, "%H:%M")

def compute_notify_slot(day_idx: int, start_hm: str, offset_min: int):
    base_monday = datetime(2000, 1, 3, 0, 0)  # Monday
    start_dt = base_monday + timedelta(days=day_idx)
    t = hhm_to_time(start_hm)
    start_dt = start_dt.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    notify_dt = start_dt - timedelta(minutes=offset_min)
    return notify_dt.weekday(), notify_dt.strftime("%H:%M")

def extract_json_from_text(text: str) -> str:
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, end = decoder.raw_decode(text[idx:])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
    raise ValueError("LINEのメッセージの返信はできませんので、ご了承ください。操作方法の確認の際は「ヘルプ」と送信してください。")

def normalize_schedule_response(data, timezone="Asia/Tokyo"):
    if isinstance(data, list):
        data = {"timezone": timezone, "schedule": data}
    elif isinstance(data, dict):
        if "schedule" not in data:
            for key in ("classes", "items", "entries", "timetable", "lessons"):
                if isinstance(data.get(key), list):
                    data["schedule"] = data[key]
                    break
        data.setdefault("timezone", timezone)
    else:
        raise ValueError("JSON形式を解釈できませんでした。")
    data = ensure_schedule_dict(data)
    return data

LOCATION_HEAD_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}$")
LOCATION_COMBINED_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}[-－]?\d{2,3}$")

def _looks_like_location_token(token: str, already_has_loc: bool) -> bool:
    token = token.strip()
    if not token:
        return False
    if "教室" in token or "号館" in token:
        return True
    if LOCATION_COMBINED_RE.match(token):
        return True
    if LOCATION_HEAD_RE.match(token):
        return True
    if already_has_loc and re.match(r"^\d{2,3}$", token):
        return True
    return False

def split_course_and_location(course_raw: str) -> tuple[str, str | None]:
    text = course_raw.strip()
    if not text:
        return "", None

    # e.g. "数学 (101教室)" -> course + location
    m = re.search(r"(?:\(|（)([^()（）]*?教室[^()（）]*?)(?:\)|）)\s*$", text)
    if m:
        course = text[:m.start()].rstrip(" ／、-－　")
        return course or text, m.group(1).strip()

    # e.g. "... C3 100", "... W2 101"
    m = re.search(r"(?:\s|　)(?P<loc>[A-Z]{1,2}\d{1,2}(?:[-－]?\d{2,3}|\s+\d{2,3}))\s*$", text)
    if m:
        course = text[:m.start()].rstrip(" ／、-－　")
        loc = re.sub(r"\s+", " ", m.group("loc")).strip()
        return course or text, loc

    tokens = re.split(r"\s+", text)
    loc_tokens: list[str] = []
    while tokens:
        last = tokens[-1]
        if not last:
            tokens.pop()
            continue
        if not loc_tokens and re.match(r"^\d{2,3}$", last) and len(tokens) >= 2 and LOCATION_HEAD_RE.match(tokens[-2]):
            loc_tokens.insert(0, tokens.pop())
            continue
        if _looks_like_location_token(last, bool(loc_tokens)):
            loc_tokens.insert(0, tokens.pop())
            continue
        break

    if loc_tokens:
        course = " ".join(tokens).rstrip(" ／、-－　")
        loc = " ".join(loc_tokens).strip()
        return (course or text).strip(), loc or None
    return text, None


def extract_command_payload(text: str, keyword: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith(keyword):
        return None
    rest = stripped[len(keyword):]
    if not rest:
        return ""
    if rest[0] in (":", "："):
        rest = rest[1:]
    return rest.strip()


def parse_replacement_parts(body: str) -> tuple[str, str] | None:
    for sep in ("->", "\u2192"):
        if sep in body:
            left, right = body.split(sep, 1)
            left = left.strip()
            right = right.strip()
            if left and right:
                return left, right

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])

    parts = re.split(r"\s{2,}|[／/|｜]", body, maxsplit=1)
    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        if left and right:
            return left, right
    return None


def format_entry_lines(entries: list[dict]) -> str:
    lines = []
    for entry in entries:
        day = entry.get("day", "?")
        start = entry.get("start", "--:--")
        end = entry.get("end") or ""
        course = entry.get("course", "").strip()
        location = entry.get("location") or "未設定"
        time_part = f"{start}-{end}" if end else start
        lines.append(f"- {day} {time_part} {course} / 教室: {location}")
    return "\n".join(lines) if lines else "対象が空です。"


def parse_entries_with_gpt(text: str, timezone: str = "Asia/Tokyo") -> dict:
    schedule_data = None
    err = None
    try:
        schedule_data = call_chatgpt_to_extract_schedule(
            text,
            model=DEFAULT_OPENAI_MODEL,
            timezone=timezone,
        )
    except Exception as e:
        err = e
    if not schedule_data or not schedule_data.get("schedule"):
        fallback = parse_timetable_locally(text, timezone=timezone)
        if fallback.get("schedule"):
            schedule_data = fallback
        else:
            raise ValueError(f"解析に失敗しました。{err or ''}".strip())
    for entry in schedule_data.get("schedule", []):
        entry.setdefault("location", None)
    return schedule_data


def entry_matches_pattern(entry: dict, pattern: dict) -> bool:
    if not pattern:
        return False
    if pattern.get("day") and entry.get("day") != pattern["day"]:
        return False
    if pattern.get("start") and entry.get("start") != pattern["start"]:
        return False
    if pattern.get("end") and entry.get("end") != pattern["end"]:
        return False
    course = (pattern.get("course") or "").strip()
    if course and course not in entry.get("course", ""):
        return False
    loc = (pattern.get("location") or "").strip()
    if loc:
        entry_loc = (entry.get("location") or "")
        if not entry_loc or loc not in entry_loc:
            return False
    return True


def _course_match(existing: dict, pattern: dict) -> bool:
    course = (pattern.get("course") or "").strip().lower()
    if not course:
        return False
    existing_course = (existing.get("course") or "").lower()
    return course in existing_course


def delete_entries_by_patterns(base: dict, patterns: list[dict]) -> tuple[dict, int]:
    targets = [dict(p, _matched=False) for p in patterns if p]
    removed = 0
    remaining = []

    for entry in base.get("schedule", []):
        matched = False
        for target in targets:
            if target["_matched"]:
                continue
            if entry_matches_pattern(entry, target):
                target["_matched"] = True
                matched = True
                removed += 1
                break
        if not matched:
            remaining.append(entry)

    if removed == len(targets) or not targets:
        return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": remaining}, removed

    new_remaining = []
    for entry in remaining:
        matched = False
        for target in targets:
            if target["_matched"]:
                continue
            if _course_match(entry, target):
                target["_matched"] = True
                matched = True
                removed += 1
                break
        if not matched:
            new_remaining.append(entry)

    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_remaining}, removed


def replace_entry_with_pattern(base: dict, old_pattern: dict, new_entry: dict) -> tuple[dict, bool]:
    replaced = False
    new_list = []
    for entry in base.get("schedule", []):
        if not replaced and entry_matches_pattern(entry, old_pattern):
            new_entry.setdefault("location", new_entry.get("location"))
            new_list.append({
                "day": new_entry.get("day", entry.get("day")),
                "start": new_entry.get("start", entry.get("start")),
                "end": new_entry.get("end") or entry.get("end"),
                "course": new_entry.get("course", entry.get("course")),
                "location": new_entry.get("location") if new_entry.get("location") is not None else entry.get("location"),
            })
            replaced = True
        else:
            new_list.append(entry)

    if replaced:
        return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_list}, True

    for idx, entry in enumerate(new_list):
        if _course_match(entry, old_pattern):
            new_list[idx] = {
                "day": new_entry.get("day", entry.get("day")),
                "start": new_entry.get("start", entry.get("start")),
                "end": new_entry.get("end") or entry.get("end"),
                "course": new_entry.get("course", entry.get("course")),
                "location": new_entry.get("location") if new_entry.get("location") is not None else entry.get("location"),
            }
            return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_list}, True

    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_list}, False


def update_locations_with_entries(base: dict, entries: list[dict]) -> tuple[dict, int]:
    targets = []
    for entry in entries:
        loc = (entry.get("location") or "").strip()
        if not loc:
            continue
        target = dict(entry)
        target["_matched"] = False
        target["location"] = loc
        targets.append(target)

    if not targets:
        return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": base.get("schedule", [])}, 0

    updated = 0
    new_list = []
    schedule_list = base.get("schedule", [])

    # First pass: strict matching by day/time/course/location pattern.
    for existing in schedule_list:
        updated_entry = dict(existing)
        for target in targets:
            if target["_matched"]:
                continue
            if entry_matches_pattern(existing, target):
                updated_entry["location"] = target["location"]
                target["_matched"] = True
                updated += 1
                break
        new_list.append(updated_entry)

    if updated == len(targets):
        return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_list}, updated

    for idx, existing in enumerate(new_list):
        for target in targets:
            if target["_matched"]:
                continue
            if _course_match(existing, target):
                new_list[idx] = dict(existing)
                new_list[idx]["location"] = target["location"]
                target["_matched"] = True
                updated += 1
                break

    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new_list}, updated


def set_pending_action(user_id: str, action: dict):
    state = ensure_user_state(user_id)
    state["pending_action"] = action


def clear_pending_action(user_id: str):
    state = ensure_user_state(user_id)
    state.pop("pending_action", None)


def send_confirmation_prompt(user_id: str, header: str, summary: str):
    msg = f"{header}\n{summary}\nこの内容で実行しますか？\nはい / いいえ"
    line_push_text(user_id, msg)


def set_pending_command(user_id: str, command_type: str):
    state = ensure_user_state(user_id)
    state["pending_command"] = command_type


def clear_pending_command(user_id: str):
    state = ensure_user_state(user_id)
    state.pop("pending_command", None)


COMMAND_GUIDANCE = {
    "add": "追加したい授業を自由な文章で教えてください。（例）月曜1限の解析学を追加して、教室はC3 101。",
    "delete": "削除したい授業を自由に書いてください。（例）火曜10:40-12:10の英語を削除。",
    "replace": "どの授業をどの授業に置き換えるかを自由に書いてください。（例）月曜1限解析学を火曜2限英語に変更。",
    "location": "教室を変更したい授業と新しい教室を自由に書いてください。（例）水曜3限情報セキュリティをW2-201に移動。",
}



def request_command_details(reply_token: str, user_id: str, command_type: str):
    guidance = COMMAND_GUIDANCE.get(command_type, "内容を自由に入力してください。")
    set_pending_command(user_id, command_type)
    line_reply_text(
        reply_token,
        [f"{guidance}\nキャンセルする場合は『いいえ』または『キャンセル』と送fてください。"],
    )



def call_chatgpt_to_extract_schedule(timetable_text: str, model=DEFAULT_OPENAI_MODEL, timezone="Asia/Tokyo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    if OpenAI is None:
        raise RuntimeError("openai ライブラリが見つかりません。pip install openai を実行してください。")

    period_hint = "\n".join(
        f"{k}限: {v['start']}~{v['end']}" for k, v in DEFAULT_PERIOD_MAP.items()
    )

    system = (
        "あなたは厳密なJSON抽出器です。"
        "大学の時間割テキスト（日本語を含む）から授業情報を抽出し、指定フォーマットのJSONのみを出力してください。"
    )
    user = f"""
以下の文章には大学の時間割が記載されています。各授業について次の項目を必ず含めてください。
- day: Mon/Tue/Wed/Thu/Fri/Sat/Sun のいずれか
- start, end: 24時間表記 HH:MM
- course: 授業名のみ（Tから始まる7桁IDや「(16T以降)」などの注記は含めない）
- location: 教室名・教室コード（例: C3 100, W1-115, 101教室）。教室が無い場合は null
- Tから始まる7桁ID（例: T2043200）は授業コードなので JSON には一切出力しない

「n限」表記は以下の対応で開始・終了時刻に変換してください（本文に具体的な時刻があればそちらを優先）。
{period_hint}

出力はJSONのみで、説明文やコードブロックは不要です。

対象テキスト:
---
{timetable_text}
---
""".strip()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    print(content)  # Debug log
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        content = "\n".join(parts)
    if not isinstance(content, str):
        raise ValueError(f"Unexpected content type: {type(content)}")

    raw_json = extract_json_from_text(content)
    data = normalize_schedule_response(json.loads(raw_json), timezone=timezone)
    for entry in data["schedule"]:
        entry.setdefault("location", None)
    return data


def call_chatgpt_to_extract_schedule_from_image(image_bytes: bytes, mime_type: str = "image/png",
                                                model=DEFAULT_VISION_MODEL, timezone="Asia/Tokyo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    if OpenAI is None:
        raise RuntimeError("openai ライブラリが見つかりません。pip install openai を実行してください。")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    system = (
        "あなたは厳密な JSON 抽出器です。"
        "時間割のスクリーンショット（日本語を含む）から授業情報を抽出し、指定フォーマットの JSON だけを出力してください。"
    )
    period_hint = "\n".join(
        f"{k}限: {v['start']}~{v['end']}" for k, v in DEFAULT_PERIOD_MAP.items()
    )
    user_text = f"""
以下の画像には大学の時間割が写っています。各授業について曜日・開始時刻・終了時刻・授業名・教室を抽出し、統一フォーマットの JSON で出力してください。

制約:
- 出力は JSON のみ（説明文やコードブロックは禁止）
- 時刻は 24 時間表記 HH:MM
- 曜日は Mon/Tue/Wed/Thu/Fri/Sat/Sun
- course には授業名のみ（Tから始まる7桁IDや「(16T以降)」などの注記は含めない）
- location には教室表記（例: C3 100, W2-101, 101教室）のみを入れる。教室が無い場合は null
- Tから始まる7桁ID（例: T2043200）は授業コードなので、JSON のどのフィールドにも出力しない
- 「n限」表記は以下のデフォルト対応で開始・終了時刻に変換する（画像内に明示的な時刻があればそちらを優先）:
{period_hint}

スキーマ:
{{
  "timezone": "{timezone}",
  "schedule": [
    {{"day": "Mon", "start": "09:00", "end": "10:30", "course": "科目名", "location": "C3 100"}}
  ]
}}
""".strip()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    )
    content = resp.choices[0].message.content

    print("[vision raw content]", content, flush=True)  # Debug log

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        content = "\n".join(parts)

    if not isinstance(content, str):
        raise ValueError(f"Unexpected content type: {type(content)}")
    
    print("[vision processed content]", content, flush=True)  # Debug log

    raw_json = extract_json_from_text(content)
    
    print("[vision extracted JSON]", raw_json, flush=True)  # Debug log

    data = normalize_schedule_response(json.loads(raw_json), timezone=timezone)
    for entry in data["schedule"]:
        entry.setdefault("location", None)
    return data


def call_chatgpt_to_extract_replacement(text: str, model=DEFAULT_OPENAI_MODEL, timezone="Asia/Tokyo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    if OpenAI is None:
        raise RuntimeError("openai ライブラリが見つかりません。pip install openai を実行してください。")

    period_hint = "\n".join(
        f"{k}限: {v['start']}~{v['end']}" for k, v in DEFAULT_PERIOD_MAP.items()
    )
    system = (
        "あなたは時間割の置換指示を抽出するエージェントです。"
        "timezone / old / new を含む JSON のみを出力してください。"
    )
    user = f"""
文章にはどの授業をどの授業に置き換えるかが記載されています。
- day は Mon/Tue/... の形
- start,end は 24時間表記 HH:MM
- course は授業名のみ（Tから始まる7桁IDや「(16T以降)」などの注記は除外）
- location は教室名のみ（無ければ null）
- Tから始まる7桁IDは JSON に出力しない

「n限」表記は以下の対応で解釈してください（本文に具体的な時刻があればそちらを優先）:
{period_hint}

出力は JSON のみです。

例:
{
  "timezone": "{timezone}",
  "old": {"day": "Mon", "start": "09:00", "end": "10:30", "course": "解析学", "location": null},
  "new": {"day": "Tue", "start": "10:40", "end": "12:10", "course": "英語", "location": "C3 100"}
}
""".strip()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    raw_json = extract_json_from_text(content)
    data = json.loads(raw_json)
    if "old" not in data or "new" not in data:
        raise ValueError("old/new が含まれていない応答でした。")
    data.setdefault("timezone", timezone)
    return data


def call_chatgpt_to_extract_location_updates(text: str, model=DEFAULT_OPENAI_MODEL, timezone="Asia/Tokyo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    if OpenAI is None:
        raise RuntimeError("openai ライブラリが見つかりません。pip install openai を実行してください。")

    period_hint = "\n".join(
        f"{k}限: {v['start']}~{v['end']}" for k, v in DEFAULT_PERIOD_MAP.items()
    )
    system = (
        "あなたは教室変更の指示を抽出するエージェントです。"
        "timezone と updates を含む JSON のみを出力してください。"
    )
    user = f"""
文章には教室を変更したい授業と新しい教室名が書かれています。複数指定される場合もあります。
曜日は Mon/Tue... または 月/火...、時刻は 24時間表記 HH:MM に直してください。
- course は授業名のみ（Tから始まる7桁IDや「(16T以降)」などの注記は除外）
- location は教室名のみを入れ、Tから始まるIDは無視する

「n限」表記は以下の対応で時刻に変換します（本文に具体的な時刻があればそちらを優先）:
{period_hint}

出力フォーマット:
{{
  "timezone": "{timezone}",
  "updates": [
    {{"day": "Mon", "start": "09:00", "end": "10:30", "course": "解析学", "location": "C3 101"}}
  ]
}}
""".strip()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    if isinstance(content, list):
        content = "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    raw_json = extract_json_from_text(content)
    data = json.loads(raw_json)
    data.setdefault("timezone", timezone)
    updates = data.get("updates") or data.get("schedule")
    if not updates:
        raise ValueError("updates が空でした。")
    for entry in updates:
        entry.setdefault("location", None)
    data["updates"] = updates
    return data


# ====== LINE API / FastAPI Webhook (Unified with async + corrections) ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"
LINE_CONTENT_URL = "https://api-data.line.me/v2/bot/message/{message_id}/content"

# ユーザーごとの状態保持（簡易版: メモリ）
USER_STATE = {}  # user_id -> {"minutes_before": int, "schedule": dict, "updated_at": datetime}

def _empty_schedule():
    return {"timezone": "Asia/Tokyo", "schedule": []}

def ensure_schedule_dict(data):
    if not isinstance(data, dict):
        return _empty_schedule()
    data.setdefault("timezone", "Asia/Tokyo")
    data.setdefault("schedule", [])
    return data

def ensure_user_state(user_id: str):
    state = USER_STATE.get(user_id)
    if not state:
        state = {"minutes_before": DEFAULT_MINUTES_BEFORE, "schedule": _empty_schedule(), "updated_at": datetime.now()}
        USER_STATE[user_id] = state
    else:
        state.setdefault("minutes_before", DEFAULT_MINUTES_BEFORE)
        state["schedule"] = ensure_schedule_dict(state.get("schedule"))
        state.setdefault("updated_at", datetime.now())
    return state

def verify_signature(body: bytes, signature: str) -> bool:
    if not LINE_SECRET:
        return False
    mac = hmac.new(LINE_SECRET.encode("utf-8"), body, hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, signature or "")

def line_push_text(to_user_id: str, text: str):
    if not LINE_TOKEN:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    body = {"to": to_user_id, "messages": [{"type": "text", "text": text[:2000]}]}
    r = requests.post(LINE_PUSH_URL, headers=headers, data=json.dumps(body), timeout=15)
    if r.status_code >= 300:
        raise RuntimeError(f"LINE Push失敗: {r.status_code} {r.text}")

def line_reply_text(reply_token: str, texts):
    if not LINE_TOKEN:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
    headers = {"Authorization": f"Bearer {LINE_TOKEN}", "Content-Type": "application/json"}
    msgs = [{"type": "text", "text": str(t)[:2000]} for t in (texts if isinstance(texts, list) else [texts])]
    body = {"replyToken": reply_token, "messages": msgs}
    r = requests.post(LINE_REPLY_URL, headers=headers, data=json.dumps(body), timeout=15)
    if r.status_code >= 300:
        print(f"LINE Reply失敗: {r.status_code} {r.text}")

def download_line_content(message_id: str) -> tuple[bytes, str]:
    if not LINE_TOKEN:
        raise RuntimeError("LINE_CHANNEL_ACCESS_TOKEN が未設定です。")
    url = LINE_CONTENT_URL.format(message_id=message_id)
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"LINE Content取得失敗: {r.status_code} {r.text}")
    mime = r.headers.get("Content-Type", "application/octet-stream")
    return r.content, mime

# 既存のschedule_jobs_for_userがなければ以下の関数で差し替え
def schedule_jobs_for_user(user_id: str, schedule_data: dict, minutes_before: int = DEFAULT_MINUTES_BEFORE):
    schedule.clear(f"user:{user_id}")  # 既存ジョブをクリア
    for entry in schedule_data.get("schedule", []):
        try:
            day_idx = normalize_day(entry["day"])
            start = entry["start"]
            end = entry.get("end")
            course = entry["course"].strip()
            location = (entry.get("location") or "").strip()
        except Exception as e:
            print(f"スキップ（不正エントリ）: {entry} -> {e}")
            continue

        notify_day_idx, notify_hm = compute_notify_slot(day_idx, start, minutes_before)
        dow_attr = DOW_ATTR[notify_day_idx]
        title = f"[授業通知] {course}"
        msg = f"{title}\n開始: {start}（{minutes_before}分前通知）"
        if end:
            msg += f"\n終了: {end}"
        msg += f"\n曜日: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day_idx]}"
        msg += f"\n教室: {location or '未設定'}"

        getattr(schedule.every(), dow_attr).at(notify_hm).tag(f"user:{user_id}").do(
            line_push_text, to_user_id=user_id, text=msg
        )
        print(f"ジョブ登録: user={user_id} {dow_attr} {notify_hm} {course}")

def init_storage():
    global DB_CONN
    if DB_CONN:
        return
    try:
        DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        DB_CONN.execute(
            """
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                minutes_before INTEGER NOT NULL,
                schedule_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        DB_CONN.commit()
        load_states_from_db()
        print(f"DB初期化完了: {DB_PATH}")
    except Exception as e:
        DB_CONN = None
        print(f"DB初期化に失敗しました: {e}")

def load_states_from_db():
    if DB_CONN is None:
        return
    try:
        cur = DB_CONN.execute("SELECT user_id, minutes_before, schedule_json, updated_at FROM user_state")
    except Exception as e:
        print(f"ユーザ状態の読込に失敗しました: {e}")
        return
    rows = cur.fetchall()
    for user_id, minutes_before, schedule_json, updated_at in rows:
        try:
            schedule_data = json.loads(schedule_json)
        except Exception as e:
            print(f"schedule_jsonのパースに失敗しました(user_id={user_id}): {e}")
            schedule_data = _empty_schedule()
        state = {
            "minutes_before": minutes_before if minutes_before is not None else DEFAULT_MINUTES_BEFORE,
            "schedule": ensure_schedule_dict(schedule_data),
            "updated_at": datetime.fromisoformat(updated_at) if updated_at else datetime.now(),
        }
        USER_STATE[user_id] = state
        if state["schedule"].get("schedule"):
            try:
                schedule_jobs_for_user(user_id, state["schedule"], minutes_before=state["minutes_before"])
            except Exception as e:
                print(f"ジョブ再登録に失敗しました(user_id={user_id}): {e}")


def get_all_user_ids(force_db: bool = True) -> list[str]:
    ids = set(USER_STATE.keys())
    if force_db:
        if DB_CONN is None:
            try:
                init_storage()
            except Exception as e:
                print(f"DB初期化に失敗しました(get_all_user_ids): {e}")
        if DB_CONN:
            try:
                cur = DB_CONN.execute("SELECT user_id FROM user_state")
                ids.update(row[0] for row in cur.fetchall())
            except Exception as e:
                print(f"user_stateテーブルからの読込に失敗しました: {e}")
    return [user_id for user_id in ids if user_id]

def persist_user_state(user_id: str):
    if DB_CONN is None:
        return
    state = ensure_user_state(user_id)
    schedule_data = ensure_schedule_dict(state.get("schedule"))
    minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)
    updated_at = datetime.now()
    state["updated_at"] = updated_at
    payload = json.dumps(schedule_data, ensure_ascii=False)
    sql = """
        INSERT INTO user_state (user_id, minutes_before, schedule_json, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            minutes_before=excluded.minutes_before,
            schedule_json=excluded.schedule_json,
            updated_at=excluded.updated_at
    """
    try:
        with DB_LOCK:
            DB_CONN.execute(sql, (user_id, minutes_before, payload, updated_at.isoformat()))
            DB_CONN.commit()
    except Exception as e:
        print(f"user_stateの保存に失敗しました(user_id={user_id}): {e}")

def reset_user_schedule(user_id: str):
    schedule.clear(f"user:{user_id}")
    state = ensure_user_state(user_id)
    state["schedule"] = _empty_schedule()
    state["updated_at"] = datetime.now()
    persist_user_state(user_id)

def summarize_schedule(schedule_data: dict, limit: int = 20) -> str:
    rows = []
    for item in schedule_data.get("schedule", [])[:limit]:
        location = item.get("location") or "未設定"
        end = item.get("end") or ""
        rows.append(f"{item['day']} {item['start']}-{end} {item['course']} / 教室: {location}")
    if len(schedule_data.get("schedule", [])) > limit:
        rows.append("…")
    return "\n".join(rows) if rows else "登録件数 0 件"


def broadcast_message(text: str, delay: float = 0.1, reload_state: bool = True):
    msg = (text or "").strip()
    if not msg:
        raise ValueError("送信するメッセージが空です。")
    if reload_state:
        if DB_CONN is None:
            init_storage()
        else:
            load_states_from_db()
    user_ids = get_all_user_ids(force_db=True)
    if not user_ids:
        print("送信対象のユーザーが見つかりませんでした。")
        return
    success = 0
    for uid in user_ids:
        try:
            line_push_text(uid, msg)
            success += 1
            if delay:
                time.sleep(delay)
        except Exception as e:
            print(f"[broadcast] 送信失敗 user={uid}: {e}")
    print(f"[broadcast] 完了: {success}/{len(user_ids)} 件送信")

# OpenAIが使えない場合の簡易パーサー（最低限の対応）
DAY_JA_TO_EN = {"月":"Mon","火":"Tue","水":"Wed","木":"Thu","金":"Fri","土":"Sat","日":"Sun"}
def parse_timetable_locally(timetable_text: str, timezone="Asia/Tokyo"):
    schedule = []
    current_day = None
    location_only_re = re.compile(r"^[A-Z]{1,2}\d{1,2}(?:[-－]?\d{2,3}|\s+\d{2,3})$")
    period_line_re = re.compile(r"^(?P<period>\d(?:-\d)?)\s+(?P<body>.+)$")
    day_header_re = re.compile(r"^(?P<day>月曜日|月曜|月|火曜日|火曜|火|水曜日|水曜|水|木曜日|木曜|木|金曜日|金曜|金|土曜日|土曜|土|日曜日|日曜|日|Mon|Tue|Wed|Thu|Fri|Sat|Sun)")

    def _day_to_en(day_str: str | None) -> str | None:
        if not day_str:
            return None
        key = day_str[:3]
        if day_str in DAY_JA_TO_EN:
            return DAY_JA_TO_EN[day_str]
        return DAY_JA_TO_EN.get(day_str[0]) or day_str[:1].upper() + day_str[1:3].lower()

    lines = timetable_text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.strip()
        i += 1
        if not line:
            continue
        day_match = day_header_re.match(line)
        if day_match and len(line) <= 4:
            current_day = _day_to_en(day_match.group("day"))
            continue
        if location_only_re.match(line):
            if schedule and not schedule[-1].get("location"):
                schedule[-1]["location"] = line.replace("  ", " ")
            continue
        period_match = period_line_re.match(line)
        if period_match and current_day:
            period_token = period_match.group("period")
            body = period_match.group("body").strip()
            start_period, end_period = period_token, period_token
            if "-" in period_token:
                start_period, end_period = period_token.split("-", 1)
            start_info = DEFAULT_PERIOD_MAP.get(start_period)
            end_info = DEFAULT_PERIOD_MAP.get(end_period)
            if not start_info:
                continue
            start_hm = start_info["start"]
            end_hm = end_info["end"] if end_info else start_info.get("end")
            body_clean = re.sub(r"T[0-9A-Z]+", " ", body)
            body_clean = re.sub(r"\d+単位", " ", body_clean)
            body_clean = body_clean.replace("再履修", " ")
            body_clean = body_clean.replace("授業毎に", " ").replace("授業中に", " ")
            body_clean = re.sub(r"\s+", " ", body_clean).strip()
            course_name, location = split_course_and_location(body_clean)
            schedule.append({
                "day": current_day,
                "start": start_hm,
                "end": end_hm,
                "course": course_name.strip(),
                "location": location,
            })
            continue
        # fallback: explicit time ranges without day prefix
        m = re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", line, re.I)
        if m:
            day, start, end, course = m.groups()
            day = day[:1].upper() + day[1:3].lower()
            start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
            end = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
            course_name, location = split_course_and_location(course)
            schedule.append({"day": day, "start": start, "end": end, "course": course_name.strip(), "location": location})
            continue
        m = re.match(r"^(月|火|水|木|金|土|日)[曜]?:?\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", line)
        if m:
            dja, start, end, course = m.groups()
            en = DAY_JA_TO_EN.get(dja)
            if en:
                start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
                end = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
                course_name, location = split_course_and_location(course)
                schedule.append({"day": en, "start": start, "end": end, "course": course_name.strip(), "location": location})
            continue
    return {"timezone": timezone, "schedule": schedule}

def merge_additions(base: dict, additions: dict) -> dict:
    keyset = {(e["day"], e["start"], e["course"]) for e in base.get("schedule", [])}
    out = list(base.get("schedule", []))
    for e in additions.get("schedule", []):
        k = (e["day"], e["start"], e["course"].strip())
        if k not in keyset:
            out.append({
                "day": e["day"],
                "start": e["start"],
                "end": e.get("end"),
                "course": e["course"].strip(),
                "location": e.get("location"),
            })
            keyset.add(k)
    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": out}

def summarize_and_push(user_id: str, minutes_before: int, schedule_data: dict, prefix: str):
    msg = f"{prefix}\n通知: {minutes_before}分前\n" + summarize_schedule(schedule_data)
    line_push_text(user_id, msg)


def finalize_action(user_id: str, new_sched: dict, minutes_before: int, prefix: str):
    schedule_jobs_for_user(user_id, new_sched, minutes_before=minutes_before)
    state = ensure_user_state(user_id)
    state["minutes_before"] = minutes_before
    state["schedule"] = ensure_schedule_dict(new_sched)
    state["updated_at"] = datetime.now()
    persist_user_state(user_id)
    summarize_and_push(user_id, minutes_before, new_sched, prefix)


def execute_pending_action(user_id: str):
    state = ensure_user_state(user_id)
    action = state.get("pending_action")
    if not action:
        line_push_text(user_id, "実行保留中の操作はありません。")
        return
    current = state.get("schedule") or _empty_schedule()
    minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)
    try:
        action_type = action.get("type")
        if action_type == "add":
            additions = {"timezone": action.get("timezone", current.get("timezone", "Asia/Tokyo")), "schedule": action.get("entries", [])}
            new_sched = merge_additions(current, additions)
            prefix = "授業の追加が完了しました。"
        elif action_type == "delete":
            new_sched, removed = delete_entries_by_patterns(current, action.get("targets", []))
            if not removed:
                line_push_text(user_id, "条件に一致する授業が見つからなかったため、削除できませんでした。")
                clear_pending_action(user_id)
                return
            prefix = f"{removed}件の授業を削除しました。"
        elif action_type == "replace":
            new_sched, replaced = replace_entry_with_pattern(current, action.get("old"), action.get("new", {}))
            if not replaced:
                line_push_text(user_id, "対象授業が見つからなかったため、置換できませんでした。")
                clear_pending_action(user_id)
                return
            prefix = "授業の置換が完了しました。"
        elif action_type == "location":
            entries = action.get("entries") or action.get("targets", [])
            new_sched, updated = update_locations_with_entries(current, entries)
            if not updated:
                line_push_text(user_id, "教室を更新できる授業が見つかりませんでした。")
                clear_pending_action(user_id)
                return
            prefix = f"{updated}件の授業の教室を更新しました。"
        else:
            line_push_text(user_id, "未対応の操作種別です。")
            clear_pending_action(user_id)
            return
    except Exception as e:
        line_push_text(user_id, f"操作の実行中にエラーが発生しました: {e}")
        clear_pending_action(user_id)
        return

    clear_pending_action(user_id)
    finalize_action(user_id, new_sched, minutes_before, prefix)


def cancel_pending_action(user_id: str) -> bool:
    state = ensure_user_state(user_id)
    if state.get("pending_action"):
        clear_pending_action(user_id)
        return True
    return False

def process_timetable_async(user_id: str, text: str, minutes_before: int):
    # 1) OpenAIで解析->失敗時は簡易パーサー
    schedule_data = None
    err = None
    try:
        schedule_data = call_chatgpt_to_extract_schedule(
            text, model=DEFAULT_OPENAI_MODEL, timezone="Asia/Tokyo"
        )
    except Exception as e:
        err = e
    if not schedule_data or not schedule_data.get("schedule"):
        schedule_data = parse_timetable_locally(text, timezone="Asia/Tokyo")
        if not schedule_data.get("schedule"):
            line_push_text(user_id, f"時間割の解析に失敗しました。{err or ''}".strip())
            return
        else:
            line_push_text(user_id, "OpenAIの解析結果から授業を取得できなかったため、簡易解析で登録します。")
    print(f"user_id={user_id}")

    # 2) スケジュール登録
    schedule_jobs_for_user(user_id, schedule_data, minutes_before=minutes_before)
    # 3) 状態保存
    state = ensure_user_state(user_id)
    state["minutes_before"] = minutes_before
    state["schedule"] = ensure_schedule_dict(schedule_data)
    state["updated_at"] = datetime.now()
    persist_user_state(user_id)
    # 4) 完了通知 + 内訳
    summarize_and_push(user_id, minutes_before, schedule_data, "時間割の登録が完了しました。内訳は以下です。誤りがあれば「追加」「削除」「置換」で修正できます。")

def process_image_timetable_async(user_id: str, message_id: str, minutes_before: int):

    try:
        image_bytes, mime_type = download_line_content(message_id)
        print(f"[image] user={user_id} bytes={len(image_bytes)} mime={mime_type}") # Debug log
    except Exception as e:
        line_push_text(user_id, f"画像のダウンロードに失敗しました: {e}")
        print(f"[image] download failed: {e}")
        return

    schedule_data = None
    err = None
    try:
        schedule_data = call_chatgpt_to_extract_schedule_from_image(
            image_bytes,
            mime_type=mime_type,
            model=DEFAULT_VISION_MODEL,
            timezone="Asia/Tokyo",
        )
    except Exception as e:
        err = e

    if not schedule_data or not schedule_data.get("schedule"):
        detail = str(err).strip() if err else "OpenAIの応答から授業が見つかりませんでした。"
        message = f"画像から時間割を抽出できませんでした。{detail}".strip()
        line_push_text(user_id, message)
        try:
            raw_preview = json.dumps(schedule_data, ensure_ascii=False)[:500] if schedule_data else "None"
        except Exception:
            raw_preview = str(schedule_data)
        print(f"[image-error] user={user_id} detail={detail} raw={raw_preview}")
        return

    schedule_jobs_for_user(user_id, schedule_data, minutes_before=minutes_before)
    state = ensure_user_state(user_id)
    state["minutes_before"] = minutes_before
    state["schedule"] = ensure_schedule_dict(schedule_data)
    state["updated_at"] = datetime.now()
    persist_user_state(user_id)
    summarize_and_push(user_id, minutes_before, schedule_data, "画像から時間割を登録しました。内訳は以下です。")


def prepare_add_action(user_id: str, body: str):
    state = ensure_user_state(user_id)
    timezone = state["schedule"].get("timezone", "Asia/Tokyo")
    try:
        parsed = parse_entries_with_gpt(body, timezone=timezone)
    except Exception as e:
        line_push_text(user_id, f"追加内容の解析に失敗しました: {e}")
        return
    entries = parsed.get("schedule", [])
    if not entries:
        line_push_text(user_id, "追加できる授業が見つかりませんでした。")
        return
    set_pending_action(user_id, {"type": "add", "entries": entries, "timezone": parsed.get("timezone", timezone)})
    summary = format_entry_lines(entries)
    send_confirmation_prompt(user_id, "追加予定の授業", summary)


def prepare_delete_action(user_id: str, body: str):
    state = ensure_user_state(user_id)
    timezone = state["schedule"].get("timezone", "Asia/Tokyo")
    try:
        parsed = parse_entries_with_gpt(body, timezone=timezone)
    except Exception as e:
        line_push_text(user_id, f"削除内容の解析に失敗しました: {e}")
        return
    targets = parsed.get("schedule", [])
    if not targets:
        line_push_text(user_id, "削除対象の授業が見つかりませんでした。")
        return
    set_pending_action(user_id, {"type": "delete", "targets": targets})
    summary = format_entry_lines(targets)
    send_confirmation_prompt(user_id, "削除予定の授業", summary)


def prepare_replace_action(user_id: str, body: str):
    state = ensure_user_state(user_id)
    timezone = state["schedule"].get("timezone", "Asia/Tokyo")
    try:
        parsed = call_chatgpt_to_extract_replacement(body, model=DEFAULT_OPENAI_MODEL, timezone=timezone)
    except Exception as e:
        line_push_text(user_id, f"置換内容の解析に失敗しました: {e}")
        return
    old_entry = parsed.get("old")
    new_entry = parsed.get("new")
    if not old_entry or not new_entry:
        line_push_text(user_id, "変更前／変更後の情報が不足しています。")
        return
    action = {"type": "replace", "old": old_entry, "new": new_entry}
    set_pending_action(user_id, action)
    summary = (
        "変更前:\n"
        f"{format_entry_lines([old_entry])}\n"
        "変更後:\n"
        f"{format_entry_lines([new_entry])}"
    )
    send_confirmation_prompt(user_id, "置換予定の授業", summary)


def prepare_location_action(user_id: str, body: str):
    state = ensure_user_state(user_id)
    timezone = state["schedule"].get("timezone", "Asia/Tokyo")
    try:
        parsed = call_chatgpt_to_extract_location_updates(body, model=DEFAULT_OPENAI_MODEL, timezone=timezone)
    except Exception as e:
        line_push_text(user_id, f"教室更新内容の解析に失敗しました: {e}")
        return
    updates = parsed.get("updates") or []
    clean_updates = [u for u in updates if u.get("location")]
    if not clean_updates:
        line_push_text(user_id, "教室を設定できる情報が見つかりませんでした。教室名まで含めて記述してください。")
        return
    action = {"type": "location", "entries": clean_updates}
    set_pending_action(user_id, action)
    summary = format_entry_lines(clean_updates)
    send_confirmation_prompt(user_id, "教室を更新する授業", summary + "\n上記の授業を記載された教室に変更します。")


COMMAND_PREPARE = {
    "add": prepare_add_action,
    "delete": prepare_delete_action,
    "replace": prepare_replace_action,
    "location": prepare_location_action,
}


def _scheduler_loop():
    while True:
        try:
            schedule.run_pending()
        except Exception as e:
            print(f"scheduleエラー: {e}")
        time.sleep(1)

from fastapi import BackgroundTasks
app = FastAPI()

@app.on_event("startup")
def _on_startup():
    init_storage()
    th = threading.Thread(target=_scheduler_loop, daemon=True)
    th.start()
    print("Scheduler thread started.")

@app.get("/", response_class=PlainTextResponse)
def health():
    return "ok"

@app.get("/callback", response_class=PlainTextResponse)
def callback_get():
    return "ok"

@app.post("/callback")
async def callback(request: Request, background_tasks: BackgroundTasks):
    body = await request.body()
    signature = request.headers.get("x-line-signature")
    if not verify_signature(body, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    data = await request.json()
    events = data.get("events", [])
    for ev in events:
        print(ev) # デバッグ用ログ
        etype = ev.get("type") # イベントタイプ
        reply_token = ev.get("replyToken")
        source = ev.get("source", {})
        user_id = source.get("userId")

        if etype == "message" and user_id:
            message = ev.get("message", {})
            msg_type = message.get("type")

            if msg_type == "text":
                text = message.get("text", "").strip()
                state = ensure_user_state(user_id)
                pending_action = state.get("pending_action")
                pending_command = state.get("pending_command")
                normalized = text.lower()

                if pending_action:
                    if normalized in CONFIRM_POSITIVE_NORMALIZED:
                        line_reply_text(reply_token, ["保留中の操作を実行します。"])
                        execute_pending_action(user_id)
                    elif normalized in CONFIRM_NEGATIVE_NORMALIZED:
                        if cancel_pending_action(user_id):
                            line_reply_text(reply_token, ["操作をキャンセルしました。"])
                        else:
                            line_reply_text(reply_token, ["キャンセルできる操作はありません。"])
                    else:
                        line_reply_text(reply_token, ["保留中の操作があります。「はい」または「いいえ」で回答してください。"])
                    continue

                if pending_command:
                    if normalized in CONFIRM_NEGATIVE_NORMALIZED:
                        clear_pending_command(user_id)
                        line_reply_text(reply_token, ["操作をキャンセルしました。"])
                        continue
                    handler = COMMAND_PREPARE.get(pending_command)
                    clear_pending_command(user_id)
                    if not handler:
                        line_reply_text(reply_token, ["内部状態の処理に失敗しました。再度コマンドを送信してください。"])
                        continue
                    line_reply_text(reply_token, ["内容を解析しています。確認メッセージをお待ちください。"])
                    background_tasks.add_task(handler, user_id, text)
                    continue

                if text in HELP_COMMANDS:
                    line_reply_text(reply_token, [HELP_TEXT])
                    continue

                if text in VIEW_COMMANDS:
                    if not state.get("schedule") or not state["schedule"].get("schedule"):
                        line_reply_text(reply_token, ["まだ時間割が登録されていません。先に時間割を送信してください。"])
                    else:
                        minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)
                        summary = summarize_schedule(state["schedule"])
                        line_reply_text(reply_token, [f"現在登録されている時間割です。\n通知: {minutes_before}分前\n{summary}"])
                    continue

                if text in RESET_COMMANDS:
                    if not state.get("schedule") or not state["schedule"].get("schedule"):
                        line_reply_text(reply_token, ["まだ時間割が登録されていません。"])
                    else:
                        reset_user_schedule(user_id)
                        line_reply_text(reply_token, ["登録済みの時間割をリセットしました。新しい時間割を送信してください。"])
                    continue

                notify_payload = extract_command_payload(text, "通知")
                if notify_payload is not None:
                    if not notify_payload:
                        line_reply_text(reply_token, ["通知 10 のように分数を続けて入力してください。"])
                        continue
                    try:
                        minutes_before = int(notify_payload.strip())
                        state["minutes_before"] = minutes_before
                        persist_user_state(user_id)
                        sched = state.get("schedule") or {}
                        if sched.get("schedule"):
                            schedule_jobs_for_user(user_id, sched, minutes_before=minutes_before)
                        line_reply_text(reply_token, [f"通知タイミングを {minutes_before} 分前に設定しました。"])
                    except Exception:
                        line_reply_text(reply_token, ["通知 10 の形式で分数を指定してください。"])
                    continue

                if text in LOCATION_COMMANDS:
                    request_command_details(reply_token, user_id, "location")
                    continue

                location_payload = None
                for keyword in LOCATION_COMMANDS:
                    payload = extract_command_payload(text, keyword)
                    if payload is not None:
                        location_payload = payload
                        break
                if location_payload is not None:
                    detail = location_payload.strip()
                    if not detail:
                        request_command_details(reply_token, user_id, "location")
                        continue
                    line_reply_text(reply_token, ["内容を解析しています。確認メッセージをお待ちください。"])
                    background_tasks.add_task(prepare_location_action, user_id, detail)
                    continue

                if text in ADD_COMMANDS:
                    request_command_details(reply_token, user_id, "add")
                    continue

                add_payload = None
                for keyword in ADD_COMMANDS:
                    payload = extract_command_payload(text, keyword)
                    if payload is not None:
                        add_payload = payload
                        break
                if add_payload is not None:
                    detail = add_payload.strip()
                    if not detail:
                        request_command_details(reply_token, user_id, "add")
                        continue
                    line_reply_text(reply_token, ["内容を解析しています。確認メッセージをお待ちください。"])
                    background_tasks.add_task(prepare_add_action, user_id, detail)
                    continue

                if text in DELETE_COMMANDS:
                    request_command_details(reply_token, user_id, "delete")
                    continue

                delete_payload = None
                for keyword in DELETE_COMMANDS:
                    payload = extract_command_payload(text, keyword)
                    if payload is not None:
                        delete_payload = payload
                        break
                if delete_payload is not None:
                    detail = delete_payload.strip()
                    if not detail:
                        request_command_details(reply_token, user_id, "delete")
                        continue
                    line_reply_text(reply_token, ["内容を解析しています。確認メッセージをお待ちください。"])
                    background_tasks.add_task(prepare_delete_action, user_id, detail)
                    continue

                if text in REPLACE_COMMANDS:
                    request_command_details(reply_token, user_id, "replace")
                    continue

                replace_payload = None
                for keyword in REPLACE_COMMANDS:
                    payload = extract_command_payload(text, keyword)
                    if payload is not None:
                        replace_payload = payload
                        break
                if replace_payload is not None:
                    detail = replace_payload.strip()
                    if not detail:
                        request_command_details(reply_token, user_id, "replace")
                        continue
                    line_reply_text(reply_token, ["内容を解析しています。確認メッセージをお待ちください。"])
                    background_tasks.add_task(prepare_replace_action, user_id, detail)
                    continue

                minutes_before = ensure_user_state(user_id).get("minutes_before", DEFAULT_MINUTES_BEFORE)
                line_reply_text(reply_token, ["入力を受けつけました。しばらくお待ちください。"])
                background_tasks.add_task(process_timetable_async, user_id, text, minutes_before)
                continue

            if msg_type == "image":
                message_id = message.get("id")
                if not message_id:
                    line_reply_text(reply_token, ["画像IDを取得できませんでした。"])
                    continue
                minutes_before = ensure_user_state(user_id).get("minutes_before", DEFAULT_MINUTES_BEFORE)
                line_reply_text(reply_token, ["画像を受け付けました。解析後に時間割をお送りします。"])
                background_tasks.add_task(process_image_timetable_async, user_id, message_id, minutes_before)
                continue
        elif etype == "follow" and user_id:
            try:
                line_push_text(user_id, "友だち追加ありがとうございます。時間割のテキストまたは画像を送ってください。時間割のテキストはかなり適当でも拾うようになっているので適当で大丈夫です。例: 月1限 解析学 / 火2限 英語演習")
                line_push_text(user_id, HELP_TEXT)
            except Exception as e:
                print(f"初回メッセージ送信失敗: {e}")

    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Timetable bot management utilities")
    parser.add_argument("--broadcast", type=str, help="全ユーザーへ送信するメッセージ")
    parser.add_argument("--list-users", action="store_true", help="登録済みユーザーIDを一覧表示")
    parser.add_argument("--no-delay", action="store_true", help="ブロードキャスト時のインターバルを省略")
    args = parser.parse_args()

    if args.list_users or args.broadcast:
        init_storage()

    if args.list_users:
        ids = get_all_user_ids(force_db=True)
        if not ids:
            print("ユーザーが登録されていません。")
        else:
            print("登録ユーザーID:")
            for uid in ids:
                print(f" - {uid}")

    if args.broadcast:
        delay = 0.0 if args.no_delay else 0.1
        broadcast_message(args.broadcast, delay=delay, reload_state=False)

    if not args.list_users and not args.broadcast:
        parser.print_help()
