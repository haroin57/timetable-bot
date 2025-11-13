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

VIEW_COMMANDS = {"一覧", "確認", "時間割", "schedule", "list"}

HELP_COMMANDS = {"help", "ヘルプ", "使い方"}

RESET_COMMANDS = {"リセット", "reset", "クリア", "clear"}



HELP_TEXT = (

    "利用できる主なコマンド:\n"

    "・通知 10 … 通知タイミングを分単位で設定\n"

    "・追加 月 1限 解析学 … 授業を追加\n"

    "・削除 火 10:40-12:10 … 授業を削除\n"

    "・置換 (1行目:変更前, 2行目以降:変更後)\n"

    "・リセット … 登録済みの時間割をすべて削除\n"

    "・一覧 … 現在登録済みの時間割を表示\n"

    "そのほか時間割テキストを送信すると解析して登録します。"

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
    first = text.find("{")
    last = text.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("JSONっぽい部分を検出できませんでした。")
    return text[first:last+1]

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

def call_chatgpt_to_extract_schedule(timetable_text: str, model="gpt-4o-mini", timezone="Asia/Tokyo"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")
    if OpenAI is None:
        raise RuntimeError("openai ライブラリが見つかりません。pip install openai を実行してください。")

    period_hint = "\n".join([f"{k}限: {v['start']}–{v['end']}" for k, v in DEFAULT_PERIOD_MAP.items()])

    system = (
        "You are a strict JSON extractor. "
        "Extract university timetable (possibly Japanese) into normalized JSON. "
        "Use only the schema and output JSON only with no extra text."
    )
    user = f"""
以下は大学の時間割テキストです。日本語の「n限」「月〜金」などが含まれます。
目的: 各授業の「曜日」「開始時刻」「終了時刻」「科目名」「教室」を抽出し、統一フォーマットのJSONで出力してください。

制約:
- 出力はJSONのみ。前後に説明文やコードブロックは不要。
- 時刻は24時間表記 "HH:MM"。
- 曜日は "Mon","Tue","Wed","Thu","Fri","Sat","Sun" のいずれか。
- locationはテキスト末尾などに記載された教室表記（例: "C3 100", "W2-101", "101教室"）をそのまま入れる。見つからなければ null。
- 日本語の「n限」は、以下のデフォルト対応を使って開始/終了を決めてください（本文中に別の時間が明記されていればそちらを優先）:
{period_hint}

スキーマ:
{{
  "timezone": "{timezone}",
  "schedule": [
    {{"day": "Mon", "start": "09:00", "end": "10:30", "course": "科目名", "location": "C3 100"}}
  ]
}}

時間割テキスト:
---
{timetable_text}
---
""".strip()

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    raw_json = extract_json_from_text(content)
    data = json.loads(raw_json)
    if "schedule" not in data or not isinstance(data["schedule"], list):
        raise ValueError("JSONにschedule配列がありません。")
    for entry in data["schedule"]:
        entry.setdefault("location", None)
    return data





# ====== LINE API / FastAPI Webhook (Unified with async + corrections) ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

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

# OpenAIが使えない場合の簡易パーサー（最低限の対応）
DAY_JA_TO_EN = {"月":"Mon","火":"Tue","水":"Wed","木":"Thu","金":"Fri","土":"Sat","日":"Sun"}
def parse_timetable_locally(timetable_text: str, timezone="Asia/Tokyo"):
    schedule = []
    tokens = re.split(r"[\n/／、]+", timetable_text)
    for tok in tokens:
        s = tok.strip()
        if not s:
            continue
        m = re.match(r"^(月|火|水|木|金|土|日)[曜]?:?\s*(\d)限\s+(.+)$", s)
        if m:
            dja, period, course = m.groups()
            en = DAY_JA_TO_EN.get(dja)
            p = DEFAULT_PERIOD_MAP.get(period)
            if en and p:
                course_name, location = split_course_and_location(course)
                schedule.append({"day": en, "start": p["start"], "end": p["end"], "course": course_name.strip(), "location": location})
            continue
        m = re.match(r"^(月|火|水|木|金|土|日)[曜]?:?\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", s)
        if m:
            dja, start, end, course = m.groups()
            en = DAY_JA_TO_EN.get(dja)
            if en:
                start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
                end   = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
                course_name, location = split_course_and_location(course)
                schedule.append({"day": en, "start": start, "end": end, "course": course_name.strip(), "location": location})
            continue
        m = re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", s, re.I)
        if m:
            day, start, end, course = m.groups()
            day = day[:1].upper() + day[1:3].lower()
            start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
            end   = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
            course_name, location = split_course_and_location(course)
            schedule.append({"day": day, "start": start, "end": end, "course": course_name.strip(), "location": location})
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

def delete_by_filter(base: dict, filt_text: str) -> dict:
    filt = filt_text.strip()
    def match(e):
        # day+period or day+start or course substring
        if "限" in filt:
            m = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun|月|火|水|木|金|土|日).{0,2}(\d)限", filt, re.I)
            if m:
                d, p = m.groups()
                d = DAY_JA_TO_EN.get(d, d.title()[:3])
                want = DEFAULT_PERIOD_MAP.get(p)
                if want:
                    return e["day"] == d and e["start"] == want["start"]
        m = re.search(r"(Mon|Tue|Wed|Thu|Fri|Sat|Sun|月|火|水|木|金|土|日).{0,2}([0-2]?\d:[0-5]\d)", filt, re.I)
        if m:
            d, hm = m.groups()
            d = DAY_JA_TO_EN.get(d, d.title()[:3])
            hm = f"{int(hm.split(':')[0]):02d}:{hm.split(':')[1]}"
            return e["day"] == d and e["start"] == hm
        if "教室" in filt and e.get("location"):
            return filt in e["location"]
        return filt in e["course"] or (e.get("location") and filt in e["location"])
    new = [e for e in base.get("schedule", []) if not match(e)]
    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new}

def summarize_and_push(user_id: str, minutes_before: int, schedule_data: dict, prefix: str):
    msg = f"{prefix}\n通知: {minutes_before}分前\n" + summarize_schedule(schedule_data)
    line_push_text(user_id, msg)

def process_timetable_async(user_id: str, text: str, minutes_before: int):
    # 1) OpenAIで解析->失敗時は簡易パーサー
    schedule_data = None
    err = None
    try:
        schedule_data = call_chatgpt_to_extract_schedule(
            text, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), timezone="Asia/Tokyo"
        )
    except Exception as e:
        err = e
    if not schedule_data or not schedule_data.get("schedule"):
        schedule_data = parse_timetable_locally(text, timezone="Asia/Tokyo")
        if not schedule_data.get("schedule"):
            line_push_text(user_id, f"時間割の解析に失敗しました。{err or ''}".strip())
            return
        else:
            line_push_text(user_id, "OpenAIが利用できないため、簡易解析で登録します。")
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

def process_correction_async(user_id: str, text: str):
    state = USER_STATE.get(user_id)
    if not state or not state.get("schedule"):
        line_push_text(user_id, "まだ時間割が登録されていません。まず時間割全体を送ってください。")
        return
    current = state["schedule"]
    minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)

    t = text.strip()
    try:
        add_body = extract_command_payload(t, "追加")
        if add_body is not None:
            body = add_body.strip()
            if not body:
                line_push_text(user_id, "追加 月 10:40-12:10 英語 のように入力してください。")
                return
            adds = parse_timetable_locally(body, timezone=current.get("timezone", "Asia/Tokyo"))
            if not adds.get("schedule"):
                try:
                    adds = call_chatgpt_to_extract_schedule(
                        body,
                        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                        timezone=current.get("timezone", "Asia/Tokyo"),
                    )
                except Exception:
                    pass
            if not adds.get("schedule"):
                line_push_text(user_id, "追加の解析に失敗しました。例: 追加 火 10:40-12:10 英語")
                return
            new_sched = merge_additions(current, adds)
        elif (delete_body := extract_command_payload(t, "削除")) is not None:
            body = delete_body.strip()
            if not body:
                line_push_text(user_id, "削除 月 1限 解析学 のように入力してください。")
                return
            new_sched = delete_by_filter(current, body)
        else:
            replace_body = extract_command_payload(t, "置換")
            if replace_body is None:
                replace_body = extract_command_payload(t, "修正")
            if replace_body is None:
                line_push_text(user_id, "修正コマンドが認識できません。次のいずれかを使ってください: 追加 / 削除 / 置換")
                return
            body = replace_body.strip()
            if not body:
                line_push_text(user_id, "置換 コマンドは 1行目=変更前, 2行目以降=変更後 の形式で入力してください。")
                return
            parsed = parse_replacement_parts(body)
            if not parsed:
                line_push_text(user_id, "置換 コマンドは 1行目=変更前, 2行目以降=変更後 です。例:\n置換\n水 13:00-14:30 物理\n木 14:40-16:10 物理")
                return
            old, new = parsed
            old_parsed = parse_timetable_locally(old, timezone=current.get("timezone", "Asia/Tokyo"))
            new_parsed = parse_timetable_locally(new, timezone=current.get("timezone", "Asia/Tokyo"))
            if not old_parsed.get("schedule") or not new_parsed.get("schedule"):
                line_push_text(user_id, "置換の解析に失敗しました。例:\n置換\n水 13:00-14:30 物理\n木 14:40-16:10 物理")
                return
            o = old_parsed["schedule"][0]
            n = new_parsed["schedule"][0]
            replaced = False
            new_list = []
            for e in current.get("schedule", []):
                if not replaced and ((e["day"] == o["day"] and e["start"] == o["start"]) or (o["course"] in e["course"])):
                    new_list.append({"day": n["day"], "start": n["start"], "end": n.get("end"), "course": n["course"]})
                    replaced = True
                else:
                    new_list.append(e)
            if not replaced:
                line_push_text(user_id, "対象が見つからず置換できませんでした。")
                return
            new_sched = {"timezone": current.get("timezone", "Asia/Tokyo"), "schedule": new_list}

        schedule_jobs_for_user(user_id, new_sched, minutes_before=minutes_before)
        state = ensure_user_state(user_id)
        state["minutes_before"] = minutes_before
        state["schedule"] = ensure_schedule_dict(new_sched)
        state["updated_at"] = datetime.now()
        persist_user_state(user_id)
        summarize_and_push(user_id, minutes_before, new_sched, "修正を反映しました。最新の時間割は以下です。")
    except Exception as e:
        line_push_text(user_id, f"修正処理でエラーが発生しました: {e}")

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

        if etype == "message" and ev.get("message", {}).get("type") == "text" and user_id:
            text = ev["message"]["text"].strip()

            if text in HELP_COMMANDS:
                line_reply_text(reply_token, [HELP_TEXT])
                continue

            if text in VIEW_COMMANDS:
                state = USER_STATE.get(user_id)
                if not state or not state.get("schedule") or not state["schedule"].get("schedule"):
                    line_reply_text(reply_token, ["まだ時間割が登録されていません。先に時間割を送信してください。"])
                else:
                    minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)
                    summary = summarize_schedule(state["schedule"])
                    line_reply_text(reply_token, [f"現在登録されている時間割です。\n通知: {minutes_before}分前\n{summary}"])
                continue

            if text in RESET_COMMANDS:
                state = USER_STATE.get(user_id)
                if not state or not state.get("schedule") or not state["schedule"].get("schedule"):
                    line_reply_text(reply_token, ["まだ時間割が登録されていません。"])
                else:
                    reset_user_schedule(user_id)
                    line_reply_text(reply_token, ["登録済みの時間割をリセットしました。新しい時間割を送信してください。"])
                continue

            # 通知分設定
            notify_payload = extract_command_payload(text, "通知")
            if notify_payload is not None:
                if not notify_payload:
                    line_reply_text(reply_token, ["通知 10 のように分数を続けて入力してください。"])
                    continue
                try:
                    minutes_before = int(notify_payload.strip())
                    st = ensure_user_state(user_id)
                    st["minutes_before"] = minutes_before
                    persist_user_state(user_id)
                    line_reply_text(reply_token, [f"通知タイミングを {minutes_before} 分前に設定しました。時間割テキストを送ってください。"])
                except Exception:
                    line_reply_text(reply_token, ["通知 10 の形式で分数を指定してください。"])
                continue

            # 修正コマンド
            add_payload = extract_command_payload(text, "追加")
            if add_payload is not None:
                if not add_payload:
                    line_reply_text(reply_token, ["追加 月 10:40-12:10 英語 のように入力してください。"])
                    continue
                line_reply_text(reply_token, ["修正内容を受け付けました。反映後に最新の時間割をお送りします。"])
                background_tasks.add_task(process_correction_async, user_id, f"追加:{add_payload}")
                continue

            delete_payload = extract_command_payload(text, "削除")
            if delete_payload is not None:
                if not delete_payload:
                    line_reply_text(reply_token, ["削除 月 1限 解析学 のように入力してください。"])
                    continue
                line_reply_text(reply_token, ["修正内容を受け付けました。反映後に最新の時間割をお送りします。"])
                background_tasks.add_task(process_correction_async, user_id, f"削除:{delete_payload}")
                continue

            replace_payload = extract_command_payload(text, "置換")
            if replace_payload is None:
                replace_payload = extract_command_payload(text, "修正")
            if replace_payload is not None:
                if not replace_payload:
                    line_reply_text(reply_token, ["置換 コマンドは 1行目=変更前, 2行目以降=変更後 の形式で入力してください。"])
                    continue
                line_reply_text(reply_token, ["修正内容を受け付けました。反映後に最新の時間割をお送りします。"])
                background_tasks.add_task(process_correction_async, user_id, f"置換:{replace_payload}")
                continue

            # 新規登録: まず即時返信、処理は非同期
            minutes_before = ensure_user_state(user_id).get("minutes_before", DEFAULT_MINUTES_BEFORE)
            line_reply_text(reply_token, ["入力を受けつけました。しばらくお待ちください。"])
            background_tasks.add_task(process_timetable_async, user_id, text, minutes_before)

        elif etype == "follow" and user_id:
            try:
                line_push_text(user_id, "友だち追加ありがとうございます。時間割のテキストを送ってください。時間割のテキストはかなり適当でも拾うようになっているので適当で大丈夫です。例: 月:1限 解析学 / 火:2限 英語\n修正は 追加 / 削除 / 置換 で指定できます。")
            except Exception as e:
                print(f"初回メッセージ送信失敗: {e}")

    return {"status": "ok"}
