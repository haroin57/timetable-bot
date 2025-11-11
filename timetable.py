from dotenv import load_dotenv; load_dotenv()
import os
import hmac
import base64
import hashlib
import json
import threading
import time
from datetime import datetime, timedelta

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
目的: 各授業の「曜日」「開始時刻」「終了時刻」「科目名」を抽出し、統一フォーマットのJSONで出力してください。

制約:
- 出力はJSONのみ。前後に説明文やコードブロックは不要。
- 時刻は24時間表記 "HH:MM"。
- 曜日は "Mon","Tue","Wed","Thu","Fri","Sat","Sun" のいずれか。
- 日本語の「n限」は、以下のデフォルト対応を使って開始/終了を決めてください（本文中に別の時間が明記されていればそちらを優先）:
{period_hint}

スキーマ:
{{
  "timezone": "{timezone}",
  "schedule": [
    {{"day": "Mon", "start": "09:00", "end": "10:30", "course": "科目名"}}
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
    return data





# ====== LINE API / FastAPI Webhook (Unified with async + corrections) ======
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_REPLY_URL = "https://api.line.me/v2/bot/message/reply"

# ユーザーごとの状態保持（簡易版: メモリ）
USER_STATE = {}  # user_id -> {"minutes_before": int, "schedule": dict, "updated_at": datetime}

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

        getattr(schedule.every(), dow_attr).at(notify_hm).tag(f"user:{user_id}").do(
            line_push_text, to_user_id=user_id, text=msg
        )
        print(f"ジョブ登録: user={user_id} {dow_attr} {notify_hm} {course}")

def summarize_schedule(schedule_data: dict, limit: int = 20) -> str:
    rows = []
    for item in schedule_data.get("schedule", [])[:limit]:
        rows.append(f"{item['day']} {item['start']}-{item.get('end','')} {item['course']}")
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
                schedule.append({"day": en, "start": p["start"], "end": p["end"], "course": course.strip()})
            continue
        m = re.match(r"^(月|火|水|木|金|土|日)[曜]?:?\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", s)
        if m:
            dja, start, end, course = m.groups()
            en = DAY_JA_TO_EN.get(dja)
            if en:
                start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
                end   = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
                schedule.append({"day": en, "start": start, "end": end, "course": course.strip()})
            continue
        m = re.match(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*([0-2]?\d:[0-5]\d)\s*-\s*([0-2]?\d:[0-5]\d)\s+(.+)$", s, re.I)
        if m:
            day, start, end, course = m.groups()
            day = day[:1].upper() + day[1:3].lower()
            start = f"{int(start.split(':')[0]):02d}:{start.split(':')[1]}"
            end   = f"{int(end.split(':')[0]):02d}:{end.split(':')[1]}"
            schedule.append({"day": day, "start": start, "end": end, "course": course.strip()})
            continue
    return {"timezone": timezone, "schedule": schedule}

def merge_additions(base: dict, additions: dict) -> dict:
    keyset = {(e["day"], e["start"], e["course"]) for e in base.get("schedule", [])}
    out = list(base.get("schedule", []))
    for e in additions.get("schedule", []):
        k = (e["day"], e["start"], e["course"].strip())
        if k not in keyset:
            out.append({"day": e["day"], "start": e["start"], "end": e.get("end"), "course": e["course"].strip()})
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
        return filt in e["course"]
    new = [e for e in base.get("schedule", []) if not match(e)]
    return {"timezone": base.get("timezone", "Asia/Tokyo"), "schedule": new}

def summarize_and_push(user_id: str, minutes_before: int, schedule_data: dict, prefix: str):
    msg = f"{prefix}\n通知: {minutes_before}分前\n" + summarize_schedule(schedule_data)
    line_push_text(user_id, msg)

def process_timetable_async(user_id: str, text: str, minutes_before: int):
    # 1) OpenAIで解析→失敗時は簡易パーサー
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

    # 2) スケジュール登録
    schedule_jobs_for_user(user_id, schedule_data, minutes_before=minutes_before)
    # 3) 状態保存
    USER_STATE[user_id] = {"minutes_before": minutes_before, "schedule": schedule_data, "updated_at": datetime.now()}
    # 4) 完了通知 + 内訳
    summarize_and_push(user_id, minutes_before, schedule_data, "時間割の登録が完了しました。内訳は以下です。誤りがあれば「追加:」「削除:」「置換:」で修正できます。")

def process_correction_async(user_id: str, text: str):
    state = USER_STATE.get(user_id)
    if not state or not state.get("schedule"):
        line_push_text(user_id, "まだ時間割が登録されていません。まず時間割全体を送ってください。")
        return
    current = state["schedule"]
    minutes_before = state.get("minutes_before", DEFAULT_MINUTES_BEFORE)

    t = text.strip()
    try:
        if t.startswith("追加:"):
            body = t.split(":", 1)[1].strip()
            # まず簡易パーサー→取れなければOpenAI
            adds = parse_timetable_locally(body, timezone=current.get("timezone", "Asia/Tokyo"))
            if not adds.get("schedule"):
                try:
                    adds = call_chatgpt_to_extract_schedule(body, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                                                            timezone=current.get("timezone", "Asia/Tokyo"))
                except Exception:
                    pass
            if not adds.get("schedule"):
                line_push_text(user_id, "追加の解析に失敗しました。例: 追加: 火 10:40-12:10 英語")
                return
            new_sched = merge_additions(current, adds)
        elif t.startswith("削除:"):
            body = t.split(":", 1)[1].strip()
            new_sched = delete_by_filter(current, body)
        elif t.startswith("置換:") or t.startswith("修正:"):
            body = t.split(":", 1)[1].strip()
            parts = [p.strip() for p in re.split(r"->|→", body, maxsplit=1)]
            if len(parts) != 2:
                line_push_text(user_id, "置換の形式は 置換: <旧> -> <新> です。")
                return
            old, new = parts
            old_parsed = parse_timetable_locally(old, timezone=current.get("timezone", "Asia/Tokyo"))
            new_parsed = parse_timetable_locally(new, timezone=current.get("timezone", "Asia/Tokyo"))
            if not old_parsed.get("schedule") or not new_parsed.get("schedule"):
                line_push_text(user_id, "置換の解析に失敗しました。例: 置換: 水 13:00-14:30 物理 -> 木 14:40-16:10 物理")
                return
            o = old_parsed["schedule"][0]
            n = new_parsed["schedule"][0]
            # day+start一致 or course一致で最初の1件を差し替え
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
        else:
            line_push_text(user_id, "修正コマンドが認識できません。次のいずれかを使ってください: 追加:/削除:/置換:")
            return

        # 登録し直し + 状態更新
        schedule_jobs_for_user(user_id, new_sched, minutes_before=minutes_before)
        USER_STATE[user_id] = {"minutes_before": minutes_before, "schedule": new_sched, "updated_at": datetime.now()}
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
        etype = ev.get("type")
        reply_token = ev.get("replyToken")
        source = ev.get("source", {})
        user_id = source.get("userId")

        if etype == "message" and ev.get("message", {}).get("type") == "text" and user_id:
            text = ev["message"]["text"].strip()

            # 通知分設定
            if text.startswith("通知:"):
                try:
                    minutes_before = int(text.split(":", 1)[1].strip())
                    st = USER_STATE.get(user_id, {})
                    st["minutes_before"] = minutes_before
                    st.setdefault("schedule", {})
                    USER_STATE[user_id] = st
                    line_reply_text(reply_token, [f"通知タイミングを {minutes_before} 分前に設定しました。時間割テキストを送ってください。"])
                except Exception:
                    line_reply_text(reply_token, ["通知:10 の形式で分数を指定してください。"])
                continue

            # 修正コマンド
            if text.startswith(("追加:", "削除:", "置換:", "修正:")):
                line_reply_text(reply_token, ["修正内容を受け付けました。反映後に最新の時間割をお送りします。"])
                background_tasks.add_task(process_correction_async, user_id, text)
                continue

            # 新規登録: まず即時返信、処理は非同期
            minutes_before = USER_STATE.get(user_id, {}).get("minutes_before", DEFAULT_MINUTES_BEFORE)
            line_reply_text(reply_token, ["入力を受けつけました。しばらくお待ちください。"])
            background_tasks.add_task(process_timetable_async, user_id, text, minutes_before)

        elif etype == "follow" and user_id:
            try:
                line_push_text(user_id, "友だち追加ありがとうございます。時間割のテキストを送ってください。例: 月:1限 解析学 / 火:2限 英語\n修正は 追加:/削除:/置換: で指定できます。")
            except Exception as e:
                print(f"初回メッセージ送信失敗: {e}")

    return {"status": "ok"}