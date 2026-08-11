#!/usr/bin/env python3
"""
CI smoke test for actual API:
    launch frozen backend
    run word
    run char predictions
"""
import asyncio, json, sys, time, uuid, urllib.request

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = "http://127.0.0.1:8000"
STARTUP_TIMEOUT = 180
TASK_TIMEOUT = 1500

def wait_for_backend():
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/models", timeout=5) as r:
                models = json.load(r)
                print("Backend up. /models:", json.dumps(models, ensure_ascii=False))
                return models
        except Exception:
            time.sleep(3)
    sys.exit("FAIL: backend never answered /models")

async def run_task(msg_type, model_name, text):
    import websockets
    task_id = str(uuid.uuid4())
    async with websockets.connect(f"ws://127.0.0.1:8000/ws/ci-{uuid.uuid4()}") as ws:
        await ws.send(json.dumps({
            "type": msg_type, "task_id": task_id,
            "request_data": {"text": text, "model_name": model_name, "text_type": "prose"},
        }))
        deadline = time.time() + TASK_TIMEOUT
        while time.time() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
            m = json.loads(raw)
            mt = m.get("type")
            if mt == "progress":
                print(f"  [{msg_type}] {m.get('percentage')}% {m.get('message')}")
            elif mt == "error":
                sys.exit(f"FAIL [{msg_type}/{model_name}]: {m.get('detail')}")
            elif mt not in ("ack", "progress", "cancelled"):
                payload = m.get("result") or m.get("predictions") or m
                print(f"PASS [{msg_type}/{model_name}] terminal message type={mt}")
                print(json.dumps(payload, ensure_ascii=False)[:500])
                return
        sys.exit(f"FAIL [{msg_type}/{model_name}]: timed out")

def pick(models, type_):
    for m in models:
        if m["type"] == type_ and m["lang"] == "grc":
            return m["name"]
    sys.exit(f"FAIL: no model of type {type_} in /models")

if __name__ == "__main__":
    models = wait_for_backend()
    word_model = pick(models, "bert")
    char_model = pick(models, "tiresias")   # excludes the canine "Character model V1" by construction
    asyncio.run(run_task("start_word_prediction", word_model,
                         "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην ἣ - Ἀχαιοῖς ἄλγε ἔθηκεν"))
    asyncio.run(run_task("start_char_prediction", char_model,
                         "μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλ-ος"))
    print("SMOKE TEST PASSED")