from __future__ import annotations

import base64
import hashlib
import hmac
import time
from dataclasses import asdict
from urllib.parse import urlencode
from uuid import uuid4

from aiohttp import web

from tg_guard_bot.config import Settings
from tg_guard_bot.store import BotStore


class SettingsWebApp:
    def __init__(
        self,
        store: BotStore,
        bot_token: str,
        public_url: str,
        host: str,
        port: int,
        settings: Settings,
    ) -> None:
        self.store = store
        self.bot_token = bot_token
        self.public_url = public_url.rstrip("/") + "/"
        self.host = host
        self.port = port
        self.settings = settings
        self.private_launches: dict[str, tuple[int, int, int]] = {}
        self.runner: web.AppRunner | None = None

    def create_private_launch(self, chat_id: int, user_id: int) -> str:
        code = uuid4().hex[:24]
        self.private_launches[code] = (chat_id, user_id, int(time.time()) + 15 * 60)
        return code

    def consume_private_launch(self, code: str, user_id: int) -> int | None:
        launch = self.private_launches.pop(code, None)
        if not launch:
            return None
        chat_id, expected_user_id, expires_at = launch
        if expected_user_id != user_id or expires_at < int(time.time()):
            return None
        return chat_id

    def launch_url(self, chat_id: int, user_id: int) -> str:
        expires_at = int(time.time()) + 24 * 60 * 60
        payload = f"{chat_id}:{user_id}:{expires_at}"
        signature = hmac.new(
            self.bot_token.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        token = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
        return f"{self.public_url}?{urlencode({'token': f'{token}.{signature}'})}"

    async def start(self) -> None:
        app = web.Application(client_max_size=32 * 1024)
        app.router.add_get("/", self.page)
        app.router.add_get("/api/settings", self.get_settings)
        app.router.add_post("/api/settings", self.update_settings)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, self.host, self.port).start()

    async def stop(self) -> None:
        if self.runner:
            await self.runner.cleanup()
            self.runner = None

    async def page(self, request: web.Request) -> web.Response:
        return web.Response(text=SETTINGS_HTML, content_type="text/html")

    async def get_settings(self, request: web.Request) -> web.Response:
        chat_id, _ = self._verify_token(request.query.get("token", ""))
        runtime = self.store.settings_for(chat_id)
        return web.json_response(public_settings(runtime, self.settings))

    async def update_settings(self, request: web.Request) -> web.Response:
        payload = await request.json()
        chat_id, _ = self._verify_token(str(payload.get("token", "")))
        values = payload.get("settings")
        if not isinstance(values, dict):
            raise web.HTTPBadRequest(text="settings must be an object")

        numeric = {
            "ask_context": (3, 50),
            "moderation_context": (3, 50),
            "ask_web_results": (1, 8),
            "silent_hours": (1, 720),
        }
        toggles = {
            "anti_bore",
            "creative_interjections",
        }
        text_settings = {
            "web_mode": "ask_web_mode",
            "ai_model": "ai_model",
            "moderation_model": "moderation_model",
            "image_model": "image_model",
            "video_model": "video_model",
            "transcription_model": "transcription_model",
            "tts_model": "tts_model",
        }
        for name, (minimum, maximum) in numeric.items():
            if name not in values:
                continue
            value = int(values[name])
            if not minimum <= value <= maximum:
                raise web.HTTPBadRequest(text=f"invalid {name}")
            self.store.update_setting(chat_id, name, value)
        for name in toggles:
            if name in values:
                self.store.update_setting(chat_id, name, int(bool(values[name])))
        for name, store_name in text_settings.items():
            if name not in values:
                continue
            value = str(values[name]).strip()
            if name == "web_mode" and not value:
                raise web.HTTPBadRequest(text=f"empty {name}")
            self.store.update_text_setting(chat_id, store_name, value)

        return web.json_response(public_settings(self.store.settings_for(chat_id), self.settings))

    def _verify_token(self, token: str) -> tuple[int, int]:
        try:
            encoded, signature = token.rsplit(".", 1)
            padding = "=" * (-len(encoded) % 4)
            payload = base64.urlsafe_b64decode(encoded + padding).decode()
            expected = hmac.new(
                self.bot_token.encode(),
                payload.encode(),
                hashlib.sha256,
            ).hexdigest()
            chat_id_text, user_id_text, expires_at_text = payload.split(":")
            if not hmac.compare_digest(signature, expected):
                raise ValueError
            if int(expires_at_text) < int(time.time()):
                raise ValueError
            return int(chat_id_text), int(user_id_text)
        except (ValueError, TypeError):
            raise web.HTTPUnauthorized(text="invalid or expired link") from None


def public_settings(runtime, defaults: Settings) -> dict[str, object]:
    data = asdict(runtime)
    transcription_model = data["transcription_model"]
    if not transcription_model or transcription_model == defaults.elevenlabs_stt_model_id:
        transcription_model = defaults.whisper_model_size
    return {
        "ask_context": data["ask_context_limit"],
        "moderation_context": data["moderation_context_limit"],
        "ask_web": data["ask_web_enabled"],
        "web_mode": "chatgpt" if data["ask_web_mode"] == "auto" else data["ask_web_mode"],
        "ask_web_results": data["ask_web_results"],
        "silent_hours": data["silent_support_hours"],
        "anti_bore": data["anti_bore_enabled"],
        "creative_interjections": data["creative_interjections_enabled"],
        "models": {
            "main": data["ai_model"] or defaults.openai_model,
            "moderation": data["moderation_model"] or defaults.openai_moderation_model,
            "image": data["image_model"] or defaults.openrouter_image_model,
            "video": data["video_model"] or defaults.openrouter_video_model,
            "transcription": transcription_model,
            "tts": data["tts_model"] or defaults.elevenlabs_model_id,
        },
    }


SETTINGS_HTML = """<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
  <title>Настройки Moder</title>
  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  <style>
    :root { color-scheme: light dark; font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--tg-theme-text-color,#18212b); background: var(--tg-theme-bg-color,#f4f6f8); }
    main { width: min(680px,100%); margin: 0 auto; padding: 20px 16px 40px; }
    h1 { margin: 0 0 4px; font-size: 26px; letter-spacing: 0; }
    .lead { margin: 0 0 22px; color: var(--tg-theme-hint-color,#6f7b87); }
    section { border-top: 1px solid var(--tg-theme-section-separator-color,#dfe4e8); padding: 18px 0; }
    h2 { margin: 0 0 14px; font-size: 15px; text-transform: uppercase; color: var(--tg-theme-hint-color,#6f7b87); }
    .row { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 16px; min-height: 54px; }
    .row + .row { border-top: 1px solid var(--tg-theme-section-separator-color,#e8ecef); }
    label { font-size: 16px; }
    small { display: block; margin-top: 3px; color: var(--tg-theme-hint-color,#6f7b87); }
    input[type=number], input[type=text], select { padding: 9px; border: 1px solid #b9c2ca; border-radius: 6px; font-size: 16px; background: var(--tg-theme-secondary-bg-color,#fff); color: inherit; }
    input[type=number] { width: 78px; text-align: center; }
    input.wide { width: min(310px,52vw); }
    input[type=checkbox] { width: 24px; height: 24px; accent-color: var(--tg-theme-button-color,#2481cc); }
    .model { padding: 8px 0; overflow-wrap: anywhere; }
    .model b { display: block; font-size: 14px; }
    .model code { color: var(--tg-theme-hint-color,#6f7b87); }
    button { width: 100%; border: 0; border-radius: 7px; padding: 13px 16px; font-size: 17px; font-weight: 650; color: var(--tg-theme-button-text-color,#fff); background: var(--tg-theme-button-color,#2481cc); cursor: pointer; }
    button:disabled { opacity: .55; }
    #status { min-height: 22px; margin: 12px 0 0; text-align: center; color: var(--tg-theme-hint-color,#6f7b87); }
  </style>
</head>
<body>
<main>
  <h1>Настройки братства</h1>
  <p class="lead">Изменения применяются сразу к этому чату.</p>
  <form id="form">
    <section>
      <h2>Контекст</h2>
      <div class="row"><label>Для /ask<small>Последние сообщения чата</small></label><input name="ask_context" type="number" min="3" max="50"></div>
      <div class="row"><label>Для модерации<small>Контекст каждой проверки</small></label><input name="moderation_context" type="number" min="3" max="50"></div>
    </section>
    <section>
      <h2>Интернет</h2>
      <div class="row"><label>Веб-поиск для /ask</label><select name="web_mode"><option value="chatgpt">ChatGPT Search</option><option value="local">Локальный поиск</option><option value="off">Выключен</option></select></div>
      <div class="row"><label>Результатов поиска</label><input name="ask_web_results" type="number" min="1" max="8"></div>
    </section>
    <section>
      <h2>Поведение</h2>
      <div class="row"><label>Поддержка молчащих<small>Через сколько часов</small></label><input name="silent_hours" type="number" min="1" max="720"></div>
      <div class="row"><label>Анти-душнила</label><input name="anti_bore" type="checkbox"></div>
      <div class="row"><label>Самостоятельно влезать в разговор</label><input name="creative_interjections" type="checkbox"></div>
    </section>
    <section>
      <h2>Модели</h2>
      <div class="row"><label>Большая модель</label><input class="wide" name="ai_model" type="text"></div>
      <div class="row"><label>Модель модерации</label><input class="wide" name="moderation_model" type="text"></div>
      <div class="row"><label>Модель картинок</label><input class="wide" name="image_model" type="text"></div>
      <div class="row"><label>Модель видео</label><input class="wide" name="video_model" type="text"></div>
      <div class="row"><label>Распознавание речи</label><input class="wide" name="transcription_model" type="text"></div>
      <div class="row"><label>Голос Арсена</label><input class="wide" name="tts_model" type="text"></div>
    </section>
    <button id="save" type="submit">Сохранить</button>
    <p id="status"></p>
  </form>
</main>
<script>
const token = new URLSearchParams(location.search).get("token");
const form = document.querySelector("#form");
const statusEl = document.querySelector("#status");
const save = document.querySelector("#save");
const tg = window.Telegram?.WebApp;
tg?.ready(); tg?.expand();
const field = name => form.elements.namedItem(name);
function fill(data) {
  for (const name of ["ask_context","moderation_context","ask_web_results","silent_hours"]) field(name).value = data[name];
  for (const name of ["anti_bore","creative_interjections"]) field(name).checked = data[name];
  field("web_mode").value = data.web_mode;
  field("ai_model").value = data.models.main;
  field("moderation_model").value = data.models.moderation;
  field("image_model").value = data.models.image;
  field("video_model").value = data.models.video;
  field("transcription_model").value = data.models.transcription;
  field("tts_model").value = data.models.tts;
}
async function load() {
  if (!token) throw new Error("Ссылка неполная. Открой панель снова через /settings.");
  const response = await fetch(`api/settings?token=${encodeURIComponent(token)}`);
  if (!response.ok) throw new Error("Ссылка устарела. Открой /settings ещё раз.");
  fill(await response.json());
}
form.addEventListener("submit", async event => {
  event.preventDefault(); save.disabled = true; statusEl.textContent = "Сохраняю...";
  const settings = {};
  for (const name of ["ask_context","moderation_context","ask_web_results","silent_hours"]) settings[name] = Number(field(name).value);
  for (const name of ["anti_bore","creative_interjections"]) settings[name] = field(name).checked;
  for (const name of ["web_mode","ai_model","moderation_model","image_model","video_model","transcription_model","tts_model"]) settings[name] = field(name).value;
  try {
    const response = await fetch("api/settings", {method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({token,settings})});
    if (!response.ok) throw new Error(await response.text());
    fill(await response.json()); statusEl.textContent = "Готово. Настройки применены.";
    tg?.HapticFeedback?.notificationOccurred("success");
  } catch (error) { statusEl.textContent = "Не сохранилось: " + error.message; }
  finally { save.disabled = false; }
});
load().catch(error => { statusEl.textContent = error.message; save.disabled = true; });
</script>
</body>
</html>
"""
