from __future__ import annotations

from pathlib import Path
from secrets import compare_digest

import uvicorn
import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates


def load_config() -> dict[str, str]:
        config_path = Path(__file__).with_name("config.yaml")
        with config_path.open(encoding="utf-8") as config_file:
                config = yaml.safe_load(config_file)

        if not isinstance(config, dict) or not all(
                isinstance(config.get(key), str) and config[key]
                for key in ("password", "session_secret")
        ):
                raise RuntimeError("config.yaml must define non-empty password and session_secret values")

        return config


config = load_config()
app = FastAPI(title="Labeling Tools API")
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


def is_authenticated(request: Request) -> bool:
        return request.session.get("authenticated") is True


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, response_model=None)
def home(request: Request) -> HTMLResponse | RedirectResponse:
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)

        return templates.TemplateResponse(request=request, name="home.html")


@app.get("/login", response_class=HTMLResponse, response_model=None)
def login(request: Request, error: bool = False) -> HTMLResponse | RedirectResponse:
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"error": error},
    )


@app.post("/login")
def authenticate(request: Request, password: str = Form()) -> RedirectResponse:
    if compare_digest(password, config["password"]):
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=303)
    return RedirectResponse(url="/login?error=1", status_code=303)


@app.post("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
