from __future__ import annotations

from pathlib import Path
from secrets import compare_digest

import uvicorn
import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware


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


def is_authenticated(request: Request) -> bool:
        return request.session.get("authenticated") is True


def login_page(error: bool = False) -> HTMLResponse:
        error_message = "<p class=\"error\">Incorrect password.</p>" if error else ""
        return HTMLResponse(
                f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sign in | Labeling Tools</title>
    <style>
        body {{ align-items: center; background: #f1f5f2; color: #17261d; display: flex; font-family: Georgia, serif; justify-content: center; margin: 0; min-height: 100vh; }}
        main {{ background: #fffdf8; border: 1px solid #cbd7cc; border-radius: 8px; box-shadow: 0 12px 30px #17261d1c; box-sizing: border-box; padding: 2.5rem; width: min(100% - 2rem, 25rem); }}
        h1 {{ font-size: 2rem; margin: 0 0 .5rem; }}
        p {{ color: #506056; line-height: 1.5; }}
        label {{ display: block; font-weight: bold; margin: 1.5rem 0 .5rem; }}
        input, button {{ box-sizing: border-box; font: inherit; width: 100%; }}
        input {{ border: 1px solid #879b89; border-radius: 4px; padding: .7rem; }}
        button {{ background: #1e6245; border: 0; border-radius: 4px; color: white; cursor: pointer; margin-top: 1rem; padding: .75rem; }}
        .error {{ color: #a12e20; font-weight: bold; }}
    </style>
</head>
<body>
    <main>
        <h1>Labeling Tools</h1>
        <p>Enter the workspace password to continue.</p>
        {error_message}
        <form action="/login" method="post">
            <label for="password">Password</label>
            <input autofocus id="password" name="password" required type="password">
            <button type="submit">Sign in</button>
        </form>
    </main>
</body>
</html>"""
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse, response_model=None)
def home(request: Request) -> HTMLResponse | RedirectResponse:
    if not is_authenticated(request):
        return RedirectResponse(url="/login", status_code=303)

    return HTMLResponse(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Labeling Tools</title>
  <style>
    body { background: #f1f5f2; color: #17261d; font-family: Georgia, serif; margin: 0; }
    header, main { margin: auto; max-width: 68rem; padding: 1.25rem 2rem; }
    header { align-items: center; display: flex; justify-content: space-between; }
    h1 { margin: 0; }
    section { background: #fffdf8; border: 1px solid #cbd7cc; border-radius: 8px; margin-top: 2rem; padding: 2rem; }
    button { background: none; border: 1px solid #1e6245; border-radius: 4px; color: #1e6245; cursor: pointer; font: inherit; padding: .45rem .7rem; }
  </style>
</head>
<body>
  <header><h1>Labeling Tools</h1><form action="/logout" method="post"><button type="submit">Sign out</button></form></header>
  <main><section><h2>Workspace</h2><p>Select a labeling operation to begin.</p></section></main>
</body>
</html>"""
    )


@app.get("/login", response_class=HTMLResponse, response_model=None)
def login(request: Request) -> HTMLResponse | RedirectResponse:
    if is_authenticated(request):
        return RedirectResponse(url="/", status_code=303)
    return login_page()


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
