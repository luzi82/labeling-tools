from __future__ import annotations

import argparse
import os
from pathlib import Path
from secrets import compare_digest

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from common.config import load_config
from compare.home import router as home_router
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates


config = load_config()
app = FastAPI(title="Labeling Tools API")
app.state.working_folder = Path.cwd()
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


def is_authenticated(request: Request) -> bool:
  return request.session.get("authenticated") is True


def require_authentication(request: Request) -> None:
  if not is_authenticated(request):
    raise HTTPException(status_code=303, headers={"Location": "/login"})


protected_pages = APIRouter(dependencies=[Depends(require_authentication)])


def parse_startup_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--working-folder",
    default=Path.cwd(),
    help="Directory used as the process working folder, created if missing (default: current directory).",
  )
  options = parser.parse_args(arguments)
  working_folder = Path(options.working_folder).expanduser().resolve()
  working_folder.mkdir(parents=True, exist_ok=True)
  if not working_folder.is_dir():
    parser.error(f"--working-folder must be a directory: {working_folder}")
  options.working_folder = working_folder
  return options


@app.get("/health")
def health_check() -> dict[str, str]:
  return {"status": "ok"}


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


protected_pages.include_router(home_router)
app.include_router(protected_pages)


if __name__ == "__main__":
  startup_arguments = parse_startup_arguments()
  os.chdir(startup_arguments.working_folder)
  app.state.working_folder = startup_arguments.working_folder
  uvicorn.run(app, host="0.0.0.0", port=8000)
