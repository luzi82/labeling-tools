from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from secrets import compare_digest

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
RESULT_FIELDNAMES = ("image_a", "image_b", "result")
RESULTS = ("A>B", "A<B", "A≈B")

config = load_config()
app = FastAPI(title="Image Comparison")
app.state.runtime = None
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


@dataclass(frozen=True)
class RuntimeConfig:
  image_folder: Path
  output_folder: Path
  source_images: frozenset[str]


def is_authenticated(request: Request) -> bool:
  return request.session.get("authenticated") is True


def require_authentication(request: Request) -> None:
  if not is_authenticated(request):
    raise HTTPException(status_code=303, headers={"Location": "/login"})


def resolve_existing_directory(value: str, argument: str) -> Path:
  path = Path(value).expanduser().resolve()
  if not path.is_dir():
    raise ValueError(f"{argument} must be an existing directory: {path}")
  return path


def parse_startup_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--image-folder", required=True, help="Folder of images to compare.")
  parser.add_argument("--output-folder", required=True, help="Directory for comparison results.")
  options = parser.parse_args(arguments)
  try:
    options.image_folder = resolve_existing_directory(options.image_folder, "--image-folder")
    options.output_folder = Path(options.output_folder).expanduser().resolve()
    options.output_folder.mkdir(parents=True, exist_ok=True)
    if not options.output_folder.is_dir():
      raise ValueError(f"--output-folder must be a directory: {options.output_folder}")
  except ValueError as error:
    parser.error(str(error))
  return options


def discover_images(image_folder: Path) -> list[Path]:
  return sorted({
    image_path.resolve() for image_path in image_folder.rglob("*")
    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
  })


def build_runtime(*, image_folder: Path, output_folder: Path) -> RuntimeConfig:
  return RuntimeConfig(
    image_folder=image_folder,
    output_folder=output_folder,
    source_images=frozenset(str(image) for image in discover_images(image_folder)),
  )


def result_path(runtime: RuntimeConfig) -> Path:
  return runtime.output_folder / "comparisons.csv"


def rewrite_rows(runtime: RuntimeConfig, rows: list[dict[str, str]]) -> None:
  csv_path = result_path(runtime)
  temporary_file = tempfile.NamedTemporaryFile(
    "w", delete=False, dir=csv_path.parent, encoding="utf-8", newline=""
  )
  temporary_path = Path(temporary_file.name)
  try:
    with temporary_file:
      writer = csv.DictWriter(temporary_file, fieldnames=RESULT_FIELDNAMES)
      writer.writeheader()
      writer.writerows(rows)
    temporary_path.replace(csv_path)
  finally:
    temporary_path.unlink(missing_ok=True)


def read_rows(runtime: RuntimeConfig) -> list[dict[str, str]]:
  path = result_path(runtime)
  if not path.exists():
    return []
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != RESULT_FIELDNAMES:
      raise ValueError("comparisons.csv must have exactly these columns: image_a,image_b,result")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      image_a = row.get("image_a")
      image_b = row.get("image_b")
      result = row.get("result")
      if not image_a or not image_b or not Path(image_a).is_absolute() or not Path(image_b).is_absolute():
        raise ValueError(f"Invalid comparison row {row_number}")
      if result not in RESULTS:
        raise ValueError(f"Invalid comparison result in row {row_number}")
      normalized_a = str(Path(image_a).resolve())
      normalized_b = str(Path(image_b).resolve())
      if normalized_a == normalized_b or normalized_a not in runtime.source_images or normalized_b not in runtime.source_images:
        raise ValueError(f"Invalid image path in comparison row {row_number}")
      if normalized_a in seen or normalized_b in seen:
        raise ValueError(f"Image appears more than once in comparison row {row_number}")
      seen.add(normalized_a)
      seen.add(normalized_b)
      rows.append({"image_a": normalized_a, "image_b": normalized_b, "result": result})
  return rows


def initialize_output(runtime: RuntimeConfig) -> None:
  runtime.output_folder.mkdir(parents=True, exist_ok=True)
  if not runtime.output_folder.is_dir():
    raise ValueError(f"Output folder is not a directory: {runtime.output_folder}")
  if result_path(runtime).exists():
    read_rows(runtime)
    return
  rewrite_rows(runtime, [])


def runtime_for(request: Request) -> RuntimeConfig:
  runtime = request.app.state.runtime
  if not isinstance(runtime, RuntimeConfig):
    raise HTTPException(status_code=503, detail="Comparison application is not configured")
  return runtime


def is_available_image(runtime: RuntimeConfig, image_path: Path) -> bool:
  return (
    str(image_path.resolve()) in runtime.source_images
    and image_path.is_file()
    and image_path.suffix.lower() in IMAGE_EXTENSIONS
  )


def unused_images(runtime: RuntimeConfig, rows: list[dict[str, str]]) -> set[str]:
  used = {row["image_a"] for row in rows} | {row["image_b"] for row in rows}
  return set(runtime.source_images - used)


def stored_pair(request: Request, unused: set[str]) -> tuple[str, str] | None:
  pair = request.session.get("pair")
  if (
    isinstance(pair, list)
    and len(pair) == 2
    and all(isinstance(image, str) for image in pair)
    and pair[0] in unused
    and pair[1] in unused
    and pair[0] != pair[1]
  ):
    return pair[0], pair[1]
  request.session.pop("pair", None)
  return None


def current_pair(request: Request, runtime: RuntimeConfig) -> tuple[str, str] | None:
  unused = unused_images(runtime, read_rows(runtime))
  pair = stored_pair(request, unused)
  if pair is not None:
    return pair
  if len(unused) < 2:
    return None
  chosen = random.sample(sorted(unused), 2)
  request.session["pair"] = chosen
  return chosen[0], chosen[1]


def append_comparison(runtime: RuntimeConfig, image_a: str, image_b: str, result: str) -> bool:
  rows = read_rows(runtime)
  available = unused_images(runtime, rows)
  if image_a not in available or image_b not in available:
    return False
  rows.append({"image_a": image_a, "image_b": image_b, "result": result})
  rewrite_rows(runtime, rows)
  return True


def undo_last_comparison(runtime: RuntimeConfig) -> dict[str, str] | None:
  rows = read_rows(runtime)
  if not rows:
    return None
  last = rows[-1]
  rewrite_rows(runtime, rows[:-1])
  return last


protected_pages = APIRouter(dependencies=[Depends(require_authentication)])


@app.get("/login", response_class=HTMLResponse, response_model=None)
def login(request: Request, error: bool = False) -> HTMLResponse | RedirectResponse:
  if is_authenticated(request):
    return RedirectResponse(url="/", status_code=303)
  return templates.TemplateResponse(request=request, name="login.html", context={"error": error})


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


@protected_pages.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
  runtime = runtime_for(request)
  image_a, image_b = current_pair(request, runtime) or (None, None)
  return templates.TemplateResponse(
    request=request,
    name="compare.html",
    context={
      "image_a": image_a,
      "image_b": image_b,
      "can_undo": bool(read_rows(runtime)),
    },
  )


@protected_pages.get("/image")
def image(request: Request, image_path: str) -> FileResponse:
  runtime = runtime_for(request)
  resolved = Path(image_path).resolve()
  if not is_available_image(runtime, resolved):
    raise HTTPException(status_code=404, detail="Image is not available")
  return FileResponse(resolved)


@protected_pages.post("/comparisons")
def create_comparison(
  request: Request,
  image_a: str = Form(),
  image_b: str = Form(),
  result: str = Form(),
) -> RedirectResponse:
  runtime = runtime_for(request)
  if result not in RESULTS:
    raise HTTPException(status_code=422, detail="Result is not a comparison choice")
  resolved_a = Path(image_a).resolve()
  resolved_b = Path(image_b).resolve()
  if resolved_a == resolved_b or not is_available_image(runtime, resolved_a) or not is_available_image(runtime, resolved_b):
    raise HTTPException(status_code=404, detail="Image is not available")
  append_comparison(runtime, str(resolved_a), str(resolved_b), result)
  request.session.pop("pair", None)
  return RedirectResponse(url="/", status_code=303)


@protected_pages.post("/comparisons/undo")
def undo_comparison(request: Request) -> RedirectResponse:
  restored = undo_last_comparison(runtime_for(request))
  if restored is None:
    raise HTTPException(status_code=409, detail="There is no comparison to undo")
  request.session["pair"] = [restored["image_a"], restored["image_b"]]
  return RedirectResponse(url="/", status_code=303)


app.include_router(protected_pages)


if __name__ == "__main__":
  startup_arguments = parse_startup_arguments()
  try:
    runtime = build_runtime(
      image_folder=startup_arguments.image_folder,
      output_folder=startup_arguments.output_folder,
    )
    initialize_output(runtime)
  except ValueError as error:
    raise SystemExit(f"Configuration error: {error}") from error
  app.state.runtime = runtime
  uvicorn.run(app, host="0.0.0.0", port=8000)
