from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import random
import shutil
import sys
import tempfile
from collections.abc import Iterable
from io import BytesIO
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from secrets import compare_digest
from urllib.parse import quote
from zoneinfo import ZoneInfo

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from PIL import Image
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
RESULT_FIELDNAMES = ("image_a", "image_b", "result")
RESULTS = ("A>B", "A<B", "A≈B")
HISTORY_FIELDNAMES = ("action", "image_a", "image_b", "result")
HISTORY_ACTIONS = ("comparison", "exclude")
EXCLUDE_SIDES = ("A", "B")
EXCLUDED_FIELDNAMES = ("image",)
DATASETS_FILE = "datasets.txt"

IMAGE_REFRESH_INTERVAL_SECONDS = 60
HALF_LIFE_IMAGES = 128
logger = logging.getLogger(__name__)

LIST_ERRORS = {
  "name": "名稱無效。",
  "datasets": "請選擇 dataset。",
  "exists": "名稱已存在。",
  "unknown": "Dataset 不存在。",
}


def hk_today() -> date:
  return datetime.now(ZoneInfo("Asia/Hong_Kong")).date()


@asynccontextmanager
async def lifespan(application: FastAPI):
  async def refresh_loop() -> None:
    while True:
      await asyncio.sleep(IMAGE_REFRESH_INTERVAL_SECONDS)
      try:
        catalog = application.state.catalog
        last_date = application.state.refresh_date
        today = hk_today()
        if not isinstance(catalog, Catalog):
          continue
        refreshed = await asyncio.to_thread(refresh_catalog, catalog, last_date, today)
        application.state.catalog = refreshed
        if today >= last_date:
          application.state.refresh_date = today
      except Exception:
        logger.exception("Image refresh failed; will retry at the next tick")

  task = None
  if application.state.refresh_date is not None:
    task = asyncio.create_task(refresh_loop())
  try:
    yield
  finally:
    if task is not None:
      task.cancel()
      with suppress(asyncio.CancelledError):
        await task


config = load_config()
app = FastAPI(title="Image Comparison", lifespan=lifespan)
app.state.catalog = None
app.state.workdir = None
app.state.refresh_date = None
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))
templates.env.filters["urlquote"] = lambda value: quote(str(value), safe="")


@dataclass(frozen=True)
class Catalog:
  images_root: Path
  datasets: dict[str, frozenset[str]]


@dataclass(frozen=True)
class VersusStore:
  name: str
  folder: Path
  known_images: frozenset[str]
  selected_images: frozenset[str]


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


def parse_port(value: str) -> int:
  try:
    port = int(value)
  except ValueError as error:
    raise argparse.ArgumentTypeError("--port must be an integer from 1 to 65535") from error
  if not 1 <= port <= 65535:
    raise argparse.ArgumentTypeError("--port must be an integer from 1 to 65535")
  return port


def parse_startup_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--images-root", required=True, help="Parent folder whose subfolders are datasets.")
  parser.add_argument("--workdir", required=True, help="Directory this app uses for versus folders.")
  parser.add_argument(
    "--port",
    type=parse_port,
    default=8000,
    help="TCP port to listen on (default: 8000).",
  )
  parser.add_argument(
    "--host",
    default="0.0.0.0",
    help="Address to bind (default: 0.0.0.0).",
  )
  options = parser.parse_args(arguments)
  try:
    options.images_root = resolve_existing_directory(options.images_root, "--images-root")
    options.workdir = Path(options.workdir).expanduser().resolve()
    options.workdir.mkdir(parents=True, exist_ok=True)
    if not options.workdir.is_dir():
      raise ValueError(f"--workdir must be a directory: {options.workdir}")
  except ValueError as error:
    parser.error(str(error))
  return options


def is_dataset_name(name: str) -> bool:
  return bool(name) and name not in {".", ".."} and "/" not in name and "\\" not in name and "\n" not in name and "\r" not in name


def is_versus_name(name: str) -> bool:
  if not name or name in {".", ".."} or name != name.strip():
    return False
  if any(ord(char) < 32 for char in name):
    return False
  return "/" not in name and "\\" not in name


def discover_images(image_folder: Path) -> list[Path]:
  return sorted({
    image_path.resolve() for image_path in image_folder.rglob("*")
    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
  })


def discover_dated_images(image_folder: Path, day: date) -> set[str]:
  folder = image_folder / day.strftime("%Y-%m") / day.isoformat()
  if not folder.is_dir():
    return set()
  return {
    str(path.resolve())
    for path in folder.iterdir()
    if path.is_file() and path.suffix.lower() == ".png"
  }


def union_images(groups: Iterable[frozenset[str]]) -> frozenset[str]:
  images: set[str] = set()
  for group in groups:
    images.update(group)
  return frozenset(images)


def build_catalog(images_root: Path) -> Catalog:
  resolved = images_root.resolve()
  datasets: dict[str, frozenset[str]] = {}
  for child in sorted((path for path in resolved.iterdir() if path.is_dir() and is_dataset_name(path.name)), key=lambda path: path.name):
    datasets[child.name] = frozenset(str(image) for image in discover_images(child))
  return Catalog(images_root=resolved, datasets=datasets)


def refresh_catalog(catalog: Catalog, last_scan_date: date, today: date) -> Catalog:
  # A backward clock leaves the cursor ahead of today; no new images are found until it catches up.
  if today < last_scan_date:
    return catalog
  updated: dict[str, frozenset[str]] = {}
  changed = False
  for name, images in catalog.datasets.items():
    found = set(images)
    day = last_scan_date
    while day <= today:
      found.update(discover_dated_images(catalog.images_root / name, day))
      day += timedelta(days=1)
    frozen = frozenset(found)
    if frozen != images:
      changed = True
    updated[name] = frozen
  if not changed:
    return catalog
  return Catalog(catalog.images_root, updated)


def result_path(folder: Path) -> Path:
  return folder / "comparisons.csv"


def history_path(folder: Path) -> Path:
  return folder / "history.csv"


def excluded_path(folder: Path) -> Path:
  return folder / "excluded.csv"


def datasets_path(folder: Path) -> Path:
  return folder / DATASETS_FILE


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
  temporary_file = tempfile.NamedTemporaryFile(
    "w", delete=False, dir=path.parent, encoding="utf-8", newline=""
  )
  temporary_path = Path(temporary_file.name)
  try:
    with temporary_file:
      writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    temporary_path.replace(path)
  finally:
    temporary_path.unlink(missing_ok=True)


def comparison_rows(history: list[dict[str, str]]) -> list[dict[str, str]]:
  return [
    {"image_a": row["image_a"], "image_b": row["image_b"], "result": row["result"]}
    for row in history
    if row["action"] == "comparison"
  ]


def excluded_images(history: list[dict[str, str]]) -> list[str]:
  return [
    row["image_a"] if row["result"] == "A" else row["image_b"]
    for row in history
    if row["action"] == "exclude"
  ]


def write_projections(folder: Path, history: list[dict[str, str]]) -> None:
  write_csv(result_path(folder), RESULT_FIELDNAMES, comparison_rows(history))
  write_csv(
    excluded_path(folder),
    EXCLUDED_FIELDNAMES,
    [{"image": image} for image in excluded_images(history)],
  )


def write_state(folder: Path, history: list[dict[str, str]]) -> None:
  write_csv(history_path(folder), HISTORY_FIELDNAMES, history)
  write_projections(folder, history)


def normalized_image_pair(
  image_a: str | None,
  image_b: str | None,
  known_images: frozenset[str],
  row_number: int,
  label: str,
) -> tuple[str, str]:
  if not image_a or not image_b or not Path(image_a).is_absolute() or not Path(image_b).is_absolute():
    raise ValueError(f"Invalid {label} row {row_number}")

  def canonical(image: str) -> str:
    if image in known_images:
      return image
    return str(Path(image).resolve())

  return canonical(image_a), canonical(image_b)


def read_legacy_comparisons(folder: Path, known_images: frozenset[str]) -> list[dict[str, str]]:
  path = result_path(folder)
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != RESULT_FIELDNAMES:
      raise ValueError("comparisons.csv must have exactly these columns: image_a,image_b,result")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      result = row.get("result")
      normalized_a, normalized_b = normalized_image_pair(
        row.get("image_a"), row.get("image_b"), known_images, row_number, "comparison"
      )
      if result not in RESULTS:
        raise ValueError(f"Invalid comparison result in row {row_number}")
      if normalized_a == normalized_b or normalized_a not in known_images or normalized_b not in known_images:
        raise ValueError(f"Invalid image path in comparison row {row_number}")
      if normalized_a in seen or normalized_b in seen:
        raise ValueError(f"Image appears more than once in comparison row {row_number}")
      seen.add(normalized_a)
      seen.add(normalized_b)
      rows.append({"image_a": normalized_a, "image_b": normalized_b, "result": result})
  return rows


def read_history(store: VersusStore) -> list[dict[str, str]]:
  path = history_path(store.folder)
  if not path.exists():
    return []
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != HISTORY_FIELDNAMES:
      raise ValueError("history.csv must have exactly these columns: action,image_a,image_b,result")
    history: list[dict[str, str]] = []
    compared: set[str] = set()
    excluded: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      action = row.get("action")
      result = row.get("result")
      if action not in HISTORY_ACTIONS:
        raise ValueError(f"Invalid history row {row_number}")
      normalized_a, normalized_b = normalized_image_pair(
        row.get("image_a"), row.get("image_b"), store.known_images, row_number, "history"
      )
      if normalized_a == normalized_b or normalized_a not in store.known_images or normalized_b not in store.known_images:
        raise ValueError(f"Invalid image path in history row {row_number}")
      consumed = compared | excluded
      if normalized_a in consumed or normalized_b in consumed:
        raise ValueError(f"Image appears more than once in history row {row_number}")
      if action == "comparison":
        if result not in RESULTS:
          raise ValueError(f"Invalid history row {row_number}")
        compared.add(normalized_a)
        compared.add(normalized_b)
      else:
        if result not in EXCLUDE_SIDES:
          raise ValueError(f"Invalid history row {row_number}")
        excluded.add(normalized_a if result == "A" else normalized_b)
      history.append({"action": action, "image_a": normalized_a, "image_b": normalized_b, "result": result})
  return history


def read_rows(store: VersusStore) -> list[dict[str, str]]:
  return comparison_rows(read_history(store))


def initialize_output(store: VersusStore) -> None:
  store.folder.mkdir(parents=True, exist_ok=True)
  if not store.folder.is_dir():
    raise ValueError(f"Output folder is not a directory: {store.folder}")
  if history_path(store.folder).exists():
    write_projections(store.folder, read_history(store))
    return
  if result_path(store.folder).exists():
    history = [
      {"action": "comparison", "image_a": row["image_a"], "image_b": row["image_b"], "result": row["result"]}
      for row in read_legacy_comparisons(store.folder, store.known_images)
    ]
    write_state(store.folder, history)
    return
  write_state(store.folder, [])


def read_dataset_names(folder: Path) -> tuple[str, ...]:
  path = datasets_path(folder)
  if not path.exists():
    return ()
  names: list[str] = []
  for line in path.read_text(encoding="utf-8").splitlines():
    name = line.strip()
    if name:
      names.append(name)
  return tuple(names)


def write_dataset_names(folder: Path, names: tuple[str, ...]) -> None:
  datasets_path(folder).write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def catalog_for(request: Request) -> Catalog:
  catalog = request.app.state.catalog
  if not isinstance(catalog, Catalog):
    raise HTTPException(status_code=503, detail="Comparison application is not configured")
  return catalog


def workdir_for(request: Request) -> Path:
  workdir = request.app.state.workdir
  if not isinstance(workdir, Path):
    raise HTTPException(status_code=503, detail="Comparison application is not configured")
  return workdir


def versus_directory(workdir: Path, name: str) -> Path:
  if not is_versus_name(name):
    raise HTTPException(status_code=404, detail="Versus was not found")
  root = workdir.resolve()
  folder = (root / name).resolve()
  if folder.parent != root:
    raise HTTPException(status_code=404, detail="Versus was not found")
  return folder


def make_store(catalog: Catalog, name: str, folder: Path) -> VersusStore:
  selected_names = read_dataset_names(folder)
  return VersusStore(
    name=name,
    folder=folder,
    known_images=union_images(catalog.datasets.values()),
    selected_images=union_images(catalog.datasets[dataset] for dataset in selected_names if dataset in catalog.datasets),
  )


def open_versus(request: Request, name: str) -> VersusStore:
  folder = versus_directory(workdir_for(request), name)
  if not folder.is_dir():
    raise HTTPException(status_code=404, detail="Versus was not found")
  store = make_store(catalog_for(request), name, folder)
  if not history_path(folder).exists():
    initialize_output(store)
  return store


def known_image(store: VersusStore, image_path: Path) -> bool:
  return (
    str(image_path.resolve()) in store.known_images
    and image_path.is_file()
    and image_path.suffix.lower() in IMAGE_EXTENSIONS
  )


def unused_images(store: VersusStore, history: list[dict[str, str]]) -> set[str]:
  consumed = set(excluded_images(history))
  for row in comparison_rows(history):
    consumed.add(row["image_a"])
    consumed.add(row["image_b"])
  return set(store.selected_images - consumed)


def recency_weights(oldest_first: list[str]) -> list[float]:
  newest = len(oldest_first) - 1
  return [0.5 ** ((newest - index) / HALF_LIFE_IMAGES) for index in range(len(oldest_first))]


def weighted_sample(oldest_first: list[str], count: int) -> list[str]:
  pool = oldest_first[:]
  chosen: list[str] = []
  for _ in range(count):
    pick = random.choices(pool, weights=recency_weights(pool), k=1)[0]
    chosen.append(pick)
    pool.remove(pick)
  return chosen


def session_pairs(request: Request) -> dict[str, list[str]]:
  pairs = request.session.get("pairs")
  if not isinstance(pairs, dict):
    return {}
  return pairs


def remember_pair(request: Request, name: str, pair: tuple[str, str] | None) -> None:
  pairs = session_pairs(request)
  if pair is None:
    pairs.pop(name, None)
  else:
    pairs[name] = [pair[0], pair[1]]
  request.session["pairs"] = pairs


def stored_pair(request: Request, name: str, unused: set[str]) -> tuple[str, str] | None:
  pairs = session_pairs(request)
  pair = pairs.get(name)
  if (
    isinstance(pair, list)
    and len(pair) == 2
    and all(isinstance(image, str) for image in pair)
    and pair[0] in unused
    and pair[1] in unused
    and pair[0] != pair[1]
  ):
    return pair[0], pair[1]
  if name in pairs:
    remember_pair(request, name, None)
  return None


def current_pair(request: Request, store: VersusStore) -> tuple[str, str] | None:
  unused = unused_images(store, read_history(store))
  pair = stored_pair(request, store.name, unused)
  if pair is not None:
    return pair
  if len(unused) < 2:
    return None
  chosen = weighted_sample(sorted(unused), 2)
  remember_pair(request, store.name, (chosen[0], chosen[1]))
  return chosen[0], chosen[1]


def append_comparison(store: VersusStore, image_a: str, image_b: str, result: str) -> bool:
  history = read_history(store)
  available = unused_images(store, history)
  if image_a not in available or image_b not in available:
    return False
  history.append({"action": "comparison", "image_a": image_a, "image_b": image_b, "result": result})
  write_state(store.folder, history)
  return True


def undo_last_action(store: VersusStore) -> dict[str, str] | None:
  history = read_history(store)
  if not history:
    return None
  last = history[-1]
  write_state(store.folder, history[:-1])
  return last


def versus_entries(workdir: Path) -> list[str]:
  if not workdir.is_dir():
    return []
  return sorted(child.name for child in workdir.iterdir() if child.is_dir() and is_versus_name(child.name))


def selected_dataset_names(catalog: Catalog, requested: list[str]) -> tuple[str, ...] | None:
  if any(name not in catalog.datasets for name in requested):
    return None
  chosen = set(requested)
  return tuple(name for name in catalog.datasets if name in chosen)


def versus_path(name: str, suffix: str = "") -> str:
  return f"/versus/{quote(name, safe='')}{suffix}"


def list_context(request: Request) -> dict[str, object]:
  catalog = catalog_for(request)
  workdir = workdir_for(request)
  versus_list: list[dict[str, object]] = []
  for name in versus_entries(workdir):
    folder = versus_directory(workdir, name)
    store = make_store(catalog, name, folder)
    if not history_path(folder).exists():
      initialize_output(store)
    history = read_history(store)
    versus_list.append({
      "name": name,
      "completed_pairs": len(comparison_rows(history)),
      "unused_count": len(unused_images(store, history)),
    })
  return {"versus_list": versus_list}


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
  return templates.TemplateResponse(
    request=request,
    name="index.html",
    context=list_context(request),
  )


@protected_pages.get("/new", response_class=HTMLResponse)
def new_versus_page(request: Request) -> HTMLResponse:
  catalog = catalog_for(request)
  error_code = request.query_params.get("error")
  return templates.TemplateResponse(
    request=request,
    name="new.html",
    context={
      "datasets": list(catalog.datasets),
      "error": LIST_ERRORS.get(error_code or "", ""),
    },
  )


@protected_pages.post("/new")
async def create_versus(request: Request) -> RedirectResponse:
  catalog = catalog_for(request)
  form = await request.form()
  name = str(form.get("name") or "").strip()
  requested = [str(item) for item in form.getlist("datasets")]
  if not is_versus_name(name):
    return RedirectResponse(url="/new?error=name", status_code=303)
  if any(dataset not in catalog.datasets for dataset in requested):
    return RedirectResponse(url="/new?error=unknown", status_code=303)
  if not requested:
    return RedirectResponse(url="/new?error=datasets", status_code=303)
  folder = versus_directory(workdir_for(request), name)
  if folder.exists():
    return RedirectResponse(url="/new?error=exists", status_code=303)
  folder.mkdir()
  selected = selected_dataset_names(catalog, requested)
  assert selected is not None
  write_dataset_names(folder, selected)
  initialize_output(make_store(catalog, name, folder))
  return RedirectResponse(url=versus_path(name), status_code=303)


@protected_pages.get("/versus/{name}/settings", response_class=HTMLResponse)
def versus_settings(request: Request, name: str) -> HTMLResponse:
  catalog = catalog_for(request)
  store = open_versus(request, name)
  error_code = request.query_params.get("error")
  return templates.TemplateResponse(
    request=request,
    name="settings.html",
    context={
      "name": store.name,
      "datasets": list(catalog.datasets),
      "selected": read_dataset_names(store.folder),
      "error": LIST_ERRORS.get(error_code or "", ""),
    },
  )


@protected_pages.post("/versus/{name}/settings")
async def update_versus_settings(request: Request, name: str) -> RedirectResponse:
  catalog = catalog_for(request)
  store = open_versus(request, name)
  form = await request.form()
  requested = [str(item) for item in form.getlist("datasets")]
  selected = selected_dataset_names(catalog, requested)
  if selected is None:
    return RedirectResponse(url=versus_path(store.name, "/settings?error=unknown"), status_code=303)
  write_dataset_names(store.folder, selected)
  return RedirectResponse(url=versus_path(store.name, "/settings"), status_code=303)


@protected_pages.get("/versus/{name}/delete", response_class=HTMLResponse)
def confirm_delete_versus(request: Request, name: str) -> HTMLResponse:
  store = open_versus(request, name)
  return templates.TemplateResponse(
    request=request,
    name="delete.html",
    context={"name": store.name},
  )


@protected_pages.post("/versus/{name}/delete")
def delete_versus(request: Request, name: str) -> RedirectResponse:
  store = open_versus(request, name)
  shutil.rmtree(store.folder)
  remember_pair(request, store.name, None)
  return RedirectResponse(url="/", status_code=303)


@protected_pages.get("/versus/{name}", response_class=HTMLResponse)
def compare(request: Request, name: str) -> HTMLResponse:
  store = open_versus(request, name)
  history = read_history(store)
  image_a, image_b = current_pair(request, store) or (None, None)
  return templates.TemplateResponse(
    request=request,
    name="compare.html",
    context={
      "name": store.name,
      "image_a": image_a,
      "image_b": image_b,
      "can_undo": bool(history),
      "completed_pairs": len(comparison_rows(history)),
      "unused_count": len(unused_images(store, history)),
    },
  )


JPEG_QUALITY = 85


def jpeg_bytes(image_path: Path) -> bytes:
  with Image.open(image_path) as image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


@protected_pages.get("/image")
def image(request: Request, image_path: str) -> Response:
  catalog = catalog_for(request)
  resolved = Path(image_path).resolve()
  known = union_images(catalog.datasets.values())
  if (
    str(resolved) not in known
    or not resolved.is_file()
    or resolved.suffix.lower() not in IMAGE_EXTENSIONS
  ):
    raise HTTPException(status_code=404, detail="Image is not available")
  return Response(
    content=jpeg_bytes(resolved),
    media_type="image/jpeg",
    headers={"Cache-Control": "private, max-age=86400"},
  )


@protected_pages.post("/versus/{name}/comparisons")
def create_comparison(
  request: Request,
  name: str,
  image_a: str = Form(),
  image_b: str = Form(),
  result: str = Form(),
) -> RedirectResponse:
  store = open_versus(request, name)
  if result not in RESULTS:
    raise HTTPException(status_code=422, detail="Result is not a comparison choice")
  resolved_a = Path(image_a).resolve()
  resolved_b = Path(image_b).resolve()
  if resolved_a == resolved_b or not known_image(store, resolved_a) or not known_image(store, resolved_b):
    raise HTTPException(status_code=404, detail="Image is not available")
  append_comparison(store, str(resolved_a), str(resolved_b), result)
  remember_pair(request, store.name, None)
  return RedirectResponse(url=versus_path(store.name), status_code=303)


@protected_pages.post("/versus/{name}/exclusions")
def create_exclusion(request: Request, name: str, image: str = Form()) -> RedirectResponse:
  store = open_versus(request, name)
  resolved = Path(image).resolve()
  if not known_image(store, resolved):
    raise HTTPException(status_code=404, detail="Image is not available")
  image_path = str(resolved)
  history = read_history(store)
  unused = unused_images(store, history)
  pair = stored_pair(request, store.name, unused)
  if pair is None or image_path not in pair:
    raise HTTPException(status_code=409, detail="Image is not in the current pair")
  side = "A" if pair[0] == image_path else "B"
  history.append({"action": "exclude", "image_a": pair[0], "image_b": pair[1], "result": side})
  write_state(store.folder, history)
  kept = pair[1] if side == "A" else pair[0]
  remaining = sorted(unused - {image_path, kept})
  if not remaining:
    remember_pair(request, store.name, None)
  else:
    replacement = weighted_sample(remaining, 1)[0]
    remember_pair(request, store.name, (replacement, kept) if side == "A" else (kept, replacement))
  return RedirectResponse(url=versus_path(store.name), status_code=303)


@protected_pages.post("/versus/{name}/comparisons/undo")
def undo_comparison(request: Request, name: str) -> RedirectResponse:
  store = open_versus(request, name)
  restored = undo_last_action(store)
  if restored is None:
    raise HTTPException(status_code=409, detail="There is nothing to undo")
  unused = unused_images(store, read_history(store))
  if restored["image_a"] in unused and restored["image_b"] in unused:
    remember_pair(request, store.name, (restored["image_a"], restored["image_b"]))
  else:
    remember_pair(request, store.name, None)
  return RedirectResponse(url=versus_path(store.name), status_code=303)


app.include_router(protected_pages)


def serve(host: str, port: int) -> None:
  uvicorn.run(app, host=host, port=port)


def main(arguments: list[str] | None = None) -> None:
  startup_arguments = parse_startup_arguments(arguments)
  app.state.catalog = build_catalog(startup_arguments.images_root)
  app.state.workdir = startup_arguments.workdir
  app.state.refresh_date = hk_today()
  serve(startup_arguments.host, startup_arguments.port)


if __name__ == "__main__":
  main()
