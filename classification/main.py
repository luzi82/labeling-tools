from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from secrets import compare_digest

import uvicorn
import yaml
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from starlette.templating import Jinja2Templates

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from common.config import load_config


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
SOURCE_FIELDNAMES = ("image_path", "label")
RESULT_FIELDNAMES = ("image_path", "label", "human_checked_state")
NOT_HUMAN_CHECKED = "NOT_HUMAN_CHECKED"
HUMAN_SCREENED = "HUMAN_SCREENED"
HUMAN_CORRECTED = "HUMAN_CORRECTED"
HUMAN_LABELED = "HUMAN_LABELED"
HUMAN_CHECKED_STATES = {NOT_HUMAN_CHECKED, HUMAN_SCREENED, HUMAN_CORRECTED, HUMAN_LABELED}

config = load_config()
app = FastAPI(title="Image Classification")
app.state.runtime = None
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


@dataclass(frozen=True)
class RuntimeConfig:
  mode: str
  labels: tuple[str, ...]
  output_folder: Path
  meta_yaml: Path
  source_images: frozenset[str]
  input_csv: Path | None = None
  image_folder: Path | None = None


class LabelSubmission(BaseModel):
  image_path: str
  label: str


def is_authenticated(request: Request) -> bool:
  return request.session.get("authenticated") is True


def require_authentication(request: Request) -> None:
  if not is_authenticated(request):
    raise HTTPException(status_code=303, headers={"Location": "/login"})


def resolve_existing_file(value: str, argument: str) -> Path:
  path = Path(value).expanduser().resolve()
  if not path.is_file():
    raise ValueError(f"{argument} must be an existing file: {path}")
  return path


def resolve_existing_directory(value: str, argument: str) -> Path:
  path = Path(value).expanduser().resolve()
  if not path.is_dir():
    raise ValueError(f"{argument} must be an existing directory: {path}")
  return path


def parse_startup_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  source_group = parser.add_mutually_exclusive_group(required=True)
  source_group.add_argument("--input-csv", help="Read-only model classification CSV.")
  source_group.add_argument("--image-folder", help="Folder of images for initial human labeling.")
  parser.add_argument("--meta-yaml", required=True, help="YAML file containing label_list.")
  parser.add_argument("--output-folder", required=True, help="Directory for human results.")
  options = parser.parse_args(arguments)
  try:
    options.meta_yaml = resolve_existing_file(options.meta_yaml, "--meta-yaml")
    if options.input_csv:
      options.input_csv = resolve_existing_file(options.input_csv, "--input-csv")
      options.image_folder = None
    else:
      options.image_folder = resolve_existing_directory(options.image_folder, "--image-folder")
      options.input_csv = None
    options.output_folder = Path(options.output_folder).expanduser().resolve()
    options.output_folder.mkdir(parents=True, exist_ok=True)
    if not options.output_folder.is_dir():
      raise ValueError(f"--output-folder must be a directory: {options.output_folder}")
  except ValueError as error:
    parser.error(str(error))
  return options


def load_labels(meta_yaml: Path) -> tuple[str, ...]:
  with meta_yaml.open(encoding="utf-8") as meta_file:
    metadata = yaml.safe_load(meta_file)
  labels = metadata.get("label_list") if isinstance(metadata, dict) else None
  if not isinstance(labels, list) or not all(isinstance(label, str) and label.strip() for label in labels):
    raise ValueError("meta YAML label_list must be a list of non-empty strings")
  if not 1 <= len(labels) <= 9 or len(set(labels)) != len(labels):
    raise ValueError("meta YAML label_list must contain 1-9 unique labels")
  return tuple(labels)


def read_source_rows(input_csv: Path, labels: tuple[str, ...]) -> list[dict[str, str]]:
  with input_csv.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != SOURCE_FIELDNAMES:
      raise ValueError("input CSV must have exactly these columns: image_path,label")
    rows: list[dict[str, str]] = []
    image_paths: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      image_path = row.get("image_path")
      label = row.get("label")
      if not image_path or not Path(image_path).is_absolute() or label not in labels:
        raise ValueError(f"Invalid input CSV row {row_number}")
      normalized_path = str(Path(image_path).resolve())
      if normalized_path in image_paths:
        raise ValueError(f"Duplicate image_path in input CSV: {normalized_path}")
      image_paths.add(normalized_path)
      rows.append({"image_path": normalized_path, "label": label})
  return rows


def discover_images(image_folder: Path) -> list[Path]:
  return sorted({
    image_path.resolve() for image_path in image_folder.rglob("*")
    if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
  })


def build_runtime(
  *, input_csv: Path | None, image_folder: Path | None, meta_yaml: Path, output_folder: Path
) -> RuntimeConfig:
  labels = load_labels(meta_yaml)
  if input_csv:
    rows = read_source_rows(input_csv, labels)
    return RuntimeConfig(
      mode="model", labels=labels, output_folder=output_folder, meta_yaml=meta_yaml,
      input_csv=input_csv, source_images=frozenset(row["image_path"] for row in rows),
    )
  if image_folder:
    return RuntimeConfig(
      mode="initial", labels=labels, output_folder=output_folder, meta_yaml=meta_yaml,
      image_folder=image_folder, source_images=frozenset(str(image) for image in discover_images(image_folder)),
    )
  raise ValueError("Provide exactly one source")


def content_hash(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source_file:
    for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def expected_manifest(runtime: RuntimeConfig) -> dict[str, object]:
  source: dict[str, object] = {"image_paths": sorted(runtime.source_images)}
  if runtime.input_csv:
    source["input_csv"] = str(runtime.input_csv)
    source["input_csv_sha256"] = content_hash(runtime.input_csv)
  else:
    source["image_folder"] = str(runtime.image_folder)
  return {
    "source_mode": runtime.mode,
    "source": source,
    "meta_yaml": str(runtime.meta_yaml),
    "meta_yaml_sha256": content_hash(runtime.meta_yaml),
  }


def rewrite_csv_rows(csv_path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
  temporary_file = tempfile.NamedTemporaryFile(
    "w", delete=False, dir=csv_path.parent, encoding="utf-8", newline=""
  )
  temporary_path = Path(temporary_file.name)
  try:
    with temporary_file:
      writer = csv.DictWriter(temporary_file, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)
    temporary_path.replace(csv_path)
  finally:
    temporary_path.unlink(missing_ok=True)


def rewrite_json(path: Path, value: object) -> None:
  temporary_file = tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8")
  temporary_path = Path(temporary_file.name)
  try:
    with temporary_file:
      json.dump(value, temporary_file, ensure_ascii=True, indent=2, sort_keys=True)
      temporary_file.write("\n")
    temporary_path.replace(path)
  finally:
    temporary_path.unlink(missing_ok=True)


def result_path(runtime: RuntimeConfig) -> Path:
  return runtime.output_folder / "human_classification.csv"


def history_path(runtime: RuntimeConfig) -> Path:
  return runtime.output_folder / "human_classification_undo_history.json"


def manifest_path(runtime: RuntimeConfig) -> Path:
  return runtime.output_folder / "human_classification_manifest.json"


def read_result_rows(runtime: RuntimeConfig) -> list[dict[str, str]]:
  path = result_path(runtime)
  if not path.exists():
    return []
  with path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != RESULT_FIELDNAMES:
      raise ValueError("human_classification.csv has invalid columns")
    rows: list[dict[str, str]] = []
    image_paths: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      image_path = row.get("image_path")
      label = row.get("label")
      state = row.get("human_checked_state")
      if not image_path or not label or not state:
        raise ValueError(f"Invalid human result row {row_number}")
      normalized_path = str(Path(image_path).resolve())
      if normalized_path not in runtime.source_images or normalized_path in image_paths:
        raise ValueError(f"Invalid image_path in human result row {row_number}")
      if label not in runtime.labels or state not in HUMAN_CHECKED_STATES:
        raise ValueError(f"Invalid label or human state in result row {row_number}")
      image_paths.add(normalized_path)
      rows.append({"image_path": normalized_path, "label": label, "human_checked_state": state})
  if runtime.mode == "model" and image_paths != runtime.source_images:
    raise ValueError("human result image set does not match model input")
  return rows


def rewrite_result_rows(runtime: RuntimeConfig, rows: list[dict[str, str]]) -> None:
  rewrite_csv_rows(result_path(runtime), RESULT_FIELDNAMES, rows)


def read_history(runtime: RuntimeConfig) -> list[dict[str, object]]:
  path = history_path(runtime)
  if not path.exists():
    return []
  with path.open(encoding="utf-8") as history_file:
    history = json.load(history_file)
  if not isinstance(history, list):
    raise ValueError("human undo history must be a list")
  return history


def rewrite_history(runtime: RuntimeConfig, history: list[dict[str, object]]) -> None:
  rewrite_json(history_path(runtime), history)


def initialize_output(runtime: RuntimeConfig) -> None:
  runtime.output_folder.mkdir(parents=True, exist_ok=True)
  if not runtime.output_folder.is_dir():
    raise ValueError(f"Output folder is not a directory: {runtime.output_folder}")
  expected = expected_manifest(runtime)
  manifest = manifest_path(runtime)
  results = result_path(runtime)
  if manifest.exists() or results.exists():
    if not manifest.exists() or not results.exists():
      raise ValueError("Output folder has incomplete human classification data")
    with manifest.open(encoding="utf-8") as manifest_file:
      actual = json.load(manifest_file)
    if actual != expected:
      raise ValueError("Output folder source or metadata does not match this run")
    read_result_rows(runtime)
    read_history(runtime)
    return
  rows: list[dict[str, str]] = []
  if runtime.mode == "model":
    assert runtime.input_csv is not None
    rows = [{**row, "human_checked_state": NOT_HUMAN_CHECKED} for row in read_source_rows(runtime.input_csv, runtime.labels)]
  rewrite_result_rows(runtime, rows)
  rewrite_history(runtime, [])
  rewrite_json(manifest, expected)


def runtime_for(request: Request) -> RuntimeConfig:
  runtime = request.app.state.runtime
  if not isinstance(runtime, RuntimeConfig):
    raise HTTPException(status_code=503, detail="Classification application is not configured")
  return runtime


def is_available_image(runtime: RuntimeConfig, image_path: Path) -> bool:
  return (
    str(image_path.resolve()) in runtime.source_images
    and image_path.is_file()
    and image_path.suffix.lower() in IMAGE_EXTENSIONS
  )


def record_action(runtime: RuntimeConfig, action: str, changes: list[dict[str, object]]) -> None:
  history = read_history(runtime)
  history.append({"action": action, "changes": changes})
  rewrite_history(runtime, history)


def create_initial_classification(runtime: RuntimeConfig, image_path: Path, label: str) -> None:
  rows = read_result_rows(runtime)
  if any(row["image_path"] == str(image_path) for row in rows):
    raise ValueError("Image already has a classification")
  row = {"image_path": str(image_path), "label": label, "human_checked_state": HUMAN_LABELED}
  rows.append(row)
  rewrite_result_rows(runtime, rows)
  record_action(runtime, "create", [{"before": None, "after": row}])


def update_model_classification(runtime: RuntimeConfig, image_path: Path, label: str) -> bool:
  rows = read_result_rows(runtime)
  row = next((row for row in rows if row["image_path"] == str(image_path)), None)
  if row is None:
    raise ValueError("Image does not have a classification")
  if row["label"] == label:
    return False
  before = dict(row)
  row["label"] = label
  row["human_checked_state"] = HUMAN_CORRECTED
  rewrite_result_rows(runtime, rows)
  record_action(runtime, "update", [{"before": before, "after": dict(row)}])
  return True


def update_initial_classification(runtime: RuntimeConfig, image_path: Path, label: str) -> bool:
  rows = read_result_rows(runtime)
  row = next((row for row in rows if row["image_path"] == str(image_path)), None)
  if row is None:
    raise ValueError("Image does not have a classification")
  if row["label"] == label:
    return False
  before = dict(row)
  row["label"] = label
  rewrite_result_rows(runtime, rows)
  record_action(runtime, "update", [{"before": before, "after": dict(row)}])
  return True


def complete_model_label(runtime: RuntimeConfig, label: str) -> int:
  rows = read_result_rows(runtime)
  changes: list[dict[str, object]] = []
  for row in rows:
    if row["label"] == label and row["human_checked_state"] == NOT_HUMAN_CHECKED:
      before = dict(row)
      row["human_checked_state"] = HUMAN_SCREENED
      changes.append({"before": before, "after": dict(row)})
  if not changes:
    return 0
  rewrite_result_rows(runtime, rows)
  record_action(runtime, "screen", changes)
  return len(changes)


def undo_last_action(runtime: RuntimeConfig) -> dict[str, object] | None:
  history = read_history(runtime)
  if not history:
    return None
  action = history[-1]
  action_name = action.get("action")
  changes = action.get("changes")
  if action_name not in {"create", "update", "screen"} or not isinstance(changes, list) or not changes:
    raise ValueError("Invalid human undo history")
  rows = read_result_rows(runtime)
  rows_by_path = {row["image_path"]: row for row in rows}
  for change in changes:
    if not isinstance(change, dict) or not isinstance(change.get("after"), dict):
      raise ValueError("Invalid human undo history change")
    after = change["after"]
    if rows_by_path.get(after.get("image_path")) != after:
      raise ValueError("Latest undo action does not match human results")
  for change in changes:
    before = change["before"]
    after = change["after"]
    if before is None:
      rows.remove(rows_by_path[after["image_path"]])
    else:
      rows_by_path[after["image_path"]].update(before)
  rewrite_result_rows(runtime, rows)
  rewrite_history(runtime, history[:-1])
  result: dict[str, object] = {"action": action_name, "changed_count": len(changes)}
  if action_name == "create" and len(changes) == 1:
    result["restored_image_path"] = changes[0]["after"]["image_path"]
  return result


def model_dashboard(runtime: RuntimeConfig) -> list[dict[str, object]]:
  rows = read_result_rows(runtime)
  dashboard: list[dict[str, object]] = []
  for label in runtime.labels:
    label_rows = [row for row in rows if row["label"] == label]
    dashboard.append({
      "label": label,
      "total": len(label_rows),
      "not_human_checked": sum(row["human_checked_state"] == NOT_HUMAN_CHECKED for row in label_rows),
      "human_screened": sum(row["human_checked_state"] == HUMAN_SCREENED for row in label_rows),
      "human_corrected": sum(row["human_checked_state"] == HUMAN_CORRECTED for row in label_rows),
      "human_labeled": sum(row["human_checked_state"] == HUMAN_LABELED for row in label_rows),
    })
  return dashboard


def get_review_items(runtime: RuntimeConfig, label: str | None, corrected_only: bool, offset: int, limit: int) -> tuple[list[dict[str, str | bool]], int]:
  if runtime.mode == "initial" and corrected_only:
    raise ValueError("Corrected-image review is available only in model mode")
  if corrected_only:
    rows = [row for row in read_result_rows(runtime) if row["human_checked_state"] == HUMAN_CORRECTED]
    if label is not None:
      if label not in runtime.labels:
        raise ValueError("Label is not configured")
      rows = [row for row in rows if row["label"] == label]
  elif label in runtime.labels:
    rows = [row for row in read_result_rows(runtime) if row["label"] == label]
  else:
    raise ValueError("Label is not configured")
  rows.sort(key=lambda row: row["image_path"])
  items = [{**row, "missing": not is_available_image(runtime, Path(row["image_path"]))} for row in rows[offset:offset + limit]]
  return items, len(rows)


protected_pages = APIRouter(dependencies=[Depends(require_authentication)])


@app.get("/health")
def health_check() -> dict[str, str]:
  return {"status": "ok"}


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
def home(request: Request, image_path: str | None = None, view: str | None = None) -> HTMLResponse:
  runtime = runtime_for(request)
  if runtime.mode == "model":
    return templates.TemplateResponse(request=request, name="home.html", context={"mode": "model", "dashboard": model_dashboard(runtime), "labels": runtime.labels})
  rows = read_result_rows(runtime)
  if view == "review":
    return templates.TemplateResponse(request=request, name="home.html", context={"mode": "initial", "dashboard": model_dashboard(runtime), "labels": runtime.labels, "review_mode": True})
  classified = {row["image_path"] for row in rows}
  unclassified = [Path(path) for path in runtime.source_images if path not in classified]
  selected = Path(image_path).resolve() if image_path else None
  if selected and selected not in unclassified:
    raise HTTPException(status_code=404, detail="Image is not available for classification")
  selected = selected or (random.choice(unclassified) if unclassified else None)
  return templates.TemplateResponse(request=request, name="home.html", context={"mode": "initial", "labels": runtime.labels, "image_path": str(selected) if selected else None})


@protected_pages.get("/image")
def image(request: Request, image_path: str) -> FileResponse:
  runtime = runtime_for(request)
  resolved = Path(image_path).resolve()
  if not is_available_image(runtime, resolved):
    raise HTTPException(status_code=404, detail="Image is not available")
  return FileResponse(resolved)


@protected_pages.post("/classifications")
def create_classification(request: Request, submission: LabelSubmission) -> dict[str, str]:
  runtime = runtime_for(request)
  image_path = Path(submission.image_path).resolve()
  if runtime.mode != "initial":
    raise HTTPException(status_code=405, detail="Initial labeling is unavailable in model mode")
  if submission.label not in runtime.labels:
    raise HTTPException(status_code=422, detail="Label is not configured")
  if not is_available_image(runtime, image_path):
    raise HTTPException(status_code=404, detail="Image is not available")
  try:
    create_initial_classification(runtime, image_path, submission.label)
  except ValueError as error:
    raise HTTPException(status_code=409, detail=str(error)) from error
  return {"image_path": str(image_path), "label": submission.label, "human_checked_state": HUMAN_LABELED}


@protected_pages.get("/review", response_class=HTMLResponse)
def review(request: Request, label: str | None = None, corrected_only: bool = False) -> HTMLResponse:
  runtime = runtime_for(request)
  if corrected_only and runtime.mode != "model":
    raise HTTPException(status_code=404, detail="Corrected-image review is available only in model mode")
  if not corrected_only and label not in runtime.labels:
    raise HTTPException(status_code=404, detail="Label is not configured")
  context = {"label": label, "labels": runtime.labels, "corrected_only": corrected_only, "can_edit": True, "dashboard": []}
  if corrected_only:
    context["dashboard"] = model_dashboard(runtime)
  return templates.TemplateResponse(request=request, name="review.html", context=context)


@protected_pages.get("/review/items")
def review_items(request: Request, label: str | None = None, corrected_only: bool = False, offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=100)) -> dict[str, object]:
  try:
    items, total = get_review_items(runtime_for(request), label, corrected_only, offset, limit)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  next_offset = offset + len(items)
  return {"items": items, "total": total, "next_offset": next_offset if next_offset < total else None, "has_more": next_offset < total}


@protected_pages.patch("/classifications")
def update_classification(request: Request, submission: LabelSubmission) -> dict[str, str | bool]:
  runtime = runtime_for(request)
  image_path = Path(submission.image_path).resolve()
  if submission.label not in runtime.labels:
    raise HTTPException(status_code=422, detail="Label is not configured")
  if not is_available_image(runtime, image_path):
    raise HTTPException(status_code=404, detail="Image is not available")
  try:
    updated = (update_model_classification if runtime.mode == "model" else update_initial_classification)(runtime, image_path, submission.label)
  except ValueError as error:
    raise HTTPException(status_code=409, detail=str(error)) from error
  return {"image_path": str(image_path), "label": submission.label, "updated": updated}


@protected_pages.post("/review/complete")
def complete_review(request: Request, label: str) -> dict[str, int]:
  runtime = runtime_for(request)
  if runtime.mode != "model" or label not in runtime.labels:
    raise HTTPException(status_code=404, detail="Label is not configured for model review")
  return {"changed_count": complete_model_label(runtime, label)}


@protected_pages.post("/classifications/undo")
def undo_last_classification(request: Request) -> dict[str, object]:
  try:
    undone = undo_last_action(runtime_for(request))
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if undone is None:
    raise HTTPException(status_code=409, detail="There is no classification action to undo")
  return undone


app.include_router(protected_pages)


if __name__ == "__main__":
  startup_arguments = parse_startup_arguments()
  try:
    runtime = build_runtime(input_csv=startup_arguments.input_csv, image_folder=startup_arguments.image_folder, meta_yaml=startup_arguments.meta_yaml, output_folder=startup_arguments.output_folder)
    initialize_output(runtime)
  except ValueError as error:
    raise SystemExit(f"Configuration error: {error}") from error
  app.state.runtime = runtime
  uvicorn.run(app, host="0.0.0.0", port=8000)
