from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import tempfile
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
CSV_FIELDNAMES = ("image_path", "label")
UNDO_HISTORY_FIELDNAMES = ("action", "image_path", "previous_label", "label")

config = load_config()
app = FastAPI(title="Image Classification")
app.state.working_folder = Path.cwd()
app.add_middleware(SessionMiddleware, secret_key=config["session_secret"], https_only=False)
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


class LabelSubmission(BaseModel):
  image_path: str
  label: str


def is_authenticated(request: Request) -> bool:
  return request.session.get("authenticated") is True


def require_authentication(request: Request) -> None:
  if not is_authenticated(request):
    raise HTTPException(status_code=303, headers={"Location": "/login"})


def parse_startup_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--working-folder",
    default=Path.cwd(),
    help="Directory containing classification_config.yaml and classification.csv.",
  )
  options = parser.parse_args(arguments)
  working_folder = Path(options.working_folder).expanduser().resolve()
  working_folder.mkdir(parents=True, exist_ok=True)
  if not working_folder.is_dir():
    parser.error(f"--working-folder must be a directory: {working_folder}")
  options.working_folder = working_folder
  return options


def load_classification_config(working_folder: Path) -> dict[str, list[str]]:
  config_path = working_folder / "classification_config.yaml"
  if not config_path.is_file():
    raise ValueError(f"Missing configuration file: {config_path}")

  with config_path.open(encoding="utf-8") as config_file:
    classification_config = yaml.safe_load(config_file)

  if not isinstance(classification_config, dict):
    raise ValueError("classification_config.yaml must contain a mapping")

  labels = classification_config.get("label_list")
  if not isinstance(labels, list) or not all(
    isinstance(label, str) and label.strip() for label in labels
  ):
    raise ValueError("label_list must be a list of non-empty strings")
  if not 1 <= len(labels) <= 9:
    raise ValueError("label_list must contain between 1 and 9 labels")
  if len(set(labels)) != len(labels):
    raise ValueError("label_list must not contain duplicate labels")

  folder_paths = classification_config.get("image_folder_path_list")
  if not isinstance(folder_paths, list) or not folder_paths or not all(
    isinstance(folder_path, str) and folder_path.strip() for folder_path in folder_paths
  ):
    raise ValueError("image_folder_path_list must be a non-empty list of paths")

  resolved_folders: list[str] = []
  for folder_path in folder_paths:
    resolved_folder = Path(folder_path).expanduser()
    if not resolved_folder.is_absolute():
      resolved_folder = working_folder / resolved_folder
    resolved_folder = resolved_folder.resolve()
    if not resolved_folder.is_dir():
      raise ValueError(f"Image folder does not exist or is not a directory: {resolved_folder}")
    resolved_folders.append(str(resolved_folder))

  return {"label_list": labels, "image_folder_path_list": resolved_folders}


def read_classification_rows(working_folder: Path) -> list[dict[str, str]]:
  csv_path = working_folder / "classification.csv"
  if not csv_path.exists():
    return []

  with csv_path.open(newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    if tuple(reader.fieldnames or ()) != CSV_FIELDNAMES:
      raise ValueError("classification.csv must have exactly these columns: image_path,label")

    rows: list[dict[str, str]] = []
    image_paths: set[str] = set()
    for row_number, row in enumerate(reader, start=2):
      image_path = row.get("image_path")
      label = row.get("label")
      if not image_path or not label or not Path(image_path).is_absolute():
        raise ValueError(f"Invalid classification.csv row {row_number}")
      normalized_path = str(Path(image_path).resolve())
      if normalized_path in image_paths:
        raise ValueError(f"Duplicate image_path in classification.csv: {normalized_path}")
      image_paths.add(normalized_path)
      rows.append({"image_path": normalized_path, "label": label})
  return rows


def read_classifications(working_folder: Path) -> dict[str, str]:
  return {
    row["image_path"]: row["label"]
    for row in read_classification_rows(working_folder)
  }


def rewrite_csv_rows(
  csv_path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
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


def rewrite_classification_rows(working_folder: Path, rows: list[dict[str, str]]) -> None:
  rewrite_csv_rows(working_folder / "classification.csv", CSV_FIELDNAMES, rows)


def read_undo_history(working_folder: Path) -> list[dict[str, str]]:
  history_path = working_folder / "classification_undo_history.csv"
  if not history_path.exists():
    return []

  with history_path.open(newline="", encoding="utf-8") as history_file:
    reader = csv.DictReader(history_file)
    if tuple(reader.fieldnames or ()) != UNDO_HISTORY_FIELDNAMES:
      raise ValueError(
        "classification_undo_history.csv must have action,image_path,previous_label,label columns"
      )

    history: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
      action = row.get("action")
      image_path = row.get("image_path")
      previous_label = row.get("previous_label")
      label = row.get("label")
      if (
        action not in {"create", "update"}
        or not image_path
        or not Path(image_path).is_absolute()
        or not label
        or (action == "create" and previous_label)
        or (action == "update" and not previous_label)
      ):
        raise ValueError(f"Invalid classification_undo_history.csv row {row_number}")
      history.append({
        "action": action,
        "image_path": str(Path(image_path).resolve()),
        "previous_label": previous_label or "",
        "label": label,
      })
  return history


def rewrite_undo_history(working_folder: Path, history: list[dict[str, str]]) -> None:
  rewrite_csv_rows(
    working_folder / "classification_undo_history.csv", UNDO_HISTORY_FIELDNAMES, history
  )


def record_undo_action(
  working_folder: Path, action: str, image_path: Path, previous_label: str, label: str
) -> None:
  history = read_undo_history(working_folder)
  history.append({
    "action": action,
    "image_path": str(image_path),
    "previous_label": previous_label,
    "label": label,
  })
  rewrite_undo_history(working_folder, history)


def discover_images(image_folder_paths: list[str]) -> list[Path]:
  image_paths: set[Path] = set()
  for folder_path in image_folder_paths:
    for image_path in Path(folder_path).rglob("*"):
      if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
        image_paths.add(image_path.resolve())
  return sorted(image_paths)


def get_unclassified_images(working_folder: Path) -> tuple[dict[str, list[str]], list[Path]]:
  classification_config = load_classification_config(working_folder)
  classifications = read_classifications(working_folder)
  images = discover_images(classification_config["image_folder_path_list"])
  return classification_config, [image for image in images if str(image) not in classifications]


def count_classifications_by_label(labels: list[str], rows: list[dict[str, str]]) -> dict[str, int]:
  counts = {label: 0 for label in labels}
  for row in rows:
    if row["label"] in counts:
      counts[row["label"]] += 1
  return counts


def is_reviewable_image(image_path: Path, image_folder_paths: list[str]) -> bool:
  return (
    image_path.is_file()
    and image_path.suffix.lower() in IMAGE_EXTENSIONS
    and is_configured_image(image_path, image_folder_paths)
  )


def get_review_items(
  working_folder: Path, label: str, offset: int, limit: int
) -> tuple[list[dict[str, str | bool]], int]:
  classification_config = load_classification_config(working_folder)
  if label not in classification_config["label_list"]:
    raise ValueError("Label is not configured")

  rows = sorted(
    (row for row in read_classification_rows(working_folder) if row["label"] == label),
    key=lambda row: row["image_path"],
  )
  items: list[dict[str, str | bool]] = []
  for row in rows[offset:offset + limit]:
    image_path = Path(row["image_path"])
    items.append({
      "image_path": row["image_path"],
      "label": row["label"],
      "missing": not is_reviewable_image(
        image_path, classification_config["image_folder_path_list"]
      ),
    })
  return items, len(rows)


def is_configured_image(image_path: Path, image_folder_paths: list[str]) -> bool:
  for folder_path in image_folder_paths:
    try:
      image_path.relative_to(Path(folder_path))
      return True
    except ValueError:
      continue
  return False


def append_classification(working_folder: Path, image_path: Path, label: str) -> None:
  rows = read_classification_rows(working_folder)
  rows.append({"image_path": str(image_path), "label": label})
  rewrite_classification_rows(working_folder, rows)


def remove_last_classification(working_folder: Path) -> dict[str, str] | None:
  history = read_undo_history(working_folder)
  if not history:
    return None

  action = history[-1]
  rows = read_classification_rows(working_folder)
  row_index = next(
    (index for index, row in enumerate(rows) if row["image_path"] == action["image_path"]),
    None,
  )
  if row_index is None or rows[row_index]["label"] != action["label"]:
    raise ValueError("Latest undo action does not match classification.csv")

  if action["action"] == "create":
    restored_row = rows.pop(row_index)
  else:
    rows[row_index]["label"] = action["previous_label"]
    restored_row = rows[row_index]
  rewrite_classification_rows(working_folder, rows)
  rewrite_undo_history(working_folder, history[:-1])
  return restored_row


def update_classification_label(working_folder: Path, image_path: Path, label: str) -> bool:
  rows = read_classification_rows(working_folder)
  row_index = next(
    (index for index, row in enumerate(rows) if row["image_path"] == str(image_path)),
    None,
  )
  if row_index is None:
    raise ValueError("Image does not have a classification")
  previous_label = rows[row_index]["label"]
  if previous_label == label:
    return False

  rows[row_index]["label"] = label
  rewrite_classification_rows(working_folder, rows)
  record_undo_action(working_folder, "update", image_path, previous_label, label)
  return True


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
def home(request: Request, image_path: str | None = None) -> HTMLResponse:
  working_folder = request.app.state.working_folder
  try:
    classification_config, unclassified_images = get_unclassified_images(working_folder)
    label_counts = count_classifications_by_label(
      classification_config["label_list"], read_classification_rows(working_folder)
    )
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error

  selected_image_path = Path(image_path).resolve() if image_path else None
  if selected_image_path and selected_image_path not in unclassified_images:
    raise HTTPException(status_code=404, detail="Image is not available for classification")
  selected_image_path = selected_image_path or (
    random.choice(unclassified_images) if unclassified_images else None
  )
  return templates.TemplateResponse(
    request=request,
    name="home.html",
    context={
      "labels": classification_config["label_list"],
      "label_counts": label_counts,
      "image_path": str(selected_image_path) if selected_image_path else None,
    },
  )


@protected_pages.get("/image")
def image(request: Request, image_path: str) -> FileResponse:
  working_folder = request.app.state.working_folder
  resolved_image_path = Path(image_path).resolve()
  try:
    classification_config, unclassified_images = get_unclassified_images(working_folder)
    classifications = read_classifications(working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  is_unclassified = resolved_image_path in unclassified_images
  is_classified = str(resolved_image_path) in classifications
  if not (is_unclassified or is_classified) or not is_reviewable_image(
    resolved_image_path, classification_config["image_folder_path_list"]
  ):
    raise HTTPException(status_code=404, detail="Image is not available for classification")
  return FileResponse(resolved_image_path)


@protected_pages.post("/classifications")
def create_classification(request: Request, submission: LabelSubmission) -> dict[str, str]:
  working_folder = request.app.state.working_folder
  image_path = Path(submission.image_path).resolve()
  try:
    classification_config = load_classification_config(working_folder)
    classifications = read_classifications(working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error

  if submission.label not in classification_config["label_list"]:
    raise HTTPException(status_code=422, detail="Label is not configured")
  if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
    raise HTTPException(status_code=404, detail="Image does not exist")
  if not is_configured_image(image_path, classification_config["image_folder_path_list"]):
    raise HTTPException(status_code=404, detail="Image is not in a configured folder")
  if str(image_path) in classifications:
    raise HTTPException(status_code=409, detail="Image already has a classification")

  append_classification(working_folder, image_path, submission.label)
  record_undo_action(working_folder, "create", image_path, "", submission.label)
  return {"image_path": str(image_path), "label": submission.label}


@protected_pages.get("/review", response_class=HTMLResponse)
def review(request: Request, label: str) -> HTMLResponse:
  try:
    classification_config = load_classification_config(request.app.state.working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if label not in classification_config["label_list"]:
    raise HTTPException(status_code=404, detail="Label is not configured")
  return templates.TemplateResponse(
    request=request,
    name="review.html",
    context={"label": label, "labels": classification_config["label_list"]},
  )


@protected_pages.get("/review/items")
def review_items(
  request: Request,
  label: str,
  offset: int = Query(default=0, ge=0),
  limit: int = Query(default=100, ge=1, le=100),
) -> dict[str, object]:
  try:
    items, total = get_review_items(request.app.state.working_folder, label, offset, limit)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  next_offset = offset + len(items)
  return {
    "items": items,
    "total": total,
    "next_offset": next_offset if next_offset < total else None,
    "has_more": next_offset < total,
  }


@protected_pages.patch("/classifications")
def update_classification(request: Request, submission: LabelSubmission) -> dict[str, str | bool]:
  working_folder = request.app.state.working_folder
  image_path = Path(submission.image_path).resolve()
  try:
    classification_config = load_classification_config(working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if submission.label not in classification_config["label_list"]:
    raise HTTPException(status_code=422, detail="Label is not configured")
  if not is_reviewable_image(image_path, classification_config["image_folder_path_list"]):
    raise HTTPException(status_code=404, detail="Image is not available for review")
  try:
    updated = update_classification_label(working_folder, image_path, submission.label)
  except ValueError as error:
    raise HTTPException(status_code=409, detail=str(error)) from error
  return {"image_path": str(image_path), "label": submission.label, "updated": updated}


@protected_pages.post("/classifications/undo")
def undo_last_classification(request: Request) -> dict[str, str]:
  try:
    history = read_undo_history(request.app.state.working_folder)
    removed_row = remove_last_classification(request.app.state.working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if removed_row is None:
    raise HTTPException(status_code=409, detail="There is no classification to undo")
  return {**removed_row, "action": history[-1]["action"]}


app.include_router(protected_pages)


if __name__ == "__main__":
  startup_arguments = parse_startup_arguments()
  try:
    load_classification_config(startup_arguments.working_folder)
    read_classifications(startup_arguments.working_folder)
  except ValueError as error:
    raise SystemExit(f"Configuration error: {error}") from error
  os.chdir(startup_arguments.working_folder)
  app.state.working_folder = startup_arguments.working_folder
  uvicorn.run(app, host="0.0.0.0", port=8000)