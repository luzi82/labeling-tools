from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from secrets import compare_digest

import uvicorn
import yaml
from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
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


def is_configured_image(image_path: Path, image_folder_paths: list[str]) -> bool:
  for folder_path in image_folder_paths:
    try:
      image_path.relative_to(Path(folder_path))
      return True
    except ValueError:
      continue
  return False


def append_classification(working_folder: Path, image_path: Path, label: str) -> None:
  csv_path = working_folder / "classification.csv"
  write_header = not csv_path.exists()
  with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    if write_header:
      writer.writeheader()
    writer.writerow({"image_path": str(image_path), "label": label})


def remove_last_classification(working_folder: Path) -> dict[str, str] | None:
  rows = read_classification_rows(working_folder)
  if not rows:
    return None

  removed_row = rows.pop()
  csv_path = working_folder / "classification.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    writer.writerows(rows)
  return removed_row


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
      "image_path": str(selected_image_path) if selected_image_path else None,
    },
  )


@protected_pages.get("/image")
def image(request: Request, image_path: str) -> FileResponse:
  working_folder = request.app.state.working_folder
  resolved_image_path = Path(image_path).resolve()
  try:
    classification_config, unclassified_images = get_unclassified_images(working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if resolved_image_path not in unclassified_images or not is_configured_image(
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
  return {"image_path": str(image_path), "label": submission.label}


@protected_pages.post("/classifications/undo")
def undo_last_classification(request: Request) -> dict[str, str]:
  try:
    removed_row = remove_last_classification(request.app.state.working_folder)
  except ValueError as error:
    raise HTTPException(status_code=422, detail=str(error)) from error
  if removed_row is None:
    raise HTTPException(status_code=409, detail="There is no classification to undo")
  return removed_row


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