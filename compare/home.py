import csv
import random
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))
IMAGE_EXTENSIONS = {".gif", ".jpeg", ".jpg", ".png", ".webp"}
IMPORT_IMAGE_COUNT = 10
MAX_IMAGE_DIMENSION = 384
TARGET_IMAGE_AREA = 65_536


class ImageFolderCreate(BaseModel):
  label: str
  path: str
  x: float
  y: float


class ImageFolder(ImageFolderCreate):
  uuid: str


class ImportedImage(BaseModel):
  path: str
  width: int
  height: int


def read_image_folders(working_folder: Path) -> list[ImageFolder]:
  csv_path = working_folder / "image_folders.csv"
  if not csv_path.exists():
    return []

  with csv_path.open(encoding="utf-8", newline="") as csv_file:
    return [ImageFolder.model_validate(row) for row in csv.DictReader(csv_file)]


def find_image_folder(working_folder: Path, folder_id: str) -> ImageFolder:
  for image_folder in read_image_folders(working_folder):
    if image_folder.uuid == folder_id:
      return image_folder
  raise HTTPException(status_code=404, detail="Image folder not found")


def display_dimensions(width: int, height: int) -> tuple[int, int]:
  scale = (TARGET_IMAGE_AREA / (width * height)) ** .5
  scale = min(scale, MAX_IMAGE_DIMENSION / width, MAX_IMAGE_DIMENSION / height)
  return max(1, round(width * scale)), max(1, round(height * scale))


@router.get("/", response_class=HTMLResponse, response_model=None)
def home(request: Request) -> HTMLResponse:
  return templates.TemplateResponse(request=request, name="home.html")


@router.get("/image-folders")
def list_image_folders(request: Request) -> list[ImageFolder]:
  return read_image_folders(request.app.state.working_folder)


@router.post("/image-folders/{folder_id}/import")
def import_images(request: Request, folder_id: str) -> list[ImportedImage]:
  image_folder = find_image_folder(request.app.state.working_folder, folder_id)
  folder_path = Path(image_folder.path).expanduser().resolve()
  if not folder_path.is_dir():
    raise HTTPException(status_code=404, detail="Image directory not found")

  image_paths = [
    path for path in folder_path.rglob("*")
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
  ]
  selected_paths = random.sample(image_paths, k=min(IMPORT_IMAGE_COUNT, len(image_paths)))
  imported_images: list[ImportedImage] = []
  for image_path in selected_paths:
    try:
      with Image.open(image_path) as image:
        width, height = display_dimensions(*image.size)
    except UnidentifiedImageError:
      continue
    imported_images.append(ImportedImage(
      path=str(image_path.relative_to(folder_path)), width=width, height=height
    ))
  return imported_images


@router.get("/image-folders/{folder_id}/images/{image_path:path}")
def get_imported_image(request: Request, folder_id: str, image_path: str) -> FileResponse:
  image_folder = find_image_folder(request.app.state.working_folder, folder_id)
  folder_path = Path(image_folder.path).expanduser().resolve()
  requested_path = (folder_path / image_path).resolve()
  if (
    not folder_path.is_dir()
    or folder_path not in requested_path.parents
    or not requested_path.is_file()
    or requested_path.suffix.lower() not in IMAGE_EXTENSIONS
  ):
    raise HTTPException(status_code=404, detail="Image not found")
  return FileResponse(requested_path)


@router.post("/image-folders")
def create_image_folder(request: Request, image_folder: ImageFolderCreate) -> dict[str, str]:
  folder_id = str(uuid4())
  csv_path = request.app.state.working_folder / "image_folders.csv"
  write_header = not csv_path.exists()

  with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=("uuid", "label", "path", "x", "y"))
    if write_header:
      writer.writeheader()
    writer.writerow({
      "uuid": folder_id,
      "label": image_folder.label,
      "path": image_folder.path,
      "x": image_folder.x,
      "y": image_folder.y,
    })

  return {"uuid": folder_id}