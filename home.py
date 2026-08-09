import csv
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


class ImageFolderCreate(BaseModel):
  label: str
  path: str
  x: float
  y: float


class ImageFolder(ImageFolderCreate):
  uuid: str


@router.get("/", response_class=HTMLResponse, response_model=None)
def home(request: Request) -> HTMLResponse:
  return templates.TemplateResponse(request=request, name="home.html")


@router.get("/image-folders")
def list_image_folders(request: Request) -> list[ImageFolder]:
  csv_path = request.app.state.working_folder / "image_folders.csv"
  if not csv_path.exists():
    return []

  with csv_path.open(encoding="utf-8", newline="") as csv_file:
    return [ImageFolder.model_validate(row) for row in csv.DictReader(csv_file)]


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