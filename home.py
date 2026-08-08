from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).with_name("templates"))


@router.get("/", response_class=HTMLResponse, response_model=None)
def home(request: Request) -> HTMLResponse:
  return templates.TemplateResponse(request=request, name="home.html")