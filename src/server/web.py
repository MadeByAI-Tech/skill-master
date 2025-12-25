from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

router = APIRouter()
# router.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="server/templates")


@router.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@router.get("/example/{id}", response_class=HTMLResponse)
async def get_example(request: Request, id:int):
    return templates.TemplateResponse(request=request, name="example.html", context={"id":id})