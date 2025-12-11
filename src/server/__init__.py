from fastapi import FastAPI
from server.item import router as item_router
from server.web  import router as  web_router
from pipeline.processing_job import extract
app = FastAPI()

extract("sdfsd")
# @app.get("/")
# def read_root():
#     return {"Hello": "World Na Ja"}

app.include_router(item_router)
app.include_router(web_router)