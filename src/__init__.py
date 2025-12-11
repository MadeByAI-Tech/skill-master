from fastapi import FastAPI
from src.item import router as item_router
from src.web  import router as  web_router
app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World Na Ja"}

app.include_router(item_router)
app.include_router(web_router)