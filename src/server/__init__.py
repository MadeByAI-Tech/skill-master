from fastapi import FastAPI
from server.web  import router as  web_router
app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"Hello": "World Na Ja"}

app.include_router(web_router)