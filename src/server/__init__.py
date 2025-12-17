from flask import Flask
from server.web  import router as  web_router # type: ignore
# from pipeline.processing_job import extract
app = Flask(__name__)
app.register_blueprint(web_router)