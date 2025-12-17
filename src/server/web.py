from flask import Blueprint, render_template

router = Blueprint('web', __name__, template_folder='../templates')

@router.route("/", methods=["GET"])
async def get_index():
    return render_template("index.html")

@router.route("/example/<int:id>", methods=["GET"])
async def get_example(id:int):
    return render_template("example.html", id=id)