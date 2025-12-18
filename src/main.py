import flask
import functions_framework
from src import llm # type: ignore
import os

print(f"MODE={os.environ['MODE']}")


@functions_framework.http # type: ignore
def main(request: flask.Request) -> flask.typing.ResponseReturnValue: # type: ignore
    function:str|None = request.args.get("function")
    if(function=="greet"):
        return greet()
    else:
        return starting_point()

def starting_point():
    result:str = "" # type: ignore
    result += "Starting Point Reached"
    result += f"</br>{greet()}"
    return result

def greet() -> str:
    return "Greetings!"
