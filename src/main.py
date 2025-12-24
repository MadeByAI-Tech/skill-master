import flask
import functions_framework
import llm # type: ignore
import os
from pipeline.processing_job import job_post_extraction_pipeline

print(f"MODE={os.environ['MODE']}")


@functions_framework.http # type: ignore
def main(request: flask.Request) -> flask.typing.ResponseReturnValue: # type: ignore
    function:str|None = request.args.get("function")
    if(function=="greet"):
        return greet()
    else:
        return job_post_extraction_pipeline()

def starting_point():
    result:str = "" # type: ignore
    result += "Starting Point Reached"
    result += f"</br>{greet()}"
    return result

def greet() -> str:
    return "Greetings!"
