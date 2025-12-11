from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

class Item(BaseModel):
    item_id: int
    q: str|None

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None) -> Item:
    item = Item(item_id=item_id, q=q)
    return item