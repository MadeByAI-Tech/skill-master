from pydantic import BaseModel
from fastapi import APIRouter
from typing import Union
router = APIRouter()

class Item(BaseModel):
    item_id: int
    q: str|None

@router.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None) -> Item:
    item = Item(item_id=item_id, q=q)
    return item