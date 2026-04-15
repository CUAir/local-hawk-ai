from enum import Enum
from typing import Tuple

class Item(Enum):
    """
    Enum to store the classification of a target.
    """
        
    TENT = 0
    MANNEQUIN = 1
    UNKNOWN = 2

    def __str__(self) -> str:
        return self.name

    def to_string(self):
        return str(self.value)

class Classification:
    """
    Class to store the classification of a target.
    """
    def __init__(
        self,
        item: str = Item.UNKNOWN,
        number_conf: float = 0.0,
    ):
        self.number: Tuple[Item, float] = (item, number_conf)