from enum import Enum
from typing import Tuple

class LabelType(Enum):
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
        label: str = LabelType.UNKNOWN,
        number_conf: float = 0.0,
    ):
        self.label: Tuple[LabelType, float] = (label, number_conf)
    
    def __str__(self) -> str:
        return "Classification(label={}, number_conf={})".format(self.label[0], self.label[1])