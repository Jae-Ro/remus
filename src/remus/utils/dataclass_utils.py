from dataclasses import asdict
from typing import Any


class BaseSerial:
    """
    Base class for dataclasses to have a `.to_dict()` method for serialization
    """

    def to_dict(self, exclude_none: bool = False) -> dict[str, Any]:
        """General method to convert dataclass instance to dictionary

        :param exclude_none: flag to exlcude fields that have a value of None, defaults to False
        :return: dictionary representation of dataclass
        """
        d = asdict(self)

        if exclude_none:
            return {k: v for k, v in d.items() if v is not None}

        return d
