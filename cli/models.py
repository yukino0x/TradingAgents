from enum import Enum
from typing import List, Optional, Dict
from pydantic import BaseModel


class AnalystType(str, Enum):
    MARKET = "market"
    PRICE_ACTION = "price_action"
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
