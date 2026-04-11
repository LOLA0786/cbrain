from dataclasses import dataclass

@dataclass
class Decision:
    action: str
    risk: float
    decision: str
