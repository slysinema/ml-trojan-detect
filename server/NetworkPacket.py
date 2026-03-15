from pydantic import BaseModel

class NetworkPacket(BaseModel):
    features: list[float]
