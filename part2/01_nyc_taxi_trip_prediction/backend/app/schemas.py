from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    pickup_location_id: int = Field(..., description="NYC TLC taxi zone LocationID for pickup")
    dropoff_location_id: int = Field(..., description="NYC TLC taxi zone LocationID for dropoff")
    pickup_datetime: datetime = Field(..., description="Requested pickup time (used for hour/day-of-week features)")
    passenger_count: int = Field(1, ge=1, le=6)


class HistoricalStats(BaseModel):
    n_trips: int
    avg_duration_min: float
    avg_fare_amount: float
    avg_trip_distance_mi: float


class PredictResponse(BaseModel):
    predicted_duration_min: float
    predicted_fare_amount: float
    haversine_km: float
    pickup_zone: str
    dropoff_zone: str
    pickup_borough: str
    dropoff_borough: str
    is_rush_hour: bool
    is_airport_trip: bool
    historical: HistoricalStats | None = None


class ZoneInfo(BaseModel):
    location_id: int
    zone: str
    borough: str
    service_zone: str
    lat: float
    lon: float
