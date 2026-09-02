import { useEffect, useRef, useState } from "react";
import { GeoJSON, MapContainer, Marker, TileLayer } from "react-leaflet";
import L, { type Layer, type LeafletMouseEvent } from "leaflet";
import type { Feature, FeatureCollection } from "geojson";
import type { ZoneInfo } from "../types";

const BOROUGH_COLORS: Record<string, string> = {
  Manhattan: "#6366f1",
  Brooklyn: "#22c55e",
  Queens: "#f59e0b",
  Bronx: "#ec4899",
  "Staten Island": "#06b6d4",
  EWR: "#94a3b8",
};

function pinIcon(color: string) {
  return L.divIcon({
    className: "",
    html: `<div style="width:18px;height:18px;border-radius:50% 50% 50% 0;background:${color};transform:rotate(-45deg);border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,0.4)"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 18],
  });
}

const PICKUP_ICON = pinIcon("#22c55e");
const DROPOFF_ICON = pinIcon("#ef4444");

interface Props {
  zones: ZoneInfo[];
  pickupId: number | null;
  dropoffId: number | null;
  onSelectZone: (locationId: number) => void;
}

export function TripMap({ zones, pickupId, dropoffId, onSelectZone }: Props) {
  const [geojson, setGeojson] = useState<FeatureCollection | null>(null);
  const geoJsonRef = useRef<L.GeoJSON | null>(null);

  useEffect(() => {
    fetch("/taxi_zones.geojson")
      .then((r) => r.json())
      .then(setGeojson);
  }, []);

  const zoneById = new Map(zones.map((z) => [z.location_id, z]));
  const pickupZone = pickupId ? zoneById.get(pickupId) : null;
  const dropoffZone = dropoffId ? zoneById.get(dropoffId) : null;

  const style = (feature?: Feature) => {
    const id = feature?.properties?.location_id;
    const borough = feature?.properties?.borough as string;
    const isSelected = id === pickupId || id === dropoffId;
    return {
      color: isSelected ? "#1e293b" : "#ffffff",
      weight: isSelected ? 2 : 0.6,
      fillColor: BOROUGH_COLORS[borough] ?? "#94a3b8",
      fillOpacity: isSelected ? 0.75 : 0.35,
    };
  };

  const onEachFeature = (feature: Feature, layer: Layer) => {
    layer.on({
      click: (e: LeafletMouseEvent) => {
        onSelectZone(feature.properties?.location_id);
        L.DomEvent.stopPropagation(e);
      },
      mouseover: (e: LeafletMouseEvent) => {
        (e.target as L.Path).setStyle({ fillOpacity: 0.9 });
      },
      mouseout: () => {
        geoJsonRef.current?.resetStyle(layer as L.Path);
      },
    });
    layer.bindTooltip(`${feature.properties?.zone} (${feature.properties?.borough})`, { sticky: true });
  };

  return (
    <MapContainer center={[40.735, -73.94]} zoom={11} className="h-full w-full rounded-2xl">
      <TileLayer
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />
      {geojson && (
        <GeoJSON
          ref={geoJsonRef}
          data={geojson}
          style={style}
          onEachFeature={onEachFeature}
          key={`${pickupId}-${dropoffId}`}
        />
      )}
      {pickupZone && <Marker position={[pickupZone.lat, pickupZone.lon]} icon={PICKUP_ICON} />}
      {dropoffZone && <Marker position={[dropoffZone.lat, dropoffZone.lon]} icon={DROPOFF_ICON} />}
    </MapContainer>
  );
}
