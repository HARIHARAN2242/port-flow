import os
from pathlib import Path
import calendar
import json
import time
import math

import streamlit as st
import pandas as pd
import requests
import websocket
import folium
from streamlit_folium import st_folium

import ml_model
import optimizer


# -----------------------------------------------------------------------------
# Distance Helper (Haversine Formula)
# -----------------------------------------------------------------------------
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates distance between two points on Earth in Nautical Miles (NM) and Kilometers (km).
    """
    R_km = 6371.0  # Earth radius in kilometers
    R_nm = 3440.06 # Earth radius in nautical miles

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R_km * c
    distance_nm = R_nm * c
    return round(distance_nm, 2), round(distance_km, 2)


# -----------------------------------------------------------------------------
# Page configuration and application data
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PortFlow - Smart Maritime & Port Operations",
    layout="wide",
)

st.title("⚓ PortFlow — Smart Maritime & Port Operations")
st.caption(
    "Historical port-dwell ML benchmark • Operational planning • "
    "Intelligent remedy assistant"
)
st.warning(
    "Model transparency: the public training dataset covers five US port complexes, "
    "not Tamil Nadu. Benchmark predictions are not validated forecasts for Chennai, "
    "Ennore, Kattupalli or Tuticorin. Ship markers are illustrative unless populated "
    "from a real position feed or ships.csv."
)

PORTS = {
    "Chennai Port": {"latitude": 13.1007, "longitude": 80.3009},
    "Kamarajar Port (Ennore)": {"latitude": 13.2470, "longitude": 80.3350},
    "Kattupalli Port": {"latitude": 13.3300, "longitude": 80.3500},
    "VOC Port (Tuticorin)": {"latitude": 8.7642, "longitude": 78.1348},
}

VESSELS = {
    "V101": {
        "vessel_name": "Chennai Express",
        "current_port": "Chennai Port",
        "container_count": 1200,
        "speed_knots": 14,
        "weather_score": 2,
        "neighbor_port": "Kamarajar Port (Ennore)",
        "neighbor_capacity": 40,
    },
    "V102": {
        "vessel_name": "Coromandel Merchant",
        "current_port": "Kamarajar Port (Ennore)",
        "container_count": 2400,
        "speed_knots": 11,
        "weather_score": 6,
        "neighbor_port": "Kattupalli Port",
        "neighbor_capacity": 55,
    },
    "V103": {
        "vessel_name": "Bay of Bengal Star",
        "current_port": "Kattupalli Port",
        "container_count": 3200,
        "speed_knots": 8,
        "weather_score": 9,
        "neighbor_port": "VOC Port (Tuticorin)",
        "neighbor_capacity": 25,
    },
    "V104": {
        "vessel_name": "Tuticorin Cargo",
        "current_port": "VOC Port (Tuticorin)",
        "container_count": 1800,
        "speed_knots": 13,
        "weather_score": 3,
        "neighbor_port": "Chennai Port",
        "neighbor_capacity": 50,
    },
}


def load_ship_positions():
    csv_path = Path(__file__).resolve().parent / "ships.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    required = {"vessel_name", "latitude", "longitude"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "ships.csv is missing required column(s): " + ", ".join(sorted(missing))
        )

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    ]
    return df


def ask_deepseek(messages):
    api_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")

    response = requests.post(
        api_url,
        json={
            "model": model_name,
            "messages": messages,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


@st.dialog("PortFlow asset details", width="large")
def show_asset_details():
    selected_key = st.session_state.get("selected_entity_key")
    all_details = st.session_state.get("portflow_asset_details", {})
    details = all_details.get(selected_key, {})

    if not details:
        st.info("No additional details are available for this selection.")
    else:
        st.subheader(str(details.get("Name", selected_key)))
        st.caption(str(details.get("Data quality note", "Details available from the current app data.")))

        if details.get("Asset type") == "Ship":
            m1, m2, m3 = st.columns(3)
            m1.metric("Containers (TEU)", details.get("Container count (TEU)", "Not provided"))
            m2.metric("Speed (knots)", details.get("Speed (knots)", "Not provided"))
            m3.metric("Weather score", details.get("Weather score (0–10)", "Not provided"))

        st.markdown("#### Details")
        display_items = [
            (label, value)
            for label, value in details.items()
            if label not in {"Name", "Data quality note", "Asset type"}
            and value is not None
            and str(value).strip() not in {"", "nan", "None"}
        ]
        left, right = st.columns(2)
        for i, (label, value) in enumerate(display_items):
            target = left if i % 2 == 0 else right
            with target:
                st.markdown(f"**{label}**")
                st.write(str(value))

    if st.button("Close details", key="close_portflow_asset_details", type="primary"):
        st.session_state["show_asset_dialog"] = False
        st.rerun()


# -----------------------------------------------------------------------------
# Main tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 ML Operations Dashboard", "🤖 Remedy Assistant", "🧪 Model Evaluation"]
)


# -----------------------------------------------------------------------------
# Tab 1: Operations dashboard, truck plan, model prediction, Leaflet map
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("PortFlow vessel operations")
    st.markdown(
        "Select a Tamil Nadu demo vessel for the operational planner, or choose "
        "a public benchmark port for a model prediction."
    )

    col_vessel, col_prediction = st.columns(2)

    with col_vessel:
        selected_vessel = st.selectbox(
            "Tamil Nadu demo vessel",
            list(VESSELS.keys()),
            format_func=lambda vessel_id: (
                f"{vessel_id} — {VESSELS[vessel_id]['vessel_name']}"
            ),
        )
        vessel = VESSELS[selected_vessel]
        st.write(
            f"**Port:** {vessel['current_port']} · "
            f"**Containers:** {vessel['container_count']:,} TEU"
        )
        st.write(
            f"**Speed:** {vessel['speed_knots']} knots · "
            f"**Weather score:** {vessel['weather_score']}/10"
        )

        try:
            truck_plan = optimizer.calculate_trucks(vessel["container_count"])
            st.metric("Planning truck count", f"{truck_plan['trucks_needed']:,}")
            with st.expander("Truck appointment plan"):
                for slot in truck_plan.get("slots", []):
                    st.write("• " + str(slot))
        except Exception as exc:
            st.error(f"Could not calculate truck plan: {exc}")

    try:
        dataset_status = ml_model.get_dataset_status()
        dataset_status = dataset_status or {}
    except Exception as exc:
        dataset_status = {"ready": False, "error": str(exc)}

    with col_prediction:
        if dataset_status.get("ready"):
            benchmark_ports = dataset_status.get("ports", [])
        else:
            try:
                data_path = getattr(ml_model, "DATA_PATH", None)
                if data_path and Path(data_path).exists():
                    benchmark_df = pd.read_csv(data_path)
                    benchmark_ports = sorted(
                        benchmark_df["Port"].dropna().astype(str).unique().tolist()
                    )
                else:
                    benchmark_ports = [
                        "Houston", "LA_Long_Beach", "NY_NJ", "Savannah", "Seattle"
                    ]
            except Exception:
                benchmark_ports = [
                    "Houston", "LA_Long_Beach", "NY_NJ", "Savannah", "Seattle"
                ]

        if not benchmark_ports:
            benchmark_ports = [
                "Houston", "LA_Long_Beach", "NY_NJ", "Savannah", "Seattle"
            ]

        selected_port = st.selectbox("Public-data benchmark port", benchmark_ports)
        selected_month = st.selectbox(
            "Prediction month",
            list(range(1, 13)),
            index=0,
            format_func=lambda month_num: calendar.month_name[month_num],
        )

        if st.button("Run trained ML prediction", type="primary"):
            try:
                prediction = ml_model.predict_port_dwell(
                    selected_port, month_num=selected_month
                )
                st.session_state["last_prediction"] = prediction
                st.session_state["last_prediction_error"] = None
            except Exception as exc:
                st.session_state["last_prediction_error"] = str(exc)

        if st.session_state.get("last_prediction_error"):
            st.error(st.session_state["last_prediction_error"])

        prediction = st.session_state.get("last_prediction")
        if prediction:
            st.metric(
                "Predicted median dwell",
                f"{prediction['predicted_dwell_days']} days",
            )
            st.metric(
                "Relative dwell-risk index",
                f"{prediction['dwell_risk_index']}/100",
            )
            st.caption(
                "Historical median for this benchmark port: "
                f"{prediction['port_historical_median_days']} days"
            )
            st.info(prediction.get("warning", "Use this as a benchmark, not a verified forecast."))
            st.session_state["portflow_context"] = {
                "Benchmark port": prediction.get("port", selected_port),
                "Predicted median dwell": (
                    f"{prediction['predicted_dwell_days']} days"
                ),
                "Relative dwell-risk index": (
                    f"{prediction['dwell_risk_index']}/100"
                ),
                "Selected Tamil Nadu demo vessel": vessel["vessel_name"],
                "Demo container count": vessel["container_count"],
            }
        else:
            st.info(
                "Run a model prediction to see a historical-data benchmark. "
                + str(dataset_status.get("error", ""))
            )

    st.divider()
    st.subheader("Decision support")
    st.write(
        "The truck plan is a simple container-to-truck calculation. "
        "It is not a live dispatch schedule."
    )
    st.write(
        "Rerouting should only be decided after checking live berth availability, "
        "safety, commercial constraints and receiving-port capacity."
    )

    # -------------------------------------------------------------------------
    # Global port and vessel map: Datalastic port search + AISStream live AIS
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("🌍 Global ports & live AIS vessel map")
    st.caption(
        "Search the global port catalogue with Datalastic and request AIS messages from AISStream. "
        "Live coverage depends on provider reception, your API plan, and the selected area; this is not a guarantee of every vessel worldwide."
    )

    def _datalastic_get(endpoint, params=None):
        api_key = os.getenv("DATALASTIC_API_KEY", "").strip() or str(st.secrets.get("DATALASTIC_API_KEY", "")).strip()
        if not api_key:
            return None, "Datalastic API key is not configured. Set DATALASTIC_API_KEY in your environment or Streamlit secrets."
        try:
            resp = requests.get(
                f"https://api.datalastic.com/api/v0/{endpoint.lstrip('/')}",
                params=params or {}, headers={"x-api-key": api_key}, timeout=25,
            )
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("meta", {}).get("success") is False:
                return None, str(payload.get("meta", {}).get("message", "Datalastic request failed"))
            return payload.get("data", payload), None
        except requests.RequestException as exc:
            return None, f"Datalastic request failed: {exc}"
        except ValueError:
            return None, "Datalastic returned an invalid JSON response."

    def _first_value(row, *names, default=None):
        for name in names:
            value = row.get(name)
            if value not in (None, "", "nan"):
                return value
        return default

    def _normalize_port_results(payload):
        if isinstance(payload, dict):
            for key in ("ports", "port", "results", "items"):
                if isinstance(payload.get(key), list):
                    return payload[key]
            if any(k in payload for k in ("name", "port_name", "unlocode")):
                return [payload]
        if isinstance(payload, list):
            return payload
        return []

    def _fetch_aisstream(area_name, bbox, seconds=8):
        api_key = os.getenv("AISSTREAM_API_KEY", "").strip() or str(st.secrets.get("AISSTREAM_API_KEY", "")).strip()
        if not api_key:
            return [], "AISStream API key is not configured. Set AISSTREAM_API_KEY in your environment or Streamlit secrets."
        ws = None
        found = {}
        try:
            ws = websocket.create_connection(
                "wss://stream.aisstream.io/v0/stream", timeout=12,
                origin="https://aisstream.io",
            )
            ws.send(json.dumps({
                "APIKey": api_key,
                "BoundingBoxes": [bbox],
                "FilterMessageTypes": ["PositionReport", "StandardClassBPositionReport", "ExtendedClassBPositionReport", "ShipStaticData"],
            }))
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except ValueError:
                    continue
                message = item.get("Message", {})
                metadata = item.get("MetaData", {}) or item.get("Metadata", {}) or {}
                pos = None
                for msg_type, body in message.items():
                    if isinstance(body, dict) and any(k in body for k in ("Latitude", "latitude")):
                        pos = body
                        break
                lat = _first_value(pos or {}, "Latitude", "latitude", default=_first_value(metadata, "latitude", "Latitude"))
                lon = _first_value(pos or {}, "Longitude", "longitude", default=_first_value(metadata, "longitude", "Longitude"))
                try:
                    lat, lon = float(lat), float(lon)
                except (TypeError, ValueError):
                    continue
                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    continue
                mmsi = str(_first_value(pos or {}, "UserID", "MMSI", default=_first_value(metadata, "MMSI", "mmsi", default="unknown")))
                name = str(_first_value(metadata, "ShipName", "shipName", "name", default="Vessel " + mmsi)).strip()
                speed = _first_value(pos or {}, "Sog", "SOG", "SpeedOverGround")
                course = _first_value(pos or {}, "Cog", "COG", "CourseOverGround")
                timestamp = _first_value(metadata, "time_utc", "TimeUTC", "timestamp", default=item.get("MessageType", "AIS message received now"))
                found[mmsi] = {
                    "MMSI": mmsi, "Ship": name or ("Vessel " + mmsi),
                    "Latitude": lat, "Longitude": lon, "Speed (knots)": speed,
                    "Course (degrees)": course, "Last message": timestamp,
                    "Data source": "AISStream live WebSocket",
                }
            return list(found.values()), None
        except Exception as exc:
            return [], f"AISStream connection/capture failed: {exc}"
        finally:
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

    with st.expander("API connection status and setup", expanded=False):
        st.markdown(
            "- **Datalastic:** set `DATALASTIC_API_KEY` to enable global port search and port details. "
            "See [Datalastic API docs](https://docs.datalastic.com/).\n"
            "- **AISStream:** set `AISSTREAM_API_KEY` to enable live AIS position capture. "
            "See [AISStream docs](https://aisstream.io/documentation/).\n"
            "- Keys stay on the Python server; do not paste them into browser-side code or commit them to Git."
        )
        st.write("Datalastic key:", "Configured" if (os.getenv("DATALASTIC_API_KEY", "").strip() or str(st.secrets.get("DATALASTIC_API_KEY", "")).strip()) else "Not configured")
        st.write("AISStream key:", "Configured" if (os.getenv("AISSTREAM_API_KEY", "").strip() or str(st.secrets.get("AISSTREAM_API_KEY", "")).strip()) else "Not configured")

    col_search, col_area, col_capture = st.columns([2.2, 1.5, 1.1])
    with col_search:
        port_query = st.text_input("Search ports or countries worldwide", placeholder="e.g. Singapore, India, Japan, Rotterdam")
    with col_area:
        area_options = {
            "Bay of Bengal": [[5.0, 75.0], [25.0, 100.0]],
            "Indian Ocean": [[-35.0, 35.0], [30.0, 120.0]],
            "Europe & Mediterranean": [[25.0, -15.0], [72.0, 45.0]],
            "North America": [[5.0, -170.0], [75.0, -50.0]],
            "South America": [[-60.0, -85.0], [15.0, -30.0]],
            "East Asia & Pacific": [[-50.0, 95.0], [65.0, 180.0]],
            "World (provider permitting)": [[-80.0, -180.0], [80.0, 180.0]],
        }
        selected_area = st.selectbox("AIS capture area", list(area_options.keys()), index=0)
    with col_capture:
        st.write("")
        st.write("")
        fetch_live = st.button("Fetch live AIS", type="primary", use_container_width=True)

    if "global_port_rows" not in st.session_state:
        st.session_state.global_port_rows = []
    if "live_ais_rows" not in st.session_state:
        st.session_state.live_ais_rows = []
    if "global_api_status" not in st.session_state:
        st.session_state.global_api_status = "No API request yet."

    search_col, details_col = st.columns([1, 1])
    with search_col:
        if st.button("Search port catalogue", use_container_width=True):
            if not port_query.strip():
                st.warning("Enter a port name, country, or UN/LOCODE to search.")
            else:
                port_data, err = _datalastic_get(
                    "port_find", 
                    {"country": port_query.strip(), "name": port_query.strip(), "fuzzy": 1}
                )
                if err:
                    st.error(err)
                    st.session_state.global_api_status = err
                else:
                    rows = _normalize_port_results(port_data)
                    cleaned = []
                    for p in rows:
                        lat = _first_value(p, "lat", "latitude", "Lat")
                        lon = _first_value(p, "lon", "longitude", "Lon", "lng")
                        try:
                            lat, lon = float(lat), float(lon)
                        except (TypeError, ValueError):
                            continue
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            cleaned.append({
                                "Port": _first_value(p, "name", "port_name", default="Unknown port"),
                                "Country": _first_value(p, "country", "country_name", "country_iso", default="Not provided"),
                                "UN/LOCODE": _first_value(p, "unlocode", "UNLOCODE", default="Not provided"),
                                "Type": _first_value(p, "type", "port_type", default="Not provided"),
                                "Latitude": lat, "Longitude": lon,
                                "UUID": _first_value(p, "uuid", default=""),
                                "Data source": "Datalastic port_find API",
                            })
                    st.session_state.global_port_rows = cleaned
                    if cleaned:
                        st.session_state.global_api_status = f"Datalastic returned {len(cleaned)} port result(s) for '{port_query.strip()}'."
                    else:
                        st.session_state.global_api_status = f"No ports found for '{port_query.strip()}'."
                    st.session_state.pop("selected_global_port_index", None)
                    st.session_state.pop("selected_ship_index", None)
        st.caption("Port search returns matching catalogue results. The provider's complete port-list export is a separate API report and may consume credits.")
    with details_col:
        st.info(st.session_state.global_api_status)

    if fetch_live:
        with st.spinner(f"Capturing AIS messages for {selected_area} for 8 seconds…"):
            rows, err = _fetch_aisstream(selected_area, area_options[selected_area], seconds=8)
        if err:
            st.error(err)
            st.session_state.global_api_status = err
        else:
            st.session_state.live_ais_rows = rows
            st.session_state.global_api_status = f"Captured {len(rows)} unique vessel(s) from AISStream in {selected_area}. This is a short live sample, not a complete census."
            st.success(st.session_state.global_api_status)

    global_ports = st.session_state.global_port_rows
    live_ships = st.session_state.live_ais_rows

    base_ports = [
        {"Port": name, "Country": "India", "UN/LOCODE": "IN", "Type": "Seaport", "Latitude": coords["latitude"], "Longitude": coords["longitude"], "Data source": "Default Regional Port"}
        for name, coords in PORTS.items()
    ]

    all_map_ports = base_ports + global_ports

    demo_map_ships = []
    if not live_ships:
        for vessel_id, info in VESSELS.items():
            lat, lon = {"V101": (13.075, 80.325), "V102": (13.225, 80.36), "V103": (13.35, 80.375), "V104": (8.74, 78.16)}[vessel_id]
            demo_map_ships.append({
                "MMSI": vessel_id, "Ship": info["vessel_name"], "Latitude": lat, "Longitude": lon,
                "Speed (knots)": info.get("speed_knots"), "Course (degrees)": None,
                "Last message": "Demo data", "Data source": "Illustrative demo data — not live AIS",
            })
    map_ships = live_ships if live_ships else demo_map_ships

    # Calculate real-time distance matrix between all ships and ports
    calculated_distances = []
    for s in map_ships:
        s_name = s.get("Ship", "Unknown vessel")
        s_lat = float(s["Latitude"])
        s_lon = float(s["Longitude"])
        for p in all_map_ports:
            p_name = p.get("Port", "Unknown port")
            p_lat = float(p["Latitude"])
            p_lon = float(p["Longitude"])
            dist_nm, dist_km = calculate_haversine_distance(s_lat, s_lon, p_lat, p_lon)
            calculated_distances.append({
                "from_vessel": s_name,
                "to_port": p_name,
                "distance_nautical_miles": dist_nm,
                "distance_km": dist_km
            })

    st.markdown("#### Port register & search results")
    port_df = pd.DataFrame(all_map_ports)
    show_port_df = port_df[["Port", "Country", "UN/LOCODE", "Type", "Latitude", "Longitude"]]
    port_event = st.dataframe(
        show_port_df, 
        hide_index=True, 
        use_container_width=True, 
        on_select="rerun", 
        selection_mode="single-row", 
        key="global_port_table"
    )
    selected_rows = list(port_event.selection.rows)
    if selected_rows:
        st.session_state.selected_global_port_index = selected_rows[0]
        st.session_state.selected_ship_index = None

    st.markdown("#### AIS vessels received")
    ship_df = pd.DataFrame(map_ships)
    if not ship_df.empty:
        ship_event = st.dataframe(
            ship_df, 
            hide_index=True, 
            use_container_width=True, 
            on_select="rerun", 
            selection_mode="single-row", 
            key="global_ship_table"
        )
        selected_ship_rows = list(ship_event.selection.rows)
        if selected_ship_rows:
            st.session_state.selected_ship_index = selected_ship_rows[0]
            st.session_state.selected_global_port_index = None

    map_center = [13.1007, 80.3009]
    map_zoom = 7

    selected_port_index = st.session_state.get("selected_global_port_index")
    selected_ship_index = st.session_state.get("selected_ship_index")

    if selected_ship_index is not None and selected_ship_index < len(map_ships):
        selected_ship = map_ships[selected_ship_index]
        map_center = [float(selected_ship["Latitude"]), float(selected_ship["Longitude"])]
        map_zoom = 13
    elif selected_port_index is not None and selected_port_index < len(all_map_ports):
        selected_port = all_map_ports[selected_port_index]
        map_center = [float(selected_port["Latitude"]), float(selected_port["Longitude"])]
        map_zoom = 13

    global_map = folium.Map(
        location=map_center, zoom_start=map_zoom, tiles=None, control_scale=True,
        min_zoom=2, max_zoom=18, max_bounds=True, world_copy_jump=True,
    )
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", no_wrap=True, max_zoom=19).add_to(global_map)

    for i, port in enumerate(all_map_ports):
        port_name = str(port["Port"])
        popup = folium.Popup(
            f"<b>{port_name}</b><br>Country: {port.get('Country', 'N/A')}<br>UN/LOCODE: {port.get('UN/LOCODE', 'N/A')}<br>Type: {port.get('Type', 'Port')}<br>Coordinates: {port['Latitude']:.5f}, {port['Longitude']:.5f}",
            max_width=350,
        )
        icon_color = "darkred" if (selected_port_index is not None and i == selected_port_index) else "red"
        folium.Marker(
            [port["Latitude"], port["Longitude"]], tooltip=f"Port: {port_name}", popup=popup,
            icon=folium.Icon(color=icon_color, icon="anchor", prefix="fa"),
        ).add_to(global_map)

    for i, ship in enumerate(map_ships):
        name = str(ship.get("Ship", "Unknown vessel"))
        mmsi = str(ship.get("MMSI", "unknown"))
        source = str(ship.get("Data source", "Not provided"))
        speed = ship.get("Speed (knots)")
        course = ship.get("Course (degrees)")
        icon_color = "darkblue" if (selected_ship_index is not None and i == selected_ship_index) else "blue"
        folium.Marker(
            [float(ship["Latitude"]), float(ship["Longitude"])],
            tooltip=f"Ship: {name}",
            popup=folium.Popup(
                f"<b>{name}</b><br>MMSI/ID: {mmsi}<br>Speed: {speed if speed is not None else 'Not provided'} knots<br>Course: {course if course is not None else 'Not provided'}<br>Source: {source}",
                max_width=350,
            ),
            icon=folium.Icon(color=icon_color, icon="ship", prefix="fa"),
        ).add_to(global_map)

    folium.LayerControl(collapsed=True).add_to(global_map)
    st_folium(
        global_map, 
        use_container_width=True, 
        height=760, 
        key=f"portflow_global_map_{selected_area}_{len(all_map_ports)}_{len(map_ships)}_{selected_port_index}_{selected_ship_index}"
    )

    map_context = {
        "map_center": map_center,
        "port_search_results": all_map_ports[:100],
        "live_ais_vessels": live_ships[:300],
        "fallback_demo_vessels": demo_map_ships if not live_ships else [],
        "precalculated_vessel_port_distances": calculated_distances,
        "api_status": st.session_state.global_api_status,
        "selected_area": selected_area,
    }
    st.session_state["portflow_live_map_context"] = map_context
    st.caption(
        "Map fills the available page width. Select a port or vessel row in the tables above to pinpoint it on the map."
    )


# -----------------------------------------------------------------------------
# Tab 2: Remedy Assistant powered by local Ollama / DeepSeek
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🤖 PortFlow AI Remedy Assistant")
    st.write(
        "Ask questions about port congestion, vessel delays, truck scheduling, "
        "weather disruptions and maritime logistics."
    )

    model_name = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    map_context_for_ai = st.session_state.get("portflow_live_map_context", {})
    map_context_json = json.dumps(map_context_for_ai, ensure_ascii=False, default=str)[:24000]
    
    system_prompt = (
        "You are PortFlow AI, an assistant for maritime logistics and port operations.\n"
        "Instructions:\n"
        "1. GROUND ALL ANSWERS directly using the provided MAP DATA CONTEXT JSON below.\n"
        "2. For distance questions, ONLY use the `precalculated_vessel_port_distances` provided in the JSON data context.\n"
        "3. Do not confuse different ports or assume vessels are at 0 NM distance unless explicitly stated in the precalculated distance matrix.\n\n"
        "CURRENT MAP DATA CONTEXT (JSON):\n" + map_context_json
    )

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            {"role": "system", "content": system_prompt}
        ]
    elif not st.session_state.ai_chat_history or st.session_state.ai_chat_history[0].get("role") != "system":
        st.session_state.ai_chat_history.insert(
            0, {"role": "system", "content": system_prompt}
        )

    for message in st.session_state.ai_chat_history:
        if message.get("role") == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask PortFlow anything about port operations...")
    if user_query:
        contextual_user_query = (
            user_query + "\n\n[Current PortFlow map/API context: " + map_context_json + "]"
        )
        st.session_state.ai_chat_history.append(
            {"role": "user", "content": contextual_user_query}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner(f"{model_name} is thinking..."):
                    response_text = ask_deepseek(st.session_state.ai_chat_history)
                st.markdown(response_text)
            except requests.exceptions.ConnectionError:
                response_text = (
                    "Cannot connect to Ollama. Check that Ollama is running at "
                    f"{ollama_url}."
                )
                st.error(response_text)
            except Exception as exc:
                response_text = f"Ollama error: {exc}"
                st.error(response_text)

        st.session_state.ai_chat_history.append(
            {"role": "assistant", "content": response_text}
        )

    if st.button("Clear conversation", key="clear_portflow_chat"):
        st.session_state.ai_chat_history = [
            {"role": "system", "content": system_prompt}
        ]
        st.rerun()


# -----------------------------------------------------------------------------
# Tab 3: Model evaluation
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("🧪 Training and evaluation")
    try:
        eval_status = ml_model.get_dataset_status() or {}
    except Exception as exc:
        eval_status = {"ready": False, "error": str(exc)}

    if eval_status.get("ready"):
        st.success("A saved trained model was found.")
        metrics = eval_status
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Held-out MAE", f"{metrics.get('mae_days', 'N/A')} days")
        col_b.metric("Held-out RMSE", f"{metrics.get('rmse_days', 'N/A')} days")
        col_c.metric("Held-out R²", str(metrics.get("r2", "N/A")))
    else:
        st.warning("The model has not been trained or the dataset is not ready.")