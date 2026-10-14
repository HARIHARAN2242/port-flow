"""PortFlow operational planning helpers; estimates are planning aids, not live capacity data."""
import math

def calculate_trucks(container_count, avg_truck_capacity=2):
    trucks_needed = math.ceil(max(0, container_count) / max(1, avg_truck_capacity))
    slot_count = 4
    trucks_per_slot = math.ceil(trucks_needed / slot_count) if trucks_needed else 0
    slots = [
        f"{hour}:00–{hour}:30: {trucks_per_slot} trucks"
        for hour in [8, 8, 9, 9]
    ]
    slots = [
        f"08:00 AM–08:30 AM: {trucks_per_slot} trucks",
        f"08:30 AM–09:00 AM: {trucks_per_slot} trucks",
        f"09:00 AM–09:30 AM: {trucks_per_slot} trucks",
        f"09:30 AM–10:00 AM: {trucks_per_slot} trucks",
    ]
    return {"trucks_needed": trucks_needed, "slots": slots}

def check_reroute(predicted_congestion, neighbor_port, neighbor_capacity_pct):
    # Risk index is a model-derived dwell-risk proxy, not actual capacity occupancy.
    if predicted_congestion >= 80.0:
        return {
            "reroute_flag": True,
            "target_port": neighbor_port,
            "target_capacity": neighbor_capacity_pct,
            "message": (
                f"High dwell-risk proxy: assess {neighbor_port} as an option only after "
                f"confirming live receiving capacity with the port operator."
            ),
        }
    return {
        "reroute_flag": False,
        "target_port": "N/A",
        "target_capacity": 0,
        "message": "The benchmark dwell-risk index is below the configured alert threshold; verify live port conditions.",
    }
