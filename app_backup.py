import os
import streamlit as st
import requests
import ml_model
import optimizer
from remedy_bot import get_bot_remedy, classify_query

st.set_page_config(page_title="PortFlow - Smart Maritime & Port Operations", layout="wide")
st.title("⚓ PortFlow — Smart Maritime & Port Operations")
st.caption("Historical port-dwell ML benchmark • Operational planning • Intelligent remedy assistant")
st.warning(
    "Model transparency: the public training dataset covers five US port complexes, not Tamil Nadu. "
    "The benchmark predictions shown here are not validated forecasts for Chennai, Ennore, Kattupalli or Tuticorin."
)

vessels = {
    "V101": {"vessel_name": "Chennai Express", "current_port": "Chennai Port", "container_count": 1200,
             "speed_knots": 14, "weather_score": 2, "neighbor_port": "Kamarajar Port (Ennore)", "neighbor_capacity": 40},
    "V102": {"vessel_name": "Coromandel Merchant", "current_port": "Kamarajar Port (Ennore)", "container_count": 2400,
             "speed_knots": 11, "weather_score": 6, "neighbor_port": "Kattupalli Port", "neighbor_capacity": 55},
    "V103": {"vessel_name": "Bay of Bengal Star", "current_port": "Kattupalli Port", "container_count": 3200,
             "speed_knots": 8, "weather_score": 9, "neighbor_port": "VOC Port (Tuticorin)", "neighbor_capacity": 25},
    "V104": {"vessel_name": "Tuticorin Cargo", "current_port": "VOC Port (Tuticorin)", "container_count": 1800,
             "speed_knots": 13, "weather_score": 3, "neighbor_port": "Chennai Port", "neighbor_capacity": 50},
}
tab1, tab2, tab3 = st.tabs(["📊 ML Operations Dashboard", "🤖 Remedy Assistant", "🧪 Model Evaluation"])

with tab1:
    st.subheader("PortFlow vessel operations")
    st.markdown("Select a Tamil Nadu demo vessel for the operational planner, or choose a public benchmark port for a model prediction.")
    c1, c2 = st.columns(2)
    with c1:
        selected_vessel = st.selectbox("Tamil Nadu demo vessel", list(vessels.keys()),
                                       format_func=lambda x: f"{x} — {vessels[x]['vessel_name']}")
        vessel = vessels[selected_vessel]
        st.write(f"**Port:** {vessel['current_port']}  ·  **Containers:** {vessel['container_count']:,} TEU")
        st.write(f"**Speed:** {vessel['speed_knots']} knots  ·  **Weather score:** {vessel['weather_score']}/10")
        truck_plan = optimizer.calculate_trucks(vessel["container_count"])
        st.metric("Planning truck count", f"{truck_plan['trucks_needed']:,}")
        with st.expander("Truck appointment plan"):
            for slot in truck_plan["slots"]:
                st.write("• " + slot)
    status = ml_model.get_dataset_status()
    with c2:
        if status.get("ready"):
            benchmark_ports = status.get("ports", [])
        else:
            try:
                import pandas as pd
                benchmark_ports = sorted(pd.read_csv(ml_model.DATA_PATH)["Port"].dropna().astype(str).unique().tolist())
            except Exception:
                benchmark_ports = ["Houston", "LA_Long_Beach", "NY_NJ", "Savannah", "Seattle"]
        selected_port = st.selectbox("Public-data benchmark port", benchmark_ports)
        month = st.selectbox("Prediction month", list(range(1, 13)), index=0,
                             format_func=lambda m: __import__("calendar").month_name[m])
        if st.button("Run trained ML prediction", type="primary"):
            try:
                prediction = ml_model.predict_port_dwell(selected_port, month_num=month)
                st.session_state["last_prediction"] = prediction
            except Exception as e:
                st.error(str(e))
        prediction = st.session_state.get("last_prediction")
        if prediction:
            st.metric("Predicted median dwell", f"{prediction['predicted_dwell_days']} days")
            st.metric("Relative dwell-risk index", f"{prediction['dwell_risk_index']}/100")
            st.caption(f"Historical median for this port: {prediction['port_historical_median_days']} days")
            st.info(prediction["warning"])
            st.session_state["portflow_context"] = {
                "Benchmark port": prediction["port"],
                "Predicted median dwell": f"{prediction['predicted_dwell_days']} days",
                "Relative dwell-risk index": f"{prediction['dwell_risk_index']}/100",
                "Selected Tamil Nadu demo vessel": vessel["vessel_name"],
                "Demo container count": vessel["container_count"],
            }
        else:
            st.info("Train the model and choose a benchmark port to see a historical-data prediction.")
    st.divider()
    st.subheader("Decision support")
    st.write("The truck plan is a simple container-to-truck calculation. It is not a live dispatch schedule.")
    st.write("Rerouting should be decided only after checking live berth availability, safety, commercial constraints and receiving-port capacity.")


# TAB 2: PORTFLOW AI CHATBOT
import os
import streamlit as st
from groq import Groq


# TAB 2: PORTFLOW AI CHATBOT
import os
import streamlit as st
from groq import Groq


# TAB 2: PORTFLOW AI CHATBOT

with tab2:
    st.subheader("🤖 PortFlow AI Remedy Assistant")
    st.write(
        "Ask questions about port congestion, vessel delays, "
        "truck scheduling, weather disruptions, and maritime logistics."
    )

    # Update this name if `ollama list` shows a different model name.
    MODEL_NAME = "deepseek-r1:1.5b"
    OLLAMA_URL = "http://localhost:11434/api/chat"

    # Initialize conversation history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            {
                "role": "system",
                "content": (
                    "You are PortFlow AI, a helpful maritime logistics "
                    "and port operations assistant. Answer in clear, "
                    "simple English. Give practical numbered steps. "
                    "Never invent live port data or claim predictions "
                    "are verified when they are not. Respect port safety."
                ),
            }
        ]

    # Display conversation, excluding system instructions
    for msg in st.session_state.ai_chat_history:
        if msg["role"] == "system":
            continue

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input(
        "Ask PortFlow anything about port operations..."
    )

    if user_query:
        st.session_state.ai_chat_history.append(
            {"role": "user", "content": user_query}
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            try:
                with st.spinner("DeepSeek is thinking..."):
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": MODEL_NAME,
                            "messages": st.session_state.ai_chat_history,
                            "stream": False,
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    response_text = response.json()["message"]["content"]

                st.markdown(response_text)

            except requests.exceptions.ConnectionError:
                response_text = (
                    "Cannot connect to Ollama. Please make sure "
                    "Ollama is running on this computer."
                )
                st.error(response_text)

            except requests.exceptions.Timeout:
                response_text = (
                    "DeepSeek took too long to respond. Please try again."
                )
                st.error(response_text)

            except (requests.exceptions.RequestException, KeyError, ValueError) as e:
                response_text = (
                    "Could not get a response from DeepSeek. "
                    "Check the model name and Ollama status."
                )
                st.error(response_text)
                st.caption(f"Technical details: {e}")

        st.session_state.ai_chat_history.append(
            {"role": "assistant", "content": response_text}
        )

    if st.button("Clear conversation"):
        st.session_state.ai_chat_history = [
            {
                "role": "system",
                "content": (
                    "You are PortFlow AI, a helpful maritime logistics "
                    "and port operations assistant. Give practical, "
                    "clear answers and never invent live port data."
                ),
            }
        ]
        st.rerun()

with tab3:
    st.subheader("🧪 Training and evaluation")
    status = ml_model.get_dataset_status()
    if status.get("ready"):
        st.success("A saved trained model was found.")
        metrics = status
        a, b, c = st.columns(3)
        a.metric("Held-out MAE", f"{metrics.get('mae_days', 'N/A')} days")
        b.metric("Held-out RMSE", f"{metrics.get('rmse_days', 'N/A')} days")
        c.metric("Held-out R²", str(metrics.get("r2", "N/A")))
        st.write(f"Training records: {metrics.get('rows', 'N/A')} · Test records: {metrics.get('test_rows', 'N/A')}")
        st.caption("Metrics use a chronological holdout from the US benchmark dataset; they do not measure performance on Tamil Nadu ports.")
    else:
        st.warning("The model has not been trained yet.")
        st.code("python train_model.py", language="bash")
        st.write(status.get("error", "Place the downloaded CSV beside app.py, then train the model."))
    st.markdown("**Public training source:** [Zenodo historical port dwell dataset](https://zenodo.org/records/21203605)")
    st.markdown("**Important:** for genuinely local Tamil Nadu forecasts, collect historical port-level records for Chennai, Kamarajar/Ennore, Kattupalli and VOC/Tuticorin, including observed dwell/arrival delay outcomes.")
