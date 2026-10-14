"""
PortFlow remedy assistant: trained text intent classifier + operational retrieval + LLM fallback.
"""
from __future__ import annotations
import os
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import google.generativeai as genai

# Configure Gemini API key from environment variable or Streamlit secrets
GENAI_API_KEY = os.getenv("GEMINI_API_KEY")
if GENAI_API_KEY:
    genai.configure(api_key=GENAI_API_KEY)

TRAINING_EXAMPLES = {
    "congestion": [
        "Why are many vessels waiting outside the port?",
        "The anchorage queue is growing and berths are full",
        "Port traffic is congested and ships cannot dock",
        "Too many ships are arriving at the same time",
        "How do we reduce vessel queue length?",
        "There is a bottleneck at the terminal",
        "The port is overloaded with incoming vessels",
        "Ships are waiting for berth allocation",
        "Congestion is delaying cargo movement",
        "Berth occupancy is too high",
        "What is the queue length at Houston berth?",
        "Are there anchorage delays at the harbor?",
    ],
    "delay": [
        "Our vessel will arrive later than planned",
        "The ship is behind schedule",
        "How can we reduce vessel arrival delay?",
        "The ETA keeps changing",
        "A vessel missed its planned arrival window",
        "The ship is moving slowly and will be late",
        "What should we do about a delayed vessel?",
        "Arrival time prediction is inaccurate",
        "There is a schedule disruption",
        "How to handle late inbound vessels?",
        "What is the delay status for vessel ID 102?",
    ],
    "weather": [
        "Strong winds are affecting port operations",
        "Heavy rain is disrupting vessel handling",
        "The sea conditions are unsafe for berthing",
        "What if a storm approaches the port?",
        "Weather conditions are delaying cranes",
        "Visibility is low due to fog",
        "How should we manage operations during bad weather?",
        "Wind speed is too high for crane operations",
        "The port is under a severe weather warning",
        "Rough seas are affecting arrivals",
        "Is fog stopping pilotage operations?",
    ],
    "truck": [
        "Trucks are queuing outside the terminal gate",
        "The container gate is overcrowded",
        "How can we reduce truck turnaround time?",
        "There are long waits at the gate",
        "Truck dispatch slots are not balanced",
        "Too many trucks arrive during the same hour",
        "How should we schedule container trucks?",
        "The OCR lane is blocked",
        "Empty container returns are causing a bottleneck",
        "Gate operations are too slow",
        "Truck congestion outside gate 3",
    ],
    "yard": [
        "The container yard is almost full",
        "We have no space to stack incoming containers",
        "Reefer containers need priority handling",
        "Yard space allocation is inefficient",
        "How can we reduce container rehandles?",
        "Cargo is not moving out of the yard quickly",
        "Container storage capacity is too low",
        "Yard congestion is affecting unloading",
        "Reefer plug capacity exhausted in yard section B",
    ],
    "equipment": [
        "A quay crane is unavailable",
        "Crane productivity has dropped",
        "Equipment breakdown is delaying unloading",
        "How should we allocate cranes between vessels?",
        "The terminal has insufficient handling equipment",
        "A gantry crane stopped working",
        "We need to improve berth productivity",
        "STS crane fault at berth 4",
    ],
}

REMEDIES = {
    "congestion": {
        "title": "Vessel congestion",
        "actions": [
            "Rank vessels by safety, contractual priority and cargo urgency; confirm the berth plan with the terminal operator.",
            "Use rolling arrival windows and coordinate virtual-arrival or speed-reduction options with carriers where safe and permitted.",
            "Check real berth and yard availability before diverting any vessel; confirm receiving-port capacity and commercial approval.",
            "Track queue length, berth occupancy and median dwell time at regular intervals.",
        ],
    },
    "delay": {
        "title": "Vessel delay",
        "actions": [
            "Recalculate ETA using the latest verified AIS position, speed, route and weather forecast.",
            "Notify the terminal and downstream service providers; update berth and pilotage plans.",
            "Avoid asking a vessel to increase speed without safety, fuel, emissions and authority checks.",
            "Record planned versus actual arrival to improve future model training.",
        ],
    },
    "weather": {
        "title": "Weather disruption",
        "actions": [
            "Follow port authority, harbour master and equipment manufacturer operating limits.",
            "Pause or restrict exposed operations when official thresholds are reached; do not override safety procedures.",
            "Update arrival windows and communicate berth/pilotage changes to affected operators.",
            "Resume operations only after the responsible authority confirms conditions are safe.",
        ],
    },
    "truck": {
        "title": "Truck gate bottleneck",
        "actions": [
            "Compare truck arrivals by 30-minute interval with gate processing capacity.",
            "Use appointment windows and communicate slot changes to transport operators.",
            "Separate exception handling, empty returns and priority cargo only where the terminal layout and policy support it.",
            "Measure truck turn time and queue length before and after the intervention.",
        ],
    },
    "yard": {
        "title": "Yard capacity",
        "actions": [
            "Review yard occupancy by zone, dwell age and planned vessel discharge.",
            "Prioritize reefer, hazardous and time-sensitive cargo according to safety rules and terminal procedures.",
            "Coordinate pickup appointments and identify containers eligible for approved evacuation or relocation.",
            "Do not recommend moving cargo without checking customs, documentation and handling restrictions.",
        ],
    },
    "equipment": {
        "title": "Equipment availability",
        "actions": [
            "Confirm the fault and estimated repair time with maintenance staff.",
            "Rebalance available cranes and gangs only after checking vessel compatibility and safety limits.",
            "Prioritize work by berth plan, cargo needs and contractual constraints.",
            "Log downtime and moves per hour for reliability analysis.",
        ],
    },
    "general": {
        "title": "General port operations",
        "actions": [
            "Identify the bottleneck using queue length, berth occupancy, dwell time and truck turn-time data.",
            "Confirm the current situation with the port control team before changing the operating plan.",
            "Choose one reversible intervention, assign an owner and measure the result.",
            "Capture the outcome so the recommendation process can be evaluated and improved.",
        ],
    },
}

def _build_classifier():
    texts, labels = [], []
    for label, examples in TRAINING_EXAMPLES.items():
        texts.extend(examples)
        labels.extend([label] * len(examples))
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline

_CLASSIFIER = _build_classifier()

def classify_query(query: str) -> dict:
    clean = re.sub(r"\s+", " ", (query or "").strip())
    if not clean:
        return {"category": "general", "confidence": 0.0}
    probs = _CLASSIFIER.predict_proba([clean])[0]
    idx = int(probs.argmax())
    return {"category": str(_CLASSIFIER.classes_[idx]), "confidence": round(float(probs[idx]), 3)}

def _llm_fallback_response(user_query: str, category: str, operational_context: dict | None = None) -> str:
    """Uses Gemini LLM to generate responses when queries ask about specific unknown data or complex situations."""
    if not GENAI_API_KEY:
        remedy = REMEDIES.get(category, REMEDIES["general"])
        return f"### {remedy['title']} remedy (Fallback Mode)\n\n" + "\n".join([f"{i+1}. {a}" for i, a in enumerate(remedy['actions'])])

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        context_str = json.dumps(operational_context, indent=2) if operational_context else "No active JSON context provided."
        
        prompt = f"""
        You are the PortFlow AI Remedy Assistant, an operational maritime assistant.
        
        User Query: "{user_query}"
        Detected Category Intent: {category}
        Current System Operational Context:
        {context_str}

        Instructions:
        1. If the user is asking for specific berth, port, or vessel data NOT found in the system context (e.g., "Houston port berth no. 7"):
           - Politely state that the specific entity/berth is not present in the currently active live dataset feed.
           - Provide actionable, general operational remedies and best practices for that category.
        2. Keep the answer clear, professional, and actionable for port operators.
        3. Do not invent fictitious real-time status data, but guide the user on how to handle that scenario.
        """
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        remedy = REMEDIES.get(category, REMEDIES["general"])
        return f"### {remedy['title']} remedy\n\n" + "\n".join([f"{i+1}. {a}" for i, a in enumerate(remedy['actions'])])

def get_bot_remedy(user_query: str, operational_context: dict | None = None) -> str:
    prediction = classify_query(user_query)
    category = prediction["category"]
    confidence = prediction["confidence"]
    
    # Check if user query contains specific entity questions or if classifier confidence is low
    specific_keywords = ["berth", "port", "vessel", "houston", "singapore", "rotterdam", "no.", "number", "#"]
    has_specific_entity = any(kw in user_query.lower() for kw in specific_keywords)

    if confidence < 0.35 or has_specific_entity:
        return _llm_fallback_response(user_query, category, operational_context)

    # Standard Rule-based response using TF-IDF Classifier
    remedy = REMEDIES.get(category, REMEDIES["general"])
    context = operational_context or {}
    lines = [f"### {remedy['title']} remedy", f"*ML intent classifier: `{category}` (confidence {confidence:.0%})*"]
    
    if context:
        lines += ["", "**Current PortFlow context**"]
        for key, value in context.items():
            lines.append(f"- **{key}:** {value}")
            
    lines += ["", "**Recommended actions**"]
    lines += [f"{i}. {action}" for i, action in enumerate(remedy["actions"], 1)]
    lines += ["", "_Operational decision support only. Verify live conditions and port-authority procedures before acting._"]
    return "\n".join(lines)

def get_retrieved_remedy(user_query: str) -> dict:
    prediction = classify_query(user_query)
    category = prediction["category"]
    return {"category": category, "confidence": prediction["confidence"], "knowledge": REMEDIES.get(category, REMEDIES["general"])}