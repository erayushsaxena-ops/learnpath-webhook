
from flask import Flask, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AQ.Ab8RN6Kgu7VIO_ioiLeSzJ2JvxOEytJ244KZ83cMYX_A2EB9eQ")
ML_API_URL = "https://learnpath-api-u92e.onrender.com/predict/bulk"

# ============================================================
# COURSE DATABASE — IIT-K CDAIO PROGRAMME
# ============================================================
COURSES = [
    {
        "id": 1,
        "module": "Understanding Digital Transformation",
        "description": "Foundation of digital transformation — how organizations grow from X to 10X using digital strategies, digitization journeys, and digital value chains.",
        "career_fit": ["Digital Leader", "CDO", "CTO", "Business Transformation", "Strategy"],
        "level": "Beginner",
        "duration": "3 sessions"
    },
    {
        "id": 2,
        "module": "Decoding Digital Transformation",
        "description": "Deep dive into digital business models, enterprise architecture, digital platforms, ecosystems and network effects that drive transformation.",
        "career_fit": ["Digital Leader", "CDO", "Product Manager", "Strategy", "CTO"],
        "level": "Beginner-Intermediate",
        "duration": "5 sessions"
    },
    {
        "id": 3,
        "module": "Data Driven Enterprise Transformation",
        "description": "Learn to build data cultures, use big data analytics, data engineering, CXO dashboards and storytelling with data to drive enterprise decisions.",
        "career_fit": ["Data Leader", "CDO", "Analytics Head", "Business Intelligence"],
        "level": "Intermediate",
        "duration": "4 sessions"
    },
    {
        "id": 4,
        "module": "Building an Intelligent Organization",
        "description": "Covers AI, Machine Learning, Generative AI, LLMs, Prompt Engineering, RAG Applications, and Agentic AI workflows for building AI-powered organizations.",
        "career_fit": ["AI Leader", "CTO", "ML Engineer", "AI Product Manager", "Chief AI Officer"],
        "level": "Intermediate-Advanced",
        "duration": "8 sessions"
    },
    {
        "id": 5,
        "module": "Tech Behind Digital Transformation",
        "description": "Explores IoT, Blockchain, Mixed Reality, Cloud computing, Low Code/No Code development, and Automation & Robotics as enablers of digital transformation.",
        "career_fit": ["CTO", "Tech Leader", "Innovation Head", "Digital Architect"],
        "level": "Intermediate",
        "duration": "4 sessions"
    },
    {
        "id": 6,
        "module": "Digital Transformation of Functions",
        "description": "How digital transformation applies to Finance, Marketing, Supply Chain, Manufacturing, and IT Strategy including Agile, DevOps and Workplace of 2030.",
        "career_fit": ["Functional Leader", "CFO", "CMO", "COO", "Digital Leader"],
        "level": "Intermediate",
        "duration": "5 sessions"
    },
    {
        "id": 7,
        "module": "Cybersecurity, Cyber and Tech Laws",
        "description": "Understand cybersecurity threats, protection strategies, cyber laws and compliance requirements in the context of digital transformation.",
        "career_fit": ["CISO", "Risk Leader", "CTO", "Compliance Head", "Digital Leader"],
        "level": "Intermediate",
        "duration": "3 sessions"
    },
    {
        "id": 8,
        "module": "Risk Compliance and Control",
        "description": "Risk management in digital context, governance structures, data privacy regulations, and ethical and responsible digital practices.",
        "career_fit": ["Risk Leader", "CRO", "Compliance Head", "CDO", "Legal Tech"],
        "level": "Intermediate",
        "duration": "3 sessions"
    },
    {
        "id": 9,
        "module": "Connecting the Dots",
        "description": "Integration module that synthesizes all learnings connecting digital strategy, technology, data, AI and leadership into a cohesive transformation framework.",
        "career_fit": ["All career paths", "Digital Leader", "CDO", "CTO", "CEO"],
        "level": "Advanced",
        "duration": "3 sessions"
    },
    {
        "id": 10,
        "module": "Leadership in the Digital & AI Era",
        "description": "Develop leadership skills for the digital age — leading digital teams, driving AI adoption, managing change and building a future-ready organization.",
        "career_fit": ["C-Suite", "Digital Leader", "CDO", "CEO", "Transformation Head"],
        "level": "Advanced",
        "duration": "3 sessions"
    }
]

# ============================================================
# SESSION STORAGE
# ============================================================
sessions = {}

# ============================================================
# HELPER — Call Gemini API
# ============================================================
def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 1000
        }
    }
    response = requests.post(url, json=payload)
    data = response.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

# ============================================================
# HELPER — Call ML API
# ============================================================
def get_skill_levels(scores):
    if not scores:
        return []
    students = [
        {
            "score": s["score"],
            "attendance": s["attendance"],
            "attempts": s.get("attempts", 2)
        }
        for s in scores
    ]
    response = requests.post(ML_API_URL, json={"students": students})
    results = response.json().get("results", [])
    for i, r in enumerate(results):
        scores[i]["skill_level"] = r["prediction"]
    return scores

# ============================================================
# HELPER — Generate Career Path
# ============================================================
def generate_career_path(session):
    name = session.get("name", "Student")
    goal = session.get("goal", "Digital Leadership")
    short_term = session.get("short_term_goal", "Not specified")
    long_term = session.get("long_term_goal", "Not specified")
    scores = session.get("scores", [])

    # Build skill summary
    skill_summary = "\n".join([
        f"- {s['course_name']}: Score {s['score']}, "
        f"Attendance: {s['attendance']}, "
        f"Attempts: {s.get('attempts', 'N/A')}, "
        f"Skill Level: {s.get('skill_level', 'Medium')}"
        for s in scores
    ]) if scores else "No scores provided"

    # Build course list
    course_list = "\n".join([
        f"Module {c['id']}: {c['module']} — "
        f"{c['description']} "
        f"(Level: {c['level']}, Duration: {c['duration']})"
        for c in COURSES
    ])

    prompt = f"""You are a personalized learning advisor for the 
IIT-K CDAIO Executive Programme on Digital Transformation.

Student Profile:
- Name: {name}
- Career Goal: {goal}
- Short Term Goal (1 year): {short_term}
- Long Term Goal (5 years): {long_term}

Current Skill Levels per Course:
{skill_summary}

Available Modules in the Programme (in order):
{course_list}

Instructions:
1. Analyze the student's goals and skill levels
2. Prioritize Weak areas that are critical for their goal
3. Leverage Strong areas for advanced application
4. Recommend the sequence of modules with clear reasoning

Format your response like this:
- Start with a 2 line personalized summary for {name}
- Show recommended module sequence with emoji numbers
- One line per module explaining WHY it fits their goal
- End with an encouraging closing message
- Keep it under 350 words
- Be friendly, specific and motivating"""

    return call_gemini(prompt)

# ============================================================
# ENDPOINT 1 — /initialize
# Called directly by Frontend with all student data
# ============================================================
@app.route("/initialize", methods=["POST"])
def initialize():
    try:
        data = request.get_json()

        # Extract student data
        name = data.get("name", "Student")
        goal = data.get("goal")
        short_term_goal = data.get("short_term_goal")
        long_term_goal = data.get("long_term_goal")
        scores = data.get("scores", [])
        session_id = data.get("session_id", "default")

        # Get skill levels from ML API
        if scores:
            scores = get_skill_levels(scores)

        # Check what's missing
        missing = []
        if not goal:
            missing.append("career goal")
        if not short_term_goal:
            missing.append("short term goal")
        if not long_term_goal:
            missing.append("long term goal")

        # Store in session
        sessions[session_id] = {
            "name": name,
            "goal": goal,
            "short_term_goal": short_term_goal,
            "long_term_goal": long_term_goal,
            "scores": scores,
            "career_path": None,
            "missing": missing
        }

        # If critical info missing → ask for it
        if "career goal" in missing:
            return jsonify({
                "status": "missing_info",
                "missing": missing,
                "message": f"Hi {name}! I have your course scores ready. "
                          f"Just need your career goal to generate "
                          f"your personalized path. "
                          f"What is your career goal? "
                          f"(Example: Chief AI Officer, CDO, CTO)"
            })

        # Generate career path
        career_path = generate_career_path(sessions[session_id])
        sessions[session_id]["career_path"] = career_path

        return jsonify({
            "status": "success",
            "session_id": session_id,
            "career_path": career_path,
            "missing": missing
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# ============================================================
# ENDPOINT 2 — /webhook
# Called by Dialogflow for conversation
# ============================================================
@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        intent = data["queryResult"]["intent"]["displayName"]
        parameters = data["queryResult"].get("parameters", {})
        session_id = data["session"].split("/")[-1]

        # Get session
        session = sessions.get(session_id, {})

        # ─────────────────────────────────────
        # Intent: AskCareerPath
        # ─────────────────────────────────────
        if intent == "AskCareerPath":

            # Session exists and has career path
            if session.get("career_path"):
                return jsonify({
                    "fulfillmentText": session["career_path"]
                })

            # Session exists but no career path yet
            elif session.get("goal"):
                career_path = generate_career_path(session)
                session["career_path"] = career_path
                sessions[session_id] = session
                return jsonify({
                    "fulfillmentText": career_path
                })

            # No session at all
            else:
                return jsonify({
                    "fulfillmentText": "I don't have your profile yet. "
                                      "Please make sure your details are "
                                      "submitted from the app first, or "
                                      "tell me your career goal to get started!"
                })

        # ─────────────────────────────────────
        # Intent: AskWhy
        # ─────────────────────────────────────
        elif intent == "AskWhy":
            career_path = session.get("career_path", "")
            goal = session.get("goal", "your career goal")
            name = session.get("name", "there")

            if not career_path:
                return jsonify({
                    "fulfillmentText": "Please generate your career path first "
                                      "by asking 'Show me my career path'!"
                })

            prompt = f"""A student named {name} with career goal '{goal}' 
received this career path recommendation:

{career_path}

They are asking WHY a particular course is recommended.
Explain in simple, motivating language (max 150 words) 
why the sequencing and recommendations make sense 
for their specific goal.
Be specific, clear and encouraging."""

            explanation = call_gemini(prompt)
            return jsonify({"fulfillmentText": explanation})

        # ─────────────────────────────────────
        # Intent: AskAlternative
        # ─────────────────────────────────────
        elif intent == "AskAlternative":
            career_path = session.get("career_path", "")
            goal = session.get("goal", "your career goal")
            name = session.get("name", "there")
