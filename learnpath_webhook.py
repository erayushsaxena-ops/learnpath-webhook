from flask import Flask, request, jsonify
import requests
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*", allow_headers="*", methods=["GET", "POST", "OPTIONS"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set!")
ML_API_URL = "https://learnpath-api-u92e.onrender.com/predict/bulk"

COURSES = [
    {
        "id": 1,
        "module": "Understanding Digital Transformation",
        "description": "Foundation of digital transformation — how organizations grow from X to 10X using digital strategies, digitization journeys, and digital value chains.",
        "level": "Beginner",
        "duration": "3 sessions"
    },
    {
        "id": 2,
        "module": "Decoding Digital Transformation",
        "description": "Deep dive into digital business models, enterprise architecture, digital platforms, ecosystems and network effects that drive transformation.",
        "level": "Beginner-Intermediate",
        "duration": "5 sessions"
    },
    {
        "id": 3,
        "module": "Data Driven Enterprise Transformation",
        "description": "Build data cultures, use big data analytics, data engineering, CXO dashboards and storytelling with data to drive enterprise decisions.",
        "level": "Intermediate",
        "duration": "4 sessions"
    },
    {
        "id": 4,
        "module": "Building an Intelligent Organization",
        "description": "Covers AI, Machine Learning, Generative AI, LLMs, Prompt Engineering, RAG Applications, and Agentic AI workflows for building AI-powered organizations.",
        "level": "Intermediate-Advanced",
        "duration": "8 sessions"
    },
    {
        "id": 5,
        "module": "Tech Behind Digital Transformation",
        "description": "Explores IoT, Blockchain, Mixed Reality, Cloud computing, Low Code/No Code development, and Automation and Robotics as enablers of digital transformation.",
        "level": "Intermediate",
        "duration": "4 sessions"
    },
    {
        "id": 6,
        "module": "Digital Transformation of Functions",
        "description": "Digital transformation in Finance, Marketing, Supply Chain, Manufacturing, and IT Strategy including Agile, DevOps and Workplace of 2030.",
        "level": "Intermediate",
        "duration": "5 sessions"
    },
    {
        "id": 7,
        "module": "Cybersecurity, Cyber and Tech Laws",
        "description": "Cybersecurity threats, protection strategies, cyber laws and compliance requirements in the context of digital transformation.",
        "level": "Intermediate",
        "duration": "3 sessions"
    },
    {
        "id": 8,
        "module": "Risk Compliance and Control",
        "description": "Risk management in digital context, governance structures, data privacy regulations, and ethical and responsible digital practices.",
        "level": "Intermediate",
        "duration": "3 sessions"
    },
    {
        "id": 9,
        "module": "Connecting the Dots",
        "description": "Synthesizes all learnings connecting digital strategy, technology, data, AI and leadership into a cohesive transformation framework.",
        "level": "Advanced",
        "duration": "3 sessions"
    },
    {
        "id": 10,
        "module": "Leadership in the Digital and AI Era",
        "description": "Leadership skills for the digital age — leading digital teams, driving AI adoption, managing change and building a future-ready organization.",
        "level": "Advanced",
        "duration": "3 sessions"
    }
]

sessions = {}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/initialize", methods=["OPTIONS"])
@app.route("/webhook", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def handle_options():
    response = app.make_default_options_response()
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def call_gemini(prompt):
    import time
    # All free tier Gemini models — tried in order
    # If one fails or is overloaded, next one is tried
    models = [
        "gemini-2.5-flash",       # Primary — best free model
        "gemini-flash-latest",    # Fallback 1
        "gemini-2.5-flash-lite",  # Fallback 2 — lighter, faster
        "gemini-flash-lite-latest" # Fallback 3 — last resort
    ]
    retry_words = [
        "high demand", "quota", "rate", "limit",
        "retry", "overload", "unavailable", "busy",
        "capacity", "try again", "resource"
    ]
    
    for model in models:
        print(f"Trying model: {model}")
        max_attempts = 2  # Reduced from 3 to stay within gunicorn timeout
        model_failed = False
        
        for attempt in range(max_attempts):
            try:
                # Wait before retry (not before first attempt)
                if attempt > 0:
                    wait_time = 5  # Reduced from 15s to 5s to avoid worker timeout
                    print(f"  Retry {attempt}/{max_attempts-1} — waiting {wait_time}s...")
                    time.sleep(wait_time)
                
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
                }
                resp = requests.post(url, json=payload, timeout=60)
                data = resp.json()
                
                # ✅ Success
                if "candidates" in data:
                    print(f"  Success with {model}!")
                    return data["candidates"][0]["content"]["parts"][0]["text"]
                
                # ❌ Error from Gemini
                elif "error" in data:
                    error_msg = data["error"].get("message", "")
                    error_code = data["error"].get("code", 0)
                    print(f"  Error from {model}: {error_msg[:80]}")
                    
                    # Model not found or not available — skip to next model
                    if error_code in [404, 400] or 'not found' in error_msg.lower():
                        print(f"  Model {model} not available — trying next model")
                        model_failed = True
                        break
                    
                    # Retryable error — wait and retry same model
                    if any(x in error_msg.lower() for x in retry_words):
                        if attempt < max_attempts - 1:
                            continue
                        else:
                            # All retries used — move to next model
                            print(f"  All retries failed for {model} — trying next model")
                            model_failed = True
                            break
                    else:
                        # Unknown error — try next model
                        model_failed = True
                        break
            
            except Exception as e:
                print(f"  Exception: {str(e)[:80]}")
                if attempt < max_attempts - 1:
                    time.sleep(10)
                    continue
                model_failed = True
                break
        
        if model_failed:
            continue
    
    # All models failed
    return "I apologize, I am temporarily unable to generate a response. Please try again in a moment."


def get_skill_levels(scores):
    if not scores:
        return []
    try:
        students = [
            {
                "score": s["score"],
                "attendance": s["attendance"],
                "attempts": s.get("attempts", 2)
            }
            for s in scores
        ]
        resp = requests.post(ML_API_URL, json={"students": students}, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            results = resp.json().get("results", [])
            for i, r in enumerate(results):
                if i < len(scores):
                    scores[i]["skill_level"] = r["prediction"]
        else:
            # ML API sleeping or unavailable — assign Medium as default
            for s in scores:
                s["skill_level"] = "Medium"
    except Exception:
        # If ML API fails for any reason — assign Medium as default
        for s in scores:
            s["skill_level"] = "Medium"
    return scores


def generate_career_path(session):
    name = session.get("name", "Student")
    goal = session.get("goal", "Digital Leadership")
    short_term = session.get("short_term_goal", "Not specified")
    long_term = session.get("long_term_goal", "Not specified")
    scores = session.get("scores", [])

    skill_summary = "\n".join([
        f"- {s['course_name']}: Score {s['score']}, "
        f"Attendance: {s['attendance']}, "
        f"Attempts: {s.get('attempts', 'N/A')}, "
        f"Skill Level: {s.get('skill_level', 'Medium')}"
        for s in scores
    ]) if scores else "No scores provided"

    course_list = "\n".join([
        f"Module {c['id']}: {c['module']} - {c['description']} "
        f"(Level: {c['level']}, Duration: {c['duration']})"
        for c in COURSES
    ])

    prompt = (
        "You are a personalized learning advisor for the "
        "IIT-K CDAIO Executive Programme on Digital Transformation.\n\n"
        f"Student Profile:\n"
        f"- Name: {name}\n"
        f"- Career Goal: {goal}\n"
        f"- Short Term Goal (1 year): {short_term}\n"
        f"- Long Term Goal (5 years): {long_term}\n\n"
        f"Current Skill Levels per Course:\n{skill_summary}\n\n"
        f"Available Modules:\n{course_list}\n\n"
        "Instructions:\n"
        "1. Analyze the student goals and skill levels\n"
        "2. Prioritize Weak areas critical for their goal\n"
        "3. Leverage Strong areas for advanced application\n"
        "4. Recommend module sequence with clear reasoning\n\n"
        "Format:\n"
        f"- Start with 2 line personalized summary for {name}\n"
        "- Show recommended module sequence with emoji numbers\n"
        "- One line per module explaining WHY it fits their goal\n"
        "- End with encouraging closing message\n"
        "- Keep under 350 words\n"
        "- Be friendly, specific and motivating"
    )

    return call_gemini(prompt)



@app.route("/initialize", methods=["POST"])
def initialize():
    try:
        data = request.get_json()
        name = data.get("name", "Student")
        goal = data.get("goal")
        short_term_goal = data.get("short_term_goal")
        long_term_goal = data.get("long_term_goal")
        scores = data.get("scores", [])
        session_id = data.get("session_id", "default")

        if scores:
            scores = get_skill_levels(scores)

        missing = []
        if not goal:
            missing.append("career goal")
        if not short_term_goal:
            missing.append("short term goal")
        if not long_term_goal:
            missing.append("long term goal")

        sessions[session_id] = {
            "name": name,
            "goal": goal,
            "short_term_goal": short_term_goal,
            "long_term_goal": long_term_goal,
            "scores": scores,
            "career_path": None,
            "missing": missing
        }

        if "career goal" in missing:
            return jsonify({
                "status": "missing_info",
                "missing": missing,
                "message": (
                    f"Hi {name}! I have your course scores ready. "
                    "What is your career goal? "
                    "(Example: Chief AI Officer, CDO, CTO)"
                )
            })

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


@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        data = request.get_json()
        intent = data["queryResult"]["intent"]["displayName"]
        parameters = data["queryResult"].get("parameters", {})
        query_text = data["queryResult"].get("queryText", "")
        session_id = data["session"].split("/")[-1]
        session = sessions.get(session_id, {})

        # If session is missing (Render restarted), rebuild from frontend context
        student_context = data.get("studentContext", {})
        if student_context and not session.get("career_path"):
            # Restore session from frontend-passed context
            if not session:
                session = {}
            if student_context.get("name") and not session.get("name"):
                session["name"] = student_context["name"]
            if student_context.get("goal") and not session.get("goal"):
                session["goal"] = student_context["goal"]
            if student_context.get("career_path") and not session.get("career_path"):
                session["career_path"] = student_context["career_path"]
            # Save restored session
            sessions[session_id] = session

        if intent == "AskCareerPath":
            if session.get("career_path"):
                return jsonify({"fulfillmentText": session["career_path"]})
            elif session.get("goal"):
                career_path = generate_career_path(session)
                session["career_path"] = career_path
                sessions[session_id] = session
                return jsonify({"fulfillmentText": career_path})
            else:
                return jsonify({
                    "fulfillmentText": (
                        "I don't have your profile yet. "
                        "Please submit your details from the app first, "
                        "or tell me your career goal to get started!"
                    )
                })

        elif intent == "AskWhy":
            # Get from session first, then fall back to studentContext
            career_path = session.get("career_path", "") or student_context.get("career_path", "")
            goal = session.get("goal", "") or student_context.get("goal", "your career goal")
            name = session.get("name", "") or student_context.get("name", "there")

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            # Short focused prompt — fewer tokens = faster Gemini response
            prompt = (
                f"You are a learning advisor. "
                f"Student: {name}, Goal: {goal}.\n"
                f"Their question: '{query_text}'\n"
                f"Their career path summary: {career_path[:500]}\n\n"
                "Answer in max 100 words. Be specific and encouraging."
            )

            return jsonify({"fulfillmentText": call_gemini(prompt)})

        elif intent == "AskAlternative":
            # Get from session first, then fall back to studentContext
            career_path = session.get("career_path", "") or student_context.get("career_path", "")
            goal = session.get("goal", "") or student_context.get("goal", "your career goal")
            name = session.get("name", "") or student_context.get("name", "there")

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            # Short focused prompt — fewer tokens = faster Gemini response
            course_names = ", ".join([c['module'] for c in COURSES])
            prompt = (
                f"You are a learning advisor. "
                f"Student: {name}, Goal: {goal}.\n"
                f"Their question: '{query_text}'\n"
                f"Current path summary: {career_path[:400]}\n"
                f"Available modules: {course_names}\n\n"
                "Suggest 1-2 alternatives in max 100 words with tradeoffs. Be friendly."
            )

            return jsonify({"fulfillmentText": call_gemini(prompt)})

        elif intent == "FillMissingInfo":
            name = parameters.get("name") or session.get("name", "Student")
            goal = parameters.get("goal") or session.get("goal")
            short_term = parameters.get("short_term_goal") or session.get("short_term_goal")
            long_term = parameters.get("long_term_goal") or session.get("long_term_goal")

            if session_id not in sessions:
                sessions[session_id] = {}

            sessions[session_id].update({
                "name": name,
                "goal": goal,
                "short_term_goal": short_term,
                "long_term_goal": long_term
            })

            session = sessions[session_id]

            if not goal:
                return jsonify({
                    "fulfillmentText": (
                        f"Thanks {name}! What is your career goal? "
                        "(Example: Chief AI Officer, CDO, CTO, "
                        "Digital Transformation Head)"
                    )
                })

            career_path = generate_career_path(session)
            sessions[session_id]["career_path"] = career_path
            return jsonify({"fulfillmentText": career_path})

        else:
            return jsonify({
                "fulfillmentText": (
                    "I am here to help with your learning path! "
                    "Ask me to 'Show my career path', "
                    "'Why is a course recommended', "
                    "or 'Show me alternatives'."
                )
            })

    except Exception as e:
        return jsonify({
            "fulfillmentText": "Something went wrong. Please try again!"
        }), 500


@app.route("/")
def home():
    return jsonify({
        "status": "LearnPath Webhook is running!",
        "endpoints": {
            "initialize": "POST /initialize",
            "webhook": "POST /webhook"
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
