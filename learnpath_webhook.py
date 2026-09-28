from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

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


def call_gemini(prompt):
    # Try these models in order until one works
    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-latest",
        "gemini-2.5-pro",
        "gemini-pro-latest"
    ]
    last_error = ""
    for model in models:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
            }
            resp = requests.post(url, json=payload, timeout=30)
            data = resp.json()
            # Check if valid response
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            # If error returned
            elif "error" in data:
                last_error = data["error"].get("message", "Unknown error")
                continue
        except Exception as e:
            last_error = str(e)
            continue
    # All models failed
    return f"Career path generation temporarily unavailable. Error: {last_error}"


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


def generate_short_summary(session):
    name = session.get("name", "Student")
    goal = session.get("goal", "Digital Leadership")
    career_path = session.get("career_path", "")

    prompt = (
        f"A student named {name} with goal '{goal}' "
        f"has this career path:\n\n{career_path}\n\n"
        "Give a SHORT 2-3 line summary of their top 3 priority modules only.\n"
        "End with exactly this line: "
        "'Ask me WHY any module is recommended or for ALTERNATIVES!'\n"
        "Keep total response under 300 characters strictly.\n"
        "No bullet points. Plain sentences only."
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

        if intent == "AskCareerPath":
            if session.get("career_path"):
                # Return short summary for Dialogflow
                # Full path already sent to frontend via /initialize
                short_summary = generate_short_summary(session)
                return jsonify({"fulfillmentText": short_summary})
            elif session.get("goal"):
                career_path = generate_career_path(session)
                session["career_path"] = career_path
                sessions[session_id] = session
                short_summary = generate_short_summary(session)
                return jsonify({"fulfillmentText": short_summary})
            else:
                return jsonify({
                    "fulfillmentText": (
                        "I don't have your profile yet. "
                        "Please submit your details from the app first, "
                        "or tell me your career goal to get started!"
                    )
                })

        elif intent == "AskWhy":
            career_path = session.get("career_path", "")
            goal = session.get("goal", "your career goal")
            name = session.get("name", "there")

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            prompt = (
                f"Student {name} with goal '{goal}' received this career path:\n\n"
                f"{career_path}\n\n"
                f"The student specifically asked: '{query_text}'\n\n"
                "Answer their specific question directly.\n"
                "Explain in simple motivating language (max 150 words) "
                "why the sequencing makes sense for their goal.\n"
                "Be specific, clear and encouraging."
            )

            return jsonify({"fulfillmentText": call_gemini(prompt)})

        elif intent == "AskAlternative":
            career_path = session.get("career_path", "")
            goal = session.get("goal", "your career goal")
            name = session.get("name", "there")

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            course_list = "\n".join([
                f"Module {c['id']}: {c['module']} - {c['description']}"
                for c in COURSES
            ])

            prompt = (
                f"Student {name} with goal '{goal}' received this career path:\n\n"
                f"{career_path}\n\n"
                f"Available modules:\n{course_list}\n\n"
                f"The student specifically asked: '{query_text}'\n\n"
                "Answer their specific question directly.\n"
                "Suggest 1-2 alternatives in max 150 words.\n"
                "Be honest about tradeoffs. Be friendly and supportive."
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
