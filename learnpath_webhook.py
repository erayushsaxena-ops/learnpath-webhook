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


def build_performance_summary(scores):
    """Build a concise performance summary from scores array."""
    if not scores:
        return "No scores available."
    lines = []
    for s in scores:
        name     = s.get("course_name", "Unknown")
        score    = s.get("score", "N/A")
        attend   = s.get("attendance", "Unknown")
        attempts = s.get("attempts", "N/A")
        level    = s.get("skill_level", "Medium")
        lines.append(
            f"- {name}: Score {score}/100, "
            f"Attendance {attend}, "
            f"Attempts {attempts}, "
            f"Skill Level: {level}"
        )
    return "\n".join(lines)


def generate_career_path(session):
    name       = session.get("name", "Student")
    goal       = session.get("goal", "Digital Leadership")
    short_term = session.get("short_term_goal", "Not specified")
    long_term  = session.get("long_term_goal", "Not specified")
    scores     = session.get("scores", [])

    # Build detailed per-module student data with urgency labels
    skill_lines = []
    for s in scores:
        score    = s.get("score", 0)
        att      = s.get("attendance", "Present")
        attempts = s.get("attempts", 1)
        level    = s.get("skill_level", "Medium")
        urgency  = "URGENT" if score < 55 or att == "Absent" else                    "REINFORCE" if score < 75 else "STRONG"
        skill_lines.append(
            f"- {s['course_name']}: Score {score}/100, "
            f"Attendance {att}, Attempts {attempts}, "
            f"Level {level} [{urgency}]"
        )
    skill_summary = "\n".join(skill_lines) if skill_lines else "No scores provided"

    module_prereqs = (
        "M1 Understanding Digital Transformation [Beginner] - Foundation. No prerequisites. Must come before M2.\n"
        "M2 Decoding Digital Transformation [Beginner-Intermediate] - Requires M1. Unlocks M3, M5, M6.\n"
        "M3 Data Driven Enterprise Transformation [Intermediate] - Requires M2. Needed before M4 (data fuels AI).\n"
        "M4 Building an Intelligent Organization [Advanced] - Requires M3 AND M2. Most complex — do last among core.\n"
        "M5 Tech Behind Digital Transformation [Intermediate] - Requires M2. Supports M4 and M6.\n"
        "M6 Digital Transformation of Functions [Intermediate] - Requires M2 and M5. Applies tech across business.\n"
        "M7 Cybersecurity, Cyber and Tech Laws [Intermediate] - Requires M5. Must come before M8.\n"
        "M8 Risk Compliance and Control [Intermediate] - Requires M7. Governance layer on top of cyber.\n"
        "M9 Connecting the Dots [Advanced] - Requires ALL prior modules. Synthesis — must be second to last.\n"
        "M10 Leadership in the Digital and AI Era [Advanced] - Final capstone. Always last."
    )

    sequencing_rules = (
        "RULE 1 — PREREQUISITES OVERRIDE RISK LEVEL:\n"
        "  Never place a module before its prerequisites, regardless of how urgent it is.\n"
        "  If M5 is Weak but M2 is Strong, M2 still comes before M5 because M5 requires M2.\n"
        "  If M7 is Weak but M5 is Medium, M5 still comes before M7 because M7 requires M5.\n"
        "  A Medium or Strong prerequisite module is more important than a Weak module that depends on it.\n\n"
        "RULE 2 — WITHIN VALID PREREQUISITE ORDER, PRIORITISE BY URGENCY:\n"
        "  a. Prerequisite modules that must be fixed before urgent dependents can be tackled.\n"
        "  b. URGENT modules (score < 55 OR Absent attendance) — critical gaps.\n"
        "  c. REINFORCE modules (score 55-74) that are prerequisites of URGENT modules — fix these before the urgent ones.\n"
        "  d. REINFORCE modules (score 55-74) — general reinforcement.\n"
        "  e. STRONG modules (score 75+) — mention briefly, study last.\n\n"
        "RULE 3 — GOAL ALIGNMENT:\n"
        "  Within the same urgency tier, modules aligned to the student's career goal come first.\n\n"
        "RULE 4 — FIXED POSITIONS:\n"
        "  M9 and M10 always go last. They synthesize everything and have all modules as prerequisites."
    )

    prompt = (
        "You are an expert learning path advisor for the "
        "IIT-K CDAIO Executive Programme on Digital Transformation.\n\n"
        f"STUDENT: {name}\n"
        f"CAREER GOAL: {goal}\n"
        f"SHORT-TERM (1 year): {short_term}\n"
        f"LONG-TERM (5 years): {long_term}\n\n"
        f"STUDENT PERFORMANCE DATA (score, attendance, attempts per module):\n{skill_summary}\n\n"
        f"MODULE PREREQUISITE MAP:\n{module_prereqs}\n\n"
        f"SEQUENCING RULES (apply all four):\n{sequencing_rules}\n\n"
        "FORMAT — follow exactly, no markdown, no asterisks, no bold:\n"
        f"One line intro for {name} referencing their goal.\n"
        "Then for EVERY module (all 10, no skipping):\n"
        "[number] Module [id]: [exact module name]\n"
        "Why: [1 line — reference their actual score, attendance, AND the prerequisite or dependency reason if it affects placement]\n"
        "\n"
        "Closing line.\n"
        "No markdown. Under 450 words. All 10 modules must appear."
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
            career_path  = session.get("career_path", "") or student_context.get("career_path", "")
            goal         = session.get("goal", "") or student_context.get("goal", "your career goal")
            name         = session.get("name", "") or student_context.get("name", "there")
            scores       = session.get("scores", [])
            performance  = build_performance_summary(scores)

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            prereq_context = (
                "Module prerequisite order for context: "
                "M1 > M2 > M3 > M4 (M3 needed first); "
                "M2 > M5 > M6 (M5 needed first); "
                "M2 > M5 > M7 > M8; "
                "M9 and M10 always last."
            )

            prompt = (
                f"You are a learning advisor for IIT-K CDAIO programme.\n"
                f"Student: {name}, Career Goal: {goal}.\n\n"
                f"Student performance (score, attendance, attempts):\n{performance}\n\n"
                f"Their full recommended career path:\n{career_path}\n\n"
                f"Module prerequisites: {prereq_context}\n\n"
                f"Student's question: '{query_text}'\n\n"
                "Answer their SPECIFIC question directly. "
                "If asking why one module is before another, explain BOTH: "
                "(1) the prerequisite dependency reason if applicable, "
                "(2) their actual scores/attendance that influenced placement. "
                "Be specific — name the modules and cite the actual scores. "
                "Max 130 words. Friendly and clear."
            )

            return jsonify({"fulfillmentText": call_gemini(prompt)})

        elif intent == "AskAlternative":
            # Get from session first, then fall back to studentContext
            career_path  = session.get("career_path", "") or student_context.get("career_path", "")
            goal         = session.get("goal", "") or student_context.get("goal", "your career goal")
            name         = session.get("name", "") or student_context.get("name", "there")
            scores       = session.get("scores", [])
            performance  = build_performance_summary(scores)

            if not career_path:
                return jsonify({
                    "fulfillmentText": (
                        "Please generate your career path first "
                        "by asking 'Show me my career path'!"
                    )
                })

            course_names = ", ".join([c['module'] for c in COURSES])
            prompt = (
                f"You are a learning advisor for IIT-K CDAIO programme.\n"
                f"Student: {name}, Career Goal: {goal}.\n\n"
                f"Their actual performance data:\n{performance}\n\n"
                f"Current recommended path:\n{career_path[:400]}\n\n"
                f"Available modules: {course_names}\n\n"
                f"Student's question: '{query_text}'\n\n"
                "Answer using their ACTUAL scores and attendance. "
                "Reference specific module names and scores when suggesting alternatives. "
                "Suggest 1-2 alternatives with honest tradeoffs based on their performance. "
                "Max 120 words. Be friendly and data-driven."
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

        elif intent == "InstructorQuery":
            # Instructor chatbot — gets full class data and answers with Gemini
            class_data  = data.get("classData", {})
            query_text_inst = data.get("queryText", query_text)

            students_summary = class_data.get("studentsSummary", "No student data provided.")
            catalog_summary  = class_data.get("catalogSummary",  "No catalog data provided.")
            insights_summary = class_data.get("insightsSummary", "No insights provided.")

            prompt = (
                "You are an AI teaching assistant for the IIT-K CDAIO Executive Programme "
                "on Digital Transformation.\n\n"
                f"CLASS DATA:\n{students_summary}\n\n"
                f"REFERENCE MATERIALS IN CATALOG:\n{catalog_summary}\n\n"
                f"CLASS INSIGHTS:\n{insights_summary}\n\n"
                f"INSTRUCTOR'S QUESTION: '{query_text_inst}'\n\n"
                "Instructions:\n"
                "1. Answer the instructor's specific question using the class data above.\n"
                "2. Be specific — name students, modules, scores, and percentages.\n"
                "3. Provide reasoning, not just lists.\n"
                "4. Suggest actionable next steps where relevant.\n"
                "5. Reference instructor materials from the catalog when relevant.\n"
                "6. Keep response under 200 words. Use bullet points for clarity.\n"
                "7. Be professional and direct."
            )

            return jsonify({"fulfillmentText": call_gemini(prompt)})

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


@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    """
    Proxy endpoint — frontend calls this instead of Dialogflow directly.
    This keeps Google credentials server-side, away from the browser.

    Request body:
    {
      "session_id":     "student-USR001-123",
      "message":        "Why is this module first?",
      "student_context": {
        "name":        "Ayush",
        "goal":        "Chief AI Officer",
        "career_path": "..."   // from /initialize response
      }
    }

    Response:
    {
      "fulfillmentText": "...",
      "intent":          "AskWhy",
      "session_id":      "student-USR001-123"
    }
    """
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        data        = request.get_json()
        session_id  = data.get("session_id", "default-session")
        message     = data.get("message", "")
        student_ctx = data.get("student_context", {})

        if not message:
            return jsonify({"error": "message field is required"}), 400

        # ── Read Dialogflow credentials from environment ──
        project_id  = os.environ.get("DIALOGFLOW_PROJECT_ID", "")
        df_key_json = os.environ.get("DIALOGFLOW_KEY_JSON", "")

        if project_id and df_key_json:
            # ── Call Dialogflow detectIntent via REST API ──
            # Get access token using service account key
            import json, time
            import base64
            import hmac, hashlib

            try:
                key_data   = json.loads(df_key_json)
                token_url  = key_data.get("token_uri", "https://oauth2.googleapis.com/token")
                client_email = key_data.get("client_email", "")
                private_key  = key_data.get("private_key", "")

                # Build JWT for Google OAuth
                now       = int(time.time())
                jwt_header  = base64.urlsafe_b64encode(
                    json.dumps({"alg":"RS256","typ":"JWT"}).encode()
                ).rstrip(b"=").decode()
                jwt_payload = base64.urlsafe_b64encode(json.dumps({
                    "iss":   client_email,
                    "scope": "https://www.googleapis.com/auth/cloud-platform",
                    "aud":   token_url,
                    "exp":   now + 3600,
                    "iat":   now
                }).encode()).rstrip(b"=").decode()

                # Sign with RSA — use cryptography library
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import padding

                private_key_obj = serialization.load_pem_private_key(
                    private_key.encode(), password=None
                )
                signing_input = f"{jwt_header}.{jwt_payload}".encode()
                signature = private_key_obj.sign(
                    signing_input, padding.PKCS1v15(), hashes.SHA256()
                )
                jwt_sig = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
                jwt_token = f"{jwt_header}.{jwt_payload}.{jwt_sig}"

                # Exchange JWT for access token
                token_resp = requests.post(token_url, data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion":  jwt_token
                }, timeout=10)
                access_token = token_resp.json().get("access_token", "")

                # Call Dialogflow detectIntent
                df_url = (
                    f"https://dialogflow.googleapis.com/v2/projects/{project_id}"
                    f"/agent/sessions/{session_id}:detectIntent"
                )
                df_resp = requests.post(df_url,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type":  "application/json"
                    },
                    json={
                        "queryInput": {
                            "text": {
                                "text":         message,
                                "languageCode": "en"
                            }
                        }
                    },
                    timeout=30
                )
                df_data      = df_resp.json()
                query_result = df_data.get("queryResult", {})
                intent_name  = query_result.get("intent", {}).get("displayName", "")
                fulfillment  = query_result.get("fulfillmentText", "")

                # If Dialogflow returned fulfillment text (from static intent response)
                # and intent was handled without webhook, return it directly
                if fulfillment and intent_name == "Welcome":
                    return jsonify({
                        "fulfillmentText": fulfillment,
                        "intent":          intent_name,
                        "session_id":      session_id
                    })

                # For all other intents, Dialogflow calls /webhook automatically
                # and returns the fulfillmentText from webhook response
                if fulfillment:
                    return jsonify({
                        "fulfillmentText": fulfillment,
                        "intent":          intent_name,
                        "session_id":      session_id
                    })

                # Fallback if Dialogflow did not return fulfillment
                intent_name = intent_name or _detect_intent_locally(message)

            except Exception as df_err:
                print(f"Dialogflow call failed: {df_err} — falling back to local intent detection")
                intent_name = _detect_intent_locally(message)

        else:
            # No Dialogflow credentials — use local intent detection
            print("No Dialogflow credentials set — using local intent detection")
            intent_name = _detect_intent_locally(message)

        # ── Call /webhook handler directly with detected intent ──
        synthetic_request = {
            "queryResult": {
                "intent":     {"displayName": intent_name},
                "parameters": {},
                "queryText":  message
            },
            "session":        f"projects/learnpath/agent/sessions/{session_id}",
            "studentContext": student_ctx
        }

        # Reuse the webhook logic by calling it internally
        return _handle_webhook_logic(synthetic_request)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _detect_intent_locally(message):
    """Local intent detection — mirrors the HTML JS logic."""
    lower = message.lower()

    if any(p in lower for p in [
        'career path', 'show my path', 'learning path',
        'what should i study', 'study next', 'recommended path'
    ]):
        return 'AskCareerPath'

    if any(p in lower for p in [
        'alternative', 'skip', 'swap', 'instead',
        'other option', 'can i take', 'replace', 'change the order'
    ]):
        return 'AskAlternative'

    if any(p in lower for p in [
        'why', 'how come', 'explain', 'inspite', 'despite',
        'even though', 'above', 'below', 'before', 'after',
        'first', 'last', 'placed', 'position', 'priority',
        'ranked', 'sequence', 'order', 'recommend',
        'higher', 'lower'
    ]):
        return 'AskWhy'

    # Check if any module name is mentioned
    module_names = [
        'understanding digital', 'decoding digital', 'data driven',
        'building an intelligent', 'tech behind', 'digital transformation of',
        'cybersecurity', 'risk compliance', 'connecting the dots', 'leadership'
    ]
    if any(m in lower for m in module_names):
        return 'AskWhy'

    return 'AskWhy'  # Default to AskWhy rather than rejecting


def _handle_webhook_logic(data):
    """Internal webhook logic — shared between /webhook and /chat."""
    try:
        intent      = data["queryResult"]["intent"]["displayName"]
        parameters  = data["queryResult"].get("parameters", {})
        query_text  = data["queryResult"].get("queryText", "")
        session_id  = data["session"].split("/")[-1]
        session     = sessions.get(session_id, {})

        student_context = data.get("studentContext", {})
        if student_context and not session.get("career_path"):
            if not session:
                session = {}
            if student_context.get("name") and not session.get("name"):
                session["name"] = student_context["name"]
            if student_context.get("goal") and not session.get("goal"):
                session["goal"] = student_context["goal"]
            if student_context.get("career_path") and not session.get("career_path"):
                session["career_path"] = student_context["career_path"]
            sessions[session_id] = session

        # Import the webhook handler logic
        # We replicate the intent handling here to avoid circular calls
        import flask
        with app.test_request_context(
            '/webhook',
            method='POST',
            json=data,
            content_type='application/json'
        ):
            # Call webhook function directly
            response = webhook()
            return response

    except Exception as e:
        return jsonify({"fulfillmentText": f"Error: {str(e)}"}), 500


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
