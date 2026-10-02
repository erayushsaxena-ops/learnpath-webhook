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
                    "generationConfig": {"temperature": 0.7, "maxOutputTokens": 8000}
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
    short_term = session.get("short_term_goal", "")
    long_term  = session.get("long_term_goal", "")
    scores     = session.get("scores", [])

    # Compact score summary
    lines = []
    for s in scores:
        sc  = s.get("score", 0)
        att = s.get("attendance", "Present")
        att_short = att[0]  # P or A
        attempts  = s.get("attempts", 1)
        lvl = "URGENT" if sc < 55 or att == "Absent" else "REINFORCE" if sc < 75 else "OK"
        # Use short module name
        short_name = s["course_name"].replace("Digital Transformation", "DT").replace("Transformation", "Trans")
        lines.append(f"{short_name}: {sc}/100 {att_short} x{attempts} [{lvl}]")
    skill_summary = "\n".join(lines)

    prompt = (
        f"You are a learning path advisor for IIT-K CDAIO Executive Programme.\n"
        f"Student: {name} | Goal: {goal}\n"
        f"Short-term: {short_term} | Long-term: {long_term}\n\n"
        f"Performance (score/attendance/attempts/urgency):\n{skill_summary}\n\n"
        "Module prerequisites:\n"
        "M1(Understanding DT) -> M2(Decoding DT) -> M3(Data Driven) -> M4(Building Intelligent Org)\n"
        "M2 -> M5(Tech Behind DT) -> M6(DT of Functions) -> M7(Cybersecurity) -> M8(Risk Compliance)\n"
        "M9(Connecting Dots) and M10(Leadership) always LAST.\n\n"
        "SEQUENCING LOGIC — apply in this order:\n"
        "1. Prerequisites always come before their dependents — even if prerequisite score is OK/Strong.\n"
        "   Example: M2 must precede M3 even if M2=Strong and M3=URGENT.\n"
        "2. Within valid prerequisite order, put URGENT first, then REINFORCE, then OK.\n"
        "3. M9 and M10 always last.\n\n"
        "OUTPUT — list ALL 10 modules. Use this exact format for each:\n"
        "<number> <Full Exact Module Name>\n"
        "Why: <score, attendance, prerequisite reason in 1 line>\n"
        "\n"
        f"Start with one intro line for {name}. End with one closing line.\n"
        "NO markdown. NO asterisks. NO bold. ALL 10 modules required. Under 600 words."
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
            # Check session first, then studentContext passed from frontend
            career_path_stored = (session.get("career_path") or
                                   student_context.get("career_path", ""))
            name_stored = session.get("name","") or student_context.get("name","")

            if career_path_stored:
                # Already generated — return it
                return jsonify({"fulfillmentText": career_path_stored})
            elif session.get("scores"):
                # Have scores but no path — generate now
                career_path = generate_career_path(session)
                session["career_path"] = career_path
                sessions[session_id]   = session
                return jsonify({"fulfillmentText": career_path})
            elif session.get("goal"):
                # Have goal but no scores — generate with what we have
                career_path = generate_career_path(session)
                session["career_path"] = career_path
                sessions[session_id]   = session
                return jsonify({"fulfillmentText": career_path})
            else:
                return jsonify({
                    "fulfillmentText": (
                        f"Hi{' '+name_stored if name_stored else ''}! "
                        "I don't have your module scores yet. "
                        "Please call /initialize with your scores first, "
                        "then I can show you your personalized career path!"
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

            # Extract goal from message text if Dialogflow parameters are empty
            # e.g. "My goal is Chief Data Officer" → goal = "Chief Data Officer"
            if not goal and query_text:
                import re
                goal_match = re.search(
                    r"(?:my goal is|i want to be(?:come)?|goal[:\s]+)\s*(.+?)\s*$",
                    query_text, re.IGNORECASE
                )
                if goal_match:
                    goal = goal_match.group(1).strip()

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

        elif intent in ("Welcome", "Default Welcome Intent"):
            name = session.get("name", "") or student_context.get("name", "")
            return jsonify({
                "fulfillmentText": (
                    f"Hi{' ' + name if name else ''}! I'm your LearnPath AI advisor. "
                    "Ask me about your career path, why a module is recommended, "
                    "or if you can skip or swap a module. How can I help you today?"
                )
            })

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

    HOW IT WORKS:
    1. We call Dialogflow ONLY for intent detection (what did the student mean?)
    2. We NEVER let Dialogflow call /webhook — that always times out (5s limit)
    3. Once we know the intent, we call our own webhook logic directly
    4. This means Gemini always runs and real answers always return

    Request body:
    {
      "session_id":      "student-USR001-123",
      "message":         "Why is this module first?",
      "student_context": { "name", "goal", "career_path" }
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

        project_id  = os.environ.get("DIALOGFLOW_PROJECT_ID", "")
        df_key_json = os.environ.get("DIALOGFLOW_KEY_JSON", "")

        # ── Step 1: Detect intent (Dialogflow if available, local otherwise) ──
        intent_name = None

        if project_id and df_key_json:
            try:
                import json, time, base64
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import padding

                key_data     = json.loads(df_key_json)
                token_url    = key_data.get("token_uri", "https://oauth2.googleapis.com/token")
                client_email = key_data.get("client_email", "")
                private_key  = key_data.get("private_key", "")

                # Build JWT
                now         = int(time.time())
                jwt_header  = base64.urlsafe_b64encode(
                    json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
                ).rstrip(b"=").decode()
                jwt_payload = base64.urlsafe_b64encode(json.dumps({
                    "iss":   client_email,
                    "scope": "https://www.googleapis.com/auth/cloud-platform",
                    "aud":   token_url,
                    "exp":   now + 3600,
                    "iat":   now
                }).encode()).rstrip(b"=").decode()

                private_key_obj = serialization.load_pem_private_key(
                    private_key.encode(), password=None
                )
                signing_input = f"{jwt_header}.{jwt_payload}".encode()
                signature     = private_key_obj.sign(
                    signing_input, padding.PKCS1v15(), hashes.SHA256()
                )
                jwt_sig   = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
                jwt_token = f"{jwt_header}.{jwt_payload}.{jwt_sig}"

                # Exchange JWT for access token
                token_resp   = requests.post(token_url, data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    "assertion":  jwt_token
                }, timeout=10)
                access_token = token_resp.json().get("access_token", "")

                # ── Call Dialogflow detectIntent for INTENT ONLY ──
                # CRITICAL: We use the result only for intent_name.
                # We NEVER use Dialogflow's fulfillmentText — it is just a placeholder.
                # Dialogflow cannot call our webhook in time (5s limit vs our 6-30s response).
                # We always call our own webhook logic after getting the intent.
                df_url  = (
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
                            "text": {"text": message, "languageCode": "en"}
                        }
                    },
                    timeout=10  # Short timeout — we only need the intent name
                )
                df_data     = df_resp.json()
                query_result_df = df_data.get("queryResult", {})
                intent_name     = query_result_df.get("intent", {}).get("displayName", "")
                df_parameters   = query_result_df.get("parameters", {})

                print(f"Dialogflow detected intent: {intent_name}, params: {df_parameters}")

                # ── Apply override rules — correct known Dialogflow mistakes ──
                corrected = _override_dialogflow_intent(intent_name, message)
                if corrected != intent_name:
                    print(f"Intent overridden: {intent_name} → {corrected}")
                    intent_name = corrected

            except Exception as df_err:
                print(f"Dialogflow intent detection failed: {df_err} — using local detection")
                intent_name  = None
                df_parameters = {}

        # ── Step 2: Fall back to local intent detection if needed ──
        if not intent_name:
            intent_name   = _detect_intent_locally(message)
            df_parameters = {}
            print(f"Local intent detection: {intent_name}")

        # ── Step 3: Always call our own webhook logic directly ──
        # This is the REAL response — Gemini-powered, no timeout issues.
        synthetic_request = {
            "queryResult": {
                "intent":     {"displayName": intent_name},
                "parameters": df_parameters if 'df_parameters' in dir() else {},
                "queryText":  message
            },
            "session":        f"projects/learnpath/agent/sessions/{session_id}",
            "studentContext": student_ctx
        }

        response = _handle_webhook_logic(synthetic_request)

        # Add intent to response for debugging
        try:
            resp_data = response.get_json()
            resp_data["intent"]     = intent_name
            resp_data["session_id"] = session_id
            return jsonify(resp_data)
        except Exception:
            return response

    except Exception as e:
        return jsonify({"error": str(e), "fulfillmentText": "Sorry, something went wrong. Please try again."}), 500



def _detect_intent_locally(message):
    """
    Comprehensive local intent detection.
    Used as: (1) primary when no Dialogflow credentials,
             (2) override when Dialogflow returns wrong/fallback intent.
    Rules are ordered — first match wins.
    """
    lower = message.lower().strip()
    import re

    # ── Welcome / greeting ──
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon',
                 'good evening', 'namaste', 'hii', 'start', 'begin',
                 'get started', 'help me', 'help']
    if lower in greetings or any(lower == g for g in greetings):
        return 'Welcome'

    # ── FillMissingInfo — student providing their goal or name ──
    fill_patterns = [
        r'my goal is', r'my career goal is', r'i want to be',
        r'i want to become', r'i am aiming', r'goal is to become',
        r'my name is', r'call me', r'i am [a-z]'
    ]
    if any(re.search(p, lower) for p in fill_patterns):
        return 'FillMissingInfo'

    # ── AskAlternative — skip / swap / optional ──
    # Check BEFORE AskWhy because "can I skip" could match "why"
    alt_keywords = [
        'skip', 'swap', 'replace', 'alternative', 'instead',
        'other option', 'can i take', 'change the order',
        'is this optional', 'is this mandatory', 'is this compulsory',
        'do i have to', 'do i need to', 'can i avoid',
        'can i leave', 'can i move', 'can i postpone',
        'can i do this later', "what if i don't",
        'what happens if i skip', 'is this necessary'
    ]
    if any(k in lower for k in alt_keywords):
        return 'AskAlternative'

    # ── AskWhy — explanation of module placement ──
    # Wide net: any question about reasoning, ordering, or a specific module
    why_keywords = [
        'why', 'how come', 'explain', 'reason',
        'inspite', 'despite', 'even though', 'although',
        'above', 'below', 'before', 'after',
        'first', 'last', 'placed', 'position', 'ranked',
        'sequence', 'order', 'priority', 'higher', 'lower',
        'tell me about', 'what is', "what's"
    ]
    module_keywords = [
        'understanding digital', 'decoding digital', 'data driven',
        'building an intelligent', 'tech behind', 'digital transformation of',
        'cybersecurity', 'risk compliance', 'connecting the dots',
        'leadership', 'module'
    ]
    has_why     = any(k in lower for k in why_keywords)
    has_module  = any(k in lower for k in module_keywords)
    # "recommended" alone → AskWhy (not AskCareerPath)
    has_recommended = 'recommended' in lower or 'recommend' in lower

    if has_why or has_module or has_recommended:
        return 'AskWhy'

    # ── AskCareerPath — student wants to see their path ──
    path_keywords = [
        'career path', 'my path', 'show path', 'learning path',
        'study plan', 'module sequence', 'show me my', 'what should i study',
        'study next', 'my sequence', 'preparation plan', 'show modules',
        'what order', 'my modules', 'give me my path', 'view my path'
    ]
    if any(k in lower for k in path_keywords):
        return 'AskCareerPath'

    # ── Default: AskWhy ──
    # Better to try to explain something than to reject
    return 'AskWhy'


def _override_dialogflow_intent(dialogflow_intent, message):
    """
    Override Dialogflow's intent when we know it got it wrong.
    Called after Dialogflow returns an intent — applies correction rules.
    """
    lower = message.lower()

    # Rule 1: If Dialogflow says AskCareerPath but message has WHY → AskWhy
    # (Dialogflow confuses "Why is X recommended?" with AskCareerPath)
    if dialogflow_intent == 'AskCareerPath':
        if any(k in lower for k in ['why', 'how come', 'explain', 'reason',
                                     'inspite', 'despite', 'even though',
                                     'tell me why', 'above', 'below', 'placed']):
            return 'AskWhy'

    # Rule 2: Default Fallback → try local detection
    if dialogflow_intent in ('Default Fallback Intent', 'Default Fallback',
                              '', None):
        return _detect_intent_locally(message)

    # Rule 3: If Dialogflow says Welcome but message is substantive → local detect
    if dialogflow_intent == 'Welcome' and len(lower) > 10:
        local = _detect_intent_locally(message)
        if local != 'Welcome':
            return local

    # Dialogflow was correct — use its intent
    return dialogflow_intent


def _handle_webhook_logic(data):
    """
    Internal — calls the webhook handler directly without going through HTTP.
    Used by /chat so we always get the real Gemini response,
    bypassing Dialogflow's 5-second fulfillment timeout entirely.
    """
    try:
        # Merge studentContext into session store so webhook has full context
        session_id      = data.get("session", "").split("/")[-1]
        student_context = data.get("studentContext", {})

        if student_context and session_id:
            session = sessions.get(session_id, {})
            for field in ["name", "goal", "career_path", "short_term_goal", "long_term_goal"]:
                if student_context.get(field) and not session.get(field):
                    session[field] = student_context[field]
            sessions[session_id] = session

        # Use Flask test request context to call webhook() directly — no HTTP round-trip
        import flask
        with app.test_request_context(
            "/webhook",
            method="POST",
            json=data,
            content_type="application/json"
        ):
            flask.g._learnpath_internal = True  # Flag so webhook knows it's internal
            return webhook()

    except Exception as e:
        print(f"_handle_webhook_logic error: {e}")
        return jsonify({"fulfillmentText": "Sorry, something went wrong. Please try again."})


@app.route("/debug-path", methods=["POST"])
def debug_path():
    """Debug endpoint — returns raw Gemini career path output."""
    try:
        data    = request.get_json()
        session = {
            "name":            data.get("name", "Test Student"),
            "goal":            data.get("goal", "Chief AI Officer"),
            "short_term_goal": data.get("short_term_goal", ""),
            "long_term_goal":  data.get("long_term_goal", ""),
            "scores":          data.get("scores", [])
        }
        raw = generate_career_path(session)
        return jsonify({"raw": raw, "lines": raw.split("\n")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
