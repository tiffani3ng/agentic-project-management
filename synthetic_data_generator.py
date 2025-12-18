# ============================================================
# 0) Imports + Config
# ============================================================
import json, random, re, math
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Optional, Tuple, Literal

import pandas as pd

SEED = 42
random.seed(SEED)

TYPE_OF_COMPANY = "a mid-sized e-commerce/fashion/retail company selling apparel, accessories, and beauty"
NUMBER_OF_EMPLOYEES = 33

TOTAL_PROJECTS = 10
COMPLETED_PROJECTS = 4
IN_PROGRESS_PROJECTS = 4
NOT_STARTED_PROJECTS = 2
TASKS_PER_PROJECT = 15

# Availability horizon
AVAIL_START_DATE = date(2025, 12, 15)
AVAIL_DAYS = 60

TIMEZONES = [
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "Europe/London",
    "Europe/Paris",
]

# "Workday" clamp used in events + task timestamps
WORKDAY_START_LOCAL = time(9, 0)
WORKDAY_END_LOCAL = time(18, 0)

OUTPUT_DIR = "/Users/tiffanie/code/agentic-project-management/data"
# EXCEL_PATH = f"{OUTPUT_DIR}/synthetic_retail_dataset.xlsx"
JSON_PATH = f"{OUTPUT_DIR}/synthetic_retail_dataset.json"

# Ollama settings (will fallback if server not reachable)
USE_OLLAMA = False
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1"  # change to your installed model name, e.g. "llama3", "mistral", etc.

# ============================================================
# 1) Light helpers
# ============================================================

def iso_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")

def parse_iso_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")

def clamp_dt(dt: datetime, start_t: time, end_t: time) -> datetime:
    # Keep same date; clamp time
    t = dt.time()
    if t < start_t:
        return dt.replace(hour=start_t.hour, minute=start_t.minute, second=0)
    if t > end_t:
        return dt.replace(hour=end_t.hour, minute=end_t.minute, second=0)
    return dt.replace(second=0)

def rand_time(min_h=9, max_h=18) -> Tuple[int,int,int]:
    hh = random.randint(min_h, max_h)
    mm = random.choice([0, 15, 30, 45])
    ss = 0
    return (hh, mm, ss)

def daterange(start: date, days: int):
    for i in range(days):
        yield start + timedelta(days=i)

def safe_int(x, default=0):
    try:
        return int(x)
    except:
        return default

# ============================================================
# 2) Ollama call (1-2 calls total) with fallback
# ============================================================

def ollama_chat(prompt: str, model: str = OLLAMA_MODEL, temperature: float = 0.6) -> str:
    """
    Calls Ollama /api/chat if available. Returns assistant content string.
    Falls back by raising on network errors.
    """
    import requests
    url = f"{OLLAMA_HOST}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You output ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]

def try_ollama_json(prompt: str, default_obj):
    if not USE_OLLAMA:
        return default_obj
    try:
        txt = ollama_chat(prompt)
        # Sometimes models wrap JSON with stray text; try to extract first JSON object/array.
        m = re.search(r"(\{.*\}|\[.*\])", txt, flags=re.S)
        if m:
            txt = m.group(1)
        return json.loads(txt)
    except Exception as e:
        print(f"[WARN] Ollama unavailable or JSON parse failed, using fallback. Reason: {e}")
        return default_obj

# ============================================================
# 3) Employees (Python-only)
# ============================================================

DEPARTMENTS = [
    "Brand Marketing",
    "Performance Marketing",
    "Merchandising & Buying",
    "Product Management",
    "Product Design (UX/UI)",
    "Creative Studio",
    "Engineering",
    "Data & Analytics",
    "Supply Chain",
    "Warehouse & Fulfillment Ops",
    "Customer Experience (CX)",
    "Sales & Partnerships",
    "Finance",
    "Legal & Compliance",
]

# 33 roles spanning all departments (ensure coverage)
ROLES = [
    # Brand Marketing (2)
    ("Brand Marketing", "PR & Communications Manager"),
    ("Brand Marketing", "Brand Partnerships & Influencer Associate"),

    # Performance Marketing (2)
    ("Performance Marketing", "Performance Marketing Specialist"),
    ("Performance Marketing", "Growth Marketing Manager (CRO/LP Testing)"),

    # Merchandising & Buying (2)
    ("Merchandising & Buying", "Merchandise Planner"),
    ("Merchandising & Buying", "Pricing & Markdown Analyst"),

    # Product Management (3)
    ("Product Management", "Product Manager (Personalization)"),
    ("Product Management", "Product Manager (Loyalty)"),
    ("Product Management", "Technical Product Manager (Integrations)"),

    # Product Design (UX/UI) (2)
    ("Product Design (UX/UI)", "UX/UI Designer"),
    ("Product Design (UX/UI)", "Product Designer (Design Systems)"),

    # Creative Studio (1)
    ("Creative Studio", "Creative Director"),

    # Engineering (4)
    ("Engineering", "Engineering Manager"),
    ("Engineering", "Frontend Engineer"),
    ("Engineering", "Backend Engineer"),
    ("Engineering", "E-commerce Platform Engineer (Shopify)"),

    # Data & Analytics (4)
    ("Data & Analytics", "Data Analyst"),
    ("Data & Analytics", "Analytics Engineer (dbt)"),
    ("Data & Analytics", "Data Engineer"),
    ("Data & Analytics", "Experimentation Analyst"),

    # Supply Chain (3)
    ("Supply Chain", "Demand Forecast Analyst"),
    ("Supply Chain", "Supply Planner"),
    ("Supply Chain", "Vendor Management Lead"),

    # Warehouse & Fulfillment Ops (3)
    ("Warehouse & Fulfillment Ops", "Warehouse Operations Associate"),
    ("Warehouse & Fulfillment Ops", "Fulfillment Process Improvement Specialist"),
    ("Warehouse & Fulfillment Ops", "Returns Operations Supervisor"),

    # Customer Experience (CX) (1)
    ("Customer Experience (CX)", "CX Operations Manager"),

    # Sales & Partnerships (1)
    ("Sales & Partnerships", "Partnerships Manager"),

    # Finance (3)
    ("Finance", "FP&A Analyst"),
    ("Finance", "Budget Analyst"),
    ("Finance", "Accounting Manager"),

    # Legal & Compliance (2)
    ("Legal & Compliance", "Legal & Compliance Associate"),
    ("Legal & Compliance", "Privacy & Data Compliance Specialist"),
]

# Department skill catalogs (15 each)
DEPT_SKILLS: Dict[str, List[str]] = {
    "Brand Marketing": [
        "Brand positioning", "Campaign planning", "Creative briefing", "Brand voice", "Influencer strategy",
        "Content calendar", "Product storytelling", "Go-to-market (GTM)", "PR coordination", "Launch planning",
    ],
    "Performance Marketing": [
        "Paid social (Meta)", "Google Ads", "TikTok Ads", "CAC/LTV analysis", "Attribution basics",
        "Creative testing", "Budget pacing", "Landing page CRO", "Pixel troubleshooting", "UTM governance",
    ],
    "Merchandising & Buying": [
        "Assortment planning", "Open-to-buy (OTB)", "SKU rationalization", "Pricing strategy", "Markdown optimization",
        "Vendor negotiations", "Trend analysis", "Category strategy", "Sell-through analysis", "Allocation",
    ],
    "Product Management": [
        "Roadmapping", "PRDs", "Backlog grooming", "Experimentation", "User stories",
        "Stakeholder alignment", "KPI definition", "Launch management", "Requirements gathering", "Cross-functional leadership",
    ],
    "Product Design (UX/UI)": [
        "Figma", "Wireframing", "Prototyping", "Design systems", "Usability testing",
        "Interaction design", "Information architecture", "Accessibility basics", "Visual hierarchy", "UX writing",
    ],
    "Creative Studio": [
        "Graphic design", "Art direction", "Adobe Photoshop", "Adobe Illustrator", "Brand identity",
        "Email design", "PDP imagery", "Retouching", "Social content design", "Storyboarding",
    ],
    "Engineering": [
        "JavaScript/TypeScript", "React", "Node.js", "Python", "REST APIs",
        "Shopify Liquid", "Kubernetes", "Docker", "CI/CD", "Observability (logs/metrics)",
    ],
    "Data & Analytics": [
        "SQL", "Looker", "dbt", "Metric definitions", "Experimentation analysis",
        "GA4", "Data modeling", "ETL pipelines", "Warehouse (Snowflake/BigQuery)", "Data QA",
    ],
    "Supply Chain": [
        "Supply planning", "Demand forecasting", "Lead time management", "PO management", "Vendor management",
        "MOQ analysis", "Inbound scheduling", "Inventory health", "Safety stock", "S&OP basics",
    ],
    "Warehouse & Fulfillment Ops": [
        "Warehouse slotting", "Cycle counts", "Pick/pack optimization", "3PL management", "Returns processing",
        "Labor planning", "SOP documentation", "Process improvement", "Scanner rollout", "Shipping exceptions",
    ],
    "Customer Experience (CX)": [
        "Zendesk", "Macros & automation", "Escalations", "Returns policy support", "Tone-of-voice support",
        "QA rubrics", "CSAT analysis", "Knowledge base", "Refund workflows", "Fraud screening basics",
    ],
    "Sales & Partnerships": [
        "Partnership development", "Outbound prospecting", "Negotiation", "Co-marketing", "Affiliate programs",
        "Wholesale basics", "Account management", "Contract basics", "Pipeline management", "ROI modeling",
    ],
    "Finance": [
        "FP&A", "Budgeting", "Forecasting", "Variance analysis", "Unit economics",
        "Gross margin analysis", "Inventory accounting basics", "Month-end close coordination", "Excel modeling", "Financial reporting",
    ],
    "Legal & Compliance": [
        "Contracting", "Legal review", "Compliance", "Privacy (GDPR/CCPA) basics", "Claims substantiation",
        "IP basics", "Vendor terms", "Risk assessment", "Policy drafting", "Template governance",
    ],
}

FIRST_NAMES = ["Ava","Mia","Sofia","Liam","Noah","Ethan","Olivia","Isabella","Aria","Zoe","Kai","Leo","Amir","Nina","Hana","Chloe","Evelyn","Maya","Jade","Sam"]
LAST_NAMES  = ["Nguyen","Patel","Kim","Garcia","Lopez","Chen","Singh","Rodriguez","Williams","Brown","Lee","Martinez","Davis","Wilson","Anderson","Thomas","Moore","Jackson","Martin","Taylor"]

def generate_employees_py() -> List[Dict]:
    employee_rows = []
    # Build E001..E033 and assign roles 1:1 from ROLES list (already 33)
    for i in range(1, NUMBER_OF_EMPLOYEES + 1):
        eid = f"E{i:03d}"
        dept, role = ROLES[i-1]
        name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
        max_hours = random.randint(20, 60)
        tz = random.choice(TIMEZONES)

        # 5 skills from own dept + 1 cross-dept skill
        dept_sk = random.sample(DEPT_SKILLS[dept], k=5)
        other_dept = random.choice([d for d in DEPARTMENTS if d != dept])
        cross = random.choice(DEPT_SKILLS[other_dept])
        skills = dept_sk + [cross]
        random.shuffle(skills)

        employee_rows.append({
            "id": eid,
            "name": name,
            "department": dept,
            "role": role,
            "skills": skills,
            "max_hours": max_hours,
            "timezone": tz,
        })
    return employee_rows

employees = generate_employees_py()
employees_df = pd.DataFrame(employees)

# Global skill pool used for tasks skill_needed selection
skill_pool = sorted({s.strip() for e in employees for s in e["skills"] if s.strip()})

# ============================================================
# 4) Availability (Python-only, "LLM-like nuance")
# ============================================================

def seniority_score(role: str) -> int:
    r = role.lower()
    if any(k in r for k in ["ceo", "chief", "vp"]): return 4
    if any(k in r for k in ["director", "head"]): return 3
    if "manager" in r or "lead" in r: return 2
    if any(k in r for k in ["senior"]): return 1
    return 0

def role_rhythm(dept: str) -> Dict[str, float]:
    """
    Returns meeting-load + weekend-likelihood by dept to shape daily free hours.
    """
    # meeting_load: higher means fewer deep-work hours on weekdays
    # weekend_bias: higher means more likely nonzero Sat/Sun hours
    base = {
        "Engineering": (0.9, 0.2),
        "Data & Analytics": (1.0, 0.1),
        "Product Management": (1.4, 0.2),
        "Brand Marketing": (1.2, 0.3),
        "Performance Marketing": (1.1, 0.35),
        "Creative Studio": (1.1, 0.45),
        "Merchandising & Buying": (1.2, 0.25),
        "Supply Chain": (1.1, 0.25),
        "Warehouse & Fulfillment Ops": (1.0, 0.2),
        "Customer Experience (CX)": (1.3, 0.35),
        "Sales & Partnerships": (1.4, 0.4),
        "Finance": (1.2, 0.15),
        "Legal & Compliance": (1.3, 0.1),
        "Product Design (UX/UI)": (1.1, 0.25),
    }
    ml, wb = base.get(dept, (1.1, 0.2))
    return {"meeting_load": ml, "weekend_bias": wb}

def weekly_pattern_for_employee(emp: Dict) -> Dict[str, int]:
    """
    Produces 0..10 hours_free per day, with:
    - seniority reduces deep-work
    - part-time reduces overall
    - midweek focus blocks (Tue/Wed) for ICs
    - weekends usually low but varies by dept
    """
    dept = emp["department"]
    role = emp["role"]
    max_hours = emp["max_hours"]

    s = seniority_score(role)
    rhythm = role_rhythm(dept)
    meeting_load = rhythm["meeting_load"] + 0.25*s  # seniors have more meetings
    weekend_bias = rhythm["weekend_bias"]

    # base weekday deep-work capacity
    # scale with max_hours (20..50 maps ~3..7 avg weekday)
    scale = (max_hours - 20) / 30.0  # 0..1
    base_weekday = 3.0 + 4.0*scale   # 3..7
    base_weekday = max(2.0, base_weekday - 0.6*s)  # senior reduction

    # convert meeting load into reduction
    base_weekday = max(1.5, base_weekday - 0.8*(meeting_load - 1.0))

    # add "focus blocks" Tue/Wed for IC-ish roles (lower seniority)
    focus_boost = 1.0 if s <= 1 and dept in ["Engineering","Data & Analytics","Product Design (UX/UI)"] else 0.4

    # daily jitter
    def jitter(x): return x + random.uniform(-0.7, 0.7)

    mon = jitter(base_weekday - 0.3)  # Monday meetings
    tue = jitter(base_weekday + focus_boost)
    wed = jitter(base_weekday + focus_boost)
    thu = jitter(base_weekday + 0.2)
    fri = jitter(base_weekday - 0.5)  # Friday lighter

    # weekends
    sat = random.uniform(0, 3) * weekend_bias
    sun = random.uniform(0, 2.5) * (weekend_bias * 0.9)

    # occasionally: crunch weeks for marketing/creative near launches
    if dept in ["Brand Marketing","Performance Marketing","Creative Studio"] and random.random() < 0.25:
        sat += random.uniform(0.5, 2.0)
        # sunday sometimes
        if random.random() < 0.5:
            sun += random.uniform(0.25, 1.5)

    # clamp to 0..10 ints
    days = [mon,tue,wed,thu,fri,sat,sun]
    days = [int(max(0, min(10, round(x)))) for x in days]

    return {"mon":days[0],"tue":days[1],"wed":days[2],"thu":days[3],"fri":days[4],"sat":days[5],"sun":days[6]}

patterns = {e["id"]: weekly_pattern_for_employee(e) for e in employees}

availability_rows = []
weekday_fields = ["mon","tue","wed","thu","fri","sat","sun"]
for d in daterange(AVAIL_START_DATE, AVAIL_DAYS):
    fld = weekday_fields[d.weekday()]
    for eid, pat in patterns.items():
        # small day-to-day variability (simulate interruptions / PTO)
        base = pat[fld]
        if random.random() < 0.05:
            base = max(0, base - random.randint(2, 5))  # partial PTO
        if random.random() < 0.05:
            base = min(10, base + random.randint(1, 3)) # extra push
        availability_rows.append({"employee_id": eid, "date": d.isoformat(), "hours_free": int(base)})

availability_df = pd.DataFrame(availability_rows)

# ============================================================
# 5) Projects (1 LLM call; Python assigns IDs + deadlines)
# ============================================================

def make_deadlines():
    """
    Deadlines relative to 2026-01-01:
    - completed (4): in 2025
    - in_progress (4): early 2026
    - not_started (2): mid-late 2026
    """
    completed = [date(2025, m, random.randint(5, 25)) for m in [8,9,10,11]]
    inprog    = [date(2026, m, random.randint(5, 28)) for m in [1,2,3,4]]
    notstart  = [date(2026, m, random.randint(5, 28)) for m in [7,10]]
    return completed, inprog, notstart

completed_dl, inprog_dl, notstart_dl = make_deadlines()

project_prompt = f"""
You are generating a portfolio of {TOTAL_PROJECTS} cross-functional projects for {TYPE_OF_COMPANY}.
Return ONLY JSON with schema:
{{
  "projects": [
    {{"name": "...", "description": "...", "priority": 1-5}},
    ...
  ]
}}

Constraints:
- Exactly {TOTAL_PROJECTS} projects.
- Names must be realistic for e-commerce/fashion/retail: marketing campaigns, replatforming, warehouse optimization, returns, personalization, creative refresh, fraud, loyalty, international shipping, etc.
- Descriptions are 1-2 sentences with a clear end result.
- Priorities: 1-5 (5 highest).
- Ensure projects span many departments and require cross-functional work.
"""

fallback_projects = {
    "projects": [
        {"name":"Returns Experience Overhaul (Self-Service + Policy Simplification)","description":"Improve self-serve returns flows, clarify policy copy, and reduce ticket volume by streamlining refunds/exchanges.","priority":5},
        {"name":"Warehouse Slotting Optimization for Peak Season","description":"Re-slot top SKUs, improve pick paths, and reduce fulfillment cycle time ahead of peak demand.","priority":4},
        {"name":"Holiday Paid Social + Influencer Launch Campaign","description":"Launch holiday campaign with paid social + creators; deliver creative system and measurement plan.","priority":4},
        {"name":"Fraud Rules Refresh + Chargeback Reduction","description":"Tune fraud rules and implement monitoring to reduce chargebacks while maintaining conversion.","priority":5},
        {"name":"Checkout Performance & Conversion Sprint","description":"Improve checkout speed and UX; ship instrumentation and experiments to lift conversion.","priority":5},
        {"name":"Inventory Accuracy Program (Cycle Counts + Scanner Rollout)","description":"Deploy scanners and cycle-count SOPs to improve inventory accuracy and reduce oversells.","priority":4},
        {"name":"Lifecycle Email Personalization MVP","description":"Build segmentation + templates for lifecycle journeys and measure engagement lifts.","priority":4},
        {"name":"Product Detail Page Creative Refresh (UGC + Image System)","description":"Refresh PDP visual system including UGC modules and updated image guidelines.","priority":3},
        {"name":"Shopify Replatform Phase 1 (Catalog + Theme Architecture)","description":"Re-architect theme and catalog structure for scalability; establish deployment pipeline.","priority":5},
        {"name":"International Shipping Expansion (Duties/Taxes + Carriers)","description":"Add carriers and duties/taxes handling for international customers, with updated CX workflows.","priority":4},
    ]
}

proj_payload = try_ollama_json(project_prompt, fallback_projects)
proj_list = proj_payload.get("projects", fallback_projects["projects"])[:TOTAL_PROJECTS]

# Assign IDs + deadlines by bucket (deterministic ordering)
project_rows = []
all_deadlines = completed_dl + inprog_dl + notstart_dl
for i in range(TOTAL_PROJECTS):
    pid = f"P{i+1:03d}"
    meta = proj_list[i]
    dl = all_deadlines[i]
    project_rows.append({
        "id": pid,
        "name": meta["name"],
        "description": meta["description"],
        "deadline": dl.isoformat(),
        "priority": int(meta["priority"]),
    })

projects_df = pd.DataFrame(project_rows).sort_values("id").reset_index(drop=True)

# bucket IDs deterministically
projects_sorted = projects_df.to_dict("records")
completed_ids = {p["id"] for p in projects_sorted[:COMPLETED_PROJECTS]}
in_progress_ids = {p["id"] for p in projects_sorted[COMPLETED_PROJECTS:COMPLETED_PROJECTS + IN_PROGRESS_PROJECTS]}
not_started_ids = {p["id"] for p in projects_sorted[-NOT_STARTED_PROJECTS:]}

# ============================================================
# 6) Tasks roadmap (1 LLM call total) -> Python fills dates/status/est_hours
# ============================================================

# Skill categories -> hour distributions
def est_hours_for_skill(skill: str) -> int:
    s = skill.lower()
    if any(k in s for k in ["sql","looker","dbt","dashboard","ga4","experiment"]):
        return random.randint(2, 20)  # mix queries + dashboards
    if any(k in s for k in ["api","backend","frontend","react","javascript","typescript","shopify","kubernetes","docker","ci/cd","integration"]):
        return random.randint(12, 40)
    if any(k in s for k in ["creative","graphic","copy","art direction","email design","pdp","photoshop","illustrator","figma"]):
        return random.randint(6, 18)
    if any(k in s for k in ["vendor","procurement","logistics","supply","warehouse","3pl","inventory"]):
        return random.randint(8, 24)
    if any(k in s for k in ["legal","compliance","contract","tax","accounting","finance","fp&a"]):
        return random.randint(6, 20)
    return random.randint(4, 16)

def status_sequence(bucket: str, n: int) -> List[str]:
    if bucket == "completed":
        return ["completed"] * n
    if bucket == "not_started":
        return ["not_started"] * n
    # in_progress: first 60% in_progress; remaining split
    k_inprog = int(round(0.60 * n))
    rem = n - k_inprog
    k_comp = int(round(0.25 * n))
    k_block = int(round(0.10 * n))
    k_ns = n - k_inprog - k_comp - k_block
    seq = (["in_progress"] * k_inprog) + (["completed"] * k_comp) + (["blocked"] * k_block) + (["not_started"] * k_ns)
    return seq[:n]

# Build single prompt for all projects (minimize calls)
projects_brief = [{"id": p["id"], "name": p["name"], "description": p["description"]} for p in projects_sorted]

tasks_prompt = f"""
You are generating task roadmaps for {TYPE_OF_COMPANY}.
Return ONLY JSON with schema:
{{
  "roadmaps": {{
     "P001": [["Task name", "Skill needed"], ... exactly {TASKS_PER_PROJECT} items],
     ...
  }}
}}

Rules:
- Provide a roadmap for each project id in the input list.
- Each roadmap must have exactly {TASKS_PER_PROJECT} tasks ordered roughly by execution order.
- "Skill needed" MUST be chosen EXACTLY from this allowed skill pool (use exact strings):
{json.dumps(skill_pool, indent=2)}

Projects:
{json.dumps(projects_brief, indent=2)}
"""

PROJECT_TASK_ROADMAPS: Dict[str, List[List[str]]] = {

    "Returns Experience Overhaul (Self-Service + Policy Simplification)": [
        ["Kickoff + align success metrics (refund time, ticket rate, CSAT)", "KPI definition"],
        ["Audit current returns flows + policy touchpoints (site + email + help center)", "Journey mapping"],
        ["Pull baseline returns drivers + ticket tags analysis", "CSAT analysis"],
        ["Map event tracking gaps for returns funnel (view policy → start return → label)", "Event tracking specs"],
        ["Draft updated returns policy copy + tone guidelines", "Brand voice"],
        ["Legal review of policy wording + claims substantiation", "Legal review"],
        ["Design self-serve returns IA + new entry points (order page, email)", "Information architecture"],
        ["Create Figma prototypes for new returns UX (refund/exchange paths)", "Prototyping"],
        ["Write PRD + requirements + edge cases (partial refunds, bundles, final sale)", "PRDs"],
        ["Implement returns UX updates in storefront/account", "Shopify Liquid"],
        ["Integrate returns logic with backend/refund workflows", "REST APIs"],
        ["Update Zendesk macros/automation + KB articles for new policy", "Macros & automation"],
        ["QA end-to-end returns flow + support escalation scenarios", "QA rubrics"],
        ["Launch plan + internal enablement (CX training + comms)", "Training enablement"],
        ["Post-launch monitoring + iterate based on ticket/returns metrics", "Dashboards"],
    ],

    "Warehouse Slotting Optimization for Peak Season": [
        ["Kickoff with Ops + define target metrics (pick rate, travel time, mis-picks)", "KPI tracking"],
        ["Extract top-SKU velocity + ABC classification for peak assortment", "Data modeling"],
        ["Analyze current pick paths + congestion zones", "Root cause analysis"],
        ["Propose new slotting strategy + zone layout changes", "Warehouse slotting"],
        ["Validate constraints (hazmat, fragility, replenishment cadence)", "Constraint management"],
        ["Coordinate with Merch on peak assortment changes + substitutions", "Stakeholder management"],
        ["Create re-slotting SOP + safety checklist", "SOP documentation"],
        ["Labor plan for re-slotting weekend/overnight shifts", "Labor planning"],
        ["Pilot re-slotting in 1 zone + capture before/after performance", "Process improvement"],
        ["Update WMS location mappings + bin labels", "WMS basics"],
        ["Scanner workflow checks for new locations + exception handling", "Shipping exceptions"],
        ["Train leads + floor staff on new pick paths and replenishment triggers", "Training enablement"],
        ["Scale re-slotting across remaining zones", "Pick/pack optimization"],
        ["Peak readiness dry run + escalation playbook", "Escalation handling"],
        ["Post-peak retro + codify improvements for next season", "Documentation"],
    ],

    "Holiday Paid Social + Influencer Launch Campaign": [
        ["Campaign kickoff + creative brief + channel plan", "Creative briefing"],
        ["Define target audience segments + offers + positioning", "Consumer insights"],
        ["Influencer shortlist + outreach + negotiation", "Influencer strategy"],
        ["Draft influencer contracts + deliverable terms", "Contracting"],
        ["Build content calendar + launch timeline across channels", "Content calendar"],
        ["Develop paid social test matrix (hooks, formats, landing pages)", "Creative testing"],
        ["Design key visual system (static + video templates)", "Graphic design"],
        ["Write copy variants + brand voice checks", "Copywriting"],
        ["Set up tracking (UTMs, pixels) + measurement plan", "UTM governance"],
        ["Analytics spec: events + naming conventions for holiday funnel", "Event tracking specs"],
        ["Launch creator content production + asset QA", "Creative QA"],
        ["Configure ad sets + budget pacing rules + guardrails", "Budget pacing"],
        ["Daily performance readouts + creative iteration loop", "GA4 reporting"],
        ["Cross-functional standup (CX, Ops) for peak comms + inventory risks", "Stakeholder management"],
        ["Post-campaign wrap: results, learnings, and next-holiday playbook", "Documentation"],
    ],

    "Fraud Rules Refresh + Chargeback Reduction": [
        ["Kickoff + define chargeback KPIs and decision thresholds", "KPI definition"],
        ["Pull baseline fraud signals + chargeback reasons by segment", "Cohort analysis"],
        ["Audit current fraud rules + manual review SOP", "Process documentation"],
        ["Identify false positive drivers impacting conversion", "Analytics interpretation"],
        ["Design new fraud ruleset + risk tiers", "Risk assessment"],
        ["Implement rule updates in fraud tooling / checkout logic", "Feature flags"],
        ["Add monitoring dashboard for fraud + CB alerts", "Looker dashboards"],
        ["Instrument events for fraud decisions (approve/deny/manual review)", "Event tracking specs"],
        ["Create CX guidance for fraud holds + escalation paths", "Escalations"],
        ["Legal/compliance review (privacy, data handling, adverse actions)", "Privacy (GDPR/CCPA) basics"],
        ["Run shadow test (new rules) against historical orders", "Experimentation analysis"],
        ["Roll out new rules with staged exposure + rollback plan", "Incident response"],
        ["Train CX + Ops on updated playbooks and documentation", "Training enablement"],
        ["Weekly review cadence for tuning + exception cases", "Stakeholder enablement"],
        ["Postmortem + finalize ongoing governance + templates", "Template governance"],
    ],

    "Checkout Performance & Conversion Sprint": [
        ["Sprint kickoff + define performance and conversion targets", "KPI definition"],
        ["Baseline checkout metrics + error rates + device breakdown", "GA4"],
        ["Create event tracking spec for checkout funnel (step-level)", "Event tracking specs"],
        ["Audit client-side performance + bundle size hot spots", "Performance optimization"],
        ["Implement frontend optimizations (lazy load, code split, caching)", "React"],
        ["Backend profiling for checkout APIs + latency sources", "Observability (logs/metrics)"],
        ["Optimize API performance + add timeouts/retries where needed", "REST APIs"],
        ["Add feature flags for risky changes + staged rollout", "Feature flags"],
        ["Design UX improvements (payment options, error messaging)", "Interaction design"],
        ["Usability test new checkout UX with 5–8 users", "Usability testing"],
        ["Set up experiment plan (A/B) for top UX hypotheses", "Experimentation"],
        ["QA automation for regression coverage (happy path + edge cases)", "QA automation"],
        ["Launch changes + monitor incidents + conversions", "Incident triage"],
        ["Daily war-room + cross-team comms (CX, Finance, Ops)", "Stakeholder alignment"],
        ["Post-sprint report + next backlog and follow-up PRDs", "Postmortems"],
    ],

    "Inventory Accuracy Program (Cycle Counts + Scanner Rollout)": [
        ["Kickoff + define accuracy KPIs and governance cadence", "KPI tracking"],
        ["Select scanner hardware + vendor coordination", "Vendor management"],
        ["Define cycle-count SOPs by zone + cadence", "Cycle counts"],
        ["Build training materials + quick-reference guides", "Training enablement"],
        ["Configure scanner workflows + exception codes", "Scanner rollout"],
        ["Update WMS processes for adjustments + approvals", "WMS basics"],
        ["Data capture spec (scan events, adjustments, reasons)", "Event tracking specs"],
        ["Run pilot in 1 site/zone + measure discrepancy reduction", "Process improvement"],
        ["Root-cause analysis on top discrepancy categories", "Root cause analysis"],
        ["Integrate inventory adjustments into reporting dashboards", "Dashboards"],
        ["CX + Merch comms process for oversells/backorders", "Cross-team feedback loops"],
        ["Scale scanner rollout across sites + staffing plan", "Labor planning"],
        ["Implement escalation path for repeated discrepancies", "Escalation handling"],
        ["Finance review for shrink and accounting implications", "Inventory accounting basics"],
        ["Program retro + finalize playbook + continuous improvement loop", "Documentation"],
    ],

    "Lifecycle Email Personalization MVP": [
        ["Kickoff + define success metrics (CTR, CVR, unsub, revenue)", "KPI definition"],
        ["Audit current lifecycle journeys + gaps (welcome, abandon, post-purchase)", "Journey mapping"],
        ["Define segmentation strategy + data requirements", "Requirements gathering"],
        ["Pull customer cohorts + baseline engagement by segment", "Cohort analysis"],
        ["Draft experiment plan for personalization hypotheses", "Experiment design"],
        ["Create email template system + modular blocks", "Email design"],
        ["Write copy variants per segment + brand voice guardrails", "Brand voice"],
        ["Define event tracking + attribution for lifecycle conversions", "Attribution basics"],
        ["Implement segmentation queries/views in warehouse", "SQL"],
        ["Build activation pipeline to ESP (audiences + triggers)", "Data integrations"],
        ["Set up dashboards for lifecycle performance monitoring", "Looker dashboards"],
        ["QA journeys end-to-end (rendering, links, suppression, edge cases)", "Creative QA"],
        ["Legal/compliance review (opt-in, privacy, claims)", "Advertising compliance"],
        ["Launch MVP + weekly optimization cadence", "Stakeholder enablement"],
        ["Post-launch report + backlog for Phase 2 improvements", "Roadmapping"],
    ],

    "Product Detail Page Creative Refresh (UGC + Image System)": [
        ["Kickoff + define PDP goals (engagement, conversion, returns)", "KPI definition"],
        ["Audit current PDP components + creative inconsistencies", "Heuristic evaluation"],
        ["Define new PDP image guidelines + shot list standards", "Brand guidelines"],
        ["UGC sourcing plan + rights management approach", "UGC curation"],
        ["Design updated PDP modules (UGC carousel, image zoom, gallery)", "Design systems"],
        ["Prototype in Figma + align on component behaviors", "Prototyping"],
        ["Engineering handoff + implementation specs", "Handoff to engineering"],
        ["Implement new PDP modules in theme/components", "Shopify Liquid"],
        ["Set up asset versioning + naming conventions for images", "Asset versioning"],
        ["Creative production coordination for refreshed imagery", "Production coordination"],
        ["QA across devices + accessibility checks", "Accessibility basics"],
        ["Set up experiment to measure uplift vs control", "Experimentation"],
        ["Launch wave 1 on top SKUs + monitor performance", "Dashboards"],
        ["Expand rollout + update internal playbook", "Documentation"],
        ["Post-launch retro + finalize governance for ongoing refreshes", "Template governance"],
    ],

    "Shopify Replatform Phase 1 (Catalog + Theme Architecture)": [
        ["Kickoff + scope Phase 1 deliverables + risks", "Roadmapping"],
        ["Audit current theme + tech debt + performance issues", "Performance optimization"],
        ["Define new theme architecture + component library plan", "Component libraries"],
        ["Catalog data model review + required schema changes", "Data modeling"],
        ["Integration mapping (ERP/WMS/3PL) + data flow diagrams", "Data integrations"],
        ["Set up CI/CD + preview environments", "CI/CD"],
        ["Implement foundational theme scaffolding + routing", "JavaScript/TypeScript"],
        ["Build core PDP/PLP components with reusable patterns", "React"],
        ["Migrate catalog structure + collections/tags strategy", "Category strategy"],
        ["Implement observability + logging for key storefront events", "Observability (logs/metrics)"],
        ["QA automation baseline (smoke tests, critical paths)", "QA automation"],
        ["Performance budget + load testing plan", "Performance optimization"],
        ["Security/privacy review for integrations and data handling", "Privacy (GDPR/CCPA) basics"],
        ["Pilot launch on a limited traffic slice + rollback plan", "Incident response"],
        ["Postmortem + finalize Phase 2 backlog + documentation", "Documentation"],
    ],

    "International Shipping Expansion (Duties/Taxes + Carriers)": [
        ["Kickoff + define target countries, timelines, and KPIs", "KPI definition"],
        ["Carrier evaluation + negotiations + onboarding requirements", "Vendor negotiations"],
        ["Define duties/taxes approach + customer messaging", "Product storytelling"],
        ["Legal/compliance review (cross-border policies, privacy, claims)", "Compliance"],
        ["Finance review for tax/VAT implications + reporting needs", "Tax"],
        ["Update checkout UX for intl shipping options + pricing transparency", "UX writing"],
        ["Engineering: integrate carriers + rate shopping + labels", "REST APIs"],
        ["Implement duties/taxes calculation workflow", "Data integrations"],
        ["Update warehouse ops SOPs for intl packing + documentation", "SOP documentation"],
        ["CX workflows for intl tracking, delays, returns, and exceptions", "Order troubleshooting"],
        ["Update help center + macros for intl FAQs", "Knowledge base"],
        ["Event tracking spec for intl checkout and delivery funnel", "Event tracking specs"],
        ["Pilot launch to 1–2 countries + monitor incidents", "Incident triage"],
        ["Expand rollout + partner performance reporting", "Reporting"],
        ["Post-launch retro + finalize escalation paths + governance", "Escalation process"],
    ],
}

fallback_roadmaps = {"roadmaps": {}}
for p in projects_sorted:
    pid = p["id"]
    pname = p["name"]

    if pname not in PROJECT_TASK_ROADMAPS:
        raise KeyError(f"No fallback roadmap found for project: {pname}")

    # Use the pre-authored roadmap directly
    items = PROJECT_TASK_ROADMAPS[pname]

    # Optional safety check: trim or pad to TASKS_PER_PROJECT
    if len(items) > TASKS_PER_PROJECT:
        items = items[:TASKS_PER_PROJECT]
    elif len(items) < TASKS_PER_PROJECT:
        # pad with low-risk generic wrap-up tasks if needed
        remaining = TASKS_PER_PROJECT - len(items)
        for i in range(remaining):
            items.append([
                f"Finalize and document remaining work for {pname}",
                random.choice(skill_pool),
            ])

    fallback_roadmaps["roadmaps"][pid] = items

# Try Ollama; if it fails, this deterministic roadmap is used
roadmap_payload = try_ollama_json(tasks_prompt, fallback_roadmaps)
roadmaps = roadmap_payload.get("roadmaps", fallback_roadmaps["roadmaps"])

def pick_due_dates(project_deadline: date, n: int, bucket: str) -> List[datetime]:
    """
    Create due dates inside window_start..deadline with clustering near the end
    and a late-gate cluster for legal/finance tasks (handled later in events too).
    """
    if bucket == "completed":
        window_start = project_deadline - timedelta(days=42)
    elif bucket == "in_progress":
        window_start = project_deadline - timedelta(days=28)
    else:
        window_start = project_deadline - timedelta(days=21)

    window_end = project_deadline
    total_days = max(7, (window_end - window_start).days)

    # Create milestones: early/mid/late, with more density late
    due_dates = []
    for i in range(n):
        phase = random.random()
        if phase < 0.25:
            offset = int(random.uniform(0.05, 0.35) * total_days)
        elif phase < 0.65:
            offset = int(random.uniform(0.35, 0.75) * total_days)
        else:
            offset = int(random.uniform(0.75, 1.00) * total_days)
        d = window_start + timedelta(days=min(total_days, max(0, offset)))
        hh, mm, ss = rand_time(10, 16)
        dt = datetime(d.year, d.month, d.day, hh, mm, ss)
        due_dates.append(dt)

    # Sort to respect roadmap order (mostly increasing)
    due_dates.sort()
    return due_dates

def start_from_due(due: datetime, est_hours: int, role_hint: str) -> datetime:
    # duration buffer: hours -> days-ish, add meeting load for senior roles
    senior = 1 if any(k in role_hint.lower() for k in ["vp","director","head","manager","lead"]) else 0
    base_days = max(1.0, est_hours / 6.0)  # rough
    base_days *= (1.2 + 0.2*senior)
    jitter = random.uniform(0.7, 1.3)
    start = due - timedelta(days=base_days*jitter)
    return clamp_dt(start, WORKDAY_START_LOCAL, WORKDAY_END_LOCAL)

tasks_rows = []
task_id_counter = 1

for p in projects_sorted:
    pid = p["id"]
    bucket = "completed" if pid in completed_ids else ("in_progress" if pid in in_progress_ids else "not_started")
    deadline = datetime.strptime(p["deadline"], "%Y-%m-%d").date()

    items = roadmaps.get(pid, fallback_roadmaps["roadmaps"][pid])
    items = items[:TASKS_PER_PROJECT]

    due_list = pick_due_dates(deadline, TASKS_PER_PROJECT, bucket)
    statuses = status_sequence(bucket, TASKS_PER_PROJECT)

    for j, (tname, skill_needed) in enumerate(items):
        est = est_hours_for_skill(skill_needed)
        due_dt = due_list[j]
        # assignee/handoffs will be inferred from events; keep placeholder for now
        assignee = None
        handoffs = []
        if bucket == "not_started" or statuses[j] == "not_started":
            start_dt = None
        else:
            # role hint: pick someone likely from dept based on skill (rough)
            role_hint = "Engineer" if any(k in skill_needed.lower() for k in ["react","api","shopify","kubernetes","docker"]) else "IC"
            start_dt = start_from_due(due_dt, est, role_hint)

        tid = f"T{task_id_counter:04d}"
        task_id_counter += 1

        tasks_rows.append({
            "id": tid,
            "project_id": pid,
            "name": str(tname),
            "skill_needed": str(skill_needed),
            "est_hours": int(est),
            "assignee": assignee,      # filled later
            "start": iso_dt(start_dt) if start_dt else None,
            "due": iso_dt(due_dt),
            "status": statuses[j],
            "_handoffs": handoffs,     # filled later
        })

tasks_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tasks_rows])

# ============================================================
# 7) Events generation (Python-only) implementing 6 bottlenecks
#    + backfill tasks.assignee and tasks._handoffs
# ============================================================

from collections import defaultdict

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in keywords)

def _skill_match(emp_skills: List[str], skill_needed: str) -> bool:
    sn = _norm(skill_needed)
    for sk in emp_skills:
        s = _norm(sk)
        if sn in s or s in sn:
            return True
    return False

# Build lookups
emp_by_id = {e["id"]: e for e in employees}
emp_skills_map = {e["id"]: e["skills"] for e in employees}

def pick_pool(role_keywords: List[str], skill_keywords: List[str], k: int) -> List[str]:
    scored = []
    for e in employees:
        score = 0
        role_t = _norm(e["role"])
        if any(rk in role_t for rk in role_keywords): score += 5
        sk_blob = " | ".join([_norm(s) for s in e["skills"]])
        if any(sk in sk_blob for sk in skill_keywords): score += 2
        score += int(e["max_hours"] >= 40)
        scored.append((score, e["id"]))
    scored.sort(reverse=True)
    chosen = [eid for sc,eid in scored if sc > 0][:k]
    if len(chosen) < k:
        chosen = [eid for _,eid in scored[:k]]
    return chosen

data_team = pick_pool(
    ["data", "analytics", "analyst", "bi", "engineer"],
    ["sql", "looker", "dbt", "ga4", "experiment"],
    2
)
engineering_team = pick_pool(
    ["engineer", "engineering", "developer", "devops", "platform"],
    ["react", "api", "shopify", "kubernetes", "docker", "qa automation", "javascript"],
    3
)
creative_team = pick_pool(
    ["designer", "creative", "brand", "copywriter", "art director"],
    ["figma", "photoshop", "illustrator", "copywriting", "graphic design", "email design"],
    3
)
ops_team = pick_pool(
    ["operations", "supply", "logistics", "warehouse", "fulfillment", "procurement"],
    ["vendor", "supply", "logistics", "warehouse", "inventory", "3pl"],
    2
)
legal_team = pick_pool(["legal", "compliance", "counsel"], ["legal", "compliance", "contract"], 1)
finance_team = pick_pool(["finance", "accounting", "fp&a", "controller"], ["finance", "accounting", "tax"], 1)

ANALYTICS_SKILLS = {"sql","looker","dbt","experimentation","ga4"}
CREATIVE_SKILLS  = {"copywriting","art direction","graphic design","email design","email marketing"}
ENG_SKILLS       = {"backend","frontend","shopify","apis","kubernetes","qa automation","javascript","react","docker"}
OPS_SKILLS       = {"vendor","vendor management","supply planning","logistics","procurement","warehouse","inventory"}
FINLEGAL_SKILLS  = {"legal","legal review","compliance","contract","contracting","accounting","tax","fp&a","finance"}

def skill_group(skill_needed: str) -> str:
    s = _norm(skill_needed)
    if any(k in s for k in ANALYTICS_SKILLS): return "analytics"
    if any(k in s for k in CREATIVE_SKILLS):  return "creative"
    if any(k in s for k in ENG_SKILLS):       return "engineering"
    if any(k in s for k in OPS_SKILLS):       return "ops"
    if any(k in s for k in FINLEGAL_SKILLS):  return "finlegal"
    return "other"

dept_by_id = {e["id"]: e["department"] for e in employees}
all_employee_ids = [e["id"] for e in employees]

# Department guardrails keep lifecycle/creative work inside the right swim lane.
GROUP_DEPARTMENT_RULES = {
    "creative": {
        "core": {
            "Creative Studio",
            "Brand Marketing",
            "Performance Marketing",
        },
        "adjacent": {
            "Product Design (UX/UI)",
            "Product Management",
        },
    },
    "analytics": {
        "core": {"Data & Analytics"},
        "adjacent": {
            "Product Management",
            "Performance Marketing",
            "Engineering",
        },
    },
    "engineering": {
        "core": {"Engineering"},
        "adjacent": {
            "Product Management",
            "Data & Analytics",
            "Product Design (UX/UI)",
        },
    },
    "ops": {
        "core": {
            "Supply Chain",
            "Warehouse & Fulfillment Ops",
        },
        "adjacent": {
            "Customer Experience (CX)",
            "Merchandising & Buying",
            "Finance",
        },
    },
    "finlegal": {
        "core": {
            "Finance",
            "Legal & Compliance",
        },
        "adjacent": {
            "Product Management",
            "Supply Chain",
            "Brand Marketing",
        },
    },
    "other": {
        "core": set(DEPARTMENTS),
        "adjacent": set(DEPARTMENTS),
    },
}

def _pool_for_group(group: str, include_adjacent: bool=False, adjacent_only: bool=False) -> List[str]:
    rules = GROUP_DEPARTMENT_RULES.get(group, GROUP_DEPARTMENT_RULES["other"])
    if adjacent_only:
        depts = set(rules.get("adjacent", set()))
    else:
        depts = set(rules.get("core", set()))
        if include_adjacent:
            depts |= set(rules.get("adjacent", set()))
    if not depts:
        depts = set(rules.get("core", set()) | rules.get("adjacent", set()))
    pool = [eid for eid in all_employee_ids if dept_by_id.get(eid) in depts]
    if not pool and include_adjacent and not adjacent_only:
        depts |= set(rules.get("adjacent", set()))
        pool = [eid for eid in all_employee_ids if dept_by_id.get(eid) in depts]
    return pool or all_employee_ids[:]

def rand_business_ts(base: datetime, min_hour=10, max_hour=16) -> datetime:
    dt = base.replace(hour=random.randint(min_hour, max_hour),
                      minute=random.choice([0,15,30,45]),
                      second=0)
    return clamp_dt(dt, WORKDAY_START_LOCAL, WORKDAY_END_LOCAL)

def add_hours(ts: datetime, hours: int) -> datetime:
    return ts + timedelta(hours=int(hours))

def nearest_burst_anchor(ts: datetime) -> datetime:
    d = ts.date()
    wd = d.weekday()  # Mon=0
    target_wd = 0 if random.random() < 0.65 else 2
    delta = (target_wd - wd) % 7
    anchor_date = d + timedelta(days=delta)
    anchor = datetime(anchor_date.year, anchor_date.month, anchor_date.day, 10, 0, 0)
    anchor += timedelta(minutes=random.choice([0,10,20,30,40]))
    return anchor

def best_fit_assignee(skill_needed: str, exclude: Optional[str]=None, pool: Optional[List[str]]=None) -> Optional[str]:
    candidates = []
    pool_ids = set(pool) if pool else None
    for eid, sks in emp_skills_map.items():
        if exclude and eid == exclude: continue
        if pool_ids is not None and eid not in pool_ids: continue
        if _skill_match(sks, skill_needed):
            candidates.append(eid)
    if not candidates:
        return None
    candidates.sort(key=lambda eid: emp_by_id[eid]["max_hours"], reverse=True)
    return candidates[0]

def pick_senior_reviewer() -> Optional[str]:
    seniors = []
    for e in employees:
        r = _norm(e["role"])
        if any(k in r for k in ["director","vp","head","lead","manager"]):
            seniors.append(e["id"])
    pref = [eid for eid in seniors if eid in creative_team]
    pool = pref or seniors
    return random.choice(pool) if pool else None

event_rows = []
def add_event(task_id: str, typ: str, ts: datetime, from_a: Optional[str], to_a: Optional[str]):
    event_rows.append({
        "task_id": task_id,
        "type": typ,
        "timestamp": iso_dt(ts),
        "from_assignee": from_a,
        "to_assignee": to_a,
    })

def _queue_label(tag: str, assignee: str) -> str:
    core = tag.upper().replace(" ", "_")
    return f"QUEUE::{core}::{assignee}"

def add_idle_and_handoff(
    task_id: str,
    current_owner: str,
    last_ts: datetime,
    next_owner: str,
    wait_range: Tuple[int, int],
    tag: str,
    service_range: Tuple[int, int] = (1, 4),
    forced_pickup: Optional[datetime] = None,
) -> Tuple[datetime, str]:
    """
    Inserts an explicit queue/idle record so stage completion and next pickup diverge.
    Returns (pickup_ts, queue_label).
    """
    service_hours = random.randint(service_range[0], service_range[1])
    complete_ts = rand_business_ts(add_hours(last_ts, service_hours))
    queue_label = _queue_label(tag, current_owner)
    add_event(task_id, "queue", complete_ts, current_owner, queue_label)

    if forced_pickup:
        pickup_ts = forced_pickup
        if pickup_ts <= complete_ts:
            pickup_ts = rand_business_ts(add_hours(complete_ts, wait_range[0]))
    else:
        wait_hours = random.randint(wait_range[0], wait_range[1])
        pickup_ts = rand_business_ts(add_hours(complete_ts, wait_hours))

    add_event(task_id, "handoff", pickup_ts, queue_label, next_owner)
    return pickup_ts, queue_label

# Start with a "default initial owner" based on skill match (can be mismatched sometimes)
def initial_owner_for_task(skill_needed: str, group: str) -> str:
    core_pool = _pool_for_group(group, include_adjacent=False)
    adjacent_only_pool = _pool_for_group(group, adjacent_only=True)
    misalign_prob = 0.15 if group != "other" else 0.10

    if random.random() < misalign_prob and adjacent_only_pool:
        pick_pool = adjacent_only_pool
    else:
        pick_pool = core_pool

    bf = best_fit_assignee(skill_needed, pool=pick_pool)
    if bf:
        return bf
    return random.choice(pick_pool or all_employee_ids)

def _random_owner_from_group(group: str, exclude: Optional[str]=None) -> str:
    pool = [eid for eid in _pool_for_group(group, include_adjacent=True) if eid != exclude]
    if not pool:
        pool = [eid for eid in all_employee_ids if eid != exclude]
    return random.choice(pool)

# Build events for each task row
task_row_by_id = {r["id"]: r for r in tasks_rows}

for row in tasks_rows:
    status = row["status"]
    if status == "not_started":
        continue
    if row["start"] is None:
        continue

    tid = row["id"]
    skill_needed = row["skill_needed"]
    group = skill_group(skill_needed)

    start_dt = clamp_dt(parse_iso_dt(row["start"]), WORKDAY_START_LOCAL, WORKDAY_END_LOCAL)
    due_dt = parse_iso_dt(row["due"])

    # initial assignee (may be mismatched)
    current = initial_owner_for_task(skill_needed, group)
    add_event(tid, "start", start_dt, None, current)
    last_ts = start_dt

    mismatch = not _skill_match(emp_skills_map.get(current, []), skill_needed)

    # ----------------------------
    # Bottleneck #6: Misallocation
    # ----------------------------
    if mismatch:
        better = best_fit_assignee(skill_needed, exclude=current)
        if better:
            h_ts = rand_business_ts(add_hours(last_ts, random.randint(6, 24)))
            add_event(tid, "handoff", h_ts, current, better)
            current = better
            last_ts = h_ts
            # occasional bounce
            if random.random() < 0.25:
                bounce_ts = rand_business_ts(add_hours(last_ts, random.randint(12, 36)))
                bounce_owner = _random_owner_from_group(group, exclude=current)
                add_event(tid, "handoff", bounce_ts, current, bounce_owner)
                last_ts = bounce_ts
                current = bounce_owner

    # ---------------------------------------------
    # Bottleneck #1: Analytics backlog + queue burst
    # ---------------------------------------------
    if group == "analytics":
        data_owner = random.choice(data_team) if data_team else current
        if current != data_owner:
            anchor = nearest_burst_anchor(last_ts)
            pickup_guess = rand_business_ts(anchor, 10, 12)
            wait_range = (24, 96)
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                data_owner,
                wait_range,
                tag="DATA_QUEUE",
                forced_pickup=pickup_guess,
            )
            current = data_owner
        else:
            # even if already on data team, force a queued pickup
            wait_range = (24, 96)
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                current,
                wait_range,
                tag="DATA_QUEUE",
            )

    # --------------------------------------------
    # Bottleneck #2: Creative review ping-pong
    # --------------------------------------------
    if group == "creative":
        designer = random.choice(creative_team) if creative_team else current
        reviewer = pick_senior_reviewer() or designer

        hops = random.randint(2, 4)
        seq = []
        if current != designer:
            seq.append(designer)
        for i in range(hops):
            seq.append(reviewer if i % 2 == 0 else designer)

        # dedupe consecutive
        clean = []
        for x in seq:
            if not clean or clean[-1] != x:
                clean.append(x)

        for nxt in clean:
            if nxt == current:
                continue
            if nxt == reviewer:
                wait_range = (48, 120)
                tag = "CREATIVE_REVIEW"
            else:
                wait_range = (12, 48)
                tag = "CREATIVE_REWORK"
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                nxt,
                wait_range,
                tag=tag,
                service_range=(2, 6),
            )
            current = nxt

    # --------------------------------------------
    # Bottleneck #4: Ops waiting on vendors
    # --------------------------------------------
    if group == "ops":
        ops_owner = random.choice(ops_team) if ops_team else current
        if current != ops_owner:
            h_ts = rand_business_ts(add_hours(last_ts, random.randint(2, 12)))
            add_event(tid, "handoff", h_ts, current, ops_owner)
            current = ops_owner
            last_ts = h_ts

        # vendor silence 72–168h
        last_ts, _ = add_idle_and_handoff(
            tid,
            current,
            last_ts,
            current,
            (72, 168),
            tag="VENDOR_WAIT",
            service_range=(1, 3),
        )

        # often handoff to finance/merch after response
        secondary = finance_team[0] if finance_team else random.choice([e["id"] for e in employees if e["id"] != current])
        if random.random() < 0.7 and secondary != current:
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                secondary,
                (12, 36),
                tag="OPS_ESCALATION",
                service_range=(2, 6),
            )
            current = secondary

    # --------------------------------------------
    # Bottleneck #5: Finance/Legal gate late
    # --------------------------------------------
    if group == "finlegal":
        gate_owner = (legal_team[0] if legal_team else None) or (finance_team[0] if finance_team else None) or current

        # late handoff within 12–60h of due
        target = due_dt - timedelta(hours=random.randint(12, 60))
        h_ts = rand_business_ts(target)
        if h_ts <= last_ts:
            h_ts = rand_business_ts(add_hours(last_ts, random.randint(2, 12)))
        if gate_owner != current:
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                gate_owner,
                (12, 48),
                tag="LEGAL_QUEUE",
                forced_pickup=h_ts,
            )
            current = gate_owner
        else:
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                current,
                (12, 48),
                tag="LEGAL_QUEUE",
                forced_pickup=h_ts,
            )

        # clarification loop sometimes
        if random.random() < 0.30:
            requester = random.choice(creative_team) if creative_team else random.choice([e["id"] for e in employees])
            if requester != current:
                last_ts, _ = add_idle_and_handoff(
                    tid,
                    current,
                    last_ts,
                    requester,
                    (12, 36),
                    tag="LEGAL_CLARIFY",
                )
                current = requester
                last_ts, _ = add_idle_and_handoff(
                    tid,
                    current,
                    last_ts,
                    gate_owner,
                    (12, 36),
                    tag="LEGAL_REVIEW",
                )
                current = gate_owner

        if status == "completed":
            last_ts_queue_label = _queue_label("LEGAL_APPROVAL", current)
            service_complete = rand_business_ts(add_hours(last_ts, random.randint(2, 6)))
            add_event(tid, "queue", service_complete, current, last_ts_queue_label)
            last_ts = rand_business_ts(add_hours(service_complete, random.randint(36, 96)))
            add_event(tid, "end", last_ts, current, None)
        continue

    # --------------------------------------------
    # Bottleneck #3: Engineering interrupts + late ends
    # --------------------------------------------
    if group == "engineering":
        if engineering_team and current not in engineering_team:
            eng_owner = random.choice(engineering_team)
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                eng_owner,
                (6, 18),
                tag="ENG_INTAKE",
            )
            current = eng_owner

        # context switching spikes
        if engineering_team and random.random() < 0.55:
            spikes = random.randint(1, 3)
            for _ in range(spikes):
                nxt = random.choice([e for e in engineering_team if e != current] or engineering_team)
                last_ts, _ = add_idle_and_handoff(
                    tid,
                    current,
                    last_ts,
                    nxt,
                    (8, 36),
                    tag="ENG_SWITCH",
                )
                current = nxt

        # blocked: no end, long gap
        if status == "blocked":
            last_ts, _ = add_idle_and_handoff(
                tid,
                current,
                last_ts,
                current,
                (96, 240),
                tag="ENG_BLOCK",
                service_range=(2, 6),
            )
            if engineering_team and random.random() < 0.4:
                nxt = random.choice([e for e in engineering_team if e != current] or engineering_team)
                last_ts, _ = add_idle_and_handoff(
                    tid,
                    current,
                    last_ts,
                    nxt,
                    (24, 48),
                    tag="ENG_ESCALATION",
                )
                current = nxt
            continue

        # late completion bias (near/after due)
        drift = random.randint(0, 36)
        end_target = due_dt + timedelta(hours=drift) if random.random() < 0.6 else due_dt - timedelta(hours=random.randint(0, 12))
        end_ts = rand_business_ts(end_target)
        if end_ts <= last_ts:
            end_ts = rand_business_ts(add_hours(last_ts, random.randint(6, 24)))

        if status == "completed":
            add_event(tid, "end", end_ts, current, None)
        continue

    # default end for completed
    if status == "completed":
        # Inject a small backlog wait so every task sees at least one queue gap
        backlog_wait = (1, 4)
        backlog_service = (1, 2)
        last_ts, _ = add_idle_and_handoff(
            tid,
            current,
            last_ts,
            current,
            backlog_wait,
            tag="GENERAL_BACKLOG",
            service_range=backlog_service,
        )
        end_ts = rand_business_ts(due_dt - timedelta(hours=random.randint(1, 12)))
        if end_ts <= last_ts:
            end_ts = rand_business_ts(add_hours(last_ts, random.randint(6, 24)))
        # mismatch inflation (30–80%)
        if mismatch:
            inflate = random.uniform(1.3, 1.8)
            extra = int(((end_ts - start_dt).total_seconds()/3600.0) * (inflate - 1.0))
            end_ts = rand_business_ts(add_hours(end_ts, max(6, extra)))
        add_event(tid, "end", end_ts, current, None)

events_df = pd.DataFrame(event_rows)

# Backfill tasks.assignee + task handoffs from events
# - assignee = first start event to_assignee
# - handoffs = sequence of to_assignee in handoff events
handoffs_map = defaultdict(list)
assignee_map = {}

for _, ev in events_df.sort_values(["task_id","timestamp"]).iterrows():
    if ev["type"] == "start" and ev["to_assignee"]:
        assignee_map.setdefault(ev["task_id"], ev["to_assignee"])
    if ev["type"] == "handoff" and ev["to_assignee"]:
        handoffs_map[ev["task_id"]].append(ev["to_assignee"])

for r in tasks_rows:
    tid = r["id"]
    r["assignee"] = assignee_map.get(tid)
    r["_handoffs"] = handoffs_map.get(tid, [])

tasks_df = pd.DataFrame([{k:v for k,v in r.items() if not k.startswith("_")} for r in tasks_rows])

# ============================================================
# 8) Exports: CSV + Excel + JSON
# ============================================================

# CSV exports
employees_df.to_csv(f"{OUTPUT_DIR}/employees.csv", index=False)
availability_df.to_csv(f"{OUTPUT_DIR}/availability.csv", index=False)
projects_df.to_csv(f"{OUTPUT_DIR}/projects.csv", index=False)
tasks_df.to_csv(f"{OUTPUT_DIR}/tasks.csv", index=False)
events_df.to_csv(f"{OUTPUT_DIR}/events.csv", index=False)

# # Excel workbook
# with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
#     employees_df.to_excel(writer, sheet_name="employees", index=False)
#     availability_df.to_excel(writer, sheet_name="availability", index=False)
#     projects_df.to_excel(writer, sheet_name="projects", index=False)
#     tasks_df.to_excel(writer, sheet_name="tasks", index=False)
#     events_df.to_excel(writer, sheet_name="events", index=False)

# JSON dump (structured)
payload = {
    "employees": employees,
    "availability": availability_rows,
    "projects": project_rows,
    "tasks": tasks_rows,
    "events": event_rows,
}
with open(JSON_PATH, "w") as f:
    json.dump(payload, f, indent=2)

print("Done.")
# print("Excel:", EXCEL_PATH)
print("JSON:", JSON_PATH)
print("CSVs:", [f"{OUTPUT_DIR}/{x}.csv" for x in ["employees","availability","projects","tasks","events"]])

# Quick sanity peek: show a few rows
print("Employee df: ", employees_df.head(5), '\n')
print("Availibility df: ", availability_df.head(5), '\n')
print("Projects df: ",projects_df.head(5), '\n')
print("Tasks df: ",tasks_df.head(5), '\n')
print("Events df: ",events_df.head(10), '\n')
