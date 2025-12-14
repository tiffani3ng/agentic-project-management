"""Data loading helpers for the workflow optimization MVP."""
import ast
from typing import List

import pandas as pd


def _parse_skills(skills_raw: str) -> List[str]:
    if pd.isna(skills_raw):
        return []
    try:
        parsed = ast.literal_eval(skills_raw)
        if isinstance(parsed, list):
            return [str(skill).strip() for skill in parsed]
    except (ValueError, SyntaxError):
        pass
    return [skill.strip() for skill in skills_raw.split(",")]


def load_employees(path: str) -> pd.DataFrame:
    employees = pd.read_csv(path)
    employees["skills"] = employees["skills"].apply(_parse_skills)
    return employees


def load_availability(path: str) -> pd.DataFrame:
    availability = pd.read_csv(path, parse_dates=["date"])
    return availability


def load_projects(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["deadline"])


def load_tasks(path: str) -> pd.DataFrame:
    tasks = pd.read_csv(path, parse_dates=["start", "due"], keep_default_na=False)
    tasks["status"] = tasks.get("status", "").fillna("")
    tasks["assignee"] = tasks["assignee"].fillna("")
    return tasks


def load_events(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"], keep_default_na=False)
