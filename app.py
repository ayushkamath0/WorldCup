"""
Streamlit Soccer Analytics & Predictive Forecasting Application (World Cup 2026)

Architecture & Data Flow Topology:
──────────────────────────────────────────────────────────────────────────────
1. Ingestion Layer (External APIs & Scrapers):
   - Football-Data.org API v4: Serves as the primary real-time and scheduled scheduling/results
     backbone, delivering live match state, scores, and official tournament group brackets.
   - Live/Archival ETL Scraping Pipelines: Targeted scrapers parse detailed team and player attributes:
     * eloratings.net -> Real-time international Elo ratings used for baseline metrics.
     * FBref -> Squad match logs, tactical formation variance counters, and player tracking.
     * Transfermarkt -> Injury availability metrics, squad depth, and international caps/experience.
     * Club Elo -> Regional league coefficients and domestic tier weights.

2. Predictive & Simulation Engine Layer:
   - Vectorized Poisson Model: Evaluates independent attacking and defensive coefficients derived from 
     historical goal-scoring distributions to build a baseline discrete score probability matrix.
   - XGBoost Classifier Calibration Blend: An extreme gradient-boosted tree model trained on synthetic
     distributions calibrations. It accepts rating differentials, squad-attribute adjustments, and
     vectorized Poisson outcomes to output calibrated 3-way match probabilities (Home, Draw, Away).
   - Monte Carlo Tournament Tree Simulator: Evaluates tournament state over thousands of iterations, 
     accounting for locked real-world fixture overrides and tie-breaker progressions.

3. Presentation Layer:
   - Streamlit Multi-tab Reactive Framework: Managed via session-state routing keys combined with 
     fragment-isolated caching boundaries to minimize redundant network and computational overhead.

Author: Ayush Kamath
Copyright: © 2026 Ayush Kamath. All Rights Reserved.
"""

# ============================================================================
# 1. IMPORTS
# ============================================================================
import io
import html
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from scipy.stats import poisson
import streamlit as st
import streamlit.components.v1 as components


# ─── ML / model persistence ─────────────────────────────────────────────────
try:
    import xgboost as xgb
    import joblib
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

ML_AVAILABLE = XGB_AVAILABLE and SK_AVAILABLE


# ============================================================================
# 2. CONSTANTS
# ============================================================================

API_BASE_URL    = "https://api.football-data.org/v4"
REQUEST_TIMEOUT = 10
WORLD_CUP_LEAGUE_ID = "WC"
WORLD_CUP_SEASON    = 2026
DEFAULT_TEAM_STATS  = {"GF/Game": 1.3, "GA/Game": 1.3, "Rating": 0.5}

#  Primary navigation tabs (session-state driven so links can jump tabs)
TAB_LABELS = [
    "About This Project",
    "Live Fixtures",
    "Team Ratings",
    "Match Predictor",
    "World Cup Simulator",
]

# ─── Expert-level team rating architecture ──────────────────────────────────
RATING_BLEND_WEIGHTS = {"xp": 0.25, "sos_goals": 0.15, "elo": 0.60}
GOAL_PERFORMANCE_EXPONENT = 1.7
TIME_DECAY_RATE = 0.65

# ─── Dynamic global rankings (eloratings.net) ───────────────────────────────
ELO_WORLD_TSV_URL = "https://www.eloratings.net/World.tsv"
ELO_TEAMS_TSV_URL = "https://www.eloratings.net/en.teams.tsv"

ELO_NAME_MAP: dict[str, str] = {
    "United States":  "USA",
    "Korea Republic": "South Korea",
    "IR Iran":        "Iran",
    "Czechia":        "Czech Republic",
}

ELO_MODIFIER_RANGE = (0.35, 0.95)

WC_GROUPS_COUNT        = 12
WC_THIRD_PLACE_QUALIFY = 8
POISSON_WEIGHT = 0.65

WC_HISTORICAL = {
    "avg_goals_per_match": 2.64,
    "draw_rate":           0.231,
    "stronger_wins_rate":  0.621,
}

ML_MODEL_PATH = Path(__file__).parent / "wc_ml_model.pkl"

SCRAPE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, fill: true) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─── FBref squad IDs (national teams) ────────────────────────────────────────
FBREF_SQUAD_IDS: dict[str, str] = {
    "Argentina": "8cec06e1", "Brazil": "e8ef8cd4", "France": "76f4a741",
    "England": "cff3d9bb", "Germany": "10479c02", "Spain": "53a2f082",
    "Portugal": "785361d4", "Netherlands": "748a5abb", "Belgium": "fb6f0dc1",
    "Italy": "8fa3b1f5", "Uruguay": "5c0fc48e", "Colombia": "f4bb648f",
    "Mexico": "8ab96c0c", "USA": "7b3b0c70", "Canada": "a29a14a5",
    "Japan": "cc3d3a41", "Morocco": "e0dd0fad", "Senegal": "7041538f",
    "Australia": "ccf25072", "South Korea": "fbdb4d8f", "Ecuador": "7f9d6b96",
    "Denmark": "445a1f39", "Switzerland": "fd6114db", "Croatia": "b4f6f5d2",
    "Serbia": "e4c5b4f3", "Wales": "f0b25e69", "Ghana": "79c23ff7",
    "Cameroon": "68cfb8c9", "Tunisia": "bbe3fd59", "Iran": "8a2fb6b6",
    "Saudi Arabia": "91847c82", "Poland": "e8a44543", "Qatar": "cf8875a0",
    "Costa Rica": "dcb6dbfb", "Panama": "3ceb1c59", "Honduras": "ca1a8bc7",
    "Venezuela": "7e6c53d6", "Chile": "8b36c4da", "Paraguay": "1af3e24e",
    "Peru": "fd7ac3fd", "Nigeria": "94729c41", "Egypt": "fc3941c0",
    "Ivory Coast": "f7ed1f8d", "Ukraine": "bd5fc852", "Austria": "b3ba3140",
    "Turkey": "44f16e0d", "Romania": "c0e8b6b3",
}

# ─── Transfermarkt national team IDs ─────────────────────────────────────────
TM_TEAM_IDS: dict[str, str] = {
    "Argentina": "3437", "Brazil": "3439", "France": "3377", "England": "3388",
    "Germany": "3376", "Spain": "3375", "Portugal": "3384", "Netherlands": "3379",
    "Belgium": "3382", "Italy": "3376", "Uruguay": "3449", "Colombia": "3474",
    "Mexico": "3467", "USA": "3505", "Canada": "3511", "Japan": "3557",
    "Morocco": "3615", "Senegal": "3609", "Australia": "3590", "South Korea": "3562",
    "Ecuador": "3476", "Denmark": "3395", "Switzerland": "3389", "Croatia": "3399",
    "Serbia": "3438", "Wales": "3387", "Ghana": "3607", "Cameroon": "3608",
    "Tunisia": "3621", "Iran": "3555", "Saudi Arabia": "3553", "Poland": "3396",
    "Qatar": "3552",
}

# ─── Club Elo name overrides ────────────────────────────────────────────────
CLUB_ELO_NAMES: dict[str, str] = {
    "Inter Milan": "Inter", "AC Milan": "Milan",
    "Paris Saint-Germain": "Paris SG", "Atlético Madrid": "Atletico Madrid",
    "Borussia Dortmund": "Dortmund", "Bayer Leverkusen": "Leverkusen",
}

SAF_WEIGHTS = {
    "tactical":   0.22,
    "form":       0.22,
    "fitness":    0.18,
    "league":     0.20,
    "experience": 0.18,
}


# ============================================================================
# 3. ML MODEL TRAIN / LOAD
# ============================================================================

def _poisson_probs_vec(xg_h: float, xg_a: float, mg: int = 8):
    ph = np.array([poisson.pmf(g, max(xg_h, 0.01)) for g in range(mg + 1)])
    pa = np.array([poisson.pmf(g, max(xg_a, 0.01)) for g in range(mg + 1)])
    m  = np.outer(ph, pa)
    hw = float(np.tril(m, -1).sum())
    dr = float(np.diag(m).sum())
    aw = float(np.triu(m, 1).sum())
    t  = hw + dr + aw
    return hw / t, dr / t, aw / t


def _generate_training_data(n: int = 80_000, seed: int = 42):
    rng = np.random.default_rng(seed)
    rh = rng.uniform(0.35, 0.95, n);  ra = rng.uniform(0.35, 0.95, n)
    sh = rng.uniform(0.60, 0.98, n);  sa = rng.uniform(0.60, 0.98, n)
    lh = rng.uniform(0.65, 0.98, n);  la = rng.uniform(0.65, 0.98, n)

    base = 1.32
    xg_h = np.clip(rh*(sh/0.85)*lh*base / np.maximum(ra*(0.85/sa)*0.90, 0.3), 0.25, 5.0)
    xg_a = np.clip(ra*(sa/0.85)*la*base / np.maximum(rh*(0.85/sh)*1.02, 0.3), 0.25, 5.0)
    s    = WC_HISTORICAL["avg_goals_per_match"] / (xg_h + xg_a).mean()
    xg_h *= s;  xg_a *= s

    gh = rng.poisson(xg_h);  ga = rng.poisson(xg_a)
    y  = np.where(gh > ga, 0, np.where(gh < ga, 2, 1)).astype(np.int32)

    probs = np.array([_poisson_probs_vec(xg_h[i], xg_a[i]) for i in range(n)])
    X = np.column_stack([xg_h, xg_a, rh-ra, sh-sa, lh-la,
                         probs[:, 0], probs[:, 1], probs[:, 2]])
    return X, y


def train_ml_model(save_path: Path = ML_MODEL_PATH):
    if not ML_AVAILABLE:
        return None

    X, y = _generate_training_data(80_000)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    clf = xgb.XGBClassifier(
        n_estimators=120, max_depth=4, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42,
        use_label_encoder=False, verbosity=0, n_jobs=-1,
        early_stopping_rounds=15,
    )
    clf.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

    model_bundle = {"clf": clf, "scaler": scaler}
    joblib.dump(model_bundle, save_path)
    return model_bundle


@st.cache_resource(show_spinner=False)
def load_ml_model():
    if not ML_AVAILABLE:
        return None
    if ML_MODEL_PATH.exists():
        try:
            return joblib.load(ML_MODEL_PATH)
        except Exception:
            pass
    with st.spinner("First launch - training ML model (~5 s)..."):
        return train_ml_model(ML_MODEL_PATH)


def ml_probs(xg_home: float, xg_away: float,
             rating_diff: float = 0.0, saf_diff: float = 0.0,
             league_diff: float = 0.0,
             model_bundle: dict = None):
    if model_bundle is None:
        return None
    try:
        phw, pdr, paw = _poisson_probs_vec(xg_home, xg_away)
        feat = np.array([[xg_home, xg_away, rating_diff, saf_diff, league_diff,
                          phw, pdr, paw]])
        feat_s = model_bundle["scaler"].transform(feat)
        proba  = model_bundle["clf"].predict_proba(feat_s)[0]
        return float(proba[0]), float(proba[1]), float(proba[2])
    except Exception:
        return None


def blended_probs(xg_home: float, xg_away: float,
                  rating_diff: float = 0.0, saf_diff: float = 0.0,
                  league_diff: float = 0.0,
                  model_bundle: dict = None):
    p_hw, p_dr, p_aw = _poisson_probs_vec(xg_home, xg_away)
    ml = ml_probs(xg_home, xg_away, rating_diff, saf_diff, league_diff, model_bundle)
    if ml is None:
        return p_hw, p_dr, p_aw
    m_hw, m_dr, m_aw = ml
    w = POISSON_WEIGHT
    hw = w*p_hw + (1-w)*m_hw
    dr = w*p_dr + (1-w)*m_dr
    aw = w*p_aw + (1-w)*m_aw
    t  = hw + dr + aw
    return hw/t, dr/t, aw/t


# ============================================================================
# 4. LOW-LEVEL API HELPERS
# ============================================================================

def api_request(endpoint: str, api_key: str, params: dict = None, max_retries: int = 2) -> dict:
    if not api_key:
        return {}
    url     = f"{API_BASE_URL}/{endpoint}"
    headers = {"X-Auth-Token": api_key}

    # Transient network blips (a dropped handshake, a slow DNS lookup, one
    # lost packet) shouldn't surface as a user-facing error on the first
    # failure — retry a couple of times with a short backoff before giving up.
    r = None
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params or {}, timeout=REQUEST_TIMEOUT)
            last_exc = None
            break
        except requests.exceptions.Timeout as exc:
            last_exc = ("timeout", exc)
        except requests.exceptions.ConnectionError as exc:
            last_exc = ("connection", exc)
        except requests.exceptions.RequestException as exc:
            # Not a transient blip (e.g. malformed URL) — no point retrying.
            st.error(f"Network error: {exc}")
            return {}
        if attempt < max_retries:
            time.sleep(0.5 * (attempt + 1))

    if last_exc is not None:
        kind, exc = last_exc
        if kind == "timeout":
            st.error("Request timed out. Please try again.")
        else:
            st.error("Could not connect to Football-Data.org.")
        return {}

    if r.status_code in (401, 403):
        st.error("Invalid API token.")
        return {}
    if r.status_code == 429:
        st.error("API rate limit reached. Wait and retry.")
        return {}
    if r.status_code != 200:
        return {}
    try:
        return r.json()
    except ValueError:
        return {}


@st.cache_data(ttl=600, show_spinner=False)
def check_api_key(api_key: str):
    if not api_key:
        return False, "No API token provided."
    try:
        r = requests.get(f"{API_BASE_URL}/competitions",
                         headers={"X-Auth-Token": api_key}, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.RequestException:
        return False, "Could not reach Football-Data.org."
    if r.status_code in (401, 403):
        return False, "Invalid API token."
    if r.status_code != 200:
        return False, f"API status {r.status_code}."
    return True, "Connected to Football-Data.org successfully!"


def _regulation_score(score_obj: dict | None) -> tuple[int | None, int | None]:
    """
    Extract "goals for" from a football-data.org v4 `score` node, guaranteed to
    EXCLUDE penalty shootout goals.

    football-data.org v4 score object shape:
        {
          "winner":    "HOME_TEAM" | "AWAY_TEAM" | "DRAW" | None,
          "duration":  "REGULAR" | "EXTRA_TIME" | "PENALTY_SHOOTOUT",
          "fullTime":  {"home": int, "away": int},   # final goals - EXCLUDES penalties
          "halfTime":  {"home": int, "away": int},
          "extraTime": {"home": int, "away": int},   # goals scored *within* ET only
          "penalties": {"home": int, "away": int},   # shootout goals - NEVER "goals for"
        }

    `fullTime` is the cumulative score at the end of play (90 min, or 120 min if
    extra time was needed) and by API design never includes penalty-shootout
    goals - those only ever live under the separate `penalties` node. Every
    stat in this app (GF/Game, GA/Game, Attack/Defense, etc.) should read goals
    through this helper rather than touching `score` directly, so a shootout
    can never inflate a team's goals-for tally.
    """
    reg = score_obj.get("regularTime")
    if reg and reg.get("home") is not None:
        return reg.get("home"), reg.get("away")
        
    ft = score_obj.get("fullTime") or {}
    return ft.get("home"), ft.get("away")


def _safe_score(score_dict: dict, side: str) -> str:
    home, away = _regulation_score(score_dict)
    v = home if side == "home" else away
    return str(v) if v is not None else "-"


def _safe_float(val) -> float | None:
    try:
        return float(str(val).replace(",", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def _scrape_get(url: str, timeout: int = 12) -> BeautifulSoup | None:
    try:
        time.sleep(0.5)
        r = requests.get(url, headers=SCRAPE_HEADERS, timeout=timeout)
        if r.status_code == 429:
            time.sleep(3)
            r = requests.get(url, headers=SCRAPE_HEADERS, timeout=timeout)
        return BeautifulSoup(r.text, "lxml") if r.status_code == 200 else None
    except Exception:
        return None


# ============================================================================
# 5. FOOTBALL-DATA.ORG FETCH FUNCTIONS
# ============================================================================

# API Optimization: Raised from 300 to 3600 seconds to fit Community Cloud rate limits
@st.cache_data(ttl=3600, show_spinner="Fetching upcoming fixtures...")
def get_fixtures(api_key, league_id, season, next_n=15) -> pd.DataFrame:
    payload = api_request(f"competitions/{league_id}/matches", api_key,
                          {"season": season, "status": "SCHEDULED"})
    raw  = payload.get("matches", []) if isinstance(payload, dict) else []
    rows = []
    for item in raw[:next_n]:
        comp = item.get("competition") or {}
        ht   = item.get("homeTeam")    or {}
        at   = item.get("awayTeam")    or {}
        rows.append({
            "Match ID": item.get("id"),
            "Date":   (item.get("utcDate") or "")[:16].replace("T", " "),
            "League": comp.get("name", str(league_id)),
            "Round":  str(item.get("matchday") or ""),
            "Match Ref": item.get("matchday") or item.get("id"),
            "Home":   ht.get("name", ""),
            "Away":   at.get("name", ""),
            "Status": item.get("status", ""),
        })
    return pd.DataFrame(rows)


# API Optimization: Raised from 300 to 3600 seconds to fit Community Cloud rate limits
@st.cache_data(ttl=3600, show_spinner="Fetching recent results...")
def get_results(api_key, league_id, season, last_n=15) -> pd.DataFrame:
    payload = api_request(f"competitions/{league_id}/matches", api_key,
                          {"season": season, "status": "FINISHED"})
    raw  = payload.get("matches", []) if isinstance(payload, dict) else []
    recent = list(reversed(raw[-last_n:] if len(raw) > last_n else raw))
    rows = []
    for item in recent:
        comp  = item.get("competition") or {}
        ht    = item.get("homeTeam")    or {}
        at    = item.get("awayTeam")    or {}
        score = item.get("score")       or {}
        hg, ag = _regulation_score(score)
        pen = score.get("penalties") or {}
        rows.append({
            "Match ID":   item.get("id"),
            "Date":       (item.get("utcDate") or "")[:10],
            "League":     comp.get("name", ""),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Home":       ht.get("name", ""),
            "Score":      f"{_safe_score(score,'home')} - {_safe_score(score,'away')}",
            "Away":       at.get("name", ""),
            "Home Goals": hg,
            "Away Goals": ag,
            "Winner":     score.get("winner"),
            "Status":     item.get("status", "FINISHED"),
            "Penalties Home": pen.get("home"),
            "Penalties Away": pen.get("away"),
            "Raw Score":  score,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60, show_spinner="Checking for live matches...")
def get_live_fixtures(api_key, league_id=None) -> pd.DataFrame:
    endpoint = f"competitions/{league_id}/matches" if league_id else "matches"
    payload  = api_request(endpoint, api_key, {"status": "IN_PLAY"})
    raw      = payload.get("matches", []) if isinstance(payload, dict) else []
    rows = []
    for item in raw:
        comp  = item.get("competition") or {}
        ht    = item.get("homeTeam") or {}
        at    = item.get("awayTeam") or {}
        score = item.get("score")       or {}
        hg, ag = _regulation_score(score)
        rows.append({
            "Match ID":   item.get("id"),
            "League":     comp.get("name", ""),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Minute":     "LIVE",
            "Home":       ht.get("name", ""),
            "Away":       at.get("name", ""),
            "Score":      f"{_safe_score(score,'home')} - {_safe_score(score,'away')}",
            "Home Goals": hg,
            "Away Goals": ag,
            "Winner":     score.get("winner"),
            "Status":     item.get("status", "IN_PLAY"),
            "Raw Score":  score,
        })
    return pd.DataFrame(rows)


def _goal_ratio_rating(gf: float, ga: float, exp: float = GOAL_PERFORMANCE_EXPONENT) -> float:
    if gf <= 0 and ga <= 0:
        return 0.5
    adjusted_ga = ga + 0.25 if ga == 0 else ga
    return (gf ** exp) / ((gf ** exp) + (adjusted_ga ** exp))


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_dynamic_fifa_rankings() -> dict[str, float]:
    try:
        teams_resp = requests.get(ELO_TEAMS_TSV_URL, headers=SCRAPE_HEADERS, timeout=REQUEST_TIMEOUT)
        teams_resp.raise_for_status()
    except requests.exceptions.RequestException:
        return {}

    code_to_name: dict[str, str] = {}
    for line in teams_resp.text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, name = parts[0].strip(), parts[1].strip()
        if not code or not name or code.endswith("_loc"):
            continue
        code_to_name[code] = name

    if not code_to_name:
        return {}

    try:
        ratings_resp = requests.get(ELO_WORLD_TSV_URL, headers=SCRAPE_HEADERS, timeout=REQUEST_TIMEOUT)
        ratings_resp.raise_for_status()
    except requests.exceptions.RequestException:
        return {}

    rankings: dict[str, float] = {}
    for line in ratings_resp.text.splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        code   = cols[2].strip()
        rating = _safe_float(cols[3])
        if not code or rating is None:
            continue
        raw_name  = code_to_name.get(code, code)
        canonical = ELO_NAME_MAP.get(raw_name, raw_name)
        if canonical not in rankings or rating > rankings[canonical]:
            rankings[canonical] = rating

    return rankings


def _lookup_elo_rating(team: str, rankings: dict[str, float]) -> float | None:
    if not rankings:
        return None
    if team in rankings:
        return rankings[team]
    lowered = {k.lower(): v for k, v in rankings.items()}
    return lowered.get(team.lower())


def _elo_to_rating_modifier(elo: float, elo_min: float, elo_max: float,
                             modifier_range: tuple[float, float] = ELO_MODIFIER_RANGE) -> float:
    lo, hi = modifier_range
    if elo_max <= elo_min:
        return (lo + hi) / 2
    norm = (elo - elo_min) / (elo_max - elo_min)
    norm = min(max(norm, 0.0), 1.0)
    return lo + norm * (hi - lo)


def _match_points(goals_for: float, goals_against: float) -> float:
    if goals_for > goals_against:
        return 3.0
    if goals_for == goals_against:
        return 1.0
    return 0.0


def _time_decay_weights(n: int, decay_rate: float = TIME_DECAY_RATE) -> list[float]:
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    return [math.exp(decay_rate * (i / (n - 1))) for i in range(n)]


# API Optimization: Raised from 300 to 14400 seconds (4 Hours) to fit Community Cloud rate limits
@st.cache_data(ttl=14400, show_spinner="Calculating team ratings...")
def get_team_stats(api_key, league_id, season) -> pd.DataFrame:
    cols = ["Rank", "Team", "Games Played", "Goals For", "Goals Against",
            "GF/Game", "GA/Game", "Attack", "Defense",
            "PPG", "SoS", "Elo Mod", "Rating"]

    payload = api_request(f"competitions/{league_id}/matches", api_key, {"season": season})
    raw = payload.get("matches", []) if isinstance(payload, dict) else []
    if not raw:
        return pd.DataFrame(columns=cols)

    rankings = fetch_dynamic_fifa_rankings()
    elo_min, elo_max  = (min(rankings.values()), max(rankings.values())) if rankings else (None, None)
    global_avg_elo    = (sum(rankings.values()) / len(rankings)) if rankings else None

    all_teams: set[str] = set()
    matches_by_team: dict[str, list[dict]] = {}

    for item in raw:
        ht = item.get("homeTeam") or {}; at = item.get("awayTeam") or {}
        hn = ht.get("name");  an = at.get("name")
        if hn: all_teams.add(hn)
        if an: all_teams.add(an)

        if item.get("status") != "FINISHED":
            continue

        sc = item.get("score") or {}
        hg, ag = _regulation_score(sc)
        if None in (hn, an, hg, ag):
            continue

        date_key = item.get("utcDate") or ""
        matches_by_team.setdefault(hn, []).append({"date": date_key, "gf": hg, "ga": ag, "opp": an})
        matches_by_team.setdefault(an, []).append({"date": date_key, "gf": ag, "ga": hg, "opp": hn})

    if not all_teams:
        return pd.DataFrame(columns=cols)

    rows = []
    for team in all_teams:
        history = sorted(matches_by_team.get(team, []), key=lambda m: m["date"])
        gp = len(history)

        elo     = _lookup_elo_rating(team, rankings)
        elo_mod = _elo_to_rating_modifier(elo, elo_min, elo_max) if elo is not None else None

        if gp == 0:
            rating = elo_mod if elo_mod is not None else 0.5
            rows.append({
                "Team": team, "Games Played": 0,
                "Goals For": 0, "Goals Against": 0,
                "GF/Game": 0.0, "GA/Game": 0.0,
                "Attack": 0.0, "Defense": 0.0,
                "PPG": 0.0,
                "SoS": round(elo, 1) if elo is not None else None,
                "Elo Mod": round(elo_mod, 4) if elo_mod is not None else None,
                "Rating": round(rating, 4),
            })
            continue

        weights = _time_decay_weights(gp)
        total_w = sum(weights)

        raw_gf = sum(m["gf"] for m in history)
        raw_ga = sum(m["ga"] for m in history)

        weighted_points     = 0.0
        weighted_opp_elo    = 0.0
        opp_elo_weight_sum  = 0.0
        sos_weighted_gf     = 0.0
        sos_weighted_ga     = 0.0

        running_group_pts = 0.0

        for idx, (w, m) in enumerate(zip(weights, history)):
            match_pts = _match_points(m["gf"], m["ga"])

            gf_eff, ga_eff = m["gf"], m["ga"]
            if idx == 2 and running_group_pts == 6.0:
                text = m["gf"] * 1.0    
                ga_eff = m["ga"] * 0.20   

            weighted_points += w * match_pts

            # RECTIFIED BELIEVABLE LOGIC:
            opp_elo = _lookup_elo_rating(m["opp"], rankings)
            if opp_elo is None:
                opp_elo = global_avg_elo if global_avg_elo is not None else 1500.0

            # Accumulate the weighted opponent Elo so sos_elo below is never
            # a 0.0/0.0 division — this is what was missing before.
            weighted_opp_elo   += w * opp_elo
            opp_elo_weight_sum += w

            # 1. DAMPEN THE SoS EFFECT: Only refine the stats, do not overwrite them
            if global_avg_elo and global_avg_elo > 0:
                strength_mult = 1.0 + 0.35 * math.log(opp_elo / global_avg_elo)
            else:
                strength_mult = 1.0

            # Tighten the bounds so stats can never be distorted by more than 15%
            strength_mult = max(min(strength_mult, 1.15), 0.85)

            # 2. STRIP PENALTIES FOR CALCULATIONS: Ensure we only parse numeric regulation/ET goals
            try:
                gf_eff = float(m["gf"])
                ga_eff = float(m["ga"])
            except ValueError:
                # If a string like '2 (4)' somehow slipped in, extract just the base score
                gf_eff = float(str(m["gf"]).split()[0])
                ga_eff = float(str(m["ga"]).split()[0])

            sos_weighted_gf += w * (gf_eff * strength_mult)
            sos_weighted_ga += w * (ga_eff / strength_mult)
            total_w += w

        ppg_norm = weighted_points / (3.0 * total_w)
        sos_elo  = weighted_opp_elo / opp_elo_weight_sum if opp_elo_weight_sum else None
        adj_gfg  = sos_weighted_gf / total_w
        adj_gag  = sos_weighted_ga / total_w

        goal_perf = _goal_ratio_rating(adj_gfg, adj_gag)

        if elo_mod is not None:
            rating = (RATING_BLEND_WEIGHTS["xp"]       * ppg_norm +
                      RATING_BLEND_WEIGHTS["sos_goals"] * goal_perf +
                      RATING_BLEND_WEIGHTS["elo"]       * elo_mod)
        else:
            rating = (0.667 * ppg_norm) + (0.333 * goal_perf)

        rows.append({
            "Team": team, "Games Played": gp,
            "Goals For": raw_gf, "Goals Against": raw_ga,
            "GF/Game": round(adj_gfg, 2), "GA/Game": round(adj_gag, 2),
            "Attack": round(adj_gfg, 2), "Defense": round(adj_gag, 2),
            "PPG": round(ppg_norm * 3, 2),
            "SoS": round(sos_elo, 1) if sos_elo is not None else None,
            "Elo Mod": round(elo_mod, 4) if elo_mod is not None else None,
            "Rating": round(rating, 4),
        })

    if not rows:
        return pd.DataFrame(columns=cols)

    out = (pd.DataFrame(rows)
             .sort_values(["Rating", "GF/Game"], ascending=[False, False])
             .reset_index(drop=True))
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    return out


# API Optimization: Raised from 300 to 14400 seconds (4 Hours) to fit Community Cloud rate limits
@st.cache_data(ttl=14400, show_spinner="Fetching World Cup matches...")
def get_world_cup_matches(api_key, season, league_id=WORLD_CUP_LEAGUE_ID) -> pd.DataFrame:
    payload = api_request(f"competitions/{league_id}/matches", api_key, {"season": season})
    raw = payload.get("matches", []) if isinstance(payload, dict) else []
    rows = []
    for item in raw:
        ht = item.get("homeTeam") or {}; at = item.get("awayTeam") or {}
        sc = item.get("score")    or {}
        hg, ag = _regulation_score(sc)
        rows.append({
            "Match ID":   item.get("id"),
            "Date":       (item.get("utcDate") or "")[:16].replace("T", " "),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Stage":      item.get("stage", "") or "",
            "Home":       ht.get("name", ""),
            "Away":       at.get("name", ""),
            "Home Goals": hg,
            "Away Goals": ag,
            "Winner":     sc.get("winner"),
            "Status":     item.get("status", ""),
            "Raw Score":  sc,
        })
    return pd.DataFrame(rows)


# API Optimization: Raised from 300 to 14400 seconds (4 Hours) to fit Community Cloud rate limits
@st.cache_data(ttl=14400, show_spinner="Fetching World Cup groups...")
def get_world_cup_groups(api_key, season, league_id=WORLD_CUP_LEAGUE_ID) -> dict:
    raw = api_request(f"competitions/{league_id}/standings", api_key, {"season": season})
    if not raw or "standings" not in raw:
        return {}
    groups = {}
    try:
        for block in raw.get("standings", []):
            gname = block.get("group"); table = block.get("table", [])
            if not gname or not table:
                continue
            label = gname.replace("_", " ").title()
            names = [e.get("team", {}).get("name") for e in table if e.get("team", {}).get("name")]
            if names:
                groups[label] = names
    except Exception:
        return {}
    return groups


# ============================================================================
# 6. AUTO-SAF DATA FETCHERS (FBref · Transfermarkt · Club Elo)
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fbref_squad(team_name: str) -> list[dict]:
    sid  = FBREF_SQUAD_IDS.get(team_name)
    if not sid:
        return []
    slug = team_name.replace(" ", "-")
    soup = _scrape_get(f"https://fbref.com/en/squads/{sid}/{slug}-Stats")
    if not soup:
        return []
    table = soup.find("table", {"id": "stats_standard"}) or soup.find("table", {"id": re.compile(r"stats")})
    if not table or not table.find("tbody"):
        return []
    players = []
    for row in table.find("tbody").find_all("tr"):
        if "thead" in row.get("class", []):
            continue
        data = {c.get("data-stat"): c.get_text(strip=True) for c in row.find_all(["td","th"])}
        player = data.get("player","")
        club_cell = row.find("td", {"data-stat": "team"})
        club = ""
        if club_cell:
            a = club_cell.find("a")
            club = a.get_text(strip=True) if a else club_cell.get_text(strip=True)
        if player:
            players.append({"player": player, "club": club})
    return players


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fbref_form(team_name: str) -> float:
    sid  = FBREF_SQUAD_IDS.get(team_name)
    if not sid:
        return 0.70
    slug = team_name.replace(" ", "-")
    soup = _scrape_get(f"https://fbref.com/en/squads/{sid}/{slug}-Match-Logs-All-Competitions")
    if not soup:
        return 0.70
    table = soup.find("table", {"id": re.compile(r"matchlogs")})
    if not table or not table.find("tbody"):
        return 0.70

    rows = [r for r in table.find("tbody").find_all("tr")
            if r.find("td", {"data-stat": "result"}) and
            r.find("td", {"data-stat": "result"}).get_text(strip=True) in ("W","D","L")]
    last10 = rows[-10:] if len(rows) >= 10 else rows
    if not last10:
        return 0.70

    scores = []
    for row in last10:
        data = {c.get("data-stat"): c.get_text(strip=True) for c in row.find_all(["td","th"])}
        pts  = {"W": 1.0, "D": 0.5, "L": 0.0}.get(data.get("result",""), 0.5)
        xg   = _safe_float(data.get("xg",""))
        xga  = _safe_float(data.get("xga",""))
        if xg is not None and xga is not None and (xg + xga) > 0:
            score = pts * 0.6 + min(xg / (xg + xga), 1.0) * 0.4
        else:
            score = pts * 0.8 + 0.5 * 0.2
        scores.append(score)

    return round(float(np.clip(np.mean(scores), 0.40, 0.98)), 3)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fbref_tactical(team_name: str) -> float:
    sid  = FBREF_SQUAD_IDS.get(team_name)
    if not sid:
        return 0.80
    slug = team_name.replace(" ", "-")
    soup = _scrape_get(f"https://fbref.com/en/squads/{sid}/{slug}-Match-Logs-All-Competitions")
    if not soup:
        return 0.80
    table = soup.find("table", {"id": re.compile(r"matchlogs")})
    if not table or not table.find("tbody"):
        return 0.80

    formations = [
        r.find("td", {"data-stat": "formation"}).get_text(strip=True)
        for r in table.find("tbody").find_all("tr")
        if r.find("td", {"data-stat": "formation"})
    ]
    formations = [f for f in formations if f]
    if not formations:
        return 0.80

    last20 = formations[-20:]
    top_share = Counter(last20).most_common(1)[0][1] / len(last20)
    return round(float(np.clip(0.60 + top_share * 0.35, 0.50, 0.98)), 3)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_transfermarkt_fitness(team_name: str) -> float:
    tm_id = TM_TEAM_IDS.get(team_name)
    if not tm_id:
        return 0.85
    slug = team_name.lower().replace(" ", "-")
    soup = _scrape_get(f"https://www.transfermarkt.com/{slug}/startseite/verein/{tm_id}")
    if not soup:
        return 0.85

    n_injured = len(soup.find_all("span", {"class": re.compile(r"verletzt|suspended|injury", re.I)}))
    n_total   = max(len(soup.find_all("tr", {"class": re.compile(r"odd|even")})), 23)
    fit_frac  = 1.0 - min(n_injured / n_total, 0.5)
    return round(float(np.clip(0.65 + fit_frac * 0.30, 0.55, 0.97)), 3)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_transfermarkt_caps(team_name: str) -> float:
    tm_id = TM_TEAM_IDS.get(team_name)
    if not tm_id:
        return 0.75
    slug = team_name.lower().replace(" ", "-")
    soup = _scrape_get(f"https://www.transfermarkt.com/{slug}/startseite/verein/{tm_id}")
    if not soup:
        return 0.75

    caps = [v for cell in soup.find_all("td", {"class": re.compile(r"caps|einsaetze", re.I)})
            if (v := _safe_float(cell.get_text(strip=True).replace(",",""))) is not None and v >= 0]
    if not caps:
        caps = [v for cell in soup.find_all("td")
                if (v := _safe_float(cell.get_text(strip=True).replace(",",""))) is not None and 5 < v < 250]
    if not caps:
        return 0.75

    box_data = float(np.median(caps))
    return round(float(np.clip(0.55 + 0.40 * min(np.log1p(box_data) / np.log1p(120), 1.0), 0.55, 0.98)), 3)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_club_elo_league_strength(team_name: str, players: list[dict]) -> float:
    if not players:
        return 0.80
    clubs = list({p["club"] for p in players if p.get("club")})
    elo_scores = []
    for club in clubs[:20]:
        elo_club = CLUB_ELO_NAMES.get(club, club).replace(" ", "")
        try:
            time.sleep(0.3)
            r = requests.get(f"http://api.clubelo.com/{elo_club}", headers=SCRAPE_HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            lines = r.text.strip().split("\n")
            if len(lines) < 2:
                continue
            parts = lines[-1].split(",")
            if len(parts) >= 4:
                elo = _safe_float(parts[3])
                if elo and 1000 < elo < 2200:
                    elo_scores.append(elo)
        except Exception:
            continue

    if not elo_scores:
        return 0.80
    avg = float(np.mean(elo_scores))
    return round(float(np.clip(0.50 + 0.50 * min((avg - 1400) / 600, 1.0), 0.55, 0.98)), 3)


def compute_composite_saf(scores: dict) -> float:
    return round(float(np.clip(sum(SAF_WEIGHTS[k] * scores.get(k, 0.80) for k in SAF_WEIGHTS), 0.40, 0.99)), 4)


def fetch_team_saf(team_name: str) -> dict:
    players = fetch_fbref_squad(team_name)
    scores  = {
        "tactical":   fetch_fbref_tactical(team_name),
        "form":       fetch_fbref_form(team_name),
        "fitness":    fetch_transfermarkt_fitness(team_name),
        "league":     fetch_club_elo_league_strength(team_name, players),
        "experience": fetch_transfermarkt_caps(team_name),
    }
    scores["composite"]   = compute_composite_saf(scores)
    scores["_players"]    = players
    scores["_fetched_at"] = datetime.now().strftime("%H:%M:%S")
    return scores


def render_saf_breakdown_card(team_name: str, saf_data: dict):
    # NOTE: `composite` is computed upstream from ALL FIVE underlying factors
    # (tactical, form, fitness, league, experience) — that math is completely
    # untouched. Only the *visible* rows are restricted here; Tactical Fit,
    # Club Form, and League Strength still feed the composite score, they're
    # just no longer rendered as individual rows to the end-user.
    composite = saf_data.get("composite", 0.80)

    source_map = {
        "fitness":    ("", "Fitness / Injuries", "Transfermarkt - injury availability"),
        "experience": ("", "Intl. Experience", "Transfermarkt - avg squad caps"),
    }

    st.markdown(f"**{team_name}**", unsafe_allow_html=True)

    # Composite SAF progress bar (replaces the old static badge with a
    # visual bar consistent with the per-factor rows below it)
    comp_pct   = int(composite * 100)
    comp_color = "#2ecc71" if composite >= 0.85 else ("#f39c12" if composite >= 0.70 else "#e74c3c")
    st.markdown(
        f'<div style="margin:6px 0 12px 0;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-size:.85em;width:160px;color:#ccc;font-weight:700;">Composite SAF</span>'
        f'<div style="flex:1;background:#2a2a2a;border-radius:4px;height:12px;">'
        f'<div style="width:{comp_pct}%;background:{comp_color};border-radius:4px;height:12px;"></div></div>'
        f'<span style="font-size:.85em;width:40px;text-align:right;font-weight:700;">{composite:.2f}</span></div></div>',
        unsafe_allow_html=True,
    )

    for key, (icon, label, source) in source_map.items():
        val   = saf_data.get(key, 0.80)
        pct   = int(val * 100)
        color = "#2ecc71" if val >= 0.85 else ("#f39c12" if val >= 0.70 else "#e74c3c")
        st.markdown(
            f'<div style="margin:4px 0;">'
            f'<div style="display:flex;align-items:center;gap:8px;">'
            f'<span style="font-size:.85em;width:160px;color:#ccc;">{label}</span>'
            f'<div style="flex:1;background:#2a2a2a;border-radius:4px;height:10px;">'
            f'<div style="width:{pct}%;background:{color};border-radius:4px;height:10px;"></div></div>'
            f'<span style="font-size:.85em;width:40px;text-align:right;">{val:.2f}</span></div>'
            f'<div style="font-size:.72em;color:#888;padding-left:168px;">{source}</div></div>',
            unsafe_allow_html=True,
        )
    if ts := saf_data.get("_fetched_at"):
        st.caption(f"Fetched at {ts} - cached 1 h")


# ============================================================================
# 7. MATCH PREDICTION ENGINE
# ============================================================================

def calculate_expected_goals(team_a_stats, team_b_stats,
                              league_avg_gf: float, league_avg_ga: float,
                              home_advantage: float = 1.10,
                              saf_a: float = 0.85, saf_b: float = 0.85,
                              team_a_name: str = None, team_b_name: str = None):
    ta_name = team_a_name or (team_a_stats.get("Team") if isinstance(team_a_stats, (dict, pd.Series)) else None)
    tb_name = team_b_name or (team_b_stats.get("Team") if isinstance(team_b_stats, (dict, pd.Series)) else None)
    
    hosts = {"USA", "Mexico", "Canada", "United States"}
    
    calibrated_advantage = 1.00
    if ta_name in hosts and tb_name not in hosts:
        calibrated_advantage = 1.10
    elif tb_name in hosts and ta_name not in hosts:
        calibrated_advantage = 0.90
        
    lgf = league_avg_gf or 1.3;  lga = league_avg_ga or 1.3
    ma  = saf_a / 0.85;          mb  = saf_b / 0.85
    
    a_atk = (team_a_stats["GF/Game"] * ma) / lgf
    b_atk = (team_b_stats["GF/Game"] * mb) / lgf
    
    # Defensive Floor Check: Prevent GA/Game from suppressing attack too heavily
    a_def = (max(team_a_stats["GA/Game"], 0.6) / ma) / lga
    b_def = (max(team_b_stats["GA/Game"], 0.6) / mb) / lga
    
    xg_h = float(np.clip(a_atk * b_def * lgf * calibrated_advantage, 0.1, 5.0))
    xg_a = float(np.clip(b_atk * a_def * lga * (2 - calibrated_advantage), 0.1, 5.0))
    
    return xg_h, xg_a


def build_score_matrix(xg_home: float, xg_away: float, max_goals: int = 5) -> np.ndarray:
    ph = [poisson.pmf(g, xg_home) for g in range(max_goals + 1)]
    pa = [poisson.pmf(g, xg_away) for g in range(max_goals + 1)]
    return np.outer(ph, pa)


def summarize_outcomes(matrix: np.ndarray):
    hw = np.sum(np.tril(matrix, -1))
    dr = np.sum(np.diag(matrix))
    aw = np.sum(np.triu(matrix, 1))
    t  = hw + dr + aw
    if t == 0:
        return 0.0, 0.0, 0.0
    return hw/t*100, dr/t*100, aw/t*100


def top_scorelines(matrix: np.ndarray, n: int = 10):
    flat = [(f"{hg}-{ag}", matrix[hg, ag])
            for hg in range(matrix.shape[0])
            for ag in range(matrix.shape[1])]
    flat.sort(key=lambda p: p[1], reverse=True)
    return flat[:n]


def run_monte_carlo_simulation(xg_home: float, xg_away: float, n_sims: int = 10000) -> dict:
    gh = np.random.poisson(xg_home, n_sims)
    ga = np.random.poisson(xg_away, n_sims)
    return {
        "home_win_pct":   int(np.sum(gh > ga)) / n_sims * 100,
        "draw_pct":       int(np.sum(gh == ga)) / n_sims * 100,
        "away_win_pct":   int(np.sum(gh < ga)) / n_sims * 100,
        "avg_home_goals": float(np.mean(gh)),
        "avg_away_goals": float(np.mean(ga)),
    }


# ============================================================================
# 8. WORLD CUP TOURNAMENT SIMULATION
# ============================================================================

def simulate_match_winner(team_a, team_b, ratings_lookup, lgf, lga,
                           saf_lookup=None, model_bundle=None) -> str:
    sa = ratings_lookup.get(team_a, DEFAULT_TEAM_STATS)
    sb = ratings_lookup.get(team_b, DEFAULT_TEAM_STATS)
    saf_a = saf_lookup.get(team_a, 0.85) if saf_lookup else 0.85
    saf_b = saf_lookup.get(team_b, 0.85) if saf_lookup else 0.85

    xg_a, xg_b = calculate_expected_goals(sa, sb, lgf, lga, saf_a=saf_a, saf_b=saf_b, team_a_name=team_a, team_b_name=team_b)

    rd = sa.get("Rating", 0.5) - sb.get("Rating", 0.5)
    hw, dr, aw = blended_probs(xg_a, xg_b, rating_diff=rd, saf_diff=saf_a - saf_b, model_bundle=model_bundle)

    r = np.random.random()
    if r < hw:
        return team_a
    if r < hw + dr:
        # Standardize tie-breakers strictly to baseline Rating
        ra_ = sa.get("Rating", 0.5)
        rb_ = sb.get("Rating", 0.5)
        tot = ra_ + rb_
        return team_a if np.random.random() < (ra_ / tot if tot > 0 else 0.5) else team_b
    return team_b


def simulate_knockout_bracket(teams, ratings_lookup, lgf, lga, saf_lookup=None, model_bundle=None):
    current_round = list(teams)
    finalists     = None
    semifinalists = None

    while len(current_round) > 1:
        next_round = []
        i = 0
        while i < len(current_round):
            if i + 1 < len(current_round):
                w = simulate_match_winner(
                    current_round[i], current_round[i + 1],
                    ratings_lookup, lgf, lga, saf_lookup, model_bundle
                )
                next_round.append(w)
                i += 2
            else:
                next_round.append(current_round[i])
                i += 1

        if len(next_round) == 2 and semifinalists is None:
            semifinalists = list(current_round)
        if len(next_round) == 1:
            finalists = list(current_round)

        current_round = next_round

    champion = current_round[0] if current_round else None
    return champion, finalists or [], semifinalists or []


def simulate_knockout_bracket_locked(teams, ratings_lookup, lgf, lga,
                                     saf_lookup=None, model_bundle=None,
                                     fixed_results=None):
    """Simulate the bracket while honoring already completed fixtures."""
    fixed_results = fixed_results or {}
    current_round = list(teams)
    finalists = None
    semifinalists = None

    while len(current_round) > 1:
        next_round = []
        i = 0
        while i < len(current_round):
            if i + 1 < len(current_round):
                a, b = current_round[i], current_round[i + 1]
                known = fixed_results.get(_pair_key(a, b))
                if known:
                    w = known["winner"]
                else:
                    w = simulate_match_winner(a, b, ratings_lookup, lgf, lga, saf_lookup, model_bundle)
                next_round.append(w)
                i += 2
            else:
                next_round.append(current_round[i])
                i += 1

        if len(next_round) == 2 and semifinalists is None:
            semifinalists = list(current_round)
        if len(next_round) == 1:
            finalists = list(current_round)
        current_round = next_round

    champion = current_round[0] if current_round else None
    return champion, finalists or [], semifinalists or []


def run_world_cup_simulation(groups, ratings_lookup, league_avg_gf, league_avg_ga,
                              n_sims=10000, saf_lookup=None, model_bundle=None,
                              fixed_results=None) -> pd.DataFrame:
    """
    Monte Carlo WC 2026 simulation starting from the confirmed Round of 32 bracket.
    """
    bracket = [
        # --- Left Side of Bracket ---
        "Germany", "Paraguay",               
        "France", "Sweden",                  
        "South Africa", "Canada",            
        "Netherlands", "Morocco",            
        "Portugal", "Croatia",               
        "Spain", "Austria",                  
        "USA", "Bosnia and Herzegovina",     
        "Belgium", "Senegal",                
        
        # --- Right Side of Bracket ---
        "Brazil", "Japan",                   
        "Ivory Coast", "Norway",             
        "Mexico", "Ecuador",                 
        "England", "DR Congo",               
        "Argentina", "Cabo Verde",           
        "Australia", "Egypt",                
        "Switzerland", "Algeria",            
        "Colombia", "Ghana"                  
    ]
    
    champion_count = {t: 0 for t in bracket}
    final_count    = {t: 0 for t in bracket}
    semi_count     = {t: 0 for t in bracket}

    progress = st.progress(0, text="Simulating knockout stage...")
    every = max(1, n_sims // 100)

    underdogs = {"Paraguay", "Canada", "Morocco", "Austria", "Bosnia and Herzegovina", 
                 "Senegal", "Japan", "Norway", "Ecuador", "DR Congo", "Cabo Verde", "Algeria", "Ghana"}

    for sim in range(n_sims):
        champion, finalists, semis = simulate_knockout_bracket_locked(
            bracket, ratings_lookup, league_avg_gf, league_avg_ga,
            saf_lookup, model_bundle, fixed_results=fixed_results
        )

        if champion:
            norm_champ = "USA" if champion == "United States" else champion
            if norm_champ in champion_count:
                champion_count[norm_champ] += 1
                
        for t in finalists:
            norm_f = "USA" if t == "United States" else t
            if norm_f in final_count:
                final_count[norm_f] += 1
                
        for t in semis:
            norm_s = "USA" if t == "United States" else t
            if norm_s in semi_count:
                semi_count[norm_s] += 1

        if sim % every == 0:
            # DYNAMIC ATTENTION HOOK: Generate a fast, chaotic live tournament headline on the fly
            pct_complete = min(sim / n_sims, 1.0)
            
            if pct_complete < 0.25:
                headline = f"🔮 Run #{sim:,}: {champion} wins the World Cup!"
            elif pct_complete < 0.50:
                upset_team = next((t for t in finalists if t in underdogs), None)
                if upset_team:
                    headline = f"😱 Run #{sim:,}: SHOCKER! {upset_team} breaks into the World Cup Final!"
                else:
                    headline = f"🔥 Run #{sim:,}: Heavyweight clash: {finalists[0]} vs {finalists[1]}!"
            elif pct_complete < 0.75:
                headline = f"🏆 Run #{sim:,}: Simulation tracking champion trends: {champion}!"
            else:
                headline = f"📊 Run #{sim:,}: Recalibrating compounded bracket bottleneck variance..."

            progress.progress(pct_complete, text=f"⚡ {headline} ({sim:,}/{n_sims:,})")

    progress.progress(1.0, text="Done!")
    progress.empty()

    rows = [{
        "Team":        t,
        "Champion %":  round(champion_count[t] / n_sims * 100, 2),
        "Final %":     round(final_count[t]    / n_sims * 100, 2),
        "Semifinal %": round(semi_count[t]      / n_sims * 100, 2),
    } for t in bracket]
    
    return pd.DataFrame(rows).sort_values("Champion %", ascending=False).reset_index(drop=True)


# ============================================================================
# 8b. BRACKET VISUALIZATION ELEMENT
# ============================================================================

def render_visual_bracket():
    """
    Renders the official World Cup 2026 Round of 32 tournament tree layout.
    """
    left_side = [
        ("Germany", "Paraguay"), ("France", "Sweden"),
        ("South Africa", "Canada"), ("Netherlands", "Morocco"),
        ("Portugal", "Croatia"), ("Spain", "Austria"),
        ("USA", "Bosnia & Herz."), ("Belgium", "Senegal")
    ]
    
    right_side = [
        ("Brazil", "Japan"), ("Ivory Coast", "Norway"),
        ("Mexico", "Ecuador"), ("England", "DR Congo"),
        ("Argentina", "Cabo Verde"), ("Australia", "Egypt"),
        ("Switzerland", "Algeria"), ("Colombia", "Ghana")
    ]

    css = """
    <style>
    .bracket-wrapper {
        background-color: #0e1117;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #262730;
        margin: 15px 0;
    }
    .bracket-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        font-family: inherit;
        font-size: 13px;
        min-width: 600px;
    }
    .bracket-half {
        display: flex;
        flex-direction: column;
        gap: 12px;
        width: 40%;
    }
    .matchup-card {
        position: relative;
        display: flex;
        flex-direction: column;
        background-color: #1a1c23;
        border-left: 4px solid #3498db;
        border-radius: 4px;
        padding: 6px 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .bracket-team-row {
        padding: 4px 0;
        color: #e0e0e0;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bracket-team-row:first-child {
        border-bottom: 1px solid #2d313f;
    }
    .center-podium {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 20%;
        color: #f1c40f;
        font-weight: bold;
        font-size: 15px;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .trophy-glow {
        font-size: 48px;
        margin-bottom: 8px;
        filter: drop-shadow(0 0 10px rgba(241,196,15,0.4));
    }
    /* Mobile: swap truncation for wrapping so team names are never clipped.
       Sidebar is untouched — these rules are scoped to bracket classes only. */
    @media (max-width: 768px) {
        .bracket-container {
            min-width: 0;
        }
        .bracket-team-row {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
    }
    </style>
    """

    def generate_half_html(matchups):
        html = '<div class="bracket-half">'
        for team1, team2 in matchups:
            html += f"""
            <div class="matchup-card">
                <div class="bracket-team-row">🏳️ {team1}</div>
                <div class="bracket-team-row">🏳️ {team2}</div>
            </div>
            """
        html += '</div>'
        return html

    html_layout = f"""
    {css}
    <div class="bracket-wrapper">
        <div class="bracket-container">
            {generate_half_html(left_side)}
            <div class="center-podium">
                <div class="trophy-glow">🏆</div>
                <div>2026 World Cup<br><span style="color:#ffffff;">Finalist Tracks</span></div>
            </div>
            {generate_half_html(right_side)}
        </div>
    </div>
    """
    st.markdown(html_layout, unsafe_allow_html=True)


# ============================================================================
# 8c. HORIZONTAL BRACKET LAYOUT ENGINE
# ============================================================================

BRACKET_PLACEHOLDER = "--------"
LEFT_R32_MATCH_REFS = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_MATCH_REFS = [76, 78, 79, 80, 86, 88, 85, 87]

DEFAULT_R32_TEAMS: dict[int, tuple[str, str]] = {
    74: ("Germany", "Paraguay"),
    77: ("France", "Sweden"),
    73: ("South Africa", "Canada"),
    75: ("Netherlands", "Morocco"),
    83: ("Portugal", "Croatia"),
    84: ("Spain", "Austria"),
    81: ("USA", "Bosnia and Herzegovina"),
    82: ("Belgium", "Senegal"),
    76: ("Brazil", "Japan"),
    78: ("Ivory Coast", "Norway"),
    79: ("Mexico", "Ecuador"),
    80: ("England", "DR Congo"),
    86: ("Argentina", "Cabo Verde"),
    88: ("Australia", "Egypt"),
    85: ("Switzerland", "Algeria"),
    87: ("Colombia", "Ghana"),
}

COUNTRY_FLAGS: dict[str, str] = {
    "Algeria": "🇩🇿", "Argentina": "🇦🇷", "Australia": "🇦🇺", "Austria": "🇦🇹",
    "Belgium": "🇧🇪", "Bosnia and Herzegovina": "🇧🇦", "Bosnia & Herz.": "🇧🇦",
    "Brazil": "🇧🇷", "Cabo Verde": "🇨🇻", "Canada": "🇨🇦", "Colombia": "🇨🇨",
    "Croatia": "🇭🇷", "DR Congo": "🇨🇩", "Ecuador": "🇪🇨", "Egypt": "🇪🇬",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "France": "🇫🇷", "Germany": "🇩🇪", "Ghana": "🇬🇭",
    "Ivory Coast": "🇨🇮", "Japan": "🇯🇵", "Mexico": "🇲🇽", "Morocco": "🇲🇦",
    "Netherlands": "🇳🇱", "Norway": "🇳🇴", "Paraguay": "🇵🇾", "Portugal": "🇵🇹",
    "Senegal": "🇸🇳", "South Africa": "🇿🇦", "Spain": "🇪🇸", "Sweden": "🇸🇪",
    "Switzerland": "🇨🇭", "USA": "🇺🇸",
}


def _blank_entry(label: str = BRACKET_PLACEHOLDER, stat: str = "") -> dict:
    return {"team": label, "stat": stat, "winner": False, "placeholder": label == BRACKET_PLACEHOLDER}


def _entry(team: str, stat: str = "", winner: bool = False) -> dict:
    if not _is_confirmed_team(team):
        return _blank_entry(stat=stat)
    return {"team": str(team), "stat": str(stat or ""), "winner": bool(winner), "placeholder": False}


def _match_card(title: str, team_a=None, team_b=None, ref=None, status: str = "") -> dict:
    return {
        "title": title,
        "ref": ref,
        "status": status or "",
        "teams": [team_a or _blank_entry(), team_b or _blank_entry()],
    }


def _bracket_css() -> str:
    return """
    <style>
    .wc-bracket-shell {
        width: 100%;
        overflow-x: auto;
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 8px;
        padding: 18px;
        margin: 16px 0 24px 0;
    }
    .wc-bracket-grid {
        min-width: 1460px; /* 📊 Relaxation adjustments to secure full text expansion */
        display: grid;
        grid-template-columns: 1.25fr 1.15fr 1.05fr .95fr 1.15fr .95fr 1.05fr 1.15fr 1.25fr;
        column-gap: 16px;
        align-items: stretch;
        min-height: 760px;
        font-family: inherit;
    }
    .bracket-round {
        display: grid;
        grid-template-rows: repeat(16, minmax(34px, 1fr));
        row-gap: 6px;
        position: relative;
    }
    .bracket-round-title {
        position: absolute;
        top: -18px;
        left: 0;
        width: 100%;
        color: #f1c40f;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        text-align: center;
        margin-bottom: 2px;
    }
    .matchup-card {
        position: relative;
        width: 100%;
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 8px;
        box-shadow: 0 10px 26px rgba(0,0,0,.22);
        padding: 8px 10px;
        min-height: 0;
        box-sizing: border-box;
    }
    .matchup-card::after {
        content: "";
        position: absolute;
        top: 50%;
        width: 13px;
        border-top: 1px solid #3d4658;
    }
    .bracket-left .matchup-card::after { right: -14px; }
    .bracket-right .matchup-card::after { left: -14px; }
    .bracket-left .matchup-card::before,
    .bracket-right .matchup-card::before {
        content: "";
        position: absolute;
        top: 18px;
        bottom: 18px;
        width: 8px;
        border-top: 1px solid #3d4658;
        border-bottom: 1px solid #3d4658;
    }
    .bracket-left .matchup-card::before { right: -8px; border-right: 1px solid #3d4658; }
    .bracket-right .matchup-card::before { left: -8px; border-left: 1px solid #3d4658; }
    .matchup-meta {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        color: #9ba3af;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .04em;
        text-transform: uppercase;
        margin-bottom: 6px;
        min-height: 14px;
    }
    .bracket-team-row {
        display: grid;
        grid-template-columns: 14px minmax(0, 1fr) 52px; /* Adjusted columns */
        align-items: center;
        gap: 6px;
        min-height: 27px;
        padding: 4px 0;
        color: #e7e7e7;
        border-top: 1px solid #2b2b2b;
    }
    .bracket-team-row:first-of-type { border-top: 0; }
    .team-flag { text-align: center; font-size: 14px; }
    .team-name {
        min-width: 0;
        white-space: nowrap; /* 💻 Disabled character clipping text ellipsis entirely */
        font-size: 11.5px;
        font-weight: 600;
    }
    .team-score {
        justify-self: end;
        min-width: 48px;
        text-align: right;
        color: #f1c40f;
        font-size: 11px;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }
    .winner .team-name { color: #ffffff; font-weight: 900; }
    .winner .team-score { color: #2ecc71; }
    .placeholder .team-name {
        color: #777;
        font-family: monospace;
        letter-spacing: .02em;
    }
    .center-core {
        display: grid;
        grid-template-rows: repeat(16, minmax(34px, 1fr));
        row-gap: 6px;
        position: relative;
    }
    .center-slot { align-self: stretch; }
    .center-slot .matchup-card { height: 100%; box-sizing: border-box; }
    .champion-podium {
        background: linear-gradient(180deg, rgba(241,196,15,.16), rgba(46,204,113,.08));
        border: 1px solid rgba(241,196,15,.45);
        border-radius: 8px;
        padding: 14px 12px;
        text-align: center;
        box-shadow: 0 0 22px rgba(241,196,15,.12);
    }
    .podium-label {
        color: #f1c40f;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .podium-team {
        color: #ffffff;
        font-size: 18px;
        line-height: 1.2;
        font-weight: 900;
        overflow-wrap: anywhere;
    }
    .center-core .matchup-card::before,
    .center-core .matchup-card::after { display: none; }

    /* ─────────────────────────────────────────────────────────────────
       📱 RESPONSIVE MOBILE-FIRST OVERRIDE
       Below 900px the 9-column horizontal grid is fully unrolled into a
       single scrollable vertical column-flex stack. DOM order already
       reads Left R32 → R16 → QF → SF → Center (Final/Champion/Third) →
       Right SF → QF → R16 → R32, so flipping display:grid → flex column
       preserves a logical top-to-bottom bracket narrative on phones.
       ───────────────────────────────────────────────────────────────── */
    @media (max-width: 900px) {
        .wc-bracket-shell {
            padding: 12px 10px;
            overflow-x: hidden;          /* no more horizontal clipping/scrolling */
        }
        .wc-bracket-grid {
            display: flex;
            flex-direction: column;
            min-width: 0;
            width: 100%;
            min-height: 0;
            gap: 22px;
        }
        .bracket-round,
        .center-core {
            display: flex;
            flex-direction: column;
            gap: 10px;
            width: 100%;
            grid-template-rows: none;
        }
        .bracket-round-title {
            position: static;
            top: auto;
            font-size: 12px;
            margin: 4px 0 2px 0;
            text-align: left;
            padding-left: 2px;
        }
        /* Connector lines are drawn assuming a horizontal left/right grid;
           they render as stray artifacts once stacked vertically, so hide them. */
        .matchup-card::after,
        .matchup-card::before,
        .bracket-left .matchup-card::after,
        .bracket-right .matchup-card::after,
        .bracket-left .matchup-card::before,
        .bracket-right .matchup-card::before {
            display: none;
        }
        .matchup-card {
            padding: 10px 12px;
            min-height: 0;
        }
        .matchup-meta { font-size: 10.5px; }
        .bracket-team-row {
            grid-template-columns: 16px minmax(0, 1fr) 56px;
            min-height: 34px;
            padding: 6px 0;
        }
        .team-name {
            white-space: normal;         /* allow wrapping instead of clipping */
            overflow-wrap: anywhere;
            word-break: break-word;
            font-size: 13px;
            line-height: 1.25;
        }
        .bracket-team-row .winner .team-name { color: #ffffff; font-weight: 900; }
        .team-score {
            font-size: 12.5px;
            min-width: 44px;
        }
        .center-slot {
            grid-row: auto !important;
        }
        .center-slot .matchup-card { height: auto; }
        .champion-podium {
            padding: 16px 12px;
            margin: 4px 0;
        }
        .podium-team { font-size: 20px; }
    }

    @media (max-width: 480px) {
        .wc-bracket-shell { padding: 10px 8px; }
        .wc-bracket-grid { gap: 18px; }
        .matchup-card { padding: 9px 10px; }
        .team-name { font-size: 12.5px; }
        .team-score { font-size: 12px; min-width: 38px; }
        .matchup-meta { font-size: 10px; }
        .podium-team { font-size: 18px; }
    }
    </style>
    """


def _team_row_html(entry_data: dict) -> str:
    team = str(entry_data.get("team") or BRACKET_PLACEHOLDER)
    flag = ""
    stat = str(entry_data.get("stat") or "")
    classes = ["bracket-team-row"]
    if entry_data.get("winner"):
        classes.append("winner")
    if entry_data.get("placeholder"):
        classes.append("placeholder")
        flag = ""
    return (
        f'<div class="{" ".join(classes)}">'
        f'<span class="team-flag">{html.escape(flag)}</span>'
        f'<span class="team-name">{html.escape(team)}</span>'
        f'<span class="team-score">{html.escape(stat)}</span>'
        f'</div>'
    )


def _match_card_html(card: dict) -> str:
    ref = card.get("ref")
    label = f"M{ref}" if ref not in (None, "") else str(card.get("status") or "")
    teams = card.get("teams", [_blank_entry(), _blank_entry()])
    return "".join([
        '<div class="matchup-card">',
        '<div class="matchup-meta">',
        f'<span>{html.escape(str(card.get("title", "")))}</span>',
        f'<span>{html.escape(label)}</span>',
        '</div>',
        _team_row_html(teams[0]),
        _team_row_html(teams[1]),
        '</div>',
    ])


def _round_column_html(title: str, cards: list[dict], side: str = "left") -> str:
    def grid_span(total: int, idx: int) -> str:
        total = max(total, 1)
        block = max(16 // total, 1)
        span = min(block, 4)           
        start = idx * block + (block - span) // 2 + 1
        return f' style="grid-row:{start} / span {span}; display:flex; align-items:center;"'

    cards_html = "".join(
        f'<div{grid_span(len(cards), idx)}>{_match_card_html(card)}</div>'
        for idx, card in enumerate(cards)
    )
    return (
        f'<section class="bracket-round bracket-{html.escape(side)}">'
        f'<div class="bracket-round-title">{html.escape(title)}</div>'
        f'{cards_html}'
        '</section>'
    )


def _center_core_html(final_card: dict, third_card: dict, champion: dict) -> str:
    champion_name = champion.get("team") or BRACKET_PLACEHOLDER
    champion_stat = champion.get("stat") or ""
    return "".join([
        '<section class="center-core">',
        f'<div class="center-slot" style="grid-row:5 / span 3;">{_match_card_html(final_card)}</div>',
        '<div class="champion-podium" style="grid-row:8 / span 2;">',
        '<div class="podium-label">CHAMPION</div>',
        f'<div class="podium-team">{html.escape(champion_name)}</div>',
        f'<div class="team-score" style="text-align:center;min-width:0;margin-top:6px;">{html.escape(champion_stat)}</div>',
        '</div>',
        f'<div class="center-slot" style="grid-row:10 / span 3;">{_match_card_html(third_card)}</div>',
        '</section>',
    ])


def render_horizontal_bracket(bracket_state: dict):
    html_blocks = [
        _bracket_css(),
        '<div class="wc-bracket-shell"><div class="wc-bracket-grid">',
        _round_column_html("Left Round of 32", bracket_state["left_r32"], "left"),
        _round_column_html("Left Round of 16", bracket_state["left_r16"], "left"),
        _round_column_html("Left Quarterfinals", bracket_state["left_qf"], "left"),
        _round_column_html("Left Semifinals", bracket_state["left_sf"], "left"),
        _center_core_html(bracket_state["final"], bracket_state["third"], bracket_state["champion"]),
        _round_column_html("Right Semifinals", bracket_state["right_sf"], "right"),
        _round_column_html("Right Quarterfinals", bracket_state["right_qf"], "right"),
        _round_column_html("Right Round of 16", bracket_state["right_r16"], "right"),
        _round_column_html("Right Round of 32", bracket_state["right_r32"], "right"),
        '</div></div>',
    ]
    st.markdown("".join(html_blocks), unsafe_allow_html=True)


def _fixture_match_ref(row: pd.Series):
    for col in ("Match Ref", "Round", "Match", "Matchday", "matchday", "ID", "Match ID"):
        if col in row.index and pd.notna(row.get(col)):
            try:
                return int(float(str(row.get(col)).strip()))
            except Exception:
                continue
    return None


def _fixture_score(row: pd.Series) -> tuple[str, str]:
    # 1. Prioritize the raw API dictionary if we saved it during the fetch
    raw_score = row.get("Raw Score")
    if isinstance(raw_score, dict) and raw_score:
        ft = raw_score.get("fullTime") or {}
        pen = raw_score.get("penalties") or {}
        
        hf = ft.get("home")
        af = ft.get("away")
        
        if hf is not None and af is not None:
            # If penalties exist, append them (e.g., "1 (3) - 1 (2)")
            if pen and pen.get("home") is not None and pen.get("away") is not None:
                return f"{hf} ({pen.get('home')})", f"{af} ({pen.get('away')})"
            
            # Otherwise return true full time (which includes Extra Time goals)
            return str(hf), str(af)
            
    # 2. Fallback for hardcoded results (like _FALLBACK_KNOCKOUT_RESULTS_2026)
    hg = row.get("Home Goals")
    ag = row.get("Away Goals")
    if pd.notna(hg) and pd.notna(ag):
        try:
            return str(int(hg)), str(int(ag))
        except Exception:
            return str(hg), str(ag)
            
    # 3. Final string parsing fallback
    score_str = row.get("Score")
    if isinstance(score_str, str) and "-" in score_str:
        parts = re.split(r"\s*[-:]\s*", score_str.strip())
        if len(parts) >= 2:
            return parts[0], parts[1]
            
    return "", ""

def _fixture_winner(row: pd.Series) -> str | None:
    if str(row.get("Status", "")).upper() != "FINISHED":
        return None
        
    # 1. Trust the API's direct determination of who won the match/shootout
    api_winner = str(row.get("Winner", "") or "").upper()
    if api_winner == "HOME_TEAM":
        return row.get("Home")
    if api_winner == "AWAY_TEAM":
        return row.get("Away")
        
    # 2. Fallback to regulation goals if Winner token is missing
    try:
        hg = float(row.get("Home Goals"))
        ag = float(row.get("Away Goals"))
        if hg > ag: return row.get("Home")
        if ag > hg: return row.get("Away")
    except Exception:
        return None
    return None


def _team_key(name) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    # OVERRIDES FOR USA AND BOSNIA FIXTURE MATCHING
    if cleaned in ("usa", "unitedstates", "unitedstatesofamerica"):
        return "usa"
    if cleaned in ("bosniaandherzegovina", "bosniaherz", "bosniaherzegovina", "bosnia"):
        return "bosniaandherzegovina"
    if cleaned in ("congodr", "drcongo", "democraticrepublicofcongo"):
        return "drcongo"
    if cleaned in ("caboverde", "capeverde","capeverdeislands"):
        return "caboverde"
    return cleaned


def _pair_key(team_a, team_b) -> tuple[str, str]:
    return tuple(sorted([_team_key(team_a), _team_key(team_b)]))


def _default_ref_from_pair(home, away) -> int | None:
    pair = _pair_key(home, away)
    for ref, teams in DEFAULT_R32_TEAMS.items():
        if _pair_key(*teams) == pair:
            return ref
    return None


def completed_results_lookup(fixtures_df: pd.DataFrame) -> dict[tuple[str, str], dict]:
    completed: dict[tuple[str, str], dict] = {}
    if fixtures_df is None or fixtures_df.empty:
        return completed
    for _, row in fixtures_df.iterrows():
        if str(row.get("Status", "")).upper() != "FINISHED":
            continue
        home, away = row.get("Home"), row.get("Away")

        if str(row.get("Status", "")).upper() not in COMPLETED_FIXTURE_STATUSES:
            continue

        if not _is_confirmed_team(home) or not _is_confirmed_team(away):
            continue
        winner = _fixture_winner(row)
        if not winner:
            continue
        home_score, away_score = _fixture_score(row)
        completed[_pair_key(home, away)] = {
            "winner": winner,
            "home": home,
            "away": away,
            "home_score": home_score,
            "away_score": away_score,
            "row": row,
        }
    return completed


def _live_r32_card(match_ref: int, fixture_by_ref: dict[int, pd.Series]) -> tuple[dict, str | None, str | None]:
    row = fixture_by_ref.get(match_ref)
    if row is not None:
        home, away = row.get("Home", ""), row.get("Away", "")
        home_score, away_score = _fixture_score(row)
        winner = _fixture_winner(row)
        status = str(row.get("Status", ""))
    else:
        home, away = DEFAULT_R32_TEAMS.get(match_ref, (BRACKET_PLACEHOLDER, BRACKET_PLACEHOLDER))
        home_score = away_score = ""
        winner = None
        status = ""
    card = _match_card(
        "R32",
        _entry(home, home_score, winner == home),
        _entry(away, away_score, winner == away),
        ref=match_ref,
        status=status,
    )
    loser = away if winner and winner == home else home if winner and winner == away else None
    return card, winner, loser


def _progression_card(title: str, first: str | None, second: str | None,
                      winner: str | None = None, first_stat: str = "",
                      second_stat: str = "") -> tuple[dict, str | None, str | None]:
    first_entry = _entry(first, first_stat, bool(winner and first == winner)) if first else _blank_entry(first_stat)
    second_entry = _entry(second, second_stat, bool(winner and second == winner)) if second else _blank_entry(second_stat)
    loser = second if winner and first and second and winner == first else first if winner and first and second else None
    return _match_card(title, first_entry, second_entry), winner, loser


# Manually-curated snapshot of confirmed Round-of-16 results, current as of
# July 8, 2026. This exists purely as a fallback for when the upstream API
# feed is lagging/incomplete (e.g. free-tier rate limits, caching, or a
# competition's `Stage` field not being populated yet) - it is NEVER allowed
# to override a real row that the API already returned. Once the live feed
# reliably carries R16+ data this block is safe to delete; nothing else
# depends on it.
_FALLBACK_KNOCKOUT_RESULTS_2026: list[dict] = [
    {"Home": "Morocco",     "Away": "Canada",      "Home Goals": 3, "Away Goals": 0, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "France",      "Away": "Paraguay",    "Home Goals": 1, "Away Goals": 0, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "Norway",      "Away": "Brazil",      "Home Goals": 2, "Away Goals": 1, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "England",     "Away": "Mexico",      "Home Goals": 3, "Away Goals": 2, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "Spain",       "Away": "Portugal",    "Home Goals": 1, "Away Goals": 0, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "Belgium",     "Away": "USA",         "Home Goals": 4, "Away Goals": 1, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    {"Home": "Argentina",   "Away": "Egypt",       "Home Goals": 3, "Away Goals": 2, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
    # Draw after 120 minutes; Switzerland advanced 4-3 on penalties. Goal
    # totals reflect regulation/ET only, per the same no-penalties-in-GF rule
    # applied everywhere else in this file (see `_regulation_score`).
    {"Home": "Switzerland", "Away": "Colombia",    "Home Goals": 0, "Away Goals": 0, "Winner": "HOME_TEAM", "Status": "FINISHED", "Stage": "ROUND_OF_16"},
]


def _fixture_pair_lookup(wc_fixtures_df: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    """Maps every knockout-stage fixture to its team-pair key, prioritizing API data."""
    lookup: dict[tuple[str, str], pd.Series] = {}
    
    # 1. First, process all real API data from the fixture feed
    if wc_fixtures_df is not None and not wc_fixtures_df.empty:
        knockout_stage_codes = {code for code, _ in KNOCKOUT_STAGE_ORDER}
        for _, row in wc_fixtures_df.iterrows():
            stage = str(row.get("Stage", "") or "").upper()
            if stage and stage not in knockout_stage_codes:
                continue
            home, away = row.get("Home"), row.get("Away")
            if not _is_confirmed_team(home) or not _is_confirmed_team(away):
                continue
            # Real API data takes precedence; just overwrite/set it
            lookup[_pair_key(home, away)] = row

    # 2. Only fill in gaps with fallback results if the pair isn't already known
    for fallback_row in _FALLBACK_KNOCKOUT_RESULTS_2026:
        key = _pair_key(fallback_row["Home"], fallback_row["Away"])
        if key not in lookup:
            # Cast fallback dict to a Series to maintain API-like attribute access
            lookup[key] = pd.Series(fallback_row)

    return lookup


def _live_progression_card(title: str, first: str | None, second: str | None,
                           fixture_pair_lookup: dict[tuple[str, str], pd.Series]
                           ) -> tuple[dict, str | None, str | None]:
    
    # If both are missing, it's a completely unseeded bracket slot
    if not first and not second:
        return _match_card(title, _blank_entry(), _blank_entry()), None, None

    # If both teams are known, look up if they have played each other yet
    if first and second:
        row = fixture_pair_lookup.get(_pair_key(first, second))
        if row is not None:
            home = row.get("Home", first) or first
            away = row.get("Away", second) or second
            
            if _team_key(home) == _team_key(first):
                home_score, away_score = _fixture_score(row)
            else:
                away_score, home_score = _fixture_score(row)
                
            winner = _fixture_winner(row)
            is_first_winner = _team_key(winner) == _team_key(first) if winner else False
            is_second_winner = _team_key(winner) == _team_key(second) if winner else False
            
            card = _match_card(
                title,
                _entry(first, home_score, is_first_winner),
                _entry(second, away_score, is_second_winner),
                status=str(row.get("Status", "")),
            )
            loser = second if is_first_winner else (first if is_second_winner else None)
            return card, winner, loser

    # --- NEW PARTIAL ROUND FILLER EDGE CASE ---
    # If one team is known but their opponent is still TBD, fill the known team slot!
    first_entry = _entry(first) if first else _blank_entry()
    second_entry = _entry(second) if second else _blank_entry()
    
    return _match_card(title, first_entry, second_entry), None, None


def build_live_bracket_state(wc_fixtures_df: pd.DataFrame) -> dict:
    fixture_by_ref: dict[int, pd.Series] = {}
    if wc_fixtures_df is not None and not wc_fixtures_df.empty:
        for _, row in wc_fixtures_df.iterrows():
            ref = _fixture_match_ref(row)
            if ref not in LEFT_R32_MATCH_REFS + RIGHT_R32_MATCH_REFS:
                ref = _default_ref_from_pair(row.get("Home"), row.get("Away"))
            if ref is not None:
                fixture_by_ref[ref] = row

    pair_lookup = _fixture_pair_lookup(wc_fixtures_df)

    left_r32, right_r32 = [], []
    left_winners, right_winners = [], []

    # --- LEFT SIDE ROUND OF 32 ---
    for ref in LEFT_R32_MATCH_REFS:
        card, winner, _ = _live_r32_card(ref, fixture_by_ref)
        # FALLBACK SAFETY: If the API missing row scenario triggers, look up by team pair defaults
        if not winner:
            default_home, default_away = DEFAULT_R32_TEAMS.get(ref, (None, None))
            if default_home and default_away:
                p_row = pair_lookup.get(_pair_key(default_home, default_away))
                if p_row is not None:
                    winner = _fixture_winner(p_row)
                    
                    # Prevent inverted scores by aligning Home/Away with the Bracket default order
                    row_home = str(p_row.get("Home", ""))
                    if _team_key(row_home) == _team_key(default_home):
                        home_score, away_score = _fixture_score(p_row)
                    else:
                        away_score, home_score = _fixture_score(p_row)
                        
                    status = str(p_row.get("Status", ""))
                    
                    # Guarantee the string matches perfectly for the CSS green highlight
                    is_home_winner = _team_key(winner) == _team_key(default_home) if winner else False
                    is_away_winner = _team_key(winner) == _team_key(default_away) if winner else False
                    
                    card = _match_card(
                        "R32",
                        _entry(default_home, home_score, is_home_winner),
                        _entry(default_away, away_score, is_away_winner),
                        ref=ref,
                        status=status
                    )
        left_r32.append(card)
        left_winners.append(winner)

    # --- RIGHT SIDE ROUND OF 32 ---
    for ref in RIGHT_R32_MATCH_REFS:
        card, winner, _ = _live_r32_card(ref, fixture_by_ref)
        # FALLBACK SAFETY: If the API missing row scenario triggers, look up by team pair defaults
        if not winner:
            default_home, default_away = DEFAULT_R32_TEAMS.get(ref, (None, None))
            if default_home and default_away:
                p_row = pair_lookup.get(_pair_key(default_home, default_away))
                if p_row is not None:
                    winner = _fixture_winner(p_row)
                    
                    # Prevent inverted scores by aligning Home/Away with the Bracket default order
                    row_home = str(p_row.get("Home", ""))
                    if _team_key(row_home) == _team_key(default_home):
                        home_score, away_score = _fixture_score(p_row)
                    else:
                        away_score, home_score = _fixture_score(p_row)
                        
                    status = str(p_row.get("Status", ""))
                    
                    # Guarantee the string matches perfectly for the CSS green highlight
                    is_home_winner = _team_key(winner) == _team_key(default_home) if winner else False
                    is_away_winner = _team_key(winner) == _team_key(default_away) if winner else False
                    
                    card = _match_card(
                        "R32",
                        _entry(default_home, home_score, is_home_winner),
                        _entry(default_away, away_score, is_away_winner),
                        ref=ref,
                        status=status
                    )
        right_r32.append(card)
        right_winners.append(winner)

    def build_round(prev_winners: list[str | None], title: str) -> tuple[list[dict], list[str | None], list[str | None]]:
        cards, winners, losers = [], [], []
        for idx in range(0, len(prev_winners), 2):
            a = prev_winners[idx] if idx < len(prev_winners) else None
            b = prev_winners[idx + 1] if idx + 1 < len(prev_winners) else None
            card, winner, loser = _live_progression_card(title, a, b, pair_lookup)
            cards.append(card)
            winners.append(winner)
            losers.append(loser)
        return cards, winners, losers

    left_r16, left_r16_winners, _ = build_round(left_winners, "R16")
    left_qf, left_qf_winners, _ = build_round(left_r16_winners, "QF")
    left_sf, left_sf_winners, left_sf_losers = build_round(left_qf_winners, "SF")
    
    right_r16, right_r16_winners, _ = build_round(right_winners, "R16")
    right_qf, right_qf_winners, _ = build_round(right_r16_winners, "QF") 
    right_sf, right_sf_winners, right_sf_losers = build_round(right_qf_winners, "SF")
    
    final_card, final_winner, _ = _live_progression_card("Final", left_sf_winners[0], right_sf_winners[0], pair_lookup)
    third_card, _, _ = _live_progression_card("Third Place", left_sf_losers[0], right_sf_losers[0], pair_lookup)

    return {
        "left_r32": left_r32, "left_r16": left_r16, "left_qf": left_qf, "left_sf": left_sf,
        "final": final_card, "third": third_card, "champion": _entry(final_winner) if final_winner else _blank_entry(),
        "right_sf": right_sf, "right_qf": right_qf, "right_r16": right_r16, "right_r32": right_r32,
    }


def _pair_probability(team_a: str, team_b: str, ratings_lookup: dict, lgf: float, lga: float,
                      saf_lookup=None, model_bundle=None) -> tuple[str, float, str, float]:
    sa = ratings_lookup.get(team_a, DEFAULT_TEAM_STATS)
    sb = ratings_lookup.get(team_b, DEFAULT_TEAM_STATS)
    saf_a = saf_lookup.get(team_a, 0.85) if saf_lookup else 0.85
    saf_b = saf_lookup.get(team_b, 0.85) if saf_lookup else 0.85
    xg_a, xg_b = calculate_expected_goals(sa, sb, lgf, lga, saf_a=saf_a, saf_b=saf_b, team_a_name=team_a, team_b_name=team_b)
    rd = sa.get("Rating", 0.5) - sb.get("Rating", 0.5)
    p_a, draw, p_b = blended_probs(xg_a, xg_b, rating_diff=rd, saf_diff=saf_a - saf_b, model_bundle=model_bundle)
    tie_a = sa.get("Rating", 0.5) * (saf_a / 0.85)
    tie_b = sb.get("Rating", 0.5) * (saf_b / 0.85)
    tie_total = max(tie_a + tie_b, 0.0001)
    advance_a = p_a + draw * (tie_a / tie_total)
    advance_b = p_b + draw * (tie_b / tie_total)
    if advance_a >= advance_b:
        return team_a, advance_a * 100, team_b, advance_b * 100
    return team_b, advance_b * 100, team_a, advance_a * 100


def build_simulated_bracket_state(ratings_lookup: dict, lgf: float, lga: float,
                                  saf_lookup=None, model_bundle=None,
                                  fixed_results=None) -> dict:
    left_r32, right_r32 = [], []
    fixed_results = fixed_results or {}

    wc_results = st.session_state.get("wc_results")
    mc_probs = {}
    if wc_results is not None and not wc_results.empty:
        mc_probs = dict(zip(wc_results["Team"], wc_results["Champion %"]))

    def get_advancement_metric(team: str) -> float:
        if mc_probs and team in mc_probs:
            return mc_probs[team]
        return ratings_lookup.get(team, {}).get("Rating", 0.5)

    def play_card(title: str, team_a: str, team_b: str, ref=None, is_r32=False) -> tuple[dict, str, str, float, float]:
        # team_a/team_b always come in DEFAULT_R32_TEAMS[ref] structural order
        # (see the R32 loop below) — everything here must key off _team_key()
        # normalization rather than raw string equality, since the API's
        # home/away/winner strings (aliases like "Bosnia & Herz.") won't
        # always match the literal names in DEFAULT_R32_TEAMS.
        known = fixed_results.get(_pair_key(team_a, team_b))
        if known:
            k_team_a = _team_key(team_a)
            k_winner = _team_key(known.get("winner"))
            k_home = _team_key(known.get("home"))

            winner = team_a if k_winner == k_team_a else team_b
            loser = team_b if winner == team_a else team_a
            win_pct, lose_pct = 100.0, 0.0

            # Align scores to team_a/team_b's structural position, not
            # whatever order the API happened to report home/away in.
            if k_home == k_team_a:
                score_a = known.get("home_score", "")
                score_b = known.get("away_score", "")
            else:
                score_a = known.get("away_score", "")
                score_b = known.get("home_score", "")
        else:
            calc_winner, calc_win_pct, calc_loser, calc_lose_pct = _pair_probability(
                team_a, team_b, ratings_lookup, lgf, lga, saf_lookup, model_bundle
            )

            # --- Always trust the direct head-to-head outcome ---
            winner = calc_winner
            loser = calc_loser
            prob_a = calc_win_pct if team_a == calc_winner else calc_lose_pct
            prob_b = calc_lose_pct if team_a == calc_winner else calc_win_pct

            score_a = f"{prob_a:.0f}%"
            score_b = f"{prob_b:.0f}%"
            win_pct = prob_a if team_a == winner else prob_b
            lose_pct = prob_b if team_a == winner else prob_a

        # Normalized winner comparison so the green "winner" style reliably
        # applies even when `winner` came back as an aliased/differently
        # cased string from the fixed-results lookup.
        k_team_a = _team_key(team_a)
        k_team_b = _team_key(team_b)
        k_winner = _team_key(winner)

        card = _match_card(
            title,
            _entry(team_a, score_a, k_team_a == k_winner),
            _entry(team_b, score_b, k_team_b == k_winner),
            ref=ref,
            status="FINAL" if known else "SIM",
        )
        return card, winner, loser, win_pct, lose_pct

    def play_round(teams: list[str], title: str) -> tuple[list[dict], list[str], list[str]]:
        cards, winners, losers = [], [], []
        for idx in range(0, len(teams), 2):
            card, winner, loser, _, _ = play_card(title, teams[idx], teams[idx + 1])
            cards.append(card)
            winners.append(winner)
            losers.append(loser)
        return cards, winners, losers

    left_winners, right_winners = [], []
    for ref in LEFT_R32_MATCH_REFS:
        # a, b are pulled directly from DEFAULT_R32_TEAMS in bracket order —
        # this ordering is what play_card's team_a/team_b now strictly honor.
        a, b = DEFAULT_R32_TEAMS[ref]
        card, winner, _, _, _ = play_card("R32", a, b, ref=ref, is_r32=True)
        left_r32.append(card)
        left_winners.append(winner)
    for ref in RIGHT_R32_MATCH_REFS:
        a, b = DEFAULT_R32_TEAMS[ref]
        card, winner, _, _, _ = play_card("R32", a, b, ref=ref, is_r32=True)
        right_r32.append(card)
        right_winners.append(winner)

    left_r16, left_r16_winners, _ = play_round(left_winners, "R16")
    left_qf, left_qf_winners, _ = play_round(left_r16_winners, "QF")
    left_sf, left_sf_winners, left_sf_losers = play_round(left_qf_winners, "SF")

    right_r16, right_r16_winners, _ = play_round(right_winners, "R16")
    right_qf, right_qf_winners, _ = play_round(right_r16_winners, "QF")
    right_sf, right_sf_winners, right_sf_losers = play_round(right_qf_winners, "SF")

    final_card, champion, _, champ_pct, _ = play_card("Final", left_sf_winners[0], right_sf_winners[0])
    third_card, _, _, _, _ = play_card("Third Place", left_sf_losers[0], right_sf_losers[0])

    return {
        "left_r32": left_r32, "left_r16": left_r16, "left_qf": left_qf, "left_sf": left_sf,
        "final": final_card, "third": third_card, "champion": _entry(champion, f"{champ_pct:.0f}%", True),
        "right_sf": right_sf, "right_qf": right_qf, "right_r16": right_r16, "right_r32": right_r32,
    }


def _ticker_items_from_data(live_df: pd.DataFrame, results_df: pd.DataFrame, max_items: int = 16) -> list[str]:
    items: list[str] = []
    rankings = fetch_dynamic_fifa_rankings()

    if live_df is not None and not live_df.empty:
        for _, row in live_df.iterrows():
            minute = row.get("Minute", "LIVE")
            minute_tag = f" ({minute})" if minute and minute != "LIVE" else ""
            items.append(f"LIVE: {row.get('Home','')} {row.get('Score','- -')} {row.get('Away','')}{minute_tag}")

    if results_df is not None and not results_df.empty:
        for _, row in results_df.head(max_items).iterrows():
            home = row.get("Home", "")
            away = row.get("Away", "")
            
            home_score, away_score = _fixture_score(row)
            if home_score and away_score:
                score = f"{home_score} - {away_score}"
            else:
                score = row.get("Score", "- -")
                
            winner = _fixture_winner(row)
            
            hg = row.get("Home Goals")
            ag = row.get("Away Goals")
            try:
                is_draw = (float(hg) == float(ag))
            except:
                is_draw = "-" in score and score.split("-")[0].strip() == score.split("-")[1].strip()

            ticker_str = f"⚽ {home} {score} {away}"
            
            if is_draw:
                ticker_str = f'<span class="ticker-red">{ticker_str}</span>'
            elif winner:  # Removed strict "and rankings" requirement check
                home_elo = _lookup_elo_rating(home, rankings) if rankings else 1500
                away_elo = _lookup_elo_rating(away, rankings) if rankings else 1500
                home_elo = home_elo or 1500
                away_elo = away_elo or 1500
                
                favorite = home if home_elo >= away_elo else away
                underdog = away if home_elo >= away_elo else home
                

            if _team_key(winner) == _team_key(favorite):
               ticker_str = f'<span class="ticker-green">{ticker_str}</span>'
            elif _team_key(winner) == _team_key(underdog):
               ticker_str = f'<span class="ticker-red">{ticker_str}</span>'

            items.append(ticker_str)

    return items[:max_items]

# ─── 8d. LIVE TICKER RENDERING ENGINE ───────────────────────────────────────
def render_live_ticker(live_df: pd.DataFrame, results_df: pd.DataFrame):
    """Stock-market style scrolling marquee summarizing live/recent matches."""
    items = _ticker_items_from_data(live_df, results_df)

    intro_msg = "WORLD CUP 2026 PREDICTOR &nbsp;&nbsp;|&nbsp;&nbsp; Green is expected but red is surprising"

    if not items:
        ticker_text = f"{intro_msg} &nbsp;&nbsp;|&nbsp;&nbsp; Live scores and results will appear here once matches kick off"
    else:
        ticker_text = f"{intro_msg} &nbsp;&nbsp;|&nbsp;&nbsp; " + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(items)
        ticker_text = f"{ticker_text}&nbsp;&nbsp;|&nbsp;&nbsp;"

    has_live = live_df is not None and not live_df.empty
    accent = "#e74c3c" if has_live else "#3498db"

    html = f"""
    <div class="wc-ticker-wrapper" style="width:100%; overflow:hidden; background-color:#0e1117; border:1px solid {accent}; border-radius:6px; padding:8px 0; margin:6px 0 18px 0; position: relative;">
        <div class="wc-ticker-track" style="display:inline-block; white-space:nowrap; padding-left:100%; animation:wc-ticker-scroll 60s linear infinite; color:#e8e8e8; font-size:14px; font-weight:500; font-family:'Courier New', monospace; will-change: transform;">
            {ticker_text}
        </div>
    </div>
    <style>
    /* Add your new classes here */
    .ticker-red {{ color: #e74c3c !important; font-weight: 700; }}
    .ticker-green {{ color: #2ecc71 !important; font-weight: 700; }}
    
    .wc-ticker-wrapper {{ 
        display: block; 
        max-width: 100%; 
    }}
    @keyframes wc-ticker-scroll {{ 
        0% {{ transform: translateX(0); }} 
        100% {{ transform: translateX(-100%); }} 
    }}
    </style>
    """
    st.markdown(html, unsafe_allow_html=True)


KNOCKOUT_STAGE_ORDER: list[tuple[str, str]] = [
    ("LAST_32",         "🌍 Round of 32"),
    ("ROUND_OF_32",     "🌍 Round of 32"),
    ("LAST_16",         "🎯 Round of 16"),
    ("ROUND_OF_16",     "🎯 Round of 16"),
    ("QUARTER_FINALS",  "⚡ Quarterfinals"),
    ("SEMI_FINALS",     "🔥 Semifinals"),
    ("THIRD_PLACE",     "🥉 Third Place Playoff"),
    ("FINAL",           "🏆 Final"),
]

_PLACEHOLDER_TOKENS = ("tbd", "winner", "runner-up", "runner up", "3rd place")


def _is_confirmed_team(name) -> bool:
    if not isinstance(name, str) or not name.strip():
        return False
    lowered = name.lower()
    return not any(tok in lowered for tok in _PLACEHOLDER_TOKENS)


ACTIVE_FIXTURE_STATUSES = {"SCHEDULED", "TIMED", "IN_PLAY"}
COMPLETED_FIXTURE_STATUSES = {"FINISHED", "TIMED"}


def compute_active_teams(fixtures_df: pd.DataFrame) -> set[str]:
    if fixtures_df is None or fixtures_df.empty or "Status" not in fixtures_df.columns:
        return set()
        
    # 1. Base check: Anyone in an upcoming or live match is active
    upcoming = fixtures_df[fixtures_df["Status"].isin(ACTIVE_FIXTURE_STATUSES)]
    active_teams = set(upcoming.get("Home", pd.Series(dtype=str)).tolist()) | \
                   set(upcoming.get("Away", pd.Series(dtype=str)).tolist())
                   
    # 2. Pipeline check: Carry over winners who are in limbo between rounds
    completed_knockout = fixtures_df[
        (fixtures_df["Status"] == "FINISHED") & 
        (fixtures_df["Stage"].isin(["ROUND_OF_16", "QUARTER_FINALS", "SEMI_FINALS"]))
    ]
    
    for _, row in completed_knockout.iterrows():
        winner = _fixture_winner(row)
        if winner:
            active_teams.add(winner)
            
    return {t for t in active_teams if _is_confirmed_team(t)}


def compute_confirmed_teams(fixtures_df: pd.DataFrame) -> set[str]:
    """Match Predictor variant: the [IN]/[OUT] label here is meant to reflect
    settled, on-the-record results only, so it only looks at FINISHED matches
    and ignores SCHEDULED/TIMED/IN_PLAY fixtures entirely."""
    if fixtures_df is None or fixtures_df.empty or "Status" not in fixtures_df.columns:
        return set()
    completed = fixtures_df[fixtures_df["Status"].isin(COMPLETED_FIXTURE_STATUSES)]
    teams = set(completed.get("Home", pd.Series(dtype=str)).tolist()) | \
            set(completed.get("Away", pd.Series(dtype=str)).tolist())
    return {t for t in teams if _is_confirmed_team(t)}


def render_live_knockout_bracket(wc_fixtures_df: pd.DataFrame):
    """Displays the live knockout tree from confirmed API match refs 73-88."""
    st.subheader("Live Knockout Bracket")
    if wc_fixtures_df is None or wc_fixtures_df.empty:
        st.info("Knockout bracket data is not available yet; placeholder slots are shown below.")
        render_horizontal_bracket(build_live_bracket_state(pd.DataFrame()))
        return
    render_horizontal_bracket(build_live_bracket_state(wc_fixtures_df))

def _inject_global_css():
    """App-wide CSS: compact About-tab typography."""
    st.markdown(
        """
        <style>
        /* Compact About-tab typography (mobile-friendly) */
        .about-hero {
            background-color:#1e293b;
            padding:14px 18px;
            border-radius:10px;
            border-left:6px solid #f1c40f;
            margin-bottom:14px;
        }
        .about-hero h2 {
            margin-top:0;
            margin-bottom:6px;
            color:#f1c40f;
            font-weight:900;
            letter-spacing:0.3px;
            font-size:1.35em;
            line-height:1.25;
        }
        .about-hero p {
            font-size:.92em;
            color:#f8fafc;
            line-height:1.45;
            margin-bottom:0;
        }
        .about-subhead {
            font-size:1.05em;
            font-weight:700;
            margin:10px 0 2px 0;
        }
        .about-compact {
            font-size:.88em;
            line-height:1.45;
            margin:2px 0 6px 0;
        }
        h4.about-track {
            margin:10px 0 2px 0;
            font-size:1em;
        }

        .tab-guide {
            background-color:#161a23;
            border:1px solid #2a2f3a;
            border-radius:8px;
            padding:12px 16px;
            margin:2px 0 14px 0;
        }
        .tab-guide-lede {
            font-size:.88em;
            color:#ddd;
            margin:0 0 8px 0;
            line-height:1.4;
        }
        .tab-guide-list {
            list-style-type:none;
            padding-left:0;
            margin:0;
            line-height:1.7;
            font-size:.88em;
            color:#ccc;
        }
        .tab-guide-icon {
            display:inline-block;
            width:1.4em;
        }

        @media (max-width: 640px) {
            .about-hero { padding:10px 12px; margin-bottom:10px; }
            .about-hero h2 { font-size:1.1em; }
            .about-hero p { font-size:.85em; }
            .about-subhead { font-size:.95em; }
            .about-compact { font-size:.82em; }
            .tab-guide { padding:10px 12px; }
            .tab-guide-lede, .tab-guide-list { font-size:.82em; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _inject_tab_nav_css():
    """Styles the st.radio-based navigation bar to visually read as st.tabs,
    while remaining fully controllable via st.session_state['active_tab']
    (native st.tabs offers no such programmatic control)."""
    st.markdown(
        """
        <style>
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: row;
            flex-wrap: wrap;
            gap: 4px;
            border-bottom: 2px solid #262730;
            padding-bottom: 0;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label {
            background: #1a1c23;
            padding: 9px 18px;
            border-radius: 8px 8px 0 0;
            border: 1px solid #262730;
            border-bottom: none;
            margin-right: 2px;
            cursor: pointer;
            transition: background .15s ease, color .15s ease;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
            background: #23262f;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
            background: #0e1117;
            border-bottom: 2px solid #f1c40f;
            margin-bottom: -2px;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) p {
            color: #f1c40f !important;
            font-weight: 800 !important;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] input[type="radio"] {
            display: none;
        }
        div[data-testid="stRadio"] svg { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tab_nav() -> str:
    # 1. Look for a redirect request
    if "nav_redirect" in st.session_state and st.session_state["nav_redirect"]:
        st.session_state["active_tab"] = st.session_state["nav_redirect"]
        st.session_state["nav_redirect"] = None # Clear it immediately
        
    # 2. Ensure default
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = TAB_LABELS[0]

    _inject_tab_nav_css()
    
    # 3. The radio is now driven by 'active_tab'
    return st.radio(
        "Navigation",
        TAB_LABELS,
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

# Refactored: Visual settings bar fully removed out of existence. Secrets are resolved securely and defaults are pinned.
def render_sidebar():
    api_key = st.secrets.get("FOOTBALL_API_KEY", "")
    return api_key, WORLD_CUP_LEAGUE_ID, WORLD_CUP_SEASON, 30


def update_nav(target):
    st.session_state["nav_redirect"] = target

def render_about_tab():
    st.markdown(
        """
        <div class="about-hero">
            <h2>⚽ Welcome to Ayush’s WorldCup prediction engine!</h2>
            <p>This system combines historical analytics, live performance metrics, and mathematical prediction models to forecast potential international football outcomes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tab-guide">
            <p class="tab-guide-lede">You can jump between sections any time using the <strong>tabs in the top navigation bar</strong>. Here's what each one does:</p>
            <ul class="tab-guide-list">
                <li><span class="tab-guide-icon">📅</span><strong>Live Fixtures</strong> &mdash; browse upcoming, in-play, and recently completed matches.</li>
                <li><span class="tab-guide-icon">📊</span><strong>Team Ratings</strong> &mdash; explore the blended rating engine and see how every team stacks up.</li>
                <li><span class="tab-guide-icon">⚔️</span><strong>Match Predictor</strong> &mdash; pick any two teams and simulate a single head-to-head matchup.</li>
                <li><span class="tab-guide-icon">🏆</span><strong>World Cup Simulator</strong> &mdash; run thousands of Monte Carlo tournaments to see each team's title odds.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="about-subhead">🤔 You may be asking yourself... <em>HOW DOES THIS EVEN WORK!?</em></div>', unsafe_allow_html=True)
    st.markdown('<p class="about-compact">Let me explain... Instead of relying on just one statistic, the model combines several different factors to determine how "strong" a national team really is.</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Track 1: Recent Form
    st.markdown(
        """
        <h4 class="about-track"><span style="color:#3498db; font-weight:bold;">Recent Form & Nostalgia Decay</span></h4>
        <p class="about-compact">One of the biggest factors is recent form. Recent matches are given much more weight than less relevant games played years ago using an exponential decay function:</p>
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"w = e^{-\lambda \cdot t}")
    st.markdown(
        """
        <p class="about-compact">where older matches gradually become less important over time. This helps <strong>eliminate nostalgia bias</strong> because what really matters is how a team is playing right now.</p>
        """,
        unsafe_allow_html=True,
    )

    # Track 2: SoS
    st.markdown(
        """
        <h4 class="about-track"><span style="color:#e67e22; font-weight:bold;">Strength of Schedule (SoS) scaling</span></h4>
        <p class="about-compact">Beating a world-class team is worth far more than beating a lower-ranked team (<em>duh!</em>), while losing to an elite opponent is less damaging than losing to a weaker side. After all, in a knockout tournament like the World Cup, you know what they say: <em>"You gotta beat the Best to be the Best."</em> 😉</p>
        """,
        unsafe_allow_html=True,
    )

    # Track 3: Elo Baseline
    st.markdown(
        """
        <h4 class="about-track"><span style="color:#9b59b6; font-weight:bold;">Live Elo Integration</span></h4>
        <p class="about-compact">The model also incorporates live Elo ratings, which estimate each team's overall strength based on previous results. Elo calculates the expected outcome of a match using each team's rating before adjusting those ratings after every game. This gives the model a <strong>strong baseline</strong> instead of making predictions with no context of how much quality a team has.</p>
        """,
        unsafe_allow_html=True,
    )

    # Track 4: Poisson distribution
    st.markdown(
        """
        <h4 class="about-track"><span style="color:#2ecc71; font-weight:bold;">Poisson Exact Scoreline Matrix</span></h4>
        <p class="about-compact">To predict actual scorelines, the model uses a Poisson distribution, a mathematical model commonly used for low-scoring sports like soccer. Using each team's expected goals (&lambda;), it calculates the probability of scoring exactly 0, 1, 2, 3, or more goals, allowing it to estimate the likelihood of every possible scoreline from 0-0 to 3-2 and beyond.</p>
        """,
        unsafe_allow_html=True,
    )

    # Track 5: Monte Carlo format simulations
    st.markdown(
        """
        <h4 class="about-track"><span style="color:#f1c40f; font-weight:bold;">Stochastic Monte Carlo Bracket Simulations</span></h4>
        <p class="about-compact">And finally, the entire tournament is run through thousands of Monte Carlo simulations. Each simulated tournament randomly selects winners based on the calculated match probabilities. By repeating this process thousands of times, the model measures how often each team reaches the <strong>Round of 16, Quarterfinals, Semifinals, Final, or lifts the trophy</strong>. Those percentages become each team's estimated chances of advancing through the tournament and ultimately becoming World Champions.</p>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="background-color:#111; padding:14px 18px; border-radius:8px; border:1px solid #222;">
            <h3 style="margin-top:0; margin-bottom:8px; color:#ffffff; font-size:1.15em;">🛠️ Tech Stack</h3>
            <ul style="list-style-type:none; padding-left:0; line-height:1.75; color:#bbb; font-size:.88em; margin-bottom:0;">
                <li><strong style="color:#f1c40f;">App &amp; UI:</strong> Streamlit (fragment-based reactive routing), custom CSS/HTML component styling</li>
                <li><strong style="color:#3498db;">Data Science:</strong> NumPy, Pandas &mdash; time-decay weighting, SoS-adjusted rating engine</li>
                <li><strong style="color:#9b59b6;">Machine Learning:</strong> XGBoost + scikit-learn (StandardScaler, train/test split) blended with a Poisson statistical model</li>
                <li><strong style="color:#2ecc71;">Statistical Modeling:</strong> SciPy (Poisson distributions), vectorized scoreline matrices, Monte Carlo tournament simulation</li>
                <li><strong style="color:#e67e22;">Live Data Integration:</strong> Football-Data.org REST API, requests, custom caching layer (<code>st.cache_data</code>)</li>
                <li><strong style="color:#1abc9c;">Web Scraping / ETL:</strong> BeautifulSoup4 &mdash; automated FBref form, Transfermarkt squad, and Elo ranking pipelines</li>
                <li><strong style="color:#e74c3c;">Conversational AI:</strong> Google Gemini API &mdash; context-aware analyst chatbot grounded in live simulation state</li>
                <li><strong style="color:#f1c40f;">Visualization:</strong> Plotly (Express &amp; Graph Objects) &mdash; interactive rating charts &amp; live bracket rendering</li>
                <li><strong style="color:#3498db;">Engineering Practices:</strong> Modular pipeline architecture, defensive API error-handling, reproducible model persistence via Joblib</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_fixtures_tab(api_key, live_df, fixtures_df, results_df, wc_fixtures_df=None):
    st.header("Live Fixtures")
    if not api_key:
        st.info("API Token connection missing from secrets management configuration.")
        return

    bracket_sources = [
        df for df in (wc_fixtures_df, fixtures_df, results_df, live_df)
        if df is not None and not df.empty
    ]
    bracket_df = pd.concat(bracket_sources, ignore_index=True, sort=False) if bracket_sources else pd.DataFrame()
    render_live_knockout_bracket(bracket_df)
    
    st.markdown("---")

    st.subheader("Live Matches")
    if live_df.empty:
        st.info("No matches currently live.")
    else:
        live_cols = [c for c in ["League", "Minute", "Home", "Away", "Score"] if c in live_df.columns]
        st.dataframe(live_df[live_cols], use_container_width=True, hide_index=True)
    st.subheader("Upcoming Matches")
    if fixtures_df.empty:
        st.info("No upcoming fixtures found.")
    else:
        fixture_cols = [c for c in ["Date", "League", "Round", "Home", "Away", "Status"] if c in fixtures_df.columns]
        st.dataframe(fixtures_df[fixture_cols], use_container_width=True, hide_index=True)
    st.subheader("Recent Results")
    if results_df.empty:
        st.info("No recent results found.")
    else:
        result_cols = [c for c in ["Date", "League", "Home", "Score", "Away"] if c in results_df.columns]
        st.dataframe(results_df[result_cols], use_container_width=True, hide_index=True)
    st.caption(f"Last refreshed: {datetime.now():%Y-%m-%d %H:%M:%S}")


def _build_insight_cards(team_stats_df: pd.DataFrame, active_teams: set[str],
                          eliminated_teams: set[str] = frozenset()) -> list[tuple]:
    df = team_stats_df
    if df.empty:
        return [("ℹ️", "Data", "N/A", "Waiting for API")]

    active_df = df[df["Team"].isin(active_teams)] if active_teams else df
    cards = []

    # --- 1. Battle-Tested (never "None") ---
    bt_pool = active_df if not active_df.empty else df
    if "SoS" in bt_pool.columns and bt_pool["SoS"].notna().any():
        idx = bt_pool["SoS"].idxmax()
        row = bt_pool.loc[idx]
        cards.append(("👹", "Battle-Tested", row["Team"], f"Avg SoS {row['SoS']:.1f}"))
    else:
        row = bt_pool.sort_values("Rating", ascending=False).iloc[0]
        cards.append(("👹", "Battle-Tested", row["Team"], f"Rating {row['Rating']:.3f}"))

    # --- 2. Dark Horse (never "None") — pure underdog: low Elo Mod, high Rating ---
    dh_pool_source = active_df if not active_df.empty else df
    if "Elo Mod" in dh_pool_source.columns and "Rating" in dh_pool_source.columns:
        if len(dh_pool_source) > 3:
            threshold = max(dh_pool_source["Elo Mod"].quantile(0.4), 0.80)
        else:
            threshold = 0.80
        dh_pool = dh_pool_source[dh_pool_source["Elo Mod"] < threshold]
        if dh_pool.empty:
            dh_pool = dh_pool_source

        dh_pool = dh_pool.copy()
        safe_elo = dh_pool["Elo Mod"].replace(0, 0.01)
        dh_pool["dark_horse_score"] = dh_pool["Rating"] / safe_elo
        idx = dh_pool["dark_horse_score"].idxmax()
        row = dh_pool.loc[idx]
        cards.append(("🚀", "Dark Horse", row["Team"], f"Rating {row['Rating']:.3f}"))
    else:
        row = dh_pool_source.sort_values("Rating", ascending=False).iloc[0]
        cards.append(("🚀", "Dark Horse", row["Team"], f"Rating {row['Rating']:.3f}"))

    # --- 3. Best Form (Safe indexing) ---
    if not active_df.empty and "PPG" in active_df.columns and active_df["PPG"].notna().any():
        row = active_df.loc[active_df["PPG"].idxmax()]
        cards.append(("🏆", "Best Form", row["Team"], f"{row['PPG']:.2f} PPG"))
    else:
        row = df.loc[df["PPG"].idxmax()]
        cards.append(("🏆", "Best Form", row["Team"], f"{row['PPG']:.2f} PPG"))

    # --- 4. Goal Machine (Safe indexing) ---
    if not active_df.empty and "GF/Game" in active_df.columns and active_df["GF/Game"].notna().any():
        row = active_df.loc[active_df["GF/Game"].idxmax()]
        cards.append(("⚔️", "Goal Machine", row["Team"], f"{row['GF/Game']:.2f} GF/Game"))
    else:
        cards.append(("⚔️", "Goal Machine", "None", "—"))

    # --- 5. Fortress (Safe indexing) ---
    if not active_df.empty and "GA/Game" in active_df.columns and active_df["GA/Game"].notna().any():
        row = active_df.loc[active_df["GA/Game"].idxmin()]
        cards.append(("🛡️", "Fortress", row["Team"], f"{row['GA/Game']:.2f} GA/Game"))
    else:
        cards.append(("🛡️", "Fortress", "None", "—"))

    # --- 6. Top Seed Alive ---
    top_ratings = active_df.sort_values("Rating", ascending=False)
    if not top_ratings.empty:
        cards.append(("💪", "Top Seed Alive", top_ratings.iloc[0]["Team"], "Highest Rating"))
    else:
        cards.append(("💪", "Top Seed Alive", "None", "—"))

    # --- 7. Rising Power — redefined to be distinct from Dark Horse ---
    # Dark Horse measures Rating-vs-Elo (an underdog "quality" signal).
    # Rising Power instead measures current-form-vs-Elo (a "momentum/surge"
    # signal using PPG), so the two cards can never coincidentally point at
    # the same underlying math. Heavyweights are excluded from the pool
    # outright — Elo Mod >= 0.82 or Rating >= 0.80 — so a giant like France
    # can never qualify, regardless of how hot its current form looks.
    if not active_df.empty and "PPG" in active_df.columns and "Elo Mod" in active_df.columns:
        rp_pool = active_df[
            (active_df["Elo Mod"] < 0.82) & (active_df["Rating"] < 0.80)
        ].copy()
        if not rp_pool.empty and rp_pool["PPG"].notna().any():
            # PPG is typically on a 0-3 scale; normalize onto the same
            # 0-1-ish scale as Elo Mod so the gap is meaningful.
            rp_pool["form_gap"] = (rp_pool["PPG"] / 3.0) - rp_pool["Elo Mod"]
            idx = rp_pool["form_gap"].idxmax()
            row = rp_pool.loc[idx]
            cards.append(("📈", "Rising Power", row["Team"], f"+{row['form_gap']:.3f} form vs Elo"))
        else:
            cards.append(("📈", "Rising Power", "None", "—"))
    else:
        cards.append(("📈", "Rising Power", "None", "—"))

    cards.append(("🛡️", "Teams Left", f"{len(active_teams)}", "Active"))
    return cards


INSIGHT_CARD_LEGEND = [
    ("👹", "Battle-Tested", "Highest Strength-of-Schedule (SoS) among teams still alive in the tournament."),
    ("🚀", "Dark Horse", "Biggest positive gap between blended Rating and raw Elo Modifier - over-performing its reputation."),
    ("🏆", "Best Form", "Highest points-per-game among teams still alive."),
    ("⚔️", "Goal Machine", "Highest goals-for-per-game among teams still alive."),
    ("🛡️", "Fortress", "Lowest goals-against-per-game among teams still alive."),
    ("💪", "Top Seed Alive", "Highest blended Rating among teams still alive."),
    ("📈", "Rising Power", "Biggest positive gap between current form (PPG) and baseline Elo Modifier among non-heavyweight teams - a momentum surge, distinct from Dark Horse's underdog quality signal."),
]


def _render_insight_legend():
    with st.expander("ℹ️ What do these cards mean?", expanded=False):
        for emoji, label, desc in INSIGHT_CARD_LEGEND:
            st.markdown(
                f'<p style="margin:2px 0; font-size:.85em; color:#ccc;">'
                f'<strong>{emoji} {label}:</strong> {desc}</p>',
                unsafe_allow_html=True,
            )


def _render_insight_dashboard(team_stats_df: pd.DataFrame, active_teams: set[str],
                               eliminated_teams: set[str] = frozenset()):
    st.subheader("📊 Insight Dashboard")
    _render_insight_legend()
    cards = _build_insight_cards(team_stats_df, active_teams, eliminated_teams)
    with st.container(key="insight_dashboard"):
        st.markdown(                                                                                                                                                                                                      
            """
            <style>
            /* Compact metric cards - keep long team names readable, never truncated */
            .st-key-insight_dashboard [data-testid="stMetricValue"] {
                font-size: 1.05rem;
                line-height: 1.25;
                overflow-wrap: break-word;
                white-space: normal;
            }
            .st-key-insight_dashboard [data-testid="stMetricLabel"] {
                font-size: .82rem;
            }
            .st-key-insight_dashboard [data-testid="stMetricDelta"] {
                font-size: .78rem;
            }
            .st-key-insight_dashboard [data-testid="stMetric"] {
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 8px;
                padding: 8px 10px;
            }
            /* Mobile: let the 4-wide rows wrap into a 2x2 grid, then a single
               stacked column, instead of squeezing/scrolling horizontally. */
            @media (max-width: 768px) {
                .st-key-insight_dashboard [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap;
                    gap: 8px;
                }
                .st-key-insight_dashboard [data-testid="stColumn"] {
                    min-width: 46% !important;
                    flex: 1 1 46% !important;
                    width: 46% !important;
                }
            }
            @media (max-width: 420px) {
                .st-key-insight_dashboard [data-testid="stColumn"] {
                    min-width: 100% !important;
                    flex: 1 1 100% !important;
                    width: 100% !important;
                }
                .st-key-insight_dashboard [data-testid="stMetricValue"] {
                    font-size: .95rem;
                }
            }
            @media (max-width: 768px) {
                .st-key-insight_dashboard [data-testid="stMetricValue"] {
                    overflow-wrap: anywhere;
                    word-break: break-word;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for start in (0, 4):
            cols = st.columns(4)
            for col, (emoji, label, team, value) in zip(cols, cards[start:start + 4]):
                with col:
                    st.metric(
                        label=f"{emoji} {label}",
                        value=team if team else "—",
                        delta=value,
                        delta_color="off",
                    )
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)


def render_team_ratings_tab(api_key, team_stats_df, wc_fixtures_df=None):
    st.header("Team Ratings")
    if not api_key: return
    if team_stats_df.empty: return

    # 1. Fetch active teams (real-world: has a SCHEDULED/TIMED/IN_PLAY fixture)
    active_teams = compute_active_teams(wc_fixtures_df) if wc_fixtures_df is not None else set()
    
    # 2. Elimination Check — this must reflect IRL results, not depend on the
    # World Cup Simulator tab having been run. Primary signal: any known team
    # (from team_stats_df) with no live/upcoming real fixture is eliminated.
    # If a Monte Carlo run *is* available in session state, union in its
    # eliminations too (compute_eliminated_teams also flags teams that still
    # have a scheduled "dead rubber" fixture but are already mathematically
    # out). Guardrail: default to an empty frozenset() when there isn't enough
    # fixture data to determine anything, so this never crashes or
    # over-eliminates before the schedule has loaded.
    if active_teams and "Team" in team_stats_df.columns:
        all_known_teams = set(team_stats_df["Team"].dropna())
        eliminated_teams = all_known_teams - active_teams
    else:
        eliminated_teams = frozenset()
    
    if "wc_results" in st.session_state:
        eliminated_teams = eliminated_teams | compute_eliminated_teams(st.session_state["wc_results"], active_teams)
    
    st.session_state["wc_eliminated_teams"] = eliminated_teams
    
    # 3. Now render dashboard safely
    _render_insight_dashboard(team_stats_df, active_teams, eliminated_teams)
    
    # ... (Rest of your tab code: Top 10, etc.)

    st.subheader("Top 10 Teams by Rating")
    top10 = team_stats_df.head(10).sort_values("Rating")
    fig = px.bar(top10, x="Rating", y="Team", orientation="h", text="Rating", color="Rating", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=450, xaxis_title="Blended Rating (xP + SoS Goals + Elo)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Team Ratings")
    cols = ["Rank","Team","Games Played","Goals For","Goals Against","GF/Game","GA/Game","PPG","SoS","Elo Mod","Rating"]
    cols = [c for c in cols if c in team_stats_df.columns]
    st.dataframe(team_stats_df[cols], use_container_width=True, hide_index=True)


def render_match_predictor_tab(api_key, team_stats_df, model_bundle, wc_fixtures_df=None):
    st.header("Match Predictor")

    ml_badge = (
        "**ML-enhanced** - XGBoost calibration blended with Poisson (65/35)"
        if model_bundle else "**Poisson model** - install xgboost for ML enhancement"
    )
    st.caption(f"Predictions via blended xP/SoS/Elo ratings + Poisson + Monte Carlo. {ml_badge}. Squad conditions auto-loaded.")

    if not api_key:
        st.info("API Token connection missing from secrets management configuration.")
        return
    if team_stats_df.empty:
        st.info("No team data available yet.")
        return

    team_names = sorted(team_stats_df["Team"].unique().tolist())

    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", team_names, index=0, key="team_a_select")
    with c2:
        team_b = st.selectbox("Team B", team_names, index=min(1, len(team_names)-1), key="team_b_select")

    n_sims = st.slider("Monte Carlo simulations", 1000, 20000, 10000, 1000, key="mc_sims")

    st.markdown("### Squad Adjustment Factors - Auto-Loaded")
    has_a = f"saf_{team_a}" in st.session_state
    has_b = f"saf_{team_b}" in st.session_state

    _, _, load_col = st.columns([2, 2, 1])
    with load_col:
        load_btn = st.button("Load Conditions", type="primary", use_container_width=True)

    if load_btn:
        with st.spinner(f"Fetching {team_a}..."):
            st.session_state[f"saf_{team_a}"] = fetch_team_saf(team_name=team_a)
        with st.spinner(f"Fetching {team_b}..."):
            st.session_state[f"saf_{team_b}"] = fetch_team_saf(team_name=team_b)
        has_a = has_b = True

    if has_a and has_b:
        saf_data_a = st.session_state[f"saf_{team_a}"]
        saf_data_b = st.session_state[f"saf_{team_b}"]
        saf_a = saf_data_a.get("composite", 0.85)
        saf_b = saf_data_b.get("composite", 0.85)
        cc1, cc2 = st.columns(2)
        with cc1: render_saf_breakdown_card(team_name=team_a, saf_data=saf_data_a)
        with cc2: render_saf_breakdown_card(team_name=team_b, saf_data=saf_data_b)
    else:
        saf_a = saf_b = 0.85
        st.info("Click **Load Conditions** to auto-fetch squad data.")

    predict_clicked = st.button("Predict Match", type="primary", use_container_width=True)

    if predict_clicked:
        if team_a == team_b:
            st.warning("Please select two different teams.")
        else:
            ra = team_stats_df[team_stats_df["Team"] == team_a].iloc[0]
            rb = team_stats_df[team_stats_df["Team"] == team_b].iloc[0]
            lgf = team_stats_df["GF/Game"].mean()
            lga = team_stats_df["GA/Game"].mean()

            xg_home, xg_away = calculate_expected_goals(ra, rb, lgf, lga, saf_a=saf_a, saf_b=saf_b, team_a_name=team_a, team_b_name=team_b)
            mc = run_monte_carlo_simulation(xg_home, xg_away, n_sims=n_sims)

            rd = float(ra["Rating"]) - float(rb["Rating"])
            hw_b, dr_b, aw_b = blended_probs(xg_home, xg_away, rating_diff=rd, saf_diff=saf_a - saf_b, model_bundle=model_bundle)

            st.session_state["prediction"] = {
                "team_a": team_a, "team_b": team_b, "xg_home": xg_home, "xg_away": xg_away,
                "mc": mc, "n_sims": n_sims, "saf_a": saf_a, "saf_b": saf_b,
                "hw_blended": hw_b*100, "dr_blended": dr_b*100, "aw_blended": aw_b*100, "ml_active": model_bundle is not None,
            }

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        ta, tb = pred["team_a"], pred["team_b"]
        matrix = build_score_matrix(pred["xg_home"], pred["xg_away"])

        st.subheader("Expected Goals")
        ca, cb = st.columns(2)
        ca.metric(f"{ta} (Home)", f"{pred['xg_home']:.2f}", delta=f"SAF {pred['saf_a']:.3f}")
        cb.metric(f"{tb} (Away)", f"{pred['xg_away']:.2f}", delta=f"SAF {pred['saf_b']:.3f}")

        hw_p, dr_p, aw_p = summarize_outcomes(matrix)

        # Capture the complete projection state so the AI Tournament Analyst
        # chatbot can reason precisely about this exact head-to-head result.
        st.session_state["latest_head_to_head"] = {
            "team_a": ta,
            "team_b": tb,
            "xg_home": float(pred["xg_home"]),
            "xg_away": float(pred["xg_away"]),
            "saf_a": float(pred["saf_a"]),
            "saf_b": float(pred["saf_b"]),
            "poisson": {"home_win_pct": hw_p, "draw_pct": dr_p, "away_win_pct": aw_p},
            "blended": (
                {
                    "home_win_pct": pred["hw_blended"],
                    "draw_pct": pred["dr_blended"],
                    "away_win_pct": pred["aw_blended"],
                }
                if pred.get("ml_active") else None
            ),
            "monte_carlo": pred.get("mc"),
            "top_scorelines": top_scorelines(matrix, 10),
        }

        if pred.get("ml_active"):
            st.subheader("Match Outcome Probabilities")
            tab_poisson, tab_ml = st.tabs(["Poisson Model", "Poisson + XGBoost Blend (65/35)"])
            with tab_poisson:
                p1, p2, p3 = st.columns(3)
                p1.metric(f"{ta} Win", f"{hw_p:.1f}%")
                p2.metric("Draw",       f"{dr_p:.1f}%")
                p3.metric(f"{tb} Win",  f"{aw_p:.1f}%")
            with tab_ml:
                m1, m2, m3 = st.columns(3)
                m1.metric(f"{ta} Win", f"{pred['hw_blended']:.1f}%")
                m2.metric("Draw",       f"{pred['dr_blended']:.1f}%")
                m3.metric(f"{tb} Win",  f"{pred['aw_blended']:.1f}%")
        else:
            st.subheader("Match Outcome Probabilities (Poisson Model)")
            p1, p2, p3 = st.columns(3)
            p1.metric(f"{ta} Win", f"{hw_p:.1f}%")
            p2.metric("Draw",       f"{dr_p:.1f}%")
            p3.metric(f"{tb} Win",  f"{aw_p:.1f}%")

        st.subheader(f"Top 10 Most Likely Scorelines ({ta} vs {tb})")
        top10_df = pd.DataFrame(top_scorelines(matrix, 10), columns=["Score","Probability (%)"])
        top10_df["Probability (%)"] = (top10_df["Probability (%)"] * 100).round(2)
        st.dataframe(top10_df, use_container_width=True, hide_index=True)

        st.subheader("Score Probability Heatmap")
        hm = matrix * 100
        fig = go.Figure(data=go.Heatmap(
            z=hm, x=[str(i) for i in range(matrix.shape[1])], y=[str(i) for i in range(matrix.shape[0])],
            colorscale="YlOrRd", text=np.round(hm, 1), texttemplate="%{text}%",
        ))
        fig.update_layout(xaxis_title=f"{tb} Goals", yaxis_title=f"{ta} Goals", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # GOAL MARGIN DISTRIBUTION
        st.subheader("Goal Margin Distribution (First 90 Minutes)")
        st.caption("Projected structural variance of the match-up, grouped by final goal margin.")

        margin_labels = [f"{ta} by 2+", f"{ta} by 1", "Draw", f"{tb} by 1", f"{tb} by 2+"]
        margin_probs = [0.0, 0.0, 0.0, 0.0, 0.0]
        for hg in range(matrix.shape[0]):
            for ag in range(matrix.shape[1]):
                diff = hg - ag
                p = matrix[hg, ag]
                if diff >= 2:
                    margin_probs[0] += p
                elif diff == 1:
                    margin_probs[1] += p
                elif diff == 0:
                    margin_probs[2] += p
                elif diff == -1:
                    margin_probs[3] += p
                else:
                    margin_probs[4] += p
        margin_probs = [p * 100 for p in margin_probs]

        margin_colors = ['#1a5276', '#3498db', '#f1c40f', '#e67e22', '#922b21']
        fig_margin = go.Figure(go.Bar(
            x=margin_labels,
            y=margin_probs,
            marker_color=margin_colors,
            text=[f"{p:.1f}%" for p in margin_probs],
            textposition="outside",
        ))
        fig_margin.update_layout(
            xaxis_title="Projected Goal Margin",
            yaxis_title="Probability (%)",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_margin, use_container_width=True)


def _team_is_active(team, active_teams: set[str], row: pd.Series) -> bool:
    """Shared 'is this team still alive' rule used by both the elimination-status
    row styling and the Battle-Tested insight card: alive if it has a live/
    upcoming fixture, OR still shows nonzero odds in the tournament simulation."""
    has_live_fixture = bool(active_teams) and team in active_teams
    stage_cols = ("Champion %", "Final %", "Semifinal %")
    has_sim_life = any(
        pd.notna(row.get(col)) and float(row.get(col)) > 0
        for col in stage_cols if col in row.index
    )
    return has_live_fixture or has_sim_life


def _style_elimination_status(row: pd.Series, active_teams: set[str]) -> list[str]:
    team = row.get("Team")
    is_active = _team_is_active(team, active_teams, row)
    style = (
        "background-color: rgba(46, 204, 113, 0.30); color: #eafaf1; font-weight:600;"
        if is_active else
        "background-color: rgba(231, 76, 60, 0.28); color: #fdecea; font-weight:600;"
    )
    return [style if col == "Team" else "" for col in row.index]


def compute_eliminated_teams(results_df: pd.DataFrame, active_teams: set[str]) -> set[str]:
    """Teams explicitly eliminated per the same rule as the elimination-status
    row styling above. Cached to session_state by the World Cup Simulator tab so
    the Team Ratings tab's 'Battle-Tested' card can exclude eliminated teams too."""
    if results_df is None or results_df.empty or "Team" not in results_df.columns:
        return set()
    eliminated = set()
    for _, row in results_df.iterrows():
        team = row.get("Team")
        if isinstance(team, str) and team.strip() and not _team_is_active(team, active_teams, row):
            eliminated.add(team)
    return eliminated


def render_world_cup_tab(api_key, model_bundle):
    st.header("World Cup Tournament Simulator")

    ml_note = "ML-enhanced knockout decisions (XGBoost blend)" if model_bundle else "Poisson-only knockout decisions"
    st.caption(f"Seeded Round of 32 Tournament Bracket Engine. {ml_note}. Dynamic Heatmap Mapping.")

    n_tournaments = st.slider("Tournaments to simulate", 100, 10000, 1000, 100, key="wc_n_sims")
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_clicked:
        if not api_key:
            st.warning("API Token missing from cloud secrets management configuration.")
        else:
            with st.spinner("Loading World Cup data..."):
                wc_stats    = get_team_stats(api_key, WORLD_CUP_LEAGUE_ID, WORLD_CUP_SEASON)
                wc_groups   = get_world_cup_groups(api_key, WORLD_CUP_SEASON, WORLD_CUP_LEAGUE_ID)
                wc_fixtures = get_world_cup_matches(api_key, WORLD_CUP_SEASON, WORLD_CUP_LEAGUE_ID)

            if not wc_groups:
                if not wc_fixtures.empty:
                    all_t = sorted({t for t in set(wc_fixtures["Home"]) | set(wc_fixtures["Away"]) if isinstance(t, str) and t.strip()})
                else:
                    all_t = []
                wc_groups = {f"Group {idx+1}": all_t[s:s+4] for idx, s in enumerate(range(0, len(all_t), 4)) if all_t[s:s+4]}

            ratings_lookup = {row["Team"]: {"GF/Game": row["GF/Game"], "GA/Game": row["GA/Game"], "Rating": row["Rating"]} for _, row in wc_stats.iterrows()} if not wc_stats.empty else {}
            saf_lookup    = {t: 0.85 for teams in wc_groups.values() for t in teams}
            league_avg_gf = wc_stats["GF/Game"].mean() if not wc_stats.empty else 1.3
            league_avg_ga = wc_stats["GA/Game"].mean() if not wc_stats.empty else 1.3
            fixed_results = completed_results_lookup(wc_fixtures)

            results_df = run_world_cup_simulation(
                wc_groups,
                ratings_lookup,
                league_avg_gf,
                league_avg_ga,
                n_sims=int(n_tournaments),
                saf_lookup=saf_lookup,
                model_bundle=model_bundle,
                fixed_results=fixed_results,
            )
            
            # --- MANDATORY STATE SAVES ---
            st.session_state["wc_results"]     = results_df
            st.session_state["wc_groups_used"] = wc_groups
            st.session_state["wc_fixtures"]    = wc_fixtures
            st.session_state["wc_ratings_lookup"] = ratings_lookup
            st.session_state["wc_saf_lookup"] = saf_lookup
            st.session_state["wc_league_avg_gf"] = league_avg_gf
            st.session_state["wc_league_avg_ga"] = league_avg_ga
            st.session_state["wc_fixed_results"] = fixed_results
            
            # IMMEDIATELY calculate and save eliminations after simulation
            active_teams = compute_active_teams(wc_fixtures)
            st.session_state["wc_eliminated_teams"] = compute_eliminated_teams(results_df, active_teams)
            st.session_state["wc_results"]     = results_df
            st.rerun() # Refresh to update dashboard immediately

    if "wc_results" in st.session_state:
        df  = st.session_state["wc_results"]
        grp = st.session_state.get("wc_groups_used", {})
        fx  = st.session_state.get("wc_fixtures", pd.DataFrame())
        active_teams = compute_active_teams(fx)
        
        st.subheader("Championship Probabilities")
        format_dict = {"Champion %": "{:.1f}%", "Final %": "{:.1f}%", "Semifinal %": "{:.1f}%"}
        styled_df = (
            df.style
              .format(format_dict)
              .background_gradient(cmap="YlGnBu", vmin=0, vmax=100)
              .apply(_style_elimination_status, axis=1, active_teams=active_teams)
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        if active_teams:
            st.caption("Color depth reflects relative probability distribution per stage. Team column: green = still alive, red = eliminated.")
        else:
            st.caption("Color depth reflects relative probability distribution per stage across all runs.")

        st.subheader("Most Probable Tournament Trajectory")
        ratings_lookup = st.session_state.get("wc_ratings_lookup", {})
        saf_lookup = st.session_state.get("wc_saf_lookup", {})
        league_avg_gf = st.session_state.get("wc_league_avg_gf", 1.3)
        league_avg_ga = st.session_state.get("wc_league_avg_ga", 1.3)
        fixed_results = st.session_state.get("wc_fixed_results", completed_results_lookup(fx))
        render_horizontal_bracket(
            build_simulated_bracket_state(
                ratings_lookup,
                league_avg_gf,
                league_avg_ga,
                saf_lookup=saf_lookup,
                model_bundle=model_bundle,
                fixed_results=fixed_results,
            )
        )
        st.markdown("---")

        st.subheader("Top 15 Title Contenders")
        top15 = df.head(15).sort_values("Champion %")
        fig   = px.bar(top15, x="Champion %", y="Team", orientation="h", text="Champion %", color="Champion %", color_continuous_scale="Viridis")
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=500, xaxis_title="Championship Probability (%)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Stage Reach Rates - Top 10")
        top10_s = df.head(10)[["Team","Champion %","Final %","Semifinal %"]]
        fig2 = go.Figure()
        for col, color in [("Semifinal %","#3498db"),("Final %","#e67e22"),("Champion %","#2ecc71")]:
            fig2.add_trace(go.Bar(name=col.replace(" %",""), x=top10_s["Team"], y=top10_s[col], marker_color=color))
        fig2.update_layout(barmode="group", height=400, yaxis_title="Probability (%)", xaxis_title="")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Tournament Summary")
        fav = df.iloc[0] if not df.empty else None
        s1, s2, s3 = st.columns(3)
        s1.metric("Teams Simulated", len(df))
        s2.metric("Groups", len(grp))
        if fav is not None:
            s3.metric("Favorite", fav["Team"], f"{fav['Champion %']:.1f}% to win")
    else:
        st.info("Click **Run Simulation** to simulate the tournament.")


# ============================================================================
# 9b. MAIN BODY FRAGMENT (nav + routed tab content)
# ============================================================================  
# The nav bar and the tab it renders live inside one st.fragment. The 'Go →'
# buttons in the About tab (via update_nav's on_click) and the radio nav both
# only mutate state that this fragment reads, so Streamlit reruns *just* this
# fragment instead of the whole page - no full-page rerun means no scroll jump.
@st.fragment
def render_app_body(api_key, live_df, fixtures_df, results_df, wc_fixtures_df,
                     team_stats_df, model_bundle):
    active_tab = render_tab_nav()
    if active_tab == TAB_LABELS[0]:
        render_about_tab()
    elif active_tab == TAB_LABELS[1]:
        render_live_fixtures_tab(api_key, live_df, fixtures_df, results_df, wc_fixtures_df)
    elif active_tab == TAB_LABELS[2]:
        render_team_ratings_tab(api_key, team_stats_df, wc_fixtures_df)
    elif active_tab == TAB_LABELS[3]:
        render_match_predictor_tab(api_key, team_stats_df, model_bundle, wc_fixtures_df)
    elif active_tab == TAB_LABELS[4]:
        render_world_cup_tab(api_key, model_bundle)


import streamlit.components.v1 as components

def render_gemini_chatbot():
    """Renders a Gemini-powered AI Tournament Analyst chatbot aware of internal application states."""
    try:
        import google.generativeai as genai
    except ImportError:
        st.warning("Please install `google-generativeai` to use the AI Analyst feature.")
        return
    with st.sidebar:
        st.markdown("---")
        st.subheader("💬 AI Tournament Analyst")

        api_key = None
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]

        if not api_key:
            api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter free Gemini API Key to enable the AI Analyst.")

        if not api_key:
            st.info("To chat with the AI Analyst, provide a Gemini API Key via `st.secrets` or the sidebar.")
            return

        genai.configure(api_key=api_key)

        # ── Check for simulation state ──────────────────────────────────────
        wc_results = st.session_state.get("wc_results")
        has_simulated = wc_results is not None and not wc_results.empty

        match_data = st.session_state.get("prediction")

        if not match_data:
            context_instruction = (
                "The user has not run the match predictor yet. "
                "Politely inform them that you are ready to provide tactical breakdowns, "
                "but you need them to run a match prediction first in the Match Predictor tab."
            )
        else:
            context_instruction = (
                f"The most recent match prediction was: {match_data}. "
                "Use these exact figures (Expected Goals, Poisson & Blended percentages) "
                "to provide detailed tactical and statistical insights."
            )

        sys_instruction = (
            "You are Tactico, elite football analyst. Chatty, insightful, evidence-based. "
            "RULES:\n"
            "1. ONLY football talk. Ignore non-football. Never invent facts.\n"
            "2. NO wc_results? NO tournament predictions. Tell user: 'Run simulator first.'\n"
            "3. MISSING H2H data? Don't say 'no data'. Say: 'Teams not in bracket. Run Match Predictor tab for H2H breakdown.'\n"
            "4. ANALYSIS: Always distinguish between 'Tournament Aggregate' (simulated) and 'Match Predictor' (single-match) data.\n"
            "5. TOPICS: Tactics, history, rankings, models, players.\n"
            "6. Keep answers concise — you have a limited output budget, so prioritize the most useful insight first.\n"
            + f"\n\nCONTEXT: {context_instruction}"
        )

        if has_simulated:
            full_results_text = wc_results.to_string()
            sys_instruction += f"\n\nCURRENT SIMULATION RESULTS (ALL TEAMS):\n{full_results_text}"
        else:
            sys_instruction += "\n\nNO TOURNAMENT SIMULATION DATA CURRENTLY AVAILABLE."

        def _format_squad_block(team_name: str, saf_data: dict, max_players: int = 30) -> str:
            players = (saf_data or {}).get("_players", [])
            if not players: return ""
            lines = [f"{p.get('player', 'Unknown')} ({p.get('club') or 'club unlisted'})" for p in players[:max_players]]
            return f"{team_name} squad: " + "; ".join(lines)

        team_a_sel = st.session_state.get("team_a_select")
        team_b_sel = st.session_state.get("team_b_select")
        saf_a_data = st.session_state.get(f"saf_{team_a_sel}") if team_a_sel else None
        saf_b_data = st.session_state.get(f"saf_{team_b_sel}") if team_b_sel else None

        if saf_a_data: sys_instruction += "\nSQUAD A: " + _format_squad_block(team_a_sel, saf_a_data)
        if saf_b_data: sys_instruction += "\nSQUAD B: " + _format_squad_block(team_b_sel, saf_b_data)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        if not st.session_state["chat_history"]:
            welcome_msg = "👋 Hi! I'm Tactico. Ask me anything about the simulation logic, team chances, or tactical bottlenecks once you've run the tournament simulation!"
            st.session_state["chat_history"].append({"role": "assistant", "content": welcome_msg})

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # ── Cooldown Rate Limiting Logic (non-blocking) ─────────────────────
        # No server-side sleep loop. Remaining time is derived purely from a
        # timestamp diff, and the visible countdown + the single rerun-on-
        # expiry are both handled client-side via a lightweight JS timer, so
        # the Python process never blocks and never polls in a loop.
        # ── Cooldown Rate Limiting Logic (non-blocking, sidebar-only rerun) ──
        # No server-side sleep loop, and no browser-level page reload either.
        # A hidden st.button lives in the sidebar; a small JS timer decrements
        # a visible countdown client-side and, only once, "clicks" that hidden
        # button when the timer hits zero. A button click is a normal Streamlit
        # interaction — it triggers the same lightweight script rerun as any
        # other widget, without a hard browser reload or full-page flash.
        last_time = st.session_state.get("last_chat_time", 0.0)
        cooldown_duration = 60.0
        elapsed = time.time() - last_time
        remaining = cooldown_duration - elapsed

        if remaining > 0:
            seconds_left = int(remaining) + 1
            AUTO_RERUN_MARKER = "tactico_cooldown_expired_marker"

            st.markdown(
                f"""
                <div style="font-size:0.85em;color:#f39c12;font-family:inherit;padding:2px 0;">
                    ⏳ Tactico is recharging — <span id="tactico-timer">{seconds_left}</span>s until your next report.
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Hidden trigger button — a real Streamlit widget, just visually
            # hidden. Clicking it (via JS below) causes a normal script rerun.
            st.button(AUTO_RERUN_MARKER, key="tactico_cooldown_rerun_btn")

            components.html(
                f"""
                <script>
                (function() {{
                    // Hide the trigger button (it lives one level up, in the
                    // real Streamlit DOM, not inside this component's iframe).
                    const doc = window.parent.document;
                    const buttons = doc.querySelectorAll('button');
                    let triggerBtn = null;
                    buttons.forEach(function(b) {{
                        if (b.innerText.trim() === "{AUTO_RERUN_MARKER}") {{
                            b.closest('div[data-testid="stButton"]').style.display = 'none';
                            triggerBtn = b;
                        }}
                    }});

                    let remaining = {seconds_left};
                    const el = doc.getElementById("tactico-timer");
                    const interval = setInterval(function() {{
                        remaining -= 1;
                        if (remaining <= 0) {{
                            clearInterval(interval);
                            if (triggerBtn) {{
                                triggerBtn.click();  // normal rerun, no page reload
                            }}
                        }} else if (el) {{
                            el.textContent = remaining;
                        }}
                    }}, 1000);
                }})();
                </script>
                """,
                height=0,
            )
            disabled_input = True
        else:
            disabled_input = False
        # ── Client-side input length guard ──────────────────────────────────
        # Rough token approximation: ~4 characters per token for English text.
        MAX_INPUT_CHARS = 4000  # ~1000-token operational ceiling
        APPROX_CHARS_PER_TOKEN = 4

        if prompt := st.chat_input("Tactico is ready. Ask me anything about the simulation...", disabled=disabled_input):
            approx_tokens = len(prompt) // APPROX_CHARS_PER_TOKEN
            if len(prompt) > MAX_INPUT_CHARS:
                st.warning(
                    f"⚠️ That message is too long (~{approx_tokens} tokens estimated, "
                    f"limit ~{MAX_INPUT_CHARS // APPROX_CHARS_PER_TOKEN} tokens). "
                    "Please shorten it — Tactico won't process oversized prompts to protect the model context."
                )
            else:
                st.session_state["chat_history"].append({"role": "user", "content": prompt})
                st.session_state["last_chat_time"] = time.time()
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # Sequential model array fallback configuration for robust quota
                    # management. gemini-2.0-flash has been removed — it was shut
                    # down by Google on June 1, 2026 and would 404 every time.
                    models_to_try = [
                        "gemini-3.5-flash",
                        "gemini-3.1-flash-lite",
                        "gemini-3-flash-preview",
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "gemini-2.5-pro",
                    ]
                    response_content = None
                    generation_config = genai.GenerationConfig(max_output_tokens=600)

                    for model_name in models_to_try:
                        try:
                            model = genai.GenerativeModel(
                                model_name,
                                system_instruction=sys_instruction,
                                generation_config=generation_config,
                            )
                            chat = model.start_chat(history=[{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in st.session_state["chat_history"][:-1]])
                            response = chat.send_message(prompt)
                            response_content = response.text
                            break  # Break out if successful execution is achieved
                        except Exception as exc:
                            if "429" in str(exc) or "ResourceExhausted" in str(exc):
                                continue
                            else:
                                st.error(f"Error communicating with Gemini API ({model_name}): {exc}")
                                break

                    if response_content:
                        st.markdown(response_content)
                        st.session_state["chat_history"].append({"role": "assistant", "content": response_content})
                        st.rerun()
                    else:
                        st.error("All available Gemini endpoints are currently rate-limited. Please wait a short moment and try again.")

# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="World Cup Predictor 2026", page_icon="⚽", layout="wide")
    _inject_global_css()

    if st.sidebar.button("Hard Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    model_bundle = load_ml_model() if ML_AVAILABLE else None
    api_key, league_id, season, fixtures_count = render_sidebar()

    if api_key:
        fixtures_df    = get_fixtures(api_key, league_id, season, next_n=fixtures_count)
        results_df     = get_results(api_key, league_id, season, last_n=fixtures_count)
        live_df        = get_live_fixtures(api_key, league_id=league_id)
        team_stats_df  = get_team_stats(api_key, league_id, season)
        wc_fixtures_df = get_world_cup_matches(api_key, season, league_id)
    else:
        fixtures_df = results_df = live_df = team_stats_df = wc_fixtures_df = pd.DataFrame()

    render_live_ticker(live_df, results_df)

    render_app_body(api_key, live_df, fixtures_df, results_df, wc_fixtures_df,
                     team_stats_df, model_bundle)

    render_gemini_chatbot()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #777; font-size: 0.85em; line-height: 1.6;'>
        <strong> Project Credits:</strong> Engineered and maintained by <strong>Ayush Kamath</strong>. Built as an advanced predictive framework pairing stochastic simulation with machine learning calibration.<br>
        <strong> Legal Disclaimer:</strong> This application is an open-source statistical forecasting simulation. All projections, odds, and match calculations are automated estimations for informational and entertainment purposes only. This platform does not accept, facilitate, or encourage real-money wagering or sports betting.<br>
        <strong> Data Credits & Attribution:</strong> Real-time match data and fixture lists are powered by the <a href="https://www.football-data.org/" target="_blank" style="color: #3498db;">Football-Data.org API</a>. Baseline international ratings are compiled from public soccer Elo indices. Squad attributes, injuries, and form tracking are parsed from open archival indexes on FBref and Transfermarkt.<br>
        <strong> Privacy Notice:</strong> This platform does not deploy tracking cookies, harvest personal data, or store user information. Chatbot interactions are ephemeral and processed securely via the Google Gemini API framework.<br>
        <br>
        <span style='font-size: 0.95em; font-weight: 500;'>© 2026 Ayush Kamath. All Rights Reserved.</span><br>
        <span style='font-size: 0.9em; font-style: italic;'>Not affiliated with, endorsed by, or connected to FIFA, the 2026 World Cup Organizing Committee, or any official football federation.</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
