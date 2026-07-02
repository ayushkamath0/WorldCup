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

# ─── Expert-level team rating architecture ──────────────────────────────────
RATING_BLEND_WEIGHTS = {"xp": 0.40, "sos_goals": 0.20, "elo": 0.40}
GOAL_PERFORMANCE_EXPONENT = 1.7
TIME_DECAY_RATE = 1.5

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
# 3. ML MODEL — TRAIN / LOAD
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

def api_request(endpoint: str, api_key: str, params: dict = None) -> dict:
    if not api_key:
        return {}
    url     = f"{API_BASE_URL}/{endpoint}"
    headers = {"X-Auth-Token": api_key}
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return {}
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to Football-Data.org.")
        return {}
    except requests.exceptions.RequestException as exc:
        st.error(f"Network error: {exc}")
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


def _safe_score(score_dict: dict, side: str) -> str:
    if not score_dict or "fullTime" not in score_dict:
        return "-"
    v = score_dict["fullTime"].get(side)
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
        ft    = score.get("fullTime")   or {}
        rows.append({
            "Match ID":   item.get("id"),
            "Date":       (item.get("utcDate") or "")[:10],
            "League":     comp.get("name", ""),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Home":       ht.get("name", ""),
            "Score":      f"{_safe_score(score,'home')} - {_safe_score(score,'away')}",
            "Away":       at.get("name", ""),
            "Home Goals": ft.get("home"),
            "Away Goals": ft.get("away"),
            "Winner":     score.get("winner"),
            "Status":     item.get("status", "FINISHED"),
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
        ft    = score.get("fullTime")   or {}
        rows.append({
            "Match ID":   item.get("id"),
            "League":     comp.get("name", ""),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Minute":     "LIVE",
            "Home":       ht.get("name", ""),
            "Away":       at.get("name", ""),
            "Score":      f"{_safe_score(score,'home')} - {_safe_score(score,'away')}",
            "Home Goals": ft.get("home"),
            "Away Goals": ft.get("away"),
            "Winner":     score.get("winner"),
            "Status":     item.get("status", "IN_PLAY"),
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

        sc = item.get("score") or {};  ft = sc.get("fullTime") or {}
        hg, ag = ft.get("home"), ft.get("away")
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

            opp_elo = _lookup_elo_rating(m["opp"], rankings)
            if opp_elo is None:
                opp_elo = global_avg_elo if global_avg_elo is not None else 1500.0

            strength_mult = 1.0 + math.log(opp_elo / global_avg_elo) if global_avg_elo else 1.0
            strength_mult = max(min(strength_mult, 1.5), 0.75)

            weighted_opp_elo   += w * opp_elo
            opp_elo_weight_sum += w

            sos_weighted_gf += w * gf_eff * strength_mult
            sos_weighted_ga += w * (ga_eff / strength_mult)

            running_group_pts += match_pts

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
        sc = item.get("score")    or {}; ft = sc.get("fullTime")   or {}
        rows.append({
            "Match ID":   item.get("id"),
            "Date":       (item.get("utcDate") or "")[:16].replace("T", " "),
            "Round":      str(item.get("matchday") or ""),
            "Match Ref":  item.get("matchday") or item.get("id"),
            "Stage":      item.get("stage", "") or "",
            "Home":       ht.get("name", ""),
            "Away":       at.get("name", ""),
            "Home Goals": ft.get("home"),
            "Away Goals": ft.get("away"),
            "Winner":     sc.get("winner"),
            "Status":     item.get("status", ""),
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


@st.cache_data(ttl=3600, show_spinner=False)
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
    composite = saf_data.get("composite", 0.80)
    source_map = {
        "tactical":   ("", "Tactical Fit", "FBref - formation & lineup stability"),
        "form":       ("", "Club Form", "FBref - last-10 match xG & results"),
        "fitness":    ("", "Fitness / Injuries", "Transfermarkt - injury availability"),
        "league":     ("", "League Strength", "Club Elo - squad-club Elo ratings"),
        "experience": ("", "Intl. Experience", "Transfermarkt - avg squad caps"),
    }
    st.markdown(
        f"**{team_name}** &nbsp;&nbsp;"
        f"<span style='background:#1f4e79;color:white;padding:2px 8px;"
        f"border-radius:4px;font-weight:bold;'>SAF {composite:.3f}</span>",
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
                              saf_a: float = 0.85, saf_b: float = 0.85):
    lgf = league_avg_gf or 1.3;  lga = league_avg_ga or 1.3
    ma  = saf_a / 0.85;          mb  = saf_b / 0.85
    a_atk = (team_a_stats["GF/Game"] * ma) / lgf
    a_def = (team_a_stats["GA/Game"] / ma) / lga
    b_atk = (team_b_stats["GF/Game"] * mb) / lgf
    b_def = (team_b_stats["GA/Game"] / mb) / lga
    xg_h = float(np.clip(a_atk * b_def * lgf * home_advantage,       0.1, 6.0))
    xg_a = float(np.clip(b_atk * a_def * lga * (2 - home_advantage), 0.1, 6.0))
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

    xg_a, xg_b = calculate_expected_goals(sa, sb, lgf, lga, home_advantage=1.0, saf_a=saf_a, saf_b=saf_b)

    rd = sa.get("Rating", 0.5) - sb.get("Rating", 0.5)
    hw, dr, aw = blended_probs(xg_a, xg_b, rating_diff=rd, saf_diff=saf_a - saf_b, model_bundle=model_bundle)

    r = np.random.random()
    if r < hw:
        return team_a
    if r < hw + dr:
        ra_ = sa.get("Rating", 0.5) * (saf_a / 0.85)
        rb_ = sb.get("Rating", 0.5) * (saf_b / 0.85)
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
    every    = max(1, n_sims // 100)

    for sim in range(n_sims):
        champion, finalists, semis = simulate_knockout_bracket_locked(
            bracket, ratings_lookup, league_avg_gf, league_avg_ga,
            saf_lookup, model_bundle, fixed_results=fixed_results
        )

        if champion: champion_count[champion] += 1
        for t in finalists: final_count[t]    += 1
        for t in semis:     semi_count[t]      += 1

        if sim % every == 0:
            progress.progress(min(sim / n_sims, 1.0), text=f"Simulating... {sim:,}/{n_sims:,}")

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
        min-width: 1320px;
        display: grid;
        grid-template-columns: 1.12fr 1.02fr .92fr .86fr 1.08fr .86fr .92fr 1.02fr 1.12fr;
        column-gap: 14px;
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
        padding: 8px 9px;
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
        grid-template-columns: 22px minmax(0, 1fr) 58px;
        align-items: center;
        gap: 6px;
        min-height: 27px;
        padding: 4px 0;
        color: #e7e7e7;
        border-top: 1px solid #2b2b2b;
    }
    .bracket-team-row:first-of-type { border-top: 0; }
    .team-flag { text-align: center; font-size: 15px; }
    .team-name {
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: 12px;
        font-weight: 600;
    }
    .team-score {
        justify-self: end;
        min-width: 54px;
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
        font-family: "Courier New", monospace;
        letter-spacing: .05em;
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
        font-size: 20px;
        line-height: 1.2;
        font-weight: 900;
        overflow-wrap: anywhere;
    }
    .center-core .matchup-card::before,
    .center-core .matchup-card::after { display: none; }
    @media (max-width: 900px) {
        .wc-bracket-shell { padding: 12px; }
        .wc-bracket-grid { min-width: 1180px; column-gap: 10px; }
        .team-name { font-size: 11px; }
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
    hg = row.get("Home Goals")
    ag = row.get("Away Goals")
    if pd.notna(hg) and pd.notna(ag):
        try:
            return str(int(hg)), str(int(ag))
        except Exception:
            return str(hg), str(ag)
    score = row.get("Score")
    if isinstance(score, str) and "-" in score:
        parts = re.split(r"\s*[-:]\s*", score.strip())
        if len(parts) >= 2:
            return parts[0], parts[1]
    return "", ""


def _fixture_winner(row: pd.Series) -> str | None:
    if str(row.get("Status", "")).upper() != "FINISHED":
        return None
    api_winner = str(row.get("Winner", "") or "").upper()
    if api_winner == "HOME_TEAM":
        return row.get("Home")
    if api_winner == "AWAY_TEAM":
        return row.get("Away")
    try:
        hg = float(row.get("Home Goals"))
        ag = float(row.get("Away Goals"))
    except Exception:
        return None
    if hg > ag:
        return row.get("Home")
    if ag > hg:
        return row.get("Away")
    return None


def _team_key(name) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "", str(name or "").lower())
    if cleaned in ("congodr", "drcongo", "democraticrepublicofcongo"):
        return "drcongo"
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


def build_live_bracket_state(wc_fixtures_df: pd.DataFrame) -> dict:
    fixture_by_ref: dict[int, pd.Series] = {}
    if wc_fixtures_df is not None and not wc_fixtures_df.empty:
        for _, row in wc_fixtures_df.iterrows():
            ref = _fixture_match_ref(row)
            if ref not in LEFT_R32_MATCH_REFS + RIGHT_R32_MATCH_REFS:
                ref = _default_ref_from_pair(row.get("Home"), row.get("Away"))
            if ref is not None:
                fixture_by_ref[ref] = row

    left_r32, right_r32 = [], []
    left_winners, right_winners = [], []

    for ref in LEFT_R32_MATCH_REFS:
        card, winner, _ = _live_r32_card(ref, fixture_by_ref)
        left_r32.append(card)
        left_winners.append(winner)
    for ref in RIGHT_R32_MATCH_REFS:
        card, winner, _ = _live_r32_card(ref, fixture_by_ref)
        right_r32.append(card)
        right_winners.append(winner)

    def build_round(prev_winners: list[str | None], title: str) -> tuple[list[dict], list[str | None], list[str | None]]:
        cards, winners, losers = [], [], []
        for idx in range(0, len(prev_winners), 2):
            a = prev_winners[idx] if idx < len(prev_winners) else None
            b = prev_winners[idx + 1] if idx + 1 < len(prev_winners) else None
            card, winner, loser = _progression_card(title, a, b)
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
    
    final_card, final_winner, _ = _progression_card("Final", left_sf_winners[0], right_sf_winners[0])
    third_card, _, _ = _progression_card("Third Place", left_sf_losers[0], right_sf_losers[0])

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
    xg_a, xg_b = calculate_expected_goals(sa, sb, lgf, lga, home_advantage=1.0, saf_a=saf_a, saf_b=saf_b)
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
        known = fixed_results.get(_pair_key(team_a, team_b))
        if known:
            winner = known["winner"]
            loser = team_b if winner == team_a else team_a
            win_pct, lose_pct = 100.0, 0.0
            score_a = known.get("home_score", "") if known.get("home") == team_a else known.get("away_score", "")
            score_b = known.get("away_score", "") if known.get("home") == team_a else known.get("home_score", "")
        else:
            calc_winner, calc_win_pct, calc_loser, calc_lose_pct = _pair_probability(
                team_a, team_b, ratings_lookup, lgf, lga, saf_lookup, model_bundle
            )
            
            prob_a = calc_win_pct if team_a == calc_winner else calc_lose_pct
            prob_b = calc_lose_pct if team_a == calc_winner else calc_win_pct
            
            if is_r32 or not mc_probs:
                winner = calc_winner
                loser = calc_loser
            else:
                metric_a = get_advancement_metric(team_a)
                metric_b = get_advancement_metric(team_b)
                winner = team_a if metric_a >= metric_b else team_b
                loser = team_b if winner == team_a else team_a

            score_a = f"{prob_a:.0f}%"
            score_b = f"{prob_b:.0f}%"
            win_pct = prob_a if team_a == winner else prob_b
            lose_pct = prob_b if team_a == winner else prob_a

        card = _match_card(
            title,
            _entry(team_a, score_a, team_a == winner),
            _entry(team_b, score_b, team_b == winner),
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


# ============================================================================
# 9. STREAMLIT UI
# ============================================================================

def _ticker_items_from_data(live_df: pd.DataFrame, results_df: pd.DataFrame, max_items: int = 12) -> list[str]:
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
            score = row.get("Score", "- -")
            winner = _fixture_winner(row)
            
            ticker_str = f"⚽ {home} {score} {away}"
            
            if winner and rankings:
                home_elo = _lookup_elo_rating(home, rankings) or 1500
                away_elo = _lookup_elo_rating(away, rankings) or 1500
                
                favorite = home if home_elo >= away_elo else away
                underdog = away if home_elo >= away_elo else home
                
                if winner == favorite:
                    ticker_str = f'<span style="color:#2ecc71;">{ticker_str}</span>'
                elif winner == underdog:
                    ticker_str = f'<span style="color:#e74c3c;">{ticker_str}</span>'

            items.append(ticker_str)

    return items[:max_items]


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
    <div class="wc-ticker-wrapper" style="width:100%; overflow:hidden; background-color:#0e1117; border:1px solid {accent}; border-radius:6px; padding:8px 0; margin:6px 0 18px 0;">
        <div class="wc-ticker-track" style="display:inline-block; white-space:nowrap; padding-left:100%; animation:wc-ticker-scroll 60s linear infinite; color:#e8e8e8; font-size:14px; font-weight:500; font-family:'Courier New', monospace;">
            {ticker_text}
        </div>
    </div>
    <style>
    @keyframes wc-ticker-scroll {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-100%); }} }}
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


def compute_active_teams(fixtures_df: pd.DataFrame) -> set[str]:
    """A team is 'alive' if it appears in any upcoming/in-play match."""
    if fixtures_df is None or fixtures_df.empty or "Status" not in fixtures_df.columns:
        return set()
    upcoming = fixtures_df[fixtures_df["Status"].isin(ACTIVE_FIXTURE_STATUSES)]
    teams = set(upcoming.get("Home", pd.Series(dtype=str)).tolist()) | \
            set(upcoming.get("Away", pd.Series(dtype=str)).tolist())
    return {t for t in teams if _is_confirmed_team(t)}


def render_live_knockout_bracket(wc_fixtures_df: pd.DataFrame):
    """Displays the live knockout tree from confirmed API match refs 73-88."""
    st.subheader("Live Knockout Bracket")
    if wc_fixtures_df is None or wc_fixtures_df.empty:
        st.info("Knockout bracket data is not available yet; placeholder slots are shown below.")
        render_horizontal_bracket(build_live_bracket_state(pd.DataFrame()))
        return
    render_horizontal_bracket(build_live_bracket_state(wc_fixtures_df))


# Refactored: Visual settings bar fully removed out of existence. Secrets are resolved securely and defaults are pinned.
def render_sidebar():
    api_key = st.secrets.get("FOOTBALL_API_KEY", "")
    return api_key, WORLD_CUP_LEAGUE_ID, WORLD_CUP_SEASON, 30


def render_about_tab():
    st.markdown(
        """
        <div style="background-color:#1e293b; padding:24px; border-radius:10px; border-left:6px solid #f1c40f; margin-bottom:25px;">
            <h2 style="margin-top:0; color:#f1c40f; font-weight:900; letter-spacing:0.5px;">⚽ Welcome to Ayush’s WorldCup prediction engine!</h2>
            <p style="font-size:1.1em; color:#f8fafc; line-height:1.6; margin-bottom:0;">
                This system combines historical analytics, live performance metrics, and mathematical prediction models to forecast potential international football outcomes.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 🤔 You may be asking yourself... *HOW DOES THIS EVEN WORK!?*")
    st.markdown("Let me explain...")
    st.write("Instead of relying on just one statistic, the model combines several different factors to determine how \"strong\" a national team really is.")
    
    st.markdown("---")
    
    # Track 1: Recent Form
    st.markdown(
        """
        <h4>📈 <span style="color:#3498db; font-weight:bold;">Recent Form & Nostalgia Decay</span></h4>
        One of the biggest factors is recent form. Recent matches are given much more weight than less relevant games played years ago using an exponential decay function:
        """,
        unsafe_allow_html=True,
    )
    st.latex(r"w = e^{-\lambda \cdot t}")
    st.markdown(
        """
        where older matches gradually become less important over time. This helps <strong>eliminate nostalgia bias</strong> because what really matters is how a team is playing right now.
        """,
        unsafe_allow_html=True,
    )

    # Track 2: SoS
    st.markdown(
        """
        <br>
        <h4>⚔️ <span style="color:#e67e22; font-weight:bold;">Strength of Schedule (SoS) scaling</span></h4>
        Beating a world-class team is worth far more than beating a lower-ranked team (<em>duh!</em>), while losing to an elite opponent is less damaging than losing to a weaker side. After all, in a knockout tournament like the World Cup, you know what they say: <em>"You gotta beat the best to be the best."</em> 😉
        """,
        unsafe_allow_html=True,
    )

    # Track 3: Elo Baseline
    st.markdown(
        """
        <br>
        <h4>🧮 <span style="color:#9b59b6; font-weight:bold;">Live Elo Integration</span></h4>
        The model also incorporates live Elo ratings, which estimate each team's overall strength based on previous results. Elo calculates the expected outcome of a match using each team's rating before adjusting those ratings after every game. This gives the model a <strong>strong baseline</strong> instead of making predictions with no context of how much quality a team has.
        """,
        unsafe_allow_html=True,
    )

    # Track 4: Poisson distribution
    st.markdown(
        """
        <br>
        <h4>📊 <span style="color:#2ecc71; font-weight:bold;">Poisson Exact Scoreline Matrix</span></h4>
        To predict actual scorelines, the model uses a Poisson distribution, a mathematical model commonly used for low-scoring sports like soccer. Using each team's expected goals (<span style="color:#2ecc71; font-weight:bold;">$\lambda$</span>), it calculates the probability of scoring exactly 0, 1, 2, 3, or more goals, allowing it to estimate the likelihood of every possible scoreline from 0-0 to 3-2 and beyond.
        """,
        unsafe_allow_html=True,
    )

    # Track 5: Monte Carlo format simulations
    st.markdown(
        """
        <br>
        <h4>🎲 <span style="color:#f1c40f; font-weight:bold;">Stochastic Monte Carlo Bracket Simulations</span></h4>
        And finally, the entire tournament is run through thousands of Monte Carlo simulations. Each simulated tournament randomly selects winners based on the calculated match probabilities. By repeating this process thousands of times, the model measures how often each team reaches the <strong>Round of 16, Quarterfinals, Semifinals, Final, or lifts the trophy</strong>. Those percentages become each team's estimated chances of advancing through the tournament and ultimately becoming World Champions.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Manual section
    st.markdown(
        """
        <div style="background-color:#1e1e24; padding:20px; border-radius:8px; border:1px solid #333;">
            <h3 style="margin-top:0; color:#ffffff;">📖 How to Use</h3>
            <ol style="line-height:1.8; color:#ddd;">
                <li>Inspect active or upcoming competitive structures inside <span style="color:#3498db; font-weight:500;">Live Fixtures</span>.</li>
                <li>Review global ratings and baseline traits under <span style="color:#9b59b6; font-weight:500;">Team Ratings</span>.</li>
                <li>Head to <span style="color:#e67e22; font-weight:500;">Match Predictor</span> to configure arbitrary single-match variables—<em>be sure to use <strong>Load Conditions</strong> to scrape dynamic squad data like injury lists, club form, and international experience factors.</em></li>
                <li>Run multi-tournament paths under <span style="color:#f1c40f; font-weight:500;">World Cup Simulator</span> to parse overall championship title likelihood metrics.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Stack section
    st.markdown(
        """
        <div style="background-color:#111; padding:20px; border-radius:8px; border:1px solid #222;">
            <h3 style="margin-top:0; color:#ffffff;">🛠️ Tech Stack</h3>
            <ul style="list-style-type:none; padding-left:0; line-height:1.9; color:#bbb;">
                <li> <strong>Application Base Framework:</strong> Streamlit Core Engine</li>
                <li> <strong>Parsers & Ingestion Tools:</strong> BeautifulSoup4, Requests Library, LXML</li>
                <li> <strong>Mathematical Core Structures:</strong> NumPy, Pandas DataFrames, SciPy Statistical Toolkits</li>
                <li> <strong>Visual Presentation Graphics:</strong> Plotly Express & Plotly Graph Objects Engine</li>
                <li> <strong>Machine Learning Pipelines:</strong> XGBoost Classifier Optimization Core</li>
                <li> <strong>Context-Aware Analytics Assistant:</strong> Google GenAI SDK (<span style="font-family:monospace; color:#2ecc71;">gemini-2.5-flash</span>)</li>
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


def render_team_ratings_tab(api_key, team_stats_df):
    st.header("Team Ratings")
    st.caption(
        "Rating = 0.40 × xP/PPG (time-decay weighted) + 0.20 × SoS-adjusted goal "
        "performance + 0.40 × baseline global Elo modifier. Ties broken by higher GF/Game."
    )
    if not api_key:
        st.info("API Token connection missing from secrets management configuration.")
        return
    if team_stats_df.empty:
        st.info("No completed matches found yet for this competition/season.")
        return

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

    team_names   = sorted(team_stats_df["Team"].unique().tolist())
    active_teams = compute_active_teams(wc_fixtures_df) if wc_fixtures_df is not None else set()

    def _team_label(name: str) -> str:
        if not active_teams:
            return name
        return f"[IN] {name}" if name in active_teams else f"[OUT] {name}"

    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A (Home)", team_names, index=0, key="team_a_select", format_func=_team_label)
    with c2:
        team_b = st.selectbox("Team B (Away)", team_names, index=min(1, len(team_names)-1), key="team_b_select", format_func=_team_label)

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

            xg_home, xg_away = calculate_expected_goals(ra, rb, lgf, lga, saf_a=saf_a, saf_b=saf_b)
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

        st.subheader("Top 10 Most Likely Scorelines")
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


def _style_elimination_status(row: pd.Series, active_teams: set[str]) -> list[str]:
    team = row.get("Team")
    has_live_fixture = bool(active_teams) and team in active_teams

    stage_cols = ("Champion %", "Final %", "Semifinal %")
    has_sim_life = any(
        pd.notna(row.get(col)) and float(row.get(col)) > 0
        for col in stage_cols if col in row.index
    )

    is_active = has_live_fixture or has_sim_life
    style = (
        "background-color: rgba(46, 204, 113, 0.30); color: #eafaf1; font-weight:600;"
        if is_active else
        "background-color: rgba(231, 76, 60, 0.28); color: #fdecea; font-weight:600;"
    )
    return [style if col == "Team" else "" for col in row.index]


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
            st.session_state["wc_results"]     = results_df
            st.session_state["wc_groups_used"] = wc_groups
            st.session_state["wc_fixtures"]    = wc_fixtures
            st.session_state["wc_ratings_lookup"] = ratings_lookup
            st.session_state["wc_saf_lookup"] = saf_lookup
            st.session_state["wc_league_avg_gf"] = league_avg_gf
            st.session_state["wc_league_avg_ga"] = league_avg_ga
            st.session_state["wc_fixed_results"] = fixed_results

    if "wc_results" in st.session_state:
        df  = st.session_state["wc_results"]
        grp = st.session_state.get("wc_groups_used", {})
        fx  = st.session_state.get("wc_fixtures", pd.DataFrame())

        st.subheader("Championship Probabilities")
        active_teams = compute_active_teams(fx)
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


def render_gemini_chatbot():
    """Renders a Gemini-powered AI Tournament Analyst chatbot aware of internal application states."""
    try:
        import google.generativeai as genai
    except ImportError:
        st.warning("Please install `google-generativeai` to use the AI Analyst feature.")
        return
        
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
    
    wc_results = st.session_state.get("wc_results")
    sys_instruction = "You are an expert AI Tournament Analyst for the 2026 World Cup app.\n"
    
    if wc_results is not None and not wc_results.empty:
        top_contenders = wc_results.head(10).to_dict(orient="records")
        sys_instruction += (
            f"The current Monte Carlo simulations have been executed. The top contenders and their odds are:\n{top_contenders}\n"
            "If a user asks about discrepancies between the visual deterministic bracket and the overall Monte Carlo probabilities, "
            "explain that the visual bracket forces raw favorites through match bottlenecks, ignoring variance. The Monte Carlo handles "
            "compounded probabilities, accounting for realistic upsets and stacked competition sides."
        )
    else:
        sys_instruction += "The user has not run the Monte Carlo simulation yet. Advise them to run it in the World Cup Simulator tab to generate detailed odds."

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Chat history UI render loop
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Cooldown Rate Limiting Logic (60 seconds)
    last_time = st.session_state.get("last_chat_time", 0.0)
    current_time = time.time()
    cooldown_limit = 60.0
    elapsed_time = current_time - last_time
    
    if elapsed_time < cooldown_limit:
        remaining_seconds = int(cooldown_limit - elapsed_time)
        st.warning(f"Active: Please wait {remaining_seconds} seconds and start a new simulation before requesting your next tactical report.")
        disabled_input = True
    else:
        disabled_input = False

    if prompt := st.chat_input("Ask the AI about simulation logic, bottlenecks, or specific team chances...", disabled=disabled_input):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        st.session_state["last_chat_time"] = time.time()
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=sys_instruction)
                formatted_history = [
                    {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} 
                    for m in st.session_state["chat_history"][:-1]
                ]
                
                chat = model.start_chat(history=formatted_history)
                response = chat.send_message(prompt)
                st.markdown(response.text)
                st.session_state["chat_history"].append({"role": "assistant", "content": response.text})
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with Gemini API: {e}")


# ============================================================================
# 10. MAIN
# ============================================================================

def main():
    st.set_page_config(page_title="World Cup Predictor 2026", page_icon="⚽", layout="wide")

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

    tab0, tab1, tab2, tab3, tab4 = st.tabs(["About This Project", "Live Fixtures", "Team Ratings", "Match Predictor", "World Cup Simulator"])
    with tab0: render_about_tab()
    with tab1: render_live_fixtures_tab(api_key, live_df, fixtures_df, results_df, wc_fixtures_df)
    with tab2: render_team_ratings_tab(api_key, team_stats_df)
    with tab3: render_match_predictor_tab(api_key, team_stats_df, model_bundle, wc_fixtures_df)
    with tab4: render_world_cup_tab(api_key, model_bundle)

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
