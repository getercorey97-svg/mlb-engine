import sqlite3
import datetime
import re
import requests
import numpy as np
import pandas as pd
from pybaseball import (
    cache,
    statcast_pitcher_expected_stats,
    statcast_pitcher_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_exitvelo_barrels
)

cache.enable()
CURRENT_YEAR = datetime.date.today().year
DB_FILE = "mlb_engine.db"

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def clean_name(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r'[^a-zA-Z\s]', '', name)
    return " ".join(name.split()).lower()

def init_tables(conn):
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS Pitcher_Advanced_Metrics (
            pitcher_name TEXT PRIMARY KEY,
            clean_name TEXT,
            k_pct REAL,
            bb_pct REAL,
            pitches_per_bf REAL,
            pitches_per_game REAL,
            csw_pct REAL,
            swstr_pct REAL,
            xera REAL,
            xba REAL,
            hardhit_pct REAL,
            barrel_pct REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS Batter_Statcast_Advanced (
            player_name TEXT PRIMARY KEY,
            clean_name TEXT,
            contact_pct REAL,
            z_contact_pct REAL,
            o_swing_pct REAL,
            xba REAL,
            xwoba REAL,
            avg_exit_velo REAL,
            hardhit_pct REAL,
            barrel_pct REAL,
            bat_speed REAL,
            swing_length REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS Batter_Decoupled_Priors (
            player_name TEXT PRIMARY KEY,
            contact_skill_mod REAL,
            babip_skill_mod REAL,
            pa_contact_sample INTEGER,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

def fetch_mlb_pitching_totals(year):
    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching&season={year}&playerPool=ALL&limit=1500"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code != 200:
            return {}
        splits = res.json().get("stats", [{}])[0].get("splits", [])
        pitchers = {}
        for s in splits:
            p = s.get("player", {})
            st = s.get("stat", {})
            name = p.get("fullName")
            if not name:
                continue
            bf = int(st.get("battersFaced", 0))
            pitches = int(st.get("numberOfPitches", 0))
            gs = int(st.get("gamesStarted", 0))
            k = int(st.get("strikeOuts", 0))
            bb = int(st.get("baseOnBalls", 0))
            era = float(st.get("era", 4.10) or 4.10)
            
            p_bf = round(pitches / bf, 2) if bf > 0 else 3.90
            p_game = round(pitches / gs, 1) if gs > 0 else 88.0
            k_pct = round(k / bf, 3) if bf > 0 else 0.224
            bb_pct = round(bb / bf, 3) if bf > 0 else 0.080

            pitchers[clean_name(name)] = {
                "name": name,
                "clean_name": clean_name(name),
                "bf": bf,
                "pitches": pitches,
                "gs": gs,
                "k_pct": k_pct,
                "bb_pct": bb_pct,
                "p_bf": p_bf,
                "p_game": p_game,
                "era": era
            }
        return pitchers
    except Exception as e:
        print(f"[WARN] MLB Stats API Pitching pull failed: {e}")
        return {}

def fetch_mlb_batting_totals(year):
    url = f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=hitting&season={year}&playerPool=ALL&limit=2000"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code != 200:
            return {}
        splits = res.json().get("stats", [{}])[0].get("splits", [])
        batters = {}
        for s in splits:
            p = s.get("player", {})
            st = s.get("stat", {})
            name = p.get("fullName")
            if not name:
                continue
            pa = int(st.get("plateAppearances", 0))
            ab = int(st.get("atBats", 0))
            hits = int(st.get("hits", 0))
            k = int(st.get("strikeOuts", 0))
            bb = int(st.get("baseOnBalls", 0))
            avg = float(st.get("avg", 0.245) or 0.245)
            obp = float(st.get("obp", 0.315) or 0.315)

            contact_rate = round(1.0 - (k / pa), 3) if pa > 0 else 0.760

            batters[clean_name(name)] = {
                "name": name,
                "clean_name": clean_name(name),
                "pa": pa,
                "ab": ab,
                "hits": hits,
                "k": k,
                "bb": bb,
                "avg": avg,
                "obp": obp,
                "contact_pct": contact_rate
            }
        return batters
    except Exception as e:
        print(f"[WARN] MLB Stats API Batting pull failed: {e}")
        return {}

def ingest_pitcher_data(conn):
    print("[INGEST] Pulling Official MLB Pitching Totals...")
    mlb_pitchers = fetch_mlb_pitching_totals(CURRENT_YEAR)

    print("[INGEST] Pulling Statcast Pitcher Expected Metrics...")
    savant_exp = pd.DataFrame()
    savant_ev = pd.DataFrame()
    try:
        savant_exp = statcast_pitcher_expected_stats(CURRENT_YEAR, minPA=1)
        if not savant_exp.empty and 'last_name, first_name' in savant_exp.columns:
            savant_exp['clean_name'] = savant_exp['last_name, first_name'].apply(
                lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
            )
    except Exception as e:
        print(f"[WARN] Statcast Pitcher Expected Stats skipped: {e}")

    try:
        savant_ev = statcast_pitcher_exitvelo_barrels(CURRENT_YEAR, minBBE=1)
        if not savant_ev.empty and 'last_name, first_name' in savant_ev.columns:
            savant_ev['clean_name'] = savant_ev['last_name, first_name'].apply(
                lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
            )
    except Exception as e:
        print(f"[WARN] Statcast Pitcher Exit Velo skipped: {e}")

    exp_map = {}
    if not savant_exp.empty and 'clean_name' in savant_exp.columns:
        for _, r in savant_exp.iterrows():
            exp_map[r['clean_name']] = {
                "xba": float(r.get('est_ba', 0.245) or 0.245),
                "xwoba": float(r.get('est_woba', 0.315) or 0.315),
                "xera": float(r.get('est_era', 4.10) or 4.10)
            }

    ev_map = {}
    if not savant_ev.empty and 'clean_name' in savant_ev.columns:
        for _, r in savant_ev.iterrows():
            ev_map[r['clean_name']] = {
                "hardhit": float(r.get('hard_hit_percent', 38.0) or 38.0),
                "barrel": float(r.get('barrel_batted_rate', 6.5) or 6.5)
            }

    c = conn.cursor()
    count = 0
    for c_name, p in mlb_pitchers.items():
        name = p["name"]
        k_pct = p["k_pct"]
        bb_pct = p["bb_pct"]
        p_bf = p["p_bf"]
        p_game = p["p_game"]
        csw_pct = round(k_pct * 1.25, 3)
        swstr_pct = round(k_pct * 0.52, 3)

        s_exp = exp_map.get(c_name, {})
        s_ev = ev_map.get(c_name, {})

        xera = s_exp.get("xera", p["era"])
        xba = s_exp.get("xba", 0.245)
        hardhit = s_ev.get("hardhit", 38.0)
        barrel = s_ev.get("barrel", 6.5)

        c.execute("""
            INSERT OR REPLACE INTO Pitcher_Advanced_Metrics (
                pitcher_name, clean_name, k_pct, bb_pct, pitches_per_bf,
                pitches_per_game, csw_pct, swstr_pct, xera, xba, hardhit_pct, barrel_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, c_name, k_pct, bb_pct, p_bf, p_game, csw_pct, swstr_pct, xera, xba, hardhit, barrel))

        c.execute("""
            UPDATE Pitcher_Stats
            SET whiff_rate = ?, pitches_per_bf = ?
            WHERE clean_name = ? OR pitcher_name = ?
        """, (round(swstr_pct * 2.1, 3), p_bf, c_name, name))
        count += 1

    conn.commit()
    print(f"[SUCCESS] Ingested Pitcher Advanced Metrics for {count} arms.")

def ingest_batter_data(conn):
    print("[INGEST] Pulling Official MLB Batting Totals...")
    mlb_batters = fetch_mlb_batting_totals(CURRENT_YEAR)

    print("[INGEST] Pulling Statcast Batter Expected Metrics...")
    savant_b_exp = pd.DataFrame()
    savant_b_ev = pd.DataFrame()
    try:
        savant_b_exp = statcast_batter_expected_stats(CURRENT_YEAR, minPA=1)
        if not savant_b_exp.empty and 'last_name, first_name' in savant_b_exp.columns:
            savant_b_exp['clean_name'] = savant_b_exp['last_name, first_name'].apply(
                lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
            )
    except Exception as e:
        print(f"[WARN] Statcast Batter Expected Stats skipped: {e}")

    try:
        savant_b_ev = statcast_batter_exitvelo_barrels(CURRENT_YEAR, minBBE=1)
        if not savant_b_ev.empty and 'last_name, first_name' in savant_b_ev.columns:
            savant_b_ev['clean_name'] = savant_b_ev['last_name, first_name'].apply(
                lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
            )
    except Exception as e:
        print(f"[WARN] Statcast Batter Exit Velo skipped: {e}")

    b_exp_map = {}
    if not savant_b_exp.empty and 'clean_name' in savant_b_exp.columns:
        for _, r in savant_b_exp.iterrows():
            b_exp_map[r['clean_name']] = {
                "xba": float(r.get('est_ba', 0.245) or 0.245),
                "xwoba": float(r.get('est_woba', 0.315) or 0.315)
            }

    b_ev_map = {}
    if not savant_b_ev.empty and 'clean_name' in savant_b_ev.columns:
        for _, r in savant_b_ev.iterrows():
            b_ev_map[r['clean_name']] = {
                "avg_ev": float(r.get('avg_hit_speed', 88.5) or 88.5),
                "hardhit": float(r.get('hard_hit_percent', 36.0) or 36.0),
                "barrel": float(r.get('barrel_batted_rate', 6.0) or 6.0)
            }

    c = conn.cursor()
    count = 0
    for c_name, b in mlb_batters.items():
        name = b["name"]
        pa = b["pa"]
        contact_pct = b["contact_pct"]
        z_contact = round(min(0.95, contact_pct * 1.10), 3)
        o_swing = 0.310

        s_exp = b_exp_map.get(c_name, {})
        s_ev = b_ev_map.get(c_name, {})

        xba = s_exp.get("xba", b["avg"])
        xwoba = s_exp.get("xwoba", b["obp"])
        avg_ev = s_ev.get("avg_ev", 88.5)
        hardhit = s_ev.get("hardhit", 36.0)
        barrel = s_ev.get("barrel", 6.0)

        c.execute("""
            INSERT OR REPLACE INTO Batter_Statcast_Advanced (
                player_name, clean_name, contact_pct, z_contact_pct, o_swing_pct,
                xba, xwoba, avg_exit_velo, hardhit_pct, barrel_pct, bat_speed, swing_length
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, c_name, contact_pct, z_contact, o_swing, xba, xwoba, avg_ev, hardhit, barrel, 71.5, 7.2))

        contact_multiplier = round(contact_pct / 0.76, 3)
        babip_multiplier = round(xba / 0.245, 3)

        c.execute("""
            INSERT INTO Batter_Decoupled_Priors (player_name, contact_skill_mod, babip_skill_mod, pa_contact_sample)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(player_name) DO UPDATE SET
                contact_skill_mod = ?,
                babip_skill_mod = ?,
                pa_contact_sample = ?
        """, (name, contact_multiplier, babip_multiplier, int(pa), contact_multiplier, babip_multiplier, int(pa)))
        count += 1

    conn.commit()
    print(f"[SUCCESS] Ingested Batter Statcast Metrics for {count} hitters.")

if __name__ == "__main__":
    conn = get_db()
    init_tables(conn)
    ingest_pitcher_data(conn)
    ingest_batter_data(conn)
    conn.close()
    print("[ALL DONE] Multilateral Statcast ingestion complete.")
