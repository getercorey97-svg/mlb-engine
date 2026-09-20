import sqlite3
import datetime
import re
import numpy as np
import pandas as pd
from pybaseball import (
    cache,
    pitching_stats,
    batting_stats,
    statcast_pitcher_expected_stats,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_arsenal_stats,
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

def ingest_pitcher_data(conn):
    print("[PYBASEBALL] Querying FanGraphs Pitching Leaderboards...")
    try:
        fg_pitch = pitching_stats(CURRENT_YEAR, qual=0)
    except Exception as e:
        print(f"[WARN] FanGraphs pitch pull failed: {e}")
        return

    print("[PYBASEBALL] Querying Statcast Pitcher Expected Stats & Exit Velo...")
    try:
        savant_exp = statcast_pitcher_expected_stats(CURRENT_YEAR, minPA=1)
        savant_ev = statcast_pitcher_exitvelo_barrels(CURRENT_YEAR, minBBE=1)
    except Exception as e:
        print(f"[WARN] Statcast pitch pull failed: {e}")
        savant_exp = pd.DataFrame()
        savant_ev = pd.DataFrame()

    fg_pitch['clean_name'] = fg_pitch['Name'].apply(clean_name)
    
    if not savant_exp.empty and 'last_name, first_name' in savant_exp.columns:
        savant_exp['clean_name'] = savant_exp['last_name, first_name'].apply(
            lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
        )
        fg_pitch = fg_pitch.merge(savant_exp[['clean_name', 'est_ba', 'est_slg', 'est_woba']], on='clean_name', how='left')
    else:
        fg_pitch['est_ba'] = np.nan

    if not savant_ev.empty and 'last_name, first_name' in savant_ev.columns:
        savant_ev['clean_name'] = savant_ev['last_name, first_name'].apply(
            lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
        )
        fg_pitch = fg_pitch.merge(savant_ev[['clean_name', 'hard_hit_percent', 'barrel_batted_rate']], on='clean_name', how='left')
    else:
        fg_pitch['hard_hit_percent'] = np.nan
        fg_pitch['barrel_batted_rate'] = np.nan

    c = conn.cursor()
    for _, r in fg_pitch.iterrows():
        name = r['Name']
        c_name = r['clean_name']
        bf = float(r.get('TBF', 0) or 0)
        pitches = float(r.get('Pitches', 0) or 0)
        starts = float(r.get('GS', 0) or 1)

        p_per_bf = round(pitches / bf, 2) if bf > 0 else 3.90
        p_per_g = round(pitches / starts, 1) if starts > 0 else 88.0

        k_pct = float(r.get('K%', 0) or 0.224)
        bb_pct = float(r.get('BB%', 0) or 0.08)
        csw_pct = float(r.get('CSW%', 0) or 0.28)
        swstr_pct = float(r.get('SwStr%', 0) or 0.11)
        xera = float(r.get('xERA', 0) or 4.10)
        xba = float(r.get('est_ba', 0) or 0.245)
        hardhit = float(r.get('hard_hit_percent', 0) or 38.0)
        barrel = float(r.get('barrel_batted_rate', 0) or 6.5)

        c.execute("""
            INSERT OR REPLACE INTO Pitcher_Advanced_Metrics (
                pitcher_name, clean_name, k_pct, bb_pct, pitches_per_bf,
                pitches_per_game, csw_pct, swstr_pct, xera, xba, hardhit_pct, barrel_pct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, c_name, k_pct, bb_pct, p_per_bf, p_per_g, csw_pct, swstr_pct, xera, xba, hardhit, barrel))

        # Update Pitcher_Stats with clean_name support
        c.execute("""
            UPDATE Pitcher_Stats
            SET whiff_rate = ?, pitches_per_bf = ?
            WHERE clean_name = ? OR pitcher_name = ?
        """, (round(swstr_pct * 2.1, 3), p_per_bf, c_name, name))

    conn.commit()
    print(f"[SUCCESS] Ingested advanced pitching metrics for {len(fg_pitch)} pitchers.")

def ingest_batter_data(conn):
    print("[PYBASEBALL] Querying FanGraphs Batting Leaderboards...")
    try:
        fg_bat = batting_stats(CURRENT_YEAR, qual=0)
    except Exception as e:
        print(f"[WARN] FanGraphs batting pull failed: {e}")
        return

    print("[PYBASEBALL] Querying Statcast Batter Expected Stats & Exit Velo...")
    try:
        savant_b_exp = statcast_batter_expected_stats(CURRENT_YEAR, minPA=1)
        savant_b_ev = statcast_batter_exitvelo_barrels(CURRENT_YEAR, minBBE=1)
    except Exception as e:
        print(f"[WARN] Statcast batter pull failed: {e}")
        savant_b_exp = pd.DataFrame()
        savant_b_ev = pd.DataFrame()

    fg_bat['clean_name'] = fg_bat['Name'].apply(clean_name)
    
    if not savant_b_exp.empty and 'last_name, first_name' in savant_b_exp.columns:
        savant_b_exp['clean_name'] = savant_b_exp['last_name, first_name'].apply(
            lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
        )
        fg_bat = fg_bat.merge(savant_b_exp[['clean_name', 'est_ba', 'est_woba']], on='clean_name', how='left')
    else:
        fg_bat['est_ba'] = np.nan
        fg_bat['est_woba'] = np.nan

    if not savant_b_ev.empty and 'last_name, first_name' in savant_b_ev.columns:
        savant_b_ev['clean_name'] = savant_b_ev['last_name, first_name'].apply(
            lambda x: clean_name(" ".join(reversed(str(x).split(", "))))
        )
        fg_bat = fg_bat.merge(savant_b_ev[['clean_name', 'avg_hit_speed', 'hard_hit_percent', 'barrel_batted_rate']], on='clean_name', how='left')
    else:
        fg_bat['avg_hit_speed'] = np.nan
        fg_bat['hard_hit_percent'] = np.nan
        fg_bat['barrel_batted_rate'] = np.nan

    c = conn.cursor()
    for _, r in fg_bat.iterrows():
        name = r['Name']
        c_name = r['clean_name']
        pa = float(r.get('PA', 0) or 0)
        contact_pct = float(r.get('Contact%', 0) or 0.76)
        z_contact = float(r.get('Z-Contact%', 0) or 0.84)
        o_swing = float(r.get('O-Swing%', 0) or 0.31)
        xba = float(r.get('est_ba', 0) or float(r.get('BA', 0.245)) or 0.245)
        xwoba = float(r.get('est_woba', 0) or float(r.get('wOBA', 0.315)) or 0.315)
        avg_ev = float(r.get('avg_hit_speed', 0) or 88.5)
        hardhit = float(r.get('hard_hit_percent', 0) or 36.0)
        barrel = float(r.get('barrel_batted_rate', 0) or 6.0)

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

    conn.commit()
    print(f"[SUCCESS] Ingested advanced batting metrics for {len(fg_bat)} hitters.")

if __name__ == "__main__":
    conn = get_db()
    init_tables(conn)
    ingest_pitcher_data(conn)
    ingest_batter_data(conn)
    conn.close()
    print("[ALL DONE] Pybaseball pipeline execution finished.")
