import sqlite3
import re
import time
import json
import requests
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="STATSCENTER PRO HUB - High-Performance Analytics")

CACHE = {
    "schedule_data": {},
    "db_frames": {},
    "last_fetched": 0,
    "db_last_read": 0
}
CACHE_TTL_SECONDS = 30
DB_CACHE_TTL_SECONDS = 60

def get_db():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def clean_team_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()

TEAM_POWER_INDEX = {
    "dodgers": (1.18, 0.88), "braves": (1.14, 0.90), "yankees": (1.15, 0.91),
    "orioles": (1.12, 0.92), "phillies": (1.11, 0.91), "astros": (1.08, 0.93),
    "guardians": (1.04, 0.89), "padres": (1.06, 0.93), "brewers": (1.05, 0.91),
    "diamondbacks": (1.09, 0.98), "twins": (1.04, 0.94), "royals": (1.03, 0.92),
    "redsox": (1.05, 0.97), "rays": (0.95, 0.93), "mariners": (0.94, 0.88),
    "tigers": (0.96, 0.94), "rangers": (0.98, 0.97), "cubs": (0.99, 0.96),
    "mets": (1.04, 0.96), "cardinals": (0.97, 0.99), "giants": (0.96, 0.98),
    "reds": (0.97, 1.02), "bluejays": (0.95, 1.01), "pirates": (0.92, 0.98),
    "nationals": (0.94, 1.05), "athletics": (0.93, 1.08), "angels": (0.92, 1.07),
    "rockies": (0.95, 1.18), "marlins": (0.88, 1.10), "whitesox": (0.82, 1.15)
}

def derive_quantitative_projection(away_team: str, home_team: str):
    away_key = next((k for k in TEAM_POWER_INDEX if k in clean_team_name(away_team)), "league")
    home_key = next((k for k in TEAM_POWER_INDEX if k in clean_team_name(home_team)), "league")
    a_off, a_def = TEAM_POWER_INDEX.get(away_key, (1.0, 1.0))
    h_off, h_def = TEAM_POWER_INDEX.get(home_key, (1.0, 1.0))

    exp_away = round(4.45 * a_off * h_def, 2)
    exp_home = round(4.45 * h_off * a_def * 1.04, 2)
    p_home = round((exp_home ** 1.83) / ((exp_home ** 1.83) + (exp_away ** 1.83)), 3)
    p_away = round(1.0 - p_home, 3)

    f5_exp_a = round(exp_away * 0.55, 2)
    f5_exp_h = round(exp_home * 0.55, 2)
    f5_median = round(f5_exp_a + f5_exp_h, 1)

    return {
        "prob_home_win": p_home, "prob_away_win": p_away,
        "expected_runs_away": exp_away, "expected_runs_home": exp_home,
        "f5_exp_away": f5_exp_a, "f5_exp_home": f5_exp_h, "f5_median_runs": f5_median
    }

def load_cached_dataframes():
    now = time.time()
    if CACHE["db_frames"] and (now - CACHE["db_last_read"]) < DB_CACHE_TTL_SECONDS:
        return CACHE["db_frames"]

    conn = get_db()
    c = conn.cursor()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    frames = {}
    target_tables = [
        "Model_Forecasts", "Historical_Forecasts", "Daily_Lineups", 
        "Daily_Batters", "Batter_Hit_Forecasts", "Pitcher_K_Forecasts",
        "Pitcher_Kalman_State", "Batter_Decoupled_Priors"
    ]

    for tbl in target_tables:
        if tbl in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {tbl}", conn).fillna(0)
                if "game_pk" in df.columns:
                    df["game_pk"] = df["game_pk"].astype(str)
                frames[tbl] = df
            except Exception as e:
                print(f"[WARN] Error loading {tbl}: {e}", flush=True)
                frames[tbl] = pd.DataFrame()
        else:
            frames[tbl] = pd.DataFrame()

    conn.close()
    CACHE["db_frames"] = frames
    CACHE["db_last_read"] = now
    return frames

def fetch_mlb_live_schedule():
    now = time.time()
    if CACHE["schedule_data"] and (now - CACHE["last_fetched"]) < CACHE_TTL_SECONDS:
        return CACHE["schedule_data"]

    now_utc = datetime.now(timezone.utc)
    start_date = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}&hydrate=lineups,linescore,probablePitcher,team"

    schedule_map = {}
    try:
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            for d in res.json().get("dates", []):
                for g in d.get("games", []):
                    schedule_map[str(g["gamePk"])] = g
            CACHE["schedule_data"] = schedule_map
            CACHE["last_fetched"] = now
    except Exception as e:
        print(f"[WARN] Live MLB API schedule pull error: {e}", flush=True)

    return CACHE["schedule_data"]

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    dfs = load_cached_dataframes()
    live_schedule = fetch_mlb_live_schedule()

    now_utc = datetime.now(timezone.utc)
    today_et = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    yest_et = (now_utc.astimezone(ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")
    tomo_et = (now_utc.astimezone(ZoneInfo("America/New_York")) + timedelta(days=1)).strftime("%Y-%m-%d")

    forecast_map = {}
    df_mf = dfs.get("Model_Forecasts", pd.DataFrame())
    if not df_mf.empty:
        for _, row in df_mf.iterrows():
            forecast_map[str(row.get("game_pk"))] = row.to_dict()

    games_payload = []
    ticker_items = []

    for pk, g in live_schedule.items():
        g_date = g.get("officialDate", "")
        status_info = g.get("status", {})
        abstract_state = status_info.get("abstractGameState", "Preview")
        
        teams = g.get("teams", {})
        away_data = teams.get("away", {})
        home_data = teams.get("home", {})
        away_name = away_data.get("team", {}).get("name", "Away")
        home_name = home_data.get("team", {}).get("name", "Home")

        linescore = g.get("linescore", {})
        curr_inning = linescore.get("currentInningOrdinal", "")
        is_top = linescore.get("isTopInning", True)
        half_sym = "▲" if is_top else "▼"
        outs = linescore.get("outs", 0)

        act_score_away = away_data.get("score", 0)
        act_score_home = home_data.get("score", 0)

        if abstract_state == "Live":
            stage = "live"
            status_badge = f"{half_sym}{curr_inning} ({outs} Out)"
        elif abstract_state == "Final":
            stage = "final"
            status_badge = f"FINAL: {away_name} {act_score_away}, {home_name} {act_score_home}"
        else:
            stage = "upcoming"
            start_iso = g.get("gameDate", "")
            if start_iso:
                try:
                    dt = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
                    status_badge = dt.astimezone(ZoneInfo("America/New_York")).strftime("%I:%M %p EDT")
                except Exception:
                    status_badge = "Scheduled"
            else:
                status_badge = "Scheduled"

        date_bucket = "today"
        if g_date == yest_et:
            date_bucket = "yesterday"
        elif g_date == tomo_et:
            date_bucket = "tomorrow"

        mf = forecast_map.get(pk, {})
        if not mf or not mf.get("prob_home_win"):
            mf = derive_quantitative_projection(away_name, home_name)

        p_home = float(mf.get("prob_home_win") or 0.50)
        p_away = round(1.0 - p_home, 3)
        exp_away = float(mf.get("expected_runs_away") or 4.5)
        exp_home = float(mf.get("expected_runs_home") or 4.5)
        f5_median = float(mf.get("f5_median_runs") or round((exp_away + exp_home) * 0.55, 1))

        if stage == "live" and linescore:
            inn = linescore.get("currentInning", 1)
            rem_pct = np.clip((9.0 - (inn - (0.5 if not is_top else 0.0))) / 9.0, 0.04, 1.0)
            proj_away = act_score_away + (exp_away * rem_pct)
            proj_home = act_score_home + (exp_home * rem_pct)
            run_diff = proj_home - proj_away
            live_p_home = round(float(1.0 / (1.0 + np.exp(-0.45 * run_diff))), 3)
            live_p_away = round(1.0 - live_p_home, 3)
        else:
            live_p_home = p_home
            live_p_away = p_away

        if stage == "live":
            ticker_items.append(f"🔴 LIVE: {away_name} {act_score_away} @ {home_name} {act_score_home} ({status_badge}) | Pick: {home_name if live_p_home >= 0.5 else away_name} ({max(live_p_home, live_p_away)*100:.1f}%)")
        elif stage == "upcoming":
            fav = home_name if p_home >= 0.5 else away_name
            ticker_items.append(f"⏳ UPCOMING: {away_name} @ {home_name} ({status_badge}) | Pick: {fav} ({max(p_home, p_away)*100:.1f}%) | Total: {exp_away+exp_home:.1f}")
        elif stage == "final":
            winner = home_name if act_score_home > act_score_away else away_name
            predicted = home_name if p_home >= 0.5 else away_name
            ticker_items.append(f"🏁 FINAL: {away_name} {act_score_away}, {home_name} {act_score_home} | Pick: {predicted} {'✅' if winner == predicted else '❌'}")

        games_payload.append({
            "game_pk": pk,
            "away_team": away_name,
            "home_team": home_name,
            "stage": stage,
            "date_bucket": date_bucket,
            "status_badge": status_badge,
            "score_away": act_score_away,
            "score_home": act_score_home,
            "prob_home": live_p_home,
            "prob_away": live_p_away,
            "pregame_prob_home": p_home,
            "pregame_prob_away": p_away,
            "exp_away": exp_away,
            "exp_home": exp_home,
            "f5_median": f5_median
        })

    # Batter Hit Forecasts
    batter_props = []
    df_batters = dfs.get("Batter_Hit_Forecasts", pd.DataFrame())
    df_daily = dfs.get("Daily_Batters", pd.DataFrame())
    df_decoupled = dfs.get("Batter_Decoupled_Priors", pd.DataFrame())

    if not df_batters.empty:
        merged_b = df_batters.copy()
        if not df_daily.empty and "player_name" in df_daily.columns:
            merged_b = pd.merge(merged_b, df_daily[["game_pk", "player_name", "is_confirmed", "batting_order"]], on=["game_pk", "player_name"], how="left").fillna(0)
        if not df_decoupled.empty and "player_name" in df_decoupled.columns:
            merged_b = pd.merge(merged_b, df_decoupled[["player_name", "contact_skill_mod", "babip_skill_mod"]], on="player_name", how="left").fillna(1.0)

        for _, row in merged_b.iterrows():
            xhits = float(row.get("expected_hits") or 0.0)
            p_over_0_5 = round(float(1.0 - stats.poisson.cdf(0, xhits)), 4) if xhits > 0 else 0.0
            p_over_1_5 = round(float(1.0 - stats.poisson.cdf(1, xhits)), 4) if xhits > 0 else 0.0

            batter_props.append({
                "game_pk": str(row.get("game_pk")),
                "player_name": str(row.get("player_name")),
                "team_name": str(row.get("team_name") or ""),
                "lineup_order": int(row.get("batting_order") or 1),
                "expected_hits": xhits,
                "over_0_5_hit_prob": p_over_0_5,
                "over_1_5_hit_prob": p_over_1_5,
                "is_confirmed": int(row.get("is_confirmed") or 0),
                "contact_mod": round(float(row.get("contact_skill_mod") or 1.000), 3),
                "babip_mod": round(float(row.get("babip_skill_mod") or 1.000), 3)
            })

    # Pitcher Strikeout Forecasts
    pitcher_props = []
    df_pitchers = dfs.get("Pitcher_K_Forecasts", pd.DataFrame())
    df_kalman = dfs.get("Pitcher_Kalman_State", pd.DataFrame())

    if not df_pitchers.empty:
        merged_p = df_pitchers.copy()
        if not df_kalman.empty and "pitcher_name" in df_kalman.columns:
            merged_p = pd.merge(merged_p, df_kalman[["pitcher_name", "latent_k_modifier", "variance_p"]], on="pitcher_name", how="left").fillna(0)

        for _, row in merged_p.iterrows():
            xk = float(row.get("expected_k") or 0.0)
            line = float(row.get("k_line") or 4.5)
            k_threshold = int(np.floor(line))
            under_prob = round(float(stats.poisson.cdf(k_threshold, xk)), 4) if xk > 0 else 1.0
            over_prob = round(1.0 - under_prob, 4)

            pitcher_props.append({
                "game_pk": str(row.get("game_pk")),
                "pitcher_name": str(row.get("pitcher_name")),
                "team_name": str(row.get("team_name") or ""),
                "k_line": line,
                "expected_k": xk,
                "over_prob": over_prob,
                "under_prob": under_prob,
                "kalman_theta": round(float(row.get("latent_k_modifier") or 1.000), 3),
                "kalman_p": round(float(row.get("variance_p") or 0.040), 4)
            })

    ticker_content = " &nbsp;&nbsp;&nbsp;&bull;&nbsp;&nbsp;&nbsp; ".join(ticker_items) if ticker_items else "Synchronizing live betting board and Statcast distributions..."

    # Template-Based Injection to Prevent F-String Collisions
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>STATSCENTER PRO HUB - High Performance Engine</title>
    <style>
        :root {
            --bg-primary: #0a0d14;
            --bg-card: #121824;
            --border: #1f2a3d;
            --text-main: #f0f3f8;
            --text-dim: #8b9bb4;
            --accent-red: #e02424;
            --accent-green: #059669;
            --accent-gold: #d97706;
            --accent-blue: #2563eb;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }
        body { background-color: var(--bg-primary); color: var(--text-main); line-height: 1.4; padding-bottom: 60px; }
        
        .ticker-wrap {
            position: sticky;
            top: 0;
            width: 100%;
            overflow: hidden;
            height: 40px;
            background-color: #000;
            border-bottom: 2px solid var(--accent-red);
            display: flex;
            align-items: center;
            z-index: 1000;
        }
        .ticker-brand {
            background: var(--accent-red);
            color: #fff;
            padding: 0 16px;
            font-weight: 900;
            font-size: 11px;
            letter-spacing: 1px;
            height: 100%;
            display: flex;
            align-items: center;
            white-space: nowrap;
            z-index: 2;
        }
        .ticker-move {
            display: inline-block;
            white-space: nowrap;
            padding-left: 100vw;
            animation: tickerScroll 120s linear infinite;
            font-size: 13px;
            font-weight: 600;
            color: #e5e7eb;
        }
        .ticker-wrap:hover .ticker-move, .ticker-wrap:active .ticker-move {
            animation-play-state: paused;
        }
        @keyframes tickerScroll {
            0% { transform: translate3d(0, 0, 0); }
            100% { transform: translate3d(-100%, 0, 0); }
        }

        .container { max-width: 1100px; margin: 0 auto; padding: 14px; }
        .header { display: flex; justify-content: space-between; align-items: center; padding: 14px 0; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
        .header h1 { font-size: 20px; font-weight: 800; letter-spacing: 0.5px; color: #fff; }
        .header .badge { background: #1e293b; color: #38bdf8; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 4px; border: 1px solid #0284c7; }

        .nav-tabs, .filter-tabs { display: flex; gap: 8px; margin-bottom: 10px; overflow-x: auto; padding-bottom: 4px; }
        .btn {
            background: var(--bg-card);
            color: var(--text-dim);
            border: 1px solid var(--border);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            white-space: nowrap;
        }
        .btn.active { background: var(--accent-red); color: #fff; border-color: var(--accent-red); }
        
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px;
            position: relative;
        }
        .card-header { display: flex; justify-content: space-between; font-size: 11px; color: var(--text-dim); margin-bottom: 8px; }
        .matchup-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
        .team-name { font-size: 15px; font-weight: 800; color: #fff; }
        .team-score { font-size: 18px; font-weight: 900; color: #f59e0b; }
        .prob-pill { font-size: 12px; font-weight: 800; padding: 3px 10px; border-radius: 4px; background: #1e293b; color: #94a3b8; }
        .prob-pill.fav { background: #064e3b; color: #34d399; border: 1px solid #059669; }

        .data-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
        .data-table th { text-align: left; padding: 10px 8px; color: var(--text-dim); border-bottom: 1px solid var(--border); font-size: 11px; text-transform: uppercase; }
        .data-table td { padding: 10px 8px; border-bottom: 1px solid rgba(255,255,255,0.04); }
        .badge-conf { background: #065f46; color: #34d399; font-size: 10px; padding: 2px 6px; border-radius: 3px; font-weight: 800; }
        .badge-proj { background: #78350f; color: #fbbf24; font-size: 10px; padding: 2px 6px; border-radius: 3px; font-weight: 800; }
    </style>
</head>
<body>

<div class="ticker-wrap">
    <div class="ticker-brand">&#9679; BOTTOMLINE</div>
    <div class="ticker-move">__TICKER_ITEMS__</div>
</div>

<div class="container">
    <div class="header">
        <div>
            <h1>STATSCENTER PRO HUB</h1>
            <div style="font-size: 11px; color: var(--text-dim); font-weight: 600;">Vectorized Discrete Prop & Kalman Architecture</div>
        </div>
        <div class="badge">FULL SCIENTIFIC STACK</div>
    </div>

    <div class="nav-tabs">
        <button class="btn" onclick="setDateFilter('yesterday', this)">Yesterday</button>
        <button class="btn active" onclick="setDateFilter('today', this)">Today</button>
        <button class="btn" onclick="setDateFilter('tomorrow', this)">Tomorrow</button>
    </div>

    <div class="filter-tabs">
        <button class="btn active" onclick="setStageFilter('all', this)" id="cnt-all">All</button>
        <button class="btn" onclick="setStageFilter('live', this)" id="cnt-live">&#128308; Live</button>
        <button class="btn" onclick="setStageFilter('upcoming', this)" id="cnt-upcoming">&#9203; Upcoming</button>
        <button class="btn" onclick="setStageFilter('final', this)" id="cnt-final">&#127937; Final</button>
    </div>

    <div class="nav-tabs" style="border-top: 1px solid var(--border); padding-top: 12px;">
        <button class="btn active" onclick="setViewMode('matchups', this)">Matchups & F5</button>
        <button class="btn" onclick="setViewMode('batters', this)" id="cnt-batters">Batter Hits (All)</button>
        <button class="btn" onclick="setViewMode('pitchers', this)" id="cnt-pitchers">Pitcher Ks (All)</button>
    </div>

    <div id="view-matchups" class="grid"></div>

    <div id="view-batters" style="display: none; overflow-x: auto;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>HITTER</th>
                    <th>SLOT</th>
                    <th>TEAM</th>
                    <th>xHITS</th>
                    <th>OVER 0.5</th>
                    <th>OVER 1.5</th>
                    <th>CONTACT / LUCK</th>
                    <th>STATUS</th>
                </tr>
            </thead>
            <tbody id="batters-tbody"></tbody>
        </table>
    </div>

    <div id="view-pitchers" style="display: none; overflow-x: auto;">
        <table class="data-table">
            <thead>
                <tr>
                    <th>STARTER</th>
                    <th>TEAM</th>
                    <th>LINE</th>
                    <th>xK</th>
                    <th>OVER %</th>
                    <th>UNDER %</th>
                    <th>KALMAN STATE (θ | P)</th>
                </tr>
            </thead>
            <tbody id="pitchers-tbody"></tbody>
        </table>
    </div>
</div>

<script>
    const games = __GAMES_DATA__;
    const batterProps = __BATTER_DATA__;
    const pitcherProps = __PITCHER_DATA__;

    let activeDate = 'today';
    let activeStage = 'all';
    let activeView = 'matchups';

    function updateCounts() {
        const filtered = games.filter(g => g.date_bucket === activeDate);
        const liveCnt = filtered.filter(g => g.stage === 'live').length;
        const upCnt = filtered.filter(g => g.stage === 'upcoming').length;
        const finCnt = filtered.filter(g => g.stage === 'final').length;

        document.getElementById('cnt-all').innerText = `All (${filtered.length})`;
        document.getElementById('cnt-live').innerText = `🔴 Live (${liveCnt})`;
        document.getElementById('cnt-upcoming').innerText = `⏳ Upcoming (${upCnt})`;
        document.getElementById('cnt-final').innerText = `🏁 Final (${finCnt})`;

        const activePks = new Set(filtered.map(g => String(g.game_pk)));
        const activeBatters = batterProps.filter(b => activePks.has(String(b.game_pk))).length;
        const activePitchers = pitcherProps.filter(p => activePks.has(String(p.game_pk))).length;

        document.getElementById('cnt-batters').innerText = `Batter Hits (${activeBatters})`;
        document.getElementById('cnt-pitchers').innerText = `Pitcher Ks (${activePitchers})`;
    }

    function renderMatchups() {
        const container = document.getElementById('view-matchups');
        container.innerHTML = '';

        const filtered = games.filter(g => {
            const matchDate = g.date_bucket === activeDate;
            const matchStage = (activeStage === 'all') || (g.stage === activeStage);
            return matchDate && matchStage;
        });

        if (filtered.length === 0) {
            container.innerHTML = '<div style="color: var(--text-dim); text-align: center; padding: 40px; font-size: 13px; grid-column: 1/-1;">No matchups found for this selection.</div>';
            return;
        }

        filtered.forEach(g => {
            const awayFav = g.prob_away >= g.prob_home;
            const homeFav = g.prob_home > g.prob_away;
            const totalRuns = Number(g.exp_away + g.exp_home).toFixed(1);
            
            const card = document.createElement('div');
            card.className = 'card';
            card.innerHTML = `
                <div class="card-header">
                    <span>${g.status_badge}</span>
                    <span style="font-family: monospace;">PK: ${g.game_pk}</span>
                </div>
                <div class="matchup-row">
                    <span class="team-name">${g.away_team}</span>
                    <span class="team-score">${g.stage === 'upcoming' ? '' : g.score_away}</span>
                    <span class="prob-pill ${awayFav ? 'fav' : ''}">${(g.prob_away * 100).toFixed(1)}%</span>
                </div>
                <div class="matchup-row">
                    <span class="team-name">${g.home_team}</span>
                    <span class="team-score">${g.stage === 'upcoming' ? '' : g.score_home}</span>
                    <span class="prob-pill ${homeFav ? 'fav' : ''}">${(g.prob_home * 100).toFixed(1)}%</span>
                </div>
                <div style="font-size: 11px; color: var(--text-dim); margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px; display: flex; justify-content: space-between;">
                    <span>Model Runs: ${g.exp_away.toFixed(1)} - ${g.exp_home.toFixed(1)} (O/U: ${totalRuns})</span>
                    <span>F5 Median: ${g.f5_median.toFixed(1)}</span>
                </div>
            `;
            container.appendChild(card);
        });
    }

    function renderBatters() {
        const tbody = document.getElementById('batters-tbody');
        tbody.innerHTML = '';

        const activePks = new Set(
            games.filter(g => g.date_bucket === activeDate && (activeStage === 'all' || g.stage === activeStage))
                 .map(g => String(g.game_pk))
        );

        const filteredBatters = batterProps.filter(b => activePks.has(String(b.game_pk)));

        if (filteredBatters.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-dim); padding: 30px;">No batter props for this selection.</td></tr>';
            return;
        }

        filteredBatters.forEach(b => {
            const tr = document.createElement('tr');
            const confBadge = b.is_confirmed == 1 
                ? '<span class="badge-conf">CONFIRMED</span>' 
                : '<span class="badge-proj">PROJECTED</span>';

            tr.innerHTML = `
                <td style="font-weight: 700; color: #fff;">${b.player_name}</td>
                <td style="color: var(--text-dim);">#${b.lineup_order || '-'}</td>
                <td>${b.team_name || ''}</td>
                <td style="font-weight: 600; font-family: monospace;">${Number(b.expected_hits || 0).toFixed(2)}</td>
                <td style="font-weight: 800; color: #34d399; font-family: monospace;">${(Number(b.over_0_5_hit_prob || 0) * 100).toFixed(1)}%</td>
                <td style="font-weight: 800; color: #38bdf8; font-family: monospace;">${(Number(b.over_1_5_hit_prob || 0) * 100).toFixed(1)}%</td>
                <td style="font-size: 11px; color: var(--text-dim); font-family: monospace;">${b.contact_mod} / ${b.babip_mod}</td>
                <td>${confBadge}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderPitchers() {
        const tbody = document.getElementById('pitchers-tbody');
        tbody.innerHTML = '';

        const activePks = new Set(
            games.filter(g => g.date_bucket === activeDate && (activeStage === 'all' || g.stage === activeStage))
                 .map(g => String(g.game_pk))
        );

        const filteredPitchers = pitcherProps.filter(p => activePks.has(String(p.game_pk)));

        if (filteredPitchers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 30px;">No pitcher props for this selection.</td></tr>';
            return;
        }

        filteredPitchers.forEach(p => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="font-weight: 700; color: #fff;">${p.pitcher_name}</td>
                <td>${p.team_name || ''}</td>
                <td style="font-weight: 800; color: #f59e0b; font-family: monospace;">${Number(p.k_line || 4.5).toFixed(1)}</td>
                <td style="font-weight: 600; font-family: monospace;">${Number(p.expected_k || 0).toFixed(2)}</td>
                <td style="color: #34d399; font-weight: 800; font-family: monospace;">${(Number(p.over_prob || 0) * 100).toFixed(1)}%</td>
                <td style="color: #94a3b8; font-weight: 700; font-family: monospace;">${(Number(p.under_prob || 0) * 100).toFixed(1)}%</td>
                <td style="color: #c084fc; font-family: monospace; font-size: 11px;">θ=${p.kalman_theta} (P:${p.kalman_p})</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function setDateFilter(d, btn) {
        activeDate = d;
        btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateCounts();
        renderActiveView();
    }

    function setStageFilter(s, btn) {
        activeStage = s;
        btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderActiveView();
    }

    function setViewMode(v, btn) {
        activeView = v;
        btn.parentElement.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        document.getElementById('view-matchups').style.display = v === 'matchups' ? 'grid' : 'none';
        document.getElementById('view-batters').style.display = v === 'batters' ? 'block' : 'none';
        document.getElementById('view-pitchers').style.display = v === 'pitchers' ? 'block' : 'none';
        
        renderActiveView();
    }

    function renderActiveView() {
        if (activeView === 'matchups') renderMatchups();
        else if (activeView === 'batters') renderBatters();
        else if (activeView === 'pitchers') renderPitchers();
    }

    updateCounts();
    renderActiveView();
</script>
</body>
</html>
"""
    html_output = template.replace("__TICKER_ITEMS__", ticker_content) \
                          .replace("__GAMES_DATA__", json.dumps(games_payload)) \
                          .replace("__BATTER_DATA__", json.dumps(batter_props)) \
                          .replace("__PITCHER_DATA__", json.dumps(pitcher_props))

    return HTMLResponse(content=html_output)
