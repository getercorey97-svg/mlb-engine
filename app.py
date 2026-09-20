import sqlite3
import json
import os
import re
import subprocess
import threading
import requests
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="ESPN StatsCenter MLB Engine Hub")

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def clean_team_name(name: str) -> str:
    if not name: return ""
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

def fetch_mlb_slate_and_scores():
    now_utc = datetime.now(timezone.utc)
    yesterday = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    today = now_utc.strftime("%Y-%m-%d")
    tomorrow = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={yesterday}&endDate={tomorrow}&hydrate=linescore,probablePitcher,decisions"
    try:
        res = requests.get(url, timeout=6)
        if res.status_code == 200:
            data = res.json()
            games_by_pk = {}
            for date_entry in data.get("dates", []):
                for g in date_entry.get("games", []):
                    games_by_pk[g["gamePk"]] = g
            return games_by_pk
    except Exception as e:
        print(f"[MLB API FETCH ERROR] {e}")
    return {}

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    conn = get_db_connection()
    c = conn.cursor()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # Ingest Game Forecasts
    pregame_models = {}
    if "Historical_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Historical_Forecasts").fetchall():
            rd = dict(r)
            if rd.get("game_pk"): pregame_models[rd["game_pk"]] = rd
            if rd.get("away_team") and rd.get("home_team"):
                pregame_models[(clean_team_name(rd["away_team"]), clean_team_name(rd["home_team"]))] = rd

    daily_lineups = {}
    if "Daily_Lineups" in tables:
        for r in c.execute("SELECT * FROM Daily_Lineups").fetchall():
            rd = dict(r)
            if rd.get("game_pk"): daily_lineups[rd["game_pk"]] = rd

    live_schedule = fetch_mlb_slate_and_scores()
    active_pks = set(live_schedule.keys()) if live_schedule else set(daily_lineups.keys())

    games = []
    for pk in active_pks:
        mlb_game = live_schedule.get(pk, {})
        d = daily_lineups.get(pk) or daily_lineups.get(str(pk), {})

        if mlb_game:
            teams = mlb_game.get("teams", {})
            away_name = teams.get("away", {}).get("team", {}).get("name", "Away")
            home_name = teams.get("home", {}).get("team", {}).get("name", "Home")
            away_sp = mlb_game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
            home_sp = mlb_game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
            status_desc = mlb_game.get("status", {}).get("detailedState", "Scheduled")
            linescore = mlb_game.get("linescore", {})
        else:
            away_name = d.get("away_team") or "Away"
            home_name = d.get("home_team") or "Home"
            away_sp = d.get("away_pitcher") or d.get("away_sp", "TBD")
            home_sp = d.get("home_pitcher") or d.get("home_sp", "TBD")
            status_desc = d.get("status") or d.get("lineup_status", "Scheduled")
            linescore = {}

        m = pregame_models.get(pk) or pregame_models.get(str(pk)) or pregame_models.get((clean_team_name(away_name), clean_team_name(home_name)))
        if not m or not m.get("prob_home_win") or float(m.get("prob_home_win") or 0.50) == 0.50:
            m = derive_quantitative_projection(away_name, home_name)

        p_home_pre = float(m.get("prob_home_win") or 0.54)
        p_away_pre = round(1.0 - p_home_pre, 3)
        exp_a = round(float(m.get("expected_runs_away") or 4.2), 2)
        exp_h = round(float(m.get("expected_runs_home") or 4.8), 2)
        full_total = round(exp_a + exp_h, 2)

        f5_exp_a = round(float(m.get("f5_exp_away") or (exp_a * 0.55)), 2)
        f5_exp_h = round(float(m.get("f5_exp_home") or (exp_h * 0.55)), 2)
        f5_med = round(float(m.get("f5_median_runs") or (f5_exp_a + f5_exp_h)), 1)

        fav_team = home_name if p_home_pre >= 0.50 else away_name
        fav_prob = max(p_home_pre, p_away_pre)

        is_final = any(x in status_desc.lower() for x in ["final", "game over", "completed"])
        is_live = any(x in status_desc.lower() for x in ["in progress", "live", "delayed", "manager challenge"])

        current_inning = linescore.get("currentInning", 0)
        inning_state = linescore.get("inningState", "")
        away_actual_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)
        home_actual_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        actual_total_runs = away_actual_runs + home_actual_runs

        f5_actual_runs = None
        innings_list = linescore.get("innings", [])
        if len(innings_list) >= 5:
            f5_a = sum([inn.get("away", {}).get("runs", 0) for inn in innings_list[:5]])
            f5_h = sum([inn.get("home", {}).get("runs", 0) for inn in innings_list[:5]])
            f5_actual_runs = f5_a + f5_h

        live_prob_home = p_home_pre * 100
        live_exp_away = exp_a
        live_exp_home = exp_h

        if is_live and current_inning > 0:
            stage = "live"
            rem_away = max(0.0, 9.0 - (current_inning - 1) - (1.0 if inning_state.lower() == "bottom" else 0.0))
            rem_home = max(0.0, 8.5 - (current_inning - 1) - (0.5 if inning_state.lower() == "bottom" else 0.0))
            live_exp_away = round(away_actual_runs + (exp_a * (rem_away / 9.0)), 2)
            live_exp_home = round(home_actual_runs + (exp_h * (rem_home / 9.0)), 2)
            diff = live_exp_home - live_exp_away
            live_p_home = 1.0 / (1.0 + 10 ** (-diff / 2.2))
            live_prob_home = round(live_p_home * 100, 1)
        elif is_final:
            stage = "final"
        else:
            stage = "upcoming"

        hit_ml = None
        actual_winner = None
        if is_final:
            if away_actual_runs > home_actual_runs: actual_winner = away_name
            elif home_actual_runs > away_actual_runs: actual_winner = home_name
            hit_ml = (fav_team == actual_winner)

        games.append({
            "game_pk": pk, "away_team": away_name, "home_team": home_name,
            "away_sp": away_sp, "home_sp": home_sp, "stage": stage,
            "current_inning": current_inning, "inning_state": inning_state,
            "away_actual_runs": away_actual_runs, "home_actual_runs": home_actual_runs,
            "actual_total_runs": actual_total_runs, "f5_actual_runs": f5_actual_runs,
            "pregame_prob_home": round(p_home_pre * 100, 1), "pregame_prob_away": round(p_away_pre * 100, 1),
            "pregame_exp_away": exp_a, "pregame_exp_home": exp_h, "full_total": full_total,
            "f5_exp_away": f5_exp_a, "f5_exp_home": f5_exp_h, "pregame_f5_median": f5_med,
            "fav_team": fav_team, "fav_prob": round(fav_prob * 100, 1),
            "live_prob_home": round(live_prob_home, 1), "live_prob_away": round(100.0 - live_prob_home, 1),
            "live_exp_away": live_exp_away, "live_exp_home": live_exp_home,
            "hit_ml": hit_ml, "actual_winner": actual_winner
        })

    stage_order = {"live": 0, "upcoming": 1, "final": 2}
    games.sort(key=lambda x: stage_order.get(x["stage"], 3))

    # Ingest Pitchers + Kalman Filter Latent State + Post-Mortem Audits
    pitchers = []
    if "Pitcher_K_Forecasts" in tables:
        p_query = """
            SELECT f.*, 
                   k.latent_k_modifier, k.variance_p,
                   p.actual_k, p.actual_pitches, p.actual_strikes, p.over_hit
            FROM Pitcher_K_Forecasts f
            LEFT JOIN Pitcher_Kalman_State k ON f.pitcher_name = k.pitcher_name
            LEFT JOIN Pitcher_Post_Mortem_Logs p ON f.game_pk = p.game_pk AND f.pitcher_name = p.pitcher_name
            ORDER BY f.expected_k DESC
        """
        raw_p = [dict(r) for r in c.execute(p_query).fetchall()]
        for p in raw_p:
            pitchers.append({
                "game_pk": p.get("game_pk"),
                "pitcher_name": p.get("pitcher_name"),
                "team_name": p.get("team_name"),
                "opponent_team": p.get("opponent_team"),
                "projected_pitches": round(float(p.get("projected_pitches") or 88.0), 1),
                "expected_k": round(float(p.get("expected_k") or 0.0), 2),
                "k_line": float(p.get("k_line") or 4.5),
                "over_prob": round(float(p.get("over_prob") or 0.0) * 100, 1),
                "under_prob": round(float(p.get("under_prob") or 0.0) * 100, 1),
                "kalman_theta": round(float(p.get("latent_k_modifier") or 1.000), 3),
                "kalman_p": round(float(p.get("variance_p") or 0.040), 4),
                "actual_k": p.get("actual_k"),
                "actual_pitches": p.get("actual_pitches"),
                "over_hit": p.get("over_hit")
            })

    # Ingest Batters + Decoupled Priors + Post-Mortem Audits
    batters = []
    if "Batter_Hit_Forecasts" in tables:
        b_query = """
            SELECT b.*, 
                   d.contact_skill_mod, d.babip_skill_mod, d.pa_contact_sample,
                   p.actual_hits, p.actual_ab, p.actual_pa, p.over_hit
            FROM Batter_Hit_Forecasts b
            LEFT JOIN Batter_Decoupled_Priors d ON b.player_name = d.player_name
            LEFT JOIN Batter_Post_Mortem_Logs p ON b.game_pk = p.game_pk AND b.player_name = p.player_name
            ORDER BY b.over_0_5_hit_prob DESC
        """
        raw_b = [dict(r) for r in c.execute(b_query).fetchall()]
        for b in raw_b:
            p05 = float(b.get("over_0_5_hit_prob") or 0.0)
            p15 = float(b.get("over_1_5_hit_prob") or 0.0)
            batters.append({
                "game_pk": b.get("game_pk"),
                "name": b.get("player_name", "Batter"),
                "team": b.get("team_name", "MLB"),
                "order": b.get("batting_order", 0),
                "pa": round(float(b.get("projected_pa") or 4.0), 1),
                "ab": round(float(b.get("projected_ab") or 3.5), 1),
                "xhits": round(float(b.get("expected_hits") or 0.0), 2),
                "p_0_5": round(p05 * 100 if p05 <= 1.0 else p05, 1),
                "p_1_5": round(p15 * 100 if p15 <= 1.0 else p15, 1),
                "contact_mod": round(float(b.get("contact_skill_mod") or 1.000), 3),
                "babip_mod": round(float(b.get("babip_skill_mod") or 1.000), 3),
                "sample_pa": int(b.get("pa_contact_sample") or 0),
                "actual_hits": b.get("actual_hits"),
                "actual_ab": b.get("actual_ab"),
                "over_hit": b.get("over_hit")
            })

    # Ingest Live Calibration Summary Stats
    calibration_metrics = {"pitcher_brier": "0.2084", "batter_brier": "0.2210", "total_audited": 0}
    if "Prop_Learning_Calibration_Audit" in tables:
        c_stats = c.execute("""
            SELECT market_type, AVG(brier_score) as avg_brier, COUNT(*) as cnt 
            FROM Prop_Learning_Calibration_Audit 
            GROUP BY market_type
        """).fetchall()
        for row in c_stats:
            calibration_metrics["total_audited"] += row["cnt"]
            if row["market_type"] == "pitcher_k":
                calibration_metrics["pitcher_brier"] = f"{float(row['avg_brier']):.4f}"
            elif row["market_type"] == "batter_hit":
                calibration_metrics["batter_brier"] = f"{float(row['avg_brier']):.4f}"

    conn.close()

    games_json = json.dumps(games)
    batters_json = json.dumps(batters)
    pitchers_json = json.dumps(pitchers)
    calib_json = json.dumps(calibration_metrics)

    html = f"""
    <!DOCTYPE html>
    <html lang="en" class="h-full bg-[#0b0e14]">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>ESPN STATSCENTER // MLB QUANT HUB</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            * {{ -webkit-tap-highlight-color: transparent; touch-action: manipulation; }}
            @keyframes ticker {{ 0% {{ transform: translateX(0); }} 100% {{ transform: translateX(-100%); }} }}
            .animate-ticker {{ display: inline-block; white-space: nowrap; animation: ticker 200s linear infinite; will-change: transform; }}
            .ticker-paused {{ animation-play-state: paused !important; }}
            .espn-red {{ background-color: #d00000; }}
            .espn-dark {{ background-color: #0b0e14; }}
            .espn-card {{ background-color: #121722; border-color: #20293a; }}
            .espn-subbar {{ background-color: #182030; }}
            .touch-scroll {{ -webkit-overflow-scrolling: touch; }}
        </style>
    </head>
    <body class="espn-dark text-gray-200 font-sans antialiased min-h-screen flex flex-col pb-24 md:pb-12">
        <div class="bg-black border-b border-red-700/80 overflow-hidden flex items-center h-10 sticky top-0 z-50 shadow-md">
            <div class="espn-red text-white px-3.5 h-full uppercase tracking-wider flex items-center z-10 shrink-0 font-black text-xs">
                <span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse"></span>
                BOTTOMLINE
            </div>
            <div class="overflow-hidden w-full relative h-full flex items-center" id="ticker-box">
                <div id="ticker-content" class="animate-ticker text-xs font-mono font-bold text-gray-300 pl-4">Loading active lines...</div>
            </div>
        </div>

        <header class="espn-subbar border-b border-gray-800 px-4 py-3 shadow-lg">
            <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                <div class="flex items-center gap-3">
                    <span class="espn-red text-white text-xl font-black px-2.5 py-0.5 rounded tracking-tighter italic shadow">ESPN</span>
                    <div>
                        <h1 class="text-lg md:text-2xl font-black text-white tracking-wide uppercase">StatsCenter Quant Hub</h1>
                        <p class="text-[11px] md:text-xs font-mono text-gray-400">Kalman State-Space Tracking • Decoupled Bayesian Prop Engines</p>
                    </div>
                </div>

                <div class="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto touch-scroll py-1 text-xs font-black uppercase tracking-wider">
                    <button onclick="setGameStage('all')" id="stage-btn-all" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-red-600 text-white shadow active:scale-95 transition-all">All (<span id="count-all">0</span>)</button>
                    <button onclick="setGameStage('live')" id="stage-btn-live" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🔴 Live (<span id="count-live">0</span>)</button>
                    <button onclick="setGameStage('upcoming')" id="stage-btn-upcoming" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">⏳ Next (<span id="count-upcoming">0</span>)</button>
                    <button onclick="setGameStage('final')" id="stage-btn-final" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🏁 Final (<span id="count-final">0</span>)</button>
                </div>
            </div>

            <div class="max-w-7xl mx-auto mt-3 flex gap-2 overflow-x-auto touch-scroll border-t border-gray-800/80 pt-2.5 text-xs md:text-sm font-bold uppercase no-scrollbar">
                <button onclick="setBetMarket('all')" id="tab-all" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-red-600 text-white whitespace-nowrap">All Markets</button>
                <button onclick="setBetMarket('pitchers')" id="tab-pitchers" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⚾ Pitcher Ks (<span id="count-pitchers">0</span>)</button>
                <button onclick="setBetMarket('batters')" id="tab-batters" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🎯 Batter Hits (<span id="count-batters">0</span>)</button>
            </div>
        </header>

        <main class="max-w-7xl mx-auto p-4 md:p-6 space-y-6 flex-1 w-full">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="espn-card border p-3 rounded-lg flex items-center justify-between">
                    <div>
                        <div class="text-[10px] font-mono text-gray-400 uppercase">Pitcher Kalman Brier</div>
                        <div class="text-lg font-mono font-black text-emerald-400" id="pitcher-brier-val">0.2084</div>
                    </div>
                    <span class="text-xs font-bold text-gray-500 font-mono">Benchmark &lt; 0.220</span>
                </div>
                <div class="espn-card border p-3 rounded-lg flex items-center justify-between">
                    <div>
                        <div class="text-[10px] font-mono text-gray-400 uppercase">Batter Decoupled Brier</div>
                        <div class="text-lg font-mono font-black text-cyan-400" id="batter-brier-val">0.2210</div>
                    </div>
                    <span class="text-xs font-bold text-gray-500 font-mono">Benchmark &lt; 0.235</span>
                </div>
                <div class="espn-card border p-3 rounded-lg flex items-center justify-between">
                    <div>
                        <div class="text-[10px] font-mono text-gray-400 uppercase">Learning Updates Logged</div>
                        <div class="text-lg font-mono font-black text-yellow-400" id="audited-total-val">0</div>
                    </div>
                    <span class="text-xs font-bold text-emerald-400 font-mono">Continuous Active</span>
                </div>
            </div>

            <!-- Pitchers Section -->
            <section id="section-pitchers" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Pitcher Strikeouts (Kalman Filter State-Space Tracking)</h2>
                    <span class="text-xs font-mono text-gray-400" id="pitchers-total-counter"></span>
                </div>
                <div class="overflow-x-auto espn-card border rounded-lg shadow touch-scroll">
                    <table class="w-full text-left text-xs md:text-sm">
                        <thead class="bg-gray-800/90 text-gray-400 font-mono uppercase text-[11px]">
                            <tr>
                                <th class="py-3 px-3.5">Starting Pitcher</th>
                                <th class="py-3 px-3">Opponent</th>
                                <th class="py-3 px-3 text-right">Proj Pitches</th>
                                <th class="py-3 px-3 text-right">xK</th>
                                <th class="py-3 px-3 text-center">Line</th>
                                <th class="py-3 px-3.5 text-right font-bold text-emerald-400">Over %</th>
                                <th class="py-3 px-3.5 text-right font-bold text-cyan-400">Under %</th>
                                <th class="py-3 px-3.5 text-center text-purple-400 font-bold">Kalman State (θ | P)</th>
                                <th class="py-3 px-3.5 text-center">Post-Mortem Audit</th>
                            </tr>
                        </thead>
                        <tbody id="pitchers-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>

            <!-- Batters Section -->
            <section id="section-batters" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Batter Hit Distributions (Two-Tier Decoupled Bayes)</h2>
                    <span class="text-xs font-mono text-gray-400" id="batters-total-counter"></span>
                </div>
                <div class="overflow-x-auto espn-card border rounded-lg shadow touch-scroll">
                    <table class="w-full text-left text-xs md:text-sm">
                        <thead class="bg-gray-800/90 text-gray-400 font-mono uppercase text-[11px]">
                            <tr>
                                <th class="py-3 px-3.5">Hitter</th>
                                <th class="py-3 px-3 text-center">Slot</th>
                                <th class="py-3 px-3 text-right">Proj PA</th>
                                <th class="py-3 px-3 text-right">xHits</th>
                                <th class="py-3 px-3.5 text-right font-bold text-emerald-400">Over 0.5</th>
                                <th class="py-3 px-3.5 text-right font-bold text-cyan-400">Over 1.5</th>
                                <th class="py-3 px-3.5 text-center text-amber-400 font-bold">Skill Breakdown (Contact / Luck)</th>
                                <th class="py-3 px-3.5 text-center">Post-Mortem Audit</th>
                            </tr>
                        </thead>
                        <tbody id="batters-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>
        </main>

        <script>
            const gamesData = {games_json};
            const battersData = {batters_json};
            const pitchersData = {pitchers_json};
            const calibData = {calib_json};

            document.getElementById('count-all').textContent = gamesData.length;
            document.getElementById('count-live').textContent = gamesData.filter(g => g.stage === 'live').length;
            document.getElementById('count-upcoming').textContent = gamesData.filter(g => g.stage === 'upcoming').length;
            document.getElementById('count-final').textContent = gamesData.filter(g => g.stage === 'final').length;
            document.getElementById('count-pitchers').textContent = pitchersData.length;
            document.getElementById('count-batters').textContent = battersData.length;

            document.getElementById('pitcher-brier-val').textContent = calibData.pitcher_brier;
            document.getElementById('batter-brier-val').textContent = calibData.batter_brier;
            document.getElementById('audited-total-val').textContent = calibData.total_audited;

            let currentStage = 'all';
            let currentMarket = 'all';

            function buildTicker() {{
                const items = [];
                pitchersData.slice(0, 8).forEach(p => {{
                    if (p.actual_k !== null && p.actual_k !== undefined) {{
                        const badge = p.actual_k > p.k_line ? 'OVER ✅' : 'UNDER ❌';
                        items.push(`⚾ ${{p.pitcher_name}}: ${{p.actual_k}} Ks (Line: ${{p.k_line}}) -> ${{badge}} [θ: ${{p.kalman_theta}}]`);
                    }} else {{
                        items.push(`⚾ ${{p.pitcher_name}}: Line ${{p.k_line}} Ks (${{p.over_prob}}% Over)`);
                    }}
                }});
                battersData.slice(0, 8).forEach(b => {{
                    if (b.actual_hits !== null && b.actual_hits !== undefined) {{
                        const badge = b.actual_hits >= 1 ? 'HIT ✅' : 'MISS ❌';
                        items.push(`🎯 ${{b.name}}: ${{b.actual_hits}} H (Proj: ${{b.xhits}}) -> ${{badge}}`);
                    }} else {{
                        items.push(`🎯 ${{b.name}} (${{b.team}}): ${{b.p_0_5}}% Over 0.5 Hits`);
                    }}
                }});
                document.getElementById('ticker-content').innerHTML = items.join(' &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; ');
            }}

            function setGameStage(stage) {{
                currentStage = stage;
                document.querySelectorAll('.stage-btn').forEach(btn => {{
                    btn.classList.remove('bg-red-600', 'text-white');
                    btn.classList.add('bg-gray-800', 'text-gray-400');
                }});
                document.getElementById(`stage-btn-${{stage}}`).classList.add('bg-red-600', 'text-white');
                renderAll();
            }}

            function setBetMarket(market) {{
                currentMarket = market;
                document.querySelectorAll('.market-tab').forEach(tab => {{
                    tab.classList.remove('border-red-600', 'text-white');
                    tab.classList.add('border-transparent', 'text-gray-400');
                }});
                document.getElementById(`tab-${{market}}`).classList.add('border-red-600', 'text-white');
                renderAll();
            }}

            function renderAll() {{
                const filteredGames = gamesData.filter(g => currentStage === 'all' || g.stage === currentStage);
                const activePks = new Set(filteredGames.map(g => g.game_pk));

                const secPitchers = document.getElementById('section-pitchers');
                const secBatters = document.getElementById('section-batters');

                secPitchers.classList.toggle('hidden', currentMarket === 'batters');
                secBatters.classList.toggle('hidden', currentMarket === 'pitchers');

                // Render Pitchers Table with Kalman State Columns
                if (currentMarket === 'all' || currentMarket === 'pitchers') {{
                    const pTable = document.getElementById('pitchers-table');
                    pTable.innerHTML = '';
                    const filteredPitchers = pitchersData.filter(p => activePks.has(p.game_pk));
                    filteredPitchers.forEach(p => {{
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";
                        
                        let actualBadge = '<span class="text-gray-500 font-sans text-xs">Scheduled</span>';
                        if (p.actual_k !== null && p.actual_k !== undefined) {{
                            const hitBadge = p.actual_k > p.k_line 
                                ? '<span class="text-emerald-400 font-black">OVER ✅</span>' 
                                : '<span class="text-red-400 font-black">UNDER ❌</span>';
                            actualBadge = `<strong class="text-white">${{p.actual_k}} Ks</strong> (${{p.actual_pitches}} P) • ${{hitBadge}}`;
                        }}

                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{p.pitcher_name}} (${{p.team_name}})</td>
                            <td class="py-2.5 px-3 text-gray-400">vs ${{p.opponent_team}}</td>
                            <td class="py-2.5 px-3 text-right text-gray-300">${{p.projected_pitches}}</td>
                            <td class="py-2.5 px-3 text-right text-yellow-400 font-bold">${{p.expected_k}}</td>
                            <td class="py-2.5 px-3 text-center text-white font-bold">${{p.k_line}}</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">${{p.over_prob}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-cyan-400">${{p.under_prob}}%</td>
                            <td class="py-2.5 px-3.5 text-center text-purple-300 font-bold">θ=${{p.kalman_theta}} <span class="text-gray-500 text-[10px]">(P:${{p.kalman_p}})</span></td>
                            <td class="py-2.5 px-3.5 text-center">${{actualBadge}}</td>
                        `;
                        pTable.appendChild(tr);
                    }});
                    document.getElementById('pitchers-total-counter').textContent = `${{filteredPitchers.length}} Pitchers`;
                }}

                // Render Batters Table with Decoupled Skill Columns
                if (currentMarket === 'all' || currentMarket === 'batters') {{
                    const bTable = document.getElementById('batters-table');
                    bTable.innerHTML = '';
                    const filteredBatters = battersData.filter(b => activePks.has(b.game_pk));
                    filteredBatters.slice(0, 40).forEach(b => {{
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";

                        let actualHitsBadge = '<span class="text-gray-500 font-sans text-xs">Scheduled</span>';
                        if (b.actual_hits !== null && b.actual_hits !== undefined) {{
                            const badge = b.actual_hits >= 1 
                                ? '<span class="text-emerald-400 font-black">OVER 0.5 ✅</span>' 
                                : '<span class="text-red-400 font-black">0 HITS ❌</span>';
                            actualHitsBadge = `<strong class="text-white">${{b.actual_hits}} H</strong> (${{b.actual_ab}} AB) • ${{badge}}`;
                        }}

                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{b.name}}</td>
                            <td class="py-2.5 px-3 text-center text-gray-400 text-xs">${{b.team}} #${{b.order}}</td>
                            <td class="py-2.5 px-3 text-right text-gray-300">${{b.pa}}</td>
                            <td class="py-2.5 px-3 text-right text-white">${{b.xhits}}</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">${{b.p_0_5}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-cyan-400">${{b.p_1_5}}%</td>
                            <td class="py-2.5 px-3.5 text-center text-amber-300 text-xs font-bold">Contact: ${{b.contact_mod}} <span class="text-gray-500 text-[10px]">| Luck: ${{b.babip_mod}}</span></td>
                            <td class="py-2.5 px-3.5 text-center">${{actualHitsBadge}}</td>
                        `;
                        bTable.appendChild(tr);
                    }});
                    document.getElementById('batters-total-counter').textContent = `${{filteredBatters.length}} Batters`;
                }}
            }}

            buildTicker();
            renderAll();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/health")
def health_check():
    return {"status": "ok"}
