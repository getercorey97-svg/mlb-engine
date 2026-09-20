import sqlite3
import json
import os
import requests
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="ESPN StatsCenter MLB Engine Hub")

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

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
                    games_by_pk[str(g["gamePk"])] = g
            return games_by_pk
    except Exception as e:
        print(f"[MLB API FETCH ERROR] {e}")
    return {}

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    conn = get_db_connection()
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # 1. Load Pre-Game Forecasts with Dual-Key Mapping (String + Int)
    pregame_models = {}
    if "Model_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Model_Forecasts").fetchall():
            row_dict = dict(r)
            pk = row_dict.get("game_pk")
            if pk is not None:
                pregame_models[str(pk)] = row_dict

    daily_lineups = {}
    if "Daily_Lineups" in tables:
        for r in c.execute("SELECT * FROM Daily_Lineups").fetchall():
            row_dict = dict(r)
            pk = row_dict.get("game_pk")
            if pk is not None:
                daily_lineups[str(pk)] = row_dict

    live_schedule = fetch_mlb_slate_and_scores()

    # Scope strictly to active 3-day window to prevent historical archives from inflating Upcoming counts
    active_pks = set(live_schedule.keys()) if live_schedule else set(daily_lineups.keys())

    games = []
    for pk_str in active_pks:
        mlb_game = live_schedule.get(pk_str, {})
        m = pregame_models.get(pk_str, {})
        d = daily_lineups.get(pk_str, {})

        if mlb_game:
            teams = mlb_game.get("teams", {})
            away_name = teams.get("away", {}).get("team", {}).get("name", "Away")
            home_name = teams.get("home", {}).get("team", {}).get("name", "Home")
            away_sp = mlb_game.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
            home_sp = mlb_game.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
            status_desc = mlb_game.get("status", {}).get("detailedState", "Scheduled")
            linescore = mlb_game.get("linescore", {})
        else:
            away_name = d.get("away_team") or m.get("away_team") or "Away"
            home_name = d.get("home_team") or m.get("home_team") or "Home"
            away_sp = d.get("away_sp", "TBD")
            home_sp = d.get("home_sp", "TBD")
            status_desc = d.get("lineup_status", "Scheduled")
            linescore = {}

        p_home_pre = float(m.get("prob_home_win") or 0.50)
        p_away_pre = 1.0 - p_home_pre
        exp_a = float(m.get("expected_runs_away") or 4.5)
        exp_h = float(m.get("expected_runs_home") or 4.5)
        f5_med = float(m.get("f5_median_runs") or 4.5)
        fav_team = home_name if p_home_pre >= 0.50 else away_name
        fav_prob = max(p_home_pre, p_away_pre)

        is_final = any(x in status_desc.lower() for x in ["final", "game over", "completed"])
        is_live = any(x in status_desc.lower() for x in ["in progress", "live", "delayed", "manager challenge"])

        current_inning = linescore.get("currentInning", 0)
        inning_state = linescore.get("inningState", "")
        away_actual_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)
        home_actual_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)

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
            if away_actual_runs > home_actual_runs:
                actual_winner = away_name
            elif home_actual_runs > away_actual_runs:
                actual_winner = home_name
            hit_ml = (fav_team == actual_winner)

        games.append({
            "game_pk": pk_str,
            "away_team": away_name,
            "home_team": home_name,
            "away_sp": away_sp,
            "home_sp": home_sp,
            "status_desc": status_desc,
            "stage": stage,
            "current_inning": current_inning,
            "inning_state": inning_state,
            "away_actual_runs": away_actual_runs,
            "home_actual_runs": home_actual_runs,
            "f5_actual_runs": f5_actual_runs,
            "pregame_prob_home": round(p_home_pre * 100, 1),
            "pregame_prob_away": round(p_away_pre * 100, 1),
            "pregame_exp_away": round(exp_a, 2),
            "pregame_exp_home": round(exp_h, 2),
            "pregame_f5_median": round(f5_med, 1),
            "fav_team": fav_team,
            "fav_prob": round(fav_prob * 100, 1),
            "live_prob_home": round(live_prob_home, 1),
            "live_prob_away": round(100.0 - live_prob_home, 1),
            "live_exp_away": live_exp_away,
            "live_exp_home": live_exp_home,
            "hit_ml": hit_ml,
            "actual_winner": actual_winner
        })

    stage_order = {"live": 0, "upcoming": 1, "final": 2}
    games.sort(key=lambda x: stage_order.get(x["stage"], 3))

    # 2. Pitcher Strikeout Props with Post-Mortem Logs
    pitchers = []
    if "Pitcher_K_Forecasts" in tables:
        raw_k = c.execute("""
            SELECT f.*, p.actual_k, p.actual_pitches, p.over_hit
            FROM Pitcher_K_Forecasts f
            LEFT JOIN Pitcher_Post_Mortem_Logs p ON f.game_pk = p.game_pk AND f.pitcher_name = p.pitcher_name
            ORDER BY f.expected_k DESC
        """).fetchall() if "Pitcher_Post_Mortem_Logs" in tables else c.execute("SELECT * FROM Pitcher_K_Forecasts ORDER BY expected_k DESC").fetchall()
        for r in raw_k:
            d_k = dict(r)
            pitchers.append({
                "game_pk": str(d_k.get("game_pk")),
                "name": d_k.get("pitcher_name", "Pitcher"),
                "team": d_k.get("team_name", "MLB"),
                "opp": d_k.get("opponent_team", "Opp"),
                "pitches": round(float(d_k.get("projected_pitches") or 85.0), 1),
                "xk": round(float(d_k.get("expected_k") or 4.5), 2),
                "k_line": round(float(d_k.get("k_line") or 4.5), 1),
                "over_prob": round(float(d_k.get("over_prob") or 0.5) * 100, 1),
                "under_prob": round(float(d_k.get("under_prob") or 0.5) * 100, 1),
                "actual_k": d_k.get("actual_k"),
                "actual_pitches": d_k.get("actual_pitches")
            })

    # 3. Batter Hit Props with Post-Mortem Logs
    batters = []
    if "Batter_Hit_Forecasts" in tables:
        raw_b = c.execute("""
            SELECT b.*, p.actual_hits, p.actual_ab, p.hit_over_0_5
            FROM Batter_Hit_Forecasts b
            LEFT JOIN Batter_Post_Mortem_Logs p ON b.game_pk = p.game_pk AND b.player_name = p.player_name
            ORDER BY b.over_0_5_hit_prob DESC
        """).fetchall() if "Batter_Post_Mortem_Logs" in tables else c.execute("SELECT * FROM Batter_Hit_Forecasts ORDER BY over_0_5_hit_prob DESC").fetchall()
        for r in raw_b:
            d_b = dict(r)
            batters.append({
                "game_pk": str(d_b.get("game_pk")),
                "name": d_b.get("player_name", "Batter"),
                "team": d_b.get("team_name", "MLB"),
                "order": d_b.get("batting_order", 0),
                "pa": round(float(d_b.get("projected_pa") or 4.0), 1),
                "ab": round(float(d_b.get("projected_ab") or 3.5), 1),
                "xhits": round(float(d_b.get("expected_hits") or 0.0), 2),
                "p_0_5": round(float(d_b.get("over_0_5_hit_prob") or 0.0) * 100, 1),
                "p_1_5": round(float(d_b.get("over_1_5_hit_prob") or 0.0) * 100, 1),
                "actual_hits": d_b.get("actual_hits"),
                "actual_ab": d_b.get("actual_ab")
            })

    # 4. Correlated SGP Edges
    sgps = []
    if "Correlated_Market_Forecasts" in tables:
        for s in c.execute("SELECT * FROM Correlated_Market_Forecasts ORDER BY correlation_edge DESC").fetchall():
            d_s = dict(s)
            sgps.append({
                "game_pk": str(d_s.get("game_pk")),
                "matchup": f"{d_s.get('away_team')} @ {d_s.get('home_team')}",
                "type": d_s.get("sgp_type"),
                "leg_1": d_s.get("leg_1"),
                "leg_2": d_s.get("leg_2"),
                "joint": round(float(d_s.get("joint_prob") or 0.0) * 100, 1),
                "indep": round(float(d_s.get("uncorrelated_prob") or 0.0) * 100, 1),
                "edge": round(float(d_s.get("correlation_edge") or 0.0) * 100, 1)
            })

    conn.close()

    games_json = json.dumps(games)
    pitchers_json = json.dumps(pitchers)
    batters_json = json.dumps(batters)
    sgps_json = json.dumps(sgps)

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
            @keyframes ticker {{
                0% {{ transform: translateX(0); }}
                100% {{ transform: translateX(-100%); }}
            }}
            .animate-ticker {{
                display: inline-block;
                white-space: nowrap;
                animation: ticker 120s linear infinite;
                will-change: transform;
            }}
            .ticker-paused {{ animation-play-state: paused !important; }}
            .espn-red {{ background-color: #d00000; }}
            .espn-dark {{ background-color: #0b0e14; }}
            .espn-card {{ background-color: #121722; border-color: #20293a; }}
            .espn-subbar {{ background-color: #182030; }}
            ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
            ::-webkit-scrollbar-thumb {{ background: #2b3548; border-radius: 4px; }}
            .touch-scroll {{ -webkit-overflow-scrolling: touch; }}
        </style>
    </head>
    <body class="espn-dark text-gray-200 font-sans antialiased min-h-screen flex flex-col selection:bg-red-600 selection:text-white pb-20">
        <!-- Top ESPN BottomLine Scrolling Ticker -->
        <div class="bg-black border-b border-red-700/80 overflow-hidden flex items-center h-10 sticky top-0 z-50 shadow-md">
            <div class="espn-red text-white px-3.5 h-full uppercase tracking-wider flex items-center z-10 shrink-0 font-black text-xs">
                <span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse"></span>
                BOTTOMLINE
            </div>
            <div class="overflow-hidden w-full relative h-full flex items-center" id="ticker-box">
                <div id="ticker-content" class="animate-ticker text-xs font-mono font-bold text-gray-300 pl-4">
                    Loading slate telemetry...
                </div>
            </div>
        </div>

        <!-- Master Navigation Header -->
        <header class="espn-subbar border-b border-gray-800 px-4 py-3 shadow-lg">
            <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                <div class="flex items-center gap-3">
                    <span class="espn-red text-white text-xl font-black px-2.5 py-0.5 rounded tracking-tighter italic shadow">ESPN</span>
                    <div>
                        <h1 class="text-lg md:text-2xl font-black text-white tracking-wide uppercase">StatsCenter Quant Hub</h1>
                        <p class="text-[11px] md:text-xs font-mono text-gray-400">Live In-Game Deduction • Pre-Match vs Post-Mortem Audit</p>
                    </div>
                </div>

                <!-- Global Game Stages Filter Buttons -->
                <div class="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto touch-scroll py-1 text-xs font-black uppercase tracking-wider">
                    <button onclick="setGameStage('all')" id="stage-btn-all" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-red-600 text-white shadow active:scale-95 transition-all">All (<span id="count-all">0</span>)</button>
                    <button onclick="setGameStage('live')" id="stage-btn-live" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🔴 Live (<span id="count-live">0</span>)</button>
                    <button onclick="setGameStage('upcoming')" id="stage-btn-upcoming" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">⏳ Upcoming (<span id="count-upcoming">0</span>)</button>
                    <button onclick="setGameStage('final')" id="stage-btn-final" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🏁 Final (<span id="count-final">0</span>)</button>
                </div>
            </div>

            <!-- Bet Market Navigation Tabs -->
            <div class="max-w-7xl mx-auto mt-3 flex gap-2 overflow-x-auto touch-scroll border-t border-gray-800/80 pt-2.5 text-xs md:text-sm font-bold uppercase">
                <button onclick="setBetMarket('all')" id="tab-all" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-red-600 text-white whitespace-nowrap">All Markets</button>
                <button onclick="setBetMarket('matchups')" id="tab-matchups" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🏟️ Matchups & F5</button>
                <button onclick="setBetMarket('pitchers')" id="tab-pitchers" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⚾ Pitcher Ks (<span id="count-pitchers">0</span>)</button>
                <button onclick="setBetMarket('batters')" id="tab-batters" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🎯 Batter Hits (<span id="count-batters">0</span>)</button>
                <button onclick="setBetMarket('sgp')" id="tab-sgp" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⚡ SGPs</button>
                <button onclick="setBetMarket('scorecard')" id="tab-scorecard" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">📊 Diagnostic Scorecard</button>
            </div>
        </header>

        <!-- Main Workspace -->
        <main class="max-w-7xl mx-auto p-4 md:p-6 space-y-6 flex-1 w-full">
            <div class="flex justify-between items-center bg-gray-900/90 border border-gray-800 px-3.5 py-2.5 rounded-lg text-xs">
                <span class="text-gray-400">View: <strong id="filter-label" class="text-yellow-400 uppercase font-mono font-bold tracking-wide">All Slates • All Markets</strong></span>
                <span class="font-mono text-gray-500 text-[11px]">Database: <strong class="text-emerald-400">mlb_engine.db</strong></span>
            </div>

            <!-- Section 1: Slate Matchups -->
            <section id="section-games" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Full Game & First 5 (F5) Board</h2>
                    <span class="text-xs font-mono text-gray-400" id="games-total-counter"></span>
                </div>
                <div id="games-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5"></div>
            </section>

            <!-- Section 2: Pitcher Strikeouts -->
            <section id="section-pitchers" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Pitcher Strikeout Props & Post-Mortem Audit</h2>
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
                                <th class="py-3 px-3.5 text-center">Outcome Audit</th>
                            </tr>
                        </thead>
                        <tbody id="pitchers-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>

            <!-- Section 3: Batter Hits -->
            <section id="section-batters" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Batter Contact Hit Props & Post-Mortem Audit</h2>
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
                                <th class="py-3 px-3.5 text-center">Outcome Audit</th>
                            </tr>
                        </thead>
                        <tbody id="batters-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>

            <!-- Section 4: SGPs -->
            <section id="section-sgp" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Cross-Market Joint Probability Edges</h2>
                </div>
                <div class="overflow-x-auto espn-card border rounded-lg shadow touch-scroll">
                    <table class="w-full text-left text-xs md:text-sm">
                        <thead class="bg-gray-800/90 text-gray-400 font-mono uppercase text-[11px]">
                            <tr>
                                <th class="py-3 px-3.5">Matchup</th>
                                <th class="py-3 px-3">Type</th>
                                <th class="py-3 px-3">Leg 1</th>
                                <th class="py-3 px-3">Leg 2</th>
                                <th class="py-3 px-3.5 text-right">Joint</th>
                                <th class="py-3 px-3.5 text-right">Indep</th>
                                <th class="py-3 px-3.5 text-right font-bold text-emerald-400">Edge</th>
                            </tr>
                        </thead>
                        <tbody id="sgp-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>

            <!-- Section 5: Diagnostic Scorecard -->
            <section id="section-scorecard" class="space-y-4">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Aggregate Diagnostic Accuracy Scorecard</h2>
                </div>
                <div id="scorecard-content" class="grid grid-cols-1 md:grid-cols-3 gap-4"></div>
            </section>
        </main>

        <script>
            const gamesData = {games_json};
            const pitchersData = {pitchers_json};
            const battersData = {batters_json};
            const sgpData = {sgps_json};

            let currentStage = 'all';
            let currentMarket = 'all';

            document.getElementById('count-all').textContent = gamesData.length;
            document.getElementById('count-live').textContent = gamesData.filter(g => g.stage === 'live').length;
            document.getElementById('count-upcoming').textContent = gamesData.filter(g => g.stage === 'upcoming').length;
            document.getElementById('count-final').textContent = gamesData.filter(g => g.stage === 'final').length;
            document.getElementById('count-pitchers').textContent = pitchersData.length;
            document.getElementById('count-batters').textContent = battersData.length;

            function buildTicker() {{
                const items = [];
                // Live games first
                gamesData.filter(g => g.stage === 'live').forEach(g => {{
                    items.push(`🔴 LIVE: ${{g.away_team}} ${{g.away_actual_runs}}, ${{g.home_team}} ${{g.home_actual_runs}} (${{g.inning_state}} ${{g.current_inning}}) | Live Win: ${{g.home_team}} ${{g.live_prob_home}}% | Proj Runs: ${{g.live_exp_away}} - ${{g.live_exp_home}}`);
                }});
                // Finals with hit/miss audit
                gamesData.filter(g => g.stage === 'final').forEach(g => {{
                    const badge = g.hit_ml ? 'HIT ✅' : 'MISS ❌';
                    items.push(`🏁 FINAL: ${{g.away_team}} ${{g.away_actual_runs}}, ${{g.home_team}} ${{g.home_actual_runs}} | Fav: ${{g.fav_team}} (${{g.fav_prob}}%) -> ${{badge}}`);
                }});
                // Upcoming games
                gamesData.filter(g => g.stage === 'upcoming').forEach(g => {{
                    items.push(`⏳ UPCOMING: ${{g.away_team}} @ ${{g.home_team}} | Fav: ${{g.fav_team}} (${{g.fav_prob}}%) | F5 Line: ${{g.pregame_f5_median}}`);
                }});
                // Top props
                pitchersData.slice(0, 6).forEach(p => {{
                    items.push(`⚾ ${{p.name}} (${{p.team}}): Line ${{p.k_line}} Ks (${{p.over_prob}}% Over)`);
                }});
                battersData.slice(0, 6).forEach(b => {{
                    items.push(`🎯 ${{b.name}} (${{b.team}} #${{b.order}}): ${{b.p_0_5}}% Over 0.5 Hits`);
                }});

                const content = items.length > 0 ? items.join(' &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; ') : 'Ingesting current slate telemetry...';
                document.getElementById('ticker-content').innerHTML = content;
            }}

            const tickerBox = document.getElementById('ticker-box');
            const tickerText = document.getElementById('ticker-content');
            tickerBox.addEventListener('touchstart', () => tickerText.classList.add('ticker-paused'), {{passive: true}});
            tickerBox.addEventListener('touchend', () => tickerText.classList.remove('ticker-paused'), {{passive: true}});

            function setGameStage(stage) {{
                currentStage = stage;
                document.querySelectorAll('.stage-btn').forEach(btn => {{
                    btn.classList.remove('bg-red-600', 'text-white');
                    btn.classList.add('bg-gray-800', 'text-gray-400');
                }});
                const active = document.getElementById(`stage-btn-${{stage}}`);
                active.classList.add('bg-red-600', 'text-white');
                active.classList.remove('bg-gray-800', 'text-gray-400');
                renderAll();
            }}

            function setBetMarket(market) {{
                currentMarket = market;
                document.querySelectorAll('.market-tab').forEach(tab => {{
                    tab.classList.remove('border-red-600', 'text-white');
                    tab.classList.add('border-transparent', 'text-gray-400');
                }});
                const active = document.getElementById(`tab-${{market}}`);
                active.classList.add('border-red-600', 'text-white');
                active.classList.remove('border-transparent', 'text-gray-400');
                renderAll();
            }}

            function renderAll() {{
                document.getElementById('filter-label').textContent = `${{currentStage}} Slates • ${{currentMarket}} Market`;

                const filteredGames = gamesData.filter(g => currentStage === 'all' || g.stage === currentStage);
                const activePks = new Set(filteredGames.map(g => g.game_pk));

                // 1. Games Grid
                const secGames = document.getElementById('section-games');
                if (currentMarket !== 'all' && currentMarket !== 'matchups') {{
                    secGames.classList.add('hidden');
                }} else {{
                    secGames.classList.remove('hidden');
                    const container = document.getElementById('games-grid');
                    container.innerHTML = '';
                    filteredGames.forEach(g => {{
                        const card = document.createElement('div');
                        card.className = "espn-card border rounded-lg p-3.5 shadow-md flex flex-col justify-between";

                        if (g.stage === 'live') {{
                            card.innerHTML = `
                                <div>
                                    <div class="flex justify-between items-center mb-2">
                                        <span class="bg-red-600 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-wider animate-pulse flex items-center gap-1">
                                            <span class="w-1.5 h-1.5 rounded-full bg-white animate-ping"></span>
                                            LIVE: ${{g.inning_state}} ${{g.current_inning}}
                                        </span>
                                        <span class="font-mono text-xs text-yellow-400 font-bold">F5 Line: ${{g.pregame_f5_median}}</span>
                                    </div>
                                    <div class="flex justify-between items-baseline mb-2">
                                        <h3 class="text-base font-black text-white uppercase">${{g.away_team}} @ ${{g.home_team}}</h3>
                                        <span class="text-lg font-mono font-black text-yellow-400">${{g.away_actual_runs}} - ${{g.home_actual_runs}}</span>
                                    </div>
                                    <div class="bg-black/60 p-2.5 rounded border border-red-900/60 mb-3 space-y-1 text-xs">
                                        <div class="text-[10px] font-mono text-red-400 uppercase font-bold tracking-wider">In-Game Bayesian Deduction</div>
                                        <div class="flex justify-between">
                                            <span class="text-gray-400">Live Win Expectancy:</span>
                                            <span class="font-bold text-white font-mono">${{g.home_team}} (${{g.live_prob_home}}%)</span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-gray-400">Live Projected Runs:</span>
                                            <span class="font-mono text-yellow-300 font-bold">${{g.live_exp_away}} - ${{g.live_exp_home}}</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="bg-gray-900/70 p-2 rounded border border-gray-800 text-[11px] flex justify-between font-mono text-gray-400">
                                    <span>Pregame ML: ${{g.fav_team}} (${{g.fav_prob}}%)</span>
                                    <span>Exp: ${{g.pregame_exp_away}} - ${{g.pregame_exp_home}}</span>
                                </div>
                            `;
                        }} else if (g.stage === 'final') {{
                            const verdictBadge = g.hit_ml 
                                ? '<span class="bg-emerald-600 text-white px-2 py-0.5 rounded text-[10px] font-black">ML HIT ✅</span>'
                                : '<span class="bg-red-700 text-white px-2 py-0.5 rounded text-[10px] font-black">ML MISS ❌</span>';
                            const f5Info = g.f5_actual_runs !== null 
                                ? `Actual F5: <strong class="text-white">${{g.f5_actual_runs}}</strong> (Line: ${{g.pregame_f5_median}})`
                                : `F5 Line: ${{g.pregame_f5_median}}`;

                            card.innerHTML = `
                                <div>
                                    <div class="flex justify-between items-center mb-2">
                                        <span class="bg-blue-600 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-wider">FINAL</span>
                                        ${{verdictBadge}}
                                    </div>
                                    <div class="flex justify-between items-baseline mb-2">
                                        <h3 class="text-base font-black text-white uppercase">${{g.away_team}} @ ${{g.home_team}}</h3>
                                        <span class="text-lg font-mono font-black text-white">${{g.away_actual_runs}} - ${{g.home_actual_runs}}</span>
                                    </div>
                                    <div class="bg-black/60 p-2.5 rounded border border-gray-800 mb-3 space-y-1.5 text-xs">
                                        <div class="text-[10px] font-mono text-gray-400 uppercase font-bold tracking-wider">Pre-Match vs Post-Mortem Outcome</div>
                                        <div class="flex justify-between">
                                            <span class="text-gray-400">Predicted Favorite:</span>
                                            <span class="font-bold ${{g.hit_ml ? 'text-emerald-400' : 'text-red-400'}}">${{g.fav_team}} (${{g.fav_prob}}%)</span>
                                        </div>
                                        <div class="flex justify-between">
                                            <span class="text-gray-400">Actual Winner:</span>
                                            <span class="font-bold text-white">${{g.actual_winner}}</span>
                                        </div>
                                        <div class="flex justify-between text-[11px] font-mono text-gray-400 pt-1 border-t border-gray-800">
                                            <span>Exp Runs: ${{g.pregame_exp_away}} - ${{g.pregame_exp_home}}</span>
                                            <span>${{f5Info}}</span>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }} else {{
                            card.innerHTML = `
                                <div>
                                    <div class="flex justify-between items-center mb-2">
                                        <span class="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[10px] font-black tracking-wider">UPCOMING</span>
                                        <span class="font-mono text-xs text-yellow-400 font-bold">F5 Total Line: ${{g.pregame_f5_median}}</span>
                                    </div>
                                    <h3 class="text-base font-black text-white uppercase tracking-tight mb-1">${{g.away_team}} @ ${{g.home_team}}</h3>
                                    <div class="text-xs text-gray-400 space-y-0.5 mb-3">
                                        <div class="truncate">SP: <span class="text-gray-200">${{g.away_sp}}</span> vs <span class="text-gray-200">${{g.home_sp}}</span></div>
                                        <div>Projected Runs: <strong class="text-white font-mono">${{g.pregame_exp_away}} - ${{g.pregame_exp_home}}</strong></div>
                                    </div>
                                </div>
                                <div class="bg-black/50 p-2.5 rounded border border-gray-800/80 space-y-1 text-xs">
                                    <div class="flex justify-between items-center">
                                        <span class="text-gray-400">Model Favorite:</span>
                                        <span class="font-bold text-emerald-400">${{g.fav_team}} (${{g.fav_prob}}%)</span>
                                    </div>
                                    <div class="flex justify-between font-mono text-[11px] text-gray-400">
                                        <span>Away: ${{g.pregame_prob_away}}%</span>
                                        <span>Home: ${{g.pregame_prob_home}}%</span>
                                    </div>
                                </div>
                            `;
                        }}
                        container.appendChild(card);
                    }});
                    document.getElementById('games-total-counter').textContent = `${{filteredGames.length}} Matchups`;
                }}

                // 2. Pitchers Table
                const secPitchers = document.getElementById('section-pitchers');
                if (currentMarket !== 'all' && currentMarket !== 'pitchers') {{
                    secPitchers.classList.add('hidden');
                }} else {{
                    secPitchers.classList.remove('hidden');
                    const pTable = document.getElementById('pitchers-table');
                    pTable.innerHTML = '';
                    const filteredPitchers = pitchersData.filter(p => activePks.has(p.game_pk));
                    filteredPitchers.forEach(p => {{
                        let badge = '<span class="text-gray-500 font-mono text-xs">Scheduled</span>';
                        if (p.actual_k !== null && p.actual_k !== undefined) {{
                            badge = p.actual_k > p.k_line 
                                ? `<strong class="text-emerald-400 font-bold">${{p.actual_k}} Ks (OVER ✅)</strong>`
                                : `<strong class="text-red-400 font-bold">${{p.actual_k}} Ks (UNDER ❌)</strong>`;
                        }}
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";
                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{p.name}} (${{p.team}})</td>
                            <td class="py-2.5 px-3 text-gray-400">vs ${{p.opp}}</td>
                            <td class="py-2.5 px-3 text-right text-gray-300">${{p.pitches}}</td>
                            <td class="py-2.5 px-3 text-right text-yellow-400 font-bold">${{p.xk}}</td>
                            <td class="py-2.5 px-3 text-center text-white font-bold">${{p.k_line}}</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">${{p.over_prob}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-cyan-400">${{p.under_prob}}%</td>
                            <td class="py-2.5 px-3.5 text-center">${{badge}}</td>
                        `;
                        pTable.appendChild(tr);
                    }});
                    document.getElementById('pitchers-total-counter').textContent = `${{filteredPitchers.length}} Pitchers`;
                }}

                // 3. Batters Table
                const secBatters = document.getElementById('section-batters');
                if (currentMarket !== 'all' && currentMarket !== 'batters') {{
                    secBatters.classList.add('hidden');
                }} else {{
                    secBatters.classList.remove('hidden');
                    const bTable = document.getElementById('batters-table');
                    bTable.innerHTML = '';
                    const filteredBatters = battersData.filter(b => activePks.has(b.game_pk));
                    filteredBatters.slice(0, 30).forEach(b => {{
                        let badge = '<span class="text-gray-500 font-mono text-xs">Scheduled</span>';
                        if (b.actual_hits !== null && b.actual_hits !== undefined) {{
                            badge = b.actual_hits >= 1 
                                ? `<strong class="text-emerald-400 font-bold">${{b.actual_hits}} H (${{b.actual_ab}} AB) • OVER ✅</strong>`
                                : `<strong class="text-red-400 font-bold">0 H (${{b.actual_ab}} AB) • UNDER ❌</strong>`;
                        }}
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";
                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{b.name}} (${{b.team}})</td>
                            <td class="py-2.5 px-3 text-center text-gray-400 text-xs">#${{b.order}}</td>
                            <td class="py-2.5 px-3 text-right text-gray-300">${{b.pa}}</td>
                            <td class="py-2.5 px-3 text-right text-white">${{b.xhits}}</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">${{b.p_0_5}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-cyan-400">${{b.p_1_5}}%</td>
                            <td class="py-2.5 px-3.5 text-center">${{badge}}</td>
                        `;
                        bTable.appendChild(tr);
                    }});
                    document.getElementById('batters-total-counter').textContent = `${{filteredBatters.length}} Batters`;
                }}

                // 4. SGPs Table
                const secSgp = document.getElementById('section-sgp');
                if (currentMarket !== 'all' && currentMarket !== 'sgp') {{
                    secSgp.classList.add('hidden');
                }} else {{
                    secSgp.classList.remove('hidden');
                    const sTable = document.getElementById('sgp-table');
                    sTable.innerHTML = '';
                    const filteredSgps = sgpData.filter(s => activePks.has(s.game_pk));
                    filteredSgps.slice(0, 20).forEach(s => {{
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";
                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{s.matchup}}</td>
                            <td class="py-2.5 px-3 text-[11px] text-yellow-400 whitespace-nowrap">${{s.type}}</td>
                            <td class="py-2.5 px-3 text-gray-200 whitespace-nowrap">${{s.leg_1}}</td>
                            <td class="py-2.5 px-3 text-gray-200 whitespace-nowrap">${{s.leg_2}}</td>
                            <td class="py-2.5 px-3.5 text-right text-white">${{s.joint}}%</td>
                            <td class="py-2.5 px-3.5 text-right text-gray-400">${{s.indep}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">+${{s.edge}}%</td>
                        `;
                        sTable.appendChild(tr);
                    }});
                }}

                // 5. Scorecard Summary
                const secScorecard = document.getElementById('section-scorecard');
                if (currentMarket !== 'all' && currentMarket !== 'scorecard') {{
                    secScorecard.classList.add('hidden');
                }} else {{
                    secScorecard.classList.remove('hidden');
                    const completed = gamesData.filter(g => g.stage === 'final' && g.hit_ml !== null);
                    const mlHits = completed.filter(g => g.hit_ml === true).length;
                    const mlTotal = completed.length;
                    const mlPct = mlTotal > 0 ? ((mlHits / mlTotal) * 100).toFixed(1) : "0.0";

                    const scoredK = pitchersData.filter(p => p.actual_k !== null && p.actual_k !== undefined);
                    const kHits = scoredK.filter(p => (p.actual_k > p.k_line && p.over_prob >= 50) || (p.actual_k <= p.k_line && p.under_prob > 50)).length;
                    const kTotal = scoredK.length;
                    const kPct = kTotal > 0 ? ((kHits / kTotal) * 100).toFixed(1) : "0.0";

                    const scoredB = battersData.filter(b => b.actual_hits !== null && b.actual_hits !== undefined);
                    const bHits = scoredB.filter(b => b.actual_hits >= 1).length;
                    const bTotal = scoredB.length;
                    const bPct = bTotal > 0 ? ((bHits / bTotal) * 100).toFixed(1) : "0.0";

                    document.getElementById('scorecard-content').innerHTML = `
                        <div class="espn-card border rounded-lg p-4 shadow">
                            <div class="text-xs text-gray-400 uppercase font-mono mb-1">Outright Moneylines</div>
                            <div class="text-2xl font-black text-white">${{mlPct}}%</div>
                            <div class="text-xs text-emerald-400 font-mono mt-1">${{mlHits}} Correct / ${{mlTotal}} Audited</div>
                        </div>
                        <div class="espn-card border rounded-lg p-4 shadow">
                            <div class="text-xs text-gray-400 uppercase font-mono mb-1">Pitcher Ks Over/Under</div>
                            <div class="text-2xl font-black text-white">${{kPct}}%</div>
                            <div class="text-xs text-cyan-400 font-mono mt-1">${{kHits}} Correct / ${{kTotal}} Audited</div>
                        </div>
                        <div class="espn-card border rounded-lg p-4 shadow">
                            <div class="text-xs text-gray-400 uppercase font-mono mb-1">Batter Over 0.5 Hits</div>
                            <div class="text-2xl font-black text-white">${{bPct}}%</div>
                            <div class="text-xs text-yellow-400 font-mono mt-1">${{bHits}} Hits / ${{bTotal}} Audited</div>
                        </div>
                    `;
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
