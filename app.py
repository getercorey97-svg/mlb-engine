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
    """Ingests official MLB Stats API schedules (past 24h + next 24h) to power Live and Final stages."""
    now_utc = datetime.now(timezone.utc)
    yesterday = (now_utc - timedelta(days=1)).strftime("%Y-%m-%d")
    today = now_utc.strftime("%Y-%m-%d")
    tomorrow = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={yesterday}&endDate={tomorrow}&hydrate=linescore,probablePitcher,decisions"
    try:
        res = requests.get(url, timeout=5)
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

    # Load local pregame forecasts
    pregame_models = {}
    if "Model_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Model_Forecasts").fetchall():
            row_dict = dict(r)
            pk = row_dict.get("game_pk")
            if pk:
                pregame_models[pk] = row_dict

    daily_lineups = {}
    if "Daily_Lineups" in tables:
        for r in c.execute("SELECT * FROM Daily_Lineups").fetchall():
            row_dict = dict(r)
            pk = row_dict.get("game_pk")
            if pk:
                daily_lineups[pk] = row_dict

    # Ingest Live & Completed Game Telemetry from MLB Stats API
    live_schedule = fetch_mlb_slate_and_scores()

    games = []
    # Merge remote live slate with engine models
    all_pks = set(list(live_schedule.keys()) + list(pregame_models.keys()) + list(daily_lineups.keys()))

    for pk in all_pks:
        mlb_game = live_schedule.get(pk, {})
        m = pregame_models.get(pk, {})
        d = daily_lineups.get(pk, {})

        # Basic Matchup Info
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

        # Extract Pregame Probabilities
        p_home_pre = float(m.get("prob_home_win") or 0.50)
        p_away_pre = 1.0 - p_home_pre
        exp_a = float(m.get("expected_runs_away") or 4.5)
        exp_h = float(m.get("expected_runs_home") or 4.5)
        f5_med = float(m.get("f5_median_runs") or 5.0)
        fav_team = home_name if p_home_pre >= 0.50 else away_name
        fav_prob = max(p_home_pre, p_away_pre)

        # Classify Stage & Deduce In-Game Analytics
        is_final = any(x in status_desc.lower() for x in ["final", "game over", "completed"])
        is_live = any(x in status_desc.lower() for x in ["in progress", "live", "delayed", "manager challenge"])

        current_inning = linescore.get("currentInning", 0)
        inning_state = linescore.get("inningState", "")
        away_actual_runs = linescore.get("teams", {}).get("away", {}).get("runs", 0)
        home_actual_runs = linescore.get("teams", {}).get("home", {}).get("runs", 0)

        # Calculate F5 Actual Runs if game reached 5th inning
        f5_actual_runs = None
        innings_list = linescore.get("innings", [])
        if len(innings_list) >= 5:
            f5_a = sum([inn.get("away", {}).get("runs", 0) for inn in innings_list[:5]])
            f5_h = sum([inn.get("home", {}).get("runs", 0) for inn in innings_list[:5]])
            f5_actual_runs = f5_a + f5_h

        # In-Game Bayesian Dynamic Deduction
        live_prob_home = p_home_pre * 100
        live_exp_away = exp_a
        live_exp_home = exp_h

        if is_live and current_inning > 0:
            stage = "live"
            rem_innings_away = max(0.0, 9.0 - (current_inning - 1) - (1.0 if inning_state.lower() == "bottom" else 0.0))
            rem_innings_home = max(0.0, 8.5 - (current_inning - 1) - (0.5 if inning_state.lower() == "bottom" else 0.0))
            
            live_exp_away = round(away_actual_runs + (exp_a * (rem_innings_away / 9.0)), 2)
            live_exp_home = round(home_actual_runs + (exp_h * (rem_innings_home / 9.0)), 2)
            
            diff = live_exp_home - live_exp_away
            live_p_home = 1.0 / (1.0 + 10 ** (-diff / 2.2))
            live_prob_home = round(live_p_home * 100, 1)

        elif is_final:
            stage = "final"
        else:
            stage = "upcoming"

        # Post-Mortem Comparison Verification
        hit_ml = None
        actual_winner = None
        if is_final:
            if away_actual_runs > home_actual_runs:
                actual_winner = away_name
            elif home_actual_runs > away_actual_runs:
                actual_winner = home_name
            hit_ml = (fav_team == actual_winner)

        games.append({
            "game_pk": pk,
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

    # Fetch Batter Hit Props
    batters = []
    if "Batter_Hit_Forecasts" in tables:
        raw_batters = [dict(r) for r in c.execute("SELECT * FROM Batter_Hit_Forecasts ORDER BY over_0_5_hit_prob DESC").fetchall()]
        for b in raw_batters:
            p05 = float(b.get("over_0_5_hit_prob") or 0.0)
            p15 = float(b.get("over_1_5_hit_prob") or 0.0)
            p25 = float(b.get("over_2_5_hit_prob") or 0.0)
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
                "p_2_5": round(p25 * 100 if p25 <= 1.0 else p25, 1)
            })

    # Fetch Cross-Market SGP Edges
    sgps = []
    if "Correlated_Market_Forecasts" in tables:
        raw_sgp = [dict(r) for r in c.execute("SELECT * FROM Correlated_Market_Forecasts ORDER BY correlation_edge DESC").fetchall()]
        for s in raw_sgp:
            joint = float(s.get("joint_prob") or 0.0)
            indep = float(s.get("uncorrelated_prob") or 0.0)
            edge = float(s.get("correlation_edge") or 0.0)
            sgps.append({
                "game_pk": s.get("game_pk"),
                "matchup": f"{s.get('away_team')} @ {s.get('home_team')}",
                "type": s.get("sgp_type"),
                "leg_1": s.get("leg_1"),
                "leg_2": s.get("leg_2"),
                "joint": round(joint * 100 if joint <= 1.0 else joint, 1),
                "indep": round(indep * 100 if indep <= 1.0 else indep, 1),
                "edge": round(edge * 100 if edge <= 1.0 else edge, 1)
            })

    conn.close()

    games_json = json.dumps(games)
    batters_json = json.dumps(batters)
    sgps_json = json.dumps(sgps)

    html = f"""
    <!DOCTYPE html>
    <html lang="en" class="h-full bg-[#0b0e14]">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <meta name="theme-color" content="#cc0000">
        <title>ESPN STATSCENTER // MLB QUANT HUB</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            * {{ -webkit-tap-highlight-color: transparent; touch-action: manipulation; }}
            @keyframes ticker {{
                0% {{ transform: translateX(100%); }}
                100% {{ transform: translateX(-100%); }}
            }}
            .animate-ticker {{
                display: inline-block;
                white-space: nowrap;
                animation: ticker 130s linear infinite; /* Authentically slowed readable ESPN pace */
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
    <body class="espn-dark text-gray-200 font-sans antialiased min-h-screen flex flex-col selection:bg-red-600 selection:text-white pb-24 md:pb-12">
        <!-- Top ESPN BottomLine Scrolling Ticker -->
        <div class="bg-black border-b border-red-700/80 overflow-hidden flex items-center h-10 sticky top-0 z-50 shadow-md">
            <div class="espn-red text-white px-3.5 h-full uppercase tracking-wider flex items-center z-10 shrink-0 font-black text-xs">
                <span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse"></span>
                BOTTOMLINE
            </div>
            <div class="overflow-hidden w-full relative h-full flex items-center" id="ticker-box">
                <div id="ticker-content" class="animate-ticker text-xs font-mono font-bold text-gray-300">
                    Loading telemetry feed...
                </div>
            </div>
        </div>

        <!-- Master Header Navigation -->
        <header class="espn-subbar border-b border-gray-800 px-4 py-3 shadow-lg">
            <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                <div class="flex items-center gap-3">
                    <span class="espn-red text-white text-xl font-black px-2.5 py-0.5 rounded tracking-tighter italic shadow">ESPN</span>
                    <div>
                        <h1 class="text-lg md:text-2xl font-black text-white tracking-wide uppercase">StatsCenter Quant Hub</h1>
                        <p class="text-[11px] md:text-xs font-mono text-gray-400">Live Bayesian Game Deduction • Post-Match Factual Audits</p>
                    </div>
                </div>

                <!-- Global Game Stages Filter Pills -->
                <div class="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto touch-scroll py-1 text-xs font-black uppercase tracking-wider">
                    <button onclick="setGameStage('all')" id="stage-btn-all" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-red-600 text-white shadow active:scale-95 transition-all">All (<span id="count-all">0</span>)</button>
                    <button onclick="setGameStage('live')" id="stage-btn-live" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🔴 Live (<span id="count-live">0</span>)</button>
                    <button onclick="setGameStage('upcoming')" id="stage-btn-upcoming" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">⏳ Next (<span id="count-upcoming">0</span>)</button>
                    <button onclick="setGameStage('final')" id="stage-btn-final" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🏁 Final (<span id="count-final">0</span>)</button>
                </div>
            </div>

            <!-- Market Type Navigation Tabs -->
            <div class="max-w-7xl mx-auto mt-3 flex gap-2 overflow-x-auto touch-scroll border-t border-gray-800/80 pt-2.5 text-xs md:text-sm font-bold uppercase no-scrollbar">
                <button onclick="setBetMarket('all')" id="tab-all" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-red-600 text-white whitespace-nowrap">All Markets</button>
                <button onclick="setBetMarket('batters')" id="tab-batters" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🎯 Batter Hits</button>
                <button onclick="setBetMarket('f5')" id="tab-f5" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⏱️ First 5 (F5)</button>
                <button onclick="setBetMarket('moneyline')" id="tab-moneyline" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🏆 Moneylines</button>
                <button onclick="setBetMarket('sgp')" id="tab-sgp" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⚡ Correlated SGPs</button>
            </div>
        </header>

        <!-- Main Display Feed -->
        <main class="max-w-7xl mx-auto p-4 md:p-6 space-y-6 flex-1 w-full">
            <div class="flex justify-between items-center bg-gray-900/90 border border-gray-800 px-3.5 py-2.5 rounded-lg text-xs">
                <span class="text-gray-400">View: <strong id="filter-label" class="text-yellow-400 uppercase font-mono font-bold tracking-wide">All Slates • All Markets</strong></span>
                <span class="font-mono text-gray-500 text-[11px]">Database: <strong class="text-emerald-400">mlb_engine.db</strong></span>
            </div>

            <!-- Section 1: Slate Matchups Grid -->
            <section id="section-games" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide" id="games-section-title">Matchup Intelligence</h2>
                    <span class="text-xs font-mono text-gray-400" id="games-total-counter"></span>
                </div>
                <div id="games-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5"></div>
            </section>

            <!-- Section 2: Batter Hit Props Table -->
            <section id="section-batters" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Endogenous Batter Hit Distributions</h2>
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
                                <th class="py-3 px-3 text-right text-gray-400">Over 2.5</th>
                            </tr>
                        </thead>
                        <tbody id="batters-table" class="divide-y divide-gray-800 font-mono"></tbody>
                    </table>
                </div>
            </section>

            <!-- Section 3: Correlated Same Game Parlays (SGP) Table -->
            <section id="section-sgp" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Cross-Market Joint Probability Edges</h2>
                    <span class="text-xs font-mono text-gray-400" id="sgp-total-counter"></span>
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
        </main>

        <script>
            const gamesData = {games_json};
            const battersData = {batters_json};
            const sgpData = {sgps_json};

            let currentStage = 'all';
            let currentMarket = 'all';

            document.getElementById('count-all').textContent = gamesData.length;
            document.getElementById('count-live').textContent = gamesData.filter(g => g.stage === 'live').length;
            document.getElementById('count-upcoming').textContent = gamesData.filter(g => g.stage === 'upcoming').length;
            document.getElementById('count-final').textContent = gamesData.filter(g => g.stage === 'final').length;

            function buildTicker() {{
                const items = [];
                // Live games first
                gamesData.filter(g => g.stage === 'live').forEach(g => {{
                    items.push(`🔴 LIVE: ${{g.away_team}} ${{g.away_actual_runs}}, ${{g.home_team}} ${{g.home_actual_runs}} (${{g.inning_state}} ${{g.current_inning}}) | Live Proj: ${{g.live_exp_away}} - ${{g.live_exp_home}} | Home Win: ${{g.live_prob_home}}%`);
                }});
                // Finals with outcome audit
                gamesData.filter(g => g.stage === 'final').forEach(g => {{
                    const badge = g.hit_ml ? 'HIT ✅' : 'MISS ❌';
                    items.push(`🏁 FINAL: ${{g.away_team}} ${{g.away_actual_runs}}, ${{g.home_team}} ${{g.home_actual_runs}} | Model Fav: ${{g.fav_team}} (${{g.fav_prob}}%) -> ${{badge}}`);
                }});
                // Top batter props
                battersData.slice(0, 8).forEach(b => {{
                    items.push(`🎯 ${{b.name}} (${{b.team}} #${{b.order}}): ${{b.p_0_5}}% Over 0.5 Hits (${{b.xhits}} xH)`);
                }});
                // Top SGPs
                sgpData.slice(0, 6).forEach(s => {{
                    items.push(`⚡ SGP: ${{s.leg_1}} + ${{s.leg_2}} [Joint: ${{s.joint}}% | +${{s.edge}}% Edge]`);
                }});
                document.getElementById('ticker-content').innerHTML = items.join(' &nbsp;&nbsp;&nbsp;•&nbsp;&nbsp;&nbsp; ');
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

                const secGames = document.getElementById('section-games');
                if (currentMarket === 'batters') {{
                    secGames.classList.add('hidden');
                }} else {{
                    secGames.classList.remove('hidden');
                    const container = document.getElementById('games-grid');
                    container.innerHTML = '';
                    
                    filteredGames.forEach(g => {{
                        const card = document.createElement('div');
                        card.className = "espn-card border rounded-lg p-3.5 shadow-md flex flex-col justify-between";

                        if (g.stage === 'live') {{
                            // LIVE IN-GAME DEDUCTION CARD
                            card.innerHTML = `
                                <div>
                                    <div class="flex justify-between items-center mb-2">
                                        <span class="bg-red-600 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-wider animate-pulse flex items-center gap-1">
                                            <span class="w-1.5 h-1.5 rounded-full bg-white animate-ping"></span>
                                            LIVE: ${{g.inning_state}} ${{g.current_inning}}
                                        </span>
                                        <span class="font-mono text-xs text-yellow-400 font-bold">F5 Pregame: ${{g.pregame_f5_median}}</span>
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
                                            <span class="text-gray-400">Live Projected Final Runs:</span>
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
                            // POST-MORTEM FACTUAL AUDIT CARD
                            const verdictBadge = g.hit_ml 
                                ? '<span class="bg-emerald-600 text-white px-2 py-0.5 rounded text-[10px] font-black">HIT ✅</span>'
                                : '<span class="bg-red-700 text-white px-2 py-0.5 rounded text-[10px] font-black">MISS ❌</span>';

                            const f5Info = g.f5_actual_runs !== null 
                                ? `Actual F5: <strong class="text-white">${{g.f5_actual_runs}}</strong> (Model: ${{g.pregame_f5_median}})`
                                : `Model F5: ${{g.pregame_f5_median}}`;

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
                                        <div class="text-[10px] font-mono text-gray-400 uppercase font-bold tracking-wider">Pre-Match vs Official Post-Mortem</div>
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
                            // UPCOMING SLATE CARD
                            card.innerHTML = `
                                <div>
                                    <div class="flex justify-between items-center mb-2">
                                        <span class="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[10px] font-black tracking-wider">UPCOMING</span>
                                        <span class="font-mono text-xs text-yellow-400 font-bold">F5 Total: ${{g.pregame_f5_median}}</span>
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
                    document.getElementById('games-total-counter').textContent = `${{filteredGames.length}} Games`;
                }}

                // 2. Batters Table
                const secBatters = document.getElementById('section-batters');
                if (currentMarket === 'f5' || currentMarket === 'moneyline') {{
                    secBatters.classList.add('hidden');
                }} else {{
                    secBatters.classList.remove('hidden');
                    const bTable = document.getElementById('batters-table');
                    bTable.innerHTML = '';
                    
                    const filteredBatters = battersData.filter(b => activePks.has(b.game_pk));
                    filteredBatters.slice(0, 30).forEach(b => {{
                        const tr = document.createElement('tr');
                        tr.className = "hover:bg-gray-800/60 transition-colors";
                        tr.innerHTML = `
                            <td class="py-2.5 px-3.5 font-sans font-bold text-white whitespace-nowrap">${{b.name}}</td>
                            <td class="py-2.5 px-3 text-center text-gray-400 text-xs">${{b.team}} #${{b.order}}</td>
                            <td class="py-2.5 px-3 text-right text-gray-300">${{b.pa}}</td>
                            <td class="py-2.5 px-3 text-right text-white">${{b.xhits}}</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-emerald-400">${{b.p_0_5}}%</td>
                            <td class="py-2.5 px-3.5 text-right font-bold text-cyan-400">${{b.p_1_5}}%</td>
                            <td class="py-2.5 px-3 text-right text-gray-400">${{b.p_2_5}}%</td>
                        `;
                        bTable.appendChild(tr);
                    }});
                    document.getElementById('batters-total-counter').textContent = `${{filteredBatters.length}} Batters`;
                }}

                // 3. SGP Table
                const secSgp = document.getElementById('section-sgp');
                if (currentMarket === 'batters' || currentMarket === 'f5' || currentMarket === 'moneyline') {{
                    secSgp.classList.add('hidden');
                }} else {{
                    secSgp.classList.remove('hidden');
                    const sTable = document.getElementById('sgp-table');
                    sTable.innerHTML = '';

                    const filteredSgps = sgpData.filter(s => activePks.has(s.game_pk));
                    filteredSgps.slice(0, 25).forEach(s => {{
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
                    document.getElementById('sgp-total-counter').textContent = `${{filteredSgps.length}} SGPs`;
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
