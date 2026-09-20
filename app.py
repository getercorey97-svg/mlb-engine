import sqlite3
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="ESPN StatsCenter MLB Engine Hub")

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    conn = get_db_connection()
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    # 1. Dynamic Game Extraction
    games = []
    try:
        dl_rows = [dict(r) for r in c.execute("SELECT * FROM Daily_Lineups").fetchall()] if "Daily_Lineups" in tables else []
        mf_rows = {dict(r)["game_pk"]: dict(r) for r in c.execute("SELECT * FROM Model_Forecasts").fetchall() if "game_pk" in dict(r)} if "Model_Forecasts" in tables else {}

        if not dl_rows and mf_rows:
            dl_rows = list(mf_rows.values())

        for g in dl_rows:
            pk = g.get("game_pk")
            away_team = g.get("away_team") or g.get("away") or "Away"
            home_team = g.get("home_team") or g.get("home") or "Home"

            away_sp = next((str(g[k]) for k in g if "away" in k.lower() and any(x in k.lower() for x in ["sp", "pitch", "starter"]) and g[k]), "TBD")
            home_sp = next((str(g[k]) for k in g if "home" in k.lower() and any(x in k.lower() for x in ["sp", "pitch", "starter"]) and "prob" not in k.lower() and g[k]), "TBD")
            status_raw = next((str(g[k]) for k in g if "status" in k.lower() or "state" in k.lower() and g[k]), "Scheduled")

            m = mf_rows.get(pk, {})
            p_home = next((float(m[k]) for k in m if "home" in k.lower() and ("prob" in k.lower() or "win" in k.lower()) and m[k] is not None), None)
            if p_home is None:
                p_home = next((float(m[k]) for k in m if ("prob" in k.lower() or "win" in k.lower()) and m[k] is not None), 0.50)
            p_away = 1.0 - p_home

            exp_away = next((float(m[k]) for k in m if "away" in k.lower() and ("run" in k.lower() or "exp" in k.lower()) and m[k] is not None), 4.5)
            exp_home = next((float(m[k]) for k in m if "home" in k.lower() and ("run" in k.lower() or "exp" in k.lower()) and "prob" not in k.lower() and m[k] is not None), 4.5)
            f5_med = next((float(m[k]) for k in m if "f5" in k.lower() and m[k] is not None), 5.0)

            if "final" in status_raw.lower():
                stage = "final"
            elif any(tag in status_raw.lower() for tag in ["live", "mark", "progress", "inning"]):
                stage = "live"
            else:
                stage = "upcoming"

            games.append({
                "game_pk": pk,
                "away_team": away_team,
                "home_team": home_team,
                "away_sp": away_sp,
                "home_sp": home_sp,
                "lineup_status": status_raw,
                "stage": stage,
                "prob_home": round(p_home * 100, 1),
                "prob_away": round(p_away * 100, 1),
                "expected_away": round(exp_away, 2),
                "expected_home": round(exp_home, 2),
                "f5_median": round(f5_med, 1),
                "fav_team": home_team if p_home >= 0.50 else away_team,
                "fav_prob": round(max(p_home, p_away) * 100, 1)
            })
    except Exception as e:
        print(f"[ERROR] Loading games: {e}")

    # 2. Dynamic Batter Hit Props Extraction
    batters = []
    try:
        raw_batters = [dict(r) for r in c.execute("SELECT * FROM Batter_Hit_Forecasts").fetchall()] if "Batter_Hit_Forecasts" in tables else []
        for b in raw_batters:
            p05 = next((float(b[k]) for k in b if "0_5" in k or "05" in k and b[k] is not None), 0.0)
            p15 = next((float(b[k]) for k in b if "1_5" in k or "15" in k and b[k] is not None), 0.0)
            p25 = next((float(b[k]) for k in b if "2_5" in k or "25" in k and b[k] is not None), 0.0)
            xhits = next((float(b[k]) for k in b if "hit" in k.lower() and "prob" not in k.lower() and b[k] is not None), 0.0)

            batters.append({
                "game_pk": b.get("game_pk"),
                "name": b.get("player_name") or b.get("name") or "Player",
                "team": b.get("team_name") or b.get("team") or "MLB",
                "order": b.get("batting_order") or b.get("order") or 0,
                "pa": round(float(b.get("projected_pa") or 4.0), 1),
                "ab": round(float(b.get("projected_ab") or 3.5), 1),
                "xhits": round(xhits, 2),
                "p_0_5": round(p05 * 100 if p05 <= 1.0 else p05, 1),
                "p_1_5": round(p15 * 100 if p15 <= 1.0 else p15, 1),
                "p_2_5": round(p25 * 100 if p25 <= 1.0 else p25, 1)
            })
        batters.sort(key=lambda x: x["p_0_5"], reverse=True)
    except Exception as e:
        print(f"[ERROR] Loading batters: {e}")

    # 3. Dynamic SGP Extraction
    sgps = []
    try:
        raw_sgp = [dict(r) for r in c.execute("SELECT * FROM Correlated_Market_Forecasts").fetchall()] if "Correlated_Market_Forecasts" in tables else []
        for s in raw_sgp:
            joint = float(s.get("joint_prob") or 0.0)
            indep = float(s.get("uncorrelated_prob") or 0.0)
            edge = float(s.get("correlation_edge") or 0.0)
            sgps.append({
                "game_pk": s.get("game_pk"),
                "matchup": f"{s.get('away_team', 'Away')} @ {s.get('home_team', 'Home')}",
                "type": s.get("sgp_type") or "Cross-Market SGP",
                "leg_1": s.get("leg_1") or "Leg 1",
                "leg_2": s.get("leg_2") or "Leg 2",
                "joint": round(joint * 100 if joint <= 1.0 else joint, 1),
                "indep": round(indep * 100 if indep <= 1.0 else indep, 1),
                "edge": round(edge * 100 if edge <= 1.0 else edge, 1)
            })
        sgps.sort(key=lambda x: x["edge"], reverse=True)
    except Exception as e:
        print(f"[ERROR] Loading SGPs: {e}")

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
                animation: ticker 40s linear infinite;
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
        <div class="bg-black border-b border-red-700/80 overflow-hidden flex items-center h-10 sticky top-0 z-50 shadow-md">
            <div class="espn-red text-white px-3.5 h-full uppercase tracking-wider flex items-center z-10 shrink-0 font-black text-xs">
                <span class="inline-block w-2 h-2 rounded-full bg-yellow-400 mr-2 animate-pulse"></span>
                LIVE LINES
            </div>
            <div class="overflow-hidden w-full relative h-full flex items-center" id="ticker-box">
                <div id="ticker-content" class="animate-ticker text-xs font-mono font-bold text-gray-300">
                    Loading live lines...
                </div>
            </div>
        </div>

        <header class="espn-subbar border-b border-gray-800 px-4 py-3 shadow-lg">
            <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-start md:items-center gap-3">
                <div class="flex items-center gap-3">
                    <span class="espn-red text-white text-xl font-black px-2.5 py-0.5 rounded tracking-tighter italic shadow">ESPN</span>
                    <div>
                        <h1 class="text-lg md:text-2xl font-black text-white tracking-wide uppercase">StatsCenter Quant Hub</h1>
                        <p class="text-[11px] md:text-xs font-mono text-gray-400">Autonomous Lineup Simulator • Cross-Market SGP Engine</p>
                    </div>
                </div>

                <div class="flex items-center gap-1.5 overflow-x-auto w-full md:w-auto touch-scroll py-1 text-xs font-black uppercase tracking-wider">
                    <button onclick="setGameStage('all')" id="stage-btn-all" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-red-600 text-white shadow active:scale-95 transition-all">All (<span id="count-all">0</span>)</button>
                    <button onclick="setGameStage('live')" id="stage-btn-live" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🔴 Live (<span id="count-live">0</span>)</button>
                    <button onclick="setGameStage('upcoming')" id="stage-btn-upcoming" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">⏳ Next (<span id="count-upcoming">0</span>)</button>
                    <button onclick="setGameStage('final')" id="stage-btn-final" class="stage-btn min-h-[40px] px-3.5 py-2 rounded-md bg-gray-800 text-gray-400 border border-gray-700 active:scale-95 transition-all">🏁 Final (<span id="count-final">0</span>)</button>
                </div>
            </div>

            <div class="max-w-7xl mx-auto mt-3 flex gap-2 overflow-x-auto touch-scroll border-t border-gray-800/80 pt-2.5 text-xs md:text-sm font-bold uppercase">
                <button onclick="setBetMarket('all')" id="tab-all" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-red-600 text-white whitespace-nowrap">All Markets</button>
                <button onclick="setBetMarket('batters')" id="tab-batters" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🎯 Batter Hits</button>
                <button onclick="setBetMarket('f5')" id="tab-f5" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⏱️ First 5 (F5)</button>
                <button onclick="setBetMarket('moneyline')" id="tab-moneyline" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">🏆 Moneylines</button>
                <button onclick="setBetMarket('sgp')" id="tab-sgp" class="market-tab min-h-[44px] px-4 py-2 border-b-2 border-transparent text-gray-400 hover:text-white whitespace-nowrap">⚡ Correlated SGPs</button>
            </div>
        </header>

        <main class="max-w-7xl mx-auto p-4 md:p-6 space-y-6 flex-1 w-full">
            <div class="flex justify-between items-center bg-gray-900/90 border border-gray-800 px-3.5 py-2.5 rounded-lg text-xs">
                <span class="text-gray-400">Filter: <strong id="filter-label" class="text-yellow-400 uppercase font-mono font-bold tracking-wide">All Slates • All Markets</strong></span>
                <span class="font-mono text-gray-500 text-[11px]">Database: <strong class="text-emerald-400">mlb_engine.db</strong></span>
            </div>

            <section id="section-games" class="space-y-3">
                <div class="flex items-center justify-between border-b border-gray-800 pb-2">
                    <h2 class="text-base md:text-lg font-black uppercase text-white tracking-wide">Slate Projections</h2>
                    <span class="text-xs font-mono text-gray-400" id="games-total-counter"></span>
                </div>
                <div id="games-grid" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5"></div>
            </section>

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
                gamesData.forEach(g => {{
                    items.push(`⚾ ${{g.away_team}} @ ${{g.home_team}} | ${{g.fav_team}} ${{g.fav_prob}}% | F5 O/U: ${{g.f5_median}} | Exp: ${{g.expected_away}} - ${{g.expected_home}}`);
                }});
                battersData.slice(0, 10).forEach(b => {{
                    items.push(`🎯 ${{b.name}} (${{b.team}} #${{b.order}}): ${{b.p_0_5}}% Over 0.5 Hits (${{b.xhits}} xH)`);
                }});
                sgpData.slice(0, 8).forEach(s => {{
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
                        let badge = '<span class="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[10px] font-black tracking-wider">UPCOMING</span>';
                        if (g.stage === 'live') badge = '<span class="bg-red-600 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-wider animate-pulse">🔴 LIVE</span>';
                        if (g.stage === 'final') badge = '<span class="bg-blue-600 text-white px-2 py-0.5 rounded text-[10px] font-black tracking-wider">FINAL</span>';

                        const card = document.createElement('div');
                        card.className = "espn-card border rounded-lg p-3.5 shadow-md flex flex-col justify-between";
                        card.innerHTML = `
                            <div>
                                <div class="flex justify-between items-center mb-2">
                                    ${{badge}}
                                    <span class="font-mono text-xs text-yellow-400 font-bold">F5 Line: ${{g.f5_median}}</span>
                                </div>
                                <h3 class="text-base font-black text-white uppercase tracking-tight mb-1">${{g.away_team}} @ ${{g.home_team}}</h3>
                                <div class="text-xs text-gray-400 space-y-0.5 mb-3">
                                    <div class="truncate">SP: <span class="text-gray-200">${{g.away_sp}}</span> vs <span class="text-gray-200">${{g.home_sp}}</span></div>
                                    <div>Runs: <strong class="text-white font-mono">${{g.expected_away}} - ${{g.expected_home}}</strong></div>
                                </div>
                            </div>
                            <div class="bg-black/50 p-2.5 rounded border border-gray-800/80 space-y-1 text-xs">
                                <div class="flex justify-between items-center">
                                    <span class="text-gray-400">Favored:</span>
                                    <span class="font-bold text-emerald-400">${{g.fav_team}} (${{g.fav_prob}}%)</span>
                                </div>
                                <div class="flex justify-between font-mono text-[11px] text-gray-400">
                                    <span>Away: ${{g.prob_away}}%</span>
                                    <span>Home: ${{g.prob_home}}%</span>
                                </div>
                            </div>
                        `;
                        container.appendChild(card);
                    }});
                    document.getElementById('games-total-counter').textContent = `${{filteredGames.length}} Games`;
                }}

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
