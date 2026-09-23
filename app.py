import json
import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="StatsCenter Pro Hub")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
def index():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    now = datetime.now(timezone.utc)
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = now.strftime("%Y-%m-%d")

    sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={yesterday_str}&endDate={today_str}&hydrate=probablePitcher,lineups,linescore"
    try:
        sched = requests.get(sched_url, timeout=6).json()
        raw_dates = sched.get("dates", [])
    except Exception:
        raw_dates = []

    forecast_map = {}
    if "Model_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Model_Forecasts").fetchall():
            forecast_map[str(r["game_pk"])] = dict(r)

    games = []
    for d in raw_dates:
        g_date = d.get("date")
        for g in d.get("games", []):
            pk = str(g.get("gamePk"))
            status = g.get("status", {}).get("abstractGameState", "")
            detailed = g.get("status", {}).get("detailedState", "")
            teams = g.get("teams", {})
            away_t = teams.get("away", {}).get("team", {}).get("name", "Away")
            home_t = teams.get("home", {}).get("team", {}).get("name", "Home")

            ls = g.get("linescore", {})
            sc_a = ls.get("teams", {}).get("away", {}).get("runs", 0)
            sc_h = ls.get("teams", {}).get("home", {}).get("runs", 0)
            inns = ls.get("innings", [])
            f5_a = sum(i.get("away", {}).get("runs", 0) for i in inns[:5] if "away" in i) if len(inns) >= 5 else None
            f5_h = sum(i.get("home", {}).get("runs", 0) for i in inns[:5] if "home" in i) if len(inns) >= 5 else None

            f_data = forecast_map.get(pk, {})
            prob_h = f_data.get("prob_home_win", 0.54)
            hit_ml = f_data.get("hit_ml")
            hit_f5 = f_data.get("hit_f5_ml")

            stage = "upcoming"
            if status == "Live" or "In Progress" in detailed:
                stage = "live"
            elif status == "Final":
                stage = "final"
                if hit_ml is None and sc_a is not None and sc_h is not None and sc_a != sc_h:
                    pred_h = (prob_h >= 0.50)
                    hit_ml = 1 if ((sc_h > sc_a and pred_h) or (sc_a > sc_h and not pred_h)) else 0
                if hit_f5 is None and f5_a is not None and f5_h is not None and f5_a != f5_h:
                    pred_h = (prob_h >= 0.50)
                    hit_f5 = 1 if ((f5_h > f5_a and pred_h) or (f5_a > f5_h and not pred_home)) else 0

            games.append({
                "game_pk": pk,
                "game_date": g_date,
                "stage": stage,
                "away_team": away_t,
                "home_team": home_t,
                "away_score": sc_a,
                "home_score": sc_h,
                "prob_home": round(prob_h * 100, 1),
                "prob_away": round((1.0 - prob_h) * 100, 1),
                "exp_runs": f_data.get("expected_runs", 8.8),
                "f5_line": f_data.get("f5_median_runs", 4.8),
                "hit_ml": hit_ml,
                "hit_f5": hit_f5,
                "time_et": g.get("gameDate", "")[11:16] + " UTC"
            })

    pitchers = []
    if "Pitcher_K_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Pitcher_K_Forecasts").fetchall():
            pitchers.append({
                "game_pk": str(r["game_pk"]),
                "pitcher": r["pitcher_name"],
                "team": r["team_name"],
                "opponent": r["opponent_team"],
                "expected_k": r["expected_k"],
                "k_line": r["k_line"],
                "over_prob": round(float(r["over_prob"] or 0.5) * 100, 1),
                "under_prob": round(float(r["under_prob"] or 0.5) * 100, 1),
                "actual_k": r["actual_k"],
                "hit_prop": r["hit_prop"]
            })

    batters = []
    if "Batter_Hit_Forecasts" in tables:
        for r in c.execute("SELECT * FROM Batter_Hit_Forecasts ORDER BY batting_order ASC").fetchall():
            batters.append({
                "game_pk": str(r["game_pk"]),
                "player": r["player_name"],
                "team": r["team_name"],
                "opponent": r["opponent_team"],
                "slot": r["batting_order"],
                "pa": r["projected_pa"],
                "x_hits": r["expected_hits"],
                "over_05": round(float(r["over_0_5_hit_prob"] or 0.6) * 100, 1),
                "actual_hits": r["actual_hits"],
                "hit_prop": r["hit_prop"]
            })

    # Build Right vs Wrong Diagnostic Audit Feed
    audit_items = []
    for g in games:
        if g["stage"] == "final" and g["hit_ml"] is not None:
            pick = g["home_team"] if g["prob_home"] >= 50 else g["away_team"]
            pick_p = max(g["prob_home"], g["prob_away"])
            win = g["home_team"] if g["home_score"] > g["away_score"] else g["away_team"]
            audit_items.append({
                "market": "Moneyline",
                "title": f"{g['away_team']} @ {g['home_team']}",
                "pick": f"Pick: {pick} ({pick_p}%)",
                "result": f"Final: {g['away_team']} {g['away_score']}, {g['home_team']} {g['home_score']} (Winner: {win})",
                "status": "HIT" if g["hit_ml"] == 1 else "MISS"
            })

    for p in pitchers:
        if p["actual_k"] is not None and p["hit_prop"] is not None:
            pick_type = "OVER" if p["over_prob"] >= 50 else "UNDER"
            audit_items.append({
                "market": "Pitcher K",
                "title": f"{p['pitcher']} ({p['team']})",
                "pick": f"{pick_type} {p['k_line']} Ks (Exp: {p['expected_k']})",
                "result": f"Recorded: {int(p['actual_k'])} Ks",
                "status": "HIT" if p["hit_prop"] == 1 else "MISS"
            })

    for b in batters:
        if b["actual_hits"] is not None and b["hit_prop"] is not None:
            audit_items.append({
                "market": "Batter Hit",
                "title": f"#{b['slot']} {b['player']} ({b['team']})",
                "pick": f"Over 0.5 Hits ({b['over_05']}%)",
                "result": f"Recorded: {int(b['actual_hits'])} Hits (PA: {b['pa']})",
                "status": "HIT" if b["hit_prop"] == 1 else "MISS"
            })

    acc = {
        "overall_right": sum(1 for a in audit_items if a["status"] == "HIT"),
        "overall_wrong": sum(1 for a in audit_items if a["status"] == "MISS"),
        "overall_pct": 0.0,
        "markets": {
            "ml": {"label": "Full Game Moneylines", "right": sum(1 for a in audit_items if a["market"] == "Moneyline" and a["status"] == "HIT"), "wrong": sum(1 for a in audit_items if a["market"] == "Moneyline" and a["status"] == "MISS")},
            "pitchers": {"label": "Pitcher Ks", "right": sum(1 for a in audit_items if a["market"] == "Pitcher K" and a["status"] == "HIT"), "wrong": sum(1 for a in audit_items if a["market"] == "Pitcher K" and a["status"] == "MISS")},
            "batters": {"label": "Batter Hits", "right": sum(1 for a in audit_items if a["market"] == "Batter Hit" and a["status"] == "HIT"), "wrong": sum(1 for a in audit_items if a["market"] == "Batter Hit" and a["status"] == "MISS")}
        }
    }
    tot = acc["overall_right"] + acc["overall_wrong"]
    if tot > 0:
        acc["overall_pct"] = round((acc["overall_right"] / tot) * 100, 1)

    for k, m in acc["markets"].items():
        sub = m["right"] + m["wrong"]
        m["pct"] = round((m["right"] / sub) * 100, 1) if sub > 0 else 0.0

    conn.close()

    template = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>STATSCENTER PRO HUB</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0b0e14; color: #f3f4f6; font-family: system-ui, -apple-system, sans-serif; }
        .ticker-wrap { overflow: hidden; white-space: nowrap; }
        .ticker-move { display: inline-block; animation: ticker 130s linear infinite; }
        .ticker-move:hover, .ticker-move:active { animation-play-state: paused; }
        @keyframes ticker { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-50%, 0, 0); } }
    </style>
</head>
<body class="p-3 max-w-4xl mx-auto space-y-4">
    <div class="bg-red-700 text-white text-xs font-bold px-2 py-1.5 rounded flex items-center shadow-md">
        <span class="bg-black text-red-400 px-1.5 py-0.5 rounded text-[10px] mr-2 uppercase tracking-wider">BOTTOMLINE</span>
        <div class="ticker-wrap flex-1">
            <div class="ticker-move text-xs font-mono">
                MLB PRO QUANTITATIVE PIPELINE ACTIVE • 50,000 MONTE CARLO ITERATIONS • 24/7 AUTONOMOUS CONTINUOUS LEARNING •
            </div>
        </div>
    </div>

    <div class="flex items-center justify-between border-b border-gray-800 pb-3">
        <div>
            <h1 class="text-xl font-black tracking-tight text-white uppercase">StatsCenter Pro Hub</h1>
            <p class="text-xs text-gray-400 font-mono">Vectorized Discrete Prop & Kalman Architecture</p>
        </div>
        <div class="bg-blue-950 border border-blue-800 px-2 py-1 rounded text-[11px] font-mono text-blue-300">
            AUTONOMOUS 24/7
        </div>
    </div>

    <div class="space-y-2">
        <div class="flex gap-2">
            <button onclick="setDateFilter('today')" id="btn-today" class="flex-1 py-1.5 rounded text-xs font-bold bg-red-600 text-white">Today</button>
            <button onclick="setDateFilter('yesterday')" id="btn-yesterday" class="flex-1 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800">Yesterday</button>
        </div>
        <div class="flex gap-1.5 overflow-x-auto pb-1">
            <button onclick="setStage('all')" id="stg-all" class="px-3 py-1 rounded text-xs font-bold bg-gray-800 text-white whitespace-nowrap">All</button>
            <button onclick="setStage('live')" id="stg-live" class="px-3 py-1 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800 whitespace-nowrap">🔴 Live</button>
            <button onclick="setStage('upcoming')" id="stg-upcoming" class="px-3 py-1 rounded text-xs font-bold bg-red-600 text-white whitespace-nowrap">⏳ Upcoming</button>
            <button onclick="setStage('final')" id="stg-final" class="px-3 py-1 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800 whitespace-nowrap">🏁 Final</button>
        </div>
    </div>

    <div class="flex gap-1.5 overflow-x-auto border-b border-gray-800 pb-2">
        <button onclick="setMarket('matchups')" id="tab-matchups" class="px-3 py-1.5 rounded text-xs font-bold bg-red-600 text-white whitespace-nowrap">Matchups & F5</button>
        <button onclick="setMarket('batters')" id="tab-batters" class="px-3 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 hover:text-white border border-gray-800 whitespace-nowrap">Batter Hits (<span id="cnt-batters">0</span>)</button>
        <button onclick="setMarket('pitchers')" id="tab-pitchers" class="px-3 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 hover:text-white border border-gray-800 whitespace-nowrap">Pitcher Ks (<span id="cnt-pitchers">0</span>)</button>
        <button onclick="setMarket('accuracy')" id="tab-accuracy" class="px-3 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 hover:text-white border border-gray-800 whitespace-nowrap">📊 Accuracy Scorecard</button>
    </div>

    <div id="sec-matchups" class="space-y-3"></div>
    <div id="sec-batters" class="space-y-2 hidden"></div>
    <div id="sec-pitchers" class="space-y-2 hidden"></div>
    
    <!-- ACCURACY SCORECARD & AUDIT FEED -->
    <div id="sec-accuracy" class="space-y-4 hidden">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <div class="text-xs font-bold uppercase text-gray-400">Multi-Market Win Rate</div>
                <div class="flex items-baseline gap-2 mt-1">
                    <span class="text-3xl font-black font-mono text-emerald-400">__ACC_PCT__%</span>
                    <span class="text-xs text-gray-400 font-mono">Accuracy</span>
                </div>
                <div class="mt-2 text-xs font-mono text-gray-300">
                    <span class="text-emerald-400 font-bold">__ACC_R__ Correct</span> • <span class="text-red-400 font-bold">__ACC_W__ Incorrect</span>
                </div>
            </div>
            <div class="bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-1.5 font-mono text-xs">
                <div class="text-[11px] font-bold uppercase text-gray-400 mb-1">Performance by Category</div>
                <div class="flex justify-between"><span>Moneylines:</span><span class="text-emerald-400 font-bold">__ML_STAT__</span></div>
                <div class="flex justify-between"><span>Pitcher Strikeouts:</span><span class="text-yellow-400 font-bold">__PK_STAT__</span></div>
                <div class="flex justify-between"><span>Batter Hits:</span><span class="text-cyan-400 font-bold">__BH_STAT__</span></div>
            </div>
        </div>

        <div class="space-y-2">
            <div class="flex justify-between items-center border-b border-gray-800 pb-2">
                <h3 class="text-xs font-bold uppercase text-white tracking-wider">Itemized Right vs. Wrong Evaluation Feed</h3>
                <span class="text-[11px] text-gray-400 font-mono">__AUDIT_COUNT__ Graded Plays</span>
            </div>
            <div id="audit-list-container" class="space-y-2"></div>
        </div>
    </div>

    <script>
        const games = __GAMES_DATA__;
        const pitchers = __PITCHERS_DATA__;
        const batters = __BATTERS_DATA__;
        const auditItems = __AUDIT_DATA__;

        const todayStr = "__TODAY_STR__";
        const yesterdayStr = "__YESTERDAY_STR__";

        let currentDate = 'today';
        let currentMarket = 'matchups';
        let currentStage = 'upcoming';

        function setDateFilter(d) {
            currentDate = d;
            document.getElementById('btn-today').className = d === 'today' ? "flex-1 py-1.5 rounded text-xs font-bold bg-red-600 text-white" : "flex-1 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800";
            document.getElementById('btn-yesterday').className = d === 'yesterday' ? "flex-1 py-1.5 rounded text-xs font-bold bg-red-600 text-white" : "flex-1 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800";
            renderAll();
        }

        function setMarket(m) {
            currentMarket = m;
            ['matchups', 'batters', 'pitchers', 'accuracy'].forEach(k => {
                const btn = document.getElementById('tab-' + k);
                const sec = document.getElementById('sec-' + k);
                if (k === m) {
                    btn.className = "px-3 py-1.5 rounded text-xs font-bold bg-red-600 text-white whitespace-nowrap";
                    sec.classList.remove('hidden');
                } else {
                    btn.className = "px-3 py-1.5 rounded text-xs font-bold bg-gray-900 text-gray-400 hover:text-white border border-gray-800 whitespace-nowrap";
                    sec.classList.add('hidden');
                }
            });
            renderAll();
        }

        function setStage(s) {
            currentStage = s;
            ['all', 'live', 'upcoming', 'final'].forEach(k => {
                const btn = document.getElementById('stg-' + k);
                if (k === s) {
                    btn.className = "px-3 py-1 rounded text-xs font-bold bg-red-600 text-white whitespace-nowrap";
                } else {
                    btn.className = "px-3 py-1 rounded text-xs font-bold bg-gray-900 text-gray-400 border border-gray-800 whitespace-nowrap";
                }
            });
            renderAll();
        }

        function renderAll() {
            const activeDateStr = currentDate === 'today' ? todayStr : yesterdayStr;
            const dateGames = games.filter(g => g.game_date === activeDateStr);
            const activeGames = dateGames.filter(g => currentStage === 'all' || g.stage === currentStage);
            const activePks = new Set(activeGames.map(g => String(g.game_pk)));

            // 1. Matchups
            const secM = document.getElementById('sec-matchups');
            secM.innerHTML = '';
            activeGames.forEach(g => {
                let badge = '';
                if (g.stage === 'final') {
                    if (g.hit_ml === 1) badge = '<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-700 font-black text-xs">HIT ✅ PICK WON</span>';
                    else if (g.hit_ml === 0) badge = '<span class="px-2 py-0.5 rounded bg-red-950 text-red-400 border border-red-700 font-black text-xs">MISS ❌ PICK LOST</span>';
                }
                const div = document.createElement('div');
                div.className = "bg-gray-900 border border-gray-800 rounded-lg p-3 space-y-2";
                div.innerHTML = `
                    <div class="flex justify-between items-center text-xs text-gray-400 border-b border-gray-800 pb-1.5">
                        <span class="font-mono font-bold">${g.time_et} • PK: ${g.game_pk}</span>
                        <div>${badge}</div>
                    </div>
                    <div class="flex justify-between items-center text-sm font-bold text-white">
                        <span>${g.away_team}</span>
                        <span class="font-mono">${g.stage === 'upcoming' ? g.prob_away + '%' : g.away_score}</span>
                    </div>
                    <div class="flex justify-between items-center text-sm font-bold text-white">
                        <span>${g.home_team}</span>
                        <span class="font-mono text-emerald-400">${g.stage === 'upcoming' ? g.prob_home + '%' : g.home_score}</span>
                    </div>
                    <div class="flex justify-between text-[11px] text-gray-400 pt-1 font-mono">
                        <span>Model Runs: ${g.exp_runs}</span>
                        <span>F5 Line: ${g.f5_line}</span>
                    </div>
                `;
                secM.appendChild(div);
            });

            // 2. Pitchers
            const secP = document.getElementById('sec-pitchers');
            secP.innerHTML = '';
            const filteredPitchers = pitchers.filter(p => activePks.has(String(p.game_pk)));
            document.getElementById('cnt-pitchers').textContent = filteredPitchers.length;
            filteredPitchers.forEach(p => {
                let pBadge = '';
                if (p.hit_prop === 1) pBadge = '<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-700 text-[10px] font-black">OVER ✅</span>';
                else if (p.hit_prop === 0 && p.actual_k !== null) pBadge = '<span class="px-2 py-0.5 rounded bg-red-950 text-red-400 border border-red-700 text-[10px] font-black">UNDER ❌</span>';
                const div = document.createElement('div');
                div.className = "bg-gray-900 border border-gray-800 rounded p-2.5 flex justify-between items-center text-xs";
                div.innerHTML = `
                    <div>
                        <div class="font-bold text-white">${p.pitcher} <span class="text-[10px] text-gray-400 font-normal">(${p.team})</span></div>
                        <div class="text-[11px] text-gray-400 font-mono">Line: ${p.k_line} Ks • Exp: ${p.expected_k}</div>
                    </div>
                    <div class="text-right font-mono">
                        <div class="text-emerald-400 font-bold">Over: ${p.over_prob}%</div>
                        <div class="text-gray-400 text-[11px]">Under: ${p.under_prob}%</div>
                        <div class="mt-0.5">${pBadge}</div>
                    </div>
                `;
                secP.appendChild(div);
            });

            // 3. Batters
            const secB = document.getElementById('sec-batters');
            secB.innerHTML = '';
            const filteredBatters = batters.filter(b => activePks.has(String(b.game_pk)));
            document.getElementById('cnt-batters').textContent = filteredBatters.length;
            filteredBatters.forEach(b => {
                let bBadge = '';
                if (b.hit_prop === 1) bBadge = '<span class="px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-700 text-[10px] font-black">HIT ✅</span>';
                else if (b.hit_prop === 0 && b.actual_hits !== null) bBadge = '<span class="px-2 py-0.5 rounded bg-red-950 text-red-400 border border-red-700 text-[10px] font-black">0 HITS ❌</span>';
                const div = document.createElement('div');
                div.className = "bg-gray-900 border border-gray-800 rounded p-2.5 flex justify-between items-center text-xs";
                div.innerHTML = `
                    <div>
                        <div class="font-bold text-white">#${b.slot} ${b.player} <span class="text-[10px] text-gray-400 font-normal">(${b.team})</span></div>
                        <div class="text-[11px] text-gray-400 font-mono">PA: ${b.pa} • xHits: ${b.x_hits}</div>
                    </div>
                    <div class="text-right font-mono">
                        <div class="text-emerald-400 font-bold">Over 0.5: ${b.over_05}%</div>
                        <div class="mt-0.5">${bBadge}</div>
                    </div>
                `;
                secB.appendChild(div);
            });

            // 4. Accuracy Audit Feed
            const contA = document.getElementById('audit-list-container');
            if (contA) {
                contA.innerHTML = '';
                if (auditItems.length === 0) {
                    contA.innerHTML = '<div class="text-gray-400 text-xs font-mono p-3 bg-gray-900 rounded border border-gray-800">No completed game boxscores graded yet today.</div>';
                } else {
                    auditItems.forEach(item => {
                        const isHit = item.status === 'HIT';
                        const div = document.createElement('div');
                        div.className = `bg-gray-900 border ${isHit ? 'border-emerald-900/50' : 'border-red-900/50'} rounded-lg p-2.5 flex justify-between items-center text-xs`;
                        div.innerHTML = `
                            <div>
                                <div class="flex items-center gap-1.5 mb-0.5">
                                    <span class="px-1.5 py-0.5 rounded bg-gray-800 text-[10px] font-mono font-bold text-gray-300 uppercase">${item.market}</span>
                                    <span class="font-bold text-white">${item.title}</span>
                                </div>
                                <div class="text-[11px] font-mono text-gray-400">${item.pick} • <span class="text-gray-300">${item.result}</span></div>
                            </div>
                            <div>
                                <span class="px-2 py-1 rounded ${isHit ? 'bg-emerald-950 text-emerald-400 border border-emerald-700' : 'bg-red-950 text-red-400 border border-red-700'} font-black text-xs font-mono">
                                    ${isHit ? 'RIGHT ✅' : 'WRONG ❌'}
                                </span>
                            </div>
                        `;
                        contA.appendChild(div);
                    });
                }
            }
        }

        renderAll();
    </script>
</body>
</html>"""

    # Populate string replacements safely
    content = template.replace("__ACC_PCT__", str(acc["overall_pct"]))
    content = content.replace("__ACC_R__", str(acc["overall_right"]))
    content = content.replace("__ACC_W__", str(acc["overall_wrong"]))
    content = content.replace("__ML_STAT__", f"{acc['markets']['ml']['right']} Right / {acc['markets']['ml']['wrong']} Wrong ({acc['markets']['ml']['pct']}%)")
    content = content.replace("__PK_STAT__", f"{acc['markets']['pitchers']['right']} Right / {acc['markets']['pitchers']['wrong']} Wrong ({acc['markets']['pitchers']['pct']}%)")
    content = content.replace("__BH_STAT__", f"{acc['markets']['batters']['right']} Right / {acc['markets']['batters']['wrong']} Wrong ({acc['markets']['batters']['pct']}%)")
    content = content.replace("__AUDIT_COUNT__", str(len(auditItems)))
    content = content.replace("__TODAY_STR__", today_str)
    content = content.replace("__YESTERDAY_STR__", yesterday_str)
    content = content.replace("__GAMES_DATA__", json.dumps(games))
    content = content.replace("__PITCHERS_DATA__", json.dumps(pitchers))
    content = content.replace("__BATTERS_DATA__", json.dumps(batters))
    content = content.replace("__AUDIT_DATA__", json.dumps(auditItems))

    return HTMLResponse(content=content)
