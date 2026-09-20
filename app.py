import sqlite3
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI(title="MLB Quantitative Hub")

def get_db():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    conn = get_db()
    c = conn.cursor()
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    games = []
    if "Model_Forecasts" in tables:
        rows = c.execute("""
            SELECT 
                m.game_pk, d.away_team, d.home_team,
                m.prob_home_win, m.expected_runs_away, m.expected_runs_home,
                m.f5_median_runs, m.f5_actual_away, m.f5_actual_home,
                m.actual_score_away, m.actual_score_home,
                m.hit_ml, m.hit_f5_ml, m.hit_f5_total
            FROM Model_Forecasts m
            LEFT JOIN Daily_Lineups d ON m.game_pk = d.game_pk
        """).fetchall()
        for r in rows:
            g = dict(r)
            p_home = float(g.get("prob_home_win") or 0.50)
            g["prob_away"] = round((1.0 - p_home) * 100, 1)
            g["prob_home"] = round(p_home * 100, 1)
            g["fav_team"] = g["home_team"] if p_home >= 0.50 else g["away_team"]
            g["fav_prob"] = max(g["prob_home"], g["prob_away"])
            g["is_final"] = (g.get("actual_score_away") is not None and g.get("actual_score_home") is not None)
            games.append(g)

    pitchers = []
    if "Pitcher_K_Forecasts" in tables:
        rows = c.execute("""
            SELECT 
                f.game_pk, f.pitcher_name, f.team_name, f.opponent_team,
                f.projected_pitches, f.expected_k, f.k_line, f.over_prob, f.under_prob,
                p.actual_k, p.actual_pitches, p.over_hit
            FROM Pitcher_K_Forecasts f
            LEFT JOIN Pitcher_Post_Mortem_Logs p ON f.game_pk = p.game_pk AND f.pitcher_name = p.pitcher_name
            ORDER BY f.expected_k DESC
        """).fetchall()
        pitchers = [dict(r) for r in rows]

    batters = []
    if "Batter_Hit_Forecasts" in tables:
        rows = c.execute("""
            SELECT 
                b.game_pk, b.player_name, b.team_name, b.batting_order,
                b.expected_hits, b.over_0_5_hit_prob,
                p.actual_hits, p.actual_ab, p.over_hit
            FROM Batter_Hit_Forecasts b
            LEFT JOIN Batter_Post_Mortem_Logs p ON b.game_pk = p.game_pk AND b.player_name = p.player_name
            ORDER BY b.over_0_5_hit_prob DESC
        """).fetchall()
        batters = [dict(r) for r in rows]

    accuracy = {
        "overall_right": 0, "overall_wrong": 0, "overall_pct": 0.0,
        "markets": {
            "moneyline": {"right": 0, "wrong": 0, "pct": 0.0, "label": "Full Game Moneyline"},
            "f5_ml": {"right": 0, "wrong": 0, "pct": 0.0, "label": "First 5 (F5) Winner"},
            "f5_total": {"right": 0, "wrong": 0, "pct": 0.0, "label": "First 5 (F5) Total O/U"},
            "pitchers": {"right": 0, "wrong": 0, "pct": 0.0, "label": "Pitcher Strikeout O/U"},
            "batters": {"right": 0, "wrong": 0, "pct": 0.0, "label": "Batter Over 0.5 Hits"}
        }
    }

    for g in games:
        if g["is_final"]:
            if g.get("hit_ml") == 1: accuracy["markets"]["moneyline"]["right"] += 1
            elif g.get("hit_ml") == 0: accuracy["markets"]["moneyline"]["wrong"] += 1
            if g.get("hit_f5_ml") == 1: accuracy["markets"]["f5_ml"]["right"] += 1
            elif g.get("hit_f5_ml") == 0: accuracy["markets"]["f5_ml"]["wrong"] += 1
            if g.get("hit_f5_total") == 1: accuracy["markets"]["f5_total"]["right"] += 1
            elif g.get("hit_f5_total") == 0: accuracy["markets"]["f5_total"]["wrong"] += 1

    for p in pitchers:
        if p.get("actual_k") is not None:
            is_over = p["actual_k"] > p["k_line"]
            model_over = p["over_prob"] >= 0.50
            if (model_over and is_over) or (not model_over and not is_over):
                accuracy["markets"]["pitchers"]["right"] += 1
            else:
                accuracy["markets"]["pitchers"]["wrong"] += 1

    for b in batters:
        if b.get("actual_hits") is not None:
            did_hit = b["actual_hits"] >= 1
            model_hit = b["over_0_5_hit_prob"] >= 0.55
            if (model_hit and did_hit) or (not model_hit and not did_hit):
                accuracy["markets"]["batters"]["right"] += 1
            else:
                accuracy["markets"]["batters"]["wrong"] += 1

    tot_r, tot_w = 0, 0
    for k, m in accuracy["markets"].items():
        sub = m["right"] + m["wrong"]
        m["pct"] = round((m["right"] / sub) * 100, 1) if sub > 0 else 0.0
        tot_r += m["right"]
        tot_w += m["wrong"]
    accuracy["overall_right"] = tot_r
    accuracy["overall_wrong"] = tot_w
    accuracy["overall_pct"] = round((tot_r / (tot_r + tot_w)) * 100, 1) if (tot_r + tot_w) > 0 else 0.0

    conn.close()

    html = f"""
    <!DOCTYPE html>
    <html lang="en" class="bg-gray-950 text-gray-100">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MLB Quantitative Hub</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="p-4 md:p-8 max-w-7xl mx-auto space-y-6 font-sans">
        <header class="flex flex-col md:flex-row justify-between items-start md:items-center pb-4 border-b border-gray-800 gap-4">
            <div>
                <h1 class="text-2xl font-black tracking-wide text-white uppercase">MLB Quantitative Engine</h1>
                <p class="text-xs text-gray-400 font-mono">Closed-Loop Learning • Moneyline, F5, Strikeouts & Hits</p>
            </div>
            <div class="flex flex-wrap gap-2">
                <button onclick="switchTab('matchups')" id="tab-btn-matchups" class="px-3 py-1.5 rounded bg-red-600 text-white font-bold text-xs">🏟️ Matchups & F5</button>
                <button onclick="switchTab('pitchers')" id="tab-btn-pitchers" class="px-3 py-1.5 rounded bg-gray-800 text-gray-400 font-bold text-xs">⚾ Pitcher Ks</button>
                <button onclick="switchTab('batters')" id="tab-btn-batters" class="px-3 py-1.5 rounded bg-gray-800 text-gray-400 font-bold text-xs">🎯 Batter Hits</button>
                <button onclick="switchTab('scorecard')" id="tab-btn-scorecard" class="px-3 py-1.5 rounded bg-gray-800 text-gray-400 font-bold text-xs">📊 Scorecard</button>
            </div>
        </header>

        <section id="section-matchups" class="space-y-4">
            <h2 class="text-sm font-black uppercase text-gray-400 font-mono">Matchups & First 5 (F5) Audit</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4" id="cards-container"></div>
        </section>

        <section id="section-pitchers" class="space-y-4 hidden">
            <h2 class="text-sm font-black uppercase text-gray-400 font-mono">Pitcher Strikeout Over/Under Audit</h2>
            <div class="overflow-x-auto bg-gray-900 border border-gray-800 rounded-lg p-2">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-gray-400 border-b border-gray-800 uppercase">
                        <tr>
                            <th class="p-2">Pitcher</th>
                            <th class="p-2">Opp</th>
                            <th class="p-2 text-right">xK</th>
                            <th class="p-2 text-center">Line</th>
                            <th class="p-2 text-right">Over %</th>
                            <th class="p-2 text-center">Actual Ks</th>
                            <th class="p-2 text-center">Audit</th>
                        </tr>
                    </thead>
                    <tbody id="pitchers-tbody" class="divide-y divide-gray-800"></tbody>
                </table>
            </div>
        </section>

        <section id="section-batters" class="space-y-4 hidden">
            <h2 class="text-sm font-black uppercase text-gray-400 font-mono">Batter Contact Props (Over 0.5 Hits) Audit</h2>
            <div class="overflow-x-auto bg-gray-900 border border-gray-800 rounded-lg p-2">
                <table class="w-full text-left text-xs font-mono">
                    <thead class="text-gray-400 border-b border-gray-800 uppercase">
                        <tr>
                            <th class="p-2">Hitter</th>
                            <th class="p-2">Team</th>
                            <th class="p-2 text-right">xHits</th>
                            <th class="p-2 text-right">Over 0.5 %</th>
                            <th class="p-2 text-center">Actual Hits</th>
                            <th class="p-2 text-center">Audit</th>
                        </tr>
                    </thead>
                    <tbody id="batters-tbody" class="divide-y divide-gray-800"></tbody>
                </table>
            </div>
        </section>

        <section id="section-scorecard" class="space-y-4 hidden">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-gray-900 border border-gray-800 p-4 rounded-lg">
                    <div class="text-xs uppercase text-gray-400">Total System Accuracy</div>
                    <div class="text-3xl font-black text-emerald-400 mt-2">{accuracy['overall_pct']}%</div>
                    <div class="text-xs text-gray-500 mt-1">{accuracy['overall_right']} Correct • {accuracy['overall_wrong']} Incorrect</div>
                </div>
            </div>
            <div class="bg-gray-900 border border-gray-800 p-4 rounded-lg space-y-4">
                <h3 class="text-xs font-bold uppercase tracking-wider text-gray-400">Market Accuracy Breakdown</h3>
                <div class="space-y-3" id="bars-container"></div>
            </div>
        </section>

        <script>
            const gamesData = {json.dumps(games)};
            const pitchersData = {json.dumps(pitchers)};
            const battersData = {json.dumps(batters)};
            const accuracyData = {json.dumps(accuracy)};

            function switchTab(tab) {{
                ['matchups', 'pitchers', 'batters', 'scorecard'].forEach(t => {{
                    document.getElementById('section-' + t).classList.add('hidden');
                    const b = document.getElementById('tab-btn-' + t);
                    b.classList.remove('bg-red-600', 'text-white');
                    b.classList.add('bg-gray-800', 'text-gray-400');
                }});
                document.getElementById('section-' + tab).classList.remove('hidden');
                const activeBtn = document.getElementById('tab-btn-' + tab);
                activeBtn.classList.add('bg-red-600', 'text-white');
                activeBtn.classList.remove('bg-gray-800', 'text-gray-400');
            }}

            const cardsBox = document.getElementById('cards-container');
            gamesData.forEach(g => {{
                let mlBadge = '<span class="text-gray-500 font-mono text-xs">Scheduled</span>';
                if (g.is_final) {{
                    mlBadge = g.hit_ml === 1 
                        ? '<span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 text-[10px] font-bold">ML HIT ✅</span>' 
                        : '<span class="px-1.5 py-0.5 rounded bg-red-950 text-red-400 border border-red-800 text-[10px] font-bold">ML MISS ❌</span>';
                }}

                let f5MlBadge = '<span class="text-gray-500 text-[10px]">Pending</span>';
                if (g.hit_f5_ml !== null && g.hit_f5_ml !== undefined) {{
                    f5MlBadge = g.hit_f5_ml === 1 
                        ? '<span class="text-emerald-400 font-bold">F5 ML ✅</span>' 
                        : '<span class="text-red-400 font-bold">F5 ML ❌</span>';
                }}

                let f5TotalBadge = '<span class="text-gray-500 text-[10px]">Pending</span>';
                if (g.hit_f5_total !== null && g.hit_f5_total !== undefined) {{
                    f5TotalBadge = g.hit_f5_total === 1 
                        ? '<span class="text-emerald-400 font-bold">F5 O/U ✅</span>' 
                        : '<span class="text-red-400 font-bold">F5 O/U ❌</span>';
                }}

                const actualF5Str = (g.f5_actual_away !== null && g.f5_actual_away !== undefined) 
                    ? `${{g.f5_actual_away}} - ${{g.f5_actual_home}} (${{g.f5_actual_away + g.f5_actual_home}} R)` 
                    : 'Scheduled';

                const card = document.createElement('div');
                card.className = "bg-gray-900 border border-gray-800 rounded-lg p-4 space-y-3";
                card.innerHTML = `
                    <div class="flex justify-between items-center text-xs font-bold">
                        <span class="text-gray-400">${{g.away_team}} @ ${{g.home_team}}</span>
                        ${{mlBadge}}
                    </div>
                    <div class="text-sm font-bold text-white flex justify-between">
                        <span>Fav: <span class="text-emerald-400">${{g.fav_team}} (${{g.fav_prob}}%)</span></span>
                        <span class="font-mono text-gray-300">${{g.actual_score_away ?? '-'}} - ${{g.actual_score_home ?? '-'}}</span>
                    </div>
                    <div class="bg-gray-950 p-2.5 rounded border border-gray-800/80 space-y-1.5 text-xs font-mono">
                        <div class="text-[10px] text-cyan-400 font-bold uppercase tracking-wider flex justify-between">
                            <span>First 5 Innings (F5)</span>
                            <span>Line: ${{g.f5_median_runs}}</span>
                        </div>
                        <div class="flex justify-between text-gray-300">
                            <span>Actual: ${{actualF5Str}}</span>
                            <div class="space-x-1">${{f5MlBadge}} • ${{f5TotalBadge}}</div>
                        </div>
                    </div>
                `;
                cardsBox.appendChild(card);
            }});

            const pTbody = document.getElementById('pitchers-tbody');
            pitchersData.forEach(p => {{
                let postMortem = '<span class="text-gray-500">Scheduled</span>';
                if (p.actual_k !== null && p.actual_k !== undefined) {{
                    const isOver = p.actual_k > p.k_line;
                    const leanedOver = p.over_prob >= 0.50;
                    const hit = (leanedOver && isOver) || (!leanedOver && !isOver);
                    postMortem = hit 
                        ? '<span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold">HIT ✅</span>' 
                        : '<span class="px-1.5 py-0.5 rounded bg-red-950 text-red-400 border border-red-800 font-bold">MISS ❌</span>';
                }}
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-2 text-white font-bold">${{p.pitcher_name}} (${{p.team_name}})</td>
                    <td class="p-2 text-gray-400">${{p.opponent_team}}</td>
                    <td class="p-2 text-right text-yellow-400">${{p.expected_k}}</td>
                    <td class="p-2 text-center text-white">${{p.k_line}}</td>
                    <td class="p-2 text-right text-emerald-400">${{(p.over_prob * 100).toFixed(1)}}%</td>
                    <td class="p-2 text-center font-bold text-white">${{p.actual_k ?? '-'}}</td>
                    <td class="p-2 text-center">${{postMortem}}</td>
                `;
                pTbody.appendChild(tr);
            }});

            const bTbody = document.getElementById('batters-tbody');
            battersData.forEach(b => {{
                let postMortem = '<span class="text-gray-500">Scheduled</span>';
                if (b.actual_hits !== null && b.actual_hits !== undefined) {{
                    const hit = (b.actual_hits >= 1 && b.over_0_5_hit_prob >= 0.55) || (b.actual_hits === 0 && b.over_0_5_hit_prob < 0.55);
                    postMortem = hit 
                        ? '<span class="px-1.5 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 font-bold">HIT ✅</span>' 
                        : '<span class="px-1.5 py-0.5 rounded bg-red-950 text-red-400 border border-red-800 font-bold">MISS ❌</span>';
                }}
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="p-2 text-white font-bold">${{b.player_name}}</td>
                    <td class="p-2 text-gray-400">${{b.team_name}} #${{b.batting_order}}</td>
                    <td class="p-2 text-right text-white">${{b.expected_hits}}</td>
                    <td class="p-2 text-right text-emerald-400">${{(b.over_0_5_hit_prob * 100).toFixed(1)}}%</td>
                    <td class="p-2 text-center font-bold text-white">${{b.actual_hits !== null ? `${{b.actual_hits}} (${{b.actual_ab}} AB)` : '-'}}</td>
                    <td class="p-2 text-center">${{postMortem}}</td>
                `;
                bTbody.appendChild(tr);
            }});

            const barsBox = document.getElementById('bars-container');
            Object.keys(accuracyData.markets).forEach(k => {{
                const m = accuracyData.markets[k];
                const total = m.right + m.wrong;
                const rPct = total > 0 ? ((m.right / total) * 100).toFixed(1) : 0;
                const wPct = total > 0 ? (100 - rPct).toFixed(1) : 0;

                const row = document.createElement('div');
                row.className = "space-y-1";
                row.innerHTML = `
                    <div class="flex justify-between text-xs font-mono">
                        <span class="text-white font-bold">${{m.label}}</span>
                        <span><span class="text-emerald-400">${{m.right}} Right</span> • <span class="text-red-400">${{m.wrong}} Wrong</span> (${{m.pct}}%)</span>
                    </div>
                    <div class="w-full bg-gray-800 h-2.5 rounded-full overflow-hidden flex">
                        <div class="bg-emerald-500 h-full" style="width: ${{rPct}}%"></div>
                        <div class="bg-red-500 h-full" style="width: ${{wPct}}%"></div>
                    </div>
                `;
                barsBox.appendChild(row);
            }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)
