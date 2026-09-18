import sqlite3
from datetime import datetime

def seed_empirical_batting_baselines():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Empirical 2025 Team Offense Baselines
    team_baselines = [
        ("Toronto Blue Jays", 0.761, 0.755, 0.778, 4.93),
        ("Philadelphia Phillies", 0.759, 0.762, 0.748, 4.80),
        ("Milwaukee Brewers", 0.736, 0.732, 0.745, 4.98),
        ("Boston Red Sox", 0.745, 0.750, 0.731, 4.85),
        ("Athletics", 0.749, 0.744, 0.760, 4.52),
        ("Los Angeles Dodgers", 0.768, 0.772, 0.758, 5.09),
        ("San Diego Padres", 0.711, 0.708, 0.718, 4.33),
        ("Arizona Diamondbacks", 0.757, 0.760, 0.748, 4.88),
        ("Tampa Bay Rays", 0.714, 0.710, 0.725, 4.40),
        ("New York Yankees", 0.787, 0.785, 0.792, 5.24),
        ("Miami Marlins", 0.708, 0.705, 0.715, 4.38),
        ("Houston Astros", 0.714, 0.718, 0.703, 4.23),
        ("Chicago Cubs", 0.751, 0.754, 0.742, 4.89),
        ("New York Mets", 0.753, 0.755, 0.748, 4.73),
        ("Kansas City Royals", 0.706, 0.708, 0.700, 4.02),
        ("Detroit Tigers", 0.711, 0.715, 0.698, 4.68),
        ("Atlanta Braves", 0.740, 0.745, 0.728, 4.65),
        ("Baltimore Orioles", 0.745, 0.748, 0.735, 4.85),
        ("Chicago White Sox", 0.640, 0.635, 0.655, 3.45),
        ("Cincinnati Reds", 0.725, 0.720, 0.738, 4.50),
        ("Cleveland Guardians", 0.728, 0.730, 0.722, 4.45),
        ("Colorado Rockies", 0.715, 0.712, 0.722, 4.20),
        ("Los Angeles Angels", 0.710, 0.715, 0.695, 4.10),
        ("Minnesota Twins", 0.735, 0.740, 0.720, 4.55),
        ("Pittsburgh Pirates", 0.695, 0.690, 0.710, 4.15),
        ("San Francisco Giants", 0.715, 0.718, 0.705, 4.35),
        ("Seattle Mariners", 0.725, 0.730, 0.710, 4.50),
        ("St. Louis Cardinals", 0.720, 0.725, 0.705, 4.45),
        ("Texas Rangers", 0.730, 0.735, 0.715, 4.60),
        ("Washington Nationals", 0.705, 0.700, 0.718, 4.25)
    ]

    for t_name, ops, ops_r, ops_l, bsr in team_baselines:
        cursor.execute('''
        INSERT OR REPLACE INTO Team_Offense 
        (team_name, ops, ops_vs_rhp, ops_vs_lhp, bsr_per_game, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (t_name, ops, ops_r, ops_l, bsr, now_ts))

    # 2. Universal Batting Slot Averages (League-Wide Initialization)
    lineup_slot_baselines = {
        1: (0.265, 0.095, 0.190, 0.310),
        2: (0.260, 0.090, 0.200, 0.305),
        3: (0.255, 0.088, 0.210, 0.300),
        4: (0.245, 0.095, 0.240, 0.290),
        5: (0.240, 0.085, 0.235, 0.285),
        6: (0.238, 0.080, 0.225, 0.280),
        7: (0.235, 0.075, 0.230, 0.280),
        8: (0.230, 0.075, 0.245, 0.275),
        9: (0.225, 0.070, 0.250, 0.270)
    }

    for t_name, ops, _, _, _ in team_baselines:
        team_avg_scalar = (ops / 0.720) 
        
        for slot, (b_avg, b_bb, b_k, b_babip) in lineup_slot_baselines.items():
            adj_avg = round(b_avg * ((team_avg_scalar + 1.0) / 2.0), 3)
            generic_name = f"Batter {slot}"
            
            cursor.execute('''
            INSERT OR IGNORE INTO Batter_Stats 
            (player_name, team_name, avg, avg_vs_rhp, avg_vs_lhp, bb_rate, k_rate, babip, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (generic_name, t_name, adj_avg, adj_avg, adj_avg, b_bb, b_k, b_babip, now_ts))

    conn.commit()
    conn.close()
    print("✅ Baseline factors successfully injected. Ready for backtesting.")

if __name__ == "__main__":
    seed_empirical_batting_baselines()
