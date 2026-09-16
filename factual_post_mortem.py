import post_match_analysis

def run_factual_post_mortem():
    """Delegates to the primary post_match_analysis module to avoid duplicate logic and database lock contentions."""
    print("[DELEGATION] Running factual post mortem via post_match_analysis...")
    post_match_analysis.run_post_match_analysis()

if __name__ == "__main__":
    run_factual_post_mortem()
