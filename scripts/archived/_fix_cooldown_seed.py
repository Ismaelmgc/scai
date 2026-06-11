"""One-time fix: seed cooldown for pre-existing trailing stop exits."""
import json

COOLDOWN_DAYS = 5

for label, path in [("BASELINE", "data/paper_trading/portfolio.json"),
                    ("ADAPTIVE", "data/paper_trading/adaptive/portfolio.json")]:
    with open(path) as f:
        state = json.load(f)

    current_idx = state["current_day_idx"]

    # Initialize cooldown fields if missing
    if "cooldown_until" not in state:
        state["cooldown_until"] = {}
    if "cooldown_days" not in state:
        state["cooldown_days"] = COOLDOWN_DAYS

    # Retroactively set cooldown for recent trailing_stop exits
    for trade in state.get("closed_trades", []):
        if trade["exit_reason"] == "trailing_stop":
            ticker = trade["ticker"]
            state["cooldown_until"][ticker] = current_idx + COOLDOWN_DAYS
            print(f"{label}: Set cooldown for {ticker} until day_idx={current_idx + COOLDOWN_DAYS}")

    # Remove pending signals that are in cooldown
    before = len(state["pending_signals"])
    state["pending_signals"] = [
        s for s in state["pending_signals"]
        if s["ticker"] not in state["cooldown_until"]
        or state["cooldown_until"][s["ticker"]] <= current_idx
    ]
    after = len(state["pending_signals"])
    if before != after:
        print(f"{label}: Removed {before - after} pending signals in cooldown")

    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    print(f"{label}: Saved")
    print()
