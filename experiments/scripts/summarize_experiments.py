import math
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

EXP_PATH = "experiments/experiment_log.csv"
FOLD_PATH = "experiments/fold_diagnostics.csv"

console = Console()


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def fmt_num(x, decimals=3):
    if pd.isna(x):
        return "nan"
    return f"{x:.{decimals}f}"


def fmt_pct(x, decimals=2):
    if pd.isna(x):
        return "nan"
    return f"{x:.{decimals}f}%"


def style_auc(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x >= 0.55:
        return (f"{x:.4f}", "bold green")
    if x >= 0.52:
        return (f"{x:.4f}", "bold yellow")
    return (f"{x:.4f}", "bold red")


def style_inv_auc(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x <= 0.45:
        return (f"{x:.4f}", "bold green")
    if x <= 0.48:
        return (f"{x:.4f}", "bold yellow")
    return (f"{x:.4f}", "bold red")


def style_gap(auc, inv_auc):
    auc = safe_float(auc)
    inv_auc = safe_float(inv_auc)
    if math.isnan(auc) or math.isnan(inv_auc):
        return ("nan", "dim")
    gap = auc - inv_auc
    if gap >= 0.05:
        return (f"{gap:.4f}", "bold green")
    if gap >= 0.02:
        return (f"{gap:.4f}", "bold yellow")
    return (f"{gap:.4f}", "bold red")


def style_logloss(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x <= 0.69:
        return (f"{x:.6f}", "bold green")
    if x <= 0.72:
        return (f"{x:.6f}", "bold yellow")
    return (f"{x:.6f}", "bold red")


def style_return(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x > 15:
        return (fmt_pct(x), "bold green")
    if x > 0:
        return (fmt_pct(x), "bold yellow")
    return (fmt_pct(x), "bold red")


def style_dd(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    ax = abs(x)
    if ax <= 10:
        return (fmt_pct(x), "bold green")
    if ax <= 20:
        return (fmt_pct(x), "bold yellow")
    return (fmt_pct(x), "bold red")


def style_sharpe(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x >= 1.5:
        return (f"{x:.2f}", "bold green")
    if x >= 0.75:
        return (f"{x:.2f}", "bold yellow")
    return (f"{x:.2f}", "bold red")


def style_pf(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x >= 1.5:
        return (f"{x:.2f}", "bold green")
    if x >= 1.0:
        return (f"{x:.2f}", "bold yellow")
    return (f"{x:.2f}", "bold red")


def style_trades(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x >= 100:
        return (str(int(x)), "bold green")
    if x >= 30:
        return (str(int(x)), "bold yellow")
    return (str(int(x)), "bold red")


def style_score(x):
    x = safe_float(x)
    if math.isnan(x):
        return ("nan", "dim")
    if x >= 1.5:
        return (f"{x:.2f}", "bold green")
    if x >= 0.75:
        return (f"{x:.2f}", "bold yellow")
    return (f"{x:.2f}", "bold red")


def style_regime(x):
    x = str(x).lower()
    if x == "bull":
        return Text("bull", style="bold green")
    if x == "bear":
        return Text("bear", style="bold red")
    return Text("sideways", style="bold yellow")


def make_run_table(row, title):
    table = Table(
        title=title,
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )

    table.add_column("Category", style="bold magenta", no_wrap=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="white")

    auc_text, auc_style = style_auc(row.get("oos_auc"))
    inv_auc_text, inv_auc_style = style_inv_auc(row.get("oos_auc_inverted"))
    gap_text, gap_style = style_gap(row.get("oos_auc"), row.get("oos_auc_inverted"))
    ll_text, ll_style = style_logloss(row.get("oos_logloss"))
    ret_text, ret_style = style_return(row.get("return_pct"))
    dd_text, dd_style = style_dd(row.get("max_drawdown_pct"))
    sharpe_text, sharpe_style = style_sharpe(row.get("sharpe"))
    pf_text, pf_style = style_pf(row.get("profit_factor"))
    trades_text, trades_style = style_trades(row.get("trades"))
    score_text, score_style = style_score(row.get("score_ret_dd", float("nan")))

    table.add_row("Run", "Timestamp", str(row.get("timestamp", "")))
    table.add_row("Run", "Git Commit", str(row.get("git_commit", ""))[:12])
    table.add_row("Run", "Source", str(row.get("data_source", "unknown")))
    table.add_row("Run", "Market", f"{row.get('symbol', '')} {row.get('interval', '')}")
    table.add_row("Run", "Model", str(row.get("model", "")))
    table.add_row("Run", "Regime", style_regime(row.get("regime_tag", "unknown")))

    table.add_row("Config", "Lookback Weeks", str(int(row["lookback_weeks"])) if not pd.isna(row.get("lookback_weeks")) else "nan")
    table.add_row("Config", "Commission", str(row.get("commission", "")))
    table.add_row("Config", "Horizon Bars", str(int(row["horizon_bars"])) if not pd.isna(row.get("horizon_bars")) else "nan")
    table.add_row("Config", "Threshold", str(row.get("threshold", "")))
    table.add_row(
        "Config",
        "Train / Test / Step",
        f"{int(row['train_size'])} / {int(row['test_size'])} / {int(row['step_size'])}"
        if not pd.isna(row.get("train_size")) else "nan",
    )
    table.add_row(
        "Config",
        "Purge / Embargo",
        f"{int(row['purge_size'])} / {int(row['embargo_size'])}"
        if not pd.isna(row.get("purge_size")) else "nan",
    )
    table.add_row(
        "Config",
        "Calibrate",
        f"{row.get('calibrate')} ({row.get('calibrator_method')}, frac={row.get('calibrator_val_frac')})",
    )

    table.add_row("Signal", "OOS AUC", Text(auc_text, style=auc_style))
    table.add_row("Signal", "OOS invAUC", Text(inv_auc_text, style=inv_auc_style))
    table.add_row("Signal", "AUC Gap", Text(gap_text, style=gap_style))
    table.add_row("Signal", "OOS LogLoss", Text(ll_text, style=ll_style))

    table.add_row("Backtest", "Return", Text(ret_text, style=ret_style))
    table.add_row("Backtest", "Buy & Hold", fmt_pct(row.get("buy_hold_return_pct", float("nan"))))
    table.add_row("Backtest", "Return - B&H", fmt_pct(row.get("return_over_bh", float("nan"))))
    table.add_row("Backtest", "Max Drawdown", Text(dd_text, style=dd_style))
    table.add_row("Backtest", "Return / |DD|", Text(score_text, style=score_style))
    table.add_row("Backtest", "Sharpe", Text(sharpe_text, style=sharpe_style))
    table.add_row("Backtest", "Profit Factor", Text(pf_text, style=pf_style))
    table.add_row("Backtest", "Trades", Text(trades_text, style=trades_style))
    table.add_row("Backtest", "Exposure", fmt_pct(row.get("exposure_pct", float("nan"))))
    table.add_row("Backtest", "Commissions Paid", f"${fmt_num(row.get('commissions_paid', float('nan')), 2)}")
    table.add_row("Backtest", "Final Equity", f"${fmt_num(row.get('equity_final', float('nan')), 2)}")
    table.add_row("Backtest", "Rows Backtest", str(int(row["rows_backtest"])) if not pd.isna(row.get("rows_backtest")) else "nan")

    return table


def make_fold_summary_table(folds: pd.DataFrame):
    table = Table(
        title="Fold Diagnostics Summary",
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )

    table.add_column("Run ID", style="bold white")
    table.add_column("Folds", justify="right")
    table.add_column("Mean AUC", justify="right")
    table.add_column("Min AUC", justify="right")
    table.add_column("Median AUC", justify="right")
    table.add_column("Max AUC", justify="right")
    table.add_column("AUC<0.50", justify="right")
    table.add_column("AUC>0.55", justify="right")
    table.add_column("Mean LogLoss", justify="right")
    table.add_column("Mean PosRate", justify="right")

    if folds.empty:
        table.add_row("No folds found", "-", "-", "-", "-", "-", "-", "-", "-", "-")
        return table

    grouped = (
        folds.groupby("run_id")
        .agg(
            folds=("fold_id", "count"),
            auc_mean=("auc", "mean"),
            auc_min=("auc", "min"),
            auc_median=("auc", "median"),
            auc_max=("auc", "max"),
            auc_lt_050=("auc", lambda s: int((s < 0.50).sum())),
            auc_gt_055=("auc", lambda s: int((s > 0.55).sum())),
            logloss_mean=("logloss", "mean"),
            pos_rate_mean=("pos_rate", "mean"),
        )
        .reset_index()
        .sort_values("run_id", ascending=False)
    )

    for _, row in grouped.iterrows():
        auc_mean_text, auc_mean_style = style_auc(row["auc_mean"])
        auc_min_text, auc_min_style = style_auc(row["auc_min"])
        auc_median_text, auc_median_style = style_auc(row["auc_median"])
        auc_max_text, auc_max_style = style_auc(row["auc_max"])
        ll_text, ll_style = style_logloss(row["logloss_mean"])

        table.add_row(
            str(row["run_id"]),
            str(int(row["folds"])),
            Text(auc_mean_text, style=auc_mean_style),
            Text(auc_min_text, style=auc_min_style),
            Text(auc_median_text, style=auc_median_style),
            Text(auc_max_text, style=auc_max_style),
            str(int(row["auc_lt_050"])),
            str(int(row["auc_gt_055"])),
            Text(ll_text, style=ll_style),
            fmt_num(row["pos_rate_mean"], 4),
        )

    return table


def make_signal_leaderboard_table(df: pd.DataFrame, title: str, top_n: int = 10, min_trades: int = 30):
    table = Table(
        title=title,
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=False,
        expand=True,
    )

    table.add_column("Timestamp", style="white")
    table.add_column("Source", style="bold white")
    table.add_column("Model", style="bold white")
    table.add_column("Market", style="bold white")
    table.add_column("Horizon", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("invAUC", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("LogLoss", justify="right")
    table.add_column("Trades", justify="right")

    if df.empty:
        table.add_row("No data", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
        return table

    filtered = df[df["trades"].fillna(0) >= min_trades].copy()
    if filtered.empty:
        table.add_row("No rows pass min_trades filter", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
        return table

    filtered["auc_gap"] = filtered["oos_auc"] - filtered["oos_auc_inverted"]
    ranked = filtered.sort_values(
        ["oos_auc", "auc_gap", "oos_logloss"],
        ascending=[False, False, True],
    ).head(top_n)

    for _, row in ranked.iterrows():
        auc_text, auc_style = style_auc(row["oos_auc"])
        inv_auc_text, inv_auc_style = style_inv_auc(row["oos_auc_inverted"])
        gap_text, gap_style = style_gap(row["oos_auc"], row["oos_auc_inverted"])
        ll_text, ll_style = style_logloss(row["oos_logloss"])
        trades_text, trades_style = style_trades(row["trades"])

        table.add_row(
            str(row["timestamp"]),
            str(row.get("data_source", "unknown")),
            str(row.get("model", "")),
            f"{row['symbol']} {row['interval']}",
            str(int(row["horizon_bars"])),
            str(row["threshold"]),
            Text(auc_text, style=auc_style),
            Text(inv_auc_text, style=inv_auc_style),
            Text(gap_text, style=gap_style),
            Text(ll_text, style=ll_style),
            Text(trades_text, style=trades_style),
        )

    return table


def make_backtest_leaderboard_table(df: pd.DataFrame, title: str, sort_col: str, ascending: bool = False, top_n: int = 10, min_trades: int = 30):
    table = Table(
        title=title,
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=False,
        expand=True,
    )

    table.add_column("Timestamp", style="white")
    table.add_column("Source", style="bold white")
    table.add_column("Model", style="bold white")
    table.add_column("Market", style="bold white")
    table.add_column("Horizon", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Return", justify="right")
    table.add_column("B&H", justify="right")
    table.add_column("R-|DD|", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("PF", justify="right")
    table.add_column("Trades", justify="right")

    if df.empty:
        table.add_row("No data", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
        return table

    filtered = df[df["trades"].fillna(0) >= min_trades].copy()
    if filtered.empty:
        table.add_row("No rows pass min_trades filter", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-")
        return table

    ranked = filtered.sort_values(sort_col, ascending=ascending).head(top_n)

    for _, row in ranked.iterrows():
        ret_text, ret_style = style_return(row["return_pct"])
        sharpe_text, sharpe_style = style_sharpe(row["sharpe"])
        pf_text, pf_style = style_pf(row["profit_factor"])
        trades_text, trades_style = style_trades(row["trades"])
        score_text, score_style = style_score(row.get("score_ret_dd", float("nan")))

        table.add_row(
            str(row["timestamp"]),
            str(row.get("data_source", "unknown")),
            str(row.get("model", "")),
            f"{row['symbol']} {row['interval']}",
            str(int(row["horizon_bars"])),
            str(row["threshold"]),
            Text(ret_text, style=ret_style),
            fmt_pct(row.get("buy_hold_return_pct", float("nan"))),
            Text(score_text, style=score_style),
            Text(sharpe_text, style=sharpe_style),
            Text(pf_text, style=pf_style),
            Text(trades_text, style=trades_style),
        )

    return table


def make_research_focus_table(df: pd.DataFrame):
    table = Table(
        title="Research Focus: What Matters Next",
        title_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True,
        expand=True,
    )

    table.add_column("Priority", style="bold magenta", no_wrap=True)
    table.add_column("Metric", style="bold white")
    table.add_column("Value", style="white")

    if df.empty:
        table.add_row("No data", "-", "-")
        return table

    filtered = df[df["trades"].fillna(0) >= 30].copy()
    if filtered.empty:
        filtered = df.copy()

    filtered["auc_gap"] = filtered["oos_auc"] - filtered["oos_auc_inverted"]

    best_signal = filtered.sort_values(
        ["oos_auc", "auc_gap", "oos_logloss"],
        ascending=[False, False, True],
    ).iloc[0]

    table.add_row("Best signal", "Timestamp", str(best_signal["timestamp"]))
    table.add_row("Best signal", "Source", str(best_signal.get("data_source", "unknown")))
    table.add_row("Best signal", "Model", str(best_signal.get("model", "")))
    table.add_row("Best signal", "Market", f"{best_signal['symbol']} {best_signal['interval']}")
    table.add_row("Best signal", "Horizon / Threshold", f"{int(best_signal['horizon_bars'])} / {best_signal['threshold']}")
    table.add_row("Best signal", "OOS AUC", style_auc(best_signal["oos_auc"])[0])
    table.add_row("Best signal", "invAUC", style_inv_auc(best_signal["oos_auc_inverted"])[0])
    table.add_row("Best signal", "AUC Gap", style_gap(best_signal["oos_auc"], best_signal["oos_auc_inverted"])[0])
    table.add_row("Best signal", "LogLoss", style_logloss(best_signal["oos_logloss"])[0])
    table.add_row("Best signal", "Edge", fmt_num(best_signal["oos_auc_inverted"] - 0.5, 4))
    table.add_row("Best signal", "Trades", str(int(best_signal["trades"])))

    return table

def export_html_dashboard(exp: pd.DataFrame):
    import os
    import webbrowser

    os.makedirs("experiments/reports", exist_ok=True)
    path = "experiments/reports/dashboard.html"

    df = exp.copy()

    # 🔥 core metrics
    df["edge"] = df["oos_auc_inverted"] - 0.5
    df["auc_gap"] = df["oos_auc"] - df["oos_auc_inverted"]

    df = df.sort_values("timestamp", ascending=False)

    def color(val, good, mid):
        if pd.isna(val):
            return "gray"
        if val >= good:
            return "limegreen"
        if val >= mid:
            return "orange"
        return "red"

    def color_inv(val, good, mid):
        if pd.isna(val):
            return "gray"
        if val <= good:
            return "limegreen"
        if val <= mid:
            return "orange"
        return "red"

    def row_html(row):
        return f"""
        <tr>
            <td>{row['timestamp']}</td>
            <td>{row.get('data_source')}</td>
            <td>{row.get('model')}</td>
            <td>{row.get('symbol')} {row.get('interval')}</td>
            <td>{int(row.get('horizon_bars', 0))}</td>
            <td>{row.get('threshold')}</td>

            <td style='color:{color(row['oos_auc'], 0.55, 0.52)}'>{row['oos_auc']:.4f}</td>
            <td style='color:{color_inv(row['oos_auc_inverted'], 0.45, 0.48)}'>{row['oos_auc_inverted']:.4f}</td>
            <td style='color:{color(row['auc_gap'], 0.05, 0.02)}'>{row['auc_gap']:.4f}</td>
            <td>{row['edge']:.4f}</td>

            <td style='color:{color(-row['oos_logloss'], -0.69, -0.72)}'>{row['oos_logloss']:.4f}</td>

            <td style='color:{color(row['return_pct'], 15, 0)}'>{row['return_pct']:.2f}%</td>
            <td style='color:{color(-abs(row['max_drawdown_pct']), -10, -20)}'>{row['max_drawdown_pct']:.2f}%</td>

            <td>{int(row.get('trades', 0))}</td>
        </tr>
        """

    def make_table(title, df_slice):
        rows = "".join([row_html(r) for _, r in df_slice.iterrows()])
        return f"""
        <h2>{title}</h2>
        <table>
        <tr>
            <th>Time</th><th>Source</th><th>Model</th><th>Market</th>
            <th>H</th><th>Thr</th>
            <th>AUC</th><th>invAUC</th><th>Gap</th><th>Edge</th>
            <th>LogLoss</th><th>Return</th><th>DD</th><th>Trades</th>
        </tr>
        {rows}
        </table>
        """

    # 🔥 sections
    latest = df.head(10)

    best_signal = df.sort_values(
        ["oos_auc", "auc_gap", "oos_logloss"],
        ascending=[False, False, True]
    ).head(10)

    best_return = df.sort_values("return_pct", ascending=False).head(10)

    best_edge = df.sort_values("edge", ascending=False).head(10)

    html = f"""
    <html>
    <head>
        <title>ML Trading Dashboard</title>
        <style>
            body {{
                font-family: Arial;
                background-color: #111;
                color: #eee;
                padding: 20px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 40px;
            }}
            th, td {{
                border: 1px solid #333;
                padding: 6px;
                text-align: right;
            }}
            th {{
                background-color: #222;
            }}
            h1, h2 {{
                color: cyan;
            }}
        </style>
    </head>
    <body>

    <h1>📊 ML Trading Research Dashboard</h1>

    {make_table("Latest Runs", latest)}
    {make_table("Best Signal (AUC)", best_signal)}
    {make_table("Best Edge", best_edge)}
    {make_table("Best Return", best_return)}

    </body>
    </html>
    """

    with open(path, "w") as f:
        f.write(html)

    print(f"📊 Dashboard updated -> {path}")

    # 🔥 auto open
    webbrowser.open(f"file://{os.path.abspath(path)}")

def main():
    try:
        exp = pd.read_csv(EXP_PATH)
    except FileNotFoundError:
        console.print(Panel(f"[bold red]Missing {EXP_PATH}[/bold red]", expand=False))
        return

    try:
        folds = pd.read_csv(FOLD_PATH, on_bad_lines="skip")
    except Exception as e:
        print(f"⚠️ Failed to read fold diagnostics: {e}")
        folds = pd.DataFrame()

    numeric_cols = [
        "commission", "lookback_weeks", "horizon_bars", "threshold", "train_size", "test_size", "step_size",
        "purge_size", "embargo_size", "calibrator_val_frac", "rows_backtest", "oos_auc",
        "oos_auc_inverted", "oos_logloss", "buy_hold_return_pct", "return_over_bh",
        "score_ret_dd", "return_pct", "max_drawdown_pct", "sharpe",
        "profit_factor", "trades", "exposure_pct", "commissions_paid", "equity_final"
    ]

    for col in numeric_cols:
        if col in exp.columns:
            exp[col] = pd.to_numeric(exp[col], errors="coerce")

    for col in ["fold_id", "auc", "auc_inverted", "logloss", "pos_rate", "n"]:
        if col in folds.columns:
            folds[col] = pd.to_numeric(folds[col], errors="coerce")

    if exp.empty:
        console.print(Panel("[bold red]No experiment rows found.[/bold red]", expand=False))
        return

    exp = exp.sort_values("timestamp", ascending=False).reset_index(drop=True)

    # 🔥 add edge metric
    if "oos_auc_inverted" in exp.columns:
        exp["edge"] = exp["oos_auc_inverted"] - 0.5

    latest = exp.iloc[0]
    best_signal = exp.assign(auc_gap=exp["oos_auc"] - exp["oos_auc_inverted"]).sort_values(
        ["oos_auc", "auc_gap", "oos_logloss"],
        ascending=[False, False, True],
    ).iloc[0]
    best_return = exp.sort_values("return_pct", ascending=False).iloc[0]
    best_score = exp.sort_values("score_ret_dd", ascending=False).iloc[0] if "score_ret_dd" in exp.columns else latest

    console.print(Panel("[bold green]Experiment Research Dashboard[/bold green]", expand=False))
    console.print(make_research_focus_table(exp))
    console.print(make_run_table(latest, "Latest Run"))
    console.print(make_run_table(best_signal, "Best by Signal Quality"))
    console.print(make_run_table(best_return, "Best by Return"))
    console.print(make_run_table(best_score, "Best by Return / |Drawdown|"))
    console.print(make_signal_leaderboard_table(exp, "Leaderboard: Signal Quality (trades>=30)", top_n=10, min_trades=30))
    console.print(make_backtest_leaderboard_table(exp, "Leaderboard: Best Return (trades>=30)", "return_pct", ascending=False, min_trades=30))
    console.print(make_backtest_leaderboard_table(exp, "Leaderboard: Best Return / |DD| (trades>=30)", "score_ret_dd", ascending=False, min_trades=30))
    console.print(make_fold_summary_table(folds))

    # 🔥 export dashboard
    export_html_dashboard(exp)

if __name__ == "__main__":
    main()