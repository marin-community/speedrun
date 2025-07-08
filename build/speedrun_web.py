# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy==2.2.6",
#   "pandas==2.3.0",
#   "plotly==6.2.0",
#   "pyarrow",
#   "fsspec==2025.5.1",
#   "requests",
#   "aiohttp",
# ]
# ///

import marimo

__generated_with = "0.14.10"
app = marimo.App(
    width="full",
    app_title="Marin Speedrun",
    css_file="custom.css",
    html_head_file="head.html",
)


@app.cell
def _():
    """Import and return the ``marimo`` module."""
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    """Render the page header with logo and title."""
    mo.Html(
        f"""
    <header class="bg-marin-dark text-white shadow-lg">
        <div class="container mx-auto px-4 py-6">
            <div class="flex items-center gap-5 group hover:opacity-90 transition-opacity duration-150">
                <img src="https://github.com/marin-community/speedrun/blob/5684db1a26feca7d32855a359fca7e0cfaec4267/assets/marin-logo.png?raw=true" alt="Marin Logo" class="h-14 w-14 object-contain">
                <div class="flex flex-col justify-center">
                    <h1 class="text-3xl font-bold leading-tight">Marin Speedrun - Leaderboard</h1>
                    <p class="text-gray-300 text-sm mt-0.5">Community-driven model training leaderboard</p>
                </div>
            </div>
        </div>
    </header>
    """
    )
    return


@app.cell
def _(mo):
    """Display introductory content explaining Speedrun."""
    mo.Html(
        """
    <div class="bg-white rounded-lg shadow p-8 mb-8">
        <div class="flex justify-between items-start mb-6">
            <h2 class="text-3xl font-extrabold text-gray-900 font-display">What is Speedrun?</h2>
            <a href="https://github.com/marin-community/marin/blob/main/docs/tutorials/submitting-speedrun.md" target="_blank" rel="noopener" class="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-marin-blue hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-marin-blue transition-colors duration-150">
                Get started with Speedrun
            </a>
        </div>
        <p class="text-gray-600 leading-relaxed text-lg mb-8">
            Speedrun is a community-driven initiative by the <a href="https://marin.community/" target="_blank" rel="noopener" class="text-marin-blue hover:text-blue-600 transition-colors duration-150">Marin project</a> to track and optimize the training efficiency of large language models. 
            Have a new architecture or training procedure that you think is more efficient? Participate in the Marin speedrun competition (inspired by 
            the <a href="https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#world-record-history" class="text-marin-blue hover:text-blue-600 transition-colors duration-150">nanogpt speedrun</a>), pick your compute budget, and create the fastest method to train a model to a 
            certain quality! 
        </p>
        <p class="text-gray-600 leading-relaxed text-lg mb-8">
            On this page, you can find leaderboards for different speedrun tracks, each targeting a specific loss threshold. You can click on any run to view the code that
            generated it, or view the Weights & Biases link for the model! We also track the overall Pareto frontier of models, allowing us to track efficiency-performance tradeoffs across all tracks.
        </p>
        <p class="text-gray-600 leading-relaxed text-lg">
            We invite you to join us in the search for more performant and efficient training methods!
        </p>
    </div>
                """
    )
    return


@app.cell
def _():
    """Load run and track data from JSON files."""
    import json
    import pandas as pd

    with open("../data/runs.json", "r") as f:
        runs = json.load(f)
    with open("../data/tracks.json", "r") as f:
        tracks = json.load(f)
    df_runs = pd.json_normalize(runs)
    df_tracks = pd.DataFrame(tracks)
    return df_runs, df_tracks, pd


@app.cell
def _(df_tracks, mo):
    """Create tabs for selecting a leaderboard track."""
    q = mo.query_params()
    tab_map = {row["name"].capitalize(): row["id"] for _, row in df_tracks.iterrows()}
    tabs = mo.ui.tabs(
        {row["name"].capitalize(): "" for _, row in df_tracks.iterrows()},
        value=(q.get("track") or "Scaling").capitalize(),
    )
    tabs.center()
    return q, tab_map, tabs


@app.cell
def _(df_runs, df_tracks, pd, q, tab_map, tabs):
    """Filter runs according to the selected track."""
    track_id = tab_map[tabs.value]
    q["track"] = tabs.value
    filtered = df_runs

    if track_id == "scaling":
        t = df_tracks.loc[df_tracks["id"] == track_id].iloc[0]
        filtered = df_runs[df_runs["run_name"].str.contains("/")]
    elif track_id != "all":
        t = df_tracks.loc[df_tracks["id"] == track_id].iloc[0]

        if pd.notna(t["target_bpb"]):
            sorted_tracks = (
                df_tracks[(df_tracks["id"] != "all") & df_tracks["target_bpb"].notna()]
                .sort_values("target_bpb", ascending=False)
                .reset_index(drop=True)
            )

            idx = sorted_tracks.index[sorted_tracks["id"] == track_id][0]

            next_lower = (
                sorted_tracks.loc[idx + 1, "target_bpb"]
                if idx < len(sorted_tracks) - 1
                else 0
            )

            filtered = df_runs[df_runs["eval_paloma_c4_en_bpb"].notna()].loc[
                lambda d: (d.eval_paloma_c4_en_bpb <= t.target_bpb)
                & (d.eval_paloma_c4_en_bpb > next_lower)
            ]
    return filtered, next_lower, t, track_id


@app.cell
def _(filtered, mo, track_id):
    """Compute summary statistics for the selected track."""
    import numpy as np

    FLOPS_BUDGET = 2e24
    best_flops_header = "Best FLOPs in Track"

    if track_id == "scaling":
        df = filtered.copy()
        df["lead_folder"] = df["run_name"].apply(lambda p: p.split("/")[0])
        preds = []
        for _, _g in df.groupby("lead_folder"):
            x = np.log(_g["training_hardware_flops"])
            y = np.log(_g["eval_paloma_c4_en_bpb"])
            _slope, _intercept = np.polyfit(x, y, 1)
            preds.append(np.exp(_intercept + _slope * np.log(FLOPS_BUDGET)))

        best_bpb_value = f"{min(preds):.4g}" if preds else "N/A"
        best_bpb_header = (
            f"Best Projected BPB @ {FLOPS_BUDGET:.0e}".replace("e+", "e")
            + " FLOPs<br/>(Approx. Llama 3 8B Compute)"
        )
        best_flops_header = "Best Compute Scaling Term"
        best_flops_value = f"{_slope:.4g}"
        num_runs = len(df.drop_duplicates(subset=["lead_folder"]))
    else:
        best_flops = (
            filtered.training_hardware_flops.min() if not filtered.empty else np.nan
        )
        best_flops_value = f"{best_flops:.4g}"
        best_bpb = (
            filtered.eval_paloma_c4_en_bpb.min() if not filtered.empty else np.nan
        )
        best_bpb_header = "Best C4-EN BPB"
        best_bpb_value = f"{best_bpb:.4g}"
        num_runs = len(filtered)

    stats = mo.Html(
        f"""
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">Total Runs in Track</h3>
            <div id="total-runs" class="text-2xl font-bold">{num_runs}</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">{best_flops_header}</h3>
            <div id="best-flops" class="text-2xl font-bold">{best_flops_value}</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">{best_bpb_header}</h3>
            <div id="best-bpb" class="text-2xl font-bold">{best_bpb_value}</div>
        </div>
    </div>
    """
    )
    stats
    return FLOPS_BUDGET, np


@app.cell
def _(df_runs, filtered, mo, next_lower, np, t, track_id, FLOPS_BUDGET):
    """Plot the Pareto frontier for runs in the selected track."""
    import plotly.graph_objects as go
    import plotly.express as px

    df_all = df_runs.copy()
    df_all["training_flops"] = df_all["training_hardware_flops"]
    df_all["in_track"] = df_all.index.isin(filtered.index)

    track_color = t.color if track_id != "all" else "#1877F2"  # fallback blue

    if track_id == "scaling":
        fig = go.Figure()
        fig.layout = None
        fig.data = None
        df_all["lead_folder"] = df_all["run_name"].apply(lambda p: p.split("/")[0])
        groups = df_all[df_all["in_track"]].groupby("lead_folder")
        colors = px.colors.qualitative.Plotly
        group_scaling = {}
        for i, (name, g) in enumerate(groups):
            color = colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=g["training_flops"],
                    y=g["eval_paloma_c4_en_bpb"],
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=name,
                    text=g["run_name"],
                    customdata=np.column_stack((g["training_flops"],)),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2e} FLOPs<br>%{y:.3f} BPB<extra></extra>",
                )
            )
            xlog = np.log(g["training_flops"])
            ylog = np.log(g["eval_paloma_c4_en_bpb"])
            slope, intercept = np.polyfit(xlog, ylog, 1)
            group_scaling[name] = {
                "slope": slope,
                "intercept": intercept,
                "projected": float(np.exp(intercept + slope * np.log(FLOPS_BUDGET))),
            }
            x_fit = np.logspace(
                np.log10(g["training_flops"].min()), np.log10(FLOPS_BUDGET), 100
            )
            y_fit = np.exp(intercept + slope * np.log(x_fit))
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    name=f"{name} fit",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=[FLOPS_BUDGET, FLOPS_BUDGET],
                y=[
                    0,
                    df_all.eval_paloma_c4_en_bpb.max(),
                ],
                mode="lines",
                line=dict(color="black", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
        group_scaling = None
        fig = go.Figure()
        fig.layout = None
        fig.data = None
        fig.add_trace(
            go.Scatter(
                x=df_all["training_flops"],
                y=df_all["eval_paloma_c4_en_bpb"],
                mode="markers",
                marker=dict(color="rgba(156,163,175,0.3)", size=8, line=dict(width=0)),
                name="All Runs",
                text=df_all["run_name"],
                customdata=np.column_stack((df_all["training_flops"],)),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2e} FLOPs<br>%{y:.3f} BPB<extra></extra>",
            )
        )
        highlight = df_all[df_all["in_track"]]
        if not highlight.empty:
            fig.add_trace(
                go.Scatter(
                    x=highlight["training_flops"],
                    y=highlight["eval_paloma_c4_en_bpb"],
                    mode="markers",
                    marker=dict(
                        color=track_color,
                        size=12,
                        opacity=0.9,
                        line=dict(color="white", width=1),
                    ),
                    name="Selected Track",
                    text=highlight["run_name"],
                    customdata=np.column_stack((highlight["training_flops"],)),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2e} FLOPs<br>%{y:.3f} BPB<extra></extra>",
                )
            )

    valid = df_all.dropna(subset=["eval_paloma_c4_en_bpb"])
    valid = valid[valid["training_hardware_flops"] > 0]

    def dominated(row):
        return (
            (valid["training_hardware_flops"] <= row.training_hardware_flops)
            & (valid["eval_paloma_c4_en_bpb"] < row.eval_paloma_c4_en_bpb)
            & (
                (valid["training_hardware_flops"] < row.training_hardware_flops)
                | (valid["eval_paloma_c4_en_bpb"] < row.eval_paloma_c4_en_bpb)
            )
        ).any()

    pareto_df = valid[~valid.apply(dominated, axis=1)].sort_values(
        "training_hardware_flops"
    )

    if not pareto_df.empty and track_id != "scaling":
        fig.add_trace(
            go.Scatter(
                x=pareto_df["training_flops"],
                y=pareto_df["eval_paloma_c4_en_bpb"],
                mode="lines+markers",
                line=dict(color="#FF2D55", width=2),
                marker=dict(size=5, color="#FF2D55"),
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

    if track_id not in ["all", "scaling"]:
        x_min, x_max = (
            df_all["training_flops"].min(),
            df_all["training_flops"].max(),
        )

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[t.target_bpb, t.target_bpb],
                mode="lines",
                line=dict(color=track_color, dash="dash", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[next_lower, next_lower],
                mode="lines",
                line=dict(color=track_color, dash="dash", width=1),
                fill="tonexty",
                fillcolor=f"rgba{(*tuple(int(track_color.lstrip('#')[i : i + 2], 16) for i in (0, 2, 4)), 0.15)}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Overall Pareto Frontier: FLOPs vs C4-EN BPB",
        xaxis=dict(
            type="log",
            title="Training FLOPs",
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            title="C4-EN BPB",
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=0.9, yanchor="bottom"),
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
    )

    mo.ui.plotly(fig, label="Runs")

    return (group_scaling,)


@app.cell
def _(filtered, group_scaling, mo, pd, t, track_id, FLOPS_BUDGET):
    """Render the leaderboard table for the current track."""

    # ───────────────────────────── helpers ─────────────────────────────
    def fmt_model_size(x: float) -> str:
        if pd.isna(x):
            return "N/A"
        return f"{x / 1e6:.1f} M" if x < 1e9 else f"{x / 1e9:.1f} B"

    def fmt_flops(x: float) -> str:
        return (
            "N/A" if pd.isna(x) else f"{x:.2E}".replace("E+0", "E").replace("E+", "E")
        )

    def fmt_date(ts: str) -> str:
        if not ts:
            return "N/A"
        ts = ts.replace(" UTC", "")
        return pd.to_datetime(ts).date().isoformat()

    # ─────────────────────────── table rows ────────────────────────────
    rows = []
    for rank, (_, r) in enumerate(
        filtered.sort_values("eval_paloma_c4_en_bpb", ascending=True).iterrows(),
        start=1,
    ):
        _website = r["author.url"]
        _name = r["author.name"]
        author = f'<a href="{_website}" title="{_website}" target="_blank" rel="noopener" class="text-marin-blue hover:text-blue-600 transition-colors duration-150">{_name}</a>'
        if pd.notna(r["author.affiliation"]):
            author += f"<br/>{r['author.affiliation']}"
        author = {"mimetype": "text/html", "data": author}

        wandb = r.wandb_link if pd.notna(r.wandb_link) else "N/A"
        experiment_file = "https://github.com/marin-community/marin/tree/main/" + (
            r["results_filepath"]
            if track_id != "scaling"
            else "/".join(r["results_filepath"].split("/")[:-1])
        )
        _run_name = (
            r["run_name"]
            if track_id != "scaling"
            else r["run_name"].split("/")[0].strip()
        )

        if track_id != "scaling":
            perf = {
                "Model Size*": fmt_model_size(r.model_size),
                "Training Time": f"{r.training_time:.1f} m",
                "Total FLOPs*": fmt_flops(r.training_hardware_flops),
                "C4-EN BPB": f"{r.eval_paloma_c4_en_bpb:.3f}",
            }
        else:
            perf = {
                "Scaling Law Intercept": f"{group_scaling[_run_name]['intercept']:.3f}",
                "Scaling Law Slope": f"{group_scaling[_run_name]['slope']:.3f}",
                (
                    f"Projected BPB @ {FLOPS_BUDGET:.0e}".replace("e+", "e") + " FLOPs"
                ): f"{group_scaling[_run_name]['projected']:.3f}",
            }

        rows.append(
            {
                "Rank": rank,
                "Run Name": {
                    "mimetype": "text/html",
                    "data": f'<a href="{experiment_file}" title="{experiment_file}" target="_blank" rel="noopener" class="text-marin-blue hover:text-blue-600 transition-colors duration-150">{_run_name}</a>',
                },
                "Author": author,
                "Date Added": fmt_date(r.run_completion_timestamp),
                **perf,
                "W&B Run": {
                    "mimetype": "text/html",
                    "data": f'<a href="{wandb}" title="{wandb}" target="_blank" rel="noopener" class="text-marin-blue hover:text-blue-600 transition-colors duration-150">View Run</a>',
                },
            }
        )

    df_disp = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["Run Name"])
        .sort_values(by=["Scaling Law Slope"] if track_id == "scaling" else ["Rank"])
    )

    # ──────────────────────────── headers ──────────────────────────────
    if track_id == "scaling":
        header = mo.Html(
            f"""
            <div style='display:flex;align-items:center;gap:6px'>
              <span style='width:14px;height:14px;border-radius:2px;background:{t.color}'></span>
              <h3 style='margin:0'>{t["name"]} Leaderboard</h3>
            </div>"""
        )
        subtitle = mo.Html(
            "<div style='color:#6b7280;font-size:0.9rem'>Runs grouped by leading folder with log-linear fits</div>"
        )
    elif track_id != "all":
        header = mo.Html(
            f"""
            <div style='display:flex;align-items:center;gap:6px'>
              <span style='width:14px;height:14px;border-radius:2px;background:{t.color}'></span>
              <h3 style='margin:0'>{t["name"]} ({t.target_bpb:.2f} BPB) Leaderboard</h3>
            </div>
            """
        )
        subtitle = mo.Html(
            f"<div style='color:#6b7280;font-size:0.9rem'>"
            f"Runs achieving ≤ {t.target_bpb:.4f} C4-EN BPB, ranked by training efficiency (FLOPs)"
            f"</div>"
        )
    else:
        header = mo.Html("<h3>All Runs Leaderboard</h3>")
        subtitle = mo.Html("")

    # ──────────────────────────── footnotes ────────────────────────────
    footnotes = mo.Html(
        "<div style='font-size:1rem;color:#6b7280;margin-top:6px'>"
        "* Model size here refers to the total number of trainable parameters<br>"
        "* Total FLOPs here refers to hardware FLOPs performed during training"
        "</div>"
    )

    # ──────────────────────────── assemble ─────────────────────────────
    df_disp["Rank"] = df_disp.reset_index().index + 1
    table = mo.ui.table(
        df_disp.set_index("Rank").sort_values(by="Rank"),
        label="Leaderboard",
        selection=None,
    )  # plain strings only
    mo.vstack([header, subtitle, table, footnotes])
    return


if __name__ == "__main__":
    app.run()
