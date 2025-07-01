# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "numpy==2.2.6",
#   "pandas==2.3.0",
#   "plotly==6.2.0",
#   "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.14.9"
app = marimo.App(
    width="full",
    layout_file="layouts/speedrun_web.grid.json",
    css_file="custom.css",
    html_head_file="head.html",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
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
    options = {row["name"]: row["id"] for _, row in df_tracks.iterrows()}
    print(options)

    track_select = mo.ui.radio(
        options=options, value="All Runs", label="Track", inline=True
    )
    track_select
    return (track_select,)


@app.cell
def _(df_runs, df_tracks, pd, track_select):
    track_id = track_select.value

    filtered = df_runs

    if track_id != "all":
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

            filtered = (
                df_runs[df_runs["eval_paloma_c4_en_bpb"].notna()]
                .loc[
                    lambda d: (d.eval_paloma_c4_en_bpb <= t.target_bpb)
                            & (d.eval_paloma_c4_en_bpb > next_lower)
                ]
            )
    return filtered, next_lower, t, track_id


@app.cell
def _(filtered, mo):
    import numpy as np

    best_flops = (
        filtered.training_hardware_flops.min() if not filtered.empty else np.nan
    )
    best_bpb = filtered.eval_paloma_c4_en_bpb.min() if not filtered.empty else np.nan
    stats = mo.Html(
        f"""
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">Total Runs in Track</h3>
            <div id="total-runs" class="text-2xl font-bold">{len(filtered)}</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">Best FLOPs in Track</h3>
            <div id="best-flops" class="text-2xl font-bold">{best_flops}</div>
        </div>
        <div class="bg-white rounded-lg shadow p-6">
            <h3 class="text-lg font-medium mb-4">Best C4-EN BPB</h3>
            <div id="best-bpb" class="text-2xl font-bold">{best_bpb}</div>
        </div>
    </div>
    """
    )
    stats
    return (np,)


@app.cell
def _(df_runs, filtered, mo, next_lower, np, t, track_id):
    import plotly.graph_objects as go

    df_all = df_runs.copy()
    df_all["training_flops_ef"] = df_all["training_hardware_flops"] / 1e18  # 1 EF = 1e18 FLOPs
    df_all["in_track"] = df_all.index.isin(filtered.index)

    track_color = (t.color if track_id != "all" else "#1877F2")  # fallback blue

    fig = go.Figure()
    fig.layout = None
    fig.data = None


    fig.add_trace(
        go.Scatter(
            x=df_all["training_flops_ef"],
            y=df_all["eval_paloma_c4_en_bpb"],
            mode="markers",
            marker=dict(color="rgba(156,163,175,0.3)", size=8, line=dict(width=0)),
            name="All Runs",
            text=df_all["run_name"],
            customdata=np.column_stack((df_all["training_flops_ef"],)),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2f} EF<br>%{y:.3f} BPB<extra></extra>",
        )
    )

    highlight = df_all[df_all["in_track"]]
    if not highlight.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight["training_flops_ef"],
                y=highlight["eval_paloma_c4_en_bpb"],
                mode="markers",
                marker=dict(color=track_color, size=12, opacity=0.9,
                            line=dict(color="white", width=1)),
                name="Selected Track",
                text=highlight["run_name"],
                customdata=np.column_stack((highlight["training_flops_ef"],)),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2f} EF<br>%{y:.3f} BPB<extra></extra>",
            )
        )

    valid = df_all.dropna(subset=["eval_paloma_c4_en_bpb"])
    valid = valid[valid["training_hardware_flops"] > 0]

    def dominated(row):
        return ((valid["training_hardware_flops"] <= row.training_hardware_flops) &
                (valid["eval_paloma_c4_en_bpb"] <  row.eval_paloma_c4_en_bpb) &
                ((valid["training_hardware_flops"] < row.training_hardware_flops) |
                 (valid["eval_paloma_c4_en_bpb"] < row.eval_paloma_c4_en_bpb))).any()

    pareto_df = valid[~valid.apply(dominated, axis=1)].sort_values("training_hardware_flops")

    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["training_flops_ef"],
                y=pareto_df["eval_paloma_c4_en_bpb"],
                mode="lines+markers",
                line=dict(color="#FF2D55", width=2),
                marker=dict(size=5, color="#FF2D55"),
                name="Pareto Frontier",
                hoverinfo="skip",
            )
        )

    print(track_id)
    if track_id != "all":
        x_min, x_max = df_all["training_flops_ef"].min(), df_all["training_flops_ef"].max()
    
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
                fill="tonexty",          # ← the magic
                fillcolor=f"rgba{(*tuple(int(track_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)), 0.15)}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Overall Pareto Frontier: FLOPs vs C4-EN BPB",
        xaxis=dict(
            type="log",
            title="Training FLOPs (ExaFLOPs)",
            tickformat=".1f",
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
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=1.05, yanchor="bottom"),
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
    )

    print(fig)
    mo.ui.plotly(fig, label="Runs")

    return


@app.cell
def _(filtered, mo, pd, t, track_id):
    # ───────────────────────────── helpers ─────────────────────────────
    def fmt_model_size(x: float) -> str:
        if pd.isna(x):
            return "N/A"
        return f"{x/1e6:.1f} M" if x < 1e9 else f"{x/1e9:.1f} B"

    def fmt_flops(x: float) -> str:
        return "N/A" if pd.isna(x) else f"{x:.2E}".replace("E+0", "E").replace("E+", "E")

    def fmt_date(ts: str) -> str:
        if not ts:
            return "N/A"
        ts = ts.replace(" UTC", "")
        return pd.to_datetime(ts).date().isoformat()

    # ─────────────────────────── table rows ────────────────────────────
    rows = []
    for rank, (_, r) in enumerate(
        filtered.sort_values("training_hardware_flops").iterrows(), start=1
    ):
        author = r["author.name"]
        if pd.notna(r["author.affiliation"]):
            author += f"\n{r['author.affiliation']}"

        wandb = r.wandb_link if pd.notna(r.wandb_link) else "N/A"

        rows.append(
            {
                "Rank": rank,
                "Run Name": r.run_name,
                "Author": author,
                "Date Added": fmt_date(r.run_completion_timestamp),
                "Model Size*": fmt_model_size(r.model_size),
                "Training Time": f"{r.training_time:.1f} m",
                "Total FLOPs*": fmt_flops(r.training_hardware_flops),
                "C4-EN BPB": f"{r.eval_paloma_c4_en_bpb:.3f}",
                "W&B Run": wandb,
            }
        )

    df_disp = pd.DataFrame(rows)

    # ──────────────────────────── headers ──────────────────────────────
    if track_id != "all":
        header = mo.Html(
            f"""
            <div style='display:flex;align-items:center;gap:6px'>
              <span style='width:14px;height:14px;border-radius:2px;background:{t.color}'></span>
              <h3 style='margin:0'>{t.name} ({t.target_bpb:.2f} BPB) Leaderboard</h3>
            </div>
            """
        )
        subtitle = mo.Html(
            f"<div style='color:#6b7280;font-size:0.9rem'>"
            f"Runs achieving ≤ {t.target_bpb:.9f} C4-EN BPB, ranked by training efficiency (FLOPs)"
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
    table = mo.ui.table(df_disp, label="Leaderboard")  # plain strings only
    mo.vstack([header, subtitle, table, footnotes])

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
