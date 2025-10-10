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

from __future__ import annotations

import marimo
from types import ModuleType
from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # Imports used only for type checking
    import pandas as pd
    import numpy as np

__generated_with = "0.14.10"
app = marimo.App(
    width="full",
    app_title="Marin Speedrun",
    css_file="custom.css",
    html_head_file="head.html",
)


@app.cell
def base_import() -> Tuple[ModuleType]:
    """Import ``marimo`` and return it.

    Returns
    -------
    tuple
        A single-element tuple containing the imported ``marimo`` module. The
        tuple form is used so that the value can be captured by subsequent
        cells in the notebook.
    """
    import marimo as mo

    return (mo,)


@app.cell
def render_header(mo: ModuleType) -> None:
    """Render the page header.

    Parameters
    ----------
    mo : ModuleType
        The ``marimo`` module used to create HTML elements.

    Returns
    -------
    None
        The header is rendered for its side effect; nothing is returned.
    """
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
def render_intro_paragraph(mo: ModuleType) -> None:
    """Display the introductory paragraph.

    Parameters
    ----------
    mo : ModuleType
        The ``marimo`` module used for HTML rendering.

    Returns
    -------
    None
        This function only produces HTML output in the notebook.
    """
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
def load_leaderboard_data() -> Tuple[pd.DataFrame, pd.DataFrame, ModuleType]:
    """Load run and track information from disk.

    Returns
    -------
    tuple
        ``(df_runs, df_tracks, pd)`` where ``df_runs`` contains run metadata
        and ``df_tracks`` describes the available leaderboard tracks. The
        ``pd`` return value is the imported :mod:`pandas` module for reuse in
        later cells.
    """
    import json
    import pandas as pd

    df_runs = pd.read_json(
        "https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/runs.json"
    )
    df_runs = pd.json_normalize(df_runs.to_dict(orient="records"))
    df_tracks = pd.read_json(
        "https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/tracks.json"
    )
    return df_runs, df_tracks, pd


@app.cell
def render_track_tabs(
    df_tracks: pd.DataFrame, mo: ModuleType
) -> Tuple[Any, Dict[str, str], Any]:
    """Create tabs allowing the user to choose a track.

    Parameters
    ----------
    df_tracks : pandas.DataFrame
        Data frame describing all available leaderboard tracks.
    mo : ModuleType
        The ``marimo`` module for constructing UI elements.

    Returns
    -------
    tuple
        ``(q, tab_map, tabs)`` where ``q`` represents the current query
        parameters, ``tab_map`` maps the displayed tab name to its underlying
        track identifier and ``tabs`` is the tab widget itself.
    """
    q = mo.query_params()
    tab_map = {row["name"].capitalize(): row["id"] for _, row in df_tracks.iterrows()}
    tabs = mo.ui.tabs(
        {row["name"].capitalize(): "" for _, row in df_tracks.iterrows()},
        value=(q.get("track") or "Scaling").capitalize(),
    )
    tabs.center()
    return q, tab_map, tabs


@app.cell
def filter_data_by_selected_track(
    df_runs: pd.DataFrame,
    df_tracks: pd.DataFrame,
    pd: ModuleType,
    q: Any,
    tab_map: Dict[str, str],
    tabs: Any,
) -> Tuple[pd.DataFrame, float, pd.Series, str]:
    """Filter runs according to the selected track.

    Parameters
    ----------
    df_runs : pandas.DataFrame
        Data for all submitted runs.
    df_tracks : pandas.DataFrame
        Metadata describing available tracks.
    pd : ModuleType
        The :mod:`pandas` module.
    q : Any
        Query parameter object used to store the currently selected track.
    tab_map : dict[str, str]
        Mapping from tab label to track identifier.
    tabs : Any
        The tab widget created by :func:`render_track_tabs`.

    Returns
    -------
    tuple
        ``(filtered, next_lower, t, track_id)`` where ``filtered`` is the subset
        of ``df_runs`` matching the selected track, ``next_lower`` is the next
        best BPB threshold for the track, ``t`` is the row in ``df_tracks``
        describing the track and ``track_id`` is the selected track identifier.
    """
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
def compute_and_render_high_level_track_stats(
    filtered: pd.DataFrame, mo: ModuleType, track_id: str
) -> Tuple[float, ModuleType]:
    """Compute and display high-level statistics for a track.

    Parameters
    ----------
    filtered : pandas.DataFrame
        Data frame containing runs restricted to the chosen track.
    mo : ModuleType
        The ``marimo`` module used to render HTML.
    track_id : str
        Identifier of the selected track.

    Returns
    -------
    tuple
        ``(FLOPS_BUDGET, np)`` where ``FLOPS_BUDGET`` is the reference
        compute budget used for projections and ``np`` is the imported
        :mod:`numpy` module for reuse by subsequent cells.
    """
    import numpy as np

    FLOPS_BUDGET = 1e22
    best_flops_header = "Best FLOPs in Track"

    if track_id == "scaling":
        df = filtered.copy()
        df["lead_folder"] = df["run_name"].apply(lambda p: p.split("/")[0])
        preds = []
        group_scaling = {}
        for _n, _g in df.groupby("lead_folder"):
            x = np.log(_g["training_hardware_flops"])
            y = np.log(_g["eval_paloma_c4_en_bpb"])
            _slope, _intercept = np.polyfit(x, y, 1)
            preds.append(np.exp(_intercept + _slope * np.log(FLOPS_BUDGET)))
            group_scaling[_n] = {
                "slope": _slope,
                "intercept": _intercept,
                "projected": float(np.exp(_intercept + _slope * np.log(FLOPS_BUDGET))),
            }

        best_bpb_value = f"{min(preds):.4g}" if preds else "N/A"
        best_bpb_header = (
            f"Best Projected BPB @ {FLOPS_BUDGET:.0e}".replace("e+", "e")
            + ' FLOPs<br/><h4 style="margin-top: -20px;">(Approx. Compute Optimal 8B Model)</h4>'
        )
        best_flops_header = "Best Compute Scaling Term"
        best_flops_value = f"{_slope:.4g}"
        num_runs = len(df.drop_duplicates(subset=["lead_folder"]))
    else:
        group_scaling = None
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
    return FLOPS_BUDGET, group_scaling, np


@app.cell
def render_speedrun_plot(
    df_runs: pd.DataFrame,
    filtered: pd.DataFrame,
    mo: ModuleType,
    next_lower: float,
    np: ModuleType,
    t: pd.Series,
    track_id: str,
    FLOPS_BUDGET: float,
) -> Tuple[Dict[str, Dict[str, float]] | None]:
    """Plot the Pareto frontier for runs in the selected track.

    Parameters
    ----------
    df_runs : pandas.DataFrame
        All run information.
    filtered : pandas.DataFrame
        Subset of ``df_runs`` belonging to the current track.
    mo : ModuleType
        ``marimo`` module for rendering the plot.
    next_lower : float
        Lower BPB bound of the track (used for shading).
    np : ModuleType
        The :mod:`numpy` module.
    t : pandas.Series
        Row of ``df_tracks`` describing the track.
    track_id : str
        Identifier of the selected track.
    FLOPS_BUDGET : float
        Reference FLOPs budget used for scaling-law projection.

    Returns
    -------
    tuple
        A one-element tuple ``(group_scaling,)`` containing a dictionary of
        scaling statistics when ``track_id`` is ``"scaling"``. Otherwise the
        tuple contains ``None``.
    """
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
        df_all["is_baseline"] = df_all["lead_folder"] == "adamw_llama_scaling"
        groups = (
            df_all[df_all["in_track"]]
            .sort_values(by=["is_baseline", "training_flops"], ascending=False)
            .groupby("lead_folder", sort=False)
        )
        colors = px.colors.qualitative.Plotly
        baseline = [g for (n, g) in groups if n == "adamw_llama_scaling"][0]
        for i, (name, g) in enumerate(groups):
            color = (
                colors[(i - 1) % len(colors)]
                if name != "adamw_llama_scaling"
                else "gray"
            )
            legend_name = (
                name if name != "adamw_llama_scaling" else "Baseline (AdamW, Llama)"
            )
            baselined = (
                g["eval_paloma_c4_en_bpb"].values
                / baseline["eval_paloma_c4_en_bpb"].values
            )
            fig.add_trace(
                go.Scatter(
                    x=g["training_flops"],
                    y=baselined,
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=legend_name,
                    legendgroup=legend_name,
                    text=g["run_name"],
                    customdata=np.column_stack(
                        (g["training_flops"], g["eval_paloma_c4_en_bpb"])
                    ),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]:.2e} FLOPs<br>%{customdata[1]:.3f} BPB<extra></extra>",
                )
            )
            xlog = np.log(g["training_flops"])
            ylog = baselined
            slope, intercept = np.polyfit(xlog, ylog, 1)
            x_fit = np.logspace(
                np.log10(g["training_flops"].min()), np.log10(FLOPS_BUDGET), 100
            )
            y_fit = intercept + slope * np.log(x_fit)
            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color=color, dash="dash"),
                    name=f"{legend_name} fit",
                    hoverinfo="skip",
                    legendgroup=legend_name,
                    showlegend=False,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=[FLOPS_BUDGET, FLOPS_BUDGET],
                y=[
                    y_fit.min() - (y_fit.min() * 0.005),
                    y_fit.max() + (y_fit.max() * 0.005),
                ],
                mode="lines",
                line=dict(color="black", dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    else:
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
        title={
            "text": "Marin Speedrun<br>Hardware FLOPs v.s. C4-EN BPB",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(
            type="log",
            title="Training FLOPs",
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        yaxis=dict(
            title="C4-EN BPB Relative to Baseline"
            if track_id == "scaling"
            else "C4-EN BPB",
            tickformat=".1%" if track_id == "scaling" else ".2f",
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
        ),
        legend=dict(x=0.8, xanchor="center", y=0.8, yanchor="bottom"),
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
    )

    mo.ui.plotly(fig, label="Runs")

    return


@app.cell
def render_speedrun_leaderboard_table(
    filtered: pd.DataFrame,
    group_scaling: Dict[str, Dict[str, float]] | None,
    mo: ModuleType,
    pd: ModuleType,
    t: pd.Series,
    track_id: str,
    FLOPS_BUDGET: float,
) -> None:
    """Render the leaderboard table for the current track.

    Parameters
    ----------
    filtered : pandas.DataFrame
        Runs belonging to the selected track.
    group_scaling : dict[str, dict[str, float]] | None
        Scaling statistics returned from :func:`render_speedrun_plot` when the
        track is ``"scaling"``.
    mo : ModuleType
        ``marimo`` module for rendering UI elements.
    pd : ModuleType
        The :mod:`pandas` module.
    t : pandas.Series
        Row of ``df_tracks`` describing the selected track.
    track_id : str
        Identifier for the current track.
    FLOPS_BUDGET : float
        Compute budget used for scaling-law projections.

    Returns
    -------
    None
        The leaderboard is rendered to the notebook; the function does not
        return a value.
    """

    # ───────────────────────────── helpers ─────────────────────────────
    def fmt_model_size(x: float) -> str:
        """Format a parameter count as a human readable string."""
        if pd.isna(x):
            return "N/A"
        return f"{x / 1e6:.1f} M" if x < 1e9 else f"{x / 1e9:.1f} B"

    def fmt_flops(x: float) -> str:
        """Format a FLOP count in scientific notation."""
        return (
            "N/A" if pd.isna(x) else f"{x:.2E}".replace("E+0", "E").replace("E+", "E")
        )

    def fmt_date(ts: str) -> str:
        """Return the ISO date component of a timestamp string."""
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
                "Training Time": f"{r.training_time/60:.1f} m",
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
        .sort_values(by=[f"Projected BPB @ {FLOPS_BUDGET:.0e}".replace("e+", "e") + " FLOPs"] if track_id == "scaling" else ["Rank"])
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
