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
    import base64
    from pathlib import Path as _Path

    # Prefer local repo asset; fall back to remote only if unavailable
    logo_src = (
        "https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/assets/marin-logo.png"
    )
    try:
        logo_path = _Path(__file__).resolve().parent.parent / "assets" / "marin-logo.png"
        if logo_path.exists():
            logo_src = "data:image/png;base64," + base64.b64encode(logo_path.read_bytes()).decode(
                "ascii"
            )
    except Exception:
        # Keep remote fallback if any issue occurs reading the local file
        pass

    mo.Html(
        f"""
    <header class="bg-marin-dark text-white shadow-lg">
        <div class="container mx-auto px-4 py-6">
            <div class="flex items-center gap-5 group hover:opacity-90 transition-opacity duration-150">
                <img src="{logo_src}" alt="Marin Logo" class="h-14 w-14 object-contain">
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
    import pandas as _pd
    import marimo as _mo
    from pathlib import Path as _Path

    REMOTE_RUNS_URL = (
        "https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/runs.json"
    )
    REMOTE_TRACKS_URL = (
        "https://raw.githubusercontent.com/marin-community/speedrun/refs/heads/main/data/tracks.json"
    )

    repo_root = _Path(__file__).resolve().parent.parent
    local_runs = repo_root / "data" / "runs.json"
    local_tracks = repo_root / "data" / "tracks.json"

    @_mo.cache
    def load_json_df(local_path: str, remote_url: str):
        """Load JSON as a DataFrame from local path, else fall back to remote URL.

        Cached to avoid refetching during reactive updates.
        """
        from pathlib import Path as __Path
        import pandas as __pd

        try:
            p = __Path(local_path)
            if p.exists():
                return __pd.read_json(p)
        except Exception:
            pass
        return __pd.read_json(remote_url)

    df_runs = load_json_df(str(local_runs), REMOTE_RUNS_URL)
    df_runs = _pd.json_normalize(df_runs.to_dict(orient="records"))
    df_tracks = load_json_df(str(local_tracks), REMOTE_TRACKS_URL)
    pd = _pd
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
) -> Tuple[float, Dict[str, Dict[str, float]] | None, ModuleType]:
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
        ``(FLOPS_BUDGET, group_scaling, np)`` where ``FLOPS_BUDGET`` is the
        reference compute budget, ``group_scaling`` stores scaling stats for
        the scaling track and ``np`` is :mod:`numpy`.
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
def render_axis_selectors(mo: ModuleType) -> Tuple[Any, Any]:
    """Create axis selection controls for the plot."""

    x_axis_options = {
        "Model FLOPs": "model_flops",
        "Training Hardware FLOPs": "training_hardware_flops",
    }
    x_axis_select = mo.ui.radio(
        options=x_axis_options,
        value="Training Hardware FLOPs",
        label="X-axis metric",
    )

    y_axis_options = {
        "C4-EN BPB (relative to baseline)": "relative",
        "C4-EN BPB (absolute)": "absolute",
    }
    y_axis_select = mo.ui.radio(
        options=y_axis_options,
        value="C4-EN BPB (relative to baseline)",
        label="Y-axis metric",
    )

    return x_axis_select, y_axis_select


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
    x_axis_select: Any,
    y_axis_select: Any,
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
    import plotly.express as px
    import plotly.graph_objects as go

    # ───────────────────────── axis mapping & selections ─────────────────────────
    x_axis_map = {
        "model_flops": (
            "model_flops",
            "Training Model FLOPs",
            "Model FLOPs",
        ),
        "training_hardware_flops": (
            "training_hardware_flops",
            "Training Hardware FLOPs",
            "Hardware FLOPs",
        ),
    }

    selected_x = x_axis_select.value
    if isinstance(selected_x, tuple):
        selected_x = selected_x[1] if len(selected_x) > 1 else selected_x[0]
    x_column, x_axis_title, x_hover_label = x_axis_map.get(
        selected_x, x_axis_map["training_hardware_flops"]
    )
    if x_column not in df_runs.columns:
        x_column, x_axis_title, x_hover_label = x_axis_map["training_hardware_flops"]

    selected_y = y_axis_select.value
    if isinstance(selected_y, tuple):
        selected_y = selected_y[1] if len(selected_y) > 1 else selected_y[0]

    relative_requested = selected_y == "relative"

    df_all = df_runs.copy()
    df_all["x_value"] = df_all[x_column]
    df_all["in_track"] = df_all.index.isin(filtered.index)

    track_color = t.color if track_id != "all" else "#1877F2"

    # ───────────────────────── baseline for relative values ──────────────────────
    baseline_bpb = None
    baseline_run_name = None
    if track_id not in ("all", "scaling"):
        baseline_run_name = t.get("run_name") if hasattr(t, "get") else None
        if baseline_run_name:
            baseline_row = df_runs[df_runs["run_name"] == baseline_run_name]
            if not baseline_row.empty:
                baseline_value = baseline_row.iloc[0]["eval_paloma_c4_en_bpb"]
                if baseline_value is not None:
                    try:
                        baseline_bpb = float(baseline_value)
                    except (TypeError, ValueError):
                        baseline_bpb = None
                    else:
                        if np.isnan(baseline_bpb):
                            baseline_bpb = None

    def build_customdata(dataframe, use_relative):
        if use_relative:
            data = np.column_stack(
                (dataframe["x_value"], dataframe["eval_paloma_c4_en_bpb"])
            )
            hover = (
                f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                "<br>Relative BPB: %{y:.3f}<br>Absolute BPB: %{customdata[1]:.3f}<extra></extra>"
            )
        elif baseline_bpb is not None:
            ratios = dataframe["eval_paloma_c4_en_bpb"] / baseline_bpb
            data = np.column_stack(
                (dataframe["x_value"], dataframe["eval_paloma_c4_en_bpb"], ratios)
            )
            hover = (
                f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                "<br>Absolute BPB: %{y:.3f}<br>Relative BPB: %{customdata[2]:.3f}<extra></extra>"
            )
        else:
            data = np.column_stack(
                (dataframe["x_value"], dataframe["eval_paloma_c4_en_bpb"])
            )
            hover = (
                f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                "<br>Absolute BPB: %{y:.3f}<extra></extra>"
            )
        return data, hover

    # ───────────────────────── uirevision keys ─────────────────────────
    layout_revision = f"track-{track_id}"
    xaxis_revision = f"x-{selected_x}-{track_id}"
    yaxis_revision = f"y-{'relative' if (relative_requested and (track_id=='scaling' or baseline_bpb is not None)) else 'absolute'}-{track_id}"

    fig = go.Figure()
    shapes = []

    # ───────────────────────── build traces ────────────────────────────
    if track_id == "scaling":
        df_all["lead_folder"] = df_all["run_name"].apply(lambda p: p.split("/")[0])
        df_all["is_baseline"] = df_all["lead_folder"] == "adamw_llama_scaling"
        groups = (
            df_all[df_all["in_track"]]
            .sort_values(by=["is_baseline", "x_value"], ascending=False)
            .groupby("lead_folder", sort=False)
        )
        colors = px.colors.qualitative.Plotly
        baseline_group = [g for (n, g) in groups if n == "adamw_llama_scaling"]
        baseline_group = baseline_group[0] if baseline_group else None

        for i, (name, g) in enumerate(groups):
            color = "gray" if name == "adamw_llama_scaling" else colors[(i - 1) % len(colors)]
            legend_name = name if name != "adamw_llama_scaling" else "Baseline (AdamW, Llama)"

            absolute_values = g["eval_paloma_c4_en_bpb"].to_numpy()
            x_vals = g["x_value"].to_numpy()
            valid = np.isfinite(x_vals) & np.isfinite(absolute_values) & (x_vals > 0)
            x_plot = x_vals[valid]
            abs_plot = absolute_values[valid]

            # y-values & hover
            if relative_requested and baseline_group is not None and not baseline_group.empty:
                try:
                    base_abs = baseline_group.sort_values("x_value")["eval_paloma_c4_en_bpb"].to_numpy()
                    cur_abs = g.sort_values("x_value")["eval_paloma_c4_en_bpb"].to_numpy()
                    if len(base_abs) == len(cur_abs) and len(cur_abs) > 0:
                        y_plot = cur_abs / base_abs
                        x_plot = g.sort_values("x_value")["x_value"].to_numpy()
                        customdata = np.column_stack((x_plot, cur_abs))
                        hovertemplate = (
                            f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                            "<br>Relative BPB: %{y:.3f}<br>Absolute BPB: %{customdata[1]:.3f}<extra></extra>"
                        )
                    else:
                        y_plot = abs_plot
                        customdata = np.column_stack((x_plot, abs_plot))
                        hovertemplate = (
                            f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                            "<br>Absolute BPB: %{y:.3f}<extra></extra>"
                        )
                except Exception:
                    y_plot = abs_plot
                    customdata = np.column_stack((x_plot, abs_plot))
                    hovertemplate = (
                        f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                        "<br>Absolute BPB: %{y:.3f}<extra></extra>"
                    )
            else:
                y_plot = abs_plot
                if baseline_group is not None and not baseline_group.empty and len(abs_plot) > 0:
                    try:
                        base_abs = baseline_group.sort_values("x_value")["eval_paloma_c4_en_bpb"].to_numpy()
                        cur_abs = g.sort_values("x_value")["eval_paloma_c4_en_bpb"].to_numpy()
                        if len(base_abs) == len(cur_abs) and len(cur_abs) > 0:
                            relative_vals = cur_abs / base_abs
                            customdata = np.column_stack((np.sort(x_plot), abs_plot, relative_vals[: len(abs_plot)]))
                            hovertemplate = (
                                f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                                "<br>Absolute BPB: %{y:.3f}<br>Relative BPB: %{customdata[2]:.3f}<extra></extra>"
                            )
                        else:
                            customdata = np.column_stack((x_plot, abs_plot))
                            hovertemplate = (
                                f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                                "<br>Absolute BPB: %{y:.3f}<extra></extra>"
                            )
                    except Exception:
                        customdata = np.column_stack((x_plot, abs_plot))
                        hovertemplate = (
                            f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                            "<br>Absolute BPB: %{y:.3f}<extra></extra>"
                        )
                else:
                    customdata = np.column_stack((x_plot, abs_plot))
                    hovertemplate = (
                        f"<b>%{{text}}</b><br>{x_hover_label}: %{{customdata[0]:.2e}}"
                        "<br>Absolute BPB: %{y:.3f}<extra></extra>"
                    )

            # Points trace — always present
            fig.add_trace(
                go.Scatter(
                    x=x_plot.tolist(),
                    y=y_plot.tolist(),
                    mode="markers",
                    marker=dict(color=color, size=10),
                    name=legend_name,
                    legendgroup=legend_name,
                    text=g["run_name"].tolist(),
                    customdata=customdata if len(x_plot) else None,
                    hovertemplate=hovertemplate if len(x_plot) else None,
                    uid=f"scaling-points-{name}",
                    visible="legendonly" if len(x_plot) == 0 else True,
                )
            )

            # Fit line — keep identity even if empty
            if len(x_plot) >= 2:
                xlog = np.log(x_plot)
                slope, intercept = np.polyfit(xlog, y_plot, 1)
                x_fit = np.logspace(np.log10(x_plot.min()), np.log10(max(FLOPS_BUDGET, x_plot.max())), 80)
                y_fit = intercept + slope * np.log(x_fit)
                fig.add_trace(
                    go.Scatter(
                        x=x_fit.tolist(),
                        y=y_fit.tolist(),
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"{legend_name} fit",
                        hoverinfo="skip",
                        legendgroup=legend_name,
                        showlegend=False,
                        uid=f"scaling-fit-{name}",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        mode="lines",
                        line=dict(color=color, dash="dash"),
                        name=f"{legend_name} fit",
                        hoverinfo="skip",
                        legendgroup=legend_name,
                        showlegend=False,
                        uid=f"scaling-fit-{name}",
                    )
                )

        # Vertical budget line as a SHAPE so it doesn't affect y autorange
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="paper",
                x0=FLOPS_BUDGET,
                x1=FLOPS_BUDGET,
                y0=0,
                y1=1,
                line=dict(color="black", dash="dash", width=1),
                layer="below",
            )
        )

    else:
        use_relative = relative_requested and baseline_bpb is not None
        if use_relative:
            df_all["y_value"] = df_all["eval_paloma_c4_en_bpb"] / baseline_bpb
        else:
            df_all["y_value"] = df_all["eval_paloma_c4_en_bpb"]

        # All runs — always present
        customdata_all, hover_all = build_customdata(df_all, use_relative)
        fig.add_trace(
            go.Scatter(
                x=df_all["x_value"],
                y=df_all["y_value"],
                mode="markers",
                marker=dict(color="rgba(156,163,175,0.3)", size=8, line=dict(width=0)),
                name="All Runs",
                text=df_all["run_name"],
                customdata=customdata_all,
                hovertemplate=hover_all,
                uid="trace-all-runs",
            )
        )

        # Selected track — always present (empty -> legendonly)
        highlight = df_all[df_all["in_track"]]
        if not highlight.empty:
            customdata_highlight, hover_highlight = build_customdata(
                highlight, use_relative
            )
            x_h = highlight["x_value"]
            y_h = highlight["y_value"]
            text_h = highlight["run_name"].tolist()
            cd_h = customdata_highlight
            hover_h = hover_highlight
        else:
            x_h, y_h, text_h, cd_h, hover_h = [], [], [], None, None

        fig.add_trace(
            go.Scatter(
                x=x_h,
                y=y_h,
                mode="markers",
                marker=dict(
                    color=track_color,
                    size=12,
                    opacity=0.9,
                    line=dict(color="white", width=1),
                ),
                name="Selected Track",
                text=text_h,
                customdata=cd_h,
                hovertemplate=hover_h,
                uid=f"trace-selected-track-{track_id}",
                visible=True if len(x_h) > 0 else "legendonly",
            )
        )

        # Pareto frontier — always present
        valid = df_all.dropna(subset=["x_value", "y_value"])
        valid = valid[valid["x_value"] > 0]

        def dominated(row):
            return (
                (valid["x_value"] <= row.x_value)
                & (valid["y_value"] < row.y_value)
                & (
                    (valid["x_value"] < row.x_value)
                    | (valid["y_value"] < row.y_value)
                )
            ).any()

        if not valid.empty:
            pareto_df = valid[~valid.apply(dominated, axis=1)].sort_values("x_value")
            x_p = pareto_df["x_value"].tolist()
            y_p = pareto_df["y_value"].tolist()
        else:
            x_p, y_p = [], []

        fig.add_trace(
            go.Scatter(
                x=x_p,
                y=y_p,
                mode="lines+markers",
                line=dict(color="#FF2D55", width=2),
                marker=dict(size=5, color="#FF2D55"),
                name="Pareto Frontier",
                hoverinfo="skip",
                uid=f"trace-pareto-{track_id}",
                visible=True if len(x_p) > 0 else "legendonly",
            )
        )

        # Target & band as SHAPES (don't affect autorange; keep trace set stable)
        if track_id not in ["all", "scaling"]:
            x_min = float(df_all["x_value"].min()) if df_all["x_value"].notna().any() else 1.0
            x_max = float(df_all["x_value"].max()) if df_all["x_value"].notna().any() else 10.0
            target_value = (
                t.target_bpb / baseline_bpb
                if (baseline_bpb is not None and use_relative)
                else t.target_bpb
            )
            lower_bound = (
                next_lower / baseline_bpb
                if (baseline_bpb is not None and use_relative)
                else next_lower
            )
            # target line
            shapes.append(
                dict(
                    type="line",
                    xref="x",
                    yref="y",
                    x0=x_min,
                    x1=x_max,
                    y0=target_value,
                    y1=target_value,
                    line=dict(color=track_color, dash="dash", width=2),
                    layer="below",
                )
            )
            # band
            y0, y1 = sorted([lower_bound, target_value])
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=x_min,
                    x1=x_max,
                    y0=y0,
                    y1=y1,
                    fillcolor=f"rgba{(*tuple(int(track_color.lstrip('#')[i:i+2], 16) for i in (0,2,4)), 0.15)}",
                    line=dict(width=0),
                    layer="below",
                )
            )

    # ───────────────────────── axis titles & formats ─────────────────────────
    if track_id == "scaling":
        y_axis_title = (
            "C4-EN BPB Relative to Baseline"
            if relative_requested
            else "C4-EN BPB"
        )
        y_tickformat = ".1%" if relative_requested else ".3f"
    else:
        use_relative = relative_requested and baseline_bpb is not None
        y_axis_title = (
            "C4-EN BPB Relative to Baseline" if use_relative else "C4-EN BPB"
        )
        y_tickformat = ".1%" if use_relative else ".3f"

    # ───────────────────────── layout (with uirevisions) ───────────────────────
    fig.update_layout(
        uirevision=layout_revision,  # persist legend & zoom across metric toggles
        title={
            "text": f"Marin Speedrun<br>{x_axis_title} vs. {y_axis_title}",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis=dict(
            type="log",
            title=x_axis_title,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            autorange=True,
            uirevision=xaxis_revision,  # refresh x range when metric changes
        ),
        yaxis=dict(
            title=y_axis_title,
            tickformat=y_tickformat,
            ticks="outside",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.05)",
            autorange=True,
            uirevision=yaxis_revision,  # refresh y range when metric changes
        ),
        legend=dict(x=0.8, xanchor="center", y=0.8, yanchor="bottom"),
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=50),
        shapes=shapes,
    )

    control_panel = mo.vstack(
        [
            mo.md("**Plot axes**"),
            mo.md("_Choose which FLOPs estimate and BPB scale to visualize._"),
            mo.hstack([x_axis_select, y_axis_select]),
        ]
    )

    fig_widget = mo.ui.plotly(fig)
    mo.vstack([fig_widget, control_panel])

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
        show_column_summaries=False,
        show_data_types=False,
    )  # plain strings only
    layout = mo.vstack([header, subtitle, table, footnotes])
    layout
    return


if __name__ == "__main__":
    app.run()
