# agents.md — How coding agents should write Marimo code for the Marin Speedrun

This guide teaches coding agents to generate **correct, reactive, and production-grade Marimo notebooks**. Marimo is a Python-first, git-friendly notebook system where notebooks are **pure Python files** with **reactive execution**.

---

## 1) Core mental model

**Notebook = DAG of cells.** Marimo statically infers read/write dependencies across cells to build a **directed acyclic graph (DAG)**. When a cell’s output changes, Marimo **re-runs only downstream dependents** and marks others stale. There’s **no hidden state**: deleting a cell removes its variables from memory.

**Implications for agents**

* Treat data as **immutable**—prefer recomputation to mutation. Mutations to objects won’t trigger recomputation in dependents.
* Write **idempotent** cells: same inputs → same outputs.
* Express dependencies **explicitly** by variable names and **return values**.

---

## 2) Minimal, valid Marimo file

```python
# my_notebook.py
import marimo

app = marimo.App()

@app.cell
def _():
    import marimo as mo
    return mo,

@app.cell
def _(mo):
    x = 21
    y = 2
    mo.md(f"**x * y = {x * y}**")
    return x, y

@app.cell
def _(x, y):
    z = x * y
    return z,

if __name__ == "__main__":
    app.run()
```

**Rules the agent must follow**

* Always create exactly one `marimo.App()` per file.
* Decorate executable cells with `@app.cell`.
* A cell’s function arguments should be the variables it **reads** (e.g., `(x, y)`), and its `return` should expose the variables it **defines** (tuple).
* Use the underscore function name `_` unless you have a reason to name it—Marimo doesn’t rely on the function name.
* End files with the `if __name__ == "__main__": app.run()` guard so the script can be run directly.

---

## 3) Returning values and dependency wiring

* **Return a tuple** of exported variables: `return a, b` or `return (a, b)`.
* If a cell defines helpers that should **not** be exported, keep them local or return only the public names.
* **Never rely on global mutation** to share values; **always return** values and consume them via parameters in downstream cells.

**Example**

```python
@app.cell
def _(df):
    # good: compute new_df without mutating df
    new_df = df.assign(total=lambda t: t.qty * t.price)
    return new_df,
```

---

## 4) Rendering & markdown

Prefer `mo.md` for rich text and layout:

```python
@app.cell
def _(mo, z):
    mo.md(f"""
    # Results
    - Computed **z** = `{z}`
    """)
    return
```

**Layout helpers**

* `marimo.hstack(a, b, ...)` and `marimo.vstack(...)` to arrange outputs.
* `mo.show_code()` if you want to display code in app view.

---

## 5) Interactivity with `mo.ui`

Use `mo.ui.*` elements; they are **reactive** (changing a widget re-runs dependents):

```python
@app.cell
def _(mo):
    slider = mo.ui.slider(0, 100, value=50, label="threshold")
    select = mo.ui.select(options=["a","b","c"], value="b", label="group")
    mo.vstack(slider, select).display()
    return slider, select

@app.cell
def _(slider, select):
    threshold = slider.value
    group = select.value
    # use threshold and group downstream
    return threshold, group
```

**Agent constraints**

* Don’t write `on_change` handlers. Reactivity is automatic; depend on `.value` in downstream cells.
* Keep UI creation and consumption in **separate** cells when possible; this clarifies dependencies and improves caching.

---

## 6) Working with data & SQL

Marimo supports SQL directly while staying pure Python:

```python
@app.cell
def _(mo, con):
    max_rows = mo.ui.slider(10, 1000, value=100, label="max rows")
    max_rows.display()
    df = mo.sql(f"SELECT * FROM my_table LIMIT {max_rows.value}", connection=con)
    return df,
```

* Keep database connections in a dedicated cell (e.g., `con = duckdb.connect()`), returned for use by SQL cells.
* Use `.display()` on UI elements to render them.

---

## 7) Caching expensive work

Use Marimo’s decorators to avoid recomputing:

```python
import marimo as mo

@mo.cache
def embed(texts, model="all-MiniLM"):
    # compute embeddings (fast cache)
    ...

@mo.persistent_cache
def load_big_model(tag: str):
    # expensive init, cache on disk
    ...
```

**Patterns**

* Call cached helpers **inside cells** and return results.
* Use `mo.persistent_cache` when results should survive restarts.

---

## 8) Performance & correctness checklist

* [ ] Cells are **idempotent** and avoid side effects.
* [ ] No reliance on mutation of shared objects to propagate changes.
* [ ] Heavy computations wrapped with `mo.cache`/`mo.persistent_cache`.
* [ ] UI elements created once; downstream cells consume `.value`.
* [ ] Clear separation of data **loading**, **transform**, **viz** cells.
* [ ] All consumed variables appear as function **parameters**; all produced variables are in the **return tuple**.

---

## 9) Conversions, exports, and running

* **Run/edit**: `marimo edit my_notebook.py` or run as a Python script with `python my_notebook.py`.
* **Convert from Jupyter**: `marimo convert notebook.ipynb -o notebook.py`.
* **Export**: `marimo export my_notebook.py --format html` (or run from the UI) to generate static HTML.
* **Autorun control**: if opening legacy notebooks that shouldn’t autorun, set `[runtime] auto_instantiate = false` in `marimo.toml`.

---

## 10) Common pitfalls for agents

* **Forgetting return values** → downstream cells won’t see variables.
* **Mutating shared state** (e.g., `df.append(...)` in-place) → dependents won’t update. Prefer new objects.
* **Embedding long-running work in UI cells** → separate UI from compute to enable caching and reduce churn.
* **Manual event handlers** (`on_change`) → unnecessary; use reactive values.
* **Non-deterministic cells** (random seeds, time) without controlling inputs → breaks idempotence. If needed, pass seeded RNGs in and return them.

---

## 11) Patterns & recipes

**A. Clean ingest → transform → visualize**

```python
@app.cell
def _():
    import duckdb, pandas as pd
    con = duckdb.connect()
    return con,

@app.cell
def _(con):
    import marimo as mo
    limit = mo.ui.slider(100, 10_000, 1000, label="Rows")
    limit.display()
    df = mo.sql(f"SELECT * FROM 'data.parquet' LIMIT {limit.value}", connection=con)
    return df,

@app.cell
def _(df):
    clean = (
        df.assign(total=lambda t: t.qty * t.price)
          .query("total > 0")
    )
    return clean,

@app.cell
def _(mo, clean):
    chart = mo.ui.plotly(data=clean, x="qty", y="total")
    chart.display()
    return chart,
```

**B. Memoized model pipeline**

```python
import marimo as mo

@mo.persistent_cache
def load_tokenizer(name: str):
    ...

@mo.cache
def tokenize(tok, texts):
    ...

@app.cell
def _():
    tok = load_tokenizer("my-model")
    return tok,

@app.cell
def _(tok):
    inputs = tokenize(tok, ["hello", "world"])
    return inputs,
```

---

## 12) Quality bar for agent outputs

When generating Marimo notebooks, **always**:

1. Produce a **single, runnable** `.py` file with `marimo.App()` and `app.run()`.
2. Keep cells **small and composable**; one responsibility each.
3. Establish **explicit dataflow** via parameters/returns.
4. Use `mo.ui` for interactivity; **avoid manual callbacks**.
5. Cache heavy work with `mo.cache`/`mo.persistent_cache`.
6. Prefer **pure functions** and immutable transformations.
7. Include brief `mo.md` documentation cells where helpful.

---

## 13) FAQ for agents

**How do I lay out elements side-by-side?** Use `marimo.hstack(a, b)` within a cell and call `.display()` on composite layouts.

**Can I mix SQL and Python?** Yes—SQL cells compile to calls like `mo.sql(...)` so the notebook stays pure Python.

**Do I need special syntax (magics)?** No. Prefer plain Python; Marimo performs static analysis to wire dependencies.

**How do I theme plots?** Query the app theme via `mo.app_meta().theme` and configure your plotting library accordingly.

---
