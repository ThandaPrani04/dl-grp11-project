"""
CS F425 Deep Learning — Data Augmentation Script
Expands the 2K training samples to ~10K with balanced coverage across:
  - all tool combinations (not just filter(year) → aggregate)
  - all columns (region, category, profit — not just year/revenue/units_sold)
  - all aggregation types (sum, mean, count)
  - all query complexities (1-filter, 2-filter, 3-filter, group-by, top-k, plot)
  - paraphrase diversity (10+ templates per query pattern)

Usage:
    python augment_queries.py \
        --input_json agent_trajectories_2k.json \
        --output_json training/data/augmented_trajectories.json \
        --target_count 10000

The output JSON matches the exact schema of agent_trajectories_2k.json.
"""

import json
import random
import argparse
import copy
from itertools import product

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE  (matches sales_data.csv column semantics)
# ─────────────────────────────────────────────────────────────────────────────

YEARS = [2021, 2022, 2023, 2024]
CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad",
          "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"]
REGIONS = ["North", "South", "East", "West", "Central"]
CATEGORIES = ["Electronics", "Clothing", "Furniture", "Groceries",
               "Sports", "Books", "Toys", "Beauty"]

METRICS = ["revenue", "profit", "units_sold"]
AGG_TYPES = ["sum", "mean", "count"]
SORT_ORDERS = ["desc", "asc"]
TOP_K_VALUES = [1, 2, 3, 5, 10]

# Dimension columns (used in filter_data / group_by)
DIM_COLS = ["year", "city", "region", "category"]

# ─────────────────────────────────────────────────────────────────────────────
# QUERY TEMPLATES  (natural language paraphrases for each pattern)
# ─────────────────────────────────────────────────────────────────────────────

# Each key is a pattern identifier; value is a list of template strings.
# Placeholders: {metric}, {agg_word}, {col}, {val}, {col2}, {val2},
#               {col3}, {val3}, {k}, {order_word}

TEMPLATES = {

    # ── Pattern 1: single filter → aggregate ─────────────────────────────────
    "filter_agg_sum": [
        "What is the total {metric} for {col} {val}?",
        "Find the sum of {metric} where {col} is {val}.",
        "Total {metric} in {val}",
        "How much {metric} was generated in {val}?",
        "Give me the total {metric} for {col} = {val}.",
        "What is the combined {metric} for {val}?",
        "Sum of {metric} for {col} {val}",
        "Calculate total {metric} for {val}.",
        "What was the total {metric} recorded in {val}?",
        "Compute the overall {metric} for {col} {val}.",
    ],
    "filter_agg_mean": [
        "What is the average {metric} for {col} {val}?",
        "Find the mean {metric} where {col} is {val}.",
        "Average {metric} in {val}",
        "What was the typical {metric} in {val}?",
        "Give me the mean {metric} for {val}.",
        "What is the average {metric} when {col} equals {val}?",
        "Compute the average {metric} for {col} {val}.",
        "Mean {metric} for {col} = {val}",
        "What is the per-record average {metric} in {val}?",
    ],
    "filter_agg_count": [
        "How many records have {col} = {val}?",
        "Count of entries for {col} {val}",
        "How many sales happened in {val}?",
        "What is the number of transactions in {val}?",
        "Count the records where {col} is {val}.",
        "How many rows correspond to {val}?",
        "Total count of records for {col} {val}",
        "Number of orders in {val}",
        "How many entries exist for {col} {val}?",
    ],

    # ── Pattern 2: two filters → aggregate ───────────────────────────────────
    "filter2_agg_sum": [
        "Total {metric} for {col} {val} and {col2} {val2}",
        "What is the total {metric} for {col2} {val2} in {val}?",
        "Sum of {metric} where {col} is {val} and {col2} is {val2}",
        "Combined {metric} for {col2} {val2} in {col} {val}",
        "Find total {metric} for {val2} in {val}.",
        "What was the {metric} for {col2} {val2} during {val}?",
        "{metric} total for {val2} and {val}",
        "How much {metric} did {val2} generate in {val}?",
    ],
    "filter2_agg_mean": [
        "Average {metric} for {col} {val} and {col2} {val2}",
        "What is the mean {metric} for {col2} {val2} in {val}?",
        "Find the average {metric} when {col} is {val} and {col2} is {val2}.",
        "Mean {metric} for {val2} during {val}",
        "What is the typical {metric} for {col2} {val2} in {val}?",
    ],
    "filter2_agg_count": [
        "How many {col2} {val2} records are there in {val}?",
        "Count of {col2} {val2} entries for {col} {val}",
        "Number of orders for {val2} in {val}",
        "How many times did {val2} appear in {val}?",
        "Count records where {col2} is {val2} and {col} is {val}.",
    ],

    # ── Pattern 3: three filters → aggregate ─────────────────────────────────
    "filter3_agg_sum": [
        "Total {metric} for {col} {val}, {col2} {val2}, and {col3} {val3}",
        "What is the {metric} for {val2} in {val3} during {val}?",
        "{metric} total for {col2} {val2}, {col3} {val3} in {val}",
        "Find total {metric} when {col} is {val}, {col2} is {val2}, {col3} is {val3}.",
        "How much {metric} for {val2} in {val3} in the year {val}?",
    ],
    "filter3_agg_count": [
        "How many records have {col} {val}, {col2} {val2}, and {col3} {val3}?",
        "Count of {val2} in {val3} for {col} {val}",
        "Number of {col2} {val2} entries in {col3} {val3} during {val}",
    ],

    # ── Pattern 4: group_by → aggregate ─────────────────────────────────────
    "group_agg_sum": [
        "Total {metric} by {col}",
        "What is the total {metric} for each {col}?",
        "Give me {metric} summed by {col}.",
        "Sum of {metric} grouped by {col}",
        "How much {metric} did each {col} generate?",
        "{metric} breakdown by {col}",
        "What is the combined {metric} per {col}?",
        "Show total {metric} per {col}.",
        "Aggregate {metric} by {col}.",
    ],
    "group_agg_mean": [
        "Average {metric} by {col}",
        "What is the mean {metric} per {col}?",
        "Mean {metric} grouped by {col}",
        "What is the average {metric} for each {col}?",
        "Give me the average {metric} per {col}.",
        "Typical {metric} per {col}",
        "What is the {col}-wise average {metric}?",
    ],
    "group_agg_count": [
        "How many records per {col}?",
        "Count of entries by {col}",
        "Give me the number of orders per {col}.",
        "Record count grouped by {col}",
        "How many sales for each {col}?",
        "Count records by {col}.",
    ],

    # ── Pattern 5: filter → group_by → aggregate (no top-k) ─────────────────
    "filter_group_agg_sum": [
        "Total {metric} by {col2} for {col} {val}",
        "What is the total {metric} per {col2} in {val}?",
        "{metric} breakdown by {col2} for {val}",
        "Sum of {metric} grouped by {col2} when {col} is {val}",
        "How much {metric} per {col2} in {val}?",
    ],
    "filter_group_agg_mean": [
        "Average {metric} by {col2} for {col} {val}",
        "Mean {metric} per {col2} in {val}",
        "What is the average {metric} per {col2} for {val}?",
    ],
    "filter_group_agg_count": [
        "Count of records per {col2} in {val}",
        "How many entries per {col2} for {col} {val}?",
        "Number of records by {col2} during {val}",
    ],

    # ── Pattern 6: top-k (no filter) ─────────────────────────────────────────
    "topk_sum_desc": [
        "Top {k} {col} by total {metric}",
        "Which {k} {col}s have the highest {metric}?",
        "List the top {k} {col}s based on {metric}.",
        "What are the {k} best-performing {col}s by {metric}?",
        "Top {k} {col}s ranked by total {metric}",
        "Which {k} {col}s generated the most {metric}?",
        "Give me the top {k} {col}s by {metric}.",
    ],
    "topk_sum_asc": [
        "Bottom {k} {col} by total {metric}",
        "Which {k} {col}s have the lowest {metric}?",
        "List the {k} least-performing {col}s by {metric}.",
        "What are the {k} {col}s with the least {metric}?",
        "Which {k} {col}s generated the least {metric}?",
    ],
    "topk_mean_desc": [
        "Top {k} {col}s by average {metric}",
        "Which {k} {col}s have the highest average {metric}?",
        "Best {k} {col}s by mean {metric}",
        "{k} {col}s with highest average {metric}",
    ],
    "topk_count_desc": [
        "Top {k} {col}s by number of orders",
        "Which {k} {col}s have the most sales records?",
        "Most frequent {k} {col}s by count",
        "Top {k} {col}s by order count",
    ],

    # ── Pattern 7: filter → top-k ─────────────────────────────────────────────
    "filter_topk_sum_desc": [
        "Top {k} {col2}s by {metric} in {val}",
        "List top {k} {col2}s based on {metric} for {col} {val}",
        "Which {k} {col2}s had the highest {metric} in {val}?",
        "Best {k} {col2}s by total {metric} for {val}",
        "Top {k} {col2}s ranked by {metric} in {val}",
        "Which {k} {col2}s generated most {metric} in {val}?",
        "Give me top {k} {col2}s for {val} by {metric}.",
    ],
    "filter_topk_sum_asc": [
        "Bottom {k} {col2}s by {metric} in {val}",
        "Lowest {k} {col2}s by {metric} for {val}",
        "Which {k} {col2}s had the least {metric} in {val}?",
    ],
    "filter_topk_mean_desc": [
        "Top {k} {col2}s by average {metric} in {val}",
        "Which {k} {col2}s had the highest average {metric} in {val}?",
        "Best {k} {col2}s by mean {metric} during {val}",
    ],
    "filter_topk_count_desc": [
        "Top {k} {col2}s by number of orders in {val}",
        "Which {k} {col2}s had the most records in {val}?",
        "Most active {k} {col2}s for {val}",
    ],

    # ── Pattern 8: two-filter → top-k ────────────────────────────────────────
    "filter2_topk_sum_desc": [
        "Top {k} {col3}s by {metric} for {col2} {val2} in {val}",
        "Which {k} {col3}s had the most {metric} in {col2} {val2} during {val}?",
        "List top {k} {col3}s by {metric} for {val2} in {val}",
    ],

    # ── Pattern 9: plot queries ───────────────────────────────────────────────
    "plot_group_sum": [
        "Plot total {metric} by {col}",
        "Show a bar chart of {metric} grouped by {col}.",
        "Visualise total {metric} per {col}.",
        "Draw a chart of {metric} by {col}.",
        "Graph the total {metric} for each {col}.",
    ],
    "plot_group_mean": [
        "Plot average {metric} by {col}",
        "Show a chart of mean {metric} per {col}.",
        "Visualise average {metric} grouped by {col}.",
        "Draw a bar chart of average {metric} by {col}.",
    ],
    "plot_filter_group_sum": [
        "Plot total {metric} by {col2} for {val}",
        "Show a chart of {metric} per {col2} in {val}.",
        "Visualise {metric} by {col2} for {col} {val}.",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# ACTION BUILDERS  (produce the exact tool invocation strings)
# ─────────────────────────────────────────────────────────────────────────────

def agg_tool(agg_type: str, metric: str) -> str:
    if agg_type == "sum":
        return f"aggregate_sum(column='{metric}')"
    elif agg_type == "mean":
        return f"aggregate_mean(column='{metric}')"
    else:
        return f"aggregate_count(column='{metric}')"


def make_actions(pattern: str, **kw) -> list[str]:
    """Return the ordered list of tool invocations for a given pattern."""
    col   = kw.get("col",   "year")
    val   = kw.get("val",   2021)
    col2  = kw.get("col2",  "city")
    val2  = kw.get("val2",  "Mumbai")
    col3  = kw.get("col3",  "region")
    val3  = kw.get("val3",  "North")
    metric = kw.get("metric", "revenue")
    agg   = kw.get("agg",   "sum")
    k     = kw.get("k",     3)
    order = kw.get("order", "desc")

    def filt(c, v):
        v_str = f"'{v}'" if isinstance(v, str) else str(v)
        return f"filter_data(column='{c}', value={v_str})"

    def grp(c):
        return f"group_by(column='{c}')"

    def srt(m, o):
        return f"sort_by(column='{m}', order='{o}')"

    def topk(k_):
        return f"top_k(k={k_})"

    # ── single filter ────────────────────────────
    if pattern in ("filter_agg_sum", "filter_agg_mean", "filter_agg_count"):
        return [filt(col, val), agg_tool(agg, metric)]

    # ── two filters ──────────────────────────────
    if pattern in ("filter2_agg_sum", "filter2_agg_mean", "filter2_agg_count"):
        return [filt(col, val), filt(col2, val2), agg_tool(agg, metric)]

    # ── three filters ────────────────────────────
    if pattern in ("filter3_agg_sum", "filter3_agg_count"):
        return [filt(col, val), filt(col2, val2), filt(col3, val3), agg_tool(agg, metric)]

    # ── group → aggregate ─────────────────────────
    if pattern in ("group_agg_sum", "group_agg_mean", "group_agg_count"):
        return [grp(col), agg_tool(agg, metric)]

    # ── filter → group → aggregate ────────────────
    if pattern in ("filter_group_agg_sum", "filter_group_agg_mean", "filter_group_agg_count"):
        return [filt(col, val), grp(col2), agg_tool(agg, metric)]

    # ── top-k (no filter) ────────────────────────
    if pattern in ("topk_sum_desc", "topk_sum_asc", "topk_mean_desc", "topk_count_desc"):
        agg_type = "mean" if "mean" in pattern else ("count" if "count" in pattern else "sum")
        ord_ = "asc" if "asc" in pattern else "desc"
        return [grp(col), agg_tool(agg_type, metric), srt(metric, ord_), topk(k)]

    # ── filter → top-k ───────────────────────────
    if pattern in ("filter_topk_sum_desc", "filter_topk_sum_asc",
                   "filter_topk_mean_desc", "filter_topk_count_desc"):
        agg_type = "mean" if "mean" in pattern else ("count" if "count" in pattern else "sum")
        ord_ = "asc" if "asc" in pattern else "desc"
        return [filt(col, val), grp(col2), agg_tool(agg_type, metric), srt(metric, ord_), topk(k)]

    # ── two-filter → top-k ───────────────────────
    if pattern == "filter2_topk_sum_desc":
        return [filt(col, val), filt(col2, val2), grp(col3), agg_tool("sum", metric), srt(metric, "desc"), topk(k)]

    # ── plot ─────────────────────────────────────
    if pattern == "plot_group_sum":
        return [grp(col), agg_tool("sum", metric), "plot()"]
    if pattern == "plot_group_mean":
        return [grp(col), agg_tool("mean", metric), "plot()"]
    if pattern == "plot_filter_group_sum":
        return [filt(col, val), grp(col2), agg_tool("sum", metric), "plot()"]

    raise ValueError(f"Unknown pattern: {pattern}")


# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def render_template(tmpl: str, **kw) -> str:
    """Fill in a template string, formatting values nicely."""
    kw = dict(kw)
    # Convert aggregation types to natural words
    agg = kw.get("agg", "sum")
    kw["agg_word"] = {"sum": "total", "mean": "average", "count": "count"}[agg]
    kw["order_word"] = "highest" if kw.get("order", "desc") == "desc" else "lowest"
    # Format year values without quotes, others as-is
    for key in ("val", "val2", "val3"):
        v = kw.get(key, "")
        kw[key] = str(v)
    return tmpl.format(**kw)


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN → GENERATOR MAPPING
# ─────────────────────────────────────────────────────────────────────────────

# Each entry: (pattern_key, param_sampler_fn)
# param_sampler_fn returns a dict of keyword args for make_actions + render_template

def _sample_year_metric_agg(agg: str):
    return dict(col="year", val=random.choice(YEARS),
                metric=random.choice(METRICS), agg=agg)

def _sample_dim_year(col: str, agg: str):
    vals = {"city": CITIES, "region": REGIONS, "category": CATEGORIES}[col]
    return dict(col="year", val=random.choice(YEARS),
                col2=col, val2=random.choice(vals),
                metric=random.choice(METRICS), agg=agg)

def _sample_group(col: str, agg: str):
    return dict(col=col, metric=random.choice(METRICS), agg=agg)

def _sample_filter_group(filter_col: str, group_col: str, agg: str):
    filter_vals = {"year": YEARS, "city": CITIES, "region": REGIONS, "category": CATEGORIES}[filter_col]
    return dict(col=filter_col, val=random.choice(filter_vals),
                col2=group_col,
                metric=random.choice(METRICS), agg=agg)

def _sample_topk(group_col: str, agg_key: str, order: str):
    agg = {"sum": "sum", "mean": "mean", "count": "count"}[agg_key]
    return dict(col=group_col, metric=random.choice(METRICS),
                agg=agg, k=random.choice(TOP_K_VALUES), order=order)

def _sample_filter_topk(filter_col: str, group_col: str, agg_key: str, order: str):
    filter_vals = {"year": YEARS, "city": CITIES, "region": REGIONS, "category": CATEGORIES}[filter_col]
    agg = {"sum": "sum", "mean": "mean", "count": "count"}[agg_key]
    return dict(col=filter_col, val=random.choice(filter_vals),
                col2=group_col,
                metric=random.choice(METRICS), agg=agg,
                k=random.choice(TOP_K_VALUES), order=order)

def _sample_filter2_topk():
    # e.g. filter(year) + filter(category) → group(city) → top_k
    col2 = random.choice(["category", "region"])
    vals2 = {"category": CATEGORIES, "region": REGIONS}[col2]
    return dict(col="year", val=random.choice(YEARS),
                col2=col2, val2=random.choice(vals2),
                col3="city",
                metric=random.choice(METRICS),
                k=random.choice(TOP_K_VALUES))

def _sample_triple_filter(agg: str):
    return dict(col="year", val=random.choice(YEARS),
                col2="category", val2=random.choice(CATEGORIES),
                col3="city", val3=random.choice(CITIES),
                metric=random.choice(METRICS), agg=agg)

def _sample_plot_group(col: str, agg: str):
    return dict(col=col, metric=random.choice(METRICS), agg=agg)

def _sample_plot_filter_group():
    return dict(col="year", val=random.choice(YEARS),
                col2=random.choice(["city", "region", "category"]),
                metric=random.choice(METRICS))


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN REGISTRY  (pattern_key → (template_key, sampler, target_count))
# ─────────────────────────────────────────────────────────────────────────────

def build_pattern_registry(target_total: int):
    """
    Returns list of (pattern_key, template_key, sampler_fn, n_samples).
    Allocation ensures balanced coverage.
    """

    # Rough allocation fractions that balance coverage (must sum to ~1.0)
    alloc = {
        # --- single filter variants (keep to ~15% total, was 60% before) ---
        ("filter_agg_sum",   "year"):     0.03,
        ("filter_agg_sum",   "city"):     0.02,
        ("filter_agg_sum",   "category"): 0.02,
        ("filter_agg_sum",   "region"):   0.02,
        ("filter_agg_mean",  "year"):     0.02,
        ("filter_agg_mean",  "category"): 0.01,
        ("filter_agg_count", "year"):     0.02,
        ("filter_agg_count", "city"):     0.01,
        # --- two-filter variants ---
        ("filter2_agg_sum",  "year+city"):     0.03,
        ("filter2_agg_sum",  "year+category"): 0.03,
        ("filter2_agg_sum",  "year+region"):   0.02,
        ("filter2_agg_mean", "year+category"): 0.02,
        ("filter2_agg_count","year+city"):     0.02,
        ("filter2_agg_count","year+category"): 0.02,
        # --- three-filter variants ---
        ("filter3_agg_sum",   "year+cat+city"):   0.03,
        ("filter3_agg_count", "year+cat+city"):   0.02,
        # --- group-by variants ---
        ("group_agg_sum",   "city"):     0.03,
        ("group_agg_sum",   "region"):   0.03,
        ("group_agg_sum",   "category"): 0.03,
        ("group_agg_sum",   "year"):     0.02,
        ("group_agg_mean",  "city"):     0.02,
        ("group_agg_mean",  "region"):   0.02,
        ("group_agg_mean",  "category"): 0.02,
        ("group_agg_count", "city"):     0.02,
        ("group_agg_count", "region"):   0.02,
        ("group_agg_count", "category"): 0.02,
        # --- filter → group → aggregate ---
        ("filter_group_agg_sum",   "year/city"):     0.03,
        ("filter_group_agg_sum",   "year/region"):   0.03,
        ("filter_group_agg_sum",   "year/category"): 0.03,
        ("filter_group_agg_mean",  "year/city"):     0.02,
        ("filter_group_agg_mean",  "year/region"):   0.02,
        ("filter_group_agg_count", "year/city"):     0.02,
        # --- top-k (no filter) ---
        ("topk_sum_desc",   "city"):     0.02,
        ("topk_sum_desc",   "region"):   0.02,
        ("topk_sum_desc",   "category"): 0.02,
        ("topk_sum_asc",    "city"):     0.01,
        ("topk_sum_asc",    "category"): 0.01,
        ("topk_mean_desc",  "region"):   0.01,
        ("topk_count_desc", "city"):     0.01,
        ("topk_count_desc", "category"): 0.01,
        # --- filter → top-k ---
        ("filter_topk_sum_desc",   "year/city"):     0.03,
        ("filter_topk_sum_desc",   "year/region"):   0.03,
        ("filter_topk_sum_desc",   "year/category"): 0.03,
        ("filter_topk_sum_desc",   "cat/city"):      0.02,
        ("filter_topk_sum_asc",    "year/city"):     0.01,
        ("filter_topk_mean_desc",  "year/region"):   0.02,
        ("filter_topk_count_desc", "year/city"):     0.02,
        ("filter_topk_count_desc", "year/category"): 0.02,
        # --- two-filter → top-k ---
        ("filter2_topk_sum_desc", "year+cat/city"):  0.03,
        # --- plot ---
        ("plot_group_sum",       "city"):     0.01,
        ("plot_group_sum",       "region"):   0.01,
        ("plot_group_sum",       "category"): 0.01,
        ("plot_group_mean",      "region"):   0.01,
        ("plot_filter_group_sum","year/*"):   0.02,
    }

    # Samplers for each (pattern_key, variant_key)
    def sampler_for(pk, vk):
        # single filter
        if pk == "filter_agg_sum" and vk == "year":
            return lambda: {**_sample_year_metric_agg("sum")}
        if pk == "filter_agg_sum" and vk in ("city","category","region"):
            vals = {"city":CITIES,"category":CATEGORIES,"region":REGIONS}[vk]
            return lambda vk=vk,vals=vals: dict(col=vk, val=random.choice(vals),
                                                metric=random.choice(METRICS), agg="sum")
        if pk == "filter_agg_mean" and vk == "year":
            return lambda: _sample_year_metric_agg("mean")
        if pk == "filter_agg_mean" and vk == "category":
            return lambda: dict(col="category", val=random.choice(CATEGORIES),
                                metric=random.choice(METRICS), agg="mean")
        if pk == "filter_agg_count" and vk == "year":
            return lambda: _sample_year_metric_agg("count")
        if pk == "filter_agg_count" and vk == "city":
            return lambda: dict(col="city", val=random.choice(CITIES),
                                metric=random.choice(METRICS), agg="count")
        # two filter
        if pk == "filter2_agg_sum" and vk == "year+city":
            return lambda: _sample_dim_year("city", "sum")
        if pk == "filter2_agg_sum" and vk == "year+category":
            return lambda: _sample_dim_year("category", "sum")
        if pk == "filter2_agg_sum" and vk == "year+region":
            return lambda: _sample_dim_year("region", "sum")
        if pk == "filter2_agg_mean" and vk == "year+category":
            return lambda: _sample_dim_year("category", "mean")
        if pk == "filter2_agg_count" and vk == "year+city":
            return lambda: _sample_dim_year("city", "count")
        if pk == "filter2_agg_count" and vk == "year+category":
            return lambda: _sample_dim_year("category", "count")
        # three filter
        if pk == "filter3_agg_sum":
            return lambda: _sample_triple_filter("sum")
        if pk == "filter3_agg_count":
            return lambda: _sample_triple_filter("count")
        # group
        if pk in ("group_agg_sum","group_agg_mean","group_agg_count"):
            agg = {"group_agg_sum":"sum","group_agg_mean":"mean","group_agg_count":"count"}[pk]
            return lambda vk=vk,agg=agg: _sample_group(vk, agg)
        # filter → group
        if pk in ("filter_group_agg_sum","filter_group_agg_mean","filter_group_agg_count"):
            agg = {"filter_group_agg_sum":"sum","filter_group_agg_mean":"mean","filter_group_agg_count":"count"}[pk]
            fc, gc = vk.split("/")
            return lambda fc=fc,gc=gc,agg=agg: _sample_filter_group(fc, gc, agg)
        # top-k no filter
        if pk in ("topk_sum_desc","topk_sum_asc","topk_mean_desc","topk_count_desc"):
            agg_key = "sum" if "sum" in pk else ("mean" if "mean" in pk else "count")
            order = "asc" if "asc" in pk else "desc"
            return lambda vk=vk,agg_key=agg_key,order=order: _sample_topk(vk, agg_key, order)
        # filter → top-k
        if pk in ("filter_topk_sum_desc","filter_topk_sum_asc",
                  "filter_topk_mean_desc","filter_topk_count_desc"):
            agg_key = "sum" if "sum" in pk else ("mean" if "mean" in pk else "count")
            order = "asc" if "asc" in pk else "desc"
            fc, gc = vk.split("/")
            return lambda fc=fc,gc=gc,agg_key=agg_key,order=order: _sample_filter_topk(fc, gc, agg_key, order)
        # two-filter → top-k
        if pk == "filter2_topk_sum_desc":
            return _sample_filter2_topk
        # plot
        if pk == "plot_group_sum":
            return lambda vk=vk: _sample_plot_group(vk, "sum")
        if pk == "plot_group_mean":
            return lambda vk=vk: _sample_plot_group(vk, "mean")
        if pk == "plot_filter_group_sum":
            return _sample_plot_filter_group
        raise ValueError(f"No sampler for ({pk}, {vk})")

    registry = []
    for (pk, vk), frac in alloc.items():
        n = max(1, int(frac * target_total))
        # map pattern key to template key
        tmpl_key = pk
        registry.append((pk, tmpl_key, sampler_for(pk, vk), n))
    return registry


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(pattern_key: str, template_key: str, sampler_fn) -> dict:
    """Generate one (query, actions) sample."""
    params = sampler_fn()
    actions = make_actions(pattern_key, **params)
    templates = TEMPLATES.get(template_key, TEMPLATES.get(pattern_key, ["{col} {val} {metric}"]))
    tmpl = random.choice(templates)
    try:
        query = render_template(tmpl, **params)
    except KeyError:
        query = f"{pattern_key}: {params}"
    return {"query": query, "actions": actions}


def deduplicate(samples: list[dict]) -> list[dict]:
    """Remove exact query duplicates (case-insensitive)."""
    seen = set()
    out = []
    for s in samples:
        key = s["query"].strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json",  required=True,
                        help="Path to original agent_trajectories_2k.json")
    parser.add_argument("--output_json", required=True,
                        help="Output path for augmented dataset")
    parser.add_argument("--target_count", type=int, default=10000,
                        help="Total samples in output (including originals)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load originals
    with open(args.input_json) as f:
        originals = json.load(f)
    print(f"Loaded {len(originals)} original samples.")

    n_to_generate = max(0, args.target_count - len(originals))
    print(f"Generating {n_to_generate} new samples (target total: {args.target_count})...")

    registry = build_pattern_registry(n_to_generate)

    new_samples = []
    for pk, tmpl_key, sampler_fn, n in registry:
        for _ in range(n):
            try:
                s = generate_sample(pk, tmpl_key, sampler_fn)
                new_samples.append(s)
            except Exception as e:
                pass  # skip bad samples silently

    # Shuffle new samples
    random.shuffle(new_samples)

    # Combine originals + new, deduplicate
    combined = originals + new_samples
    combined = deduplicate(combined)
    random.shuffle(combined)

    # Print distribution stats
    from collections import Counter
    tool_counts = Counter()
    for s in combined:
        for a in s["actions"]:
            tool = a.split("(")[0]
            tool_counts[tool] += 1
    
    print(f"\nFinal dataset: {len(combined)} samples")
    print("Tool distribution:")
    for tool, cnt in sorted(tool_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / sum(tool_counts.values())
        print(f"  {tool:<25} {cnt:5d}  ({pct:.1f}% of tool calls)")

    # Save
    with open(args.output_json, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
