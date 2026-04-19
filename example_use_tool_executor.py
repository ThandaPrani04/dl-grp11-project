import pandas as pd

# Load data
df = pd.read_csv("sales_data.csv")

executor = ToolExecutor(df)

actions = [
    {"tool": "filter", "args": {"column": "year", "op": "==", "value": 2022}},
    {"tool": "groupby", "args": {"column": "city"}},
    {"tool": "aggregate", "args": {"column": "revenue", "agg": "sum"}},
    {"tool": "sort", "args": {"column": "revenue", "ascending": False}},
    {"tool": "topk", "args": {"k": 3}}
]

result = executor.execute(actions)

print(result)