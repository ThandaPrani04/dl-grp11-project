import pandas as pd

class ToolExecutor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def execute(self, actions):
        """
        actions: list of dicts
        each dict = {"tool": ..., "args": {...}}
        """
        df = self.df

        for step_id, action in enumerate(actions):
            tool = action["tool"]
            args = action.get("args", {})

            try:
                if tool == "filter":
                    df = self._filter(df, **args)

                elif tool == "groupby":
                    df = self._groupby(df, **args)

                elif tool == "aggregate":
                    df = self._aggregate(df, **args)

                elif tool == "sort":
                    df = self._sort(df, **args)

                elif tool == "topk":
                    df = self._topk(df, **args)

                else:
                    raise ValueError(f"Unknown tool: {tool}")

            except Exception as e:
                raise RuntimeError(
                    f"Error at step {step_id} ({tool}): {str(e)}"
                )

        return df

    # -----------------------------
    # Tool implementations
    # -----------------------------

    def _filter(self, df, column, op, value):
        if op == "==":
            return df[df[column] == value]
        elif op == ">":
            return df[df[column] > value]
        elif op == "<":
            return df[df[column] < value]
        else:
            raise ValueError(f"Unsupported op: {op}")

    def _groupby(self, df, column):
        return df.groupby(column)

    def _aggregate(self, df, column, agg):
        # 1. Calculate the result based on the aggregation type
        if agg == "sum":
            result = df[column].sum()
        elif agg == "mean":
            result = df[column].mean()
        elif agg == "count":
            result = df[column].count()
        else:
            raise ValueError(f"Unsupported aggregation: {agg}")

        # 2. If the result has an index (like a grouped series), make it a dataframe
        if hasattr(result, "reset_index"):
            return result.reset_index()
            
        # 3. If the result is a single number (scalar), wrap it in a 1x1 dataframe
        else:
            return pd.DataFrame({column: [result]})

    def _sort(self, df, column, ascending=False):
        return df.sort_values(by=column, ascending=ascending)

    def _topk(self, df, k):
        return df.head(k)