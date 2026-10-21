
import re
import pandas as pd


def normalize(value):
    """Normalize values so IDs can be compared consistently."""
    value = str(value).strip().casefold()

    # Handle IDs read by pandas as floats, e.g. 15694829.0
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    return value


def find_id_column(df):
    """Find a column that represents a record ID."""
    for column in df.columns:
        name = re.sub(r"[^a-z0-9]", "", str(column).casefold())

        if name in {
            "id", "userid", "employeeid", "studentid",
            "customerid", "recordid", "loanid", "productid"
        }:
            return column

    # Fallback for other column names containing an ID suffix.
    for column in df.columns:
        name = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        if name.endswith("id"):
            return column

    return None



def find_requested_column(df, question):
    q = question.casefold()

    # Check what value the user wants first.
    aliases = {
        "EstimatedSalary": [
            "estimated salary", "estimatedsalary", "salary",
            "income", "pay", "earnings"
        ],
        "Age": ["age"],
        "Gender": ["gender", "sex"],
        "Purchased": ["purchased", "purchase", "bought"],
    }

    # Match the user's requested field to a real CSV column.
    for column in df.columns:
        name = str(column).casefold()
        for alias in aliases.get(column, []):
            if alias in q:
                return column

    # Check explicitly mentioned column names, but skip ID columns
    # so that "user id" isn't mistaken for the requested answer.
    for column in sorted(df.columns, key=lambda c: len(str(c)), reverse=True):
        name = str(column).casefold()
        if name.replace("_", "").replace(" ", "").endswith("id"):
            continue

        normalized_name = name.replace("_", " ").replace("-", " ")
        if normalized_name in q:
            return column

    return None



def answer_csv_question(df, question):
    if df.empty:
        return "The CSV contains no rows."

    question = question.strip()
    if not question:
        return "Please enter a question."

    id_column = find_id_column(df)
    requested_column = find_requested_column(df, question)

    # Look for IDs in the question, including numeric-only IDs.
    tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", question)

    # If the question looks like a record lookup, never guess using RAG.
    mentions_id = bool(
        re.search(r"\b(id|record|user|employee|student|customer)\b",
                  question, re.IGNORECASE)
    )

    if mentions_id and id_column is not None:
        candidates = [
            token for token in tokens
            if any(char.isdigit() for char in token)
        ]

        for token in candidates:
            matches = df[
                df[id_column].apply(normalize) == normalize(token)
            ]

            if not matches.empty:
                if len(matches) > 1:
                    return f"ID {token} appears more than once in the CSV."

                row = matches.iloc[0]

                if requested_column is not None:
                    value = row[requested_column]

                    if pd.isna(value):
                        return f"{requested_column} is empty for ID {token}."

                    return f"{requested_column} for ID {token}: {value}"

                return "Matching record:\n" + row.to_string()

        if candidates:
            return (
                f"No matching record found for ID "
                f"{', '.join(candidates)} in column '{id_column}'."
            )

    # Handle simple questions that don't require a record ID.
    if requested_column is not None:
        series = df[requested_column]
        q = question.casefold()

        if "average" in q or "mean" in q:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                return f"Average {requested_column}: {numeric.mean():.2f}"

        if "total" in q or "sum" in q:
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                return f"Total {requested_column}: {numeric.sum():.2f}"

    return None
