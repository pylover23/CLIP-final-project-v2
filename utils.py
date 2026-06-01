import csv
import json
import os


def format_csv_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def round_json_value(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: round_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_json_value(item) for item in value]
    return value


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows([{key: format_csv_value(value) for key, value in row.items()} for row in rows])


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_json_value(data), f, indent=2, ensure_ascii=False)

