from __future__ import annotations

import ast
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_sql_constants() -> dict[str, str]:
    module = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    constants: dict[str, str] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.endswith("_SQL"):
            continue
        try:
            constants[target.id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return constants


SQL_PARAMS = {
    "PROBLEM_CASES_BY_CANDIDATE_SQL": ("Access", "password reset"),
    "FCR_CASES_BY_ISSUE_SQL": ("Access",),
}



def test_app_sql_constants_execute_against_rebuilt_schema(db_conn, insert_events) -> None:
    insert_events(
        db_conn,
        [
            {
                "case_id": "APP-SMOKE-1",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-04T08:00:00Z",
                "event": "Ticket created",
                "issue_type": "Access",
                "resolver": "Desk",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": None,
            },
            {
                "case_id": "APP-SMOKE-1",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-04T08:05:00Z",
                "event": "Ticket solved by level 1 support",
                "issue_type": "Access",
                "resolver": "L1",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": None,
            },
            {
                "case_id": "APP-SMOKE-1",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-04T08:10:00Z",
                "event": "Ticket closed",
                "issue_type": "Access",
                "resolver": "L1",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": 4.5,
            },
            {
                "case_id": "APP-SMOKE-1",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-04T08:12:00Z",
                "event": "Customer feedback received",
                "issue_type": "Access",
                "resolver": "L1",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": 4.5,
            },
        ],
    )

    sql_constants = _load_sql_constants()
    assert sql_constants

    with db_conn.cursor() as cur:
        for name, sql_text in sql_constants.items():
            params = SQL_PARAMS.get(name)
            cur.execute(sql_text, params)
            assert cur.description is not None, name
            assert len(cur.description) > 0, name
            cur.fetchall()

