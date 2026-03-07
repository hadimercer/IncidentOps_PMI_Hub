from __future__ import annotations

from pipeline import apply_sql


def test_case_rollup_uses_explicit_attribute_rules(db_conn, insert_events) -> None:
    insert_events(
        db_conn,
        [
            {
                "case_id": "CASE-ROLLUP",
                "variant": "Legacy",
                "priority": "Medium",
                "reporter": "Ava",
                "ts": "2024-01-01T08:00:00Z",
                "event": "Ticket created",
                "issue_type": "Access",
                "resolver": "Desk",
                "report_channel": "Email",
                "short_description": "Password reset",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-ROLLUP",
                "variant": "Modern",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-01T08:05:00Z",
                "event": "Ticket assigned to level 2 support",
                "issue_type": "Access",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-ROLLUP",
                "variant": "Modern",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-01T08:10:00Z",
                "event": "Ticket solved by level 2 support",
                "issue_type": "Hardware",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-ROLLUP",
                "variant": "Modern",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-01T08:15:00Z",
                "event": "Ticket closed",
                "issue_type": "Hardware",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Password reset",
                "customer_satisfaction": 4.5,
            },
        ],
    )

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT variant, priority, issue_type, report_channel, escalation_count, customer_satisfaction
            FROM im.v_case
            WHERE case_id = 'CASE-ROLLUP'
            """
        )
        variant, priority, issue_type, report_channel, escalation_count, customer_satisfaction = cur.fetchone()

    assert variant == "Modern"
    assert priority == "Medium"
    assert issue_type == "Access"
    assert report_channel == "Email"
    assert escalation_count == 1
    assert float(customer_satisfaction) == 4.5


def test_canonical_mapping_drives_resolution_and_fcr(db_conn, insert_events) -> None:
    insert_events(
        db_conn,
        [
            {
                "case_id": "CASE-FCR-TRUE",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-02T09:00:00Z",
                "event": "Ticket created",
                "issue_type": "Access",
                "resolver": "Desk",
                "report_channel": "Portal",
                "short_description": "VPN issue",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-TRUE",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-02T09:10:00Z",
                "event": "Ticket solved by level 1 support",
                "issue_type": "Access",
                "resolver": "L1",
                "report_channel": "Portal",
                "short_description": "VPN issue",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-TRUE",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ava",
                "ts": "2024-01-02T09:15:00Z",
                "event": "Ticket closed",
                "issue_type": "Access",
                "resolver": "L1",
                "report_channel": "Portal",
                "short_description": "VPN issue",
                "customer_satisfaction": 4.0,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:00:00Z",
                "event": "Ticket created",
                "issue_type": "Email",
                "resolver": "Desk",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:05:00Z",
                "event": "Ticket solved by level 1 support",
                "issue_type": "Email",
                "resolver": "L1",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:10:00Z",
                "event": "Ticket closed",
                "issue_type": "Email",
                "resolver": "L1",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:20:00Z",
                "event": "Ticket reopened",
                "issue_type": "Email",
                "resolver": "Desk",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:25:00Z",
                "event": "Ticket solved by level 1 support",
                "issue_type": "Email",
                "resolver": "L1",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-REOPEN",
                "variant": "Standard",
                "priority": "High",
                "reporter": "Ben",
                "ts": "2024-01-02T10:30:00Z",
                "event": "Ticket closed",
                "issue_type": "Email",
                "resolver": "L1",
                "report_channel": "Phone",
                "short_description": "Mailbox error",
                "customer_satisfaction": 3.5,
            },
            {
                "case_id": "CASE-FCR-L2",
                "variant": "Escalated",
                "priority": "High",
                "reporter": "Cara",
                "ts": "2024-01-02T11:00:00Z",
                "event": "Ticket created",
                "issue_type": "Hardware",
                "resolver": "Desk",
                "report_channel": "Portal",
                "short_description": "Laptop issue",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-L2",
                "variant": "Escalated",
                "priority": "High",
                "reporter": "Cara",
                "ts": "2024-01-02T11:03:00Z",
                "event": "Level 1 escalates to level 2 support",
                "issue_type": "Hardware",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Laptop issue",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-L2",
                "variant": "Escalated",
                "priority": "High",
                "reporter": "Cara",
                "ts": "2024-01-02T11:10:00Z",
                "event": "Ticket solved by level 2 support",
                "issue_type": "Hardware",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Laptop issue",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-FCR-L2",
                "variant": "Escalated",
                "priority": "High",
                "reporter": "Cara",
                "ts": "2024-01-02T11:15:00Z",
                "event": "Ticket closed",
                "issue_type": "Hardware",
                "resolver": "L2",
                "report_channel": "Portal",
                "short_description": "Laptop issue",
                "customer_satisfaction": 2.5,
            },
        ],
    )

    with db_conn.cursor() as cur:
        cur.execute(
            """
            SELECT case_id, resolution_level, fcr
            FROM im.v_fcr_cases
            WHERE case_id IN ('CASE-FCR-TRUE', 'CASE-FCR-REOPEN', 'CASE-FCR-L2')
            ORDER BY case_id
            """
        )
        rows = cur.fetchall()

    assert rows == [
        ("CASE-FCR-L2", "L2", False),
        ("CASE-FCR-REOPEN", "L1", False),
        ("CASE-FCR-TRUE", "L1", True),
    ]


def test_verify_objects_fails_when_required_view_is_missing(db_conn) -> None:
    with db_conn.cursor() as cur:
        cur.execute("DROP VIEW im.v_fcr_summary")
    db_conn.commit()

    assert apply_sql._verify_objects(db_conn) is False



def test_verify_objects_fails_when_unmapped_events_exist(db_conn, insert_events) -> None:
    insert_events(
        db_conn,
        [
            {
                "case_id": "CASE-UNMAPPED",
                "variant": "Unknown",
                "priority": "Low",
                "reporter": "Drew",
                "ts": "2024-01-03T09:00:00Z",
                "event": "Completely new raw event",
                "issue_type": "Other",
                "resolver": "Desk",
                "report_channel": "Portal",
                "short_description": "Unexpected workflow",
                "customer_satisfaction": None,
            },
            {
                "case_id": "CASE-UNMAPPED",
                "variant": "Unknown",
                "priority": "Low",
                "reporter": "Drew",
                "ts": "2024-01-03T09:05:00Z",
                "event": "Ticket closed",
                "issue_type": "Other",
                "resolver": "Desk",
                "report_channel": "Portal",
                "short_description": "Unexpected workflow",
                "customer_satisfaction": None,
            },
        ],
    )

    assert apply_sql._verify_objects(db_conn) is False

