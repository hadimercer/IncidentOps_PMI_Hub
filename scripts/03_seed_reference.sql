-- IncidentOps PMI Hub - seed reference data
-- Idempotent seed inserts for Postgres

CREATE SCHEMA IF NOT EXISTS im;

INSERT INTO im.sla_policy (priority, target_hours)
VALUES
    ('High', 8),
    ('Medium', 16),
    ('Low', 24)
ON CONFLICT (priority) DO UPDATE
SET target_hours = EXCLUDED.target_hours;

DELETE FROM im.event_alias;
DELETE FROM im.event_catalog;

INSERT INTO im.event_catalog (event_name, description, stakeholder_owner)
VALUES
    ('ticket_created', 'Ticket is created in the ITSM platform.', 'Service Desk'),
    ('ticket_closed', 'Ticket is formally closed in the ITSM platform.', 'Service Desk'),
    ('customer_feedback_received', 'Customer feedback or satisfaction is recorded.', 'Service Desk'),
    ('ticket_reopened', 'The incident is reopened after closure or failed resolution.', 'Service Desk'),
    ('ticket_rejected', 'The proposed resolution or closure is rejected.', 'Service Desk'),
    ('ticket_solved_l1', 'The incident is resolved by level 1 support.', 'L1 Support'),
    ('ticket_solved_l2', 'The incident is resolved by level 2 support.', 'L2 Support'),
    ('ticket_solved_l3', 'The incident is resolved by level 3 support.', 'L3 Support'),
    ('escalated_to_l2', 'The incident is escalated from level 1 to level 2 support.', 'L1 Support'),
    ('assigned_to_l2', 'The incident is assigned to level 2 support.', 'Service Desk'),
    ('escalated_to_l3', 'The incident is escalated from level 2 to level 3 support.', 'L2 Support'),
    ('assigned_to_l3', 'The incident is assigned to level 3 support.', 'L2 Support'),
    ('assigned_to_l1', 'The incident is assigned to level 1 support.', 'Service Desk'),
    ('work_in_progress_l1', 'The incident is actively being worked by level 1 support.', 'L1 Support'),
    ('work_in_progress_l2', 'The incident is actively being worked by level 2 support.', 'L2 Support'),
    ('work_in_progress_l3', 'The incident is actively being worked by level 3 support.', 'L3 Support')
ON CONFLICT (event_name) DO UPDATE
SET
    description = EXCLUDED.description,
    stakeholder_owner = EXCLUDED.stakeholder_owner;

INSERT INTO im.event_alias (raw_event_name, event_code)
VALUES
    ('Ticket created', 'ticket_created'),
    ('Ticket closed', 'ticket_closed'),
    ('Customer feedback received', 'customer_feedback_received'),
    ('Ticket reopened by customer', 'ticket_reopened'),
    ('Ticket reopened', 'ticket_reopened'),
    ('Ticket rejected', 'ticket_rejected'),
    ('Ticket rejected by customer', 'ticket_rejected'),
    ('Ticket solved by level 1 support', 'ticket_solved_l1'),
    ('Ticket solved by level 2 support', 'ticket_solved_l2'),
    ('Ticket solved by level 3 support', 'ticket_solved_l3'),
    ('Ticket escalated to level 2 support', 'escalated_to_l2'),
    ('Level 1 escalates to level 2 support', 'escalated_to_l2'),
    ('Ticket assigned to level 2 support', 'assigned_to_l2'),
    ('Level 2 escalates to level 3 support', 'escalated_to_l3'),
    ('Ticket escalated to level 3 support', 'escalated_to_l3'),
    ('Ticket assigned to level 3 support', 'assigned_to_l3'),
    ('Ticket assigned to level 1 support', 'assigned_to_l1'),
    ('WIP - level 1 support', 'work_in_progress_l1'),
    ('WIP - level 2 support', 'work_in_progress_l2'),
    ('WIP - level 3 support', 'work_in_progress_l3'),
    ('Ticket rejected by level 1 support', 'ticket_rejected'),
    ('Level 2 support rejects the ticket', 'ticket_rejected')
ON CONFLICT (raw_event_name) DO UPDATE
SET event_code = EXCLUDED.event_code;

