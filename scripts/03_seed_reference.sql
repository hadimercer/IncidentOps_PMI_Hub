-- IncidentOps PMI Hub - seed reference data
-- Idempotent seed inserts for Postgres

CREATE SCHEMA IF NOT EXISTS im;

INSERT INTO im.sla_policy (priority, target_hours)
VALUES
    ('High', 8),
    ('Medium', 16),
    ('Low', 24)
ON CONFLICT (priority) DO NOTHING;

INSERT INTO im.event_catalog (event_name, description, stakeholder_owner)
VALUES
    ('incident_reported', 'Incident is reported by requester or monitoring.', 'Service Desk'),
    ('ticket_logged', 'Ticket is created in the ITSM platform.', 'Service Desk'),
    ('categorize_incident', 'Incident category and impact are identified.', 'Service Desk'),
    ('prioritize_incident', 'Priority is set using urgency and impact.', 'Incident Manager'),
    ('initial_diagnosis_started', 'Initial triage and diagnosis begin.', 'L1 Support'),
    ('assignment_to_l1', 'Case is assigned to first-line resolver group.', 'Service Desk'),
    ('investigation_in_progress', 'Resolver team investigates probable cause.', 'Resolver Team'),
    ('assignment_to_l2', 'Case is escalated to second-line resolver group.', 'L2 Support'),
    ('assignment_to_l3', 'Case is escalated to specialist or engineering.', 'L3 Support'),
    ('workaround_provided', 'Temporary workaround is communicated.', 'Resolver Team'),
    ('vendor_engaged', 'External vendor is engaged for support.', 'Vendor Manager'),
    ('root_cause_identified', 'Root cause is identified and documented.', 'Resolver Team'),
    ('resolution_implemented', 'Permanent fix is implemented.', 'Resolver Team'),
    ('resolution_validated', 'Resolution is validated against acceptance criteria.', 'Service Owner'),
    ('customer_notified', 'Requester is informed about resolution status.', 'Service Desk'),
    ('reopen_incident', 'Incident is reopened after failed resolution.', 'Service Desk'),
    ('closure_approval', 'Closure approval is obtained from requester/owner.', 'Service Owner'),
    ('incident_closed', 'Ticket is formally closed in ITSM.', 'Service Desk')
ON CONFLICT (event_name) DO NOTHING;
