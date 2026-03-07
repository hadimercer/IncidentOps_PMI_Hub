# Product Brief

## Overview
IncidentOps PMI Hub is a fictional enterprise service-desk analytics product built to demonstrate requirements-driven delivery, process mining concepts, and decision-support design. It turns an exported incident event log into a structured analytics layer and reviewer-friendly operational dashboard.

## Problem Statement
The service desk captures detailed incident events, but leaders and operators often work from manual exports and surface-level averages. That leaves them without a reliable view of:
- how incidents actually flow end to end
- where time is lost
- which patterns drive escalations, reopens, and customer satisfaction outcomes

The result is operational decision-making without process visibility.

## Objectives
- Make incident flow visible through variants and transition analysis
- Reduce cycle time by surfacing dwell-time and routing bottlenecks
- Measure escalation and handoff churn, including ping-pong behavior
- Improve closure governance through feedback and reopen signals
- Produce repeatable exports and KPI packs for operational review
- Present a portfolio-grade full-stack deliverable that a reviewer can run and inspect end to end

## Primary Stakeholders
- Service Owner / ITSM Director: needs outcome KPIs and improvement visibility
- Process Owner / Operations Manager: needs bottleneck, handoff, and routing signals
- L1/L2/L3 Team Leads: need exception queues and operational drilldowns
- Quality / CX Lead: needs closure compliance and customer outcome monitoring
- Problem / Knowledge / Channel stakeholders: need evidence-based follow-up lists

## In-Scope Capabilities
- CSV event-log ingestion with validation and repeatable loading
- PostgreSQL storage plus analytical SQL views
- Process mining metrics: variants, transitions, dwell time, cycle time, escalation/handoff cost
- Quality and CX metrics: feedback capture, reject/reopen, closure compliance
- FCR and enablement opportunity views
- Problem-candidate generation using repeat themes and impact signals
- Streamlit dashboard with multi-page operational review surfaces
- CSV exports for follow-up actions

## Important Constraints
- CSV input only; no live ITSM integration
- Analytics-only product; it does not update tickets
- SLA compliance is policy-based, not sourced from an authoritative event flag
- Designed for local / portfolio-scale execution rather than enterprise multi-user deployment
- Process mining outputs are descriptive, not causal proof

## Important Assumptions
- Event timestamps are reliable enough for per-case sequencing
- Customer satisfaction is interpreted as a stable case-level outcome
- Reopen events are meaningful quality signals
- Escalation events represent actual tier transfers rather than administrative relabeling
- Reviewer value comes from reproducibility, clarity, and business framing rather than production hardening
