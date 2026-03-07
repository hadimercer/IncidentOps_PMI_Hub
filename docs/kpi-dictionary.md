# KPI Dictionary

## Cycle Time
Elapsed time from the earliest case creation signal to the final close signal for a case.

Use it to:
- compare flow variants
- quantify bottlenecks
- review queue and routing impact

## SLA Met
Whether case cycle time is less than or equal to the configured priority-based SLA target.

Use it to:
- track service-level compliance
- compare issue types, channels, and variants against target performance

## Reopen Rate
Share of cases that include a reopen signal after an earlier closure outcome.

Use it to:
- detect premature closure
- assess quality and resolution durability

## Reject Rate
Share of cases that include a reject signal in the lifecycle.

Use it to:
- identify low-confidence resolutions
- monitor closure governance and acceptance quality

## Escalation Count
Count of escalation or higher-tier assignment events within a case.

Use it to:
- measure specialist-capacity demand
- compare costly cohorts

## Resolver Changes
Count of ownership/resolver changes across the case lifecycle.

Use it to:
- detect handoff churn
- review routing quality and operational friction

## Ping-Pong Cases
Cases with repeated back-and-forth movement, typically reflected in high resolver churn and escalation behavior.

Use it to:
- identify avoidable operational churn
- prioritize routing or process interventions

## Closure Governance
Quality-oriented closure signals such as:
- closed without feedback
- reopened shortly after closure
- reject/reopen exceptions

Use it to:
- monitor quality controls
- create exception queues for follow-up

## Customer Satisfaction
Case-level satisfaction outcome derived from available customer feedback values.

Use it to:
- evaluate outcome quality
- compare speed versus customer experience

## FCR (First Contact Resolution Proxy)
A proxy for resolution at L1 without escalation to L2/L3 and without reopen behavior.

Use it to:
- identify enablement gaps
- prioritize knowledge and training investment

## Enablement Candidates
Issue cohorts where low FCR, high cycle burden, or higher-tier resolution mix suggest knowledge or training opportunities.

Use it to:
- target L1 enablement
- focus KB or playbook work

## Problem Candidates
Ranked repeat incident themes based on issue type, normalized short description, and impact signals such as volume, cycle time, and quality outcomes.

Use it to:
- feed problem-management intake
- separate repeat operational pain from isolated incidents
