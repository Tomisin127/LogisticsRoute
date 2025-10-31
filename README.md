<img width="1024" height="1024" alt="1001224629" src="https://github.com/user-attachments/assets/3274020a-968d-41df-832a-24f0435e8c49" />
# LogisticsRoute
Logistics Route: An AI-powered agent for logistics and supply chain management, enabling users to plan deliveries, report route incidents, and list shipments using natural language commands. It provides route details, cost estimates, CO₂ emissions, and risk assessments for US-based logistics operations.

![tag:innovationlab](https://img.shields.io/badge/innovationlab-3D8BD3)

![tag:hackathon](https://img.shields.io/badge/hackathon-5F43F1)



# Logistics Agent: AI-Powered Delivery Planning and Incident Reporting

## Description
The *Logistics Agent* is an AI-powered assistant designed for logistics and supply chain management. It enables users to plan deliveries across US states, report incidents on routes, and list past shipments using natural language commands. Leveraging an LLM for flexible command parsing, it supports logistics professionals, shippers, and supply chain managers by providing route details, cost estimates, CO₂ emissions, and risk assessments. Interact via direct messages to streamline logistics operations.

## Use Case Examples
1. *Plan a Delivery*: Request a delivery plan with details like distance, estimated time, cost, and CO₂ emissions (e.g., "Arrange a delivery from New York to Los Angeles").
2. *Report Route Incidents*: Log issues on a route with severity and notes (e.g., "Report an issue on the route from NY to CA, severity 3, note Road closure").
3. *List Shipments*: Retrieve a history of planned shipments with IDs, routes, weights, costs, and timestamps (e.g., "List shipments").

## Capabilities and APIs
The Logistics Agent offers the following major functions:

- *Plan Delivery*:
  - *Input*: Origin and destination (US state abbreviations or city names like NY, Los Angeles), optional weight (kg), cargo type (e.g., general, perishable), and preferred mode.
  - *Output*: Delivery plan including distance (miles), ETA (hours), cost (USD), CO₂ emissions (kg), carrier, risk level, and notes.
  - *Process*: Uses a knowledge graph for route lookup, OpenRouteService API for distances, and pricing calculations based on weight, cargo, and route conditions.
  - Example: "Plan delivery from NY to CA" returns a detailed plan with cost and risk assessment.

- *Report Incident*:
  - *Input*: Origin and destination, optional severity (1–5), and note (e.g., "Road closure").
  - *Output*: Confirmation of the reported incident with route details, incident count, severity score, and forecast data (surge multiplier, expected delay).
  - Example: "Report incident from NY to CA severity 3 note Road closure" logs the incident and provides route status.

- *List Shipments*:
  - *Input*: None.
  - *Output*: List of stored shipments with IDs, routes, weights, costs, and timestamps, or "No shipments found."
  - Example: "List shipments" retrieves all past delivery plans.

### Input Data Model

```python
class PlanDeliveryRequest(Model):
    origin: str          # e.g., "NY" or "New York" (mapped to NY)
    destination: str     # e.g., "CA" or "Los Angeles" (mapped to CA)
    weight_kg: float     # default: 100.0
    cargo: str           # default: "general"
    prefer_mode: str     # default: None


class ReportIncidentRequest(Model):
    origin: str          # e.g., "NY"
    destination: str     # e.g., "CA"
    severity: int        # default: 1
    note: str            # default: ""

List Shipments requires no input parameters

Output Data Model

class PlanDeliveryResponse(Model):
    distance_miles: int
    est_time_hours: float
    cost_usd: float
    co2_kg: float
    carrier: str
    risk: str
    notes: str
    error: str              # if applicable


class ReportIncidentResponse(Model):
    incidents: int
    severity_score: int
    last_incident: str
    surge_multiplier: float
    expected_delay_hours: float


class ListShipmentsResponse(Model):
    shipments: List[Shipment]  # List of shipment objects or empty list
    # Shipment: {id: str, origin: str, dest: str, weight_kg: float, cost_usd: float, ts: str}
```
Interaction Modes

Direct Message: Send commands via Agentverse messages to the agent’s address:
agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx or @LogisticsRoute.

Supported Commands:

Plan Delivery: "Arrange a delivery from [origin] to [destination]"

Report Incident: "Report incident from NY to CA severity 3 note Road closure"

List Shipments: "List shipments" or "Show me all deliveries"


Note: The agent uses an LLM to parse natural language, so variations like "ship" or "schedule" are supported. Use US state abbreviations or city names for locations.

Limitations and Scope

Geographic Scope: US states/cities only.

Command Parsing: Invalid commands return "Please use a valid command."

Response Issue: Possible mailbox/uagents library responses may fail.

External APIs: Depends on OpenRouteService + ASI API.

Not Supported: International shipping, real-time tracking, weather queries.


Usage Guidelines

Send commands to the agent’s address.

Use natural language or structured commands.

If no response, enable debug logs and verify Agentverse connectivity.

Test with "list shipments" to confirm connectivity.


Licensing

MIT License. Contact the developer for commercial licensing inquiries.

Contact Information

Developer: https://x.com/paulraimi11
Email: paul.raimi.pr@gmail.com
Agentverse Profile: https://agentverse.ai/agents/details/agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx/profile

Acknowledgments

Built using the uagents library

Powered by ASI:One

Utilizes OpenRouteService API


Keywords and Tags

Logistics
Supply Chain
Delivery Planning
Incident Reporting
Shipment Tracking
Route Optimization
CO₂ Emissions
US Logistics

