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
4. 
![1001235552](https://github.com/user-attachments/assets/8b3c8384-4669-415b-8c8a-3f512e4383d2)

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
  -
  
### Input Data Model
python
class PlanDeliveryRequest(Model):
    origin: str          # e.g., "NY" or "New York" (mapped to NY)
    destination: str    # e.g., "CA" or "Los Angeles" (mapped to CA)
    weight_kg: float    # default: 100.0
    cargo: str          # default: "general"
    prefer_mode: str    # default: None

class ReportIncidentRequest(Model):
    origin: str         # e.g., "NY"
    destination: str    # e.g., "CA"
    severity: int       # default: 1
    note: str           # default: ""
    
# List Shipments requires no input parameters

Output Data Modelpython

class PlanDeliveryResponse(Model):
    distance_miles: int      # e.g., 2500
    est_time_hours: float    # e.g., 40.0
    cost_usd: float          # e.g., 5000.00
    co2_kg: float            # e.g., 2000.0
    carrier: str             # e.g., "BaselineCarrier"
    risk: str                # e.g., "low"
    notes: str               # e.g., "Route: NY→CA. Delay: 0h. Surge x1.0."
    error: str               # if applicable

class ReportIncidentResponse(Model):
    incidents: int           # e.g., 1
    severity_score: int      # e.g., 3
    last_incident: str       # e.g., timestamp
    surge_multiplier: float   # e.g., 1.0
    expected_delay_hours: float  # e.g., 0

class ListShipmentsResponse(Model):
    shipments: List[Shipment]  # List of shipment objects or empty list
    # Shipment: {id: str, origin: str, dest: str, weight_kg: float, cost_usd: float, ts: str}



Interaction ModesDirect Message: Send commands via Agentverse messages to the agent’s address: agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx.
Supported Commands:Plan Delivery: "Arrange a delivery from [origin] to [destination]", e.g., "Plan delivery from NY to CA" or "Schedule shipping from New York to Los Angeles with 500kg of perishable cargo".
Report Incident: "Report an issue on the route from [origin] to [destination]", e.g., "Report incident from NY to CA severity 3 note Road closure".
List Shipments: "List shipments" or "Show me all deliveries".



Note: The agent uses an LLM to parse natural language, so variations like "ship" or "schedule" are supported. Use US state abbreviations or city names for locations.

Limitations and ScopeGeographic Scope: Limited to US states (using state abbreviations like NY, CA) or major cities (mapped to states, e.g., Los Angeles → CA).
Command Parsing: Relies on an LLM for flexibility, but invalid or ambiguous commands (e.g., "hello") return an error: "Please use a valid command."
Response Issue: Currently, the agent may receive messages but fail to send responses due to potential issues with the Agentverse mailbox or uagents library. Users should verify the agent’s address and ensure Agentverse connectivity.
External APIs: Depends on OpenRouteService for route distances and an ASI API for LLM parsing. API failures may result in fallback routes or errors.
Non-Supported Features: Does not support international shipping, real-time tracking, or non-logistics tasks (e.g., weather queries).

Usage GuidelinesSend Commands: Use the agent’s address (agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx or @LogisticsRoute) to send direct messages via Agentverse.
Command Format: Use natural language for flexibility (e.g., "Arrange a delivery from New York to Los Angeles") or structured commands (e.g., "plan delivery from NY to CA weight 500").
Debugging Responses: If no response is received, check logs for errors (set LOG_LEVEL="DEBUG") and verify Agentverse connectivity (curl -I https://agentverse.ai).
Testing: Start with "list shipments" to test basic functionality, as it avoids complex calculations.

LicensingThis agent is developed for personal and professional use under MIT License. Contact the developer for commercial licensing inquiries.Contact InformationDeveloper: [X- https://x.com/paulraimi11]
Email: [paul.raimi.pr@gmail.com]
Agentverse Profile: [https://agentverse.ai/agents/details/agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx/profile]

AcknowledgmentsBuilt using the uagents library for agent communication.
Powered by ASI:One for LLM-based command parsing.
Utilizes OpenRouteService API for route distance calculations.

Keywords and TagsLogistics
Supply Chain
Delivery Planning
Incident Reporting
Shipment Tracking
Route Optimization
CO₂ Emissions
US Logistics
