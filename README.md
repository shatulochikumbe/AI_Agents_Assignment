# AI_Agents_Assignment

# AI Agent Strategy for Smart Manufacturing: AutoParts Inc.

## 📌 Project Overview
This repository contains the simulation and documentation for the comprehensive AI Agent implementation strategy proposed for AutoParts Inc., addressing challenges in defect reduction, predictive maintenance, and production orchestration.

## 🤖 Simulated Workflow: Predictive Maintenance Agent
A core agent was prototyped using n8n to demonstrate the autonomous "Observe-Reason-Act" cycle for predictive maintenance.

### **Live Simulation Link**
🔗 **[https://chikumbes.app.n8n.cloud/workflow/oJ4qXZ7iU6MoKFeA]**

### **Workflow Logic**
1.  **Observe:** Simulated IoT sensor (webhook) sends machine vibration data.
2.  **Reason:** AI Agent node analyzes the data against failure thresholds.
3.  **Act:** Based on the risk level, the workflow:
    *   Creates a maintenance ticket (CRITICAL).
    *   Sends an alert to the team lead (WARNING).
    *   Logs the event for monitoring (NORMAL).

### **How to Use the Simulation**
1.  The link above leads to a **read-only view** of the workflow.
2.  To test execution, you will need to **copy the workflow to your own n8n instance** (use the "Copy to n8n" button).
3.  In your n8n editor, you can trigger the workflow manually using the **"Execute Workflow"** button on the Webhook node.

## 🛠️ Repository Contents
| File | Description |
|------|-------------|
| `README.md` | This documentation file. |
| `workflow_export.json` | The exported JSON definition of the n8n workflow. Can be imported into any n8n instance. |
| `docs/system_architecture.png` | Diagram illustrating the multi-agent system architecture. |

## 📚 Full Assignment Report
The comprehensive analysis—covering the three-agent strategy, ROI, implementation timeline, and risk assessment—is available in the [Full Report](docs/full_report.pdf).