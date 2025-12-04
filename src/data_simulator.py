"""
simulation.py - Smart Manufacturing AI Agent Simulation for AutoParts Inc.

This file implements a comprehensive multi-agent system for AutoParts Inc.'s
smart manufacturing transformation. The simulation includes three specialized
AI agents that work collaboratively to address production challenges.

Author: AI Assignment Implementation
Date: 2025-12-04
Version: 1.0
"""

import json
import random
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from enum import Enum
import time

# ----------------------------------------------------------------------------
# Data Models and Configuration
# ----------------------------------------------------------------------------

class MachineStatus(Enum):
    """Machine operational status enumeration."""
    OPERATIONAL = "operational"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"

class OrderPriority(Enum):
    """Production order priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class MachineMetrics:
    """Data model for machine operational metrics."""
    machine_id: str
    machine_name: str
    machine_type: str
    temperature: float
    vibration: float
    power_consumption: float
    running_hours: int
    last_maintenance: str
    efficiency: float
    status: str
    
@dataclass
class QualityMetrics:
    """Data model for product quality measurements."""
    batch_id: str
    product_type: str
    inspection_date: str
    defect_rate: float
    dimensional_accuracy: float
    surface_finish: float
    total_units_inspected: int
    defects_found: int
    defect_types: List[str]

@dataclass
class ProductionSchedule:
    """Data model for production scheduling."""
    order_id: str
    product_type: str
    quantity: int
    scheduled_start: str
    scheduled_end: str
    priority: str
    assigned_machines: List[str]
    current_progress: int

@dataclass
class CustomOrder:
    """Data model for custom customer orders."""
    custom_order_id: str
    customer_name: str
    product_type: str
    specifications: Dict[str, Any]
    quantity: int
    deadline: str
    status: str
    estimated_cost: float

@dataclass
class ManufacturingConfig:
    """Configuration settings for the manufacturing simulation."""
    defect_threshold: float = 0.15  # 15% threshold
    maintenance_alert_threshold: float = 0.7
    production_target_efficiency: float = 0.85
    customization_lead_time_days: int = 5
    simulation_interval_minutes: int = 15

# ----------------------------------------------------------------------------
# Base AI Agent Class (Implements ReAct Pattern)
# ----------------------------------------------------------------------------

class AIAgent:
    """
    Base class for AI agents implementing the ReAct pattern (Reason + Act).
    This pattern enables agents to reason about their environment and take
    appropriate actions based on their observations[citation:7][citation:8].
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.action_history = []
        
    def think(self, observation: Dict[str, Any]) -> str:
        """Reason about the current observation."""
        raise NotImplementedError("Subclasses must implement think method")
    
    def act(self, thought: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Take action based on reasoning."""
        raise NotImplementedError("Subclasses must implement act method")
    
    def observe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process observation data."""
        return data
    
    def run_react_cycle(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the complete ReAct pattern cycle[citation:7]:
        1. Thought: Reason about the observation
        2. Action: Take appropriate action
        3. Observation: Process results
        4. Answer: Generate final response
        """
        # Step 1: Thought
        thought = self.think(observation)
        self.action_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "thought": thought
        })
        
        # Step 2: Action
        action_result = self.act(thought, observation)
        
        # Step 3: Observation
        processed_result = self.observe(action_result)
        
        # Step 4: Answer
        answer = self._generate_answer(thought, processed_result)
        
        return {
            "thought": thought,
            "action_result": action_result,
            "final_answer": answer
        }
    
    def _generate_answer(self, thought: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final answer from thought and result."""
        return {
            "summary": f"{self.name} completed analysis",
            "details": result,
            "recommendation": self._extract_recommendation(thought, result)
        }
    
    def _extract_recommendation(self, thought: str, result: Dict[str, Any]) -> str:
        """Extract recommendation from thought and result."""
        return "Implement suggested actions based on analysis."

# ----------------------------------------------------------------------------
# Specialized AI Agents
# ----------------------------------------------------------------------------

class QualityControlAgent(AIAgent):
    """
    Autonomous Quality Inspector Agent.
    
    Analyzes quality metrics to identify defect patterns and recommend
    corrective actions. Goal-based agent targeting defect rate reduction[citation:1].
    """
    
    def __init__(self):
        super().__init__(
            name="Autonomous Quality Inspector",
            description="Analyzes quality data to calculate defect rates, identify patterns, and classify defect types"
        )
        self.defect_patterns = {}
        
    def think(self, observation: Dict[str, Any]) -> str:
        """Reason about quality metrics and defect patterns."""
        quality_data = observation.get("quality_metrics", [])
        
        if not quality_data:
            return "No quality data available for analysis."
        
        # Analyze defect patterns
        total_defects = sum(q.defects_found for q in quality_data)
        total_units = sum(q.total_units_inspected for q in quality_data)
        overall_defect_rate = (total_defects / total_units) * 100 if total_units > 0 else 0
        
        thought = f"Analyzing {len(quality_data)} quality batches. "
        thought += f"Overall defect rate: {overall_defect_rate:.2f}%. "
        
        if overall_defect_rate > 15:  # 15% threshold from AutoParts Inc.
            thought += "CRITICAL: Defect rate exceeds 15% threshold. "
            thought += "Need to identify root causes and implement immediate corrective actions."
        elif overall_defect_rate > 10:
            thought += "WARNING: Defect rate above 10%. "
            thought += "Monitor closely and consider process adjustments."
        else:
            thought += "NORMAL: Defect rate within acceptable range. "
            thought += "Continue monitoring for early detection of issues."
            
        return thought
    
    def act(self, thought: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed defect analysis."""
        quality_data = data.get("quality_metrics", [])
        config = data.get("config", ManufacturingConfig())
        
        analysis = {
            "timestamp": datetime.datetime.now().isoformat(),
            "overall_defect_rate": 0,
            "defects_by_component": {},
            "defects_by_production_line": {},
            "patterns": [],
            "recommendations": []
        }
        
        if not quality_data:
            return analysis
        
        # Calculate overall defect rate
        total_defects = sum(q.defects_found for q in quality_data)
        total_units = sum(q.total_units_inspected for q in quality_data)
        analysis["overall_defect_rate"] = (total_defects / total_units * 100) if total_units > 0 else 0
        
        # Analyze by product type
        defect_by_type = {}
        for qm in quality_data:
            if qm.product_type not in defect_by_type:
                defect_by_type[qm.product_type] = {"defects": 0, "units": 0}
            defect_by_type[qm.product_type]["defects"] += qm.defects_found
            defect_by_type[qm.product_type]["units"] += qm.total_units_inspected
        
        for product_type, stats in defect_by_type.items():
            defect_rate = (stats["defects"] / stats["units"] * 100) if stats["units"] > 0 else 0
            analysis["defects_by_component"][product_type] = {
                "produced": stats["units"],
                "defects": stats["defects"],
                "defect_rate": f"{defect_rate:.2f}%"
            }
        
        # Identify patterns
        if analysis["overall_defect_rate"] > 5:
            analysis["patterns"].append("High overall defect rate detected - exceeds 5% threshold")
        
        # Find component with highest defect rate
        highest_defect_component = None
        highest_rate = 0
        for component, data in analysis["defects_by_component"].items():
            rate = float(data["defect_rate"].rstrip("%"))
            if rate > highest_rate:
                highest_rate = rate
                highest_defect_component = component
        
        if highest_defect_component:
            analysis["patterns"].append(
                f"{highest_defect_component} has the highest defect rate at {highest_rate:.2f}%"
            )
        
        # Generate recommendations
        if analysis["overall_defect_rate"] > config.defect_threshold * 100:
            analysis["recommendations"].append(
                "Implement immediate quality control review"
            )
            analysis["recommendations"].append(
                "Increase inspection frequency on high-defect components"
            )
        
        if highest_rate > 10:
            analysis["recommendations"].append(
                f"Focus on {highest_defect_component} - consider process review and operator training"
            )
        
        return analysis

class PredictiveMaintenanceAgent(AIAgent):
    """
    Predictive Maintenance Agent.
    
    Analyzes machine health metrics and predicts potential failures before they occur.
    Model-based reflex agent that maintains internal models of machine health[citation:1].
    """
    
    def __init__(self):
        super().__init__(
            name="Predictive Maintenance Agent",
            description="Predicts machine failure probability based on operational metrics using statistical analysis"
        )
        self.machine_health_history = {}
        
    def think(self, observation: Dict[str, Any]) -> str:
        """Reason about machine health and maintenance needs."""
        machine_data = observation.get("machine_metrics", [])
        
        if not machine_data:
            return "No machine data available for analysis."
        
        # Analyze overall machine health
        warning_count = sum(1 for m in machine_data if m.status == MachineStatus.WARNING.value)
        critical_count = sum(1 for m in machine_data if m.status == MachineStatus.CRITICAL.value)
        
        thought = f"Analyzing {len(machine_data)} machines. "
        thought += f"Found {warning_count} machines in warning state, {critical_count} in critical state. "
        
        if critical_count > 0:
            thought += "CRITICAL: Machines require immediate maintenance. "
            thought += "Schedule maintenance to prevent production downtime."
        elif warning_count > 0:
            thought += "WARNING: Some machines showing early signs of degradation. "
            thought += "Schedule preventive maintenance during next available window."
        else:
            thought += "NORMAL: All machines operating within normal parameters. "
            thought += "Continue routine monitoring."
            
        return thought
    
    def act(self, thought: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform machine health analysis and failure prediction."""
        machine_data = data.get("machine_metrics", [])
        
        # Health thresholds
        thresholds = {
            "vibration": {"normal": 2.5, "warning": 5.0, "critical": 7.5},
            "temperature": {"normal": 70, "warning": 85, "critical": 95},
            "runtimeHours": {"normal": 2000, "warning": 5000, "critical": 8000},
            "errorCounts": {"normal": 5, "warning": 15, "critical": 30}
        }
        
        analysis_results = []
        maintenance_recommendations = []
        
        for machine in machine_data:
            # Calculate individual risk scores
            vibration_risk = self._calculate_risk_score(
                machine.vibration, thresholds["vibration"]
            )
            temperature_risk = self._calculate_risk_score(
                machine.temperature, thresholds["temperature"]
            )
            runtime_risk = self._calculate_risk_score(
                machine.running_hours, thresholds["runtimeHours"]
            )
            
            # Calculate weighted failure probability
            failure_probability = (
                vibration_risk * 0.35 +
                temperature_risk * 0.30 +
                runtime_risk * 0.20
            )
            
            # Determine maintenance urgency
            urgency, action, timeframe = self._determine_urgency(failure_probability)
            
            # Identify risk factors
            risk_factors = []
            if vibration_risk > 50:
                risk_factors.append(f"High vibration ({machine.vibration:.2f} mm/s)")
            if temperature_risk > 50:
                risk_factors.append(f"Elevated temperature ({machine.temperature:.1f}°C)")
            if runtime_risk > 50:
                risk_factors.append(f"Extended runtime ({machine.running_hours} hours)")
            
            machine_analysis = {
                "machine_id": machine.machine_id,
                "machine_name": machine.machine_name,
                "failure_probability": round(failure_probability, 1),
                "urgency_level": urgency,
                "recommended_action": action,
                "maintenance_timeframe": timeframe,
                "risk_factors": risk_factors if risk_factors else ["No significant risk factors detected"],
                "health_score": 100 - failure_probability
            }
            
            analysis_results.append(machine_analysis)
            
            if urgency in ["HIGH", "CRITICAL"]:
                maintenance_recommendations.append({
                    "machine_id": machine.machine_id,
                    "machine_name": machine.machine_name,
                    "priority": urgency,
                    "action": action,
                    "timeframe": timeframe
                })
        
        return {
            "machine_analysis": analysis_results,
            "maintenance_recommendations": maintenance_recommendations,
            "summary": {
                "total_machines": len(machine_data),
                "machines_requiring_maintenance": len(maintenance_recommendations),
                "overall_health_score": self._calculate_overall_health(analysis_results)
            }
        }
    
    def _calculate_risk_score(self, value: float, threshold: Dict[str, float]) -> float:
        """Calculate risk score based on threshold values."""
        if value <= threshold["normal"]:
            return 0
        elif value <= threshold["warning"]:
            return 25 + ((value - threshold["normal"]) / (threshold["warning"] - threshold["normal"])) * 25
        elif value <= threshold["critical"]:
            return 50 + ((value - threshold["warning"]) / (threshold["critical"] - threshold["warning"])) * 30
        else:
            return 80 + min(20, ((value - threshold["critical"]) / threshold["critical"]) * 20)
    
    def _determine_urgency(self, failure_probability: float) -> tuple:
        """Determine maintenance urgency based on failure probability."""
        if failure_probability < 25:
            return "LOW", "Continue normal operations with routine monitoring", "30+ days"
        elif failure_probability < 50:
            return "MODERATE", "Schedule preventive maintenance within next maintenance window", "14-30 days"
        elif failure_probability < 75:
            return "HIGH", "Prioritize maintenance - schedule within next week", "3-7 days"
        else:
            return "CRITICAL", "IMMEDIATE maintenance required - risk of imminent failure", "0-2 days"
    
    def _calculate_overall_health(self, analysis_results: List[Dict]) -> float:
        """Calculate overall factory health score."""
        if not analysis_results:
            return 100.0
        return sum(m["health_score"] for m in analysis_results) / len(analysis_results)

class ProductionOrchestratorAgent(AIAgent):
    """
    Production Orchestration Agent.
    
    Optimizes production schedules to maximize efficiency while accommodating
    custom orders. Utility-based agent that balances multiple objectives[citation:1].
    """
    
    def __init__(self):
        super().__init__(
            name="Production Orchestration Agent",
            description="Optimizes production schedules to maximize efficiency while accommodating custom orders"
        )
        self.schedule_history = []
        
    def think(self, observation: Dict[str, Any]) -> str:
        """Reason about production scheduling and optimization opportunities."""
        production_schedule = observation.get("production_schedule", [])
        custom_orders = observation.get("custom_orders", [])
        machine_metrics = observation.get("machine_metrics", [])
        
        thought = f"Analyzing {len(production_schedule)} production orders "
        thought += f"and {len(custom_orders)} custom orders. "
        
        # Check for scheduling conflicts
        urgent_custom_orders = len([
            o for o in custom_orders 
            if o.status == "pending" and self._days_until_deadline(o.deadline) <= 5
        ])
        
        if urgent_custom_orders > 0:
            thought += f"CRITICAL: {urgent_custom_orders} urgent custom orders require scheduling. "
            thought += "Need to optimize production schedule to accommodate."
        
        # Check machine utilization
        operational_machines = len([m for m in machine_metrics if m.status == "operational"])
        if operational_machines < len(machine_metrics) * 0.9:
            thought += f"WARNING: Only {operational_machines}/{len(machine_metrics)} machines operational. "
            thought += "Consider maintenance scheduling impact on production."
        
        # Check order progress
        avg_progress = sum(o.current_progress for o in production_schedule) / len(production_schedule) if production_schedule else 0
        if avg_progress < 50:
            thought += f"WARNING: Average order progress is {avg_progress:.1f}%. "
            thought += "Monitor closely for potential delays."
        
        thought += "Will optimize schedule for maximum efficiency and on-time delivery."
        return thought
    
    def act(self, thought: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize production schedule and manage custom orders."""
        production_schedule = data.get("production_schedule", [])
        custom_orders = data.get("custom_orders", [])
        machine_metrics = data.get("machine_metrics", [])
        config = data.get("config", ManufacturingConfig())
        
        # Prepare schedule data for optimization
        schedule_data = {
            "orders": production_schedule,
            "machines": machine_metrics,
            "custom_orders": custom_orders
        }
        
        # Calculate current efficiency metrics
        current_efficiency = self._calculate_schedule_efficiency(production_schedule, machine_metrics)
        
        # Optimize schedule
        optimized_schedule = self._optimize_schedule(production_schedule, custom_orders, machine_metrics)
        optimized_efficiency = self._calculate_schedule_efficiency(optimized_schedule, machine_metrics)
        
        # Analyze custom orders
        custom_order_analysis = self._analyze_custom_orders(custom_orders, config)
        
        # Calculate improvements
        efficiency_improvement = (
            (optimized_efficiency - current_efficiency) / current_efficiency * 100 
            if current_efficiency > 0 else 0
        )
        
        # Generate recommendations
        recommendations = []
        if efficiency_improvement > 0:
            recommendations.append(f"Schedule optimization can improve efficiency by {efficiency_improvement:.2f}%")
        
        if custom_order_analysis["urgent_orders"] > 0:
            recommendations.append(
                f"Prioritize {custom_order_analysis['urgent_orders']} urgent custom orders"
            )
        
        # Check for machine bottlenecks
        machine_utilization = self._calculate_machine_utilization(optimized_schedule, machine_metrics)
        for machine_id, utilization in machine_utilization.items():
            if utilization > 80:  # 80% utilization threshold
                recommendations.append(f"Machine {machine_id} is overutilized - consider load balancing")
        
        return {
            "current_efficiency": f"{current_efficiency:.2f}%",
            "optimized_efficiency": f"{optimized_efficiency:.2f}%",
            "efficiency_improvement": f"{efficiency_improvement:.2f}%",
            "custom_order_analysis": custom_order_analysis,
            "machine_utilization": machine_utilization,
            "recommendations": recommendations,
            "optimized_schedule_summary": {
                "total_orders": len(optimized_schedule),
                "high_priority_orders": len([o for o in optimized_schedule if o.priority == "high"]),
                "estimated_completion_time": self._estimate_completion_time(optimized_schedule)
            }
        }
    
    def _calculate_schedule_efficiency(self, schedule: List[ProductionSchedule], machines: List[MachineMetrics]) -> float:
        """Calculate overall schedule efficiency."""
        if not schedule or not machines:
            return 0.0
        
        # Calculate machine utilization
        machine_hours = {m.machine_id: 0 for m in machines}
        for order in schedule:
            for machine_id in order.assigned_machines:
                if machine_id in machine_hours:
                    # Estimate hours based on quantity
                    machine_hours[machine_id] += order.quantity / 100  # Simplified calculation
        
        # Calculate efficiency based on utilization and machine health
        total_efficiency = 0
        count = 0
        for machine in machines:
            if machine.machine_id in machine_hours:
                utilization = min(100, machine_hours[machine.machine_id] / 8 * 100)  # Assume 8-hour day
                machine_efficiency = machine.efficiency * (utilization / 100)
                total_efficiency += machine_efficiency
                count += 1
        
        return total_efficiency / count if count > 0 else 0.0
    
    def _optimize_schedule(self, schedule: List[ProductionSchedule], 
                          custom_orders: List[CustomOrder],
                          machines: List[MachineMetrics]) -> List[ProductionSchedule]:
        """Optimize production schedule."""
        # Combine regular and custom orders
        all_orders = schedule.copy()
        
        # Convert custom orders to production schedule format
        for custom_order in custom_orders:
            if custom_order.status in ["pending", "in-progress"]:
                production_order = ProductionSchedule(
                    order_id=custom_order.custom_order_id,
                    product_type=custom_order.product_type,
                    quantity=custom_order.quantity,
                    scheduled_start=datetime.datetime.now().strftime("%Y-%m-%d"),
                    scheduled_end=custom_order.deadline,
                    priority="high" if self._days_until_deadline(custom_order.deadline) <= 5 else "normal",
                    assigned_machines=self._assign_machines(custom_order.product_type, machines),
                    current_progress=50 if custom_order.status == "in-progress" else 0
                )
                all_orders.append(production_order)
        
        # Sort by priority and deadline
        all_orders.sort(key=lambda x: (
            0 if x.priority == "high" else 1 if x.priority == "normal" else 2,
            x.scheduled_end
        ))
        
        return all_orders
    
    def _analyze_custom_orders(self, custom_orders: List[CustomOrder], config: ManufacturingConfig) -> Dict[str, Any]:
        """Analyze custom order requirements and urgency."""
        analysis = {
            "total_custom_orders": len(custom_orders),
            "pending_orders": len([o for o in custom_orders if o.status == "pending"]),
            "in_progress_orders": len([o for o in custom_orders if o.status == "in-progress"]),
            "urgent_orders": 0,
            "total_estimated_revenue": sum(o.estimated_cost for o in custom_orders),
            "order_details": []
        }
        
        for order in custom_orders:
            days_until_deadline = self._days_until_deadline(order.deadline)
            is_urgent = days_until_deadline <= config.customization_lead_time_days
            
            if is_urgent:
                analysis["urgent_orders"] += 1
            
            analysis["order_details"].append({
                "custom_order_id": order.custom_order_id,
                "customer_name": order.customer_name,
                "product_type": order.product_type,
                "deadline": order.deadline,
                "days_until_deadline": days_until_deadline,
                "is_urgent": is_urgent,
                "status": order.status
            })
        
        return analysis
    
    def _assign_machines(self, product_type: str, machines: List[MachineMetrics]) -> List[str]:
        """Assign appropriate machines for product type."""
        # Simplified machine assignment logic
        assigned = []
        for machine in machines:
            if machine.status == "operational":
                if "CNC" in machine.machine_name and "Engine" in product_type:
                    assigned.append(machine.machine_id)
                elif "Lathe" in machine.machine_name and "Gear" in product_type:
                    assigned.append(machine.machine_id)
                elif "Press" in machine.machine_name and "Rotor" in product_type:
                    assigned.append(machine.machine_id)
        
        return assigned[:2] if assigned else ["M001"]  # Default to machine 1
    
    def _calculate_machine_utilization(self, schedule: List[ProductionSchedule], 
                                      machines: List[MachineMetrics]) -> Dict[str, float]:
        """Calculate utilization percentage for each machine."""
        utilization = {m.machine_id: 0 for m in machines}
        
        for order in schedule:
            for machine_id in order.assigned_machines:
                if machine_id in utilization:
                    utilization[machine_id] += order.quantity / 1000  # Simplified calculation
        
        # Convert to percentage (assuming 8-hour work day)
        for machine_id in utilization:
            utilization[machine_id] = min(100, utilization[machine_id] / 8 * 100)
        
        return utilization
    
    def _estimate_completion_time(self, schedule: List[ProductionSchedule]) -> str:
        """Estimate when all orders will be completed."""
        if not schedule:
            return "No orders scheduled"
        
        end_dates = [datetime.datetime.strptime(o.scheduled_end, "%Y-%m-%d") for o in schedule]
        latest_date = max(end_dates)
        return latest_date.strftime("%Y-%m-%d")
    
    def _days_until_deadline(self, deadline: str) -> int:
        """Calculate days until deadline."""
        try:
            deadline_date = datetime.datetime.strptime(deadline, "%Y-%m-%d")
            days = (deadline_date - datetime.datetime.now()).days
            return max(0, days)
        except:
            return 999  # Far future if date parsing fails

# ----------------------------------------------------------------------------
# Production Data Simulator
# ----------------------------------------------------------------------------

class ProductionDataSimulator:
    """
    Simulates realistic production data for AutoParts Inc.
    Generates synthetic manufacturing data including machine metrics,
    quality measurements, production schedules, and custom orders[citation:2].
    """
    
    def __init__(self, config: ManufacturingConfig = None):
        self.config = config or ManufacturingConfig()
        self.machine_templates = [
            {"id": "M001", "name": "CNC Mill A", "type": "Milling"},
            {"id": "M002", "name": "CNC Mill B", "type": "Milling"},
            {"id": "M003", "name": "Lathe A", "type": "Turning"},
            {"id": "M004", "name": "Press A", "type": "Stamping"},
            {"id": "M005", "name": "Welder A", "type": "Welding"}
        ]
        
        self.product_types = ["Engine Block", "Transmission Gear", "Brake Rotor"]
        self.custom_product_types = ["Custom Engine Block", "Custom Transmission Housing", "Custom Suspension Part"]
        self.defect_types = {
            "Engine Block": ["surface scratches", "dimensional variance", "material defects"],
            "Transmission Gear": ["tooth alignment", "hardness variance", "surface finish"],
            "Brake Rotor": ["surface porosity", "thickness variance", "warping"]
        }
    
    def generate_machine_metrics(self) -> List[MachineMetrics]:
        """Generate realistic machine operational metrics."""
        metrics = []
        
        for template in self.machine_templates:
            # Simulate realistic variations
            temperature = random.uniform(65, 95)
            vibration = random.uniform(0.5, 3.5)
            power_consumption = random.uniform(15, 45)
            running_hours = random.randint(1200, 8500)
            
            # Determine status based on metrics
            if vibration > 7.0 or temperature > 90:
                status = MachineStatus.CRITICAL.value
            elif vibration > 5.0 or temperature > 85 or running_hours > 7000:
                status = MachineStatus.WARNING.value
            else:
                status = MachineStatus.OPERATIONAL.value
            
            # Calculate efficiency
            efficiency = 100 - (vibration * 2) - (max(0, temperature - 70) * 0.5)
            efficiency = max(65, min(98, efficiency))
            
            metrics.append(MachineMetrics(
                machine_id=template["id"],
                machine_name=template["name"],
                machine_type=template["type"],
                temperature=round(temperature, 1),
                vibration=round(vibration, 2),
                power_consumption=round(power_consumption, 1),
                running_hours=running_hours,
                last_maintenance=(datetime.datetime.now() - 
                                datetime.timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d"),
                efficiency=round(efficiency, 1),
                status=status
            ))
        
        return metrics
    
    def generate_quality_metrics(self) -> List[QualityMetrics]:
        """Generate quality inspection data."""
        metrics = []
        
        for product_type in self.product_types:
            total_units = random.randint(100, 500)
            defects_found = random.randint(1, 15)
            defect_rate = (defects_found / total_units) * 100
            
            # Select defect types for this batch
            available_defects = self.defect_types.get(product_type, ["unknown defect"])
            num_defect_types = random.randint(1, min(3, len(available_defects)))
            selected_defects = random.sample(available_defects, num_defect_types)
            
            metrics.append(QualityMetrics(
                batch_id=f"B{random.randint(1000, 9999)}",
                product_type=product_type,
                inspection_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                defect_rate=round(defect_rate, 2),
                dimensional_accuracy=round(random.uniform(96, 99.9), 2),
                surface_finish=round(random.uniform(85, 99), 1),
                total_units_inspected=total_units,
                defects_found=defects_found,
                defect_types=selected_defects
            ))
        
        return metrics
    
    def generate_production_schedule(self) -> List[ProductionSchedule]:
        """Generate production schedule data."""
        schedule = []
        
        for product_type in self.product_types:
            days_from_now = random.randint(1, 21)
            schedule.append(ProductionSchedule(
                order_id=f"ORD{random.randint(10000, 99999)}",
                product_type=product_type,
                quantity=random.randint(500, 2000),
                scheduled_start=(datetime.datetime.now() + 
                               datetime.timedelta(days=random.randint(1, 7))).strftime("%Y-%m-%d"),
                scheduled_end=(datetime.datetime.now() + 
                              datetime.timedelta(days=days_from_now)).strftime("%Y-%m-%d"),
                priority="high" if random.random() > 0.7 else "normal",
                assigned_machines=[f"M00{random.randint(1, 5)}" for _ in range(random.randint(1, 2))],
                current_progress=random.randint(0, 85)
            ))
        
        return schedule
    
    def generate_custom_orders(self) -> List[CustomOrder]:
        """Generate custom order data."""
        orders = []
        customers = ["Premium Auto Corp", "Elite Motors Ltd", "Performance Parts Inc", "Custom Auto Works"]
        
        for i in range(2):  # Generate 2 custom orders
            product_type = random.choice(self.custom_product_types)
            customer = random.choice(customers)
            
            # Generate specifications based on product type
            if "Engine" in product_type:
                specifications = {
                    "material": "Aluminum Alloy 7075",
                    "tolerance": "±0.001 inches",
                    "surfaceFinish": "Ra 0.8 μm",
                    "specialRequirements": "Heat treated, anodized finish"
                }
            elif "Transmission" in product_type:
                specifications = {
                    "material": "Cast Iron Grade 60",
                    "tolerance": "±0.002 inches",
                    "surfaceFinish": "Ra 1.6 μm",
                    "specialRequirements": "Pressure tested, coated"
                }
            else:
                specifications = {
                    "material": "Steel Alloy",
                    "tolerance": "±0.003 inches",
                    "surfaceFinish": "Ra 3.2 μm",
                    "specialRequirements": "Stress relieved"
                }
            
            orders.append(CustomOrder(
                custom_order_id=f"CUST{random.randint(1000, 9999)}",
                customer_name=customer,
                product_type=product_type,
                specifications=specifications,
                quantity=random.randint(50, 300),
                deadline=(datetime.datetime.now() + 
                         datetime.timedelta(days=random.randint(14, 45))).strftime("%Y-%m-%d"),
                status=random.choice(["pending", "in-progress"]),
                estimated_cost=random.randint(15000, 50000)
            ))
        
        return orders
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate complete production dataset."""
        machine_metrics = self.generate_machine_metrics()
        quality_metrics = self.generate_quality_metrics()
        production_schedule = self.generate_production_schedule()
        custom_orders = self.generate_custom_orders()
        
        # Calculate summary statistics
        operational_machines = len([m for m in machine_metrics if m.status == "operational"])
        avg_efficiency = sum(m.efficiency for m in machine_metrics) / len(machine_metrics) if machine_metrics else 0
        avg_defect_rate = sum(q.defect_rate for q in quality_metrics) / len(quality_metrics) if quality_metrics else 0
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "facility": "AutoParts Inc - Main Production Facility",
            "machine_metrics": machine_metrics,
            "quality_metrics": quality_metrics,
            "production_schedule": production_schedule,
            "custom_orders": custom_orders,
            "summary": {
                "total_machines": len(machine_metrics),
                "operational_machines": operational_machines,
                "average_efficiency": round(avg_efficiency, 1),
                "total_active_orders": len(production_schedule),
                "total_custom_orders": len(custom_orders),
                "average_defect_rate": round(avg_defect_rate, 2)
            }
        }

# ----------------------------------------------------------------------------
# Main Simulation Orchestrator
# ----------------------------------------------------------------------------

class ManufacturingSimulation:
    """
    Main orchestrator for the smart manufacturing simulation.
    Coordinates all AI agents and manages the simulation workflow[citation:4].
    """
    
    def __init__(self):
        self.config = ManufacturingConfig()
        self.data_simulator = ProductionDataSimulator(self.config)
        
        # Initialize AI agents
        self.quality_agent = QualityControlAgent()
        self.maintenance_agent = PredictiveMaintenanceAgent()
        self.orchestrator_agent = ProductionOrchestratorAgent()
        
        self.simulation_data = None
        self.agent_results = {}
        
    def run_simulation_cycle(self) -> Dict[str, Any]:
        """
        Run a complete simulation cycle including:
        1. Data generation
        2. Agent analysis
        3. Result compilation
        """
        print("=" * 70)
        print("SMART MANUFACTURING AI AGENT SIMULATION - AutoParts Inc.")
        print("=" * 70)
        
        # Step 1: Generate production data
        print("\n📊 STEP 1: Generating production data...")
        self.simulation_data = self.data_simulator.generate_complete_dataset()
        print(f"   • Generated data for {self.simulation_data['summary']['total_machines']} machines")
        print(f"   • {self.simulation_data['summary']['operational_machines']} machines operational")
        print(f"   • {self.simulation_data['summary']['total_active_orders']} active production orders")
        print(f"   • {self.simulation_data['summary']['total_custom_orders']} custom orders")
        
        # Prepare observation data for agents
        observation_data = {
            "quality_metrics": self.simulation_data["quality_metrics"],
            "machine_metrics": self.simulation_data["machine_metrics"],
            "production_schedule": self.simulation_data["production_schedule"],
            "custom_orders": self.simulation_data["custom_orders"],
            "config": self.config
        }
        
        # Step 2: Run Quality Control Agent
        print("\n🤖 STEP 2: Running Quality Control Agent...")
        quality_result = self.quality_agent.run_react_cycle(observation_data)
        self.agent_results["quality"] = quality_result
        print(f"   • Overall defect rate: {quality_result['action_result'].get('overall_defect_rate', 0):.2f}%")
        print(f"   • Recommendations: {len(quality_result['action_result'].get('recommendations', []))}")
        
        # Step 3: Run Predictive Maintenance Agent
        print("\n🔧 STEP 3: Running Predictive Maintenance Agent...")
        maintenance_result = self.maintenance_agent.run_react_cycle(observation_data)
        self.agent_results["maintenance"] = maintenance_result
        critical_machines = len([
            m for m in maintenance_result['action_result'].get('machine_analysis', [])
            if m.get('urgency_level') in ['HIGH', 'CRITICAL']
        ])
        print(f"   • Machines analyzed: {len(maintenance_result['action_result'].get('machine_analysis', []))}")
        print(f"   • Machines requiring maintenance: {critical_machines}")
        
        # Step 4: Run Production Orchestrator Agent
        print("\n📅 STEP 4: Running Production Orchestrator Agent...")
        orchestration_result = self.orchestrator_agent.run_react_cycle(observation_data)
        self.agent_results["orchestration"] = orchestration_result
        efficiency_improvement = orchestration_result['action_result'].get('efficiency_improvement', '0%')
        print(f"   • Schedule efficiency improvement: {efficiency_improvement}")
        print(f"   • Urgent custom orders: {orchestration_result['action_result'].get('custom_order_analysis', {}).get('urgent_orders', 0)}")
        
        # Step 5: Compile final report
        print("\n📋 STEP 5: Compiling final report...")
        final_report = self._compile_final_report()
        
        print("\n✅ Simulation complete!")
        print("=" * 70)
        
        return final_report
    
    def _compile_final_report(self) -> Dict[str, Any]:
        """Compile comprehensive manufacturing report from all agent results."""
        report = {
            "report_title": "AutoParts Inc - Manufacturing Intelligence Report",
            "generated_at": datetime.datetime.now().isoformat(),
            "facility": self.simulation_data["facility"],
            "executive_summary": {
                "quality_status": self._determine_quality_status(),
                "maintenance_status": self._determine_maintenance_status(),
                "schedule_status": self._determine_schedule_status(),
                "total_machines": self.simulation_data["summary"]["total_machines"],
                "operational_machines": self.simulation_data["summary"]["operational_machines"],
                "average_efficiency": self.simulation_data["summary"]["average_efficiency"],
                "overall_health_score": self._calculate_overall_health_score()
            },
            "quality_analysis": self.agent_results.get("quality", {}).get("action_result", {}),
            "machine_health_analysis": self.agent_results.get("maintenance", {}).get("action_result", {}),
            "schedule_optimization": self.agent_results.get("orchestration", {}).get("action_result", {}),
            "action_items": self._generate_action_items(),
            "simulation_metadata": {
                "agents_used": [
                    self.quality_agent.name,
                    self.maintenance_agent.name,
                    self.orchestrator_agent.name
                ],
                "data_points_generated": self._count_data_points(),
                "simulation_duration": "Instant (simulated)"
            }
        }
        
        return report
    
    def _determine_quality_status(self) -> str:
        """Determine overall quality status based on agent analysis."""
        quality_result = self.agent_results.get("quality", {}).get("action_result", {})
        defect_rate = quality_result.get("overall_defect_rate", 0)
        
        if defect_rate > 15:
            return "CRITICAL - Immediate action required"
        elif defect_rate > 10:
            return "WARNING - Monitor closely"
        elif defect_rate > 5:
            return "ATTENTION - Review recommended"
        else:
            return "GOOD - Within acceptable range"
    
    def _determine_maintenance_status(self) -> str:
        """Determine overall maintenance status."""
        maintenance_result = self.agent_results.get("maintenance", {}).get("action_result", {})
        machine_analysis = maintenance_result.get("machine_analysis", [])
        
        critical_count = len([m for m in machine_analysis if m.get("urgency_level") == "CRITICAL"])
        high_count = len([m for m in machine_analysis if m.get("urgency_level") == "HIGH"])
        
        if critical_count > 0:
            return f"CRITICAL - {critical_count} machines require immediate maintenance"
        elif high_count > 0:
            return f"WARNING - {high_count} machines need preventive maintenance"
        else:
            return "HEALTHY - All machines operating normally"
    
    def _determine_schedule_status(self) -> str:
        """Determine production schedule status."""
        orchestration_result = self.agent_results.get("orchestration", {}).get("action_result", {})
        custom_analysis = orchestration_result.get("custom_order_analysis", {})
        
        urgent_orders = custom_analysis.get("urgent_orders", 0)
        efficiency_improvement = float(orchestration_result.get("efficiency_improvement", "0%").rstrip("%"))
        
        if urgent_orders > 0:
            return f"URGENT - {urgent_orders} custom orders approaching deadline"
        elif efficiency_improvement > 10:
            return "OPTIMIZATION AVAILABLE - Significant efficiency gains possible"
        else:
            return "STABLE - Schedule operating efficiently"
    
    def _calculate_overall_health_score(self) -> float:
        """Calculate overall factory health score (0-100)."""
        quality_result = self.agent_results.get("quality", {}).get("action_result", {})
        maintenance_result = self.agent_results.get("maintenance", {}).get("action_result", {})
        
        # Quality component (40%)
        defect_rate = quality_result.get("overall_defect_rate", 0)
        quality_score = max(0, 100 - defect_rate)
        
        # Maintenance component (30%)
        maintenance_summary = maintenance_result.get("summary", {})
        maintenance_score = maintenance_summary.get("overall_health_score", 100)
        
        # Efficiency component (30%)
        efficiency = self.simulation_data["summary"]["average_efficiency"]
        
        # Weighted average
        overall_score = (quality_score * 0.4) + (maintenance_score * 0.3) + (efficiency * 0.3)
        
        return round(overall_score, 1)
    
    def _generate_action_items(self) -> List[Dict[str, Any]]:
        """Generate prioritized action items from all agent recommendations."""
        action_items = []
        
        # Quality action items
        quality_result = self.agent_results.get("quality", {}).get("action_result", {})
        for rec in quality_result.get("recommendations", []):
            action_items.append({
                "category": "Quality",
                "priority": "HIGH" if "immediate" in rec.lower() else "MEDIUM",
                "action": rec,
                "assigned_to": "Quality Control Team",
                "deadline": "ASAP" if "immediate" in rec.lower() else "Within 7 days"
            })
        
        # Maintenance action items
        maintenance_result = self.agent_results.get("maintenance", {}).get("action_result", {})
        for machine in maintenance_result.get("machine_analysis", []):
            if machine.get("urgency_level") in ["HIGH", "CRITICAL"]:
                action_items.append({
                    "category": "Maintenance",
                    "priority": machine.get("urgency_level", "MEDIUM"),
                    "action": f"{machine.get('machine_name')}: {machine.get('recommended_action')}",
                    "assigned_to": "Maintenance Team",
                    "deadline": machine.get("maintenance_timeframe", "Within 7 days"),
                    "machine_id": machine.get("machine_id")
                })
        
        # Schedule optimization action items
        orchestration_result = self.agent_results.get("orchestration", {}).get("action_result", {})
        for rec in orchestration_result.get("recommendations", []):
            priority = "HIGH" if "urgent" in rec.lower() else "MEDIUM"
            action_items.append({
                "category": "Production Schedule",
                "priority": priority,
                "action": rec,
                "assigned_to": "Production Planning Team",
                "deadline": "Within 3 days" if priority == "HIGH" else "Within 14 days"
            })
        
        # Sort by priority
        priority_order = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
        action_items.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return action_items
    
    def _count_data_points(self) -> int:
        """Count total data points generated in simulation."""
        count = 0
        if self.simulation_data:
            count += len(self.simulation_data.get("machine_metrics", []))
            count += len(self.simulation_data.get("quality_metrics", []))
            count += len(self.simulation_data.get("production_schedule", []))
            count += len(self.simulation_data.get("custom_orders", []))
        return count
    
    def save_report_to_file(self, report: Dict[str, Any], filename: str = "manufacturing_report.json"):
        """Save simulation report to JSON file."""
        # Convert dataclasses to dictionaries
        def convert_dataclasses(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert_dataclasses(v) for k, v in asdict(obj).items() if not k.startswith('_')}
            elif isinstance(obj, list):
                return [convert_dataclasses(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_dataclasses(v) for k, v in obj.items()}
            else:
                return obj
        
        serializable_report = convert_dataclasses(report)
        
        with open(filename, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        print(f"\n💾 Report saved to '{filename}'")
        return filename
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the simulation results."""
        print("\n" + "=" * 70)
        print("SIMULATION SUMMARY - AutoParts Inc.")
        print("=" * 70)
        
        summary = report.get("executive_summary", {})
        print(f"\n🏭 Overall Factory Health Score: {summary.get('overall_health_score', 'N/A')}/100")
        print(f"📊 Operational Machines: {summary.get('operational_machines', 0)}/{summary.get('total_machines', 0)}")
        print(f"⚡ Average Efficiency: {summary.get('average_efficiency', 'N/A')}%")
        
        print(f"\n🔴 Quality Status: {summary.get('quality_status', 'N/A')}")
        print(f"🟡 Maintenance Status: {summary.get('maintenance_status', 'N/A')}")
        print(f"🟢 Schedule Status: {summary.get('schedule_status', 'N/A')}")
        
        action_items = report.get("action_items", [])
        print(f"\n📝 Action Items: {len(action_items)} total")
        
        critical_items = [a for a in action_items if a.get("priority") == "CRITICAL"]
        high_items = [a for a in action_items if a.get("priority") == "HIGH"]
        
        if critical_items:
            print("   ⚠️  CRITICAL Actions:")
            for item in critical_items[:3]:  # Show first 3 critical items
                print(f"     • {item.get('action', 'Unknown action')}")
        
        if high_items:
            print("   🔴 HIGH Priority Actions:")
            for item in high_items[:3]:  # Show first 3 high priority items
                print(f"     • {item.get('action', 'Unknown action')}")
        
        print("\n🤖 AI Agents Executed:")
        for agent in report.get("simulation_metadata", {}).get("agents_used", []):
            print(f"   • {agent}")
        
        print("=" * 70)

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------

def main():
    """
    Main execution function for the smart manufacturing simulation.
    This function demonstrates the complete AI agent system in action.
    """
    print("Initializing Smart Manufacturing AI Agent Simulation...")
    
    # Create simulation instance
    simulation = ManufacturingSimulation()
    
    try:
        # Run simulation cycle
        report = simulation.run_simulation_cycle()
        
        # Print human-readable summary
        simulation.print_summary(report)
        
        # Save detailed report to file
        report_filename = simulation.save_report_to_file(report)
        
        print(f"\n🎯 Simulation completed successfully!")
        print(f"📄 Detailed report saved to: {report_filename}")
        print(f"🤖 AI Agents deployed: 3")
        print(f"📊 Data points analyzed: {simulation._count_data_points()}")
        
        # Demonstrate agent capabilities
        print("\n" + "=" * 70)
        print("AI AGENT CAPABILITIES DEMONSTRATION")
        print("=" * 70)
        
        # Show sample agent output
        if simulation.agent_results.get("quality"):
            defect_rate = simulation.agent_results["quality"]["action_result"].get("overall_defect_rate", 0)
            print(f"📈 Quality Agent identified defect rate: {defect_rate:.2f}%")
        
        if simulation.agent_results.get("maintenance"):
            machines = simulation.agent_results["maintenance"]["action_result"].get("machine_analysis", [])
            critical = len([m for m in machines if m.get("urgency_level") == "CRITICAL"])
            print(f"🔧 Maintenance Agent flagged {critical} machines for immediate attention")
        
        if simulation.agent_results.get("orchestration"):
            improvement = simulation.agent_results["orchestration"]["action_result"].get("efficiency_improvement", "0%")
            print(f"📅 Orchestrator Agent identified {improvement} potential efficiency improvement")
        
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Simulation error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Run the simulation
    exit_code = main()
    exit(exit_code)