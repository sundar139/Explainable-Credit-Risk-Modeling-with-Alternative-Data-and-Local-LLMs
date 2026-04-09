"""Explainability workflows package."""

from credit_risk_altdata.explainability import workflow

ExplainabilityWorkflowResult = workflow.ExplainabilityWorkflowResult
run_explainability_workflow = workflow.run_explainability_workflow

__all__ = ["ExplainabilityWorkflowResult", "run_explainability_workflow"]
