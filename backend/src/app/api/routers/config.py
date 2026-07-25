import logging

from app.models.helpers.compute_stats import (
    SHAPE_METRICS,
    SHAPE_OPERATORS,
    ShapeClassificationConfig,
    ShapeCondition,
    ShapeMetric,
    ShapeOperator,
    ShapeRule,
    dump_shape_classification_config,
    load_shape_classification_config,
)
from app.models.helpers.settings import (
    app_support_shape_config_path,
    bundled_shape_config_path,
    settings,
)
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/config")
logger = logging.getLogger("routes.config")


class ShapeConditionIn(BaseModel):
    metric: ShapeMetric
    op: ShapeOperator
    value: float


class ShapeRuleIn(BaseModel):
    label: str = Field(min_length=1)
    conditions: list[ShapeConditionIn] = Field(min_length=1)


class ShapeRulesUpdate(BaseModel):
    rules: list[ShapeRuleIn]


def _config_response(config: ShapeClassificationConfig, is_default: bool) -> dict:
    return {
        "default_shape": config.default_shape,
        "rules": [
            {
                "label": rule.label,
                "conditions": [
                    {"metric": c.metric, "op": c.op, "value": c.value}
                    for c in rule.conditions
                ],
            }
            for rule in config.rules
        ],
        "is_default": is_default,
        "available_metrics": list(SHAPE_METRICS),
        "available_operators": list(SHAPE_OPERATORS),
    }


@router.get("/shape-rules")
async def get_shape_rules():
    """actively override if one exists, else the bundled default."""
    config = load_shape_classification_config(settings.SHAPE_CONFIG_PATH)
    is_default = not app_support_shape_config_path().exists()
    return _config_response(config, is_default)


@router.put("/shape-rules")
async def update_shape_rules(req: ShapeRulesUpdate):
    """
    Overwrite the user's shape rules. default_shape is left untouched 
    and only the ordered rule list is editable from the client. Written to the app-support
    override path; the bundled shape_config.toml is never modified.
    """
    current = load_shape_classification_config(settings.SHAPE_CONFIG_PATH)
    updated = ShapeClassificationConfig(
        default_shape=current.default_shape,
        rules=[
            ShapeRule(
                label=rule.label,
                conditions=[
                    ShapeCondition(metric=c.metric, op=c.op, value=c.value)
                    for c in rule.conditions
                ],
            )
            for rule in req.rules
        ],
    )
    override_path = app_support_shape_config_path()
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(dump_shape_classification_config(updated))
    logger.info(f"Saved {len(updated.rules)} shape classification rule(s)")
    return _config_response(updated, is_default=False)


@router.post("/shape-rules/reset")
async def reset_shape_rules():
    """Delete the user override, reverting to the bundled default rules."""
    app_support_shape_config_path().unlink(missing_ok=True)
    config = load_shape_classification_config(bundled_shape_config_path())
    logger.info("Reset shape classification rules to default")
    return _config_response(config, is_default=True)
