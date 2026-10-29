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
    app_support_shape_configs_dir,
    app_support_yolo_config_path,
    app_support_ui_preferences_path,
    bundled_shape_config_path,
    settings,
)
from fastapi import APIRouter, HTTPException
import json
import re
from pathlib import Path
from pydantic import BaseModel, Field

from app.logutils import get_logger

router = APIRouter(prefix="/config")
logger = get_logger("config")


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


def _rules_from_request(req: ShapeRulesUpdate) -> ShapeClassificationConfig:
    """Build a config from the editable rule list, keeping the active default_shape."""
    current = load_shape_classification_config(settings.SHAPE_CONFIG_PATH)
    return ShapeClassificationConfig(
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


_PRESET_NAME_RE = re.compile(r"^[^/\\\x00-\x1f]+$")


def _preset_path(name: str) -> Path:
    if not name.strip() or name in (".", "..") or not _PRESET_NAME_RE.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail="Preset names can't be empty or contain path separators",
        )
    return app_support_shape_configs_dir() / f"{name.strip()}.toml"


@router.get("/shape-rules/presets")
async def list_shape_rule_presets():
    """Names of saved presets, alphabetical."""
    presets_dir = app_support_shape_configs_dir()
    if not presets_dir.exists():
        return {"presets": []}
    return {"presets": sorted(p.stem for p in presets_dir.glob("*.toml"))}


@router.post("/shape-rules/presets/{name}")
async def save_shape_rule_preset(name: str, req: ShapeRulesUpdate):
    """Store the given rules under a name; active rules are left untouched."""
    config = _rules_from_request(req)
    path = _preset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_shape_classification_config(config))
    logger.info(f"Saved shape-rule preset {name!r}")
    return _config_response(config, is_default=False)


@router.post("/shape-rules/presets/{name}/load")
async def load_shape_rule_preset(name: str):
    """Copy a named preset into the active override slot."""
    preset_path = _preset_path(name)
    if not preset_path.exists():
        raise HTTPException(
            status_code=404, detail=f"No shape-rule preset named {name!r}"
        )
    config = load_shape_classification_config(preset_path)
    override_path = app_support_shape_config_path()
    override_path.parent.mkdir(parents=True, exist_ok=True)
    override_path.write_text(dump_shape_classification_config(config))
    logger.info(f"Loaded shape-rule preset {name!r}")
    return _config_response(config, is_default=False)


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
    updated = _rules_from_request(req)
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


# shipped out of the box values. Seeded into yolo_config.json once and never
# changed after that, so a user can always go back to them.
YOLO_SHIPPED_DEFAULTS = {"conf": 0.01, "iou": 0.5, "max_det": 4000}


class YoloOverridesIn(BaseModel):
    # user overrides, one per tunable; null means use the stored default
    conf: float | None = Field(None, ge=0.0, le=1.0)
    iou: float | None = Field(None, ge=0.0, le=1.0)
    max_det: int | None = Field(None, ge=1)


class UIPreferences(BaseModel):
    hide_comparison_reset_warning: bool = False


@router.get("/ui-preferences", response_model=UIPreferences)
async def get_ui_preferences() -> UIPreferences:
    path = app_support_ui_preferences_path()
    if not path.exists():
        return UIPreferences()
    return UIPreferences.model_validate_json(path.read_text())


@router.put("/ui-preferences", response_model=UIPreferences)
async def update_ui_preferences(req: UIPreferences) -> UIPreferences:
    path = app_support_ui_preferences_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(req.model_dump_json(indent=2))
    logger.info("Saved UI preferences")
    return req


def _load_yolo_config() -> dict:
    """File shape: {field: {"default": x, "override": x|null}}.

    The default is seeded from the shipped value on first use and is immutable
    after that. The override is the only field the app writes.
    """
    try:
        data = json.loads(app_support_yolo_config_path().read_text())
    except (OSError, ValueError):
        data = None
    config = {}
    for field, shipped_default in YOLO_SHIPPED_DEFAULTS.items():
        entry = data.get(field) if isinstance(data, dict) else None
        entry = entry if isinstance(entry, dict) else {}
        default = entry.get("default")
        override = entry.get("override")
        config[field] = {
            "default": default
            if isinstance(default, (int, float))
            else shipped_default,
            "override": override if isinstance(override, (int, float)) else None,
        }
    return config


def _write_yolo_config(config: dict) -> None:
    path = app_support_yolo_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


@router.get("/yolo")
async def get_yolo_settings():
    """{field: {default, override}} for the configure modal. The effective
    value is the override when set, else the default."""
    config = _load_yolo_config()
    if not app_support_yolo_config_path().exists():
        _write_yolo_config(config)  # seed so the shipped defaults stay on disk
    return config


@router.put("/yolo")
async def update_yolo_settings(req: YoloOverridesIn):
    """Replace the user overrides. Stored defaults are immutable."""
    config = _load_yolo_config()
    for field, override in req.model_dump().items():
        config[field]["override"] = override
    _write_yolo_config(config)
    logger.info(f"Saved yolo detection overrides: {req.model_dump()}")
    return config
