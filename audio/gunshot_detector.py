"""Public gunshot detector.

Production gunshot inference is PANNs + AST. Legacy YAMNet-named exports remain
only so existing UI/config/tests can start unchanged during the AST trial.
"""
from audio.gunshot_detector_runtime import (
    EnergyGate,
    FusionDecision,
    FusionResult,
    GunshotFusion,
    TemporalEvidenceResult,
    TemporalEvidenceTracker,
)
from audio.gunshot_detector_ast import (
    ASTClassifier,
    ASTResult,
    AST_GROUPS,
    PANNsClassifier,
)
from audio.gunshot_detector_ast_rescue import (
    GunshotDetector,
    PANNsASTFusion,
)

# Compatibility names. They all point to AST implementations; YAMNet is not
# instantiated by the production GunshotDetector.
PANNsYAMNetFusion = PANNsASTFusion
YAMNetClassifier = ASTClassifier


class YAMNetVeto:
    VETO_CLASSES = (
        AST_GROUPS["clap"]
        | AST_GROUPS["click"]
        | AST_GROUPS["metal_impact"]
        | AST_GROUPS["door_slam"]
    )
    GUN_CLASSES = set(AST_GROUPS["gunshot"])


__all__ = [
    "ASTClassifier",
    "ASTResult",
    "AST_GROUPS",
    "EnergyGate",
    "FusionDecision",
    "FusionResult",
    "GunshotDetector",
    "GunshotFusion",
    "PANNsASTFusion",
    "PANNsClassifier",
    "PANNsYAMNetFusion",
    "TemporalEvidenceResult",
    "TemporalEvidenceTracker",
    "YAMNetClassifier",
    "YAMNetVeto",
]
