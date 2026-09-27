"""Distance/low-PANNs rescue policy for the AST trial.

Some real distant/reproduced gunshots are acoustically impulsive but PANNs can
assign almost all probability to Music/Wind chime/Speech.  The base AST trial
skipped AST in that situation, so the independent semantic model never got a
chance to recover the event.

This module keeps the normal PANNs+AST consensus unchanged and adds one very
strict rescue path.  AST is allowed to rescue a low-PANNs event only when there
is still a tiny non-zero firearm trace and AST strongly, cleanly prefers the
real-gunshot bucket over every nuisance/explosive/other bucket.
"""
from __future__ import annotations

from audio.gunshot_detector_runtime import FusionDecision
from audio.gunshot_detector_ast import (
    AST_NAMES,
    GunshotDetector as _BaseGunshotDetector,
    PANNsASTFusion as _BasePANNsASTFusion,
)


class PANNsASTFusion(_BasePANNsASTFusion):
    # Field log 2026-08-27: real shots reached only 0.001-0.004 PANNs firearm.
    AST_RESCUE_PANNS_FLOOR = 0.0005
    AST_RESCUE_GUNSHOT = 0.52
    AST_RESCUE_MARGIN = 0.18
    AST_RESCUE_NUISANCE_MAX = 0.28
    AST_RESCUE_EXPLOSIVE_MAX = 0.30

    def decide_ast_rescue(self, firearm, explosive, nuisance, ast):
        margin = self.margin(firearm, explosive, nuisance)
        if float(firearm) < self.AST_RESCUE_PANNS_FLOOR:
            return FusionDecision(False, "ast_rescue_panns_floor", margin)

        gun = ast.gunshot
        if gun < max(self.AST_RESCUE_GUNSHOT, self.fa + 0.08):
            return FusionDecision(False, "ast_rescue_threshold", margin)
        if ast.top_class != "gunshot":
            return FusionDecision(False, f"ast_rescue_{ast.top_class}", margin)

        strongest_other = max(
            ast.score(name) for name in AST_NAMES if name != "gunshot"
        )
        required_margin = max(self.AST_RESCUE_MARGIN, self.fam + 0.04)
        if gun < strongest_other + required_margin:
            return FusionDecision(False, "ast_rescue_margin", margin)

        # Explicit hard-negative vetoes remain stronger than the rescue.
        if ast.nuisance >= min(self.AST_RESCUE_NUISANCE_MAX, gun - 0.14):
            return FusionDecision(False, "ast_rescue_nuisance", margin)
        if ast.firecracker >= min(self.AST_RESCUE_EXPLOSIVE_MAX, gun - 0.14):
            return FusionDecision(False, "ast_rescue_firecracker", margin)

        return FusionDecision(True, "accepted_ast_strong_rescue", margin)


class GunshotDetector(_BaseGunshotDetector):
    """Base detector plus strict AST rescue for PANNs false negatives."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve thresholds/config prepared by the base constructor while
        # upgrading only the decision policy.
        base = self.fusion
        upgraded = PANNsASTFusion(
            panns_threshold=base.p,
            panns_margin_threshold=base.pm,
            ast_threshold=base.a,
            ast_veto_threshold=base.av,
            ast_veto_margin=base.avm,
            ast_margin_threshold=base.am,
            far_panns_threshold=base.fp,
            far_panns_margin=base.fpm,
            far_ast_threshold=base.fa,
            far_ast_margin=base.fam,
        )
        self.fusion = upgraded

    def _evaluate(self, panns_audio, ast_audio, allow_far=True):
        pf, pe, pn, ptop, _ = self.panns.check(
            panns_audio, self.sample_rate
        )
        strong = self.fusion.strong_candidate(pf, pe, pn)
        far = self.fusion.far_candidate(pf, pe, pn)

        # Normal paths are unchanged.
        if strong.accepted or (allow_far and far.accepted):
            ast = self.ast.check(ast_audio, self.ast_rate)
            decision = (
                self.fusion.decide(pf, pe, pn, ast)
                if strong.accepted
                else self.fusion.decide_far(pf, pe, pn, ast)
            )
            return pf, pe, pn, ptop, ast, decision

        # Important: this is only the full-window/far-gate path.  The short
        # peak rescue remains two-model consensus and cannot become AST-only.
        if not allow_far or pf < self.fusion.AST_RESCUE_PANNS_FLOOR:
            return pf, pe, pn, ptop, None, strong

        ast = self.ast.check(ast_audio, self.ast_rate)
        decision = self.fusion.decide_ast_rescue(pf, pe, pn, ast)
        return pf, pe, pn, ptop, ast, decision
