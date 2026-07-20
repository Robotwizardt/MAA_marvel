from collections.abc import Iterable
from dataclasses import dataclass
import re
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context, JRecognitionType
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR

from agent.strategies.ocr import CardCandidate, choose_card


ENERGY_ROI = (290, 1120, 140, 160)
CARD_COST_ROIS = (
    (0, 920, 180, 140),
    (180, 920, 180, 140),
    (360, 920, 180, 140),
    (540, 920, 180, 140),
)
HAND_BOXES = (
    (50, 1010, 80, 100),
    (230, 1010, 80, 100),
    (410, 1010, 80, 100),
    (590, 1010, 80, 100),
)
MINIMUM_CONFIDENCE = 0.80


@dataclass(frozen=True, slots=True)
class ParsedDigit:
    value: int | None
    confidence: float
    reason: str


def parse_digit_results(results: Iterable[Any]) -> ParsedDigit:
    valid: list[tuple[int, float]] = []
    for result in results:
        text = str(getattr(result, "text", "")).strip()
        if re.fullmatch(r"[0-7]", text):
            valid.append((int(text), float(getattr(result, "score", 0.0))))

    if not valid:
        return ParsedDigit(None, 0.0, "no_single_digit")
    values = {value for value, _ in valid}
    if len(values) != 1:
        return ParsedDigit(None, max(score for _, score in valid), "conflicting_digits")
    value = valid[0][0]
    return ParsedDigit(value, max(score for _, score in valid), "recognized")


def _results(detail: Any | None) -> list[Any]:
    if detail is None:
        return []
    return list(getattr(detail, "filtered_results", []))


@AgentServer.custom_recognition("MarvelCardSelection")
class CardSelection(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        energy_detail = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(expected=["^[0-7]$"], roi=ENERGY_ROI, threshold=0.80),
            argv.image,
        )
        if energy_detail is None:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={
                    "energy": None,
                    "candidates": [],
                    "selected_slot": None,
                    "reason": "energy_missing",
                },
            )

        energy = parse_digit_results(_results(energy_detail))
        if energy.value is None:
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={
                    "energy": None,
                    "candidates": [],
                    "selected_slot": None,
                    "reason": f"energy_{energy.reason}",
                },
            )

        candidates: list[CardCandidate] = []
        diagnostic_candidates: list[dict[str, object]] = []
        for slot, roi in enumerate(CARD_COST_ROIS):
            detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(expected=["^[0-7]$"], roi=roi, threshold=0.30),
                argv.image,
            )
            parsed = parse_digit_results(_results(detail))
            diagnostic_candidates.append(
                {
                    "slot": slot,
                    "cost": parsed.value,
                    "confidence": parsed.confidence,
                    "reason": parsed.reason,
                }
            )
            if parsed.value is not None:
                candidates.append(
                    CardCandidate(slot, parsed.value, parsed.confidence)
                )

        decision = choose_card(
            energy=energy.value,
            cards=candidates,
            minimum_confidence=MINIMUM_CONFIDENCE,
        )
        selected_slot = None if decision.card is None else decision.card.slot
        return CustomRecognition.AnalyzeResult(
            box=None if selected_slot is None else HAND_BOXES[selected_slot],
            detail={
                "energy": energy.value,
                "candidates": diagnostic_candidates,
                "selected_slot": selected_slot,
                "reason": decision.reason,
            },
        )
