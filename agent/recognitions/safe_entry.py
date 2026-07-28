from collections.abc import Iterable
import re

from maa.agent.agent_server import AgentServer
from maa.context import Context, JRecognitionType
from maa.custom_recognition import CustomRecognition
from maa.pipeline import JOCR

from agent.conquest.tier_policy import EntryEvidence, is_safe_entry
from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE
from agent.session.config import ConquestTier


TICKET_COUNT_ROI = (170, 900, 380, 250)


def parse_ticket_count(texts: Iterable[str]) -> int | None:
    """从“6/1”或“已拥有0/1”等 OCR 文本中提取当前拥有数量。"""
    counts: list[int] = []
    for text in texts:
        match = re.search(r"(?:已拥有)?\s*(\d+)\s*/\s*1", str(text))
        if match:
            counts.append(int(match.group(1)))
    # 同屏若出现冲突数字则拒绝猜测，交给下一级回退。
    return counts[0] if counts and len(set(counts)) == 1 else None


def reserve_for_tier(tier: ConquestTier) -> int:
    config = STORE.require_state().config
    return {
        ConquestTier.SILVER: config.reserve_silver_tickets,
        ConquestTier.GOLD: config.reserve_gold_tickets,
        ConquestTier.INFINITE: config.reserve_infinite_tickets,
    }.get(tier, 0)


@AgentServer.custom_recognition("MarvelSafeEntry")
class SafeEntry(CustomRecognition):
    """组合多个 Pipeline 识别节点，判断当前入口是否绝对安全。"""
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        values = parse_json_object(argv.custom_recognition_param)
        tier_value = str(values.get("tier", ""))
        if tier_value == "current":
            tier = STORE.current_tier()
            if tier is None:
                return CustomRecognition.AnalyzeResult(
                    box=None,
                    detail={"safe": False, "reason": "no_current_tier"},
                )
        else:
            tier = ConquestTier(tier_value)

        def matched(entry: str) -> bool:
            # 复用 Pipeline 中已有 OCR/模板节点，避免在 Python 重复维护 ROI。
            result = context.run_recognition(entry, argv.image)
            return result is not None and result.box is not None

        ticket_count = None
        if tier is not ConquestTier.PROVING_GROUNDS:
            detail = context.run_recognition_direct(
                JRecognitionType.OCR,
                JOCR(roi=TICKET_COUNT_ROI, threshold=0.30, order_by="Horizontal"),
                argv.image,
            )
            results = [] if detail is None else getattr(detail, "filtered_results", [])
            ticket_count = parse_ticket_count(
                getattr(result, "text", "") for result in results
            )

        evidence = EntryEvidence(
            tier=tier,
            free_label=matched("征服-证据-免费进入"),
            ticket_count=ticket_count,
            reserve_count=reserve_for_tier(tier),
            gold_icon=matched("征服-证据-金块图标"),
            gold_amount=matched("征服-证据-金块金额"),
            paid_confirmation=matched("征服-证据-付费确认"),
        )
        safe = is_safe_entry(evidence)
        # box 不为空表示 CustomRecognition 命中；这里使用全屏框仅作为布尔信号。
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 720, 1280) if safe else None,
            detail={
                "tier": tier.value,
                "safe": safe,
                "free_label": evidence.free_label,
                "ticket_count": evidence.ticket_count,
                "reserve_count": evidence.reserve_count,
                "gold_icon": evidence.gold_icon,
                "gold_amount": evidence.gold_amount,
                "paid_confirmation": evidence.paid_confirmation,
            },
        )
