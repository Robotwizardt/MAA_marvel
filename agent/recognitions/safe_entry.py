from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from agent.conquest.tier_policy import EntryEvidence, is_safe_entry
from agent.runtime.commands import parse_json_object
from agent.runtime.store import STORE
from agent.session.config import ConquestTier


@AgentServer.custom_recognition("MarvelSafeEntry")
class SafeEntry(CustomRecognition):
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
            return context.run_recognition(entry, argv.image) is not None

        evidence = EntryEvidence(
            tier=tier,
            free_label=matched("征服-证据-免费进入"),
            ticket_label=matched("征服-证据-门票可用"),
            gold_icon=matched("征服-证据-金块图标"),
            gold_amount=matched("征服-证据-金块金额"),
            paid_confirmation=matched("征服-证据-付费确认"),
        )
        safe = is_safe_entry(evidence)
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 720, 1280) if safe else None,
            detail={
                "tier": tier.value,
                "safe": safe,
                "free_label": evidence.free_label,
                "ticket_label": evidence.ticket_label,
                "gold_icon": evidence.gold_icon,
                "gold_amount": evidence.gold_amount,
                "paid_confirmation": evidence.paid_confirmation,
            },
        )
