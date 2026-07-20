from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition

from agent.runtime.commands import parse_json_object
from agent.session.config import ConquestTier


@AgentServer.custom_recognition("MarvelSafeEntry")
class SafeEntry(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        values = parse_json_object(argv.custom_recognition_param)
        tier = ConquestTier(str(values.get("tier", "")))
        return CustomRecognition.AnalyzeResult(
            box=None,
            detail={"tier": tier.value, "reason": "entry_evidence_not_collected"},
        )
