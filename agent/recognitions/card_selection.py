from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition


@AgentServer.custom_recognition("MarvelCardSelection")
class CardSelection(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        return CustomRecognition.AnalyzeResult(
            box=None,
            detail={"reason": "ocr_adapter_not_configured"},
        )
