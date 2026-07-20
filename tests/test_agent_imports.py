import importlib
import unittest

from maa.agent.agent_server import AgentServer


class AgentImportTests(unittest.TestCase):
    def test_main_registers_expected_adapters_without_starting_socket(self) -> None:
        importlib.import_module("agent.main")

        self.assertTrue(
            {
                "MarvelConfigureSession",
                "MarvelPlayTurn",
                "MarvelRecordEvent",
                "MarvelRouteConquestTier",
                "MarvelRecoveryAction",
            }.issubset(AgentServer._custom_action_holder)
        )
        self.assertTrue(
            {
                "MarvelSessionGate",
                "MarvelCardSelection",
                "MarvelSafeEntry",
            }.issubset(AgentServer._custom_recognition_holder)
        )


if __name__ == "__main__":
    unittest.main()
