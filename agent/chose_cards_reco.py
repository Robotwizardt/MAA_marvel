from abc import abstractmethod
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context, JRecognitionType
from maa.pipeline import JOCR  
import re

@AgentServer.custom_recognition("chose_cards_reco")
class MyRecognition(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        print(f"开始分析图像: {argv.image}")
        # 1. 识别当前可用费用
        cost_result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(
                roi=[304, 1141, 109, 111],
                expected="^[0-7]$"
            ),
            argv.image
        )
        
        current_cost = self._extract_cost(cost_result.best_result.text)
        print(f"当前可用费用: {current_cost}")
        
        if current_cost is None or current_cost <= 0:
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 0, 0), 
                detail="No cost available"
            )

        # 2. 识别手牌区域中所有卡牌的费用（整体识别）
        card_cost_result = context.run_recognition_direct(
            JRecognitionType.OCR,
            JOCR(
                roi=[14, 955, 697, 67],  # 手牌区域
                expected="^[0-7]$"  # 识别所有文本
            ),
            argv.image
        )
        
        # 3. 从整体识别结果中提取所有数字
        # 应该遍历 filtered_results 提取每个结果的 text  
        all_numbers = []  
        for result in card_cost_result.filtered_results:  
            if hasattr(result, 'text'):  
                numbers = re.findall(r'(\d+)', result.text)  
                all_numbers.extend([int(num) for num in numbers])

        print(f"识别到的所有数字: {all_numbers}")
        
        # 提取偶数索引的数字作为费用（索引0,2,4...是费用）
        # 例如: [3, 5, 4, 2] -> 索引0是费用3，索引1是战力5，索引2是费用4，索引3是战力2
        card_costs = []
        for i in range(0, len(all_numbers), 2):  # 从索引0开始，步长2
            card_costs.append(card_cost_result.filtered_results[i])  # 获取对应的OCR结果对象
        
        print(f"提取的卡牌费用: {card_costs}")
        
        # 4. 过滤出可用费用范围内的卡牌
        eligible_cards = []
        
        for idx, cost in enumerate(card_costs):
            if cost is not None and self._extract_cost(cost.text) <= current_cost:
                eligible_cards.append({
                    "index": idx,
                    "cost": cost.text, # 卡牌的费用
                    "roi": cost.box, # 记录卡牌费用的ROI
                    "position": self._get_card_center(cost.box)
                })
        
        # 5. 选择费用最高的卡牌
        if not eligible_cards:
            return CustomRecognition.AnalyzeResult(
                box=(0, 0, 0, 0),
                detail="No eligible cards"
            )
        
        # 按费用降序排序
        best_card = max(eligible_cards, key=lambda x: x["cost"])
        print(f"选择卡牌 {best_card['index']+1}，费用: {best_card['cost']}")
        
        # 6. 点击选中的卡牌
        # click_x, click_y = best_card["position"]
        # click_job = context.tasker.controller.post_click(click_x, click_y)
        # click_job.wait()
        # print(f"已点击卡牌位置: ({click_x}, {click_y})")
        
        # 7. 返回识别结果
        return CustomRecognition.AnalyzeResult(
            box=best_card["position"],
            detail=f"Play card with cost {best_card['cost']}"
        )
    
    
    
    def _get_card_center(self, roi: list) -> tuple:
        """
        计算卡牌中心点坐标
        """
        x, y, width, height = roi
        return (x + width // 2, y + height // 2,width, height)
    
    def _extract_cost(self, text: str) -> int:
        """从OCR识别的文本中提取数字"""
        if not text:
            return None
        
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        return None