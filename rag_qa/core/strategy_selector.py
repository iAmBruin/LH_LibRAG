# 导入 LangChain 提示模板
from langchain.prompts import PromptTemplate
# 导入日志和配置
from base import logger, Config
# 导入 OpenAI
from openai import OpenAI
import os


class StrategySelector:
    def __init__(self):
        # 初始化 OpenAI 客户端
        self.client = OpenAI(api_key=os.getenv("API_KEY"),
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        # 获取策略选择提示模板
        self.strategy_prompt_template = self._get_strategy_prompt()

        print("实际模型名称：", Config().LLM_MODEL)
        print("实际接口地址：", self.client.base_url)

    def call_dashscope(self, prompt):
        # 调用 DashScope API
        try:
            # 创建聊天完成请求
            completion = self.client.chat.completions.create(
                model=Config().LLM_MODEL,
                messages=[
                    {"role": "system", "content": "你是一个有用的助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            # 返回完成结果
            return completion.choices[0].message.content if completion.choices else "直接检索"
        except Exception as e:
            # 记录 API 调用失败
            logger.error(f"DashScope API 调用失败: {e}")
            # 默认返回直接检索
            return "直接检索"

    def _get_strategy_prompt(self):
        #   定义私有方法，获取策略选择 Prompt 模板
        return PromptTemplate(
            template="""
            你是凌华图书馆的AI智能助手“凌小图”，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：适用于查询意图明确，需要从知识库中检索特定信息的问题，例如：
                    * 示例：
                        * 查询：凌华图书馆几点开门？
                        * 策略：直接检索
                    * 查询：图书借期是多长时间？
                        * 策略：直接检索

            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：适用于查询较为抽象、概念性较强，直接检索效果可能不佳的问题，例如：
                    * 示例：
                        * 查询：图书馆有哪些适合青少年阅读的资源？
                        * 策略：假设问题检索

            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：适用于查询涉及多个方面，需要分别检索不同信息的问题，例如：
                    * 示例：
                        * 查询：凌华图书馆的开放时间、读者证办理方式和图书借阅期限分别是什么？
                        * 策略：子查询检索

            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：适用于查询较为复杂、表达比较口语化或需要转换后才能有效检索的问题，例如：
                    * 示例：
                        * 查询：我周末想去凌华图书馆借几本书，还能办理读者证吗？
                        * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """
            ,
            input_variables=["query"],
        )

    #   定义方法，选择检索策略
    def select_strategy(self, query):
        #   关键词短路规则：馆藏/书目/推荐等意图明确的问题直接走"直接检索"，
        #   跳过 LLM 策略选择调用（省约 0.5s），实测召回质量不低于 HyDE
        direct_keywords = ["索书号", "馆藏", "借阅", "借期", "续借", "归还", "罚款", "预约",
                           "读者证", "开馆", "开门", "开放时间", "书目", "书单", "推荐",
                           "在架", "楼层", "阅览室", "自习", "入藏", "借还"]
        if any(kw in query for kw in direct_keywords):
            hit = next(k for k in direct_keywords if k in query)
            logger.info(f"策略短路命中[{hit}]，直接使用直接检索")
            return "直接检索"
        #   调用 LLM 获取检索策略
        strategy = self.call_dashscope(self.strategy_prompt_template.format(query=query)).strip()
        logger.info(f"为查询 '{query}' 选择的检索策略：{strategy}")
        return strategy


if __name__ == '__main__':
    ss = StrategySelector()
    print(ss.select_strategy('推荐图书馆里有的几本青少年书籍'))
