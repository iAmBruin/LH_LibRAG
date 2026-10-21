#   导入 PromptTemplate 类，用于创建 Prompt 模板
from langchain.prompts import PromptTemplate


#   定义 RAGPrompts 类，用于管理所有 Prompt 模板
class RAGPrompts:
    #   定义 RAG 提示模板
    # @staticmethod
    # def rag_prompt():
    #     #   创建并返回 PromptTemplate 对象
    #     return PromptTemplate(
    #         template="""
    #         你是一个智能助手，帮助用户回答问题。
    #         如果提供了上下文，请基于上下文回答；如果没有上下文，请直接根据你的知识回答。
    #         如果答案来源于检索到的文档，请在回答中说明。
    #
    #         上下文: {context}
    #         问题: {question}
    #
    #         如果无法回答，请回复：“信息不足，无法回答，请联系人工客服，电话：{phone}。”
    #         回答:
    #         """,
    #         #   定义输入变量
    #         input_variables=["context", "question", "phone"],
    #     )
    # # 优化后再添加：如果添加了历史消息
    @staticmethod
    def rag_prompt():
        return PromptTemplate(
            template="""
        你是凌华图书馆的AI智能助手“凌小图”，负责帮助读者解答图书馆相关问题。请按照以下步骤处理：

        1. **分析问题和上下文**：
           - 优先根据提供的上下文回答问题。
           - 如果答案来源于检索到的图书馆文档，请在回答中明确说明，例如：“根据提供的资料，……”。
           - 不要编造图书馆不存在的规定、服务或馆藏信息。

        2. **评估对话历史**：
           - 检查对话历史是否与当前问题相关（例如，是否涉及相同的图书、借阅规则、读者证、馆内服务或其他问题背景）。
           - 如果对话历史与当前问题相关，请结合历史信息生成更准确的回答。
           - 如果对话历史无关（例如，仅包含问候或不相关的内容），忽略历史，仅基于上下文和当前问题回答。

        3. **生成回答**：
           - 提供清晰、准确、简洁的回答，优先回答读者最关心的信息。
           - 涉及时间、地点、费用、借阅期限、续借次数等具体信息时，应以检索到的资料为准。
           - 如果上下文和历史消息均不足以回答问题，请回复：“抱歉，我暂时无法从现有资料中找到可靠答案，建议您前往凌华图书馆服务台咨询工作人员。”

        **上下文**: {context}
        **对话历史**:
        {history}
        **问题**: {question}

        **回答**:
        """,
            input_variables=["context", "history", "question"],
        )

    #   定义假设问题生成的 Prompt 模板
    @staticmethod
    def hyde_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            假设你是用户，想了解以下问题，请生成一个简短的假设答案：  
            问题: {query}  
            假设答案:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义子查询生成的 Prompt 模板
    @staticmethod
    def subquery_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询分解为多个简单子查询，每行一个子查询：  
            查询: {query}  
            子查询:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )

    #   定义回溯问题生成的 Prompt 模板
    @staticmethod
    def backtracking_prompt():
        #   创建并返回 PromptTemplate 对象
        return PromptTemplate(
            template="""  
            将以下复杂查询简化为一个更简单的问题：  
            查询: {query}  
            简化问题:  
            """,
            #   定义输入变量
            input_variables=["query"],
        )