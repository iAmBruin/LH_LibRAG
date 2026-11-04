# 凌华图书馆 RAG 智能问答系统 · 凌小图

**LH_LibRAG** 是一款面向图书馆业务场景的 RAG 智能问答系统，融合 FAQ 精确问答、关键词检索、向量检索与大语言模型生成，为读者提供图书馆业务咨询与知识问答服务。

## 项目展示

<img width="3196" height="2160" alt="Image" src="https://github.com/user-attachments/assets/8ed8a7e8-8f30-48e3-8dc4-18cedc851a8c" />

### 项目演示

[<video src="https://github.com/user-attachments/assets/7699d5c9-e595-4be3-a8a7-6a222d1f51e5" controls></video>](https://github.com/user-attachments/assets/ac3fe689-6cae-428f-a439-204cb309b080)

![无法播放？点击查看演示视频](https://github.com/user-attachments/assets/ac3fe689-6cae-428f-a439-204cb309b080)

## 问答流程

<img width="1200" height="3520" alt="Image" src="https://github.com/user-attachments/assets/2545b45d-6105-476e-9647-c86e4c15c024" />

## 技术架构

```text
┌────────────────────────────────────────────┐
│              FastAPI 接口层                 │
│       RESTful API / WebSocket              │
├────────────────────────────────────────────┤
│          IntegratedQASystem 集成层          │
│    FAQ 检索 / 意图分类 / RAG 流程编排        │
├────────────────────────────────────────────┤
│                  数据层                     │
│       MySQL       Redis       Milvus        │
└────────────────────────────────────────────┘
```

## 技术栈

| 模块 | 技术选型 | 主要用途 |
|---|---|---|
| 开发语言 | Python | 后端及 RAG 业务开发 |
| 后端框架 | FastAPI、Uvicorn | 提供 RESTful API 与 WebSocket 服务 |
| 向量数据库 | Milvus | 存储文档向量，支撑稠密 + 稀疏混合检索 |
| 嵌入模型 | BGE-M3 | 将文本编码为稠密 / 稀疏向量 |
| 重排模型 | BGE-Reranker | 对召回候选内容做相关性重排 |
| 关键词检索 | rank_bm25、jieba | 中文分词与 BM25 检索 |
| 意图分类 | BERT 中文分类器 | 用户问题意图识别与路由 |
| 大语言模型 | 通义千问 qwen-plus | 基于检索上下文生成回答 |
| 模型调用 | DashScope、OpenAI SDK | 模型服务调用与流式生成 |
| 数据存储 | MySQL、Redis | FAQ、会话历史与缓存管理 |
| 深度学习 | PyTorch | 本地模型加载与推理 |
| 前端 | HTML、JavaScript | Web 问答交互界面 |

## 检索增强策略

针对不同类型的问题，系统支持多种检索增强策略，改善用户提问与知识库内容之间的语义匹配效果。

| 检索策略 | 核心思路 | 适用场景 |
|---|---|---|
| 直接检索 | 使用原始问题进行检索 | 问题明确、关键词具体 |
| HyDE 假设问题检索 | 生成假设性文本，辅助向量检索 | 提问抽象、表达与知识库差异较大 |
| 子查询检索 | 将复杂问题拆分为多个子问题，分别检索后汇总 | 一个问题包含多个查询目标 |
| 回溯问题检索 | 对原问题进行改写、归纳或分解后检索 | 问题复杂、口语化或直接检索效果不佳 |

读完啦~ 如有问题，欢迎交流与指正 👋

