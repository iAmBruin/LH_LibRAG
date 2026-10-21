# LH_LibRAG — 凌华图书馆 RAG 智能问答系统

> 凌华图书馆 AI 智能助手「凌小图」：基于 **RAG（检索增强生成）** 架构，结合 MySQL 结构化知识库与 Milvus 向量库，提供图书馆问答、馆藏查询、借阅咨询等智能化服务。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 接口](#api-接口)
- [GPU 加速](#gpu-加速)
- [常见问题](#常见问题)

---

## 项目简介

本项目是面向「凌华图书馆」场景的智能问答系统。用户提出问题时，系统按照**多级路由**策略处理：

1. **问候语识别** → 返回模板回复；
2. **MySQL/BM25 精确匹配** → 命中馆藏知识库直接返回答案；
3. **RAG 检索增强生成** → 向量检索 + 大模型流式生成最终答案。

系统采用**混合检索**（稠密向量 + 稀疏关键词）并经过 **BGE-Reranker 重排**，显著提升答案的准确率与相关性。

## 功能特性

- 🗂️ **双层知识源**：MySQL 结构化 QA 库（BM25 精确检索） + Milvus 向量库（文档语义检索）
- 🧠 **混合检索**：BGE-M3 稠密向量 + 稀疏关键词 双路召回，BGE-Reranker 精排
- 🎯 **意图分类**：BERT 模型自动识别查询类别，支持按馆藏信息 / 借阅规则 / 读者证 / 馆内服务 / 相关规定等来源过滤
- ⚡ **流式输出**：FastAPI + WebSocket 逐 token 流式返回大模型答案
- 💬 **多轮对话**：MySQL 持久化会话历史，保留最近 5 轮上下文
- 🌐 **Web 前端**：内置静态页面，开箱即用
- 🚀 **GPU 加速**：嵌入与重排模型支持 NVIDIA CUDA 加速（本机 RTX 5060 已验证）

## 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| 语言 | Python 3.11 |
| 关系数据库 | MySQL（知识库、会话历史） |
| 缓存 | Redis |
| 向量数据库 | Milvus 2.x（集合 `proj_rag`） |
| 嵌入模型 | BGE-M3（本地 `rag_qa/models/bge-m3`） |
| 重排模型 | BGE-Reranker-large（本地 `rag_qa/models/bge-reranker-large`） |
| 意图分类 | BERT 中文分类器（本地 `rag_qa/core/bert_query_classifier`） |
| 大模型 | 通义千问 qwen-plus（DashScope OpenAI 兼容接口） |
| 检索算法 | BM25（rank_bm25 + jieba）、Milvus HybridSearch |
| 深度学习 | PyTorch 2.11.0+cu128（CUDA 12.8） |

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                        前端 (static/)                          │
└───────────────┬───────────────────────────────┬───────────────┘
                │ HTTP /api/query                │ WebSocket /api/stream
┌───────────────▼───────────────────────────────▼───────────────┐
│                       app.py (FastAPI)                          │
└───────────────┬────────────────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────────┐
│              new_main.py — IntegratedQASystem                   │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│  │ 问候语正则匹配   │──│ 检查 GREETING_PATTERNS → 模板回复      │ │
│  └─────────────────┘  └─────────────────────────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│  │ BM25Search      │──│ MySQL 精确匹配 (threshold=0.85)      │ │
│  └─────────────────┘  └─────────────────────────────────────┘ │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│  │ RAGSystem       │──│ 意图分类 → 混合检索 → 重排 → LLM 流式  │ │
│  └─────────────────┘  └─────────────────────────────────────┘ │
└───────────────┬───────────────────────────────────────────────┘
                │
   ┌────────────┼────────────┐
   ▼            ▼            ▼
  MySQL        Redis       Milvus
(知识库/会话)  (缓存)   (proj_rag 向量集合)
```

**查询处理流程**：

1. 问候语正则匹配 → 直接返回模板回复（如「你好」「再见」等 8 类）；
2. `BM25Search.search(query, threshold=0.85)` 在 MySQL QA 库中精确检索；
   - 命中 → 直接返回答案（非流式）；
   - 未命中 → 进入 RAG 流程；
3. RAG 流程：
   - `QueryClassifier`（BERT）识别查询意图类别；
   - `VectorStore.hybrid_search_with_rerank` 在 Milvus 中执行稠密 + 稀疏混合检索，`k=3` 召回；
   - `BGE-Reranker` 对候选重排，取 `candidate_m=2` 个最优父文档；
   - 组装 prompt 调用 DashScope（qwen-plus）**流式**生成答案；
4. 答案与问题写入 MySQL `conversations` 表（每会话保留最近 5 轮）。

## 目录结构

```
LH_LibRAG/
├── app.py                      # FastAPI 入口（REST + WebSocket + 静态页面），端口 8003
├── new_main.py                 # IntegratedQASystem 集成类 / 命令行交互入口
├── config.ini                  # 全局配置文件（MySQL/Redis/Milvus/LLM/检索参数）
├── requirements.txt            # 依赖清单（含 GPU 版 PyTorch 说明）
├── base/
│   ├── config.py               # Config 类：读取 config.ini
│   └── logger.py               # 日志配置（输出到 logs/app.log）
├── mysql_qa/
│   ├── db/mysql_client.py      # MySQL 客户端（知识库、会话历史）
│   ├── cache/redis_client.py   # Redis 客户端
│   └── retrieval/bm25_search.py# BM25 精确检索（rank_bm25 + jieba 分词）
├── rag_qa/
│   ├── __init__.py             # 导出 VectorStore / RAGSystem
│   ├── rag_main.py             # 旧入口（faiss 向量库入库/检索，需 faiss-cpu）
│   ├── models/                 # 本地模型：bge-m3、bge-reranker-large、文档分割模型
│   ├── core/
│   │   ├── vector_store.py     # BGE-M3 嵌入 + BGE-Reranker + Milvus 混合检索
│   │   ├── new_rag_system.py   # RAGSystem：意图分类 → 检索 → prompt → LLM 流式
│   │   ├── query_classifier.py # BERT 意图分类器（CPU 运行）
│   │   ├── bert_query_classifier/  # 训练好的 BERT 分类模型权重
│   │   └── strategy_selector.py# 按类别选择回答策略 / prompt 模板
│   └── edu_*                   # 文档处理模块（PDF/图片/OCR/文本分割，入库用）
└── static/
    └── index.html              # Web 前端页面
```

## 环境要求

| 依赖 | 版本 / 说明 |
|------|------------|
| Python | 3.10 ~ 3.11（本项目验证于 3.11.9） |
| MySQL | 8.x（本机 `localhost:3306`，库 `lhlib_kg`） |
| Redis | 任意版本（本机 `localhost:6379`，密码 `1234`） |
| Milvus | 2.x（本机 `localhost:19530`，库 `lhlib_kg`，集合 `proj_rag`） |
| 大模型 | 通义千问（DashScope）API Key |
| GPU（可选） | NVIDIA 显卡 + CUDA 驱动（Blackwell/RTX 50 系需 torch≥2.11） |

## 快速开始

### 1. 创建虚拟环境并安装依赖

```bash
# 进入项目目录，创建虚拟环境
python -m venv .venv

# 激活（Windows）
.venv\Scripts\activate

# 安装依赖（含 GPU 版 PyTorch 说明，见 requirements.txt 头部）
pip install -r requirements.txt
```

> 本项目已适配 NVIDIA GPU（RTX 5060 / CUDA 12.8）。PyTorch 需使用 cu128 构建，安装命令：

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
```

### 2. 启动依赖服务

确保 MySQL、Redis、Milvus 三个服务已在本机启动，端口与 `config.ini` 一致。

### 3. 配置 API Key（必需）

```powershell
# 方式一：设置环境变量（代码读取环境变量 API_KEY）
$env:API_KEY = "sk-你的dashscope密钥"
```

> 注意：`base/config.py` 通过 `os.getenv("API_KEY")` 读取密钥，`config.ini` 中的 `dashscope_api_key` 字段**不生效**。

### 4. 运行

```bash
# 方式一：启动 Web 服务（推荐）
python app.py
# 浏览器访问 http://localhost:8003

# 方式二：命令行交互问答
python new_main.py
```

## 配置说明

`config.ini` 各字段：

| 小节 | 字段 | 说明 |
|------|------|------|
| `[mysql]` | host / user / password / database | MySQL 连接信息（库 `lhlib_kg`） |
| `[redis]` | host / port / password / db | Redis 连接信息 |
| `[milvus]` | host / port / database_name / collection_name | 向量库配置（集合 `proj_rag`） |
| `[llm]` | model / dashscope_base_url | 大模型：`qwen-plus`，DashScope OpenAI 兼容地址 |
| `[retrieval]` | parent_chunk_size / child_chunk_size / chunk_overlap | 文档分块参数（父块 1200 / 子块 300 / 重叠 50） |
| `[retrieval]` | retrieval_k / candidate_m | 召回数量 3 / 最终候选 2 |
| `[logger]` | log_file | 日志文件路径 |
| `[app]` | valid_sources / customer_service_phone | 支持的来源类别 / 客服电话 |

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 返回前端页面 `static/index.html` |
| POST | `/api/query` | 非流式问答（`{query, source_filter?, session_id?}`）；若需 RAG 返回流式提示 |
| WS | `/api/stream` | WebSocket 流式问答（逐 token 返回，含开始/结束/错误帧） |
| POST | `/api/create_session` | 创建新会话，返回 `session_id` |
| GET | `/api/history/{session_id}` | 获取会话历史（最近 5 轮） |
| DELETE | `/api/history/{session_id}` | 清除会话历史 |
| GET | `/api/sources` | 获取有效的学科/来源类别 |
| GET | `/health` | 健康检查 |

## GPU 加速

本项目已在本机 **NVIDIA GeForce RTX 5060（8GB 显存，Blackwell 架构）** 上验证通过。

### 模型部署策略

| 模型 | 设备 | 说明 |
|------|------|------|
| BGE-M3 嵌入 | **GPU**（fp16 加载，显存约 1.1GB） | 检索主路径，GPU 加速收益最大 |
| BGE-Reranker 重排 | **CPU** | 每查询仅重排 2~3 个候选，CPU 毫秒级完成，为 GPU 腾显存 |
| BERT 意图分类 | **CPU** | 轻量短文本分类，无需 GPU |

---

## 许可证

本项目为凌华图书馆内部学习/开发项目，仅供学习交流使用。
