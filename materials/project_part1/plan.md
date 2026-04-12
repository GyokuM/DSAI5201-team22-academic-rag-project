DSAI5201 项目讨论总纲
题目方向：Lightweight RAG for Academic Paper QA
一、先说结论：这个题能做，而且很适合这门课
老师在 Option 2 的参考题里，直接列了
“Building a Lightweight RAG System for Academic Papers”，任务描述就是：
用 FAISS 和 LLM 搭建一个基于本地 PDF 的学术论文问答系统。
所以从选题合法性上，这个方向是完全符合老师要求的。
而且这门课本身强调的是 practice / real-world impact / 从 data ingestion 到 model deployment 的完整 pipeline，老师也在课程介绍里强调项目要尽量贴近真实应用，而不是只停留在概念层面。

---
二、老师最看重什么
1）Option 2 不是“跑通现成代码”就行
老师写得很明确：
- 可以用 public code / datasets
- 但必须加入 new contributions
- 例如：
  - improved methods
  - ablation studies
  - new insights
也就是说，这个项目不能只是：
“我用 LangChain + FAISS + 某个 LLM 做了个聊天机器人。”
这样太弱了。

---
2）项目报告评分看三件事
老师的 rubric 主要看：
- Depth of Analysis（分析深度）
- Technical Contribution（技术贡献）
- Organization & Clarity（组织与表达清晰度）
所以我们做项目时，必须从一开始就按这三项倒推。

---
3）提交物和展示要求
最终要满足这些硬要求：
- Option 2 报告至少 5 页
- 单栏 PDF
- 要有 code submission
- 要有 demo results
- PDF 里还要单独附一页成员贡献说明
- presentation 10 分钟 + Q&A 5 分钟
- 每个组员都必须讲自己负责的部分

---
三、我们这个项目的最推荐定位
不要把它说成：
做一个学术论文聊天机器人
而要说成：
构建并系统评估一个轻量级 Academic Paper RAG 系统，重点研究 PDF parsing、chunking、retrieval 和 grounded answering 对问答质量的影响。
这样一来，我们项目就不只是“做系统”，而是“做系统 + 做实验 + 做分析”。
这才是最符合老师口味的版本。

---
四、项目最终目标
最终系统要能做到什么
输入：本地学术论文 PDF
输出：
- 回答用户问题
- 给出引用依据
- 尽量减少 hallucination
- 展示检索到的相关文本块
我们真正要研究的问题
不是“能不能问答”，而是：
1. 不同 PDF 解析方式 会不会影响结果
2. 不同 chunking 策略 哪个更适合学术论文
3. 不同 retrieval 设置 哪个效果最好
4. 加入 grounding prompt / citation constraint 后，答案是否更可靠
5. 是否可以在保证效果的同时保持 lightweight（轻量部署、低资源）

---
五、项目完成流程
这个部分就是你开会时最需要讲的。
阶段 1：确定数据与任务范围
我们先不做“大而全”，而是做一个text-first 的 academic RAG。
数据来源
优先考虑 Open RAG Benchmark。
这个数据集就是为 PDF-based RAG 设计的，Hugging Face 页面显示它当前草稿版包含：
- 1000 篇 arXiv PDF 论文
- 3045 组 QA
- 内容包含 text、tables、images
- 组织方式类似 BEIR 格式，适合做 RAG 评测。 (Hugging Face)
但为了控制工作量，我们一开始建议：
- 先做 text-only / text-dominant 部分
- 先不碰复杂图像问答
- 表格最多作为 bonus
任务范围
先做：
- 从 PDF 提取文本
- 文本切块
- 检索
- 问答生成
- 引用来源展示

---
阶段 2：做 baseline 系统
先搭一个最基础可运行版本：
1. PDF parser
2. text splitter
3. embedding model
4. FAISS vector store
5. top-k retrieval
6. LLM answer generation
7. answer + source chunk 展示
技术栈建议
- Python
- FAISS
- sentence-transformers / BGE embedding
- 本地或 API LLM
- Streamlit / Gradio 做 demo
FAISS 官方定位就是高效的向量相似度检索库，很适合这个项目。(Hugging Face)

---
阶段 3：做“自己的贡献”
这是整个项目最核心的部分。
六、我们推荐的主要贡献点
贡献点 1：PDF Parsing + Chunking Strategy Ablation
这是最稳的主贡献。
为什么重要
学术论文不是普通网页文本，里面有：
- 标题层级
- 摘要
- 方法
- 实验
- 表格
- 公式
- 引用
如果 chunking 做得差，检索就会崩。
我们怎么做
比较至少 3 种 chunking 方式：
- 固定长度 chunk
- RecursiveCharacterTextSplitter
- 结构化 / section-aware chunking
LangChain 的 RecursiveCharacterTextSplitter 本身就是常用 RAG splitter，支持 chunkSize 和 chunkOverlap 这类参数。(LangChain 参考文档)
我们要输出什么
- Recall@k / hit rate 对比
- 示例问题的检索片段对比
- 分析哪种分块更适合 academic PDF
这点为什么算贡献
因为这是标准的 ablation study + improved method，完全符合老师要求。

---
贡献点 2：Retrieval Configuration Optimization
第二个主贡献。
我们怎么做
对比：
- 不同 embedding model
- 不同 top-k
- 是否加 reranker
- 是否做 hybrid retrieval
我们想回答的问题
- top-5 还是 top-10 更合适
- 小模型 embedding 能不能已经够用
- rerank 是否真的值得加
- 检索速度和效果如何 trade-off
输出结果
- 检索准确率表格
- 检索延迟表格
- 最佳配置建议
这点非常适合写在实验章节，也很像老师要的 new insights。

---
贡献点 3：Grounded Answer Generation
第三个主贡献。
为什么要做
RAG 最大问题不是“答不出来”，而是：
- 检索到了但没用好
- 乱编
- 引用不准
- hallucination
我们怎么做
对比两种或三种 prompt 模式：
- 普通问答 prompt
- 强制“仅根据检索内容回答”
- 强制“给出引用依据 / source chunk”
我们看什么
- faithfulness
- answer correctness
- hallucination cases
输出
- grounded prompt 前后对比
- 错误案例分析
- 最佳 prompt template
这点很适合写成：
“Effect of grounding constraints on academic RAG answer faithfulness.”

---
贡献点 4：Lightweight Deployment + Error Analysis
这个可以保留，但建议当第四贡献 / bonus contribution，不是主轴。
轻量化部分
如果时间和设备允许，可以测试：
- 小模型 / API 模型 vs 本地量化模型
- 响应时间
- 显存 / 内存占用
Hugging Face 文档说明，bitsandbytes 支持 8-bit 和 4-bit 量化，4-bit 可以明显降低内存占用，适合做“轻量部署”实验。(Hugging Face)
但注意
这部分不建议当你们最核心的贡献。
因为你们主线是 Academic Paper RAG，不是 “模型压缩项目”。
所以更推荐的写法是：
- deployment optimization
- system efficiency analysis
- resource-performance tradeoff
Error Analysis
这一块非常值得做，而且很加分。
可以总结错误类型：
- PDF 解析失败
- chunk 太碎 / 太大
- 检索错段落
- 检索对了但 LLM 编造
- 有证据但回答没对齐问题
这个就是非常典型的 Depth of Analysis。

---
七、需不需要训练模型？
结论：不需要从零训练大模型
这题最稳的方案是：
- 不用自己训练 LLM
- 不用自己训练 embedding model
- 直接使用公开模型 / API / 公共框架
- 把重点放在 系统设计 + ablation + analysis
因为老师明确允许 public code / datasets can be used，但关键是要有 new contributions。
那训练什么？
严格说，这个项目更多是：
- 配置 embedding
- 建 FAISS index
- 做 retrieval experiments
- 做 prompt and pipeline optimization
如果你们有余力，最多可以：
- 微调一个小 reranker
- 或自己构造一点 evaluation set
但这不是必须的。

---
八、前端后端要不要做？
结论：要有 demo，但不需要重工程
老师要的是 code submission + demo results。
所以最推荐做法是：
后端
- Python
- RAG pipeline 封装成函数或简单 API
- 支持输入 question
- 输出：
  - retrieved chunks
  - final answer
  - citations
前端
- Streamlit 或 Gradio
- 简洁即可：
  - 选择 PDF / 选择 paper
  - 输入问题
  - 输出答案
  - 展示来源
重点
前端不是拿分核心。
真正拿分的是方法、实验、分析。

---
九、最终交付结构怎么设计最符合老师 rubric
报告目录建议
这个最适合直接照着写。
1. Introduction
- 项目背景
- 为什么 academic PDF QA 有价值
- RAG 在这个场景的难点
2. Related Background
- RAG 基本原理
- FAISS
- PDF parsing/chunking
- grounded generation
3. Baseline System
- 数据
- parser
- chunking
- embedding
- retrieval
- generation
4. Our Contributions
这里单独列你们自己的改进：
- parsing/chunking ablation
- retrieval optimization
- grounded answering
- lightweight deployment / error analysis
5. Experimental Setup
- 数据集
- 问题集
- 评测指标
- 硬件环境
- 所有对比设置
6. Results and Analysis
- 表格
- 图
- case study
- 错误分析
- 最佳配置总结
7. Limitations and Future Work
- 当前系统只做 text-first
- 图表和多模态部分还不够
- 未来可扩展到 multimodal PDF QA
8. Conclusion
- 总结贡献
- 总结发现
9. Statement of Contribution
单独一页写：
- 成员姓名学号
- 负责章节
- 具体任务
这是老师强制要求。

---
十、presentation 怎么讲最稳
老师要求：
- 10 分钟 total
- 每个人都要讲
- 讲自己负责的部分
- 要保证整体像一个 cohesive story
- 之后 5 分钟 Q&A。
最推荐故事线
1. 问题：学术 PDF QA 难在哪
2. baseline：我们先搭了一个基础 RAG
3. 改进：我们比较了 chunking / retrieval / grounding
4. 结果：哪些设置最好
5. insight：为什么会这样
6. demo：现场问一个问题
7. limitation：还没做什么

---
十一、建议分工（3–4人版）
人 1：数据与 parsing
- PDF 读取
- 清洗
- chunking 实验
人 2：retrieval
- embedding
- FAISS
- top-k / reranker 对比
人 3：generation 与 evaluation
- prompt
- faithfulness / correctness 评测
- 错误分析
人 4：demo 与 report integration
- Streamlit/Gradio
- 图表整理
- 报告合并
- presentation 统稿

---
十二、开会时最该讨论的 6 个决策
你待会开会最该拍板的是这六件事：
1. 题目最终版本叫什么
我建议：
Improving Lightweight RAG for Academic Paper Question Answering:
A Comparative Study of Chunking, Retrieval, and Grounding Strategies
2. 数据集用什么
- 直接用 Open RAG Bench 为主
- 先做 text-first
- 多模态先不展开太大
3. 我们的核心贡献保留哪 3 个
最推荐：
- chunking ablation
- retrieval optimization
- grounded answering
4. 轻量化要不要做
- 有余力做
- 没余力就降级为 bonus
- 不要让它抢主线
5. demo 形式
- Streamlit / Gradio 简单网页
- 不做复杂前后端分离
6. 每个人负责哪块
这个一定早点定，因为最终 contribution statement 和 presentation 都要对应。

---