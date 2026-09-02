项目地址：
<请替换为最终公开 Git 仓库地址>

项目简介：
本项目实现了一个轻量级本地 Coding Agent，不依赖 LangChain、AutoGen、OpenAI Agents SDK 等 Agent 框架，而是基于大语言模型原生 Tool Calling，从底层实现“模型决策—本地执行—环境反馈—继续决策”的完整 Agent 闭环。系统支持自主读取、搜索、创建和修改代码，并能够执行命令、运行测试和根据失败结果自动纠错。

运行方式：
环境要求：Python 3.11+，推荐使用 uv。

1. 进入目录：
   cd packages/my-coding-agent

2. 复制环境配置：
   Copy-Item .env.example .env

3. 在 .env 中配置模型：
   AGENT_MODEL=gpt-5.6-sol
   AGENT_API_KEY=<YOUR_API_KEY>
   AGENT_BASE_URL=<OPENAI_COMPATIBLE_BASE_URL>

4. 安装依赖：
   uv sync

5. 启动：
   uv run my-coding-agent <workspace>

示例：
uv run my-coding-agent ../../examples/demo-calculator

主要功能与特色：

1. 原生 Agent Loop：自主解析模型 Tool Call，本地执行后将 Tool Result 重新加入上下文，实现连续多步决策。
2. 本地工具系统：实现 list_files、search、read、write、edit、bash 六种 Coding Tools，并通过 ToolRegistry 统一注册、参数校验和执行。
3. Session 与 Context：使用 JSONL 持久化会话，并通过大结果落盘、消息裁剪、Tool Result 压缩和 LLM 摘要四层机制控制长上下文。
4. Hook 权限控制：提供 ask、auto、plan 三种模式，对 write、edit、bash 等有副作用操作进行拦截和授权。
5. 结构化错误恢复：处理非法参数、未知工具、文件错误、命令失败、超时、空响应和最大循环次数等异常，并将错误作为 Observation 返回模型进行自我修正。
6. 长期 Memory：通过 USER.md 和 MEMORY.md 保存跨 Session 的用户偏好及项目知识。
7. Subagent：支持 reviewer、tester 等只读子代理，使用独立 Session 完成代码审查和测试分析，再将结果返回主 Agent。
8. Visual TUI：基于 Agent 生命周期事件实时展示步骤、工具调用、执行结果、权限状态、Memory 和 Subagent 信息。

本项目重点不是简单封装模型接口，而是独立实现 Coding Agent 的核心运行、状态管理、安全控制与错误恢复机制。
