# Git 提交规范 (Git Commit Standards)

为了追踪不同 Agent 对代码库的贡献，请在执行 Git Commit 时通过 `--author` 选项指定相应的作者信息。

## 作者配置规则
每个 Agent 在提交代码时，应根据自身身份设置作者信息。

- **Name**: 使用当前 Agent 的工具名称（例如：`Gemini`, `Claude`, `Codex`, `OpenCode`）。
- **Email**: 格式为 `<name>@llm-meeting.com`（建议全小写，例如：`gemini@llm-meeting.com`）。

## 执行命令建议
在提交时，请使用以下格式附加 `--author` 选项，无需修改仓库的 `user.name` 或 `user.email` 配置：
```bash
git commit --author="Gemini <gemini@llm-meeting.com>" -m "feat: your commit message"
```
*注：请将 "Gemini" 及邮箱前缀替换为您实际的 Agent 名称。*
