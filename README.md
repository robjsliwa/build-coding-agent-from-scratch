# Build a Coding Agent From Scratch

Companion code for the blog series **[Build a Coding Agent From Scratch](https://www.shiftleftai.dev/posts/coding-agent-introduction/)** on [shiftleftai.dev](https://www.shiftleftai.dev).

> "What I cannot create, I do not understand."

## Introduction

Most people learn about AI agents by reaching for a framework. This series takes the opposite route: we build the agent ourselves, one small piece at a time, so every moving part is something you can read, debug and reason about.

The project is **`loom`**, a terminal-based coding agent in the spirit of Claude Code or Codex. It is written in Go with **no SDKs and no agent frameworks**, and it talks to local models through [Ollama](https://ollama.com) (the series uses `qwen3.6:27b` and `gemma4:e4b`).

Along the way you will learn how to implement:

- The agent loop
- Tool calling
- Skills and `AGENTS.md` integration
- Session persistence and management

Start with the [series introduction](https://www.shiftleftai.dev/posts/coding-agent-introduction/) for the full motivation and background.

## Lessons

Each lesson is a self-contained snapshot of `loom` as it stands at the end of that post, so you can read a lesson and run or diff the matching code.

| # | Lesson | Code |
|---|--------|------|
| 1 | [The Agent Loop](https://www.shiftleftai.dev/posts/coding-agent-0001/) | [`lesson-0001/`](lesson-0001) |
| 2 | [Tool Calling](https://www.shiftleftai.dev/posts/coding-agent-0002/) | [`lesson-0002/`](lesson-0002) |
| 3 | [The Tool Suite](https://www.shiftleftai.dev/posts/coding-agent-0003/) | [`lesson-0003/`](lesson-0003) |
| 4 | [The Edit Tool](https://www.shiftleftai.dev/posts/coding-agent-0004/) | [`lesson-0004/`](lesson-0004) |
| 5 | [The System Prompt](https://www.shiftleftai.dev/posts/coding-agent-0005/) | [`lesson-0005/`](lesson-0005) |
| 6 | [Streaming and UX](https://www.shiftleftai.dev/posts/coding-agent-0006/) | [`lesson-0006/`](lesson-0006) |
| 7 | [Safety and Permissions](https://www.shiftleftai.dev/posts/coding-agent-0007/) | [`lesson-0007/`](lesson-0007) |
| 8 | [Compaction](https://www.shiftleftai.dev/posts/coding-agent-0008/) | [`lesson-0008/`](lesson-0008) |
| 9 | [Slash Commands](https://www.shiftleftai.dev/posts/coding-agent-0009/) | [`lesson-0009/`](lesson-0009) |
| 10 | [AGENTS.md, Skills and The Trust Store](https://www.shiftleftai.dev/posts/coding-agent-0010/) | [`lesson-0010/`](lesson-0010) |
| 11 | [Sessions and Persistence](https://www.shiftleftai.dev/posts/coding-agent-0011/) | [`lesson-0011/`](lesson-0011) |

## Repository layout

```
lesson-NNNN/
└── loom/
    ├── go.mod
    └── cmd/loom/main.go   # entry point (later lessons add more packages)
```

## Getting started

**Prerequisites**

- [Go](https://go.dev/dl/) (see `go.mod` in each lesson for the required version)
- [Ollama](https://ollama.com) running locally with a model pulled, e.g. `ollama pull gemma4:e4b`

**Run a lesson**

```sh
cd lesson-0001/loom
go run ./cmd/loom
```

Swap `lesson-0001` for any lesson folder. To see what a lesson changed, diff it against the previous one:

```sh
diff -ru lesson-0003/loom lesson-0004/loom
```

## About the authors

- **Rob Sliwa**: coder, book enthusiast and continuous learner.
- **Pawan Tripathi**: writes about infrastructure, agentic coding and minimalist design.

More posts at **[shiftleftai.dev](https://www.shiftleftai.dev)**.
