package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	ollamaURL = "http://localhost:11434/api/chat"
	// model     = "qwen3.6:27b-mlx"
	model = "gemma4:e4b-mlx"
)

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	ToolName  string     `json:"tool_name,omitempty"`
}

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Tools    []Tool    `json:"tools,omitempty"`
	Stream   bool      `json:"stream"`
}

type chatResponse struct {
	Message Message `json:"message"`
}

type ToolDef struct {
	Tool Tool
	Run  func(args map[string]any) string
}

var registry = []ToolDef{readFileDef, listFilesDef, bashDef}

var readFileDef = ToolDef{
	Tool: Tool{
		Type: "function",
		Function: ToolFunction{
			Name: "read_file",
			Description: "Read a file and return its contents as text. " +
				"Use this whenever you need to see what a file contains.",
			Parameters: json.RawMessage(`{
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "Relative path to the file"}
			},
			"required": ["path"]
		}`),
		},
	},
	Run: readFile,
}

func readFile(args map[string]any) string {
	path, ok := args["path"].(string)
	if !ok {
		return "error: read_file requires a string 'path' argument"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "error: " + err.Error()
	}
	return string(data)
}

var listFilesDef = ToolDef{
	Tool: Tool{
		Type: "function",
		Function: ToolFunction{
			Name: "list_files",
			Description: "Recursively list files under a directory. " +
				"Use this to discover what exists before reading files.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "Directory to list; defaults to the current directory"}
				}
			}`),
		},
	},
	Run: listFiles,
}

func listFiles(args map[string]any) string {
	dir, _ := args["path"].(string)
	if dir == "" {
		dir = "."
	}
	var b strings.Builder
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() && d.Name() == ".git" {
			return filepath.SkipDir
		}
		if !d.IsDir() {
			b.WriteString(p + "\n")
		}
		return nil
	})
	if err != nil {
		return "error: " + err.Error()
	}
	return b.String()
}

var bashDef = ToolDef{
	Tool: Tool{
		Type: "function",
		Function: ToolFunction{
			Name: "bash",
			Description: "Execute a shell command and return its combined stdout and stderr. " +
				"Use for running tests, builds, git, and anything without a dedicated tool.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"command": {"type": "string", "description": "The shell command to run"}
				},
				"required": ["command"]
			}`),
		},
	},
	Run: bashTool,
}

func bashTool(args map[string]any) string {
	command, ok := args["command"].(string)
	if !ok {
		return "error: bash requires a string 'command' argument"
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, "bash", "-c", command).CombinedOutput()
	if err != nil {
		return string(out) + "\nerror: " + err.Error()
	}
	if len(out) == 0 {
		return "(no output)"
	}
	return string(out)
}

func chat(messages []Message) (Message, error) {
	body, err := json.Marshal(chatRequest{
		Model:    model,
		Messages: messages,
		Tools: func() []Tool {
			tools := make([]Tool, len(registry))
			for i, def := range registry {
				tools[i] = def.Tool
			}
			return tools
		}(),
		Stream: false,
	})
	if err != nil {
		return Message{}, err
	}

	resp, err := http.Post(
		ollamaURL,
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return Message{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Message{}, fmt.Errorf("ollama returned %s", resp.Status)
	}

	var out chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return Message{}, err
	}

	return out.Message, nil
}

func main() {
	fmt.Printf("loom v0.3 — chatting with %s (ctrl-c to quit)\n", model)
	scanner := bufio.NewScanner(os.Stdin)
	var conversation []Message

	for {
		fmt.Print("\nyou: ")
		if !scanner.Scan() {
			break
		}
		conversation = append(conversation,
			Message{Role: "user", Content: scanner.Text()})

		for {
			reply, err := chat(conversation)
			if err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
				break
			}
			conversation = append(conversation, reply)

			if len(reply.ToolCalls) == 0 {
				fmt.Println("\nloom:", reply.Content)
				break
			}

			for _, tc := range reply.ToolCalls {
				fmt.Printf("  ⚙ %s(%v)\n", tc.Function.Name, tc.Function.Arguments)
				var result string
				var toolDef ToolDef
				var toolFound bool
				for _, def := range registry {
					if def.Tool.Function.Name == tc.Function.Name {
						toolDef = def
						toolFound = true
					}
				}

				if toolFound {
					result = toolDef.Run(tc.Function.Arguments)
				} else {
					result = "error: unknown tool " + tc.Function.Name
				}

				conversation = append(conversation, Message{
					Role:     "tool",
					ToolName: tc.Function.Name,
					Content:  result,
				})
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "\nreading standard input:", err)
	}
}
