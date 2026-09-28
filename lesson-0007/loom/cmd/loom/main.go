package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const (
	ollamaURL = "http://localhost:11434/api/chat"
	// model     = "qwen3.6:27b-mlx"
	model = "gemma4:e4b-mlx"

	numCtx = 16384
)

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Thinking  string     `json:"thinking,omitempty"`
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
	Model    string         `json:"model"`
	Messages []Message      `json:"messages"`
	Tools    []Tool         `json:"tools,omitempty"`
	Stream   bool           `json:"stream"`
	Options  map[string]any `json:"options,omitempty"`
}

type chatResponse struct {
	Message         Message `json:"message"`
	PromptEvalCount int     `json:"prompt_eval_count"`
	EvalCount       int     `json:"eval_count"`
	Done            bool    `json:"done"`
}

type ToolDef struct {
	Tool Tool
	Safe bool
	Run  func(args map[string]any) string
}

var approved = map[string]bool{}

var registry = []ToolDef{readFileDef, listFilesDef, editFileDef, bashDef}

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
	Safe: true,
	Run:  readFile,
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
	Safe: true,
	Run:  listFiles,
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

var editFileDef = ToolDef{
	Tool: Tool{
		Type: "function",
		Function: ToolFunction{
			Name: "edit_file",
			Description: "Edit a file by replacing old_string (which must occur exactly once) " +
				"with new_string. If old_string is empty and the file does not exist, " +
				"the file is created with new_string as its contents.",
			Parameters: json.RawMessage(`{
				"type": "object",
				"properties": {
					"path": {"type": "string", "description": "Relative path to the file"},
					"old_string": {"type": "string", "description": "The exact text to replace. Must match exactly once, including whitespace and indentation. Empty to create a new file."},
					"new_string": {"type": "string", "description": "The replacement text"}
				},
				"required": ["path", "new_string"]
			}`),
		},
	},
	Run: editFile,
}

func editFile(args map[string]any) string {
	path, ok := args["path"].(string)
	if !ok {
		return "error: edit_file requires a string 'path' argument"
	}
	oldStr, _ := args["old_string"].(string)
	newStr, ok := args["new_string"].(string)
	if !ok {
		return "error: edit_file requires a string 'new_string' argument"
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) && oldStr == "" {
			if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
				return "error: " + err.Error()
			}
			if err := os.WriteFile(path, []byte(newStr), 0o644); err != nil {
				return "error: " + err.Error()
			}
			return "ok, created " + path
		}
		return "error: " + err.Error()
	}

	content := string(data)
	switch n := strings.Count(content, oldStr); {
	case oldStr == "" || n == 0:
		return "error: old_string not found in " + path +
			" — re-read the file; it may have changed"
	case n > 1:
		return fmt.Sprintf("error: old_string appears %d times in %s — "+
			"include more surrounding context to make it unique", n, path)
	}

	content = strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "error: " + err.Error()
	}
	return "ok, edited " + path
}

func systemPrompt() string {
	cwd, _ := os.Getwd()
	return fmt.Sprintf(`You are loom, a coding agent. You complete tasks by calling tools,
not by describing what could be done.

Environment:
- Working directory: %s
- Platform: %s/%s
- Today's date: %s

Rules:
- Before editing a file, read it first. Quote old_string exactly, including whitespace.
- After any code change, verify it: build or run tests with bash. Never claim a success you have not seen.
- If a tool returns an error, read it and change your approach; never repeat the same call unchanged.
- Prefer small, targeted edits over rewriting whole files.
- When done, summarize what you changed and how you verified it, in a sentence or two.`,
		cwd, runtime.GOOS, runtime.GOARCH, time.Now().Format("2006-01-02"))
}

func askPermission(scanner *bufio.Scanner, name string) bool {
	fmt.Printf("  allow %s? [y]es once / [a]lways / [n]o: ", name)
	if !scanner.Scan() {
		return false
	}

	switch strings.ToLower(strings.TrimSpace(scanner.Text())) {
	case "y":
		return true
	case "a":
		approved[name] = true
		return true
	}

	return false
}

func chat(ctx context.Context, messages []Message) (Message, error) {
	chatReq := chatRequest{
		Model:    model,
		Messages: messages,
		Tools: func() []Tool {
			tools := make([]Tool, len(registry))
			for i, def := range registry {
				tools[i] = def.Tool
			}
			return tools
		}(),
		Stream:  true,
		Options: map[string]any{"num_ctx": numCtx},
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return Message{}, err
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		ollamaURL,
		bytes.NewReader(body),
	)
	if err != nil {
		return Message{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Message{}, err
	}

	if resp.StatusCode != http.StatusOK {
		return Message{}, fmt.Errorf("request failed with status %s", resp.Status)
	}

	dec := json.NewDecoder(resp.Body)
	var (
		assembled = Message{Role: "assistant"}
		content   strings.Builder
		prefixed  bool
	)

	for {
		var chunk chatResponse
		if err := dec.Decode(&chunk); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			assembled.Content = content.String()
			return assembled, err // return partial result with error
		}

		if chunk.Message.Thinking != "" {
			// \033[90m sets the text color to grey
			// \033[0m resets it back to default
			fmt.Printf("\033[90m%s\033[0m", chunk.Message.Thinking)
		}

		if chunk.Message.Content != "" {
			if !prefixed {
				fmt.Print("\nloom: ")
				prefixed = true
			}
			fmt.Print(chunk.Message.Content)
			content.WriteString(chunk.Message.Content)
		}

		assembled.ToolCalls = append(assembled.ToolCalls, chunk.Message.ToolCalls...)
		if chunk.Done {
			used := chunk.PromptEvalCount + chunk.EvalCount
			marker := ""
			if used > numCtx*8/10 {
				marker = " ⚠ context nearly full"
			}
			fmt.Printf("\n  [ctx %d/%d%s]\n", used, numCtx, marker)
			break
		}
	}

	assembled.Content = content.String()
	return assembled, nil
}

func main() {
	fmt.Printf("loom v0.7 — chatting with %s (ctrl-c to quit)\n", model)
	scanner := bufio.NewScanner(os.Stdin)
	conversation := []Message{{Role: "system", Content: systemPrompt()}}

	for {
		fmt.Print("\nyou: ")
		if !scanner.Scan() {
			break
		}
		conversation = append(conversation,
			Message{Role: "user", Content: scanner.Text()})

		for {
			ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
			reply, err := chat(ctx, conversation)
			stop()
			if err != nil {
				if ctx.Err() != nil {
					fmt.Println("\n(interrupted)")
					if reply.Content != "" {
						reply.ToolCalls = nil
						conversation = append(conversation, reply)
					}
					break
				}
				fmt.Fprintln(os.Stderr, "error:", err)
				break
			}
			conversation = append(conversation, reply)

			if len(reply.ToolCalls) == 0 {
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

				switch {
				case !toolFound:
					result = "error: unknown tool " + tc.Function.Name
				case toolDef.Safe || approved[tc.Function.Name] || askPermission(scanner, tc.Function.Name):
					result = toolDef.Run(tc.Function.Arguments)
				default:
					result = "permission denied by user. Do not retry the same call; " +
						"explain what you wanted to do, or try a different approach."
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
