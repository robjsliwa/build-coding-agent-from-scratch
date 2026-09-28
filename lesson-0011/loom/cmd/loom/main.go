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
	"slices"
	"strings"
	"time"
)

const (
	ollamaURL = "http://localhost:11434/api/chat"
	// model     = "qwen3.6:27b-mlx"
	model = "gemma4:e4b-mlx"
)

const (
	numCtx    = 128 * 1024
	compactAt = numCtx * 8 / 10
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
	Stream   bool           `json:"stream"`
	Tools    []Tool         `json:"tools,omitempty"`
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

// A Command is the harness-side twin of a ToolDef: the user invokes it,
// never the model, and it runs before the conversation array is touched.
type Command struct {
	Name, Desc string
	Run        func(arg string)
}

var approved = map[string]bool{}

func askPermission(scanner *bufio.Scanner, name string) bool {
	fmt.Printf("   allow %s? [y]ess once / [a]lways / [n]o: ", name)
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

var registry = []ToolDef{readFileDef, listFilesDef, bashDef, editFileDef}

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
			" - re-read the file; it may have changed"

	case n > 1:
		return fmt.Sprintf("error: old_string appears %d times in %s - "+
			"include more surrounding context to make it unique", n, path)
	}

	content = strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "error: " + err.Error()
	}

	return "ok, edited " + path
}

type Skill struct {
	Name, Desc, Body, Dir string
}

// Both are decided once at startup in main and read by systemPrompt().
var (
	repoTrusted bool
	skills      []Skill
)

// loomHome is the global config directory, ~/.loom.
func loomHome() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".loom")
}

// contextSection reads one context file and wraps it in a provenance
// header. Config is optional everywhere, so a missing file is not an
// error — it just contributes nothing.
func contextSection(label, path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return "\n\n## " + label + "\n" + strings.TrimSpace(string(data))
}

// parseSkill reads one SKILL.md: optional frontmatter between ---
// fences holding name: and description: lines, then the body. The
// name falls back to the skill's directory name.
func parseSkill(path string) (Skill, bool) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Skill{}, false
	}

	// Absolute, because repo skills are discovered under a relative
	// ".loom/skills" and a cwd-relative dir is the same guess again.
	dir := filepath.Dir(path)
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}

	s := Skill{
		Name: filepath.Base(dir),
		Desc: "(no description)",
		Dir:  dir,
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 && strings.TrimSpace(lines[0]) == "---" {
		i := 1
		for ; i < len(lines) && strings.TrimSpace(lines[i]) != "---"; i++ {
			if v, ok := strings.CutPrefix(lines[i], "name:"); ok {
				s.Name = strings.TrimSpace(v)
			}

			if v, ok := strings.CutPrefix(lines[i], "description:"); ok {
				s.Desc = strings.TrimSpace(v)
			}
		}

		if i < len(lines) {
			i++ // step past the closing ---
		}

		s.Body = strings.TrimSpace(strings.Join(lines[i:], "\n"))
	} else {
		s.Body = strings.TrimSpace(string(data))
	}

	return s, true
}

// discoverSkills finds every <dir>/<name>/SKILL.md under one skills home.
func discoverSkills(dir string) []Skill {
	matches, _ := filepath.Glob(filepath.Join(dir, "*", "SKILL.md"))
	var out []Skill
	for _, m := range matches {
		if s, ok := parseSkill(m); ok {
			out = append(out, s)
		}
	}
	return out
}

type trustEntry struct {
	Trusted bool   `json:"trusted"`
	Decided string `json:"decided"`
}

func trustPath() string {
	return filepath.Join(loomHome(), "trusted-dirs.json")
}

func loadTrust() map[string]trustEntry {
	trust := map[string]trustEntry{}
	if data, err := os.ReadFile(trustPath()); err == nil {
		json.Unmarshal(data, &trust)
	}
	return trust // missing or unreadable file = empty map = ask
}

func saveTrust(trust map[string]trustEntry) {
	data, _ := json.MarshalIndent(trust, "", "  ")
	err := os.MkdirAll(loomHome(), 0o755)
	if err == nil {
		err = os.WriteFile(trustPath(), data, 0o644)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning: trust store not saved:", err)
	}
}

// decideTrust returns whether repo config may load this session: a
// remembered decision answers silently; otherwise one y/a/n prompt —
// the Lesson 7 gate shape, aimed at a directory instead of a tool.
func decideTrust(scanner *bufio.Scanner) bool {
	_, agentsErr := os.Stat("AGENTS.md")
	_, loomDirErr := os.Stat(".loom")
	if agentsErr != nil && loomDirErr != nil {
		return false // no repo config on offer, nothing to decide
	}

	cwd, _ := os.Getwd()
	key, _ := filepath.Abs(cwd)
	trust := loadTrust()
	if e, ok := trust[key]; ok {
		return e.Trusted
	}

	var found []string
	if agentsErr == nil {
		found = append(found, "AGENTS.md")
	}
	if n := len(discoverSkills(filepath.Join(".loom", "skills"))); n > 0 {
		found = append(found, fmt.Sprintf("%d skill(s)", n))
	}
	fmt.Printf("  this directory offers loom config: %s\n", strings.Join(found, ", "))
	fmt.Print("  load it? [y]es this session / [a]lways / [n]o: ")
	if !scanner.Scan() {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(scanner.Text())) {
	case "y":
		return true
	case "a":
		trust[key] = trustEntry{Trusted: true, Decided: time.Now().Format(time.RFC3339)}
		saveTrust(trust)
		return true
	}
	return false
}

func systemPrompt() string {
	cwd, _ := os.Getwd()
	prompt := fmt.Sprintf(`You are loom, a coding agent. You complete tasks by calling tools,
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

	// Context files are re-read on every call: the reads are
	// microseconds, and it makes /clear double as a config reload.
	prompt += contextSection("User notes (~/.loom/AGENTS.md)",
		filepath.Join(loomHome(), "AGENTS.md"))
	if repoTrusted {
		prompt += contextSection("Project notes (AGENTS.md)", "AGENTS.md")
	}

	// The skills menu: names and descriptions only — bodies stay on
	// disk until /skill:<name> loads them (progressive disclosure).
	if len(skills) > 0 {
		var menu strings.Builder
		menu.WriteString("\n\n## Skills\n")
		menu.WriteString("If a task matches a skill, ask the user to load it with /skill:<name>.\n")
		for _, s := range skills {
			fmt.Fprintf(&menu, "- %s: %s\n", s.Name, s.Desc)
		}
		prompt += strings.TrimRight(menu.String(), "\n")
	}

	return prompt
}

func chat(ctx context.Context, messages []Message) (Message, int, error) {
	var used int
	chatReq := chatRequest{
		Model:    model,
		Messages: messages,
		Stream:   true,
		Tools: func() []Tool {
			tools := make([]Tool, len(registry))
			for i, def := range registry {
				tools[i] = def.Tool
			}
			return tools
		}(),
		Options: map[string]any{"num_ctx": numCtx},
	}
	body, err := json.Marshal(chatReq)
	if err != nil {
		return Message{}, used, err
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		ollamaURL,
		bytes.NewReader(body),
	)
	if err != nil {
		return Message{}, used, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return Message{}, used, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return Message{}, used, fmt.Errorf("request failed with status: %s", resp.Status)
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
			return assembled, used, err // partial result travels with the error
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
			used = chunk.PromptEvalCount + chunk.EvalCount
			break
		}
	}

	assembled.Content = content.String()
	return assembled, used, nil
}

func summarize(ctx context.Context, messages []Message) (string, error) {
	instruction := Message{Role: "user", Content: "Summarize this conversation " +
		"for your own future reference. Preserve: the user's goals, decisions made, " +
		"file paths touched and how they changed, tool results that still matter, " +
		"and unfinished work. Dense bullet points. Omit pleasantries and dead ends."}

	msgs := append(append([]Message{}, messages[1:]...), instruction)

	body, err := json.Marshal(chatRequest{
		Model:    model,
		Messages: msgs,
		Stream:   false,
		Options:  map[string]any{"num_ctx": numCtx},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(
		ctx,
		"POST",
		ollamaURL,
		bytes.NewReader(body),
	)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("summarize failed with status: %s", resp.Status)
	}

	var res chatResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", err
	}

	return res.Message.Content, nil
}

func compact(conversation []Message, summary string) []Message {
	const keep = 6
	start := len(conversation) - keep

	if start < 1 {
		start = 1
	}

	for start > 1 && conversation[start].Role == "tool" {
		start--
	}

	fresh := []Message{
		{Role: "system", Content: systemPrompt()},
		{Role: "user", Content: "[Context summary - earlier messages were compacted " +
			"to save space. File contents mentioned below may be stale, re-read " +
			"files before editing.]\n\n" + summary},
	}

	return append(fresh, conversation[start:]...)
}

func estimateTokens(messages []Message) int {
	n := 0
	for _, m := range messages {
		n += messageTokens(m)
	}

	return n
}

func messageTokens(m Message) int {
	n := len(m.Content) + len(m.Thinking)
	for _, tc := range m.ToolCalls {
		b, _ := json.Marshal(tc)
		n += len(b)
	}

	return n / 4
}

func roleField(m Message) string {
	switch {
	case m.Role == "assistant" && len(m.ToolCalls) > 0:
		names := make([]string, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			names[i] = tc.Function.Name
		}
		return "assistant  ⚙ " + strings.Join(names, ", ")

	case m.Role == "tool":
		return "tool ⇐ " + m.ToolName
	}

	return m.Role
}

func preview(m Message, n int) string {
	if m.Content == "" && len(m.ToolCalls) > 0 {
		return fmt.Sprintf("(%d tool call(s), no text)", len(m.ToolCalls))
	}

	text := strings.ReplaceAll(m.Content, "\n", "␤")
	r := []rune(text)
	if len(r) > n {
		return string(r[:n]) + "…"
	}

	return text
}

// xray prints the conversation array, one row per message, then the
// totals the compaction trigger works from. Since the model has no
// state but this array, the printout is the model's whole mind.
func xray(conversation []Message) {
	fmt.Println("── conversation x-ray ──────────────────────────────────────────")
	for i, m := range conversation {
		fmt.Printf(" #%-3d%-28s%9s  %s\n",
			i, roleField(m), fmt.Sprintf("~%d tok", messageTokens(m)), preview(m, 40))
	}
	fmt.Println("────────────────────────────────────────────────────────────────")
	fmt.Printf(" %d messages · ~%d tokens estimated · budget %d · compacts at %d\n",
		len(conversation), estimateTokens(conversation), numCtx, compactAt)
}

const maxLogLine = 16 * 1024 * 1024 // a logged tool result can be enormous

// record is one line of the log. Kind says how to read the rest: "meta"
// opens a file, "system" and "message" carry a Message, and "compact"
// carries the summary the window was rebuilt around.
type record struct {
	Kind       string   `json:"kind"`
	Time       string   `json:"time"`
	Model      string   `json:"model,omitempty"`
	Cwd        string   `json:"cwd,omitempty"`
	ForkedFrom string   `json:"forked_from,omitempty"`
	Message    *Message `json:"message,omitempty"`
	Summary    string   `json:"summary,omitempty"`
}

type Session struct {
	ID   string
	Path string
	f    *os.File
}

// session is nil when logging is unavailable and every method below
// tolerates a nil receiver, so no call site has to check.
var session *Session

func sessionsDir() string {
	return filepath.Join(loomHome(), "sessions")
}

func (s *Session) log(r record) {
	if s == nil {
		return
	}

	r.Time = time.Now().Format(time.RFC3339)
	data, err := json.Marshal(r)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning: session log encode failed:", err)
		return
	}

	// json.Marshal escapes newlines, so one record is always one line.
	if _, err := s.f.Write(append(data, '\n')); err != nil {
		fmt.Fprintln(os.Stderr, "warning: session log write failed:", err)
	}
}

func (s *Session) close() {
	if s != nil {
		s.f.Close()
	}
}

// openSession creates a new log file and writes its meta line. The id is
// a timestamp, so ids sort chronologically as plain strings.
func openSession(forkedFrom string) *Session {
	if err := os.MkdirAll(sessionsDir(), 0o700); err != nil {
		fmt.Fprintln(os.Stderr, "warning: session logging disabled:", err)
		return nil
	}

	base := time.Now().Format("20060102-150405")
	for n := 0; n < 100; n++ {
		id := base
		if n > 0 {
			id = fmt.Sprintf("%s-%d", base, n)
		}

		// O_EXCL: two looms started in the same second get separate logs.
		path := filepath.Join(sessionsDir(), id+".jsonl")
		f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY|os.O_APPEND, 0o600)
		if os.IsExist(err) {
			continue
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "warning: session logging disabled:", err)
			return nil
		}

		cwd, _ := os.Getwd()
		s := &Session{ID: id, Path: path, f: f}
		s.log(record{Kind: "meta", Model: model, Cwd: cwd, ForkedFrom: forkedFrom})
		return s
	}

	fmt.Fprintln(os.Stderr, "warning: session logging disabled: no free id")
	return nil
}

// reopenSession appends to an existing log, so resuming a session
// continues its record instead of starting a second one beside it.
func reopenSession(id string) *Session {
	path := filepath.Join(sessionsDir(), id+".jsonl")
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_APPEND, 0o600)
	if err != nil {
		fmt.Fprintln(os.Stderr, "warning: cannot append to session log:", err)
		return nil
	}

	return &Session{ID: id, Path: path, f: f}
}

// newWindow starts a fresh conversation and records the system prompt it
// was built with. The prompt is logged for the human reading the
// transcript later. Replay regenerates it rather than replaying it.
func newWindow() []Message {
	sys := Message{Role: "system", Content: systemPrompt()}
	session.log(record{Kind: "system", Message: &sys})
	return []Message{sys}
}

func add(conversation *[]Message, msgs ...Message) {
	for _, m := range msgs {
		*conversation = append(*conversation, m)
		session.log(record{Kind: "message", Message: &m})
	}
}

func scanLog(path string, fn func(record)) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), maxLogLine)

	bad := 0
	for sc.Scan() {
		if strings.TrimSpace(sc.Text()) == "" {
			continue
		}

		var r record
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			bad++
			continue
		}

		fn(r)
	}

	if bad > 0 {
		fmt.Fprintf(os.Stderr, "  warning: skipped %d unreadable line(s) in %s\n", bad, path)
	}

	return sc.Err()
}

func replay(id string) ([]Message, error) {
	window := []Message{{Role: "system", Content: systemPrompt()}}

	err := scanLog(filepath.Join(sessionsDir(), id+".jsonl"), func(r record) {
		switch r.Kind {
		case "message":
			if r.Message != nil {
				window = append(window, *r.Message)
			}

		case "compact":
			// The same function the live loop called, on the same
			// messages, with the same summary. This is why compact() must
			// not write to the log: replaying an event is not the event.
			window = compact(window, r.Summary)
		}
	})

	return window, err
}

func dropOrphanedCalls(msgs []Message) []Message {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "tool" {
			continue
		}

		if msgs[i].Role == "assistant" && len(msgs[i].ToolCalls) > len(msgs[i+1:]) {
			return msgs[:i]
		}

		break
	}

	return msgs
}

func resumeWindow(id string) ([]Message, error) {
	window, err := replay(id)
	if err != nil {
		return nil, err
	}

	window = dropOrphanedCalls(window)
	window = append(window, Message{Role: "user", Content: fmt.Sprintf(
		"[Session %s reopened. Time has passed since the messages above and "+
			"file contents may have changed; re-read any file before editing it.]", id)})

	return window, nil
}

type sessionInfo struct {
	ID, Started, First string
	Msgs               int
}

// listSessions summarizes the most recent logs, newest first.
func listSessions(limit int) []sessionInfo {
	paths, _ := filepath.Glob(filepath.Join(sessionsDir(), "*.jsonl"))
	slices.Sort(paths) // ids are timestamps, so lexical order is chronological

	var out []sessionInfo
	for i := len(paths) - 1; i >= 0 && len(out) < limit; i-- {
		info := sessionInfo{ID: strings.TrimSuffix(filepath.Base(paths[i]), ".jsonl")}
		err := scanLog(paths[i], func(r record) {
			switch r.Kind {
			case "meta":
				info.Started = r.Time

			case "message":
				if r.Message == nil {
					return
				}
				info.Msgs++
				if info.First == "" && r.Message.Role == "user" {
					info.First = preview(*r.Message, 44)
				}
			}
		})
		// A loom that started and was never spoken to leaves a log with no
		// messages in it; only the current one is worth showing.
		if err != nil || (info.Msgs == 0 && (session == nil || info.ID != session.ID)) {
			continue
		}

		out = append(out, info)
	}

	return out
}

func printSessions() {
	infos := listSessions(10)
	if len(infos) == 0 {
		fmt.Println("  no saved sessions yet")
		return
	}

	fmt.Println("  recent sessions (newest first, * = current):")
	for _, in := range infos {
		marker := " "
		if session != nil && in.ID == session.ID {
			marker = "*"
		}
		fmt.Printf("  %s %-17s %4d msg  %s\n", marker, in.ID, in.Msgs, in.First)
	}

	fmt.Println("  /resume <id> to continue one - /fork <id> to branch it")
}

func runTurn(scanner *bufio.Scanner, conversation *[]Message, ctxSize *int) {
	for {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		reply, used, err := chat(ctx, *conversation)
		stop()
		if err != nil {
			if ctx.Err() != nil {
				fmt.Println("\n(interrupted)")
				if reply.Content != "" {
					reply.ToolCalls = nil
					add(conversation, reply)
				}
				return
			}
			fmt.Fprintln(os.Stderr, "error:", err)
			return
		}

		add(conversation, reply)

		for _, tc := range reply.ToolCalls {
			fmt.Printf("  ⚙ %s(%v)\n", tc.Function.Name, tc.Function.Arguments)
			var result string
			var toolDef ToolDef
			var toolFound bool
			for _, def := range registry {
				if def.Tool.Function.Name == tc.Function.Name {
					toolDef = def
					toolFound = true
					break
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

			add(conversation, Message{
				Role:     "tool",
				ToolName: tc.Function.Name,
				Content:  result,
			})
		}

		*ctxSize = max(estimateTokens(*conversation), used)

		if *ctxSize > compactAt {
			fmt.Printf("\n   [compacting %d messages, ctx %d/%d]\n",
				len(*conversation), *ctxSize, numCtx)
			summary, err := summarize(context.Background(), *conversation)
			if err != nil {
				fmt.Fprintln(os.Stderr, "compaction failed, continuing:", err)
			} else {
				*conversation = compact(*conversation, summary)
				session.log(record{Kind: "compact", Summary: summary})
				*ctxSize = estimateTokens(*conversation)
			}
		}

		if len(reply.ToolCalls) == 0 {
			break
		}
	}
	fmt.Printf("\n  [ctx %d/%d]\n", *ctxSize, numCtx)
}

func main() {
	fmt.Printf("loom v0.11 - chatting with %s (ctrl-c to quit)\n", model)
	scanner := bufio.NewScanner(os.Stdin)

	session = openSession("")
	defer func() { session.close() }()

	repoTrusted = decideTrust(scanner)
	globalSkills := discoverSkills(filepath.Join(loomHome(), "skills"))
	skills = globalSkills
	if repoTrusted {
		skills = append(skills, discoverSkills(filepath.Join(".loom", "skills"))...)
	}

	// The startup report: say what loaded and what didn't — silence
	// about active config is how injection stays invisible.
	var report []string
	if _, err := os.Stat(filepath.Join(loomHome(), "AGENTS.md")); err == nil {
		report = append(report, "user AGENTS.md")
	}
	if _, err := os.Stat("AGENTS.md"); err == nil {
		if repoTrusted {
			report = append(report, "project AGENTS.md (trusted)")
		} else {
			report = append(report, "project AGENTS.md (not loaded — untrusted)")
		}
	}
	report = append(report, fmt.Sprintf("skills: %d global, %d repo",
		len(globalSkills), len(skills)-len(globalSkills)))

	if session != nil {
		report = append(report, "session "+session.ID)
	}
	fmt.Println("  context: " + strings.Join(report, " · "))

	conversation := newWindow()
	var ctxSize int

	var commands []Command
	commands = []Command{
		{"/help", "list commands", func(string) {
			for _, c := range commands {
				fmt.Printf("  %-10s %s\n", c.Name, c.Desc)
			}
		}},
		{"/clear", "start a fresh session", func(string) {
			session.close()
			session = openSession("")
			conversation = newWindow()
			ctxSize = 0
			fmt.Println("  session cleared")
			if session != nil {
				fmt.Println("  now logging to " + session.ID)
			}
		}},
		{"/resume", "list sessions, or resume one by id", func(arg string) {
			id := strings.TrimSpace(arg)
			if id == "" {
				printSessions()
				return
			}

			window, err := resumeWindow(id)
			if err != nil {
				fmt.Fprintln(os.Stderr, "  cannot resume:", err)
				return
			}

			s := reopenSession(id)
			if s == nil {
				return
			}

			session.close()
			session = s

			session.log(record{Kind: "system", Message: &window[0]})
			conversation = window
			ctxSize = estimateTokens(conversation)
			fmt.Printf("  resumed %s - window rebuilt: %d messages, ~%d tokens\n",
				id, len(conversation), ctxSize)
		}},
		{"/fork", "branch a session into a new log", func(arg string) {
			id := strings.TrimSpace(arg)
			if id == "" {
				if session == nil {
					fmt.Println("  no current session to fork")
					return
				}
				id = session.ID
			}

			window, err := resumeWindow(id)
			if err != nil {
				fmt.Fprintln(os.Stderr, "  cannot fork:", err)
				return
			}

			s := openSession(id)
			if s == nil {
				return
			}

			session.close()
			session = s

			session.log(record{Kind: "system", Message: &window[0]})
			conversation = []Message{window[0]}
			add(&conversation, window[1:]...)
			ctxSize = estimateTokens(conversation)
			fmt.Printf("  forked %s into %s - %d messages carried over\n",
				id, s.ID, len(conversation)-1)
		}},
		{"/compact", "compact the conversation now", func(string) {
			before := estimateTokens(conversation)
			summary, err := summarize(context.Background(), conversation)
			if err != nil {
				fmt.Fprintln(os.Stderr, "compaction failed, conversation untouched:", err)
				return
			}
			conversation = compact(conversation, summary)
			session.log(record{Kind: "compact", Summary: summary})
			fmt.Printf("  compacted: ~%d → ~%d tokens\n", before, estimateTokens(conversation))
		}},
		{"/context", "x-ray the conversation array", func(string) {
			xray(conversation)
		}},
	}

	for _, s := range skills {
		commands = append(commands, Command{"/skill:" + s.Name, s.Desc, func(arg string) {
			body := fmt.Sprintf("[Skill loaded: %s]\nSkill directory: %s\n"+
				"Files this skill refers to relatively (./foo.md) live in that "+
				"directory - read them from there, do not search the disk.\n\n%s",
				s.Name, s.Dir, s.Body)
			if arg := strings.TrimSpace(arg); arg != "" {
				body += "\n\n---\n\n" + arg
			}
			add(&conversation, Message{Role: "user", Content: body})
			fmt.Printf("  skill %s loaded into context (~%d tok)\n",
				s.Name, estimateTokens([]Message{{Content: body}}))

			runTurn(scanner, &conversation, &ctxSize)
		}})
	}

	for {
		fmt.Print("\n❯ ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(input, "/") {
			name, arg, _ := strings.Cut(input, " ")
			found := false
			for _, c := range commands {
				if c.Name == name {
					c.Run(arg)
					found = true
					break
				}
			}
			if !found {
				fmt.Printf("  unknown command %q — try /help\n", name)
			}
			continue
		}

		add(&conversation, Message{Role: "user", Content: input})

		runTurn(scanner, &conversation, &ctxSize)
	}
}
