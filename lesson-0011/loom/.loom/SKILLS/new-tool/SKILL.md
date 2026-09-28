---
name: new-tool
description: How to add a new tool to loom's registry
---
To add a tool to loom, in cmd/loom/main.go:

1. Write the run function: func myTool(args map[string]any) string —
   validate arguments, return failures as "error: ..." strings.
2. Declare a ToolDef: the Tool schema (name, description, JSON
   parameters) plus Run set to your function.
3. Decide the Safe flag: true only if the tool cannot modify anything.
   The zero value is gated — leave it unset when in doubt.
4. Append the def to the registry slice.
5. go build ./cmd/loom, then test with a prompt that forces the tool.