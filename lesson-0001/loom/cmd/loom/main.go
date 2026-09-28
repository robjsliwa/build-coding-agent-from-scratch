package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
)

const (
	ollamaURL = "http://localhost:11434/api/chat"
	// model     = "qwen3.6:27b-mlx"
	model = "gemma4:e4b-mlx"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type chatResponse struct {
	Message Message `json:"message"`
}

func chat(messages []Message) (Message, error) {
	body, err := json.Marshal(chatRequest{
		Model:    model,
		Messages: messages,
		Stream:   false,
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
	fmt.Printf("loom v0.1 — chatting with %s (ctrl-c to quit)\n", model)
	scanner := bufio.NewScanner(os.Stdin)
	var conversation []Message

	for {
		fmt.Print("\nyou: ")
		if !scanner.Scan() {
			break
		}
		conversation = append(conversation,
			Message{Role: "user", Content: scanner.Text()})

		reply, err := chat(conversation)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			continue
		}
		conversation = append(conversation, reply)

		fmt.Println("\nloom:", reply.Content)
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "\nreading standard input:", err)
	}
}
