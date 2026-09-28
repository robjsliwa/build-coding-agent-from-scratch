package playground

import "testing"

func TestAbs(t *testing.T) {
	tests := []struct {
		input int
		want  int
	}{
		{-5, 5},
		{5, 5},
		{0, 0},
		{-10, 10},
	}

	for _, tt := range tests {
		t.Run(string(rune(tt.input)), func(t *testing.T) {
			got := Abs(tt.input)
			if got != tt.want {
				t.Errorf("Abs(%d) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}
