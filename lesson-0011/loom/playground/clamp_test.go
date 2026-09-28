package playground

import "testing"

func TestClamp(t *testing.T) {
	for _, tc := range []struct{ v, lo, hi, want int }{
		{5, 0, 10, 5}, {-3, 0, 10, 0}, {15, 0, 10, 10},
	} {
		if got := Clamp(tc.v, tc.lo, tc.hi); got != tc.want {
			t.Errorf("Clamp(%d,%d,%d) = %d, want %d", tc.v, tc.lo, tc.hi, got, tc.want)
		}
	}
}
