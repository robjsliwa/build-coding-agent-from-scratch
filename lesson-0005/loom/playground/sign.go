package playground

// Sign returns the sign of x: -1 if x < 0, 0 if x == 0, and 1 if x > 0.
func Sign(x float64) int {
	if x < 0 {
		return -1
	} else if x > 0 {
		return 1
	}
	return 0
}
