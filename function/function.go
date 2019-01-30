package function

import "math"

// Loss function
func Sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

//内積計算
func InnerProduct(w, x []float64) (f float64) {
	if len(w) != len(x) {
		panic("エラーですよ")
	}

	for i := range w {
		f += w[i] * x[i]
	}

	return
}

func Multiplication(x float64, y []float64) []float64 {
	for i, tmp := range y {
		y[i] = x * tmp
	}
	return y
}
