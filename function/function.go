package function

import (
	"math"
)

// Loss function
func Sigmoid(x float64) (y float64) {
	return 1.0 / (1.0 + math.Exp(-1*x))
}

//内積計算
func InnerProduct(w, x []float64) (f float64) {
	if len(w) != len(x) {
		panic("エラーですよ")
	}

	for i, _ := range w {
		n := w[i] * x[i]
		f += n
	}

	return
}

func Multiplication(x float64, y []float64) (f float64) {
	for _, tmp := range y {
		n := x * tmp
		f += n
	}
	return
}

//func Round(f float64, places int) float64 {
//	shift := math.Pow(10, float64(places))
//	return math.Floor(f*shift+.5) / shift
//}

// RoundDown 切り捨て

func Round(num, places float64) float64 {

	shift := math.Pow(10, places)

	return math.Trunc(num*shift) / shift

}
