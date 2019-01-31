package layer

import (
	"fmt"
	"math"

	"github.com/mytheta/neural-network-go/function"
)

var (
	wm1 = []float64{0.0, 1.09, 0.1}
	wm2 = []float64{1.0, 1.5, 1.87}
	wm3 = []float64{3.6, 2.03, 1.01}
)

func MiddleLayer1(x []float64) (y float64) {
	h := function.InnerProduct(wm1, x)
	y = function.Sigmoid(h)
	return
}

func MiddleLayer2(x []float64) (y float64) {
	h := function.InnerProduct(wm2, x)
	y = function.Round(function.Sigmoid(h), 8)
	return
}

func MiddleLayer3(x []float64) (y float64) {
	h := function.InnerProduct(wm3, x)
	y = function.Round(function.Sigmoid(h), 8)

	return
}

func MiddleErrorFunc(e []float64, g float64) float64 {
	var h1, h2 float64
	for _, epsilon := range e {
		epsilon = function.Round(epsilon, 10)
		h1 += function.Multiplication(epsilon, wo1)
		h2 += function.Multiplication(epsilon, wo2)
	}
	r := (h1 + h2) * g
	if math.IsNaN(r) {
		panic("エラーですよ")
	}
	d := 1 - g
	if math.IsNaN(d) {
		panic("エラーですよ")
	}
	r = r * d
	if math.IsNaN(r) {
		panic("エラーですよ")
	}
	return r
}

func MiddleWeightCalc1(p, e, g float64) {
	n := p * e * g
	wm1[0] = wm1[0] - n
	wm1[1] = wm1[1] - n
	wm1[2] = wm1[2] - n
}

func MiddleWeightCalc2(p, e, g float64) {
	n := p * e * g
	wm2[0] = wm2[0] - n
	wm2[1] = wm2[1] - n
	wm2[2] = wm2[2] - n
}

func MiddleWeightCalc3(p, e, g float64) {
	n := p * e * g
	wm3[0] = wm3[0] - n
	wm3[1] = wm3[1] - n
	wm3[2] = wm3[2] - n
}

func PrintW() {
	fmt.Println(wm1)
	fmt.Println(wm2)
	fmt.Println(wm3)
	fmt.Println(wo1)
	fmt.Println(wo2)

}
