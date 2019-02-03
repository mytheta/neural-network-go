package layer

import (
	"fmt"
	"math"

	"github.com/mytheta/neural-network-go/function"
)

var (
	wm1 = []float64{1.14, -1.09, 1.1}
	wm2 = []float64{4.5, 4.5, -5.87}
	wm3 = []float64{-3.6, 3.23, 1.51}
)

func MiddleLayer1(x []float64) (y float64) {
	h := function.InnerProduct(wm1, x)
	y = function.Sigmoid(h)
	return
}

func MiddleLayer2(x []float64) (y float64) {
	h := function.InnerProduct(wm2, x)
	y = function.Sigmoid(h)
	return
}

func MiddleLayer3(x []float64) (y float64) {
	h := function.InnerProduct(wm3, x)
	y = function.Sigmoid(h)

	return
}

//中間層の誤差関数
func MiddleErrorFunc(e, g float64) float64 {
	sigma := function.Multiplication(e, wo1)
	r := sigma * g
	d := 1 - g
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

}
