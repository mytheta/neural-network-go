package layer

import "github.com/mytheta/neural-network-go/function"

var (
	wm1 = []float64{18.6, 11.1, 81.1}
	wm2 = []float64{18.6, 11.1, 81.1}
	wm3 = []float64{18.6, 11.1, 81.1}
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

func MiddleErrorFunc1(e []float64, g float64) float64 {
	h := function.InnerProduct(e, wo1)
	return h * g * (1 - g)
}

func MiddleErrorFunc2(e []float64, g float64) float64 {
	h := function.InnerProduct(e, wo2)
	return h * g * (1 - g)
}

func MiddleErrorFunc3(e []float64, g float64) float64 {
	h := function.InnerProduct(e, wo3)
	return h * g * (1 - g)
}

func MiddleWeightCalc1(p, e, g float64) {
	wm1[0] = wm1[0] - p*e*g
	wm1[1] = wm1[1] - p*e*g
	wm1[2] = wm1[2] - p*e*g
}

func MiddleWeightCalc2(p, e, g float64) {
	wm2[0] = wm2[0] - p*e*g
	wm2[1] = wm2[1] - p*e*g
	wm2[2] = wm2[2] - p*e*g
}

func MiddleWeightCalc3(p, e, g float64) {
	wm3[0] = wm3[0] - p*e*g
	wm3[1] = wm3[1] - p*e*g
	wm3[2] = wm3[2] - p*e*g
}
