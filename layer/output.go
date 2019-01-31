package layer

import (
	"github.com/mytheta/neural-network-go/function"
)

var (
	wo1 = []float64{2.6, 1.1, 3.1}
	wo2 = []float64{0.001, 0.07, 0.23}
)

func OutPutLayer1(x []float64) (y float64) {
	h := function.InnerProduct(wo1, x)
	y = function.Sigmoid(h)
	return
}

func OutPutLayer2(x []float64) (y float64) {
	h := function.InnerProduct(wo2, x)
	y = function.Sigmoid(h)
	return
}

func OutPutErrorFunc(g, b float64) float64 {
	return (g - b) * g * (1 - g)
}

func OutWeightCalc1(p, e, g float64) {
	n := p * e * g
	wo1[0] = wo1[0] - n
	wo1[1] = wo1[1] - n
	wo1[2] = wo1[2] - n
}

func OutWeightCalc2(p, e, g float64) {
	n := p * e * g
	wo2[0] = wo2[0] - n
	wo2[1] = wo2[1] - n
	wo2[2] = wo2[2] - n
}
