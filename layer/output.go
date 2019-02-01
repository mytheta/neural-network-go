package layer

import (
	"github.com/mytheta/neural-network-go/function"
)

var (
	wo1 = []float64{0.6, 0.1, -1.1}
)

func OutPutLayer1(x []float64) (y float64) {
	h := function.InnerProduct(wo1, x)
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
