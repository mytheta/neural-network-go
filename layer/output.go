package layer

import "github.com/mytheta/neural-network-go/function"

var (
	wo1 = []float64{18.6, 11.1, 81.1}
	wo2 = []float64{18.6, 11.1, 81.1}
)

func OutPutLayer1(x []float64) (h float64) {
	h = function.InnerProduct(wo1, x)
	return
}

func OutPutLayer2(x []float64) (h float64) {
	h = function.InnerProduct(wo2, x)
	return
}

func ErrorFunc(g, b float64) float64 {
	return (g - b) * g * (1 - g)
}

func OutWeightCalc1(p, e, g float64) {
	wo1[0] = wo1[0] - p*e*g
	wo1[1] = wo1[1] - p*e*g
	wo1[2] = wo1[2] - p*e*g
}

func OutWeightCalc2(p, e, g float64) {
	wo2[0] = wo2[0] - p*e*g
	wo2[1] = wo2[1] - p*e*g
	wo2[2] = wo2[2] - p*e*g
}
