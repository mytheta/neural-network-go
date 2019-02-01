package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/plot"

	"gonum.org/v1/plot/vg"

	"github.com/mytheta/neural-network-go/layer"

	"gonum.org/v1/plot/plotter"
)

const (
	min, max, p = 0.0, 3.9, 0.07
)

func main() {

	dots := make(plotter.XYs, 4)

	//クラス1
	x1, y1 := 0.0, 0.0
	dots[0].X = x1
	dots[0].Y = y1

	//クラス2
	x2, y2 := 20.0, 20.0
	dots[1].X = x2
	dots[1].Y = y2

	//クラス3
	x3, y3 := 0.0, 20.0
	dots[2].X = x3
	dots[2].Y = y3

	//クラス4
	x4, y4 := 20.0, 0.0
	dots[3].X = x4
	dots[3].Y = y4

	class := [][]float64{{1.0, x1, y1}, {1.0, x2, y2}, {1.0, x3, y3}, {1.0, x4, y4}}

	//教師データ作成
	b := []float64{1, 1, 0, 0}

	var errGraph []float64
	var beforeError float64
	var afterError float64
	var count int

	for {
		//0~199をランダムに生成
		rand := randomCount(min, max)

		beforeError = afterError

		errGraph = append(errGraph, beforeError)

		x := class[int(rand)]
		in1 := layer.InputLayer1(x[0])
		in2 := layer.InputLayer2(x[1])
		in3 := layer.InputLayer3(x[2])

		in := []float64{in1, in2, in3}

		mid1 := layer.MiddleLayer1(in)
		mid2 := layer.MiddleLayer2(in)
		mid3 := layer.MiddleLayer3(in)
		mid := []float64{mid1, mid2, mid3}

		out1 := layer.OutPutLayer1(mid)

		e21 := layer.OutPutErrorFunc(out1, b[int(rand)])
		e2 := []float64{e21}
		layer.OutWeightCalc1(p, e21, out1)

		afterError = e21 * e21

		e11 := layer.MiddleErrorFunc(e2, mid1)
		e12 := layer.MiddleErrorFunc(e2, mid2)
		e13 := layer.MiddleErrorFunc(e2, mid3)
		if math.IsNaN(e11) {
			panic("エラーですよ")
		}

		layer.MiddleWeightCalc1(p, e11, mid1)
		layer.MiddleWeightCalc2(p, e12, mid2)
		layer.MiddleWeightCalc3(p, e13, mid3)

		//前回と今回の誤差の二乗が閾値以下だったら終了
		if count == 500 {
			break
		}
		count++
	}
	for i, test := range class {
		in1 := layer.InputLayer1(test[0])
		in2 := layer.InputLayer2(test[1])
		in3 := layer.InputLayer3(test[2])

		in := []float64{in1, in2, in3}

		mid1 := layer.MiddleLayer1(in)
		mid2 := layer.MiddleLayer2(in)
		mid3 := layer.MiddleLayer3(in)
		mid := []float64{mid1, mid2, mid3}

		out1 := layer.OutPutLayer1(mid)

		fmt.Printf("%dのclass1の確率:%f\n", i, out1)

	}

	layer.PrintW()

	// 図の生成
	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	//label
	p.Title.Text = "Points Example"
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	// Draw a grid behind the data
	p.Add(plotter.NewGrid())

	// Make a scatter plotter and set its style.
	r, err := plotter.NewScatter(dots)
	if err != nil {
		panic(err)
	}

	r.GlyphStyle.Color = color.RGBA{R: 128, B: 0, A: 0}
	p.Add(r)

	// Axis ranges
	p.X.Min = 0
	p.X.Max = 20
	p.Y.Min = 0
	p.Y.Max = 20

	// Save the plot to a PNG file.
	if err := p.Save(6*vg.Inch, 6*vg.Inch, "report.png"); err != nil {
		panic(err)
	}

	p2, err := plot.New()
	if err != nil {
		panic(err)
	}

	// Make a line plotter and set its style.
	l, err := plotter.NewLine(lineGraph(errGraph))
	if err != nil {
		panic(err)
	}
	l.LineStyle.Width = vg.Points(1)
	l.LineStyle.Dashes = []vg.Length{vg.Points(5), vg.Points(0)}
	l.LineStyle.Color = color.RGBA{B: 255, A: 255}

	p2.Add(l)
	p2.Title.Text = "Plotutil example"
	p2.X.Label.Text = "X"
	p2.Y.Label.Text = "Y"

	p2.Legend.Add("line", l)
	// Save the plot to a PNG file.
	if err := p2.Save(4*vg.Inch, 4*vg.Inch, "points.png"); err != nil {
		panic(err)
	}

}

func randomCount(min, max float64) float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()*(max-min) + min
}

// 誤差関数の出力
func lineGraph(n []float64) plotter.XYs {
	pts := make(plotter.XYs, len(n))
	for i, m := range n {
		//fmt.Println(m)
		pts[i].X = float64(i)
		pts[i].Y = m
	}
	return pts
}
