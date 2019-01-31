package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/plot"

	"gonum.org/v1/plot/vg"

	"github.com/mytheta/neural-network-go/function"

	"github.com/mytheta/neural-network-go/layer"

	"gonum.org/v1/plot/plotter"
)

const (
	max, min, p = 0.0, 199.0, 0.00000002
)

func main() {

	var class [][]float64
	var test1, test2 float64

	dots := make(plotter.XYs, 2)

	//クラス1
	x1, y1 := 10.0, 10.0
	dots[0].X = x1
	dots[0].Y = y1

	//クラス2
	x2, y2 := 10.0, 10.0
	dots[1].X = x2
	dots[1].Y = y2

	//各クラスのサンプル
	n := 100
	class1, plotdata1 := randomPoints1(n, x1, y1)
	class2, plotdata2 := randomPoints2(n, x2, y2)
	class = append(class1, class2...)

	testClass1, _ := randomPoints1(10, x1, y1)
	testClass2, _ := randomPoints2(10, x2, y2)

	//教師データ作成
	b := make([]float64, n*2)
	makeTrainData(b, n)

	var errGraph []float64
	var beforeError float64
	var afterError float64
	var count int
	afterError = 10000

	for {
		//0~199をランダムに生成
		rand := randomCount(max, min)

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
		if math.IsNaN(mid3) {
			panic("エラーですよ")
		}

		out1 := function.Round(layer.OutPutLayer1(mid), 5)
		out2 := function.Round(layer.OutPutLayer2(mid), 5)
		if math.IsNaN(out1) {
			panic("エラーですよ")
		}

		if b[int(rand)] == 1 {
			test1 = 1.0
			test2 = 0.0
		} else {
			test1 = 0.0
			test2 = 1.0
		}

		e21 := layer.OutPutErrorFunc(out1, test1)
		e22 := layer.OutPutErrorFunc(out2, test2)
		e2 := []float64{e21, e22}
		layer.OutWeightCalc1(p, e21, out1)
		layer.OutWeightCalc2(p, e22, out2)

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
		if count == 700000000 {
			break
		}
		count++
	}
	for _, test := range testClass1 {
		in1 := layer.InputLayer1(test[0])
		in2 := layer.InputLayer2(test[1])
		in3 := layer.InputLayer3(test[2])

		in := []float64{in1, in2, in3}

		mid1 := layer.MiddleLayer1(in)
		mid2 := layer.MiddleLayer2(in)
		mid3 := layer.MiddleLayer3(in)
		mid := []float64{mid1, mid2, mid3}

		out1 := layer.OutPutLayer1(mid)
		out2 := layer.OutPutLayer2(mid)

		fmt.Print("クラス1の確率:")
		fmt.Println(out1)
		fmt.Print("クラス2の確率:")
		fmt.Println(out2)

	}
	fmt.Println("test2")
	for _, test := range testClass2 {
		in1 := layer.InputLayer1(test[0])
		in2 := layer.InputLayer2(test[1])
		in3 := layer.InputLayer3(test[2])

		in := []float64{in1, in2, in3}

		mid1 := layer.MiddleLayer1(in)
		mid2 := layer.MiddleLayer2(in)
		mid3 := layer.MiddleLayer3(in)
		mid := []float64{mid1, mid2, mid3}

		out1 := layer.OutPutLayer1(mid)
		out2 := layer.OutPutLayer2(mid)

		fmt.Print("クラス1の確率:")
		fmt.Println(out1)
		fmt.Print("クラス2の確率:")
		fmt.Println(out2)

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
	s, err := plotter.NewScatter(plotdata1)
	if err != nil {
		panic(err)
	}

	y, err := plotter.NewScatter(plotdata2)
	if err != nil {
		panic(err)
	}

	r, err := plotter.NewScatter(dots)
	if err != nil {
		panic(err)
	}

	s.GlyphStyle.Color = color.RGBA{R: 255, B: 128, A: 55}
	y.GlyphStyle.Color = color.RGBA{R: 155, B: 128, A: 255}
	r.GlyphStyle.Color = color.RGBA{R: 128, B: 0, A: 0}
	p.Add(s)
	p.Add(y)
	p.Add(r)
	p.Legend.Add("class1", s)
	p.Legend.Add("class2", y)

	// Axis ranges
	p.X.Min = 0
	p.X.Max = 20
	p.Y.Min = 0
	p.Y.Max = 20

	// Save the plot to a PNG file.
	if err := p.Save(6*vg.Inch, 6*vg.Inch, "report.png"); err != nil {
		panic(err)
	}
}

func randomCount(min, max float64) float64 {
	rand.Seed(time.Now().UnixNano())
	return rand.Float64()*(max-min) + min
}

//ガウス分布
func randomClass1(axis float64) float64 {
	//分散
	dispersion := 1.0
	rand.Seed(time.Now().UnixNano())
	return rand.NormFloat64()*dispersion + axis
}

func randomClass2(axis float64) float64 {
	//分散
	dispersion := 3.0
	rand.Seed(time.Now().UnixNano())
	n := rand.NormFloat64()*dispersion + axis
	if n > 8 && n < 12 {
		n = n + n
	}
	return n
}

//学習データの生成
func randomPoints1(n int, x, y float64) ([][]float64, plotter.XYs) {
	matrix := make([][]float64, n)
	pts := make(plotter.XYs, n)
	for i := range matrix {
		l := randomClass1(x)
		m := randomClass1(y)
		matrix[i] = []float64{1.0, l, m}
		pts[i].X = l
		pts[i].Y = m
	}
	return matrix, pts
}

func randomPoints2(n int, x, y float64) ([][]float64, plotter.XYs) {
	matrix := make([][]float64, n)
	pts := make(plotter.XYs, n)
	for i := range matrix {
		l := randomClass2(x)
		m := randomClass2(y)
		matrix[i] = []float64{1.0, l, m}
		pts[i].X = l
		pts[i].Y = m
	}
	return matrix, pts
}

//学習
//前半に-1,後半に1を格納
func makeTrainData(b []float64, n int) {
	for i := 0; i < n*2; i++ {
		if i >= n {
			b[i] = 1
		} else {
			b[i] = -1
		}
	}
}
