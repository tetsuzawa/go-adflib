package main

import (
	"fmt"
	"github.com/tetsuzawa/go-adf/adf"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"math/rand"
	"os"
)

func init() {
	rand.Seed(1)
}

func unset(s []float64, i int) []float64 {
	if i >= len(s) {
		return s
	}
	return append(s[:i], s[i+1:]...)
}

const (
	//step size of filter
	mu = 0.1
	//length of filter
	L = 64
)

func main() {
	//creation of data
	//number of samples
	n := 512
	//input value
	var x = make([][]float64, n)
	//noise
	var v = make([]float64, n)
	//desired value
	var d = make([]float64, n)
	var xRow = make([]float64, L)
	for i := 0; i < n; i++ {
		xRow = unset(xRow, 0)
		xRow = append(xRow, rand.NormFloat64())
		x[i] = xRow
		v[i] = rand.NormFloat64() * 0.1
		d[i] = x[i][0]
	}

	//identification
	f, err := adf.NewFiltLMS(L, mu, "zeros")
	if err != nil {
		log.Fatalln(err)
	}
	y, e, _, err := f.Run(d, x)
	if err != nil {
		log.Fatalln(err)
	}

	// show results
	p, err := plot.New()
	if err != nil {
		log.Fatalln(err)
	}
	//label
	p.Title.Text = "LMS Sample"
	p.X.Label.Text = "sample"
	p.Y.Label.Text = "y"

	p.Add(plotter.NewGrid())

	ptsD := make(plotter.XYs, n)
	ptsY := make(plotter.XYs, n)
	ptsE := make(plotter.XYs, n)
	for i := 0; i < n; i++ {
		ptsD[i].X = float64(i)
		ptsD[i].Y = d[i]
		ptsY[i].X = float64(i)
		ptsY[i].Y = y[i]
		ptsE[i].X = float64(i)
		ptsE[i].Y = e[i]
	}

	plotD, err := plotter.NewLine(ptsD)
	//plotD, err := plotter.NewScatter(ptsD)
	if err != nil {
		log.Fatalln(err)
	}
	plotY, err := plotter.NewLine(ptsY)
	//plotY, err := plotter.NewScatter(ptsY)
	if err != nil {
		log.Fatalln(err)
	}
	plotE, err := plotter.NewLine(ptsE)
	//plotE, err := plotter.NewScatter(ptsE)
	if err != nil {
		log.Fatalln(err)
	}
	plotD.Color = color.RGBA{R: 87, G: 209, B: 201, A: 1}
	plotY.Color = color.RGBA{R: 237, G: 84, B: 133, A: 1}
	plotE.Color = color.RGBA{R: 255, G: 232, B: 105, A: 1}
	//plotD.GlyphStyle.Color = color.RGBA{R: 87, G: 209, B: 201, A: 1}
	//plotY.GlyphStyle.Color = color.RGBA{R: 237, G: 84, B: 133, A: 1}
	//plotE.GlyphStyle.Color = color.RGBA{R: 255, G: 232, B: 105, A: 1}

	// \plot
	p.Add(plotD)
	p.Add(plotY)
	p.Add(plotE)

	//label
	p.Legend.Add("Desired", plotD)
	p.Legend.Add("Output", plotY)
	p.Legend.Add("Error", plotE)

	//座標範囲
	//p.Y.Min = -10
	//p.Y.Max = 10

	name := fmt.Sprintf("lms_ex_mu-%v_L-%v.png", mu, L)
	if err := p.Save(20*vg.Centimeter, 20*vg.Centimeter, name); err != nil {
		log.Fatalln(err)
	}
	fw, err := os.Create(name)
	if err != nil{
		log.Fatalln(err)
	}
	defer fw.Close()
	for i:=0; i<n; i++ {
		_, err = fw.Write([]byte(fmt.Sprintf("%f,%f,%f\n", d[i], y[i], e[i])))
		if err != nil{
			log.Fatalln(err)
		}
	}
}
