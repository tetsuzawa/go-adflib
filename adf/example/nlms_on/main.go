package main

import (
	"fmt"
	"github.com/gonum/floats"
	"github.com/tetsuzawa/go-research/go-adf/adf"
	"github.com/tetsuzawa/go-research/go-adf/misc"
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
	mu = 1
	//length of filter
	L = 8
	//eps
	eps = 0.001
)

func main() {
	ExploreLearning_nlms()
}

func m() {
	//creation of data
	//number of samples
	n := 512
	//input value
	var x = make([]float64, L)
	//noise
	var v float64
	//desired value
	var d float64
	//output value
	var y float64
	//error
	var e float64
	var dBuf = make([]float64, 0)
	var yBuf = make([]float64, 0)
	var eBuf = make([]float64, 0)

	f, err := adf.NewFiltNLMS(L, mu, eps, "zeros")
	//identification
	if err != nil {
		log.Fatalln(err)
	}

	for i := 0; i < n; i++ {
		x = unset(x, 0)
		x = append(x, rand.NormFloat64())
		v = 0.1 * rand.NormFloat64()
		d = x[L-1] + v
		f.Adapt(d, x)
		y = f.Predict(x)
		e = d - y
		dBuf = append(dBuf, d)
		yBuf = append(yBuf, y)
		eBuf = append(eBuf, e)
	}

	name := fmt.Sprintf("nlms_ex_on_mu-%v_L-%v.csv", mu, L)
	fw, err := os.Create(name)
	if err != nil {
		log.Fatalln(err)
	}
	defer fw.Close()
	for i := 0; i < n; i++ {
		_, err = fw.Write([]byte(fmt.Sprintf("%f,%f,%f\n", dBuf[i], yBuf[i], eBuf[i])))
		if err != nil {
			log.Fatalln(err)
		}
	}
}

func ExploreLearning_nlms() {
	rand.Seed(1)
	//creation of data
	//number of samples
	n := 64
	L := 4
	mu := 1.0
	eps := 1e-5
	//input value
	var x = make([][]float64, n)
	//noise
	var v = make([]float64, n)
	//desired value
	var d = make([]float64, n)
	var xRow = make([]float64, L)
	for i := 0; i < n; i++ {
		xRow = misc.Unset(xRow, 0)
		xRow = append(xRow, rand.NormFloat64())
		x[i] = append([]float64{}, xRow...)
		v[i] = rand.NormFloat64() * 0.1
		d[i] = x[i][L-1]
	}

	af, err := adf.NewFiltNLMS(L, mu, eps, "zeros")
	checkError(err)
	es, mus, err := adf.ExploreLearning(af, d, x, 0.00001, 2.0, 100, 0.5, 100, "MSE", nil)
	checkError(err)
	res := make(map[float64]float64, len(es))
	for i:=0;i<len(es);i++{
		res[es[i]] = mus[i]
	}
	eMin := floats.Min(es)
	fmt.Printf("the step size mu with the smallest error is %.3f\n", res[eMin])
	//output:
	//the step size mu with the smallest error is 1.030

	//sort.Float64s(es)
	//for i, k := range es {
	//	fmt.Println(i, k, res[k])
	//}
}

func checkError(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}
