package main

import (
	"fmt"
	"gonum.org/v1/gonum/floats"
	"log"
	"math/rand"

	"github.com/tetsuzawa/go-research/go-adf/fdadf"
	"github.com/tetsuzawa/go-research/go-adf/misc"
)

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)
	ExploreLearning_fblms()

}

func ExploreLearning_fblms() {
	rand.Seed(1)
	//creation of data
	//number of samples
	n := 512
	L := 32
	m := n / L
	mu := 0.0000001

	//input value
	var x = make([][]float64, m)
	for i := range x {
		x[i] = make([]float64, L)
	}
	//desired value
	var d = make([][]float64, m)
	for i := range d {
		d[i] = make([]float64, L)
	}
	var xRow = make([]float64, L)
	for i := 0; i < m; i++ {
		for j := 0; j < L; j++ {
			xRow = misc.Unset(xRow, 0)
			xRow = append(xRow, rand.NormFloat64())
		}
		copy(x[i], xRow)
		copy(d[i], x[i])
	}

	af, err := fdadf.NewFiltFBLMS(L, mu, "zeros")
	check(err)
	es, mus, err := fdadf.ExploreLearning(af, d, x, 0.00001, 2.0, 100, 0.5, 100, "MSE", nil)
	check(err)

	res := make(map[float64]float64, len(es))
	for i := 0; i < len(es); i++ {
		res[es[i]] = mus[i]
	}
	eMin := floats.Min(es)
	fmt.Printf("the step size mu with the smallest error is %.3f\n", res[eMin])
	//output:
	//the step size mu with the smallest error is 1.313
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
