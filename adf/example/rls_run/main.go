package main

import (
	"fmt"
	"github.com/tetsuzawa/go-adf/adf"
	"github.com/tetsuzawa/go-adf/misc"
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
	L = 16
	//eps
	eps = 0.001
)

func check(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}

func main() {
	run()
}

func run() {
	rand.Seed(1)
	//creation of data
	//number of samples
	n := 128
	//L := 4
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
		d[i] = x[i][0]
	}

	af := adf.Must(adf.NewFiltRLS(L, mu, 1e-5, "random"))

	y, e, _, err := af.Run(d, x)
	check(err)
	//fmt.Println(y)
	//fmt.Println(e)
	//fmt.Println(w)

	name := fmt.Sprintf("rls_ex_off_mu-%v_L-%v.csv", mu, L)
	fw, err := os.Create(name)
	check(err)
	defer fw.Close()
	for i := 0; i < n; i++ {
		_, err = fw.Write([]byte(fmt.Sprintf("%f,%f,%f\n", d[i], y[i], e[i])))
		check(err)
	}
}
