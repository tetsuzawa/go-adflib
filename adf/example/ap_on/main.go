package main

import (
	"fmt"
	"github.com/tetsuzawa/go-adf/adf"
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
	L = 64
	//number of order
	order = 1
	//eps
	eps = 0.001
)

func main() {
	//creation of data
	//number of samples
	n := 512
	//input value
	var x = make([]float64, L)
	//noise
	//var v float64
	//desired value
	var d float64
	//output value
	var y float64
	//error
	var e float64
	var dBuf = make([]float64, 0)
	var yBuf = make([]float64, 0)
	var eBuf = make([]float64, 0)

	f, err := adf.NewFiltAP(L, mu, order, eps, "zeros")
	//identification
	if err != nil {
		log.Fatalln(err)
	}

	for i := 0; i < n; i++ {
		x = unset(x, 0)
		x = append(x, rand.NormFloat64())
		//v = 0.1 * rand.NormFloat64()
		//d = x[L-1] + v
		d = x[L-1]
		f.Adapt(d, x)
		y = f.Predict(x)
		e = d - y
		dBuf = append(dBuf, d)
		yBuf = append(yBuf, y)
		eBuf = append(eBuf, e)
	}

	name := fmt.Sprintf("ap_ex_on_mu-%v_L-%v_order-%v.csv", mu, L, order)
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
