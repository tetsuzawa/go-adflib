package adf

import (
	"errors"

	"gonum.org/v1/gonum/floats"
)

//FiltLMS is base struct for LMS filter.
//Use NewFiltLMS to make instance.
type FiltLMS struct {
	filtBase
	wHistory [][]float64
}

//NewFiltLMS is constructor of LMS filter.
//This func initialize filter length `n`, update step size `mu` and filter weight `w`.
func NewFiltLMS(n int, mu float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(FiltLMS)
	p.kind = "LMS filter"
	p.n = n
	p.muMin = 0
	p.muMax = 2
	p.mu, err = p.checkFloatParam(mu, p.muMin, p.muMax, "mu")
	if err != nil {
		return nil, err
	}
	err = p.initWeights(w, n)
	if err != nil {
		return nil, err
	}
	return p, nil
}

//Adapt calculates the error `e` between desired value `d` and estimated value `y`,
//and update filter weights according to error `e`.
func (af *FiltLMS) Adapt(d float64, x []float64) {
	w := af.w.RawRowView(0)
	y := floats.Dot(w, x)
	e := d - y
	for i := 0; i < len(x); i++ {
		w[i] += af.mu * e * x[i]
	}
}

//Run calculates the errors `e` between desired values `d` and estimated values `y` in a row,
//while updating filter weights according to error `e`.
func (af *FiltLMS) Run(d []float64, x [][]float64) (y []float64, e []float64, wHist [][]float64, err error) {
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])
	af.wHistory = make([][]float64, N)
	for i := 0; i < N; i++ {
		af.wHistory[i] = make([]float64, af.n)
	}

	y = make([]float64, N)
	e = make([]float64, N)
	//adaptation loop
	for i := 0; i < N; i++ {
		w := af.w.RawRowView(0)
		copy(af.wHistory[i], w)
		y[i] = floats.Dot(w, x[i])
		e[i] = d[i] - y[i]
		for j := 0; j < af.n; j++ {
			w[j] += af.mu * e[i] * x[i][j]
		}
	}
	wHist = af.wHistory
	return y, e, wHist, nil
}

func (af *FiltLMS) clone() AdaptiveFilter {
	altaf := *af
	return &altaf
}
