package adf

import (
	"errors"

	"gonum.org/v1/gonum/floats"
)

//FiltNLMS is base struct for NLMS filter.
//Use NewFiltNLMS to make instance.
type FiltNLMS struct {
	filtBase
	eps      float64
	wHistory [][]float64
}

//NewFiltLMS is constructor of LMS filter.
//This func initialize filter length `n`, update step size `mu` and filter weight `w`.
func NewFiltNLMS(n int, mu float64, eps float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(FiltNLMS)
	p.kind = "NLMS filter"
	p.n = n
	p.muMin = 0
	p.muMax = 2
	p.mu, err = p.checkFloatParam(mu, p.muMin, p.muMax, "mu")
	if err != nil {
		return nil, err
	}
	p.eps, err = p.checkFloatParam(eps, 0, 1, "eps")
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
func (af *FiltNLMS) Adapt(d float64, x []float64) {
	w := af.w.RawRowView(0)
	y := floats.Dot(w, x)
	e := d - y
	nu := af.mu / (af.eps + floats.Dot(x, x))
	for i := 0; i < len(x); i++ {
		w[i] += nu * e * x[i]
	}
}

//Run calculates the errors `e` between desired values `d` and estimated values `y` in a row,
//while updating filter weights according to error `e`.
func (af *FiltNLMS) Run(d []float64, x [][]float64) (y []float64, e []float64, wHist [][]float64, err error) {
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
	w := af.w.RawRowView(0)
	//adaptation loop
	for i := 0; i < N; i++ {
		copy(af.wHistory[i], w)
		y[i] = floats.Dot(w, x[i])
		e[i] = d[i] - y[i]
		nu := af.mu / (af.eps + floats.Dot(x[i], x[i]))
		for j := 0; j < af.n; j++ {
			w[j] += nu * e[i] * x[i][j]
		}
	}
	wHist = af.wHistory
	return y, e, af.wHistory, nil
}

func (af *FiltNLMS) clone() AdaptiveFilter {
	altaf := *af
	return &altaf
}
