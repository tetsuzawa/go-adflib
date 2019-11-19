package adf

import (
	"errors"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type FiltRLS struct {
	filtBase
	kind     string
	wHistory [][]float64
	eps      float64
	R        *mat.Dense
}

func NewFiltRLS(n int, mu float64, eps float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(FiltRLS)
	p.kind = "RLS filter"
	p.n = n
	p.mu, err = p.checkFloatParam(mu, 0, 1, "mu")
	if err != nil {
		return nil, err
	}
	p.eps, err = p.checkFloatParam(mu, 0, 1, "eps")
	if err != nil {
		return nil, err
	}
	err = p.initWeights(w, n)
	if err != nil {
		return nil, err
	}
	var Rs = make([]float64, n*n)
	for i := 0; i < n; i++ {
		Rs[i*(n+1)] = 1 / eps
	}
	p.R = mat.NewDense(n, n, Rs)
	return p, nil
}

func (af *FiltRLS) Adapt(d float64, x []float64) {
	w := af.w.RawRowView(0)
	y := floats.Dot(w, x)
	e := d - y
	for i := 0; i < len(x); i++ {
		w[i] += af.mu * e * x[i]
	}
}

func (af *FiltRLS) Run(d []float64, x [][]float64) ([]float64, []float64, [][]float64, error) {
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])
	af.wHistory = make([][]float64, N)

	y := make([]float64, N)
	e := make([]float64, N)
	w := af.w.RawRowView(0)
	R1 := mat.NewDense(af.n, af.n, nil)
	var R2 float64
	xVec := mat.NewDense(1, af.n, nil)
	aux1 := mat.NewDense(af.n, 1, nil)
	aux4 := mat.NewDense(1, af.n, nil)
	//aux2 := mat.NewDense(af.n, af.n, nil)
	var aux2 float64
	aux3 := mat.NewDense(af.n, af.n, nil)
	dwT := mat.NewDense(af.n, 1, nil)
	//adaptation loop
	for i := 0; i < N; i++ {
		af.wHistory[i] = w
		y[i] = floats.Dot(w, x[i])
		e[i] = d[i] - y[i]
		xVec.SetRow(0, x[i])
		aux1.Mul(af.R, xVec.T())
		aux2 = floats.Dot(mat.Col(nil, 0, aux1), mat.Row(nil, 0, xVec))
		R1 = mat.DenseCopyOf(af.R)
		R1.Scale(aux2, R1)
		//R1.Product(aux1.T(), xVec.T())
		aux4.Mul(xVec, af.R)
		R2 = af.mu + mat.Dot(aux4.RowView(0), mat.DenseCopyOf(xVec.T()).ColView(0))
		//for j:=0;j<af.n;j++{
		//	floats.AddConst(af.R.RawRowView(j))
		//}
		R1.Scale(1/R2, R1)
		aux3.Sub(af.R, R1)
		af.R.Scale(1/af.mu, aux3)
		dwT.Mul(af.R, xVec.T())
		dwT.Scale(e[i], dwT)
		floats.Add(w, mat.Col(nil, 0, dwT))
	}
	return y, e, af.wHistory, nil
}
