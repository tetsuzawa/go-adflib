package adf

import (
	"errors"

	"gonum.org/v1/gonum/mat"
)

type FiltAP struct {
	filtBase
	kind     string
	order    int
	eps      float64
	wHistory *mat.Dense
	xMem     *mat.Dense
	dMem     *mat.Dense
	yMem     *mat.Dense
	eMem     *mat.Dense
	epsIDE   *mat.Dense
	ide      *mat.Dense
}

func NewFiltAP(n int, mu float64, order int, eps float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(FiltAP)
	p.kind = "AP filter"
	p.n = n
	p.mu, err = p.CheckFloatParam(mu, 0, 1000, "mu")
	if err != nil {
		return nil, err
	}
	p.order = order
	p.eps, err = p.CheckFloatParam(eps, 0, 1000, "eps")
	if err != nil {
		return nil, err
	}
	err = p.InitWeights(w, n)
	if err != nil {
		return nil, err
	}
	p.xMem = mat.NewDense(n, order, nil)
	p.dMem = mat.NewDense(1, order, nil)

	elmNum := order * order

	//make diagonal matrix
	diaMat := make([]float64, elmNum)
	for i := 0; i < order; i++ {
		diaMat[i*(order+1)] = eps
	}
	p.epsIDE = mat.NewDense(order, order, diaMat)

	for i := 0; i < order; i++ {
		diaMat[i*(order+1)] = 1
	}
	p.ide = mat.NewDense(order, order, diaMat)
	//p.wHistory = mat.NewDense(len(x), order, nil)
	p.yMem = mat.NewDense(1, order, nil)
	p.eMem = mat.NewDense(1, order, nil)

	return p, nil
}

func (af *FiltAP) Adapt(d float64, x []float64) {
	xr, _ := af.xMem.Dims()
	xCol := make([]float64, xr)
	dr, _ := af.dMem.Dims()
	dCol := make([]float64, dr)
	// create input matrix and target vector
	// shift column
	for i := af.order - 1; i > 0; i-- {
		mat.Col(xCol, i-1, af.xMem)
		af.xMem.SetCol(i, xCol)
		mat.Col(dCol, i-1, af.dMem)
		af.dMem.SetCol(i, dCol)
	}
	af.xMem.SetCol(0, x)
	af.dMem.Set(0, 0, d)

	// estimate output and error
	//wd := mat.NewDense(1, len(af.w), af.w)
	af.yMem.Mul(af.w, af.xMem)
	af.eMem.Sub(af.dMem, af.yMem)

	// update
	dw1 := mat.NewDense(af.order, af.order, nil)
	dw1.Mul(af.xMem.T(), af.xMem)
	dw1.Add(dw1, af.epsIDE)
	dw2 := mat.NewDense(af.order, af.order, nil)
	err := dw2.Solve(dw1, af.ide)
	if err != nil {
		panic(err)
	}
	dw3 := mat.NewDense(1, af.order, nil)
	dw3.Mul(af.eMem, dw2)
	dw := mat.NewDense(1, af.n, nil)
	dw.Mul(dw3, af.xMem.T())
	dw.Scale(af.mu, dw)
	af.w.Add(af.w, dw)
}

func (af *FiltAP) Run(d []float64, x [][]float64) ([]float64, []float64, [][]float64, error) {
	//TODO
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])
	af.wHistory = mat.NewDense(N, af.n, nil)

	y := make([]float64, N)
	e := make([]float64, N)

	xr, _ := af.xMem.Dims()
	xCol := make([]float64, xr)
	dr, _ := af.dMem.Dims()
	dCol := make([]float64, dr)

	//adaptation loop
	for i := 0; i < N; i++ {
		//af.wHistory[i] = af.w
		af.wHistory.SetRow(i, af.w.RawRowView(0))

		// create input matrix and target vector
		// shift column
		for i := af.order - 1; i > 0; i-- {
			mat.Col(xCol, i-1, af.xMem)
			af.xMem.SetCol(i, xCol)
			mat.Col(dCol, i-1, af.dMem)
			af.dMem.SetCol(i, dCol)
		}
		af.xMem.SetCol(0, x[i])
		af.dMem.Set(0, 0, d[i])

		// estimate output and error
		// same as af.yMem.Mul(af.xMem, af.w.T()).T()
		af.yMem.Mul(af.w, af.xMem.T())
		af.eMem.Sub(af.dMem, af.yMem)
		y[i] = af.yMem.At(0, 0)
		e[i] = af.eMem.At(0, 0)

		// update
		dw1 := mat.NewDense(af.order, af.order, nil)
		dw1.Mul(af.xMem.T(), af.xMem)
		dw1.Add(dw1, af.epsIDE)
		dw2 := mat.NewDense(af.order, af.order, nil)
		err := dw2.Solve(dw1, af.ide)
		if err != nil {
			return nil, nil, nil, err
		}
		dw3 := mat.NewDense(1, af.order, nil)
		dw3.Mul(af.eMem, dw2)
		dw := mat.NewDense(af.n, 1, nil)
		dw.Mul(af.xMem, dw3.T())
		dw.Scale(af.mu, dw)
		af.w.Add(af.w, dw.T())
	}
	wHistory := make([][]float64, N)
	for i := 0; i < N; i++ {
		wHistory[i] = af.wHistory.RawRowView(i)
	}
	return y, e, wHistory, nil
}
