package adf

import (
	"errors"
	"gonum.org/v1/gonum/mat"
)

//FiltAP is base struct for AP filter.
//Use NewFiltAP to make instance.
type FiltAP struct {
	filtBase
	order  int
	eps    float64
	wHist  [][]float64
	xMem   *mat.Dense
	dMem   *mat.Dense
	yMem   *mat.Dense
	eMem   *mat.Dense
	epsIDE *mat.Dense
	ide    *mat.Dense
}

//NewFiltAP is constructor of AP filter.
//This func initialize filter length `n`, update step size `mu`, projection order `order` and filter weight `w`.
func NewFiltAP(n int, mu float64, order int, eps float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(FiltAP)
	p.kind = "AP filter"
	p.n = n
	p.muMin = 0
	p.muMax = 1000
	p.mu, err = p.checkFloatParam(mu, p.muMin, p.muMax, "mu")
	if err != nil {
		return nil, err
	}
	p.order = order
	p.eps, err = p.checkFloatParam(eps, 0, 1000, "eps")
	if err != nil {
		return nil, err
	}
	err = p.initWeights(w, n)
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

	p.yMem = mat.NewDense(order, 1, nil)
	p.eMem = mat.NewDense(1, order, nil)

	return p, nil
}

//Adapt calculates the error `e` between desired value `d` and estimated value `y`,
//and update filter weights according to error `e`.
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
	af.yMem.Mul(af.xMem.T(), af.w.T())
	af.eMem.Sub(af.dMem, af.yMem.T())

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

//Run calculates the errors `e` between desired values `d` and estimated values `y` in a row,
//while updating filter weights according to error `e`.
func (af *FiltAP) Run(d []float64, x [][]float64) (y []float64, e []float64, wHist [][]float64, err error) {
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])
	af.wHist = make([][]float64, N)
	for i := 0; i < N; i++ {
		af.wHist[i] = make([]float64, af.n)
	}

	y = make([]float64, N)
	e = make([]float64, N)
	w := af.w.RawRowView(0)

	xr, _ := af.xMem.Dims()
	xCol := make([]float64, xr)
	dr, _ := af.dMem.Dims()
	dCol := make([]float64, dr)

	//adaptation loop
	for i := 0; i < N; i++ {
		copy(af.wHist[i], w)

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
		af.yMem.Mul(af.xMem.T(), af.w.T())
		af.eMem.Sub(af.dMem, af.yMem.T())
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
	wHist = af.wHist
	return y, e, wHist, nil
}
