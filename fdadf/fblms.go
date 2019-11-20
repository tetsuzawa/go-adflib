package fdadf

import (
	"math/cmplx"

	"github.com/mjibson/go-dsp/fft"
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

//FiltFBLMS is base struct for FBLMS filter
//(Fast Block Least Mean Square filter).
//Use NewFiltFBLMS to make instance.
type FiltFBLMS struct {
	filtBase
	wHistory [][]float64
	xMem     *mat.Dense
}

//NewFiltFBLMS is constructor of FBLMS filter.
//This func initialize filter length `n`, update step size `mu` and filter weight `w`.
func NewFiltFBLMS(n int, mu float64, w interface{}) (FDAdaptiveFilter, error) {
	var err error
	p := new(FiltFBLMS)
	p.kind = "FBLMS filter"
	p.n = n
	p.mu, err = p.checkFloatParam(mu, 0, 1000, "mu")
	if err != nil {
		return nil, errors.Wrap(err, "Parameter error at checkFloatParam()")
	}
	err = p.initWeights(w, 2*n)
	if err != nil {
		return nil, err
	}
	p.xMem = mat.NewDense(1, n, make([]float64, n))
	return p, nil
}

//Adapt calculates the error `e` between desired value `d` and estimated value `y`,
//and update filter weights according to error `e`.
func (af *FiltFBLMS) Adapt(d []float64, x []float64) {
	zeros := make([]float64, af.n)
	Y := make([]complex128, 2*af.n)
	y := make([]float64, af.n)
	e := make([]float64, af.n)
	EU := make([]complex128, 2*af.n)

	w := af.w.RawRowView(0)
	// 1 compute the output of the filter for the block kM, ..., KM + M -1
	W := fft.FFT(float64sToComplex128s(append(w[:af.n], zeros...)))
	xSet := append(af.xMem.RawRowView(0), x...)
	U := fft.FFT(float64sToComplex128s(xSet))
	af.xMem.SetRow(0, x)
	for i := 0; i < 2*af.n; i++ {
		Y[i] = W[i] * U[i]
	}
	yc := fft.IFFT(Y)[af.n:]
	for i := 0; i < af.n; i++ {
		y[i] = real(yc[i])
		e[i] = d[i] - y[i]
	}

	// 2 compute the correlation vector
	aux1 := fft.FFT(float64sToComplex128s(append(zeros, e...)))
	aux2 := fft.FFT(float64sToComplex128s(xSet))
	for i := 0; i < 2*af.n; i++ {
		EU[i] = aux1[i] * cmplx.Conj(aux2[i])
	}
	phi := fft.IFFT(EU)[:af.n]

	// 3 update the parameters of the filter
	aux1 = fft.FFT(float64sToComplex128s(append(w[:af.n], zeros...)))
	aux2 = fft.FFT(append(phi, float64sToComplex128s(zeros)...))
	for i := 0; i < 2*af.n; i++ {
		W[i] = aux1[i] + complex(af.mu, 0)*aux2[i]
	}
	aux3 := fft.IFFT(W)
	for i := 0; i < 2*af.n; i++ {
		w[i] = real(aux3[i])
	}
}

//Predict calculates the new output value `y` from input array `x`.
func (af *FiltFBLMS) Predict(x []float64) (y []float64) {
	zeros := make([]float64, af.n)
	y = make([]float64, af.n)
	Y := make([]complex128, 2*af.n)
	W := fft.FFT(float64sToComplex128s(append(af.w.RawRowView(0)[:af.n], zeros...)))
	U := fft.FFT(float64sToComplex128s(append(af.xMem.RawRowView(0), x...)))
	for i := 0; i < 2*af.n; i++ {
		Y[i] = W[i] * U[i]
	}
	yc := fft.IFFT(Y)[af.n:]
	for i := 0; i < af.n; i++ {
		y[i] = real(yc[i])
	}
	return
}

//Run calculates the errors `e` between desired values `d` and estimated values `y` in a row,
//while updating filter weights according to error `e`.
//The arg `x`: rows are samples sets, columns are input values.
func (af *FiltFBLMS) Run(d [][]float64, x [][]float64) ([][]float64, [][]float64, [][]float64, error) {
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])
	af.wHistory = make([][]float64, N)
	for i := range af.wHistory {
		af.wHistory[i] = make([]float64, af.n)
	}

	zeros := make([]float64, af.n)
	Y := make([]complex128, 2*af.n)
	y := make([][]float64, N)
	for i := range y {
		y[i] = make([]float64, af.n)
	}
	e := make([][]float64, N)
	for i := range e {
		e[i] = make([]float64, af.n)
	}

	EU := make([]complex128, 2*af.n)

	for k := 0; k < N; k++ {
		w := af.w.RawRowView(0)
		copy(af.wHistory[k], w)

		// 1 compute the output of the filter for the block kM, ..., KM + M -1
		W := fft.FFT(float64sToComplex128s(append(w[:af.n], zeros...)))
		xSet := append(af.xMem.RawRowView(0), x[k]...)
		U := fft.FFT(float64sToComplex128s(xSet))
		af.xMem.SetRow(0, x[k])

		for i := 0; i < 2*af.n; i++ {
			Y[i] = W[i] * U[i]
		}
		yc := fft.IFFT(Y)[af.n:]
		for i := 0; i < af.n; i++ {
			y[k][i] = real(yc[i])
			e[k][i] = x[k][i] - y[k][i]
		}

		// 2 compute the correlation vector
		aux1 := fft.FFT(float64sToComplex128s(append(zeros, e[k]...)))
		aux2 := fft.FFT(float64sToComplex128s(xSet))
		for i := 0; i < 2*af.n; i++ {
			EU[i] = aux1[i] * cmplx.Conj(aux2[i])
		}
		phi := fft.IFFT(EU)[:af.n]

		// 3 update the parameters of the filter
		aux1 = fft.FFT(float64sToComplex128s(append(w[:af.n], zeros...)))
		aux2 = fft.FFT(append(phi, float64sToComplex128s(zeros)...))
		for i := 0; i < 2*af.n; i++ {
			W[i] = aux1[i] + complex(af.mu, 0)*aux2[i]
		}
		aux3 := fft.IFFT(W)
		for i := 0; i < 2*af.n; i++ {
			w[i] = real(aux3[i])
		}
	}

	return y, e, af.wHistory, nil
}
