/*
This package is designed to simplify adaptive signal processing tasks
with golang (filtering, prediction, reconstruction, classification).
For code optimisation, this library uses gonum/floats and gonum/mat for slice operations.

This package is created with reference to https://github.com/matousc89/padasip.
*/
package adf

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/tetsuzawa/go-adflib/misc"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// AdaptiveFilter is the basic Adaptive Filter interface type.
type AdaptiveFilter interface {
	initWeights(w interface{}, n int) error
	//Predict calculates the new estimated value `y` from input slice `x`.
	Predict(x []float64) (y float64)

	//Adapt calculates the error `e` between desired value `d` and estimated value `y`,
	//and update filter weights according to error `e`.
	Adapt(d float64, x []float64)

	//Run calculates the errors `e` between desired values `d` and estimated values `y` in a row,
	//while updating filter weights according to error `e`.
	Run(d []float64, x [][]float64) (y []float64, e []float64, wHist [][]float64, err error)
	checkFloatParam(p, low, high float64, name string) (float64, error)
	checkIntParam(p, low, high int, name string) (int, error)

	//SetStepSize sets the step size of adaptive filter.
	SetStepSize(mu float64)

	//GetParams returns the parameters at the time this func is called.
	//parameters contains `n`: filter length, `mu`: filter update step size and `w`: filter weights.
	GetParams() (n int, mu float64, w []float64)

	//GetParams returns the name of ADF.
	GetKindName() (kind string)
}

//Must checks whether err is nil or not. If err in not nil, this func causes panic.
func Must(af AdaptiveFilter, err error) AdaptiveFilter {
	if err != nil {
		panic(err)
	}
	return af
}

//PreTrainedRun use part of the data for few epochs of learning.
//The arg `d` is desired values.
//`x` is input matrix. columns are bunch of samples and rows are set of samples.
//`nTrain` is train to test ratio, typical value is 0.5. (that means 50% of data is used for training).
//`epochs` is number of training epochs, typical value is 1. This number describes how many times the training will be repeated.
func PreTrainedRun(af AdaptiveFilter, d []float64, x [][]float64, nTrain float64, epochs int) (y, e []float64, w [][]float64, err error) {
	var nTrainI = int(float64(len(d)) * nTrain)
	//train
	for i := 0; i < epochs; i++ {
		_, _, _, err = af.Run(d[:nTrainI], x[:nTrainI])
		if err != nil {
			return nil, nil, nil, err
		}
	}
	//run
	y, e, w, err = af.Run(d[:nTrainI], x[:nTrainI])
	if err != nil {
		return nil, nil, nil, err
	}
	return y, e, w, nil
}

//ExploreLearning searches the `mu` with the smallest error value from the input matrix `x` and desired values `d`.
//The arg `d` is desired value.
//`x` is input matrix.
//`muStart` is starting learning rate.
//`muEnd` is final learning rate.
//`steps` : how many learning rates should be tested between `muStart` and `muEnd`.
//`nTrain` is train to test ratio, typical value is 0.5. (that means 50% of data is used for training)
//`epochs` is number of training epochs, typical value is 1. This number describes how many times the training will be repeated.
//`criteria` is how should be measured the mean error. Available values are "MSE", "MAE" and "RMSE".
//`target_w` is target weights. If the slice is nil, the mean error is estimated from prediction error.
// If an slice is provided, the error between weights and `target_w` is used.
func ExploreLearning(af AdaptiveFilter, d []float64, x [][]float64, muStart, muEnd float64, steps int,
	nTrain float64, epochs int, criteria string, targetW []float64) ([]float64, []float64, error) {
	mus := misc.LinSpace(muStart, muEnd, steps)
	es := make([]float64, len(mus))
	zeros := make([]float64, int(float64(len(x))*nTrain))
	for i, mu := range mus {
		//init
		err := af.initWeights("zeros", len(x[0]))
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to init weights at InitWights()")
		}
		af.SetStepSize(mu)
		//run
		_, e, _, err := PreTrainedRun(af, d, x, nTrain, epochs)
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to pre train at PreTrainedRun()")
		}
		es[i], err = misc.GetMeanError(e, zeros, criteria)
		//fmt.Println(es[i])
		if err != nil {
			return nil, nil, errors.Wrap(err, "failed to get mean error at GetMeanError()")
		}
	}
	return es, mus, nil
}

//FiltBase is base struct for adaptive filter structs.
//It puts together some functions used by all adaptive filters.
type filtBase struct {
	kind string
	n    int
	mu   float64
	w    *mat.Dense
}

//NewFiltBase is constructor of base adaptive filter only for development.
func newFiltBase(n int, mu float64, w interface{}) (AdaptiveFilter, error) {
	var err error
	p := new(filtBase)
	p.kind = "Base filter"
	p.n = n
	p.mu, err = p.checkFloatParam(mu, 0, 1000, "mu")
	if err != nil {
		return nil, err
	}
	err = p.initWeights(w, n)
	if err != nil {
		return nil, err
	}
	return p, nil
}

//initWeights initialises the adaptive weights of the filter.
//The arg `w` is initial weights of filter.
// Possible value "random":  create random weights with stddev 0.5 and mean is 0.
// "zeros": create zero value weights.
//`n` is size of filter. Note that it is often mistaken for the sample length.
func (af *filtBase) initWeights(w interface{}, n int) error {
	if n <= 0 {
		n = af.n
	}
	switch v := w.(type) {
	case string:
		if v == "random" {
			w := make([]float64, n)
			for i := 0; i < n; i++ {
				w[i] = misc.NewRandn(0.5, 0)
			}
			af.w = mat.NewDense(1, n, w)
		} else if v == "zeros" {
			w := make([]float64, n)
			af.w = mat.NewDense(1, n, w)
		} else {
			return errors.New("impossible to understand the w")
		}
	case []float64:
		if len(v) != n {
			return errors.New("length of w is different from n")
		}
		af.w = mat.NewDense(1, n, v)
	default:
		return errors.New(`args w must be "random" or "zeros" or []float64{...}`)
	}
	return nil
}

//Predict calculates the new estimated value `y` from input slice `x`.
func (af *filtBase) Predict(x []float64) (y float64) {
	y = floats.Dot(af.w.RawRowView(0), x)
	return y
}

//Adapt is just a method to satisfy the interface.
//It is used by overriding.
func (af *filtBase) Adapt(d float64, x []float64) {
	//TODO
}

//Run is just a method to satisfy the interface.
//It is used by overriding.
func (af *filtBase) Run(d []float64, x [][]float64) ([]float64, []float64, [][]float64, error) {
	//TODO
	//measure the data and check if the dimension agree
	N := len(x)
	if len(d) != N {
		return nil, nil, nil, errors.New("the length of slice d and x must agree")
	}
	af.n = len(x[0])

	y := make([]float64, N)
	e := make([]float64, N)
	w := make([]float64, af.n)
	wHist := make([][]float64, N)
	//adaptation loop
	for i := 0; i < N; i++ {
		w = af.w.RawRowView(0)
		y[i] = floats.Dot(w, x[i])
		e[i] = d[i] - y[i]
		copy(wHist[i], w)
	}
	return y, e, wHist, nil
}

//checkFloatParam check if the value of the given parameter
//is in the given range and a float.
func (af *filtBase) checkFloatParam(p, low, high float64, name string) (float64, error) {
	if low <= p && p <= high {
		return p, nil
	} else {
		err := fmt.Errorf("parameter %v is not in range <%v, %v>", name, low, high)
		return 0, err
	}
}

//checkIntParam check if the value of the given parameter
//is in the given range and a int.
func (af *filtBase) checkIntParam(p, low, high int, name string) (int, error) {
	if low <= p && p <= high {
		return p, nil
	} else {
		err := fmt.Errorf("parameter %v is not in range <%v, %v>", name, low, high)
		return 0, err
	}
}

//SetStepSize set a update step size mu.
func (af *filtBase) SetStepSize(mu float64) {
	af.mu = mu
}

//GetParams returns the parameters at the time this func is called.
//parameters contains `n`: filter length, `mu`: filter update step size and `w`: filter weights.
func (af *filtBase) GetParams() (int, float64, []float64) {
	return af.n, af.mu, af.w.RawRowView(0)
}

//GetParams returns the name of ADF.
func (af *filtBase) GetKindName() string {
	return af.kind
}
