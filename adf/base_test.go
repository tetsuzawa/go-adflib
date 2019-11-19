package adf

import (
	"log"
	"math/rand"
	"reflect"
	"testing"

	"github.com/tetsuzawa/go-research/go-adf/misc"
	"gonum.org/v1/gonum/mat"
)

func init() {
	rand.Seed(1)
}

func TestAdaptiveFilter_CheckFloatParam(t *testing.T) {
	type fields struct {
		n  int
		mu float64
		w  interface{}
	}
	type args struct {
		p    float64
		low  float64
		high float64
		name string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    float64
		wantErr bool
	}{
		{
			name: "valid",
			fields: fields{
				n:  1,
				mu: 1.0,
				w:  nil,
			},
			args: args{
				p:    1.5,
				low:  0,
				high: 1000,
				name: "mu",
			},
			want:    1.5,
			wantErr: false,
		},
		{
			name: "invalid",
			fields: fields{
				n:  1,
				mu: 1.0,
				w:  nil,
			},
			args: args{
				p:    1.5,
				low:  0,
				high: 1,
				name: "mu",
			},
			want:    0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				n:  tt.fields.n,
				mu: tt.fields.mu,
				//w:  tt.fields.w,
			}
			//af, err := newAdaptiveFilter(tt.fields.n, tt.fields.mu, tt.fields.w)
			//if err != nil {
			//	log.Fatalln(err)
			//}
			got, err := af.CheckFloatParam(tt.args.p, tt.args.low, tt.args.high, tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("CheckFloatParam() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("CheckFloatParam() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAdaptiveFilter_CheckIntParam(t *testing.T) {
	type fields struct {
		w  *mat.Dense
		n  int
		mu float64
	}
	type args struct {
		p    int
		low  int
		high int
		name string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    int
		wantErr bool
	}{
		{
			name: "valid",
			fields: fields{
				n:  1,
				mu: 1.0,
				w:  nil,
			},
			args: args{
				p:    1,
				low:  0,
				high: 1000,
				name: "mu",
			},
			want:    1,
			wantErr: false,
		},
		{
			name: "invalid",
			fields: fields{
				n:  1,
				mu: 1.0,
				w:  nil,
			},
			args: args{
				p:    2,
				low:  0,
				high: 1,
				name: "mu",
			},
			want:    0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				w:  tt.fields.w,
				n:  tt.fields.n,
				mu: tt.fields.mu,
			}
			got, err := af.CheckIntParam(tt.args.p, tt.args.low, tt.args.high, tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("CheckIntParam() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("CheckIntParam() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMust(t *testing.T) {
	type args struct {
		adf AdaptiveFilter
		err error
	}
	tests := []struct {
		name string
		args args
		want AdaptiveFilter
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Must(tt.args.adf, tt.args.err); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Must() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestExploreLearning(t *testing.T) {
	rand.Seed(1)
	//creation of data
	//number of samples
	n := 64
	L := 4
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
		x[i] = xRow
		v[i] = rand.NormFloat64() * 0.1
		d[i] = x[i][0]
	}
	//targetW := NewRandSlice(L)
	//targetW := nil
	//t.Log(targetW)

	type fields struct {
		n  int
		mu float64
		w  interface{}
	}
	type args struct {
		d        []float64
		x        [][]float64
		muStart  float64
		muEnd    float64
		steps    int
		nTrain   float64
		epochs   int
		criteria string
		targetW  []float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []float64
		want1   []float64
		wantErr bool
	}{
		{
			name: "random",
			fields: fields{
				n:  L,
				mu: 1.0,
				w:  "random",
			},
			args: args{
				d:        d,
				x:        x,
				muStart:  0.000001,
				muEnd:    1.0,
				steps:    100,
				nTrain:   0.5,
				epochs:   1,
				criteria: "MSE",
				targetW:  nil,
			},
			want: []float64{0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666, 0.7203269776822666},
			//want1:   []float64{1e-06, 0.010101999999999998, 0.020203, 0.030303999999999998, 0.040404999999999996, 0.050505999999999995, 0.060606999999999994, 0.070708, 0.08080899999999999, 0.09090999999999999, 0.10101099999999999, 0.11111199999999999, 0.12121299999999999, 0.131314, 0.14141499999999999, 0.15151599999999998, 0.16161699999999998, 0.17171799999999998, 0.18181899999999998, 0.19191999999999998, 0.20202099999999998, 0.21212199999999998, 0.22222299999999998, 0.23232399999999997, 0.24242499999999997, 0.252526, 0.26262699999999994, 0.2727279999999999, 0.28282899999999994, 0.29292999999999997, 0.30303099999999994, 0.3131319999999999, 0.32323299999999994, 0.33333399999999996, 0.34343499999999993, 0.3535359999999999, 0.36363699999999993, 0.37373799999999996, 0.38383899999999993, 0.3939399999999999, 0.40404099999999993, 0.41414199999999995, 0.4242429999999999, 0.4343439999999999, 0.4444449999999999, 0.45454599999999995, 0.4646469999999999, 0.4747479999999999, 0.4848489999999999, 0.49494999999999995, 0.505051, 0.5151519999999999, 0.525253, 0.535354, 0.5454549999999999, 0.5555559999999999, 0.565657, 0.575758, 0.585859, 0.5959599999999999, 0.606061, 0.616162, 0.6262629999999999, 0.6363639999999999, 0.646465, 0.656566, 0.666667, 0.6767679999999999, 0.686869, 0.69697, 0.7070709999999999, 0.7171719999999999, 0.727273, 0.737374, 0.747475, 0.7575759999999999, 0.7676769999999999, 0.777778, 0.7878789999999999, 0.7979799999999999, 0.8080809999999999, 0.818182, 0.828283, 0.8383839999999999, 0.8484849999999999, 0.858586, 0.8686869999999999, 0.8787879999999999, 0.8888889999999999, 0.89899, 0.909091, 0.9191919999999999, 0.9292929999999999, 0.939394, 0.9494949999999999, 0.9595959999999999, 0.9696969999999999, 0.979798, 0.989899, 0.9999999999999999},
			want1:   []float64{1e-06, 0.010101999999999998, 0.020203, 0.030303999999999998, 0.040404999999999996, 0.050505999999999995, 0.060606999999999994, 0.070708, 0.08080899999999999, 0.09090999999999999, 0.10101099999999999, 0.11111199999999999, 0.12121299999999999, 0.131314, 0.14141499999999999, 0.15151599999999998, 0.16161699999999998, 0.17171799999999998, 0.18181899999999998, 0.19191999999999998, 0.20202099999999998, 0.21212199999999998, 0.22222299999999998, 0.23232399999999997, 0.24242499999999997, 0.252526, 0.26262699999999994, 0.2727279999999999, 0.28282899999999994, 0.29292999999999997, 0.30303099999999994, 0.3131319999999999, 0.32323299999999994, 0.33333399999999996, 0.34343499999999993, 0.3535359999999999, 0.36363699999999993, 0.37373799999999996, 0.38383899999999993, 0.3939399999999999, 0.40404099999999993, 0.41414199999999995, 0.4242429999999999, 0.4343439999999999, 0.4444449999999999, 0.45454599999999995, 0.4646469999999999, 0.4747479999999999, 0.4848489999999999, 0.49494999999999995, 0.505051, 0.5151519999999999, 0.525253, 0.535354, 0.5454549999999999, 0.5555559999999999, 0.565657, 0.575758, 0.585859, 0.5959599999999999, 0.606061, 0.616162, 0.6262629999999999, 0.6363639999999999, 0.646465, 0.656566, 0.666667, 0.6767679999999999, 0.686869, 0.69697, 0.7070709999999999, 0.7171719999999999, 0.727273, 0.737374, 0.747475, 0.7575759999999999, 0.7676769999999999, 0.777778, 0.7878789999999999, 0.7979799999999999, 0.8080809999999999, 0.818182, 0.828283, 0.8383839999999999, 0.8484849999999999, 0.858586, 0.8686869999999999, 0.8787879999999999, 0.8888889999999999, 0.89899, 0.909091, 0.9191919999999999, 0.9292929999999999, 0.939394, 0.9494949999999999, 0.9595959999999999, 0.9696969999999999, 0.979798, 0.989899, 0.9999999999999999},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//af := &filtBase{
			//	w:  tt.fields.w,
			//	n:  tt.fields.n,
			//	mu: tt.fields.mu,
			//}
			af, err := newAdaptiveFilter(tt.fields.n, tt.fields.mu, tt.fields.w)
			if err != nil {
				log.Fatalln(err)
			}
			got, got1, err := ExploreLearning(af, tt.args.d, tt.args.x, tt.args.muStart, tt.args.muEnd, tt.args.steps, tt.args.nTrain, tt.args.epochs, tt.args.criteria, tt.args.targetW)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExploreLearning() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ExploreLearning() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("ExploreLearning() got1 = %v, want %v", got1, tt.want1)
			}
		})
	}
}

func TestAdaptiveFilter_InitWeights(t *testing.T) {
	type fields struct {
		w  *mat.Dense
		n  int
		mu float64
	}
	type args struct {
		w interface{}
		n int
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				w:  tt.fields.w,
				n:  tt.fields.n,
				mu: tt.fields.mu,
			}
			if err := af.InitWeights(tt.args.w, tt.args.n); (err != nil) != tt.wantErr {
				t.Errorf("InitWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestPreTrainedRun(t *testing.T) {
	type fields struct {
		w  *mat.Dense
		n  int
		mu float64
	}
	type args struct {
		d      []float64
		x      [][]float64
		nTrain float64
		epochs int
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantY   []float64
		wantE   []float64
		wantW   [][]float64
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				w:  tt.fields.w,
				n:  tt.fields.n,
				mu: tt.fields.mu,
			}
			gotY, gotE, gotW, err := PreTrainedRun(af, tt.args.d, tt.args.x, tt.args.nTrain, tt.args.epochs)
			if (err != nil) != tt.wantErr {
				t.Errorf("PreTrainedRun() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(gotY, tt.wantY) {
				t.Errorf("PreTrainedRun() gotY = %v, want %v", gotY, tt.wantY)
			}
			if !reflect.DeepEqual(gotE, tt.wantE) {
				t.Errorf("PreTrainedRun() gotE = %v, want %v", gotE, tt.wantE)
			}
			if !reflect.DeepEqual(gotW, tt.wantW) {
				t.Errorf("PreTrainedRun() gotW = %v, want %v", gotW, tt.wantW)
			}
		})
	}
}

func TestAdaptiveFilter_Predict(t *testing.T) {
	type fields struct {
		w  *mat.Dense
		n  int
		mu float64
	}
	type args struct {
		x []float64
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		wantY  float64
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				w:  tt.fields.w,
				n:  tt.fields.n,
				mu: tt.fields.mu,
			}
			if gotY := af.Predict(tt.args.x); gotY != tt.wantY {
				t.Errorf("Predict() = %v, want %v", gotY, tt.wantY)
			}
		})
	}
}

func TestAdaptiveFilter_Run(t *testing.T) {
	type fields struct {
		w  *mat.Dense
		n  int
		mu float64
	}
	type args struct {
		d []float64
		x [][]float64
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []float64
		want1   []float64
		want2   [][]float64
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				w:  tt.fields.w,
				n:  tt.fields.n,
				mu: tt.fields.mu,
			}
			got, got1, got2, err := af.Run(tt.args.d, tt.args.x)
			if (err != nil) != tt.wantErr {
				t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Run() got = %v, want %v", got, tt.want)
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Run() got1 = %v, want %v", got1, tt.want1)
			}
			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Run() got2 = %v, want %v", got2, tt.want2)
			}
		})
	}
}

func TestAdaptiveFilter_GetParams(t *testing.T) {
	type fields struct {
		n  int
		mu float64
		w  interface{}
	}
	tests := []struct {
		name   string
		fields fields
		want   int
		want1  float64
		want2  []float64
	}{
		{
			name: "GetParams",
			fields: fields{
				n:  8,
				mu: 1.0,
				w:  "zeros",
			},
			want:  8,
			want1: 1.0,
			want2: []float64{0, 0, 0, 0, 0, 0, 0, 0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af, _ := newAdaptiveFilter(tt.fields.n, tt.fields.mu, tt.fields.w)
			got, got1, got2 := af.GetParams()
			if got != tt.want {
				t.Errorf("GetParams() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("GetParams() got1 = %v, want %v", got1, tt.want1)
			}
			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("GetParams() got2 = %v, want %v", got2, tt.want2)
			}
		})
	}
}
