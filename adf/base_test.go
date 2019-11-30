package adf

import (
	"reflect"
	"testing"

	"gonum.org/v1/gonum/mat"
)

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
			got, err := af.checkFloatParam(tt.args.p, tt.args.low, tt.args.high, tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkFloatParam() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("checkFloatParam() got = %v, want %v", got, tt.want)
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
			got, err := af.checkIntParam(tt.args.p, tt.args.low, tt.args.high, tt.args.name)
			if (err != nil) != tt.wantErr {
				t.Errorf("checkIntParam() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("checkIntParam() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMust(t *testing.T) {
	adf , _ := NewFiltLMS(4, 1.0, nil)
	type args struct {
		adf AdaptiveFilter
		err error
	}
	tests := []struct {
		name string
		args args
		want AdaptiveFilter
	}{
		{
			name: "success",
			args: args{
				adf: adf,
				err: nil,
			},
			want: adf,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Must(tt.args.adf, tt.args.err); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Must() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_filtBase_initWeights(t *testing.T) {
	type fields struct {
		kind  string
		n     int
		muMin float64
		muMax float64
		mu    float64
		w     *mat.Dense
	}
	type args struct {
		w []float64
		n int
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		wantErr bool
	}{
		{
			name: "zeros success",
			fields: fields{
				kind:  "Base filter",
				n:     8,
				muMin: 0,
				muMax: 1000,
				mu:    1.0,
				w:     mat.NewDense(1, 8, []float64{0, 0, 0, 0, 0, 0, 0, 0}),
			},
			args: args{
				n: 8,
				w: nil,
			},
			wantErr: false,
		},
		{
			name: "zeros failed",
			fields: fields{
				kind:  "Base filter",
				n:     8,
				muMin: 0,
				muMax: 1000,
				mu:    1.0,
				w:     nil,
			},
			args: args{
				n: 4,
				w: []float64{0, 0, 0, 0, 0, 0, 0, 0,},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			af := &filtBase{
				kind:  tt.fields.kind,
				n:     tt.fields.n,
				muMin: tt.fields.muMin,
				muMax: tt.fields.muMax,
				mu:    tt.fields.mu,
				w:     tt.fields.w,
			}
			if err := af.initWeights(tt.args.w, tt.args.n); (err != nil) != tt.wantErr {
				t.Errorf("initWeights() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

