package misc

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestElmAbs(t *testing.T) {
	type args struct {
		fs []float64
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "made by myself",
			args: args{fs: []float64{1, -2, -3, 4, 5, 6, -7, -8, -9}},
			want: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ElmAbs(tt.args.fs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ElmAbs() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFloor(t *testing.T) {
	type args struct {
		fs [][]float64
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "made by myself",
			args: args{
				fs: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			},
			want: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Floor(tt.args.fs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Floor() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetMeanError(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
		fn string
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			name: "MAE",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
				fn: "MAE",
			},
			want:    3.5,
			wantErr: false,
		},
		{
			name: "MSE",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
				fn: "MSE",
			},
			want:    15.166666666666666,
			wantErr: false,
		},
		{
			name: "RMSE",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
				fn: "RMSE",
			},
			want:    1.5898986690282426,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetMeanError(tt.args.x1, tt.args.x2, tt.args.fn)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetMeanError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("GetMeanError() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetValidError(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
	}
	tests := []struct {
		name    string
		args    args
		want    []float64
		wantErr bool
	}{
		{
			//name:    "",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
			},
			want:    []float64{1, 2, 3, 4, 5, 6},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetValidError(tt.args.x1, tt.args.x2)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetValidError() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetValidError() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLinSpace(t *testing.T) {
	{
		type args struct {
			start float64
			end   float64
			n     int
		}
		tests := []struct {
			name string
			args args
			want []float64
		}{
			{
				args: args{start: 0, end: 10, n: 21},
				want: []float64{0., 0.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.,
					5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10.},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				if got := LinSpace(tt.args.start, tt.args.end, tt.args.n); !reflect.DeepEqual(got, tt.want) {
					t.Errorf("LinSpace() = %v, want %v", got, tt.want)
				}
			})
		}
	}
}

func TestLogSE(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
	}
	tests := []struct {
		name    string
		args    args
		want    []float64
		wantErr bool
	}{
		{
			//name:    "",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
			},
			want:    []float64{0, 6.020599913279624, 9.54242509439325, 12.041199826559248, 13.979400086720375, 15.563025007672874},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LogSE(tt.args.x1, tt.args.x2)
			if (err != nil) != tt.wantErr {
				t.Errorf("LogSE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("LogSE() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMAE(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			//name:    "",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
			},
			want:    3.5,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MAE(tt.args.x1, tt.args.x2)
			if (err != nil) != tt.wantErr {
				t.Errorf("MAE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("MAE() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMSE(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			//name:    "",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
			},
			want:    15.166666666666666,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MSE(tt.args.x1, tt.args.x2)
			if (err != nil) != tt.wantErr {
				t.Errorf("MSE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("MSE() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewRandn(t *testing.T) {
	rand.Seed(1)
	type args struct {
		stddev float64
		mean   float64
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			args: args{stddev: 0.5, mean: 0},
			want: -0.6168790887989735,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewRandn(tt.args.stddev, tt.args.mean); got != tt.want {
				t.Errorf("NewRandn() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewNormRand2dSlice(t *testing.T) {
	rand.Seed(1)
	type args struct {
		n int
		m int
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			//name: "made by myself",
			args: args{
				n: 2,
				m: 3,
			},
			want: [][]float64{{-1.233758177597947, -0.12634751070237293}, {-0.5209945711531503, 2.28571911769958}, {0.3228052526115799, 0.5900672875996937}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewNormRand2dSlice(tt.args.n, tt.args.m); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewNormRand2dSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewNormRandSlice(t *testing.T) {
	rand.Seed(1)
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			//name: ,
			args: args{
				n: 2,
			},
			want: []float64{-1.233758177597947, -0.12634751070237293},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewNormRandSlice(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewNormRandSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewRand2dSlice(t *testing.T) {
	rand.Seed(1)
	type args struct {
		n int
		m int
	}
	tests := []struct {
		name string
		args args
		want [][]float64
	}{
		{
			name: "made by myself",
			args: args{
				n: 2,
				m: 3,
			},
			want: [][]float64{{0.6046602879796196, 0.9405090880450124}, {0.6645600532184904, 0.4377141871869802}, {0.4246374970712657, 0.6868230728671094}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewRand2dSlice(tt.args.n, tt.args.m); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewRand2dSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestNewRandSlice(t *testing.T) {
	rand.Seed(1)
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			name: "made by myself",
			args: args{n: 2},
			want: []float64{0.6046602879796196, 0.9405090880450124},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewRandSlice(tt.args.n); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewRandSlice() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRMSE(t *testing.T) {
	type args struct {
		x1 []float64
		x2 []float64
	}
	tests := []struct {
		name    string
		args    args
		want    float64
		wantErr bool
	}{
		{
			//name:    "",
			args: args{
				x1: []float64{1, 2, 3, 4, 5, 6},
				x2: []float64{0, 0, 0, 0, 0, 0},
			},
			want:    1.5898986690282426,
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := RMSE(tt.args.x1, tt.args.x2)
			if (err != nil) != tt.wantErr {
				t.Errorf("RMSE() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("RMSE() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUnset(t *testing.T) {
	type args struct {
		s []float64
		i int
	}
	tests := []struct {
		name string
		args args
		want []float64
	}{
		{
			//name: "made by myself",
			args: args{
				s: []float64{0, 1, 2, 3, 4, 5},
				i: 4,
			},
			want: []float64{0, 1, 2, 3, 5},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Unset(tt.args.s, tt.args.i); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Unset() = %v, want %v", got, tt.want)
			}
		})
	}
}
