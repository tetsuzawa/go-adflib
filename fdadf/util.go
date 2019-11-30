package fdadf

func float64sToComplex128s(fs []float64) []complex128 {
	cs := make([]complex128, len(fs))
	for i, f := range fs {
		cs[i] = complex(f, 0)
	}
	return cs
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
