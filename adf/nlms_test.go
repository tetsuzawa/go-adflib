package adf

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"

	"github.com/gonum/floats"
	"github.com/tetsuzawa/go-adf/misc"
)

func TestFiltNLMS_Run(t *testing.T) {
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
		x[i] = append([]float64{}, xRow...)
		v[i] = rand.NormFloat64() * 0.1
		d[i] = x[i][0]
	}
	//type fields struct {
	//	filtBase filtBase
	//	kind           string
	//	eps            float64
	//	wHistory       [][]float64
	//}
	type args struct {
		d []float64
		x [][]float64
	}
	tests := []struct {
		name string
		//fields  fields
		args    args
		want    []float64
		want1   []float64
		want2   [][]float64
		wantErr bool
	}{
		{
			name: "Run NLMS Filter",
			args: args{
				d: d,
				x: x,
			},
			want:    []float64{-0.3916108209122005, 0.16536933657985348, -0.0437579947155992, 0.012124947341295034, -0.2654555121700104, 0.3802942691699313, -0.015105143735420865, 0.06338390297796329, -0.35634062205002437, 1.5687950887982662, -0.13433136700109533, 0.2233351641247334, 0.19138298538219461, -0.24457092974407102, 0.014496883486147707, -0.16389871128240147, -0.5703468062521424, 0.12325158184348353, -0.0001354304685272777, 0.41477231714128937, 0.1284912512030528, -0.10594739642904036, -0.6771124260933329, 0.10022071550746274, -0.0007792286060597231, -0.16438259032950334, 0.804033784658881, 0.05122531204556498, -0.3013910495723021, -0.405364406872704, -0.3028385843710033, 0.8416786588506451, -0.1280995968578023, 0.1724792387475776, 0.6525066762526404, -0.17330306888775004, 0.023775619911505473, -0.009171657096201392, 0.7178877185849712, 0.2772808354985652, 0.27493273050125877, 0.31136108806500473, 1.238565646136781, 0.004408117980679305, 0.013802766783230742, -0.7373018741864878, 0.03857756458510833, 0.801736882960992, -0.049745493036917665, 0.5057272250487606, 0.022230842171991626, -0.07618246803444377, -0.8511889527808415, -0.05578977998644839, -0.028604330951522903, 0.0024669022429527057, 0.2697704378237238, 0.17988582851926682, 0.02031316996455955, 0.05439092562840408, 0.034151431454050384, -0.01906135272886425, 0.027614795022829132, 0.6631737158019635,},
			want1:   []float64{0.3916108209122005, -0.16536933657985348, 0.0437579947155992, -1.245883124939242, -0.2555390589831399, -0.057489016558351425, 0.1739128839118565, -0.7946669191554423, 1.9417445843306473, -0.269954241280832, 0.8667732928056084, 0.4767857383152514, 0.8082431356290678, -0.07196631315001722, 1.0862323102638731, 1.1536091314909331, -0.8647001158800858, 0.01409879172986725, -0.8459589429870699, -0.25864582185847473, 0.1512524674601187, 0.8086976579912216, -0.39243297625968754, 0.230068670105199, -1.1199348637505735, 1.0916810610935865, 0.16900059718430427, 0.45889079844045116, 0.5529310936314513, 0.5720551257910016, -1.3358024934854311, 0.31255215485939214, -0.6424613509919914, -0.9486924486213446, 0.7677728205362397, -0.15715369358721953, -0.2708103367417415, 1.744804913750763, -0.9237216458418311, -1.3289461331025092, -0.952456276764398, -2.2729989991499133, 0.7473536358443511, -0.04554622604967179, 1.1899288477031864, -0.10129088833286226, -1.4416702929587735, 0.17254628092762958, -1.9432168769545648, -0.12449348474881006, 0.38734035238745806, 2.6064209113868504, -0.4190537657146517, -0.3648380326348751, 0.03076499955416054, 2.380393443188988, -0.07850432002311614, -0.1698796305344279, -0.4786260181707988, -0.5308485045945931, 2.696970439570396, 0.07153199137613384, 0.7722616071307364, 0.06216377214174784,},
			want2:   [][]float64{{-0.6507507226658572, 0.657011901126719, -0.5912859617953181, 0.31741294852014135,}, {0, 0, 0, -0.3174108632530106,}, {-0, -0, 0.11375181193994623, 0.04803540722617919,}, {0, -0.028446918274250707, -0.012012637691915408, 0.00744296151894114,}, {0.799322070609082, 0.3375397763936327, -0.2091377124027139, -0.10288769226536705,}, {0.14229189613619306, -0.0881632439569553, -0.04337291734306064, 0.1997250120164878,}, {-0.005839971201952878, -0.0028730407010865646, 0.013229870705023707, -0.028682040977572033,}, {0.005801643435633767, -0.026715595258033083, 0.05791876693399332, 0.047449900543642703,}, {0.11023407771369016, -0.2389848249165679, -0.19578811452692585, -0.11040877250498643,}, {0.5889353192819465, 0.4824847593598468, 0.2720826550784235, 0.26007625628822134,}, {-0.09443521340261717, -0.05325387610081662, -0.050903901702756854, -0.07268011799980124,}, {0.2986037814549917, 0.2854270646981752, 0.40753011161572267, -0.12904670577901747,}, {0.11916457912415569, 0.17014207914186727, -0.05387644790369343, 0.18735039997486194,}, {0.24553074938938638, -0.07774869506092075, 0.27036394721539364, 0.24309522935080186,}, {0.0052359469091108795, -0.018207524546209313, -0.016371126405936333, 0.023737584328683278,}, {0.28005271511885127, 0.2518067949270409, -0.36511140900779226, 0.03494533011408528,}, {0.30255663653537435, -0.43869697758591447, 0.041988309112148466, -0.25865289745084535,}, {0.4402666024234409, -0.04213854012862557, 0.2595783380918343, -0.04789897846985562,}, {0.0023125316258581015, -0.014245465419369593, 0.0026286601818602284, 0.004709970418815463,}, {0.5453934641846012, -0.10063932911588348, -0.1803231419474703, -0.4529936749782048,}, {-0.023202040685922194, -0.04157286135384428, -0.10443608646483826, 0.1589457055055494,}, {0.023182802367556037, 0.058238020518918555, -0.08863491128368477, 0.027371601361957227,}, {0.18925540462373483, -0.288035820059052, 0.08894916834053446, -0.3018158948030631,}, {0.12458770457330179, -0.03847428665288717, 0.13054817115989817, -0.1080178435367605,}, {0.023957919926665624, -0.08129228384242893, 0.06726265959009288, 0.07058016642844833,}, {0.37771951719421365, -0.3125317447694616, -0.3279463329941081, -0.17192682083670394,}, {0.4752248907176659, 0.4986637769335893, 0.2614259383597243, 0.12891004753909283,}, {0.1266817679483245, 0.06641328604738443, 0.03274862439172055, 0.021701879570942367,}, {0.07709299913316188, 0.03801482838903416, 0.025191691038400802, -0.24764510000358553,}, {0.03385313704451124, 0.022433818729324415, -0.22053403537766364, 0.15534041135945476,}, {0.020555559310402226, -0.20206994176351817, 0.14233461888624752, -0.0950221546141974,}, {0.4198384168920495, -0.2957270167973296, 0.19742644855900418, 0.1988746221536197,}, {0.07936191209968624, -0.05298176888393917, -0.05337040373632984, 0.09765472836307586,}, {0.14899302004300946, 0.15008592202743543, -0.2746203686057948, 0.06389598534959591,}, {0.2639440185586067, -0.4829527159423341, 0.11236872129720016, 0.08400183745167616,}, {0.2097081756222304, -0.04879285023640367, -0.036475355659889366, 0.25627102596722373,}, {0.016102960062266677, 0.01203785375526083, -0.08457636879731653, 0.010030163962305896,}, {0.015846116731118947, -0.11133272092383922, 0.013203279605345062, 0.06745938903539302,}, {0.6555080789410107, -0.07773865920102146, -0.3971893810389761, -0.25588479394794394,}, {0.03485205419179252, 0.17806926405276916, 0.1147191216981039, 0.33214694823604407,}, {0.14936566050917263, 0.09622714777094615, 0.2786070272895655, -0.28205565586963344,}, {0.07819456776917938, 0.22639719227905086, -0.22919956175026027, 0.004747844701543868,}, {0.48241999866101243, -0.4883914467295121, 0.010116977209409633, -0.2960302717271558,}, {0.2433972185588744, -0.005041947662031054, 0.1475311355882997, -0.10277917524280915,}, {0.00045449353241463195, -0.013298818521979212, 0.00926476701980472, 0.015501358724478859,}, {0.282509393121383, -0.19681325103055097, -0.3292983838054229, 0.22865906489960452,}, {0.01118679267846193, 0.018717198815094125, -0.012996897006115605, 0.02658603537441859,}, {0.2875274194465035, -0.19965403444701374, 0.40840588487821716, -0.07812395527232115,}, {0.03211725807239562, -0.06569803229495916, 0.012567375560764484, 0.013501520135258111,}, {0.3623767823499851, -0.06931904897619132, -0.07447159758883326, -0.4600687295888074,}, {-0.00569848810659633, -0.0061220619643087195, -0.03782071771756654, 0.018986942284127357,}, {0.018975786901951813, 0.11722813066438043, -0.05885144136138155, -0.01948804955856776,}, {0.8049818257448452, -0.40412092597835164, -0.13382048852058107, 0.0006874051578618607,}, {0.07127284567086212, 0.02360126984970256, -0.0001212343102773835, -0.13370140618191662,}, {0.026047654801344036, -0.00013380082869668639, -0.14756019895788533, -0.011844280530609095,}, {1.1631883555357885e-05, 0.012828045000935567, 0.0010296744293068001, 5.386801550659771e-05,}, {0.9573617317701056, 0.07684498259311218, 0.004020189873722311, -0.1841363397113727,}, {-0.03169403097370452, -0.0016580916291254234, 0.07594539887006133, 0.0789520979411163,}, {-0.00021527435597031152, 0.009860188992863058, 0.01025055656649305, -0.0587576322947086,}, {0.027771180463491585, 0.028870649078365063, -0.16549062206058432, -0.003179425540023825,}, {0.03036845016966141, -0.17407622862761518, -0.0033443732358867423, -0.05098251708661009,}, {0.8537326682551111, 0.016402013697243114, 0.2500366689344085, 0.2267362418123674,}, {0.0015215023968540317, 0.023194188110515004, 0.02103276717962851, -0.033038727386681066,}, {0.17676338782682077, 0.1602911541602286, -0.251788825482059, -0.22433053452929985,}},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			//af := &FiltNLMS{
			//	filtBase: tt.fields.filtBase,
			//	kind:           tt.fields.kind,
			//	eps:            tt.fields.eps,
			//	wHistory:       tt.fields.wHistory,
			//}
			af := Must(NewFiltNLMS(L, 1.0, 1e-5, "random"))
			got, got1, got2, err := af.Run(tt.args.d, tt.args.x)
			if (err != nil) != tt.wantErr {
				t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Run() got = %v\n, want %v\n", got, tt.want)
				for i := 0; i < n; i++ {
					fmt.Printf("%g, ", got[i])
				}
				fmt.Println("")
			}
			if !reflect.DeepEqual(got1, tt.want1) {
				t.Errorf("Run() got1 = %v\n, want %v\n", got1, tt.want1)
				for i := 0; i < n; i++ {
					fmt.Printf("%g, ", got1[i])
				}
				fmt.Println("")
			}
			if !reflect.DeepEqual(got2, tt.want2) {
				t.Errorf("Run() got2 = %v\n, want %v\n", got2, tt.want2)
				for i := 0; i < n; i++ {
					fmt.Print("{")
					for k := 0; k < L; k++ {
						fmt.Printf("%g, ", got2[i][k])
					}
					fmt.Print("}, ")
				}
				fmt.Println("")
			}
		})
	}
}

/*
func TestNewFiltNLMS(t *testing.T) {
	type args struct {
		n   int
		mu  float64
		eps float64
		w   interface{}
	}
	tests := []struct {
		name    string
		args    args
		want    AdaptiveFilter
		wantErr bool
	}{
		{
			name: "TestNewFiltNLMS",
			args: args{
				n:   4,
				mu:  1.0,
				eps: 1e-5,
				w:   "zeros",
			},
			want: &FiltLMS{
				filtBase: filtBase{
					w:  mat.NewDense(1, 4, []float64{0, 0, 0, 0}),
					n:  4,
					mu: 1.0,
				},
				kind:     "LMS Filter",
				wHistory: nil,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewFiltNLMS(tt.args.n, tt.args.mu, tt.args.eps, tt.args.w)
			if (err != nil) != tt.wantErr {
				t.Errorf("NewFiltNLMS() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewFiltNLMS() got = %v, want %v", got, tt.want)
			}
		})
	}
}
*/

func ExampleExploreLearning_nlms() {
	rand.Seed(1)
	//creation of data
	//number of samples
	//n := 64
	n := 512
	L := 8
	mu := 1.0
	eps := 0.001
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
		x[i] = append([]float64{}, xRow...)
		v[i] = rand.NormFloat64() * 0.1
		d[i] = x[i][L-1]
	}

	af, err := NewFiltNLMS(L, mu, eps, "zeros")
	check(err)
	es, mus, err := ExploreLearning(af, d, x, 0.00001, 2.0, 100, 0.5, 100, "MSE", nil)
	check(err)

	res := make(map[float64]float64, len(es))
	for i := 0; i < len(es); i++ {
		res[es[i]] = mus[i]
	}
	eMin := floats.Min(es)
	fmt.Printf("the step size mu with the smallest error is %.3f\n", res[eMin])
	//output:
	//the step size mu with the smallest error is 1.313
}
