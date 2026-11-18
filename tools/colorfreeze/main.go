// Command colorfreeze is the runnable numerical companion for the color
// configuration freeze analysis (docs/color-freeze.md). It replays the exact
// production paint arithmetic under hypothetical future color configurations
// and quantifies what each freeze granularity can promise across machines.
//
// Usage:
//
//	go run ./tools/colorfreeze            # full text report
//	go run ./tools/colorfreeze -json      # machine-readable report
//
// The command never writes outside its process and needs no network.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"prismplate/internal/geom"
	"prismplate/internal/raster"
	"prismplate/internal/scene"
	"prismplate/internal/validate"

	"prismplate/tools/colorfreeze/paintmodel"
)

type layerSpec struct {
	opacity float64
	blend   string
	brush   paintmodel.Brush
}

type sceneSpec struct {
	name   string
	width  int
	height int
	bg     paintmodel.Color
	layers []layerSpec
}

// gradientScene is the transparent-gradient control: alpha falls 1.0 -> 0.25
// along the full diagonal while hue also shifts, so interpolation space and
// premultiplication both matter.
func gradientScene() sceneSpec {
	w, h := 64, 48
	return sceneSpec{
		name: "transparent-gradient", width: w, height: h,
		bg: paintmodel.Color{R: 0.04, G: 0.055, B: 0.09, A: 1},
		layers: []layerSpec{{
			opacity: 1, blend: "normal",
			brush: paintmodel.Brush{
				Kind: "linear", FromX: 0, FromY: 0, ToX: float64(w), ToY: float64(h),
				Stops: []paintmodel.Stop{
					{At: 0, Color: paintmodel.Color{R: 1, G: 0.7, B: 0.22, A: 1}},
					{At: 1, Color: paintmodel.Color{R: 0.9, G: 0.18, B: 0.14, A: 0.25}},
				},
			},
		}},
	}
}

// blendScene adds the screen-blended cyan glaze at group opacity 0.72 so the
// mixed-background path (unpremultiply inside paint.Over) is exercised too.
func blendScene() sceneSpec {
	base := gradientScene()
	base.name = "gradient-plus-screen-glaze"
	base.layers = append(base.layers, layerSpec{
		opacity: 0.72, blend: "screen",
		brush: paintmodel.Brush{Kind: "solid", Color: paintmodel.Color{R: 0.1, G: 0.65, B: 0.84, A: 0.82}},
	})
	return base
}

// renderLinear evaluates the ported pipeline at pixel centers. All shapes are
// full-canvas rectangles in the diagnostic scenes, so the four coverage
// samples per production pixel are identical and one center evaluation is
// exactly equivalent to raster's 2x2 accumulation.
func renderLinear(cfg paintmodel.Config, s sceneSpec, naive bool) [][]paintmodel.Pixel {
	out := make([][]paintmodel.Pixel, s.height)
	for y := 0; y < s.height; y++ {
		row := make([]paintmodel.Pixel, s.width)
		for x := 0; x < s.width; x++ {
			px, py := float64(x)+0.5, float64(y)+0.5
			dst := cfg.From(s.bg)
			for _, l := range s.layers {
				var src paintmodel.Pixel
				if naive {
					src = cfg.NaiveSRGBSample(l.brush, px, py)
				} else {
					src = cfg.Sample(l.brush, px, py)
				}
				dst = paintmodel.Over(dst, src.Scale(l.opacity), l.blend)
			}
			row[x] = dst
		}
		out[y] = row
	}
	return out
}

type nrgba struct{ R, G, B, A uint8 }

func quantize(cfg paintmodel.Config, grid [][]paintmodel.Pixel) [][]nrgba {
	out := make([][]nrgba, len(grid))
	for y, row := range grid {
		orow := make([]nrgba, len(row))
		for x, p := range row {
			r, g, b, a := cfg.RGBA8(p)
			orow[x] = nrgba{r, g, b, a}
		}
		out[y] = orow
	}
	return out
}

func maxByteDelta(a, b [][]nrgba) (int, int, int, int) {
	var dr, dg, db, da int
	for y := range a {
		for x := range a[y] {
			dr = max(dr, absInt(int(a[y][x].R)-int(b[y][x].R)))
			dg = max(dg, absInt(int(a[y][x].G)-int(b[y][x].G)))
			db = max(db, absInt(int(a[y][x].B)-int(b[y][x].B)))
			da = max(da, absInt(int(a[y][x].A)-int(b[y][x].A)))
		}
	}
	return dr, dg, db, da
}

func maxLinearDelta(a, b [][]paintmodel.Pixel) (float64, float64) {
	var dc, da float64
	for y := range a {
		for x := range a[y] {
			dc = max(dc, math.Abs(a[y][x].R-b[y][x].R), math.Abs(a[y][x].G-b[y][x].G), math.Abs(a[y][x].B-b[y][x].B))
			da = max(da, math.Abs(a[y][x].A-b[y][x].A))
		}
	}
	return dc, da
}

func pngBytes(img [][]nrgba) ([]byte, error) {
	nrgbaImg := image.NewNRGBA(image.Rect(0, 0, len(img[0]), len(img)))
	for y, row := range img {
		for x, p := range row {
			nrgbaImg.SetNRGBA(x, y, color.NRGBA{R: p.R, G: p.G, B: p.B, A: p.A})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, nrgbaImg); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func sha(b []byte) string { v := sha256.Sum256(b); return hex.EncodeToString(v[:])[:16] }

// ---- port fidelity: ported arithmetic vs production raster package ----

func productionScene(s sceneSpec) scene.Compiled {
	rect := func() []geom.Op {
		return []geom.Op{
			{Kind: "M", Points: []geom.Point{{X: 0, Y: 0}}},
			{Kind: "L", Points: []geom.Point{{X: float64(s.width), Y: 0}}},
			{Kind: "L", Points: []geom.Point{{X: float64(s.width), Y: float64(s.height)}}},
			{Kind: "L", Points: []geom.Point{{X: 0, Y: float64(s.height)}}},
			{Kind: "Z"},
		}
	}
	doc := scene.Document{
		Width: s.width, Height: s.height,
		Background: scene.Color{R: s.bg.R, G: s.bg.G, B: s.bg.B, A: s.bg.A},
	}
	for i, l := range s.layers {
		sl := scene.Layer{ID: fmt.Sprintf("l%d", i), Opacity: l.opacity, Blend: l.blend, Shapes: []scene.Shape{{
			ID:        fmt.Sprintf("s%d", i),
			Path:      rect(),
			FillRule:  "nonzero",
			Transform: geom.Matrix{1, 0, 0, 1, 0, 0},
			Brush: scene.Brush{
				Kind:  l.brush.Kind,
				Color: scene.Color{R: l.brush.Color.R, G: l.brush.Color.G, B: l.brush.Color.B, A: l.brush.Color.A},
				From:  geom.Point{X: l.brush.FromX, Y: l.brush.FromY},
				To:    geom.Point{X: l.brush.ToX, Y: l.brush.ToY},
			},
		}}}
		for _, st := range l.brush.Stops {
			sl.Shapes[0].Brush.Stops = append(sl.Shapes[0].Brush.Stops, scene.Stop{At: st.At, Color: scene.Color{R: st.Color.R, G: st.Color.G, B: st.Color.B, A: st.Color.A}})
		}
		doc.Layers = append(doc.Layers, sl)
	}
	compiled, err := validate.Compile(doc)
	if err != nil {
		panic(err)
	}
	return compiled
}

func verifyPort() (mismatches int, err error) {
	cfg := paintmodel.ProductionConfig()
	for _, spec := range []sceneSpec{gradientScene(), blendScene()} {
		prod, err := raster.Render(context.Background(), productionScene(spec))
		if err != nil {
			return 0, err
		}
		prodImg, err := png.Decode(bytes.NewReader(prod))
		if err != nil {
			return 0, err
		}
		port := quantize(cfg, renderLinear(cfg, spec, false))
		var nrgbaProd *image.NRGBA
		switch p := prodImg.(type) {
		case *image.NRGBA:
			nrgbaProd = p
		case *image.RGBA:
			// The PNG encoder promotes fully opaque images to straight RGBA.
			nrgbaProd = image.NewNRGBA(p.Bounds())
			for y := 0; y < spec.height; y++ {
				for x := 0; x < spec.width; x++ {
					nrgbaProd.SetNRGBA(x, y, color.NRGBAModel.Convert(p.At(x, y)).(color.NRGBA))
				}
			}
		default:
			return 0, fmt.Errorf("unexpected decoded image type %T", prodImg)
		}
		for y := 0; y < spec.height; y++ {
			for x := 0; x < spec.width; x++ {
				got := nrgbaProd.NRGBAAt(x, y)
				want := port[y][x]
				if got.R != want.R || got.G != want.G || got.B != want.B || got.A != want.A {
					mismatches++
				}
			}
		}
	}
	return mismatches, nil
}

// ---- report data structures ----

type FreezeRow struct {
	Scene    string  `json:"scene"`
	Strategy string  `json:"strategy"`
	Config   string  `json:"config"`
	MaxDR    int     `json:"max_d_r"`
	MaxDG    int     `json:"max_d_g"`
	MaxDB    int     `json:"max_d_b"`
	MaxDA    int     `json:"max_d_a"`
	MaxLinDC float64 `json:"max_linear_d_color"`
	MaxLinDA float64 `json:"max_linear_d_alpha"`
}

type MidpointRow struct {
	Scene  string  `json:"scene"`
	Method string  `json:"method"`
	Config string  `json:"config"`
	R      uint8   `json:"r"`
	G      uint8   `json:"g"`
	B      uint8   `json:"b"`
	A      uint8   `json:"a"`
	LinR   float64 `json:"linear_r_premult"`
	LinG   float64 `json:"linear_g_premult"`
	LinB   float64 `json:"linear_b_premult"`
	LinA   float64 `json:"linear_a"`
}

type BoundaryRow struct {
	Case           string  `json:"case"`
	Config         string  `json:"config"`
	Input          float64 `json:"input"`
	R              uint8   `json:"r"`
	G              uint8   `json:"g"`
	B              uint8   `json:"b"`
	A              uint8   `json:"a"`
	ReachableCodes int     `json:"reachable_codes,omitempty"`
	Note           string  `json:"note"`
}

type StorageRow struct {
	Scene    string `json:"scene"`
	Level    string `json:"level"`
	Bytes    int    `json:"bytes"`
	Editable bool   `json:"editable_svg_possible"`
	Note     string `json:"note"`
}

type Report struct {
	Configs      []string                     `json:"configs"`
	PortMismatch int                          `json:"port_vs_raster_mismatches"`
	Boundaries   []BoundaryRow                `json:"quantization_boundaries"`
	Freeze       []FreezeRow                  `json:"freeze_replay"`
	Midpoints    []MidpointRow                `json:"gradient_midpoints"`
	Storage      []StorageRow                 `json:"storage_footprint"`
	PNGDigests   map[string]map[string]string `json:"png_digest_prefixes"`
	ExampleJSON  int                          `json:"example_scene_json_bytes"`
	Level1At512  int                          `json:"level1_linear_grid_bytes_at_512_sq_one_layer"`
	RawNRGBA512  int                          `json:"raw_nrgba_bytes_at_512_sq"`
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	flag.Parse()

	cfgs := []paintmodel.Config{paintmodel.ProductionConfig(), paintmodel.AltGamma(), paintmodel.AltKnee()}
	scenes := []sceneSpec{gradientScene(), blendScene()}

	rep := Report{Configs: []string{cfgs[0].Name, cfgs[1].Name, cfgs[2].Name}, PNGDigests: map[string]map[string]string{}}

	// 1. Port fidelity against the production raster package.
	mismatch, err := verifyPort()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	rep.PortMismatch = mismatch

	// 2. Quantization boundary probes through each config.
	transparent := paintmodel.Pixel{R: 0.6, G: 0.3, B: 0.1, A: 0}
	for _, cfg := range cfgs {
		r, g, b, a := cfg.RGBA8(transparent)
		rep.Boundaries = append(rep.Boundaries, BoundaryRow{Case: "zero-alpha-ignores-rgb", Config: cfg.Name, Input: 0, R: r, G: g, B: b, A: a, Note: "paint.RGBA8 returns 0,0,0,0 whenever A<=0"})
	}
	probes := []float64{0.0, 0.0031308, 0.04045, 0.5, 1.0}
	for _, cfg := range cfgs {
		for _, v := range probes {
			// Opaque gray so the transfer function is isolated from alpha math.
			r, g, b, a := cfg.RGBA8(cfg.From(paintmodel.Color{R: v, G: v, B: v, A: 1}))
			note := ""
			if cfg.Name == "srgb-iec" {
				switch v {
				case 0.0031308:
					note = "production linear-segment encode knee"
				case 0.04045:
					note = "production decode knee"
				case 1.0:
					note = "clamped max; encode can never yield 256"
				}
			} else if v == 1.0 {
				note = "clamped max; encode can never yield 256"
			}
			rep.Boundaries = append(rep.Boundaries, BoundaryRow{Case: "opaque-gray", Config: cfg.Name, Input: v, R: r, G: g, B: b, A: a, Note: note})
		}
	}
	// Round-trip reachability: every 8-bit code must decode/encode back to itself.
	for _, cfg := range cfgs {
		worst := 0
		distinct := map[uint8]bool{}
		for k := 0; k <= 255; k++ {
			r, _, _, _ := cfg.RGBA8(cfg.From(paintmodel.Color{R: float64(k) / 255, G: 0, B: 0, A: 1}))
			distinct[r] = true
			worst = max(worst, absInt(int(r)-k))
		}
		rep.Boundaries = append(rep.Boundaries, BoundaryRow{Case: "code-roundtrip", Config: cfg.Name, Input: -1, ReachableCodes: len(distinct), Note: fmt.Sprintf("max |reencoded-k| over k=0..255 = %d; distinct reachable codes=%d", worst, len(distinct))})
	}

	// 3. Freeze-granularity replay. Reference world is always config[0].
	for _, s := range scenes {
		refLin := renderLinear(cfgs[0], s, false)
		ref8 := quantize(cfgs[0], refLin)
		rep.PNGDigests[s.name] = map[string]string{}
		for _, cfg := range cfgs {
			// L0: only input colors are frozen; the machine re-runs everything.
			l0lin := renderLinear(cfg, s, false)
			l08 := quantize(cfg, l0lin)
			dc, da := maxLinearDelta(l0lin, refLin)
			d := maxByteDeltaRow(l08, ref8)
			rep.Freeze = append(rep.Freeze, freeze(s.name, "L0-input-colors", cfg, d, dc, da))
			if b, err := pngBytes(l08); err == nil {
				rep.PNGDigests[s.name]["L0/"+cfg.Name] = sha(b)
			}

			// L1: converted linear premultiplied materials are frozen; the replay
			// machine only re-encodes through its own output transfer function.
			l1lin := refLin
			l18 := quantize(cfg, l1lin)
			dc, da = maxLinearDelta(l1lin, refLin)
			d = maxByteDeltaRow(l18, ref8)
			rep.Freeze = append(rep.Freeze, freeze(s.name, "L1-linear-materials", cfg, d, dc, da))
			if b, err := pngBytes(l18); err == nil {
				rep.PNGDigests[s.name]["L1/"+cfg.Name] = sha(b)
			}

			// L2: final pixels are frozen; replay is identity by construction.
			dc, da = 0.0, 0.0
			d = maxByteDeltaRow(ref8, ref8)
			rep.Freeze = append(rep.Freeze, freeze(s.name, "L2-final-pixels", cfg, d, dc, da))
			if b, err := pngBytes(ref8); err == nil {
				rep.PNGDigests[s.name]["L2/"+cfg.Name] = sha(b)
			}
		}

		// 4. Gradient midpoint: canonical premultiplied-linear vs naive sRGB mix.
		mx, my := s.width/2, s.height/2
		for _, cfg := range cfgs {
			lin := renderLinear(cfg, s, false)
			naive := renderLinear(cfg, s, true)
			lin8 := quantize(cfg, lin)
			naive8 := quantize(cfg, naive)
			p, n := lin[my][mx], naive[my][mx]
			rep.Midpoints = append(rep.Midpoints,
				MidpointRow{Scene: s.name, Method: "canonical-linear-premult", Config: cfg.Name, R: lin8[my][mx].R, G: lin8[my][mx].G, B: lin8[my][mx].B, A: lin8[my][mx].A, LinR: p.R, LinG: p.G, LinB: p.B, LinA: p.A},
				MidpointRow{Scene: s.name, Method: "naive-sRGB-mix", Config: cfg.Name, R: naive8[my][mx].R, G: naive8[my][mx].G, B: naive8[my][mx].B, A: naive8[my][mx].A, LinR: n.R, LinG: n.G, LinB: n.B, LinA: n.A},
			)
		}

		// 5. Storage footprint of each freeze level.
		l0 := len(mustMarshal(sceneJSON(s)))
		rawNRGBA := s.width * s.height * 4
		pngB := len(must(pngBytes(ref8)))
		l1perLayer := len(s.layers) * s.width * s.height * 4 * 8 // float64 premult RGBA grid per layer
		rep.Storage = append(rep.Storage,
			StorageRow{Scene: s.name, Level: "L0 input scene JSON", Bytes: l0, Editable: true, Note: "same size as a revision document; smallest and only editable form"},
			StorageRow{Scene: s.name, Level: "L1 linear premult grid (float64, per layer)", Bytes: l1perLayer, Editable: false, Note: "geometry, brushes, blend modes and SVG are not recoverable from a flattened grid"},
			StorageRow{Scene: s.name, Level: "L2 raw NRGBA grid", Bytes: rawNRGBA, Editable: false, Note: "8-bit quantized; identical information class to the PNG bytes"},
			StorageRow{Scene: s.name, Level: "L2 PNG artifact", Bytes: pngB, Editable: false, Note: "what the service already persists per artifact today"},
		)
		rep.Level1At512 = 512 * 512 * 4 * 8
		rep.RawNRGBA512 = 512 * 512 * 4
	}

	// Reference size of the real revision document shipped in the repo.
	if b, err := os.ReadFile("examples/amber-proof.json"); err == nil {
		rep.ExampleJSON = len(b)
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	printText(rep)
}

func freeze(sceneName, strategy string, cfg paintmodel.Config, d [4]int, dc, da float64) FreezeRow {
	return FreezeRow{Scene: sceneName, Strategy: strategy, Config: cfg.Name, MaxDR: d[0], MaxDG: d[1], MaxDB: d[2], MaxDA: d[3], MaxLinDC: dc, MaxLinDA: da}
}

func maxByteDeltaRow(a, b [][]nrgba) [4]int {
	r, g, bl, al := maxByteDelta(a, b)
	return [4]int{r, g, bl, al}
}

func sceneJSON(s sceneSpec) any {
	// JSON size proxy: marshal the same structural shape as a revision document.
	type stopJSON struct {
		At    float64          `json:"at"`
		Color paintmodel.Color `json:"color"`
	}
	type brushJSON struct {
		Kind  string           `json:"kind"`
		Color paintmodel.Color `json:"color"`
		From  [2]float64       `json:"from"`
		To    [2]float64       `json:"to"`
		Stops []stopJSON       `json:"stops,omitempty"`
	}
	type layerJSON struct {
		ID      string    `json:"id"`
		Opacity float64   `json:"opacity"`
		Blend   string    `json:"blend"`
		Brush   brushJSON `json:"brush"`
	}
	type doc struct {
		Width      int              `json:"width"`
		Height     int              `json:"height"`
		Background paintmodel.Color `json:"background"`
		Layers     []layerJSON      `json:"layers"`
	}
	d := doc{Width: s.width, Height: s.height, Background: s.bg}
	for i, l := range s.layers {
		lj := layerJSON{ID: fmt.Sprintf("layer-%d", i), Opacity: l.opacity, Blend: l.blend, Brush: brushJSON{
			Kind:  l.brush.Kind,
			Color: l.brush.Color,
			From:  [2]float64{l.brush.FromX, l.brush.FromY},
			To:    [2]float64{l.brush.ToX, l.brush.ToY},
		}}
		for _, st := range l.brush.Stops {
			lj.Brush.Stops = append(lj.Brush.Stops, stopJSON{At: st.At, Color: st.Color})
		}
		d.Layers = append(d.Layers, lj)
	}
	return d
}

func mustMarshal(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func printText(r Report) {
	fmt.Println("prism-plate color-freeze numerical report")
	fmt.Println("==========================================")
	fmt.Printf("configs under test: %v\n", r.Configs)
	fmt.Printf("ported-pipeline vs internal/raster pixel mismatches: %d (0 means the port is arithmetic-faithful)\n\n", r.PortMismatch)

	fmt.Println("[1] quantization boundaries (NRGBA8 output codes)")
	for _, b := range r.Boundaries {
		if b.Case == "code-roundtrip" {
			fmt.Printf("  %-22s %-9s %-19s codes=%d/256  %s\n", b.Case, b.Config, "", b.ReachableCodes, b.Note)
			continue
		}
		fmt.Printf("  %-22s %-9s in=%-9g -> (%3d,%3d,%3d,%3d)  %s\n", b.Case, b.Config, b.Input, b.R, b.G, b.B, b.A, b.Note)
	}

	fmt.Println("\n[2] freeze replay vs reference config srgb-iec (max per-channel byte delta over the whole canvas; linear deltas in 0..1 premult space)")
	fmt.Printf("  %-26s %-22s %-10s %5s %5s %5s %5s %12s %12s\n", "scene", "strategy", "config", "dR", "dG", "dB", "dA", "linColor", "linAlpha")
	for _, f := range r.Freeze {
		fmt.Printf("  %-26s %-22s %-10s %5d %5d %5d %5d %12.6f %12.6f\n", f.Scene, f.Strategy, f.Config, f.MaxDR, f.MaxDG, f.MaxDB, f.MaxDA, f.MaxLinDC, f.MaxLinDA)
	}

	fmt.Println("\n[3] gradient midpoint (center pixel): interpolation-space comparison")
	for _, m := range r.Midpoints {
		fmt.Printf("  %-26s %-24s %-9s NRGBA=(%3d,%3d,%3d,%3d) linPremult=(%.6f,%.6f,%.6f) a=%.6f\n", m.Scene, m.Method, m.Config, m.R, m.G, m.B, m.A, m.LinR, m.LinG, m.LinB, m.LinA)
	}

	fmt.Println("\n[4] storage footprint per freeze level")
	for _, s := range r.Storage {
		fmt.Printf("  %-26s %-42s %8d bytes  editableSVG=%-5v %s\n", s.Scene, s.Level, s.Bytes, s.Editable, s.Note)
	}
	fmt.Printf("  extrapolated one-layer 512x512: L1 float64 grid=%d bytes; L2 raw NRGBA=%d bytes (%.1fx)\n", r.Level1At512, r.RawNRGBA512, float64(r.Level1At512)/float64(r.RawNRGBA512))
	fmt.Printf("  examples/amber-proof.json revision document: %d bytes\n", r.ExampleJSON)

	fmt.Println("\n[5] PNG digest prefixes (content identity the artifact key is built from)")
	for sceneName, m := range r.PNGDigests {
		for k, v := range m {
			fmt.Printf("  %-26s %-22s sha256:%s\n", sceneName, k, v)
		}
	}
}
