// Package paintmodel is a line-by-line port of the production paint pipeline
// (internal/paint) with one extension: every transfer-function constant is
// supplied by a Config so that hypothetical future color configurations can be
// replayed against the same arithmetic the service uses today.
//
// Keep this in lockstep with internal/paint/paint.go while the service stays
// on the prism-plate-v1 contract; the --mode=verify report entry exists to
// make divergence visible.
package paintmodel

import "math"

// Pixel is linear-light, premultiplied alpha, exactly like paint.Pixel.
type Pixel struct{ R, G, B, A float64 }

// Config is the complete set of transfer-function constants a renderer needs.
// It deliberately contains no ICC/profile machinery: a configuration is just
// named, versioned arithmetic over the existing 0..1 sRGB-encoded inputs.
type Config struct {
	Name      string  // stable identifier persisted alongside rendered facts
	Version   string  // semver-style tag used in artifact/bundle lineage
	Thresh    float64 // linear/sRGB knee (production: 0.04045)
	LinearDiv float64 // production: 12.92
	Offset    float64 // production: 0.055
	Scale     float64 // production: 1.055
	Gamma     float64 // production: 2.4
}

// ProductionConfig returns the exact constants baked into internal/paint.
func ProductionConfig() Config {
	return Config{Name: "srgb-iec", Version: "1.0.0", Thresh: 0.04045, LinearDiv: 12.92, Offset: 0.055, Scale: 1.055, Gamma: 2.4}
}

// AltGamma is a plausible future device-consistency config: a pure 2.2 power
// law, the kind of change a "color configuration" rollout would introduce.
func AltGamma() Config {
	return Config{Name: "gamma-2.2", Version: "1.1.0", Thresh: 0.0, LinearDiv: 1, Offset: 0, Scale: 1, Gamma: 2.2}
}

// AltKnee keeps the 2.4 power law but widens the linear segment, simulating a
// vendor-tuned low-light transfer function.
func AltKnee() Config {
	return Config{Name: "wide-knee", Version: "1.2.0", Thresh: 0.06, LinearDiv: 12.0, Offset: 0.055, Scale: 1.055, Gamma: 2.4}
}

func (cfg Config) linear(v float64) float64 {
	if v <= cfg.Thresh {
		return v / cfg.LinearDiv
	}
	return math.Pow((v+cfg.Offset)/cfg.Scale, cfg.Gamma)
}

func (cfg Config) srgb(v float64) float64 {
	v = math.Max(0, math.Min(1, v)) // production clamps before encoding
	encThresh := cfg.Thresh / cfg.LinearDiv
	if v <= encThresh {
		return v * cfg.LinearDiv
	}
	return cfg.Scale*math.Pow(v, 1/cfg.Gamma) - cfg.Offset
}

// Color is an input-space (document) color, 0..1 encoded values plus alpha.
type Color struct{ R, G, B, A float64 }

// From mirrors paint.From: transfer function per channel, then premultiply.
func (cfg Config) From(c Color) Pixel {
	return Pixel{cfg.linear(c.R) * c.A, cfg.linear(c.G) * c.A, cfg.linear(c.B) * c.A, c.A}
}

// Scale mirrors paint.Pixel.Scale (group opacity applied after layer assembly).
func (p Pixel) Scale(a float64) Pixel { return Pixel{p.R * a, p.G * a, p.B * a, p.A * a} }

// RGBA8 mirrors paint.Pixel.RGBA8, including the transparent-pixel rule and
// round-half-away-from-zero quantization.
func (cfg Config) RGBA8(p Pixel) (uint8, uint8, uint8, uint8) {
	if p.A <= 0 {
		return 0, 0, 0, 0
	}
	r := uint8(math.Round(cfg.srgb(p.R/p.A) * 255))
	g := uint8(math.Round(cfg.srgb(p.G/p.A) * 255))
	b := uint8(math.Round(cfg.srgb(p.B/p.A) * 255))
	a := uint8(math.Round(math.Min(1, p.A) * 255))
	return r, g, b, a
}

func mix(a, b, t float64) float64 { return a + (b-a)*t }

// Brush is the subset of scene.Brush exercised by the diagnostic.
type Brush struct {
	Kind         string // "solid" | "linear"
	Color        Color
	FromX, FromY float64
	ToX, ToY     float64
	Stops        []Stop
}

type Stop struct {
	At    float64
	Color Color
}

// Sample mirrors paint.Sample: interpolation happens on premultiplied linear
// pixels, including the alpha channel itself.
func (cfg Config) Sample(b Brush, x, y float64) Pixel {
	if b.Kind == "solid" {
		return cfg.From(b.Color)
	}
	dx, dy := b.ToX-b.FromX, b.ToY-b.FromY
	t := ((x-b.FromX)*dx + (y-b.FromY)*dy) / (dx*dx + dy*dy)
	t = math.Max(0, math.Min(1, t))
	for i := 1; i < len(b.Stops); i++ {
		a, z := b.Stops[i-1], b.Stops[i]
		if t <= z.At {
			u := (t - a.At) / (z.At - a.At)
			x0, x1 := cfg.From(a.Color), cfg.From(z.Color)
			return Pixel{mix(x0.R, x1.R, u), mix(x0.G, x1.G, u), mix(x0.B, x1.B, u), mix(x0.A, x1.A, u)}
		}
	}
	return cfg.From(b.Stops[len(b.Stops)-1].Color)
}

// NaiveSRGBSample is the counterfactual most renderers (and SVG viewers using
// sRGB gradient interpolation) implement: mix encoded values, including alpha,
// then decode once. It quantifies the interpolation-space promise gap.
func (cfg Config) NaiveSRGBSample(b Brush, x, y float64) Pixel {
	if b.Kind == "solid" {
		return cfg.From(b.Color)
	}
	dx, dy := b.ToX-b.FromX, b.ToY-b.FromY
	t := ((x-b.FromX)*dx + (y-b.FromY)*dy) / (dx*dx + dy*dy)
	t = math.Max(0, math.Min(1, t))
	for i := 1; i < len(b.Stops); i++ {
		a, z := b.Stops[i-1], b.Stops[i]
		if t <= z.At {
			u := (t - a.At) / (z.At - a.At)
			enc := Color{
				R: mix(a.Color.R, z.Color.R, u),
				G: mix(a.Color.G, z.Color.G, u),
				B: mix(a.Color.B, z.Color.B, u),
				A: mix(a.Color.A, z.Color.A, u),
			}
			return cfg.From(enc)
		}
	}
	return cfg.From(b.Stops[len(b.Stops)-1].Color)
}

// Over mirrors paint.Over exactly, including the unpremultiply/re-premultiply
// inside the non-normal blend branch.
func Over(dst, src Pixel, mode string) Pixel {
	blend := func(d, s float64) float64 {
		if mode == "normal" || src.A == 0 || dst.A == 0 {
			return s + d*(1-src.A)
		}
		a, b := d/dst.A, s/src.A
		v := a * b
		if mode == "screen" {
			v = a + b - a*b
		}
		return (1-src.A)*d + (1-dst.A)*s + src.A*dst.A*v
	}
	return Pixel{blend(dst.R, src.R), blend(dst.G, src.G), blend(dst.B, src.B), src.A + dst.A*(1-src.A)}
}
