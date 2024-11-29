package pipelineBackends

import (
	"fmt"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

type Tokenizer struct {
	Runtime          string
	RustTokenizer    *RustTokenizer
	GoTokenizer      *GoTokenizer
	TokenizerTimings *timings
	Destroy          func() error
}

func LoadTokenizer(model *Model, s *options.Options) error {
	tokenizerBytes, err := util.ReadFileBytes(util.PathJoinSafe(model.Path, "tokenizer.json"))
	if err != nil {
		return err
	}

	switch s.Runtime {
	case "ORT", "XLA":
		return loadRustTokenizer(tokenizerBytes, model)
	case "GO":
		return loadGoTokenizer(tokenizerBytes, model)
	default:
		return fmt.Errorf("runtime %s not recognized", s.Runtime)
	}
}

func TokenizeInputs(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	switch tk.Runtime {
	case "RUST":
		tokenizeInputsRust(batch, tk, inputs)
	case "GO":
		tokenizeInputsGo(batch, tk, inputs)
	}
}

func AllInputTokens(pipeline *BasePipeline) {
	if pipeline.Model.Tokenizer.Runtime == "RUST" {
		allInputTokensRust(pipeline)
	}
}

func Decode(tokens []uint32, tokenizer *Tokenizer) string {
	switch tokenizer.Runtime {
	case "RUST":
		return decodeRust(tokens, tokenizer)
	case "GO":
		return decodeGo(tokens, tokenizer)
	}
	return ""
}
