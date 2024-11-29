//go:build GO || ALL

package pipelineBackends

import (
	"bytes"
	"log"

	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

type GoTokenizer struct {
	Tokenizer *tokenizer.Tokenizer
}

func loadGoTokenizer(tokenizerBytes []byte, model *Model) error {
	tk, tkErr := pretrained.FromReader(bytes.NewReader(tokenizerBytes))
	if tkErr != nil {
		return tkErr
	}
	model.Tokenizer = &Tokenizer{Runtime: "GO", GoTokenizer: &GoTokenizer{Tokenizer: tk}, TokenizerTimings: &timings{}, Destroy: func() error {
		return nil
	}}
	return nil
}

func tokenizeInputsGo(batch *PipelineBatch, tk *Tokenizer, inputs []string) {
	outputs := make([]TokenizedInput, len(inputs))
	maxSequence := 0
	goTK := tk.GoTokenizer.Tokenizer
	for i, input := range inputs {

		output, err := goTK.EncodeSingle(input, true)
		if err != nil {
			log.Fatal(err)
		}

		maxAttentionIndex := 0
		for j, attentionMaskValue := range output.AttentionMask {
			if attentionMaskValue != 0 {
				maxAttentionIndex = j
			}
		}

		outputs[i] = TokenizedInput{
			Raw:               input,
			Tokens:            output.Tokens,
			TokenIDs:          convertIntsToUints(output.Ids),
			TypeIDs:           convertIntsToUints(output.TypeIds),
			AttentionMask:     convertIntsToUints(output.AttentionMask),
			MaxAttentionIndex: maxAttentionIndex,
			SpecialTokensMask: convertIntsToUints(output.SpecialTokenMask),
			Offsets:           convertGoOffsets(output.Offsets), // we need the offsets here for postprocessing later
		}
		if maxAttentionIndex > maxSequence {
			maxSequence = maxAttentionIndex
		}

	}
	batch.Input = outputs
	batch.MaxSequenceLength = maxSequence + 1
}

func decodeGo(tokens []uint32, tokenizer *Tokenizer) string {
	return tokenizer.GoTokenizer.Tokenizer.Decode(convertUintsToInts(tokens), false)
}

func convertIntsToUints(input []int) []uint32 {
	output := make([]uint32, len(input))
	for i, x := range input {
		output[i] = uint32(x)
	}
	return output
}

func convertUintsToInts(input []uint32) []int {
	output := make([]int, len(input))
	for i, x := range input {
		output[i] = int(x)
	}
	return output
}

func convertGoOffsets(input [][]int) [][2]uint {
	output := make([][2]uint, len(input))
	for i, x := range input {
		output[i] = [2]uint{uint(x[0]), uint(x[1])}
	}
	return output
}
