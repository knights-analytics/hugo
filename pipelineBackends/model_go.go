//go:build GO || ALL

package pipelineBackends

import (
	"fmt"

	"github.com/knights-analytics/hugot/options"

	"github.com/advancedclimatesystems/gonnx"
	"gorgonia.org/tensor"
)

type GoModel struct {
	Model *gonnx.Model
}

func createGoModelBackend(model *Model, _ *options.Options) error {

	modelParsed, err := gonnx.NewModelFromBytes(model.OnnxBytes)
	if err != nil {
		return err
	}

	inputs, outputs := loadInputOutputMetaGo(modelParsed)

	model.GoModel = &GoModel{
		Model: modelParsed,
	}
	model.InputsMeta = inputs
	model.OutputsMeta = outputs

	return err
}

func loadInputOutputMetaGo(model *gonnx.Model) ([]InputOutputInfo, []InputOutputInfo) {

	var inputs, outputs []InputOutputInfo

	inputShapes := model.InputShapes()
	for _, name := range model.InputNames() {
		shape := inputShapes[name]
		dimensions := make([]int64, len(shape))
		for i, y := range shape {
			if y.IsDynamic {
				dimensions[i] = -1
			} else {
				dimensions[i] = y.Size
			}
		}
		inputs = append(inputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	outputShapes := model.OutputShapes()
	for _, name := range model.OutputNames() {
		shape := outputShapes[name]
		dimensions := make([]int64, len(shape))
		for i, y := range shape {
			if y.IsDynamic {
				dimensions[i] = -1
			} else {
				dimensions[i] = y.Size
			}
		}
		outputs = append(outputs, InputOutputInfo{
			Name:       name,
			Dimensions: dimensions,
		})
	}
	return inputs, outputs
}

func createInputTensorsGo(batch *PipelineBatch, inputsMeta []InputOutputInfo) error {
	batchSize := len(batch.Input)
	tensorSize := batchSize * batch.MaxSequenceLength

	paddingMasks := make([][]bool, batchSize)
	inputMap := map[string]tensor.Tensor{}
	for _, inputMeta := range inputsMeta {
		backingSlice := make([]int64, tensorSize)
		counter := 0

		for j, input := range batch.Input {
			inputPaddingMask := make([]bool, batch.MaxSequenceLength)
			length := len(input.TokenIDs)
			for k := 0; k < batch.MaxSequenceLength; k++ {
				if k+1 <= length {
					switch inputMeta.Name {
					case "input_ids":
						backingSlice[counter] = int64(input.TokenIDs[k])
						inputPaddingMask[k] = true
					case "token_type_ids":
						backingSlice[counter] = int64(input.TypeIDs[k])
					case "attention_mask":
						backingSlice[counter] = int64(input.AttentionMask[k])
					default:
						return fmt.Errorf("input %s not recognized", inputMeta.Name)
					}
				} else {
					backingSlice[counter] = 0 // pad with zero
				}
				counter++
			}

			if inputMeta.Name == "input_ids" {
				paddingMasks[j] = inputPaddingMask
			}
		}
		inputMap[inputMeta.Name] = tensor.New(
			tensor.WithShape(batchSize, batch.MaxSequenceLength),
			tensor.WithBacking(backingSlice),
		)
	}
	batch.InputValues = inputMap
	batch.PaddingMask = paddingMasks
	return nil
}

func runGoSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {

	tensors, err := p.Model.GoModel.Model.Run(batch.InputValues.(map[string]tensor.Tensor))
	if err != nil {
		return err
	}

	convertedOutput := make([]OutputArray, len(p.Model.OutputsMeta))
	for i, meta := range p.Model.OutputsMeta {
		result := tensors[meta.Name]
		rawOutput := result.Data().([]float32)
		convertedOutput[i] = ReshapeOutput(&rawOutput, p.Model.OutputsMeta[i], batch.PaddingMask, batch.MaxSequenceLength)
	}

	// store resulting tensors
	batch.OutputValues = convertedOutput
	return nil
}
