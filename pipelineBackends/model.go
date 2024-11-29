package pipelineBackends

import (
	"errors"

	"github.com/knights-analytics/hugot/options"
)

type Model struct {
	Path         string
	OnnxFilename string
	OnnxBytes    []byte
	ORTModel     *ORTModel
	XLAModel     *XLAModel
	GoModel      *GoModel
	Tokenizer    *Tokenizer
	InputsMeta   []InputOutputInfo
	OutputsMeta  []InputOutputInfo
	Destroy      func() error
}

func ReshapeOutput(input *[]float32, meta InputOutputInfo, paddingMask [][]bool, sequenceLength int) OutputArray {
	outArray := OutputArray{}

	dimensions := meta.Dimensions.ValuesInt()
	lenDimensions := len(dimensions)
	switch lenDimensions {
	case 2:
		outArray.Result2D = flatDataTo2D(input, paddingMask, dimensions[lenDimensions-1])
	case 3:
		outArray.Result3D = flatDataTo3D(input, paddingMask, sequenceLength, dimensions[lenDimensions-1])
	}
	return outArray
}

func flatDataTo2D(input *[]float32, paddingMask [][]bool, dimension int) [][]float32 {
	// Input string, token, dimension
	output := make([][]float32, len(paddingMask))

	counter := 0
	for batchIndex := range paddingMask {
		inputEmbedding := make([]float32, dimension)

		for i := 0; i < dimension; i++ {
			inputEmbedding[i] = (*input)[counter]
			counter++
		}

		output[batchIndex] = inputEmbedding
	}

	return output
}

func flatDataTo3D(input *[]float32, paddingMask [][]bool, sequenceLength int, dimension int) [][][]float32 {
	// Input string, token, dimension
	output := make([][][]float32, len(paddingMask))

	counter := 0

	for batchIndex, mask := range paddingMask {
		tokenEmbeddings := make([][]float32, 0, sequenceLength)

		for _, isValid := range mask {
			if !isValid {
				// skip whole token
				counter = counter + dimension
				continue
			}

			// valid token, create embedding
			embedding := make([]float32, dimension)

			for i := 0; i < dimension; i++ {
				embedding[i] = (*input)[counter]
				counter++
			}

			tokenEmbeddings = append(tokenEmbeddings, embedding)
		}

		output[batchIndex] = tokenEmbeddings
	}

	return output
}

func LoadModel(path string, onnxFilename string, options *options.Options) (*Model, error) {
	model := &Model{
		Path:         path,
		OnnxFilename: onnxFilename,
	}

	err := LoadOnnxModelBytes(model)
	if err != nil {
		return nil, err
	}

	err = CreateModelBackend(model, options)
	if err != nil {
		return nil, err
	}

	tkErr := LoadTokenizer(model, options)
	if tkErr != nil {
		return nil, tkErr
	}

	model.Destroy = func() error {
		destroyErr := model.Tokenizer.Destroy()
		switch options.Runtime {
		case "ORT":
			destroyErr = errors.Join(destroyErr, model.ORTModel.Destroy())
		case "XLA":
			model.XLAModel.Destroy()
		}
		return destroyErr
	}
	return model, nil
}
