package pipelineBackends

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/util"
)

// BasePipeline can be embedded by a pipeline.
type BasePipeline struct {
	PipelineName    string
	Runtime         string
	Model           *Model
	PipelineTimings *timings
}

type InputOutputInfo struct {
	// The name of the input or output
	Name string
	// The input or output's dimensions, if it's a tensor. This should be
	// ignored for non-tensor types.
	Dimensions Shape
}

type Shape []int64

func (s Shape) String() string {
	return fmt.Sprintf("%v", []int64(s))
}

func (s Shape) ValuesInt() []int {
	output := make([]int, len(s))
	for i, v := range s {
		output[i] = int(v)
	}
	return output
}

// NewShape Returns a Shape, with the given dimensions.
func NewShape(dimensions ...int64) Shape {
	return dimensions
}

type OutputInfo struct {
	Name       string
	Dimensions []int64
}

type PipelineMetadata struct {
	OutputsInfo []OutputInfo
}

type PipelineBatchOutput interface {
	GetOutput() []any
}

// Pipeline is the interface that any pipeline must implement.
type Pipeline interface {
	GetStats() []string                        // Get the pipeline running stats
	Validate() error                           // Validate the pipeline for correctness
	GetMetadata() PipelineMetadata             // Return metadata information for the pipeline
	GetModel() *Model                          // Return the model used by the pipeline
	Run([]string) (PipelineBatchOutput, error) // Run the pipeline on an input
}

// PipelineOption is an option for a pipeline type.
type PipelineOption[T Pipeline] func(eo T)

// PipelineConfig is a configuration for a pipeline type that can be used
// to create that pipeline.
type PipelineConfig[T Pipeline] struct {
	ModelPath    string
	Name         string
	OnnxFilename string
	Options      []PipelineOption[T]
}

type timings struct {
	NumCalls uint64
	TotalNS  uint64
}

// TokenizedInput holds the result of running tokenizer on an input.
type TokenizedInput struct {
	Raw               string
	Tokens            []string
	TokenIDs          []uint32
	TypeIDs           []uint32
	AttentionMask     []uint32
	SpecialTokensMask []uint32
	MaxAttentionIndex int
	Offsets           [][2]uint
}

// PipelineBatch represents a batch of inputs that runs through the pipeline.
type PipelineBatch struct {
	Input             []TokenizedInput
	MaxSequenceLength int
	InputValues       any
	PaddingMask       [][]bool
	DestroyInputs     func() error
	OutputValues      []OutputArray
}

type OutputArray struct {
	Result2D [][]float32
	Result3D [][][]float32
}

func (b *PipelineBatch) Destroy() error {
	return b.DestroyInputs()
}

// NewBatch initializes a new batch for inference.
func NewBatch() *PipelineBatch {
	return &PipelineBatch{DestroyInputs: func() error {
		return nil
	}}
}

func GetNames(info []InputOutputInfo) []string {
	names := make([]string, 0, len(info))
	for _, v := range info {
		names = append(names, v.Name)
	}
	return names
}

func RunSessionOnBatch(batch *PipelineBatch, p *BasePipeline) error {
	switch p.Runtime {
	case "ORT":
		return runORTSessionOnBatch(batch, p)
	case "XLA":
		return runXLASessionOnBatch(batch, p)
	case "GO":
		return runGoSessionOnBatch(batch, p)
	}
	return nil
}

// CreateInputTensorsTraining creates input tensors for training. Same as CreateInputTensors but
// we never pad the batch size as we expect regular batch sizes from the dataset.
func CreateInputTensorsTraining(batch *PipelineBatch, inputsMeta []InputOutputInfo, runtime string) error {
	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, inputsMeta)
	case "XLA":
		return createInputTensorsXLA(batch, inputsMeta, false)
	}
	return nil
}

func CreateInputTensors(batch *PipelineBatch, inputsMeta []InputOutputInfo, runtime string) error {

	switch runtime {
	case "ORT":
		return createInputTensorsORT(batch, inputsMeta)
	case "XLA":
		return createInputTensorsXLA(batch, inputsMeta, true)
	case "GO":
		return createInputTensorsGo(batch, inputsMeta)
	}
	return nil
}

func NewBasePipeline[T Pipeline](config PipelineConfig[T], s *options.Options, model *Model) (*BasePipeline, error) {
	pipeline := &BasePipeline{}
	pipeline.Runtime = s.Runtime
	pipeline.PipelineName = config.Name
	pipeline.Model = model
	pipeline.PipelineTimings = &timings{}
	return pipeline, nil
}

func LoadOnnxModelBytes(model *Model) error {
	var modelOnnxFile string
	onnxFiles, err := getOnnxFiles(model.Path)
	if err != nil {
		return err
	}
	if len(onnxFiles) == 0 {
		return fmt.Errorf("no .onnx file detected at %s. There should be exactly .onnx file", model.Path)
	}
	if len(onnxFiles) > 1 {
		if model.OnnxFilename == "" {
			return fmt.Errorf("multiple .onnx file detected at %s and no OnnxFilename specified", model.Path)
		}
		modelNameFound := false
		for i := range onnxFiles {
			if onnxFiles[i][1] == model.OnnxFilename {
				modelNameFound = true
				modelOnnxFile = util.PathJoinSafe(onnxFiles[i]...)
			}
		}
		if !modelNameFound {
			return fmt.Errorf("file %s not found at %s", model.OnnxFilename, model.Path)
		}
	} else {
		modelOnnxFile = util.PathJoinSafe(onnxFiles[0]...)
	}

	onnxBytes, err := util.ReadFileBytes(modelOnnxFile)
	if err != nil {
		return err
	}

	model.OnnxBytes = onnxBytes

	return err
}

func getOnnxFiles(path string) ([][]string, error) {
	var onnxFiles [][]string
	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if strings.HasSuffix(info.Name(), ".onnx") {
			onnxFiles = append(onnxFiles, []string{util.PathJoinSafe(path, parent), info.Name()})
		}
		return true, nil
	}
	err := util.FileSystem.Walk(context.Background(), path, walker)
	return onnxFiles, err
}

func CreateModelBackend(model *Model, s *options.Options) error {
	// creation of the session. Only one output (either token or sentence embedding).
	var err error
	switch s.Runtime {
	case "ORT":
		err = createORTModelBackend(model, s)
	case "XLA":
		err = createXLAModelBackend(model, s)
	case "GO":
		err = createGoModelBackend(model, s)
	}
	return err
}
