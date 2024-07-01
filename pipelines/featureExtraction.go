package pipelines

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"time"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/daulet/tokenizers"
	util "github.com/knights-analytics/hugot/utils"
)

// FeatureExtractionPipeline A feature extraction pipeline is a go version of
// https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/feature_extraction.py
type FeatureExtractionPipeline struct {
	basePipeline
	Normalization bool
	OutputName    string
	Output        ort.InputOutputInfo
}

type FeatureExtractionOutput struct {
	Embeddings [][]float32
}

func (t *FeatureExtractionOutput) GetOutput() []any {
	out := make([]any, len(t.Embeddings))
	for i, embedding := range t.Embeddings {
		out[i] = any(embedding)
	}
	return out
}

// PIPELINE OPTIONS

// WithNormalization applies normalization to the mean pooled output of the feature pipeline.
func WithNormalization() PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.Normalization = true
	}
}

// WithOutputName if there are multiple outputs from the underlying model, which output should
// be returned. If not passed, the first output from the feature pipeline is returned.
func WithOutputName(outputName string) PipelineOption[*FeatureExtractionPipeline] {
	return func(pipeline *FeatureExtractionPipeline) {
		pipeline.OutputName = outputName
	}
}

// NewFeatureExtractionPipeline init a feature extraction pipeline.
func NewFeatureExtractionPipeline(config PipelineConfig[*FeatureExtractionPipeline], ortOptions *ort.SessionOptions) (*FeatureExtractionPipeline, error) {
	pipeline := &FeatureExtractionPipeline{}
	pipeline.ModelPath = config.ModelPath
	pipeline.PipelineName = config.Name
	pipeline.OrtOptions = ortOptions
	pipeline.OnnxFilename = config.OnnxFilename

	for _, o := range config.Options {
		o(pipeline)
	}

	// tokenizer init
	pipeline.TokenizerOptions = []tokenizers.EncodeOption{tokenizers.WithReturnTypeIDs(), tokenizers.WithReturnAttentionMask()}
	tk, err := loadTokenizer(pipeline.ModelPath)
	if err != nil {
		return nil, err
	}
	pipeline.Tokenizer = tk

	// onnx model init
	model, err := loadOnnxModelBytes(pipeline.ModelPath, pipeline.OnnxFilename)
	if err != nil {
		return nil, err
	}

	// init of inputs and outputs
	inputs, outputs, err := loadInputOutputMeta(model)
	if err != nil {
		return nil, err
	}
	pipeline.InputsMeta = inputs
	pipeline.OutputsMeta = outputs

	// filter outputs
	if pipeline.OutputName != "" {
		for _, output := range outputs {
			if output.Name == pipeline.OutputName {
				pipeline.Output = output
				break
			}
		}
		if pipeline.Output.Name == "" {
			return nil, fmt.Errorf("output %s is not available, outputs are: %s", pipeline.OutputName, strings.Join(getNames(outputs), ", "))
		}
	} else {
		pipeline.Output = outputs[0] // we take the first output otherwise, like transformers does
	}

	// creation of the session. Only one output (either token or sentence embedding).
	session, err := createSession(model, inputs, []ort.InputOutputInfo{pipeline.Output}, ortOptions)
	if err != nil {
		return nil, err
	}
	pipeline.OrtSession = session

	// initialize timings

	pipeline.PipelineTimings = &timings{}
	pipeline.TokenizerTimings = &timings{}

	// validate pipeline
	err = pipeline.Validate()
	if err != nil {
		errDestroy := pipeline.Destroy()
		return nil, errors.Join(err, errDestroy)
	}
	return pipeline, nil
}

// INTERFACE IMPLEMENTATION

// GetMetadata returns metadata information about the pipeline, in particular:
// OutputInfo: names and dimensions of the output layer.
func (p *FeatureExtractionPipeline) GetMetadata() PipelineMetadata {
	return PipelineMetadata{
		OutputsInfo: []OutputInfo{
			{
				Name:       p.OutputName,
				Dimensions: p.Output.Dimensions,
			},
		},
	}
}

// Destroy frees the feature extraction pipeline resources.
func (p *FeatureExtractionPipeline) Destroy() error {
	return destroySession(p.Tokenizer, p.OrtSession)
}

// GetStats returns the runtime statistics for the pipeline.
func (p *FeatureExtractionPipeline) GetStats() []string {
	return []string{
		fmt.Sprintf("Statistics for pipeline: %s", p.PipelineName),
		fmt.Sprintf("Tokenizer: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.TokenizerTimings.TotalNS),
			p.TokenizerTimings.NumCalls,
			time.Duration(float64(p.TokenizerTimings.TotalNS)/math.Max(1, float64(p.TokenizerTimings.NumCalls)))),
		fmt.Sprintf("ONNX: Total time=%s, Execution count=%d, Average query time=%s",
			time.Duration(p.PipelineTimings.TotalNS),
			p.PipelineTimings.NumCalls,
			time.Duration(float64(p.PipelineTimings.TotalNS)/math.Max(1, float64(p.PipelineTimings.NumCalls)))),
	}
}

// Validate checks that the pipeline is valid.
func (p *FeatureExtractionPipeline) Validate() error {
	var validationErrors []error

	for _, input := range p.InputsMeta {
		dims := []int64(input.Dimensions)
		if len(dims) > 3 {
			validationErrors = append(validationErrors, fmt.Errorf("inputs and outputs currently can have at most 3 dimensions"))
		}
		nDynamicDimensions := 0
		for _, d := range dims {
			if d == -1 {
				nDynamicDimensions++
			}
		}
		if nDynamicDimensions > 2 {
			validationErrors = append(validationErrors, fmt.Errorf(`input %s has dimensions: %s. 
			There can only be max 2 dynamic dimensions (batch size and sequence length)`,
				input.Name, input.Dimensions.String()))
		}
	}
	return errors.Join(validationErrors...)
}

// Preprocess tokenizes the input strings.
func (p *FeatureExtractionPipeline) Preprocess(batch *PipelineBatch, inputs []string) error {
	start := time.Now()
	tokenizeInputs(batch, p.Tokenizer, inputs, p.TokenizerOptions)
	atomic.AddUint64(&p.TokenizerTimings.NumCalls, 1)
	atomic.AddUint64(&p.TokenizerTimings.TotalNS, uint64(time.Since(start)))
	err := createInputTensors(batch, p.InputsMeta)
	return err
}

// Forward performs the forward inference of the feature extraction pipeline.
func (p *FeatureExtractionPipeline) Forward(batch *PipelineBatch) error {
	start := time.Now()
	err := runSessionOnBatch(batch, p.OrtSession, []ort.InputOutputInfo{p.Output})
	if err != nil {
		return err
	}
	atomic.AddUint64(&p.PipelineTimings.NumCalls, 1)
	atomic.AddUint64(&p.PipelineTimings.TotalNS, uint64(time.Since(start)))
	return nil
}

// Postprocess parses the first output from the network similar to the transformers implementation.
func (p *FeatureExtractionPipeline) Postprocess(batch *PipelineBatch) (*FeatureExtractionOutput, error) {
	// TODO: this works if token embeddings are returned or sentence embeddings are returned.
	// in the former case embeddings are mean pooled. In the latter they are just returned.
	// to make this more general for other pipelines and to allow return of raw token embeddings,
	// we need an ndarray type that can be the return type of this pipeline. Need to think
	// about how to do this in a lightweight manner.

	batchEmbeddings := make([][]float32, len(batch.Input))
	outputDimensions := []int64(p.Output.Dimensions)
	embeddingDimension := outputDimensions[len(outputDimensions)-1]
	maxSequenceLength := batch.MaxSequenceLength

	// now take the output slice and gather the results as a "matrix"
	outputEmbedding := make([]float32, embeddingDimension)
	outputEmbeddingCounter := 0
	tokenEmbeddings := make([][]float32, maxSequenceLength)
	tokenEmbeddingsCounter := 0
	batchInputCounter := 0
	for _, result := range batch.OutputTensors[0].GetData() {
		outputEmbedding[outputEmbeddingCounter] = result
		if outputEmbeddingCounter == int(embeddingDimension)-1 {
			// we gathered one embedding
			if len(outputDimensions) <= 2 {
				// it is already a sentence embedding, just add it to batch outputs
				batchEmbeddings[batchInputCounter] = outputEmbedding
				outputEmbedding = make([]float32, embeddingDimension)
				batchInputCounter++
			} else {
				// output is embedding for a token, add to token embeddings
				tokenEmbeddings[tokenEmbeddingsCounter] = outputEmbedding
				outputEmbedding = make([]float32, embeddingDimension)
				if tokenEmbeddingsCounter == maxSequenceLength-1 {
					// computed all embeddings for the tokens, calculate sentence embedding, add to batch outputs, and reset token embeddings and counter
					batchEmbeddings[batchInputCounter] = meanPooling(tokenEmbeddings, batch.Input[batchInputCounter], maxSequenceLength, int(embeddingDimension))
					tokenEmbeddings = make([][]float32, maxSequenceLength)
					tokenEmbeddingsCounter = 0
					batchInputCounter++
				} else {
					// still more tokens to go
					tokenEmbeddingsCounter++
				}
			}
			outputEmbeddingCounter = 0
		} else {
			// still more elements of the embedding to go
			outputEmbeddingCounter++
		}
	}

	// Normalize embeddings (if asked), like in https://huggingface.co/sentence-transformers/all-mpnet-base-v2
	if p.Normalization {
		for i, output := range batchEmbeddings {
			batchEmbeddings[i] = util.Normalize(output, 2)
		}
	}

	return &FeatureExtractionOutput{Embeddings: batchEmbeddings}, nil
}

func meanPooling(tokens [][]float32, input tokenizedInput, maxSequence int, dimensions int) []float32 {
	length := len(input.AttentionMask)
	vector := make([]float32, dimensions)
	for j := 0; j < maxSequence; j++ {
		if j+1 <= length && input.AttentionMask[j] != 0 {
			for k, vectorValue := range tokens[j] {
				vector[k] = vector[k] + vectorValue
			}
		}
	}

	numAttentionTokens := float32(input.MaxAttentionIndex + 1)
	for v, vectorValue := range vector {
		vector[v] = vectorValue / numAttentionTokens
	}

	return vector
}

// Run the pipeline on a batch of strings.
func (p *FeatureExtractionPipeline) Run(inputs []string) (PipelineBatchOutput, error) {
	return p.RunPipeline(inputs)
}

// RunPipeline is like Run, but returns the concrete feature extraction output type rather than the interface.
func (p *FeatureExtractionPipeline) RunPipeline(inputs []string) (*FeatureExtractionOutput, error) {
	var runErrors []error
	batch := NewBatch()
	defer func(*PipelineBatch) {
		runErrors = append(runErrors, batch.Destroy())
	}(batch)

	runErrors = append(runErrors, p.Preprocess(batch, inputs))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	runErrors = append(runErrors, p.Forward(batch))
	if e := errors.Join(runErrors...); e != nil {
		return nil, e
	}

	result, postErr := p.Postprocess(batch)
	runErrors = append(runErrors, postErr)
	return result, errors.Join(runErrors...)
}
