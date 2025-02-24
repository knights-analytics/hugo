package hugot

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/gomlx/go-huggingface/hub"
	"github.com/stretchr/testify/assert"

	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
)

//go:embed testData/tokenExpected.json
var tokenExpectedByte []byte

//go:embed testData/vectors.json
var resultsByte []byte

// use the system library for the tests
const onnxRuntimeSharedLibrary = "/usr/lib64/onnxruntime.so"

// test download validation

func TestDownloadValidation(t *testing.T) {

	downloadOptions := NewDownloadOptions()

	_, err := validateDownloadHfModel(hub.New("KnightsAnalytics/distilbert-base-uncased-finetuned-sst-2-english"), downloadOptions)
	assert.NoError(t, err)
	// a model without tokenizer.json or .onnx model should error
	_, err = validateDownloadHfModel(hub.New("ByteDance/SDXL-Lightning"), downloadOptions)
	assert.Error(t, err)
	// a model with the required files in a subfolder should not error
	_, err = validateDownloadHfModel(hub.New("distilbert/distilbert-base-uncased-finetuned-sst-2-english"), downloadOptions)
	assert.NoError(t, err)
}

// FEATURE EXTRACTION

func featureExtractionPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"

	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(resultsByte, &expectedResults)
	check(t, err)
	var testResults [][]float32

	// test 'robert smith'
	testResults = expectedResults["test1output"]
	batchResult, err := pipeline.RunPipeline([]string{"robert smith"})
	if err != nil {
		t.Fatal(err)
	}
	for i := range batchResult.Embeddings {
		e := floatsEqual(batchResult.Embeddings[i], testResults[i])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// test ['robert smith junior', 'francis ford coppola']
	testResults = expectedResults["test2output"]
	batchResult, err = pipeline.RunPipeline([]string{"robert smith junior", "francis ford coppola"})
	if err != nil {
		t.FailNow()
	}
	for i := range batchResult.Embeddings {
		e := floatsEqual(batchResult.Embeddings[i], testResults[i])
		if e != nil {
			t.Logf("Test 1: The neural network didn't produce the correct result on loop %d: %s\n", i, e)
			t.FailNow()
		}
	}

	// determinism test to make sure embeddings of a string are not influenced by other strings in the batch
	testPairs := map[string][][]string{}
	testPairs["identity"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo"}}
	testPairs["contextOverlap"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "yo mama yo"}}
	testPairs["contextDisjoint"] = [][]string{{"sinopharm", "yo"}, {"sinopharm", "another test"}}

	for k, sentencePair := range testPairs {
		// these vectors should be the same
		firstBatchResult, err2 := pipeline.RunPipeline(sentencePair[0])
		check(t, err2)
		firstEmbedding := firstBatchResult.Embeddings[0]

		secondBatchResult, err3 := pipeline.RunPipeline(sentencePair[1])
		check(t, err3)
		secondEmbedding := secondBatchResult.Embeddings[0]
		e := floatsEqual(firstEmbedding, secondEmbedding)
		if e != nil {
			t.Logf("Equality failed for determinism test %s test with pairs %s and %s", k, strings.Join(sentencePair[0], ","), strings.Join(sentencePair[1], ","))
			t.Log("First vector", firstEmbedding)
			t.Log("second vector", secondEmbedding)
			t.Fail()
		}
	}

	zero := uint64(0)
	assert.Greater(t, pipeline.PipelineTimings.NumCalls, zero, "PipelineTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.PipelineTimings.TotalNS, zero, "PipelineTimings.TotalNS should be greater than 0")
	assert.Greater(t, pipeline.Model.Tokenizer.TokenizerTimings.NumCalls, zero, "TokenizerTimings.NumCalls should be greater than 0")
	assert.Greater(t, pipeline.Model.Tokenizer.TokenizerTimings.TotalNS, zero, "TokenizerTimings.TotalNS should be greater than 0")

	// test normalization
	testResults = expectedResults["normalizedOutput"]
	config = FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineNormalise",
		OnnxFilename: "model.onnx",
		Options: []FeatureExtractionOption{
			pipelines.WithNormalization(),
		},
	}
	pipeline, err = NewPipeline(session, config)
	check(t, err)

	normalizationStrings := []string{"Onnxruntime is a great inference backend"}
	normalizedEmbedding, err := pipeline.RunPipeline(normalizationStrings)
	check(t, err)
	for i, embedding := range normalizedEmbedding.Embeddings {
		e := floatsEqual(embedding, testResults[i])
		if e != nil {
			t.Fatalf("Normalization test failed: %s", normalizationStrings[i])
		}
	}

	// test getting output by name
	configSentence := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineSentence",
		OnnxFilename: "model.onnx",
		Options:      []FeatureExtractionOption{pipelines.WithOutputName("last_hidden_state")},
	}
	pipelineSentence, err := NewPipeline(session, configSentence)
	check(t, err)

	outputSentence, err := pipelineSentence.RunPipeline([]string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
	fmt.Println(outputSentence.Embeddings[0])
	configSentence = FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipelineToken",
		OnnxFilename: "model.onnx",
	}
	pipelineToken, err := NewPipeline(session, configSentence)
	check(t, err)
	_, err = pipelineToken.RunPipeline([]string{"Onnxruntime is a great inference backend"})
	if err != nil {
		t.FailNow()
	}
}

func featureExtractionPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		OnnxFilename: "model.onnx",
		Name:         "testPipeline",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = pipelineBackends.NewShape(-1, -1, -1)

	err = pipeline.Validate()
	assert.Error(t, err)

	pipeline.Model.InputsMeta[0].Dimensions = pipelineBackends.NewShape(1, 1, 1, 1)
	err = pipeline.Validate()
	assert.Error(t, err)
}

// Text classification

func textClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSoftmax(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	check(t, err)

	test := struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		pipeline: sentimentPipeline,
		name:     "Basic tests",
		strings:  []string{"This movie is disgustingly good!", "The director tried too much"},
		expected: pipelines.TextClassificationOutput{
			ClassificationOutputs: [][]pipelines.ClassificationOutput{
				{
					{
						Label: "POSITIVE",
						Score: 0.9998536109924316,
					},
				},
				{
					{
						Label: "NEGATIVE",
						Score: 0.9975218176841736,
					},
				},
			},
		},
	}

	t.Run(test.name, func(t *testing.T) {
		batchResult, err := test.pipeline.RunPipeline(test.strings)
		check(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check get stats
	session.GetStats()
}

func textClassificationPipelineMulti(t *testing.T, session *Session) {
	t.Helper()

	modelPathMulti := "./models/KnightsAnalytics_roberta-base-go_emotions"

	configMulti := TextClassificationConfig{
		ModelPath:    modelPathMulti,
		Name:         "testPipelineSimpleMulti",
		OnnxFilename: "model.onnx",
		Options: []TextClassificationOption{
			pipelines.WithMultiLabel(),
			pipelines.WithSigmoid(),
		},
	}
	sentimentPipelineMulti, err := NewPipeline(session, configMulti)
	check(t, err)

	test := struct {
		pipeline *pipelines.TextClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TextClassificationOutput
	}{
		pipeline: sentimentPipelineMulti,
		name:     "Multiclass pipeline test",
		strings:  []string{"ONNX is seriously fast for small batches. Impressive"},
		expected: pipelines.TextClassificationOutput{
			ClassificationOutputs: [][]pipelines.ClassificationOutput{
				{
					{
						Label: "admiration",
						Score: 0.9217681,
					},
					{
						Label: "amusement",
						Score: 0.001201711,
					},
					{
						Label: "anger",
						Score: 0.001109502,
					},
					{
						Label: "annoyance",
						Score: 0.0034009134,
					},
					{
						Label: "approval",
						Score: 0.05643816,
					},
					{
						Label: "caring",
						Score: 0.0011591336,
					},
					{
						Label: "confusion",
						Score: 0.0018672282,
					},
					{
						Label: "curiosity",
						Score: 0.0026787464,
					},
					{
						Label: "desire",
						Score: 0.00085846696,
					},
					{
						Label: "disappointment",
						Score: 0.0027759627,
					},
					{
						Label: "disapproval",
						Score: 0.004615115,
					},
					{
						Label: "disgust",
						Score: 0.00075303164,
					},
					{
						Label: "embarrassment",
						Score: 0.0003314704,
					},
					{
						Label: "excitement",
						Score: 0.005340109,
					},
					{
						Label: "fear",
						Score: 0.00042834174,
					},
					{
						Label: "gratitude",
						Score: 0.013405683,
					},
					{
						Label: "grief",
						Score: 0.00029952865,
					},
					{
						Label: "joy",
						Score: 0.0026875956,
					},
					{
						Label: "love",
						Score: 0.00092915917,
					},
					{
						Label: "nervousness",
						Score: 0.00012843,
					},
					{
						Label: "optimism",
						Score: 0.006792505,
					},
					{
						Label: "pride",
						Score: 0.0033409835,
					},
					{
						Label: "realization",
						Score: 0.007224476,
					},
					{
						Label: "relief",
						Score: 0.00071489986,
					},
					{
						Label: "remorse",
						Score: 0.00026071363,
					},
					{
						Label: "sadness",
						Score: 0.0009562365,
					},
					{
						Label: "surprise",
						Score: 0.0037120024,
					},
					{
						Label: "neutral",
						Score: 0.04079749,
					},
				},
			},
		},
	}

	t.Run(test.name, func(t *testing.T) {
		batchResult, err := test.pipeline.RunPipeline(test.strings)
		check(t, err)
		for i, expected := range test.expected.ClassificationOutputs {
			checkClassificationOutput(t, expected, batchResult.ClassificationOutputs[i])
		}
	})

	// check get stats
	session.GetStats()
}

func textClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-base-uncased-finetuned-sst-2-english"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TextClassificationOption{
			pipelines.WithSingleLabel(),
		},
	}
	sentimentPipeline, err := NewPipeline(session, config)
	check(t, err)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := sentimentPipeline.IDLabelMap
		defer func() {
			sentimentPipeline.IDLabelMap = labelMapInitial
		}()
		sentimentPipeline.IDLabelMap = map[int]string{}
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := sentimentPipeline.Model.OutputsMeta[0].Dimensions
		defer func() {
			sentimentPipeline.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		sentimentPipeline.Model.OutputsMeta[0].Dimensions = pipelineBackends.NewShape(-1, -1, -1)
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})
}

// Zero shot

func zeroShotClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := ZeroShotClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
		Options: []pipelineBackends.PipelineOption[*pipelines.ZeroShotClassificationPipeline]{
			pipelines.WithHypothesisTemplate("This example is {}."),
			pipelines.WithLabels([]string{"fun", "dangerous"}),
			pipelines.WithMultilabel(false), // Gets overridden per test, but included for coverage
		},
	}

	classificationPipeline, err := NewPipeline(session, config)
	check(t, err)

	tests := []struct {
		pipeline   *pipelines.ZeroShotClassificationPipeline
		name       string
		sequences  []string
		labels     []string
		multilabel bool
		expected   pipelines.ZeroShotOutput
	}{
		{
			pipeline:   classificationPipeline,
			name:       "single sequence, single label, no multilabel",
			sequences:  []string{"I am going to the park"},
			labels:     []string{"fun"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, multiple labels, no multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun", "movie"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.7746766209602356,
							},
							{
								Key:   "movie",
								Value: 0.2253233790397644,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "movie",
								Value: 0.9984978437423706,
							},
							{
								Key:   "fun",
								Value: 0.001502170693129301,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, multiple labels, multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun", "movie"},
			multilabel: true,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
							{
								Key:   "movie",
								Value: 0.00009480675362283364,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "movie",
								Value: 0.9985591769218445,
							},
							{
								Key:   "fun",
								Value: 0.0006653196760453284,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "multiple sequences, single label, multilabel",
			sequences:  []string{"I am going to the park", "I will watch Interstellar tonight"},
			labels:     []string{"fun"},
			multilabel: true,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "I am going to the park",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0009069009101949632,
							},
						},
					},
					{
						Sequence: "I will watch Interstellar tonight",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "fun",
								Value: 0.0006653196760453284,
							},
						},
					},
				},
			},
		},
		{
			pipeline:   classificationPipeline,
			name:       "single sequence, multiple labels, multilabel=false",
			sequences:  []string{"Please don't bother me, I'm in a rush"},
			labels:     []string{"busy", "relaxed", "stressed"},
			multilabel: false,
			expected: pipelines.ZeroShotOutput{
				ClassificationOutputs: []pipelines.ZeroShotClassificationOutput{
					{
						Sequence: "Please don't bother me, I'm in a rush",
						SortedValues: []struct {
							Key   string
							Value float64
						}{
							{
								Key:   "stressed",
								Value: 0.8865461349487305,
							},
							{
								Key:   "busy",
								Value: 0.10629364103078842,
							},
							{
								Key:   "relaxed",
								Value: 0.007160270120948553,
							},
						},
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			classificationPipeline.Labels = tt.labels
			classificationPipeline.Multilabel = tt.multilabel
			batchResult, err := tt.pipeline.RunPipeline(tt.sequences)
			check(t, err)
			assert.Equal(t, len(batchResult.GetOutput()), len(tt.expected.ClassificationOutputs))

			for ind, expected := range tt.expected.ClassificationOutputs {
				expectedResult := expected.SortedValues
				testResult := batchResult.ClassificationOutputs[ind].SortedValues
				assert.Equal(t, len(expectedResult), len(testResult))
				assert.Equal(t, tt.expected.ClassificationOutputs[ind].Sequence, batchResult.ClassificationOutputs[ind].Sequence)
				for i := range testResult {
					assert.True(t, almostEqual(testResult[i].Value, expectedResult[i].Value))
				}
			}
		})
	}
}

func zeroShotClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_deberta-v3-base-zeroshot-v1"

	config := TextClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
	}
	sentimentPipeline, err := NewPipeline(session, config)
	check(t, err)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := sentimentPipeline.IDLabelMap
		defer func() {
			sentimentPipeline.IDLabelMap = labelMapInitial
		}()
		sentimentPipeline.IDLabelMap = map[int]string{}
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := sentimentPipeline.Model.OutputsMeta[0].Dimensions
		defer func() {
			sentimentPipeline.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		sentimentPipeline.Model.OutputsMeta[0].Dimensions = pipelineBackends.NewShape(-1, -1, -1)
		err = sentimentPipeline.Validate()
		assert.Error(t, err)
	})
}

// Token classification

func tokenClassificationPipeline(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	check(t, err2)

	configNone := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineNone",
		Options: []TokenClassificationOption{
			pipelines.WithoutAggregation(),
		},
	}
	pipelineNone, err3 := NewPipeline(session, configNone)
	check(t, err3)

	var expectedResults map[int]pipelines.TokenClassificationOutput
	err4 := json.Unmarshal(tokenExpectedByte, &expectedResults)
	check(t, err4)

	tests := []struct {
		pipeline *pipelines.TokenClassificationPipeline
		name     string
		strings  []string
		expected pipelines.TokenClassificationOutput
	}{
		{
			pipeline: pipelineSimple,
			name:     "Simple aggregation",
			strings:  []string{"My name is Wolfgang and I live in Berlin."},
			expected: expectedResults[0],
		},
		{
			pipeline: pipelineNone,
			name:     "No aggregation",
			strings:  []string{"My name is Wolfgang and I live in Berlin."},
			expected: expectedResults[1],
		},
		{
			pipeline: pipelineSimple,
			name:     "Parsing of batch with different token length",
			strings:  []string{"Microsoft incorporated.", "Yesterday I went to Berlin and met with Jack Brown."},
			expected: expectedResults[2],
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			batchResult, err := tt.pipeline.RunPipeline(tt.strings)
			check(t, err)
			printTokenEntities(batchResult)
			for i, predictedEntities := range batchResult.Entities {
				assert.Equal(t, len(tt.expected.Entities[i]), len(predictedEntities))
				for j, entity := range predictedEntities {
					expectedEntity := tt.expected.Entities[i][j]
					assert.Equal(t, expectedEntity.Entity, entity.Entity)
					assert.Equal(t, expectedEntity.Word, entity.Word)
				}
			}
		})
	}
}

func tokenClassificationPipelineValidation(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	pipelineSimple, err2 := NewPipeline(session, configSimple)
	check(t, err2)

	t.Run("id-label-map", func(t *testing.T) {
		labelMapInitial := pipelineSimple.IDLabelMap
		defer func() {
			pipelineSimple.IDLabelMap = labelMapInitial
		}()
		pipelineSimple.IDLabelMap = map[int]string{}
		err := pipelineSimple.Validate()
		assert.Error(t, err)
	})

	t.Run("output-shape", func(t *testing.T) {
		dimensionInitial := pipelineSimple.Model.OutputsMeta[0].Dimensions
		defer func() {
			pipelineSimple.Model.OutputsMeta[0].Dimensions = dimensionInitial
		}()
		pipelineSimple.Model.OutputsMeta[0].Dimensions = pipelineBackends.NewShape(-1, -1, -1)
		err := pipelineSimple.Validate()
		assert.Error(t, err)
	})
}

// No same name

func noSameNamePipeline(t *testing.T, session *Session) {
	t.Helper()
	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testPipelineSimple",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err2 := NewPipeline(session, configSimple)
	if err2 != nil {
		t.FailNow()
	}
	_, err3 := NewPipeline(session, configSimple)
	assert.Error(t, err3)
}

// destroy pipelines

func destroyPipelines(t *testing.T, session *Session) {
	t.Helper()

	modelPath := "./models/KnightsAnalytics_distilbert-NER"
	configSimple := TokenClassificationConfig{
		ModelPath: modelPath,
		Name:      "testClosePipeline",
		Options: []TokenClassificationOption{
			pipelines.WithSimpleAggregation(),
			pipelines.WithIgnoreLabels([]string{"O"}),
		},
	}
	_, err2 := NewPipeline(session, configSimple)
	if err2 != nil {
		t.FailNow()
	}

	if len(session.models) != 1 {
		t.Fatal("Session should have 1 model")
	}

	for _, model := range session.models {
		if _, ok := model.Pipelines["testClosePipeline"]; !ok {
			t.Fatal("Pipeline alias was not added to the model")
		}
	}

	if err := ClosePipeline[*pipelines.TokenClassificationPipeline](session, "testClosePipeline"); err != nil {
		t.Fatal(err)
	}

	if len(session.models) != 0 {
		t.Fatal("Session should have 0 models")
	}
	if len(session.tokenClassificationPipelines) != 0 {
		t.Fatal("Session should have 0 token classification pipelines")
	}
}

// Thread safety

func threadSafety(t *testing.T, session *Session) {
	const numWorkers = 4
	const numEmbeddings = 500
	const numResults = numWorkers * numEmbeddings

	t.Helper()
	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	check(t, err)

	var expectedResults map[string][][]float32
	err = json.Unmarshal(resultsByte, &expectedResults)
	check(t, err)
	expectedResult1 := expectedResults["test1output"]
	expectedResult2 := expectedResults["test2output"]

	outputChannel1 := make(chan [][]float32, numResults)
	outputChannel2 := make(chan [][]float32, numResults)
	errChannel := make(chan error, numWorkers)

	worker := func() {
		for range numEmbeddings {
			batchResult, threadErr := pipeline.RunPipeline([]string{"robert smith"})
			if threadErr != nil {
				errChannel <- threadErr
			}
			outputChannel1 <- batchResult.Embeddings
			batchResult, threadErr = pipeline.RunPipeline([]string{"robert smith junior", "francis ford coppola"})
			if threadErr != nil {
				errChannel <- threadErr
			}
			outputChannel2 <- batchResult.Embeddings
		}
	}

	for range numWorkers {
		go worker()
	}

	correctResults1 := 0
	correctResults2 := 0
loop:
	for {
		if correctResults1 == numResults && correctResults2 == numResults {
			break loop
		}
		select {
		case vectors := <-outputChannel1:
			for i, vector := range vectors {
				e := floatsEqual(vector, expectedResult1[i])
				if e != nil {
					t.Logf("Test 1: The threaded neural network didn't produce the correct result: %s\n", e)
					t.FailNow()
				}
			}
			correctResults1++
		case vectors := <-outputChannel2:
			for i, vector := range vectors {
				e := floatsEqual(vector, expectedResult2[i])
				if e != nil {
					t.Logf("Test 1: The threaded neural network didn't produce the correct result: %s\n", e)
					t.FailNow()
				}
			}
			correctResults2++
		case threadErr := <-errChannel:
			t.Fatal(threadErr)
		}
	}
}

// Utilities

func checkClassificationOutput(t *testing.T, inputResult []pipelines.ClassificationOutput, inputExpected []pipelines.ClassificationOutput) {
	t.Helper()
	assert.Equal(t, len(inputResult), len(inputExpected))
	for i, output := range inputResult {
		resultExpected := inputExpected[i]
		assert.Equal(t, output.Label, resultExpected.Label)
		assert.True(t, almostEqual(float64(output.Score), float64(resultExpected.Score)))
	}
}

// Returns an error if any element between a and b don't match.
func floatsEqual(a, b []float32) error {
	if len(a) != len(b) {
		return fmt.Errorf("length mismatch: %d vs %d", len(a), len(b))
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		// Arbitrarily chosen precision. Large enough not to be affected by quantization
		if diff >= 0.0007 {
			return fmt.Errorf("data element %d doesn't match: %.12f vs %.12f",
				i, a[i], b[i])
		}
	}
	return nil
}

func almostEqual(a, b float64) bool {
	return math.Abs(a-b) <= 0.0001
}

func check(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("Test failed with error %s", err.Error())
	}
}

func printTokenEntities(o *pipelines.TokenClassificationOutput) {
	for i, entities := range o.Entities {
		fmt.Printf("Input %d\n", i)
		for _, entity := range entities {
			fmt.Printf("%+v\n", entity)
		}
	}
}
