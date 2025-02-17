package hugot

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"

	"github.com/knights-analytics/hugot/datasets"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelineBackends"
	"github.com/knights-analytics/hugot/pipelines"
	"github.com/knights-analytics/hugot/util"
)

type TrainingSession struct {
	runtime  string
	pipeline pipelineBackends.Pipeline
	config   TrainingConfig
}

func (s *TrainingSession) Destroy() error {
	err := s.pipeline.GetModel().Destroy()
	if err != nil {
		return err
	}
	s.pipeline = nil
	return nil
}

type TrainingOption func(eo *TrainingSession) error

type TrainingConfig struct {
	ModelPath          string
	OnnxFilename       string
	Cuda               bool
	Epochs             int
	XlaTrainingOptions *XLATrainingOptions
	Dataset            datasets.Dataset
	Verbose            bool
}

func newTrainingSession[T pipelineBackends.Pipeline](runtime string, config TrainingConfig) (*TrainingSession, error) {
	session := &TrainingSession{
		config:  config,
		runtime: runtime,
	}

	var trainingPipeline T
	var model *pipelineBackends.Model
	var err error

	opts := options.Defaults()
	opts.Runtime = runtime

	switch runtime {
	case "XLA":
		opts.XLAOptions.Cuda = config.Cuda
	default:
		return nil, fmt.Errorf("runtime %s is not supported", runtime)
	}

	if config.Epochs <= 0 {
		config.Epochs = 1
	}

	model, err = pipelineBackends.LoadModel(config.ModelPath, config.OnnxFilename, opts)
	if err != nil {
		return nil, err
	}

	switch any(trainingPipeline).(type) {
	case *pipelines.FeatureExtractionPipeline:
		pipelineConfig := FeatureExtractionConfig{}
		pipeline := any(trainingPipeline).(*pipelines.FeatureExtractionPipeline)
		pipeline, _, err = InitializePipeline(pipeline, pipelineConfig, opts, model)
		if err != nil {
			return nil, err
		}
		session.pipeline = pipeline

		// hook the dataset up with the pipeline for tokenization
		if d, ok := session.config.Dataset.(*datasets.SemanticSimilarityDataset); !ok {
			return nil, fmt.Errorf("expected SemanticSimilarityDataset, got %T", d)
		} else {
			if e := d.SetTokenizationPipeline(pipeline); e != nil {
				return nil, e
			}
		}
	default:
		return nil, fmt.Errorf("training for pipeline type is not supported")
	}

	if session.config.Verbose {
		session.config.Dataset.SetVerbose(true)
	}

	return session, nil
}

func (s *TrainingSession) Train() error {
	switch s.runtime {
	case "XLA":
		return TrainXLA(s)
	default:
		return fmt.Errorf("training runtime %s is not supported", s.runtime)
	}
}

// Save serializes the trained model as an onnx model.
// If a tokenizer is present, the tokenizer files are copied from the untrained model directory to the trained model.
// Path is the full path to the directory where the model will be saved.
func (s *TrainingSession) Save(path string) error {
	if path == "" {
		return fmt.Errorf("path is required")
	}

	var deferError error

	model := s.pipeline.GetModel()
	if model != nil {
		if s.runtime == "XLA" {
			xlaModel := model.XLAModel

			if xlaModel != nil {
				// we save the model locally in a temp directory first so we can support s3
				tempDir, err := os.MkdirTemp("", "hugot")
				if err != nil {
					return err
				}
				defer func() {
					deferError = errors.Join(deferError, os.RemoveAll(tempDir))
				}()

				if err := xlaModel.Save(util.PathJoinSafe(tempDir, "model.onnx")); err != nil {
					return err
				}

				if err := util.MoveFile(util.PathJoinSafe(tempDir, "model.onnx"), path); err != nil {
					return err
				}

				if model.Tokenizer != nil {
					if err := copyTokenizer(model.Path, path); err != nil {
						return err
					}
				}
				return nil
			}

			return fmt.Errorf("XLA model is nil")
		} else {
			return fmt.Errorf("XLA runtime is required for saving a training model")
		}
	} else {
		return fmt.Errorf("pipeline model is nil")
	}
}

func copyTokenizer(from, to string) error {
	toCopy := map[string]bool{
		"special_tokens_map.json": true,
		"tokenizer_config.json":   true,
		"tokenizer.json":          true,
	}

	walker := func(_ context.Context, _ string, parent string, info os.FileInfo, _ io.Reader) (toContinue bool, err error) {
		if toCopy[info.Name()] {
			if err := util.CopyFile(util.PathJoinSafe(from, parent, info.Name()), to); err != nil {
				return false, err
			}
		}
		return true, nil
	}
	return util.WalkDir()(context.Background(), from, walker)
}
