//go:build !GO && !ALL

package pipelineBackends

import "github.com/knights-analytics/hugot/options"

type GoModel struct{}

func createGoModelBackend(_ *Model, _ *options.Options) error {
	return nil
}

func createInputTensorsGo(_ *PipelineBatch, _ []InputOutputInfo) error {
	return nil
}

func runGoSessionOnBatch(_ *PipelineBatch, _ *BasePipeline) error {
	return nil
}
