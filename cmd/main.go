package main

// package hugot

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"strings"
	"sync"

	"github.com/mattn/go-isatty"
	"github.com/urfave/cli/v2"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	util "github.com/knights-analytics/hugot/utils"
)

var modelPath string
var inputPath string
var outputPath string
var pipelineType string
var sharedLibraryPath string
var batchSize int
var modelsDir string

var runCommand = &cli.Command{
	Name:  "run",
	Usage: "Run a huggingface pipeline on input data",
	Description: `Run expects a path to a file with input in .jsonl format. Each json line in the file must be of the format {"input": "input string"} to be processed.
				`,
	ArgsUsage: `
				--input: path to a .jsonl file or a folder with .jsonl files to process. If omitted, the input will be read from stdin.
				--output: path to a folder where to write the output. If omitted, the output will be sent to stdout.
				--model: model name or path to the .onnx model to load. The hugot cli looks for models with this chain: first use the provided path. If the path does not exist, look for a model
				with this name at $HOME/hugot/models. Finally, try to download the model from Huggingface and use it.
				--type: pipeline type. Currently implemented types are: featureExtraction, tokenClassification, and textClassification (only single label)
				--onnxruntimeSharedLibrary: path to the onnxruntime.so library. If not provided, the cli will try to load it from $HOME/lib/hugot/onnxruntime.so, and from /usr/lib/onnxruntime.so in the last instance.
				`,
	Flags: []cli.Flag{
		&cli.StringFlag{
			Name:        "model",
			Usage:       "Path to the model",
			Aliases:     []string{"p"},
			Destination: &modelPath,
			Required:    true,
		},
		&cli.StringFlag{
			Name:        "input",
			Usage:       "Path to the input data",
			Aliases:     []string{"i"},
			Destination: &inputPath,
		},
		&cli.StringFlag{
			Name:        "output",
			Usage:       "Path to output",
			Aliases:     []string{"o"},
			Destination: &outputPath,
		},
		&cli.StringFlag{
			Name:        "type",
			Usage:       "Pipeline type",
			Aliases:     []string{"t"},
			Destination: &pipelineType,
			Required:    true,
		},
		&cli.StringFlag{
			Name:        "onnxruntimeSharedLibrary",
			Usage:       "Path to onnxruntime.so",
			Aliases:     []string{"s"},
			Destination: &sharedLibraryPath,
			Required:    false,
		},
		&cli.IntFlag{
			Name:        "batchSize",
			Usage:       "Number of inputs to process in a batch",
			Aliases:     []string{"b"},
			Destination: &batchSize,
			Required:    false,
			Value:       20,
		},
		&cli.StringFlag{
			Name:        "modelFolder",
			Usage:       "Folder where to store downloaded models. Falls back to $HOME/hugot/models if not specified",
			Aliases:     []string{"f"},
			Destination: &modelsDir,
			Required:    false,
			Value:       "",
		},
	},
	Action: func(ctx *cli.Context) error {
		var opts []hugot.WithOption

		if modelsDir == "" {
			userDir, err := os.UserHomeDir()
			if err != nil {
				return err
			}
			modelsDir = util.PathJoinSafe(userDir, "hugot", "models")
		}

		if sharedLibraryPath != "" {
			opts = append(opts, hugot.WithOnnxLibraryPath(sharedLibraryPath))
		} else {
			homeDir, err := os.UserHomeDir()
			if err != nil {
				if exists, err := util.FileSystem.Exists(ctx.Context, path.Join(homeDir, "lib", "hugot", "onnxruntime.so")); err != nil && exists {
					opts = append(opts, hugot.WithOnnxLibraryPath(path.Join(homeDir, "lib", "hugot", "onnxruntime.so")))
				}
			}
		}

		session, err := hugot.NewSession(opts...)
		if err != nil {
			return err
		}

		var setupErrs []error

		defer func() {
			err := session.Destroy()
			setupErrs = append(setupErrs, err)
		}()

		var pipe pipelines.Pipeline

		// is the model a full path to a model
		ok, err := util.FileSystem.Exists(ctx.Context, modelPath)
		if err != nil {
			return err
		}
		if !ok {
			// is the model the name of a model previously downloaded
			downloadedModelName := strings.Replace(modelPath, "/", "_", -1)
			ok, err = util.FileSystem.Exists(ctx.Context, util.PathJoinSafe(modelsDir, downloadedModelName))
			if err != nil {
				return err
			}
			if ok {
				modelPath = util.PathJoinSafe(modelsDir, downloadedModelName)
			} else {
				// is the model the name of a model to download
				if strings.Contains(modelPath, ":") {
					return fmt.Errorf("filters with : are currently not supported")
				}
				err = util.FileSystem.Create(context.Background(), modelsDir, os.ModePerm, true)
				if err != nil {
					return err
				}
				modelPath, err = session.DownloadModel(modelPath, modelsDir, hugot.NewDownloadOptions())
				if err != nil {
					return err
				}
			}
		}

		switch pipelineType {
		case "tokenClassification":
			config := hugot.TokenClassificationConfig{
				ModelPath: modelPath,
				Name:      "cliPipeline",
			}
			pipe, err = hugot.NewPipeline(session, config)
			setupErrs = append(setupErrs, err)
		case "textClassification":
			config := hugot.TextClassificationConfig{
				ModelPath: modelPath,
				Name:      "cliPipeline",
			}
			pipe, err = hugot.NewPipeline(session, config)
			setupErrs = append(setupErrs, err)
		case "featureExtraction":
			config := hugot.FeatureExtractionConfig{
				ModelPath: modelPath,
				Name:      "cliPipeline",
			}
			pipe, err = hugot.NewPipeline(session, config)
			setupErrs = append(setupErrs, err)
		default:
			setupErrs = append(setupErrs, fmt.Errorf("pipeline type %s not implemented", pipelineType))
		}
		if e := errors.Join(setupErrs...); e != nil {
			return e
		}

		inputChannel := make(chan []input, 1000)
		processedChannel := make(chan []byte, 1000)
		errorsChannel := make(chan error, 1000)
		nWriteWorkers := 1
		nProcessWorkers := 1
		var processedWg, writeWg sync.WaitGroup

		for i := 0; i < nProcessWorkers; i++ {
			go processWithPipeline(&processedWg, inputChannel, processedChannel, errorsChannel, pipe)
			processedWg.Add(1)
		}

		var writers []struct {
			Writer io.WriteCloser
			Type   string
		}

		for i := 0; i < nWriteWorkers; i++ {
			var writer io.WriteCloser

			if outputPath != "" {
				dest := util.PathJoinSafe(outputPath, fmt.Sprintf("result-%d.jsonl", i))
				writer, err = util.FileSystem.NewWriter(ctx.Context, dest, os.ModePerm)
				if err != nil {
					return err
				}
			} else {
				writer = os.Stdout
			}

			writers = append(writers, struct {
				Writer io.WriteCloser
				Type   string
			}{
				Writer: writer,
				Type:   "stdout",
			})
			writeWg.Add(1)
			go writeOutputs(&writeWg, processedChannel, errorsChannel, writer)
		}

		defer func() {
			for _, writer := range writers {
				if writer.Type != "stdout" {
					err = errors.Join(err, writer.Writer.Close())
				}
			}
		}()

		// read inputs

		exists, err := util.FileSystem.Exists(ctx.Context, inputPath)
		if err != nil {
			return err
		}
		exists = inputPath != "" && exists

		if exists {
			fileWalker := func(_ context.Context, _ string, _ string, info os.FileInfo, reader io.Reader) (toContinue bool, err error) {
				extension := filepath.Ext(info.Name())
				if extension == ".jsonl" {
					err := readInputs(reader, inputChannel)
					if err != nil {
						return false, err
					}
				}
				return true, nil
			}

			err := util.FileSystem.Walk(ctx.Context, inputPath, fileWalker)
			if err != nil {
				return err
			}
		} else {
			if inputPath != "" {
				return fmt.Errorf("file %s does not exist", inputPath)
			}

			if !isatty.IsTerminal(os.Stdin.Fd()) && !isatty.IsCygwinTerminal(os.Stdin.Fd()) {
				// there is something to process on stdin
				err := readInputs(os.Stdin, inputChannel)
				if err != nil {
					return err
				}
			}
		}

		close(inputChannel)
		processedWg.Wait()
		close(processedChannel)
		close(errorsChannel)
		writeWg.Wait()
		return err
	},
}

func main() {
	app := &cli.App{
		Name:     "hugot",
		Usage:    "Huggingface transformers from the command line - alpha",
		Commands: []*cli.Command{runCommand},
	}
	if err := app.Run(os.Args); err != nil {
		panic(err)
	}
}

func writeOutputs(wg *sync.WaitGroup, processedChannel chan []byte, errorChannel chan error, writeTarget io.WriteCloser) {
	for processedChannel != nil || errorChannel != nil {
		select {
		case output, ok := <-processedChannel:
			if !ok {
				processedChannel = nil
			}
			_, err := writeTarget.Write(output)
			if err != nil {
				panic(err)
			}
			_, err = writeTarget.Write([]byte("\n"))
			if err != nil {
				panic(err)
			}
		case err, ok := <-errorChannel:
			if !ok {
				errorChannel = nil
			}
			if err != nil {
				_, err = os.Stderr.WriteString(err.Error())
				if err != nil {
					panic(err)
				}
			}
		}
	}
	wg.Done()
}

func processWithPipeline(wg *sync.WaitGroup, inputChannel chan []input, processedChannel chan []byte, errorsChannel chan error, p pipelines.Pipeline) {
	for inputBatch := range inputChannel {
		inputStrings := make([]string, len(inputBatch))
		for i := 0; i < len(inputBatch); i++ {
			inputStrings[i] = inputBatch[i].Input
		}
		output, err := p.Run(inputStrings)
		if err != nil {
			errorsChannel <- err
		} else {
			batchOutputs := output.GetOutput()
			for i, batchOutput := range batchOutputs {
				out := inputBatch[i]
				out.Output = batchOutput
				outputBytes, marshallErr := json.Marshal(out)
				if marshallErr != nil {
					errorsChannel <- marshallErr
				} else {
					processedChannel <- outputBytes
				}
			}
		}
	}
	wg.Done()
}

func readInputs(inputSource io.Reader, inputChannel chan []input) error {
	inputBatch := make([]input, 0, 20)

	scanner := bufio.NewScanner(inputSource)
	for scanner.Scan() {
		var line input
		err := json.Unmarshal(scanner.Bytes(), &line)
		if err != nil {
			return err
		}
		inputBatch = append(inputBatch, line)
		if len(inputBatch) == batchSize {
			inputChannel <- inputBatch
			inputBatch = []input{}
		}
	}
	// flush
	if len(inputBatch) > 0 {
		inputChannel <- inputBatch
	}
	return nil
}

type input struct {
	Input  string `json:"input"`
	Output any    `json:"output"`
}
