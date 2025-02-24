//go:build !NODOWNLOAD

package hugot

import (
	"errors"
	"fmt"
	"path"
	"path/filepath"
	"strings"
	"time"

	"github.com/gomlx/go-huggingface/hub"

	"github.com/knights-analytics/hugot/util"
)

// DownloadOptions is a struct of options that can be passed to DownloadModel.
type DownloadOptions struct {
	AuthToken             string
	OnnxFilePath          string
	Branch                string
	MaxRetries            int
	RetryInterval         int
	ConcurrentConnections int
	Verbose               bool
}

// NewDownloadOptions creates new DownloadOptions struct with default values.
// Override the values to specify different download options.
func NewDownloadOptions() DownloadOptions {
	d := DownloadOptions{}
	d.Branch = "main"
	d.MaxRetries = 5
	d.RetryInterval = 5
	d.ConcurrentConnections = 5
	return d
}

// DownloadModel can be used to download a model directly from huggingface. Before the model is downloaded,
// validation occurs to ensure there is an .onnx and tokenizers.json file. Hugot only works with onnx models.
func DownloadModel(modelName string, destination string, options DownloadOptions) (string, error) {

	// replicates code in hf downloader
	modelP := modelName
	if strings.Contains(modelP, ":") {
		modelP = strings.Split(modelName, ":")[0]
	}
	modelPath := path.Join(destination, strings.Replace(modelP, "/", "_", -1))

	repo := hub.New(modelName)
	if options.AuthToken != "" {
		repo = repo.WithAuth(options.AuthToken)
	}
	if options.ConcurrentConnections != 0 {
		repo.MaxParallelDownload = options.ConcurrentConnections
	}
	if options.Verbose {
		repo.Verbosity = 1
	} else {
		repo.Verbosity = 0
		repo.WithProgressBar(false)
	}
	if options.Branch != "" {
		repo.WithRevision(options.Branch)
	}

	// make sure it's an onnx model with tokenizer
	downloadFiles, err := validateDownloadHfModel(repo, options)
	if err != nil {
		return "", err
	}

	for i := 0; i < options.MaxRetries; i++ {
		downloadPaths, downloadErr := repo.DownloadFiles(downloadFiles...)
		if downloadErr != nil {
			if options.Verbose {
				fmt.Printf("Warning: attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, downloadErr)
			}
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
			continue
		}

		for j, downloadPath := range downloadPaths {
			truePath, symErr := filepath.EvalSymlinks(downloadPath)
			if symErr != nil {
				return "", symErr
			}
			moveErr := util.CopyFile(truePath, fmt.Sprintf("%s/%s", modelPath, path.Base(downloadFiles[j])))
			if moveErr != nil {
				return "", moveErr
			}
		}

		if options.Verbose {
			fmt.Printf("\nDownload of %s completed successfully\n", modelName)
		}
		return modelPath, nil
	}

	return "", fmt.Errorf("failed to download %s after %d attempts", modelName, options.MaxRetries)
}

func validateDownloadHfModel(repo *hub.Repo, options DownloadOptions) ([]string, error) {

	onnxFilePath := options.OnnxFilePath

	for i := 0; i < options.MaxRetries; i++ {
		err := repo.DownloadInfo(false)
		if err != nil {
			if options.Verbose {
				fmt.Printf("Warning: list repo attempt %d / %d failed, error: %s\n", i+1, options.MaxRetries, err)
			}
			if i+1 == options.MaxRetries {
				return nil, err
			}
			time.Sleep(time.Duration(options.RetryInterval) * time.Second)
		}
	}

	tokenizerPath := ""
	onnxPath := ""
	var toDownload []string
	var allOnnx []string
	for fileName, err := range repo.IterFileNames() {
		if err != nil {
			return nil, err
		}

		baseFileName := filepath.Base(fileName)
		if baseFileName == "tokenizer.json" {
			tokenizerPath = fileName
		} else if baseFileName == "special_tokens_map.json" ||
			baseFileName == "tokenizer_config.json" ||
			baseFileName == "config.json" ||
			baseFileName == "vocab.txt" {
			toDownload = append(toDownload, fileName)
		} else if filepath.Ext(baseFileName) == ".onnx" {
			if onnxFilePath != "" {
				if fileName == onnxFilePath {
					onnxPath = fileName
				}
			} else {
				onnxPath = fileName
			}
			allOnnx = append(allOnnx, fileName)
		}
	}

	var errs []error
	if tokenizerPath == "" {
		errs = append(errs, fmt.Errorf("model does not have a tokenizer.json file"))
	}
	if onnxFilePath != "" {
		if onnxPath == "" {
			errs = append(errs, fmt.Errorf("model .onnx file not found at %s", onnxFilePath))
		}
	} else {
		numModels := len(allOnnx)
		if numModels == 0 {
			errs = append(errs, fmt.Errorf("model does not have a .onnx file, Hugot only works with onnx models"))
		} else if numModels > 1 {
			errs = append(errs, fmt.Errorf("model has multiple .onnx files, please specify one of the following onnxFilePaths: %s", strings.Join(allOnnx, " ")))
		}
	}

	return append(toDownload, tokenizerPath, onnxPath), errors.Join(errs...)
}
