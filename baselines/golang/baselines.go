package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"time"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
	// "reflect"
)

func check(err error) {
	if err != nil {
		panic(err.Error())
	}
}

func bToMb(b uint64) uint64 {
	return b / 1024 / 1024
}

func printMemUsage() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
	fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
	fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
	fmt.Printf("\tNumGC = %v\n", m.NumGC)
}

func extractFeatures(records [][]string, featurePipeline *pipelines.FeatureExtractionPipeline) ([][]float32, time.Duration, int) {
	start := time.Now()

	const batchSize = 32
	var output [][]float32
	totalProcessed := 0
	// records = records[:100]
	batch := make([]string, 0, batchSize) // Pre-allocate capacity for the batch

	fmt.Println("starting pipeline loop")

	for _, row := range records {
		batch = append(batch, row[5])
		if len(batch) == batchSize {
			batchResult, err := featurePipeline.RunPipeline(batch)
			if err != nil {
				fmt.Println("Error running pipeline:", err)
				continue // Skip this batch and continue
			}

			output = append(output, batchResult.Embeddings...)
			totalProcessed += batchSize
			batch = batch[:0] // Reset batch without reallocating
		}
	}

	// Process any remaining records in the last batch
	if len(batch) > 0 {
		fmt.Println("processing remaining batch")
		batchResult, err := featurePipeline.RunPipeline(batch)
		if err != nil {
			fmt.Println("Error running pipeline:", err)
			panic(err.Error())
		}
		output = append(output, batchResult.Embeddings...)
		totalProcessed += len(batch)
	}

	duration := time.Since(start)
	return output, duration, totalProcessed
}

func main() {
	metrics := make(map[string]interface{})

	// new hugot instance
	startInitialization := time.Now()
	// session, err := hugot.NewSession()
	session, err := hugot.NewSession(
		hugot.WithInterOpNumThreads(1),
		hugot.WithIntraOpNumThreads(1),
		hugot.WithCpuMemArena(false),
		hugot.WithMemPattern(false),
	)
	check(err)
	defer func(session *hugot.Session) {
		err := session.Destroy()
		check(err)
	}(session)

	// access appropriate huggingface model
	modelPath, err := session.DownloadModel("KnightsAnalytics/all-MiniLM-L6-v2", "./", hugot.NewDownloadOptions())
	check(err)
	config := hugot.FeatureExtractionConfig{
		ModelPath: modelPath,
		Name:      "testPipeline",
	}

	// download data
	file, err := os.Open("text_data.csv")
	if err != nil {
		fmt.Println("Error:", err)
		panic(err.Error())
	}
	defer file.Close()
	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		fmt.Println("Error:", err)
		panic(err.Error())
	}
	fmt.Println("csv downloaded")

	// create new pipeline
	featurePipeline, err := hugot.NewPipeline(session, config)
	endInitialization := time.Since(startInitialization)
	fmt.Println("feature pipeline created in ", endInitialization, "seconds")
	metrics["startup time"] = endInitialization

	// run over multiple iters and find avg runtime
	var totalTime float64 = 0
	numIters := 1
	timePerIter := make([]float64, 0, numIters)
	var vector [][]float32
	for i := 0; i < numIters; i++ {
		output, duration, totalProcessed := extractFeatures(records, featurePipeline)
		// fmt.Println("expected 100 embeddings, got ", len(output))
		seconds := duration.Seconds()
		timePerIter = append(timePerIter, seconds)
		totalTime = totalTime + duration.Seconds()
		fmt.Printf("Iteration %d: Processed %d inputs in %f seconds\n", i+1, totalProcessed, duration.Seconds())
		vector = output

		fmt.Printf("Memory usage after iteration %d:\n", i+1)
		printMemUsage() // Track memory usage after each iteration
	}

	metrics["time per iteration"] = timePerIter
	metrics["average runtime"] = totalTime / float64(numIters)
	fmt.Println(metrics)
	fmt.Println(len(vector))

	/**
		uncomment following lines to compare Golang/python outputs
	**/
	// file1, err1 := os.Open("output.csv")
	// if err1 != nil {
	// 	fmt.Println("Error:", err)
	// 	panic(err.Error())
	// }
	// defer file.Close()

	// reader1 := csv.NewReader(file1)
	// records1, err := reader1.ReadAll()

	// compareOutputs(vector, records1)
}

// func compareOutputs(records [][]float32, otherRecords [][]string) {
// 	for row_num, row := range otherRecords {
// 		for ind, entry := range row {
// 			floatValue, _ := strconv.ParseFloat(entry, 32)
// 			diff := (float32(floatValue) - records[row_num][ind])
// 			if diff >= 0.0001 {
// 				fmt.Println("error on row", row, "entry", ind)
// 				panic("diff too large")
// 			}
// 		}
// 		fmt.Println("row successfully compared")
// 	}

// }
