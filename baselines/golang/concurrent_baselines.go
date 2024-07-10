package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/pipelines"
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
	records = records[:5000]

	type result struct {
		embeddings [][]float32
		err        error
	}

	batchCh := make(chan []string, len(records)/batchSize+1)
	resultCh := make(chan result, len(records)/batchSize+1)
	var wg sync.WaitGroup

	// Worker function
	worker := func() {
		defer wg.Done()
		for batch := range batchCh {
			batchResult, err := featurePipeline.RunPipeline(batch)
			resultCh <- result{batchResult.Embeddings, err}
		}
	}

	// Start workers
	numWorkers := 4 // Adjust based on your available CPU cores
	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go worker()
	}

	// Distribute batches to workers
	go func() {
		var batch []string
		for _, row := range records {
			batch = append(batch, row[5])
			if len(batch) == batchSize {
				batchCh <- batch
				batch = nil
			}
		}
		if len(batch) > 0 {
			batchCh <- batch
		}
		close(batchCh)
	}()

	// Collect results
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	for res := range resultCh {
		if res.err != nil {
			fmt.Println("Error running pipeline:", res.err)
			continue // Skip this batch and continue
		}
		output = append(output, res.embeddings...)
		totalProcessed += len(res.embeddings)
	}

	duration := time.Since(start)
	return output, duration, totalProcessed
}

func main() {
	metrics := make(map[string]interface{})
	fmt.Println(runtime.NumCPU())

	// new hugot instance
	startInitialization := time.Now()
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
		seconds := duration.Seconds()
		timePerIter = append(timePerIter, seconds)
		totalTime += seconds
		fmt.Printf("Iteration %d: Processed %d inputs in %f seconds\n", i+1, totalProcessed, seconds)
		vector = output

		fmt.Printf("Memory usage after iteration %d:\n", i+1)
		printMemUsage() // Track memory usage after each iteration
	}

	metrics["time per iteration"] = timePerIter
	metrics["average runtime"] = totalTime / float64(numIters)

	fmt.Println(metrics)
	fmt.Println(len(vector))

	file3, err := os.Create("golang_embeddings.csv")
	if err != nil {
		fmt.Println("failed")
	}
	defer file3.Close()

	// Create a CSV writer
	writer := csv.NewWriter(file3)
	defer writer.Flush()

	// sort embeddings because currently appened out of order
	sort.Slice(vector, func(i, j int) bool {
		return vector[i][0] < vector[j][0]
	})

	// fmt.Println((vector))
	// fmt.Println(vector)

	// file1, err1 := os.Open("python_embeddings.csv")
	// if err1 != nil {
	// 	fmt.Println("Error:", err)
	// 	panic(err.Error())
	// }
	// defer file1.Close()

	// reader1 := csv.NewReader(file1)
	// records1, err := reader1.ReadAll()
	// if err != nil {
	// 	fmt.Println("Error:", err)
	// 	panic(err.Error())
	// }

	// compareOutputs(vector, records1)
}

// func compareOutputs(records [][]float32, otherRecords [][]string) {
// 	for rowNum, row := range otherRecords {
// 		for ind, entry := range row {
// 			floatValue, _ := strconv.ParseFloat(entry, 32)
// 			diff := float32(floatValue) - records[rowNum][ind]
// 			if diff >= 0.0001 {
// 				fmt.Println("error on row", rowNum, "entry", ind)
// 				panic("diff too large")
// 			}
// 		}
// 		fmt.Println("row successfully compared")
// 	}
// }
