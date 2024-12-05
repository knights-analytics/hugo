//go:build ALL

package hugot

import (
	"errors"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

var inputStrings = []string{
	"Tech Innovators Inc. Launches Revolutionary AI Platform",
	"Green Energy Solutions Unveils Next-Gen Solar Panels",
	"Global Ventures Co. Secures $2 Billion in Funding",
	"Creative Minds Studio Launches Virtual Creativity Hub",
	"Healthcare Partners Ltd. Introduces AI-Driven Diagnostics",
	"Future Finance Group Predicts Key Market Trends for 2024",
	"Premier Logistics LLC Expands Into New International Markets",
	"Dynamic Marketing Agency Announces Strategic Partnership",
	"Eco-Friendly Products Corp. Debuts Sustainable Tech Line",
	"Blue Ocean Enterprises Leads the Way in Marine Technology",
	"NextGen Software Solutions Rolls Out New Cloud Suite",
	"Innovative Construction Co. Breaks Ground on Green Projects",
	"Precision Engineering Ltd. Redefines Robotics Efficiency",
	"Elite Consulting Group Forecasts Industry Growth in 2024",
	"Urban Development LLC Transforms City Skylines Nationwide",
	"Digital Media Concepts Sets New Standards for AI Content Delivery",
	"Community Builders Inc. Wins National Housing Award",
	"Trusted Insurance Brokers Introduces Smart Policy Options",
	"Advanced Manufacturing Corp. Showcases Cutting-Edge Automation",
	"Visionary Design Studio Redefines Modern Architecture",
	"Strategic Investment Partners Reveals Key Acquisitions",
	"Modern Retail Solutions Integrates AI Shopping Experiences",
	"Efficient Energy Systems Revolutionizes Grid Technology",
	"High-Tech Components Inc. Develops Next-Gen Processors",
	"Education Outreach Network Empowers Communities with New Programs",
	"Healthcare Innovations Ltd. Drives Breakthrough in Medical Research",
	"Creative Film Productions Wins Prestigious Global Awards",
	"Global Trade Services Expands Globalized Shipping Network",
	"NextLevel Sports Management Signs High-Profile Athletes",
	"Sustainable Agriculture Group Promotes Organic Farming",
	"Cloud Based Solutions Unveils New Secure Data Services",
	"Tech Innovators Inc. to Host Annual Tech Summit This Fall",
}

func checkBench(b *testing.B, err error) {
	b.Helper()
	if err != nil {
		b.Fatalf("Test failed with error %s", err.Error())
	}
}

func worker(inputChannel chan []string, pipeline *pipelines.FeatureExtractionPipeline, b *testing.B, wg *sync.WaitGroup) {
	for inputs := range inputChannel {
		outputs, threadErr := pipeline.RunPipeline(inputs)
		if threadErr != nil {
			b.Error(threadErr)
		}
		if len(outputs.Embeddings) != len(inputs) {
			b.Error(errors.New("number of outputs does not match number of inputs"))
		}
	}
	wg.Done()
}

func threadBenchmark(b *testing.B, session *Session, multiThread bool, randomBatches bool) {

	b.Helper()
	b.StopTimer()

	numWorkers := 1
	if multiThread {
		numWorkers = runtime.NumCPU()
	}

	modelPath := "./models/KnightsAnalytics_all-MiniLM-L6-v2"
	config := FeatureExtractionConfig{
		ModelPath:    modelPath,
		Name:         "testPipeline",
		OnnxFilename: "model.onnx",
	}
	pipeline, err := NewPipeline(session, config)
	checkBench(b, err)

	inputChannel := make(chan []string, b.N)

	wg := sync.WaitGroup{}
	for range numWorkers {
		wg.Add(1)
		go worker(inputChannel, pipeline, b, &wg)
	}

	// warmup
	for range 10 {
		_, threadErr := pipeline.RunPipeline(inputStrings)
		if threadErr != nil {
			b.Error(threadErr)
		}
	}

	b.StartTimer()
	for range b.N {
		if !randomBatches {
			inputChannel <- inputStrings
		} else {
			randomNumber := rand.Intn(32) + 1
			inputChannel <- inputStrings[0:randomNumber]
		}
	}
	close(inputChannel)
	wg.Wait()
	b.StopTimer()
}

func BenchmarkORTThreadBenchmarkSingle(b *testing.B) {
	b.StopTimer()
	session, err := NewORTSession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 128 {
		b.N = 128
	}
	threadBenchmark(b, session, false, false)
}

func BenchmarkXLAThreadBenchmarkSingle(b *testing.B) {
	b.StopTimer()
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 128 {
		b.N = 128
	}
	threadBenchmark(b, session, false, false)
}

func BenchmarkORTThreadBenchmarkMulti(b *testing.B) {
	b.StopTimer()
	session, err := NewORTSession(
		options.WithInterOpNumThreads(1),
		options.WithIntraOpNumThreads(1),
		options.WithCpuMemArena(false),
		options.WithMemPattern(false),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 256 {
		b.N = 256
	}
	threadBenchmark(b, session, true, false)
}

func BenchmarkXLAThreadBenchmarkMulti(b *testing.B) {
	b.StopTimer()
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 256 {
		b.N = 256
	}
	threadBenchmark(b, session, true, false)
}

func BenchmarkORTThreadBenchmarkRandomBatchSize(b *testing.B) {
	b.StopTimer()
	session, err := NewORTSession(
		options.WithInterOpNumThreads(1),
		options.WithIntraOpNumThreads(1),
		options.WithCpuMemArena(false),
		options.WithMemPattern(false),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 256 {
		b.N = 256
	}
	threadBenchmark(b, session, true, true)
}

func BenchmarkXLAThreadBenchmarkRandomBatchSize(b *testing.B) {
	b.StopTimer()
	session, err := NewXLASession()
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 256 {
		b.N = 256
	}
	threadBenchmark(b, session, true, true)
}

func BenchmarkORTThreadBenchmarkCuda(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	b.StopTimer()
	session, err := NewORTSession(
		options.WithOnnxLibraryPath("/usr/lib64/onnxruntime-gpu/libonnxruntime.so"),
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 512 {
		b.N = 512
	}
	threadBenchmark(b, session, true, false)
}

func BenchmarkXLAThreadBenchmarkCuda(b *testing.B) {
	if os.Getenv("CI") != "" {
		b.SkipNow()
	}
	b.StopTimer()
	session, err := NewXLASession(
		options.WithCuda(map[string]string{
			"device_id": "0",
		}),
	)
	checkBench(b, err)
	defer func(session *Session) {
		destroyErr := session.Destroy()
		checkBench(b, destroyErr)
	}(session)
	if b.N < 512 {
		b.N = 512
	}
	threadBenchmark(b, session, true, false)
}
