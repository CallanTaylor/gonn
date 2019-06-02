package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"os"
	"strconv"
	"time"
)

func main() {
	net := CreateNet(784, 16, 10, 0.1)
	mnistTrain(&net)
	mnistPredict(&net)
}

func mnistTrain(net *Network) {
	rand.Seed(time.Now().UTC().UnixNano())
	start := time.Now()

	for epochs := 0; epochs < 1; epochs++ {
		training_data, err := os.Open("mnist_dataset/mnist_train.csv")
		if err == nil {
			reader := csv.NewReader(bufio.NewReader(training_data))
			for {
				record, err := reader.Read()
				if err == io.EOF {
					break
				}

				inputs := make([]float64, net.inputs)
				for i := range inputs {
					x, _ := strconv.ParseFloat(record[i], 64)
					inputs[i] = (x / 255.0 * 0.99) + 0.01
				}

				tragets := make([]float64, 10)
				for i := range tragets {
					tragets[i] = 0.1
				}
				x, _ := strconv.Atoi(record[0])
				tragets[x] = 0.9

				net.Train(inputs, tragets)
			}
		}
		training_data.Close()
	}
	elapsed := time.Since(start)
	fmt.Printf("Time taken to train = %s\n", elapsed)
}

func mnistPredict(net *Network) {
	start := time.Now()
	test_file, _ := os.Open("mnist_dataset/mnist_test.csv")
	defer test_file.Close()

	score := 0
	reader := csv.NewReader(bufio.NewReader(test_file))
	for {

		record, err := reader.Read()
		if err == io.EOF {
			break
		}

		inputs := make([]float64, net.inputs)
		for i := range inputs {
			x, _ := strconv.ParseFloat(record[i], 64)
			inputs[i] = (x / 255 * 0.99) + 0.01
		}

		outputs := net.Predict(inputs)
		best := 0
		maxTarget := 0.0
		for i := 0; i < net.outputs; i++ {
			if outputs.At(i, 0) > maxTarget {
				maxTarget = outputs.At(i, 0)
				best = i
			}
		}
		target, _ := strconv.Atoi(record[0])
		if target == best {
			score++
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("Time taken to predict 10000 digits = %s\n", elapsed)
	fmt.Println("score: ", score)
}
