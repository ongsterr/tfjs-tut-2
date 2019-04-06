import React, { Component } from 'react'
import * as tfvis from '@tensorflow/tfjs-vis'
import * as tf from '@tensorflow/tfjs'

import './App.css'
import { MnistData } from './data.js'

class App extends Component {
  classNames = [
    'Zero',
    'One',
    'Two',
    'Three',
    'Four',
    'Five',
    'Six',
    'Seven',
    'Eight',
    'Nine',
  ]

  async componentDidMount() {
    await this.run()
  }

  async showExamples(data) {
    // Create a container in the visor
    const surface = tfvis
      .visor()
      .surface({ name: 'Input Data Examples', tab: 'Input Data' })

    // Get the examples
    const examples = data.nextTestBatch(20)
    const numExamples = examples.xs.shape[0]

    // Create a canvas element to render each example
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        // reshape the tensor to 28x28 px
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1])
      })

      const canvas = document.createElement('canvas')
      canvas.width = 28
      canvas.height = 28
      canvas.style = 'margin: 4px;'
      await tf.browser.toPixels(imageTensor, canvas)
      surface.drawArea.appendChild(canvas)

      imageTensor.dispose()
    }
  }

  getModel() {
    const model = tf.sequential()

    const IMAGE_WIDTH = 28
    const IMAGE_HEIGHT = 28
    const IMAGE_CHANNELS = 1

    // In the first layer of our convolutional neural network, we have to specify the input shape.
    // Then, we specify some parametersfor the convolution operation that takes place in this layer.
    model.add(
      tf.layers.conv2d({
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
        kernelSize: 5,
        filters: 8,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    )

    // The MaxPooling layer acts as a sort of downsampling using max values in a region instead of averaging
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

    // Repeat another conv2d + maxpooling stack
    // Note that we have more filters in convolution
    model.add(
      tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
      })
    )
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }))

    // Now we flatten the output from the 2D filters into a 1D vector to prepare it for input into our last layer
    // This is common practice when feeding higher dimensional data to a final classification output layer
    model.add(tf.layers.flatten())

    // Our last layer is a dense layer which has 10 outputs units, one for each output class
    const NUM_OUTPUT_CLASSES = 10
    model.add(
      tf.layers.dense({
        units: NUM_OUTPUT_CLASSES,
        kernelInitializer: 'varianceScaling',
        activation: 'softmax',
      })
    )

    // Choose an optimizer, loss function and accuracy metric, then compile and return the model
    const optimizer = tf.train.adam()
    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metric: ['accuracy'],
    })

    return model
  }

  async train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc']
    const container = {
      name: 'Model Training',
      styles: {
        height: '1000px',
      },
    }

    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics)

    const BATCH_SIZE = 512
    const TRAIN_DATA_SIZE = 5500
    const TEST_DATA_SIZE = 1000

    // This is to preapre data as tensors
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TRAIN_DATA_SIZE)
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels]
    })

    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE)
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels]
    })

    //  We also pass a validationData property to indicate which data the model should use to test itself after each epoch (but not use for training).
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks,
    })
  }

  doPrediction(model, data, testDataSize = 500) {
    const IMAGE_WIDTH = 28
    const IMAGE_HEIGHT = 28

    const testData = data.nextTestBatch(testDataSize)
    const testxs = testData.xs.reshape([
      testDataSize,
      IMAGE_WIDTH,
      IMAGE_HEIGHT,
      1,
    ])
    const labels = testData.labels.argMax([-1])

    const preds = model.predict(testxs).argMax([-1])

    testxs.dispose()
    return [preds, labels]
  }

  async showAccuracy(model, data) {
    const [preds, labels] = this.doPrediction(model, data)
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds)
    const container = { name: 'Accuracy', tab: 'Evaluation' }
    tfvis.show.perClassAccuracy(container, classAccuracy, this.classNames)

    labels.dispose()
  }

  async showConfusion(model, data) {
    const [preds, labels] = this.doPrediction(model, data)
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds)
    const container = { name: 'Confusion Matrix', tab: 'Evaluation' }
    tfvis.render.confusionMatrix(
      container,
      { values: confusionMatrix },
      this.classNames
    )

    labels.dispose()
  }

  async run() {
    const data = new MnistData()
    await data.load()
    await this.showExamples(data)

    const model = this.getModel()
    tfvis.show.modelSummary({ name: 'Model Architecture' }, model)

    await this.train(model, data)

    await this.showAccuracy(model, data)
    await this.showConfusion(model, data)
  }

  render() {
    return <div className="App">Hello Tensorflow</div>
  }
}

export default App
