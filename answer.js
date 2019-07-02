/**
 * 取回資料
 */
async function getData() {
    const carsDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataReq.json();
    let cleaned = new Array();
    for (let car of carsData) {
        if (car.Miles_per_Gallon != null && car.Horsepower != null) {
            cleaned.push({
                mpg: car.Miles_per_Gallon,
                horsepower: car.Horsepower
            });
        }
    }
    return cleaned;
}
/**
 * 執行的核心
 */
async function run() {
    const data = await getData();
    // Create the model
    const model = createModel();
    const { normalizedInputs, normalizedOutputs, inputMax,
        inputMin, outputMax, outputMin } = prepareData(data);
    await trainModel(model, normalizedInputs, normalizedOutputs);
    testModel(model, inputMin, inputMax, outputMin, outputMax, data);
}
/**
 * 建立模型的形狀
 */
function createModel() {
    // Create a sequential model
    const model = tf.sequential();
    // Add a single hidden layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));
    // Add an output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));
    return model;
}
/**
 * 將資料轉成 tensor 並且標準化
 */
function prepareData(data) {
    let inputs = tf.tensor(data.map(d => d.horsepower));
    let outputs = tf.tensor(data.map(d => d.mpg));
    const inputMax = inputs.max();
    const inputMin = inputs.min();
    const outputMax = outputs.max();
    const outputMin = outputs.min();
    let normalizedInputs = inputs.sub(inputMin).div(inputMax.sub(inputMin));
    let normalizedOutputs = outputs.sub(outputMin).div(outputMax.sub(outputMin));
    return {
        normalizedInputs,
        normalizedOutputs,
        inputMax,
        inputMin,
        outputMax,
        outputMin
    };
}
/**
 * 
 * 設定參數並進行訓練
 */
async function trainModel(model, input, output) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    });
    const batchSize = 28;
    const epochs = 50;
    const history = await model.fit(input, output, {
        batchSize,
        epochs,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            {
                height: 200,
                callbacks: ['onEpochEnd']
            }
        )
    });
    console.log("done");
    console.log(history);
}

function testModel(model, inputMin, inputMax, outputMin, outputMax, data) {
    const xs = tf.linspace(0, 1, 100);
    const preds = model.predict(xs);
    const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);

    const unNormPreds = preds
        .mul(outputMax.sub(outputMin))
        .add(outputMin);

    let newXs=unNormXs.dataSync();//轉成陣列
    let newPreds=unNormPreds.dataSync();

    const predictedPoints = new Array();
    for(let i=0; i<newXs.length; i++){
        predictedPoints.push({
            x: newXs[i],
            y: newPreds[i]
        });
    }

    const originalPoints = data.map(d => ({
        x: d.horsepower, y: d.mpg,
    }));

    tfvis.render.scatterplot(
        {name: 'Model Predictions vs Original Data'}, 
        {values: [originalPoints, predictedPoints], series: ['original', 'predicted']}, 
        {
          xLabel: 'Horsepower',
          yLabel: 'MPG',
          height: 300
        }
      );
}
document.addEventListener('DOMContentLoaded', run);