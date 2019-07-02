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
    
}
/**
 * 帶入測試資料，測試模型
 */
function testModel(model, inputMin, inputMax, outputMin, outputMax, data) {
    
}
document.addEventListener('DOMContentLoaded', run);