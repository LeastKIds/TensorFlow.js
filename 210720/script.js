// const tf = require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs');
// import tf from '@tensorflow/tfjs';

async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map(car => ({
        mpg : car.Miles_per_Gallon,
        horsepower : car.Horsepower,
    })).filter(car => (car.mpg != null && car.horsepower != null));
    return cleaned;
}

async function run() {
    const data = await getData();
    const values = data.map(d => ({
        x : d.horsepower,
        y : d.mpg,
    }));

    tfvis.render.scatterplot(
        {name : 'Horsepower v MPG'},
        {values},
        {
            xLabel : 'Horsepower',
            yLabel : 'MPG',
            height : 300,
        },
    );

    const model = createModel();
    tfvis.show.modelSummary({name : 'Model Summary'}, model);
}
// DOMContentLoaded : 브라우저가 html을 전부 읽고 DOM트리를 완성하는 즉시 발생
document.addEventListener('DOMContentLoaded', run);

function createModel() {
    const model = tf.sequential();  // 모델 객체를 인스턴스화 함. 입력이 바로 출력으로 가기 때문에 sequential모델
    model.add(tf.layers.dense({inputShape : [1], units : 1, useBias : true}));
    model.add(tf.layers.dense({units : 1, useBias : true}));
    return model;
}