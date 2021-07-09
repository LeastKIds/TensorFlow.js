const tf = require('@tensorflow/tfjs');

// 1. 데이터 값 넣어주기
const temp = [20,21,22,23];
const sales = [40,42,44,46];
const reason = tf.tensor(temp);
const result = tf.tensor(sales);

// console.log(reason.print());    // .print() : 안에 들어간 내용 표시
// console.log(result.print());

// 2. 모델의 모양 만들기
const X = tf.input({shape: [1]});   // shape : 원인의 개수
const Y = tf.layers.dense({ units : 1 }).apply(X);  // units : 결과의 개수, apply : 원인과 겨과를 이어주는 것
const model = tf.model({ inputs : X, outputs : Y });
// optimizer : 좀 더 효율적으로 모델을 만드는 방법, loss : 모델이 잘 만들어 졌는지 측정하는 방법
// meanSquaredError : 평균 제곱 오차 (MSE), rootMeanSquaredError(RMSE) : 평균 제곱근 오차
const compileParam = { optimizer : tf.train.adam(), loss : tf.losses.meanSquaredError }
model.compile(compileParam);

// 3. 데이터로 모델을 학습시키기
// const fitParam = {epochs : 30000 }    // epochs : 몇 번 학습을 할지, 30000번에 정확한 값 도출

// 학습을 시킬 때 얼마나 돌려야 되는지 알려줌
const fitParam = {
    epochs : 10000,
    callbacks : {
        onEpochEnd : function (epoch, logs) {
            console.log('epoch', epoch, logs, 'RMSE => ' , Math.sqrt(logs.loss));
        }
    }
}

model.fit(reason, result, fitParam).then(function (result) {

//    4. 모델을 이용
//    4.1 기존의 데이터를 이용
    const predictResult = model.predict(reason);
    console.log(predictResult.print());
//    실제에서는 20~23 까지의 값을 넣어서 결과를 도출하는것이 아닌
//    20,21의 값을 넣어서 학습 시킨 뒤
//    22,23의 값을 넣어서 나온 결과가 정답인지 아닌지 비교하는게
//    좀 더 정확해 진다.


    // 5. 사용
    const nextTemp = [15,16,17,18,19];
    const nextReason = tf.tensor(nextTemp);
    const nextResult = model.predict(nextReason);

    console.log(nextResult.print());
});

