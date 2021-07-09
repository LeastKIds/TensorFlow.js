const tf = require('@tensorflow/tfjs');

// 1. 데이터 값 넣어주기
const temp = [20,21,22,23];
const sales = [40,42,44,46];
const reason = tf.tensor(temp);
const result = tf.tensor(sales);

console.log(reason.print());    // .print() : 안에 들어간 내용 표시
console.log(result.print());

// 2.
// 2. 모델의 모양 만들기
const X = tf.input({shape: [1]});
const Y = tf.layers.dense({ units : 1 }).apply(X);
const model = tf.model({ inputs : X, outputs : Y });
// optimizer : 좀 더 효율적으로 모델을 만드는 방법, loss : 모델이 잘 만들어 졌는지 측정하는 방법
const compileParam = { optimizer : tf.train.adam(), loss : tf.losses.meanSquaredError }
model.compile(compileParam);