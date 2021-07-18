const tf = require('@tensorflow/tfjs-node');

const tsOne = tf.tensor1d([10,13,16]);
const tsTwo = tf.tensor1d([1,3,5]);
console.log(tsOne.add(tsTwo).print());
console.log(tsOne.sub(tsTwo).print());

const tsOne_2 = tf.scalar(7);
const tsTwo_2 = tf.tensor1d([1,3,5]);
console.log(tsOne_2.add(tsTwo_2).print());

const tsOne_3 = tf.tensor1d([1,2,3]);
const tsTwo_3 = tf.tensor1d([4,5,6]);
console.log(tsOne_3.dot(tsTwo_3).print());

const tsOne_4 = tf.tensor1d([1,2,3]);
const tsTwo_4 = tf.tensor1d([4,5,6]);
console.log(tf.outerProduct(tsOne_4, tsTwo_4).print());

const tsOne_5 = tf.tensor2d([1,2,3,4],[2,2]);
const tsTwo_5 = tf.tensor2d([5,6,7,8],[2,2]);
console.log(tsOne_5.add(tsTwo_5).print());

// const tsOne_6 = tf.tensor2d([1,2,3,4,5,6,7,8,9],[3,3]);
// const tsTwo_6 = tf.tensor2d([9,8,7,1,2,3,4,5,6],[3,3]);
const tsOne_6 = tf.tensor2d([[1,2,3],[4,5,6],[7,8,9]]);
const tsTwo_6 = tf.tensor2d([[9,8,7],[1,2,3],[4,5,6]]);
console.log(tsOne_6.add(tsTwo_6).print());

const tsOne_7 = tf.tensor2d([[3,4],[5,6],[7,8]]);
const tsTwo_7 = tf.tensor1d([1,2]);
console.log(tsOne_7.add(tsTwo_7).print());

const tsOne_8 = tf.tensor2d([[1,2],[3,4]]);
const tsTwo_8 = tf.tensor2d([[5],[6]]);
console.log(tsOne_8.matMul(tsTwo_8).print());
console.log(tsOne_8.mul(tsTwo_8).print());

const tsOne_9 = tf.tensor2d([[1,2],[3,4]]);
const tsTwo_9 = tf.tensor2d([[5,6],[7,8]]);
console.log(tsOne_9.matMul(tsTwo_9).print());

const tsOne_10 = tf.tensor2d([[1,2,3],[4,5,6]]);
const tsTwo_10 = tf.tensor2d([[1,2],[3,4],[5,6]]);
console.log(tsOne_10.matMul(tsTwo_10).print());