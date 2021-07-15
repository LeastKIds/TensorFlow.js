const tf = require('@tensorflow/tfjs-node');

const typed32 = new Int32Array(2);
typed32[0] = 123;
console.log(tf.tensor(typed32).print());

const buffer = new ArrayBuffer(12);
const buffer32 = new Float32Array(buffer);
buffer32[0] = 4.5, buffer32[2] = 9.1;
console.log(tf.tensor(buffer32).print());

const buffer8 = new Uint8Array(2);
buffer8[0] = 1;
console.log(tf.tensor(buffer8).print());

console.log(tf.tensor([[11,12], [21,22]]).print(true));

console.log(tf.scalar(123).print(true));

console.log(tf.scalar(true).print(true));

console.log(tf.tensor1d([1,2,3]).print(true));

console.log(tf.tensor1d([1,-3,0],'bool').print(true));

console.log(tf.tensor2d([1,2,3,4,5,6],[3,2]).print(true));