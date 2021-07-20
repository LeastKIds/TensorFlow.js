const tf = require('@tensorflow/tfjs-node');
const { reason, result } = require('./data');

async function test(){
    try {
        const currentDirectory = __dirname;
        const model =  await tf.loadLayersModel('file://' + currentDirectory +'/path/to/lemon/model.json');

        const A = model.predict(tf.tensor([20])).print();
        console.log(A);
    } catch (error) {
        console.error(error);
    }
}

// test();
// console.log(123);
// console.log(reason);
// console.log(result);
console.log(__dirname);


