const tf = require('@tensorflow/tfjs-node');

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

test();

