console.log('Hello TensorWorld')
const getData=async()=>{
    const carDataReq=await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const carsData=await carDataReq.json()
    const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      })).filter((car)=>{
          return car.mpg!=null&&car.horsepower!=null;
      })
      return cleaned
   
}
const run=async()=>{
    const data=await getData();
    const values=data.map(d=>({
         x:d.horsepower,
         y:d.mpg
    }))
    tfvis.render.scatterplot(
        {name:'Horsepower v MPG'},
        {values},
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height:300
        }
    )
    const model=createModel()
    tfvis.show.modelSummary({name:'Model Summary'},model)
    const tensorData=convertToTensor(data);
    const {inputs,labels}=tensorData;
    await trainModel(model,inputs,labels)
    testModel(model,data,tensorData)
}
document.addEventListener('DOMContentLoaded',run)
const createModel=()=>{
    const model=tf.sequential();
    model.add(tf.layers.dense({inputShape:[1],units:1,useBias:true}))
    model.add(tf.layers.dense({units:1,useBias:true}))
    return model;
}
const convertToTensor=(data)=>{
    return tf.tidy(()=>{
        tf.util.shuffle(data)
        const inputs =data.map(d=>d.horsepower)
        const labels=data.map(d=>d.mpg)

        const inputTensor=tf.tensor2d(inputs,[inputs.length,1])
        const labelTensor=tf.tensor2d(labels,[labels.length,1])

        const inputMax=inputTensor.max()
        const inputMin=inputTensor.min()
        const labelMax=labelTensor.max()
        const labelMin=labelTensor.min()

        const normalizedInputs=inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels=labelTensor.sub(labelMin).div(inputMin.sub(labelMin))

        return{
        inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    }

 
    })
}

async function trainModel(model, inputs, labels) {
    // Prepare the model for training.  
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });
    
    const batchSize = 32;
    const epochs = 50;
    
    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'], 
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }

  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;  
    
    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling 
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {
      
      const xs = tf.linspace(0, 1, 100);      
      const preds = model.predict(xs.reshape([100, 1]));      
      
      const unNormXs = xs
        .mul(inputMax.sub(inputMin))
        .add(inputMin);
      
      const unNormPreds = preds
        .mul(labelMax.sub(labelMin))
        .add(labelMin);
      
      // Un-normalize the data
      return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });
    
   
    const predictedPoints = Array.from(xs).map((val, i) => {
      return {x: val, y: preds[i]}
    });
    
    const originalPoints = inputData.map(d => ({
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