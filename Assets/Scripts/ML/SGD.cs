using System;
using System.Threading;
namespace ML
{
    public class SGD:Optimizer
    {
        
        #region Fields

        private int batchSize;
        
        #endregion
        
        #region Constructor

        public SGD(int batchSize,double learningRate = 0.00001)
        {
            this.batchSize = batchSize;
            this.learningRate = learningRate;
        }
        #endregion
        
        public override (Tensor[],Tensor[]) backwards(Tensor[] features, Tensor[] labels)
        {
            // put batch_size random elements from data and labels into mini batches.
            Tensor[] dataBatch = new Tensor[batchSize];
            Tensor[] labelsBatch = new Tensor[batchSize];
            //put the selected items in the batch arrays
            int index = (int)Math.Floor(rand.NextDouble() * labels.Length);
            for (int i = 0; i < batchSize; i++)
            {
                dataBatch[i] = features[index];
                labelsBatch[i] = labels[index];
                // getting another random index
                index = (int)Math.Floor(rand.NextDouble() * labels.Length);
            }
            Tensor[][] weightGrads = new Tensor[batchSize][];
            Tensor[][] biasGrads= new Tensor[batchSize][];
            // getting the gradiants
            for (int i = 0; i < batchSize; i++)
            {
                (weightGrads[i], biasGrads[i]) = GetGrad(dataBatch[i], labelsBatch[i]);
            }
            
            
            // computing the final grad:
            Tensor[] finalWeightGrad = new  Tensor[weightGrads[0].Length];
            Tensor[] finalBiasGrad = new  Tensor[biasGrads[0].Length];
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                // adding the weight and bias gradiants
                finalWeightGrad[i] = new Tensor(weightGrads[0][i],0);
                finalBiasGrad[i] = new Tensor(biasGrads[0][i],0);
                for (int j = 0; j < batchSize; j++)
                {
                    finalWeightGrad[i] += weightGrads[j][i];
                    finalBiasGrad[i] += biasGrads[j][i];
                }
            }
            // returning the final gradiants
            return (finalWeightGrad, finalBiasGrad);
        }
        
    }
}