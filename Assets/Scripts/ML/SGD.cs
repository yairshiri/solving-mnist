using System;

namespace ML
{
    public class SGD:Optimizer
    {
        
        #region Fields

        private int batchSize;
        
        #endregion
        
        #region Constructor

        public SGD(int batchSize)
        {
            this.batchSize = batchSize;
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
                (weightGrads[i], biasGrads[i]) = GetGrad(dataBatch[i],labelsBatch[i]);
            }
            // computing the final grad:
            Tensor[] finalWeightGrad = new  Matrix[weightGrads[0].Length];
            Tensor[] finalBiasGrad = new  Vector[biasGrads[0].Length];
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                // adding the weight and bias gradiants
                finalWeightGrad[i] = new Matrix(weightGrads[0][i],true);
                finalBiasGrad[i] = new Vector(biasGrads[0][i],true);
                for (int j = 0; j < batchSize; j++)
                {
                    finalWeightGrad[i] += (Matrix)weightGrads[j][i];
                    finalBiasGrad[i] += (Vector)biasGrads[j][i];
                }
            }
            // returning the final gradiants
            return (finalWeightGrad, finalBiasGrad);
        }

    }
}