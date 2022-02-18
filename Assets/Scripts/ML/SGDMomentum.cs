using System;

namespace ML
{
    public class SGDMomentum:Optimizer
    {
        
        #region Fields

        private int batchSize;
        private Tensor[] Vw;
        private Tensor[] Vb;
        private double beta;
        private bool inited = false;
        #endregion
        
        #region Constructor

        public SGDMomentum(int batchSize,double beta=0.9)
        {
            this.batchSize = batchSize;
            this.beta = beta;
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
            Tensor[] finalWeightGrad = new  Tensor[weightGrads[0].Length];
            Tensor[] finalBiasGrad = new  Tensor[biasGrads[0].Length];
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                // adding the weight and bias gradiants
                finalWeightGrad[i] = new Tensor(weightGrads[0][i].Shape);
                finalBiasGrad[i] = new Tensor(biasGrads[0][i].Shape);
                for (int j = 0; j < batchSize; j++)
                {
                    finalWeightGrad[i].Value += weightGrads[j][i].Value;
                    finalBiasGrad[i].Value += biasGrads[j][i].Value;
                }
            }
            // if the momentum tensors are not initialised, we need to init them with the shape of the gradients
            // and 0 values
            if (inited == false)
            {
                Vw = new Tensor[finalWeightGrad.Length];
                for (int i = 0; i < finalWeightGrad.Length; i++)
                {
                    Vw[i] = new Tensor(finalWeightGrad[i].Shape);
                }
                Vb = new Tensor[finalBiasGrad.Length];
                for (int i = 0; i < finalBiasGrad.Length; i++)
                {
                    Vb[i] = new Tensor(finalBiasGrad[i].Shape);
                }

                inited = true;
            }
            
            // updating vw,vb
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                Vw[i] = new Tensor(beta*Vw[i]+(1-beta)*finalWeightGrad[i]);
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                Vb[i] = new Tensor(beta*Vb[i]+(1-beta)*finalBiasGrad[i]);
            }

            
            // returning the final gradiants
            return (Vw, Vb);
        }

    }
}