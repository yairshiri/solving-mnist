using System;
using UnityEngine;

namespace ML
{
    public class SGDMomentum:SGD
    {
        
        #region Fields

        private Tensor[] Vw;
        private Tensor[] Vb;
        private double beta;
        private bool inited;
        #endregion
        
        #region Constructor

        public SGDMomentum(int batchSize,double beta=0.9,double learningRate=0.00001):base(batchSize,learningRate)
        {
            this.beta = beta;
        }
        #endregion
        
        public override (Tensor[],Tensor[]) backwards(Tensor[] features, Tensor[] labels)
        {

            (Tensor[] finalWeightGrad, Tensor[] finalBiasGrad) = base.backwards(features, labels);
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
                Vw[i] = beta*Vw[i]+(1-beta)*finalWeightGrad[i];
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                Vb[i] = beta*Vb[i]+(1-beta)*finalBiasGrad[i];
            }

            // returning the final gradiants
            return (Vw, Vb);
        }

    }
}