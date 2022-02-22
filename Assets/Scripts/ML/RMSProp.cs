using System;
using UnityEngine;

namespace ML
{
    public class RMSProp:SGD
    {
        private Tensor[] Sw;
        private Tensor[] Sb;
        private double beta;
        private bool inited;
        private double NOISE = 1E-8;
        private int counter = 1;
        public RMSProp(int batchSize,double beta = 0.9,double learningRate = 1E-2) : base(batchSize,learningRate)
        {
            this.beta = beta;
        }
        
        public override (Tensor[],Tensor[]) backwards(Tensor[] features, Tensor[] labels)
        {

            (Tensor[] finalWeightGrad, Tensor[] finalBiasGrad) = base.backwards(features, labels);
            // if the momentum tensors are not initialised, we need to init them with the shape of the gradients
            // and 0 values
            if (inited == false)
            {
                Sw = new Tensor[finalWeightGrad.Length];
                for (int i = 0; i < finalWeightGrad.Length; i++)
                {
                    Sw[i] = new Tensor(finalWeightGrad[i].Shape);
                }
                Sb = new Tensor[finalBiasGrad.Length];
                for (int i = 0; i < finalBiasGrad.Length; i++)
                {
                    Sb[i] = new Tensor(finalBiasGrad[i].Shape);
                }

                inited = true;
            }
            
            // updating Sw,Sb
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                Sw[i] = beta*Sw[i]+(1-beta)*finalWeightGrad[i].Pow(2);
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                Sb[i] = beta*Sb[i]+(1-beta)*finalBiasGrad[i].Pow(2);
            }

            double biasCurrection = 1 / (1 - Math.Pow(beta, counter));
            // doing the dividing
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                finalWeightGrad[i] /= (Sw[i]*biasCurrection).Pow(0.5)+NOISE;
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                finalBiasGrad[i] /= (Sb[i]*biasCurrection).Pow(0.5)+NOISE;
            }
            counter++;
            // returning the final gradiants
            return (finalWeightGrad, finalBiasGrad);
        }

        
    }
}