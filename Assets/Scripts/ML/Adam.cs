using System;
using UnityEngine;

namespace ML
{
    public class Adam:SGD
    {
        private double beta1 = 0.9;
        private double beta2 = 0.999;
        private Tensor[] Vw;
        private Tensor[] Vb;
        private Tensor[] Sw;
        private Tensor[] Sb;
        private double NOISE = 1E-8;
        private bool inited = false;
        private int counter = 1;
        
        public Adam(int batchSize,double beta1 = 0.9,double beta2 = 0.999 ,double learningRate = 1E-02) : base(batchSize, learningRate)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
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
            
            // updating Sw,Sb
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                Sw[i] = beta2*Sw[i]+(1-beta2)*finalWeightGrad[i].Pow(2);
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                Sb[i] = beta2*Sb[i]+(1-beta2)*finalBiasGrad[i].Pow(2);
            }
            // updating Vw,Vb
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                Vw[i] = beta1*Vw[i]+(1-beta1)*finalWeightGrad[i];
            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                Vb[i] = beta1*Vb[i]+(1-beta1)*finalBiasGrad[i];
            }

            double biasCurrectionFactor = Math.Sqrt(1 - Math.Pow(beta2, counter)) / (1 - Math.Pow(beta1, counter));
            // doing the dividing
            for (int i = 0; i < finalWeightGrad.Length; i++)
            {
                finalWeightGrad[i] = Vw[i]/(Sw[i].Pow(0.5)+NOISE);
                finalWeightGrad[i] *= biasCurrectionFactor;

            }
            for (int i = 0; i < finalBiasGrad.Length; i++)
            {
                finalBiasGrad[i] =Vb[i]/( Sb[i].Pow(0.5)+NOISE);
                finalBiasGrad[i] *= biasCurrectionFactor;
            }
            counter++;
            // returning the final gradiants
            return (finalWeightGrad, finalBiasGrad);
        }

        
    }
}