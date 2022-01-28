using System;

namespace ML
{
    public class SoftMaxLayer:ActionLayer
    {
        #region Fields

        private static Function<Tensor, Tensor> softmax = new Function<Tensor, Tensor>(softmaxFunc, softmaxDeriv,"softmax");

        #endregion
        
        public SoftMaxLayer(int[] size, string name = "") : base(size,softmax,name)
        {
            
        }

        #region Methods

        private static Tensor softmaxFunc(Tensor x) 
        {
            // initializing a new vector with the length of x.
            Vector ret = new Vector(x.Length);
            float sum = 0;
            // finding the max value in x
            float max = 0;
            for (int i = 0; i < ret.Length; i++)
                if (x[i] > max)
                    max = x[i];
            // every element is e to the power of the elements devided by the sum of e to the poewr of all the elements.
            // implementation from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
            for (int i = 0; i < ret.Length; i++)
            {   
                // x = e^x
                ret[i] = (float)Math.Exp(x[i]-max);
                // adding to the sum
                sum += ret[i];
            }
            // deviding by the sum
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] /= sum;
            }
            return ret;
        }
        // softmax deriv from https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        private static Tensor softmaxDeriv(Tensor x)
        {
            Vector ret =(Vector) softmaxFunc(x);
            for (int i = 0; i < x.Length; i++)
            {
                ret[i] = ret[i] * (1 - ret[i]);
            }
            return ret;
        }

        #endregion
        
    }
}