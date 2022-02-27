using System;

namespace ML
{
    public class MaxPooling:ActionLayer
    {
        private int size;
        private int stride;
        private Tensor deltaVals;
        public MaxPooling(int size ,int stride = 1,string name = "") : base(new int[]{}, name, false)
        {
            this.size = size;
            this.stride = stride;
            base.Init(new Function<Tensor, Tensor>(maxPoolingFunc, maxPoolingDeriv, "max pooling 2D"));
        }

        public override void Init(int[] shape)
        {
            base.Init(shape);
            outputShape = new[] { (shape[0]-size)/stride+1, (shape[1] - size)/stride+1,shape[2]};
        }



        public Tensor maxPoolingFunc(Tensor x)
        {
            Tensor ret = new Tensor(outputShape);
            // this will be returned in the backprop
            deltaVals = new Tensor(inputShape);
            double max = Double.NegativeInfinity;
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    for (int k = 0; k < ret.Shape[2]; k++)
                    {
                        for (int l = 0; l < size; l++)
                        {
                            for (int m = 0; m < size; m++)
                            {
                                if (x[i * stride + l][j * stride + m][k].Value >= max)
                                {   
                                    max = x[i * stride + l][j * stride + m][k].Value;
                                    ret[i][j][k].Value = max;
                                    // saving for backprop!
                                    deltaVals[i * stride + l][j * stride + m][k].Value = 1;
                                }
                            }
                        }
                        max = 0;
                    }
                }
            }
            return ret;
        }

        public Tensor maxPoolingDeriv(Tensor x)
        {
            return deltaVals;
        }

        public override Tensor bPass(Tensor loss)
        {
            Tensor ret = new Tensor(inputShape);
            for (int i = 0; i < outputShape[0]; i++)
            {
                for (int j = 0; j < outputShape[1]; j++)
                {
                    for (int k = 0; k < outputShape[2]; k++)
                    {
                        for (int l = 0; l < size; l++)
                        {
                            for (int m = 0; m < size; m++)
                            {
                                ret[i * stride + l][j * stride + m][k] +=
                                    deltaVals[i * stride + l][j * stride + m][k] * loss[i][j][k];
                            }
                        }
                    }
                }
            }
            return ret;
        }
    }
}