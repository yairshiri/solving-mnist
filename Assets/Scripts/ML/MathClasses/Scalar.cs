using System;
using UnityEngine.Assertions;

namespace ML
{
    public class Scalar:Tensor
    {
        #region variables

        public new float Data
        {
            get => _data[0];
            set => _data[0]= value;
        }
        
        #endregion
        
        
        #region  constructors
        // data, name constructor
        public Scalar(float data, string name) : base(new []{1}, name)
        {
            Data = data;
        }
        // data constructor
        public Scalar(float data) : base(new []{1})
        {
            Data = data;
        }

        public Scalar(Tensor data) : base(new []{1})
        {
            // make sure that data is a scalar
            Assert.AreEqual(data.Dimension, 0);
            // copy it
            Data = data.Data;
        }
        // copy constructor 
        public Scalar(Scalar a) : base(new []{1}, a.Name)
        {
            Data = a.Data;
        }
        #endregion
        
        #region operators

        public static Scalar operator +(Scalar a,Scalar b)
        {
            return new Scalar(a.Data + b.Data);
        }
        public static float operator +(Scalar a,float b)
        {
            return a.Data + b;
        }
        public static float operator +(float a,Scalar b)
        {
            return b.Data + a;
        }  
        public static float operator -(Scalar a,float b)
        {
            return a.Data - b;
        }
        public static float operator -(float a,Scalar b)
        {
            return a-b.Data;
        }
        public static float operator *(Scalar a,float b)
        {
            return a.Data * b;
        }
        public static float operator *(Scalar a,Scalar b)
        {
            return a.Data * b.Data;
        }
        
        
        #endregion
        
        #region Methods

        public override Tensor ElementWiseFunction(Func<Tensor, Tensor> func)
        {
            return new Scalar(func(this));
        }

        public override Tensor ElementWiseMultiply(Tensor a)
        {
            // make sure they are both the same size
            Assert.AreEqual(a.Dimension,0);
            // create the return Scalar
            Scalar ret = new Scalar(this*a.Data);
            return ret;
        }

        public override Tensor Clone()
        {
            return new Scalar(this);
        }

        public override Tensor Transpose()
        {
            return this.Clone();
        }

        #endregion
    }
}