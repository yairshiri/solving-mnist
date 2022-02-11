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
        public Scalar(float data, string name) : base(0, name)
        {
            Data = data;
        }
        // data constructor
        public Scalar(float data) : base(0)
        {
            Data = data;
        }

        public Scalar(Tensor data) : base(0)
        {
            // make sure that data is a scalar
            Assert.AreEqual(data.Dimension, 0);
            // copy it
            Data = data.Data;
        }
        // empty constructor
        public Scalar() : base(0)
        {
            
        }
        // copy constructor 
        public Scalar(Scalar a) : base(0, a.Name)
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

        #endregion
    }
}