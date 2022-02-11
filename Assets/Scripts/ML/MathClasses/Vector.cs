using System;
using System.Runtime.CompilerServices;
using UnityEngine;
using UnityEngine.Assertions;

namespace ML
{
    public class Vector:Tensor
    {
        #region variables
        // Vector data is a list of floats
        private new Scalar[] _data;
        public new Scalar[] Data
        {
            get => _data;
            set
            {   
                _data = new Scalar[value.Length];
                // creating a new scalar array, setting the value and changing the length
                for (int i = 0; i < value.Length; i++)
                {
                    _data[i] = new Scalar(value[i]);
                }        
                Length = value.Length;
            }
        }
        #endregion

        #region constructors

        // constructor without name and data as float to be applied to all elements
        public Vector(int size, float data,string name="") : base(new []{size},name)
        {
            _data = new Scalar[size];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(data);
            }
            Length = Data.Length;
        }
        
        // constructor with size and name
        public Vector(int size, string name="") : base(new []{size}, name)
        {
            Length = size;
            _data = new Scalar[size];
            for (int i = 0; i < Length; i++)
            {
                _data[i] = new Scalar(0);
            }
        }
        
        
        // copy constructor
        public Vector(Vector a) : base(a.Shape,a.Name)
        {
            this.Data = a.Data;
        }
        //copy from tensor constructor
        public Vector(Tensor a,bool copySize = false) : base(a.Shape,a.Name)
        {
            
            // make sure that the tensor a is actually a vector:
            Assert.AreEqual(a.Dimension, 1);
            // copying the data from a to this
            if (copySize == false)
            {
                Data = ((Vector)a).Data;
            }
            else
            {
                Length = a.Length;
                _data = new Scalar[Length];
                for (int i = 0; i < Length; i++)
                {
                    _data[i] = new Scalar(0);
                }

            }
        }
        
        #endregion

        #region operators
        // vector multiplication, dot product.
        public static Scalar operator *(Vector a, Vector b)
        {
            // a and b have to be vectors of equal lengths
            Assert.AreEqual(a.Length,b.Length);
            
            // calculating the magnitudes of a and b
            float sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }
            
            // returning the value
            return new Scalar(sum);
        }
        // vector scalar element-wise multiplication.
        public static Vector operator *(Vector a, Scalar b)
        {
            Vector ret = new Vector(a);
            for (int i = 0; i < a.Length; i++)
            {
                ret[i] = b * ret[i] ;
            }

            return ret;
        }
        public static Vector operator *(Vector a, float b)
        {
            Vector ret = new Vector(a);
            for (int i = 0; i < a.Length; i++)
            {
                ret[i] = b * ret[i] ;
            }

            return ret;
        }

        public static Vector operator +(Vector a, Tensor b)
        {
            //make sure b is a vector
            Assert.AreEqual(a.Dimension,b.Dimension);
            Vector ret = new Vector(b);
            for (int i = 0; i < a.Length; i++)
            {
                ret[i] += a[i];
            }

            return ret;

        }

        public static Vector operator +(Tensor a, Vector b)
        {
            return b + a;
        }
        
        // square brackets operator
        public override float this[int i]
        {
            get => _data[i].Data;
            set => Data[i].Data = value;
        }

        #endregion

        #region methods

        public float sum()
        {
            float ret = 0;
            for (int i = 0; i < Length; i++)
            {
                ret += Data[i];
            }

            return ret;
        }


        public override string ToString()
        {
            // getting the name, if present and adding the first |
            string ret = base.ToString() + "|";
            // adding the values of the vector
            for (int i = 0; i < Length; i++)
            {
                // adding a | between every item so it'll look like this: |this[0]|this[1]|....|this[n-2]|this[n-1]|
                ret +=  this[i] + "|";
            }
            // adding a new line
            ret += "\n";
            return ret;
        }

        public override Tensor ElementWiseFunction(Func<Tensor, Tensor> func)
        {
            // creating a copy of this
            Vector ret = new Vector(this);
            // applying the function
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = func(ret.Data[i]).Data;
            }
            return ret;
        }

        public override Tensor ElementWiseMultiply(Tensor a)
        {
            // making sure a is a vector:
            Assert.AreEqual(a.Dimension,1);
            // making sure this vector and a have the same size
            Assert.AreEqual(a.Length, Length);
            // creating the return vector with the length of a (and of this vector)
            Vector ret = new Vector(a.Length);
            //copying the values
            for (int i = 0; i < ret.Length; i++)
            {
                ret[i] = this[i] * a[i];
            }
            return ret;
        }

        public override Tensor Clone()
        {
            return new Vector(this);
        }

        public override Tensor Transpose()
        {
            return this.Clone();
        }

        #endregion
        
    }
}