using System;
using System.Runtime.CompilerServices;
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
                // creating a new scalar array, setting the value and changing the length
                _data = new Scalar[value.Length];
                for (int i = 0; i < value.Length; i++)
                    _data[i] = new Scalar(value[i]);
                Length = value.Length;
            }
        }
        #endregion

        #region constructors

        // constructor with name and data
        public Vector(Scalar[]data,string name) : base(1, name)
        {
            this.Data = data;
            this.Length = this.Data.Length;
        }
        // constructor without name
        public Vector(Scalar[] data) : base(1)
        {
            this.Data = data;
            this.Length = this.Data.Length;
        }

        // constructor with name and data as  floats
        public Vector(float[]data,string name) : base(1, name)
        {
            this._data = new Scalar[data.Length];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(data[i]);
            }
            this.Length = this.Data.Length;
        }
        // constructor without name and data as floats
        public Vector(float[] data) : base(1)
        {
            this._data = new Scalar[data.Length];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(data[i]);
            }
            this.Length = this.Data.Length;
        }
        // constructor without name and data as float to be applied to all elements
        public Vector(int size, float data) : base(1)
        {
            this._data = new Scalar[size];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(data);
            }
            this.Length = this.Data.Length;
        }
        
        // constructor with size and name
        public Vector(int size, string name) : base(1, name)
        {
            this.Length = size;
            this._data = new Scalar[size];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(0);
            }
            this.Name = name;
        }
        
        // constructor only size
        public Vector(int size) : base(1)
        {
            Length = size;
            this._data = new Scalar[size];
            for (int i = 0; i < Length; i++)
            {
                _data[i] = new Scalar(0);
            }
        }
        
        // copy constructor
        public Vector(Vector a) : base(1)
        {
            this.Data = new Scalar[a.Length];
            for (int i = 0; i < Length; i++)
            {
                Data[i] = new Scalar(a[i]);
            }
            this.Name = a.Name;
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
        #endregion
        
    }
}