using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using ML;
using UnityEngine;
using UnityEngine.Assertions;

public   class Tensor
{
    #region variables

    // every tensor needs dimensionality 
    private int[] _shape;

    // only the final tensor has a value
    private double _value = 0;

    // name, doesn't need to have it
    private string _name = "";

    // need to add explanation here
    protected Tensor[] _data ;

    #endregion

    #region GettersAndSetters
    
    public int Height => Shape[0];

    public int Width => Shape[1];

    public string Name
    {
        get => _name;
        set => _name = value;
    }

    public int[] Shape
    {
        get => _shape;
        set => _shape = value;
    }

    public int Dimension
    {
        get => _shape.Length;
    }

    public  Tensor[] Data
    {
        get => _data;
        set => _data = value;
    }

    public double Value
    {
        get
        {
            if (Data == null)
                return _value;
            return this[0].Value;
        }
        set
        {
            if (Data == null||Data[0] == null)
                _value = value;
            else
            {
                this[0].Value = value;
            }
        }
    }

    public int Length
    {
        get => Shape[0];
    }

    public bool IsScalar => (Data == null&&Length==1&&Dimension==1);

    #endregion

    #region constructors

    // constructor with name
    public Tensor(int[] shape, double defaultValue=0.0,string name="")
    {
        Shape = shape;
        Name = name;
        Data = new Tensor[Shape[0]];
        // if we are a scalar
        if (Dimension==1)
        {
            for (int i = 0; i < Shape[0]; i++)
            {
                Data[i] = new Tensor(defaultValue,Name+"["+i+"]");
            }
            return;
        }
        for (int i = 0; i < shape[0]; i++)
        {
            Data[i] = new Tensor(Shape.Skip(1).ToArray(),defaultValue ,name+"["+i+"]");
        }
    }


    // copying constructor
    public Tensor(Tensor a)
    {
        Name = a.Name;
        Shape = a.Shape;
        if (a.IsScalar)
        {
            Value = a.Value;
            return;
        }
            Data = new Tensor[Shape[0]];
        if (Dimension==1)
        {
            for (int i = 0; i < Shape[0]; i++)
            {
                Data[i] = new Tensor(a[i].Value,Name+"["+i+"]");
            }
            return;
        }
        for (int i = 0; i < Shape[0]; i++)
        {
            Data[i] = new Tensor(a[i]);
        }
    }
    
    public Tensor(int size, double defaultValue= 0.0,string name="")
    {
        Shape = new []{size};
        Name = name;
        Data = new Tensor[Shape[0]];
        for (int i = 0; i < size; i++)
        {
            Data[i] = new Tensor(defaultValue,  name+"["+i+"]");
        }
        
    }
    public Tensor(double value, string name = "")
    {
        Shape = new[] {1};
        Value = value;
        Name = name;
    }

    #endregion

    #region operators

    // trying to make a general + operator for tensors

    public static Tensor operator +(Tensor a, Tensor b)
    {
        // if we add a scalar to a scalar
        if (a.IsScalar && b.IsScalar)
            return new Tensor(a.Value + b.Value);
        Tensor ret;
        // we add scalar to non scalar
        if (a.IsScalar || b.IsScalar)
        {
            // we find who is the scalar
            Tensor scalar = a;
            Tensor nonScalar = b;
            if (b.IsScalar)
            {
                scalar = b;
                nonScalar = a;
            }

            ret = new Tensor(nonScalar);
            // add the scalar to the non scalar
            for (int i = 0; i < a.Length; i++)
            {
                ret[i] += scalar;
            }

            return ret;
        }
        // add two same shape tensors
        Assert.IsTrue(Enumerable.SequenceEqual(a.Shape,b.Shape));
        ret = new Tensor(a);
        for (int i = 0; i < ret.Shape[0]; i++)
        {
            ret[i] += b[i];
        }
        return ret;
    }

    public static Tensor operator *(Tensor a, Tensor b)
    {
        //if a,b are scalars just do scalar multiplication
        if (a.IsScalar && b.IsScalar)
            return new Tensor(a.Value * b.Value);
        if (Math.Abs(a.Dimension - b.Dimension) > 1)
            throw new Exception("The dimentions of a and b need to be equal or 1 less of each other!");
        Tensor max;
        Tensor min;
        if (a.Dimension > b.Dimension)
        {
            max = a;
            min = b;
        }
        else
        {
            max = b;
            min = a;
        }
        Tensor ret = new Tensor(max);
        // i is the max index, j is min.
        for (int i = 0; i < min.Length; i++)
        {
            ret[i] =  min[i];
            
        }
        return ret;
    }

    // square brackets operator
    public virtual Tensor this[int i]
    {
        get => Data[i];
        set => Data[i] = value;
    }

    #endregion

    #region methods

    public new virtual  string ToString()
    {
        // all tensors may have names, so instead of implementing this (the name adding) for every claas we implement here
        // and use in other classes (with base.ToString())
        string ret = "";
        if (Name != "")
            ret += Name+": ";
        return ret;
    }

    public virtual Tensor ElementWiseFunction(Func<Tensor, Tensor> func)
    {
        //stopping clause
        if (this.IsScalar)
        {
            return func(this);
        }
        // creating a copy of this
        Tensor ret = new Tensor(this);
        // applying the function recursively
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] = ret[i].ElementWiseFunction(func);
        }
        return ret;
    }


    public virtual Tensor ElementWiseMultiply(Tensor a)
    {
        //stopping clause
        if (this.IsScalar)
        {
            return this*a;
        }
        // creating a copy of this
        Tensor ret = new Tensor(this);
        // applying the function recursively
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] = ret[i].ElementWiseMultiply(a[i]);
        }
        return ret;
    }

    public virtual Tensor Clone()
    {
        return new Tensor(this);
    }


    public virtual bool Multiplyable(Tensor a)
    {
        if(a.IsScalar || this.IsScalar || a.Dimension==Dimension)
            return true;
        return false;
    }


    public static Tensor MatrixMult(Tensor a, Tensor b)
    {
        if (a.Dimension<2  && b.Dimension < 2)
            return a * b;
        Assert.AreEqual(a.Shape[1],b.Shape[0]);
        Tensor ret = new Tensor(new[] { a.Shape[0], b.Shape[1] },name:a.Name + " * " + b.Name);
        for (int i = 0; i < ret.Shape[0]; i++)
        {
            for (int j = 0; j < ret.Shape[1]; j++)
            {
                for (int k = 0; k < a.Shape[1]; k++)
                {
                    ret[i][j] = new Tensor(ret[i][j].Value+a[i][k].Value*b[k][j].Value);
                }
            }
        }

        return ret;
    }

    #endregion
}