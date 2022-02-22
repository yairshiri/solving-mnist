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
    private double _value;

    // name, doesn't need to have it
    private string _name = "";

    // need to add explanation here
    protected Tensor[] _data ;

    #endregion

    #region GettersAndSetters
    
    public int Height => Shape[0];

    public int Width
    {
        get
        {
            if (Dimension < 2)
                return 1;
            return Shape[1];
        }
    }

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

    public int NumOfElements
    {
        get
        {
            int ret = 1;
            for (int i = 0; i < Dimension; i++)
            {
                ret *= Shape[i];
            }
            return ret;
        }
    }

    public bool IsScalar => (Dimension == 0);

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
        Shape = Array.Empty<int>();
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
        // add two same shape tensors or one is a scalar (if both then we don't even get here)
        Assert.IsTrue(Enumerable.SequenceEqual(a.Shape,b.Shape)||b.IsScalar||a.IsScalar);
        // find the one with the bigger dimension. 
        Tensor max = a;
        Tensor min = b;
        if (b.Dimension > a.Dimension)
        {
            max = b;
            min = a;
        }
        ret = new Tensor(max);
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] += min[i];
        }
        return ret;
    }

    public static Tensor operator *(Tensor a, Tensor b)
    {
        //if a,b are scalars , just return what we need to 
        if (a.IsScalar && b.IsScalar)
            return new Tensor(a.Value * b.Value);
        Assert.IsTrue(Enumerable.SequenceEqual(a.Shape,b.Shape)||b.IsScalar||a.IsScalar);
        Tensor max = a;
        Tensor min = b;
        if (b.Dimension > a.Dimension)
        {
            max = b;
            min = a;
        }
        Tensor ret = new Tensor(max);
        // multiplying the values
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] *=  min[i];
        }
        return ret;
    }

    public static Tensor operator /(Tensor a, Tensor b)
    {
        //if a,b are scalars , just return what we need to 
        if (a.IsScalar && b.IsScalar)
            return new Tensor(a.Value / b.Value);
        Assert.IsTrue(Enumerable.SequenceEqual(a.Shape,b.Shape)||b.IsScalar);
        Tensor ret = new Tensor(a);
        for (int i = 0; i < a.Length; i++)
        {
            ret[i] /=  b[i];
        }
        return ret;
    }
    public static Tensor operator *(Tensor a, double b)
    {
        if (a.IsScalar)
            return new Tensor(a.Value * b);
        Tensor ret = new Tensor(a);
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] *= b;
        }
        return ret;
    }
    public static Tensor operator *(double a, Tensor b)
    {
        return b * a;
    }
    public static Tensor operator /(Tensor a, double b)
    {
        if (a.IsScalar)
            return new Tensor(a.Value / b);
        Tensor ret = new Tensor(a);
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] /= b;
        }
        return ret;
    }
    public static Tensor operator /(double a, Tensor b)
    {
        return b / a;
    }
    public static Tensor operator +(Tensor a, double b)
    {
        if (a.IsScalar)
            return new Tensor(a.Value + b);
        Tensor ret = new Tensor(a);
        for (int i = 0; i < ret.Length; i++)
        {
            ret[i] += b;
        }
        return ret;
    }
    public static Tensor operator +(double a, Tensor b)
    {
        return b + a;
    }

    // square brackets operator
    public virtual Tensor this[int i]
    {
        get
        {
            if (IsScalar)
                return this;
            return Data[i];
        }
        set
        {
            if (IsScalar)
            {
                Value = value.Value;
                Name = value.Name;
            }
            Data[i] = value;
        }
    }

    #endregion

    #region methods

    public  virtual  string ToString(bool Addname=true)
    {
        string ret = "";
        if (Name != ""&&Addname)
            ret += Name+":\n";
        if (IsScalar)
            return ret + Value;
        if (Dimension <= 1 && !IsScalar)
        {
            ret += "[";
            for (int i = 0; i < this.Length; i++)
            {
                ret+=this[i].Value+",";
            }

            ret= ret.Remove(ret.Length-1,1);
            ret += "]\n";
        }
        else
        {
            for (int i = 0; i < this.Length; i++)
            {
                ret+=this[i].ToString(false);
            }
        }
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
    
    public virtual Tensor Atleast2d(int axis = 0)
    {
        // if we are atleast 2d, just return this.
        if (this.Dimension >= 2)
            return this;
        Tensor ret;
        if (axis == 0)
        {
            ret = new Tensor(new[]{1,this.Length});
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j].Value = this[j].Value;
                }
            }
        }
        else
        {
            ret = new Tensor(new[]{this.Length,1});
            for (int i = 0; i < ret.Height; i++)
            {
                for (int j = 0; j < ret.Width; j++)
                {
                    ret[i][j].Value = this[i].Value;
                }
            }
        }
        
        return ret;
    }


    public static Tensor MatrixMult(Tensor a, Tensor b)
    {
        if (a.Dimension < 2 && b.Dimension < 2)
            return a * b;
        Tensor ret;
        Assert.AreEqual(a.Width,b.Height);
        // if b is a vector (its dimension is 1), then we want to return a vector. else, obviously, a matrix.
        if (b.Dimension == 1)
        {
            ret = new Tensor( a.Height,name:a.Name + " * " + b.Name);
        }
        else
        {
            ret= new Tensor(new[] { a.Height, b.Width },name:a.Name + " * " + b.Name);
        }
        for (int i = 0; i < ret.Height; i++)
        {
            for (int j = 0; j < ret.Width; j++)
            {
                for (int k = 0; k < a.Width; k++)
                {
                    ret[i][j].Value += a[i][k].Value*b[k][j].Value;
                }
            }
        }
        return ret;
    }

    public virtual Tensor Transpose()
    {
        // we need to make sure we return a matrix.
        Tensor temp = this.Atleast2d();
        // create a new rotated matrix:
        Tensor ret = new Tensor(new []{temp.Shape[1],temp.Shape[0]},name:this.Name+" transposed");
        // copy the values
        for (int i = 0; i < temp.Height; i++)
        {
            for (int j = 0; j < temp.Width; j++)
            {
                ret[j][i].Value = temp[i][j].Value;
            }
        }
        return ret;
    }


    public virtual Tensor Pow(double pow)
    {
        if (IsScalar)
            return new Tensor(Math.Pow(Value, pow));
        Tensor ret = Clone();
        for (int i = 0; i < Length; i++)
        {
            ret[i] = ret[i].Pow(pow);
        }
        return ret;
    }

    public virtual Tensor Concat(Tensor x)
    {
        Assert.IsTrue(Enumerable.SequenceEqual(x.Shape, this.Shape));
        Tensor ret = new Tensor(new[] { Shape[0] }.Concat(Shape.Skip(1)).ToArray());
        for (int i = 0; i < ret.Height/2; i++)
        {
            ret[i] = this[i].Clone();
            ret[ret.Height+i] = x[i].Clone();
        }
        return ret;
    }

    #endregion
}