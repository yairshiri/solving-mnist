using System;
using System.Collections;
using System.Collections.Generic;
using ML;
using UnityEngine;
using Network = ML.Network;
using Random = System.Random;

public class MLMain : MonoBehaviour
{
    private static int sampleSize = 100000;
    private static int batchSize = 100;

    private float noise = 0.00001f;
    
    
    // counts loops through update
    private int counter = 0;
    // log every LOG_INTERVAL loops through update
    private int LOG_INTERVAL = 20;
    
    private float lr = 0.00001f;
    private Tensor[] features = new Tensor[sampleSize];
    private Tensor[] labels = new Tensor[sampleSize];
    private Random rand = new Random();
    private float x ;

    // creating the softmax activation func 
    //private ML.Function<Tensor,Tensor> softmax = new Function<Tensor,Tensor>( softmaxFunc,softmaxDeriv,"softmax");
    // creating the mse loss
    private ML.Loss mse = new Loss( mseFunc,mseDeriv,"mse");
    // creating the Categorical crossEntropy loss
    private ML.Loss CE = new Loss( CEFunc,CEDeriv,"categorical crossentropy");
    // creating the Binary Catergorical crossEntropy loss
    private ML.Loss BCE = new Loss( BCCEFunc,CEDeriv,"binary categorical crossentropy");
    private Network net;
    
    
    // Start is called before the first frame update
    void Start()
    {
        x = (float)rand.NextDouble() * 10;
        // generating the data
        Debug.Log("Generating data...");
        
        for (int i = 0; i < sampleSize; i++)
        {
            features[i] = new Tensor(1, x,"Data "+i);
            labels[i] = new Tensor(1,x*2+2,  "Label " + i);
            /*labels[i][0].Value = 0;
            if (x > 5)
                labels[i][0].Value = 1;
            labels[i][1].Value = 1 - labels[i][0].Value;*/
            x = (float)rand.NextDouble() * 10;

        }

        Debug.Log("Done!");
        Layer[] layers=
        {
            new DenseLayer(3,"softrelu","d1"),
            new DenseLayer(4,"softrelu","d2"),
            new DenseLayer(labels[0].Length,"linear","d3"),
        };
        Optimizer optimizer = new SGD(batchSize);
        net = new Network(layers,lr,1,mse,optimizer);

    }
    #region activations
    
    private static Tensor mseFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0].Value += Math.Pow(input.pred[i].Value - input.label[i].Value,2);
        }

        return ret;
    }

    private static Tensor mseDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = 2*(input.pred[i].Value - input.label[i].Value);
        }
        return ret;

    }

    private static Tensor CEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[0].Value += -input.label[i].Value * (float)Math.Log(input.pred[i].Value);
        }
        return ret;
    }

    private static Tensor CEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        
        Tensor ret = new Tensor(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = -(input.label[i].Value / input.pred[i].Value);
        }
        return ret;

    }
    private static Tensor BCCEFunc((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(1);
        //L = -t1*log(s1) - (1-t1)*log(1-s1)
        ret[0].Value = -input.label[0].Value * (float)Math.Log(input.pred[0].Value) - input.label[1].Value * (float)Math.Log(input.pred[1].Value);
        return ret;
    }

    private static Tensor BCCEDeriv((Tensor pred, Tensor label) input)
    {
        // x,y need to have the same length
        Debug.Assert(input.pred.Length==input.label.Length);
        Tensor ret = new Tensor(input.pred.Length);
        for (int i = 0; i < input.pred.Length; i++)
        {
            ret[i].Value = -(input.label[i].Value / input.pred[i].Value) + (1 - input.label[i].Value)/(1-input.pred[i].Value);
        }
        return ret;

    }
    
    #endregion
    
    // a method that does the logging. yeah it's a great comment.
    void log()
    {
        Debug.Log(net.ToString());
    }
    
    
    // Update is called once per frame
    void Update()
    {
        x = (float)rand.NextDouble() * 10;
        net.backwards(features,labels);
        counter++;
        if (counter % LOG_INTERVAL == 0)
        {
            log();
        }
        
    }
}
