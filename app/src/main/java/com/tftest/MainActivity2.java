package com.tftest;

import android.content.res.AssetManager;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;

/**
 * Created by petar on 28.6.17..
 */

public class MainActivity2 extends AppCompatActivity {

    private static final String MODEL_FILE = "file:///android_asset/saved_model.pbtxt";
//    private static final String MODEL_FILE = "file:///android_asset/saved_model.pbtxt";
    private static final String INPUT_NODE = "I";
    private static final String OUTPUT_NODE = "X";

    private static final int[] INPUT_SIZE = {1,4};

    private TensorFlowInferenceInterface inferenceInterface;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        inferenceInterface = new TensorFlowInferenceInterface();
        inferenceInterface.initializeTensorFlow(getAssets(), MODEL_FILE);

        final Button button = (Button) findViewById(R.id.button);

        button.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {

                final EditText editNum1 = (EditText) findViewById(R.id.editNum1);
                final EditText editNum2 = (EditText) findViewById(R.id.editNum2);
                final EditText editNum3 = (EditText) findViewById(R.id.editNum3);
                final EditText editNum4 = (EditText) findViewById(R.id.editNum4);

                float num1 = Float.parseFloat(editNum1.getText().toString());
                float num2 = Float.parseFloat(editNum2.getText().toString());
                float num3 = Float.parseFloat(editNum3.getText().toString());
                float num4 = Float.parseFloat(editNum4.getText().toString());

                float[] inputFloats = {num1, num2, num3, num4};
                final TextView textViewR = (TextView) findViewById(R.id.txtViewResult);

                /*SavedModelBundle bundle=SavedModelBundle.load("/java/workspace/APIJavaSampleCode/tfModels/dnn/ModelSave","serve");
                Session s = bundle.session();*/
//                Graph graph=new Graph();
                /*Session s= new Session(inferenceInterface.graph());

                FloatBuffer.wrap(inputFloats);
                float[][] data= new float[1][4];
                data[0]=inputFloats;
                Tensor inputTensor=Tensor.create(data);

                Tensor result = s.runner()
                        .feed(INPUT_NODE, inputTensor)
//                        .feed("saved_model", inputTensor)
                        //.fetch("tensorflow/serving/classify")
                        .fetch("dnn/multi_class_head/predictions/probabilities")
                        //.fetch("dnn/zero_fraction_3/Cast")
                        .run().get(0);


                float[][] m = new float[1][5];
                float[][] vector = result.copyTo(m);
                float maxVal = 0;
                int inc = 0;
                int predict = -1;
                for(float val : vector[0])
                {
                    textViewR.setText(val+"  ");
                    if(val > maxVal) {
                        predict = inc;
                        maxVal = val;
                    }
                    inc++;
                }
                System.out.println(predict);*/

                //"dnn/input_from_feature_columns/input_from_feature_columns/concat"

                /*if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.KITKAT) {
                    test();
                }*/
                inferenceInterface.fillNodeFloat(/*INPUT_NODE*/"nn/input_from_feature_columns/input_from_feature_columns/concat", INPUT_SIZE, inputFloats);
                inferenceInterface.runInference(new String[] {OUTPUT_NODE});

                float[] resu = {0};
                inferenceInterface.readNodeFloat(OUTPUT_NODE, resu);

                textViewR.setText(String.valueOf(resu[0]));
            }
        });
    }

    @RequiresApi(api = Build.VERSION_CODES.KITKAT)
    void test()
    {
        try (Graph graph = new Graph()) {

            AssetManager am = getAssets();
            InputStream is = am.open("saved_model.pb");
            byte[] bytes = convert(is);
                graph.importGraphDef(bytes);
                try (Session sess = new Session(graph)) {
                    try (Tensor x = Tensor.create(1.0f);
                         Tensor y = sess.runner().feed("x", x).fetch("y").run().get(0)) {
                        System.out.println(y.floatValue());
                    }
            }
            catch (Exception ex){}
} catch (IOException e) {
            e.printStackTrace();
        }
    }

    private byte[] convert(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();

        int nRead;
        byte[] data = new byte[16384];

        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }

        buffer.flush();

        return buffer.toByteArray();
    }
}