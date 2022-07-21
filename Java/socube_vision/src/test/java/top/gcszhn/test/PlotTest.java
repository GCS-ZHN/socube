package top.gcszhn.test;
import top.gcszhn.jvision.JvisionException;
import top.gcszhn.jvision.chart.RingDiagram;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.junit.Test;

import com.alibaba.fastjson.JSONObject;

public class PlotTest {
    @Test
    public void test() throws IOException, JvisionException {
        String configStr;
        try (FileInputStream input = new FileInputStream("data/config.json")) {
            configStr = new String(input.readAllBytes());
        }
        int width = 600, height = 600;
        String fontFamily = "Calibri";
        int df = 15;
        float[] radioRange = new float[]{50, 200};
        int startAngle = 90, arcAngle = -(360 - 360 / df);
        float gapRatio = 0.2f;
        int step = df - 1;
        boolean withTitle = false;
        boolean balance = true;
        JSONObject config = JSONObject.parseObject(configStr);
        String format = "pdf";
        for (String type: config.getJSONObject("precision").keySet()) {
            try {
                JSONObject subConfig = config.getJSONObject("precision").getJSONObject(type);
                float[] valueRange = new float[]{
                    subConfig.getFloatValue("min"), 
                    subConfig.getFloatValue("max")
                };
                RingDiagram ringDiagram = new RingDiagram(
                    withTitle ? "Precision": null, 
                    width, 
                    height, 
                    gapRatio, 
                    radioRange, 
                    valueRange, 
                    startAngle, 
                    arcAngle, 
                    (valueRange[1]-valueRange[0]) / step,
                    balance);
                ringDiagram.setFontFamily(fontFamily);
                ringDiagram.loadData(String.format("data/data-precision-%s.csv", type));
                ringDiagram.draw(String.format("data/data-precision-%s.%s", type, format));
            } catch (JvisionException e) {
                if (e.getCause() instanceof FileNotFoundException) {
                    System.out.println(e.getCause().getMessage());
                } else {
                    throw e;
                }
            }
        }

        for (String type: config.getJSONObject("recall").keySet()) {
            try{
                JSONObject subConfig = config.getJSONObject("recall").getJSONObject(type);
                float[] valueRange = new float[]{
                    subConfig.getFloatValue("min"), 
                    subConfig.getFloatValue("max")
                };
                RingDiagram ringDiagram = new RingDiagram(
                    withTitle ? "Recall": null, 
                    width, 
                    height, 
                    gapRatio, 
                    radioRange, 
                    valueRange, 
                    startAngle, 
                    arcAngle, 
                    (valueRange[1]-valueRange[0]) / step,
                    balance);
                ringDiagram.setFontFamily(fontFamily);
                ringDiagram.loadData(String.format("data/data-recall-%s.csv", type));
                ringDiagram.draw(String.format("data/data-recall-%s.%s", type, format));
            } catch (JvisionException e) {
                if (e.getCause() instanceof FileNotFoundException) {
                    System.out.println(e.getCause().getMessage());
                } else {
                    throw e;
                }
            }
        }

        for (String type: config.getJSONObject("TNR").keySet()) {
            try {
                JSONObject subConfig = config.getJSONObject("TNR").getJSONObject(type);
                float[] valueRange = new float[]{
                    subConfig.getFloatValue("min"), 
                    subConfig.getFloatValue("max")
                };
                RingDiagram ringDiagram = new RingDiagram(
                    withTitle ? "TNR": null, 
                    width, 
                    height, 
                    gapRatio, 
                    radioRange, 
                    valueRange, 
                    startAngle, 
                    arcAngle, 
                    (valueRange[1]-valueRange[0]) / step,
                    balance);
                ringDiagram.setFontFamily(fontFamily);
                ringDiagram.loadData(String.format("data/data-TNR-%s.csv", type));
                ringDiagram.draw(String.format("data/data-TNR-%s.%s", type, format));
            } catch (JvisionException e) {
                if (e.getCause() instanceof FileNotFoundException) {
                    System.out.println(e.getCause().getMessage());
                } else {
                    throw e;
                }
            }
        }
    }
}
