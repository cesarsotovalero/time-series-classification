package weka.core;

import java.io.Serializable;
import java.util.Enumeration;
import java.util.Vector;
import weka.core.neighboursearch.PerformanceStats;

/**
 * <!-- globalinfo-start --> Implementing DTW distance (dissimilarity) function
 * for time series.<br/> <br/> Attention: For efficiency reasons the
 * Sakoe-Chiba's band is used as global constraint<br/> <br/> For more
 * information, see:<br/> <br/> UCR Time Series Classification/Clustering
 * Homepage, available from URL = {http://www.cs.ucr.edu}
 *
 * <p/>
 * <!-- globalinfo-end -->
 *
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 *
 * <pre> -W
 * The warping window size of the Sakoe-Chiba band (used as global constraint)
 * </pre>
 *
 * <pre> -R &lt;col1,col2-col4,...&gt;
 *  Specifies list of columns to used in the calculation of the
 *  distance. 'first' and 'last' are valid indices.
 *  (default: first-last)</pre>
 *
 * <pre> -V
 *  Invert matching sense of column indices.</pre>
 *
 * <!-- options-end -->
 *
 * @author César Soto (csoto@uclv.edu.cu)
 *
 */
public class DTWDistance implements DistanceFunction, Serializable, Cloneable, OptionHandler {

    /**
     * The instances used internally.
     */
    private Instances m_Data = null;
    /**
     * The window size representing the the Sakoe-Chiba Band.
     */
    private int m_WindowSize = 10;
    /**
     * The range of attributes to use for calculating the distance.
     */
    private Range m_AttributeIndices = new Range("first-last");

    /**
     * Calculates the distance between two instances.
     *
     * @param first the first instance
     * @param second the second instance
     * @return the distance between the two given instances
     */
    @Override
    public double distance(Instance first, Instance second) {

        return distance(first, second, Double.POSITIVE_INFINITY);

    }

    /**
     * Calculates the distance between two instances.
     *
     * @param first the first instance
     * @param second the second instance
     * @param stats the performance stats object
     * @return the distance between the two given instances
     */
    @Override
    public double distance(Instance first, Instance second, PerformanceStats stats) throws Exception {
        return distance(first, second, Double.POSITIVE_INFINITY, stats);
    }

    /**
     * Calculates the distance between two instances. Offers speed up (if the
     * distance function class in use supports it) in nearest neighbour search
     * by taking into account the cutOff or maximum distance. Depending on the
     * distance function class, post processing of the distances by
     * postProcessDistances(double []) may be required if this function is used.
     *
     * @param first the first instance
     * @param second the second instance
     * @param cutOffValue If the distance being calculated becomes larger than
     * cutOffValue then the rest of the calculation is discarded.
     * @return the distance between the two given instances or
     * Double.POSITIVE_INFINITY if the distance being calculated becomes larger
     * than cutOffValue.
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue) {
        return distance(first, second, cutOffValue, null);
    }

    /**
     * Calculates the distance between two instances. Offers speed up (if the
     * distance function class in use supports it) in nearest neighbour search
     * by taking into account the cutOff or maximum distance. Depending on the
     * distance function class, post processing of the distances by
     * postProcessDistances(double []) may be required if this function is used.
     *
     * @param first the first instance
     * @param second the second instance
     * @param cutOffValue If the distance being calculated becomes larger than
     * cutOffValue then the rest of the calculation is discarded.
     * @param stats the performance stats object
     * @return the distance between the two given instances or
     * Double.POSITIVE_INFINITY if the distance being calculated becomes larger
     * than cutOffValue.
     */
    @Override
    public double distance(Instance first, Instance second, double cutOffValue, PerformanceStats stats) {

        double attsFirst[] = first.toDoubleArray();
        double attsSecond[] = second.toDoubleArray();

        int classIndexOfFirst = first.classAttribute().index();

        double attsFirstF[] = new double[attsFirst.length - 1];
        double attsSecondF[] = new double[attsFirst.length - 1];

        int j = 0;
        for (int i = 0; i < attsFirst.length; i++) {
            if (i != classIndexOfFirst) {
                attsFirstF[j] = attsFirst[i];
                attsSecondF[j] = attsSecond[i];
                j++;
            }
        }
        return distanceDTW(attsFirstF, attsSecondF, getM_WindowSize());

    }

    /**
     * Internal method to calculate the DTW distance between two time series
     * using the Sakoe-Chiba Band.
     *
     * @param ts1 A time series.
     * @param ts2 A time series.
     * @param window The size of the Sakoe-Chiba Band.
     * @return
     */
    private double distanceDTW(double[] ts1, double[] ts2, int window) {


        window = (int) m_WindowSize * (ts1.length) / 100;
        window = Math.min(window, ts1.length - 1);

        // treating the case when the two time series have length = 1
        if (ts1.length == 1) {
            return (ts1[0] - ts2[0]) * (ts1[0] - ts2[0]);
        }

        double[][] array = new double[ts1.length][ts2.length];

        // fill the array with infinite values first
        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array.length; j++) {
                array[i][j] = Double.MAX_VALUE;
            }
        }

        // set the value in [0][0] position
        array[0][0] = (ts1[0] - ts2[0]) * (ts1[0] - ts2[0]);

        for (int i = 1; i <= window; i++) {
            array[0][i] = (ts1[i] - ts2[0]) * (ts1[i] - ts2[0]) + array[0][i - 1];
        }

        for (int i = 1; i <= window; i++) {
            array[i][0] = (ts1[0] - ts2[i]) * (ts1[0] - ts2[i]) + array[i - 1][0];
        }

        for (int i = 1; i < array.length; i++) {
            for (int j = Math.max(1, i - window); j <= Math.min(i + window, array.length - 1); j++) {
                if (j == i + window) {
                    array[i][j] = (ts1[j] - ts2[i]) * (ts1[j] - ts2[i]) + Math.min(array[i - 1][j - 1], array[i][j - 1]);
                } else if (j == i - window) {
                    array[i][j] = (ts1[j] - ts2[i]) * (ts1[j] - ts2[i]) + Math.min(array[i - 1][j - 1], array[i - 1][j]);
                } else {
                    array[i][j] = (ts1[j] - ts2[i]) * (ts1[j] - ts2[i]) + Math.min(Math.min(array[i - 1][j - 1], array[i][j - 1]), array[i - 1][j]);
                }
            }
        }

        return Math.sqrt(array[array.length - 1][array.length - 1]);
    }

    
    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration listOptions() {

        Vector result = new Vector();
        result.addElement(new Option(
                "\tSet the size of the Sakoe-Chiba Band for DTW algorithm.",
                "W", 1, "-W"));

        result.addElement(new Option(
                "\tSpecifies list of columns to used in the calculation of the \n"
                + "\tdistance. 'first' and 'last' are valid indices.\n"
                + "\t(default: first-last)",
                "R", 1, "-R <col1,col2-col4,...>"));

        result.addElement(new Option(
                "\tInvert matching sense of column indices.",
                "V", 0, "-V"));

        return result.elements();
    }

    /**
     * Parses a given list of options.
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        String tmpStr;

        setWarpingWindowSize(Integer.parseInt(Utils.getOption('W', options)));

        tmpStr = Utils.getOption('R', options);
        if (tmpStr.length() != 0) {
            setAttributeIndices(tmpStr);
        } else {
            setAttributeIndices("first-last");
        }

        setInvertSelection(Utils.getFlag('V', options));

    }

    /**
     * Gets the current settings. Returns empty array.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        // TODO Auto-generated method stub

        Vector<String> result;

        result = new Vector<String>();

        result.add("-W");
        result.add(Integer.toString(getWarpingWindowSize()));

        result.add("-R");
        result.add(getAttributeIndices());

        if (getInvertSelection()) {
            result.add("-V");
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * Sets the warping window size.
     *
     * @param windowSize the length of the window size to use
     */
    public void setWarpingWindowSize(int windowSize) {
        m_WindowSize = windowSize;
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String WarpingWindowSizeTipText() {
        return "Set the size of the warping window, uses the Sakoe-Chiba as "
                + "warping window global constraint.";

    }

    /**
     *
     *
     * @return the the warping window size
     */
    public int getWarpingWindowSize() {
        return getM_WindowSize();
    }

    /**
     * Does post processing of the distances (if necessary) returned by
     * distance(distance(Instance first, Instance second, double cutOffValue).
     * It may be necessary, depending on the distance function, to do post
     * processing to set the distances on the correct scale. Some distance
     * function classes may not return correct distances using the cutOffValue
     * distance function to minimize the inaccuracies resulting from floating
     * point comparison and manipulation.
     *
     * @param distances	the distances to post-process
     */
    @Override
    public void postProcessDistances(double distances[]) {
    }

    /**
     * returns the instances currently set.
     *
     * @return the current instances
     */
    @Override
    public Instances getInstances() {
        return m_Data;
    }

    /**
     * Sets the range of attributes to use in the calculation of the distance.
     * The indices start from 1, 'first' and 'last' are valid as well. E.g.:
     * first-3,5,6-last
     *
     * @param value	the new attribute index range
     */
    @Override
    public void setAttributeIndices(String value) {
        m_AttributeIndices.setRanges(value);

    }

    /**
     * Gets the range of attributes used in the calculation of the distance.
     *
     * @return	the attribute index range
     */
    @Override
    public String getAttributeIndices() {
        return m_AttributeIndices.getRanges();
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String attributeIndicesTipText() {
        return "Specify range of attributes to act on. "
                + "This is a comma separated list of attribute indices, with "
                + "\"first\" and \"last\" valid values. Specify an inclusive "
                + "range with \"-\". E.g: \"first-3,5,6-10,last\".";
    }

    /**
     * Sets the instances.
     *
     * @param insts the instances to use
     */
    @Override
    public void setInstances(Instances insts) {
        m_Data = insts;
    }

    /**
     * Sets whether the matching sense of attribute indices is inverted or not.
     *
     * @param value	if true the matching sense is inverted
     */
    @Override
    public void setInvertSelection(boolean value) {
        m_AttributeIndices.setInvert(value);

    }

    /**
     * Gets whether the matching sense of attribute indices is inverted or not.
     *
     * @return	true if the matching sense is inverted
     */
    @Override
    public boolean getInvertSelection() {
        return m_AttributeIndices.getInvert();
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String invertSelectionTipText() {
        return "Set attribute selection mode. If false, only selected "
                + "attributes in the range will be used in the distance calculation; if "
                + "true, only non-selected attributes will be used for the calculation.";
    }

    /**
     * Update the distance function (if necessary) for the newly added instance.
     *
     * @param ins	the instance to add
     */
    @Override
    public void update(Instance ins) {
    }

    /**
     * Returns a string describing this object.
     *
     * @return a description of the evaluator suitable for displaying in the
     * explorer/experimenter gui
     */
    public String globalInfo() {
        return "Implementing DTW distance function for time series."
                + "Uses the Sakoe-Chiba band as global constraint to limit the "
                + "DTW warping path.\n";
    }

    /**
     * @return the m_WindowSize
     */
    public int getM_WindowSize() {
        return m_WindowSize;
    }
}
