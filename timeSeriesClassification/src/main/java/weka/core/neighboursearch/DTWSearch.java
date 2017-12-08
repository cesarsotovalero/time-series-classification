package weka.core.neighboursearch;

import java.util.Enumeration;
import java.util.Vector;
import weka.core.*;

/**
 * <!-- globalinfo-start --> Class implementing the LB_Keogh algorithm as a
 * lower bounding measure to improve the DTWDistance function for nearest
 * neighbor search classification in time series data.
 *
 * <p/>
 * <!-- globalinfo-end -->
 *
 * @author César Soto-Valero (cesarsotovalero@gmail.com)
 */
public class DTWSearch extends NearestNeighbourSearch {

    /**
     * Whether to skip instances from the neighbor that are identical to the
     * query instance.
     */
    protected boolean m_SkipIdentical = false;
    /**
     * Array holding the distances of the nearest neighbor. It is filled up by
     * both nearestNeighbour() and kNearestNeighbours().
     */
    protected double[] m_Distances;

    /**
     * Constructor: Needs that setInstances(Instances) to be called before the
     * class is usable.
     */
    public DTWSearch() {
        m_DistanceFunction = new DTWDistance();

    }

    /**
     * Constructor that uses the supplied set of instances.
     *
     * @param insts the instances to use
     */
    public DTWSearch(Instances insts) {
        super(insts);
        m_DistanceFunction.setInstances(insts);
    }

    /**
     * Returns the nearest instance in the current neighbourhood to the supplied
     * instance.
     *
     * @param target The instance to find the nearest neighbour for.
     * @return The nearest neighbor
     * @throws Exception if the nearest neighbour could not be found.
     */
    @Override
    public Instance nearestNeighbour(Instance target) throws Exception {
        return (kNearestNeighbours(target, 1)).instance(0);
    }

    /**
     * Returns the k nearest instances in the current neighbourhood to the
     * supplied instance.
     *
     * @param target The instance to find the k nearest neighbours for.
     * @param k	The number of nearest neighbours to find.
     * @return The k nearest neighbors.
     * @throws Exception if the neighbours could not be found.
     */
    @Override
    public Instances kNearestNeighbours(Instance target, int k) {
        Instances neighbours = new Instances(m_Instances, k);

        int sizeW = (int) (((DTWDistance) m_DistanceFunction).getM_WindowSize() * (m_Instances.numAttributes() - 1) / 100); // the envelope width
        sizeW = Math.min(sizeW, m_Instances.numAttributes() - 2);

        double bestDistance = Double.MAX_VALUE;

        double[] lowerB;
        double[] upperB;

        lowerB = computeL(target, sizeW); //the lower bounding
        upperB = computeU(target, sizeW); //the upper bounding

        for (int i = 0; i < m_Instances.numInstances(); i++) {

            double distanceLB = computeLB_Keogh(lowerB, upperB, m_Instances.get(i)); // compute LB_Keogh

            if (distanceLB < bestDistance) {
                double distanceDTW = m_DistanceFunction.distance(m_Instances.get(i), target); //calculate DTW

                if (distanceDTW < bestDistance) {
                    bestDistance = distanceDTW;
                    neighbours.delete(); // removes all instances form the set
                    neighbours.add(m_Instances.get(i));
                } else if (distanceDTW == bestDistance) {
                    neighbours.add(m_Instances.get(i));
                }
            }

        }
        m_Distances = new double[neighbours.numInstances()];
        for (int i = 0; i < neighbours.numInstances(); i++) {
            m_Distances[i] = bestDistance;
        }

        return neighbours;
    }

    /**
     * Returns a string describing this nearest neighbour search algorithm.
     *
     * @return a description of the algorithm for displaying in the
     * explorer/experimenter gui
     */
    @Override
    public String globalInfo() {
        return "Class implementing LB_Keogh as lower bounding "
                + "measure to improve DTWDistance function for nearest neighbor search "
                + "classification in time series data.";
    }

    /**
     * Returns an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    @Override
    public Enumeration listOptions() {
        Vector<Option> result = new Vector<Option>();

        result.add(new Option(
                "\tSkip identical instances (distances equal to zero).\n",
                "S", 1, "-S"));

        return result.elements();
    }

    /**
     * Parses a given list of options.
     * <p/>
     *
     * <!-- options-start --> Valid options are:
     * <p/>
     *
     * <pre> -S
     *  Skip identical instances (distances equal to zero).
     * </pre>
     *
     * <!-- options-end -->
     *
     * @param options the list of options as an array of strings
     * @throws Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);

        setSkipIdentical(Utils.getFlag('S', options));
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    @Override
    public String distanceFunctionTipText() {
        return "The distance function to use for finding neighbours"
                + "(must be DTWDistance). ";
    }

    /**
     * sets the distance function to use for nearest neighbour search.
     *
     * @param df The new distance function to use.
     * @throws Exception if instances cannot be processed.
     */
    @Override
    public void setDistanceFunction(DistanceFunction df) throws Exception {
        if (df instanceof DTWDistance) {
            m_DistanceFunction = df;
        } else {
            throw new Exception("The distance function to use must be DTWDistance");
        }

    }

    /**
     * Gets the current settings.
     *
     * @return an array of strings suitable for passing to setOptions()
     */
    @Override
    public String[] getOptions() {
        Vector<String> result;
        String[] options;
        int i;

        result = new Vector<String>();

        options = super.getOptions();
        for (i = 0; i < options.length; i++) {
            result.add(options[i]);
        }

        if (getSkipIdentical()) {
            result.add("-S");
        }

        return result.toArray(new String[result.size()]);
    }

    /**
     * Returns the tip text for this property.
     *
     * @return tip text for this property suitable for displaying in the
     * explorer/experimenter gui
     */
    public String skipIdenticalTipText() {
        return "Whether to skip identical instances (with distance 0 to the target)";
    }

    /**
     * Sets the property to skip identical instances (with distance zero from
     * the target) from the set of neighbours returned.
     *
     * @param skip if true, identical instances are skipped
     */
    public void setSkipIdentical(boolean skip) {
        m_SkipIdentical = skip;
    }

    /**
     * Gets whether if identical instances are skipped from the neighbourhood.
     *
     * @return true if identical instances are skipped
     */
    public boolean getSkipIdentical() {
        return m_SkipIdentical;
    }

    /**
     * Returns the distances of the k nearest neighbours. The kNearestNeighbours
     * or nearestNeighbour needs to be called first for this to work.
     *
     * @return	the distances
     * @throws Exception if called before calling kNearestNeighbours or
     * nearestNeighbours.
     */
    @Override
    public double[] getDistances() throws Exception {
        if (m_Distances == null) {
            throw new Exception("No distances available. Please call either "
                    + "kNearestNeighbours or nearestNeighbours first.");
        }
        return m_Distances;
    }

    /**
     * Updates the LinearNNSearch to cater for the new added instance. This
     * implementation only updates the ranges of the DistanceFunction class,
     * since our set of instances is passed by reference and should already have
     * the newly added instance.
     *
     * @param ins The instance to add. Usually this is the instance that is
     * added to our neighbourhood i.e. the training instances.
     * @throws Exception if the given instances are null
     */
    @Override
    public void update(Instance ins) throws Exception {
        if (m_Instances == null) {
            throw new Exception("No instances supplied yet. Cannot update without"
                    + "supplying a set of instances first.");
        }
        m_DistanceFunction.update(ins);
    }

    /**
     * Returns the revision string.
     *
     * @return	the revision
     */
    @Override
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    /**
     * Compute the lower bound of the envelope
     *
     * @param query the instance to compute the envelope
     * @param sizeW percent of series length used to control the envelope width
     * @return an array of values with the lower bound of the query
     */
    private double[] computeL(Instance query, int sizeW) {

        double[] lowerB = new double[query.numAttributes() - 1]; //para excluir la clase

        int index = 0;
        double[] aux = new double[query.numAttributes() - 1];
        for (int i = 0; i < query.numAttributes(); i++) {
            if (i != query.classIndex()) {
                aux[index] = query.value(i);
                index++;
            }
        }

        for (int i = 0; i < lowerB.length; i++) {

            double min = Double.MAX_VALUE;
            for (int j = Math.max(0, i - sizeW); j <= Math.min(i + sizeW, query.numAttributes() - 2); j++) {
                if (aux[j] < min) {
                    min = aux[j];
                }
            }
            lowerB[i] = min;
        }

        return lowerB;
    }

    /**
     * Compute the upper bound of the envelope
     *
     * @param query the instance to compute the envelope
     * @param sizeW percent of series length used to control the width of the
     * envelope
     * @return an array of values with the upper bound of the query
     */
    private double[] computeU(Instance query, int sizeW) {

        double[] upperB = new double[query.numAttributes() - 1]; //para excluir la clase
        int index = 0;
        double[] aux = new double[query.numAttributes() - 1];
        for (int i = 0; i < query.numAttributes(); i++) {
            if (i != query.classIndex()) {
                aux[index] = query.value(i);
                index++;
            }
        }
        for (int i = 0; i < upperB.length; i++) {
            double max = Double.MIN_VALUE;
            for (int j = Math.max(0, i - sizeW); j <= Math.min(i + sizeW, query.numAttributes() - 2); j++) {
                if (aux[j] > max) {
                    max = aux[j];
                }
            }
            upperB[i] = max;
        }
        return upperB;
    }

    /**
     * Compute the LB_Keogh value between an envelope and a query sequence
     *
     * @param query
     * @param data
     * @param sizeW percent of series length used to control the width of the
     * envelope
     * @return the Euclidean distance between the target and the envelope
     */
    private double computeLB_Keogh(double[] lowerB, double[] upperB, Instance query) {

        double sumL = 0;
        double sumU = 0;
        //double[] theArrayTarget = query.toDoubleArray();

        int index = 0;
        double[] theArrayTarget = new double[query.numAttributes() - 1];

        for (int i = 0; i < query.numAttributes(); i++) {
            if (i != query.classIndex()) {
                theArrayTarget[index] = query.value(i);
                index++;
            }
        }

        for (int i = 0; i < theArrayTarget.length - 1; i++) { //excluding the class
            double p = theArrayTarget[i];

            if (p > upperB[i]) {
                sumL += Math.pow(p - upperB[i], 2);
            }

            if (p < lowerB[i]) {
                sumU += Math.pow(p - lowerB[i], 2);
            }
        }

        return Math.sqrt(sumU + sumL);

    }
}
